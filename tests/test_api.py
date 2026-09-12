import asyncio
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave

import httpx
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from qingming_api.app import create_app
from qingming_api.audio import encode, pcm16, tempo_filters
from qingming_api.contract import SpeechRequest, VoiceCatalog, split_text
from qingming_api.worker import HEADER, NativeWorker


CONFIG = {"tts_model_size": "1b7", "tts_model_type": "custom_voice", "talker_config": {
    "spk_id": {"vivian": 3065, "Ryan": 3061, "dylan": 2878},
    "spk_is_dialect": {"vivian": False, "Ryan": False, "dylan": "beijing_dialect"},
    "codec_language_id": {"english": 2050, "chinese": 2055, "beijing_dialect": 2074},
}}
FLOATS = struct.pack("<" + "f" * 1920, *([0.25, -0.25] * 960))


class FakeWorker:
    def __init__(self):
        self.ready = False
        self.error = None
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.completed = False
        self.closed = False

    async def start(self):
        self.ready = True

    async def close(self):
        self.ready = False
        self.closed = True

    async def generate(self, text, speaker, language, instruct):
        self.calls.append((text, speaker, language, instruct))
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        self.completed = False
        try:
            yield FLOATS
            await asyncio.sleep(0.001)
            yield FLOATS
            self.completed = True
        finally:
            self.active -= 1


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        (self.model / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
        self.worker = FakeWorker()
        self.app = create_app(self.model, self.worker)

    def client(self):
        return TestClient(self.app)

    def test_catalog_preserves_case_filters_aliases_and_languages(self):
        with self.client() as client:
            voices = client.get("/v1/voices").json()["data"]
            self.assertEqual(voices, client.get("/v1/audio/voices").json()["data"])
            self.assertIn("Ryan", {v["id"] for v in voices})
            self.assertEqual({v["id"] for v in voices if v["alias"]}, {"alloy", "echo"})
            names = {m["id"] for m in client.get("/v1/models").json()["data"]}
            self.assertIn("tts-1-hd-en", names)
            self.assertNotIn("tts-1-da", names)

    def test_pcm_and_wav_are_valid_signed_16_bit(self):
        with self.client() as client:
            pcm = client.post("/v1/audio/speech", json={"input": "Hello.", "response_format": "pcm"})
            self.assertEqual(pcm.status_code, 200)
            self.assertEqual(pcm.content, pcm16(FLOATS) * 2)
            wav = client.post("/v1/audio/speech", json={"input": "Hello.", "response_format": "wav"})
            with wave.open(io.BytesIO(wav.content)) as decoded:
                self.assertEqual(decoded.getparams()[:4], (1, 2, 24000, 3840))
                self.assertEqual(decoded.readframes(3840), pcm.content)

    def test_true_pcm_response_and_instruction_alias(self):
        with self.client() as client:
            response = client.post("/v1/audio/speech", json={"input": "Hello.", "voice": "rYaN", "stream": True,
                "response_format": "pcm", "model": "tts-1-en", "instructions": "Calm."})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-audio-sample-format"], "s16le")
            self.assertEqual(self.worker.calls[-1][1:], ("Ryan", "english", "Calm."))
            self.assertEqual(response.content, pcm16(FLOATS) * 2)

    def test_invalid_requests_never_reach_worker(self):
        cases = [{"voice": "clone:someone"}, {"ref_audio": "file.wav"}, {"voice": "fable"},
                 {"model": "tts-1-da"}, {"language": "Danish"}, {"speed": 0.1}, {"speed": 4.1},
                 {"stream": True}, {"input": "x" * 4097}, {"input": " "},
                 {"instruct": "a", "instructions": "b"}, {"stream_format": "sse"}]
        with self.client() as client:
            for extra in cases:
                with self.subTest(extra=extra):
                    response = client.post("/v1/audio/speech", json={"input": "Hello.", **extra})
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertIn("error", response.json())
            self.assertEqual(self.worker.calls, [])
            clone = client.post("/v1/audio/speech", json={"input": "Hello.", "voice": "clone:a"}).json()
            self.assertEqual(clone["error"]["code"], "voice_cloning_not_supported")

    def test_body_limit_and_bad_json(self):
        with self.client() as client:
            self.assertEqual(client.post("/v1/audio/speech", content=b"x" * 70000).status_code, 413)
            self.assertEqual(client.post("/v1/audio/speech", content=b"{").status_code, 400)

    def test_authentication_and_readiness(self):
        app = create_app(self.model, self.worker, api_key="local-test-key")
        with TestClient(app) as client:
            self.assertEqual(client.get("/healthz").status_code, 200)
            self.assertEqual(client.get("/v1/models").status_code, 401)
            headers = {"Authorization": "Bearer local-test-key"}
            self.assertEqual(client.get("/v1/models", headers=headers).status_code, 200)
            self.worker.ready = False
            self.assertEqual(client.get("/readyz", headers=headers).status_code, 503)

    def test_chunking_preserves_speaker_style_and_gap(self):
        text = "This is the first sentence to speak clearly. " * 20
        with self.client() as client:
            response = client.post("/v1/audio/speech", json={"input": text, "response_format": "pcm", "instruct": "Calm."})
            self.assertEqual(response.status_code, 200)
            count = len(split_text(text))
            self.assertGreater(count, 1)
            self.assertEqual(len(self.worker.calls), count)
            self.assertEqual([call[0] for call in self.worker.calls], split_text(text))
            self.assertTrue(all(len(call[0]) <= 400 and call[1:] == ("vivian", "Auto", "Calm.") for call in self.worker.calls))
            self.assertEqual(len(response.content), count * len(FLOATS) + (count - 1) * 5760)

    def test_short_multisentence_input_is_one_generation_without_added_gap(self):
        text = "This is the first sentence to speak clearly.\n" * 3
        self.assertGreater(len(text), 70)
        with self.client() as client:
            response = client.post("/v1/audio/speech", json={"input": text, "response_format": "pcm"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual([call[0] for call in self.worker.calls], [text.strip()])
            self.assertEqual(response.content, pcm16(FLOATS) * 2)

    def test_segment_cap_and_existing_fallbacks_preserve_text(self):
        self.assertEqual(split_text("x" * 400), ["x" * 400])
        self.assertEqual(split_text("x" * 401), ["x" * 400, "x"])
        self.assertEqual(split_text(" \n\t "), [])
        sentence = "Read this whole sentence calmly. "
        text = sentence * 30
        parts = split_text(text)
        self.assertTrue(all(part.endswith(".") for part in parts))
        self.assertEqual(" ".join(parts), text.strip())
        text = "ordinary words without punctuation " * 30
        parts = split_text(text)
        self.assertEqual(" ".join(parts), text.strip())
        self.assertTrue(all(0 < len(part) <= 400 for part in parts))

    def test_incomplete_nonstreamed_audio_is_not_returned_as_success(self):
        class IncompleteWorker(FakeWorker):
            async def generate(self, *args):
                yield FLOATS
                raise RuntimeError("Native generation reached its token limit before EOS")
        app = create_app(self.model, IncompleteWorker())
        with TestClient(app) as client:
            response = client.post("/v1/audio/speech", json={"input": "A longer passage. " * 15,
                                                          "response_format": "pcm"})
            self.assertEqual(response.status_code, 502)
            self.assertIn("error", response.json())

    def test_checkpoint_type_guard_and_alias_configuration(self):
        catalog = VoiceCatalog(self.model, {"narrator": "RYAN", "invalid": "Sophia"})
        self.assertEqual(catalog.aliases, {"narrator": "Ryan"})
        bad = dict(CONFIG, tts_model_type="base")
        (self.model / "config.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "CustomVoice"):
            VoiceCatalog(self.model)

    def test_audio_conversion_and_unicode_chunking(self):
        self.assertEqual(pcm16(struct.pack("<4f", -2, -0.5, 0.5, 2)), struct.pack("<4h", -32768, -16384, 16384, 32767))
        with self.assertRaises(ValueError):
            pcm16(struct.pack("<f", float("nan")))
        for text in ("你好世界。" * 100, "a" * 801, "🌤️" * 300):
            parts = split_text(text)
            self.assertEqual("".join(parts), text)
            self.assertTrue(all(0 < len(p) <= 400 for p in parts))
        self.assertEqual(tempo_filters(0.25), "atempo=0.5,atempo=0.5")
        self.assertEqual(tempo_filters(4), "atempo=2,atempo=2")


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        (self.model / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
        self.worker = FakeWorker()
        self.worker.ready = True

    async def test_openai_sdk_default_mp3_and_all_encoders(self):
        import imageio_ffmpeg
        app = create_app(self.model, self.worker, ffmpeg=imageio_ffmpeg.get_ffmpeg_exe())
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test/v1")
        async with AsyncOpenAI(api_key="not-needed", base_url="http://test/v1", http_client=http) as sdk:
            response = await sdk.audio.speech.create(model="tts-1", voice="Ryan", input="Hello.")
            self.assertGreater(len(response.content), 100)
            self.assertTrue(response.content.startswith(b"ID3") or response.content[0] == 255)
            for format in ("opus", "aac", "flac", "wav", "pcm"):
                with self.subTest(format=format):
                    response = await sdk.audio.speech.create(model="tts-1", voice="Ryan", input="Hello.", response_format=format, speed=0.5)
                    self.assertGreater(len(response.content), 100)

    async def test_concurrent_requests_are_serialized(self):
        app = create_app(self.model, self.worker)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            results = await asyncio.gather(*(client.post("/v1/audio/speech", json={"input": f"Hello {i}.", "response_format": "pcm"}) for i in range(3)))
            self.assertTrue(all(r.status_code == 200 for r in results))
            self.assertEqual(self.worker.max_active, 1)

    async def test_stream_reaches_asgi_client_before_native_completion(self):
        app = create_app(self.model, self.worker)
        body = json.dumps({"input": "Hello.", "stream": True, "response_format": "pcm"}).encode()
        delivered = False
        finished = asyncio.Event()
        observed = []
        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await finished.wait()
            return {"type": "http.disconnect"}
        async def send(event):
            if event["type"] == "http.response.body" and event.get("body"):
                observed.append(self.worker.completed)
            if event["type"] == "http.response.body" and not event.get("more_body", False):
                finished.set()
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": "/v1/audio/speech", "raw_path": b"/v1/audio/speech",
                 "query_string": b"", "headers": [(b"content-type", b"application/json")], "server": ("test", 80), "client": ("test", 123)}
        await asyncio.wait_for(app(scope, receive, send), 5)
        self.assertTrue(observed)
        self.assertFalse(observed[0])

    async def test_native_frame_parser_and_completion(self):
        class Input:
            def write(self, data):
                self.data = data
            async def drain(self):
                pass
        class Process:
            returncode = None
            stdin = Input()
        worker = NativeWorker("unused", self.model)
        worker.process = Process()
        worker.ready = True
        worker.reader = asyncio.StreamReader()
        event = json.dumps({"event": "completed", "request_id": 1, "result": {"status": "ok", "eos": True, "frames": 1}}).encode()
        worker.reader.feed_data(HEADER.pack(b"QAF1", 1, 1, len(FLOATS), 0) + FLOATS + HEADER.pack(b"QAF1", 2, 1, len(event), 0) + event)
        chunks = [c async for c in worker.generate("你好", "Ryan", "auto", "Calm.")]
        self.assertEqual(chunks, [FLOATS])
        self.assertIn("你好".encode(), worker.process.stdin.data)
        self.assertNotIn(b"ref_audio", worker.process.stdin.data)

    async def test_http_disconnect_releases_stream_and_worker_lock(self):
        class HeldWorker(FakeWorker):
            async def generate(self, *args):
                self.active += 1
                try:
                    yield FLOATS
                    await asyncio.Event().wait()
                finally:
                    self.active -= 1
        worker = HeldWorker()
        worker.ready = True
        app = create_app(self.model, worker, queue_timeout=0.05)
        body = json.dumps({"input": "Hello.", "stream": True, "response_format": "pcm"}).encode()
        delivered = False
        disconnect = asyncio.Event()
        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}
        async def send(event):
            if event["type"] == "http.response.body" and event.get("body"):
                disconnect.set()
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": "/v1/audio/speech", "raw_path": b"/v1/audio/speech",
                 "query_string": b"", "headers": [(b"content-type", b"application/json")], "server": ("test", 80), "client": ("test", 123)}
        await asyncio.wait_for(app(scope, receive, send), 5)
        self.assertEqual(worker.active, 0)
        # A later request can acquire the lock; it must not remain stranded.
        worker.generate = self.worker.generate
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/audio/speech", json={"input": "Again.", "response_format": "pcm"})
            self.assertEqual(response.status_code, 200)

    async def test_tempo_stream_does_not_wait_for_source_completion(self):
        import imageio_ffmpeg
        for speed in (0.25, 0.5, 2.0, 4.0):
            with self.subTest(speed=speed):
                proceed = asyncio.Event()
                async def source():
                    yield FLOATS * 8  # Native first packet: 640 ms of audio.
                    await proceed.wait()
                    yield FLOATS * 16
                stream = encode(source(), "pcm", speed, imageio_ffmpeg.get_ffmpeg_exe())
                try:
                    first = await asyncio.wait_for(anext(stream), 3)
                    self.assertTrue(first)
                    proceed.set()
                    async for _ in stream:
                        pass
                finally:
                    await stream.aclose()

    async def test_bad_native_frame_closes_owned_worker(self):
        from unittest.mock import AsyncMock
        class Input:
            def write(self, data):
                pass
            async def drain(self):
                pass
        class Process:
            returncode = None
            stdin = Input()
        worker = NativeWorker("unused", self.model)
        worker.process = Process()
        worker.ready = True
        worker.reader = asyncio.StreamReader()
        worker.close = AsyncMock()
        worker.reader.feed_data(HEADER.pack(b"QAF1", 1, 999, 4, 0))
        with self.assertRaisesRegex(RuntimeError, "misrouted"):
            await anext(worker.generate("Hello", "Ryan", "English", "Calm."))
        worker.close.assert_awaited_once()

    async def test_native_generation_without_eos_fails_closed(self):
        from unittest.mock import AsyncMock
        class Input:
            def write(self, data):
                self.data = data
            async def drain(self):
                pass
        class Process:
            returncode = None
            stdin = Input()
        worker = NativeWorker("unused", self.model)
        worker.process = Process()
        worker.ready = True
        worker.reader = asyncio.StreamReader()
        worker.close = AsyncMock()
        event = json.dumps({"event": "completed", "request_id": 1,
                            "result": {"status": "ok", "eos": False, "frames": 1}}).encode()
        worker.reader.feed_data(HEADER.pack(b"QAF1", 1, 1, len(FLOATS), 0) + FLOATS
                                + HEADER.pack(b"QAF1", 2, 1, len(event), 0) + event)
        with self.assertRaisesRegex(RuntimeError, "before EOS"):
            _ = [chunk async for chunk in worker.generate("A longer passage. " * 15, "Ryan", "English", "")]
        self.assertEqual(json.loads(worker.process.stdin.data)["max_new_tokens"], 512)
        worker.close.assert_awaited_once()
        self.assertIsNone(worker.last_result)

    async def test_native_eof_timeout_and_abandoned_stream_fail_closed(self):
        from unittest.mock import AsyncMock
        class Input:
            def write(self, data):
                pass
            async def drain(self):
                pass
        class Process:
            returncode = None
            stdin = Input()
        for mode in ("eof", "timeout", "abandoned"):
            with self.subTest(mode=mode):
                worker = NativeWorker("unused", self.model, request_timeout=0.02)
                worker.process = Process()
                worker.ready = True
                worker.reader = asyncio.StreamReader()
                worker.close = AsyncMock()
                stream = worker.generate("Hello", "Ryan", "English", "Calm.")
                if mode == "eof":
                    worker.reader.feed_eof()
                    with self.assertRaises(asyncio.IncompleteReadError):
                        await anext(stream)
                elif mode == "timeout":
                    with self.assertRaises(asyncio.TimeoutError):
                        await anext(stream)
                else:
                    worker.reader.feed_data(HEADER.pack(b"QAF1", 1, 1, len(FLOATS), 0) + FLOATS)
                    self.assertEqual(await anext(stream), FLOATS)
                    await stream.aclose()
                worker.close.assert_awaited_once()
                self.assertIn("interrupted", worker.error)


if __name__ == "__main__":
    unittest.main()
