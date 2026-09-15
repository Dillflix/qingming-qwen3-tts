"""Real worker parser/API/encoders with a scripted native pipe, not a HIP test."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import httpx
import imageio_ffmpeg

from qingming_api.app import create_app
from qingming_api.audio import pcm16
from qingming_api.worker import GenerationLimitError, HEADER, NativeWorker
from test_api import CONFIG, FLOATS


def response(request, *, limited=False, updates=None, kind=2, event_name="completed", frames=None, event_id=None):
    count = request["max_new_tokens"] if limited else 1
    result = {"status": "ok", "eos": not limited, "eos_frame": -1 if limited else count,
              "frames": count, "max_new_tokens": request["max_new_tokens"]}
    if updates:
        result.update(updates)
    identifier = request["request_id"]
    terminal = json.dumps({"event": event_name, "request_id": identifier if event_id is None else event_id,
                           "result": result}).encode()
    # Each native packet stays below MAX_PAYLOAD, even in the 512-frame test.
    audio = HEADER.pack(b"QAF1", 1, identifier, len(FLOATS), 0) + FLOATS
    return (audio * (count if frames is None else frames)
            + HEADER.pack(b"QAF1", kind, identifier, len(terminal), 0) + terminal)


class ScriptedWorker(NativeWorker):
    """Use generate() unchanged; replace only the owned process/pipe boundary."""

    def __init__(self, model, replies, *, budget=2):
        super().__init__("unused", model, max_new_tokens=budget, request_timeout=2)
        self.replies = iter(replies)
        self.requests = []
        self.starts = 0
        self.closes = 0
        self.first_entered = asyncio.Event()
        self.release_first = asyncio.Event()
        self.release_first.set()

    async def start(self):
        self.starts += 1
        self.reader = asyncio.StreamReader()
        worker = self

        class Input:
            def write(self, data):
                request = json.loads(data)
                worker.requests.append(request)
                worker.reader.feed_data(next(worker.replies)(request))

            async def drain(self):
                if len(worker.requests) == 1:
                    worker.first_entered.set()
                    await worker.release_first.wait()

        self.process = SimpleNamespace(stdin=Input(), returncode=None)
        self.ready = True
        self.error = None

    async def close(self):
        self.closes += 1
        self.ready = False
        if self.process is not None:
            self.process.returncode = -15


async def eventually(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(wait(), 2)


class LimitRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        (self.model / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")

    async def test_full_512_frame_limit_keeps_pipe_and_next_request_usable(self):
        worker = ScriptedWorker(self.model, [lambda r: response(r, limited=True), response], budget=512)
        await worker.start()
        process, reader = worker.process, worker.reader
        with self.assertLogs("qingming.worker", level="WARNING") as logs:
            with self.assertRaises(GenerationLimitError) as caught:
                _ = [c async for c in worker.generate("Sensitive text", "Ryan", "English", "")]
        self.assertEqual((caught.exception.request_id, caught.exception.frames), (1, 512))
        self.assertTrue(worker.ready)
        self.assertIsNone(worker.error)
        self.assertIsNone(worker.last_result)  # Never present a truncation as a successful result.
        self.assertEqual(worker.closes, 0)
        self.assertIn("worker retained", logs.output[0])
        self.assertNotIn("Sensitive text", logs.output[0])
        chunks = [c async for c in worker.generate("Next.", "Ryan", "English", "")]
        self.assertEqual(chunks, [FLOATS])
        self.assertIs(worker.process, process)
        self.assertIs(worker.reader, reader)
        self.assertTrue(worker.last_result["eos"])
        self.assertEqual([r["request_id"] for r in worker.requests], [1, 2])
        self.assertTrue(all(not Path(r["output"]).parent.exists() for r in worker.requests))

    async def test_completion_id_must_be_an_integer_not_a_boolean(self):
        worker = ScriptedWorker(self.model, [lambda r: response(r, limited=True, event_id=True)])
        await worker.start()
        with self.assertRaisesRegex(RuntimeError, "wrong request ID"):
            _ = [c async for c in worker.generate("Hello.", "Ryan", "English", "")]
        self.assertEqual(worker.closes, 1)
        self.assertFalse(worker.ready)

    async def test_suspicious_limit_completions_still_discard_worker(self):
        cases = [
            {"updates": {"eos": "false"}}, {"updates": {"eos": None}},
            {"updates": {"status": "error"}}, {"kind": 3}, {"event_name": "error"},
            {"updates": {"frames": 1}}, {"updates": {"frames": 3}},
            {"updates": {"frames": True}}, {"updates": {"frames": 2.0}},
            {"updates": {"max_new_tokens": None}}, {"updates": {"max_new_tokens": 1}},
            {"updates": {"max_new_tokens": 2.0}}, {"updates": {"max_new_tokens": True}},
            {"updates": {"eos_frame": None}}, {"updates": {"eos_frame": 0}},
            {"updates": {"eos_frame": -1.0}}, {"frames": 1}, {"frames": 0},
        ]
        for case in cases:
            with self.subTest(case=case):
                worker = ScriptedWorker(self.model, [lambda r: response(r, limited=True, **case)])
                await worker.start()
                with self.assertRaises(RuntimeError) as caught:
                    _ = [c async for c in worker.generate("Hello.", "Ryan", "English", "")]
                self.assertNotIsInstance(caught.exception, GenerationLimitError)
                self.assertEqual(worker.closes, 1)
                self.assertFalse(worker.ready)
                self.assertIsNone(worker.last_result)

    async def test_limit_fails_only_one_http_request_including_queued_followers(self):
        for format in ("pcm", "wav", "mp3"):
            with self.subTest(format=format):
                worker = ScriptedWorker(self.model, [lambda r: response(r, limited=True)] + [response] * 5)
                worker.release_first.clear()
                app = create_app(self.model, worker, ffmpeg=imageio_ffmpeg.get_ffmpeg_exe(),
                                 supervisor_options={"poll_interval": 0.001, "initial_delay": 0.001})
                async with app.router.lifespan_context(app):
                    await eventually(lambda: worker.ready)
                    closes_at_start = worker.closes
                    process = worker.process
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                        first = asyncio.create_task(client.post("/v1/audio/speech", json={
                            "input": "A longer passage. " * 30, "response_format": format}))
                        await asyncio.wait_for(worker.first_entered.wait(), 2)
                        followers = [asyncio.create_task(client.post("/v1/audio/speech", json={
                            "input": f"Following {i}.", "response_format": format})) for i in range(5)]
                        await asyncio.sleep(0.02)
                        self.assertEqual(len(worker.requests), 1)
                        worker.release_first.set()
                        results = await asyncio.wait_for(asyncio.gather(first, *followers), 10)
                        self.assertEqual([r.status_code for r in results], [422, 200, 200, 200, 200, 200])
                        error = results[0].json()["error"]
                        self.assertEqual(error["code"], "speech_generation_limit")
                        self.assertEqual(error["param"], "input")
                        self.assertTrue(results[0].headers["content-type"].startswith("application/json"))
                        self.assertEqual((await client.get("/readyz")).status_code, 200)
                    self.assertEqual(worker.closes, closes_at_start)
                    self.assertEqual(worker.starts, 1)
                    self.assertEqual(app.state.supervisor.starts, 1)
                    self.assertIs(worker.process, process)
                    # No replay of the failed request or generation of its remaining segments.
                    self.assertEqual(len(worker.requests), 6)
                    self.assertEqual([r["text"] for r in worker.requests[1:]], [f"Following {i}." for i in range(5)])
                    self.assertTrue(all(not Path(r["output"]).parent.exists() for r in worker.requests))

    async def test_streamed_limit_aborts_body_but_next_request_works(self):
        worker = ScriptedWorker(self.model, [lambda r: response(r, limited=True), response])
        await worker.start()
        app = create_app(self.model, worker, api_key="test-key")
        body = json.dumps({"input": "Hello.", "stream": True, "response_format": "pcm"}).encode()
        delivered = False
        events = []

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(event):
            events.append(event)

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": "/v1/audio/speech", "raw_path": b"/v1/audio/speech",
                 "query_string": b"", "headers": [(b"content-type", b"application/json"),
                                                     (b"authorization", b"Bearer test-key")],
                 "server": ("test", 80), "client": ("test", 123)}
        with self.assertRaises(Exception) as caught:
            await asyncio.wait_for(app(scope, receive, send), 3)

        def contains_limit(error):
            return isinstance(error, GenerationLimitError) or any(
                contains_limit(child) for child in getattr(error, "exceptions", ()))

        self.assertTrue(contains_limit(caught.exception), repr(caught.exception))
        self.assertTrue(any(e["type"] == "http.response.start" and e["status"] == 200 for e in events))
        self.assertTrue(any(e.get("body") for e in events))
        self.assertFalse(any(e["type"] == "http.response.body" and not e.get("more_body", False) for e in events))
        self.assertTrue(worker.ready)
        self.assertEqual(worker.closes, 0)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                                     headers={"Authorization": "Bearer test-key"}) as client:
            next_response = await client.post("/v1/audio/speech", json={"input": "Next.", "response_format": "pcm"})
            self.assertEqual(next_response.status_code, 200)
            self.assertEqual(next_response.content, pcm16(FLOATS))


if __name__ == "__main__":
    unittest.main()
