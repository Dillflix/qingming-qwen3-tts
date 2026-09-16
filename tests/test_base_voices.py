import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx
import imageio_ffmpeg

from qingming_api.app import create_app
from qingming_api.base_voices import BaseVoiceCatalog
from qingming_api.config import parse_args
from qingming_api.contract import SpeechRequest, APIError
from qingming_api.worker import GenerationLimitError
from scripts.clone_voice import fingerprint_model, sha256, SCHEMA
from scripts.configure_base_service import render
from test_api import FakeWorker, FLOATS
from test_clone_voice import write_reference, EMBEDDING
from test_limit_recovery import ScriptedWorker, response


class Fixture:
    def __init__(self, root):
        self.root = root
        self.model, self.library = root / "model", root / "voices"
        self.model.mkdir()
        self.library.mkdir()
        (self.model / "config.json").write_text(json.dumps({"tts_model_type": "base", "tts_model_size": "1b7",
            "talker_config": {"codec_language_id": {"english": 1, "chinese": 2}}}))
        (self.model / "model.safetensors").write_bytes(b"fixture-weights")
        model = fingerprint_model(self.model)
        for name, slug in {"Samuel": "samuel-v1", "Will": "will-v1"}.items():
            directory = self.library / slug
            directory.mkdir()
            write_reference(directory / "reference.wav")
            (directory / "speaker.bf16").write_bytes(EMBEDDING if name == "Samuel" else EMBEDDING[::-1])
            (directory / "profile.json").write_text(json.dumps({"schema": SCHEMA, "status": "ENROLLED",
                "voice_id": slug, "method": "x_vector_only", "consent_confirmed": True,
                "model_files_sha256": model, "files": {file: sha256(directory / file) for file in ("reference.wav", "speaker.bf16")}}))
        self.registry = self.library / "production.json"
        self.spec = {"schema": "qingming-base-registry-v1", "default": "Samuel", "voices": {"Samuel": "samuel-v1", "Will": "will-v1"}}
        self.save()

    def save(self):
        self.registry.write_text(json.dumps(self.spec))

    def catalog(self):
        return BaseVoiceCatalog(self.model, self.library, self.registry)


class BaseCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))

    def test_named_voices_default_no_aiden_and_no_styles(self):
        catalog = self.fixture.catalog()
        for name, expected in (("sAmUeL", "Samuel"), ("Will", "Will"), ("echo", "Samuel"), ("alloy", "Samuel")):
            self.assertEqual(catalog.resolve(SpeechRequest(input="Hi", voice=name))[0], expected)
        for request in ({"voice": "Aiden"}, {"voice": "../../private"}, {"voice": "Will", "instructions": "Laugh"},
                        {"voice": "Will", "model": "bad"}, {"voice": "Will", "language": "bad"}):
            with self.assertRaises(APIError):
                catalog.resolve(SpeechRequest(input="Hi", **request))
        self.assertNotIn("speaker.bf16", json.dumps(catalog.voices()))
        self.assertNotEqual(catalog.embeddings["Samuel"], catalog.embeddings["Will"])

    def test_registry_and_payload_integrity(self):
        original = json.loads(json.dumps(self.fixture.spec))
        for voices in ({"../bad": "samuel-v1"}, {"Samuel": "../escape"}, {"Aiden": "samuel-v1"},
                       {"Samuel": "samuel-v1", "samuel": "will-v1"}, {}):
            self.fixture.spec["voices"] = voices
            self.fixture.save()
            with self.assertRaises(ValueError):
                self.fixture.catalog()
        self.fixture.spec = original
        self.fixture.save()
        (self.fixture.library / "will-v1/speaker.bf16").write_bytes(b"bad")
        with self.assertRaises(ValueError):
            self.fixture.catalog()

    def test_checkpoint_is_hashed_once_and_vectors_snapshotted(self):
        with patch("qingming_api.base_voices.fingerprint_model", wraps=fingerprint_model) as fingerprint:
            catalog = self.fixture.catalog()
            self.assertEqual(fingerprint.call_count, 1)
        before = dict(catalog.embeddings)
        (self.fixture.library / "will-v1/speaker.bf16").write_bytes(b"changed later")
        self.assertEqual(before, dict(catalog.embeddings))
        with self.assertRaises(TypeError):
            catalog.embeddings["Will"] = "overwrite"

    def test_config_requires_explicit_base_registry_and_service_only_overrides_exec(self):
        args = parse_args(["--task", "base-xvector", "--voice-library", str(self.fixture.library),
                           "--voice-registry", str(self.fixture.registry)], {})
        self.assertEqual(args.task, "base-xvector")
        content = render(Path("/home/test"), Path("/opt/base/binary"), Path("/opt/base/model"),
                         Path("/home/test/voices"), Path("/home/test/voices/production.json"), 0)
        self.assertIn("ExecStart=\nExecStart=", content)
        self.assertIn("--task base-xvector", content)
        self.assertIn("--api-key-file %d/api-key", content)
        self.assertNotIn("LoadCredential=", content)
        self.assertNotIn("QINGMING_HOST=", content)

    def test_http_voice_switching_default_and_rejection_before_gpu(self):
        worker = FakeWorker()
        worker.task = "base-xvector"
        catalog = self.fixture.catalog()
        app = create_app(self.fixture.model, worker, catalog=catalog, api_key="test", ffmpeg=imageio_ffmpeg.get_ffmpeg_exe())
        with TestClient(app, headers={"Authorization": "Bearer test"}) as client:
            self.assertEqual(client.get("/readyz").json()["task"], "base-xvector")
            self.assertEqual(client.get("/v1/voices").json()["default"], "Samuel")
            for voice in (None, "Will", "Samuel"):
                body = {"input": "Hello!", "response_format": "pcm"}
                if voice:
                    body["voice"] = voice
                self.assertEqual(client.post("/v1/audio/speech", json=body).status_code, 200)
            self.assertEqual([call[1] for call in worker.calls], ["Samuel", "Will", "Samuel"])
            count = len(worker.calls)
            for extra in ({"voice": "Aiden"}, {"voice": "Will", "instruct": "cheerfully"},
                          {"voice": "Will", "speaker_embedding_hex": "bad"}, {"voice": "Will", "ref_audio": "/private"}):
                self.assertEqual(client.post("/v1/audio/speech", json={"input": "hello", **extra}).status_code, 400)
            self.assertEqual(len(worker.calls), count)


class BaseWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_base_payload_switching_and_clean_limit_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder))
            catalog = fixture.catalog()
            worker = ScriptedWorker(fixture.model, [response, response, lambda r: response(r, limited=True), response])
            worker.task = "base-xvector"
            worker.voice_embeddings = dict(catalog.embeddings)
            await worker.start()
            process = worker.process
            for name in ("Samuel", "Will"):
                raw = [part async for part in worker.generate("Hi", name, "English", "")]
                self.assertEqual(raw, [FLOATS])
            with self.assertRaises(GenerationLimitError):
                _ = [part async for part in worker.generate("Hi", "Will", "English", "")]
            _ = [part async for part in worker.generate("Hi", "Samuel", "English", "")]
            self.assertIs(worker.process, process)
            self.assertEqual(worker.closes, 0)
            self.assertEqual([req["speaker_embedding_hex"] for req in worker.requests],
                             [catalog.embeddings[name] for name in ("Samuel", "Will", "Will", "Samuel")])
            for req in worker.requests:
                self.assertFalse({"speaker", "instruct", "ref_audio"} & req.keys())
            with self.assertRaises(ValueError):
                _ = [part async for part in worker.generate("Hi", "Aiden", "English", "")]
            self.assertEqual(len(worker.requests), 4)
