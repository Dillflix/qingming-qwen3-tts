import asyncio
import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import imageio_ffmpeg
from qingming_api.worker import GenerationLimitError
from scripts import validate_base as validator
from scripts.clone_voice import sha256
from test_api import FLOATS
from test_base_voices import Fixture
from test_clone_voice import write_native_audio


class BaseValidatorTests(unittest.TestCase):
    def test_complete_gate_and_failure_modes_with_real_api_encoders(self):
        for mode in ("pass", "drift", "replaced", "early-eos"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                fixture = Fixture(root)
                binary = root / "binary"
                binary.write_bytes(b"fake")
                args = SimpleNamespace(binary=binary, model_dir=fixture.model, voice_library=fixture.library,
                                       voice_registry=fixture.registry, hip_device=0, max_new_tokens=512,
                                       ffmpeg=imageio_ffmpeg.get_ffmpeg_exe())
                out = root / "output"
                out.mkdir()
                once_calls = []
                def once(args, out, label, text, **kwargs):
                    self.assertFalse(worker.ready)  # Never coexist with the resident model.
                    once_calls.append(kwargs)
                    wav = out / f"{label}.wav"
                    write_native_audio(wav, samples=12 * 1920)
                    return {"native": {"status": "ok", "task": "base-xvector", "eos": True, "frames": 12},
                            "wav_sha256": sha256(wav)}

                class Worker:
                    task = "base-xvector"
                    def __init__(self):
                        self.ready, self.closed = False, False
                        self.process = SimpleNamespace(returncode=None)
                        self.info = {"task": self.task}
                        self.max_new_tokens = 512
                        self.calls = []
                    async def start(self):
                        self.ready = True
                    async def close(self):
                        self.ready, self.closed = False, True
                    async def generate(self, text, speaker, language, instruct, *, retain_dir=None):
                        self.calls.append((speaker, self.max_new_tokens))
                        if self.max_new_tokens == 1:
                            yield FLOATS
                            if mode != "early-eos":
                                raise GenerationLimitError(len(self.calls), 1)
                            return
                        if mode == "replaced":
                            self.process = SimpleNamespace(returncode=None)
                        wav = out / f"resident-{len(self.calls)}.wav"
                        write_native_audio(wav, samples=12 * 1920)
                        self.last_result = {"status": "ok", "task": "base-xvector", "eos": True, "frames": 12, "output": str(wav)}
                        yield (b"\0" * len(FLOATS) if mode == "drift" else FLOATS) * 8
                        await asyncio.sleep(0.001)
                        yield FLOATS * 4
                worker = Worker()
                report = {"status": "FAIL"}
                # Deterministic clock: Windows monotonic ticks can exceed the fake
                # worker's entire request duration. Real GPU timing is not mocked
                # by the host qualification command.
                ticks = itertools.count()
                with patch.object(validator, "run_native", side_effect=once), \
                     patch.object(validator, "NativeWorker", return_value=worker), \
                     patch.object(validator, "time", SimpleNamespace(monotonic=lambda: next(ticks))):
                    if mode == "pass":
                        asyncio.run(validator.validate(args, out, report))
                        self.assertEqual(report["status"], "PASS")
                        self.assertEqual([name for name, budget in worker.calls[:3]], ["Samuel", "Will", "Samuel"])
                        self.assertEqual([budget for name, budget in worker.calls].count(1), 1)
                        self.assertTrue((out / "api.mp3").is_file())
                    else:
                        with self.assertRaises(RuntimeError):
                            asyncio.run(validator.validate(args, out, report))
                        self.assertEqual(report["status"], "FAIL")
                self.assertTrue(worker.closed)
                self.assertEqual(len(once_calls), 2)
                self.assertTrue(all("embedding" in call and "reference" not in call for call in once_calls))
