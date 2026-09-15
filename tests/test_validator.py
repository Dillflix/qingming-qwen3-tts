import asyncio
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts import validate_customvoice as validator
from qingming_api.worker import GenerationLimitError


class ValidatorTests(unittest.TestCase):
    def test_limit_recovery_gate_checks_reuse_and_identical_control_audio(self):
        for mode in ("pass", "drift", "unavailable", "early-eos"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                from test_api import CONFIG, FLOATS
                (root / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
                binary = root / "fake-binary"
                binary.write_bytes(b"test")
                args = SimpleNamespace(model_dir=root, binary=binary, hip_device=None, expected_calm_sha256=None)

                class Worker:
                    def __init__(self):
                        self.process = SimpleNamespace(pid=123, returncode=None)
                        self.info = {"max_new_tokens": 512}
                        self.max_new_tokens = 512
                        self.ready = False
                        self.closed = False
                        self.budgets = []

                    async def start(self):
                        self.ready = True

                    async def close(self):
                        self.closed = True
                        self.ready = False

                    async def generate(self, *unused, retain_dir):
                        self.budgets.append(self.max_new_tokens)
                        identifier = len(self.budgets)
                        limited = self.max_new_tokens == 8 and mode != "early-eos"
                        frames = 8 if limited else 1
                        pcm = FLOATS * frames
                        if mode == "drift" and identifier > 2:
                            pcm = b"\0" * len(pcm)
                        path = retain_dir / f"{identifier}.wav"
                        header = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
                                  + struct.pack("<IHHIIHH", 16, 3, 1, 24000, 96000, 4, 32)
                                  + b"data" + struct.pack("<I", len(pcm)))
                        path.write_bytes(header + pcm)
                        yield pcm
                        if limited:
                            if mode == "unavailable":
                                self.ready = False
                            raise GenerationLimitError(identifier, frames)
                        self.last_result = {"output": str(path), "eos": True}

                worker = Worker()
                report = {"status": "FAIL"}
                with patch.object(validator, "NativeWorker", return_value=worker):
                    if mode == "pass":
                        asyncio.run(validator.validate_limit_recovery(args, root, report))
                        self.assertEqual(report["status"], "PASS")
                        self.assertEqual(worker.budgets, [512, 8, 512, 512, 512, 512, 512])
                    else:
                        with self.assertRaises(RuntimeError):
                            asyncio.run(validator.validate_limit_recovery(args, root, report))
                        self.assertEqual(report["status"], "FAIL")
                self.assertTrue(worker.closed)
                self.assertEqual(worker.max_new_tokens, 512)

    def test_native_wav_contract_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.wav"
            pcm = struct.pack("<2f", 0.25, -0.25)
            raw = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
                   + struct.pack("<IHHIIHH", 16, 3, 1, 24000, 96000, 4, 32)
                   + b"data" + struct.pack("<I", len(pcm)) + pcm)
            path.write_bytes(raw)
            self.assertEqual(validator.wav_payload(path), pcm)
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                validator.wav_payload(path)

    def test_failure_is_packaged_without_a_gpu_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ["validate_customvoice.py", "--binary", str(root / "missing-binary"),
                    "--model-dir", str(root / "missing-model")]
            with patch.object(validator, "ROOT", root), patch("sys.argv", args):
                self.assertEqual(validator.main(), 1)
            archive, = root.glob("*.tar.gz")
            with tarfile.open(archive) as bundle:
                report_name, = [name for name in bundle.getnames() if name.endswith("/report.json")]
                report = json.load(bundle.extractfile(report_name))
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(report["errors"])
            self.assertEqual(archive.with_name(archive.name + ".sha256").read_text().split()[0],
                             validator.sha256(archive))


if __name__ == "__main__":
    unittest.main()
