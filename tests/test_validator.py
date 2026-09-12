import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import validate_customvoice as validator


class ValidatorTests(unittest.TestCase):
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
