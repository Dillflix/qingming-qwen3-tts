from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import configure_base_service as installer
from scripts.validate_base import identity
from test_base_voices import Fixture


class BaseDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)
        self.binary = self.root / "base-binary"
        self.binary.write_bytes(b"binary")
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"python")
        self.unit = self.root / "qingming-tts.service"
        self.unit.write_text("old unit preserved")
        self.dropin = self.root / "qingming-tts.service.d/50-base-voices.conf"
        self.report = self.root / "report.json"
        args = SimpleNamespace(binary=self.binary, voice_registry=self.fixture.registry, hip_device=0)
        self.report.write_text(json.dumps({"status": "PASS", **identity(args, self.fixture.catalog())}))
        self.argv = ["--binary", str(self.binary), "--model-dir", str(self.fixture.model),
                     "--voice-library", str(self.fixture.library), "--voice-registry", str(self.fixture.registry),
                     "--hip-device", "0", "--validation-report", str(self.report)]

    def invoke(self, *, install=False, failing=False):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            if failing and command[0] == "systemd-analyze":
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        with patch.object(installer, "ROOT", self.root), patch.object(installer, "UNIT", self.unit), \
             patch.object(installer, "DROPIN", self.dropin), patch.object(installer.sys, "platform", "linux"), \
             patch.object(installer.os, "geteuid", return_value=0, create=True), \
             patch.object(installer, "atomic_write", side_effect=lambda p, content, mode: p.write_text(content)), \
             patch.object(installer.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            installer.main(self.argv + (["--install"] if install else []))
        return calls

    def test_installs_only_dropin_preserving_unit_without_service_control(self):
        calls = self.invoke(install=True)
        self.assertEqual(self.unit.read_text(), "old unit preserved")
        self.assertIn("--task base-xvector", self.dropin.read_text())
        self.assertEqual(calls[1], ["systemctl", "daemon-reload"])
        self.assertEqual(len(calls), 2)
        self.assertFalse(self.invoke(install=True))  # Idempotent exact match.

    def test_venv_interpreter_symlink_is_not_resolved(self):
        python = self.root / '.venv/bin/python'
        system_python = self.root / 'usr/bin/python3.14'
        original_resolve = Path.resolve

        # Simulate the Linux venv symlink on every test platform, including
        # Windows hosts without permission to create filesystem symlinks.
        def resolve(path, *args, **kwargs):
            return system_python if path == python else original_resolve(path, *args, **kwargs)

        with patch.object(Path, 'resolve', resolve):
            self.invoke(install=True)
        content = self.dropin.read_text()
        expected = installer.quote(python.absolute()).replace('$', '$$')
        self.assertIn(f'ExecStart={expected} -m qingming_api', content)
        self.assertNotIn(str(system_python), content)

    def test_dependency_preflight_precedes_service_stop(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / 'scripts/deploy_base_voices.sh').read_text()
        self.assertLess(script.index('import uvicorn; from scripts import validate_base'),
                        script.index('sudo systemctl stop'))
        requirements = (root / 'requirements-base-production.txt').read_text()
        self.assertIn('-r requirements-api.txt', requirements)
        self.assertIn('httpx>=0.27,<1', requirements)

    def test_failed_systemd_validation_removes_only_new_dropin(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(install=True, failing=True)
        self.assertFalse(self.dropin.exists())
        self.assertEqual(self.unit.read_text(), "old unit preserved")

    def test_missing_or_stale_qualification_never_writes(self):
        original = self.report.read_text()
        for change in ({"status": "FAIL"}, {"binary_sha256": "wrong"}, {"registry_sha256": "wrong"},
                       {"model_files_sha256": {}}, {"embeddings_sha256": {}}, {"hip_device": 1}):
            report = json.loads(original)
            report.update(change)
            self.report.write_text(json.dumps(report))
            with self.subTest(change=change), self.assertRaises(SystemExit):
                self.invoke(install=True)
            self.assertFalse(self.dropin.exists())

    def test_different_existing_dropin_is_not_overwritten(self):
        self.dropin.parent.mkdir()
        self.dropin.write_text("user-owned configuration")
        with self.assertRaises(SystemExit):
            self.invoke(install=True)
        self.assertEqual(self.dropin.read_text(), "user-owned configuration")
