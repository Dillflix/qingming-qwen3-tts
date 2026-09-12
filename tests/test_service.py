import asyncio
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import ExitStack, redirect_stdout

import httpx

from qingming_api.app import create_app
from qingming_api.config import parse_args, read_key
from qingming_api.supervisor import WorkerSupervisor
from qingming_api.worker import NativeWorker
from scripts.install_service import render_unit, render_environment
from scripts import install_service
from test_api import CONFIG, FakeWorker, FLOATS


async def eventually(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(wait(), 2)


class ConfigurationTests(unittest.TestCase):
    def test_environment_and_explicit_overrides(self):
        args = parse_args(["--port", "8881"], {"QINGMING_API_KEY": "test-key", "QINGMING_HOST": "0.0.0.0",
                          "QINGMING_PORT": "8880", "QINGMING_HIP_DEVICE": "0"})
        self.assertEqual((args.host, args.port, args.hip_device), ("0.0.0.0", 8881, 0))
        self.assertIsInstance(args.binary, Path)
        self.assertIsInstance(args.model_dir, Path)

    def test_file_credential_and_fail_closed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key"
            path.write_text("private-test-token\n", encoding="utf-8")
            args = parse_args(["--host", "0.0.0.0", "--api-key-file", str(path)], {})
            self.assertEqual(args.api_key, "private-test-token")
            cases = [(["--host", "0.0.0.0"], {}), (["--api-key-file", str(path)], {"QINGMING_API_KEY": "secret"}),
                     (["--api-key-file", str(path / "missing")], {}), (["--port", "0"], {}),
                     (["--hip-device", "-1"], {}), ([], {"QINGMING_API_KEY": ""})]
            for argv, env in cases:
                with self.subTest(argv=argv), redirect_stderr(io.StringIO()) as errors:
                    with self.assertRaises(SystemExit):
                        parse_args(argv, env)
                    self.assertNotIn("private-test-token", errors.getvalue())
            for value in ("", "x" * 4097, "first\nsecond", "bad key", "你好"):
                path.write_text(value, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_key(path)

    def test_service_rendering_is_scoped_and_credentials_are_not_inline(self):
        unit = render_unit("/home/jdillman/qingming-qwen3-tts", "jdillman")
        self.assertIn("User=jdillman", unit)
        self.assertIn("WorkingDirectory=/home/jdillman/qingming-qwen3-tts\n", unit)
        self.assertIn("LoadCredential=api-key:/etc/qingming-tts/api-key", unit)
        self.assertIn("--api-key-file %d/api-key", unit)
        self.assertIn("KillMode=control-group", unit)
        self.assertNotIn("qwen-flash-next", unit)
        self.assertNotIn("QINGMING_API_KEY=", unit)
        self.assertNotIn("PrivateDevices", unit)  # ROCm needs /dev/kfd and render nodes.
        unit = render_unit('/home/a b/%s/$abc/"test', "jdillman")
        self.assertIn('%%s/$$abc/\\"test/.venv/bin/python"', unit)
        with self.assertRaises(ValueError):
            render_unit("/home/test\nExecStart=/bad", "user")
        with self.assertRaises(ValueError):
            render_unit("/home/test", "root\nExecStart=/bad")
        self.assertEqual(render_environment("0.0.0.0", 8880, 0),
                         "QINGMING_HOST=0.0.0.0\nQINGMING_PORT=8880\nPYTHONUNBUFFERED=1\nQINGMING_HIP_DEVICE=0\n")


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_failures_back_off_to_a_cap(self):
        worker = FakeWorker()
        worker.start = AsyncMock(side_effect=RuntimeError("startup failed"))
        supervisor = WorkerSupervisor(worker, asyncio.Lock(), initial_delay=1, max_delay=4)
        delays = []
        async def fake_sleep(delay):
            delays.append(delay)
            if len(delays) == 4:
                raise asyncio.CancelledError
        with patch("qingming_api.supervisor.asyncio.sleep", fake_sleep), \
             self.assertLogs("qingming.supervisor", level="WARNING"):
            with self.assertRaises(asyncio.CancelledError):
                await supervisor.run()
        self.assertEqual(delays, [1, 2, 4, 4])
        self.assertTrue(worker.closed)

    async def test_disconnected_stream_recovers_before_next_request(self):
        class InterruptedWorker(FakeWorker):
            attempts = 0
            async def generate(self, *args):
                self.attempts += 1
                if self.attempts > 1:
                    async for chunk in super().generate(*args):
                        yield chunk
                    return
                try:
                    yield FLOATS
                    await asyncio.Event().wait()
                finally:
                    await self.close()
        worker = InterruptedWorker()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
            app = create_app(directory, worker, supervisor_options={"poll_interval": 0.001, "initial_delay": 0.01})
            async with app.router.lifespan_context(app):
                await eventually(lambda: worker.ready)
                body = json.dumps({"input": "Interrupted.", "response_format": "pcm", "stream": True}).encode()
                sent = False
                disconnect = asyncio.Event()
                async def receive():
                    nonlocal sent
                    if not sent:
                        sent = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    await disconnect.wait()
                    return {"type": "http.disconnect"}
                async def send(event):
                    if event["type"] == "http.response.body" and event.get("body"):
                        disconnect.set()
                scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                         "method": "POST", "scheme": "http", "path": "/v1/audio/speech", "raw_path": b"/v1/audio/speech",
                         "query_string": b"", "headers": [(b"content-type", b"application/json")], "server": ("test", 80), "client": ("test", 123)}
                await asyncio.wait_for(app(scope, receive, send), 3)
                await eventually(lambda: worker.ready and app.state.supervisor.starts == 2)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    result = await client.post("/v1/audio/speech", json={"input": "Next.", "response_format": "pcm"})
                    self.assertEqual(result.status_code, 200)
                self.assertEqual(worker.attempts, 2)

    async def test_replacement_waits_for_request_cleanup_lock(self):
        worker = FakeWorker()
        lock = asyncio.Lock()
        supervisor = WorkerSupervisor(worker, lock, poll_interval=0.001, initial_delay=0.001)
        task = asyncio.create_task(supervisor.run())
        try:
            await eventually(lambda: worker.ready)
            async with lock:
                worker.ready = False
                await asyncio.sleep(0.02)
                self.assertEqual(supervisor.starts, 1)
            await eventually(lambda: supervisor.starts == 2 and worker.ready)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(worker.ready)
        self.assertTrue(worker.closed)
        self.assertEqual(supervisor.starts, 2)

    async def test_failed_startup_retries_and_shutdown_cancels_reload(self):
        worker = FakeWorker()
        worker.start = AsyncMock(side_effect=[RuntimeError("startup failed"), RuntimeError("failed again"), None])
        supervisor = WorkerSupervisor(worker, asyncio.Lock(), initial_delay=0.01, max_delay=0.02)
        task = asyncio.create_task(supervisor.run())
        await eventually(lambda: supervisor.starts >= 3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(worker.closed)
        self.assertFalse(worker.ready)
        worker.start = AsyncMock(side_effect=lambda: None)
        entered = asyncio.Event()
        async def held_start():
            entered.set()
            await asyncio.Event().wait()
        worker.start = held_start
        supervisor = WorkerSupervisor(worker, asyncio.Lock())
        task = asyncio.create_task(supervisor.run())
        await entered.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(worker.ready)

    async def test_http_failure_recovers_without_replaying_failed_text(self):
        class FailingWorker(FakeWorker):
            attempts = 0
            async def generate(self, *args):
                self.attempts += 1
                if self.attempts == 1:
                    await self.close()
                    raise RuntimeError("simulated native EOF")
                async for chunk in super().generate(*args):
                    yield chunk
        worker = FailingWorker()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
            app = create_app(directory, worker, api_key="test-key", supervisor_options={"poll_interval": 0.001, "initial_delay": 0.05})
            async with app.router.lifespan_context(app):
                await eventually(lambda: worker.ready)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                                             headers={"Authorization": "Bearer test-key"}) as client:
                    result = await client.post("/v1/audio/speech", json={"input": "Fails once.", "response_format": "pcm"})
                    self.assertEqual(result.status_code, 502)
                    self.assertEqual((await client.get("/readyz")).status_code, 503)
                    self.assertEqual((await client.get("/healthz")).status_code, 200)
                    await eventually(lambda: worker.ready)
                    result = await client.post("/v1/audio/speech", json={"input": "Next request.", "response_format": "pcm"})
                    self.assertEqual(result.status_code, 200)
                    self.assertEqual(worker.attempts, 2)
                    self.assertEqual([c[0] for c in worker.calls], ["Next request."])
                    # Idle exit is also recovered; not just request failures.
                    worker.ready = False
                    await eventually(lambda: worker.ready)
                    self.assertEqual(app.state.supervisor.starts, 3)
            self.assertFalse(worker.ready)

    async def test_queued_request_rechecks_ready_and_does_not_use_dead_worker(self):
        entered, fail = asyncio.Event(), asyncio.Event()
        class HeldWorker(FakeWorker):
            attempts = 0
            async def generate(self, *args):
                self.attempts += 1
                entered.set()
                await fail.wait()
                self.ready = False
                raise RuntimeError("worker exited")
                yield FLOATS
        worker = HeldWorker()
        worker.ready = True
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
            app = create_app(directory, worker)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                first = asyncio.create_task(client.post("/v1/audio/speech", json={"input": "First.", "response_format": "pcm"}))
                await entered.wait()
                second = asyncio.create_task(client.post("/v1/audio/speech", json={"input": "Second.", "response_format": "pcm"}))
                await asyncio.sleep(0.02)
                fail.set()
                responses = await asyncio.gather(first, second)
                self.assertEqual([r.status_code for r in responses], [502, 503])
                self.assertEqual(worker.attempts, 1)

    async def test_intentional_stop_not_reported_as_crash(self):
        worker = NativeWorker("unused", "unused")
        process = type("Process", (), {"wait": AsyncMock(return_value=-15)})()
        worker.process = process
        worker._closing = True
        with self.assertLogs("qingming.worker", level="INFO") as logs:
            await worker._watch(process)
        self.assertFalse(any("ERROR" in line for line in logs.output))
        worker._closing = False
        with self.assertLogs("qingming.worker", level="ERROR"):
            await worker._watch(process)
        self.assertIn("unexpectedly", worker.error)


class InstallerTests(unittest.TestCase):
    def test_installer_preserves_key_and_config_and_never_controls_services(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory) / "checkout"
            root.mkdir()
            etc = Path(directory) / "etc"
            etc.mkdir()
            unit = etc / "qingming-tts.service"
            settings = etc / "qingming-tts"
            for name in (".venv/bin/python", "build/rx7900xtx-24g-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b",
                         "models/Qwen3-TTS-12Hz-1.7B-CustomVoice/config.json"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            stack.enter_context(patch.object(install_service, "ROOT", root))
            stack.enter_context(patch.object(install_service, "CONFIG_DIR", settings))
            stack.enter_context(patch.object(install_service, "UNIT", unit))
            # The installer is Linux-only; emulate its OS/account/file boundary.
            stack.enter_context(patch.object(install_service.sys, "platform", "linux"))
            stack.enter_context(patch.object(install_service.os, "geteuid", return_value=0, create=True))
            stack.enter_context(patch.dict("sys.modules", {"pwd": SimpleNamespace(getpwnam=lambda user: SimpleNamespace(pw_uid=1000))}))
            stack.enter_context(patch.object(install_service.shutil, "which", return_value="/usr/bin/tool"))
            stack.enter_context(patch.object(install_service, "render_unit", side_effect=lambda root, user: render_unit("/home/jdillman/qingming-qwen3-tts", user)))
            written_modes = []
            def write(path, content, mode):
                written_modes.append((path, mode))
                path.write_text(content, encoding="utf-8")
            stack.enter_context(patch.object(install_service, "atomic_write", side_effect=write))
            commands = stack.enter_context(patch.object(install_service.subprocess, "run"))
            prompt = stack.enter_context(patch.object(install_service.getpass, "getpass", return_value="private-test-token"))
            stack.enter_context(patch.object(install_service.sys, "argv", ["installer", "--user", "jdillman", "--host", "0.0.0.0", "--install"]))
            with redirect_stdout(io.StringIO()) as output:
                install_service.main()
            self.assertNotIn("private-test-token", output.getvalue())
            self.assertEqual(read_key(settings / "api-key"), "private-test-token")
            self.assertIn((settings / "api-key", 0o600), written_modes)
            self.assertIn("QINGMING_HOST=0.0.0.0", (settings / "environment").read_text())
            settings.joinpath("environment").write_text("QINGMING_HOST=127.0.0.1\n", encoding="utf-8")
            prompt.side_effect = AssertionError("Reinstall must not prompt or rotate the key")
            with redirect_stdout(io.StringIO()):
                install_service.main()
            self.assertEqual(read_key(settings / "api-key"), "private-test-token")
            self.assertEqual((settings / "environment").read_text(), "QINGMING_HOST=127.0.0.1\n")
            self.assertEqual(len(list(settings.glob("*.bak"))), 1)
            calls = [call.args[0] for call in commands.call_args_list]
            self.assertEqual([c[:2] for c in calls], [["systemd-analyze", "verify"], ["systemctl", "daemon-reload"]] * 2)


if __name__ == "__main__":
    unittest.main()
