"""One owned Linux native process; no access to or restarts of other services."""
import asyncio
from contextlib import suppress
import json
import logging
import os
from pathlib import Path
import struct
import tempfile

LOG = logging.getLogger("qingming.worker")
HEADER = struct.Struct("<4sIQII")
MAX_PAYLOAD = 1024 * 1024


class NativeWorker:
    def __init__(self, binary, model_dir, *, hip_device=None, max_new_tokens=512,
                 startup_timeout=300, request_timeout=180):
        self.binary = Path(binary).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.hip_device = hip_device
        self.max_new_tokens = max_new_tokens
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.process = None
        self.transport = None
        self.ready = False
        self.error = None
        self.info = {}
        self.last_result = None
        self.counter = 0
        self.tasks = []

    async def start(self):
        if os.name != "posix":
            raise RuntimeError("The native ROCm worker requires Linux")
        read_fd, write_fd = os.pipe()
        env = os.environ.copy()
        if self.hip_device is not None:
            env["HIP_VISIBLE_DEVICES"] = str(self.hip_device)
        args = [str(self.binary), "--lifecycle", "resident", "--model-dir", str(self.model_dir),
                "--task", "custom-voice", "--text-mode", "streaming", "--max-new-tokens", str(self.max_new_tokens),
                "--resident-cu-partition", "none", "--protocol", "jsonl-v1", "--audio-fd", str(write_fd)]
        pipe = os.fdopen(read_fd, "rb", buffering=0)
        try:
            self.reader = asyncio.StreamReader(limit=MAX_PAYLOAD + HEADER.size)
            self.transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(self.reader), pipe)
            self.process = await asyncio.create_subprocess_exec(
                *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, pass_fds=(write_fd,), env=env,
            )
            self.tasks.append(asyncio.create_task(self._drain(self.process.stderr, "stderr")))
            line = await asyncio.wait_for(self.process.stdout.readline(), self.startup_timeout)
            self.info = json.loads(line)
            expected = {"event": "ready", "protocol": "jsonl-v1", "task": "custom-voice", "family": "1.7b",
                        "sample_rate": 24000, "channels": 1, "sample_format": "f32le", "cu_partition": "none"}
            if any(self.info.get(k) != v for k, v in expected.items()):
                raise RuntimeError(f"Native startup did not report the required CustomVoice protocol: {self.info}")
            self.ready = True
            self.tasks.append(asyncio.create_task(self._drain(self.process.stdout, "stdout")))
            self.tasks.append(asyncio.create_task(self._watch()))
            LOG.info("CustomVoice worker ready: %s", self.info)
        except BaseException:
            await self.close()
            raise
        finally:
            os.close(write_fd)
            if self.transport is None:
                pipe.close()

    async def _drain(self, stream, label):
        while line := await stream.readline():
            LOG.info("native %s: %s", label, line.decode("utf-8", "replace").rstrip())

    async def _watch(self):
        rc = await self.process.wait()
        self.ready = False
        self.error = f"Native worker exited with status {rc}; restart the API after inspecting its log"
        LOG.error(self.error)

    async def generate(self, text, speaker, language, instruct, *, seed=1234, retain_dir=None):
        """Caller holds the single-worker lock through the whole HTTP response."""
        if not self.ready or self.process.returncode is not None:
            raise RuntimeError(self.error or "Native worker is not ready")
        self.counter += 1
        request_id = self.counter
        complete = False
        temporary = None
        if retain_dir is None:
            temporary = tempfile.TemporaryDirectory(prefix="qingming-audio-")
            directory = Path(temporary.name)
        else:
            directory = Path(retain_dir)
            directory.mkdir(parents=True, exist_ok=True)
        try:
            self.last_result = None
            body = {"request_id": request_id, "text": text, "speaker": speaker, "language": language,
                    "instruct": instruct, "seed": seed, "max_new_tokens": self.max_new_tokens,
                    "output": str(directory / f"{request_id}.wav"), "stream_audio": True}
            # The resident parser accepts literal UTF-8; never turn it into \u escapes.
            data = (json.dumps(body, ensure_ascii=False) + "\n").encode("utf-8")
            if len(data) > 65536:
                raise ValueError("Native request exceeds 64 KiB")
            deadline = asyncio.get_running_loop().time() + self.request_timeout
            async def bounded(awaitable):
                return await asyncio.wait_for(awaitable, max(0.001, deadline - asyncio.get_running_loop().time()))

            self.process.stdin.write(data)
            await bounded(self.process.stdin.drain())
            samples = 0
            while True:
                raw = await bounded(self.reader.readexactly(HEADER.size))
                magic, kind, received_id, length, reserved = HEADER.unpack(raw)
                if magic != b"QAF1" or received_id != request_id or reserved or length > MAX_PAYLOAD:
                    raise RuntimeError("Invalid or misrouted native audio frame")
                payload = await bounded(self.reader.readexactly(length))
                if kind == 1:
                    if not length or length % 4:
                        raise RuntimeError("Invalid native float32 chunk length")
                    samples += length // 4
                    yield payload
                elif kind in (2, 3):
                    event = json.loads(payload)
                    if event.get("request_id") != request_id:
                        raise RuntimeError("Native completion has the wrong request ID")
                    if kind == 3:
                        raise RuntimeError(event.get("error", "Native generation failed"))
                    result = event.get("result", {})
                    if event.get("event") != "completed" or result.get("status") != "ok" or not result.get("eos"):
                        raise RuntimeError("Native generation failed or reached its token limit before EOS")
                    if samples == 0 or samples != result.get("frames", -1) * 1920:
                        raise RuntimeError("Native audio length disagrees with its codec frame count")
                    self.last_result = result
                    complete = True
                    return
                else:
                    raise RuntimeError("Unknown native audio frame kind")
        finally:
            if not complete:
                self.error = "Generation failed or was interrupted; restart the API worker"
                await self.close()
            if temporary is not None:
                temporary.cleanup()

    async def close(self):
        self.ready = False
        if self.process and self.process.returncode is None:
            with suppress(ProcessLookupError):
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except asyncio.TimeoutError:
                with suppress(ProcessLookupError):
                    self.process.kill()
                await self.process.wait()
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        if self.transport:
            self.transport.close()
            self.transport = None
