import asyncio
from contextlib import asynccontextmanager
import hmac
import io
import json
import logging
from pathlib import Path
import wave

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from .audio import MEDIA_TYPES, encode
from .contract import APIError, CLONE_ERROR, SpeechRequest, VoiceCatalog, split_text
from .supervisor import WorkerSupervisor

LOG = logging.getLogger("qingming.api")
MAX_BODY = 65536
MAX_AUDIO_BYTES = 128 * 1024 * 1024


class OwnedStream(StreamingResponse):
    def __init__(self, source, cleanup, **kwargs):
        super().__init__(source, **kwargs)
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await asyncio.shield(self.cleanup())


def create_app(model_dir, worker, *, aliases=None, api_key=None, ffmpeg="ffmpeg", queue_timeout=30,
               supervisor_options=None):
    catalog = VoiceCatalog(Path(model_dir), aliases)
    lock = asyncio.Lock()
    waiting = 0

    supervisor = WorkerSupervisor(worker, lock, **(supervisor_options or {}))

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(supervisor.run())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    app = FastAPI(title="Qingming CustomVoice", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.worker = worker
    app.state.supervisor = supervisor

    @app.exception_handler(APIError)
    async def api_error_handler(request, error):
        return JSONResponse(error.body, status_code=error.status)

    @app.middleware("http")
    async def authenticate(request, call_next):
        # Liveness is public and contains no model, prompt or credential details.
        if api_key and request.url.path != "/healthz":
            supplied = request.headers.get("authorization", "")
            if not hmac.compare_digest(supplied.encode(), ("Bearer " + api_key).encode()):
                error = APIError("Invalid API key", code="invalid_api_key", status=401)
                return JSONResponse(error.body, status_code=401)
        return await call_next(request)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        return JSONResponse({"ready": worker.ready, "task": "custom-voice"}, status_code=200 if worker.ready else 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": name, "object": "model", "created": 0,
                 "owned_by": "qingming", "language": language} for name, language in catalog.models.items()]}

    @app.get("/v1/voices")
    @app.get("/v1/audio/voices")
    async def voices():
        return catalog.voices()

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        nonlocal waiting
        data = bytearray()
        async for part in request.stream():
            data.extend(part)
            if len(data) > MAX_BODY:
                raise APIError("Request body exceeds 64 KiB", code="request_too_large", status=413)
        try:
            body = json.loads(data)
            if isinstance(body, dict) and isinstance(body.get("voice"), str) and body["voice"].casefold().startswith("clone:"):
                raise APIError(CLONE_ERROR, "voice", "voice_cloning_not_supported")
            value = SpeechRequest.model_validate(body)
        except (ValueError, UnicodeError) as error:
            param = None
            if isinstance(error, ValidationError):
                param = ".".join(map(str, error.errors()[0]["loc"]))
            raise APIError("Invalid speech request", param, "invalid_request") from error
        speaker, language, instruct = catalog.resolve(value)
        if not worker.ready:
            raise APIError("CustomVoice worker is not ready; inspect the service log", code="engine_not_ready", status=503)
        if waiting >= 8:
            raise APIError("The speech queue is full", code="queue_full", status=429)
        waiting += 1
        try:
            try:
                await asyncio.wait_for(lock.acquire(), queue_timeout)
            except asyncio.TimeoutError as error:
                raise APIError("Timed out waiting for the speech worker", code="queue_timeout", status=429) from error
        finally:
            waiting -= 1
        # A queued request may have arrived before its predecessor killed the worker.
        if not worker.ready:
            lock.release()
            raise APIError("CustomVoice worker is recovering", code="engine_not_ready", status=503)
        released = False
        cleanup_task = None

        async def native_chunks():
            for index, text in enumerate(split_text(value.input)):
                if index:
                    yield b"\0" * (24000 * 120 // 1000 * 4)
                source = worker.generate(text, speaker, language, instruct)
                try:
                    async for chunk in source:
                        yield chunk
                finally:
                    await source.aclose()

        # A seekable WAV header is written after collecting PCM, not an unknown-size
        # FFmpeg pipe header. All other non-streaming formats are encoded directly.
        target_format = "pcm" if value.response_format == "wav" else value.response_format
        encoded = encode(native_chunks(), target_format, value.speed, ffmpeg)

        async def finish_cleanup():
            nonlocal released
            try:
                await encoded.aclose()
            finally:
                if not released:
                    lock.release()
                    released = True

        async def cleanup():
            nonlocal cleanup_task
            # Stream finalization and ASGI disconnect may both reach here. Keep
            # cleanup alive through cancellation and never close a generator twice
            # concurrently or release its lock before the process has been reaped.
            if cleanup_task is None:
                cleanup_task = asyncio.create_task(finish_cleanup())
            await asyncio.shield(cleanup_task)

        try:
            # Verify there is real audio before sending HTTP 200/stream headers.
            first = await anext(encoded)
            if value.stream:
                async def stream():
                    try:
                        total = len(first)
                        yield first
                        async for chunk in encoded:
                            total += len(chunk)
                            if total > MAX_AUDIO_BYTES:
                                raise RuntimeError("Audio response exceeds 128 MiB")
                            yield chunk
                    finally:
                        await cleanup()
                return OwnedStream(stream(), cleanup, media_type=MEDIA_TYPES["pcm"],
                                   headers={"X-Audio-Sample-Rate": "24000", "X-Audio-Sample-Format": "s16le",
                                            "X-Audio-Channels": "1", "Cache-Control": "no-store"})
            content = bytearray(first)
            async for chunk in encoded:
                content.extend(chunk)
                if len(content) > MAX_AUDIO_BYTES:
                    raise RuntimeError("Audio response exceeds 128 MiB")
            if value.response_format == "wav":
                output = io.BytesIO()
                with wave.open(output, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(content)
                content = output.getvalue()
            return Response(bytes(content), media_type=MEDIA_TYPES[value.response_format], headers={"Cache-Control": "no-store"})
        except Exception as error:
            LOG.exception("Speech generation failed")
            await cleanup()
            raise APIError("Speech generation failed; inspect the service log", code="generation_failed", status=502) from error
        except BaseException:
            await asyncio.shield(cleanup())
            raise
        finally:
            if not value.stream:
                await cleanup()

    return app
