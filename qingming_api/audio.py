import asyncio
from contextlib import suppress
import math
import struct


MEDIA_TYPES = {"mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac",
               "flac": "audio/flac", "wav": "audio/wav", "pcm": "application/octet-stream"}


def pcm16(chunk):
    if len(chunk) % 4:
        raise ValueError("Incomplete native float32 sample")
    out = bytearray()
    for (value,) in struct.iter_unpack("<f", chunk):
        if not math.isfinite(value):
            raise ValueError("Nonfinite native audio sample")
        out.extend(struct.pack("<h", max(-32768, min(32767, round(value * 32768)))))
    return bytes(out)


def tempo_filters(speed):
    values = []
    while speed < 0.5:
        values.append("atempo=0.5")
        speed /= 0.5
    while speed > 2:
        values.append("atempo=2")
        speed /= 2
    values.append(f"atempo={speed:g}")
    return ",".join(values)


def encoder_command(ffmpeg, format, speed):
    codec, container = {"mp3": ("libmp3lame", "mp3"), "opus": ("libopus", "ogg"),
                        "aac": ("aac", "adts"), "flac": ("flac", "flac"),
                        "wav": ("pcm_s16le", "wav"), "pcm": ("pcm_s16le", "s16le")}[format]
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-probesize", "32", "-analyzeduration", "0",
            "-f", "f32le", "-ar", "24000", "-ac", "1",
            "-i", "pipe:0", "-af", tempo_filters(speed), "-c:a", codec, "-f", container,
            "-flush_packets", "1", "pipe:1"]


async def encode(source, format, speed, ffmpeg="ffmpeg"):
    """Consume bounded native chunks; concurrently drain encoder output."""
    if format == "pcm" and speed == 1.0:
        try:
            async for chunk in source:
                yield pcm16(chunk)
        finally:
            await source.aclose()
        return
    process = None
    feed = None
    errors = None
    try:
        process = await asyncio.create_subprocess_exec(
            *encoder_command(ffmpeg, format, speed), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

        async def write_input():
            try:
                async for chunk in source:
                    process.stdin.write(chunk)
                    await process.stdin.drain()
            finally:
                process.stdin.close()
                with suppress(BrokenPipeError, ConnectionResetError):
                    await process.stdin.wait_closed()

        feed = asyncio.create_task(write_input())
        errors = asyncio.create_task(process.stderr.read())
        deadline = asyncio.get_running_loop().time() + 600
        async def bounded(awaitable):
            return await asyncio.wait_for(awaitable, max(0.001, deadline - asyncio.get_running_loop().time()))

        while chunk := await bounded(process.stdout.read(4096)):
            yield chunk
        await bounded(feed)
        rc = await bounded(process.wait())
        message = await bounded(errors)
        if rc:
            raise RuntimeError("Audio encoding failed: " + message.decode("utf-8", "replace")[:1000])
    finally:
        if feed:
            feed.cancel()
            await asyncio.gather(feed, return_exceptions=True)
        if process and process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
        if errors:
            errors.cancel()
            await asyncio.gather(errors, return_exceptions=True)
        await source.aclose()
