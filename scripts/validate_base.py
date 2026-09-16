#!/usr/bin/env python3
"""Real HIP gate: all voices once/resident, A/B/A, stream/WAV, limit recovery and API audio."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
from qingming_api.app import create_app
from qingming_api.audio import pcm16
from qingming_api.base_voices import BaseVoiceCatalog
from qingming_api.worker import NativeWorker, GenerationLimitError
from scripts.clone_voice import BASE_TEXT, native_wav, run_native, sha256, write_json


def identity(args, catalog):
    return {"hip_device": args.hip_device, "binary_sha256": sha256(args.binary), "registry_sha256": sha256(args.voice_registry),
            "model_files_sha256": catalog.model_fingerprint,
            "embeddings_sha256": {name: hashlib.sha256(bytes.fromhex(value)).hexdigest()
                                  for name, value in catalog.embeddings.items()}}


async def validate(args, out, report):
    catalog = BaseVoiceCatalog(args.model_dir, args.voice_library, args.voice_registry)
    report.update(identity(args, catalog))
    report["cases"] = {}
    # Never load once controls while the resident model is alive.
    baseline = {}
    for index, (name, profile) in enumerate(catalog.profiles.items()):
        label = f"once-{index}"
        record = run_native(args, out, label, BASE_TEXT, embedding=profile / "speaker.bf16")
        baseline[name] = native_wav(out / f"{label}.wav", record["native"], args.max_new_tokens)
        report["cases"][label] = {"voice": name, **record}
    worker = NativeWorker(args.binary, args.model_dir, hip_device=args.hip_device,
                          task="base-xvector", voice_embeddings=catalog.embeddings)
    try:
        await worker.start()
        process = worker.process
        report["resident_ready"] = worker.info

        async def generate(name, label):
            start = time.monotonic()
            first, chunks = None, []
            print(f"Resident {label}: {name}", flush=True)
            async for chunk in worker.generate(BASE_TEXT, name, "English", "", retain_dir=out):
                if first is None:
                    first = time.monotonic() - start
                chunks.append(chunk)
            elapsed = time.monotonic() - start
            raw = b"".join(chunks)
            if worker.process is not process or not worker.ready:
                raise RuntimeError("Resident worker was replaced or became unavailable")
            if raw != baseline[name] or native_wav(Path(worker.last_result["output"]), worker.last_result, 512) != raw:
                raise RuntimeError(f"Once/resident/stream/WAV mismatch for {name}")
            if len(chunks) < 2 or first is None or first >= elapsed:
                raise RuntimeError("Early native streaming was not observed")
            report["cases"][label] = {"voice": name, "native": worker.last_result,
                                      "first_audio_seconds": first, "completed_seconds": elapsed,
                                      "pcm_matches_once": True, "chunks": len(chunks)}
            return raw

        for index, name in enumerate(catalog.embeddings):
            await generate(name, f"resident-{index}")
        first_voice = next(iter(catalog.embeddings))
        await generate(first_voice, "return-to-first-voice")

        print("Testing clean generation-limit recovery without a worker restart", flush=True)
        worker.max_new_tokens = 1
        try:
            _ = [part async for part in worker.generate(BASE_TEXT, first_voice, "English", "")]
        except GenerationLimitError:
            if not worker.ready or worker.process is not process:
                raise RuntimeError("Clean generation limit discarded the worker")
        else:
            raise RuntimeError("Limit fixture reached EOS; recovery was not exercised")
        finally:
            worker.max_new_tokens = 512
        await generate(first_voice, "after-limit")

        app = create_app(args.model_dir, worker, catalog=catalog, ffmpeg=args.ffmpeg)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://trial") as client:
            ready = await client.get("/readyz")
            if ready.status_code != 200 or ready.json()["task"] != "base-xvector":
                raise RuntimeError("Base readiness failed")
            for stream in (False, True):
                result = await client.post("/v1/audio/speech", json={"input": BASE_TEXT, "voice": first_voice,
                                           "language": "English", "response_format": "pcm", "stream": stream})
                if result.status_code != 200 or result.content != pcm16(baseline[first_voice]):
                    raise RuntimeError("API streamed/nonstreamed PCM mismatch")
            result = await client.post("/v1/audio/speech", json={"input": BASE_TEXT, "voice": first_voice, "language": "English"})
            if result.status_code != 200 or not result.content:
                raise RuntimeError("API default MP3 failed")
            (out / "api.mp3").write_bytes(result.content)
            subprocess.run([args.ffmpeg, "-v", "error", "-i", str(out / "api.mp3"), "-f", "null", "-"],
                           check=True, timeout=30, capture_output=True)
            rejected = await client.post("/v1/audio/speech", json={"input": "Hello.", "voice": "Aiden"})
            if rejected.status_code != 400:
                raise RuntimeError("Aiden must not silently become another speaker")
        if not worker.ready or worker.process is not process:
            raise RuntimeError("API checks replaced the worker")
        report["status"] = "PASS"
        report["scope"] = "All registered voices once/resident parity; A/B/A; native streaming; forced 1-frame recovery; in-process API PCM/MP3. Proxy/browser and endurance are separate."
    finally:
        await worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/base-production-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--voice-library", type=Path, default=ROOT / "voice-library")
    parser.add_argument("--voice-registry", type=Path, default=ROOT / "voice-library/production.json")
    parser.add_argument("--hip-device", type=int, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--output-dir", type=Path, help="An empty private results directory")
    args = parser.parse_args()
    args.seed, args.language, args.max_new_tokens, args.timeout = 1234, "English", 512, 180
    if args.hip_device < 0:
        parser.error("HIP device must be nonnegative")
    os.umask(0o077)
    if args.output_dir:
        out = args.output_dir.resolve()
        if not out.is_dir() or any(out.iterdir()):
            parser.error("--output-dir must be an existing empty directory")
    else:
        out = Path(tempfile.mkdtemp(prefix="benchmark-base-production.", dir=ROOT))
    print(f"Private results: {out}", flush=True)
    report = {"status": "FAIL", "errors": []}
    try:
        asyncio.run(validate(args, out, report))
    except (Exception, KeyboardInterrupt) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        print(f"ERROR: {error}", flush=True)
    finally:
        write_json(out / "report.json", report)
        print(f"Status: {report['status']}\nReport: {out / 'report.json'}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
