#!/usr/bin/env python3
"""Short native CLI/resident regression gate; always package logs, including failures."""
import argparse
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qingming_api.contract import VoiceCatalog
from qingming_api.worker import GenerationLimitError, NativeWorker

TEXT = "Welcome to Dillflix, mailboxhead!"
CALM = "Speak in a calm and confident tone."
CHEERFUL = "Speak cheerfully and enthusiastically."


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def wav_payload(path):
    raw = path.read_bytes()
    if (len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:16] != b"WAVEfmt " or raw[36:40] != b"data"
            or struct.unpack_from("<HHIIHH", raw, 20) != (3, 1, 24000, 96000, 4, 32)
            or struct.unpack_from("<I", raw, 4)[0] != len(raw) - 8
            or struct.unpack_from("<I", raw, 40)[0] != len(raw) - 44
            or len(raw) == 44 or (len(raw) - 44) % 4):
        raise ValueError(f"Unexpected or truncated native float32 WAV: {path}")
    return raw[44:]


async def validate(args, out, report):
    VoiceCatalog(args.model_dir)
    env = os.environ.copy()
    if args.hip_device is not None:
        env["HIP_VISIBLE_DEVICES"] = str(args.hip_device)
    report["binary_sha256"] = sha256(args.binary)
    report["model_config_sha256"] = sha256(args.model_dir / "config.json")
    report["hip_device_override"] = args.hip_device
    revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
    report["source_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    report["cases"] = {}
    for label in ("once-calm-1", "once-calm-2"):
        print(f"Starting {label}", flush=True)
        output = out / f"{label}.wav"
        command = [str(args.binary), "--lifecycle", "once", "--model-dir", str(args.model_dir),
                   "--task", "custom-voice", "--text-mode", "streaming", "--speaker", "Ryan",
                   "--language", "English", "--instruct", CALM, "--text", TEXT,
                   "--output", str(output), "--seed", "1234", "--max-new-tokens", "512"]
        with (out / f"{label}.log").open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=180)
        result.check_returncode()
        metadata = json.loads((out / f"{label}.log").read_text(encoding="utf-8"))
        if metadata.get("status") != "ok" or not metadata.get("eos"):
            raise RuntimeError(f"{label} failed or did not reach natural EOS")
        wav_payload(output)
        report["cases"][label] = {"sha256": sha256(output), "native": metadata}
    baseline = report["cases"]["once-calm-1"]["sha256"]
    if baseline != report["cases"]["once-calm-2"]["sha256"]:
        raise RuntimeError("Repeated single-shot calm WAVs differ")
    if args.expected_calm_sha256 and baseline != args.expected_calm_sha256:
        raise RuntimeError("Single-shot output differs from the supplied pre-change baseline hash")
    print("Single-shot repeatability and baseline checks passed", flush=True)

    worker = NativeWorker(args.binary, args.model_dir, hip_device=args.hip_device)
    try:
        print("Starting persistent CustomVoice worker (shared streams, no XTX partition)", flush=True)
        await worker.start()
        report["resident_ready"] = worker.info
        print(json.dumps(worker.info), flush=True)
        for label, instruction in (("resident-calm-1", CALM), ("resident-calm-2", CALM), ("resident-cheerful", CHEERFUL)):
            print(f"Starting {label}", flush=True)
            start = time.perf_counter()
            first = None
            chunks = []
            async for chunk in worker.generate(TEXT, "Ryan", "English", instruction, retain_dir=out):
                if first is None:
                    first = time.perf_counter() - start
                    print(f"First native audio received after {first:.3f}s", flush=True)
                chunks.append(chunk)
            completed = time.perf_counter() - start
            output = Path(worker.last_result["output"])
            raw = b"".join(chunks)
            if not chunks or wav_payload(output) != raw:
                raise RuntimeError(f"{label}: streamed native chunks do not match the accumulated WAV")
            if first is None or first >= completed:
                raise RuntimeError(f"{label}: audio was not observed before completion")
            digest = sha256(output)
            report["cases"][label] = {"sha256": digest, "chunk_count": len(chunks),
                "first_audio_received_s": first, "completion_received_s": completed,
                "native_chunks_match_wav": True, "native": worker.last_result}
            if "calm" in label and digest != baseline:
                raise RuntimeError(f"{label}: resident and single-shot calm WAVs differ")
        if report["cases"]["resident-cheerful"]["sha256"] == baseline:
            raise RuntimeError("Changing instruct produced an identical waveform")
        report["status"] = "PASS"
        report["scope"] = "Native once/resident repeatability, style sample, framed stream/WAV equality; HTTP and endurance remain separate checks"
    finally:
        await worker.close()


async def validate_limit_recovery(args, out, report):
    """Short forced-limit gate in one real worker; no production config changes."""
    VoiceCatalog(args.model_dir)
    report["binary_sha256"] = sha256(args.binary)
    report["model_config_sha256"] = sha256(args.model_dir / "config.json")
    report["cases"] = {}
    worker = NativeWorker(args.binary, args.model_dir, hip_device=args.hip_device)
    try:
        print("Starting dedicated limit-recovery worker (512-frame capacity)", flush=True)
        await worker.start()
        process = worker.process
        report["resident_ready"] = worker.info
        report["worker_pid"] = process.pid

        def require_same_worker():
            if worker.process is not process or not worker.ready or process.returncode is not None:
                raise RuntimeError("The native worker was stopped, replaced, or became unavailable")

        async def control(label):
            print(f"Starting {label}", flush=True)
            chunks = [chunk async for chunk in worker.generate(TEXT, "Ryan", "English", CALM, retain_dir=out)]
            require_same_worker()
            raw = b"".join(chunks)
            output = Path(worker.last_result["output"])
            if not raw or wav_payload(output) != raw:
                raise RuntimeError(f"{label}: native stream/WAV mismatch")
            report["cases"][label] = {"sha256": sha256(output), "native": worker.last_result}
            return raw

        baseline = await control("before-limit")
        if args.expected_calm_sha256 and report["cases"]["before-limit"]["sha256"] != args.expected_calm_sha256:
            raise RuntimeError("Pre-limit control differs from the supplied baseline hash")
        print("Forcing an 8-frame request limit; a GenerationLimitError is expected", flush=True)
        worker.max_new_tokens = 8  # Request budget only; resident capacity stays 512.
        chunks = []
        try:
            async for chunk in worker.generate(TEXT, "Ryan", "English", CALM, retain_dir=out):
                chunks.append(chunk)
        except GenerationLimitError as error:
            require_same_worker()
            raw = b"".join(chunks)
            output = out / f"{error.request_id}.wav"
            if error.frames != 8 or len(raw) != 8 * 1920 * 4 or wav_payload(output) != raw:
                raise RuntimeError("Forced-limit audio length/WAV mismatch")
            report["cases"]["forced-limit"] = {"status": "EXPECTED_LIMIT", "frames": error.frames,
                                                 "worker_retained": True, "sha256": sha256(output)}
        else:
            raise RuntimeError("The fixture reached EOS within 8 frames; the limit path was not exercised")
        finally:
            worker.max_new_tokens = 512
        for index in range(1, 6):
            if await control(f"after-limit-{index}") != baseline:
                raise RuntimeError(f"Post-limit control {index} differs from the pre-limit PCM")
        report["status"] = "PASS"
        report["scope"] = ("One forced 8-frame limit at 512-frame capacity; five identical natural-EOS controls "
                           "in the same HIP process. Not an HTTP/proxy, full-512-frame exhaustion, or endurance test.")
    finally:
        await worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/rx7900xtx-24g-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--hip-device", type=int)
    parser.add_argument("--expected-calm-sha256")
    parser.add_argument("--limit-recovery-only", action="store_true",
                        help="Run a short native before/forced-limit/five-after test instead of the full baseline suite")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.model_dir = args.model_dir.resolve()
    out = Path(tempfile.mkdtemp(prefix="benchmark-customvoice-api.", dir=ROOT))
    print(f"Results: {out}", flush=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(out / "resident.log", encoding="utf-8")])
    report = {"status": "FAIL", "errors": []}
    try:
        check = validate_limit_recovery if args.limit_recovery_only else validate
        asyncio.run(check(args, out, report))
    except BaseException as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logging.shutdown()
        archive = out.with_name(out.name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(out, arcname=out.name, recursive=True)
        archive.with_name(archive.name + ".sha256").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
        print(f"Status: {report['status']}\nArchive: {archive}\nChecksum: {archive}.sha256", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
