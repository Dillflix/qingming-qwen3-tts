#!/usr/bin/env python3
"""Offline 1.7B Base voice enrollment/audition. Never changes services or HTTP routing."""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "qingming-base-xvector-v1"
BASE_TEXT = "Welcome to Dillflix. This is a short test of my new voice, with a steady pace and a natural finish."
SAMPLES = {
    "conversation": "Good morning! I have a few ideas for today. Let's take our time, compare the options, and choose the one that works best.",
    "technical": "The server returned error ten sixty-two because two rows had the same key. Remove the duplicate, then run the test again. Does the result look correct?",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    # All artifacts have new, private destinations. Do not overwrite a previous enrollment.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint_model(model_dir):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if (str(config.get("tts_model_size", "")).lower() not in ("1b7", "1.7b")
            or config.get("tts_model_type") != "base"):
        raise ValueError("Enrollment requires a 1.7B Base checkpoint, not CustomVoice")
    if not any(model_dir.glob("*.safetensors")):
        raise ValueError("Base model safetensors are missing")
    files = sorted(p for p in model_dir.rglob("*") if p.is_file()
                   and not any(part.startswith(".") for part in p.relative_to(model_dir).parts)
                   and (p.suffix in (".safetensors", ".json") or p.name == "merges.txt"))
    # Hash actual weights, not just config.json: unrelated checkpoints can share a config.
    return {p.relative_to(model_dir).as_posix(): sha256(p) for p in files}


def check_embedding(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(4097)
    if len(raw) != 4096:
        raise ValueError("Voice embedding must be exactly 2048 little-endian BF16 values (4096 bytes)")
    words = struct.unpack("<2048H", raw)
    if any(word & 0x7F80 == 0x7F80 for word in words) or not any(word & 0x7FFF for word in words):
        raise ValueError("Voice embedding contains non-finite values or is all zero")


def validate_reference(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 24000, "NONE"):
            raise ValueError("Reference must be mono PCM16 WAV at 24000 Hz")
        frames = wav.getnframes()
        if not 3 * 24000 <= frames <= 30 * 24000:
            raise ValueError("Select a 3..30-second reference; 10..20 seconds is a useful starting point")
        raw = wav.readframes(frames)
    if len(raw) != frames * 2:
        raise ValueError("Truncated reference WAV")
    samples = array("h", raw)
    if sys.byteorder != "little":
        samples.byteswap()
    peak = max(abs(x) for x in samples) / 32768
    rms = math.sqrt(sum(x * x for x in samples) / frames) / 32768
    clipping = sum(abs(x) >= 32760 for x in samples) / frames
    if peak < 0.005 or rms < 0.0005:
        raise ValueError("Reference is silent or too quiet; choose a clean speech recording")
    if clipping > 0.01:
        raise ValueError("More than 1% of reference samples clip; choose a cleaner recording")
    return {"duration_seconds": frames / 24000, "sample_rate": 24000, "channels": 1,
            "sample_format": "pcm_s16le", "peak": peak, "rms": rms, "clipped_fraction": clipping}


def normalize_reference(args, destination, log):
    reference = args.reference.resolve(strict=True)
    if not reference.is_file():
        raise ValueError("Reference must be a local audio file")
    command = [args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n"]
    if args.start_seconds:
        command += ["-ss", str(args.start_seconds)]
    # Limit protocols to local input; do not let a playlist fetch network references.
    command += ["-protocol_whitelist", "file,pipe", "-i", str(reference)]
    if args.duration_seconds is not None:
        command += ["-t", str(args.duration_seconds)]
    else:
        command += ["-t", "31"]  # Detect too-long input without decoding a whole recording.
    command += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
                "-map_metadata", "-1", str(destination)]
    with log.open("xb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    result.check_returncode()
    return validate_reference(destination)


def load_profile(directory, model_dir):
    directory = Path(directory).resolve(strict=True)
    profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
    if (profile.get("schema") != SCHEMA or profile.get("status") != "ENROLLED"
            or profile.get("method") != "x_vector_only" or profile.get("consent_confirmed") is not True):
        raise ValueError("Not a completed, authorized Base x-vector enrollment")
    # Fixed names only; never accept an arbitrary path supplied by profile metadata.
    for name in ("reference.wav", "speaker.bf16"):
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256(path) != profile.get("files", {}).get(name):
            raise ValueError(f"Voice profile integrity check failed: {name}")
    validate_reference(directory / "reference.wav")
    check_embedding(directory / "speaker.bf16")
    if profile.get("model_files_sha256") != fingerprint_model(model_dir):
        raise ValueError("Base checkpoint fingerprint changed; re-enroll/revalidate rather than silently reusing this voice")
    return profile


def native_wav(path, metadata, budget):
    # Native output is IEEE float32 mono, unlike the normalized PCM16 reference.
    raw = path.read_bytes()
    if (len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:16] != b"WAVEfmt " or raw[36:40] != b"data"
            or struct.unpack_from("<HHIIHH", raw, 20) != (3, 1, 24000, 96000, 4, 32)
            or struct.unpack_from("<I", raw, 4)[0] != len(raw) - 8
            or struct.unpack_from("<I", raw, 40)[0] != len(raw) - 44):
        raise ValueError("Malformed native float32 WAV")
    frames = metadata.get("frames")
    if (metadata.get("status") != "ok" or metadata.get("task") != "base-xvector"
            or metadata.get("eos") is not True or type(frames) is not int
            or not 0 < frames <= budget or len(raw) - 44 != frames * 1920 * 4):
        raise ValueError("Generation failed, hit its limit, or returned inconsistent frame metadata; not a complete sample")
    samples = struct.iter_unpack("<f", raw[44:])
    peak = 0.0
    for (value,) in samples:
        if not math.isfinite(value):
            raise ValueError("Native output contains non-finite audio")
        peak = max(peak, abs(value))
    if peak < 1e-5:
        raise ValueError("Native output is silent")
    return raw[44:]


def run_native(args, out, label, text, *, reference=None, embedding=None, export=None):
    text_path = out / f"{label}.txt"
    with text_path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    wav = out / f"{label}.wav"
    command = [str(args.binary), "--lifecycle", "once", "--task", "base-xvector", "--model-dir", str(args.model_dir),
               "--text-mode", "streaming", "--language", args.language, "--text-file", str(text_path),
               "--output", str(wav), "--seed", str(args.seed), "--max-new-tokens", str(args.max_new_tokens)]
    if reference is not None:
        command += ["--ref-audio", str(reference)]
    if embedding is not None:
        command += ["--speaker-embedding-bf16", str(embedding)]
    if export is not None:
        command += ["--save-speaker-embedding-bf16", str(export)]
    environment = os.environ.copy()
    environment["HIP_VISIBLE_DEVICES"] = str(args.hip_device)
    print(f"Generating {label} (one Base process; timeout {args.timeout}s)...", flush=True)
    started = time.monotonic()
    with (out / f"{label}.log").open("xb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=args.timeout)
    wall = time.monotonic() - started
    result.check_returncode()
    metadata = json.loads((out / f"{label}.log").read_text(encoding="utf-8"))
    pcm = native_wav(wav, metadata, args.max_new_tokens)
    duration = len(pcm) / (24000 * 4)
    record = {"native": metadata, "wav_sha256": sha256(wav), "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
              "audio_duration_seconds": duration, "wall_seconds_including_model_load": wall,
              "wall_rtf_including_model_load": wall / duration}
    print(f"{label}: {duration:.2f}s audio; {wall:.2f}s including model load; EOS confirmed", flush=True)
    return record


def enroll(args, out, report):
    print("Fingerprinting the Base checkpoint (including weights)...", flush=True)
    model = fingerprint_model(args.model_dir)
    args.library.mkdir(parents=True, exist_ok=True)
    directory = args.library / args.voice_id
    directory.mkdir(mode=0o700)  # Refuse reuse/overwrite, including an incomplete attempt.
    report["profile_directory"] = str(directory)
    reference = directory / "reference.wav"
    embedding = directory / "speaker.bf16"
    ref_stats = normalize_reference(args, reference, out / "normalize.log")
    report["reference"] = ref_stats
    cases = report["cases"]
    cases["reference"] = run_native(args, out, "reference", BASE_TEXT, reference=reference, export=embedding)
    check_embedding(embedding)
    cases["saved-voice"] = run_native(args, out, "saved-voice", BASE_TEXT, embedding=embedding)
    if cases["reference"]["pcm_sha256"] != cases["saved-voice"]["pcm_sha256"]:
        raise RuntimeError("Saved embedding did not reproduce reference-conditioned PCM; enrollment withheld")
    report["reference_vs_saved_pcm_match"] = True
    for label, text in SAMPLES.items():
        cases[label] = run_native(args, out, label, text, embedding=embedding)
    profile = {"schema": SCHEMA, "status": "ENROLLED", "voice_id": args.voice_id, "method": "x_vector_only",
               "consent_confirmed": True, "created_unix": int(time.time()), "language": args.language,
               "reference": ref_stats, "reference_selection": {"start_seconds": args.start_seconds,
               "duration_seconds": args.duration_seconds}, "embedding": {"dtype": "bf16-le", "dimension": 2048},
               "files": {p.name: sha256(p) for p in (reference, embedding)},
               "model_files_sha256": model, "enrollment_binary_sha256": report["binary_sha256"],
               "validation": {"reference_vs_saved_pcm_match": True, "natural_eos_all_cases": True,
               "listening_approved": False, "production_qualified": False}}
    # This completion marker is written LAST. Partial attempts cannot be loaded as voices.
    write_json(directory / "profile.json", profile)
    report["status"] = "PASS"
    report["note"] = "Saved voice is enrolled, but similarity/pronunciation need your listening review. No production changes."


def audition(args, out, report):
    print("Checking voice/profile and Base checkpoint fingerprints...", flush=True)
    profile = load_profile(args.profile, args.model_dir)
    if args.language is None:
        args.language = profile["language"]
    text = args.text_file.read_text(encoding="utf-8")
    if not text.strip() or "\0" in text or len(text) > 2000:
        raise ValueError("Audition text must contain 1..2000 characters and no NUL; use a short passage initially")
    report["voice_id"] = profile["voice_id"]
    report["binary_matches_enrollment"] = report["binary_sha256"] == profile["enrollment_binary_sha256"]
    report["cases"]["audition"] = run_native(args, out, "audition", text, embedding=args.profile / "speaker.bf16")
    report["status"] = "PASS"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("enroll", "audition"):
        command = commands.add_parser(name)
        command.add_argument("--binary", type=Path, default=ROOT / "build/base-clone-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b")
        command.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3-TTS-12Hz-1.7B-Base")
        command.add_argument("--hip-device", type=int, required=True, help="Physical HIP GPU index (RX 7900 XT), confirmed before testing")
        command.add_argument("--language", default="English" if name == "enroll" else None)
        command.add_argument("--seed", type=int, default=1234)
        command.add_argument("--max-new-tokens", type=int, default=512)
        command.add_argument("--timeout", type=int, default=180)
        command.add_argument("--results-root", type=Path, default=ROOT)
        command.add_argument("--archive", action="store_true", help="Opt in to a private results archive containing generated voice audio; do not publish it")
        if name == "enroll":
            command.add_argument("--voice-id", required=True)
            command.add_argument("--reference", type=Path, required=True)
            command.add_argument("--library", type=Path, default=ROOT / "voice-library")
            command.add_argument("--consent-confirmed", action="store_true", required=True, help="Confirm you own this voice or have the speaker's permission")
            command.add_argument("--start-seconds", type=float, default=0)
            command.add_argument("--duration-seconds", type=float)
            command.add_argument("--ffmpeg", default="ffmpeg")
        else:
            command.add_argument("--profile", type=Path, required=True)
            command.add_argument("--text-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.seed <= 4294967295 or args.hip_device < 0 or not 1 <= args.max_new_tokens <= 8192 or not 10 <= args.timeout <= 900:
        parser.error("invalid seed/device/budget/timeout")
    if args.command == "enroll":
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", args.voice_id):
            parser.error("voice-id must be a lowercase slug (1..48 characters), not a path")
        if (not math.isfinite(args.start_seconds) or args.start_seconds < 0
                or (args.duration_seconds is not None and not 3 <= args.duration_seconds <= 30)):
            parser.error("start-seconds must be nonnegative and duration-seconds 3..30")
    for name in ("binary", "model_dir", "library", "profile", "text_file", "results_root"):
        if hasattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    args.results_root.mkdir(parents=True, exist_ok=True)
    old_umask = os.umask(0o077)
    try:
        out = Path(tempfile.mkdtemp(prefix="benchmark-base-clone.", dir=args.results_root))
    except BaseException:
        os.umask(old_umask)
        raise
    print(f"Private results: {out}", flush=True)
    report = {"status": "FAIL", "errors": [], "cases": {}, "workflow": args.command,
              "hip_device_override": args.hip_device, "scope": "Offline once-path tests only; not resident, streaming, HTTP, concurrency or production qualification"}
    try:
        report["binary_sha256"] = sha256(args.binary)
        (enroll if args.command == "enroll" else audition)(args, out, report)
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "FAIL"
        report["errors"].append(f"{type(error).__name__}: {error}")
        print(f"ERROR: {error}. See private logs; no automatic retry.", file=sys.stderr, flush=True)
    finally:
        try:
            write_json(out / "report.json", report)
            if args.archive:
                archive = out.with_suffix(out.suffix + ".tar.gz")
                with tarfile.open(archive, "x:gz") as bundle:
                    bundle.add(out, arcname=out.name)
                archive.with_name(archive.name + ".sha256").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
                print(f"Private archive (generated voice audio): {archive}", flush=True)
        finally:
            os.umask(old_umask)
    print(f"Status: {report['status']}. Results: {out}\nNo services were started, stopped, or reconfigured.", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
