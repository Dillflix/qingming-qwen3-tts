"""CLI/environment configuration; never print credentials in validation errors."""
import argparse
import os
from pathlib import Path


def validate_key(value):
    if not value or len(value) > 4096 or any(not 33 <= ord(c) <= 126 for c in value):
        raise ValueError("API key must be a nonempty, single ASCII token (no whitespace)")
    return value


def read_key(path):
    with Path(path).open(encoding="utf-8") as source:
        value = source.read(4098).rstrip("\r\n")
    return validate_key(value)


def parse_args(argv=None, environ=None):
    env = os.environ if environ is None else environ
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="CustomVoice-only API; one supervised gfx1100 worker")
    parser.add_argument("--binary", type=Path, default=env.get("QINGMING_BINARY", str(root / "build/rx7900xtx-24g-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b")))
    parser.add_argument("--model-dir", type=Path, default=env.get("QINGMING_MODEL_DIR", str(root / "models/Qwen3-TTS-12Hz-1.7B-CustomVoice")))
    parser.add_argument("--host", default=env.get("QINGMING_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=env.get("QINGMING_PORT", "8880"))
    parser.add_argument("--api-key", default=env.get("QINGMING_API_KEY"), help="Prefer environment or --api-key-file to avoid process-list exposure")
    parser.add_argument("--api-key-file", type=Path, default=env.get("QINGMING_API_KEY_FILE"))
    parser.add_argument("--hip-device", type=int, default=env.get("QINGMING_HIP_DEVICE"))
    parser.add_argument("--voice-aliases", type=Path, default=env.get("QINGMING_VOICE_ALIASES"))
    parser.add_argument("--ffmpeg", default=env.get("QINGMING_FFMPEG", "ffmpeg"))
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or (args.hip_device is not None and args.hip_device < 0):
        parser.error("Port must be 1..65535 and HIP device index must be nonnegative")
    if not args.host or any(c.isspace() or ord(c) < 32 for c in args.host):
        parser.error("Invalid listen address")
    if args.api_key is not None and args.api_key_file is not None:
        parser.error("Choose one API key source: argument/environment OR file")
    try:
        if args.api_key_file is not None:
            args.api_key = read_key(args.api_key_file)
        elif args.api_key is not None:
            args.api_key = validate_key(args.api_key)
    except (ValueError, OSError, UnicodeError):
        parser.error("API key is empty, invalid, or its file is unreadable; credential contents are not logged")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.api_key:
        parser.error("Set QINGMING_API_KEY or --api-key-file before binding a non-loopback address")
    return args
