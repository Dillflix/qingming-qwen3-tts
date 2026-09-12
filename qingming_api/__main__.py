import argparse
import json
import logging
import os
from pathlib import Path
import shutil

import uvicorn

from .app import create_app
from .worker import NativeWorker


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="CustomVoice-only Qingming speech API; one persistent gfx1100 worker")
    parser.add_argument("--binary", type=Path, default=root / "build/rx7900xtx-24g-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b")
    parser.add_argument("--model-dir", type=Path, default=root / "models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8880)
    parser.add_argument("--api-key", default=os.environ.get("QINGMING_API_KEY"))
    parser.add_argument("--hip-device", type=int)
    parser.add_argument("--voice-aliases", type=Path, help="JSON object mapping public aliases to checkpoint speakers")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.api_key:
        parser.error("Set QINGMING_API_KEY or --api-key before binding a non-loopback address")
    if not args.binary.is_file():
        parser.error("Native binary not found; build the 1.7B HIP target first")
    if not shutil.which(args.ffmpeg):
        parser.error("FFmpeg is required for MP3 and other encoded output formats")
    aliases = json.loads(args.voice_aliases.read_text(encoding="utf-8")) if args.voice_aliases else None
    if aliases is not None and not isinstance(aliases, dict):
        parser.error("--voice-aliases must contain a JSON object")
    logging.basicConfig(level=logging.INFO)
    worker = NativeWorker(args.binary, args.model_dir, hip_device=args.hip_device)
    app = create_app(args.model_dir, worker, aliases=aliases, api_key=args.api_key, ffmpeg=args.ffmpeg)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, limit_concurrency=32, timeout_keep_alive=5)


if __name__ == "__main__":
    main()
