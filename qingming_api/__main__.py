import json
import logging
import shutil

import uvicorn

from .app import create_app
from .config import parse_args
from .worker import NativeWorker


def main():
    args = parse_args()
    if not args.binary.is_file():
        raise SystemExit("Native binary not found; build the 1.7B HIP target first")
    if not shutil.which(args.ffmpeg):
        raise SystemExit("FFmpeg is required for MP3 and other encoded output formats")
    aliases = json.loads(args.voice_aliases.read_text(encoding="utf-8")) if args.voice_aliases else None
    if aliases is not None and not isinstance(aliases, dict):
        raise SystemExit("--voice-aliases must contain a JSON object")
    logging.basicConfig(level=logging.INFO)
    worker = NativeWorker(args.binary, args.model_dir, hip_device=args.hip_device)
    app = create_app(args.model_dir, worker, aliases=aliases, api_key=args.api_key, ffmpeg=args.ffmpeg)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, limit_concurrency=32, timeout_keep_alive=5)


if __name__ == "__main__":
    main()
