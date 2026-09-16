#!/usr/bin/env python3
"""Validate an explicit private voice registry and create it without overwriting one."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qingming_api.base_voices import BaseVoiceCatalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default", required=True)
    parser.add_argument("--voice", action="append", required=True, help="PublicName=private-profile-slug; repeat per voice")
    parser.add_argument("--library", type=Path, default=ROOT / "voice-library")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3-TTS-12Hz-1.7B-Base")
    args = parser.parse_args()
    voices = {}
    for entry in args.voice:
        name, separator, slug = entry.partition("=")
        if not separator or name in voices:
            parser.error("Use distinct PublicName=profile-slug entries")
        voices[name] = slug
    spec = {"schema": "qingming-base-registry-v1", "default": args.default, "voices": voices}
    target = args.library / "production.json"
    os.umask(0o077)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=args.library, delete=False, encoding="utf-8") as file:
        temporary = Path(file.name)
        json.dump(spec, file)
    try:
        catalog = BaseVoiceCatalog(args.model_dir, args.library, temporary)
        if target.exists():
            if target.is_symlink() or json.loads(target.read_text()) != spec:
                parser.error("Different production.json already exists; back it up and review changes explicitly")
        else:
            with target.open("x", encoding="utf-8") as output:
                json.dump(spec, output, indent=2)
                output.write("\n")
        print(f"Registry: {target}\nDefault: {catalog.default}\nVoices: {', '.join(catalog.embeddings)}")
    finally:
        temporary.unlink()


if __name__ == "__main__":
    main()
