#!/usr/bin/env python3
"""Install a reversible Base-only drop-in for an EXISTING service; never restart it."""
import argparse
import json
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qingming_api.base_voices import BaseVoiceCatalog
from scripts.install_service import atomic_write, quote
from scripts.validate_base import identity

UNIT = Path("/etc/systemd/system/qingming-tts.service")
DROPIN = Path("/etc/systemd/system/qingming-tts.service.d/50-base-voices.conf")


def render(root, binary, model_dir, library, registry, hip_device):
    if hip_device < 0:
        raise ValueError("HIP device must be nonnegative")
    def argument(value):
        value = Path(value).resolve()
        return quote(value).replace("$", "$$")
    return ("[Unit]\nDescription=Qingming Base cloned-voice speech API (RX 7900 XT)\n\n[Service]\n"
            "UnsetEnvironment=QINGMING_VOICE_ALIASES QINGMING_VOICE_LIBRARY QINGMING_VOICE_REGISTRY\n"
            "ExecStart=\n"
            f"ExecStart={argument(root / '.venv/bin/python')} -m qingming_api --api-key-file %d/api-key "
            f"--task base-xvector --binary {argument(binary)} --model-dir {argument(model_dir)} "
            f"--voice-library {argument(library)} --voice-registry {argument(registry)} --hip-device {hip_device}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/base-production-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--voice-library", type=Path, default=ROOT / "voice-library")
    parser.add_argument("--voice-registry", type=Path, default=ROOT / "voice-library/production.json")
    parser.add_argument("--hip-device", type=int, required=True)
    parser.add_argument("--validation-report", type=Path, required=True, help="Successful validate_base.py report for these exact weights, binary and registered voices")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    catalog = BaseVoiceCatalog(args.model_dir, args.voice_library, args.voice_registry)
    report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or any(report.get(k) != v for k, v in identity(args, catalog).items()):
        parser.error("Validation did not pass or binary/checkpoint/registry/voices changed; run validate_base.py again")
    for file in (args.binary, ROOT / ".venv/bin/python"):
        if not file.is_file():
            parser.error(f"Required file missing: {file}")
    content = render(ROOT, args.binary, args.model_dir, args.voice_library, args.voice_registry, args.hip_device)
    if not args.install:
        print(content)
        return 0
    if sys.platform != "linux" or os.geteuid() != 0:
        parser.error("Installation requires Linux and sudo")
    if not UNIT.is_file() or UNIT.is_symlink():
        parser.error("An existing regular qingming-tts.service is required; no new service will be created")
    if DROPIN.parent.is_symlink() or DROPIN.is_symlink():
        parser.error("Refusing symlinked drop-in directory/file")
    if DROPIN.exists():
        if DROPIN.read_text(encoding="utf-8") != content:
            parser.error("A different Base drop-in exists; refusing overwrite. Back it up and review explicitly.")
        print("Identical Base drop-in already installed; no restart performed.")
        return 0
    DROPIN.parent.mkdir(mode=0o755, exist_ok=True)
    atomic_write(DROPIN, content, 0o644)
    try:
        subprocess.run(["systemd-analyze", "verify", "--man=no", str(UNIT)], check=True)
        subprocess.run(["systemctl", "daemon-reload"], check=True)
    except BaseException:
        DROPIN.unlink()  # Only the new, exact file created by this invocation.
        subprocess.run(["systemctl", "daemon-reload"], check=False)
        raise
    print(f"Installed {DROPIN}. No service was started, stopped, or restarted.")
    print("Old unit, environment, credentials, binary, checkpoint and profiles are preserved for rollback.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
