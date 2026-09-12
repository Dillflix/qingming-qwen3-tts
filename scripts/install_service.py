#!/usr/bin/env python3
"""Install only Qingming's unit/config. Never start/stop services or touch the LLM."""
import argparse
from datetime import datetime, timezone
import getpass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qingming_api.config import read_key, validate_key

CONFIG_DIR = Path("/etc/qingming-tts")
UNIT = Path("/etc/systemd/system/qingming-tts.service")


def quote(value):
    value = str(value)
    if any(ord(c) < 32 for c in value):
        raise ValueError("Control characters are not allowed in service configuration")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def render_unit(root, user):
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*\$?", user):
        raise ValueError("Invalid service user")
    # ExecStart additionally performs dollar expansion; there is no shell.
    executable = quote(str(root) + "/.venv/bin/python").replace("$", "$$")
    # Unlike ExecStart, WorkingDirectory takes the whole path, not a quoted word.
    working = str(root)
    if not working.startswith("/") or "\\" in working or working != working.strip():
        raise ValueError("The checkout needs an absolute Linux path without backslashes or edge whitespace")
    working = working.replace("%", "%%")
    return f"""[Unit]
Description=Qingming CustomVoice speech API (RX 7900 XT)
After=network.target
StartLimitIntervalSec=120
StartLimitBurst=10

[Service]
Type=simple
User={user}
WorkingDirectory={working}
EnvironmentFile=/etc/qingming-tts/environment
LoadCredential=api-key:/etc/qingming-tts/api-key
ExecStart={executable} -m qingming_api --api-key-file %d/api-key
Restart=on-failure
RestartSec=5
KillMode=control-group
TimeoutStopSec=30
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
LimitCORE=0
StandardOutput=journal
StandardError=journal
SyslogIdentifier=qingming-tts

[Install]
WantedBy=multi-user.target
"""


def render_environment(host, port, hip_device):
    if not re.fullmatch(r"[a-zA-Z0-9.:-]+", host) or not 1 <= port <= 65535:
        raise ValueError("Invalid host or port")
    if hip_device is not None and hip_device < 0:
        raise ValueError("HIP device index must be nonnegative")
    # EnvironmentFile does not interpret systemd specifiers or shell substitutions.
    text = f"QINGMING_HOST={host}\nQINGMING_PORT={port}\nPYTHONUNBUFFERED=1\n"
    if hip_device is not None:
        text += f"QINGMING_HIP_DEVICE={hip_device}\n"
    return text


def atomic_write(path, content, mode):
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            target.write(content)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Existing non-root Linux user that ran the HIP baseline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8880)
    parser.add_argument("--hip-device", type=int)
    parser.add_argument("--api-key-file", type=Path, help="Copy a key on first install; otherwise prompt without echo")
    parser.add_argument("--install", action="store_true", help="Write files and daemon-reload; default only prints nonsecret configuration")
    args = parser.parse_args()
    unit = render_unit(ROOT, args.user)
    environment = render_environment(args.host, args.port, args.hip_device)
    if not args.install:
        print(unit)
        print("# Initial /etc/qingming-tts/environment (preserved on reinstall):")
        print(environment)
        return
    if sys.platform != "linux" or os.geteuid() != 0:
        parser.error("Installation requires Linux and sudo")
    import pwd
    account = pwd.getpwnam(args.user)
    if account.pw_uid == 0:
        parser.error("Run the API as an existing non-root user")
    for path in (ROOT / ".venv/bin/python", ROOT / "build/rx7900xtx-24g-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b",
                 ROOT / "models/Qwen3-TTS-12Hz-1.7B-CustomVoice/config.json"):
        if not path.is_file():
            parser.error(f"Required file is missing: {path}")
    if not all(shutil.which(name) for name in ("ffmpeg", "systemctl", "systemd-analyze")):
        parser.error("Install FFmpeg and systemd (including systemd-analyze) before installing the service")
    # Let the host's actual parser reject invalid directives before touching /etc.
    with tempfile.TemporaryDirectory(prefix="qingming-unit-check.") as directory:
        candidate = Path(directory) / UNIT.name
        candidate.write_text(unit, encoding="utf-8")
        subprocess.run(["systemd-analyze", "verify", "--man=no", str(candidate)], check=True)
    if CONFIG_DIR.is_symlink() or any(p.is_symlink() for p in (UNIT, CONFIG_DIR / "environment", CONFIG_DIR / "api-key")):
        parser.error("Refusing symlinked service configuration")
    CONFIG_DIR.mkdir(mode=0o700, parents=False, exist_ok=True)
    os.chmod(CONFIG_DIR, 0o700)
    key_path = CONFIG_DIR / "api-key"
    if key_path.exists():
        read_key(key_path)  # Fail closed on empty/invalid existing credentials.
        if args.api_key_file and read_key(args.api_key_file) != read_key(key_path):
            parser.error("Existing API key differs; refusing implicit credential rotation")
        os.chmod(key_path, 0o600)
        print("Preserving existing API key.")
    else:
        key = read_key(args.api_key_file) if args.api_key_file else validate_key(getpass.getpass("New Qingming backend API key (hidden): "))
        if not args.api_key_file and key != getpass.getpass("Confirm backend API key (hidden): "):
            parser.error("Keys do not match")
        atomic_write(key_path, key + "\n", 0o600)
    config = CONFIG_DIR / "environment"
    if not config.exists():
        atomic_write(config, environment, 0o600)
    else:
        print("Preserving /etc/qingming-tts/environment; edit it explicitly to change bind/device settings.")
    if UNIT.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = CONFIG_DIR / ("qingming-tts.service." + stamp + ".bak")
        shutil.copy2(UNIT, backup)
        print(f"Previous unit saved to {backup}")
    atomic_write(UNIT, unit, 0o644)
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    print("Installed qingming-tts.service. No service was started, stopped, or restarted.")
    print("Stop only the previous foreground TTS instance, then: sudo systemctl enable --now qingming-tts.service")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        # Key validation errors never contain the credential value.
        raise SystemExit(f"Installation failed: {error}")
