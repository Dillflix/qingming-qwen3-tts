#!/usr/bin/env python3
"""Authenticated local Base readiness, without printing or passing keys as arguments."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qingming_api.config import read_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key-file", type=Path, default=Path("/etc/qingming-tts/api-key"))
    parser.add_argument("--port", type=int, default=8880)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    key = read_key(args.api_key_file)
    request = urllib.request.Request(f"http://127.0.0.1:{args.port}/readyz", headers={"Authorization": "Bearer " + key})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with opener.open(request, timeout=5) as response:
                value = json.load(response)
                if value == {"ready": True, "task": "base-xvector"}:
                    print("Base worker ready.")
                    return 0
                if value.get("task") != "base-xvector":
                    raise RuntimeError("Service is not running Base")
        except urllib.error.HTTPError as error:
            if error.code != 503:
                raise RuntimeError(f"Readiness returned HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError("Base did not become ready within 180 seconds")


if __name__ == "__main__":
    raise SystemExit(main())
