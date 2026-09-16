#!/usr/bin/env python3
"""MP3 transport check and listening fixture, direct or through LiteLLM. No service mutations."""
import argparse
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


FIXTURES = {
    "smoke": "Welcome to Dillflix, mailboxhead!",
    "narration": (
        "Welcome back to Dillflix. Tonight, we are taking a quiet walk through the history of cinema. "
        "A good story does not need to rush: it gives each character room to breathe, and lets each scene "
        "lead naturally into the next.\n"
        "Think of a small theater on a rainy evening, with the lights dimmed and the audience settling "
        "into their seats. Outside, the city carries on as usual. Inside, a single voice begins the story, "
        "steady and clear. By the time the final scene arrives, familiar details have taken on new meaning. "
        "That is the pleasure of watching closely, and of allowing a story to unfold at its own pace."
    ),
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    # Never forward a backend/proxy credential to a redirect target.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8880/v1")
    parser.add_argument("--model", default="tts-1")
    parser.add_argument("--voice", default="echo", help="Registered/preset voice; echo maps to Ryan on CustomVoice or the registry default on Base")
    parser.add_argument("--fixture", choices=FIXTURES, default="smoke",
                        help="narration exercises longer segments; listen for completeness and steady delivery")
    parser.add_argument("--api-key-env", default="QINGMING_TEST_API_KEY")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
        parser.error("Use an http(s) base URL without credentials, query, or fragment")
    if not shutil.which(args.ffmpeg):
        parser.error("FFmpeg is required to validate the returned MP3")
    key = args.api_key_file.read_text(encoding="utf-8").strip() if args.api_key_file else os.environ.get(args.api_key_env)
    if key is None:
        key = getpass.getpass("API key for this endpoint (hidden): ")
    if not key or any(not 33 <= ord(c) <= 126 for c in key):
        parser.error("A nonempty ASCII API key is required")
    body = json.dumps({"model": args.model, "voice": args.voice,
                       "input": FIXTURES[args.fixture], "response_format": "mp3"}).encode()
    request = urllib.request.Request(args.base_url.rstrip("/") + "/audio/speech", data=body,
                                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    output = Path(tempfile.mkdtemp(prefix="benchmark-customvoice-check.", dir=Path.cwd()))
    report = {"status": "FAIL", "model": args.model, "voice": args.voice,
              "fixture": args.fixture, "input_characters": len(FIXTURES[args.fixture]),
              "scope": "MP3 transport/decoding only; listen for complete text, pace and tone"}
    try:
        started = time.monotonic()
        with urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({})).open(request, timeout=120) as response:
            audio = response.read(8 * 1024 * 1024 + 1)
        elapsed = time.monotonic() - started
        if not 100 < len(audio) <= 8 * 1024 * 1024:
            raise RuntimeError("Empty, degenerate, or oversized audio response")
        if not (audio.startswith(b"ID3") or (audio[0] == 255 and audio[1] & 224 == 224)):
            raise RuntimeError("Response is not MP3")
        path = output / "speech.mp3"
        path.write_bytes(audio)
        decoded = subprocess.run([args.ffmpeg, "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", "24000", "-"],
                                 capture_output=True, timeout=30, check=True)
        if len(decoded.stdout) < 24000 // 5:
            raise RuntimeError("Decoded audio is too short")
        report.update(status="PASS", mp3_bytes=len(audio), response_complete_ms=round(elapsed * 1000, 2),
                      audio_duration_s=round(len(decoded.stdout) / 48000, 3))
    except urllib.error.HTTPError as error:
        report["error"] = f"HTTP {error.code}; check this endpoint's model mapping, credentials, readiness and logs"
    except Exception as error:
        # No raw proxy error body, URL, request headers, or key in the saved report.
        report["error"] = type(error).__name__ + "; inspect the service/proxy logs"
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Results: {output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
