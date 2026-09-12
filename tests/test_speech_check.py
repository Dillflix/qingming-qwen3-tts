from contextlib import chdir, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import imageio_ffmpeg

from scripts import check_speech
from test_api import FLOATS


class SpeechCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cls.mp3 = subprocess.run([cls.ffmpeg, "-v", "error", "-f", "f32le", "-ar", "24000", "-ac", "1",
                                  "-i", "pipe:0", "-f", "mp3", "pipe:1"],
                                 input=FLOATS * 4, capture_output=True, check=True).stdout

    def exercise(self, status):
        calls = []
        payload = self.mp3
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                calls.append((self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.send_response(status)
                if status == 302:
                    self.send_header("Location", "/credential-must-not-follow")
                self.end_headers()
                self.wfile.write(payload if status == 200 else b"sensitive proxy error contents")
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory, chdir(directory), redirect_stdout(io.StringIO()) as output:
                with patch("sys.argv", ["check_speech", "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                                       "--model", "qingming-tts", "--ffmpeg", self.ffmpeg]), \
                     patch.dict("os.environ", {"QINGMING_TEST_API_KEY": "private-test-token"}):
                    code = check_speech.main()
                report_path, = Path(directory).glob("benchmark-customvoice-check.*/report.json")
                report = json.loads(report_path.read_text())
                self.assertNotIn("private-test-token", output.getvalue())
                self.assertNotIn("sensitive proxy error contents", output.getvalue())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "/v1/audio/speech")
        self.assertEqual(calls[0][1], "Bearer private-test-token")
        self.assertEqual(calls[0][2]["model"], "qingming-tts")
        self.assertEqual(calls[0][2]["voice"], "echo")
        return code, report

    def test_mp3_check_on_real_http_transport(self):
        code, report = self.exercise(200)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "PASS")
        self.assertGreater(report["audio_duration_s"], 0.2)

    def test_auth_failure_is_reported_without_proxy_error_body(self):
        code, report = self.exercise(401)
        self.assertEqual(code, 1)
        self.assertIn("HTTP 401", report["error"])

    def test_redirects_do_not_receive_credentials(self):
        code, report = self.exercise(302)
        self.assertEqual(code, 1)
        self.assertIn("HTTP 302", report["error"])


if __name__ == "__main__":
    unittest.main()
