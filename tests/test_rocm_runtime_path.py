"""Exercise real CMake configuration without compiling or requiring ROCm."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CMAKE = os.environ.get("QINGMING_TEST_CMAKE") or shutil.which("cmake")
NINJA = os.environ.get("QINGMING_TEST_NINJA") or shutil.which("ninja")


@unittest.skipUnless(CMAKE and NINJA, "CMake and Ninja are required")
class RocmRuntimePathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qingming-runtime-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rocm = self.root / "rocm with spaces"
        self.build = self.root / "build"

    def runtime(self, directory):
        library = self.rocm / directory / "libamdhip64.so"
        library.parent.mkdir(parents=True, exist_ok=True)
        library.touch()
        return library

    def configure(self, family="1.7b", device="rx7900xtx-24g"):
        # project(... NONE) only generates commands; these compilers are never run.
        return subprocess.run(
            [
                CMAKE, "-S", str(ROOT), "-B", str(self.build), "-G", "Ninja",
                f"-DCMAKE_MAKE_PROGRAM={NINJA}",
                f"-DQINGMING_DEVICE={device}",
                f"-DQINGMING_MODEL_FAMILY={family}",
                f"-DROCM_PATH={self.rocm.as_posix()}",
                f"-DHIP_CLANG={Path(sys.executable).as_posix()}",
                f"-DHOST_CXX={Path(sys.executable).as_posix()}",
                f"-DNVCC={Path(sys.executable).as_posix()}",
            ],
            capture_output=True, text=True, timeout=60,
        )

    def generated(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return (self.build / "build.ninja").read_text(encoding="utf-8")

    def assert_runtime_flag(self, generated, directory):
        flag = "-Wl,--enable-new-dtags,-rpath,"
        self.assertIn(flag + (self.rocm / directory).as_posix(), generated)
        # Only the HIP production binary needs this, not the host benchmark.
        self.assertEqual(generated.count(flag), 1)
        self.assertIn("--offload-arch=gfx1100", generated)

    def test_lib_and_lib64_for_both_model_families(self):
        for directory in ("lib", "lib64"):
            library = self.runtime(directory)
            for family in ("0.6b", "1.7b"):
                with self.subTest(directory=directory, family=family):
                    self.assert_runtime_flag(
                        self.generated(self.configure(family)), directory
                    )
            library.unlink()

    def test_lib_takes_precedence(self):
        self.runtime("lib")
        self.runtime("lib64")
        self.assert_runtime_flag(self.generated(self.configure()), "lib")

    def test_missing_runtime_fails_configuration(self):
        result = self.configure()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("libamdhip64.so not found under", result.stdout + result.stderr)

    def test_reconfigure_does_not_keep_stale_runtime_path(self):
        library = self.runtime("lib")
        self.assert_runtime_flag(self.generated(self.configure()), "lib")
        library.unlink()
        self.runtime("lib64")
        self.assert_runtime_flag(self.generated(self.configure()), "lib64")

    def test_cuda_does_not_require_or_link_hip_runtime(self):
        for family in ("0.6b", "1.7b"):
            with self.subTest(family=family):
                generated = self.generated(self.configure(family, "rtx4090-24g"))
                self.assertIn("-arch=sm_89", generated)
                self.assertNotIn("--enable-new-dtags", generated)
                self.assertNotIn("libamdhip64", generated)


if __name__ == "__main__":
    unittest.main()
