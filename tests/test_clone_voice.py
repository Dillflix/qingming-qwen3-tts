"""Offline clone-profile checks with a fake native process; no HIP qualification."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import wave

from scripts import clone_voice


BASE_CONFIG = {"tts_model_type": "base", "tts_model_size": "1b7"}
EMBEDDING = struct.pack("<2048H", *([0x3E80, 0xBE80] * 1024))


def write_reference(path, *, seconds=4, rate=24000, channels=1, width=2, silent=False):
    """A comfortably audible, unclipped PCM fixture with an exact duration."""
    frames = int(seconds * rate)
    sample = (b"\x00" * width if silent else
              {1: b"\xa0", 2: struct.pack("<h", 8192), 4: struct.pack("<i", 536870912)}[width])
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(sample * channels * frames)


def write_native_audio(path, *, samples=24000):
    pcm = struct.pack("<2f", 0.25, -0.25) * (samples // 2)
    header = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
              + struct.pack("<IHHIIHH", 16, 3, 1, 24000, 96000, 4, 32)
              + b"data" + struct.pack("<I", len(pcm)))
    path.write_bytes(header + pcm)


class FakeNative:
    """Capture the actual CLI contract while producing deterministic native artifacts."""
    def __init__(self, binary, *, eos=True, mismatch=False, returncode=0, timeout=False):
        self.binary = binary
        self.eos = eos
        self.mismatch = mismatch
        self.returncode = returncode
        self.timeout = timeout
        self.calls = []

    def __call__(self, command, **kwargs):
        command = [str(item) for item in command]
        # This allowlist also prevents a passing workflow from starting/stopping
        # services, changing GPU settings, installing packages, or invoking a shell.
        if command[0] == "git":
            if command[-2:] != ["rev-parse", "HEAD"]:
                raise AssertionError(f"Unexpected git mutation: {command}")
            return subprocess.CompletedProcess(command, 0, "test-source-revision\n", "")
        if command[0] == "fake-ffmpeg":
            if "-protocol_whitelist" not in command or command[command.index("-protocol_whitelist") + 1] != "file,pipe":
                raise AssertionError("Reference decoding must be restricted to local input")
            shutil.copyfile(command[command.index("-i") + 1], command[-1])
            return subprocess.CompletedProcess(command, 0)
        if command[0] != str(self.binary):
            raise AssertionError(f"Unexpected external command: {command}")
        self.calls.append((command, kwargs))
        if self.timeout:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        value = lambda flag: command[command.index(flag) + 1]
        output = Path(value("--output"))
        write_native_audio(output, samples=12 * 1920)
        if self.mismatch and len(self.calls) > 1:
            raw = output.read_bytes()
            output.write_bytes(raw[:44] + struct.pack("<f", 0.125) + raw[48:])
        if "--save-speaker-embedding-bf16" in command:
            Path(value("--save-speaker-embedding-bf16")).write_bytes(EMBEDDING)
        budget = int(value("--max-new-tokens"))
        metadata = {
            "status": "ok", "eos": self.eos, "frames": 12 if self.eos else budget,
            "eos_frame": 12 if self.eos else -1, "max_new_tokens": budget,
            "lifecycle": "once", "task": "base-xvector", "family": "1.7b",
            "audio_duration_s": 1.0, "output": str(output), "e2e_ms": 50,
        }
        content = json.dumps(metadata) + "\n"
        destination = kwargs.get("stdout")
        if hasattr(destination, "write"):
            try:
                destination.write(content)
            except TypeError:
                destination.write(content.encode("utf-8"))
            destination.flush()
            stdout = None
        else:
            stdout = content if kwargs.get("text") or kwargs.get("encoding") else content.encode("utf-8")
        result = subprocess.CompletedProcess(command, self.returncode, stdout, "")
        if kwargs.get("check"):
            result.check_returncode()
        return result


class CloneInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_embedding_requires_exact_shape_and_finite_bfloat16_values(self):
        path = self.root / "speaker.bf16"
        path.write_bytes(EMBEDDING)
        clone_voice.check_embedding(path)
        for invalid in (b"", EMBEDDING[:-1], EMBEDDING + b"\0\0", b"\0" * 4096,
                        struct.pack("<H", 0x7F80) + EMBEDDING[2:],
                        struct.pack("<H", 0xFF80) + EMBEDDING[2:],
                        struct.pack("<H", 0x7FC1) + EMBEDDING[2:]):
            with self.subTest(length=len(invalid), prefix=invalid[:2]):
                path.write_bytes(invalid)
                with self.assertRaises((ValueError, RuntimeError)):
                    clone_voice.check_embedding(path)

    def test_reference_must_be_pcm16_mono_24khz_and_within_duration_bounds(self):
        path = self.root / "reference.wav"
        write_reference(path)
        self.assertIsInstance(clone_voice.validate_reference(path), dict)
        for kwargs in ({"channels": 2}, {"rate": 16000}, {"width": 1}, {"width": 4},
                       {"seconds": 2.99}, {"seconds": 30.01}, {"silent": True}):
            with self.subTest(kwargs=kwargs):
                write_reference(path, **kwargs)
                with self.assertRaises((ValueError, RuntimeError)):
                    clone_voice.validate_reference(path)

    def test_reference_rejects_truncated_wave_payload(self):
        path = self.root / "reference.wav"
        write_reference(path)
        path.write_bytes(path.read_bytes()[:-10])
        with self.assertRaises((ValueError, RuntimeError, EOFError)):
            clone_voice.validate_reference(path)

    def test_native_audio_requires_natural_eos_and_consistent_frames_within_budget(self):
        path = self.root / "native.wav"
        write_native_audio(path, samples=12 * 1920)
        metadata = {"status": "ok", "task": "base-xvector", "eos": True, "frames": 12}
        self.assertEqual(len(clone_voice.native_wav(path, metadata, 12)), 12 * 1920 * 4)
        for changed, budget in (({"eos": False}, 12), ({"eos": "true"}, 12), ({}, 11),
                                ({"frames": 11}, 12), ({"frames": True}, 12),
                                ({"status": "error"}, 12), ({"task": "custom-voice"}, 12)):
            with self.subTest(changed=changed, budget=budget):
                with self.assertRaises(ValueError):
                    clone_voice.native_wav(path, {**metadata, **changed}, budget)

    def test_native_audio_rejects_silence_nonfinite_samples_and_truncation(self):
        path = self.root / "native.wav"
        write_native_audio(path, samples=12 * 1920)
        original = path.read_bytes()
        metadata = {"status": "ok", "task": "base-xvector", "eos": True, "frames": 12}
        for invalid in (original[:-1], original[:44] + b"\0" * (len(original) - 44),
                        original[:44] + struct.pack("<f", float("nan")) + original[48:],
                        original[:44] + struct.pack("<f", float("inf")) + original[48:]):
            with self.subTest(length=len(invalid), first_sample=invalid[44:48]):
                path.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    clone_voice.native_wav(path, metadata, 512)


class CloneWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps(BASE_CONFIG), encoding="utf-8")
        for filename in ("model.safetensors", "tokenizer.json", "tokenizer_config.json",
                         "vocab.json", "merges.txt", "speech_tokenizer/config.json",
                         "speech_tokenizer/model.safetensors"):
            target = self.model / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"{}" if filename.endswith(".json") else b"test model artifact")
        self.binary = self.root / "native-binary"
        self.binary.write_bytes(b"fake native executable")
        self.reference = self.root / "source.wav"
        write_reference(self.reference)
        self.library = self.root / "voices"
        self.profile_dir = self.library / "test-speaker"
        self.text_file = self.root / "text.txt"
        self.text_file.write_text("A short voice clone audition.", encoding="utf-8")
        self.common = ["--model-dir", str(self.model), "--binary", str(self.binary), "--hip-device", "0",
                       "--results-root", str(self.root / "results")]
        self.enroll_args = ["enroll", "--voice-id", "test-speaker", "--reference", str(self.reference),
                            "--consent-confirmed", "--library", str(self.library), "--ffmpeg", "fake-ffmpeg", *self.common]
        self.audition_args = ["audition", "--profile", str(self.profile_dir), "--text-file", str(self.text_file), *self.common]

    def run_main(self, args, native):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                patch.object(clone_voice.subprocess, "run", side_effect=native):
            try:
                return clone_voice.main(args)
            except SystemExit as error:
                return error.code

    def test_enrollment_requires_consent_before_starting_native(self):
        native = FakeNative(self.binary)
        args = [arg for arg in self.enroll_args if arg != "--consent-confirmed"]
        self.assertNotEqual(self.run_main(args, native), 0)
        self.assertFalse(native.calls)
        self.assertFalse(self.profile_dir.exists())

    def test_voice_id_cannot_escape_the_library(self):
        native = FakeNative(self.binary)
        for voice_id in ("../escape", "..", "nested/escape", "nested\\escape", "A Voice"):
            with self.subTest(voice_id=voice_id):
                args = list(self.enroll_args)
                args[args.index("--voice-id") + 1] = voice_id
                self.assertNotEqual(self.run_main(args, native), 0)
                self.assertFalse(native.calls)
        self.assertFalse((self.root / "escape").exists())

    def test_checkpoint_must_be_17b_base_before_starting_native(self):
        native = FakeNative(self.binary)
        for family, size in (("custom_voice", "1b7"), ("voice_design", "1b7"), ("base", "0b6")):
            with self.subTest(family=family, size=size):
                (self.model / "config.json").write_text(
                    json.dumps({"tts_model_type": family, "tts_model_size": size}), encoding="utf-8")
                self.assertNotEqual(self.run_main(self.enroll_args, native), 0)
                self.assertFalse(native.calls)
                self.assertFalse(self.profile_dir.exists())

    def test_saved_voice_roundtrip_and_audition_use_embedding_without_reference(self):
        native = FakeNative(self.binary)
        self.assertEqual(self.run_main(self.enroll_args, native), 0)
        self.assertEqual(len(native.calls), 4)
        profile = clone_voice.load_profile(self.profile_dir, self.model)
        self.assertEqual(profile["voice_id"], "test-speaker")
        self.assertTrue(profile["validation"]["reference_vs_saved_pcm_match"])
        self.assertTrue(profile["validation"]["natural_eos_all_cases"])
        self.assertFalse(profile["validation"]["production_qualified"])
        self.assertFalse(profile["validation"]["listening_approved"])
        self.assertEqual((self.profile_dir / "speaker.bf16").read_bytes(), EMBEDDING)
        self.assertIn("--ref-audio", native.calls[0][0])
        self.assertIn("--save-speaker-embedding-bf16", native.calls[0][0])
        self.assertNotIn("--speaker-embedding-bf16", native.calls[0][0])
        self.assertEqual(self.run_main(self.audition_args, native), 0)
        self.assertEqual(len(native.calls), 5)
        for command, kwargs in native.calls:
            self.assertEqual(command[command.index("--lifecycle") + 1], "once")
            self.assertEqual(command[command.index("--task") + 1], "base-xvector")
            self.assertEqual(kwargs["env"]["HIP_VISIBLE_DEVICES"], "0")
            self.assertNotIn("--instruct", command)
            self.assertNotIn("--ref-text", command)
        for command, _ in native.calls[1:]:
            self.assertIn("--speaker-embedding-bf16", command)
            self.assertNotIn("--ref-audio", command)
            self.assertNotIn("--save-speaker-embedding-bf16", command)
        reports = list((self.root / "results").glob("*/report.json"))
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(json.loads(path.read_text())["status"] == "PASS" for path in reports))
        self.assertFalse(list((self.root / "results").glob("*.tar.gz")))

    def test_archive_opt_in_packages_report_and_listening_audio_without_voice_profile(self):
        native = FakeNative(self.binary)
        self.assertEqual(self.run_main([*self.enroll_args, "--archive"], native), 0)
        archive, = (self.root / "results").glob("*.tar.gz")
        checksum = archive.with_name(archive.name + ".sha256").read_text().split()[0]
        self.assertEqual(checksum, hashlib.sha256(archive.read_bytes()).hexdigest())
        with tarfile.open(archive, "r:gz") as bundle:
            files = {Path(member.name).name: bundle.extractfile(member).read()
                     for member in bundle.getmembers() if member.isfile()}
        self.assertEqual(json.loads(files["report.json"])["status"], "PASS")
        self.assertEqual({name for name in files if name.endswith(".wav")},
                         {"reference.wav", "saved-voice.wav", "conversation.wav", "technical.wav"})
        for name in (self.reference.name, "speaker.bf16", "profile.json"):
            self.assertNotIn(name, files)
        # reference.wav in results is synthesized listening audio. The source
        # recording with the same profile filename must never enter the archive.
        self.assertNotIn(self.reference.read_bytes(), files.values())
        self.assertNotIn((self.profile_dir / "reference.wav").read_bytes(), files.values())
        self.assertNotIn(EMBEDDING, files.values())
        for name, content in files.items():
            if name.endswith(".wav"):
                self.assertEqual(content[:4], b"RIFF")
                self.assertGreater(len(content), 44)

    def test_native_timeout_retains_failure_report_without_retry_or_completed_profile(self):
        native = FakeNative(self.binary, timeout=True)
        self.assertEqual(self.run_main(self.enroll_args, native), 1)
        self.assertEqual(len(native.calls), 1)
        self.assertFalse((self.profile_dir / "profile.json").exists())
        report_path, = (self.root / "results").glob("*/report.json")
        report = json.loads(report_path.read_text())
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["cases"], {})
        self.assertIn("TimeoutExpired", " ".join(report["errors"]))

    def test_offline_workflow_preserves_production_api_and_service_artifacts(self):
        repo = Path(clone_voice.__file__).resolve().parents[1]
        protected = [path for directory in (repo / "qingming_api", repo / "deployment")
                     for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
        protected += [repo / "scripts/install_service.py"]
        before = {path: hashlib.sha256(path.read_bytes()).digest() for path in protected}
        native = FakeNative(self.binary)
        self.assertEqual(self.run_main(self.enroll_args, native), 0)
        self.assertEqual(self.run_main(self.audition_args, native), 0)
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).digest() for path in protected})

    def test_missing_model_weights_fail_before_decoding_or_native_generation(self):
        (self.model / "model.safetensors").unlink()
        native = FakeNative(self.binary)
        self.assertEqual(self.run_main(self.enroll_args, native), 1)
        self.assertFalse(native.calls)
        self.assertFalse(self.profile_dir.exists())

    def test_enrollment_does_not_overwrite_an_existing_voice(self):
        native = FakeNative(self.binary)
        self.assertEqual(self.run_main(self.enroll_args, native), 0)
        before = {path.name: path.read_bytes() for path in self.profile_dir.iterdir()}
        self.assertEqual(self.run_main(self.enroll_args, native), 1)
        self.assertEqual(len(native.calls), 4)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.profile_dir.iterdir()})

    def test_incomplete_native_generation_cannot_publish_or_replay_a_profile(self):
        native = FakeNative(self.binary, eos=False)
        self.assertEqual(self.run_main(self.enroll_args, native), 1)
        self.assertEqual(len(native.calls), 1)
        self.assertFalse((self.profile_dir / "profile.json").exists())
        self.assertEqual(self.run_main(self.audition_args, native), 1)
        self.assertEqual(len(native.calls), 1)
        report_path, = (self.root / "results").glob("*/reference.log")
        self.assertFalse(json.loads(report_path.read_text())["eos"])

    def test_saved_embedding_must_reproduce_reference_conditioned_audio(self):
        native = FakeNative(self.binary, mismatch=True)
        self.assertEqual(self.run_main(self.enroll_args, native), 1)
        self.assertEqual(len(native.calls), 2)
        self.assertFalse((self.profile_dir / "profile.json").exists())
        report_path, = (self.root / "results").glob("*/report.json")
        report = json.loads(report_path.read_text())
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("did not reproduce", " ".join(report["errors"]))

    def test_nonzero_native_exit_is_not_accepted_even_with_success_metadata(self):
        native = FakeNative(self.binary, returncode=1)
        self.assertEqual(self.run_main(self.enroll_args, native), 1)
        self.assertEqual(len(native.calls), 1)
        self.assertFalse((self.profile_dir / "profile.json").exists())

    def test_profile_payload_tampering_prevents_native_replay(self):
        native = FakeNative(self.binary)
        self.assertEqual(self.run_main(self.enroll_args, native), 0)
        for filename in ("reference.wav", "speaker.bf16"):
            with self.subTest(filename=filename):
                path = self.profile_dir / filename
                original = path.read_bytes()
                path.write_bytes(original[:-2] + b"\0\0")
                with self.assertRaises(ValueError):
                    clone_voice.load_profile(self.profile_dir, self.model)
                self.assertEqual(self.run_main(self.audition_args, native), 1)
                self.assertEqual(len(native.calls), 4)
                path.write_bytes(original)

    def test_profile_metadata_must_confirm_completed_authorized_enrollment(self):
        native = FakeNative(self.binary)
        self.assertEqual(self.run_main(self.enroll_args, native), 0)
        path = self.profile_dir / "profile.json"
        original = json.loads(path.read_text())
        for key, value in (("schema", "future-schema"), ("status", "FAILED"),
                           ("method", "icl"), ("consent_confirmed", False), ("consent_confirmed", "yes")):
            with self.subTest(key=key, value=value):
                path.write_text(json.dumps({**original, key: value}), encoding="utf-8")
                self.assertEqual(self.run_main(self.audition_args, native), 1)
                self.assertEqual(len(native.calls), 4)
        path.write_text(json.dumps(original), encoding="utf-8")

    def test_changed_model_weights_or_tokenizer_prevent_native_replay(self):
        native = FakeNative(self.binary)
        self.assertEqual(self.run_main(self.enroll_args, native), 0)
        for filename in ("model.safetensors", "tokenizer.json", "speech_tokenizer/model.safetensors"):
            with self.subTest(filename=filename):
                path = self.model / filename
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    clone_voice.load_profile(self.profile_dir, self.model)
                self.assertEqual(self.run_main(self.audition_args, native), 1)
                self.assertEqual(len(native.calls), 4)
                path.write_bytes(original)

    def test_model_fingerprint_covers_weight_and_tokenizer_assets(self):
        expected = {path.relative_to(self.model).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in self.model.rglob("*") if path.is_file()}
        self.assertEqual(clone_voice.fingerprint_model(self.model), expected)

    def test_audition_rejects_blank_nul_or_excessive_text_before_native(self):
        native = FakeNative(self.binary)
        self.assertEqual(self.run_main(self.enroll_args, native), 0)
        for text in (" \n\t", "test\0text", "x" * 2001):
            with self.subTest(length=len(text)):
                self.text_file.write_text(text, encoding="utf-8")
                self.assertEqual(self.run_main(self.audition_args, native), 1)
                self.assertEqual(len(native.calls), 4)


if __name__ == "__main__":
    unittest.main()
