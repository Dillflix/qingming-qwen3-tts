"""Administrator-enrolled Base voices, validated and snapshotted before GPU startup."""
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

from scripts.clone_voice import fingerprint_model, load_profile
from .contract import APIError, LANGUAGE_SUFFIXES


class BaseVoiceCatalog:
    task = "base-xvector"

    def __init__(self, model_dir, library, registry):
        model_dir, library = Path(model_dir), Path(library).resolve(strict=True)
        spec = json.loads(Path(registry).read_text(encoding="utf-8"))
        if spec.get("schema") != "qingming-base-registry-v1":
            raise ValueError("Unsupported Base voice registry schema")
        voices = spec.get("voices")
        if not isinstance(voices, dict) or not 1 <= len(voices) <= 100:
            raise ValueError("Register 1..100 named voices explicitly")
        fingerprint = fingerprint_model(model_dir)
        self.model_fingerprint = fingerprint
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        languages = config.get("talker_config", {}).get("codec_language_id", {})
        if not isinstance(languages, dict) or not languages or any(not isinstance(k, str) for k in languages):
            raise ValueError("Base checkpoint language metadata is missing")
        self.languages = {name.casefold(): name for name in languages}
        self.lookup, embeddings, self.profiles = {}, {}, {}
        for name, slug in voices.items():
            if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,63}", name)
                    or name.casefold() in {"default", "alloy", "echo", "aiden"}
                    or name.casefold() in self.lookup):
                raise ValueError("Invalid, reserved, or duplicate registered voice name")
            if not isinstance(slug, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", slug):
                raise ValueError("Voice profile must be a library slug, not a path")
            directory = library / slug
            if directory.is_symlink() or directory.resolve().parent != library:
                raise ValueError("Voice profile must stay within the private library")
            profile = load_profile(directory, model_dir, model_fingerprint=fingerprint)
            if profile.get("voice_id") != slug:
                raise ValueError("Voice profile ID disagrees with registry")
            payload = (directory / "speaker.bf16").read_bytes()
            if hashlib.sha256(payload).hexdigest() != profile["files"]["speaker.bf16"]:
                raise ValueError("Voice embedding changed during startup")
            self.lookup[name.casefold()] = name
            embeddings[name] = payload.hex()
            self.profiles[name] = directory
        default = spec.get("default")
        if not isinstance(default, str) or default.casefold() not in self.lookup:
            raise ValueError("Registry default must name a registered voice")
        self.default = self.lookup[default.casefold()]
        self.embeddings = MappingProxyType(embeddings)
        self.aliases = {name: self.default for name in ("default", "alloy", "echo")}
        self.models = {name: None for name in ("qwen3-tts", "tts-1", "tts-1-hd", "qingming-qwen3-tts-1.7b-base")}
        for suffix, language in LANGUAGE_SUFFIXES.items():
            if language in self.languages:
                for prefix in ("tts-1", "tts-1-hd"):
                    self.models[f"{prefix}-{suffix}"] = self.languages[language]

    def resolve(self, request):
        speaker = self.lookup.get(request.voice.casefold()) or self.aliases.get(request.voice.casefold())
        if speaker is None:
            raise APIError("Unknown registered voice; see /v1/voices. Aiden is no longer available on Base.", "voice")
        if request.model not in self.models:
            raise APIError("Unknown model alias; see /v1/models", "model")
        if request.instruct is not None or request.instructions is not None:
            raise APIError("Base cloned voices do not support style instructions", "instructions", "unsupported_parameter")
        language = request.language or "auto"
        suffix = self.models[request.model]
        if suffix:
            if language.casefold() != "auto" and language.casefold() != suffix.casefold():
                raise APIError("Language conflicts with model suffix", "language")
            language = suffix
        if language.casefold() == "auto":
            language = "Auto"
        else:
            language = self.languages.get(language.casefold())
            if language is None:
                raise APIError("Unsupported language", "language")
        if request.stream and request.response_format != "pcm":
            raise APIError("True streaming currently supports response_format=pcm only", "response_format")
        return speaker, language, ""

    def voices(self):
        return {"object": "list", "default": self.default, "data": [
            {"id": name, "speaker": name, "alias": False} for name in self.embeddings
        ] + [{"id": name, "speaker": target, "alias": True} for name, target in self.aliases.items()]}
