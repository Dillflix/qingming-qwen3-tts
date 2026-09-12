import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


MODEL_NAMES = ("qwen3-tts", "tts-1", "tts-1-hd", "qingming-qwen3-tts-1.7b-customvoice")
DEFAULT_ALIASES = {
    "alloy": "Vivian", "echo": "Ryan", "fable": "Sophia",
    "nova": "Isabella", "onyx": "Evan", "shimmer": "Lily",
}
LANGUAGE_SUFFIXES = {
    "en": "english", "zh": "chinese", "de": "german", "it": "italian",
    "pt": "portuguese", "es": "spanish", "ja": "japanese", "ko": "korean",
    "fr": "french", "ru": "russian",
}
CLONE_ERROR = (
    "Reference-audio voice cloning requires a Base checkpoint and is not enabled "
    "on this CustomVoice server."
)


class APIError(Exception):
    def __init__(self, message, param=None, code="invalid_value", status=400):
        super().__init__(message)
        self.status = status
        self.body = {"error": {"message": message, "type": (
            "invalid_request_error" if status < 500 else "server_error"
        ), "param": param, "code": code}}


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str = "qwen3-tts"
    input: str = Field(min_length=1, max_length=4096)
    voice: str = "Vivian"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0, strict=True)
    stream: bool = Field(default=False, strict=True)
    stream_format: Literal["audio"] = "audio"
    language: str | None = "Auto"
    instruct: str | None = Field(default=None, max_length=4096)
    # OpenAI spells this plural; accept either, but never silently choose between them.
    instructions: str | None = Field(default=None, max_length=4096)

    @field_validator("input", "voice", "model", "language", "instruct", "instructions")
    @classmethod
    def valid_string(cls, value):
        if value is not None and ("\0" in value or not value.strip()):
            raise ValueError("must be nonblank and contain no NUL characters")
        if value is not None:
            value.encode("utf-8")  # Reject lone surrogate code points before IPC.
        return value


class VoiceCatalog:
    def __init__(self, model_dir: Path, aliases=None):
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("tts_model_type") != "custom_voice" or str(config.get("tts_model_size")).lower() not in ("1b7", "1.7b"):
            raise ValueError("This server requires a 1.7B CustomVoice checkpoint; no Base fallback exists")
        talker = config.get("talker_config", {})
        if not isinstance(talker, dict):
            raise ValueError("CustomVoice talker_config must be an object")
        self.speakers = talker.get("spk_id", {})
        self.dialects = talker.get("spk_is_dialect", {})
        self.languages = talker.get("codec_language_id", {})
        if (not isinstance(self.speakers, dict) or not self.speakers
                or not isinstance(self.languages, dict) or not self.languages
                or not isinstance(self.dialects, dict)
                or any(not isinstance(name, str) or not name.strip()
                       for name in (*self.speakers, *self.languages))):
            raise ValueError("CustomVoice config is missing speaker/language metadata")
        self.lookup = {name.casefold(): name for name in self.speakers}
        if len(self.lookup) != len(self.speakers):
            raise ValueError("Ambiguous case-insensitive speaker names")
        self.language_lookup = {name.casefold(): name for name in self.languages}
        self.aliases = {}
        for name, target in (DEFAULT_ALIASES if aliases is None else aliases).items():
            if not isinstance(name, str) or not isinstance(target, str):
                raise ValueError("Voice aliases must map strings to strings")
            if name.casefold() in self.lookup or name.casefold().startswith("clone:"):
                raise ValueError("Voice aliases must not shadow native speakers or use clone:")
            if target.casefold() in self.lookup:
                self.aliases[name.casefold()] = self.lookup[target.casefold()]
        self.models = {name: None for name in MODEL_NAMES}
        for suffix, language in LANGUAGE_SUFFIXES.items():
            if language in self.language_lookup:
                for prefix in ("tts-1", "tts-1-hd"):
                    self.models[f"{prefix}-{suffix}"] = self.language_lookup[language]

    def resolve(self, request):
        if request.voice.casefold().startswith("clone:"):
            raise APIError(CLONE_ERROR, "voice", "voice_cloning_not_supported")
        speaker = self.lookup.get(request.voice.casefold()) or self.aliases.get(request.voice.casefold())
        if not speaker:
            raise APIError("Unknown preset voice; see /v1/voices", "voice")
        if request.model not in self.models:
            raise APIError("Unknown model alias; see /v1/models", "model")
        language = request.language or "auto"
        suffix = self.models[request.model]
        if suffix:
            if language.casefold() != "auto" and language.casefold() != suffix.casefold():
                raise APIError("Language conflicts with the model suffix", "language")
            language = suffix
        if language.casefold() != "auto":
            language = self.language_lookup.get(language.casefold())
            if not language:
                raise APIError("Language is not supported by this checkpoint", "language")
        if request.instruct is not None and request.instructions is not None and request.instruct != request.instructions:
            raise APIError("instruct and instructions disagree", "instruct")
        if request.stream and request.response_format != "pcm":
            raise APIError("True streaming currently supports response_format=pcm only", "response_format")
        return speaker, language, request.instruct or request.instructions or ""

    def voices(self):
        native = [{"id": name, "speaker": name, "alias": False, "dialect": self.dialects.get(name, False)}
                  for name in self.speakers]
        aliases = [{"id": name, "speaker": target, "alias": True, "dialect": self.dialects.get(target, False)}
                   for name, target in self.aliases.items()]
        return {"object": "list", "data": native + aliases}


def split_text(text, minimum=20, maximum=70):
    """Prefer sentence punctuation, then whitespace, then a Unicode-codepoint cut."""
    text = text.strip()
    result = []
    while len(text) > maximum:
        candidates = [m.end() for m in re.finditer(r"[.!?。！？;；,:：，](?:\s+|$)|[。！？；，：]", text[:maximum + 1])
                      if minimum <= m.end() <= maximum]
        if not candidates:
            candidates = [m.start() for m in re.finditer(r"\s+", text[:maximum + 1])
                          if minimum <= m.start() <= maximum]
        end = candidates[-1] if candidates else maximum
        result.append(text[:end].strip())
        text = text[end:].strip()
    if text:
        result.append(text)
    return result
