"""Text-to-speech / speech-synthesis output modality.

A neutral ``synthesize_speech`` surface — voice / format / speed — over OpenAI
TTS, Gemini TTS, and ElevenLabs/Cartesia, with a deterministic
:class:`MockSpeechProvider` for offline tests. Synthetic speech is marked with
audio provenance and metered against the budget exactly like every other output,
unifying generated audio with the realtime audio path.
"""

from __future__ import annotations

import base64
import struct
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import MediaGenerationError
from ..core.types import AudioRef
from ..governance.transparency import ProvenanceManifest
from .media import attach_media_provenance, speech_cost

__all__ = [
    "SpeechRequest",
    "GeneratedAudio",
    "SpeechResponse",
    "SpeechProvider",
    "MockSpeechProvider",
    "OpenAISpeechProvider",
    "GoogleSpeechProvider",
    "ElevenLabsSpeechProvider",
]

AudioFormat = Literal["mp3", "wav", "opus", "aac", "flac", "pcm"]

_AUDIO_MEDIA = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


class SpeechRequest(BaseModel):
    text: str
    voice: str = "alloy"
    format: AudioFormat = "mp3"
    speed: float = 1.0
    language: str | None = None
    instructions: str | None = None  # delivery/style steering (OpenAI gpt-4o-mini-tts)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def media_type(self) -> str:
        return _AUDIO_MEDIA.get(self.format, "audio/mpeg")


class GeneratedAudio(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    data: bytes
    media_type: str = "audio/mpeg"
    voice: str = ""
    manifest: ProvenanceManifest | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def save(self, path: str) -> str:
        from pathlib import Path

        Path(path).write_bytes(self.data)
        return path

    def to_ref(self, path: str) -> AudioRef:
        self.save(path)
        return AudioRef(path=path, media_type=self.media_type)


class SpeechResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    audio: GeneratedAudio
    model: str = ""
    provider: str = ""
    cost_usd: float = 0.0
    characters: int = 0


class SpeechProvider(ABC):
    name: str = "speech"

    @abstractmethod
    async def synthesize_speech(self, request: SpeechRequest, *, model: str) -> SpeechResponse: ...

    async def aclose(self) -> None:
        return None

    def _wrap(
        self, data: bytes, request: SpeechRequest, *, model: str
    ) -> SpeechResponse:
        stamped, manifest = attach_media_provenance(
            data, media_type=request.media_type, model=model, provider=self.name
        )
        audio = GeneratedAudio(
            data=stamped, media_type=request.media_type, voice=request.voice, manifest=manifest
        )
        return SpeechResponse(
            audio=audio,
            model=model,
            provider=self.name,
            cost_usd=speech_cost(model, characters=len(request.text)),
            characters=len(request.text),
        )


# -- WAV helper (dependency-free, for the mock) -------------------------------


def _silent_wav(seconds: float = 0.2, sample_rate: int = 16000) -> bytes:
    """A minimal valid mono 16-bit PCM WAV of near-silence (offline tests)."""
    frames = max(1, int(seconds * sample_rate))
    data = b"\x00\x00" * frames
    block_align = 2
    byte_rate = sample_rate * block_align
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, block_align, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


class MockSpeechProvider(SpeechProvider):
    """Deterministic offline TTS: a real WAV whose length scales with the text."""

    name = "mock-tts"

    def __init__(self, *, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.requests: list[SpeechRequest] = []

    async def synthesize_speech(
        self, request: SpeechRequest, *, model: str = "mock-tts"
    ) -> SpeechResponse:
        self.requests.append(request)
        seconds = max(0.05, len(request.text) / 16.0 / max(0.1, request.speed))
        data = _silent_wav(seconds=seconds, sample_rate=self.sample_rate)
        # The mock always emits WAV; reflect that in the bound media type.
        wav_request = request.model_copy(update={"format": "wav"})
        return self._wrap(data, wav_request, model=model)


# -- OpenAI TTS ---------------------------------------------------------------


class OpenAISpeechProvider(SpeechProvider):
    """OpenAI audio/speech endpoint over httpx."""

    name = "openai"

    def __init__(self, api_key: str | None = None, *, base_url: str = "https://api.openai.com/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client: Any = None

    def _http(self) -> Any:
        import httpx

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def synthesize_speech(
        self, request: SpeechRequest, *, model: str = "gpt-4o-mini-tts"
    ) -> SpeechResponse:
        if not self.api_key:
            from ..core.errors import ProviderAuthError

            raise ProviderAuthError("missing OpenAI API key", provider=self.name)
        body: dict[str, Any] = {
            "model": model,
            "input": request.text,
            "voice": request.voice,
            "response_format": request.format,
            "speed": request.speed,
        }
        if request.instructions:
            body["instructions"] = request.instructions
        response = await self._http().post(
            f"{self.base_url}/audio/speech",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
        )
        if response.status_code >= 400:
            raise MediaGenerationError(f"OpenAI TTS error {response.status_code}: {response.text[:500]}")
        return self._wrap(response.content, request, model=model)

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


# -- Google TTS ---------------------------------------------------------------


class GoogleSpeechProvider(SpeechProvider):
    """Gemini TTS (``generateContent`` with an audio response modality)."""

    name = "google"

    def __init__(
        self, api_key: str | None = None,
        *, base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client: Any = None

    def _http(self) -> Any:
        import httpx

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def synthesize_speech(
        self, request: SpeechRequest, *, model: str = "gemini-2.5-flash-preview-tts"
    ) -> SpeechResponse:
        body = {
            "contents": [{"parts": [{"text": request.text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": request.voice}}
                },
            },
        }
        response = await self._http().post(
            f"{self.base_url}/models/{model}:generateContent",
            headers={"x-goog-api-key": self.api_key or ""},
            json=body,
        )
        if response.status_code >= 400:
            raise MediaGenerationError(f"Gemini TTS error {response.status_code}: {response.text[:500]}")
        data = response.json()
        audio_b64: str | None = None
        for candidate in data.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    audio_b64 = inline["data"]
                    break
        if audio_b64 is None:
            raise MediaGenerationError("Gemini TTS returned no audio data")
        # Gemini returns raw PCM; reflect that media type.
        pcm_request = request.model_copy(update={"format": "pcm"})
        return self._wrap(base64.b64decode(audio_b64), pcm_request, model=model)

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


# -- ElevenLabs / Cartesia ----------------------------------------------------


class ElevenLabsSpeechProvider(SpeechProvider):
    """ElevenLabs text-to-speech over httpx (also fits Cartesia-style endpoints
    via ``base_url`` / ``path`` overrides)."""

    name = "elevenlabs"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.elevenlabs.io/v1",
        default_voice: str = "Rachel",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_voice = default_voice
        self._client: Any = None

    def _http(self) -> Any:
        import httpx

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def synthesize_speech(
        self, request: SpeechRequest, *, model: str = "eleven_multilingual_v2"
    ) -> SpeechResponse:
        voice = request.voice or self.default_voice
        response = await self._http().post(
            f"{self.base_url}/text-to-speech/{voice}",
            headers={"xi-api-key": self.api_key or "", "Accept": request.media_type},
            json={"text": request.text, "model_id": model},
        )
        if response.status_code >= 400:
            raise MediaGenerationError(
                f"ElevenLabs TTS error {response.status_code}: {response.text[:500]}"
            )
        return self._wrap(response.content, request, model=model)

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
