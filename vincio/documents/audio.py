"""Audio transcript ingestion.

Turns the dead "audio" file classification into a real ingestion path: a
:class:`Transcriber` (mirroring :class:`~vincio.documents.ocr.OCREngine`)
produces a timestamped, optionally speaker-diarized transcript that
``load_media`` assembles into a :class:`~vincio.core.types.Document`. The
:class:`MockTranscriber` keeps offline runs deterministic; Whisper/Deepgram or a
provider-audio backend handle real audio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from ..core.errors import DocumentError

__all__ = [
    "TranscriptSegment",
    "Transcript",
    "Transcriber",
    "MockTranscriber",
    "WhisperTranscriber",
    "ProviderAudioTranscriber",
]


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None

    @property
    def timestamp(self) -> str:
        def fmt(seconds: float) -> str:
            minutes, sec = divmod(int(seconds), 60)
            return f"{minutes:02d}:{sec:02d}"

        return f"[{fmt(self.start)}–{fmt(self.end)}]"


class Transcript(BaseModel):
    text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    language: str | None = None
    duration_s: float = 0.0


class Transcriber(Protocol):
    async def transcribe(self, audio_path: str | Path) -> Transcript:  # pragma: no cover
        ...


class MockTranscriber:
    """Deterministic offline transcriber.

    Produces fixed segments (its own text, or a supplied script) so the audio
    ingestion path is exercised without a model or network. ``diarize`` alternates
    two speakers across segments.
    """

    def __init__(self, *, script: list[str] | None = None, diarize: bool = False) -> None:
        self.script = script or [
            "This is a deterministic mock transcript segment.",
            "It exercises the audio ingestion path offline.",
        ]
        self.diarize = diarize

    async def transcribe(self, audio_path: str | Path) -> Transcript:
        segments: list[TranscriptSegment] = []
        cursor = 0.0
        for index, line in enumerate(self.script):
            duration = max(1.0, len(line.split()) / 3.0)
            segments.append(
                TranscriptSegment(
                    start=round(cursor, 2),
                    end=round(cursor + duration, 2),
                    text=line,
                    speaker=(f"Speaker {index % 2 + 1}" if self.diarize else None),
                )
            )
            cursor += duration
        return Transcript(
            text=" ".join(self.script),
            segments=segments,
            language="en",
            duration_s=round(cursor, 2),
        )


class WhisperTranscriber:
    """OpenAI Whisper transcription (``audio/transcriptions``, verbose JSON)."""

    def __init__(
        self, api_key: str | None = None, *, model: str = "whisper-1",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def transcribe(self, audio_path: str | Path) -> Transcript:
        import httpx

        path = Path(audio_path)
        async with httpx.AsyncClient(timeout=300.0) as client:
            with path.open("rb") as fh:
                response = await client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key or ''}"},
                    data={"model": self.model, "response_format": "verbose_json"},
                    files={"file": (path.name, fh, "application/octet-stream")},
                )
        if response.status_code >= 400:
            raise DocumentError(f"transcription error {response.status_code}: {response.text[:500]}")
        data = response.json()
        segments = [
            TranscriptSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
            )
            for seg in data.get("segments") or []
        ]
        return Transcript(
            text=data.get("text", ""),
            segments=segments,
            language=data.get("language"),
            duration_s=float(data.get("duration", 0.0) or 0.0),
        )


class ProviderAudioTranscriber:
    """Transcribe via an audio-capable :class:`~vincio.providers.base.ModelProvider`.

    Sends the clip as a chat ``input_audio`` part (the 1.9 audio-input wiring),
    so any provider that accepts audio input can transcribe without a dedicated
    speech-to-text endpoint. Segment timing is not recovered (one segment).
    """

    PROMPT = "Transcribe this audio exactly. Output only the transcription text."

    def __init__(self, provider: object, *, model: str) -> None:
        self.provider = provider
        self.model = model

    async def transcribe(self, audio_path: str | Path) -> Transcript:
        from ..core.types import AudioRef, ContentPart, Message, ModelRequest

        request = ModelRequest(
            model=self.model,
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentPart(type="text", text=self.PROMPT),
                        ContentPart(type="audio", audio=AudioRef(path=str(audio_path))),
                    ],
                )
            ],
            temperature=0.0,
        )
        response = await self.provider.generate(request)  # type: ignore[attr-defined]
        text = response.text.strip()
        return Transcript(text=text, segments=[TranscriptSegment(start=0.0, end=0.0, text=text)])
