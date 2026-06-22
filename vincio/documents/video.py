"""Video understanding: deterministic frame sampling, temporal segmentation,
and analysis into citable, time-stamped evidence.

Video is a first-class modality beside image, table, and text. A
:class:`VideoAnalyzer` turns a clip into a :class:`VideoAnalysis` — a timeline of
:class:`VideoSegment`\\ s, each carrying a transcript/caption and the frames
sampled from its window — which :func:`video_evidence_items` lowers into
``modality="video"`` :class:`~vincio.core.types.EvidenceItem` records that the
context compiler scores, budgets, orders, and cites alongside everything else,
with the segment's ``time_range`` preserved through to the citation.

The frame-sampling and temporal-segmentation primitives are deterministic and
dependency-free, so they address a clip without decoding it. The
:class:`MockVideoAnalyzer` keeps offline runs deterministic; the real path
(:class:`PyAVFrameExtractor` + :class:`ProviderVideoAnalyzer`) decodes frames
behind the ``vincio[video]`` extra and captions them with a vision provider.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from ..core.errors import DocumentError, LoaderError
from ..core.types import EvidenceItem, ImageRef, TrustLevel, VideoRef

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .audio import Transcriber
    from .multimodal import ImageAnalyzer

__all__ = [
    "VideoFrame",
    "VideoSegment",
    "VideoAnalysis",
    "sample_frame_times",
    "segment_timeline",
    "FrameExtractor",
    "PyAVFrameExtractor",
    "VideoAnalyzer",
    "MockVideoAnalyzer",
    "ProviderVideoAnalyzer",
    "video_evidence_items",
]


# -- deterministic temporal primitives ----------------------------------------


def sample_frame_times(
    duration_s: float, *, fps: float | None = None, count: int | None = None
) -> list[float]:
    """Deterministic frame-sampling timestamps (seconds) over ``[0, duration_s)``.

    ``count`` samples that many evenly-spaced frames at the centre of each equal
    slice (stable regardless of duration); ``fps`` samples one frame every
    ``1/fps`` seconds. With neither, defaults to one frame per second. Always
    returns at least one timestamp for a positive duration.
    """
    if duration_s <= 0:
        return []
    if count is not None and count > 0:
        return [round(duration_s * (i + 0.5) / count, 3) for i in range(count)]
    rate = fps if (fps and fps > 0) else 1.0
    step = 1.0 / rate
    times: list[float] = []
    t = 0.0
    while t < duration_s - 1e-9:
        times.append(round(t, 3))
        t += step
    return times or [0.0]


def segment_timeline(
    duration_s: float, *, window_s: float, stride_s: float | None = None
) -> list[tuple[float, float]]:
    """Deterministic temporal segmentation of ``[0, duration_s)`` into windows.

    Each window spans ``window_s`` seconds, advancing by ``stride_s`` (defaults
    to ``window_s`` for non-overlapping segments; a smaller stride overlaps). The
    final window is clamped to ``duration_s`` so the whole clip is covered.
    """
    if duration_s <= 0 or window_s <= 0:
        return []
    stride = stride_s if (stride_s and stride_s > 0) else window_s
    segments: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s - 1e-9:
        end = min(start + window_s, duration_s)
        segments.append((round(start, 3), round(end, 3)))
        if end >= duration_s:
            break
        start += stride
    return segments


# -- data models --------------------------------------------------------------


class VideoFrame(BaseModel):
    """A single frame sampled from a clip at ``timestamp`` seconds."""

    timestamp: float
    image: ImageRef | None = None
    caption: str = ""


class VideoSegment(BaseModel):
    """A temporal segment of a clip with its transcript/caption and frames."""

    start: float
    end: float
    text: str = ""
    frames: list[VideoFrame] = Field(default_factory=list)
    speaker: str | None = None

    @property
    def timestamp(self) -> str:
        def fmt(seconds: float) -> str:
            minutes, sec = divmod(int(seconds), 60)
            return f"{minutes:02d}:{sec:02d}"

        return f"[{fmt(self.start)}–{fmt(self.end)}]"


class VideoAnalysis(BaseModel):
    """The structured understanding of a clip: a segmented, frame-sampled
    timeline plus the full transcript and clip-level metadata."""

    duration_s: float = 0.0
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    language: str | None = None
    transcript: str = ""
    segments: list[VideoSegment] = Field(default_factory=list)

    @property
    def frames(self) -> list[VideoFrame]:
        """Every sampled frame across all segments, in time order."""
        return [frame for segment in self.segments for frame in segment.frames]


# -- frame extraction (real backend) ------------------------------------------


class FrameExtractor(Protocol):
    def extract(
        self, video_path: str | Path, timestamps: list[float]
    ) -> list[ImageRef]:  # pragma: no cover - protocol
        ...


class PyAVFrameExtractor:
    """Decode frames at given timestamps with PyAV (``vincio[video]``).

    Seeks to each timestamp, decodes the nearest frame, writes it as a PNG into
    ``out_dir`` (a temp dir by default), and returns an :class:`ImageRef` per
    frame. Decoding needs the optional ``vincio[video]`` extra.
    """

    def __init__(self, *, out_dir: str | Path | None = None) -> None:
        self.out_dir = Path(out_dir) if out_dir is not None else None

    def extract(self, video_path: str | Path, timestamps: list[float]) -> list[ImageRef]:
        try:
            import av  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise LoaderError(
                'video frame extraction requires PyAV: pip install "vincio[video]"'
            ) from exc
        import tempfile

        out_dir = self.out_dir or Path(tempfile.mkdtemp(prefix="vincio-frames-"))
        out_dir.mkdir(parents=True, exist_ok=True)
        refs: list[ImageRef] = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            time_base = stream.time_base or 1
            for index, ts in enumerate(timestamps):
                offset = int(ts / float(time_base))
                container.seek(offset, stream=stream, any_frame=False, backward=True)
                frame = next((f for f in container.decode(stream)), None)
                if frame is None:
                    continue
                path = out_dir / f"frame_{index:04d}_{ts:.3f}.png"
                frame.to_image().save(str(path))
                refs.append(
                    ImageRef(path=str(path), media_type="image/png", metadata={"timestamp": ts})
                )
        return refs


# -- analyzers ----------------------------------------------------------------


class VideoAnalyzer(Protocol):
    async def analyze(self, video_path: str | Path) -> VideoAnalysis:  # pragma: no cover
        ...


class MockVideoAnalyzer:
    """Deterministic offline video analyzer.

    Produces a fixed, segmented timeline (its own scene script, or a supplied
    one) so the whole video-understanding path — segmentation, frame sampling,
    evidence, temporal citation — runs without a model, network, or codec.
    """

    def __init__(
        self,
        *,
        script: list[str] | None = None,
        segment_seconds: float = 5.0,
        fps: float = 1.0,
        frames_per_segment: int = 1,
        transcript: list[str] | None = None,
    ) -> None:
        self.script = script or [
            "A title card introduces the product demo.",
            "A presenter speaks to the camera about the roadmap.",
            "A bar chart of quarterly results is shown on screen.",
        ]
        self.segment_seconds = max(0.1, segment_seconds)
        self.fps = fps
        self.frames_per_segment = max(1, frames_per_segment)
        self.transcript = transcript

    async def analyze(self, video_path: str | Path) -> VideoAnalysis:
        segments: list[VideoSegment] = []
        for index, caption in enumerate(self.script):
            start = round(index * self.segment_seconds, 3)
            end = round(start + self.segment_seconds, 3)
            spoken = (
                self.transcript[index]
                if self.transcript and index < len(self.transcript)
                else caption
            )
            frame_times = [
                round(start + offset, 3)
                for offset in sample_frame_times(
                    self.segment_seconds, count=self.frames_per_segment
                )
            ]
            frames = [
                VideoFrame(
                    timestamp=ts,
                    image=ImageRef(metadata={"caption": caption, "timestamp": ts}),
                    caption=caption,
                )
                for ts in frame_times
            ]
            segments.append(VideoSegment(start=start, end=end, text=spoken, frames=frames))
        duration = round(len(self.script) * self.segment_seconds, 3)
        return VideoAnalysis(
            duration_s=duration,
            fps=self.fps,
            language="en",
            transcript=" ".join(self.transcript or self.script),
            segments=segments,
        )


class ProviderVideoAnalyzer:
    """Analyze a clip with a frame extractor + a vision provider (+ a transcriber).

    Segments the timeline, samples and decodes frames per segment, captions each
    frame with an :class:`~vincio.documents.multimodal.ImageAnalyzer`, and folds
    in an optional audio transcript — assembling a :class:`VideoAnalysis`. The
    frame decode runs behind ``vincio[video]``; offline tests use
    :class:`MockVideoAnalyzer`.
    """

    def __init__(
        self,
        analyzer: ImageAnalyzer,
        *,
        extractor: FrameExtractor | None = None,
        transcriber: Transcriber | None = None,
        segment_seconds: float = 5.0,
        frames_per_segment: int = 1,
        duration_s: float | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.extractor = extractor or PyAVFrameExtractor()
        self.transcriber = transcriber
        self.segment_seconds = max(0.1, segment_seconds)
        self.frames_per_segment = max(1, frames_per_segment)
        self.duration_s = duration_s

    async def analyze(self, video_path: str | Path) -> VideoAnalysis:
        duration = self.duration_s if self.duration_s is not None else _probe_duration(video_path)
        if duration <= 0:
            raise DocumentError(
                "could not determine video duration; pass duration_s to the analyzer"
            )
        transcript_segments: list[Any] = []
        full_transcript = ""
        language: str | None = None
        if self.transcriber is not None:
            transcript = await self.transcriber.transcribe(video_path)
            transcript_segments = list(transcript.segments)
            full_transcript = transcript.text
            language = transcript.language
        windows = segment_timeline(duration, window_s=self.segment_seconds)
        segments: list[VideoSegment] = []
        for start, end in windows:
            frame_times = [
                round(start + offset, 3)
                for offset in sample_frame_times(end - start, count=self.frames_per_segment)
            ]
            refs = self.extractor.extract(video_path, frame_times)
            frames: list[VideoFrame] = []
            captions: list[str] = []
            for ref, ts in zip(refs, frame_times, strict=False):
                observations = await self.analyzer.observe(ref)
                caption = "; ".join(o.observation for o in observations)
                captions.append(caption)
                frames.append(VideoFrame(timestamp=ts, image=ref, caption=caption))
            spoken = _transcript_for_window(transcript_segments, start, end)
            text = spoken or " ".join(c for c in captions if c)
            segments.append(VideoSegment(start=start, end=end, text=text, frames=frames))
        return VideoAnalysis(
            duration_s=round(duration, 3),
            language=language,
            transcript=full_transcript or " ".join(s.text for s in segments),
            segments=segments,
        )


def _probe_duration(video_path: str | Path) -> float:
    try:
        import av  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise LoaderError(
            'probing a video duration requires PyAV: pip install "vincio[video]" '
            "(or pass duration_s explicitly)"
        ) from exc
    with av.open(str(video_path)) as container:  # pragma: no cover - needs a real clip
        if container.duration:
            return float(container.duration) / 1_000_000.0
        stream = container.streams.video[0]
        if stream.duration and stream.time_base:
            return float(stream.duration) * float(stream.time_base)
    return 0.0


def _transcript_for_window(segments: list[Any], start: float, end: float) -> str:
    """The transcript text overlapping ``[start, end)``, joined in time order."""
    parts: list[str] = []
    for seg in segments:
        seg_start = float(getattr(seg, "start", 0.0))
        seg_end = float(getattr(seg, "end", 0.0))
        if seg_end > start and seg_start < end:
            text = str(getattr(seg, "text", "")).strip()
            if text:
                parts.append(text)
    return " ".join(parts)


# -- analysis → evidence ------------------------------------------------------


def video_evidence_items(
    analysis: VideoAnalysis,
    *,
    source_id: str,
    video_path: str | None = None,
    media_type: str = "video/mp4",
    url: str | None = None,
    trust_level: TrustLevel = TrustLevel.UNTRUSTED_DOCUMENT,
) -> list[EvidenceItem]:
    """Lower a :class:`VideoAnalysis` into citable, time-stamped evidence.

    One ``modality="video"`` :class:`~vincio.core.types.EvidenceItem` per
    segment: ``text`` is the segment's transcript/caption surrogate (scored and
    deduped like any text), ``video`` carries the clip reference plus the
    segment's sampled-frame captions, and ``time_range`` pins the segment so the
    citation resolves to the moment (``citation_ref`` → ``<source>:t<start>-<end>``).
    """
    items: list[EvidenceItem] = []
    for index, segment in enumerate(analysis.segments, start=1):
        frame_captions = [f.caption for f in segment.frames if f.caption]
        video_ref = VideoRef(
            path=video_path,
            url=url,
            media_type=media_type,
            duration_seconds=analysis.duration_s or None,
            fps=analysis.fps,
            metadata={
                "transcript": segment.text,
                "caption": frame_captions[0] if frame_captions else segment.text,
                "frame_captions": frame_captions,
                "frame_timestamps": [f.timestamp for f in segment.frames],
            },
        )
        text = segment.text or (frame_captions[0] if frame_captions else "")
        items.append(
            EvidenceItem(
                id=f"{source_id}:seg{index}",
                source_id=source_id,
                source_type="document",
                modality="video",
                text=f"{segment.timestamp} {text}".strip(),
                video=video_ref,
                time_range=(segment.start, segment.end),
                media_ref=video_path or url,
                trust_level=trust_level,
                provenance=0.9 if (video_path or url) else 0.5,
                metadata={
                    "segment": index,
                    "speaker": segment.speaker,
                    "frame_count": len(segment.frames),
                },
            )
        )
    return items
