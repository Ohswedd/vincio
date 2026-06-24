"""Real-behavior coverage for vincio.documents.video.

Targets the uncovered paths: ProviderVideoAnalyzer.analyze (with/without a
transcriber, duration probe failure, caption assembly, transcript-vs-caption
fallback), _transcript_for_window overlap logic, _probe_duration's missing-PyAV
error, PyAVFrameExtractor without the codec extra, and segment_timeline's
loop-exit branch. Everything runs offline with the deterministic MockProvider.
"""

from __future__ import annotations

import asyncio

import pytest

from vincio.core.errors import DocumentError, LoaderError
from vincio.core.types import ImageRef, TrustLevel
from vincio.documents.audio import MockTranscriber
from vincio.documents.multimodal import ImageAnalyzer
from vincio.documents.video import (
    MockVideoAnalyzer,
    ProviderVideoAnalyzer,
    PyAVFrameExtractor,
    VideoAnalysis,
    VideoFrame,
    VideoSegment,
    _probe_duration,
    _transcript_for_window,
    sample_frame_times,
    segment_timeline,
    video_evidence_items,
)
from vincio.providers import MockProvider

# -- real frame extractor stub (implements the FrameExtractor protocol) -------


class _StubExtractor:
    """A deterministic, codec-free FrameExtractor: one ImageRef per timestamp.

    Records the timestamps it was asked for so tests can assert what windows
    drove extraction. Not a mock object — a real class with real behavior.
    """

    def __init__(self, *, drop: bool = False) -> None:
        self.drop = drop
        self.calls: list[list[float]] = []

    def extract(self, video_path, timestamps):  # noqa: ANN001 - protocol shape
        self.calls.append(list(timestamps))
        if self.drop:
            return []
        return [
            ImageRef(path=f"{video_path}#{ts:.3f}.png", metadata={"timestamp": ts})
            for ts in timestamps
        ]


def _vision_analyzer() -> ImageAnalyzer:
    return ImageAnalyzer(MockProvider(), model="mock-vision")


# -- segment_timeline: the loop-exit (no break) branch ------------------------


def test_segment_timeline_exits_loop_when_window_exceeds_duration():
    # window_s > duration_s: first end == duration triggers the `end >= duration`
    # break path; verify the single clamped window.
    assert segment_timeline(3.0, window_s=10.0) == [(0.0, 3.0)]


def test_segment_timeline_exact_tiling_exits_via_condition_not_break():
    # An evenly divisible timeline: the last window ends exactly at duration, so
    # the loop body breaks there and start never advances past duration.
    segs = segment_timeline(9.0, window_s=3.0)
    assert segs == [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]


def test_segment_timeline_zero_window_returns_empty():
    assert segment_timeline(10.0, window_s=0.0) == []
    assert segment_timeline(10.0, window_s=-1.0) == []


def test_segment_timeline_default_stride_is_non_overlapping():
    # stride_s None / <= 0 falls back to window_s.
    assert segment_timeline(8.0, window_s=4.0, stride_s=0.0) == [(0.0, 4.0), (4.0, 8.0)]


# -- _transcript_for_window ---------------------------------------------------


class _Seg:
    def __init__(self, start, end, text):  # noqa: ANN001
        self.start = start
        self.end = end
        self.text = text


def test_transcript_for_window_joins_overlapping_in_time_order():
    segs = [
        _Seg(0.0, 4.0, "intro line"),
        _Seg(4.0, 8.0, "middle line"),
        _Seg(8.0, 12.0, "outro line"),
    ]
    # Window [3, 9) overlaps the first three boundaries: intro (ends 4 > 3) and
    # middle and outro (starts 8 < 9).
    assert _transcript_for_window(segs, 3.0, 9.0) == "intro line middle line outro line"


def test_transcript_for_window_excludes_non_overlapping_and_blank():
    segs = [
        _Seg(0.0, 2.0, "before"),
        _Seg(5.0, 7.0, "   "),  # overlaps but blank → skipped
        _Seg(5.0, 7.0, "kept"),
    ]
    # Window [5, 7): "before" ends at 2 (not > 5) → excluded; blank stripped out.
    assert _transcript_for_window(segs, 5.0, 7.0) == "kept"


def test_transcript_for_window_empty_segments():
    assert _transcript_for_window([], 0.0, 10.0) == ""


def test_transcript_for_window_uses_getattr_defaults():
    # Objects missing start/end/text fall back to 0.0/"" and are excluded.
    class _Bare:
        pass

    assert _transcript_for_window([_Bare()], 1.0, 5.0) == ""


# -- _probe_duration: missing PyAV --------------------------------------------


def test_probe_duration_without_pyav_raises_loader_error():
    with pytest.raises(LoaderError, match="requires PyAV"):
        _probe_duration("/no/such/clip.mp4")


# -- PyAVFrameExtractor -------------------------------------------------------


def test_pyav_extractor_stores_out_dir(tmp_path):
    ext = PyAVFrameExtractor(out_dir=tmp_path / "frames")
    assert ext.out_dir == tmp_path / "frames"
    default = PyAVFrameExtractor()
    assert default.out_dir is None


def test_pyav_extractor_without_codec_raises_loader_error():
    ext = PyAVFrameExtractor()
    with pytest.raises(LoaderError, match='vincio\\[video\\]'):
        ext.extract("/clip.mp4", [0.0, 1.0])


# -- ProviderVideoAnalyzer.__init__ defaults ----------------------------------


def test_provider_analyzer_defaults_clamp_and_fallback():
    pa = ProviderVideoAnalyzer(
        _vision_analyzer(),
        segment_seconds=0.0,  # clamped up to 0.1
        frames_per_segment=0,  # clamped up to 1
    )
    assert pa.segment_seconds == 0.1
    assert pa.frames_per_segment == 1
    # No extractor supplied → defaults to a PyAVFrameExtractor.
    assert isinstance(pa.extractor, PyAVFrameExtractor)
    assert pa.transcriber is None


# -- ProviderVideoAnalyzer.analyze: duration probe failure --------------------


def test_provider_analyze_unknown_duration_raises_document_error():
    # duration_s omitted → _probe_duration runs and, without PyAV, raises
    # LoaderError before the DocumentError guard is reached.
    pa = ProviderVideoAnalyzer(_vision_analyzer(), extractor=_StubExtractor())
    with pytest.raises(LoaderError, match="requires PyAV"):
        asyncio.run(pa.analyze("/clip.mp4"))


def test_provider_analyze_nonpositive_duration_raises_document_error():
    pa = ProviderVideoAnalyzer(
        _vision_analyzer(), extractor=_StubExtractor(), duration_s=0.0
    )
    with pytest.raises(DocumentError, match="could not determine video duration"):
        asyncio.run(pa.analyze("/clip.mp4"))


# -- ProviderVideoAnalyzer.analyze: caption assembly (no transcriber) ---------


def test_provider_analyze_builds_segments_from_frame_captions():
    extractor = _StubExtractor()
    pa = ProviderVideoAnalyzer(
        _vision_analyzer(),
        extractor=extractor,
        segment_seconds=5.0,
        frames_per_segment=2,
        duration_s=12.0,
    )
    analysis = asyncio.run(pa.analyze("/clip.mp4"))

    # 12s / 5s windows → [0,5), [5,10), [10,12).
    assert [(s.start, s.end) for s in analysis.segments] == [
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 12.0),
    ]
    # Each window sampled frames_per_segment frames; extractor saw 2 per window.
    assert all(len(call) == 2 for call in extractor.calls)
    assert len(analysis.frames) == 6
    # With no transcriber, segment text is the joined frame captions (non-empty).
    assert all(seg.text for seg in analysis.segments)
    assert analysis.language is None
    assert analysis.duration_s == 12.0
    # transcript falls back to the joined segment texts.
    assert analysis.transcript == " ".join(s.text for s in analysis.segments)


def test_provider_analyze_dropped_frames_yield_empty_text():
    # An extractor that returns no frames → no captions → segment text "".
    pa = ProviderVideoAnalyzer(
        _vision_analyzer(),
        extractor=_StubExtractor(drop=True),
        segment_seconds=5.0,
        duration_s=5.0,
    )
    analysis = asyncio.run(pa.analyze("/clip.mp4"))
    assert len(analysis.segments) == 1
    assert analysis.segments[0].frames == []
    assert analysis.segments[0].text == ""
    # Whole transcript collapses to empty.
    assert analysis.transcript == ""


# -- ProviderVideoAnalyzer.analyze: with a transcriber ------------------------


def test_provider_analyze_folds_in_transcript_over_captions():
    transcriber = MockTranscriber(
        script=["spoken intro here", "spoken middle here"]
    )
    pa = ProviderVideoAnalyzer(
        _vision_analyzer(),
        extractor=_StubExtractor(),
        transcriber=transcriber,
        segment_seconds=5.0,
        duration_s=10.0,
    )
    analysis = asyncio.run(pa.analyze("/clip.mp4"))

    # Transcriber set language and full transcript directly.
    assert analysis.language == "en"
    assert analysis.transcript == "spoken intro here spoken middle here"
    # The spoken transcript wins over frame captions for the overlapping window.
    assert "spoken intro here" in analysis.segments[0].text


# -- video_evidence_items: round-trip through ProviderVideoAnalyzer output ----


def test_provider_output_lowers_to_temporal_evidence():
    pa = ProviderVideoAnalyzer(
        _vision_analyzer(),
        extractor=_StubExtractor(),
        segment_seconds=5.0,
        duration_s=10.0,
    )
    analysis = asyncio.run(pa.analyze("/clip.mp4"))
    items = video_evidence_items(
        analysis,
        source_id="VID",
        url="https://x/clip.mp4",
        trust_level=TrustLevel.USER,
    )
    assert len(items) == len(analysis.segments)
    first = items[0]
    assert first.modality == "video"
    assert first.time_range == (0.0, 5.0)
    assert first.trust_level == TrustLevel.USER
    # url provided (no path) → provenance 0.9 and media_ref is the url.
    assert first.provenance == 0.9
    assert first.media_ref == "https://x/clip.mp4"
    # frame captions are threaded into the video ref metadata.
    assert first.video.metadata["frame_captions"]
    assert first.metadata["frame_count"] == len(analysis.segments[0].frames)


def test_video_evidence_no_source_lowers_provenance_and_uses_caption_text():
    # No path and no url → provenance 0.5; a segment with empty text falls back
    # to the first frame caption for both text and metadata.
    frame = VideoFrame(timestamp=1.0, caption="a bar chart")
    seg = VideoSegment(start=0.0, end=2.0, text="", frames=[frame], speaker="host")
    analysis = VideoAnalysis(duration_s=2.0, segments=[seg])
    items = video_evidence_items(analysis, source_id="S")
    assert len(items) == 1
    item = items[0]
    assert item.provenance == 0.5
    assert item.media_ref is None
    # text = timestamp + caption surrogate.
    assert item.text == "[00:00–00:02] a bar chart"
    assert item.video.metadata["caption"] == "a bar chart"
    assert item.metadata["speaker"] == "host"


def test_video_evidence_empty_analysis_yields_no_items():
    assert video_evidence_items(VideoAnalysis(), source_id="S") == []


# -- MockVideoAnalyzer: supplied transcript path ------------------------------


def test_mock_analyzer_uses_supplied_transcript_per_segment():
    analyzer = MockVideoAnalyzer(
        script=["scene one", "scene two"],
        transcript=["narration one", "narration two"],
        segment_seconds=3.0,
    )
    analysis = asyncio.run(analyzer.analyze("clip.mp4"))
    assert analysis.duration_s == 6.0
    # Segment text comes from the transcript, frame captions from the script.
    assert analysis.segments[0].text == "narration one"
    assert analysis.segments[0].frames[0].caption == "scene one"
    assert analysis.transcript == "narration one narration two"


def test_mock_analyzer_transcript_shorter_than_script_falls_back():
    # transcript shorter than script → later segments fall back to the caption.
    analyzer = MockVideoAnalyzer(
        script=["one", "two", "three"],
        transcript=["spoken one"],
        segment_seconds=2.0,
    )
    analysis = asyncio.run(analyzer.analyze("clip.mp4"))
    assert analysis.segments[0].text == "spoken one"
    assert analysis.segments[1].text == "two"  # past the transcript → caption
    assert analysis.segments[2].text == "three"


def test_mock_analyzer_clamps_segment_seconds_and_frames():
    analyzer = MockVideoAnalyzer(
        script=["only"], segment_seconds=0.0, frames_per_segment=0
    )
    assert analyzer.segment_seconds == 0.1
    assert analyzer.frames_per_segment == 1
    analysis = asyncio.run(analyzer.analyze("c.mp4"))
    assert len(analysis.segments[0].frames) == 1


# -- VideoAnalysis / VideoSegment helpers -------------------------------------


def test_video_segment_timestamp_formats_minutes_and_seconds():
    seg = VideoSegment(start=65.0, end=130.0)
    assert seg.timestamp == "[01:05–02:10]"


def test_video_analysis_frames_flattens_in_time_order():
    a = VideoAnalysis(
        segments=[
            VideoSegment(start=0, end=1, frames=[VideoFrame(timestamp=0.5)]),
            VideoSegment(
                start=1,
                end=2,
                frames=[VideoFrame(timestamp=1.2), VideoFrame(timestamp=1.8)],
            ),
        ]
    )
    assert [f.timestamp for f in a.frames] == [0.5, 1.2, 1.8]


def test_sample_frame_times_count_centres_independent_of_duration():
    # count branch wins even when fps is also given.
    assert sample_frame_times(8.0, fps=4.0, count=4) == [1.0, 3.0, 5.0, 7.0]


def test_sample_frame_times_fps_steps_by_inverse_rate():
    # fps>0 path: one frame every 1/fps seconds across [0, duration).
    assert sample_frame_times(2.0, fps=2.0) == [0.0, 0.5, 1.0, 1.5]


def test_sample_frame_times_defaults_to_one_per_second():
    # Neither count nor fps → default rate 1.0 fps.
    assert sample_frame_times(4.0) == [0.0, 1.0, 2.0, 3.0]


def test_sample_frame_times_zero_fps_and_count_use_default_rate():
    # count<=0 and fps<=0 both fall through to the 1.0 fps default.
    assert sample_frame_times(3.0, fps=0.0, count=0) == [0.0, 1.0, 2.0]


def test_sample_frame_times_short_duration_returns_single_zero():
    # The while loop never appends (step exceeds duration) → guaranteed [0.0].
    assert sample_frame_times(0.4, fps=1.0) == [0.0]


def test_sample_frame_times_negative_duration_is_empty():
    assert sample_frame_times(-5.0, count=3) == []


def test_segment_timeline_overlap_exits_via_loop_condition():
    # stride < window with a final start that lands inside the last window:
    # the last appended window has end < duration is false at the clamp, but an
    # earlier iteration leaves end < duration so no break, then start advances
    # past duration and the while condition ends the loop.
    segs = segment_timeline(7.0, window_s=4.0, stride_s=4.0)
    # [0,4) end<7 → no break, start→4; [4,7) end==7 → clamp+break path.
    assert segs == [(0.0, 4.0), (4.0, 7.0)]
    # A stride that overshoots past duration before reaching the clamp.
    overshoot = segment_timeline(5.0, window_s=2.0, stride_s=10.0)
    # [0,2) end<5 → no break, start→10 ≥ 5 → loop condition exits (no clamp).
    assert overshoot == [(0.0, 2.0)]
