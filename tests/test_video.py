"""Native video understanding & generation: video is a first-class evidence
modality the compiler scores, budgets, orders, and cites alongside text/image/
table; deterministic frame sampling and temporal segmentation address a clip
without decoding it; a claim grounds to a time range preserved through to the
citation; and generated/edited video carries a C2PA manifest bound to its bytes.
"""

from __future__ import annotations

import pytest

from vincio import ContextApp, VincioConfig
from vincio.context.compiler import ContextCompiler
from vincio.context.ir import ContextIR
from vincio.context.packet import ContextPacket
from vincio.context.scoring import ContextCandidate, ContextScorer
from vincio.core.media import encode_video_bytes
from vincio.core.types import ContentPart, EvidenceItem, Objective, VideoRef
from vincio.documents import (
    MockVideoAnalyzer,
    load_video,
    sample_frame_times,
    segment_timeline,
    video_evidence_items,
)
from vincio.generation.report import CitedReportBuilder
from vincio.generation.video import MockVideoProvider, VideoGenRequest
from vincio.governance.transparency import verify_manifest
from vincio.providers import MockProvider

# -- deterministic temporal primitives -------------------------------------


def test_sample_frame_times_count_and_fps():
    # `count` → evenly-spaced frame centres, deterministic regardless of duration.
    assert sample_frame_times(10.0, count=2) == [2.5, 7.5]
    # `fps` → one frame every 1/fps seconds.
    assert sample_frame_times(3.0, fps=1.0) == [0.0, 1.0, 2.0]
    # A positive duration always yields at least one frame.
    assert sample_frame_times(0.5, fps=0.1) == [0.0]
    assert sample_frame_times(0.0) == []


def test_segment_timeline_windows_and_overlap():
    assert segment_timeline(12.0, window_s=5.0) == [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]
    # Smaller stride overlaps; the final window is clamped to the duration.
    overlapped = segment_timeline(10.0, window_s=6.0, stride_s=3.0)
    assert overlapped[0] == (0.0, 6.0)
    assert overlapped[-1][1] == 10.0
    assert segment_timeline(0.0, window_s=5.0) == []


# -- typed video modality on evidence --------------------------------------


def test_video_ref_and_content_part():
    ref = VideoRef(path="/x/demo.mp4", duration_seconds=12.0, fps=24.0)
    part = ContentPart(type="video", video=ref)
    assert part.type == "video"
    assert part.video.media_type == "video/mp4"


def test_video_evidence_scorable_text_and_token_cost():
    ev = EvidenceItem(
        source_id="d1",
        modality="video",
        video=VideoRef(path="/x.mp4", detail="high", metadata={"transcript": "a chart is shown"}),
    )
    # The transcript/caption is the scorable surrogate across the pipeline.
    assert ev.scorable_text == "a chart is shown"
    # Modality-aware token cost: a high-detail clip ships several frames.
    assert ev.estimated_token_cost() == 2048


def test_video_citation_ref_is_temporal():
    ev = EvidenceItem(source_id="VID1", modality="video", time_range=(12.0, 18.5), text="…")
    # The time range is preserved into the citation, not just the document.
    assert ev.citation_ref == "VID1:t12-18.5"


def test_scorer_video_token_cost():
    scorer = ContextScorer(max_token_cost=4000)
    clip = ContextCandidate(
        id="c", type="evidence", content="transcript", modality="video",
        video=VideoRef(path="/x.mp4", detail="low"),
    )
    assert scorer.modality_token_cost(clip) == 256
    assert 0.0 < scorer.normalized_token_cost(clip) <= 1.0


# -- analyzer → evidence ---------------------------------------------------


async def test_mock_analyzer_produces_segmented_timeline():
    analyzer = MockVideoAnalyzer(segment_seconds=5.0, frames_per_segment=2)
    analysis = await analyzer.analyze("demo.mp4")
    assert analysis.duration_s == 15.0
    assert len(analysis.segments) == 3
    assert len(analysis.frames) == 6  # 2 frames per segment
    # Segments tile the timeline contiguously.
    assert analysis.segments[0].start == 0.0
    assert analysis.segments[1].start == 5.0


def test_video_evidence_items_carry_time_range_and_payload():
    import asyncio

    analyzer = MockVideoAnalyzer(segment_seconds=4.0)
    analysis = asyncio.run(analyzer.analyze("demo.mp4"))
    items = video_evidence_items(analysis, source_id="VID1", video_path="/x/demo.mp4")
    assert all(it.modality == "video" for it in items)
    first = items[0]
    assert first.time_range == (0.0, 4.0)
    assert first.citation_ref == "VID1:t0-4"
    assert first.video is not None and first.video.path == "/x/demo.mp4"
    assert first.scorable_text  # transcript surrogate present


# -- loader: temporally-segmented document ---------------------------------


def test_load_video_builds_timestamped_sections(tmp_path):
    clip = tmp_path / "demo.mp4"
    clip.write_bytes(b"not-a-real-codec-stream")
    doc = load_video(str(clip), analyzer=MockVideoAnalyzer(segment_seconds=5.0))
    assert doc.media_type == "video/analysis"
    assert doc.metadata["segment_count"] == 3
    assert doc.metadata["duration_s"] == 15.0
    # Each section carries its start/end timestamps for downstream grounding.
    assert doc.sections[0]["start"] == 0.0
    assert doc.sections[0]["end"] == 5.0
    assert doc.sections[1]["start"] == 5.0


def test_load_video_missing_file_raises(tmp_path):
    from vincio.core.errors import LoaderError

    with pytest.raises(LoaderError):
        load_video(str(tmp_path / "nope.mp4"), analyzer=MockVideoAnalyzer())


# -- compiler: video is a first-class candidate ----------------------------


async def test_compiler_selects_and_serializes_video_evidence():
    compiler = ContextCompiler()
    evidence = [
        EvidenceItem(source_id="d1", text="The annual fee is $99.", relevance=0.6),
        EvidenceItem(
            source_id="VID1", modality="video", relevance=0.9, time_range=(2.0, 7.0),
            video=VideoRef(path="/demo.mp4", metadata={"transcript": "the annual fee is shown on screen"}),
            text="[00:02–00:07] the annual fee is shown on screen",
        ),
    ]
    candidates = compiler._collect(evidence=evidence, memory=[], tool_results=[])
    video_candidate = next(c for c in candidates if c.modality == "video")
    assert video_candidate.video is not None
    assert video_candidate.token_cost > 0

    # The packet serializes the video payload and the temporal locator.
    ir = ContextIR(objective=Objective("fees"), evidence=evidence)
    packet = ContextPacket.from_ir(ir, slim=True)
    entry = next(e for e in packet.evidence_items if e["source_id"] == "VID1")
    assert entry["modality"] == "video"
    assert entry["video"]["path"] == "/demo.mp4"
    assert entry["time_range"] == [2.0, 7.0]


# -- temporal grounding through to the citation ----------------------------


async def test_cited_report_renders_temporal_footnote():
    evidence = video_evidence_items(
        await MockVideoAnalyzer(segment_seconds=6.0).analyze("demo.mp4"),
        source_id="VID1",
        video_path="/x/demo.mp4",
    )
    answer = f"The roadmap is discussed early in the clip [{evidence[1].citation_ref}]."
    report = await CitedReportBuilder().build_report(answer, evidence)
    assert len(report.citations) == 1
    citation = report.citations[0]
    assert citation.time_range == (6.0, 12.0)
    # The footnote points at the moment (seconds), not just the source.
    assert "t6–12s" in citation.footnote()


# -- generation: C2PA-bound synthetic video --------------------------------


async def test_mock_video_provider_stamps_provenance():
    provider = MockVideoProvider()
    response = await provider.generate_video(VideoGenRequest(prompt="a cat playing", seconds=4))
    assert len(response.videos) == 1
    clip = response.videos[0]
    assert clip.media_type == "video/mp4"
    assert clip.manifest is not None
    assert clip.manifest.media_type == "video/mp4"
    # The manifest binds to the exact bytes the consumer receives.
    assert verify_manifest(clip.manifest, clip.data)


async def test_video_generation_is_deterministic_and_edit_marks_edited():
    provider = MockVideoProvider()
    req = VideoGenRequest(prompt="a sunset", seconds=3, seed=7)
    first = (await provider.generate_video(req)).videos[0]
    second = (await provider.generate_video(req)).videos[0]
    assert first.data == second.data  # deterministic
    edited = (await provider.edit_video(VideoRef(path="/in.mp4"), req)).videos[0]
    assert edited.data != first.data
    assert edited.manifest is not None and edited.manifest.is_synthetic


# -- app surface: metered, audited, provenance-stamped ---------------------


async def test_app_generate_video_meters_and_audits(tmp_path):
    config = VincioConfig()
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    app = ContextApp(name="vid", provider=MockProvider(), config=config)

    response = await app.agenerate_video("a product demo", provider=MockVideoProvider(), seconds=5)
    assert response.videos and response.videos[0].manifest is not None
    entries = [e for e in app.audit.entries if e.action == "video_generate"]
    assert len(entries) == 1
    assert entries[0].details["assets"] == 1
    assert entries[0].details["content_sha256"]  # provenance binding recorded
    assert app.audit.verify_chain()


def test_app_load_video_offline(tmp_path):
    clip = tmp_path / "demo.mp4"
    clip.write_bytes(b"stub")
    config = VincioConfig()
    config.observability.exporter = "memory"
    app = ContextApp(name="vid", provider=MockProvider(), config=config)
    doc = app.load_video(str(clip), analyzer=MockVideoAnalyzer())
    assert doc.media_type == "video/analysis"


def test_encode_video_bytes_enforces_cap(tmp_path):
    from vincio.core.errors import InputError

    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"abcdef")
    media_type, encoded = encode_video_bytes(VideoRef(path=str(clip)))
    assert media_type == "video/mp4"
    assert encoded
    with pytest.raises(InputError):
        encode_video_bytes(VideoRef(path=str(clip)), max_bytes=2)
    with pytest.raises(InputError):
        encode_video_bytes(VideoRef())  # no path
