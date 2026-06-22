"""Native video understanding & generation — video as first-class evidence.

A recorded meeting, a screen capture, a product demo is evidence — but reduced to
a transcript or a handful of stills it loses the temporal structure that makes it
evidence. This example makes video a first-class modality on the *existing*
multimodal packet: a clip is sampled, segmented, scored, and cited beside text
and images, with the timestamp preserved all the way through to the citation, and
generated/edited video carries a C2PA manifest bound to its bytes — exactly the
way generated images and audio already do.

Five steps, all offline and deterministic:

  1. Understand: a `MockVideoAnalyzer` turns a clip into a segmented, frame-sampled
     timeline; `video_evidence_items` lowers it into typed video evidence.
  2. First-class evidence: the context compiler scores and cites the clip beside
     text — same packet, same budget, same citations.
  3. Temporal grounding: a clip-grounded claim resolves to a *time range*; the
     cited-report footnote points at the moment, not just the document.
  4. Generate with provenance: a `MockVideoProvider` produces a clip whose C2PA
     manifest binds to its exact bytes — tampering fails verification.
  5. Metered & audited: generation is metered against the budget and lands on the
     hash-chained audit log with the provenance binding, like every other asset.

Everything is opt-in and additive; the real path decodes frames behind the
`vincio[video]` extra and captions them with a vision provider, but the offline
mock exercises the whole flow without a model, network, or codec.
"""

from __future__ import annotations

import asyncio

from vincio import ContextApp, VincioConfig
from vincio.context.compiler import ContextCompiler
from vincio.documents import MockVideoAnalyzer, video_evidence_items
from vincio.generation.report import CitedReportBuilder
from vincio.generation.video import MockVideoProvider, VideoGenRequest
from vincio.governance import verify_manifest
from vincio.providers import MockProvider


def _app() -> ContextApp:
    config = VincioConfig()
    config.observability.exporter = "memory"
    return ContextApp(name="video_demo", provider=MockProvider(), config=config)


async def main() -> None:
    print("Native video understanding & generation — video as first-class evidence\n")

    # 1. Understand — sample, segment, and lower the clip into typed evidence.
    print("1. Understand — a segmented, frame-sampled timeline becomes video evidence")
    analysis = await MockVideoAnalyzer(segment_seconds=5.0, frames_per_segment=2).analyze("demo.mp4")
    evidence = video_evidence_items(analysis, source_id="DEMO", video_path="/clips/demo.mp4")
    print(f"   duration {analysis.duration_s}s | {len(analysis.segments)} segments | "
          f"{len(analysis.frames)} frames sampled")
    for item in evidence:
        print(f"     {item.citation_ref:14} {item.scorable_text[:48]}")

    # 2. First-class evidence — the compiler scores/cites the clip beside text.
    print("\n2. First-class evidence — scored and budgeted in the same packet as text")
    from vincio.core.types import EvidenceItem

    mixed = [EvidenceItem(source_id="d1", text="Unrelated background note.", relevance=0.2), *evidence]
    candidates = ContextCompiler()._collect(evidence=mixed, memory=[], tool_results=[])
    clip = next(c for c in candidates if c.modality == "video")
    print(f"   video candidate selected: modality={clip.modality} token_cost={clip.token_cost}")

    # 3. Temporal grounding — a claim resolves to a time range, cited to the moment.
    print("\n3. Temporal grounding — the citation points at the moment, not the document")
    answer = f"The quarterly chart is shown near the end of the clip [{evidence[-1].citation_ref}]."
    report = await CitedReportBuilder().build_report(answer, evidence)
    citation = report.citations[0]
    print(f"   claim grounds to t={citation.time_range[0]}–{citation.time_range[1]}s")
    print(f"   footnote: {citation.footnote()}")

    # 4. Generate with provenance — a C2PA manifest bound to the clip's bytes.
    print("\n4. Generate with provenance — synthetic video is tamper-evident")
    app = _app()
    response = await app.agenerate_video("a 4-second product teaser", provider=MockVideoProvider(), seconds=4)
    generated = response.videos[0]
    print(f"   generated {len(generated.data)} bytes | manifest media_type={generated.manifest.media_type}")
    print(f"   manifest binds to the bytes: {verify_manifest(generated.manifest, generated.data)}")
    print(f"   a single altered byte fails: {verify_manifest(generated.manifest, generated.data + b'x')}")

    # 5. Metered & audited — the generation is on the verifiable chain.
    print("\n5. Metered & audited — the generation lands on the hash-chained audit log")
    entry = next(e for e in app.audit.entries if e.action == "video_generate")
    print(f"   audit action={entry.action} assets={entry.details['assets']} "
          f"provenance={entry.details['content_sha256'][0][:16]}…")
    print(f"   chain verifies = {app.audit.verify_chain()}")

    print("\nVideo rides the existing multimodal packet — scored, cited at the moment, and provenance-bound.")


if __name__ == "__main__":
    asyncio.run(main())
