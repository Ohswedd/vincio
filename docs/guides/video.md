# Native video understanding & generation

The multimodal packet already scores, budgets, orders, and cites image and table
evidence beside text, and generation flows images and audio *out* with C2PA
provenance. Video joins them as a **first-class modality on the same packet** — not
a new plane. A recorded meeting, a screen capture, a product demo is evidence; the
job is to keep the *temporal* structure that makes it evidence, all the way through
to the citation.

Everything here is additive and opt-in. The dependency-free offline path uses a
deterministic mock; the real frame-decode path installs behind the `vincio[video]`
extra (PyAV + Pillow).

```bash
pip install "vincio[video]"   # only needed for the real frame-decode analyzer
```

## Video as typed evidence

A `VideoRef` joins `ImageRef` / `AudioRef`, a `ContentPart` gains a `video` part, and
`EvidenceItem` gains `modality="video"`, a `video` carrier, and a `time_range`
temporal locator. The scorable surrogate (a transcript or caption) drives relevance,
dedup, and ordering the way it does for an image's caption — so video competes for the
budget and gets cited uniformly.

```python
from vincio.core.types import EvidenceItem, VideoRef

clip = EvidenceItem(
    source_id="DEMO",
    modality="video",
    time_range=(10.0, 15.0),
    video=VideoRef(path="/clips/demo.mp4", metadata={"transcript": "a chart is shown"}),
    text="[00:10–00:15] a chart is shown",
)
clip.citation_ref   # "DEMO:t10-15"  — the time range, not just the document
```

## Understanding a clip

Two deterministic, dependency-free primitives address a clip without decoding it:

- `sample_frame_times(duration_s, *, fps=None, count=None)` — frame-sampling
  timestamps (evenly-spaced centres for `count`, every `1/fps` seconds for `fps`).
- `segment_timeline(duration_s, *, window_s, stride_s=None)` — temporal
  segmentation into windows (overlapping when `stride_s < window_s`), clamped to the
  duration so the whole clip is covered.

A `VideoAnalyzer` turns a clip into a `VideoAnalysis` — a `VideoSegment` timeline of
transcripts/captions and sampled `VideoFrame`s. `MockVideoAnalyzer` keeps offline runs
deterministic; `ProviderVideoAnalyzer` (with a `PyAVFrameExtractor` and a vision
provider) decodes and captions frames on the real path. `video_evidence_items` lowers
an analysis into typed, time-stamped, citable evidence.

```python
import asyncio
from vincio.documents import MockVideoAnalyzer, video_evidence_items

analysis = asyncio.run(MockVideoAnalyzer(segment_seconds=5.0).analyze("demo.mp4"))
evidence = video_evidence_items(analysis, source_id="DEMO", video_path="/clips/demo.mp4")
# Each item: modality="video", time_range=(start, end), citation_ref "DEMO:t<start>-<end>"
```

`app.load_video(path, *, analyzer)` ingests a clip as a temporally-segmented
`Document` whose sections carry their `start` / `end` timestamps — so the same
retrieval, chunking, and citation machinery that handles a PDF page handles a clip
segment.

## Temporal grounding

`EvidenceItem.time_range` is preserved end-to-end: retrieval chunking copies a
transcript segment's `(start, end)` onto the chunk and the evidence it yields, and the
`CitedReportBuilder` resolves a claim to a `ResolvedCitation.time_range` and renders
the footnote at the **moment** — `, t10–15s` — not just the document. A video-grounded
answer is auditable at sub-clip resolution.

```python
from vincio.generation.report import CitedReportBuilder

answer = f"The chart appears near the end [{evidence[-1].citation_ref}]."
report = asyncio.run(CitedReportBuilder().build_report(answer, evidence))
report.citations[0].footnote()   # "DEMO [...], t10–15s — /clips/demo.mp4: \"…\""
```

## Generating video with provenance

A `VideoProvider` mirrors the image and speech surfaces: `generate_video` /
`edit_video` over a deterministic `MockVideoProvider`, OpenAI Sora
(`OpenAIVideoProvider`), Google Veo (`GoogleVideoProvider`), and a generic
`HTTPVideoProvider`. Every clip carries a C2PA `ProvenanceManifest` bound to its bytes,
priced by `video_cost` / `VideoPrice`, metered against the run `Budget`, and audited —
exactly the way generated images and audio are.

```python
from vincio.generation.video import MockVideoProvider, VideoGenRequest
from vincio.governance import verify_manifest

response = app.generate_video("a 4-second product teaser", provider=MockVideoProvider(), seconds=4)
clip = response.videos[0]
verify_manifest(clip.manifest, clip.data)        # True — bound to the exact bytes
verify_manifest(clip.manifest, clip.data + b"x") # False — a single altered byte fails
```

Editing (`app.edit_video` / `aedit_video`) marks the manifest synthetic-and-edited and
records a `video_edit` audit event; generation records `video_generate`. Both land on
the hash-chained audit log with the provenance binding.

## See also

- `examples/02_retrieval_rag.py` — a fully offline walkthrough.
- [Generate documents & media](generate-documents.md) — the image / speech / document
  output modalities video sits beside.
- The `video` VincioBench family and its SLOs (`video_temporal_grounding`,
  `video_generation_provenance_bound`, `video_first_class_evidence`).
