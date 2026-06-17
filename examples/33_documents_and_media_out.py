"""Documents & images flow OUT — cited, governed, eval-gated artifacts (1.9).

Vincio reads a DOCX, a PDF, and a scanned packet, and validates a JSON answer —
1.9 closes the loop so the *deliverable* comes out under the same guarantees:

  1. DocumentBuilder — turn a validated result into a structurally-contracted,
     provenance-audited artifact (Markdown/HTML dependency-free; DOCX/PDF/PPTX
     behind extras).
  2. CitedReportBuilder — resolve [E1] markers to footnotes + a bibliography,
     with sentence-level citation coverage and per-claim entailment.
  3. Image generation & TTS — first-class output modalities, every asset
     C2PA-stamped, metered against the budget, and audited.
  4. Richer inputs — new-format loaders (PPTX/EPUB/RTF/ODT/…), audio transcript
     ingestion, and offline forms/KYC extraction.
  5. EU AI Act conformity pack — risk-tier classification and an Annex IV
     technical-documentation artifact, generated from the live system.

Runs fully offline with deterministic mock providers and mock media providers.
"""

from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path

from _shared import example_provider

from vincio import ContextApp
from vincio.core.types import EvidenceItem, TrustLevel
from vincio.documents import HeuristicFormExtractor, MockTranscriber, load_document
from vincio.generation import (
    CitationContract,
    DocumentContract,
    ImageGenRequest,
    MockImageProvider,
    MockSpeechProvider,
    SpeechRequest,
    TableSpec,
    generate_redline,
)
from vincio.governance import verify_manifest, write_sidecar_manifest


def _app() -> ContextApp:
    provider, model = example_provider()
    return ContextApp(name="deliverables", provider=provider, model=model)


MEMO = """# Q2 Board Memo

## Summary

Revenue grew 30% year over year [E1]. Operating costs fell materially [E2].

## Outlook

Full-year guidance is unchanged [E1].

| Metric | Q1 | Q2 |
| --- | --- | --- |
| Revenue ($M) | 10 | 13 |
| Margin (%) | 22 | 27 |
"""

EVIDENCE = [
    EvidenceItem(id="E1", source_id="10-Q", page=4, trust_level=TrustLevel.UNTRUSTED_DOCUMENT,
                 text="Revenue grew 30% year over year and full-year guidance is unchanged."),
    EvidenceItem(id="E2", source_id="earnings-call", trust_level=TrustLevel.USER,
                 text="Operating costs fell materially this quarter."),
]


def document_demo() -> None:
    print("== 1. Document generation with a structural contract ==")
    app = _app()
    contract = DocumentContract(
        required_sections=["Summary", "Outlook"],
        table_specs=[TableSpec(required_columns=["Metric", "Q1", "Q2"], min_rows=2)],
        min_words=20,
    )
    md = app.build_document(MEMO, format="markdown", contract=contract)
    html = app.build_document(MEMO, format="html")
    print(f"  rendered markdown ({len(md.content)}B) + html ({len(html.content)}B)")
    print(f"  document_generate audit events: "
          f"{sum(1 for e in app.audit.entries if e.action == 'document_generate')}")
    try:
        app.build_document("# Tiny\n\ntoo short", format="markdown",
                           contract=DocumentContract(required_sections=["Appendix"]))
    except Exception as exc:  # noqa: BLE001
        print(f"  deficient document correctly rejected: {type(exc).__name__}")


def cited_report_demo() -> None:
    print("\n== 2. Cited report — resolved citations + per-claim entailment ==")
    app = _app()
    art = app.cited_report(
        "Revenue grew 30% [E1]. Operating costs fell materially [E2].",
        EVIDENCE,
        format="markdown",
        contract=CitationContract(min_coverage=1.0, require_entailment=True, min_entailment_rate=0.5),
    )
    text = art.text
    print("  inline markers resolved to numbered footnotes:", "[1]" in text and "Notes" in text)
    note = next((line for line in text.splitlines() if "[1]" in line and "10-Q" in line), "")
    print("  footnote:", note.strip())
    print("  has bibliography:", "Sources" in text)


def redline_demo() -> None:
    print("\n== 3. Redline (DOCUMENT_COMPARISON → tracked changes) ==")
    art = generate_redline(
        "The initial term is 24 months.",
        "The initial term is 36 months with auto-renewal.",
        format="markdown",
    )
    print("  ", art.text.splitlines()[-1])


async def media_demo() -> None:
    print("\n== 4. Image generation & TTS — provenance + budget + audit ==")
    app = _app()
    image_resp = await app.agenerate_image(
        ImageGenRequest(prompt="a minimalist revenue growth chart, blue palette"),
        provider=MockImageProvider(),
    )
    image = image_resp.images[0]
    print(f"  image: {len(image.data)}B PNG, cost ${image_resp.cost_usd:.4f}, "
          f"provenance verifies: {verify_manifest(image.manifest, image.data)}")
    print(f"  digital source type: {image.manifest.digital_source_type.rsplit('/', 1)[-1]}")

    speech_resp = await app.asynthesize_speech(
        "Revenue grew thirty percent year over year.",
        provider=MockSpeechProvider(),
    )
    audio = speech_resp.audio
    print(f"  audio: {len(audio.data)}B {audio.media_type}, "
          f"provenance verifies: {verify_manifest(audio.manifest, audio.data)}")

    # Attach a C2PA sidecar next to the saved asset.
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    image.save(str(out / "chart.png"))
    sidecar = write_sidecar_manifest(out / "chart.png", image.manifest)
    print(f"  content credentials written: {sidecar.name}")
    print(f"  media audit events: "
          f"{[e.action for e in app.audit.entries if e.action in ('image_generate', 'speech_synthesize')]}")


def inputs_demo() -> None:
    print("\n== 5. Richer inputs — new formats, transcripts, forms ==")
    # A dependency-free PPTX (OOXML zip).
    deck = Path("deck.pptx")
    with zipfile.ZipFile(deck, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", "<p><a:t>Quarterly results</a:t></p>")
        archive.writestr("ppt/slides/slide2.xml", "<p><a:t>Outlook unchanged</a:t></p>")
    doc = load_document(deck)
    print(f"  PPTX → {len(doc.sections)} slides, e.g. {doc.sections[0]['text']!r}")

    # Audio transcript ingestion (offline mock transcriber).
    clip = Path("call.wav")
    clip.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    transcript_doc = _app().load_media(str(clip), transcriber=MockTranscriber(diarize=True))
    print(f"  audio → {transcript_doc.metadata['segment_count']} timestamped segments")

    # Offline KYC/forms extraction.
    fields = HeuristicFormExtractor().extract(
        "Name: Jordan Lee\nDate of Birth: 1988-04-02\nID Number: X1234567\nTotal: $4,200"
    )
    print(f"  KYC fields: {[(f.name, f.value) for f in fields if f.confidence > 0.8]}")


def conformity_demo() -> None:
    print("\n== 6. EU AI Act conformity pack — risk tier + Annex IV ==")
    app = _app()
    assessment = app.risk_tier(purpose="credit scoring for loan applications",
                               domains=["creditworthiness"])
    print(f"  risk tier (advisory): {assessment.tier.value} — {len(assessment.obligations)} obligations")
    annex = app.annex_iv(purpose="credit scoring", domains=["creditworthiness"], format="markdown")
    print(f"  Annex IV technical documentation: {len(annex.content)}B, "
          f"sections include 'Risk-management system': {'Risk-management system' in annex.text}")
    fria = app.fria(purpose="credit scoring", affected_groups=["Loan applicants"])
    print(f"  Art. 27 FRIA generated: {len(fria.content)}B")
    print(f"  conformity_doc audit events: "
          f"{sum(1 for e in app.audit.entries if e.action == 'conformity_doc')}")


def main() -> None:
    document_demo()
    cited_report_demo()
    redline_demo()
    asyncio.run(media_demo())
    inputs_demo()
    conformity_demo()
    print("\nDocuments and media flow OUT under the same guarantees as text IN — "
          "cited, provenance-stamped, budget-metered, audited, on one chain.")


if __name__ == "__main__":
    main()
