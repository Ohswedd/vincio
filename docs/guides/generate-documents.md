# Generate documents & media (`vincio.generation`)

> **Experimental (1.9).** Additive behind a new `vincio.generation` subpackage and
> opt-in extras on the frozen 1.0 API. Symbols are marked `@experimental`.

Vincio reads a DOCX, a PDF, and a scanned packet, and validates a JSON answer.
1.9 closes the loop so the **deliverable** comes out under the same guarantees
Vincio applies to text *in*: cited, structurally-validated, provenance-stamped,
budget-metered, eval-gated artifacts — all on one trace and one audit chain,
in-process and never a service.

Install the renderer extras you need (Markdown and HTML are dependency-free):

```bash
pip install "vincio[gen-docx]"   # DOCX (python-docx)
pip install "vincio[gen-pdf]"    # PDF (reportlab)
pip install "vincio[gen-pptx]"   # PPTX (python-pptx)
```

## Document generation engine

`DocumentBuilder` turns a **validated result** — an `OutputContract` output, a
`RunResult`, a structured mapping, or Markdown — into a rendered artifact.
Because the input already passed validation, the document is grounded by
construction; the builder lays it out and never invents content.

```python
from vincio.generation import DocumentBuilder, DocumentContract, TableSpec

builder = DocumentBuilder(audit_log=app.audit)  # or app.build_document(...)

artifact = builder.build(
    "# Q2 Board Memo\n\n## Summary\n\nRevenue grew 30% [E1].\n\n"
    "| Metric | Q1 | Q2 |\n| --- | --- | --- |\n| Revenue | 10 | 13 |\n",
    format="markdown",                 # markdown | html | docx | pdf | pptx
    contract=DocumentContract(
        required_sections=["Summary"],
        table_specs=[TableSpec(required_columns=["Metric", "Q1"], min_rows=1)],
        min_words=10,
        citations_per_section=True,
    ),
)
print(artifact.text)        # textual formats; .content for binary
artifact.save("memo.md")
```

A `DocumentContract` is to a document what an `OutputContract` is to model text.
Repair is **formatting-only** (the document mirror of the JSON-repair path): it
normalizes heading levels, derives a missing title, and trims whitespace, but a
missing required section or an uncited section is a `DocumentContractError`, never
silently padded. Every render records a `document_generate` audit event carrying
the source evidence ids.

**Template / form filling** keeps the same grounding guarantee for fixed forms —
a `{{slot}}` marked `must_cite` must carry a valid citation:

```python
from vincio.generation import Slot, fill_text_template, fill_docx_form, fill_pdf_form

fill_text_template("Total due: {{total}}", {"total": 4200},
                   slots=[Slot(name="total", type="currency")])  # "Total due: $4,200.00"
```

**Redlines** pair the `DOCUMENT_COMPARISON` intent with tracked-change output
(visual insertions/deletions in DOCX/PDF, `**ins**`/`~~del~~` in Markdown/HTML):

```python
from vincio.generation import generate_redline
generate_redline(original_text, revised_text, format="docx").save("redline.docx")
```

## Cited reports

`CitedReportBuilder` resolves inline `[E1]`-style markers to numbered
footnotes/endnotes and a generated bibliography, computes **sentence-level
citation coverage**, and optionally verifies **per-claim entailment** — replacing
the flat "one valid citation anywhere" check with "every claim cited *and*
supported".

```python
report = await app.acited_report(
    "Revenue grew 30% [E1]. Costs fell [E2].",
    evidence,                          # the EvidenceItems that grounded the answer
    format="markdown",
    contract=CitationContract(min_coverage=1.0, require_entailment=True),
)
# Or synchronously: app.cited_report(answer, evidence, ...)
```

A `CitationContract` enforces a coverage floor, rejects unresolved markers, and
(when `require_entailment=True`) requires the cited evidence to support each
claim. The entailment backend defaults to a strict lexical+numeric check; pass
your own `entailment=callable` (sync or async) to plug in an NLI model or judge.

## Image generation & TTS

Image generation/editing and speech synthesis are first-class output modalities.
Every asset is C2PA-stamped (provenance bound to its bytes), metered against the
budget, and audited (`image_generate` / `speech_synthesize`).

```python
from vincio.generation import MockImageProvider, MockSpeechProvider
from vincio.governance import verify_manifest

resp = await app.agenerate_image("a revenue chart", provider=MockImageProvider())
image = resp.images[0]
image.save("chart.png")                # carries an embedded C2PA manifest (PNG)
assert verify_manifest(image.manifest, image.data)

speech = await app.asynthesize_speech("Revenue grew thirty percent.",
                                      provider=MockSpeechProvider())
speech.audio.save("vo.wav")
```

Real backends: `OpenAIImageProvider` (`gpt-image-1`), `GoogleImageProvider`
(Imagen), `HTTPImageProvider` (Replicate-style); `OpenAISpeechProvider`,
`GoogleSpeechProvider`, `ElevenLabsSpeechProvider`. The `Mock*` providers produce
real PNG/WAV bytes for offline tests.

`mark_synthetic_content` is now **media-aware** — it accepts raw bytes and binds
by SHA-256, marks edits with `compositeWithTrainedAlgorithmicMedia`, and pairs
with `embed_provenance` (PNG metadata, dependency-free) and
`write_sidecar_manifest` (a `*.c2pa.json` for any format).

## Richer inputs

OCR auto-fallback, audio transcripts, new formats, and forms close the
documents-*in* gap so the classifier's promises match the loader's reality:

```python
from vincio.documents import load_document, load_media, MockTranscriber, HeuristicFormExtractor

load_document("deck.pptx")             # PPTX/EPUB/RTF/ODT dependency-free; Parquet via vincio[parquet]
load_pdf("scan.pdf", ocr_engine=ocr)   # low-text pages OCR'd (vincio[ocr]); extractor='ocr' per page
load_media("call.wav", transcriber=MockTranscriber())   # timestamped, diarized transcript Document
HeuristicFormExtractor().extract("Name: Jane\nTotal: $5")  # offline KYC/invoice key-values
```

Formats now register through a `ParserRegistry` (`register_loader(...)`), so new
formats add additively instead of editing a suffix chain. HTML parses with a real
structural path (table extraction); JSON/JSONL/YAML structure into
sections/tables; PDF figure regions become citable evidence via `figure_evidence`.

## EU AI Act conformity pack

The document engine's first governance application: a `RiskTierClassifier` places
a configured app into the Act's risk tiers, and `AnnexIVBuilder` / `FRIAGenerator`
render **Annex IV technical documentation** and the **Article 27 FRIA** as cited
documents — every field drawn from the live config, the model/system cards, the
compliance matrix, and the eval/red-team evidence Vincio already holds, so they
are grounded by construction and regenerate on every config change.

```python
app.risk_tier(purpose="credit scoring", domains=["creditworthiness"]).tier   # high_risk
app.annex_iv(purpose="credit scoring", domains=["creditworthiness"]).save("annex_iv.md")
app.fria(purpose="credit scoring", affected_groups=["Loan applicants"]).save("fria.md")
```

The classification is **advisory** — the operator makes the final call. ISO/IEC
42001 controls join the `ComplianceMapper` family; the pack is recorded as a
`conformity_doc` audit event.

See `examples/33_documents_and_media_out.py` for an end-to-end, offline run.
