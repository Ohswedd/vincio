# Enterprise governance & compliance

Regulated buyers ask for evidence: a model card, a control-coverage matrix, an
AI bill of materials, proof you can erase a customer's data, a data-residency
guarantee. Vincio's `vincio.governance` module (1.6) generates that evidence
**in the library, from the running system** — there is no hosted compliance
program. Every artifact is a view over data Vincio already holds: the
hash-chained audit log, the evidence ledger, eval reports, and the price table.

Everything here is additive and opt-in, behind `@experimental` 1.6 entry points
on the frozen 1.0 public API.

## Model & system cards

A **model card** documents one model; a **system card** documents the whole
context-engineering system around it (model + retrieval + memory + safety
filters + human-oversight points). Both are generated from the live config and,
optionally, measured eval evidence — so they cannot drift from what the app does.

```python
from vincio import ContextApp

app = ContextApp("support", model="gpt-5.2-mini")
app.add_source("kb", path="./docs")

model_card = app.model_card()          # pricing pulled from the live table
system_card = app.system_card()        # safety filters + governance controls
print(model_card.to_json())            # native schema
print(model_card.to_dict("open_model_card"))  # or "ai_card"
```

Attach evaluation evidence so the card states *measured* quality:

```python
report = app.evaluate(dataset)         # an EvalReport
card = app.model_card(eval_report=report)   # card.evaluation has the metric means
```

The schema is pluggable (`CardFormat.VINCIO`, `OPEN_MODEL_CARD`, `AI_CARD`)
because no single machine-readable format has won. Set a default with
`governance.card_format` in `vincio.yaml`.

From the CLI: `vincio governance card app.py --kind system --format ai_card`.

## Compliance-framework mapping

`app.compliance_report()` maps Vincio's controls onto four frameworks — **OWASP
LLM Top 10 (2025)**, **OWASP Agentic AI**, **NIST AI RMF (GenAI profile)**, and
**MITRE ATLAS** — and backs each claim with evidence: red-team probe outcomes,
the security configuration, and eval results. An uncovered control is reported
honestly, not hidden in an aggregate.

```python
from vincio.evals.redteam import RedTeamSuite

redteam = RedTeamSuite().run(app)                       # behavioural evidence
report = app.compliance_report(redteam=redteam)
print(report.summary())          # frameworks, controls_total, coverage_rate, gaps
print(report.to_markdown())      # an auditor-ready matrix
```

Each `ControlCoverage` is `covered` / `partial` / `not_covered` with the exact
evidence strings that justify it (`red-team injection: 3/3 probes defended`,
`eval faithfulness=0.95 (≥ 0.7)`, `injection detector enabled`, …).

From the CLI: `vincio governance report app.py --red-team --markdown`.

## AI-BOM & supply chain

The release pipeline already ships a CycloneDX **SBOM** (dependencies) and SLSA
provenance. The **AI-BOM** adds the AI layer: the base model and version, the
embedding and rerank models, fine-tune datasets, and prompt/registry versions —
each with an optional **SHA-256 hash** for blast-radius assessment.

```python
from vincio.governance import AIComponent, sha256_file

bom = app.aibom(
    datasets=[AIComponent(type="data", name="finetune-v3", role="dataset",
                          sha256=sha256_file("finetune-v3.jsonl"))],
)
print(bom.to_json())                       # CycloneDX 1.6 JSON
print(bom.verify_all({"dataset:finetune-v3": "finetune-v3.jsonl"}))  # hash check
```

From the CLI: `vincio governance aibom app.py --output vincio.aibom.cdx.json`.

## EU AI Act transparency

For the 2 Aug 2026 GenAI transparency duties, Vincio supplies the artifacts and
hooks — deadline-agnostic, no signing authority assumed:

```python
from vincio.governance import ai_disclosure, data_summary, mark_synthetic_content

manifest = mark_synthetic_content(result.raw_text, model_id=app.model)  # C2PA-style
print(ai_disclosure(language="es"))        # localized interaction disclosure
print(data_summary(result))                # grounding/training-data summary
```

Enable automatic marking on every run with `governance.content_marking: true`
(or `app.content_marking = True`); the manifest and disclosure are attached to
`result.metadata["content_credentials"]` / `["ai_disclosure"]`.

## Data lineage & erasure-by-source

Vincio records `source → document → chunk → evidence → output` as the app
ingests and runs, so two questions have a mechanical answer:

```python
lineage = app.trace_lineage("kb")          # where did this come from?
result = app.erase_source("kb")            # GDPR right-to-erasure
```

`erase_source` removes the source's chunks from **every index**, its memories,
and its cache entries, then writes an `erase_source` entry to the hash-chained
audit log. It is idempotent — a second call finds nothing left to erase.

From the CLI: `vincio governance lineage app.py kb` / `vincio governance erase app.py kb`.

## Data-residency-aware routing

When a tenant requires in-jurisdiction processing, pin allowed provider regions
and Vincio **refuses egress** to others — deterministically, as a blocking
policy decision on the audit path, before any request leaves the process.

```python
app.set_residency(["eu"], provider_regions={"openai": "us"})
# A run now raises ResidencyViolationError (region 'us' not in {'eu'}),
# recorded as a residency_check deny on the audit log.
```

Or configure it: `governance.allowed_regions: ["eu"]` plus
`governance.provider_regions: {openai: us}`. Vincio can refuse to *send* a
request; it cannot guarantee where a global provider runs it — the control is
egress refusal, which is what an in-jurisdiction policy needs from the client.

## Multilingual PII & the token tax

The built-in PII detector is English/US-centric. Non-English **locale packs**
add national-ID and locale phone formats without changing the English path:

```python
from vincio.security import PIIDetector, available_locales

detector = PIIDetector(locales=["fr", "in", "sg"])   # or governance.locales in config
detector.detect("DNI 12345678Z, PAN ABCDE1234F, NRIC S1234567D")
```

Packs ship for France, Germany, Spain, India, Singapore, Brazil, and the UK.
Configured locales flow through the policy engine automatically.

The **token tax** — non-English text costing more tokens per character — is
tracked per language and tenant by `app.fertility`, so the cost is visible and
routable rather than hidden in an aggregate:

```python
app.fertility.token_tax("ja")     # e.g. 2.1x the English baseline
app.fertility.report()            # per-language and per-tenant fertility
```

Per-language **eval slicing** surfaces the high-vs-low-resource accuracy gap:

```python
report.slice_by_tag("lang:")      # one EvalReport per language
report.tag_gap("accuracy", prefix="lang:")  # best, worst, and the gap
```

## RAG-poisoning detection

A handful of crafted documents can flip many answers. `PoisoningDetector` scans
retrieved evidence and flags likely-poisoned items from **authority/provenance**
signals — embedded instructions, low-authority/high-promotion sources, and
consensus outliers — before they reach the model. An optional async classifier
hook (PromptArmor-class) blends in; the deterministic layers never depend on it.

```python
from vincio.security import PoisoningDetector

report = PoisoningDetector().scan(result.evidence)
print(report.flagged_ids)
print(report.telemetry(poisoned_ids={"bad1"}))   # precision/recall/FP/FN
```

## How it interconnects

Every artifact reads from data Vincio already holds — the audit chain, the
evidence ledger, eval reports, the price table, the prompt registry — so
governance is a *view* over the running system, not a parallel bookkeeping
burden. Residency and erasure are `PolicyViolation`s and audit entries on the
same hash-chained path as every other decision. See the
[threat model](../security/threat-model.md) for the trust boundaries these
controls operate within, and the runnable
[`30_governance_compliance.py`](../../examples/30_governance_compliance.py) example.
