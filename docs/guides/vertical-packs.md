# Vertical packs: a regulated domain in one line

A [pack](plugins.md) is an opt-in, dependency-free bundle you apply with
`app.use_pack(...)`. Two tiers ship in the box:

- **Domain packs** — `support`, `engineering`, `finance`, `legal`. A light prompt
  + schema + policy starting point for a domain.
- **Vertical packs** — `healthcare`, `ediscovery`, `kyc`, `customer_support`,
  `code_review`. A *full-stack* configuration for a regulated or high-stakes use
  case: on top of the prompt, schema, policies, deterministic rails, and domain
  metrics, a vertical also preconfigures **retrieval**, **scoped memory**, and a
  **data-residency posture**, and ships a larger **golden eval set**.

Everything is wired through the public `ContextApp` API (`configure`,
`set_policy`, `add_evaluator`, `add_rail`, `add_memory`, `set_residency`), so a
pack never reaches past the contract it documents and you can layer your own
settings on top.

```python
from vincio import ContextApp

app = ContextApp(name="kyc_desk").use_pack("kyc")
app.add_source("cases", path="./due-diligence")   # bring your own corpus
result = app.run("Screen this customer against sanctions and adverse media.")
```

## What a vertical configures

| Vertical | Schema | Rails | Memory | Residency |
|---|---|---|---|---|
| `healthcare` | `clinical_answer` (`phi_detected`, `needs_clinician`) | PHI redact + secrets (output) | user-scoped | `us` |
| `ediscovery` | `ediscovery_review` (`responsive`, `privileged`, `privilege_basis`) | secrets (output) | team-scoped | `us` |
| `kyc` | `kyc_assessment` (`risk_rating`, `sanctions_hit`, `pep`, `sar_recommended`) | PII redact + secrets (output) | user-scoped | `us` |
| `customer_support` | `support_resolution` (`category`, `priority`, `resolution_steps`) | PII redact + secrets (output) | user-scoped | — |
| `code_review` | `code_review` (`findings[]`, `security_risk`, `approve`) | secrets (output) | team-scoped | — |

Each also sets retrieval knobs suited to the domain (e.g. `sentence_window`
chunking for clinical notes, `parent_document` for long litigation records,
`code_aware` for diffs) and ships a golden eval set you can gate quality on from
day one:

```python
from vincio import load_pack
from vincio.evals import EvalRunner

pack = load_pack("healthcare")
report = EvalRunner(app).run(pack.dataset())   # the pack's golden set
report.print_summary()
```

## PII / PHI redaction on structured output

A vertical's `redact` rail masks detected identifiers in the deliverable —
including the **string fields of a structured output**, not just free text — so a
`clinical_answer` or `kyc_assessment` never ships an SSN or account number the
rail caught. The schema and field types are preserved; the raw model emission is
left intact on the trace (trace content capture is
[off by default](../guides/cost-and-reliability.md)).

## Residency, offline-first

A residency-pinned vertical (`healthcare`, `ediscovery`, `kyc`) applies
`set_residency([...region, "on_prem"], deny_on_unknown=False)`. Self-hosted /
in-process processing is in jurisdiction by construction, so `on_prem` is always
admitted and the dependency-free offline path still runs. The pack fails *open*
on an unknown region rather than hard-blocking; the strict in-jurisdiction
posture comes from pinning a region-bearing endpoint (then the region is always
known) and tightening with `app.set_residency(..., deny_on_unknown=True)`. Even
fail-open, an *identifiable* out-of-jurisdiction endpoint is still refused egress.

The `purpose` field on a vertical (e.g. `treatment`, `legal_obligation`) is
advisory metadata — pair it with a [`ConsentLedger`](governance.md) to enforce a
GDPR lawful basis.

## Writing your own vertical

Construct a `Pack` with the vertical fields and `register_pack` it:

```python
from vincio.packs import Pack, register_pack

register_pack(Pack(
    name="claims",
    description="Insurance claims adjudication.",
    role="claims adjudicator",
    objective="Decide the claim from the policy and the submitted evidence.",
    rules=["Decide only from the policy text and the claim file.", "Cite each clause."],
    output_schema={...},
    evaluators=["groundedness", "schema_validity", "citation_accuracy"],
    rails=[{"name": "pii_redact", "kind": "safety", "direction": "output",
            "detectors": ["pii"], "action": "redact"}],
    retrieval={"mode": "hybrid", "chunking": "sentence_window", "top_k": 10},
    memory={"scope": "user", "strategy": "semantic"},
    residency=["us"],
    eval_cases=[...],   # your golden set
))
```

See [`examples/42_vertical_packs.py`](../../examples/42_vertical_packs.py) for a
runnable tour, and the [cookbook](cookbook.md) for task-shaped recipes built on
these packs.
