# Guide: reliability & guardrails

0.7 makes reliability a guarantee, not a hope: deterministic rails before
and after every generation, schema-constrained decoding, streaming
validation, bounded self-correction, and typed signatures — all on the same
packet, trace, and audit log as the rest of the pipeline.

## Rails as policies

Rails are NeMo-Guardrails-style programmable guardrails expressed in the
deterministic policy engine. A rail is plain data — kind, direction,
action, parameters — and every check is plain code: no rail depends on
model judgment, so enforcement is exact, explainable, and free.

```python
from vincio import ContextApp

app = ContextApp(name="support")

# topic: deny inputs/outputs that mention (or stray from) given topics
app.add_rail(name="stay_on_topic", kind="topic", direction="input",
             blocked_topics=["legal advice", "medical advice"])
app.add_rail(name="scoped", kind="topic", direction="input",
             allowed_topics=["refund", "invoice", "subscription"])

# format: length and regex constraints on the output
app.add_rail(name="bounded", kind="format", direction="output", max_chars=4000)
app.add_rail(name="ticket_ref", kind="format", direction="output",
             require_pattern=r"TICKET-\d+")

# safety: reuse the security engine's detectors (PII, secrets, injection);
# action="redact" masks instead of blocking
app.add_rail(name="no_leaks", kind="safety", direction="output",
             detectors=["pii", "secrets"], action="redact")

# custom: any registered predicate — (text, params) -> falsy | message
app.register_rail_predicate(
    "max_words", lambda text, p: "too many words" if len(text.split()) > p["limit"] else None
)
app.add_rail(name="brevity", kind="custom", direction="output",
             predicate="max_words", params={"limit": 400})
```

Input rails run before the model is called (a blocking violation denies the
run and is audited); output rails run inside the validation pipeline's
policy step, so a violation fails validation like any other contract breach.
Every rail violation is a `PolicyViolation` named `rail:<name>` — on the
trace, in `result.validation`, and in the hash-chained audit log.

## Self-correcting loops

```python
app.enable_self_correction(max_cycles=2, max_cost_usd=0.05)
```

When validation fails, Vincio runs bounded validate → critique → repair
cycles. Three properties hold by construction:

- **The critique is deterministic** — derived from the `ValidationReport`,
  not model judgment.
- **Facts are never invented** — the repair request is structure-only
  (rename, retype, re-serialize), and semantic/citation/policy validators
  re-run every cycle, so an output whose facts changed still fails.
- **The loop is bounded twice** — by `max_cycles` and by a hard
  `max_cost_usd` ceiling.

The standalone `SelfCorrector` gives the same loop outside the app runtime:

```python
from vincio.output import OutputContract, OutputSchema, OutputValidator, SelfCorrector

schema = OutputSchema.from_pydantic(Invoice)
corrector = SelfCorrector(OutputValidator(OutputContract.from_schema(schema), schema=schema),
                          provider=provider, model="gpt-5.2", max_cycles=2)
outcome = await corrector.correct(raw_text)
outcome.valid, outcome.cycles, outcome.cost_usd, outcome.stopped_reason
```

## Streaming validation

`StreamingValidator` parses the balanced partial JSON as deltas arrive and
prefix-checks it against the schema: tolerant of what hasn't arrived yet,
strict about what definitely cannot match. `app.astream()` wires it in
automatically — `partial_output` events carry `valid_prefix` /
`validation_errors`, so consumers can abort a doomed generation early
instead of paying for the rest of it.

## Typed signatures

Signatures declare what a model call computes; the prompt, schema, and
validation come for free, and every signature is an optimization target:

```python
from vincio import Signature, InputField, OutputField, signature
from vincio.optimize import PromptOptimizer

class Triage(Signature):
    """Classify a support ticket."""
    ticket: str = InputField(desc="the raw ticket text")
    label: str = OutputField(desc="bug | billing | feature | other")
    confidence: float = OutputField()

QA = signature("question, context -> answer, confidence: float")  # string form

predict = app.predictor(Triage)            # or Predict(Triage, provider=..., model=...)
result = predict(ticket="The export 500s") # typed: result.label, result.confidence

# Signatures feed the optimizer: their PromptSpec is a search target like
# any hand-written prompt (formats, examples, reasoning modes, rewrites).
spec = Triage.to_prompt_spec()
report = await PromptOptimizer(evaluate_variant).optimize(spec, dataset)
```

## What lands where (interconnection)

| Event | Trace | Audit log |
|---|---|---|
| Constrained decoding mode | `prompt_render` / `output_validation` span attrs | — |
| Schema route chosen | `prompt_render` span (`schema=`) | — |
| Repair action | `repair` event on the validation span | `output_validation` entry (`decision=repair`) |
| Validation failure | `validation_failed` events + span attrs | `output_validation` entry (`decision=deny`) |
| Self-correction | `self_correction` event (cycles, cost, outcome) | `correction_cycles` in the entry details |
| Input rail block | `policy` span violations | `run` entry (`decision=deny`, `rail:<name>`) |
| Output rail block | `policy` step in `result.validation` | `output_validation` entry |

The VincioBench `reliability` family measures all of it offline — strict
schema closure, mid-stream invalid detection (and the abort savings),
correction recovery rate, rail catch rate with zero false positives,
signature validity, and routing accuracy — under CI-gated budgets.
