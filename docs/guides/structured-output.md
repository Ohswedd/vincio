# Guide: structured output

## Define the contract

```python
from pydantic import BaseModel
from vincio import ContextApp

class RiskItem(BaseModel):
    clause: str
    risk_level: str
    explanation: str
    evidence_ids: list[str]

class ContractRiskReport(BaseModel):
    summary: str
    risks: list[RiskItem]
    missing_information: list[str]

app = ContextApp(name="contract_review", output_schema=ContractRiskReport)
result = app.run("Find risky renewal and liability clauses", files=["msa.pdf"])
report: ContractRiskReport = result.output   # validated instance
```

## Constrained generation

When the provider supports native structured output (OpenAI json_schema,
Anthropic forced tool use, Gemini responseSchema), Vincio sends a
**strict-sanitized** schema (`to_strict_json_schema`: every object closed,
every property required, optional fields nullable) so the provider's
constrained decoder can enforce it exactly, and omits the schema from the
prompt text. Otherwise the schema is rendered into the prompt and the output
is parsed robustly (fenced JSON, lenient repair). The decoding mode is
negotiated per run from the provider capability matrix and recorded on the
trace (`decoding=native|prompt`); validation always runs against the
*original* schema.

Grammar-style constraints are JSON schemas too, so they ride the same path:

```python
from vincio.output import choice_schema, regex_schema
app = ContextApp(name="labels", output_schema=choice_schema(["bug", "billing", "other"]))
invoice_id = regex_schema(r"^INV-\d{4}$")   # post-hoc validated everywhere
```

## The validation pipeline

parse → schema_validate → semantic_validate → citation_validate →
policy_validate → repair_if_allowed → final_validate

Inspect it per run: `result.validation["steps"]`.

## Repair policy, what may and may not be fixed

Allowed: malformed JSON, missing optional fields, safe type coercion
("0.9" → 0.9), markdown formatting.
**Never repaired:** factual claims, unsafe content, missing required
evidence, failed business rules, those fail validation loudly.

```python
from vincio.output import RepairPolicy
app.output_contract.repair_policy = RepairPolicy(
    allow_json_repair=True, allow_type_coercion=True,
    allow_fill_optional=True, allow_llm_repair=False,
)
```

## Semantic validators (business rules)

```python
def risk_levels_valid(report, ctx):
    bad = [r.clause for r in report.risks if r.risk_level not in ("low", "medium", "high")]
    return f"invalid risk levels for: {bad}" if bad else None

app.add_validator("risk_levels", risk_levels_valid)
```

## Streaming validation

`vincio.output.parse_partial_json(text)` balances truncated JSON during
streaming so UIs can render structured output incrementally. On top of it,
the `StreamingValidator` prefix-checks the partial output against the
schema as it streams: missing required fields are tolerated (they may still
arrive), but a definite mismatch, wrong type, unknown field on a closed
object, is reported immediately so you can abort generation early.

`app.astream()` does this automatically when a schema is set: every
`partial_output` event carries `valid_prefix` and `validation_errors`.

```python
async for event in app.astream("Extract the invoice"):
    if event.type == "partial_output" and event.valid_prefix is False:
        break  # stop paying for an answer that can no longer be valid
```

## Multi-schema routing

One app can produce different shapes for different tasks. Routes match by
task type, keywords, or a predicate; content-side, `classify`/`validate_any`
find which registered schema some data actually matches:

```python
app.add_output_schema(BugReport, keywords=["bug", "crash"])
app.add_output_schema(BillingIssue, keywords=["invoice", "refund"])
result = app.run("Refund invoice INV-100")     # validates as BillingIssue
```

## Typed signatures

DSPy-style input → output signatures compile to a `PromptSpec` over the
prompt AST, see [reliability & guardrails](reliability-guardrails.md) and
the [DSPy comparison](../comparisons/dspy.md):

```python
from vincio import Signature, InputField, OutputField

class Triage(Signature):
    """Classify a support ticket."""
    ticket: str = InputField(desc="the raw ticket text")
    label: str = OutputField(desc="bug | billing | feature | other")
    confidence: float = OutputField()

result = app.predictor(Triage)(ticket="The export button 500s")
result.label, result.confidence            # typed, schema-validated
```

## Self-correcting loops

`app.enable_self_correction(max_cycles=2, max_cost_usd=0.05)` turns failed
validations into bounded validate → critique → repair cycles. The critique
is built deterministically from the validation report; the repair prompt is
structure-only (facts are never invented), every cycle re-runs all
validators, and the loop stops at the first valid output, `max_cycles`, or
the cost ceiling. Cycles, cost, and outcome land on the trace and in the
audit log.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Example: 06_structured_output.py](../../examples/06_structured_output.py)
- [Concept: Prompt compiler](../concepts/prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
