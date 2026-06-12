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

When the provider supports native structured output (OpenAI json_schema,
Anthropic forced tool use, Gemini responseSchema), Vincio uses it and omits
the schema from the prompt text. Otherwise the schema is rendered into the
prompt and the output is parsed robustly (fenced JSON, lenient repair).

## The validation pipeline

parse → schema_validate → semantic_validate → citation_validate →
policy_validate → repair_if_allowed → final_validate

Inspect it per run: `result.validation["steps"]`.

## Repair policy — what may and may not be fixed

Allowed: malformed JSON, missing optional fields, safe type coercion
("0.9" → 0.9), markdown formatting.
**Never repaired:** factual claims, unsafe content, missing required
evidence, failed business rules — those fail validation loudly.

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

## Streaming partial output

`vincio.output.parse_partial_json(text)` balances truncated JSON during
streaming so UIs can render structured output incrementally.
