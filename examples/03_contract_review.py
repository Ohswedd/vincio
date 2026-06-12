"""Full contract review app."""

import json
import tempfile
from pathlib import Path

from _shared import example_provider, write_sample_docs
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


def responder(request):
    import re

    text = "\n".join(m.text for m in request.messages)
    refs = re.findall(r"\[([\w.:-]+:C\d+)\]", text)[:2] or ["E1"]
    return json.dumps(
        {
            "summary": f"Auto-renewal and long initial term present commercial risk. [{refs[0]}]",
            "risks": [
                {
                    "clause": "auto-renewal",
                    "risk_level": "high",
                    "explanation": "Renews unless terminated 60 days before renewal.",
                    "evidence_ids": [refs[0]],
                },
                {
                    "clause": "initial term",
                    "risk_level": "medium",
                    "explanation": "24-month lock-in.",
                    "evidence_ids": [refs[-1]],
                },
            ],
            "missing_information": ["liability cap", "payment terms detail"],
        }
    )


docs_dir = write_sample_docs(Path(tempfile.mkdtemp()) / "contracts")
provider, model = example_provider(responder)

app = ContextApp(name="contract_review", output_schema=ContractRiskReport, provider=provider, model=model)
app.configure(
    role="commercial_contract_risk_reviewer",
    objective="Review contracts for commercial and legal risk",
    rules=[
        "Use only provided documents",
        "Cite evidence IDs for every risk",
        "If evidence is missing, report missing_information",
    ],
)
app.add_source("contracts", path=str(docs_dir), chunking="heading_aware", retrieval="hybrid")
app.add_evaluator("schema_validity")
app.add_evaluator("groundedness")
app.add_evaluator("citation_accuracy")

if __name__ == "__main__":
    result = app.run("Find risky renewal, termination, and payment clauses.")
    report = result.output
    print("summary:", report.summary)
    for risk in report.risks:
        print(f"  [{risk.risk_level}] {risk.clause}: {risk.explanation} ({risk.evidence_ids})")
    print("missing:", report.missing_information)
    print("trace:", result.trace_id)
