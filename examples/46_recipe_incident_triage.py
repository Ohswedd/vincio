"""Cookbook recipe — incident triage.

Turn an alert plus runbook excerpts into a typed, grounded triage decision:
severity, affected component, and the next mitigation step — answered only from
the attached runbooks, so the on-call action is cited, not improvised.
"""

from pydantic import BaseModel, Field

from _shared import example_provider, json_responder

from vincio import ContextApp
from vincio.core.types import Document


class Triage(BaseModel):
    severity: str = Field(description="sev1 | sev2 | sev3")
    component: str
    mitigation: str
    runbook_ref: str


decision = {
    "severity": "sev2",
    "component": "checkout-api",
    "mitigation": "Roll back the latest checkout-api deploy and drain the bad pods.",
    "runbook_ref": "RB-17",
}
provider, model = example_provider(json_responder(decision))

app = ContextApp(name="incident_triage", provider=provider, model=model, output_schema=Triage)
app.add_source("runbooks", documents=[
    Document(id="RB-17", title="Checkout incidents",
             text="If checkout-api error rate exceeds 5%, roll back the latest deploy "
                  "and drain affected pods. Page the payments on-call for sev1."),
])
app.set_policy("answer_only_from_sources", False)


if __name__ == "__main__":
    alert = "PagerDuty: checkout-api 5xx error rate at 12% for 4 minutes after a deploy."
    result = app.run(alert)
    out = result.output
    out = out.model_dump() if hasattr(out, "model_dump") else out
    print("severity :", out["severity"])
    print("component:", out["component"])
    print("mitigate :", out["mitigation"])
    print("runbook  :", out["runbook_ref"])
    print("trace    :", result.trace_id)
