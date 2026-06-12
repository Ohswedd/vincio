"""Support triage with typed output."""

from _shared import example_provider, json_responder
from pydantic import BaseModel

from vincio import ContextApp


class TicketClassification(BaseModel):
    label: str
    confidence: float
    reason: str


provider, model = example_provider(
    json_responder({"label": "billing", "confidence": 0.93, "reason": "duplicate charge reported"})
)

app = ContextApp(name="support_triage", output_schema=TicketClassification, provider=provider, model=model)
app.configure(
    role="support_ticket_triage_engine",
    objective="Classify support tickets",
    rules=["Answer with exactly one label: bug, billing, feature, or other."],
)

if __name__ == "__main__":
    result = app.run("I was charged twice this month")
    print(f"label={result.output.label} confidence={result.output.confidence}")
    print(f"reason: {result.output.reason}")
    print(f"trace: {result.trace_id}  cost: ${result.cost_usd:.6f}")
