"""Structured output, guardrails & reliability (0.7).

Typed signatures, constrained decoding, streaming validation, programmable
rails, bounded self-correction, and multi-schema routing — reliability as a
guarantee, not a hope.
"""

import asyncio

from _shared import example_provider
from pydantic import BaseModel

from vincio import ContextApp, InputField, OutputField, Signature

provider, model = example_provider()


# --- 1. Multi-schema routing: one app, several output shapes ---------------


class BugReport(BaseModel):
    title: str
    severity: str
    component: str


class BillingIssue(BaseModel):
    invoice_id: str
    amount: float
    action: str


app = ContextApp(name="reliable_intake", provider=provider, model=model)
app.configure(objective="Convert support messages into structured records")
app.add_output_schema(BugReport, keywords=["bug", "crash", "error"])
app.add_output_schema(BillingIssue, keywords=["invoice", "refund", "charge"])

# --- 2. Rails: deterministic input/output guardrails ------------------------

app.add_rail(name="stay_on_topic", kind="topic", direction="input",
             blocked_topics=["legal advice", "medical advice"])
app.add_rail(name="no_leaked_pii", kind="safety", direction="output",
             action="redact", detectors=["pii"])
app.register_rail_predicate(
    "too_long", lambda text, params: "answer too long" if len(text) > params["limit"] else None
)
app.add_rail(name="bounded_answer", kind="custom", direction="output",
             predicate="too_long", params={"limit": 4000})

# --- 3. Self-correction: bounded validate → critique → repair ---------------

app.enable_self_correction(max_cycles=2, max_cost_usd=0.05)


# --- 4. Typed signatures (DSPy-style) ---------------------------------------


class Summarize(Signature):
    """Summarize a support thread in one sentence."""

    thread: str = InputField(desc="the full support thread")
    summary: str = OutputField(desc="one-sentence summary")
    sentiment: str = OutputField(desc="positive | neutral | negative")


async def main() -> None:
    # Constrained decoding: the schema rides the provider's native
    # structured-output path (strict-sanitized), and the run's trace records
    # which decoding mode was used.
    result = await app.arun("The app crashes whenever I export a report")
    print(f"routed to: {type(result.output).__name__}")
    print(f"output: {result.output}")

    result = await app.arun("Please refund invoice INV-2041 for $129")
    print(f"routed to: {type(result.output).__name__}")

    # A rail violation denies the run before the model is ever called.
    denied = await app.arun("Can you give me legal advice about my lease?")
    print(f"rail verdict: {denied.status.value} — {denied.error}")

    # Streaming validation: partial output is parsed and prefix-checked as
    # it streams; valid_prefix flips to False the moment the output cannot
    # match the schema anymore, so callers can abort early.
    async for event in app.astream("There is a bug in the billing export"):
        if event.type == "partial_output":
            print(f"partial (valid_prefix={event.valid_prefix}): {event.partial_output}")
        elif event.type == "done":
            print(f"final status: {event.result.status.value}")

    # Signatures: typed input → output programs over the prompt AST.
    summarize = app.predictor(Summarize)
    outcome = await summarize.acall(thread="Customer reported double billing; resolved with credit.")
    print(f"summary: {outcome.summary!r}  sentiment: {outcome.sentiment!r}")

    # Every signature is an optimization target: its PromptSpec feeds the
    # same optimizer as any hand-written prompt.
    spec = Summarize.to_prompt_spec()
    print(f"optimizable spec: {spec.name} (schema: {spec.output_schema is not None})")

    # Interconnection: validation failures/repairs are audit entries.
    validations = [e for e in app.audit.entries if e.action == "output_validation"]
    print(f"audited validation events: {len(validations)}")


if __name__ == "__main__":
    asyncio.run(main())
