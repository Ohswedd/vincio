"""Reliable structured output.

How Vincio turns "the model usually returns JSON" into a guarantee:
Pydantic output contracts, provider-native constrained decoding, streaming
validation that aborts a doomed generation mid-stream, a bounded
self-correcting repair loop (structure-only — it never invents facts),
multi-schema routing on one app, and DSPy-style typed Signatures / Predict.

Runs fully offline on the deterministic mock provider — no API keys, no network.
"""

from __future__ import annotations

import asyncio

from _shared import example_provider
from pydantic import BaseModel, Field

from vincio import ContextApp, InputField, OutputField, Signature
from vincio.output.constrained import DecodingMode, negotiate_decoding


def banner(title: str) -> None:
    print(f"\n== {title} ==")


# --- Output contracts ------------------------------------------------------
# A Pydantic model IS the contract: its JSON schema rides the provider's
# structured-output path, and the model's reply is parsed/validated back into
# an instance of it. Field constraints (ge/le, enums) are enforced post-hoc.


class BugReport(BaseModel):
    """A triaged engineering bug."""

    title: str
    severity: str = Field(description="low | medium | high | critical")
    component: str


class BillingIssue(BaseModel):
    """A billing/refund action item."""

    invoice_id: str
    amount: float
    action: str = Field(description="refund | credit | investigate")


# --- 1. Constrained decoding negotiation -----------------------------------
def section_constrained_decoding() -> None:
    """The strongest enforcement available is chosen per provider + contract.

    ``negotiate_decoding`` inspects the model's capabilities. If the provider
    enforces a JSON schema natively (the mock advertises this, as real
    structured-output models do), the schema becomes a hard grammar during
    decoding — NATIVE. Otherwise the schema is rendered into the prompt and a
    robust parser + repair cleans up the reply — PROMPT. With no schema at all
    it is NONE. The same code path runs offline and against a real model.
    """
    banner("Constrained decoding — pick the strongest enforcement")
    provider, model = example_provider()
    caps = provider.capabilities(model)

    native = negotiate_decoding(caps, BugReport.model_json_schema())
    none = negotiate_decoding(caps, None)
    print(f"  with a schema:    {native}  (provider enforces the grammar)")
    print(f"  without a schema: {none}")
    # NATIVE means the model literally cannot emit a non-conforming token.
    assert native is DecodingMode.NATIVE


# --- 2. Multi-schema routing ----------------------------------------------
def build_intake_app() -> ContextApp:
    """One app, several output shapes; the right schema is chosen per message."""
    provider, model = example_provider()
    app = ContextApp(name="intake", provider=provider, model=model)
    app.configure(objective="Convert support messages into structured records")
    # Each schema declares the keywords that route an inbound message to it.
    # First match wins; the app's base schema (if any) is the fallback.
    app.add_output_schema(BugReport, keywords=["bug", "crash", "error", "export"])
    app.add_output_schema(BillingIssue, keywords=["invoice", "refund", "charge"])
    return app


async def section_multi_schema_routing() -> None:
    banner("Multi-schema routing — one app, many output shapes")
    app = build_intake_app()

    bug = await app.arun("The app crashes whenever I export a report")
    print(f"  '...crashes on export' -> {type(bug.output).__name__}: {bug.output}")

    billing = await app.arun("Please refund invoice INV-2041")
    print(f"  '...refund invoice...'  -> {type(billing.output).__name__}: {billing.output}")


# --- 3. Streaming validation with mid-stream early abort -------------------
class StreamedAnswer(BaseModel):
    answer: str
    confidence: float  # a number — a string here is a definite schema violation


async def section_streaming_early_abort() -> None:
    """Validate the partial JSON as it streams and bail the moment it's doomed.

    Each ``partial_output`` event carries ``valid_prefix``: True while the
    bytes seen so far *could* still complete into a valid instance, and False
    the instant they can't (a wrong type on an arrived field, an unknown field
    on a closed object). Strings are never length/enum-checked mid-stream
    (they may be prefixes) — only definite, unfixable mismatches flip it.

    A caller that sees ``valid_prefix is False`` can stop generation early
    instead of paying for the rest of an answer it will reject anyway.
    """
    banner("Streaming validation — abort a doomed generation mid-stream")

    # This responder violates the contract: confidence arrives as a string.
    def bad_stream(_request: object) -> str:
        return (
            '{"answer": "the refund window is thirty calendar days for the '
            'pro plan", "confidence": "very high indeed"}'
        )

    provider, model = example_provider(bad_stream)
    app = ContextApp("streaming", provider=provider, model=model, output_schema=StreamedAnswer)

    aborted = False
    async for event in app.astream("What is the refund window?"):
        if event.type == "partial_output":
            print(f"  partial valid_prefix={event.valid_prefix}: {event.partial_output}")
            if event.valid_prefix is False:
                # In a real client this is where you cancel the request.
                aborted = True
                break
        elif event.type == "done":
            print(f"  final status: {event.result.status.value}")
    print(f"  aborted early on a definite mismatch: {aborted}")


# --- 4. Bounded self-correcting repair loop --------------------------------
class Record(BaseModel):
    name: str
    score: int  # an integer; "high" can't be coerced, forcing a re-prompt


async def section_self_correction() -> None:
    """Validate -> critique -> repair, bounded by cycles and cost.

    When the first reply fails validation in a way the structural parser can't
    fix on its own (here ``score`` came back as the word "high"), the corrector
    re-prompts the model with the specific errors. Crucially it is
    STRUCTURE-ONLY: the critique forbids changing factual content, so it fixes
    the shape, never invents data. The loop is hard-capped (max_cycles /
    max_cost_usd) so a stubborn model can't spin forever.

    We script two replies: an invalid one, then a valid one — proving the loop
    converged rather than the model getting lucky on the first try.
    """
    banner("Bounded self-correction — repair structure, never invent facts")
    provider, model = example_provider(
        script=[
            '{"name": "alpha", "score": "high"}',  # invalid: wrong type
            '{"name": "alpha", "score": 7}',       # repaired on the next cycle
        ]
    )
    app = ContextApp("repair", provider=provider, model=model, output_schema=Record)
    app.enable_self_correction(max_cycles=2, max_cost_usd=0.05)

    result = await app.arun("Emit a record")
    print(f"  status: {result.status.value}  output: {result.output}")

    # The validation outcome and every repair cycle land in the audit chain.
    events = [e for e in app.audit.entries if e.action == "output_validation"]
    for e in events:
        print(f"  audit decision={e.decision} cycles={e.details.get('correction_cycles')}")


# --- 5. DSPy-style typed Signatures / Predict ------------------------------
class Triage(Signature):
    """Triage a support ticket into a label and urgency."""

    # InputField / OutputField mark the direction; the docstring is the
    # instruction. Predict turns the signature into a typed callable program.
    ticket: str = InputField(desc="the raw ticket text")
    label: str = OutputField(desc="bug | billing | feature | other")
    urgency: str = OutputField(desc="low | high")


async def section_signatures() -> None:
    """Typed input -> output programs, compiled from a class, not a prompt string."""
    banner("Typed Signatures / Predict — programs over prompts")
    provider, model = example_provider()
    app = ContextApp("triage", provider=provider, model=model)

    # app.predictor binds the signature to the app's provider/model. The call
    # is keyword-typed (unknown/missing inputs raise) and the result exposes
    # the declared output fields as attributes.
    triage = app.predictor(Triage)
    out = await triage.acall(ticket="App crashes on export of large reports")
    print(f"  predicted label={out.label!r} urgency={out.urgency!r}")

    # A signature is also an optimization target: it compiles to the same
    # PromptSpec / prompt AST that the optimizer rewrites for any prompt.
    spec = Triage.to_prompt_spec()
    print(f"  optimizable spec: {spec.name} (carries schema: {spec.output_schema is not None})")


async def main() -> None:
    section_constrained_decoding()
    await section_multi_schema_routing()
    await section_streaming_early_abort()
    await section_self_correction()
    await section_signatures()
    print("\nStructured output as a guarantee: typed, constrained, streamed, repaired.")


if __name__ == "__main__":
    asyncio.run(main())
