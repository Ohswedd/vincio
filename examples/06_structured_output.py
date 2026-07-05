"""Reliable structured output — from "usually JSON" to a guarantee.

How Vincio makes typed output dependable: a Pydantic model IS the contract,
provider-native constrained decoding turns the schema into a hard grammar,
streaming validation aborts a doomed generation mid-stream, a bounded repair
loop fixes STRUCTURE ONLY (it never invents facts), one app routes many output
shapes, and DSPy-style typed Signatures compile programs instead of prompts.
Runs fully offline on the deterministic mock provider.
"""

from __future__ import annotations

import asyncio

from _shared import example_provider
from pydantic import BaseModel, Field

from vincio import ContextApp, InputField, OutputField, Signature
from vincio.output.constrained import DecodingMode, negotiate_decoding


class BugReport(BaseModel):
    title: str
    severity: str = Field(description="low | medium | high | critical")
    component: str


class BillingIssue(BaseModel):
    invoice_id: str
    amount: float
    action: str = Field(description="refund | credit | investigate")


class StreamedAnswer(BaseModel):
    answer: str
    confidence: float  # a number — a string here is a definite schema violation


class Record(BaseModel):
    name: str
    score: int  # an integer; the word "high" can't be coerced, forcing a re-prompt


class Triage(Signature):
    """Triage a support ticket into a label and urgency."""

    # InputField / OutputField mark direction; the docstring is the instruction.
    ticket: str = InputField(desc="the raw ticket text")
    label: str = OutputField(desc="bug | billing | feature | other")
    urgency: str = OutputField(desc="low | high")


async def main() -> None:
    # 1. negotiate_decoding picks the strongest enforcement for this provider +
    #    contract. If the provider enforces a JSON schema natively (the mock does,
    #    as real structured-output models do), the schema becomes a hard grammar
    #    during decoding — NATIVE means it literally cannot emit a bad token.
    #    Otherwise the schema is rendered into the prompt + a repair parser (PROMPT).
    caps = (provider := example_provider()[0]).capabilities("mock-1")
    print("1. constrained decoding: with schema", negotiate_decoding(caps, BugReport.model_json_schema()),
          "| without", negotiate_decoding(caps, None))
    assert negotiate_decoding(caps, BugReport.model_json_schema()) is DecodingMode.NATIVE

    # 2. Multi-schema routing: one app, several output shapes. Each schema declares
    #    the keywords that route an inbound message to it (first match wins). Use
    #    this to converge a mixed intake stream into the right typed record.
    app = ContextApp(name="intake", provider=provider, model="mock-1")
    app.configure(objective="Convert support messages into structured records")
    app.add_output_schema(BugReport, keywords=["bug", "crash", "error", "export"])
    app.add_output_schema(BillingIssue, keywords=["invoice", "refund", "charge"])
    bug = await app.arun("The app crashes whenever I export a report")
    billing = await app.arun("Please refund invoice INV-2041")
    print(f"2. routing: crash -> {type(bug.output).__name__}, refund -> {type(billing.output).__name__}")

    # 3. Streaming validation checks the partial JSON as it arrives and bails the
    #    instant it's doomed. Each partial_output carries valid_prefix: True while
    #    the bytes COULD still complete validly, False on a definite mismatch (a
    #    wrong type on an arrived field). Seeing False, a real client cancels the
    #    request instead of paying for the rest of an answer it will reject.
    def bad_stream(_request):  # confidence arrives as a string — unfixable
        return ('{"answer": "the refund window is thirty calendar days for the pro plan", '
                '"confidence": "very high indeed"}')

    stream_app = ContextApp("streaming", provider=example_provider(bad_stream)[0],
                            model="mock-1", output_schema=StreamedAnswer)
    aborted = False
    async for event in stream_app.astream("What is the refund window?"):
        if event.type == "partial_output" and event.valid_prefix is False:
            aborted = True
            break
    print("3. streaming abort on a definite type mismatch:", aborted)

    # 4. Bounded self-correction: validate -> critique -> repair, hard-capped by
    #    cycles and cost. When the first reply fails in a way the structural parser
    #    can't fix (score came back as the word "high"), the corrector re-prompts
    #    with the specific errors. It is STRUCTURE-ONLY — the critique forbids
    #    changing facts, so it fixes the shape, never invents data.
    repair_app = ContextApp("repair", model="mock-1", output_schema=Record,
                            provider=example_provider(script=['{"name": "alpha", "score": "high"}',
                                                              '{"name": "alpha", "score": 7}'])[0])
    repair_app.enable_self_correction(max_cycles=2, max_cost_usd=0.05)
    result = await repair_app.arun("Emit a record")
    cycles = [e.details.get("correction_cycles") for e in repair_app.audit.entries
              if e.action == "output_validation"]
    print(f"4. self-correction: {result.status.value} -> {result.output} (cycles {cycles})")

    # 5. Typed Signatures / Predict compile a program from a class, not a prompt
    #    string. The call is keyword-typed (unknown/missing inputs raise) and the
    #    result exposes the declared output fields as attributes. A signature is
    #    also an optimization target: it lowers to the same PromptSpec the optimizer
    #    rewrites, so DSPy-style programs share Vincio's tuning machinery.
    triage = ContextApp("triage", provider=provider, model="mock-1").predictor(Triage)
    out = await triage.acall(ticket="App crashes on export of large reports")
    spec = Triage.to_prompt_spec()
    print(f"5. signature: label={out.label!r} urgency={out.urgency!r} | "
          f"optimizable spec {spec.name!r} carries schema={spec.output_schema is not None}")


if __name__ == "__main__":
    asyncio.run(main())
