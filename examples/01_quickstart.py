"""Quickstart: the five-minute tour.

The first file to read. One runnable program that walks the whole core loop:
build a ``ContextApp``, get typed Pydantic output, ground answers in your own
documents with citations, read the trace_id + cost stamped on every result,
shape behaviour with role/objective/rules, and hold a short multi-turn chat.
Everything runs offline on the deterministic mock provider — no keys, no network.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _shared import citing_responder, example_provider, json_responder, write_sample_docs
from pydantic import BaseModel

from vincio import ContextApp


def banner(title: str) -> None:
    """Print a section header so the run reads as a guided tour."""
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 1. Hello, ContextApp — the smallest useful run.
# ---------------------------------------------------------------------------
def section_hello() -> None:
    """A ContextApp wraps a provider+model and runs a single governed turn.

    The mock provider returns a fixed string offline, so the same code that
    would call a real model here is fully deterministic.
    """
    banner("1. Hello, ContextApp")

    # example_provider() returns a deterministic mock unless VINCIO_PROVIDER is
    # set; the responder is the canned reply the mock will produce.
    provider, model = example_provider(lambda req: "Hi! I'm a Vincio app running fully offline.")
    app = ContextApp(name="hello", provider=provider, model=model)

    result = app.run("Say hello.")
    print("answer:", result.output)


# ---------------------------------------------------------------------------
# 2. Typed output — get a validated Pydantic object back, not a string.
# ---------------------------------------------------------------------------
class TicketClassification(BaseModel):
    """The structured shape we want every classification to conform to."""

    label: str
    confidence: float
    reason: str


def section_typed_output() -> None:
    """Passing ``output_schema=`` makes the app parse and validate the model's
    reply into that Pydantic type. ``result.output`` is then a typed object —
    attribute access, IDE completion, and a validation error if the model
    strays off-schema."""
    banner("2. Typed output")

    # json_responder makes the mock emit this exact JSON; with a real provider
    # the schema is sent to the model and the same parsing/validation applies.
    provider, model = example_provider(
        json_responder({"label": "billing", "confidence": 0.93, "reason": "duplicate charge reported"})
    )
    app = ContextApp(
        name="support_triage",
        output_schema=TicketClassification,
        provider=provider,
        model=model,
    )

    result = app.run("I was charged twice this month.")
    out = result.output  # a TicketClassification instance, already validated
    print(f"label={out.label!r}  confidence={out.confidence}")
    print("reason:", out.reason)


# ---------------------------------------------------------------------------
# 3. Configure role / objective / rules — shape behaviour declaratively.
# ---------------------------------------------------------------------------
def section_configure() -> None:
    """``app.configure(...)`` sets the system persona declaratively: who the app
    is (role), what it's for (objective), and the hard constraints it must obey
    (rules). This compiles into the context the model sees on every run, so you
    steer behaviour from config rather than ad-hoc prompt strings."""
    banner("3. Configure role / objective / rules")

    provider, model = example_provider(
        json_responder({"label": "billing", "confidence": 0.88, "reason": "refund request"})
    )
    app = ContextApp(
        name="triage_configured",
        output_schema=TicketClassification,
        provider=provider,
        model=model,
    )
    app.configure(
        role="support_ticket_triage_engine",
        objective="Classify incoming support tickets into one routing label.",
        rules=[
            "Answer with exactly one label: bug, billing, feature, or other.",
            "Never invent account details that were not provided.",
        ],
    )

    result = app.run("Can I get a refund for last month?")
    print("configured app classified as:", result.output.label)
    print("(role/objective/rules are now part of every run's compiled context)")


# ---------------------------------------------------------------------------
# 4. Grounded document QA with citations.
# ---------------------------------------------------------------------------
def section_grounded_qa(docs_dir: Path) -> None:
    """Attach your own documents as a source and the app retrieves, grounds, and
    cites. With ``answer_only_from_sources`` on, the model may only use retrieved
    evidence, and ``result.citations`` reports exactly which chunks backed the
    answer — the core of trustworthy RAG."""
    banner("4. Grounded document QA with citations")

    # citing_responder echoes a real evidence ref ([{ref}]) from the prompt, so
    # the citation pipeline has something concrete to bind to offline.
    provider, model = example_provider(
        citing_responder("The Pro plan refund window is 30 days with no fee. [{ref}]")
    )
    app = ContextApp(name="docs_qa", provider=provider, model=model)

    # add_source indexes a folder: adaptive chunking + hybrid (keyword+vector)
    # retrieval are sensible defaults for mixed prose/tables.
    app.add_source("docs", path=str(docs_dir), chunking="adaptive", retrieval="hybrid")

    # Policies are guardrails. Grounding-only + measured groundedness/citation
    # accuracy turn "the model said so" into "the evidence says so".
    app.set_policy("answer_only_from_sources", True)
    app.add_evaluator("groundedness")
    app.add_evaluator("citation_accuracy")

    result = app.run("What is the refund window for the Pro plan?")
    print("answer:", result.output)
    print("citations:", result.citations)
    print("eval scores:", result.eval_scores)


# ---------------------------------------------------------------------------
# 5. Every result carries a trace_id and a cost.
# ---------------------------------------------------------------------------
def section_trace_and_cost() -> None:
    """Observability is built in: every RunResult is stamped with a ``trace_id``
    (to correlate logs/audit) and a ``cost_usd`` (token spend), plus latency and
    token usage. You get accounting and traceability for free on every call."""
    banner("5. trace_id + cost on every result")

    provider, model = example_provider(lambda req: "Tracked and accounted for.")
    app = ContextApp(name="observable", provider=provider, model=model)

    result = app.run("Anything.")
    print("output     :", result.output)
    print("trace_id   :", result.trace_id)
    print(f"cost_usd   : ${result.cost_usd:.6f}")
    print(f"latency_ms : {result.latency_ms}")
    print("tokens     :", result.usage.total_tokens, "total")


# ---------------------------------------------------------------------------
# 6. A short multi-turn conversation.
# ---------------------------------------------------------------------------
def _chat_responder(request):
    """Deterministic offline chat behaviour keyed off the running transcript.

    A real model would decide these replies itself; here we branch on what the
    user has said so far so the multi-turn flow is reproducible.
    """
    convo = "\n".join(m.text for m in request.messages).lower()
    if "thank" in convo:
        return "You're welcome! Anything else about your account?"
    if "pro" in convo and "refund" in convo:
        return "On the Pro plan you have a 30-day refund window with no fee."
    return "Happy to help — what's your plan and question?"


def section_conversation() -> None:
    """``app.assistant(...)`` is a session-aware chat layer: each ``send`` is a
    full governed run, but turns are threaded under one session with memory, so
    later turns can reference earlier context. A chat product is a short loop,
    not hand-wired plumbing."""
    banner("6. A short multi-turn conversation")

    provider, model = example_provider(_chat_responder)
    app = ContextApp(name="support_chat", provider=provider, model=model)
    chat = app.assistant(user_id="cust-42")

    for user_msg in [
        "What's my refund window? My plan is Pro.",
        "Thanks, that's all I needed.",
    ]:
        turn = chat.send(user_msg)
        print(f"user      : {user_msg}")
        print(f"assistant : {turn.text}")

    # The session retains the whole exchange (system + user + assistant turns).
    print(f"\nsession {chat.session_id}: {len(chat.history())} messages retained")


# ---------------------------------------------------------------------------
# main: run the tour top to bottom.
# ---------------------------------------------------------------------------
def main() -> None:
    # write_sample_docs lays down refund_policy.md + terms.md in a temp folder
    # for the grounded-QA section; nothing touches the user's project tree.
    docs_dir = write_sample_docs(Path(tempfile.mkdtemp()) / "docs")

    section_hello()
    section_typed_output()
    section_configure()
    section_grounded_qa(docs_dir)
    section_trace_and_cost()
    section_conversation()


if __name__ == "__main__":
    main()
