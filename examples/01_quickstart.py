"""Quickstart — the five-minute tour of the core loop.

The first file to read. One runnable program: build a ``ContextApp``, get a
typed Pydantic object back, shape behaviour with role/objective/rules, ground
answers in your own documents with citations, read the trace_id + cost stamped
on every result, and hold a short multi-turn chat. Runs offline on the mock
provider — set ``VINCIO_PROVIDER`` (and a key) to point it at a real model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _shared import citing_responder, example_provider, json_responder, write_sample_docs
from pydantic import BaseModel

from vincio import ContextApp
from vincio.core.config import VincioConfig


class TicketClassification(BaseModel):
    """The typed shape we want every classification to validate against."""

    label: str
    confidence: float
    reason: str


def _isolated_config() -> VincioConfig:
    """A per-run in-process store so the chat demo is hermetic (memory never
    leaks across runs) — the offline-first default for examples and tests."""
    tmp = tempfile.mkdtemp(prefix="vincio_quickstart_")
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp}/vincio.db"
    config.observability.exporter = "memory"
    return config


def _chat_responder(request):
    """Deterministic offline chat behaviour keyed off the running transcript.
    A real model decides these replies itself; here we branch so the flow is
    reproducible."""
    convo = "\n".join(m.text for m in request.messages).lower()
    if "thank" in convo:
        return "You're welcome! Anything else about your account?"
    if "pro" in convo and "refund" in convo:
        return "On the Pro plan you have a 30-day refund window with no fee."
    return "Happy to help — what's your plan and question?"


def main() -> None:
    docs_dir = write_sample_docs(Path(tempfile.mkdtemp()) / "docs")

    # 1. A ContextApp wraps a provider+model and runs one governed turn.
    #    example_provider() returns a deterministic mock offline; its argument is
    #    the canned reply, so the whole tour is reproducible with no network.
    provider, model = example_provider(lambda req: "Hi! I'm a Vincio app running offline.")
    app = ContextApp(name="hello", provider=provider, model=model)
    print("1. hello:", app.run("Say hello.").output)

    # 2. output_schema= makes the app parse and validate the reply into that
    #    Pydantic type, so result.output is a typed object (attribute access, IDE
    #    completion, a validation error if the model strays off-schema).
    provider, model = example_provider(
        json_responder({"label": "billing", "confidence": 0.93, "reason": "duplicate charge"})
    )
    app = ContextApp(name="triage", output_schema=TicketClassification, provider=provider, model=model)
    out = app.run("I was charged twice this month.").output  # a validated instance
    print(f"2. typed output: label={out.label!r} confidence={out.confidence}")

    # 3. configure(...) sets the system persona declaratively — who the app is
    #    (role), what it's for (objective), the hard constraints (rules). It
    #    compiles into every run's context, so you steer from config, not ad-hoc
    #    prompt strings glued together per call.
    provider, model = example_provider(
        json_responder({"label": "billing", "confidence": 0.88, "reason": "refund request"})
    )
    app = ContextApp(name="triage2", output_schema=TicketClassification, provider=provider, model=model)
    app.configure(
        role="support_ticket_triage_engine",
        objective="Classify incoming support tickets into one routing label.",
        rules=["Answer with exactly one label: bug, billing, feature, or other.",
               "Never invent account details that were not provided."],
    )
    print("3. configured classify:", app.run("Can I get a refund for last month?").output.label)

    # 4. Attach documents as a source and the app retrieves, grounds and cites.
    #    answer_only_from_sources restricts the model to retrieved evidence, and
    #    the two evaluators turn "the model said so" into "the evidence says so".
    #    citing_responder echoes a real evidence ref so citations bind offline.
    provider, model = example_provider(
        citing_responder("The Pro plan refund window is 30 days with no fee. [{ref}]")
    )
    app = ContextApp(name="docs_qa", provider=provider, model=model)
    app.add_source("docs", path=str(docs_dir), chunking="adaptive", retrieval="hybrid")
    app.set_policy("answer_only_from_sources", True)
    app.add_evaluator("groundedness")
    app.add_evaluator("citation_accuracy")
    result = app.run("What is the refund window for the Pro plan?")
    print("4. grounded QA:", result.output, "| cites", result.citations, "| eval", result.eval_scores)

    # 5. Observability is built in: every result carries a trace_id (to correlate
    #    logs/audit), a cost_usd, latency, and token usage — accounting for free.
    provider, model = example_provider(lambda req: "Tracked and accounted for.")
    r = ContextApp(name="observable", provider=provider, model=model).run("Anything.")
    print(f"5. trace {r.trace_id} | ${r.cost_usd:.6f} | {r.latency_ms}ms | {r.usage.total_tokens} tokens")

    # 6. assistant(...) is a session-aware chat layer: each .send is a full
    #    governed run, but turns are threaded under one session with memory so
    #    later turns see earlier context. The isolated config keeps it hermetic.
    provider, model = example_provider(_chat_responder)
    app = ContextApp(name="support_chat", provider=provider, model=model, config=_isolated_config())
    chat = app.assistant(user_id="cust-42")
    for msg in ["What's my refund window? My plan is Pro.", "Thanks, that's all I needed."]:
        print(f"6. chat: {msg!r} -> {chat.send(msg).text}")
    print(f"   session {chat.session_id}: {len(chat.history())} messages retained")


if __name__ == "__main__":
    main()
