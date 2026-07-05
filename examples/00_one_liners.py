"""The one-liners — the ergonomic front door.

The five jobs a newcomer actually has, each as ONE expression: grounded RAG
Q&A, typed extraction, an approval-gated tool agent, an offline eval, and a
multi-turn chat — plus the fluent, immutable ``Flow`` (Vincio's answer to LCEL).

Each one-liner is a thin facade that lowers to the *same* governed
``ContextApp.run`` packet as the verbose builder form: retrieval, grounding,
validation, rails, budgets, tracing and the audit chain all apply unchanged.
``.app`` is the escape hatch to every deep method, and the last block *proves*
the no-fork guarantee by lowering both forms to a byte-identical signature.
Runs fully offline on the deterministic mock provider.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _shared import citing_responder, example_provider, json_responder, write_sample_docs
from pydantic import BaseModel

from vincio import ContextApp, Flow, chat, evaluation, extractor, rag, tool_agent
from vincio.core.config import VincioConfig
from vincio.documents import load_directory
from vincio.evals import Dataset as EvalDataset
from vincio.evals import EvalCase
from vincio.testing import run_signature


def _isolated_config() -> VincioConfig:
    """A per-run, in-process store so the demo is hermetic — memory never leaks
    across runs. Best practice for any example or test that writes memory."""
    tmp = tempfile.mkdtemp(prefix="vincio_oneliners_")
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp}/vincio.db"
    config.observability.exporter = "memory"
    return config


class TicketClassification(BaseModel):
    """The typed shape every classification must validate against."""

    label: str
    confidence: float
    reason: str


def create_ticket(summary: str) -> str:
    """Open a support ticket — a *write* tool, so it is approval-gated below."""
    return f"TICKET-{abs(hash(summary)) % 1000:03d}"


def _chat_responder(request):
    """Deterministic offline chat behaviour keyed off the running transcript."""
    convo = "\n".join(m.text for m in request.messages).lower()
    if "thank" in convo:
        return "You're welcome! Anything else about your account?"
    if "pro" in convo and "refund" in convo:
        return "On the Pro plan you have a 30-day refund window with no fee."
    return "Happy to help — what's your plan and question?"


def main() -> None:
    docs_dir = write_sample_docs(Path(tempfile.mkdtemp()) / "docs")

    # 1. rag(...) is the canonical six-call RAG path as one expression: it
    #    indexes the sources, turns on grounding-only answering with citations,
    #    and wires the groundedness + citation-accuracy evaluators. .ask runs a
    #    full grounded, cited, eval-scored governed run.
    provider, model = example_provider(
        citing_responder("The Pro plan refund window is 30 days with no fee. [{ref}]")
    )
    answer = rag(str(docs_dir), provider=provider, model=model, chunking="adaptive").ask(
        "What is the refund window for the Pro plan?"
    )
    print("1. rag:", answer.output, "| cites", answer.citations, "| eval", answer.eval_scores)

    # 2. extractor(schema) is a typed extractor — .extract parses and validates
    #    each reply into the Pydantic type, so you get an object, not a string.
    provider, model = example_provider(
        json_responder({"label": "billing", "confidence": 0.93, "reason": "duplicate charge"})
    )
    ticket = extractor(TicketClassification, provider=provider, model=model).extract(
        "I was charged twice this month."
    )
    print(f"2. extractor: label={ticket.label!r} confidence={ticket.confidence}")

    # 3. tool_agent(writes=[...]) registers write tools as approval-required:
    #    denied by default and surfaced as pending approvals, so a one-shot reply
    #    can never silently fire a write. approve=[...] pre-allows a trusted tool.
    #    The mock is scripted to request the tool, then reply once it returns.
    script = [
        {"tool_call": {"name": "create_ticket", "arguments": {"summary": "duplicate charge"}}},
        "Done — I've opened a ticket for the duplicate charge.",
    ]
    provider, model = example_provider(script=list(script))
    blocked = tool_agent(writes=[create_ticket], provider=provider, model=model)
    blocked.run("Open a ticket for the duplicate charge")
    provider, model = example_provider(script=list(script))
    approved = tool_agent(
        writes=[create_ticket], approve=["create_ticket"], provider=provider, model=model
    )
    approved.run("Open a ticket for the duplicate charge")
    print("3. tool_agent: denied by default", [a.tool for a in blocked.pending_approvals],
          "| after approve", [(a.tool, a.status) for a in approved.approvals])

    # 4. evaluation(dataset, gates=...) bundles the dataset, evaluators and
    #    pass/fail gates so scoring is one call over a configured app. Reuse the
    #    rag facade's .app so the eval runs against a grounded RAG pipeline.
    provider, model = example_provider(
        citing_responder("The Pro plan refund window is 30 days. [{ref}]")
    )
    app = rag(str(docs_dir), provider=provider, model=model).app
    dataset = EvalDataset(
        name="refund-qa",
        cases=[EvalCase(id="c1", input="What is the refund window for the Pro plan?",
                        expected="30 days")],
    )
    report = evaluation(dataset, gates={"groundedness": ">= 0.5"}, app=app).run()
    print("4. evaluation: gates", {m: g["passed"] for m, g in report.gates.items()},
          "| scores", {m: round(s["mean"], 3) for m, s in report.summary().items()})

    # 5. chat(...) opens a session-aware multi-turn conversation: every .send is
    #    a full governed run threaded under one session with memory write-back.
    provider, model = example_provider(_chat_responder)
    bot = chat(provider=provider, model=model, user_id="cust-42", config=_isolated_config())
    for message in ["What's my refund window? My plan is Pro.", "Thanks, that's all."]:
        print(f"5. chat: {message!r} -> {bot.send(message).text}")
    print(f"   session {bot.session_id}: {len(bot.history())} messages retained")

    # 6. Flow threads retrieve -> ground -> call -> evaluate as an immutable
    #    value: every step returns a NEW Flow (nothing mutates in place), and
    #    .run lowers the whole pipeline to one governed run.
    provider, model = example_provider(
        citing_responder("The Pro plan refund window is 30 days. [{ref}]")
    )
    flow = (
        Flow(provider=provider, model=model)
        .retrieve(str(docs_dir), chunking="adaptive")
        .ground()
        .evaluate("groundedness", "citation_accuracy")
    )
    answer = flow.run("What is the refund window for the Pro plan?")
    print(f"6. Flow ({flow.steps} immutable steps):", answer.output, "| eval", answer.eval_scores)

    # 7. The no-fork guarantee: the one-liner and the verbose builder form lower
    #    to the SAME packet and result — proven byte-for-byte via run_signature.
    #    Hold the input constant (in-memory docs) since the facade never loads.
    documents = list(load_directory(docs_dir))
    question = "What is the refund window for the Pro plan?"

    def responder():
        return citing_responder("The Pro plan refund window is 30 days. [{ref}]")

    verbose = ContextApp(name="rag", provider=example_provider(responder())[0], model="mock-1")
    verbose.add_source("docs", documents=documents, chunking="adaptive", retrieval="hybrid")
    verbose.set_policy("answer_only_from_sources", True)
    verbose.add_evaluator("groundedness")
    verbose.add_evaluator("citation_accuracy")
    ad_hoc = rag(documents, provider=example_provider(responder())[0], model="mock-1",
                 chunking="adaptive").app
    identical = run_signature(verbose, question) == run_signature(ad_hoc, question)
    print("7. byte-identical (same packet AND result):", identical)

    # .app is the escape hatch — anything the verbose path can do, reach it here.
    ad_hoc.add_rail(name="no_pii", kind="safety", detectors=["pii"])
    print("   escape hatch: added a rail via .app; rails now", len(ad_hoc.rail_engine.rails))


if __name__ == "__main__":
    main()
