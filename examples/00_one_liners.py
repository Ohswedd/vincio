"""The one-liners: the ergonomic 'ad-hoc' front door.

Read this before the quickstart. The five jobs a newcomer actually has — grounded
RAG Q&A, a tool-using agent, structured extraction, an eval, and a multi-step flow
— each as ONE expression, plus the fluent ``Flow`` (the Vincio answer to LCEL).

Every one-liner is a thin, transparent facade that lowers to the exact same
governed ``ContextApp.run`` packet as the verbose builder form — so retrieval,
grounding, validation, rails, budgets, tracing, and the audit chain all apply
unchanged. ``.app`` is the escape hatch to every deep method. The last section
*proves* the no-behavioral-fork guarantee by lowering the one-liner and the
six-call verbose form to a byte-identical signature.

Everything runs offline on the deterministic mock provider — no keys, no network.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _shared import citing_responder, example_provider, json_responder, write_sample_docs
from pydantic import BaseModel

from vincio import ContextApp, Flow, chat, evaluation, extractor, rag, tool_agent
from vincio.documents import load_directory
from vincio.evals import Dataset as EvalDataset
from vincio.evals import EvalCase
from vincio.testing import run_signature


def banner(title: str) -> None:
    """Print a section header so the run reads as a guided tour."""
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 1. Grounded RAG Q&A — one line instead of six coupled builder calls.
# ---------------------------------------------------------------------------
def section_rag(docs_dir: Path) -> None:
    """``rag(...)`` indexes sources, turns on grounding-only answering with
    citations, and adds the groundedness + citation-accuracy evaluators — the
    canonical six-call RAG path, as one expression. ``.ask`` runs a full,
    grounded, cited, eval-scored governed run."""
    banner("1. Grounded RAG Q&A — rag(...).ask(...)")

    provider, model = example_provider(
        citing_responder("The Pro plan refund window is 30 days with no fee. [{ref}]")
    )
    answer = rag(str(docs_dir), provider=provider, model=model, chunking="adaptive").ask(
        "What is the refund window for the Pro plan?"
    )
    print("answer   :", answer.output)
    print("citations:", answer.citations)
    print("eval     :", answer.eval_scores)


# ---------------------------------------------------------------------------
# 2. Typed structured extraction — text in, a validated object out.
# ---------------------------------------------------------------------------
class TicketClassification(BaseModel):
    """The structured shape every classification must conform to."""

    label: str
    confidence: float
    reason: str


def section_extractor() -> None:
    """``extractor(schema)`` builds a typed extractor: ``.extract`` parses and
    validates each reply into the schema and returns the object."""
    banner("2. Structured extraction — extractor(schema).extract(...)")

    provider, model = example_provider(
        json_responder({"label": "billing", "confidence": 0.93, "reason": "duplicate charge"})
    )
    get_ticket = extractor(TicketClassification, provider=provider, model=model)
    ticket = get_ticket.extract("I was charged twice this month.")
    print(f"label={ticket.label!r}  confidence={ticket.confidence}  reason={ticket.reason!r}")


# ---------------------------------------------------------------------------
# 3. Approval-gated tool agent — writes are denied until approved.
# ---------------------------------------------------------------------------
def create_ticket(summary: str) -> str:
    """Open a support ticket (a write tool — approval-gated)."""
    return f"TICKET-{abs(hash(summary)) % 1000:03d}"


def section_tool_agent() -> None:
    """``tool_agent(writes=[...])`` registers write tools as approval-required:
    denied by default and surfaced as pending approvals, so a one-shot reply can
    never silently run a write tool. ``approve`` pre-allows a trusted tool."""
    banner("3. Approval-gated tool agent — tool_agent(...).run(...)")

    # The mock asks to call the write tool, then replies once the tool returns.
    script = [
        {"tool_call": {"name": "create_ticket", "arguments": {"summary": "duplicate charge"}}},
        "Done — I've opened a ticket for the duplicate charge.",
    ]

    provider, model = example_provider(script=script)
    blocked = tool_agent(writes=[create_ticket], provider=provider, model=model)
    blocked.run("Open a ticket for the duplicate charge")
    print("denied by default — pending:", [a.tool for a in blocked.pending_approvals])

    provider, model = example_provider(script=list(script))
    approved = tool_agent(
        writes=[create_ticket], approve=["create_ticket"], provider=provider, model=model
    )
    result = approved.run("Open a ticket for the duplicate charge")
    print("after approve       — reply:", result.raw_text)
    print("approvals           :", [(a.tool, a.status) for a in approved.approvals])


# ---------------------------------------------------------------------------
# 4. An offline eval — metrics and gates bundled with the dataset.
# ---------------------------------------------------------------------------
def section_evaluation(docs_dir: Path) -> None:
    """``evaluation(dataset, metrics=..., gates=...)`` bundles the dataset, the
    evaluators, and the pass/fail gates so scoring is one call over a configured
    (here grounded) app."""
    banner("4. An offline eval — evaluation(...).run()")

    provider, model = example_provider(
        citing_responder("The Pro plan refund window is 30 days. [{ref}]")
    )
    # The eval runs against a grounded RAG app — reuse the rag(...) facade's app.
    app = rag(str(docs_dir), provider=provider, model=model).app
    dataset = EvalDataset(
        name="refund-qa",
        cases=[
            EvalCase(
                id="c1",
                input="What is the refund window for the Pro plan?",
                expected="30 days",
            )
        ],
    )
    report = evaluation(dataset, gates={"groundedness": ">= 0.5"}, app=app).run()
    print("metrics :", evaluation(dataset, app=app).metrics)
    print("gates   :", {m: g["passed"] for m, g in report.gates.items()})
    print("scores  :", {m: round(s["mean"], 3) for m, s in report.summary().items()})


# ---------------------------------------------------------------------------
# 5. A multi-turn chat — a re-presentation of app.assistant.
# ---------------------------------------------------------------------------
def _chat_responder(request):
    """Deterministic offline chat behaviour keyed off the running transcript."""
    convo = "\n".join(m.text for m in request.messages).lower()
    if "thank" in convo:
        return "You're welcome! Anything else about your account?"
    if "pro" in convo and "refund" in convo:
        return "On the Pro plan you have a 30-day refund window with no fee."
    return "Happy to help — what's your plan and question?"


def section_chat() -> None:
    """``chat(...)`` opens a session-aware multi-turn conversation: every turn is
    a full governed run, threaded under one session with memory write-back."""
    banner("5. A multi-turn chat — chat(...).send(...)")

    provider, model = example_provider(_chat_responder)
    bot = chat(provider=provider, model=model, user_id="cust-42")
    for message in ["What's my refund window? My plan is Pro.", "Thanks, that's all."]:
        print(f"user      : {message}")
        print(f"assistant : {bot.send(message).text}")
    print(f"session {bot.session_id}: {len(bot.history())} messages retained")


# ---------------------------------------------------------------------------
# 6. The fluent, immutable Flow — the Vincio answer to LCEL.
# ---------------------------------------------------------------------------
def section_flow(docs_dir: Path) -> None:
    """``Flow`` threads retrieve → ground → call → validate → evaluate as a value:
    every step returns a NEW Flow (nothing mutates in place), and ``.run`` lowers
    the whole pipeline to one governed run."""
    banner("6. The fluent Flow — Flow().retrieve().ground().evaluate().run()")

    provider, model = example_provider(
        citing_responder("The Pro plan refund window is 30 days. [{ref}]")
    )
    flow = (
        Flow(provider=provider, model=model)
        .retrieve(str(docs_dir), chunking="adaptive")
        .ground()
        .evaluate("groundedness", "citation_accuracy")
    )
    print("flow steps:", flow.steps, "(immutable — each step returns a new Flow)")
    answer = flow.run("What is the refund window for the Pro plan?")
    print("answer    :", answer.output)
    print("eval      :", answer.eval_scores)


# ---------------------------------------------------------------------------
# 7. The escape hatch + the no-behavioral-fork guarantee.
# ---------------------------------------------------------------------------
def section_escape_hatch_and_lowering(docs_dir: Path) -> None:
    """``.app`` reaches every deep method, and the one-liner lowers to the *same*
    packet and result as the verbose six-call form — proven byte-for-byte."""
    banner("7. Escape hatch + byte-identical lowering")

    # Shared in-memory documents so both forms index identical evidence (the
    # front door never touches loading, so we hold that input constant).
    documents = load_directory(docs_dir)
    question = "What is the refund window for the Pro plan?"

    def fresh_responder():
        return citing_responder("The Pro plan refund window is 30 days. [{ref}]")

    # The verbose six-call RAG path a newcomer writes by hand.
    provider, model = example_provider(fresh_responder())
    verbose = ContextApp(name="rag", provider=provider, model=model)
    verbose.add_source("docs", documents=list(documents), chunking="adaptive", retrieval="hybrid")
    verbose.set_policy("answer_only_from_sources", True)
    verbose.add_evaluator("groundedness")
    verbose.add_evaluator("citation_accuracy")

    # The one-liner.
    provider, model = example_provider(fresh_responder())
    ad_hoc = rag(list(documents), provider=provider, model=model, chunking="adaptive").app

    verbose_sig = run_signature(verbose, question)
    ad_hoc_sig = run_signature(ad_hoc, question)
    print("verbose packet spec_hash:", verbose_sig["spec_hash"])
    print("one-liner packet spec_hash:", ad_hoc_sig["spec_hash"])
    print("byte-identical (same packet AND result):", verbose_sig == ad_hoc_sig)

    # The escape hatch: anything the verbose path can do, the one-liner reaches.
    task = rag(list(documents), provider=example_provider(fresh_responder())[0], model="mock-1")
    task.app.add_rail(name="no_pii", kind="safety", detectors=["pii"])
    print("escape hatch — added a rail via .app; rails now:", len(task.app.rail_engine.rails))


# ---------------------------------------------------------------------------
# main: run the tour top to bottom.
# ---------------------------------------------------------------------------
def main() -> None:
    docs_dir = write_sample_docs(Path(tempfile.mkdtemp()) / "docs")

    section_rag(docs_dir)
    section_extractor()
    section_tool_agent()
    section_evaluation(docs_dir)
    section_chat()
    section_flow(docs_dir)
    section_escape_hatch_and_lowering(docs_dir)


if __name__ == "__main__":
    main()
