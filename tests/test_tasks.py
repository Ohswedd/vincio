"""The ergonomic 'ad-hoc' front door (``vincio.tasks``).

These tests hold the three guarantees the ErgonomicsBench family publishes:

* **compiles-byte-identical** — every one-liner lowers to the *same* governed
  ``ContextApp.run`` packet and ``RunResult`` as the verbose builder form (proven
  with the shared ``vincio.testing.run_signature`` lowering harness), so there is
  no behavioral fork.
* **escape-hatch-total** — every facade exposes ``.app``, the fully-configured
  ``ContextApp``, reaching every deep method; nothing is shadowed.
* **conciseness** — each of the five jobs (plus the fluent ``Flow``) is one entry
  point.

Plus the per-facade behavior: grounded RAG, typed extraction, approval-gated tool
use, an offline eval, a threaded chat, and an immutable Flow. Everything runs on
the deterministic mock provider — offline, no keys.
"""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel

import vincio
from vincio import (
    Assistant,
    ContextApp,
    Flow,
    StabilityLevel,
    chat,
    evaluation,
    extractor,
    rag,
    stability_of,
    tool_agent,
)
from vincio.documents import load_directory
from vincio.evals import Dataset as EvalDataset
from vincio.evals import EvalCase
from vincio.providers import MockProvider
from vincio.tasks import Evaluation, Extractor, RagTask, ToolAgent
from vincio.testing import run_signature


# --------------------------------------------------------------------------- #
# Fixtures & helpers
# --------------------------------------------------------------------------- #
def _citing_responder():
    """A mock that echoes the first real evidence ref so citations bind offline."""

    def responder(request):
        text = "\n".join(m.text for m in request.messages)
        match = re.search(r"\[([\w.:-]+:C\d+)\]", text)
        ref = match.group(1) if match else "E1"
        return f"The Pro plan refund window is 30 days. [{ref}]"

    return responder


def _provider():
    return MockProvider(responder=_citing_responder())


@pytest.fixture()
def docs(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "refund.md").write_text(
        "# Refund Policy\n\nPro plan customers may request refunds within 30 days, no fee.\n",
        encoding="utf-8",
    )
    return folder


@pytest.fixture()
def documents(docs):
    # Shared, already-loaded documents so the verbose and ad-hoc forms index
    # byte-identical evidence (loading is outside the front door, held constant).
    return load_directory(docs)


class TicketClassification(BaseModel):
    label: str
    confidence: float


# --------------------------------------------------------------------------- #
# rag — grounded RAG Q&A
# --------------------------------------------------------------------------- #
def test_rag_answers_grounded_with_citations(docs):
    task = rag(str(docs), provider=_provider(), model="mock-1", chunking="adaptive")
    assert isinstance(task, RagTask)
    answer = task.ask("What is the refund window for the Pro plan?")
    assert "30 days" in answer.output
    assert answer.citations
    assert answer.eval_scores.get("groundedness") == pytest.approx(1.0)
    assert answer.eval_scores.get("citation_accuracy") == pytest.approx(1.0)


def test_rag_call_is_alias_for_ask(docs):
    task = rag(str(docs), provider=_provider(), model="mock-1")
    assert (
        task("What is the refund window?").output == task.ask("What is the refund window?").output
    )


def test_rag_lowers_byte_identical_to_the_verbose_six_call_form(documents):
    question = "What is the refund window for the Pro plan?"

    verbose = ContextApp(name="rag", provider=_provider(), model="mock-1")
    verbose.add_source("docs", documents=list(documents), chunking="adaptive", retrieval="hybrid")
    verbose.set_policy("answer_only_from_sources", True)
    verbose.add_evaluator("groundedness")
    verbose.add_evaluator("citation_accuracy")

    ad_hoc = rag(list(documents), provider=_provider(), model="mock-1", chunking="adaptive").app

    assert run_signature(verbose, question) == run_signature(ad_hoc, question)


def test_rag_accepts_a_documents_list_and_a_name_mapping(documents):
    by_list = rag(list(documents), provider=_provider(), model="mock-1")
    assert "docs" in by_list.app.sources

    by_mapping = rag({"policies": list(documents)}, provider=_provider(), model="mock-1")
    assert "policies" in by_mapping.app.sources


def test_rag_grounded_false_skips_the_answer_only_policy(documents):
    task = rag(list(documents), provider=_provider(), model="mock-1", grounded=False)
    assert task.app.policies.answer_only_from_sources is False


def test_rag_evaluators_empty_adds_no_metrics(documents):
    task = rag(list(documents), provider=_provider(), model="mock-1", evaluators=())
    assert task.app.evaluators == []


# --------------------------------------------------------------------------- #
# extractor — typed structured extraction
# --------------------------------------------------------------------------- #
def _json_provider():
    import json

    payload = json.dumps({"label": "billing", "confidence": 0.9})
    return MockProvider(responder=lambda request: payload)


def test_extractor_returns_a_validated_object():
    get = extractor(TicketClassification, provider=_json_provider(), model="mock-1")
    assert isinstance(get, Extractor)
    out = get.extract("I was charged twice this month.")
    assert isinstance(out, TicketClassification)
    assert out.label == "billing"
    assert out.confidence == pytest.approx(0.9)


def test_extractor_lowers_byte_identical_to_output_schema_constructor():
    verbose = ContextApp(
        name="extractor",
        provider=_json_provider(),
        model="mock-1",
        output_schema=TicketClassification,
    )
    ad_hoc = extractor(TicketClassification, provider=_json_provider(), model="mock-1").app
    assert run_signature(verbose, "x") == run_signature(ad_hoc, "x")


def test_extractor_run_returns_the_full_result():
    get = extractor(TicketClassification, provider=_json_provider(), model="mock-1")
    result = get.run("I was charged twice.")
    assert result.output.label == "billing"
    assert result.trace_id


# --------------------------------------------------------------------------- #
# tool_agent — approval-gated tool use
# --------------------------------------------------------------------------- #
def _ticket_tool(calls):
    def create_ticket(summary: str) -> str:
        """Open a support ticket."""
        calls.append(summary)
        return "TICKET-1"

    return create_ticket


def _tool_script():
    return [
        {"tool_call": {"name": "create_ticket", "arguments": {"summary": "dup charge"}}},
        "Done, I opened a ticket.",
    ]


def test_tool_agent_denies_write_tools_by_default():
    calls: list[str] = []
    provider = MockProvider(script=_tool_script())
    agent = tool_agent(writes=[_ticket_tool(calls)], provider=provider, model="mock-1")
    assert isinstance(agent, ToolAgent)
    agent.run("Open a ticket for the duplicate charge")
    assert calls == []  # the write tool never ran
    assert [a.tool for a in agent.pending_approvals] == ["create_ticket"]


def test_tool_agent_runs_an_approved_write_tool():
    calls: list[str] = []
    provider = MockProvider(script=_tool_script())
    agent = tool_agent(
        writes=[_ticket_tool(calls)], approve=["create_ticket"], provider=provider, model="mock-1"
    )
    agent.run("Open a ticket for the duplicate charge")
    assert calls == ["dup charge"]
    assert agent.pending_approvals == []
    assert [(a.tool, a.status) for a in agent.approvals] == [("create_ticket", "approved")]


def test_tool_agent_approve_method_grants_a_standing_approval():
    calls: list[str] = []
    agent = tool_agent(
        writes=[_ticket_tool(calls)], provider=MockProvider(script=_tool_script()), model="mock-1"
    )
    agent.approve("create_ticket")  # standing approval, before the run
    agent.run("Open a ticket")
    assert calls == ["dup charge"]
    assert [(a.tool, a.status) for a in agent.approvals] == [("create_ticket", "approved")]


def test_tool_agent_read_tools_run_freely():
    seen: list[str] = []

    def search_docs(query: str) -> str:
        """Search the docs (read-only)."""
        seen.append(query)
        return "Pro plan: 30 day refund window."

    script = [
        {"tool_call": {"name": "search_docs", "arguments": {"query": "refund"}}},
        "The Pro plan has a 30-day refund window.",
    ]
    agent = tool_agent(tools=[search_docs], provider=MockProvider(script=script), model="mock-1")
    agent.run("What is the refund window?")
    assert seen == ["refund"]
    assert agent.pending_approvals == []


# --------------------------------------------------------------------------- #
# evaluation — an offline eval
# --------------------------------------------------------------------------- #
def _eval_dataset():
    return EvalDataset(
        name="refund-qa",
        cases=[
            EvalCase(
                id="c1",
                input="What is the refund window for the Pro plan?",
                expected="30 days",
            )
        ],
    )


def test_evaluation_runs_metrics_and_gates(documents):
    app = rag(list(documents), provider=_provider(), model="mock-1").app
    ev = evaluation(_eval_dataset(), gates={"groundedness": ">= 0.5"}, app=app)
    assert isinstance(ev, Evaluation)
    assert ev.metrics == ["groundedness", "citation_accuracy"]
    report = ev.run()
    assert report.gates["groundedness"]["passed"] is True


def test_evaluation_registers_its_metrics_as_evaluators():
    ev = evaluation(_eval_dataset(), metrics=["groundedness"], provider=_provider(), model="mock-1")
    assert ev.app.evaluators == ["groundedness"]


def test_evaluation_without_a_dataset_raises():
    ev = evaluation(provider=_provider(), model="mock-1")
    with pytest.raises(ValueError, match="dataset"):
        ev.run()


# --------------------------------------------------------------------------- #
# chat — a re-presentation of app.assistant
# --------------------------------------------------------------------------- #
def test_chat_returns_an_assistant_that_threads():
    bot = chat(provider=MockProvider(responder=lambda r: "Hello!"), model="mock-1", user_id="u-1")
    assert isinstance(bot, Assistant)
    turn = bot.send("Hi there")
    assert turn.text == "Hello!"
    assert turn.trace_id
    assert len(bot.history()) == 2


# --------------------------------------------------------------------------- #
# Flow — the fluent, immutable pipeline
# --------------------------------------------------------------------------- #
def test_flow_is_immutable_each_step_returns_a_new_flow():
    base = Flow(provider=_provider(), model="mock-1")
    retrieved = base.retrieve("./docs")
    grounded = retrieved.ground()
    assert base.steps == []
    assert retrieved.steps == ["retrieve"]
    assert grounded.steps == ["retrieve", "ground"]
    assert base is not retrieved is not grounded


def test_flow_lowers_byte_identical_to_the_verbose_form(documents):
    question = "What is the refund window for the Pro plan?"

    verbose = ContextApp(name="flow", provider=_provider(), model="mock-1")
    verbose.add_source("docs", documents=list(documents), chunking="adaptive", retrieval="hybrid")
    verbose.set_policy("answer_only_from_sources", True)
    verbose.add_evaluator("groundedness")
    verbose.add_evaluator("citation_accuracy")

    flow = (
        Flow(provider=_provider(), model="mock-1")
        .retrieve(documents=list(documents), chunking="adaptive")
        .ground()
        .evaluate("groundedness", "citation_accuracy")
    )
    assert run_signature(verbose, question) == run_signature(flow.app, question)


def test_flow_runs_and_invoke_is_an_alias(documents):
    flow = Flow(provider=_provider(), model="mock-1").retrieve(documents=list(documents)).ground()
    assert "30 days" in flow.run("What is the refund window?").output
    assert "30 days" in flow.invoke("What is the refund window?").output


def test_flow_validate_sets_a_typed_contract():
    flow = Flow(provider=_json_provider(), model="mock-1").validate(TicketClassification)
    assert flow.app.output_contract.schema_def is not None
    result = flow.run("I was charged twice.")
    assert isinstance(result.output, TicketClassification)


def test_flow_app_is_memoized():
    flow = Flow(provider=_provider(), model="mock-1").call(objective="answer questions")
    assert flow.app is flow.app  # built once, reused


# --------------------------------------------------------------------------- #
# Escape hatch — total reach through `.app`
# --------------------------------------------------------------------------- #
def test_every_facade_exposes_the_configured_contextapp(documents):
    facades = [
        rag(list(documents), provider=_provider(), model="mock-1"),
        extractor(TicketClassification, provider=_json_provider(), model="mock-1"),
        tool_agent(provider=_provider(), model="mock-1"),
        evaluation(provider=_provider(), model="mock-1"),
    ]
    for facade in facades:
        assert isinstance(facade.app, ContextApp)
    assert isinstance(Flow(provider=_provider(), model="mock-1").app, ContextApp)


def test_escape_hatch_reaches_deep_methods(documents):
    task = rag(list(documents), provider=_provider(), model="mock-1")
    before = len(task.app.rail_engine.rails)
    task.app.add_rail(name="no_pii", kind="safety", detectors=["pii"])
    task.app.enable_self_correction(max_cycles=2)
    assert len(task.app.rail_engine.rails) == before + 1


def test_app_argument_layers_the_task_onto_a_prebuilt_app(documents):
    prebuilt = ContextApp(name="custom", provider=_provider(), model="mock-1")
    prebuilt.add_rail(name="no_pii", kind="safety", detectors=["pii"])
    task = rag(list(documents), app=prebuilt)
    assert task.app is prebuilt
    assert task.app.policies.answer_only_from_sources is True  # task config layered on


# --------------------------------------------------------------------------- #
# Surface & stability
# --------------------------------------------------------------------------- #
def test_entry_points_are_in_the_public_surface():
    for name in ("rag", "extractor", "tool_agent", "evaluation", "chat", "Flow"):
        assert name in vincio.__all__
        assert getattr(vincio, name) is not None


def test_entry_points_are_tagged_experimental():
    for symbol in (rag, extractor, tool_agent, evaluation, chat, Flow):
        assert stability_of(symbol)["level"] == StabilityLevel.EXPERIMENTAL


def test_tasks_namespace_exports_the_facades():
    import vincio.tasks as tasks

    for name in (
        "rag",
        "extractor",
        "tool_agent",
        "evaluation",
        "chat",
        "Flow",
        "RagTask",
        "Extractor",
        "ToolAgent",
        "Evaluation",
    ):
        assert name in tasks.__all__
