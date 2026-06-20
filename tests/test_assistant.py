"""The Assistant layer, covered by a multi-turn simulator suite.

Each test drives the conversational layer over the deterministic mock provider,
either directly or through :class:`~vincio.evals.simulator.Simulator`, so the
multi-turn machinery — session threading, memory write-back, tool approvals, and
the recorded transcript — is exercised offline and reproducibly.
"""

from __future__ import annotations

import pytest

from vincio import Assistant, ContextApp
from vincio.evals.simulator import Persona, Simulator
from vincio.providers import MockProvider


@pytest.fixture()
def chat_app():
    provider = MockProvider(
        responder=lambda r: "Here are the steps to resolve that, with a follow-up if needed."
    )
    return ContextApp(name="chat", provider=provider, model="mock-1")


def test_assistant_threads_a_session(chat_app):
    chat = chat_app.assistant(user_id="u-1")
    sid = chat.session_id
    t1 = chat.send("How do I reset my password?")
    t2 = chat.send("And update my email?")
    assert t1.trace_id and t2.trace_id
    # Both turns ran under the same session.
    assert chat.session_id == sid
    assert len(chat.history()) == 4  # 2 user + 2 assistant
    assert chat.history()[0] == {"role": "user", "content": "How do I reset my password?"}


def test_assistant_writes_back_to_session_memory(chat_app):
    chat = chat_app.assistant(user_id="u-7")
    turn = chat.send("My plan is Enterprise and my region is EU.")
    assert turn.memory_writes, "a turn should be written back to memory"
    # The written turn is recallable in the same session.
    recalled = chat_app.memory.recall("plan region", session_id=chat.session_id)
    assert recalled
    assert any("Enterprise" in item.content for item in recalled)


def test_memory_writeback_can_be_disabled():
    app = ContextApp(name="c", provider=MockProvider(responder=lambda r: "ok"), model="mock-1")
    chat = app.assistant(memory_writeback=False)
    turn = chat.send("hello")
    assert turn.memory_writes == []


def test_reset_rotates_the_session(chat_app):
    chat = chat_app.assistant(user_id="u-1")
    first = chat.session_id
    chat.send("hi")
    chat.reset()
    assert chat.session_id != first
    assert chat.history() == []


def test_write_tool_is_denied_until_approved():
    def refund_create(invoice: str) -> str:
        return f"refunded {invoice}"

    provider = MockProvider(
        script=[
            {"tool_call": {"name": "refund_create", "arguments": {"invoice": "INV-1"}}},
            "Your refund is being processed.",
            {"tool_call": {"name": "refund_create", "arguments": {"invoice": "INV-1"}}},
            "Done — the duplicate charge is refunded.",
        ]
    )
    app = ContextApp(name="refunds", provider=provider, model="mock-1")
    app.add_tool(refund_create, permissions=["billing:write"], approval_required=True,
                 side_effects="write")
    chat = app.assistant(user_id="u-2")

    turn = chat.send("Please refund invoice INV-1")
    assert turn.needs_approval
    assert [a.tool for a in chat.pending_approvals] == ["refund_create"]
    # The write tool did not actually run.
    assert all(tr.status != "ok" for tr in turn.result.tool_results if tr.tool_name == "refund_create")

    # Approve it; the next turn runs the tool.
    chat.approve("refund_create")
    turn2 = chat.send("Yes, go ahead")
    ran = [tr for tr in turn2.result.tool_results if tr.tool_name == "refund_create"]
    assert ran and ran[0].status == "ok"
    assert not turn2.needs_approval


def test_auto_approve_runs_the_tool_immediately():
    def refund_create(invoice: str) -> str:
        return f"refunded {invoice}"

    provider = MockProvider(
        script=[
            {"tool_call": {"name": "refund_create", "arguments": {"invoice": "INV-2"}}},
            "Refunded.",
        ]
    )
    app = ContextApp(name="refunds", provider=provider, model="mock-1")
    app.add_tool(refund_create, permissions=["billing:write"], approval_required=True,
                 side_effects="write")
    chat = app.assistant(user_id="u-3", auto_approve=["refund_create"])
    turn = chat.send("refund INV-2")
    assert not turn.needs_approval
    approved = [a for a in turn.approvals if a.status == "approved"]
    assert approved and approved[0].tool == "refund_create"


def test_on_approval_callback_decides():
    def write_record(value: str) -> str:
        return "written"

    provider = MockProvider(
        script=[{"tool_call": {"name": "write_record", "arguments": {"value": "x"}}}, "done"]
    )
    app = ContextApp(name="rec", provider=provider, model="mock-1")
    app.add_tool(write_record, permissions=["data:write"], approval_required=True,
                 side_effects="write")
    seen = []

    def decide(request):
        seen.append(request.tool)
        return True

    chat = app.assistant(on_approval=decide)
    turn = chat.send("write x")
    assert seen == ["write_record"]
    assert any(a.status == "approved" for a in turn.approvals)


# -- multi-turn simulator suite ---------------------------------------------


def test_simulator_drives_the_assistant(chat_app):
    """The Simulator drives the Assistant as a multi-turn agent; the whole
    conversation is recorded and convertible to an EvalCase."""
    chat = chat_app.assistant(user_id="sim-1")

    def agent(messages: list[dict[str, str]]) -> str:
        # The simulator passes the running thread; reply to the latest user turn.
        return chat.send(messages[-1]["content"]).text

    sim = Simulator(seed=7, max_turns=4)
    persona = Persona(
        name="customer",
        goal="resolve a duplicate billing charge",
        facts={"plan": "Pro"},
        max_turns=4,
    )
    convo = sim.simulate(agent, persona)
    assert convo.rounds >= 1
    assert len(convo.turns) >= 2
    # The assistant kept its own transcript in lockstep with the simulation.
    assert len(chat.history()) == len(convo.turns)
    # The conversation converts to a scorable multi-turn eval case.
    case = convo.to_eval_case(id="sim_billing")
    assert case.context["messages"]
    assert case.metadata["persona"] == "customer"


def test_simulated_session_accumulates_memory(chat_app):
    chat = chat_app.assistant(user_id="sim-2")

    def agent(messages: list[dict[str, str]]) -> str:
        return chat.send(messages[-1]["content"]).text

    sim = Simulator(seed=3, max_turns=3)
    convo = sim.simulate(
        agent, Persona(name="user", goal="understand the refund policy", max_turns=3)
    )
    # Every user turn produced a memory write-back, so session memory grew.
    user_turns = [t for t in convo.turns if t["role"] == "user"]
    recalled = chat_app.memory.recall("refund", session_id=chat.session_id, top_k=10)
    assert 1 <= len(recalled) <= len(user_turns)


def test_assistant_is_constructed_via_factory(chat_app):
    chat = chat_app.assistant()
    assert isinstance(chat, Assistant)
    assert chat.session_id.startswith("sess")
