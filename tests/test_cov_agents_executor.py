"""Real-behavior coverage for vincio.agents.executor.

Every test drives the real AgentExecutor with the deterministic MockProvider
(scripted/responder), a real ToolRegistry/ToolRuntime, and real Planner modes.
Targets the uncovered branches: budget enforcement / termination, tool-call
planning branches (unknown tool, tool error, budget cap), step error and repair
paths, ask_human gate, output-validator finalize branches, retry, replan loop,
astream error surfacing, and the ReAct tool-error branch.
"""

from __future__ import annotations

import pytest

from vincio.agents import AgentExecutor, AgentStep, Planner
from vincio.agents.state import AgentState
from vincio.core.errors import ToolValidationError
from vincio.core.types import (
    Budget,
    EvidenceItem,
    Objective,
    TaskType,
)
from vincio.output.schemas import OutputContract
from vincio.output.validators import OutputValidator
from vincio.providers import MockProvider
from vincio.tools import ToolRegistry, ToolRuntime


def _runtime_with_tool(*, fail: bool = False, approval: bool = False):
    """A real ToolRuntime exposing one ``lookup`` tool."""
    registry = ToolRegistry()

    @registry.register(approval_required=approval, side_effects="write" if approval else "read")
    def lookup(invoice_id: str) -> dict:
        """Look up an invoice."""
        if fail:
            raise RuntimeError("backend exploded")
        return {"invoice_id": invoice_id, "amount": 42.0}

    return ToolRuntime(registry, cache_enabled=False), registry.specs()


# ---------------------------------------------------------------------------
# budget / termination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_steps_termination_sets_reason():
    # budget allows so few steps that the static plan can't finish.
    executor = AgentExecutor(
        MockProvider(default_text="draft"), model="m", planner=Planner(mode="static")
    )
    state = await executor.run(
        Objective("answer", task_type=TaskType.DOCUMENT_QA),
        budget=Budget(max_steps=1),
    )
    # _check_termination tripped on usage.steps >= max_steps before finalize ran.
    assert state.terminated is True
    assert state.termination_reason == "max_steps"
    assert state.final_answer is None


@pytest.mark.asyncio
async def test_budget_exhausted_records_unrecoverable_error():
    # zero cost budget: the very first model call pushes cost over the limit.
    provider = MockProvider(
        responder=lambda req: "draft",
        latency_ms=1,
    )
    # Force a non-zero cost so the cost dimension is breached.
    executor = AgentExecutor(provider, model="m", planner=Planner(mode="static"))
    state = await executor.run(
        Objective("answer", task_type=TaskType.DOCUMENT_QA),
        # 1 step lets the first think run, then termination check trips on steps.
        budget=Budget(max_steps=2, max_cost_usd=0.0, max_input_tokens=0),
    )
    assert state.terminated is True
    # input_tokens exhausted -> budget_exhausted + recorded unrecoverable error.
    assert state.termination_reason == "budget_exhausted"
    assert any(not e.recoverable and "budget exhausted" in e.message for e in state.errors)


# ---------------------------------------------------------------------------
# plan_and_call_tools branches
# ---------------------------------------------------------------------------


def _tool_plan_responder(tool_name: str, args: dict):
    """Responder that emits a tool-plan structured call for the plan_tools step
    and plain text otherwise."""

    def responder(req):
        if req.output_schema_name == "tool_calls":
            return {"calls": [{"tool_name": tool_name, "arguments": args}]}
        return "draft answer"

    return responder


@pytest.mark.asyncio
async def test_plan_tools_executes_known_tool_and_records_result():
    runtime, specs = _runtime_with_tool()
    provider = MockProvider(responder=_tool_plan_responder("lookup", {"invoice_id": "INV-9"}))
    executor = AgentExecutor(
        provider,
        model="m",
        planner=Planner(mode="static"),
        tool_runtime=runtime,
        tool_specs=specs,
    )
    state = await executor.run(Objective("do it", task_type=TaskType.TOOL_ACTION))
    assert state.usage.tool_calls == 1
    assert len(state.tool_results) == 1
    assert state.tool_results[0].status == "ok"
    assert state.tool_results[0].output == {"invoice_id": "INV-9", "amount": 42.0}


@pytest.mark.asyncio
async def test_plan_tools_unknown_tool_records_error_entry():
    runtime, specs = _runtime_with_tool()
    provider = MockProvider(responder=_tool_plan_responder("ghost", {"x": 1}))
    executor = AgentExecutor(
        provider,
        model="m",
        planner=Planner(mode="static"),
        tool_runtime=runtime,
        tool_specs=specs,
    )
    state = await executor.run(Objective("do it", task_type=TaskType.TOOL_ACTION))
    # unknown tool was never executed.
    assert state.usage.tool_calls == 0
    plan_step = next(s for s in state.steps if s.name == "plan_tools")
    assert plan_step.result == [{"tool": "ghost", "status": "error", "error": "unknown tool"}]


@pytest.mark.asyncio
async def test_plan_tools_handler_error_becomes_error_result():
    runtime, specs = _runtime_with_tool(fail=True)
    provider = MockProvider(responder=_tool_plan_responder("lookup", {"invoice_id": "X"}))
    executor = AgentExecutor(
        provider,
        model="m",
        planner=Planner(mode="static"),
        tool_runtime=runtime,
        tool_specs=specs,
    )
    state = await executor.run(Objective("do it", task_type=TaskType.TOOL_ACTION))
    # the handler exception is captured by the runtime as an error ToolResult.
    assert state.usage.tool_calls == 1
    assert state.tool_results[0].status == "error"
    assert "backend exploded" in (state.tool_results[0].error or "")


@pytest.mark.asyncio
async def test_plan_tools_validation_error_caught_as_error_result():
    # missing required ``invoice_id`` -> ToolRuntime raises ToolValidationError,
    # which _plan_and_call_tools catches and folds into an error ToolResult.
    runtime, specs = _runtime_with_tool()
    provider = MockProvider(responder=_tool_plan_responder("lookup", {}))
    executor = AgentExecutor(
        provider,
        model="m",
        planner=Planner(mode="static"),
        tool_runtime=runtime,
        tool_specs=specs,
    )
    state = await executor.run(Objective("do it", task_type=TaskType.TOOL_ACTION))
    assert state.usage.tool_calls == 1
    assert state.tool_results[0].status == "error"
    assert "invalid arguments" in (state.tool_results[0].error or "")


@pytest.mark.asyncio
async def test_plan_tools_respects_tool_call_budget_cap():
    runtime, specs = _runtime_with_tool()

    def responder(req):
        if req.output_schema_name == "tool_calls":
            return {
                "calls": [
                    {"tool_name": "lookup", "arguments": {"invoice_id": "A"}},
                    {"tool_name": "lookup", "arguments": {"invoice_id": "B"}},
                    {"tool_name": "lookup", "arguments": {"invoice_id": "C"}},
                ]
            }
        return "draft"

    executor = AgentExecutor(
        MockProvider(responder=responder),
        model="m",
        planner=Planner(mode="static"),
        tool_runtime=runtime,
        tool_specs=specs,
    )
    state = await executor.run(
        Objective("do it", task_type=TaskType.TOOL_ACTION),
        budget=Budget(max_tool_calls=1),
    )
    # the slice + the in-loop guard cap execution at the budget.
    assert state.usage.tool_calls == 1


@pytest.mark.asyncio
async def test_plan_tools_approval_required_propagates_and_terminates():
    runtime, specs = _runtime_with_tool(approval=True)
    provider = MockProvider(responder=_tool_plan_responder("lookup", {"invoice_id": "Z"}))
    executor = AgentExecutor(
        provider,
        model="m",
        planner=Planner(mode="static"),
        tool_runtime=runtime,
        tool_specs=specs,
    )
    state = await executor.run(Objective("do it", task_type=TaskType.TOOL_ACTION))
    # ToolApprovalRequiredError raised inside _plan_and_call_tools bubbles to
    # _run_step which terminates the run as approval_required.
    assert state.termination_reason == "approval_required"
    assert any("approval" in (e.message or "").lower() for e in state.errors)


# ---------------------------------------------------------------------------
# _step_tool branches (explicit tool step)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_tool_step_skipped_without_runtime():
    executor = AgentExecutor(
        MockProvider(default_text="x"), model="m", planner=Planner(mode="static")
    )
    step = AgentStep(type="tool", name="t", tool_name="lookup", instruction="go")
    state = AgentState(objective=Objective("o"))
    await executor._step_tool(state, step)
    assert step.status == "skipped"
    assert step.error == "no tool runtime or tool name"


@pytest.mark.asyncio
async def test_explicit_tool_step_error_marks_step_failed():
    runtime, specs = _runtime_with_tool(fail=True)
    executor = AgentExecutor(
        MockProvider(default_text="x"),
        model="m",
        planner=Planner(mode="static"),
        tool_runtime=runtime,
        tool_specs=specs,
        repair=False,
    )
    state = AgentState(objective=Objective("o"))
    step = AgentStep(
        type="tool",
        name="t",
        tool_name="lookup",
        instruction="go",
        tool_arguments={"invoice_id": "INV-1"},
    )
    await executor._step_tool(state, step)
    # an error ToolResult marks the step failed so dependents aren't built on it.
    assert step.status == "failed"
    assert step.metadata["tool_status"] == "error"
    assert "backend exploded" in (step.error or "")
    assert step.result == {"status": "error", "error": step.error}


@pytest.mark.asyncio
async def test_explicit_tool_step_raises_validation_error_for_bad_args():
    runtime, specs = _runtime_with_tool()
    executor = AgentExecutor(
        MockProvider(default_text="x"),
        model="m",
        planner=Planner(mode="static"),
        tool_runtime=runtime,
        tool_specs=specs,
    )
    state = AgentState(objective=Objective("o"))
    step = AgentStep(
        type="tool",
        name="t",
        tool_name="lookup",
        instruction="go",
        tool_arguments={"invoice_id": 123},  # wrong type -> schema rejects
    )
    with pytest.raises(ToolValidationError, match="invalid arguments"):
        await executor._step_tool(state, step)


@pytest.mark.asyncio
async def test_explicit_tool_step_synthesizes_arguments_when_missing():
    runtime, specs = _runtime_with_tool()

    def responder(req):
        if req.output_schema_name == "tool_arguments":
            return {"arguments": {"invoice_id": "SYNTH"}}
        return "draft"

    executor = AgentExecutor(
        MockProvider(responder=responder),
        model="m",
        planner=Planner(mode="static"),
        tool_runtime=runtime,
        tool_specs=specs,
    )
    state = AgentState(objective=Objective("o"))
    step = AgentStep(type="tool", name="t", tool_name="lookup", instruction="look up")
    await executor._step_tool(state, step)
    assert step.result == {"invoice_id": "SYNTH", "amount": 42.0}
    assert state.tool_results[0].output["invoice_id"] == "SYNTH"


# ---------------------------------------------------------------------------
# _step_validate confidence + revised answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_failure_writes_revision_and_low_confidence():
    def responder(req):
        if req.output_schema_name == "validation":
            return {
                "passed": False,
                "issues": ["unsupported claim", "missing citation"],
                "revised_answer": "corrected answer",
            }
        return "draft"

    executor = AgentExecutor(
        MockProvider(responder=responder), model="m", planner=Planner(mode="static")
    )
    state = AgentState(objective=Objective("o"))
    state.working_memory["analyze"] = "first draft"
    step = AgentStep(type="validate", name="validate", instruction="check")
    await executor._step_validate(state, step)
    assert state.working_memory["analyze"] == "corrected answer"
    # confidence = 1 - 0.34 * 2 issues = 0.32
    assert state.working_memory["_confidence"] == pytest.approx(0.32)
    assert state.working_memory["validation"]["passed"] is False


@pytest.mark.asyncio
async def test_validate_uses_last_think_result_when_no_draft():
    def responder(req):
        if req.output_schema_name == "validation":
            # echo back the draft we saw to prove it was sourced from the think step.
            text = "\n".join(m.text for m in req.messages)
            return {"passed": True, "issues": [], "revised_answer": text[-20:]}
        return "draft"

    executor = AgentExecutor(
        MockProvider(responder=responder), model="m", planner=Planner(mode="static")
    )
    state = AgentState(objective=Objective("o"))
    think = AgentStep(type="think", name="reason", instruction="x")
    think.result = "THE-THINK-DRAFT"
    state.steps.append(think)
    step = AgentStep(type="validate", name="validate", instruction="check")
    await executor._step_validate(state, step)
    # passed=True keeps confidence at 1.0.
    assert state.working_memory["_confidence"] == 1.0


# ---------------------------------------------------------------------------
# ask_human branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_human_no_gate_terminates_approval_required():
    executor = AgentExecutor(
        MockProvider(default_text="x"), model="m", planner=Planner(mode="static")
    )
    state = AgentState(objective=Objective("o"))
    step = AgentStep(type="ask_human", name="ask", instruction="ok?")
    await executor._step_ask_human(state, step)
    assert step.status == "failed"
    assert state.terminated is True
    assert state.termination_reason == "approval_required"


@pytest.mark.asyncio
async def test_ask_human_denied_terminates():
    async def gate(prompt, ctx):
        return False

    executor = AgentExecutor(
        MockProvider(default_text="x"),
        model="m",
        planner=Planner(mode="static"),
        human_gate=gate,
    )
    state = AgentState(objective=Objective("o"))
    step = AgentStep(type="ask_human", name="ask", instruction="ok?")
    await executor._step_ask_human(state, step)
    assert step.result == {"approved": False}
    assert state.termination_reason == "approval_required"


@pytest.mark.asyncio
async def test_ask_human_approved_continues():
    seen = {}

    async def gate(prompt, ctx):
        seen.update(ctx)
        return True

    executor = AgentExecutor(
        MockProvider(default_text="x"),
        model="m",
        planner=Planner(mode="static"),
        human_gate=gate,
    )
    state = AgentState(objective=Objective("the objective"))
    step = AgentStep(type="ask_human", name="ask", instruction="ok?")
    await executor._step_ask_human(state, step)
    assert step.result == {"approved": True}
    assert state.terminated is False
    assert seen["objective"] == "the objective"


# ---------------------------------------------------------------------------
# finalize with output validator (both branches)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_validation_passed_sets_final_answer():
    contract = OutputContract(format="text")
    validator = OutputValidator(contract)
    executor = AgentExecutor(
        MockProvider(default_text="The final answer."),
        model="m",
        planner=Planner(mode="direct"),
        output_validator=validator,
    )
    state = await executor.run(Objective("answer please"))
    assert state.termination_reason == "validation_passed"
    assert state.final_answer == "The final answer."
    assert state.terminated is True


@pytest.mark.asyncio
async def test_finalize_validation_failed_records_error_objective_complete():
    # require_citations on a text contract -> the citation-free draft fails.
    contract = OutputContract(format="text", require_citations=True)
    validator = OutputValidator(contract)
    executor = AgentExecutor(
        MockProvider(default_text="No citations here at all."),
        model="m",
        planner=Planner(mode="direct"),
        output_validator=validator,
    )
    state = await executor.run(Objective("answer please"))
    assert state.termination_reason == "objective_complete"
    assert state.final_answer == "No citations here at all."
    assert any("output validation failed" in (e.message or "") for e in state.errors)


# ---------------------------------------------------------------------------
# retry path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_failure_retries_once():
    calls = {"n": 0}

    async def retrieve_fn(query):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient retrieval error")
        return [EvidenceItem(id="e1", source_id="D", text="found")]

    executor = AgentExecutor(
        MockProvider(default_text="draft"),
        model="m",
        planner=Planner(mode="static"),
        retrieve_fn=retrieve_fn,
        repair=False,
    )
    state = await executor.run(
        Objective("answer", task_type=TaskType.DOCUMENT_QA),
        budget=Budget(max_retries=2),
    )
    # first attempt raised, retry succeeded -> evidence gathered + a retry counted.
    assert calls["n"] == 2
    assert state.usage.retries == 1
    assert any(e.id == "e1" for e in state.evidence)


# ---------------------------------------------------------------------------
# plan_and_execute replan loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_and_execute_replans_when_validation_fails():
    # The validate step always fails -> _needs_replan True -> heuristic replan
    # runs revise+refinalize. The planner has no provider so replan is heuristic.
    def responder(req):
        if req.output_schema_name == "validation":
            return {"passed": False, "issues": ["needs work"], "revised_answer": "v2"}
        if req.output_schema_name == "agent_plan":
            # force dynamic plan to fall back to static so the validate step exists.
            return "not json"
        return "draft"

    executor = AgentExecutor(
        MockProvider(responder=responder),
        model="m",
        planner=Planner(mode="plan_and_execute"),
        max_replans=2,
    )
    state = await executor.run(
        Objective("answer", task_type=TaskType.DOCUMENT_QA),
        budget=Budget(max_steps=40, max_cost_usd=100.0, max_input_tokens=10_000_000),
    )
    # replans recorded; at least one corrective revise step was added.
    assert state.working_memory.get("_replans", 0) >= 1
    assert any(s.name == "revise" for s in state.steps)


@pytest.mark.asyncio
async def test_needs_replan_false_on_approval_required():
    executor = AgentExecutor(
        MockProvider(default_text="x"), model="m", planner=Planner(mode="plan_and_execute")
    )
    state = AgentState(objective=Objective("o"))
    state.termination_reason = "approval_required"
    assert executor._needs_replan(state) is False


@pytest.mark.asyncio
async def test_needs_replan_true_when_no_final_answer():
    executor = AgentExecutor(
        MockProvider(default_text="x"), model="m", planner=Planner(mode="plan_and_execute")
    )
    state = AgentState(objective=Objective("o"))
    state.final_answer = None
    assert executor._needs_replan(state) is True


# ---------------------------------------------------------------------------
# astream error surfacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_surfaces_error_event_on_exception():
    # An exception that escapes run() itself (planning, not a single step) is
    # surfaced by astream as a terminal ``error`` event.
    class BoomPlanner(Planner):
        async def plan(self, *args, **kwargs):
            raise RuntimeError("planner down")

    executor = AgentExecutor(
        MockProvider(default_text="x"), model="m", planner=BoomPlanner(mode="static")
    )
    events = [ev async for ev in executor.astream(Objective("o"))]
    types = [e.type for e in events]
    assert "run_start" in types
    assert "done" not in types
    error_events = [e for e in events if e.type == "error"]
    assert error_events and "planner down" in (error_events[0].error or "")


@pytest.mark.asyncio
async def test_astream_done_event_carries_final_state():
    executor = AgentExecutor(
        MockProvider(default_text="final"), model="m", planner=Planner(mode="direct")
    )
    events = [ev async for ev in executor.astream(Objective("o"))]
    done = [e for e in events if e.type == "done"]
    assert len(done) == 1
    assert done[0].result == "final"
    assert done[0].payload["termination_reason"] == "objective_complete"
    # real token deltas were streamed for the finalize answer.
    assert any(e.type == "text_delta" for e in events)


# ---------------------------------------------------------------------------
# ReAct loop branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_tool_error_recorded_in_transcript():
    runtime, specs = _runtime_with_tool(fail=True)

    calls = {"n": 0}

    def responder(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"tool_call": {"name": "lookup", "arguments": {"invoice_id": "Q"}}}
        return "done answering"

    executor = AgentExecutor(
        MockProvider(responder=responder),
        model="m",
        planner=Planner(mode="react"),
        tool_runtime=runtime,
        tool_specs=specs,
    )
    state = await executor.run("solve it", budget=Budget(max_steps=5, max_tool_calls=3))
    # the failing tool produced an error ToolResult (not a crash) and the loop
    # continued to a final text answer.
    assert any(r.status == "error" for r in state.tool_results)
    assert state.final_answer == "done answering"
    assert state.termination_reason == "objective_complete"


@pytest.mark.asyncio
async def test_react_tool_call_budget_breaks_loop():
    runtime, specs = _runtime_with_tool()

    def responder(req):
        # always ask for a tool call -> would loop, but tool-call budget stops it.
        return {"tool_call": {"name": "lookup", "arguments": {"invoice_id": "L"}}}

    executor = AgentExecutor(
        MockProvider(responder=responder),
        model="m",
        planner=Planner(mode="react"),
        tool_runtime=runtime,
        tool_specs=specs,
    )
    state = await executor.run("loop", budget=Budget(max_steps=10, max_tool_calls=2))
    assert state.usage.tool_calls == 2
    assert state.termination_reason == "budget_exhausted"


@pytest.mark.asyncio
async def test_react_with_evidence_and_validator_passes():
    contract = OutputContract(format="text")
    validator = OutputValidator(contract)
    executor = AgentExecutor(
        MockProvider(default_text="grounded answer"),
        model="m",
        planner=Planner(mode="react"),
        output_validator=validator,
    )
    state = await executor.run(
        Objective("answer"),
        initial_evidence=[EvidenceItem(id="e1", source_id="D", text="fact")],
        budget=Budget(max_steps=3),
    )
    assert state.termination_reason == "validation_passed"
    assert state.final_answer == "grounded answer"


@pytest.mark.asyncio
async def test_react_tool_validation_error_becomes_error_result():
    # The model asks for a tool call with the wrong arg type -> ToolRuntime
    # raises ToolValidationError, which the ReAct loop catches into an error
    # ToolResult (the loop's VincioError branch) rather than crashing.
    runtime, specs = _runtime_with_tool()

    calls = {"n": 0}

    def responder(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"tool_call": {"name": "lookup", "arguments": {}}}  # missing arg
        return "final react answer"

    executor = AgentExecutor(
        MockProvider(responder=responder),
        model="m",
        planner=Planner(mode="react"),
        tool_runtime=runtime,
        tool_specs=specs,
    )
    state = await executor.run("solve", budget=Budget(max_steps=5, max_tool_calls=3))
    err = next(r for r in state.tool_results if r.status == "error")
    assert "invalid arguments" in (err.error or "")
    assert state.final_answer == "final react answer"


# ---------------------------------------------------------------------------
# retrieve skip, step error handlers, selector, repair recording
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_step_skipped_without_retrieve_fn():
    executor = AgentExecutor(
        MockProvider(default_text="x"), model="m", planner=Planner(mode="static")
    )
    state = AgentState(objective=Objective("o"))
    step = AgentStep(type="retrieve", name="retrieve", instruction="find")
    await executor._step_retrieve(state, step)
    assert step.status == "skipped"
    assert step.error == "no retrieval configured"
    assert state.evidence == []


@pytest.mark.asyncio
async def test_run_step_vincio_error_recorded_recoverable():
    # A retrieve_fn that raises a VincioError exercises the typed-error handler
    # in _run_step (recoverable error recorded, step failed).
    from vincio.core.errors import RetrievalError

    async def retrieve_fn(query):
        raise RetrievalError("index offline")

    executor = AgentExecutor(
        MockProvider(default_text="x"),
        model="m",
        planner=Planner(mode="static"),
        retrieve_fn=retrieve_fn,
        repair=False,
    )
    state = AgentState(objective=Objective("o"))
    step = AgentStep(type="retrieve", name="retrieve", instruction="find")
    await executor._run_step(state, step)
    assert step.status == "failed"
    assert step.error == "index offline"
    err = next(e for e in state.errors if e.step_id == step.id)
    assert err.recoverable is True
    assert state.usage.steps == 1


@pytest.mark.asyncio
async def test_run_step_unexpected_exception_recorded():
    # A generic (non-Vincio) exception is caught and surfaced with its type name.
    async def retrieve_fn(query):
        raise ValueError("boom value")

    executor = AgentExecutor(
        MockProvider(default_text="x"),
        model="m",
        planner=Planner(mode="static"),
        retrieve_fn=retrieve_fn,
        repair=False,
    )
    state = AgentState(objective=Objective("o"))
    step = AgentStep(type="retrieve", name="retrieve", instruction="find")
    await executor._run_step(state, step)
    assert step.status == "failed"
    assert step.error == "ValueError: boom value"


@pytest.mark.asyncio
async def test_cost_aware_selector_records_selection_metadata():
    from vincio.agents.selection import CostAwareSelector

    executor = AgentExecutor(
        MockProvider(default_text="answer"),
        model="m",
        planner=Planner(mode="direct"),
        selector=CostAwareSelector(["m"]),
    )
    state = await executor.run(Objective("answer"))
    # the selector chose a model and stamped a selection record in working memory.
    selections = state.working_memory.get("_selections")
    assert selections and selections[0]["model"] == "m"
    assert "tier" in selections[0]


@pytest.mark.asyncio
async def test_select_model_returns_none_without_selector():
    executor = AgentExecutor(MockProvider(default_text="x"), model="m")
    state = AgentState(objective=Objective("o"))
    from vincio.core.types import Message

    chosen = executor._select_model(
        state, [Message(role="user", content="hi")], needs_structured=False
    )
    assert chosen is None


@pytest.mark.asyncio
async def test_tool_failure_repair_publishes_event_and_records_repair():
    # repair on (default): a failing tool step triggers the repairer, which
    # records a PlanRepair and publishes a typed PlanRepaired event.
    from vincio.core.events import EventBus, PlanRepaired

    bus = EventBus()
    seen: list = []
    bus.subscribe(PlanRepaired.event, lambda ev: seen.append(ev))

    runtime, specs = _runtime_with_tool(fail=True)

    def responder(req):
        if req.output_schema_name == "tool_calls":
            return {"calls": [{"tool_name": "lookup", "arguments": {"invoice_id": "X"}}]}
        if req.output_schema_name == "tool_arguments":
            return {"arguments": {"invoice_id": "X"}}
        return "draft"

    executor = AgentExecutor(
        MockProvider(responder=responder),
        model="m",
        planner=Planner(mode="static"),
        tool_runtime=runtime,
        tool_specs=specs,
        events=bus,
    )
    # an explicit tool step is what the repairer can rebind/substitute on failure.
    from vincio.agents.dag import StepDAG

    dag = StepDAG()
    tool_step = AgentStep(
        type="tool",
        name="call_tool",
        tool_name="lookup",
        instruction="look up",
        tool_arguments={"invoice_id": "X"},
    )
    dag.add(tool_step)
    state = AgentState(objective=Objective("o", task_type=TaskType.TOOL_ACTION))
    state.steps = list(dag.steps.values())
    executor.attribution = {"run_id": state.id}
    await executor._execute_dag(state, dag)
    # the tool step failed and a repair was recorded on the trajectory.
    # the tool failed; the repairer substituted it (reasoning instead) and the
    # repair was recorded on the trajectory and published on the bus.
    assert state.repairs, "expected the repairer to record a repair"
    assert state.repairs[0].trigger == "tool_failure"
    assert state.repairs[0].action == "substitute"
    assert seen and seen[0].payload["step_name"] == "call_tool"
    assert seen[0].payload["trigger"] == "tool_failure"


@pytest.mark.asyncio
async def test_replan_loop_breaks_when_replan_returns_no_steps():
    # planner.replan returns None -> the loop breaks immediately, _replans == 0.
    class NoReplan(Planner):
        async def replan(self, *args, **kwargs):
            return None

    executor = AgentExecutor(
        MockProvider(default_text="draft"),
        model="m",
        planner=NoReplan(mode="plan_and_execute"),
    )
    state = AgentState(objective=Objective("o"))
    # force _needs_replan True: no final answer yet.
    await executor._replan_loop(state, state.objective)
    assert state.working_memory["_replans"] == 0


@pytest.mark.asyncio
async def test_replan_loop_breaks_when_budget_exhausted():
    executor = AgentExecutor(
        MockProvider(default_text="draft"),
        model="m",
        planner=Planner(mode="plan_and_execute"),
    )
    state = AgentState(objective=Objective("o"), budget=Budget(max_steps=2))
    state.usage.steps = 5  # already over the step budget
    await executor._replan_loop(state, state.objective)
    assert state.working_memory["_replans"] == 0


@pytest.mark.asyncio
async def test_run_unrecoverable_error_when_finalize_fails():
    # A provider that fails the finalize call leaves the run not-terminated with
    # an error and no answer -> run() stamps termination_reason unrecoverable_error.
    class FailFinalize(MockProvider):
        async def generate(self, request):
            raise RuntimeError("model unavailable")

        async def stream(self, request):
            raise RuntimeError("model unavailable")
            yield  # pragma: no cover - generator marker

    executor = AgentExecutor(
        FailFinalize(), model="m", planner=Planner(mode="direct")
    )
    state = await executor.run(Objective("answer"))
    assert state.terminated is True
    assert state.final_answer is None
    assert state.termination_reason == "unrecoverable_error"
    assert any("model unavailable" in (e.message or "") for e in state.errors)


@pytest.mark.asyncio
async def test_cost_ledger_attributes_model_calls():
    from vincio.observability.finops import CostLedger

    ledger = CostLedger()
    executor = AgentExecutor(
        MockProvider(default_text="answer"),
        model="m",
        planner=Planner(mode="direct"),
        cost_ledger=ledger,
    )
    state = await executor.run(
        Objective("answer"),
        attribution={"tenant_id": "acme", "feature": "qa"},
    )
    # every model call was recorded against the supplied attribution + run id.
    assert ledger.events, "expected a cost event recorded"
    ev = ledger.events[0]
    assert ev.tenant_id == "acme"
    assert ev.feature == "qa"
    assert ev.run_id == state.id


@pytest.mark.asyncio
async def test_system_prompt_is_prepended_to_context():
    seen = {}

    def responder(req):
        seen["roles"] = [m.role for m in req.messages]
        seen["system"] = next((m.text for m in req.messages if m.role == "system"), None)
        return "answer"

    executor = AgentExecutor(
        MockProvider(responder=responder),
        model="m",
        planner=Planner(mode="direct"),
        system_prompt="You are terse.",
    )
    await executor.run(Objective("answer"))
    assert seen["roles"][0] == "system"
    assert seen["system"] == "You are terse."


@pytest.mark.asyncio
async def test_finalize_uses_contract_output_schema():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    contract = OutputContract(format="json", schema_def=schema, schema_name="final")
    validator = OutputValidator(contract)

    seen = {}

    def responder(req):
        # the finalize call must carry the contract's schema.
        if req.output_schema is not None:
            seen["name"] = req.output_schema_name
            seen["schema"] = req.output_schema
        return {"answer": "structured result"}

    executor = AgentExecutor(
        MockProvider(responder=responder),
        model="m",
        planner=Planner(mode="direct"),
        output_validator=validator,
    )
    state = await executor.run(Objective("answer"))
    assert seen["name"] == "final"
    assert seen["schema"] == schema
    assert state.final_answer == {"answer": "structured result"}
    assert state.termination_reason == "validation_passed"


# ---------------------------------------------------------------------------
# compaction, budget-shock repair, low-level guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_compacts_large_evidence_into_summary():
    seen = {}

    def responder(req):
        text = "\n".join(m.text for m in req.messages)
        seen["text"] = text
        return "answer"

    # small context budget -> section_budget = 512; many big evidence blocks
    # force the older ones into a rolling summary.
    executor = AgentExecutor(
        MockProvider(responder=responder),
        model="m",
        planner=Planner(mode="direct"),
        max_context_tokens=1024,
    )
    big = "lorem ipsum dolor sit amet consectetur " * 40
    evidence = [
        EvidenceItem(id=f"e{i}", source_id="D", text=f"{big} item {i}") for i in range(12)
    ]
    await executor.run(Objective("answer"), initial_evidence=evidence)
    assert "(earlier evidence summarized)" in seen["text"]


@pytest.mark.asyncio
async def test_budget_shock_repair_drops_optional_tail():
    from vincio.agents.dag import StepDAG

    executor = AgentExecutor(
        MockProvider(default_text="x"), model="m", planner=Planner(mode="static")
    )
    dag = StepDAG()
    think = dag.add(AgentStep(type="think", name="analyze", instruction="reason"))
    dag.add(
        AgentStep(type="finalize", name="finalize", instruction="answer"),
        depends_on=[think.id],
    )
    state = AgentState(objective=Objective("o"), budget=Budget(max_cost_usd=1.0))
    state.usage.cost_usd = 0.9  # 90% spent -> over the 75% shock fraction
    state.steps = list(dag.steps.values())
    executor.attribution = {"run_id": state.id}
    executor._maybe_repair_budget_shock(state, dag)
    assert state.repairs and state.repairs[0].trigger == "budget_shock"
    assert think.status == "skipped"
    assert think.error == "dropped under budget pressure"


@pytest.mark.asyncio
async def test_execute_dag_breaks_when_no_ready_steps():
    # A DAG whose only step is already failed has no ready steps -> the loop
    # breaks without running anything (repair disabled so nothing reopens it).
    from vincio.agents.dag import StepDAG

    executor = AgentExecutor(
        MockProvider(default_text="x"),
        model="m",
        planner=Planner(mode="static"),
        repair=False,
    )
    dag = StepDAG()
    a = dag.add(AgentStep(type="think", name="a"))
    dag.add(AgentStep(type="think", name="b"), depends_on=[a.id])
    a.status = "failed"  # b becomes unreachable; ready_steps() is empty
    state = AgentState(objective=Objective("o"))
    state.steps = list(dag.steps.values())
    await executor._execute_dag(state, dag)
    # nothing executed: no model calls were charged.
    assert state.usage.steps == 0


@pytest.mark.asyncio
async def test_run_step_bounded_skips_when_already_terminated():
    import asyncio

    executor = AgentExecutor(
        MockProvider(default_text="x"), model="m", planner=Planner(mode="static")
    )
    state = AgentState(objective=Objective("o"))
    state.terminated = True
    step = AgentStep(type="think", name="t", instruction="x")
    await executor._run_step_bounded(state, step, asyncio.Semaphore(1))
    # the guard short-circuited: the step never ran.
    assert step.status == "pending"
    assert step.attempts == 0


@pytest.mark.asyncio
async def test_react_terminates_immediately_when_budget_already_exhausted():
    executor = AgentExecutor(
        MockProvider(default_text="x"), model="m", planner=Planner(mode="react")
    )
    # max_steps=0 -> the very first _check_termination at the top of the ReAct
    # loop trips and no model call is ever made.
    state = await executor.run("solve", budget=Budget(max_steps=0))
    assert state.terminated is True
    assert state.termination_reason == "max_steps"
    assert state.usage.steps == 0
    assert state.final_answer is None


@pytest.mark.asyncio
async def test_context_compacts_large_tool_results_into_summary():
    from vincio.core.types import ToolResult

    seen = {}

    def responder(req):
        seen["text"] = "\n".join(m.text for m in req.messages)
        return "answer"

    executor = AgentExecutor(
        MockProvider(responder=responder),
        model="m",
        planner=Planner(mode="static"),
        max_context_tokens=1024,
    )
    state = AgentState(objective=Objective("o"))
    blob = "word " * 300
    state.tool_results = [
        ToolResult(call_id=str(i), tool_name="lookup", status="ok", output={"data": blob, "i": i})
        for i in range(12)
    ]
    step = AgentStep(type="think", name="analyze", instruction="reason")
    await executor._step_think(state, step)
    assert "(earlier tool results summarized)" in seen["text"]


@pytest.mark.asyncio
async def test_plan_tools_inner_budget_guard_breaks_loop():
    # usage already at the tool-call cap -> the in-loop guard breaks before
    # executing even the first planned call.
    runtime, specs = _runtime_with_tool()
    provider = MockProvider(responder=_tool_plan_responder("lookup", {"invoice_id": "X"}))
    executor = AgentExecutor(
        provider,
        model="m",
        planner=Planner(mode="static"),
        tool_runtime=runtime,
        tool_specs=specs,
    )
    state = AgentState(objective=Objective("o"), budget=Budget(max_tool_calls=2))
    state.usage.tool_calls = 2  # already spent the whole tool-call budget
    step = AgentStep(type="think", name="plan_tools", instruction="call tools")
    await executor._plan_and_call_tools(state, step)
    # no new tool calls were made; the guard stopped the loop.
    assert state.usage.tool_calls == 2
    assert state.tool_results == []
    assert step.result == []
