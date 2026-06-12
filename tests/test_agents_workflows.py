"""Agent + workflow engine tests."""

import pytest

from vincio.agents import AgentExecutor, AgentStep, HandoffRouter, Planner, StepDAG
from vincio.core.errors import AgentEngineError
from vincio.core.types import Budget, EvidenceItem, Objective, TaskType
from vincio.providers import MockProvider
from vincio.tools import ToolRegistry, ToolRuntime
from vincio.workflows import Workflow


@pytest.fixture()
def tool_registry():
    registry = ToolRegistry()

    @registry.register()
    def lookup_invoice(invoice_id: str) -> dict:
        """Get invoice details."""
        return {"invoice_id": invoice_id, "amount": 42.0, "status": "paid"}

    return registry


class TestDAG:
    def test_topology_and_cycle_detection(self):
        dag = StepDAG()
        a = dag.add(AgentStep(type="retrieve", name="a"))
        dag.add(AgentStep(type="think", name="b"), depends_on=[a.id])
        levels = dag.topological_levels()
        assert [s.name for s in levels[0]] == ["a"]
        assert [s.name for s in levels[1]] == ["b"]
        with pytest.raises(AgentEngineError):
            dag.add(AgentStep(type="think", name="c"), depends_on=["missing"])

    def test_failed_upstream_skips(self):
        dag = StepDAG()
        a = dag.add(AgentStep(type="think", name="a"))
        dag.add(AgentStep(type="think", name="b"), depends_on=[a.id])
        a.status = "failed"
        assert dag.ready_steps() == []
        assert dag.steps[[s for s in dag.steps if dag.steps[s].name == "b"][0]].status == "skipped"


class TestPlanner:
    @pytest.mark.asyncio
    async def test_static_plan_shape(self):
        planner = Planner(mode="static")
        dag = await planner.plan(
            Objective("answer", task_type=TaskType.DOCUMENT_QA), has_retrieval=True
        )
        names = [s.name for level in dag.topological_levels() for s in level]
        assert names[0] == "retrieve" and names[-1] == "finalize"

    @pytest.mark.asyncio
    async def test_dynamic_plan_falls_back_safely(self):
        planner = Planner(mode="dynamic", provider=MockProvider(default_text="not json"), model="m")
        dag = await planner.plan(Objective("answer"))
        assert any(s.type == "finalize" for s in dag.steps.values())


class TestExecutor:
    @pytest.mark.asyncio
    async def test_dag_run(self, tool_registry):
        async def retrieve(query):
            return [
                EvidenceItem(id="E1", source_id="D1", text="Refunds allowed within 30 days.", relevance=0.9)
            ]

        executor = AgentExecutor(
            MockProvider(),
            model="mock-1",
            planner=Planner(mode="static"),
            tool_runtime=ToolRuntime(tool_registry),
            tool_specs=tool_registry.specs(),
            retrieve_fn=retrieve,
        )
        state = await executor.run(
            Objective("Decide refund eligibility for INV-1", task_type=TaskType.TOOL_ACTION),
            budget=Budget(max_steps=12),
        )
        assert state.terminated
        assert state.final_answer is not None
        assert all(s.status in ("done", "skipped") for s in state.steps)
        metrics = state.metrics()
        assert metrics["success"] and metrics["steps_done"] >= 4

    @pytest.mark.asyncio
    async def test_react_with_tools(self, tool_registry):
        provider = MockProvider(
            script=[
                {"tool_call": {"name": "lookup_invoice", "arguments": {"invoice_id": "INV-1"}}},
                "Invoice INV-1 is paid; refund eligible.",
            ]
        )
        executor = AgentExecutor(
            provider,
            model="mock-1",
            planner=Planner(mode="react"),
            tool_runtime=ToolRuntime(tool_registry),
            tool_specs=tool_registry.specs(),
        )
        state = await executor.run("Is INV-1 refundable?")
        assert [r.tool_name for r in state.tool_results] == ["lookup_invoice"]
        assert "eligible" in str(state.final_answer)

    @pytest.mark.asyncio
    async def test_budget_exhaustion_terminates(self, tool_registry):
        # Provider always asks for another tool call -> must hit the budget.
        provider = MockProvider(
            responder=lambda req: {
                "tool_call": {"name": "lookup_invoice", "arguments": {"invoice_id": "X"}}
            }
        )
        executor = AgentExecutor(
            provider,
            model="mock-1",
            planner=Planner(mode="react"),
            tool_runtime=ToolRuntime(tool_registry, cache_enabled=False),
            tool_specs=tool_registry.specs(),
        )
        state = await executor.run("loop forever", budget=Budget(max_steps=3, max_tool_calls=2))
        assert state.termination_reason in ("budget_exhausted", "max_steps")


class TestHandoffs:
    @pytest.mark.asyncio
    async def test_select_and_run(self):
        router = HandoffRouter()
        billing = AgentExecutor(MockProvider(default_text="billing answer"), model="m", planner=Planner(mode="direct"))
        legal = AgentExecutor(MockProvider(default_text="legal answer"), model="m", planner=Planner(mode="direct"))
        router.register("billing", billing, description="billing refunds invoices payments", keywords=["invoice", "refund"])
        router.register("legal", legal, description="contracts legal clauses", keywords=["contract", "clause"])
        assert router.select("Is this invoice refundable?") == "billing"
        state = await router.run("Review the termination clause in the contract")
        assert "legal" in str(state.final_answer)


class TestWorkflows:
    @pytest.mark.asyncio
    async def test_dag_with_named_bindings(self):
        workflow = Workflow("calc")
        workflow.step("double", lambda input: input * 2)
        workflow.step("plus_one", lambda double: double + 1, depends_on=["double"])
        result = await workflow.arun(5)
        assert result.status == "succeeded"
        assert result.context.output_of("plus_one") == 11

    @pytest.mark.asyncio
    async def test_retries(self):
        attempts = {"n": 0}

        def flaky(input):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("flaky")
            return "ok"

        workflow = Workflow("retry")
        workflow.step("flaky", flaky, retries=3, retry_delay_s=0.01)
        result = await workflow.arun(1)
        assert result.status == "succeeded"
        assert result.context.results["flaky"].attempts == 3

    @pytest.mark.asyncio
    async def test_branching(self):
        workflow = Workflow("branch")
        workflow.step("classify", lambda input: input)
        workflow.step(
            "high_risk", lambda ctx: "escalated", depends_on=["classify"],
            when=lambda ctx: ctx.output_of("classify") == "high",
        )
        workflow.step(
            "low_risk", lambda ctx: "auto-approved", depends_on=["classify"],
            when=lambda ctx: ctx.output_of("classify") == "low",
        )
        result = await workflow.arun("high")
        assert result.context.results["high_risk"].status == "done"
        assert result.context.results["low_risk"].status == "skipped"

    @pytest.mark.asyncio
    async def test_compensation_on_failure(self):
        undone = []
        workflow = Workflow("comp")
        workflow.step("reserve", lambda input: "reserved", compensation=lambda ctx: undone.append("reserve"))
        workflow.step("charge", lambda reserve: 1 / 0, depends_on=["reserve"])
        result = await workflow.arun(1)
        assert result.status == "partial"
        assert result.failed_steps == ["charge"]
        assert undone == ["reserve"]
        assert result.context.results["reserve"].status == "compensated"

    @pytest.mark.asyncio
    async def test_parallel_level(self):
        import asyncio

        order = []

        async def make(name, delay):
            await asyncio.sleep(delay)
            order.append(name)
            return name

        workflow = Workflow("par")
        workflow.step("slow", lambda input: make("slow", 0.05))

        async def slow_step(input):
            return await make("slow", 0.05)

        async def fast_step(input):
            return await make("fast", 0.0)

        workflow2 = Workflow("par2")
        workflow2.step("a", slow_step)
        workflow2.step("b", fast_step)
        result = await workflow2.arun(1)
        assert result.status == "succeeded"
        assert order == ["fast", "slow"]  # ran concurrently

    @pytest.mark.asyncio
    async def test_approval_gate(self):
        async def deny(step, context):
            return False

        workflow = Workflow("appr", approval_fn=deny)
        workflow.step("dangerous", lambda input: "done", approval=True)
        result = await workflow.arun(1)
        assert result.status == "failed"
        assert "approval denied" in result.context.results["dangerous"].error
