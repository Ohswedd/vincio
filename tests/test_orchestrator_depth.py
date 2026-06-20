"""Orchestrator & planner depth: hierarchical (HTN) planning, in-place plan
repair, cost-aware action selection, parallel sub-graph scheduling, and durable
timers / scheduled steps. All offline and deterministic on the mock provider."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from vincio.agents import (
    AgentExecutor,
    CostAwareSelector,
    HTNDomain,
    HTNPlanNode,
    Planner,
    PlanRepairer,
    StateGraph,
    SubgraphScheduler,
    SubgraphTask,
    TimerService,
    dag_from_plan_node,
    deliver_event,
    due_timers,
    pending_timers,
    sleep_for,
    sleep_until,
    wait_for_event,
)
from vincio.agents.dag import StepDAG
from vincio.agents.graph import Checkpointer
from vincio.agents.state import AgentState, AgentStep
from vincio.core.errors import CapabilityMismatchError
from vincio.core.events import EventBus
from vincio.core.types import (
    Budget,
    ModelCapabilities,
    ModelProfile,
    Objective,
    TaskType,
)
from vincio.observability.costs import CostTracker, ModelPrice, PriceTable
from vincio.providers import MockProvider
from vincio.providers.capabilities import RequestNeeds
from vincio.providers.registry import ModelRegistry
from vincio.storage.base import InMemoryMetadataStore
from vincio.tools.registry import ToolRegistry
from vincio.tools.runtime import ToolRuntime

# ---------------------------------------------------------------------------
# Hierarchical (HTN) planning
# ---------------------------------------------------------------------------


def _sample_domain() -> HTNDomain:
    return (
        HTNDomain()
        .method("root", ["gather", "analyze", "answer"])
        .method("gather", ["search", "lookup"], ordering="parallel")
        .operator("search", step_type="retrieve", instruction="search the kb")
        .operator("lookup", step_type="tool", tool_name="probe", instruction="look up")
        .operator("analyze", step_type="think", instruction="analyze")
        .operator("answer", step_type="finalize", instruction="final answer")
    )


class TestHierarchicalPlanner:
    def test_deterministic_decomposition_tree(self):
        root = _sample_domain().decompose("root", context={"has_retrieval": True})
        assert isinstance(root, HTNPlanNode)
        assert [op.name for op in root.leaves()] == ["search", "lookup", "analyze", "answer"]
        # parallel sub-goal exposes both children on one level
        gather = root.children[0]
        assert gather.ordering == "parallel"

    def test_parallel_method_lands_on_one_level(self):
        dag = dag_from_plan_node(
            _sample_domain().decompose("root"), available_tools={"probe"}
        )
        levels = dag.topological_levels()
        assert {s.name for s in levels[0]} == {"search", "lookup"}  # parallel siblings
        assert levels[-1][0].type == "finalize"

    def test_unbound_tool_degrades_to_reasoning(self):
        dag = dag_from_plan_node(_sample_domain().decompose("root"), available_tools=set())
        lookup = next(s for s in dag.steps.values() if s.name == "lookup")
        assert lookup.type == "think" and lookup.tool_name is None

    def test_precondition_selects_method(self):
        domain = (
            HTNDomain()
            .method("root", ["online"], when={"has_retrieval": True})
            .method("root", ["offline"], when={"has_retrieval": False})
            .operator("online", step_type="retrieve")
            .operator("offline", step_type="think")
        )
        on = domain.decompose("root", context={"has_retrieval": True})
        off = domain.decompose("root", context={"has_retrieval": False})
        assert on.leaves()[0].name == "online"
        assert off.leaves()[0].name == "offline"

    def test_recursion_guard_bottoms_out(self):
        cyclic = HTNDomain().method("root", ["root"])  # self-referential
        root = cyclic.decompose("root")
        # Bounded: never recurses forever; bottoms out as a reasoning leaf.
        assert root.leaves()

    async def test_planner_hierarchical_mode_with_domain(self):
        planner = Planner(mode="hierarchical", domain=_sample_domain())
        dag = await planner.plan(
            Objective("do it", task_type=TaskType.AGENT_WORKFLOW), has_retrieval=True
        )
        assert planner.last_plan_tree is not None
        assert isinstance(dag, StepDAG)
        assert any(s.type == "finalize" for s in dag.steps.values())

    async def test_planner_offline_no_domain_falls_back_to_static(self):
        planner = Planner(mode="hierarchical")  # no domain, no provider
        dag = await planner.plan(Objective("answer"), has_retrieval=False)
        assert planner.last_plan_tree is None
        assert any(s.type == "finalize" for s in dag.steps.values())

    async def test_planner_llm_decomposition(self):
        def responder(request):
            if request.output_schema_name == "htn_plan":
                return {
                    "subgoals": [
                        {
                            "name": "research",
                            "method": "parallel",
                            "steps": [
                                {"name": "s1", "type": "think", "instruction": "a", "tool_name": ""},
                                {"name": "s2", "type": "think", "instruction": "b", "tool_name": ""},
                            ],
                        },
                        {
                            "name": "write",
                            "method": "sequence",
                            "steps": [
                                {"name": "fin", "type": "finalize", "instruction": "answer", "tool_name": ""},
                            ],
                        },
                    ]
                }
            return None

        planner = Planner(mode="hierarchical", provider=MockProvider(responder=responder), model="mock-1")
        dag = await planner.plan(Objective("write a brief"))
        levels = dag.topological_levels()
        assert {s.name for s in levels[0]} == {"s1", "s2"}  # parallel sub-goal
        assert any(s.type == "finalize" for s in dag.steps.values())

    async def test_executor_runs_hierarchical_plan(self):
        registry = ToolRegistry()

        @registry.register()
        def probe(q: str = "x") -> dict:
            """Probe."""
            return {"q": q}

        from vincio.core.types import EvidenceItem

        async def retrieve_fn(query: str) -> list[EvidenceItem]:
            return [
                EvidenceItem(id="e1", source_id="src", text="the invoice was paid", citation_ref="E1")
            ]

        executor = AgentExecutor(
            MockProvider(),
            model="mock-1",
            planner=Planner(mode="hierarchical", domain=_sample_domain()),
            tool_runtime=ToolRuntime(registry, cache_enabled=False),
            tool_specs=registry.specs(),
            retrieve_fn=retrieve_fn,
        )
        state = await executor.run("Investigate the dispute", budget=Budget(max_steps=12))
        assert state.terminated
        assert state.final_answer is not None
        # the parallel "gather" sub-goal genuinely ran both leaves
        assert {s.name for s in state.steps if s.status == "done"} >= {"search", "lookup"}


# ---------------------------------------------------------------------------
# Plan repair & replanning
# ---------------------------------------------------------------------------


def _tool_dag(tool_name: str, *, metadata: dict | None = None) -> tuple[StepDAG, AgentStep, AgentStep]:
    dag = StepDAG()
    tool_step = AgentStep(
        type="tool", name="lookup", instruction="look up", tool_name=tool_name, metadata=metadata or {}
    )
    dag.add(tool_step)
    finalize = AgentStep(type="finalize", name="finalize", instruction="answer")
    dag.add(finalize, depends_on=[tool_step.id])
    return dag, tool_step, finalize


def _failing_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register()
    def billing_lookup_primary(invoice: str = "x") -> dict:
        """Primary (flaky)."""
        raise RuntimeError("upstream 503")

    @registry.register()
    def billing_lookup_backup(invoice: str = "x") -> dict:
        """Backup."""
        return {"invoice": invoice, "amount": 42}

    return registry


class TestPlanRepair:
    def _executor(self, registry: ToolRegistry, *, repair: bool = True, events: EventBus | None = None):
        return AgentExecutor(
            MockProvider(),
            model="mock-1",
            planner=Planner(mode="static"),
            tool_runtime=ToolRuntime(registry, cache_enabled=False),
            tool_specs=registry.specs(),
            events=events,
            repair=repair,
        )

    async def test_rebind_to_explicit_fallback(self):
        registry = _failing_registry()
        executor = self._executor(registry)
        dag, tool_step, finalize = _tool_dag(
            "billing_lookup_primary", metadata={"fallback_tools": ["billing_lookup_backup"]}
        )
        state = AgentState(objective=Objective("refund"), budget=Budget(max_steps=12))
        state.steps = list(dag.steps.values())
        await executor._execute_dag(state, dag)
        assert [r.action for r in state.repairs] == ["rebind"]
        assert tool_step.tool_name == "billing_lookup_backup" and tool_step.status == "done"
        assert finalize.status == "done"

    async def test_rebind_to_name_overlap_sibling(self):
        registry = _failing_registry()
        executor = self._executor(registry)
        # No explicit fallback — repair finds the name-overlap sibling.
        dag, tool_step, _ = _tool_dag("billing_lookup_primary")
        state = AgentState(objective=Objective("x"), budget=Budget(max_steps=12))
        state.steps = list(dag.steps.values())
        await executor._execute_dag(state, dag)
        assert state.repairs[0].action == "rebind"
        assert tool_step.tool_name == "billing_lookup_backup"

    async def test_rebind_then_substitute_when_backup_also_fails(self):
        registry = ToolRegistry()

        @registry.register()
        def lookup_primary(q: str = "x") -> dict:
            """Primary (fails)."""
            raise RuntimeError("boom1")

        @registry.register()
        def lookup_backup(q: str = "x") -> dict:
            """Backup (also fails)."""
            raise RuntimeError("boom2")

        executor = self._executor(registry)
        dag, tool_step, finalize = _tool_dag(
            "lookup_primary", metadata={"fallback_tools": ["lookup_backup"]}
        )
        state = AgentState(objective=Objective("x"), budget=Budget(max_steps=12))
        state.steps = list(dag.steps.values())
        await executor._execute_dag(state, dag)
        # re-binds through the backup, then substitutes to reasoning — never dead-ends
        assert [r.action for r in state.repairs] == ["rebind", "substitute"]
        assert tool_step.type == "think" and finalize.status == "done"

    async def test_substitute_to_reasoning_when_no_alternative(self):
        registry = ToolRegistry()

        @registry.register()
        def only_tool(x: str = "x") -> dict:
            """Only tool (errors)."""
            raise RuntimeError("boom")

        executor = self._executor(registry)
        dag, tool_step, finalize = _tool_dag("only_tool")
        state = AgentState(objective=Objective("x"), budget=Budget(max_steps=12))
        state.steps = list(dag.steps.values())
        await executor._execute_dag(state, dag)
        assert state.repairs[0].action == "substitute"
        assert tool_step.type == "think" and tool_step.tool_name is None
        assert finalize.status == "done"

    async def test_contradiction_inserts_corrective_step(self):
        repairer = PlanRepairer()
        dag = StepDAG()
        validate = AgentStep(type="validate", name="validate")
        dag.add(validate)
        validate.status = "failed"
        finalize = AgentStep(type="finalize", name="finalize")
        dag.add(finalize, depends_on=[validate.id])
        state = AgentState(objective=Objective("x"))
        state.working_memory["validation"] = {"passed": False, "issues": ["unsupported claim"]}
        repair = repairer.repair_failure(state, dag, validate, tools=[])
        assert repair is not None and repair.action == "reorder"
        # the finalize now depends on the inserted revise step
        assert finalize.input_refs == repair.added_steps

    async def test_budget_shock_drops_optional_tail(self):
        repairer = PlanRepairer()
        dag = StepDAG()
        a = AgentStep(type="think", name="a")
        dag.add(a)
        a.status = "done"
        b = AgentStep(type="retrieve", name="b")
        dag.add(b, depends_on=[a.id])
        finalize = AgentStep(type="finalize", name="finalize")
        dag.add(finalize, depends_on=[b.id])
        state = AgentState(objective=Objective("x"), budget=Budget(max_cost_usd=1.0))
        state.usage.cost_usd = 0.9
        state.steps = list(dag.steps.values())
        repair = repairer.repair_budget_shock(state, dag, state.budget)
        assert repair is not None and repair.action == "drop"
        assert b.status == "skipped" and finalize.input_refs == [a.id]

    async def test_repairs_recorded_as_events_everywhere(self):
        registry = _failing_registry()
        events = EventBus()
        seen: list[dict] = []
        events.subscribe("plan.repaired", lambda e: seen.append(e.payload))
        executor = self._executor(registry, events=events)
        dag, tool_step, _ = _tool_dag(
            "billing_lookup_primary", metadata={"fallback_tools": ["billing_lookup_backup"]}
        )
        state = AgentState(objective=Objective("x"), budget=Budget(max_steps=12))
        state.steps = list(dag.steps.values())
        await executor._execute_dag(state, dag)
        # trajectory event
        assert state.repairs and state.metrics()["repairs"] == 1
        # event bus
        assert seen and seen[0]["action"] == "rebind"
        # trajectory projection surfaces the repair as a step
        from vincio.evals.trajectory import Trajectory

        traj = Trajectory.from_agent_state(state)
        assert any(s.type == "plan_repair" for s in traj.steps)

    async def test_repair_disabled_falls_back_to_skip(self):
        registry = ToolRegistry()

        @registry.register()
        def only_tool(x: str = "x") -> dict:
            """Errors."""
            raise RuntimeError("boom")

        executor = self._executor(registry, repair=False)
        dag, tool_step, finalize = _tool_dag("only_tool")
        state = AgentState(objective=Objective("x"), budget=Budget(max_steps=12))
        state.steps = list(dag.steps.values())
        await executor._execute_dag(state, dag)
        assert state.repairs == []
        assert finalize.status == "skipped"  # upstream-skip cascade

    def test_max_repairs_bounds_repairs(self):
        repairer = PlanRepairer(max_repairs=0)
        dag, tool_step, _ = _tool_dag("only_tool")
        tool_step.status = "failed"
        state = AgentState(objective=Objective("x"))
        state.steps = list(dag.steps.values())
        assert repairer.repair_failure(state, dag, tool_step, tools=[]) is None


# ---------------------------------------------------------------------------
# Cost-aware action selection
# ---------------------------------------------------------------------------


def _priced_registry() -> ModelRegistry:
    caps = ModelCapabilities(structured_output=True, tool_calling=True, reasoning=True)
    fast = ModelProfile(
        name="Fast", provider="mock", model="fast-x", tier="fast",
        capabilities=caps, input_cost_per_mtok=0.15, output_cost_per_mtok=0.60,
    )
    strong = ModelProfile(
        name="Strong", provider="mock", model="strong-x", tier="strong",
        capabilities=caps, input_cost_per_mtok=3.0, output_cost_per_mtok=15.0,
    )
    return ModelRegistry([fast, strong])


def _priced_table() -> PriceTable:
    table = PriceTable()
    table.set("fast-x", ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60))
    table.set("strong-x", ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0))
    return table


class TestCostAwareSelection:
    def test_picks_cheapest_capable(self):
        selector = CostAwareSelector(["fast-x", "strong-x"], registry=_priced_registry())
        decision = selector.select(needs=RequestNeeds(), input_tokens=500, output_tokens=256, confidence=0.9)
        assert decision.model == "fast-x" and not decision.escalated

    def test_escalates_on_low_confidence(self):
        selector = CostAwareSelector(["fast-x", "strong-x"], registry=_priced_registry())
        decision = selector.select(needs=RequestNeeds(), input_tokens=500, output_tokens=256, confidence=0.2)
        assert decision.model == "strong-x" and decision.escalated

    def test_downgrade_flag_when_over_budget(self):
        selector = CostAwareSelector(["fast-x", "strong-x"], registry=_priced_registry())
        decision = selector.select(
            needs=RequestNeeds(), input_tokens=500, output_tokens=256, remaining_budget_usd=1e-9
        )
        assert decision.downgraded

    def test_capability_filter_raises_when_none_capable(self):
        registry = ModelRegistry([
            ModelProfile(name="F", provider="mock", model="text-only", tier="fast",
                         capabilities=ModelCapabilities(structured_output=True))
        ])
        selector = CostAwareSelector(["text-only"], registry=registry)
        with pytest.raises(CapabilityMismatchError):
            selector.select(needs=RequestNeeds(vision=True), input_tokens=10, output_tokens=10)

    async def test_savings_versus_always_strong(self):
        registry, prices = _priced_registry(), _priced_table()

        def make(selector):
            return AgentExecutor(
                MockProvider(), model="strong-x", planner=Planner(mode="static"),
                cost_tracker=CostTracker(prices), selector=selector,
            )

        strong_run = await make(None).run("Summarize", budget=Budget(max_cost_usd=1.0))
        selector = CostAwareSelector(["fast-x", "strong-x"], registry=registry)
        cheap_run = await make(selector).run("Summarize", budget=Budget(max_cost_usd=1.0))
        assert cheap_run.usage.cost_usd < strong_run.usage.cost_usd
        assert cheap_run.working_memory.get("_selections")

    def test_empty_models_rejected(self):
        with pytest.raises(ValueError):
            CostAwareSelector([])


# ---------------------------------------------------------------------------
# Parallel sub-graph scheduling
# ---------------------------------------------------------------------------


def _two_step_graph(name: str) -> StateGraph:
    graph = StateGraph(name)
    graph.add_node("a", lambda s: {"n": s.get("n", 0) + 1})
    graph.add_node("b", lambda s: {"n": s["n"] * 10})
    graph.add_edge("a", "b")
    return graph


class TestSubgraphScheduling:
    async def test_concurrency_across_workers(self):
        tasks = [SubgraphTask(_two_step_graph(f"sg{i}"), {"n": i}) for i in range(4)]
        result = await SubgraphScheduler(workers=4).run(tasks)
        assert result.peak_concurrency == 4
        assert sorted(o.result.state["n"] for o in result.completed) == [10, 20, 30, 40]
        assert result.speedup > 1.0

    async def test_equivalent_to_serial(self):
        parallel = await SubgraphScheduler(workers=4).run(
            [SubgraphTask(_two_step_graph(f"p{i}"), {"n": i}) for i in range(4)]
        )
        serial = await SubgraphScheduler(workers=1).run(
            [SubgraphTask(_two_step_graph(f"s{i}"), {"n": i}) for i in range(4)]
        )
        assert serial.peak_concurrency == 1
        assert sorted(o.result.state["n"] for o in serial.completed) == sorted(
            o.result.state["n"] for o in parallel.completed
        )

    async def test_fair_share_sums_to_budget(self):
        tasks = [SubgraphTask(_two_step_graph(f"sg{i}"), {"n": i}) for i in range(4)]
        result = await SubgraphScheduler(workers=4, budget=Budget(max_cost_usd=1.0)).run(tasks)
        assert result.shares_usd == [0.25, 0.25, 0.25, 0.25]
        assert abs(sum(result.shares_usd) - 1.0) < 1e-9

    async def test_weighted_fair_share(self):
        tasks = [
            SubgraphTask(_two_step_graph("a"), {"n": 0}, weight=1.0),
            SubgraphTask(_two_step_graph("b"), {"n": 0}, weight=3.0),
        ]
        result = await SubgraphScheduler(workers=2, budget=Budget(max_cost_usd=1.0)).run(tasks)
        assert sorted(result.shares_usd) == [0.25, 0.75]

    async def test_deadline_returns_partial(self):
        class FakeClock:
            def __init__(self) -> None:
                self.t = 0

            def __call__(self) -> int:
                self.t += 1
                return self.t

        tasks = [SubgraphTask(_two_step_graph(f"sg{i}"), {"n": i}) for i in range(4)]
        result = await SubgraphScheduler(workers=1, deadline_s=2, clock=FakeClock()).run(tasks)
        assert result.deadline_hit
        assert len(result.completed) >= 1 and len(result.partial) >= 1
        assert all(o.status == "deadline" for o in result.partial)

    async def test_empty_tasks(self):
        result = await SubgraphScheduler(workers=2).run([])
        assert result.completed == [] and result.peak_concurrency == 0


# ---------------------------------------------------------------------------
# Durable timers & scheduled steps
# ---------------------------------------------------------------------------


_BASE = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


def _sleep_graph(store, seconds: float = 3600) -> StateGraph:
    graph = StateGraph("timed")
    graph.add_node("start", lambda s: {"stage": "started"})

    def wait(s):
        sleep_for(s, seconds, clock=lambda: _BASE)
        return {"stage": "woke"}

    graph.add_node("wait", wait)
    graph.add_node("done", lambda s: {"stage": "done"})
    graph.add_edge("start", "wait")
    graph.add_edge("wait", "done")
    return graph


class TestDurableTimers:
    def test_sleep_pauses_until_due(self):
        store = InMemoryMetadataStore()
        compiled = _sleep_graph(store).compile(checkpointer=Checkpointer(store))
        paused = compiled.invoke({}, thread_id="t1")
        assert paused.status == "interrupted" and paused.state.get("stage") == "started"
        # not due 30 minutes in
        early = TimerService(compiled, clock=lambda: _BASE + timedelta(minutes=30))
        assert early.due() == [] and early.tick() == []

    def test_sleep_resumes_after_restart(self):
        store = InMemoryMetadataStore()
        first = _sleep_graph(store).compile(checkpointer=Checkpointer(store))
        first.invoke({}, thread_id="t1")
        # Simulate a process restart: a fresh compiled graph + checkpointer, same store.
        restarted = _sleep_graph(store).compile(checkpointer=Checkpointer(store))
        results = TimerService(restarted, clock=lambda: _BASE + timedelta(hours=2)).tick()
        assert len(results) == 1
        assert results[0].status == "done" and results[0].state["stage"] == "done"

    def test_sleep_until_absolute_time(self):
        store = InMemoryMetadataStore()
        graph = StateGraph("g")
        graph.add_node("a", lambda s: {"x": 1})

        def wait(s):
            sleep_until(s, _BASE + timedelta(hours=1))
            return {"woke": True}

        graph.add_node("w", wait)
        graph.add_edge("a", "w")
        compiled = graph.compile(checkpointer=Checkpointer(store))
        compiled.invoke({}, thread_id="t")
        assert due_timers(compiled, now=_BASE + timedelta(minutes=10)) == []
        results = TimerService(compiled).tick(now=_BASE + timedelta(hours=2))
        assert results[0].state.get("woke") is True

    def test_wait_for_event(self):
        store = InMemoryMetadataStore()
        graph = StateGraph("evt")
        graph.add_node("a", lambda s: {"x": 1})

        def wait(s):
            return {"approval": wait_for_event(s, "approved")}

        graph.add_node("w", wait)
        graph.add_edge("a", "w")
        compiled = graph.compile(checkpointer=Checkpointer(store))
        paused = compiled.invoke({}, thread_id="t2")
        assert paused.status == "interrupted"
        # a non-matching event does not wake the thread
        assert deliver_event(compiled, "t2", "rejected") is None
        done = deliver_event(compiled, "t2", "approved", payload={"by": "alice"})
        assert done.status == "done" and done.state["approval"] == {"by": "alice"}

    def test_pending_timers_scan(self):
        store = InMemoryMetadataStore()
        compiled = _sleep_graph(store).compile(checkpointer=Checkpointer(store))
        compiled.invoke({}, thread_id="t1")
        pending = pending_timers(compiled)
        assert len(pending) == 1 and pending[0].timer.kind == "sleep_until"

    def test_no_timers_no_resumes(self):
        store = InMemoryMetadataStore()
        graph = StateGraph("plain")
        graph.add_node("a", lambda s: {"x": 1})
        compiled = graph.compile(checkpointer=Checkpointer(store))
        compiled.invoke({}, thread_id="t")
        assert TimerService(compiled).tick(now=_BASE) == []
