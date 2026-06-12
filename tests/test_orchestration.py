"""0.6 milestone tests: multi-agent crews over a shared blackboard, durable
stateful graphs (checkpoint/resume/time-travel), human-in-the-loop interrupts
on graphs and workflows, declarative composition with streaming node events,
and runtime backend adapters."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from vincio.agents import (
    END,
    AgentExecutor,
    AgentRole,
    Blackboard,
    Checkpointer,
    Crew,
    LangGraphBackend,
    OpenAIAgentsBackend,
    Planner,
    StateGraph,
    branch,
    compose,
    interrupt,
    parallel,
)
from vincio.core.errors import AgentEngineError, ConfigError, GraphError
from vincio.core.events import EventBus
from vincio.core.types import Budget
from vincio.observability.exporters import InMemoryExporter
from vincio.observability.traces import Tracer
from vincio.providers import MockProvider
from vincio.storage.sqlite import SQLiteMetadataStore
from vincio.workflows import Workflow


def direct_agent(text: str) -> AgentExecutor:
    return AgentExecutor(MockProvider(default_text=text), model="mock-1", planner=Planner(mode="direct"))


class TestBlackboard:
    def test_versioned_posts_and_history(self):
        board = Blackboard()
        board.post("finding", "v1", author="researcher")
        entry = board.post("finding", "v2", author="writer")
        assert entry.version == 2
        assert board.get("finding") == "v2"
        assert [e.value for e in board.history("finding")] == ["v1", "v2"]
        assert board.entry("finding").author == "writer"

    def test_snapshot_restore_roundtrip(self):
        board = Blackboard()
        board.post("a", {"x": 1}, author="agent_a")
        board.post("a", {"x": 2}, author="agent_a")
        snapshot = json.loads(json.dumps(board.snapshot()))  # must be JSON-serializable
        restored = Blackboard.restore(snapshot)
        assert restored.get("a") == {"x": 2}
        assert len(restored.history("a")) == 2

    def test_as_context_renders_authors(self):
        board = Blackboard()
        board.post("summary", "short text", author="analyst")
        rendered = board.as_context()
        assert "summary" in rendered and "analyst" in rendered

    def test_posts_emit_events(self):
        bus = EventBus()
        seen = []
        bus.subscribe("blackboard.posted", lambda e: seen.append(e.payload["key"]))
        board = Blackboard(event_bus=bus)
        board.post("k", 1, author="a")
        assert seen == ["k"]


class TestCrew:
    @pytest.mark.asyncio
    async def test_sequential_shares_blackboard(self):
        crew = Crew("team")
        crew.add("researcher", direct_agent("research notes"), goal="gather evidence")
        crew.add("writer", direct_agent("final report"))
        result = await crew.arun("write a report")
        assert result.status == "succeeded"
        assert result.output == "final report"
        assert [r.role for r in result.reports] == ["researcher", "writer"]
        assert list(result.blackboard["entries"]) == ["researcher", "writer"]
        # the writer's objective included the researcher's post
        assert result.reports[0].metrics["success"]

    @pytest.mark.asyncio
    async def test_tasks_by_name_and_validation(self):
        crew = Crew("team")
        crew.add("a", direct_agent("A"))
        crew.add("b", direct_agent("B"))
        result = await crew.arun("obj", tasks={"b": "do b things"})
        assert [r.role for r in result.reports] == ["b"]
        with pytest.raises(AgentEngineError):
            await crew.arun("obj", tasks={"missing": "x"})
        with pytest.raises(AgentEngineError):
            await crew.arun("obj", tasks=["only one task for two members", "x", "y"])

    @pytest.mark.asyncio
    async def test_parallel_returns_dict(self):
        crew = Crew("team", process="parallel")
        crew.add("a", direct_agent("answer a"))
        crew.add("b", direct_agent("answer b"))
        result = await crew.arun("obj")
        assert result.output == {"a": "answer a", "b": "answer b"}
        assert result.status == "succeeded"

    @pytest.mark.asyncio
    async def test_budget_guarantees_termination(self):
        crew = Crew("team")
        for name in ("a", "b", "c"):
            crew.add(name, direct_agent(name))
        result = await crew.arun("obj", budget=Budget(max_steps=1))
        assert result.status == "budget_exhausted"
        assert len(result.reports) < 3
        assert result.usage.steps >= 1

    @pytest.mark.asyncio
    async def test_member_budget_fraction(self):
        crew = Crew("team")
        crew.add(AgentRole(name="a", budget_fraction=0.25), direct_agent("a"))
        member_budget = crew._member_budget(crew._members["a"].role, Budget(max_input_tokens=1000))
        assert member_budget.max_input_tokens == 250

    @pytest.mark.asyncio
    async def test_hierarchical_delegation_with_manager(self):
        manager = MockProvider(
            script=[
                json.dumps({"assignments": [{"agent": "analyst", "task": "dig in", "reason": "fits"}]}),
                json.dumps({"done": True, "final_answer": "all set", "follow_ups": []}),
            ]
        )
        crew = Crew("h", process="hierarchical", manager_provider=manager, manager_model="m")
        crew.add("analyst", direct_agent("analysis"), keywords=["analyze"])
        crew.add("writer", direct_agent("draft"))
        result = await crew.arun("analyze the contract")
        assert result.status == "succeeded"
        assert result.output == "all set"
        assert [d.to_agent for d in result.delegations] == ["analyst"]
        assert result.rounds == 1

    @pytest.mark.asyncio
    async def test_hierarchical_offline_fallback_selects_by_keywords(self):
        crew = Crew("h", process="hierarchical")  # no manager provider
        crew.add("billing", direct_agent("billing answer"), keywords=["invoice", "refund"])
        crew.add("legal", direct_agent("legal answer"), keywords=["contract", "clause"])
        result = await crew.arun("Is this invoice refundable?")
        assert result.status == "succeeded"
        assert result.delegations[0].to_agent == "billing"
        assert result.output == "billing answer"

    @pytest.mark.asyncio
    async def test_hierarchical_max_rounds_bound(self):
        follow_up = json.dumps(
            {"done": False, "final_answer": "", "follow_ups": [{"agent": "a", "task": "again", "reason": "more"}]}
        )
        manager = MockProvider(
            responder=lambda req: follow_up
            if req.output_schema and "done" in req.output_schema.get("properties", {})
            else json.dumps({"assignments": [{"agent": "a", "task": "go", "reason": "r"}]})
        )
        crew = Crew("h", process="hierarchical", manager_provider=manager, manager_model="m", max_rounds=2)
        crew.add("a", direct_agent("a"))
        result = await crew.arun("obj")
        assert result.status == "max_rounds"
        assert result.rounds == 2
        assert len(result.delegations) == 2

    @pytest.mark.asyncio
    async def test_member_runs_emit_crew_spans(self):
        exporter = InMemoryExporter()
        tracer = Tracer("t", exporter)
        crew = Crew("team", tracer=tracer)
        crew.add("solo", direct_agent("x"))
        with tracer.trace():
            await crew.arun("obj")
        spans = exporter.traces[0].spans
        assert any(s.type == "crew" for s in spans)
        assert any(s.type == "crew_agent" and s.name == "solo" for s in spans)

    def test_empty_and_duplicate_members_rejected(self):
        crew = Crew("team")
        with pytest.raises(AgentEngineError):
            crew.run("obj")
        crew.add("a", direct_agent("a"))
        with pytest.raises(AgentEngineError):
            crew.add("a", direct_agent("a"))


class TestStateGraph:
    def build_linear(self) -> StateGraph:
        graph = StateGraph("flow")
        graph.add_node("a", lambda s: {"x": s.get("x", 0) + 1})
        graph.add_node("b", lambda s: {"x": s["x"] * 10})
        graph.add_edge("a", "b")
        return graph

    @pytest.mark.asyncio
    async def test_linear_run_and_checkpoints(self):
        compiled = self.build_linear().compile()
        result = await compiled.ainvoke({"x": 1})
        assert result.status == "done"
        assert result.state == {"x": 20}
        history = compiled.history(result.thread_id)
        assert history[-1].status == "done"
        assert [c.step for c in history] == sorted(c.step for c in history)

    @pytest.mark.asyncio
    async def test_conditional_routing_with_mapping(self):
        graph = StateGraph("route")
        graph.add_node("classify", lambda s: {"kind": s["kind"]})
        graph.add_node("hot", lambda s: {"out": "escalated"})
        graph.add_node("cold", lambda s: {"out": "archived"})
        graph.add_conditional_edge("classify", lambda s: s["kind"], {"urgent": "hot", "other": "cold"})
        result = await graph.compile().ainvoke({"kind": "urgent"})
        assert result.state["out"] == "escalated"
        result = await graph.compile().ainvoke({"kind": "other"})
        assert result.state["out"] == "archived"

    @pytest.mark.asyncio
    async def test_reducers_merge_parallel_branches(self):
        import operator

        graph = StateGraph("fan", reducers={"items": operator.add})
        graph.add_node("seed", lambda s: {"items": ["seed"]})
        graph.add_node("left", lambda s: {"items": ["left"]})
        graph.add_node("right", lambda s: {"items": ["right"]})
        graph.add_edge("seed", "left")
        graph.add_edge("seed", "right")
        result = await graph.compile().ainvoke({})
        assert result.state["items"] == ["seed", "left", "right"]

    @pytest.mark.asyncio
    async def test_max_steps_bounds_cycles(self):
        graph = StateGraph("loop")
        graph.add_node("a", lambda s: {"n": s.get("n", 0) + 1})
        graph.add_node("b", lambda s: {})
        graph.add_edge("a", "b")
        graph.add_edge("b", "a")
        result = await graph.compile(max_steps=5).ainvoke({})
        assert result.status == "max_steps"
        assert result.steps == 5

    @pytest.mark.asyncio
    async def test_interrupt_before_pause_and_resume(self):
        graph = self.build_linear()
        compiled = graph.compile(interrupt_before=["b"])
        paused = await compiled.ainvoke({"x": 1})
        assert paused.status == "interrupted"
        assert paused.next_nodes == ["b"]
        done = await compiled.aresume(paused.thread_id)
        assert done.status == "done"
        assert done.state["x"] == 20

    @pytest.mark.asyncio
    async def test_interrupt_after(self):
        compiled = self.build_linear().compile(interrupt_after=["a"])
        paused = await compiled.ainvoke({"x": 1})
        assert paused.status == "interrupted"
        assert paused.state["x"] == 2
        done = await compiled.aresume(paused.thread_id)
        assert done.state["x"] == 20

    @pytest.mark.asyncio
    async def test_dynamic_interrupt_receives_resume_value(self):
        graph = StateGraph("hitl")
        graph.add_node("draft", lambda s: {"draft": "v1"})
        graph.add_node("gate", lambda s: {"approved": interrupt(s, {"question": "ship it?"})})
        graph.add_node("ship", lambda s: {"shipped": s["approved"]})
        graph.add_edge("draft", "gate")
        graph.add_edge("gate", "ship")
        compiled = graph.compile()
        paused = await compiled.ainvoke({})
        assert paused.status == "interrupted"
        assert paused.interrupt_payload == {"question": "ship it?"}
        done = await compiled.aresume(paused.thread_id, value=True)
        assert done.status == "done"
        assert done.state["shipped"] is True

    @pytest.mark.asyncio
    async def test_update_state_edit_and_resume(self):
        graph = StateGraph("edit")
        graph.add_node("draft", lambda s: {"draft": "v1"})
        graph.add_node("publish", lambda s: {"published": s["draft"]})
        graph.add_edge("draft", "publish")
        compiled = graph.compile(interrupt_before=["publish"])
        paused = await compiled.ainvoke({})
        compiled.update_state(paused.thread_id, {"draft": "v2-edited"})
        done = await compiled.aresume(paused.thread_id)
        assert done.state["published"] == "v2-edited"

    @pytest.mark.asyncio
    async def test_fork_time_travel_is_deterministic(self):
        compiled = self.build_linear().compile()
        original = await compiled.ainvoke({"x": 1})
        # fork from the checkpoint taken after node "a" and re-execute
        after_a = compiled.history(original.thread_id)[1]
        forked_thread = compiled.fork(after_a.id)
        replayed = await compiled.aresume(forked_thread)
        assert replayed.status == "done"
        assert replayed.state == original.state

    @pytest.mark.asyncio
    async def test_durable_resume_across_instances(self, tmp_path):
        store = SQLiteMetadataStore(tmp_path / "graph.db")
        graph = self.build_linear()
        first = graph.compile(checkpointer=Checkpointer(store), interrupt_before=["b"])
        paused = await first.ainvoke({"x": 1})
        # a fresh process: same graph definition, same store, new compile
        second = self.build_linear().compile(checkpointer=Checkpointer(store), interrupt_before=["b"])
        done = await second.aresume(paused.thread_id)
        assert done.status == "done"
        assert done.state["x"] == 20

    @pytest.mark.asyncio
    async def test_astream_yields_node_and_terminal_events(self):
        compiled = self.build_linear().compile()
        events = [event async for event in compiled.astream({"x": 1})]
        kinds = [e.type for e in events]
        assert kinds.count("node_start") == 2 and kinds.count("node_end") == 2
        assert kinds[-1] == "done"
        assert any(e.type == "checkpoint" for e in events)

    @pytest.mark.asyncio
    async def test_state_schema_validation(self):
        from pydantic import BaseModel

        class State(BaseModel):
            x: int

        graph = StateGraph("typed", state_schema=State)
        graph.add_node("bad", lambda s: {"x": "not an int"})
        with pytest.raises(GraphError):
            await graph.compile().ainvoke({"x": 1})

    @pytest.mark.asyncio
    async def test_nodes_emit_graph_node_spans(self):
        exporter = InMemoryExporter()
        tracer = Tracer("t", exporter)
        compiled = self.build_linear().compile(tracer=tracer)
        with tracer.trace():
            await compiled.ainvoke({"x": 1})
        spans = exporter.traces[0].spans
        assert [s.name for s in spans if s.type == "graph_node"] == ["a", "b"]

    def test_definition_errors(self):
        graph = StateGraph("bad")
        with pytest.raises(GraphError):
            graph.compile()  # no nodes
        graph.add_node("a", lambda s: {})
        with pytest.raises(GraphError):
            graph.add_node("a", lambda s: {})
        with pytest.raises(GraphError):
            graph.add_edge("a", "missing")
        with pytest.raises(GraphError):
            graph.add_node(END, lambda s: {})

    @pytest.mark.asyncio
    async def test_router_unknown_target_raises(self):
        graph = StateGraph("bad_route")
        graph.add_node("a", lambda s: {})
        graph.add_conditional_edge("a", lambda s: "nowhere")
        with pytest.raises(GraphError):
            await graph.compile().ainvoke({})

    @pytest.mark.asyncio
    async def test_resume_requires_checkpoints(self):
        compiled = self.build_linear().compile()
        with pytest.raises(GraphError):
            await compiled.aresume("thread_missing")
        done = await compiled.ainvoke({"x": 1})
        with pytest.raises(GraphError):
            await compiled.aresume(done.thread_id)  # already completed


class TestCompose:
    @pytest.mark.asyncio
    async def test_pipe_order_and_call(self):
        pipeline = compose(lambda v: v + 1) | (lambda v: v * 2)
        assert await pipeline.acall(3) == 8
        assert pipeline.call(3) == 8
        assert pipeline.nodes == ["<lambda>", "<lambda>"]

    @pytest.mark.asyncio
    async def test_ror_wraps_left_operand(self):
        def double(v):
            return v * 2

        pipeline = double | compose(lambda v: v + 1)
        assert await pipeline.acall(3) == 7

    @pytest.mark.asyncio
    async def test_astream_events_per_node(self):
        pipeline = compose(lambda v: v + 1, lambda v: v * 2, name="math")
        events = [e async for e in pipeline.astream(3)]
        assert [e.type for e in events] == ["node_start", "node_end", "node_start", "node_end", "done"]
        assert events[-1].value == 8

    @pytest.mark.asyncio
    async def test_error_event_and_raise(self):
        def boom(v):
            raise ValueError("nope")

        pipeline = compose(boom)
        events = [e async for e in pipeline.astream(1)]
        assert events[-1].type == "error" and "nope" in events[-1].error
        with pytest.raises(AgentEngineError):
            await pipeline.acall(1)

    @pytest.mark.asyncio
    async def test_parallel_and_branch(self):
        fan = parallel(double=lambda v: v * 2, triple=lambda v: v * 3)
        assert await fan.acall(2) == {"double": 4, "triple": 6}
        router = branch(lambda v: "big" if v > 10 else "small", {"big": lambda v: "B", "small": lambda v: "S"})
        assert await router.acall(20) == "B"
        assert await router.acall(2) == "S"
        strict = branch(lambda v: "missing", {"a": lambda v: v})
        with pytest.raises(AgentEngineError):
            await strict.acall(1)
        defaulted = branch(lambda v: "missing", {"a": lambda v: "fallback"}, default="a")
        assert await defaulted.acall(1) == "fallback"

    @pytest.mark.asyncio
    async def test_composes_engines_and_normalizes_results(self):
        agent = direct_agent("agent says hi")
        workflow = Workflow("wf").step("only", lambda input: f"wf({input})")
        pipeline = compose(agent) | workflow
        assert await pipeline.acall("question") == "wf(agent says hi)"

    @pytest.mark.asyncio
    async def test_composes_graphs(self):
        graph = StateGraph("g")
        graph.add_node("n", lambda s: {"answer": s["input"] * 2})
        pipeline = compose(graph) | (lambda state: state["answer"])
        assert await pipeline.acall(21) == 42

    @pytest.mark.asyncio
    async def test_node_spans_emitted(self):
        exporter = InMemoryExporter()
        tracer = Tracer("t", exporter)
        pipeline = compose(lambda v: v, name="p", tracer=tracer)
        with tracer.trace():
            await pipeline.acall(1)
        assert any(s.type == "compose_node" for s in exporter.traces[0].spans)


class _FakeLangGraphBuilder:
    def __init__(self, schema):
        self.schema = schema
        self.nodes: list[str] = []
        self.edges: list[tuple[str, str]] = []
        self.conditional: list[str] = []
        self.entry: str | None = None

    def add_node(self, name, fn):
        self.nodes.append(name)

    def add_edge(self, source, target):
        self.edges.append((source, target))

    def add_conditional_edges(self, source, router):
        self.conditional.append(source)
        self.router = router

    def set_entry_point(self, name):
        self.entry = name

    def compile(self, **kwargs):
        async def ainvoke(input, **kw):
            return {"ran": True, "input": input}

        return SimpleNamespace(ainvoke=ainvoke)


class TestBackends:
    def fake_langgraph(self):
        return SimpleNamespace(StateGraph=_FakeLangGraphBuilder, END="__lg_end__", START="__lg_start__")

    def vincio_graph(self) -> StateGraph:
        graph = StateGraph("x")
        graph.add_node("n1", lambda s: {})
        graph.add_node("n2", lambda s: {})
        graph.add_edge("n1", "n2")
        graph.add_conditional_edge("n2", lambda s: END)
        return graph

    @pytest.mark.asyncio
    async def test_langgraph_export_and_run(self):
        backend = LangGraphBackend(self.fake_langgraph())
        builder = backend.export(self.vincio_graph())
        assert builder.nodes == ["n1", "n2"]
        assert builder.entry == "n1"
        assert builder.edges == [("n1", "n2")]
        assert builder.conditional == ["n2"]
        # Vincio END translates to the runtime's END sentinel
        assert builder.router({}) == "__lg_end__"
        result = await backend.run(self.vincio_graph(), {"q": 1})
        assert result == {"ran": True, "input": {"q": 1}}

    @pytest.mark.asyncio
    async def test_openai_agents_export_and_run(self):
        class FakeAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        async def fake_run(agent, input, **kwargs):
            return SimpleNamespace(final_output=f"ran:{agent.kwargs['name']}")

        module = SimpleNamespace(
            Agent=FakeAgent,
            function_tool=lambda fn: ("tool", fn),
            Runner=SimpleNamespace(run=fake_run),
        )
        backend = OpenAIAgentsBackend(module)

        def search(q: str) -> str:
            return q

        exported = backend.export_agent("solo", instructions="do it", tools=[search], model="gpt-x")
        assert exported.kwargs["tools"] == [("tool", search)]
        assert exported.kwargs["model"] == "gpt-x"

        crew = Crew("squad")
        crew.add("researcher", direct_agent("r"), description="finds facts")
        crew.add("writer", direct_agent("w"), goal="write well")
        manager = backend.export_crew(crew)
        assert manager.kwargs["name"] == "squad_manager"
        assert [a.kwargs["name"] for a in manager.kwargs["handoffs"]] == ["researcher", "writer"]
        assert "finds facts" in manager.kwargs["handoffs"][0].kwargs["instructions"]
        assert await backend.run(crew, "go") == "ran:squad_manager"

    def test_missing_runtime_raises_config_error(self):
        from vincio.agents.backends import _import

        with pytest.raises(ConfigError):
            _import("definitely_not_installed_runtime_xyz", hint="nothing")


class TestWorkflowInterrupts:
    @pytest.mark.asyncio
    async def test_pause_and_resume_with_approvals(self):
        runs = {"build": 0}

        def build(input):
            runs["build"] += 1
            return "built"

        workflow = Workflow("deploy")
        workflow.step("build", build)
        workflow.step("ship", lambda build: f"{build}+shipped", depends_on=["build"], approval=True)
        paused = await workflow.arun(1)
        assert paused.status == "paused"
        assert paused.pending_approvals == ["ship"]
        resumed = await workflow.aresume(paused, approvals={"ship": True})
        assert resumed.status == "succeeded"
        assert resumed.context.output_of("ship") == "built+shipped"
        assert runs["build"] == 1  # done steps are not re-executed on resume

    @pytest.mark.asyncio
    async def test_resume_with_denial_fails_step(self):
        workflow = Workflow("deploy")
        workflow.step("build", lambda input: "built")
        workflow.step("ship", lambda build: build, depends_on=["build"], approval=True)
        paused = await workflow.arun(1)
        denied = await workflow.aresume(paused, approvals={"ship": False})
        assert denied.status == "partial"
        assert denied.failed_steps == ["ship"]

    @pytest.mark.asyncio
    async def test_edit_and_resume_steers_downstream(self):
        workflow = Workflow("review")
        workflow.step("draft", lambda input: "draft-v1")
        workflow.step("publish", lambda draft: f"published:{draft}", depends_on=["draft"], approval=True)
        paused = await workflow.arun(None)
        paused.context.results["draft"].output = "draft-v2-edited"
        resumed = await workflow.aresume(paused, approvals={"publish": True})
        assert resumed.context.output_of("publish") == "published:draft-v2-edited"

    @pytest.mark.asyncio
    async def test_approval_fn_still_consulted_when_configured(self):
        async def deny(step, context):
            return False

        workflow = Workflow("appr", approval_fn=deny)
        workflow.step("dangerous", lambda input: "done", approval=True)
        result = await workflow.arun(1)
        assert result.status == "failed"
        assert "approval denied" in result.context.results["dangerous"].error


class TestAppIntegration:
    def make_app(self):
        from vincio import ContextApp

        return ContextApp("orch_test", provider=MockProvider(default_text="answer"), model="mock-1")

    @pytest.mark.asyncio
    async def test_app_crew_runs_and_snapshots_blackboard(self):
        app = self.make_app()
        crew = app.crew(members=[{"name": "analyst", "goal": "analyze"}, {"name": "writer"}])
        result = await crew.arun("objective")
        assert result.status == "succeeded"
        assert list(result.blackboard["entries"]) == ["analyst", "writer"]
        assert result.metrics()["members_succeeded"] == 2

    @pytest.mark.asyncio
    async def test_app_graph_checkpoints_in_app_store(self):
        app = self.make_app()
        graph = app.graph("flow")
        graph.add_node("a", lambda s: {"v": 1})
        result = await graph.compile().ainvoke({})
        assert result.status == "done"
        records = app.store.query("graph_checkpoints", where={"thread_id": result.thread_id})
        assert records, "checkpoints should persist in the app metadata store"

    @pytest.mark.asyncio
    async def test_compose_with_app_agent_handle(self):
        app = self.make_app()
        pipeline = compose(app.agent(planner="direct")) | (lambda answer: f"wrapped:{answer}")
        out = await pipeline.acall("hi")
        assert out.startswith("wrapped:")

    @pytest.mark.asyncio
    async def test_parallel_crew_members_run_concurrently(self):
        # two slow members in a parallel crew should overlap
        class SlowProvider(MockProvider):
            async def generate(self, request):
                await asyncio.sleep(0.05)
                return await super().generate(request)

        crew = Crew("par", process="parallel")
        for name in ("a", "b"):
            crew.add(name, AgentExecutor(SlowProvider(default_text=name), model="m", planner=Planner(mode="direct")))
        import time

        started = time.monotonic()
        await crew.arun("obj")
        assert time.monotonic() - started < 0.095  # < 2 * 0.05 means concurrent
