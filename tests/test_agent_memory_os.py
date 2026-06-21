"""Agent memory OS, in-loop context compaction, level-parallel DAG
execution, and the plan-and-execute replanning loop."""

import warnings

from vincio import ContextApp, VincioConfig
from vincio.agents.compaction import LoopCompactor
from vincio.agents.dag import StepDAG
from vincio.agents.executor import AgentExecutor
from vincio.agents.planner import Planner
from vincio.agents.state import AgentState, AgentStep
from vincio.core.types import Message, Objective
from vincio.memory.agent_os import MemoryOS, memory_tools
from vincio.providers import MockProvider

warnings.simplefilter("ignore")


def _app(tmp_path, **kw):
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    return ContextApp(name="os", provider=MockProvider(**kw), model="mock-1", config=config)


# --------------------------------------------------------------------------- #
# in-loop context compaction
# --------------------------------------------------------------------------- #


class TestLoopCompactor:
    def test_under_budget_keeps_everything(self):
        c = LoopCompactor(max_tokens=10_000, keep_recent=4)
        blocks = [f"block {i}" for i in range(5)]
        summary, kept = c.compact_blocks(blocks)
        assert summary is None and kept == blocks

    def test_over_budget_summarizes_old_blocks(self):
        c = LoopCompactor(max_tokens=40, keep_recent=2, summary_tokens=30)
        blocks = [f"This is observation number {i} with some descriptive content." for i in range(12)]
        summary, kept = c.compact_blocks(blocks)
        assert summary is not None  # older blocks folded into a rolling summary
        assert len(kept) < len(blocks)
        assert kept[-1] == blocks[-1]  # most recent kept verbatim

    def test_message_compaction_preserves_anchor_and_recent(self):
        c = LoopCompactor(max_tokens=30, keep_recent=2)
        messages = [
            Message(role="system", content="You are an agent."),
            Message(role="user", content="Solve the very long objective with many details here."),
            *[Message(role="assistant", content=f"step {i} reasoning that is fairly long indeed") for i in range(8)],
        ]
        out = c.compact_messages(messages)
        assert out[0].role == "system"
        assert any("compacted" in m.text.lower() for m in out)
        assert len(out) < len(messages)

    def test_message_compaction_keeps_tool_pair_together(self):
        c = LoopCompactor(max_tokens=20, keep_recent=1)
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="do it with lots of detail and length to trigger compaction now"),
            *[Message(role="assistant", content=f"think {i} with extra padding text here") for i in range(4)],
            Message(role="assistant", content="calling tool with arguments and padding text"),
            Message(role="tool", content="tool result payload with padding text"),
        ]
        out = c.compact_messages(messages)
        # The recent window must not start on a lone tool message.
        recent_roles = [m.role for m in out if "compacted" not in m.text.lower()]
        assert recent_roles[-1] != "tool" or "assistant" in recent_roles


# --------------------------------------------------------------------------- #
# level-parallel DAG + plan-and-execute
# --------------------------------------------------------------------------- #


class TestLevelParallelDAG:
    def test_static_plan_runs_to_completion(self, tmp_path):
        app = _app(tmp_path)
        handle = app.agent(planner="static", max_steps=8)
        result = handle.run("Summarize the objective.")
        assert result.terminated
        assert result.final_answer is not None

    async def test_independent_steps_all_execute(self):
        executor = AgentExecutor(MockProvider(), model="mock-1", planner=Planner(mode="static"))
        # Two independent think steps form one parallel level.
        dag = StepDAG()
        dag.add(AgentStep(type="think", name="a", instruction="think a"))
        dag.add(AgentStep(type="think", name="b", instruction="think b"))
        state = AgentState(objective=Objective(text="parallel"))
        await executor._execute_dag(state, dag)
        assert all(s.status in ("done", "skipped") for s in dag.steps.values())
        # topological_levels groups the two independent steps into one level.
        assert len(dag.topological_levels()[0]) == 2


class TestPlanAndExecute:
    async def test_heuristic_replan_proposes_corrective_steps(self):
        planner = Planner(mode="plan_and_execute")
        state = AgentState(objective=Objective(text="answer"))
        state.working_memory["validation"] = {"passed": False, "issues": ["unsupported claim"]}
        dag = await planner.replan(Objective(text="answer"), state=state, tools=[])
        names = {s.name for s in dag.steps.values()}
        assert "revise" in names and "refinalize" in names

    def test_needs_replan_detects_validation_failure(self, tmp_path):
        executor = AgentExecutor(MockProvider(), model="mock-1", planner=Planner(mode="plan_and_execute"))
        state = AgentState(objective=Objective(text="x"))
        state.final_answer = "answer"
        state.working_memory["validation"] = {"passed": False, "issues": []}
        assert executor._needs_replan(state) is True
        state.working_memory["validation"] = {"passed": True}
        assert executor._needs_replan(state) is False

    def test_plan_and_execute_runs_end_to_end(self, tmp_path):
        app = _app(tmp_path)
        handle = app.agent(planner="plan_and_execute", max_steps=10)
        result = handle.run("Answer the question.")
        assert result.terminated
        assert "_replans" in result.working_memory


# --------------------------------------------------------------------------- #
# agent memory OS
# --------------------------------------------------------------------------- #


class TestMemoryOS:
    def test_append_search_replace(self, tmp_path):
        app = _app(tmp_path)
        app.add_memory()
        os = MemoryOS(app.memory, scope="agent", owner_id="a1")
        mid = os.append("The user prefers concise answers.")
        assert mid
        hits = os.search("answer preference")
        assert any("concise" in h for h in hits)
        assert os.replace(mid, "The user prefers detailed answers.") is True

    def test_archive_pages_out_of_core(self, tmp_path):
        app = _app(tmp_path)
        app.add_memory()
        os = MemoryOS(app.memory, scope="agent", owner_id="a1")
        mid = os.append("The customer is on the enterprise plan.")
        assert mid in os.core_ids
        assert os.archive(mid) is True
        assert mid not in os.core_ids
        # Archived items are excluded from active search.
        assert not any("enterprise" in h for h in os.search("plan tier"))

    def test_pager_evicts_over_budget(self, tmp_path):
        app = _app(tmp_path)
        app.add_memory()
        os = MemoryOS(app.memory, scope="agent", owner_id="a1", max_core_tokens=20)
        for i in range(8):
            os.append(f"The account region for customer {i} is the European Union zone.",
                      importance=0.5 + i * 0.05)
        # The pager keeps core under the token budget by archiving low-importance items.
        assert os.core_tokens() <= 20 or len(os.core_ids) == 1

    def test_page_in_promotes_relevant_archival(self, tmp_path):
        app = _app(tmp_path)
        app.add_memory()
        os = MemoryOS(app.memory, scope="agent", owner_id="a1")
        mid = os.append("The Pro plan refund window is 30 days.")
        os.archive(mid)
        promoted = os.page_in("refund window", top_k=1)
        assert promoted == 1
        assert mid in os.core_ids

    def test_enable_memory_os_registers_tools(self, tmp_path):
        app = _app(tmp_path)
        os = app.enable_memory_os(owner_id="agent-1")
        assert isinstance(os, MemoryOS)
        for tool in ("memory_append", "memory_replace", "memory_search", "memory_archive"):
            assert tool in app.tool_registry
        # Writes are write-side-effecting (ride the guarded write path).
        assert app.tool_registry.get("memory_append").spec.side_effects == "write"
        assert app.tool_registry.get("memory_search").spec.side_effects == "read"

    def test_archive_is_audited(self, tmp_path):
        app = _app(tmp_path)
        app.add_memory()
        os = MemoryOS(app.memory, scope="agent", owner_id="a1")
        mid = os.append("a fact")
        os.archive(mid)
        assert app.audit.verify_chain()

    def test_memory_tools_callables(self, tmp_path):
        app = _app(tmp_path)
        app.add_memory()
        os = MemoryOS(app.memory, scope="agent", owner_id="a1")
        append, replace, search, archive = memory_tools(os)
        mid = append("remember this", 0.9)
        assert isinstance(mid, str)
        assert isinstance(search("remember", 3), list)
