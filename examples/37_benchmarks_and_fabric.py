"""Prove it on the world's benchmarks + the agent fabric (2.2).

The additive 2.2 surface on the frozen 2.0 API — measurability, composability,
and embeddability, all offline on the deterministic mock:

  1. Stateful-environment eval: an Environment (reset/step/observe/verify) drives
     an agent policy through a *mutable* world; a task-success ORACLE verifies the
     end state — not turn-by-turn plausibility — and projects onto a Trajectory.
  2. Agentic benchmark adapters: SWE-bench Verified, τ-bench, GAIA, WebArena, and
     BFCL behind one BenchmarkAdapter contract, pinned by a task-set hash and
     replayed offline against the benchmark's own verifiable scorer.
  3. Retrieval eval + index-version regression: a golden-set harness scores
     recall@k / nDCG / MRR, records a versioned artifact keyed on
     (embedder, chunker, corpus hash), and gates a regression on recall deltas
     using the same significance test as a model swap.
  4. The governed agent fabric: an AgentDirectory over A2A Agent Cards + AGNTCY/ACP
     + the MCP registry, where every resolution passes an allow-list gate and is
     recorded as an access decision on the audit chain.
  5. Generative UI: an agent run streams token/tool events, translated into the
     AG-UI protocol for an interactive frontend — one streamed run, with the run's
     provenance and audit, not a bolt-on UI layer.

Everything here is opt-in and additive behind ``@experimental(since="2.2")``;
nothing below is required to run Vincio.
"""

from __future__ import annotations

import asyncio

from _shared import example_provider

from vincio.a2a.protocol import AgentCard, AgentSkill
from vincio.agents import AgentExecutor
from vincio.agents.planner import Planner
from vincio.core.types import Budget, EvidenceItem
from vincio.evals import (
    EnvAction,
    EnvironmentSimulator,
    GAIAAdapter,
    RetrievalConfig,
    RetrievalGoldenSet,
    RetrievalQuery,
    SWEBenchAdapter,
    TauBenchAdapter,
    make_agent_solver,
    make_retail_environment,
    retrieval_regression,
    scripted_policy,
)
from vincio.registry import (
    ACPAgentManifest,
    ACPClient,
    AgentDirectory,
    MCPRegistryClient,
    MCPServerRecord,
)
from vincio.security.access import AllowListGate
from vincio.security.audit import AuditLog
from vincio.server.agui import agent_stream_to_agui
from vincio.storage import IndexRegressionStore
from vincio.tools.registry import ToolRegistry
from vincio.tools.runtime import ToolRuntime


def banner(title: str) -> None:
    print(f"\n== {title} ==")


async def environment_eval() -> None:
    banner("1. Stateful-environment eval + task-success oracle")
    env = make_retail_environment("cancel_refund")
    # A correct policy: cancel first, then refund (the env rejects refunding a
    # non-cancelled order — a policy the agent must respect).
    policy = scripted_policy(
        [
            EnvAction(tool="cancel_order", arguments={"order_id": "O1002"}),
            EnvAction(tool="refund_order", arguments={"order_id": "O1002"}),
            EnvAction(kind="finish", text="cancelled and refunded"),
        ]
    )
    result = EnvironmentSimulator().run(env, policy)
    print(f"  task success (oracle): {result.success}  score={result.verification.score}")
    for check in result.verification.checks:
        print(f"    [{'ok' if check.passed else 'XX'}] {check.name}: {check.detail}")
    print(f"  trajectory steps={len(result.trajectory.steps)} success={result.trajectory.success}")


async def benchmark_adapters() -> None:
    banner("2. Agentic benchmark adapters (verifiable, pinned, offline replay)")
    # SWE-bench Verified: a patch resolves an issue iff fail-to-pass turns green
    # and pass-to-pass stays green.
    swe = SWEBenchAdapter(
        [
            {"id": "ok", "gold": {"fail_to_pass": ["t1"], "pass_to_pass": ["t2"]},
             "recorded": {"tests": {"t1": "passed", "t2": "passed"}}},
            {"id": "no", "gold": {"fail_to_pass": ["t3"], "pass_to_pass": ["t4"]},
             "recorded": {"tests": {"t3": "failed", "t4": "passed"}}},
        ]
    )
    swe_report = await swe.replay()
    print(f"  swebench_verified: resolved={swe_report.success_rate}  hash={swe_report.task_set_hash}")

    # τ-bench: scored on the environment's database end state.
    tau = TauBenchAdapter(
        [
            {"id": "cancel-refund", "inputs": {"env": "retail", "env_task": "cancel_refund"},
             "recorded": [{"tool": "cancel_order", "arguments": {"order_id": "O1002"}},
                          {"tool": "refund_order", "arguments": {"order_id": "O1002"}}]},
        ]
    )
    tau_report = await tau.replay()
    print(f"  tau_bench:         success={tau_report.success_rate}  (end-state oracle)")
    # The score projects onto an EvalReport, so the optimizer/gates consume it.
    print(f"  -> EvalReport success mean = {tau_report.to_eval_report().summary()['success']['mean']}")

    # LIVE run — the identical scorer grades fresh agent output (not a recording).
    # `make_agent_solver` drives any ContextApp/AgentExecutor; here a stand-in
    # callable plays the agent. `make_env_solver` runs a policy through the world.
    gaia = GAIAAdapter([{"id": "g", "prompt": "capital of France?", "gold": "Paris"}])
    live = await gaia.run(make_agent_solver(lambda prompt: "Paris"))
    print(f"  gaia (live solve): success={live.success_rate}  replayed={live.replayed}")


async def retrieval_eval() -> None:
    banner("3. Retrieval eval + index-version regression (gated on recall deltas)")
    corpus = {
        "d0": "refund policy window thirty days",
        "d1": "shipping address change order",
        "d2": "cancel order before dispatch",
        "d3": "warranty coverage twelve months",
        "d4": "reset password email link",
        "d5": "loyalty points redemption rewards",
        "d6": "invoice download tax pdf",
        "d7": "subscription renewal billing cycle",
    }
    queries = [
        RetrievalQuery(id="q0", query="refund window", relevant_ids=["d0"]),
        RetrievalQuery(id="q1", query="change shipping address", relevant_ids=["d1"]),
        RetrievalQuery(id="q2", query="cancel my order", relevant_ids=["d2"]),
        RetrievalQuery(id="q3", query="warranty length", relevant_ids=["d3"]),
        RetrievalQuery(id="q4", query="reset password", relevant_ids=["d4"]),
        RetrievalQuery(id="q5", query="redeem loyalty points", relevant_ids=["d5"]),
        RetrievalQuery(id="q6", query="download invoice pdf", relevant_ids=["d6"]),
        RetrievalQuery(id="q7", query="subscription billing", relevant_ids=["d7"]),
    ]
    items = [EvidenceItem(id=c, source_id=c, text=t) for c, t in corpus.items()]
    golden = RetrievalGoldenSet(name="demo", queries=queries, corpus_hash=RetrievalGoldenSet.corpus_hash_of(items))

    def lexical(query: str, top_k: int) -> list[EvidenceItem]:
        qs = set(query.lower().split())
        ranked = sorted(corpus.items(), key=lambda kv: len(qs & set(kv[1].split())), reverse=True)
        return [EvidenceItem(id=c, source_id=c, text=t) for c, t in ranked[:top_k]]

    def degraded(query: str, top_k: int) -> list[EvidenceItem]:
        return [EvidenceItem(id=c, source_id=c, text=t) for c, t in list(corpus.items())[:top_k]]

    store = IndexRegressionStore()
    config = RetrievalConfig(embedder="hash", chunker="fixed")
    baseline = await retrieval_regression(lexical, golden, config, store=store)
    print(f"  baseline recall@3={baseline.current.get('recall_at_3')} (first measurement, key={baseline.key})")
    regressed = await retrieval_regression(degraded, golden, config, store=store, metrics=("recall_at_3",))
    print(f"  candidate passed={regressed.passed}  regressions={regressed.regressions}")
    print(f"  delta recall@3 = {regressed.deltas.get('recall_at_3')}  (caught as a CI gate)")


async def agent_fabric() -> None:
    banner("4. The governed agent fabric (A2A + AGNTCY/ACP + MCP registry)")
    audit = AuditLog(directory=None)
    gate = AllowListGate(allow=["researcher", "acp-planner", "filesystem"], deny=["evil*"])
    directory = AgentDirectory(allow_list=gate, audit=audit)

    directory.register(
        AgentCard(name="researcher", description="web research",
                  skills=[AgentSkill(id="research", name="research", tags=["research", "web"])]),
        url="https://researcher.example",
    )
    directory.register(AgentCard(name="evil-bot", skills=[AgentSkill(id="x", name="x")]))

    # Discover an AGNTCY/ACP agent and an MCP server into the same directory.
    await ACPClient(
        catalog=[ACPAgentManifest(id="acp-planner", name="acp-planner", capabilities=["planning"])]
    ).register_into_directory(directory)
    await MCPRegistryClient(
        catalog=[MCPServerRecord(name="filesystem", url="https://fs.example/mcp")]
    ).register_into_directory(directory)

    print(f"  discovered: {directory.names}")
    print(f"  find(tag='research') -> {[r.name for r in directory.find(tag='research')]}")
    for name in ("researcher", "acp-planner", "filesystem", "evil-bot"):
        decision = directory.try_resolve(name)
        print(f"  resolve {name:14s} -> {'ALLOW' if decision.allowed else 'DENY '}  ({decision.decision.reason})")
    print(f"  audited resolutions: {len(audit.query(action='agent_resolve'))} on the hash-chained log")


async def generative_ui() -> None:
    banner("5. Generative UI — an agent run streamed as AG-UI events")
    registry = ToolRegistry()

    @registry.register()
    def lookup(q: str) -> dict:
        """Look up a fact."""
        return {"answer": f"about {q}"}

    provider, model = example_provider(
        default_responder=lambda req: {"tool_call": {"name": "lookup", "arguments": {"q": "refunds"}}}
    )
    executor = AgentExecutor(
        provider, model=model, planner=Planner(mode="react"),
        tool_runtime=ToolRuntime(registry, cache_enabled=False), tool_specs=registry.specs(),
    )
    stream = executor.astream("look up the refund policy", budget=Budget(max_steps=3, max_tool_calls=2))
    async for ui_event in agent_stream_to_agui(stream):
        detail = ui_event.tool_call_name or ui_event.delta or ui_event.step_name or ""
        print(f"  {ui_event.type:22s} {detail}")


async def main() -> None:
    await environment_eval()
    await benchmark_adapters()
    await retrieval_eval()
    await agent_fabric()
    await generative_ui()
    print("\nMeasurable on the leaderboards, composable into a governed fabric, embeddable in a UI.")


if __name__ == "__main__":
    asyncio.run(main())
