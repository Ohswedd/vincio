"""Orchestrator & planner depth: hierarchical planning, in-place plan repair,
cost-aware action selection, parallel sub-graph scheduling, and durable timers.

Runs fully offline on the deterministic mock provider.
"""

import asyncio
from datetime import UTC, datetime, timedelta

from _shared import example_provider

from vincio import (
    Budget,
    ContextApp,
    StateGraph,
    SubgraphScheduler,
    SubgraphTask,
    TimerService,
    sleep_for,
    wait_for_event,
)
from vincio.agents import Checkpointer, HTNDomain
from vincio.storage.base import InMemoryMetadataStore

PROVIDER, MODEL = example_provider()  # MockProvider offline; real provider with env vars


def hierarchical_and_repair() -> None:
    """1+2. An HTN domain decomposes a goal into a sub-goal tree (a parallel
    sub-goal runs its leaves concurrently). One leaf binds a flaky tool with a
    backup; when it fails, the agent re-binds in place and still finishes —
    the repair is recorded as a trajectory event, not a silent retry."""
    app = ContextApp(name="incident_triage", provider=PROVIDER, model=MODEL)

    def enrich_primary(alert: str = "") -> dict:
        """Primary enrichment service (currently failing)."""
        raise RuntimeError("enrichment upstream 503")

    def enrich_backup(alert: str = "") -> dict:
        """Backup enrichment service."""
        return {"alert": alert, "owner": "db-team", "severity": "high"}

    app.add_tool(enrich_primary)
    app.add_tool(enrich_backup)

    # The HTN domain's entry task is "root" (the planner's default entry point).
    domain = (
        HTNDomain()
        .method("root", ["assess", "respond"])
        .method("assess", ["classify", "enrich"], ordering="parallel")
        .operator("classify", step_type="think", instruction="classify the severity")
        .operator(
            "enrich", step_type="tool", tool_name="enrich_primary",
            instruction="enrich the alert", fallbacks=["enrich_backup"],
        )
        .operator("respond", step_type="finalize", instruction="recommend a response")
    )
    agent = app.agent(
        tools=["enrich_primary", "enrich_backup"],
        planner="hierarchical",
        domain=domain,
        max_steps=10,
    )
    state = agent.run("Triage the database CPU alert")
    print("1+2. hierarchical run:", state.termination_reason)
    print("     repairs:", [f"{r.action}: {r.from_binding} -> {r.to_binding}" for r in state.repairs])


def cost_aware_selection() -> None:
    """3. The agent spends a cheaper capable model on the easy steps and escalates
    to a stronger one only when confidence is low, reading pricing and capability
    from the model registry."""
    app = ContextApp(name="summarize", provider=PROVIDER, model=MODEL)
    agent = app.agent(cost_aware_models=["gpt-5.2-mini", "gpt-5.2"])
    state = agent.run("Summarize the incident report")
    picks = [s["model"] for s in state.working_memory.get("_selections", [])]
    print("3. cost-aware: model per step ->", picks or "(no model steps)")


def parallel_subgraphs() -> None:
    """4. Independent sub-graphs run concurrently across a worker pool under one
    fair-share budget, with an SLA deadline that returns partial results."""

    def make_branch(name: str) -> StateGraph:
        graph = StateGraph(name)
        graph.add_node("fetch", lambda s: {"n": s.get("n", 0) + 1})
        graph.add_node("score", lambda s: {"n": s["n"] * 10})
        graph.add_edge("fetch", "score")
        return graph

    tasks = [SubgraphTask(make_branch(f"region_{i}"), {"n": i}) for i in range(4)]
    result = asyncio.run(SubgraphScheduler(workers=4, budget=Budget(max_cost_usd=1.0)).run(tasks))
    print(
        f"4. parallel sub-graphs: {len(result.completed)} done at "
        f"{result.peak_concurrency}x concurrency, {result.speedup}x speedup, "
        f"fair shares={result.shares_usd}"
    )


def durable_timers() -> None:
    """5. A graph pauses on a durable timer and on an external event, survives a
    process restart, and resumes when the timer is due / the event is delivered —
    without holding a worker while it waits."""
    store = InMemoryMetadataStore()
    now = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)

    def build() -> StateGraph:
        graph = StateGraph("scheduled_review")
        graph.add_node("draft", lambda s: {"stage": "drafted"})

        def cool_off(s):
            sleep_for(s, 86_400, clock=lambda: now)  # wait a day, durably
            return {"stage": "cooled_off"}

        graph.add_node("cool_off", cool_off)
        graph.add_node("approve", lambda s: {"approval": wait_for_event(s, "approved")})
        graph.add_node("publish", lambda s: {"stage": "published"})
        graph.add_edge("draft", "cool_off")
        graph.add_edge("cool_off", "approve")
        graph.add_edge("approve", "publish")
        return graph

    paused = build().compile(checkpointer=Checkpointer(store)).invoke({}, thread_id="rev-1")
    print("5. durable timer paused:", paused.status, "— no worker held while waiting")

    # A fresh process (new compiled graph + checkpointer, same store) resumes it.
    restarted = build().compile(checkpointer=Checkpointer(store))
    after_sleep = TimerService(restarted, clock=lambda: now + timedelta(days=2)).tick()[0]
    print("   woke after restart, now paused at:", after_sleep.next_nodes or "(done)")

    # The approval gate waits for a named event; delivering it resumes the graph.
    done = TimerService(restarted).deliver("rev-1", "approved", payload={"by": "alice"})
    print("   delivered approval ->", done.state.get("stage"), done.state.get("approval"))


if __name__ == "__main__":
    hierarchical_and_repair()
    cost_aware_selection()
    parallel_subgraphs()
    durable_timers()
