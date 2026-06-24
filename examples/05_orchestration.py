"""Multi-agent orchestration & durable execution.

Four ways to coordinate work in Vincio, all offline on the deterministic mock:

  1. Crews    - role-based agents share a blackboard, each runs on its slice of
                the crew budget, and the team is guaranteed to terminate.
  2. Graphs   - durable stateful graphs: checkpoint / resume / edit-and-resume /
                time-travel forks, and a human-in-the-loop interrupt gate.
  3. Workflows- deterministic DAGs the model can't steer: retries, branching,
                compensation (saga rollback), and a pausing approval gate.
  4. Backend  - the distributed durable-execution plane: sub-graph scheduling
                under a fair-share budget, plus durable timers that pause a
                graph without pinning a worker.

Pick a crew when you want agents to collaborate, a graph when you need durable
resumable state with human gates, and a workflow when control flow must be fixed
and reproducible.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from _shared import example_provider

from vincio import ContextApp
from vincio.agents import SubgraphScheduler, SubgraphTask, interrupt
from vincio.agents.timers import TimerService, sleep_for
from vincio.core.types import Budget
from vincio.core.utils import utcnow


def banner(title: str) -> None:
    print(f"\n== {title} ==")


# --------------------------------------------------------------------------- #
# 1. Multi-agent crew: roles, delegation, shared blackboard, bounded budget.
# --------------------------------------------------------------------------- #
def section_crew() -> None:
    banner("1. multi-agent crew")

    # The mock answers differently per role by sniffing the system prompt, so a
    # sequential crew shows real hand-off: the writer reads what the researcher
    # posted to the blackboard.
    provider, model = example_provider(
        default_responder=lambda request: (
            # The system prompt names the active member ("You are <name>."), so
            # we route on that to give each role a distinct, deterministic voice.
            "Findings: refunds spiked 14% in Q3, driven by the Basic plan's $5 fee."
            if "You are researcher" in "\n".join(m.text for m in request.messages)
            else "Report: Q3 refunds rose 14%; recommend waiving the Basic plan fee."
        )
    )
    app = ContextApp(name="crew_demo", provider=provider, model=model)

    # Sequential process: members run in order over a shared blackboard. Each
    # member gets a slice of the crew budget (budget_fraction; default is an
    # equal split) so the whole team is guaranteed to terminate.
    crew = app.crew(
        name="refund_team",
        members=[
            {"name": "researcher", "goal": "gather the relevant numbers",
             "keywords": ["find", "data"], "budget_fraction": 0.6},
            {"name": "writer", "goal": "draft a crisp recommendation",
             "budget_fraction": 0.4},
        ],
        process="sequential",
    )
    result = crew.run("Explain the Q3 refund trend and recommend a fix")
    print("status:", result.status)
    print("output:", result.output)
    for report in result.reports:
        # termination_reason tells you WHY a member stopped (done / budget / rounds).
        print(f"  - {report.role}: {str(report.answer)[:70]} ({report.termination_reason})")
    print("blackboard keys:", list(result.blackboard["entries"]))
    print("metrics:", result.metrics())

    # Hierarchical process: a manager delegates each task to the best-matching
    # member. Offline this is deterministic keyword routing; with a real
    # provider the manager plans the delegation with the LLM.
    triage = app.crew(
        name="support_triage",
        members=[
            {"name": "billing", "description": "invoices, refunds, payments",
             "keywords": ["invoice", "refund"]},
            {"name": "legal", "description": "contracts and clauses",
             "keywords": ["contract", "clause"]},
        ],
        process="hierarchical",
    )
    routed = triage.run("Is invoice INV-77 refundable under the contract?")
    print("delegations:", [(d.to_agent, d.reason) for d in routed.delegations])


# --------------------------------------------------------------------------- #
# 2. Durable stateful graph: checkpoint / resume / edit / time-travel / HITL.
# --------------------------------------------------------------------------- #
def section_graph() -> None:
    banner("2. durable stateful graph")

    app = ContextApp(name="graph_demo")

    # A contract-review pipeline as a graph. The app-bound graph persists a
    # checkpoint after every node into the app's metadata store, so the thread
    # survives interrupts (and process restarts with a SQLite/Postgres store).
    graph = app.graph("contract_review")
    graph.add_node("ingest", lambda s: {"clauses": ["auto-renewal", "late fees"]})
    graph.add_node("analyze", lambda s: {"risk": "high" if "auto-renewal" in s["clauses"] else "low"})
    # interrupt() pauses the graph and surfaces a payload to a human; the value
    # passed to resume() becomes this call's return on re-run (human-in-the-loop).
    graph.add_node(
        "approve",
        lambda s: {"approved": interrupt(s, {"question": f"Risk is {s['risk']} - proceed?"})},
    )
    graph.add_node("report", lambda s: {"report": f"risk={s['risk']} approved={s['approved']}"})
    graph.add_edge("ingest", "analyze")
    graph.add_edge("analyze", "approve")
    graph.add_edge("approve", "report")
    review = graph.compile()

    # (a) Run until the human gate pauses the graph.
    paused = review.invoke({})
    print("status:", paused.status, "| asked:", paused.interrupt_payload)

    # (b) Edit-and-resume: change persisted state before continuing.
    review.update_state(paused.thread_id, {"risk": "medium"})

    # (c) Resume with the human's answer; the gate re-runs and receives `value`.
    done = review.resume(paused.thread_id, value=True)
    print("resumed report:", done.state["report"])

    # (d) Time-travel: every step left a checkpoint; fork the post-analyze one
    #     and replay the tail with a different human decision.
    history = review.history(paused.thread_id)
    print("checkpoints:", [(c.step, c.status) for c in history])
    forked = review.fork(history[2].id)
    replayed = review.resume(forked, value=False)
    print("forked report:", replayed.state["report"])


# --------------------------------------------------------------------------- #
# 3. Deterministic workflow: retries, branching, compensation, approval gate.
# --------------------------------------------------------------------------- #
def section_workflow() -> None:
    banner("3. deterministic workflow (saga)")

    app = ContextApp(name="workflow_demo")

    # A booking saga. Unlike an agent, the control flow is fixed: the engine
    # decides the next step, never the model. Steps run with retries; a failure
    # triggers compensation in reverse completion order (the saga rollback).
    attempts = {"n": 0}

    def reserve_inventory(ctx) -> dict:
        return {"sku": "WIDGET-1", "reserved": True}

    def charge_card(ctx) -> dict:
        # Fail once to demonstrate retry-with-backoff, then succeed.
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("payment gateway timeout")
        return {"charge_id": "ch_123", "amount": 49}

    def ship(ctx) -> dict:
        return {"tracking": "1Z-999"}

    wf = app.workflow("place_order")
    wf.step("reserve", reserve_inventory,
            # compensation runs if a LATER step fails: release the reservation.
            compensation=lambda ctx: print("    compensate: released inventory reservation"))
    wf.step("charge", charge_card, depends_on=["reserve"], retries=3, retry_delay_s=0.0,
            compensation=lambda ctx: print("    compensate: refunded the charge"))
    # Branching: this step only runs when the `when` predicate is true.
    wf.step("ship", ship, depends_on=["charge"],
            when=lambda ctx: ctx.output_of("charge")["amount"] > 0)

    result = wf.run()
    print("status:", result.status)
    print("charge attempts:", result.context.results["charge"].attempts, "(retried then succeeded)")
    print("output:", result.output)

    # Now force a failure downstream to watch compensation unwind the saga.
    wf2 = app.workflow("place_order_fail")
    wf2.step("reserve", reserve_inventory,
             compensation=lambda ctx: print("    compensate: released inventory"))
    wf2.step("charge", lambda ctx: {"charge_id": "ch_9", "amount": 49}, depends_on=["reserve"],
             compensation=lambda ctx: print("    compensate: refunded the charge"))
    wf2.step("ship", lambda ctx: (_ for _ in ()).throw(RuntimeError("carrier outage")),
             depends_on=["charge"])
    failed = wf2.run()
    print("failed status:", failed.status, "| compensated:", failed.compensated_steps)

    # Approval gate: a step marked approval=True with no approver PAUSES the run.
    # Resume by supplying an approvals map — a first-class human interrupt.
    wf3 = app.workflow("publish")
    wf3.step("draft", lambda ctx: {"text": "Q3 results look strong."})
    wf3.step("legal_review", lambda ctx: {"ok": True}, depends_on=["draft"], approval=True)
    wf3.step("publish", lambda ctx: {"url": "/posts/q3"}, depends_on=["legal_review"])
    pending = wf3.run()
    print("gate status:", pending.status, "| pending approvals:", pending.pending_approvals)
    approved = wf3.resume(pending, approvals={"legal_review": True})
    print("after approval:", approved.status, "| output:", approved.output)


# --------------------------------------------------------------------------- #
# 4. Distributed backend: sub-graph scheduling + durable timers.
# --------------------------------------------------------------------------- #
async def section_distributed() -> None:
    banner("4. distributed backend: sub-graph scheduling")

    app = ContextApp(name="dist_demo")

    # Three independent sub-graphs (no shared state) we want to run concurrently
    # over a worker pool under ONE budget. Each is a normal durable graph thread.
    def make_branch(label: str):
        g = app.graph(f"branch_{label}")
        g.add_node("work", lambda s: {"done": f"{label}:{s['n'] * 2}"})
        g.set_entry("work")
        from vincio.agents import END
        g.add_edge("work", END)
        return g.compile()

    tasks = [
        SubgraphTask(make_branch("alpha"), {"n": 1}, weight=2.0),  # heavier weight
        SubgraphTask(make_branch("beta"), {"n": 2}, weight=1.0),
        SubgraphTask(make_branch("gamma"), {"n": 3}, weight=1.0),
    ]
    # Fair-share budget: the cap is split across sub-graphs by weight (shares sum
    # to the cap), so no branch starves and the whole job stays inside one budget.
    scheduler = SubgraphScheduler(workers=3, budget=Budget(max_cost_usd=0.30))
    sched = await scheduler.run(tasks)
    print("completed:", [o.result.state["done"] for o in sched.completed])
    print("peak concurrency:", sched.peak_concurrency,
          "| budget shares:", [round(s, 3) for s in sched.shares_usd])

    # --- Durable timers ----------------------------------------------------- #
    banner("4b. durable timers (pause without pinning a worker)")

    # A node can pause the graph for a wall-clock delay. The absolute wake time
    # is persisted into the checkpoint, so nothing holds a worker while we wait.
    timed = app.graph("nightly_job")
    timed.add_node("queue", lambda s: {"queued": True})
    timed.add_node("wait", lambda s: (sleep_for(s, 3600), {"woke": True})[1])  # ~1h, durably
    timed.add_node("run", lambda s: {"ran": True})
    timed.add_edge("queue", "wait")
    timed.add_edge("wait", "run")
    job = timed.compile()

    paused = job.invoke({})
    print("status:", paused.status, "(durably waiting, no worker held)")

    # A TimerService (often in a fresh process) scans for due timers and resumes
    # them. Nothing is due yet, so a tick now does nothing.
    service = TimerService(job)
    print("due right now:", len(service.tick()))

    # Simulate time passing by ticking with a future clock: the timer fires.
    future = utcnow() + timedelta(seconds=3600)
    woken = service.tick(now=future)
    print("after an hour:", [r.state.get("ran") for r in woken])


# --------------------------------------------------------------------------- #
async def main() -> None:
    section_crew()
    section_graph()
    section_workflow()
    await section_distributed()
    print("\nAll four orchestration models ran offline on the deterministic mock.")


if __name__ == "__main__":
    asyncio.run(main())
