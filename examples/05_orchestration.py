"""Multi-agent orchestration & durable execution.

Four ways to coordinate work, all offline on the deterministic mock. Pick a
CREW when agents should collaborate over a shared blackboard under a bounded
budget; a GRAPH when you need durable, resumable state with human gates
(checkpoint / resume / edit / time-travel); a WORKFLOW when control flow must be
fixed and reproducible (retries, branching, saga compensation, approval gates);
and the distributed BACKEND to schedule sub-graphs under a fair-share budget
with durable timers that pause a run without pinning a worker.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from _shared import example_provider

from vincio import ContextApp
from vincio.agents import END, SubgraphScheduler, SubgraphTask, interrupt
from vincio.agents.timers import TimerService, sleep_for
from vincio.core.types import Budget
from vincio.core.utils import utcnow


def section_crew() -> None:
    # Role-based members run over a SHARED blackboard, each on a slice of the crew
    # budget (budget_fraction), so the team is guaranteed to terminate. The mock
    # routes on the system prompt ("You are <name>.") to give each role a distinct
    # voice, showing a real hand-off: the writer reads what the researcher posted.
    provider, model = example_provider(default_responder=lambda request: (
        "Findings: refunds spiked 14% in Q3, driven by the Basic plan's $5 fee."
        if "You are researcher" in "\n".join(m.text for m in request.messages)
        else "Report: Q3 refunds rose 14%; recommend waiving the Basic plan fee."))
    app = ContextApp(name="crew_demo", provider=provider, model=model)

    crew = app.crew(name="refund_team", process="sequential", members=[
        {"name": "researcher", "goal": "gather the relevant numbers",
         "keywords": ["find", "data"], "budget_fraction": 0.6},
        {"name": "writer", "goal": "draft a crisp recommendation", "budget_fraction": 0.4},
    ])
    result = crew.run("Explain the Q3 refund trend and recommend a fix")
    print("1. sequential crew:", result.output)
    for r in result.reports:  # termination_reason tells you WHY a member stopped
        print(f"     {r.role}: {str(r.answer)[:60]} ({r.termination_reason})")

    # Hierarchical process: a manager delegates each task to the best-matching
    # member (deterministic keyword routing offline; LLM-planned with a real model).
    triage = app.crew(name="support_triage", process="hierarchical", members=[
        {"name": "billing", "keywords": ["invoice", "refund"]},
        {"name": "legal", "keywords": ["contract", "clause"]},
    ])
    routed = triage.run("Is invoice INV-77 refundable under the contract?")
    print("   hierarchical delegations:", [(d.to_agent, d.reason) for d in routed.delegations])


def section_graph() -> None:
    # A contract-review pipeline as a durable graph. The app-bound graph persists a
    # checkpoint after EVERY node, so the thread survives interrupts (and process
    # restarts with a SQLite/Postgres store). interrupt() pauses and surfaces a
    # payload to a human; the value passed to resume() becomes its return on re-run.
    graph = ContextApp(name="graph_demo").graph("contract_review")
    graph.add_node("ingest", lambda s: {"clauses": ["auto-renewal", "late fees"]})
    graph.add_node("analyze", lambda s: {"risk": "high" if "auto-renewal" in s["clauses"] else "low"})
    graph.add_node("approve", lambda s: {"approved": interrupt(s, {"question": f"Risk {s['risk']} - ok?"})})
    graph.add_node("report", lambda s: {"report": f"risk={s['risk']} approved={s['approved']}"})
    graph.add_edge("ingest", "analyze")
    graph.add_edge("analyze", "approve")
    graph.add_edge("approve", "report")
    review = graph.compile()

    paused = review.invoke({})  # (a) run until the human gate pauses
    review.update_state(paused.thread_id, {"risk": "medium"})  # (b) edit persisted state
    done = review.resume(paused.thread_id, value=True)  # (c) resume with the human's answer
    # (d) time-travel: every step left a checkpoint; fork one and replay with a
    #     different decision — the durable state makes counterfactuals free.
    history = review.history(paused.thread_id)
    replayed = review.resume(review.fork(history[2].id), value=False)
    print(f"2. durable graph: paused asking {paused.interrupt_payload}")
    print(f"     resumed -> {done.state['report']} | forked -> {replayed.state['report']}")


def section_workflow() -> None:
    # A booking saga. Unlike an agent, control flow is FIXED — the engine decides
    # the next step, never the model. Steps run with retries; a failure triggers
    # compensation in reverse completion order (the saga rollback).
    app = ContextApp(name="workflow_demo")
    attempts = {"n": 0}

    def charge_card(ctx) -> dict:
        attempts["n"] += 1  # fail once to show retry-with-backoff, then succeed
        if attempts["n"] < 2:
            raise RuntimeError("payment gateway timeout")
        return {"charge_id": "ch_123", "amount": 49}

    wf = app.workflow("place_order")
    wf.step("reserve", lambda ctx: {"sku": "WIDGET-1", "reserved": True},
            compensation=lambda ctx: print("     compensate: released inventory"))
    wf.step("charge", charge_card, depends_on=["reserve"], retries=3, retry_delay_s=0.0)
    # Branching: a step runs only when its `when` predicate is true.
    wf.step("ship", lambda ctx: {"tracking": "1Z-999"}, depends_on=["charge"],
            when=lambda ctx: ctx.output_of("charge")["amount"] > 0)
    result = wf.run()
    print(f"3. workflow saga: {result.status} after "
          f"{result.context.results['charge'].attempts} charge attempts -> {result.output}")

    # Force a downstream failure to watch compensation unwind completed steps.
    wf2 = app.workflow("place_order_fail")
    wf2.step("reserve", lambda ctx: {"reserved": True},
             compensation=lambda ctx: print("     compensate: released inventory"))
    wf2.step("charge", lambda ctx: {"amount": 49}, depends_on=["reserve"],
             compensation=lambda ctx: print("     compensate: refunded the charge"))
    wf2.step("ship", lambda ctx: (_ for _ in ()).throw(RuntimeError("carrier outage")),
             depends_on=["charge"])
    failed = wf2.run()
    print(f"     forced failure: {failed.status}, compensated {failed.compensated_steps}")

    # Approval gate: a step marked approval=True with no approver PAUSES the run;
    # resume by supplying an approvals map — a first-class human interrupt.
    wf3 = app.workflow("publish")
    wf3.step("draft", lambda ctx: {"text": "Q3 results look strong."})
    wf3.step("legal_review", lambda ctx: {"ok": True}, depends_on=["draft"], approval=True)
    wf3.step("publish", lambda ctx: {"url": "/posts/q3"}, depends_on=["legal_review"])
    pending = wf3.run()
    approved = wf3.resume(pending, approvals={"legal_review": True})
    print(f"     approval gate: {pending.status} -> after approval {approved.status} {approved.output}")


async def section_distributed() -> None:
    app = ContextApp(name="dist_demo")

    # Three independent sub-graphs run concurrently over a worker pool under ONE
    # budget. Fair-share splits the cap by weight (shares sum to the cap), so no
    # branch starves and the whole job stays inside a single budget.
    def make_branch(label: str):
        g = app.graph(f"branch_{label}")
        g.add_node("work", lambda s: {"done": f"{label}:{s['n'] * 2}"})
        g.set_entry("work")
        g.add_edge("work", END)
        return g.compile()

    tasks = [SubgraphTask(make_branch("alpha"), {"n": 1}, weight=2.0),
             SubgraphTask(make_branch("beta"), {"n": 2}, weight=1.0),
             SubgraphTask(make_branch("gamma"), {"n": 3}, weight=1.0)]
    sched = await SubgraphScheduler(workers=3, budget=Budget(max_cost_usd=0.30)).run(tasks)
    print("4. sub-graph scheduling:", [o.result.state["done"] for o in sched.completed],
          f"| peak concurrency {sched.peak_concurrency}")

    # Durable timers: a node pauses the graph for a wall-clock delay. The absolute
    # wake time is persisted into the checkpoint, so NOTHING holds a worker while
    # we wait. A TimerService (often a fresh process) scans for due timers.
    timed = app.graph("nightly_job")
    timed.add_node("queue", lambda s: {"queued": True})
    timed.add_node("wait", lambda s: (sleep_for(s, 3600), {"woke": True})[1])  # ~1h, durably
    timed.add_node("run", lambda s: {"ran": True})
    timed.add_edge("queue", "wait")
    timed.add_edge("wait", "run")
    job = timed.compile()
    paused = job.invoke({})
    service = TimerService(job)
    due_now = len(service.tick())  # nothing is due yet
    woken = service.tick(now=utcnow() + timedelta(seconds=3600))  # simulate an hour passing
    print(f"   durable timer: paused={paused.status} (no worker held), "
          f"due-now={due_now}, after-1h ran={[r.state.get('ran') for r in woken]}")


async def main() -> None:
    section_crew()
    section_graph()
    section_workflow()
    await section_distributed()


if __name__ == "__main__":
    asyncio.run(main())
