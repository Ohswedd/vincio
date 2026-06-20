"""Parallel sub-graph scheduling (agents/scheduling).

A decomposed goal often has independent sub-graphs — branches that share no
state until they rejoin. Running them one after another wastes the worker pool;
running them with no coordination overspends the budget or blows the latency
SLA. The :class:`SubgraphScheduler` runs independent sub-graphs concurrently
across the lock-free distributed backend under one **fair-share budget** and an
optional **graph-level deadline**:

* **work-stealing** — sub-graphs are pulled off one shared queue by a pool of
  lease-guarded workers, so an idle worker immediately steals the next sub-graph
  instead of waiting on a static assignment;
* **fair-share budget** — a single budget is split across the sub-graphs by
  weight (the shares sum to exactly the cap), so no branch starves and the whole
  job stays inside one budget;
* **SLA deadline** — once the deadline passes the scheduler stops dispatching new
  sub-graphs and returns the completed results plus the durable partial state of
  the rest, rather than running everything past the deadline.

Every sub-graph is a normal durable :class:`~vincio.agents.graph.StateGraph`
thread, lease-guarded and CAS-committed by the worker pool, so a sub-graph that
does not finish before the deadline is not lost — its latest checkpoint is the
partial result and it can be resumed later.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.concurrency import gather_bounded
from ..core.types import Budget
from ..core.utils import new_id
from .backends import WorkerPoolBackend
from .distributed import GraphCoordinator
from .graph import Checkpointer, GraphResult

__all__ = ["SubgraphTask", "SubgraphOutcome", "ScheduleResult", "SubgraphScheduler"]


class SubgraphTask:
    """One independent sub-graph to schedule.

    ``graph`` is a :class:`~vincio.agents.graph.StateGraph` (or compiled form);
    ``input`` is its initial state; ``weight`` biases its fair-share allocation
    (default equal). Not a Pydantic model — it holds the live graph object.
    """

    __slots__ = ("id", "graph", "input", "thread_id", "weight")

    def __init__(
        self,
        graph: Any,
        input: dict[str, Any] | None = None,
        *,
        id: str | None = None,
        thread_id: str | None = None,
        weight: float = 1.0,
    ) -> None:
        self.id = id or new_id("subgraph")
        self.graph = graph
        self.input = input or {}
        self.thread_id = thread_id or new_id("thread")
        self.weight = max(0.0, weight)


class SubgraphOutcome(BaseModel):
    """The result of scheduling one sub-graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    thread_id: str
    status: str  # done | interrupted | max_steps | deadline
    partial: bool = False
    steps: int = 0
    budget_share_usd: float = 0.0
    result: GraphResult | None = None
    partial_state: dict[str, Any] = Field(default_factory=dict)


class ScheduleResult(BaseModel):
    """The aggregate result of one scheduling pass."""

    completed: list[SubgraphOutcome] = Field(default_factory=list)
    partial: list[SubgraphOutcome] = Field(default_factory=list)
    deadline_hit: bool = False
    peak_concurrency: int = 0
    total_steps: int = 0
    makespan_steps: int = 0
    speedup: float = 1.0
    shares_usd: list[float] = Field(default_factory=list)
    budget_usd: float | None = None

    @property
    def all(self) -> list[SubgraphOutcome]:
        return [*self.completed, *self.partial]


class SubgraphScheduler:
    """Runs independent sub-graphs concurrently under a fair-share budget + SLA.

    Builds on :class:`~vincio.agents.backends.WorkerPoolBackend`, so each
    sub-graph is a lease-guarded, CAS-committed durable thread — the scheduler
    adds the cross-sub-graph concerns (work-stealing pool, budget fairness,
    deadline) without changing how a single graph runs. ``clock`` is injectable
    for deterministic deadline tests.
    """

    def __init__(
        self,
        *,
        workers: int = 4,
        store: Any = None,
        coordinator: GraphCoordinator | None = None,
        lease_ttl_s: float = 30.0,
        budget: Budget | None = None,
        deadline_s: float | None = None,
        clock: Any = None,
    ) -> None:
        self.backend = WorkerPoolBackend(
            workers=workers, store=store, coordinator=coordinator, lease_ttl_s=lease_ttl_s
        )
        self.workers = max(1, workers)
        self.budget = budget
        self.deadline_s = deadline_s
        self._clock = clock or time.monotonic

    async def run(self, tasks: list[SubgraphTask]) -> ScheduleResult:
        """Schedule ``tasks`` and return completed + partial outcomes."""
        if not tasks:
            return ScheduleResult(budget_usd=self.budget.max_cost_usd if self.budget else None)

        total_budget = self.budget.max_cost_usd if self.budget is not None else None
        total_weight = sum(t.weight for t in tasks) or float(len(tasks))
        start = self._clock()
        deadline = start + self.deadline_s if self.deadline_s is not None else None

        queue: asyncio.Queue[int] = asyncio.Queue()
        for index in range(len(tasks)):
            queue.put_nowait(index)
        outcomes: list[SubgraphOutcome | None] = [None] * len(tasks)
        shares: list[float] = []
        worker_steps = [0] * self.workers

        peak = 0
        active = 0
        deadline_hit = False
        lock = asyncio.Lock()

        async def worker(wid: int) -> None:
            nonlocal peak, active, deadline_hit
            while True:
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                task = tasks[index]
                if deadline is not None and self._clock() >= deadline:
                    deadline_hit = True
                    outcomes[index] = self._partial_outcome(task)
                    continue
                # Weighted fair share of one budget: the shares sum to the cap, so
                # no sub-graph starves and the whole job stays inside one budget.
                share = (
                    total_budget * (task.weight / total_weight)
                    if total_budget is not None
                    else 0.0
                )
                async with lock:
                    shares.append(share)
                    active += 1
                    peak = max(peak, active)
                try:
                    enriched = dict(task.input)
                    if share:
                        enriched.setdefault("_budget_share_usd", share)
                    result: GraphResult = await self.backend.run(
                        task.graph, enriched, thread_id=task.thread_id
                    )
                    worker_steps[wid] += result.steps
                    outcomes[index] = SubgraphOutcome(
                        id=task.id,
                        thread_id=task.thread_id,
                        status=result.status,
                        partial=result.status != "done",
                        steps=result.steps,
                        budget_share_usd=share,
                        result=result,
                        partial_state=dict(result.state) if result.status != "done" else {},
                    )
                finally:
                    async with lock:
                        active -= 1

        await gather_bounded([worker(wid) for wid in range(self.workers)], limit=self.workers)

        resolved = [o for o in outcomes if o is not None]
        completed = [o for o in resolved if o.status == "done"]
        partial = [o for o in resolved if o.status != "done"]
        total_steps = sum(o.steps for o in completed)
        makespan = max((s for s in worker_steps if s), default=0)
        speedup = (total_steps / makespan) if makespan else 1.0
        return ScheduleResult(
            completed=completed,
            partial=partial,
            deadline_hit=deadline_hit,
            peak_concurrency=peak,
            total_steps=total_steps,
            makespan_steps=makespan,
            speedup=round(speedup, 4),
            shares_usd=shares,
            budget_usd=total_budget,
        )

    def _partial_outcome(self, task: SubgraphTask) -> SubgraphOutcome:
        """A sub-graph not dispatched before the deadline: its durable last state.

        The latest checkpoint (if the thread ever started) is the partial result;
        otherwise the partial state is the sub-graph's input. Either way the
        thread can be resumed later — nothing is lost to the deadline."""
        latest = Checkpointer(self.backend.store).latest(task.thread_id)
        state = dict(latest.state) if latest is not None else dict(task.input)
        return SubgraphOutcome(
            id=task.id,
            thread_id=task.thread_id,
            status="deadline",
            partial=True,
            steps=latest.step if latest is not None else 0,
            partial_state=state,
        )
