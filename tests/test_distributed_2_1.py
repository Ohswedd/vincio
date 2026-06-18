"""Distributed durable execution (2.1): lease + CAS coordination, BSP parallel
super-steps, Send map-reduce, worker-pool fan-out, Ray/Temporal export adapters,
and workflow map-reduce. All offline and deterministic.

The 2.1.1 follow-up adds a cross-backend conformance battery
(:func:`vincio.testing.assert_backend_conformance`) that holds every backend to
the native engine's semantics, and exercises the channel-default reducer so a
map-reduce needs no upstream seed node."""

from __future__ import annotations

import asyncio
import operator

import pytest
from pydantic import BaseModel, Field

from vincio.agents import (
    END,
    CompiledGraph,
    DistributedCheckpointer,
    InMemoryGraphCoordinator,
    RayBackend,
    Send,
    StateGraph,
    TemporalBackend,
    WorkerPoolBackend,
)
from vincio.core.errors import CheckpointConflictError
from vincio.storage.base import InMemoryMetadataStore
from vincio.testing import assert_backend_conformance
from vincio.workflows import Workflow


def _inc_graph(name: str = "inc") -> StateGraph:
    g = StateGraph(name)
    g.add_node("inc", lambda s: {"n": s.get("n", 0) + 1})
    g.add_edge("inc", END)
    return g


# ---------------------------------------------------------------------------
# Coordinator: lease + optimistic-concurrency CAS
# ---------------------------------------------------------------------------


class TestCoordinator:
    def test_cas_rejects_stale_commit(self):
        coord = InMemoryGraphCoordinator()
        assert coord.acquire("t1", "A", ttl_s=100)
        assert coord.version("t1") == 0
        assert coord.commit("t1", "A", expected_version=0) == 1
        with pytest.raises(CheckpointConflictError):
            coord.commit("t1", "A", expected_version=0)  # stale
        assert coord.commit("t1", "A", expected_version=1) == 2

    def test_lease_excludes_other_owner(self):
        coord = InMemoryGraphCoordinator()
        assert coord.acquire("t1", "A", ttl_s=100)
        assert not coord.acquire("t1", "B", ttl_s=100)
        # B cannot commit even at the right version: lease is coupled into CAS.
        with pytest.raises(CheckpointConflictError):
            coord.commit("t1", "B", expected_version=0)

    def test_expired_lease_is_reclaimed(self):
        clock = {"t": 0.0}
        coord = InMemoryGraphCoordinator(clock=lambda: clock["t"])
        assert coord.acquire("t1", "A", ttl_s=10)
        assert not coord.acquire("t1", "B", ttl_s=10)
        clock["t"] = 11.0  # A's lease expired
        assert coord.acquire("t1", "B", ttl_s=10)
        with pytest.raises(CheckpointConflictError):
            coord.commit("t1", "A", expected_version=0)  # A lost it
        assert coord.commit("t1", "B", expected_version=0) == 1

    def test_release_frees_the_thread(self):
        coord = InMemoryGraphCoordinator()
        coord.acquire("t1", "A", ttl_s=100)
        coord.release("t1", "A")
        assert coord.acquire("t1", "B", ttl_s=100)


class TestDistributedCheckpointer:
    def test_lease_prevents_double_execution(self):
        store = InMemoryMetadataStore()
        coord = InMemoryGraphCoordinator()
        a = DistributedCheckpointer(store, coordinator=coord, owner="A", lease_ttl_s=100)
        a.on_thread_start("t1")  # A holds the lease for the whole run
        b = DistributedCheckpointer(store, coordinator=coord, owner="B", lease_ttl_s=100)
        with pytest.raises(CheckpointConflictError):
            b.on_thread_start("t1")
        a.on_thread_end("t1")
        b.on_thread_start("t1")  # now free

    async def test_single_run_commits_monotonic_versions(self):
        store = InMemoryMetadataStore()
        coord = InMemoryGraphCoordinator()
        ckpt = DistributedCheckpointer(store, coordinator=coord, owner="A")
        g = StateGraph("steps")
        g.add_node("a", lambda s: {"x": 1})
        g.add_node("b", lambda s: {"y": 2})
        g.add_edge("a", "b")
        g.add_edge("b", END)
        result = await g.compile(checkpointer=ckpt).ainvoke({}, thread_id="t1")
        assert result.status == "done"
        versions = [c.version for c in ckpt.history("t1")]
        assert versions == sorted(versions) and versions[0] >= 1
        assert all(c.lease_owner == "A" for c in ckpt.history("t1"))


# ---------------------------------------------------------------------------
# Cross-restart durability: resume on a different worker
# ---------------------------------------------------------------------------


async def test_durable_resume_across_workers():
    store = InMemoryMetadataStore()
    coord = InMemoryGraphCoordinator()
    g = StateGraph("pipeline")
    g.add_node("ingest", lambda s: {"stage": "ingested"})
    g.add_node("finish", lambda s: {"stage": "finished"})
    g.add_edge("ingest", "finish")
    g.add_edge("finish", END)

    worker_a = DistributedCheckpointer(store, coordinator=coord, owner="A")
    compiled_a = g.compile(checkpointer=worker_a, interrupt_before=["finish"])
    r1 = await compiled_a.ainvoke({}, thread_id="t1")
    assert r1.status == "interrupted"
    assert r1.state["stage"] == "ingested"

    # A released its lease on pause; a *different* worker B resumes the thread.
    worker_b = DistributedCheckpointer(store, coordinator=coord, owner="B")
    compiled_b = g.compile(checkpointer=worker_b, interrupt_before=["finish"])
    r2 = await compiled_b.aresume("t1")
    assert r2.status == "done"
    assert r2.state["stage"] == "finished"
    # The ledger is one continuous, version-monotonic history across both workers.
    versions = [c.version for c in worker_b.history("t1")]
    assert versions == sorted(versions)
    owners = {c.lease_owner for c in worker_b.history("t1")}
    assert owners == {"A", "B"}


# ---------------------------------------------------------------------------
# BSP parallel super-steps + Send map-reduce
# ---------------------------------------------------------------------------


async def test_bsp_parallel_superstep_runs_branches_concurrently():
    barrier = asyncio.Barrier(2)  # only releases if both branches run at once
    g = StateGraph("bsp", reducers={"acc": lambda a, b: (a or []) + b})
    g.add_node("start", lambda s: {"acc": []})

    async def branch_a(s):
        await barrier.wait()
        return {"acc": ["a"]}

    async def branch_b(s):
        await barrier.wait()
        return {"acc": ["b"]}

    g.add_node("a", branch_a)
    g.add_node("b", branch_b)
    g.add_node("join", lambda s: {"joined": True})
    g.add_edge("start", "a")
    g.add_edge("start", "b")
    g.add_edge("a", "join")
    g.add_edge("b", "join")
    g.add_edge("join", END)

    compiled = g.compile(parallel=True)
    result = await asyncio.wait_for(compiled.ainvoke({}), timeout=2.0)
    assert result.status == "done"
    assert sorted(result.state["acc"]) == ["a", "b"]
    assert result.state["joined"] is True


async def test_send_map_reduce_fans_out():
    g = StateGraph("mapreduce", reducers={"results": lambda a, b: (a or []) + b})
    g.add_node("start", lambda s: {"results": []})
    g.add_node("dispatch", lambda s: [Send("worker", {"x": v}) for v in s["items"]])
    g.add_node("worker", lambda s: {"results": [s["x"] * 2]})
    g.add_node("reduce", lambda s: {"total": sum(s["results"])})
    g.set_entry("start")
    g.add_edge("start", "dispatch")
    g.add_edge("dispatch", "reduce")
    g.add_edge("reduce", END)

    result = await g.compile().ainvoke({"items": [1, 2, 3]})
    assert result.status == "done"
    assert sorted(result.state["results"]) == [2, 4, 6]
    assert result.state["total"] == 12


async def test_send_fanout_rejects_interrupt_inside_target():
    from vincio.agents import interrupt
    from vincio.core.errors import GraphError

    g = StateGraph("bad")
    g.add_node("dispatch", lambda s: [Send("worker", {})])
    g.add_node("worker", lambda s: interrupt(s, "nope"))
    g.add_edge("dispatch", END)
    with pytest.raises(GraphError):
        await g.compile().ainvoke({})


# ---------------------------------------------------------------------------
# Worker-pool fan-out
# ---------------------------------------------------------------------------


async def test_worker_pool_fans_out_batch():
    backend = WorkerPoolBackend(workers=3)
    inputs = [{"n": i} for i in range(8)]
    results = await backend.run_batch(_inc_graph(), inputs)
    assert [r.state["n"] for r in results] == [i + 1 for i in range(8)]
    assert all(r.status == "done" for r in results)


async def test_worker_pool_single_run():
    backend = WorkerPoolBackend(workers=2)
    result = await backend.run(_inc_graph(), {"n": 41})
    assert result.state["n"] == 42


# ---------------------------------------------------------------------------
# Ray / Temporal export adapters (injected fakes, offline)
# ---------------------------------------------------------------------------


class _FakeRay:
    class _Task:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    def remote(self, fn):
        return _FakeRay._Task(fn)

    def get(self, ref):
        return ref


async def test_ray_backend_runs_graph():
    backend = RayBackend(module=_FakeRay())
    result = await backend.run(_inc_graph(), {"n": 1})
    assert result.state["n"] == 2
    assert isinstance(backend.export(_inc_graph()), StateGraph)


class _FakeTemporalClient:
    def __init__(self):
        self.calls = []

    async def execute_workflow(self, name, arg, *, id, task_queue):
        self.calls.append((name, id, task_queue))
        graph = arg["graph"]
        return await graph.compile().ainvoke(arg["input"])


async def test_temporal_backend_runs_graph():
    client = _FakeTemporalClient()
    backend = TemporalBackend(client=client, task_queue="q1")
    result = await backend.run(_inc_graph(), {"n": 1}, thread_id="wf-1")
    assert result.state["n"] == 2
    assert client.calls == [("VincioGraphWorkflow", "wf-1", "q1")]


def test_temporal_backend_requires_client():
    from vincio.core.errors import ConfigError

    backend = TemporalBackend()
    with pytest.raises(ConfigError):
        asyncio.run(backend.run(_inc_graph(), {"n": 1}))


# ---------------------------------------------------------------------------
# Workflow map-reduce fan-out
# ---------------------------------------------------------------------------


async def test_workflow_map_step_fans_out():
    wf = Workflow("mapreduce")
    wf.step("seed", lambda input: [1, 2, 3, 4])
    wf.map_step("double", lambda x: x * 2, over="seed")
    wf.step("total", lambda double: sum(double), depends_on=["double"])
    result = await wf.arun()
    assert result.status == "succeeded"
    assert result.context.output_of("double") == [2, 4, 6, 8]
    assert result.context.output_of("total") == 20


async def test_workflow_map_step_over_callable():
    wf = Workflow("mr2")
    wf.step("seed", lambda input: {"ids": [10, 20]})
    wf.map_step(
        "fetch",
        lambda i: i + 1,
        over=lambda ctx: ctx.output_of("seed")["ids"],
        depends_on=["seed"],
    )
    result = await wf.arun()
    assert result.context.output_of("fetch") == [11, 21]


def test_compiled_graph_preserves_parallel_flag():
    compiled = _inc_graph().compile(parallel=True, fanout_limit=4)
    assert isinstance(compiled, CompiledGraph)
    assert compiled.parallel is True and compiled.fanout_limit == 4


# ---------------------------------------------------------------------------
# Cross-backend conformance: every backend reproduces the native engine
# ---------------------------------------------------------------------------


def _make_backend(label):
    """Each named distributed backend, wired to its offline driver/fake."""
    return {
        "worker_pool": lambda: WorkerPoolBackend(workers=3),
        "ray": lambda: RayBackend(module=_FakeRay()),
        "temporal": lambda: TemporalBackend(client=_FakeTemporalClient()),
    }[label]()


@pytest.mark.parametrize("label", ["worker_pool", "ray", "temporal"])
async def test_backend_conformance(label):
    # The reference is the native engine; the battery includes the no-seed
    # map-reduce, so this also proves the channel default survives export.
    await assert_backend_conformance(_make_backend(label), label=label)


async def test_backend_conformance_flags_a_divergent_backend():
    class _BrokenBackend:
        name = "broken"

        async def run(self, graph, payload, **kwargs):
            return await graph.compile().ainvoke({})  # ignores the input -> wrong state

    with pytest.raises(AssertionError, match="diverged from the native engine"):
        await assert_backend_conformance(_BrokenBackend())


# ---------------------------------------------------------------------------
# Channel-default reducer: map-reduce without an upstream seed node (2.1.1)
# ---------------------------------------------------------------------------


async def test_send_map_reduce_without_seed_node():
    # operator.add is non-defensive; the declared default is what makes the
    # first write fold correctly — no "start" node seeding ``results``.
    g = StateGraph("mr", reducers={"results": operator.add}, defaults={"results": list})
    g.add_node("dispatch", lambda s: [Send("worker", {"x": v}) for v in s["items"]])
    g.add_node("worker", lambda s: {"results": [s["x"] * 2]})
    g.add_node("reduce", lambda s: {"total": sum(s["results"])})
    g.set_entry("dispatch")
    g.add_edge("dispatch", "reduce")
    g.add_edge("reduce", END)

    result = await g.compile().ainvoke({"items": [1, 2, 3]})
    assert sorted(result.state["results"]) == [2, 4, 6]
    assert result.state["total"] == 12


async def test_channel_default_inferred_from_state_schema():
    class S(BaseModel):
        log: list = Field(default_factory=list)

    g = StateGraph("schema_default", state_schema=S, reducers={"log": operator.add})
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_edge("a", "b")
    g.add_edge("b", END)
    result = await g.compile().ainvoke({})
    assert result.state["log"] == ["a", "b"]  # reducer applied from the first write


def test_no_default_keeps_legacy_first_write_passthrough():
    g = StateGraph("legacy", reducers={"items": operator.add})
    assert g.reducer_default("items").__class__.__name__ == "_Missing"
    # the bare operator.add reducer still passes the first raw value through
    g.add_node("seed", lambda s: {"items": ["seed"]})
    g.add_node("more", lambda s: {"items": ["more"]})
    g.add_edge("seed", "more")
    g.add_edge("more", END)
    result = asyncio.run(g.compile().ainvoke({}))
    assert result.state["items"] == ["seed", "more"]
