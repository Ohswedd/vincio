"""Runtime backends (agents/backends).

Adapters that export Vincio orchestration definitions to external runtimes,
so Vincio orchestrates **without lock-in**: the compiler layer stays
provider-neutral, and the same :class:`StateGraph` or :class:`Crew` can run
on the native engine, on LangGraph, or on the OpenAI Agents SDK.

Both adapters import their runtime lazily (``langgraph`` /
``openai-agents``) and accept an injected module for offline tests. Nothing
in core Vincio depends on either package.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..core.concurrency import gather_bounded
from ..core.errors import ConfigError
from ..core.utils import new_id
from .crew import Crew
from .distributed import DistributedCheckpointer, GraphCoordinator, InMemoryGraphCoordinator
from .graph import END, Checkpointer, CompiledGraph, GraphResult, StateGraph

__all__ = [
    "RuntimeBackend",
    "LangGraphBackend",
    "OpenAIAgentsBackend",
    "WorkerPoolBackend",
    "RayBackend",
    "TemporalBackend",
]


@runtime_checkable
class RuntimeBackend(Protocol):
    """A runtime that can execute an exported Vincio orchestration."""

    name: str

    async def run(self, exported: Any, input: Any, **kwargs: Any) -> Any: ...


def _import(module_name: str, *, hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ConfigError(
            f"{module_name!r} is not installed; install it with `pip install {hint}` "
            "or inject a compatible module"
        ) from exc


class LangGraphBackend:
    """Exports a Vincio :class:`StateGraph` to a LangGraph ``StateGraph``.

    Node functions transfer as-is (both engines use dict-in/dict-update-out
    nodes); edges, conditional edges, and the entry point are translated,
    with Vincio's ``END`` mapped to LangGraph's.
    """

    name = "langgraph"

    def __init__(self, module: Any | None = None) -> None:
        self._module = module

    @property
    def module(self) -> Any:
        if self._module is None:
            self._module = _import("langgraph.graph", hint="langgraph")
        return self._module

    def export(self, graph: StateGraph) -> Any:
        """Build (but don't compile) the equivalent LangGraph builder."""
        module = self.module
        builder = module.StateGraph(dict)
        for name, fn in graph.nodes.items():
            builder.add_node(name, fn)
        if graph.entry is None:
            raise ConfigError(f"graph {graph.name!r} has no entry node")
        builder.set_entry_point(graph.entry)
        for source, targets in graph.edges.items():
            if source in graph.routers:
                continue  # the router has exclusive precedence, as in the native engine
            for target in targets:
                builder.add_edge(source, module.END if target == END else target)
        for source, (router, mapping) in graph.routers.items():
            translated = {
                key: (module.END if target == END else target)
                for key, target in (mapping or {}).items()
            }

            def routed(state: dict, _router: Any = router, _map: dict = translated) -> Any:
                choice = _router(state)
                if choice in _map:
                    return _map[choice]
                return module.END if choice == END else choice

            builder.add_conditional_edges(source, routed)
        return builder

    def compile(self, graph: StateGraph, **kwargs: Any) -> Any:
        return self.export(graph).compile(**kwargs)

    async def run(self, exported: Any, input: Any, **kwargs: Any) -> Any:
        """Run a Vincio graph (or a pre-exported builder) on LangGraph."""
        if isinstance(exported, StateGraph):
            exported = self.compile(exported)
        elif hasattr(exported, "compile"):
            exported = exported.compile()
        return await exported.ainvoke(input, **kwargs)


class OpenAIAgentsBackend:
    """Exports Vincio agents and crews to OpenAI Agents SDK ``Agent`` objects.

    Crew members become agents; the crew becomes a triage/manager agent with
    handoffs to every member, mirroring Vincio's hierarchical process.
    """

    name = "openai_agents"

    def __init__(self, module: Any | None = None) -> None:
        self._module = module

    @property
    def module(self) -> Any:
        if self._module is None:
            self._module = _import("agents", hint="openai-agents")
        return self._module

    def export_agent(
        self,
        name: str,
        *,
        instructions: str = "",
        tools: list[Callable[..., Any]] | None = None,
        handoffs: list[Any] | None = None,
        model: str | None = None,
    ) -> Any:
        module = self.module
        wrapped_tools = [module.function_tool(fn) for fn in (tools or [])]
        kwargs: dict[str, Any] = {
            "name": name,
            "instructions": instructions,
            "tools": wrapped_tools,
            "handoffs": handoffs or [],
        }
        if model is not None:
            kwargs["model"] = model
        return module.Agent(**kwargs)

    def export_crew(self, crew: Crew, *, model: str | None = None) -> Any:
        members = []
        for member in crew._members.values():
            role = member.role
            instructions = "\n".join(
                part
                for part in (
                    role.description,
                    f"Your goal: {role.goal}" if role.goal else "",
                    getattr(member.executor, "system_prompt", ""),
                )
                if part
            )
            members.append(self.export_agent(role.name, instructions=instructions, model=model))
        roster = ", ".join(crew.names)
        return self.export_agent(
            f"{crew.name}_manager",
            instructions=(
                "You manage a team. Hand each task off to the member best suited "
                f"for it. Members: {roster}."
            ),
            handoffs=members,
            model=model,
        )

    async def run(self, exported: Any, input: Any, **kwargs: Any) -> Any:
        if isinstance(exported, Crew):
            exported = self.export_crew(exported)
        result = await self.module.Runner.run(exported, input, **kwargs)
        return getattr(result, "final_output", result)


def _graph_compile_kwargs(exported: Any) -> dict[str, Any]:
    """Preserve a pre-compiled graph's options when recompiling for a backend."""
    if isinstance(exported, CompiledGraph):
        return {
            "interrupt_before": sorted(exported.interrupt_before),
            "interrupt_after": sorted(exported.interrupt_after),
            "tracer": exported.tracer,
            "max_steps": exported.max_steps,
            "parallel": exported.parallel,
            "fanout_limit": exported.fanout_limit,
        }
    return {}


def _as_state_graph(exported: Any) -> StateGraph:
    graph = exported.graph if isinstance(exported, CompiledGraph) else exported
    if not isinstance(graph, StateGraph):
        raise ConfigError(
            f"expected a StateGraph or CompiledGraph, got {type(graph).__name__}"
        )
    return graph


class WorkerPoolBackend:
    """In-process reference distributed executor — lock-free, durable, fan-out.

    The worker-pool backend is the substantive distributed engine: it runs
    the **same** :class:`~vincio.agents.graph.StateGraph` across a pool of async
    workers that share one :class:`~vincio.agents.distributed.GraphCoordinator`
    and checkpoint store, so each graph thread is lease-guarded and every
    super-step is CAS-committed. Two workers can never double-execute a step
    (the loser raises :class:`~vincio.core.errors.CheckpointConflictError`), a
    crashed worker's thread is reclaimed once its lease expires and resumes from
    the last checkpoint, and many threads fan out across the pool. It needs no
    external service — the same coordinator interface is backed by Redis
    (:class:`~vincio.agents.distributed.RedisGraphCoordinator`) for a real
    multi-process deployment.
    """

    name = "worker_pool"

    def __init__(
        self,
        *,
        workers: int = 4,
        store: Any = None,
        coordinator: GraphCoordinator | None = None,
        lease_ttl_s: float = 30.0,
    ) -> None:
        if store is None:
            from ..storage.base import InMemoryMetadataStore

            store = InMemoryMetadataStore()
        self.workers = max(1, workers)
        self.store = store
        self.coordinator = coordinator or InMemoryGraphCoordinator()
        self.lease_ttl_s = lease_ttl_s

    def _checkpointer(self, owner: str) -> DistributedCheckpointer:
        return DistributedCheckpointer(
            self.store,
            coordinator=self.coordinator,
            owner=owner,
            lease_ttl_s=self.lease_ttl_s,
        )

    async def run(
        self, exported: Any, input: Any = None, *, thread_id: str | None = None, **kwargs: Any
    ) -> GraphResult:
        """Run one graph thread on a freshly-leased worker checkpointer."""
        graph = _as_state_graph(exported)
        compiled = graph.compile(
            checkpointer=self._checkpointer(new_id("worker")), **_graph_compile_kwargs(exported)
        )
        return await compiled.ainvoke(input, thread_id=thread_id)

    async def run_batch(
        self, exported: Any, inputs: list[Any], *, thread_ids: list[str] | None = None
    ) -> list[GraphResult]:
        """Fan a batch of inputs out across the worker pool, one thread each.

        Each worker holds its own lease-guarded checkpointer over the shared
        coordinator/store and pulls work off a queue, so the batch is executed
        with bounded, coordinated parallelism. Results are returned in input
        order.
        """
        graph = _as_state_graph(exported)
        compile_kwargs = _graph_compile_kwargs(exported)
        ids = thread_ids or [new_id("thread") for _ in inputs]
        if len(ids) != len(inputs):
            raise ConfigError("thread_ids must align 1:1 with inputs")
        results: list[GraphResult | None] = [None] * len(inputs)
        queue: asyncio.Queue[int] = asyncio.Queue()
        for idx in range(len(inputs)):
            queue.put_nowait(idx)

        async def worker() -> None:
            checkpointer = self._checkpointer(new_id("worker"))
            while True:
                try:
                    idx = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                compiled = graph.compile(checkpointer=checkpointer, **compile_kwargs)
                results[idx] = await compiled.ainvoke(inputs[idx], thread_id=ids[idx])

        await gather_bounded([worker() for _ in range(self.workers)], limit=self.workers)
        return [r for r in results if r is not None]


class RayBackend:
    """Run a Vincio graph on a Ray cluster — node bodies as Ray tasks.

    Each node's function is dispatched as a ``ray.remote`` task so the worker
    pool is Ray's, while the durable driver and checkpoint ledger stay local and
    lock-free — Vincio keeps provenance and the swap-gate discipline, Ray
    provides horizontal compute. Imports ``ray`` lazily and accepts an injected
    module for offline tests, exactly like the LangGraph/OpenAI-Agents adapters.
    """

    name = "ray"

    def __init__(self, module: Any | None = None) -> None:
        self._module = module

    @property
    def module(self) -> Any:
        if self._module is None:
            self._module = _import("ray", hint="ray")
        return self._module

    def export(self, graph: StateGraph) -> StateGraph:
        """Build an equivalent graph whose nodes run as Ray tasks."""
        ray = self.module
        wrapped = StateGraph(
            graph.name,
            state_schema=graph.state_schema,
            reducers=dict(graph.reducers),
            defaults=dict(graph.defaults),
        )
        for name, fn in graph.nodes.items():
            task = ray.remote(fn)

            async def node(state: dict[str, Any], _task: Any = task) -> Any:
                ref = _task.remote(state)
                get = ray.get
                result = await asyncio.get_running_loop().run_in_executor(None, lambda: get(ref))
                if inspect.iscoroutine(result):
                    result = await result
                return result

            wrapped.add_node(name, node)
        wrapped.entry = graph.entry
        wrapped.edges = {k: list(v) for k, v in graph.edges.items()}
        wrapped.routers = dict(graph.routers)
        return wrapped

    async def run(
        self, exported: Any, input: Any = None, *, checkpointer: Checkpointer | None = None,
        thread_id: str | None = None, **kwargs: Any,
    ) -> GraphResult:
        graph = _as_state_graph(exported)
        compiled = self.export(graph).compile(
            checkpointer=checkpointer or Checkpointer(), **_graph_compile_kwargs(exported)
        )
        return await compiled.ainvoke(input, thread_id=thread_id)


class TemporalBackend:
    """Run a Vincio graph as a Temporal workflow for cross-restart durability.

    A thin client adapter: it submits the graph to a Temporal worker through an
    injected/lazy ``temporalio`` client (mirroring the OpenAI-Agents ``Runner``
    adapter), so Temporal provides the durable-execution backbone while the
    graph definition, checkpoints, and trace remain Vincio's. Offline tests
    inject a client whose ``execute_workflow`` runs the graph in-process.
    """

    name = "temporal"

    def __init__(
        self, client: Any | None = None, *, task_queue: str = "vincio-graphs"
    ) -> None:
        self._client = client
        self.task_queue = task_queue

    def export(self, graph: StateGraph) -> dict[str, Any]:
        """Return the workflow descriptor a Temporal worker registers."""
        return {"workflow": "VincioGraphWorkflow", "graph": graph.name, "task_queue": self.task_queue}

    async def run(
        self, exported: Any, input: Any = None, *, client: Any | None = None,
        thread_id: str | None = None, **kwargs: Any,
    ) -> Any:
        client = client or self._client
        if client is None:
            raise ConfigError(
                "TemporalBackend.run needs a Temporal client; pass client=... "
                "(a temporalio.client.Client) or construct the backend with one"
            )
        graph = _as_state_graph(exported)
        return await client.execute_workflow(
            "VincioGraphWorkflow",
            {"graph": graph, "input": input},
            id=thread_id or new_id("temporal"),
            task_queue=self.task_queue,
        )
