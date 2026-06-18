"""Durable stateful graphs (agents/graph).

A :class:`StateGraph` is a typed, checkpointed execution graph: nodes are
functions (sync or async) over a shared state dict, edges (static or
conditional) decide what runs next, and a :class:`Checkpointer` persists a
checkpoint after every super-step on the existing storage layer (in-memory,
SQLite, or Postgres via the URL factory). That makes runs durable:

- **resume** — continue an interrupted thread from its latest checkpoint;
- **time-travel** — fork any historical checkpoint into a new thread and
  re-execute deterministically from that step;
- **human-in-the-loop** — pause before/after named nodes
  (``interrupt_before`` / ``interrupt_after``) or dynamically from inside a
  node (:func:`interrupt`), then edit state (``update_state``) and resume.

Execution is bounded (``max_steps``) and every node emits a ``graph_node``
span, so durable graphs get the same trace/eval treatment as the rest of
Vincio.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.concurrency import gather_bounded
from ..core.errors import GraphError
from ..core.utils import new_id, utcnow
from ..observability.traces import Tracer
from ..providers.base import run_sync
from ..storage.base import MetadataStore

__all__ = [
    "START",
    "END",
    "GraphInterrupt",
    "interrupt",
    "Send",
    "Checkpoint",
    "Checkpointer",
    "GraphEvent",
    "GraphResult",
    "StateGraph",
    "CompiledGraph",
]

START = "__start__"
END = "__end__"

NodeFn = Callable[[dict[str, Any]], Any]
RouterFn = Callable[[dict[str, Any]], str]
Reducer = Callable[[Any, Any], Any]

_CHECKPOINT_KIND = "graph_checkpoints"
_RESUME_KEY = "__resume__"


class _Missing:
    """Sentinel: a reduced key has no declared empty, so the legacy first-write
    passthrough applies. Distinct from ``None`` (a legitimate channel default)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<no-default>"


_MISSING = _Missing()


class GraphInterrupt(Exception):
    """Control-flow signal raised by :func:`interrupt`; not an error."""

    def __init__(self, payload: Any = None) -> None:
        super().__init__("graph interrupted")
        self.payload = payload


def interrupt(state: dict[str, Any], payload: Any = None) -> Any:
    """Pause the graph from inside a node and surface ``payload`` to the caller.

    On resume with a value (``graph.resume(thread_id, value=...)``), the node
    re-runs and this call returns that value instead of pausing.
    """
    if _RESUME_KEY in state:
        return state.pop(_RESUME_KEY)
    raise GraphInterrupt(payload)


class Send(BaseModel):
    """Dynamic fan-out instruction for map-reduce super-steps (2.1).

    A node returns a list of :class:`Send` to spawn ``node`` once per item with
    its own ``state`` overlay; the spawned instances run concurrently (bounded)
    within the dispatching super-step and their updates reduce back into the
    shared state in deterministic order. Pair an additive reducer on the
    collected key (e.g. ``operator.add`` over a list) with a downstream reduce
    node to complete the map-reduce. Declare the channel's empty value once via
    ``StateGraph(..., defaults={"collected": list})`` (or a ``state_schema``
    field default) and the reducer is applied from the very first write — no
    need to seed the key with an upstream node. Mirrors LangGraph's ``Send`` so a
    graph that fans out runs the same on the native engine and a distributed
    backend.
    """

    node: str
    state: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, node: str, state: dict[str, Any] | None = None, **data: Any) -> None:
        super().__init__(node=node, state=state or {}, **data)


class Checkpoint(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ckpt"))
    thread_id: str
    graph: str = ""
    step: int = 0
    # Optimistic-concurrency token (2.1): the head checkpoint's version is the
    # value a distributed worker must compare-and-swap against to commit the
    # next super-step. ``0`` for the single-process path, which never CASes.
    version: int = 0
    state: dict[str, Any] = Field(default_factory=dict)
    next_nodes: list[str] = Field(default_factory=list)
    status: Literal["running", "interrupted", "done", "max_steps"] = "running"
    interrupt_payload: Any = None
    parent_id: str | None = None
    # Distributed lease holder that wrote this checkpoint (2.1), for audit/debug.
    lease_owner: str | None = None
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())


class Checkpointer:
    """Persists checkpoints on any :class:`MetadataStore` (memory/SQLite/Postgres)."""

    def __init__(self, store: MetadataStore | None = None) -> None:
        if store is None:
            from ..storage.base import InMemoryMetadataStore

            store = InMemoryMetadataStore()
        self.store = store

    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        self.store.save(_CHECKPOINT_KIND, checkpoint.model_dump(mode="json"))
        return checkpoint

    def on_thread_start(self, thread_id: str) -> None:
        """Hook called once when a thread begins/resumes execution.

        A no-op for the single-process checkpointer; the distributed
        checkpointer (2.1) overrides it to acquire the thread's lease.
        """

    def on_thread_end(self, thread_id: str) -> None:
        """Hook called once when a thread reaches a terminal state or pauses.

        A no-op here; the distributed checkpointer releases the lease.
        """

    def get(self, checkpoint_id: str) -> Checkpoint | None:
        record = self.store.get(_CHECKPOINT_KIND, checkpoint_id)
        return Checkpoint.model_validate(record) if record else None

    def history(self, thread_id: str) -> list[Checkpoint]:
        """All checkpoints of a thread, oldest first."""
        records = self.store.query(
            _CHECKPOINT_KIND, where={"thread_id": thread_id}, limit=10_000
        )
        checkpoints = [Checkpoint.model_validate(r) for r in records]
        return sorted(checkpoints, key=lambda c: (c.step, c.created_at))

    def latest(self, thread_id: str) -> Checkpoint | None:
        history = self.history(thread_id)
        return history[-1] if history else None


class GraphEvent(BaseModel):
    type: Literal["node_start", "node_end", "checkpoint", "interrupt", "done"]
    node: str = ""
    step: int = 0
    thread_id: str = ""
    payload: Any = None


class GraphResult(BaseModel):
    thread_id: str
    status: Literal["done", "interrupted", "max_steps"]
    state: dict[str, Any] = Field(default_factory=dict)
    steps: int = 0
    next_nodes: list[str] = Field(default_factory=list)
    interrupt_payload: Any = None
    checkpoint_id: str | None = None


class StateGraph:
    """Build-time graph definition; ``compile()`` produces the runnable form."""

    def __init__(
        self,
        name: str = "graph",
        *,
        state_schema: type[BaseModel] | None = None,
        reducers: dict[str, Reducer] | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.state_schema = state_schema
        self.reducers = reducers or {}
        # Per-channel empty value: a literal or a zero-arg factory (``list``,
        # ``dict``, ...). When a reduced key is absent on its first write, the
        # reducer is applied against this default instead of passing the raw
        # value through — so a map-reduce needs no upstream seed node. Defaults
        # are also inferred from ``state_schema`` field defaults (see
        # :meth:`reducer_default`); the explicit map wins.
        self.defaults = defaults or {}
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, list[str]] = {}
        self.routers: dict[str, tuple[RouterFn, dict[str, str] | None]] = {}
        self.entry: str | None = None
        # compile-time defaults, bound by app.graph()
        self.default_tracer: Tracer | None = None
        self.default_checkpointer: Checkpointer | None = None

    def add_node(self, name: str, fn: NodeFn) -> StateGraph:
        if name in (START, END):
            raise GraphError(f"{name!r} is reserved")
        if name in self.nodes:
            raise GraphError(f"duplicate node {name!r}")
        self.nodes[name] = fn
        if self.entry is None:
            self.entry = name
        return self

    def add_edge(self, source: str, target: str) -> StateGraph:
        if source == START:
            return self.set_entry(target)
        self._check_known(source, target)
        self.edges.setdefault(source, []).append(target)
        return self

    def add_conditional_edge(
        self, source: str, router: RouterFn, targets: dict[str, str] | None = None
    ) -> StateGraph:
        """Route from ``source`` by ``router(state)``; ``targets`` optionally
        maps router outputs to node names (identity mapping when omitted)."""
        self._check_known(source)
        for target in (targets or {}).values():
            self._check_known(target)
        self.routers[source] = (router, targets)
        return self

    def set_entry(self, name: str) -> StateGraph:
        self._check_known(name)
        self.entry = name
        return self

    def _check_known(self, *names: str) -> None:
        for name in names:
            if name != END and name not in self.nodes:
                raise GraphError(f"unknown node {name!r}")

    def reducer_default(self, key: str) -> Any:
        """The empty value a reducer folds the first write into, or ``_MISSING``.

        Resolution order: an explicit entry in ``defaults`` (a zero-arg factory
        is called) wins; otherwise a non-required ``state_schema`` field supplies
        its default. ``_MISSING`` means no default is known, so the legacy
        first-write passthrough applies and the raw value is stored as-is.
        """
        if key in self.defaults:
            factory = self.defaults[key]
            return factory() if callable(factory) else factory
        schema = self.state_schema
        if schema is not None:
            field = schema.model_fields.get(key)
            if field is not None and not field.is_required():
                return field.get_default(call_default_factory=True)
        return _MISSING

    def compile(
        self,
        *,
        checkpointer: Checkpointer | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        tracer: Tracer | None = None,
        max_steps: int = 64,
        parallel: bool = False,
        fanout_limit: int = 8,
    ) -> CompiledGraph:
        if not self.nodes:
            raise GraphError(f"graph {self.name!r} has no nodes")
        self._check_known(*(interrupt_before or []), *(interrupt_after or []))
        return CompiledGraph(
            self,
            checkpointer=checkpointer or self.default_checkpointer or Checkpointer(),
            interrupt_before=interrupt_before or [],
            interrupt_after=interrupt_after or [],
            tracer=tracer or self.default_tracer or Tracer(),
            max_steps=max_steps,
            parallel=parallel,
            fanout_limit=fanout_limit,
        )


class CompiledGraph:
    def __init__(
        self,
        graph: StateGraph,
        *,
        checkpointer: Checkpointer,
        interrupt_before: list[str],
        interrupt_after: list[str],
        tracer: Tracer,
        max_steps: int,
        parallel: bool = False,
        fanout_limit: int = 8,
    ) -> None:
        self.graph = graph
        self.checkpointer = checkpointer
        self.interrupt_before = set(interrupt_before)
        self.interrupt_after = set(interrupt_after)
        self.tracer = tracer
        self.max_steps = max_steps
        self.parallel = parallel
        self.fanout_limit = fanout_limit

    # -- state handling --------------------------------------------------------

    def _merge(self, state: dict[str, Any], updates: Any) -> dict[str, Any]:
        if updates is None:
            return state
        if not isinstance(updates, dict):
            raise GraphError(
                f"node returned {type(updates).__name__}; nodes must return a dict of state updates"
            )
        for key, value in updates.items():
            reducer = self.graph.reducers.get(key)
            if reducer is None:
                state[key] = value
            elif key in state:
                state[key] = reducer(state[key], value)
            else:
                # First write to a reduced key: fold into the channel's declared
                # empty so the reducer runs from the start (no seed node needed).
                # When no default is known, keep the legacy passthrough.
                default = self.graph.reducer_default(key)
                state[key] = value if default is _MISSING else reducer(default, value)
        if self.graph.state_schema is not None:
            try:
                self.graph.state_schema.model_validate(state)
            except Exception as exc:
                raise GraphError(f"state failed schema validation: {exc}") from exc
        return state

    async def _run_node(self, name: str, state: dict[str, Any]) -> Any:
        fn = self.graph.nodes[name]
        if inspect.iscoroutinefunction(fn):
            return await fn(state)
        return await asyncio.get_running_loop().run_in_executor(None, lambda: fn(state))

    async def _execute_node(
        self, name: str, state: dict[str, Any], step: int
    ) -> tuple[list[Any], list[str]]:
        """Run node ``name`` against ``state``, returning ``(updates_list, spawned)``.

        Opens the node's ``graph_node`` span (so the call works the same in the
        sequential and parallel super-step paths). A node that returns a list of
        :class:`Send` fans out: each target runs concurrently (bounded by
        ``fanout_limit``) over a copy of the state with the send's overlay, each
        in its own span, and their update dicts are returned in send order so the
        caller merges them deterministically (reducers collect the map results).
        A normal node returns a single-element list. :class:`GraphInterrupt`
        propagates; interrupting from inside a fan-out target is unsupported.
        """
        with self.tracer.span(name, type="graph_node") as span:
            span.set(graph=self.graph.name, step=step)
            try:
                result = await self._run_node(name, state)
            except GraphInterrupt:
                span.set(status="interrupted")
                raise
            is_sends = (
                isinstance(result, list) and bool(result) and all(isinstance(s, Send) for s in result)
            )
            if is_sends:
                span.set(fanned_out=len(result))
        if not is_sends:
            return [result], []
        sends: list[Send] = result

        async def _one(send: Send) -> Any:
            with self.tracer.span(send.node, type="graph_node") as span:
                span.set(graph=self.graph.name, step=step, spawned_by=name)
                try:
                    return await self._run_node(send.node, {**state, **send.state})
                except GraphInterrupt as exc:
                    raise GraphError(
                        f"node {send.node!r} called interrupt() inside a Send fan-out; "
                        "interrupts are only supported in top-level super-step nodes"
                    ) from exc

        outs = await gather_bounded([_one(s) for s in sends], limit=self.fanout_limit)
        return list(outs), [s.node for s in sends]

    def _next_frontier(self, ran: list[str], state: dict[str, Any]) -> list[str]:
        targets: list[str] = []
        for name in ran:
            if name in self.graph.routers:
                router, mapping = self.graph.routers[name]
                choice = router(state)
                choice = (mapping or {}).get(choice, choice)
                if choice != END and choice not in self.graph.nodes:
                    raise GraphError(f"router at {name!r} returned unknown node {choice!r}")
                targets.append(choice)
            else:
                targets.extend(self.graph.edges.get(name, []) or [END])
        # Deterministic, deduplicated; END only survives if nothing else runs.
        seen: set[str] = set()
        frontier: list[str] = []
        for target in targets:
            if target != END and target not in seen:
                seen.add(target)
                frontier.append(target)
        return frontier or [END]

    def _checkpoint(
        self,
        thread_id: str,
        step: int,
        state: dict[str, Any],
        next_nodes: list[str],
        *,
        status: Literal["running", "interrupted", "done", "max_steps"] = "running",
        interrupt_payload: Any = None,
        parent_id: str | None = None,
    ) -> Checkpoint:
        return self.checkpointer.save(
            Checkpoint(
                thread_id=thread_id,
                graph=self.graph.name,
                step=step,
                state=dict(state),
                next_nodes=list(next_nodes),
                status=status,
                interrupt_payload=interrupt_payload,
                parent_id=parent_id,
            )
        )

    # -- execution -------------------------------------------------------------

    async def _events(
        self,
        state: dict[str, Any],
        frontier: list[str],
        step: int,
        thread_id: str,
        *,
        parent_checkpoint: str | None,
        skip_interrupt_check: bool,
    ) -> AsyncIterator[GraphEvent]:
        parent_id = parent_checkpoint
        self.checkpointer.on_thread_start(thread_id)
        try:
            async for event in self._run_loop(
                state, frontier, step, thread_id, parent_id=parent_id,
                skip_interrupt_check=skip_interrupt_check,
            ):
                yield event
        finally:
            self.checkpointer.on_thread_end(thread_id)

    async def _run_loop(
        self,
        state: dict[str, Any],
        frontier: list[str],
        step: int,
        thread_id: str,
        *,
        parent_id: str | None,
        skip_interrupt_check: bool,
    ) -> AsyncIterator[GraphEvent]:
        with self.tracer.span(self.graph.name, type="custom") as run_span:
            run_span.set(graph=self.graph.name, thread_id=thread_id)
            while frontier != [END]:
                if step >= self.max_steps:
                    ckpt = self._checkpoint(
                        thread_id, step, state, frontier, status="max_steps", parent_id=parent_id
                    )
                    result = GraphResult(
                        thread_id=thread_id, status="max_steps", state=state,
                        steps=step, next_nodes=frontier, checkpoint_id=ckpt.id,
                    )
                    run_span.set(status="max_steps", steps=step)
                    yield GraphEvent(type="done", step=step, thread_id=thread_id, payload=result)
                    return
                pause_before = [n for n in frontier if n in self.interrupt_before]
                if pause_before and not skip_interrupt_check:
                    ckpt = self._checkpoint(
                        thread_id, step, state, frontier, status="interrupted", parent_id=parent_id
                    )
                    result = GraphResult(
                        thread_id=thread_id, status="interrupted", state=state, steps=step,
                        next_nodes=frontier, checkpoint_id=ckpt.id,
                    )
                    run_span.set(status="interrupted", steps=step)
                    yield GraphEvent(
                        type="interrupt", node=pause_before[0], step=step,
                        thread_id=thread_id, payload=result,
                    )
                    return
                skip_interrupt_check = False
                ran: list[str] = []
                interrupted: GraphInterrupt | None = None
                interrupted_node = ""
                if self.parallel and len(frontier) > 1:
                    # True BSP super-step: every frontier node reads the same
                    # pre-step state and runs concurrently; writes are applied at
                    # the barrier in deterministic frontier order. If any node
                    # interrupts, the whole super-step re-runs on resume, so the
                    # partial writes of this step are discarded.
                    pre_state = dict(state)
                    for name in frontier:
                        yield GraphEvent(type="node_start", node=name, step=step, thread_id=thread_id)
                    outcomes = await gather_bounded(
                        [self._execute_node(name, dict(pre_state), step) for name in frontier],
                        limit=self.fanout_limit,
                        return_exceptions=True,
                    )
                    staged: list[tuple[str, list[Any]]] = []
                    for name, outcome in zip(frontier, outcomes, strict=True):
                        if isinstance(outcome, GraphInterrupt):
                            interrupted, interrupted_node = outcome, name
                            break
                        if isinstance(outcome, BaseException):
                            raise outcome
                        staged.append((name, outcome[0]))
                    if interrupted is None:
                        for name, updates_list in staged:
                            for updates in updates_list:
                                state = self._merge(state, updates)
                            ran.append(name)
                            yield GraphEvent(
                                type="node_end", node=name, step=step, thread_id=thread_id
                            )
                    else:
                        state = pre_state  # re-run the whole super-step on resume
                else:
                    for name in frontier:  # deterministic order; updates merged in sequence
                        yield GraphEvent(type="node_start", node=name, step=step, thread_id=thread_id)
                        try:
                            updates_list, _spawned = await self._execute_node(name, state, step)
                        except GraphInterrupt as signal:
                            interrupted, interrupted_node = signal, name
                            break
                        for updates in updates_list:
                            state = self._merge(state, updates)
                        ran.append(name)
                        yield GraphEvent(type="node_end", node=name, step=step, thread_id=thread_id)
                state.pop(_RESUME_KEY, None)  # consumed (or unused) once the step ran
                if interrupted is not None:
                    # Re-queue the interrupting node and the rest of the frontier,
                    # and carry the successors of siblings that already ran this
                    # super-step — otherwise their branches would be lost on resume.
                    carried = [
                        t for t in (self._next_frontier(ran, state) if ran else []) if t != END
                    ]
                    pending = [interrupted_node] + [
                        n for n in frontier if n not in ran and n != interrupted_node
                    ]
                    pending += [t for t in carried if t not in pending]
                    ckpt = self._checkpoint(
                        thread_id, step, state, pending, status="interrupted",
                        interrupt_payload=interrupted.payload, parent_id=parent_id,
                    )
                    result = GraphResult(
                        thread_id=thread_id, status="interrupted", state=state, steps=step,
                        next_nodes=pending, interrupt_payload=interrupted.payload,
                        checkpoint_id=ckpt.id,
                    )
                    run_span.set(status="interrupted", steps=step)
                    yield GraphEvent(
                        type="interrupt", node=interrupted_node, step=step,
                        thread_id=thread_id, payload=result,
                    )
                    return
                step += 1
                frontier = self._next_frontier(ran, state)
                ckpt = self._checkpoint(thread_id, step, state, frontier, parent_id=parent_id)
                parent_id = ckpt.id
                yield GraphEvent(type="checkpoint", step=step, thread_id=thread_id, payload=ckpt.id)
                pause_after = [n for n in ran if n in self.interrupt_after]
                if pause_after and frontier != [END]:
                    ckpt = self._checkpoint(
                        thread_id, step, state, frontier, status="interrupted", parent_id=parent_id
                    )
                    result = GraphResult(
                        thread_id=thread_id, status="interrupted", state=state, steps=step,
                        next_nodes=frontier, checkpoint_id=ckpt.id,
                    )
                    run_span.set(status="interrupted", steps=step)
                    yield GraphEvent(
                        type="interrupt", node=pause_after[0], step=step,
                        thread_id=thread_id, payload=result,
                    )
                    return
            ckpt = self._checkpoint(thread_id, step, state, [END], status="done", parent_id=parent_id)
            result = GraphResult(
                thread_id=thread_id, status="done", state=state, steps=step, checkpoint_id=ckpt.id
            )
            run_span.set(status="done", steps=step)
            yield GraphEvent(type="done", step=step, thread_id=thread_id, payload=result)

    def _start_args(
        self, input: dict[str, Any] | None, thread_id: str | None
    ) -> tuple[dict[str, Any], list[str], int, str, str | None, bool]:
        thread_id = thread_id or new_id("thread")
        latest = self.checkpointer.latest(thread_id)
        if latest is not None and latest.status == "done":
            # Re-running a finished thread would interleave checkpoints with
            # the completed history and break resume — branch instead.
            raise GraphError(
                f"thread {thread_id!r} already completed; fork() one of its "
                "checkpoints or use a new thread_id"
            )
        if latest is not None:  # running (crashed), interrupted, or max_steps
            state = dict(latest.state)
            state.update(input or {})
            return state, list(latest.next_nodes), latest.step, thread_id, latest.id, True
        if self.graph.entry is None:
            raise GraphError(f"graph {self.graph.name!r} has no entry node")
        return dict(input or {}), [self.graph.entry], 0, thread_id, None, False

    async def astream(
        self, input: dict[str, Any] | None = None, *, thread_id: str | None = None
    ) -> AsyncIterator[GraphEvent]:
        """Stream node/checkpoint/interrupt events; terminal event carries the result."""
        state, frontier, step, thread_id, parent, resumed = self._start_args(input, thread_id)
        async for event in self._events(
            state, frontier, step, thread_id,
            parent_checkpoint=parent, skip_interrupt_check=resumed,
        ):
            yield event

    async def ainvoke(
        self, input: dict[str, Any] | None = None, *, thread_id: str | None = None
    ) -> GraphResult:
        """Run to completion or to the next interrupt; resumes existing threads."""
        result: GraphResult | None = None
        async for event in self.astream(input, thread_id=thread_id):
            if event.type in ("done", "interrupt"):
                result = event.payload
        assert result is not None  # _events always ends with a terminal event
        return result

    def invoke(self, input: dict[str, Any] | None = None, *, thread_id: str | None = None) -> GraphResult:
        return run_sync(self.ainvoke(input, thread_id=thread_id))

    # -- durability: resume / edit / time-travel -----------------------------------

    async def aresume(self, thread_id: str, *, value: Any = None) -> GraphResult:
        """Continue an interrupted thread; ``value`` answers a node-level
        :func:`interrupt` (the paused node re-runs and receives it)."""
        latest = self.checkpointer.latest(thread_id)
        if latest is None:
            raise GraphError(f"no checkpoints for thread {thread_id!r}")
        if latest.status == "done":
            raise GraphError(f"thread {thread_id!r} already completed")
        input = {_RESUME_KEY: value} if value is not None else None
        return await self.ainvoke(input, thread_id=thread_id)

    def resume(self, thread_id: str, *, value: Any = None) -> GraphResult:
        return run_sync(self.aresume(thread_id, value=value))

    def update_state(self, thread_id: str, values: dict[str, Any]) -> Checkpoint:
        """Edit-and-resume: merge ``values`` into the latest checkpoint's state
        as a new checkpoint, then ``resume(thread_id)`` to continue."""
        latest = self.checkpointer.latest(thread_id)
        if latest is None:
            raise GraphError(f"no checkpoints for thread {thread_id!r}")
        state = self._merge(dict(latest.state), values)
        return self._checkpoint(
            thread_id, latest.step, state, latest.next_nodes,
            status=latest.status if latest.status != "done" else "interrupted",
            parent_id=latest.id,
        )

    def history(self, thread_id: str) -> list[Checkpoint]:
        return self.checkpointer.history(thread_id)

    def fork(self, checkpoint_id: str, *, thread_id: str | None = None) -> str:
        """Time-travel: branch a new thread from any historical checkpoint.
        ``resume(new_thread_id)`` then re-executes deterministically from that step."""
        checkpoint = self.checkpointer.get(checkpoint_id)
        if checkpoint is None:
            raise GraphError(f"unknown checkpoint {checkpoint_id!r}")
        new_thread = thread_id or new_id("thread")
        self._checkpoint(
            new_thread, checkpoint.step, dict(checkpoint.state), list(checkpoint.next_nodes),
            status="interrupted", parent_id=checkpoint.id,
        )
        return new_thread
