"""Declarative composition (agents/compose).

A small, typed composition API so chains and graphs read like data: wrap any
step in :class:`Composable` (or call :func:`compose`) and pipe with ``|``.
Steps can be plain functions (sync or async), :class:`AgentExecutor`s, crews,
workflows, or compiled graphs — results are normalized automatically
(``AgentState`` → final answer, ``WorkflowResult``/``CrewResult`` → output,
``GraphResult`` → state).

Every node emits a ``compose_node`` span and a streaming :class:`NodeEvent`,
so a composed pipeline is observable node by node::

    pipeline = compose(fetch) | summarize | app.agent(planner="direct")
    result = await pipeline.acall("Q3 refunds")
    async for event in pipeline.astream("Q3 refunds"):
        print(event.type, event.node)

Combinators: :func:`parallel` fans one input out to named branches and
returns a dict of their results; :func:`branch` routes the input to one of
several steps by a router function.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ..core.concurrency import gather_bounded
from ..core.errors import AgentEngineError
from ..observability.traces import Tracer
from ..providers.base import run_sync

__all__ = ["NodeEvent", "Composable", "compose", "parallel", "branch"]

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class NodeEvent(BaseModel):
    type: str  # node_start | node_end | error | done
    node: str = ""
    index: int = 0
    value: Any = None
    error: str | None = None
    duration_ms: int = 0


def _normalize(result: Any) -> Any:
    """Unwrap engine results into plain values for the next step."""
    from ..workflows.engine import WorkflowResult
    from .crew import CrewResult
    from .graph import GraphResult
    from .state import AgentState

    if isinstance(result, AgentState):
        return result.final_answer if result.final_answer is not None else result.raw_answer_text
    if isinstance(result, (WorkflowResult, CrewResult)):
        return result.output
    if isinstance(result, GraphResult):
        return result.state
    return result


def _wrap(step: Any) -> tuple[str, Callable[[Any], Any]]:
    """Return (name, async runner) for any supported step type."""
    from .crew import Crew
    from .executor import AgentExecutor
    from .graph import CompiledGraph, StateGraph

    if isinstance(step, Composable):
        return step.name, step.acall
    if isinstance(step, StateGraph):
        step = step.compile()
    if isinstance(step, CompiledGraph):
        graph = step

        async def run_graph(value: Any) -> Any:
            payload = value if isinstance(value, dict) else {"input": value}
            return _normalize(await graph.ainvoke(payload))

        return graph.graph.name, run_graph
    if isinstance(step, (AgentExecutor, Crew)):
        runner = step

        async def run_engine(value: Any) -> Any:
            return _normalize(await runner.run(str(value))) if isinstance(
                runner, AgentExecutor
            ) else _normalize(await runner.arun(str(value)))

        name = getattr(runner, "name", None) or type(runner).__name__.lower()
        return str(name), run_engine
    if hasattr(step, "arun"):  # Workflow, app handles, custom engines
        engine = step
        # App agent handles wrap an AgentExecutor and take a string objective.
        takes_objective = isinstance(getattr(engine, "_executor", None), AgentExecutor)

        async def run_arun(value: Any) -> Any:
            return _normalize(await engine.arun(str(value) if takes_objective else value))

        return str(getattr(engine, "name", None) or type(engine).__name__.lower()), run_arun
    if callable(step):
        fn = step

        async def run_callable(value: Any) -> Any:
            out = fn(value)
            return _normalize(await out if inspect.isawaitable(out) else out)

        return getattr(fn, "__name__", None) or type(fn).__name__, run_callable
    raise AgentEngineError(f"cannot compose object of type {type(step).__name__}")


class Composable(Generic[TIn, TOut]):
    """A pipeline of named steps; build with ``compose()`` and the ``|`` operator."""

    def __init__(self, *steps: Any, name: str | None = None, tracer: Tracer | None = None) -> None:
        self._nodes: list[tuple[str, Callable[[Any], Any]]] = [_wrap(s) for s in steps]
        self.name = name or "pipeline"
        self.tracer = tracer or Tracer()

    @property
    def nodes(self) -> list[str]:
        return [name for name, _ in self._nodes]

    def __or__(self, other: Any) -> Composable:
        merged = Composable(name=self.name, tracer=self.tracer)
        merged._nodes = list(self._nodes)
        if isinstance(other, Composable):
            merged._nodes.extend(other._nodes)
        else:
            merged._nodes.append(_wrap(other))
        return merged

    def __ror__(self, other: Any) -> Composable:
        merged = Composable(name=self.name, tracer=self.tracer)
        merged._nodes = [_wrap(other), *self._nodes]
        return merged

    async def acall(self, value: TIn = None) -> TOut:  # type: ignore[assignment]
        result: Any = value
        async for event in self.astream(result):
            if event.type == "error":
                raise AgentEngineError(f"compose node {event.node!r} failed: {event.error}")
            if event.type == "done":
                result = event.value
        return result

    def call(self, value: TIn = None) -> TOut:  # type: ignore[assignment]
        return run_sync(self.acall(value))

    async def astream(self, value: Any = None) -> AsyncIterator[NodeEvent]:
        """Stream a node_start/node_end pair per node, then ``done``."""
        if not self._nodes:
            raise AgentEngineError(f"pipeline {self.name!r} has no steps")
        for index, (name, runner) in enumerate(self._nodes):
            yield NodeEvent(type="node_start", node=name, index=index)
            started = time.monotonic()
            error: str | None = None
            # The span closes before any yield: abandoning the generator must
            # not finalize an open span in a foreign context.
            with self.tracer.span(name, type="compose_node") as span:
                span.set(pipeline=self.name, index=index)
                try:
                    value = await runner(value)
                except Exception as exc:  # noqa: BLE001 - surfaced as an event
                    error = f"{type(exc).__name__}: {exc}"
                    span.set(status="error")
            duration_ms = int((time.monotonic() - started) * 1000)
            if error is not None:
                yield NodeEvent(type="error", node=name, index=index, error=error, duration_ms=duration_ms)
                return
            yield NodeEvent(type="node_end", node=name, index=index, value=value, duration_ms=duration_ms)
        yield NodeEvent(type="done", node=self.name, index=len(self._nodes), value=value)


def compose(*steps: Any, name: str | None = None, tracer: Tracer | None = None) -> Composable:
    """Compose steps left to right: ``compose(a, b) == compose(a) | b``."""
    return Composable(*steps, name=name, tracer=tracer)


def parallel(
    branches: dict[str, Any] | None = None,
    *,
    name: str = "parallel",
    tracer: Tracer | None = None,
    limit: int = 8,
    **named: Any,
) -> Composable:
    """Fan the input out to named branches concurrently; returns a dict of results."""
    routes = {**(branches or {}), **named}
    if not routes:
        raise AgentEngineError("parallel() needs at least one branch")
    wrapped = {key: _wrap(step)[1] for key, step in routes.items()}

    async def fan_out(value: Any) -> dict[str, Any]:
        results = await gather_bounded(
            [runner(value) for runner in wrapped.values()], limit=limit
        )
        return dict(zip(wrapped.keys(), results, strict=True))

    fan_out.__name__ = name
    return Composable(fan_out, name=name, tracer=tracer)


def branch(
    router: Callable[[Any], str],
    routes: dict[str, Any],
    *,
    default: str | None = None,
    name: str = "branch",
    tracer: Tracer | None = None,
) -> Composable:
    """Route the input to one step chosen by ``router(value)``."""
    wrapped = {key: _wrap(step)[1] for key, step in routes.items()}

    async def route(value: Any) -> Any:
        choice = router(value)
        if choice not in wrapped:
            if default is None:
                raise AgentEngineError(f"router chose unknown branch {choice!r}")
            choice = default
        return await wrapped[choice](value)

    route.__name__ = name
    return Composable(route, name=name, tracer=tracer)
