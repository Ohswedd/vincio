"""Distributed-backend conformance harness (2.1.1).

The native durable engine is the reference semantics for every runtime backend.
A backend that runs a Vincio :class:`~vincio.agents.graph.StateGraph` elsewhere —
the in-process :class:`~vincio.agents.backends.WorkerPoolBackend`, the
:class:`~vincio.agents.backends.RayBackend` / :class:`~vincio.agents.backends.TemporalBackend`
export adapters, or a user's own wiring to a real cluster — must produce the
**same** final state as ``graph.compile().ainvoke(input)``.

:func:`assert_backend_conformance` runs a battery of canonical graphs through a
backend and asserts each result matches the native engine. The Ray/Temporal
adapters can only be exercised offline against injected fakes (a real cluster
can't run here), so this is the contract those fakes — and a real deployment —
are held to: it is the offline proof that the adapter preserves sequential,
conditional, and map-reduce (``Send``) semantics, including the channel-default
reducer behavior, rather than only that it runs a single graph.

    from vincio.testing import assert_backend_conformance

    async def test_my_backend():
        await assert_backend_conformance(MyClusterBackend(client=fake))
"""

from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable
from typing import Any

__all__ = ["assert_backend_conformance", "conformance_cases"]

# A case is (name, build_graph, input, expected_keys): ``build_graph`` returns a
# *fresh* StateGraph each call (export adapters may rebuild/rewrap nodes, so the
# reference run and the backend run must not share an instance), and
# ``expected_keys`` is the subset of state compared (nodes may add private keys).
Case = tuple[str, "Callable[[], Any]", dict[str, Any], tuple[str, ...]]


def conformance_cases() -> list[Case]:
    """The canonical battery every runtime backend must reproduce."""
    from vincio.agents import END, Send, StateGraph

    def sequential() -> Any:
        g = StateGraph("seq")
        g.add_node("a", lambda s: {"n": s.get("n", 0) + 1})
        g.add_node("b", lambda s: {"n": s["n"] * 10})
        g.add_edge("a", "b")
        g.add_edge("b", END)
        return g

    def conditional() -> Any:
        g = StateGraph("cond")
        g.add_node("classify", lambda s: {"kind": "even" if s["x"] % 2 == 0 else "odd"})
        g.add_node("even", lambda s: {"label": "even"})
        g.add_node("odd", lambda s: {"label": "odd"})
        g.set_entry("classify")
        g.add_conditional_edge("classify", lambda s: s["kind"], {"even": "even", "odd": "odd"})
        g.add_edge("even", END)
        g.add_edge("odd", END)
        return g

    def map_reduce() -> Any:
        # No seed node and a non-defensive reducer: the channel default is what
        # makes the first write fold correctly (limitation (c)), and it must
        # survive export to the backend (limitation (a)).
        g = StateGraph("mr", reducers={"results": operator.add}, defaults={"results": list})
        g.add_node("dispatch", lambda s: [Send("worker", {"v": v}) for v in s["items"]])
        g.add_node("worker", lambda s: {"results": [s["v"] * 2]})
        g.add_node("reduce", lambda s: {"total": sum(s["results"])})
        g.set_entry("dispatch")
        g.add_edge("dispatch", "reduce")
        g.add_edge("reduce", END)
        return g

    return [
        ("sequential", sequential, {"n": 4}, ("n",)),
        ("conditional_even", conditional, {"x": 2}, ("label",)),
        ("conditional_odd", conditional, {"x": 3}, ("label",)),
        ("map_reduce", map_reduce, {"items": [1, 2, 3]}, ("results", "total")),
    ]


async def assert_backend_conformance(
    backend: Any,
    *,
    label: str | None = None,
    cases: list[Case] | None = None,
) -> None:
    """Assert ``backend.run`` reproduces the native engine on every case.

    For each graph the reference is ``graph.compile().ainvoke(input)``; the
    backend gets a freshly built graph and its final state is compared on the
    case's tracked keys. The ``results`` collection is order-insensitive (map
    fan-out reduces concurrently). Raises :class:`AssertionError` on the first
    divergence, naming the case and backend so the failure is actionable.
    """
    name = label or getattr(backend, "name", type(backend).__name__)
    for case_name, build, payload, keys in cases or conformance_cases():
        reference = await build().compile().ainvoke(payload)
        result = await _run(backend, build(), payload)
        state = result.state if hasattr(result, "state") else result
        for key in keys:
            expected, got = reference.state[key], state.get(key)
            if key == "results" and isinstance(expected, list) and isinstance(got, list):
                expected, got = sorted(expected), sorted(got)
            assert got == expected, (
                f"backend {name!r} diverged from the native engine on case "
                f"{case_name!r}: state[{key!r}] = {got!r}, expected {expected!r}"
            )


def _run(backend: Any, graph: Any, payload: dict[str, Any]) -> Awaitable[Any]:
    return backend.run(graph, payload)
