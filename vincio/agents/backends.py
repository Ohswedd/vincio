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

import importlib
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..core.errors import ConfigError
from .crew import Crew
from .graph import END, StateGraph

__all__ = ["RuntimeBackend", "LangGraphBackend", "OpenAIAgentsBackend"]


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
