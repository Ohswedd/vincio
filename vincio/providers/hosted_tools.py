"""Provider-native hosted tools as namespaced Vincio tools (1.10).

OpenAI's Responses API ships built-in, server-executed tools — ``web_search``,
``file_search``, ``code_interpreter``, ``computer_use`` — that the model invokes
without a local handler. This module surfaces them as ordinary Vincio
:class:`~vincio.core.types.ToolSpec`\\ s (namespaced ``openai:web_search`` …) so
they register on the tool registry, carry explicit permissions, and ride the
same RBAC + audit path as any local tool. Each spec is marked
``metadata={"hosted": True, ...}``; the Responses adapter
(:mod:`vincio.providers.openai_responses`) recognizes the marker and emits the
built-in tool descriptor instead of a function tool.
"""

from __future__ import annotations

from ..core.types import ToolSpec

__all__ = ["HOSTED_TOOLS", "hosted_tool_specs", "is_hosted", "hosted_payload"]

# name -> (Responses built-in `type`, default permission, side-effect, cost hint)
_HOSTED = {
    "web_search": ("web_search", "web:search", "external", 0.01),
    "file_search": ("file_search", "files:read", "read", 0.0),
    "code_interpreter": ("code_interpreter", "code:execute", "external", 0.03),
    "computer_use": ("computer_use_preview", "computer:use", "external", 0.05),
}


def _spec(name: str, *, namespace: str = "openai") -> ToolSpec:
    hosted_type, permission, side_effects, cost = _HOSTED[name]
    return ToolSpec(
        name=f"{namespace}:{name}",
        description=f"Provider-native hosted tool: {name} (executed by {namespace}).",
        permissions=[permission],
        side_effects=side_effects,  # type: ignore[arg-type]
        cost_estimate=cost,
        # computer_use can take consequential real-world actions: gate it.
        approval_required=name == "computer_use",
        metadata={"hosted": True, "hosted_type": hosted_type, "namespace": namespace, "tool": name},
    )


HOSTED_TOOLS: dict[str, ToolSpec] = {name: _spec(name) for name in _HOSTED}


def hosted_tool_specs(names: list[str] | None = None, *, namespace: str = "openai") -> list[ToolSpec]:
    """Return ToolSpecs for the named hosted tools (all of them by default)."""
    selected = names or list(_HOSTED)
    specs: list[ToolSpec] = []
    for name in selected:
        if name not in _HOSTED:
            raise KeyError(f"unknown hosted tool {name!r}; known: {sorted(_HOSTED)}")
        specs.append(_spec(name, namespace=namespace))
    return specs


def is_hosted(spec: ToolSpec) -> bool:
    return bool(spec.metadata.get("hosted"))


def hosted_payload(spec: ToolSpec) -> dict[str, object]:
    """The Responses built-in tool descriptor for a hosted ToolSpec."""
    return {"type": spec.metadata.get("hosted_type", spec.name.split(":")[-1])}
