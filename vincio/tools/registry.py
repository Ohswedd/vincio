"""Tool registry: typed, permissioned tool registration.

Tools register with explicit contracts (input/output schemas, permissions,
side effects, timeout, cost). Plain functions, async functions, and Pydantic
models are all supported; schemas are derived from type hints when not
given explicitly.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model

from ..core.errors import ToolNotFoundError
from ..core.types import ToolSpec

__all__ = ["RegisteredTool", "ToolRegistry"]


def _schema_from_model(model: type[BaseModel] | dict[str, Any] | None) -> dict[str, Any]:
    if model is None:
        return {}
    if isinstance(model, dict):
        return model
    return model.model_json_schema()


def _schema_from_signature(fn: Callable) -> dict[str, Any]:
    """Build an input schema from a function signature."""
    signature = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name in ("self", "cls"):
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, str)
        default = parameter.default if parameter.default is not inspect.Parameter.empty else ...
        fields[name] = (annotation, default)
    if not fields:
        return {"type": "object", "properties": {}, "additionalProperties": True}
    model = create_model(f"{fn.__name__}_input", **fields)
    return model.model_json_schema()


class RegisteredTool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    spec: ToolSpec
    handler: Callable
    input_model: Any = None  # type[BaseModel] | None
    output_model: Any = None

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.handler)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self.stats: dict[str, dict[str, float]] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def register(
        self,
        handler: Callable | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: type[BaseModel] | dict[str, Any] | None = None,
        output_schema: type[BaseModel] | dict[str, Any] | None = None,
        permissions: list[str] | None = None,
        side_effects: str = "read",
        timeout_ms: int = 30_000,
        cost_estimate: float = 0.0,
        approval_required: bool = False,
        cacheable: bool | None = None,
        idempotent: bool = False,
    ):
        """Register a tool. Usable directly or as a decorator::

            @registry.register(permissions=["crm:read"])
            def crm_lookup(customer_id: str) -> dict: ...
        """

        def wrap(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            resolved_input = (
                _schema_from_model(input_schema)
                if input_schema is not None
                else _schema_from_signature(fn)
            )
            spec = ToolSpec(
                name=tool_name,
                description=description or (inspect.getdoc(fn) or tool_name).split("\n")[0],
                input_schema=resolved_input,
                output_schema=_schema_from_model(output_schema),
                permissions=permissions or [],
                side_effects=side_effects,  # type: ignore[arg-type]
                timeout_ms=timeout_ms,
                cost_estimate=cost_estimate,
                approval_required=approval_required,
                cacheable=cacheable,
                idempotent=idempotent,
            )
            self._tools[tool_name] = RegisteredTool(
                spec=spec,
                handler=fn,
                input_model=input_schema if isinstance(input_schema, type) else None,
                output_model=output_schema if isinstance(output_schema, type) else None,
            )
            self.stats.setdefault(
                tool_name,
                {"calls": 0.0, "successes": 0.0, "failures": 0.0, "total_ms": 0.0, "quality_lift_sum": 0.0, "quality_samples": 0.0},
            )
            return fn

        if handler is not None:
            return wrap(handler)
        return wrap

    def register_spec(self, spec: ToolSpec, *, handler: Callable | None = None) -> RegisteredTool:
        """Register a pre-built :class:`ToolSpec` (e.g. a provider-native hosted
        tool that executes server-side, so it has no local handler). The spec —
        including its ``metadata`` marker — is preserved verbatim."""

        def _hosted_stub(*args: Any, **kwargs: Any) -> Any:
            raise ToolNotFoundError(
                f"tool {spec.name!r} is a provider-native hosted tool; it is executed "
                "by the provider, not locally",
                tool=spec.name,
            )

        registered = RegisteredTool(spec=spec, handler=handler or _hosted_stub)
        self._tools[spec.name] = registered
        self.stats.setdefault(
            spec.name,
            {"calls": 0.0, "successes": 0.0, "failures": 0.0, "total_ms": 0.0,
             "quality_lift_sum": 0.0, "quality_samples": 0.0},
        )
        return registered

    def get(self, name: str) -> RegisteredTool:
        if name not in self._tools:
            raise ToolNotFoundError(f"tool {name!r} not registered; known: {self.names}", tool=name)
        return self._tools[name]

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        if names is None:
            return [tool.spec for tool in self._tools.values()]
        return [self.get(name).spec for name in names]

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    # -- reliability scoring ------------------------------------------

    def record_call(self, name: str, *, success: bool, duration_ms: int) -> None:
        stats = self.stats.setdefault(
            name,
            {"calls": 0.0, "successes": 0.0, "failures": 0.0, "total_ms": 0.0, "quality_lift_sum": 0.0, "quality_samples": 0.0},
        )
        stats["calls"] += 1
        stats["successes" if success else "failures"] += 1
        stats["total_ms"] += duration_ms
        if name in self._tools:
            tool = self._tools[name]
            tool.spec.reliability_score = stats["successes"] / stats["calls"]

    def record_quality_lift(self, name: str, lift: float) -> None:
        stats = self.stats.get(name)
        if stats is not None:
            stats["quality_lift_sum"] += lift
            stats["quality_samples"] += 1

    def reliability(self, name: str) -> dict[str, float]:
        stats = self.stats.get(name, {})
        calls = stats.get("calls", 0.0)
        return {
            "reliability": (stats.get("successes", 0.0) / calls) if calls else 1.0,
            "avg_latency_ms": (stats.get("total_ms", 0.0) / calls) if calls else 0.0,
            "usefulness": (
                stats.get("quality_lift_sum", 0.0) / stats.get("quality_samples", 1.0)
                if stats.get("quality_samples")
                else 0.0
            ),
            "calls": calls,
        }
