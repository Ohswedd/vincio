"""Multi-schema routing (0.7).

Choose and validate against alternative output schemas by task or content.
A :class:`SchemaRouter` holds named routes — each a schema plus the
conditions under which it applies (task types, keywords, or a predicate) —
and picks the contract for a run before generation. Content-side,
:meth:`SchemaRouter.classify` finds which registered schema some structured
data actually matches, so heterogeneous outputs can be validated without
knowing the shape in advance.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import OutputSchemaError
from .schemas import OutputSchema

__all__ = ["SchemaRoute", "SchemaRouter"]


class SchemaRoute(BaseModel):
    """One routable schema and the conditions that select it."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    schema_obj: OutputSchema = Field(exclude=True)
    task_types: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    predicate: Any = Field(default=None, exclude=True)  # Callable[[str], bool] | None
    priority: int = 100  # lower wins ties

    def matches(self, text: str, task_type: str | None = None) -> float:
        """Match score for routing: 0 = no match; higher = stronger match."""
        score = 0.0
        if self.task_types:
            if task_type is None or task_type not in self.task_types:
                return 0.0
            score += 2.0
        if self.predicate is not None:
            if not self.predicate(text):
                return 0.0
            score += 2.0
        if self.keywords:
            lowered = text.lower()
            # Keywords match at word starts, tolerating suffixes ("crash"
            # matches "crashed") so simple morphology doesn't break routing.
            hits = sum(
                1
                for keyword in self.keywords
                if re.search(rf"\b{re.escape(keyword.lower())}", lowered)
            )
            if hits == 0:
                return 0.0
            score += hits
        return score


class SchemaRouter:
    """Routes a run (or a piece of structured data) to one of several schemas."""

    def __init__(self, *, default: OutputSchema | None = None) -> None:
        self.routes: list[SchemaRoute] = []
        self.default = default

    def add(
        self,
        schema: OutputSchema | type | dict[str, Any],
        *,
        name: str | None = None,
        task_types: list[str] | None = None,
        keywords: list[str] | None = None,
        when: Callable[[str], bool] | None = None,
        priority: int = 100,
    ) -> SchemaRoute:
        schema_obj = _coerce_schema(schema)
        route = SchemaRoute(
            name=name or schema_obj.name,
            schema_obj=schema_obj,
            task_types=list(task_types or []),
            keywords=list(keywords or []),
            predicate=when,
            priority=priority,
        )
        self.routes.append(route)
        return route

    def route(self, text: str, *, task_type: str | None = None) -> SchemaRoute | None:
        """Pick the best-matching route for a task (None → use the default)."""
        scored = [
            (route.matches(text or "", task_type), route)
            for route in self.routes
        ]
        candidates = [(score, route) for score, route in scored if score > 0]
        if not candidates:
            return None
        candidates.sort(key=lambda pair: (-pair[0], pair[1].priority))
        return candidates[0][1]

    def classify(self, data: Any) -> SchemaRoute | None:
        """Content-based routing: the first registered schema *data* validates
        against (routes are tried in priority order)."""
        for route in sorted(self.routes, key=lambda r: r.priority):
            if route.schema_obj.is_valid(data):
                return route
        return None

    def validate_any(self, data: Any) -> tuple[str, Any]:
        """Validate *data* against the alternatives; returns (route name,
        validated output) or raises :class:`OutputSchemaError`."""
        route = self.classify(data)
        if route is not None:
            return route.name, route.schema_obj.validate(data)
        if self.default is not None and self.default.is_valid(data):
            return self.default.name, self.default.validate(data)
        raise OutputSchemaError(
            "output matches none of the registered schemas",
            errors=[route.name for route in self.routes],
        )


def _coerce_schema(schema: OutputSchema | type | dict[str, Any]) -> OutputSchema:
    if isinstance(schema, OutputSchema):
        return schema
    if isinstance(schema, dict):
        return OutputSchema.from_json_schema(schema)
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return OutputSchema.from_pydantic(schema)
    raise OutputSchemaError(f"unsupported schema type: {type(schema).__name__}")
