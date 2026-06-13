"""Streaming validation (0.7).

Validate and repair partial structured output while it streams. The
:class:`StreamingValidator` accumulates text deltas, parses the
balanced-partial JSON, and checks the *prefix* against the schema: fields
that have fully arrived must already have the right shape, while missing
required fields and in-progress values are tolerated until the stream ends.

A definite mismatch (wrong type, unknown field on a closed object) is
therefore known mid-stream — callers can abort generation early instead of
paying for the rest of an invalid answer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .parsers import parse_partial_json
from .repair import Repairer
from .schemas import OutputSchema, RepairPolicy

__all__ = ["StreamingValidationEvent", "StreamingValidator", "validate_partial"]


def validate_partial(data: Any, schema: dict[str, Any]) -> list[str]:
    """Prefix-check *data* against *schema*; returns definite errors only.

    Tolerant of streaming truncation: missing required fields are fine (they
    may still arrive) and string values are never length/enum-checked (they
    may be prefixes). Wrong types and unknown properties on closed objects
    are definite errors no further tokens can fix.
    """
    errors: list[str] = []
    _check_partial(data, schema, path="$", errors=errors, root=schema)
    return errors


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    node: Any = root
    for part in ref[2:].split("/"):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return node if isinstance(node, dict) else {}

def _check_partial(
    data: Any, schema: dict[str, Any], *, path: str, errors: list[str], root: dict[str, Any]
) -> None:
    if not isinstance(schema, dict):
        return
    schema = _resolve_ref(schema, root)
    for combiner in ("anyOf", "oneOf"):
        options = schema.get(combiner)
        if isinstance(options, list) and options:
            # Valid if any branch has no definite errors.
            for option in options:
                branch_errors: list[str] = []
                _check_partial(data, option, path=path, errors=branch_errors, root=root)
                if not branch_errors:
                    return
            errors.append(f"{path}: matches no {combiner} branch")
            return
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        if data is None:
            if "null" in schema_type:
                return
        schema_type = non_null[0] if non_null else None
    if schema_type == "object" or "properties" in schema:
        if not isinstance(data, dict):
            if data is not None:
                errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        properties = schema.get("properties") or {}
        closed = schema.get("additionalProperties") is False
        for key, value in data.items():
            if key in properties:
                _check_partial(value, properties[key], path=f"{path}.{key}", errors=errors, root=root)
            elif closed and properties:
                errors.append(f"{path}: unknown field {key!r}")
        return
    if schema_type == "array":
        if not isinstance(data, list):
            if data is not None:
                errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        item_schema = schema.get("items") or {}
        for index, item in enumerate(data):
            _check_partial(item, item_schema, path=f"{path}[{index}]", errors=errors, root=root)
        return
    if data is None:
        return  # value not arrived yet (or nullable)
    if schema_type == "string" and not isinstance(data, str):
        errors.append(f"{path}: expected string, got {type(data).__name__}")
    elif schema_type == "integer" and (isinstance(data, bool) or not isinstance(data, int)):
        errors.append(f"{path}: expected integer, got {type(data).__name__}")
    elif schema_type == "number" and (isinstance(data, bool) or not isinstance(data, (int, float))):
        errors.append(f"{path}: expected number, got {type(data).__name__}")
    elif schema_type == "boolean" and not isinstance(data, bool):
        errors.append(f"{path}: expected boolean, got {type(data).__name__}")


class StreamingValidationEvent(BaseModel):
    """Result of one incremental validation pass over the stream so far."""

    data: Any = None
    complete: bool = False
    valid_prefix: bool = True
    errors: list[str] = Field(default_factory=list)
    repaired: bool = False
    repair_actions: list[str] = Field(default_factory=list)
    chars_seen: int = 0


class StreamingValidator:
    """Incremental schema validation over a streaming structured output.

    Feed text deltas as they arrive; each :meth:`feed` (past ``min_interval``
    new characters) parses the balanced partial JSON and prefix-checks it.
    :meth:`finalize` runs the full parse with structural repair when the
    stream ends.
    """

    def __init__(
        self,
        schema: OutputSchema | None = None,
        *,
        repair_policy: RepairPolicy | None = None,
        min_interval_chars: int = 24,
    ) -> None:
        self.schema = schema
        self.repairer = Repairer(repair_policy)
        self.min_interval_chars = min_interval_chars
        self._parts: list[str] = []
        self._length = 0
        self._last_parse_length = 0
        self.last_event: StreamingValidationEvent | None = None

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def feed(self, delta: str) -> StreamingValidationEvent | None:
        """Add a text delta; returns a validation event when a parse ran."""
        if delta:
            self._parts.append(delta)
            self._length += len(delta)
        if self._length - self._last_parse_length < self.min_interval_chars:
            return None
        return self._parse()

    def _parse(self) -> StreamingValidationEvent | None:
        self._last_parse_length = self._length
        partial, complete = parse_partial_json(self.text)
        if partial is None:
            return None
        errors = (
            validate_partial(partial, self.schema.json_schema) if self.schema is not None else []
        )
        event = StreamingValidationEvent(
            data=partial,
            complete=complete,
            valid_prefix=not errors,
            errors=errors,
            chars_seen=self._length,
        )
        self.last_event = event
        return event

    def finalize(self) -> StreamingValidationEvent:
        """Full parse + allowed structural repair once the stream has ended."""
        text = self.text
        partial, complete = parse_partial_json(text)
        repaired = False
        actions: list[str] = []
        if partial is not None and self.schema is not None and not self.schema.is_valid(partial):
            outcome = self.repairer.repair_structure(partial, self.schema)
            if outcome.repaired:
                partial = outcome.data
                repaired = True
                actions = outcome.actions
        errors: list[str] = []
        if self.schema is not None:
            if partial is None:
                errors = ["could not parse structured output"]
            elif not self.schema.is_valid(partial):
                errors = ["output does not match schema after streaming repair"]
        event = StreamingValidationEvent(
            data=partial,
            complete=complete or partial is not None,
            valid_prefix=not errors,
            errors=errors,
            repaired=repaired,
            repair_actions=actions,
            chars_seen=self._length,
        )
        self.last_event = event
        return event
