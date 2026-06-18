"""Constrained generation.

Provider-native schema-constrained decoding where the provider supports it,
with the robust-parser fallback everywhere else. The decoding mode is
negotiated per run from the provider's capability matrix and recorded on the
model span, so "how was this output constrained?" is always answerable from
the trace.

Strict-mode schema sanitization (:func:`to_strict_json_schema`) converts any
JSON schema into the shape strict constrained decoders require: every object
closed (``additionalProperties: false``), every property required, optional
fields made nullable instead of absent. Grammar-style constraints (fixed
choices, regex-shaped strings) are expressed as JSON schemas so they ride the
same native path.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from ..core.types import ModelCapabilities

__all__ = [
    "DecodingMode",
    "negotiate_decoding",
    "to_strict_json_schema",
    "choice_schema",
    "regex_schema",
]


class DecodingMode(StrEnum):
    """How structured output is enforced for a given provider + contract."""

    NATIVE = "native"  # provider-enforced grammar/JSON-schema decoding
    PROMPT = "prompt"  # schema rendered in the prompt; robust parser + repair
    NONE = "none"  # no schema on the contract


def negotiate_decoding(
    capabilities: ModelCapabilities, schema_def: dict[str, Any] | None
) -> DecodingMode:
    """Pick the strongest decoding mode the provider supports for this contract."""
    if schema_def is None:
        return DecodingMode.NONE
    if capabilities.structured_output:
        return DecodingMode.NATIVE
    return DecodingMode.PROMPT


# Keywords that strict constrained decoders commonly reject; dropping them
# only widens the schema (post-hoc validation still enforces them).
_STRICT_UNSUPPORTED_KEYS = ("default", "format")


def to_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform a JSON schema for strict constrained decoding.

    Strict decoders (e.g. OpenAI ``strict: true``) require every object to be
    closed and every property to be required. Optional properties become
    required-but-nullable so the constrained grammar stays exactly as
    expressive as the original schema. The original schema keeps enforcing
    optionality at validation time; this transform only affects decoding.
    """
    return _strictify(schema, root=schema)


def _strictify(node: Any, *, root: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_strictify(item, root=root) for item in node]
    if not isinstance(node, dict):
        return node
    result = {k: v for k, v in node.items() if k not in _STRICT_UNSUPPORTED_KEYS}
    properties = result.get("properties")
    if isinstance(properties, dict):
        previously_required = set(result.get("required") or [])
        new_properties: dict[str, Any] = {}
        for name, prop_schema in properties.items():
            strict_prop = _strictify(prop_schema, root=root)
            if name not in previously_required:
                strict_prop = _nullable(strict_prop)
            new_properties[name] = strict_prop
        result["properties"] = new_properties
        result["required"] = list(new_properties)
        result["additionalProperties"] = False
    if "items" in result:
        result["items"] = _strictify(result["items"], root=root)
    for defs_key in ("$defs", "definitions"):
        if isinstance(result.get(defs_key), dict):
            result[defs_key] = {
                name: _strictify(sub_schema, root=root)
                for name, sub_schema in result[defs_key].items()
            }
    for combiner in ("anyOf", "oneOf", "allOf"):
        if combiner in result:
            result[combiner] = _strictify(result[combiner], root=root)
    return result


def _nullable(prop_schema: Any) -> Any:
    """Widen a property schema to accept null (for previously-optional fields)."""
    if not isinstance(prop_schema, dict):
        return prop_schema
    if "anyOf" in prop_schema:
        options = prop_schema["anyOf"]
        if not any(isinstance(o, dict) and o.get("type") == "null" for o in options):
            prop_schema = {**prop_schema, "anyOf": [*options, {"type": "null"}]}
        return prop_schema
    schema_type = prop_schema.get("type")
    if isinstance(schema_type, str) and schema_type != "null":
        return {**prop_schema, "type": [schema_type, "null"]}
    if isinstance(schema_type, list) and "null" not in schema_type:
        return {**prop_schema, "type": [*schema_type, "null"]}
    return prop_schema


def choice_schema(choices: list[str], *, name: str = "choice") -> dict[str, Any]:
    """A grammar-style constraint: the output must be exactly one of *choices*."""
    return {
        "type": "object",
        "properties": {name: {"type": "string", "enum": list(choices)}},
        "required": [name],
        "additionalProperties": False,
    }


def regex_schema(pattern: str, *, name: str = "value") -> dict[str, Any]:
    """A grammar-style constraint: a single string field shaped by *pattern*.

    Providers with native pattern support enforce it during decoding; the
    validation pipeline enforces it everywhere via post-hoc schema checking.
    """
    return {
        "type": "object",
        "properties": {name: {"type": "string", "pattern": pattern}},
        "required": [name],
        "additionalProperties": False,
    }
