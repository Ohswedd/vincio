"""Output repair.

Repair is allowed only for: malformed JSON, missing optional fields, type
coercion, markdown formatting. Repair is NOT allowed for: unsupported
factual claims, unsafe content, missing required evidence, failed business
rules — those must fail validation and surface to the caller.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.errors import OutputRepairForbiddenError
from ..core.types import Message, ModelRequest
from ..providers.base import ModelProvider
from .parsers import lenient_json_loads
from .schemas import OutputSchema, RepairPolicy

__all__ = ["RepairOutcome", "Repairer"]


class RepairOutcome:
    def __init__(self, data: Any, *, repaired: bool, actions: list[str]) -> None:
        self.data = data
        self.repaired = repaired
        self.actions = actions


def _coerce_types(data: Any, schema: dict[str, Any]) -> tuple[Any, list[str]]:
    """Safe type coercions: '3.5'→3.5, 1→True for booleans, scalar→[scalar]."""
    actions: list[str] = []
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), None)
    if schema_type == "number" and isinstance(data, str):
        try:
            return float(data), [f"coerced {data!r} to number"]
        except ValueError:
            return data, []
    if schema_type == "integer" and isinstance(data, (str, float)) and not isinstance(data, bool):
        try:
            value = float(data)
            if value.is_integer():
                return int(value), [f"coerced {data!r} to integer"]
        except ValueError:
            pass
        return data, []
    if schema_type == "string" and isinstance(data, (int, float)) and not isinstance(data, bool):
        return str(data), [f"coerced {data!r} to string"]
    if schema_type == "boolean" and isinstance(data, str):
        lowered = data.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True, [f"coerced {data!r} to boolean"]
        if lowered in ("false", "no", "0"):
            return False, [f"coerced {data!r} to boolean"]
        return data, []
    if schema_type == "array" and not isinstance(data, list) and data is not None:
        coerced_item, item_actions = _coerce_types(data, schema.get("items") or {})
        return [coerced_item], ["wrapped scalar in array", *item_actions]
    if schema_type == "object" and isinstance(data, dict):
        properties = schema.get("properties") or {}
        result = dict(data)
        for key, prop_schema in properties.items():
            if key in result:
                result[key], item_actions = _coerce_types(result[key], prop_schema)
                actions.extend(f"{key}: {a}" for a in item_actions)
        return result, actions
    if schema_type == "array" and isinstance(data, list):
        item_schema = schema.get("items") or {}
        result_list = []
        for index, item in enumerate(data):
            coerced, item_actions = _coerce_types(item, item_schema)
            result_list.append(coerced)
            actions.extend(f"[{index}]: {a}" for a in item_actions)
        return result_list, actions
    return data, actions


def _fill_optional(data: Any, schema: dict[str, Any]) -> tuple[Any, list[str]]:
    """Fill missing optional fields with schema defaults / nulls."""
    if not isinstance(data, dict) or schema.get("type") not in (None, "object"):
        return data, []
    actions: list[str] = []
    required = set(schema.get("required") or [])
    properties = schema.get("properties") or {}
    result = dict(data)
    for key, prop_schema in properties.items():
        if key in result or key in required:
            continue
        if "default" in prop_schema:
            result[key] = prop_schema["default"]
            actions.append(f"filled optional {key!r} with default")
        elif isinstance(prop_schema.get("type"), list) and "null" in prop_schema["type"]:
            result[key] = None
            actions.append(f"filled optional {key!r} with null")
        elif "anyOf" in prop_schema and any(o.get("type") == "null" for o in prop_schema["anyOf"]):
            result[key] = None
            actions.append(f"filled optional {key!r} with null")
    return result, actions


_LLM_REPAIR_PROMPT = (
    "The following output failed schema validation. Re-serialize it so it "
    "matches the JSON schema EXACTLY. Do not add, remove, or change any "
    "factual content, claims, or citations — only fix structure, field "
    "names, and types. Output JSON only."
)


class Repairer:
    def __init__(
        self,
        policy: RepairPolicy | None = None,
        *,
        provider: ModelProvider | None = None,
        model: str | None = None,
    ) -> None:
        self.policy = policy or RepairPolicy()
        self.provider = provider
        self.model = model

    def repair_parse(self, text: str) -> RepairOutcome:
        """Repair malformed JSON text."""
        if not self.policy.allow_json_repair:
            raise OutputRepairForbiddenError("JSON repair disabled by repair policy")
        data = lenient_json_loads(text)
        return RepairOutcome(data, repaired=True, actions=["lenient JSON parse"])

    def repair_structure(self, data: Any, schema: OutputSchema) -> RepairOutcome:
        """Apply allowed structural repairs until the schema validates."""
        actions: list[str] = []
        current = data
        for _attempt in range(self.policy.max_repair_attempts):
            if schema.is_valid(current):
                return RepairOutcome(current, repaired=bool(actions), actions=actions)
            progressed = False
            if self.policy.allow_type_coercion:
                coerced, coercion_actions = _coerce_types(current, schema.json_schema)
                if coercion_actions:
                    current = coerced
                    actions.extend(coercion_actions)
                    progressed = True
            if self.policy.allow_fill_optional:
                filled, fill_actions = _fill_optional(current, schema.json_schema)
                if fill_actions:
                    current = filled
                    actions.extend(fill_actions)
                    progressed = True
            if not progressed:
                break
        return RepairOutcome(current, repaired=bool(actions), actions=actions)

    async def repair_with_model(self, raw_text: str, schema: OutputSchema) -> RepairOutcome:
        """LLM reserialization (structure-only)."""
        if not self.policy.allow_llm_repair:
            raise OutputRepairForbiddenError("LLM repair disabled by repair policy")
        if self.provider is None or self.model is None:
            raise OutputRepairForbiddenError("LLM repair requires a provider and model")
        request = ModelRequest(
            model=self.model,
            messages=[
                Message(role="system", content=_LLM_REPAIR_PROMPT),
                Message(
                    role="user",
                    content=f"Schema:\n{json.dumps(schema.json_schema)}\n\nOutput to fix:\n{raw_text[:12_000]}",
                ),
            ],
            output_schema=schema.json_schema,
            output_schema_name=schema.name,
            temperature=0.0,
        )
        response = await self.provider.generate(request)
        data = response.structured if response.structured is not None else lenient_json_loads(response.text)
        return RepairOutcome(data, repaired=True, actions=["model reserialization"])
