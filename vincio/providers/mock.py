"""Deterministic mock provider.

Used by tests, trace replay, and offline development. Responses can be:

- scripted per request hash (``responses={hash: ...}``),
- queued in order (``script=[...]``),
- computed by a function (``responder=lambda request: ...``),
- or synthesized deterministically (default), including schema-valid
  structured output generated from the request's JSON schema.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from typing import Any

from ..core.tokens import count_tokens
from ..core.types import (
    ModelCapabilities,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
)
from .base import ModelProvider

__all__ = ["MockProvider", "instance_from_schema"]


def instance_from_schema(schema: dict[str, Any], *, seed: str = "") -> Any:
    """Generate a deterministic, schema-valid instance from a JSON schema."""
    if not isinstance(schema, dict):
        return None
    if "$ref" in schema:
        # Resolve local refs against $defs/definitions when present.
        ref: str = schema["$ref"]
        root = schema.get("_root", schema)
        if ref.startswith("#/"):
            node: Any = root
            for part in ref[2:].split("/"):
                node = node.get(part, {}) if isinstance(node, dict) else {}
            merged = dict(node)
            merged["_root"] = root
            return instance_from_schema(merged, seed=seed)
        return None
    for combiner in ("anyOf", "oneOf"):
        if combiner in schema and schema[combiner]:
            options = [o for o in schema[combiner] if o.get("type") != "null"] or schema[combiner]
            merged = dict(options[0])
            merged["_root"] = schema.get("_root", schema)
            return instance_from_schema(merged, seed=seed)
    if "allOf" in schema and schema["allOf"]:
        merged_schema: dict[str, Any] = {}
        for part in schema["allOf"]:
            merged_schema.update(part)
        merged_schema["_root"] = schema.get("_root", schema)
        return instance_from_schema(merged_schema, seed=seed)
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), "null")
    root = schema.get("_root", schema)
    if schema_type == "object" or "properties" in schema:
        result: dict[str, Any] = {}
        for name, prop in (schema.get("properties") or {}).items():
            prop_with_root = dict(prop)
            prop_with_root["_root"] = root
            result[name] = instance_from_schema(prop_with_root, seed=f"{seed}.{name}")
        return result
    if schema_type == "array":
        item_schema = dict(schema.get("items") or {"type": "string"})
        item_schema["_root"] = root
        min_items = schema.get("minItems", 1)
        return [instance_from_schema(item_schema, seed=f"{seed}[{i}]") for i in range(max(1, min_items))]
    if schema_type == "string":
        if schema.get("format") == "date-time":
            return "2026-01-01T00:00:00Z"
        return f"mock_{seed.strip('.') or 'value'}"
    if schema_type == "integer":
        return schema.get("minimum", 1)
    if schema_type == "number":
        return float(schema.get("minimum", 0.5))
    if schema_type == "boolean":
        return True
    if schema_type == "null":
        return None
    return f"mock_{seed.strip('.') or 'value'}"


class MockProvider(ModelProvider):
    name = "mock"

    def __init__(
        self,
        *,
        responses: dict[str, ModelResponse | str] | None = None,
        script: list[ModelResponse | str | dict[str, Any]] | None = None,
        responder: Callable[[ModelRequest], ModelResponse | str | dict[str, Any]] | None = None,
        default_text: str | None = None,
        latency_ms: int = 1,
        embedding_dim: int = 64,
        reasoning: bool = False,
    ) -> None:
        self.responses = responses or {}
        self.script = list(script or [])
        self.responder = responder
        self.default_text = default_text
        self.latency_ms = latency_ms
        self.embedding_dim = embedding_dim
        self.reasoning = reasoning
        self.requests: list[ModelRequest] = []
        self.call_count = 0

    # -- helpers ---------------------------------------------------------------

    def _coerce(self, value: ModelResponse | str | dict[str, Any], request: ModelRequest) -> ModelResponse:
        if isinstance(value, ModelResponse):
            response = value.model_copy(deep=True)
        elif isinstance(value, dict):
            if "tool_call" in value:
                tc = value["tool_call"]
                response = ModelResponse(
                    text=value.get("text", ""),
                    tool_calls=[ToolCallRequest(name=tc["name"], arguments=tc.get("arguments", {}))],
                    finish_reason="tool_calls",
                )
            else:
                response = ModelResponse(text=json.dumps(value), structured=value)
        else:
            response = ModelResponse(text=value)
            if request.output_schema is not None:
                try:
                    response.structured = json.loads(value)
                except json.JSONDecodeError:
                    response.structured = None
        response.model = request.model
        response.provider = self.name
        response.latency_ms = self.latency_ms
        if not response.usage.input_tokens:
            input_text = "\n".join(m.text for m in request.messages)
            response.usage = TokenUsage(
                input_tokens=count_tokens(input_text),
                output_tokens=count_tokens(response.text),
            )
        # Emulate thinking tokens when reasoning is requested and supported, so
        # the reasoning surface (cost accounting, span attributes) is exercised
        # offline. Deterministic: scales with the requested effort.
        if self.reasoning and request.reasoning_effort and not response.usage.reasoning_tokens:
            think = {"minimal": 8, "low": 24, "medium": 64, "high": 128}.get(
                request.reasoning_effort, 32
            )
            response.usage.reasoning_tokens = think
            response.usage.output_tokens += think
        return response

    def _default_response(self, request: ModelRequest) -> ModelResponse:
        if request.output_schema is not None:
            instance = instance_from_schema(request.output_schema, seed=request.hash[:6])
            text = json.dumps(instance)
            return self._coerce(
                ModelResponse(text=text, structured=instance if isinstance(instance, dict) else {"value": instance}),
                request,
            )
        if self.default_text is not None:
            return self._coerce(self.default_text, request)
        user_text = next(
            (m.text for m in reversed(request.messages) if m.role == "user"), ""
        )
        return self._coerce(f"[mock:{request.model}] {user_text[:200]}", request)

    # -- API ---------------------------------------------------------------------

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        self.call_count += 1
        if request.hash in self.responses:
            return self._coerce(self.responses[request.hash], request)
        if self.script:
            return self._coerce(self.script.pop(0), request)
        if self.responder is not None:
            return self._coerce(self.responder(request), request)
        return self._default_response(request)

    async def stream(self, request: ModelRequest):
        """Real chunked streaming: the response text arrives in small deltas
        (concatenating them reproduces the text exactly) so streaming
        consumers are exercised meaningfully offline."""
        response = await self.generate(request)
        chunk_size = 16
        for start in range(0, len(response.text), chunk_size):
            yield ModelEvent(type="text_delta", text=response.text[start : start + chunk_size])
        for tool_call in response.tool_calls:
            yield ModelEvent(type="tool_call_delta", tool_call=tool_call)
        yield ModelEvent(type="usage", usage=response.usage)
        yield ModelEvent(type="done", response=response)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Deterministic pseudo-embedding: bag-of-token hashes, L2-normalized.

        Similar texts share tokens and therefore direction, which makes this
        a usable (if crude) semantic signal for tests and offline mode.
        """
        vector = [0.0] * self.embedding_dim
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.embedding_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(
            structured_output=True,
            tool_calling=True,
            vision=True,
            audio=True,
            prompt_caching=True,
            reasoning=self.reasoning,
            max_context_tokens=200_000,
            max_output_tokens=32_768,
            supports_system_message=True,
            supports_developer_message=True,
        )
