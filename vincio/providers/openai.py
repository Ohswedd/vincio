"""OpenAI provider (Chat Completions API) implemented over httpx.

Works with api.openai.com and any OpenAI-compatible endpoint. Supports tool
calling, structured outputs (json_schema), streaming, and embeddings.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from ..core.errors import ProviderResponseError
from ..core.types import (
    Message,
    ModelCapabilities,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
    ToolSpec,
)
from ..observability.costs import PriceTable, default_price_table
from .base import HTTPProvider, parse_sse_lines

__all__ = ["OpenAIProvider"]


class OpenAIProvider(HTTPProvider):
    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    default_embedding_model = "text-embedding-3-small"

    def __init__(self, *args: Any, price_table: PriceTable | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.price_table = price_table or default_price_table()

    # -- rendering -------------------------------------------------------------

    def _render_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for message in messages:
            role = "system" if message.role == "developer" else message.role
            item: dict[str, Any] = {"role": role}
            if isinstance(message.content, str):
                item["content"] = message.content
            else:
                parts: list[dict[str, Any]] = []
                for part in message.content:
                    if part.type == "text":
                        parts.append({"type": "text", "text": part.text or ""})
                    elif part.type == "image" and part.image is not None:
                        url = part.image.url or f"file://{part.image.path}"
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": url, "detail": part.image.detail},
                            }
                        )
                item["content"] = parts
            if message.role == "tool":
                item["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in message.tool_calls
                ]
            rendered.append(item)
        return rendered

    def _render_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                    or {"type": "object", "properties": {}, "additionalProperties": True},
                },
            }
            for tool in tools
        ]

    def _payload(self, request: ModelRequest, *, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self._render_messages(request.messages),
        }
        if request.tools:
            payload["tools"] = self._render_tools(request.tools)
        if request.output_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema_name or "output",
                    "schema": request.output_schema,
                    "strict": True,
                },
            }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_output_tokens is not None:
            payload["max_completion_tokens"] = request.max_output_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.seed is not None:
            payload["seed"] = request.seed
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    # -- parsing -----------------------------------------------------------------

    def _parse_usage(self, usage: dict[str, Any] | None) -> TokenUsage:
        if not usage:
            return TokenUsage()
        details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        return TokenUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cached_input_tokens=details.get("cached_tokens", 0),
            reasoning_tokens=completion_details.get("reasoning_tokens", 0),
        )

    def _parse_response(self, data: dict[str, Any], request: ModelRequest, latency_ms: int) -> ModelResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderResponseError("no choices in response", provider=self.name)
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        tool_calls: list[ToolCallRequest] = []
        for tc in message.get("tool_calls") or []:
            function = tc.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": function.get("arguments")}
            tool_calls.append(
                ToolCallRequest(id=tc.get("id") or "", name=function.get("name") or "", arguments=arguments)
            )
        finish_map = {
            "stop": "stop",
            "length": "length",
            "tool_calls": "tool_calls",
            "content_filter": "content_filter",
        }
        usage = self._parse_usage(data.get("usage"))
        structured: dict[str, Any] | None = None
        if request.output_schema is not None and text:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None
        return ModelResponse(
            model=data.get("model", request.model),
            text=text,
            tool_calls=tool_calls,
            structured=structured,
            finish_reason=finish_map.get(choice.get("finish_reason"), "stop"),
            usage=usage,
            cost_usd=self.price_table.cost(request.model, usage),
            latency_ms=latency_ms,
            provider=self.name,
            raw=data,
        )

    # -- API --------------------------------------------------------------------

    async def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.monotonic()
        data = await self._post_json("/chat/completions", self._payload(request))
        latency_ms = int((time.monotonic() - started) * 1000)
        return self._parse_response(data, request, latency_ms)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        started = time.monotonic()
        text_parts: list[str] = []
        usage = TokenUsage()
        tool_accumulator: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        model_name = request.model
        async for line in self._post_stream("/chat/completions", self._payload(request, stream=True)):
            data_str = parse_sse_lines(line)
            if data_str is None or data_str == "[DONE]":
                continue
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            model_name = chunk.get("model", model_name)
            if chunk.get("usage"):
                usage = self._parse_usage(chunk["usage"])
                yield ModelEvent(type="usage", usage=usage)
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text_parts.append(delta["content"])
                    yield ModelEvent(type="text_delta", text=delta["content"])
                for tc in delta.get("tool_calls") or []:
                    index = tc.get("index", 0)
                    acc = tool_accumulator.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.get("id"):
                        acc["id"] = tc["id"]
                    function = tc.get("function") or {}
                    if function.get("name"):
                        acc["name"] = function["name"]
                    if function.get("arguments"):
                        acc["arguments"] += function["arguments"]
        tool_calls: list[ToolCallRequest] = []
        for index in sorted(tool_accumulator):
            acc = tool_accumulator[index]
            try:
                arguments = json.loads(acc["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": acc["arguments"]}
            tool_call = ToolCallRequest(id=acc["id"] or f"tc_{index}", name=acc["name"], arguments=arguments)
            tool_calls.append(tool_call)
            yield ModelEvent(type="tool_call_delta", tool_call=tool_call)
        text = "".join(text_parts)
        structured: dict[str, Any] | None = None
        if request.output_schema is not None and text:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None
        finish_map = {"stop": "stop", "length": "length", "tool_calls": "tool_calls", "content_filter": "content_filter"}
        response = ModelResponse(
            model=model_name,
            text=text,
            tool_calls=tool_calls,
            structured=structured,
            finish_reason=finish_map.get(finish_reason, "stop"),
            usage=usage,
            cost_usd=self.price_table.cost(request.model, usage),
            latency_ms=int((time.monotonic() - started) * 1000),
            provider=self.name,
        )
        yield ModelEvent(type="done", response=response)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        data = await self._post_json(
            "/embeddings", {"model": model or self.default_embedding_model, "input": texts}
        )
        items = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]

    def capabilities(self, model: str) -> ModelCapabilities:
        is_mini = "mini" in model or "nano" in model
        return ModelCapabilities(
            structured_output=True,
            tool_calling=True,
            vision=not model.startswith("gpt-3.5"),
            audio="audio" in model,
            prompt_caching=True,
            max_context_tokens=128_000 if "gpt-4o" in model else 272_000,
            max_output_tokens=16_384 if is_mini else 32_768,
            supports_system_message=True,
            supports_developer_message=True,
        )
