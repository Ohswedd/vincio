"""Anthropic provider (Messages API) implemented over httpx.

Supports tool calling, structured output (via forced tool use), streaming,
and prompt cache control on stable prefixes.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any, cast

from ..core.errors import ProviderResponseError
from ..core.media import encode_image_bytes
from ..core.types import (
    FinishReason,
    ModelCapabilities,
    ModelEvent,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
    ToolSpec,
)
from ..observability.costs import PriceTable, default_price_table
from .base import HTTPProvider, parse_sse_lines, reasoning_budget_from_effort

__all__ = ["AnthropicProvider"]

STRUCTURED_TOOL_NAME = "emit_structured_output"
ANTHROPIC_VERSION = "2023-06-01"
# Beta that unlocks the 1-hour cache TTL; harmless when only 5-minute (default
# ephemeral) breakpoints are used, so it is sent whenever caching is in play.
EXTENDED_CACHE_TTL_BETA = "extended-cache-ttl-2025-04-11"


def _cache_control(ttl: str | None) -> dict[str, str]:
    """Build an Anthropic ``cache_control`` block for a cache breakpoint.

    A bare ``ephemeral`` block is the 5-minute cache; ``ttl="1h"`` requests the
    one-hour cache (needs the extended-cache-ttl beta header).
    """
    control = {"type": "ephemeral"}
    if ttl == "1h":
        control["ttl"] = "1h"
    return control


class AnthropicProvider(HTTPProvider):
    name = "anthropic"
    default_base_url = "https://api.anthropic.com/v1"

    def __init__(self, *args: Any, price_table: PriceTable | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.price_table = price_table or default_price_table()

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": EXTENDED_CACHE_TTL_BETA,
            "Content-Type": "application/json",
        }

    # -- rendering -------------------------------------------------------------

    def _render(self, request: ModelRequest) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Returns (system_blocks, messages). System/developer roles become system blocks."""
        system_blocks: list[dict[str, Any]] = []
        rendered: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role in ("system", "developer"):
                block: dict[str, Any] = {"type": "text", "text": message.text}
                if message.cache_hint:
                    block["cache_control"] = _cache_control(message.cache_ttl)
                system_blocks.append(block)
                continue
            if message.role == "tool":
                rendered.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": message.text,
                            }
                        ],
                    }
                )
                continue
            content: list[dict[str, Any]] = []
            if isinstance(message.content, str):
                if message.content:
                    content.append({"type": "text", "text": message.content})
            else:
                for part_model in message.content:
                    if part_model.type == "text":
                        content.append({"type": "text", "text": part_model.text or ""})
                    elif part_model.type == "image" and part_model.image is not None:
                        if part_model.image.url:
                            content.append(
                                {
                                    "type": "image",
                                    "source": {"type": "url", "url": part_model.image.url},
                                }
                            )
                        elif part_model.image.path:
                            media_type, data = encode_image_bytes(part_model.image)
                            content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    },
                                }
                            )
            if message.role == "assistant" and message.tool_calls:
                for tc in message.tool_calls:
                    content.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    )
            # A cache breakpoint caches everything up to and including the last
            # block of this message — works for single-text and multi-part alike.
            if message.cache_hint and content:
                content[-1] = {**content[-1], "cache_control": _cache_control(message.cache_ttl)}
            rendered.append({"role": message.role, "content": content or [{"type": "text", "text": ""}]})
        return system_blocks, rendered

    def _render_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
                or {"type": "object", "properties": {}, "additionalProperties": True},
            }
            for tool in tools
        ]

    def _payload(self, request: ModelRequest, *, stream: bool = False) -> dict[str, Any]:
        system_blocks, messages = self._render(request)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens or 4096,
        }
        if system_blocks:
            payload["system"] = system_blocks
        tools = self._render_tools(request.tools)
        if request.output_schema is not None:
            # Structured output via a forced tool call carrying the schema.
            tools.append(
                {
                    "name": STRUCTURED_TOOL_NAME,
                    "description": "Emit the final structured output matching the required schema.",
                    "input_schema": request.output_schema,
                }
            )
            if not request.tools:
                payload["tool_choice"] = {"type": "tool", "name": STRUCTURED_TOOL_NAME}
        if tools:
            payload["tools"] = tools
        thinking = (
            request.reasoning_effort is not None or request.thinking_budget_tokens is not None
        ) and self.capabilities(request.model).reasoning
        if thinking:
            # Extended thinking: budget must be < max_tokens, and the API only
            # accepts the default sampling settings, so temperature/top_p are
            # dropped while thinking is on.
            budget = reasoning_budget_from_effort(
                request.reasoning_effort, request.thinking_budget_tokens
            )
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["max_tokens"] = max(payload["max_tokens"], budget + 1024)
        else:
            if request.temperature is not None:
                payload["temperature"] = request.temperature
            if request.top_p is not None:
                payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop
        if stream:
            payload["stream"] = True
        return payload

    # -- parsing -----------------------------------------------------------------

    def _parse_usage(self, usage: dict[str, Any] | None) -> TokenUsage:
        if not usage:
            return TokenUsage()
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        cache_create = usage.get("cache_creation_input_tokens", 0) or 0
        return TokenUsage(
            input_tokens=(usage.get("input_tokens", 0) or 0) + cache_read + cache_create,
            output_tokens=usage.get("output_tokens", 0) or 0,
            cached_input_tokens=cache_read,
        )

    def _parse_response(self, data: dict[str, Any], request: ModelRequest, latency_ms: int) -> ModelResponse:
        content = data.get("content")
        if content is None:
            raise ProviderResponseError(
                "no content in response", provider=self.name, retryable=True
            )
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        structured: dict[str, Any] | None = None
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block.get("text") or "")
            elif block.get("type") == "tool_use":
                if block.get("name") == STRUCTURED_TOOL_NAME and request.output_schema is not None:
                    structured = block.get("input") or {}
                else:
                    tool_calls.append(
                        ToolCallRequest(
                            id=block.get("id") or "",
                            name=block.get("name") or "",
                            arguments=block.get("input") or {},
                        )
                    )
        stop_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "refusal": "content_filter",
        }
        finish: FinishReason = cast(FinishReason, stop_map.get(data.get("stop_reason") or "", "stop"))
        if finish == "tool_calls" and not tool_calls and structured is not None:
            finish = "stop"
        text = "".join(text_parts)
        if structured is not None and not text:
            text = json.dumps(structured)
        usage = self._parse_usage(data.get("usage"))
        return ModelResponse(
            model=data.get("model", request.model),
            text=text,
            tool_calls=tool_calls,
            structured=structured,
            finish_reason=finish,
            usage=usage,
            cost_usd=self.price_table.cost(request.model, usage),
            latency_ms=latency_ms,
            provider=self.name,
            raw=data,
        )

    # -- API --------------------------------------------------------------------

    async def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.monotonic()
        data = await self._post_json("/messages", self._payload(request))
        latency_ms = int((time.monotonic() - started) * 1000)
        return self._parse_response(data, request, latency_ms)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        started = time.monotonic()
        text_parts: list[str] = []
        usage = TokenUsage()
        finish = "stop"
        model_name = request.model
        blocks: dict[int, dict[str, Any]] = {}
        async for line in self._post_stream("/messages", self._payload(request, stream=True)):
            data_str = parse_sse_lines(line)
            if data_str is None:
                continue
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "message_start":
                message = event.get("message") or {}
                model_name = message.get("model", model_name)
                usage = self._parse_usage(message.get("usage"))
            elif etype == "content_block_start":
                index = event.get("index", 0)
                blocks[index] = dict(event.get("content_block") or {})
                blocks[index].setdefault("_json", "")
            elif etype == "content_block_delta":
                index = event.get("index", 0)
                delta = event.get("delta") or {}
                block = blocks.setdefault(index, {"type": "text", "_json": ""})
                if delta.get("type") == "text_delta":
                    chunk_text = delta.get("text") or ""
                    text_parts.append(chunk_text)
                    yield ModelEvent(type="text_delta", text=chunk_text)
                elif delta.get("type") == "input_json_delta":
                    block["_json"] += delta.get("partial_json") or ""
            elif etype == "message_delta":
                delta = event.get("delta") or {}
                stop_map = {
                    "end_turn": "stop",
                    "stop_sequence": "stop",
                    "max_tokens": "length",
                    "tool_use": "tool_calls",
                }
                if delta.get("stop_reason"):
                    finish = stop_map.get(delta["stop_reason"], "stop")
                delta_usage = self._parse_usage(event.get("usage"))
                usage.output_tokens = max(usage.output_tokens, delta_usage.output_tokens)
        tool_calls: list[ToolCallRequest] = []
        structured: dict[str, Any] | None = None
        for index in sorted(blocks):
            block = blocks[index]
            if block.get("type") != "tool_use":
                continue
            try:
                arguments = json.loads(block.get("_json") or "{}") if block.get("_json") else (block.get("input") or {})
            except json.JSONDecodeError:
                arguments = {"_raw": block.get("_json")}
            if block.get("name") == STRUCTURED_TOOL_NAME and request.output_schema is not None:
                structured = arguments
                continue
            tool_call = ToolCallRequest(
                id=block.get("id") or f"tc_{index}", name=block.get("name") or "", arguments=arguments
            )
            tool_calls.append(tool_call)
            yield ModelEvent(type="tool_call_delta", tool_call=tool_call)
        yield ModelEvent(type="usage", usage=usage)
        text = "".join(text_parts)
        if structured is not None and not text:
            text = json.dumps(structured)
        if finish == "tool_calls" and not tool_calls and structured is not None:
            finish = "stop"
        response = ModelResponse(
            model=model_name,
            text=text,
            tool_calls=tool_calls,
            structured=structured,
            finish_reason=finish,  # type: ignore[arg-type]
            usage=usage,
            cost_usd=self.price_table.cost(request.model, usage),
            latency_ms=int((time.monotonic() - started) * 1000),
            provider=self.name,
        )
        yield ModelEvent(type="done", response=response)

    @staticmethod
    def _parse_models_list(data: dict[str, Any]) -> list[ModelProfile]:
        """Map an Anthropic ``/v1/models`` payload onto sparse profiles."""
        out: list[ModelProfile] = []
        for item in data.get("data") or []:
            model_id = item.get("id")
            if not model_id:
                continue
            out.append(
                ModelProfile(name=item.get("display_name") or model_id,
                             provider="anthropic", model=model_id)
            )
        return out

    async def list_models(self) -> list[ModelProfile]:
        data = await self._get_json("/models")
        return self._parse_models_list(data)

    def capabilities(self, model: str) -> ModelCapabilities:
        from .registry import default_model_registry

        profile = default_model_registry().resolve(model)
        if profile is not None:
            return profile.capabilities
        # Fallback: substring sniffing for ids not in the registry.
        return ModelCapabilities(
            structured_output=True,  # via forced tool use
            tool_calling=True,
            vision=True,
            audio=False,
            prompt_caching=True,
            reasoning="opus" in model or "sonnet" in model or "fable" in model,
            max_context_tokens=200_000,
            max_output_tokens=64_000 if "haiku" not in model else 32_000,
            supports_system_message=True,
            supports_developer_message=False,
        )
