"""OpenAI Responses API adapter (``/v1/responses``).

The Responses API is OpenAI's stateful successor to Chat Completions: it keeps
reasoning state across tool calls via ``previous_response_id`` (so reasoning
tokens are not re-billed), exposes built-in tools, and returns a typed
``output`` array. This adapter speaks it behind the same
:class:`~vincio.providers.base.ModelProvider` interface — Chat Completions
(:class:`~vincio.providers.openai.OpenAIProvider`) stays the portable default,
and this is opt-in via ``provider="openai_responses"``.

Only the request/response shape differs from Chat Completions; rendering of
messages and tools, reasoning negotiation, and cost tracking reuse the parent.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.errors import ProviderResponseError
from ..core.types import (
    Message,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
    ToolSpec,
)
from .openai import OpenAIProvider

__all__ = ["OpenAIResponsesProvider"]


class OpenAIResponsesProvider(OpenAIProvider):
    """OpenAI provider that targets the Responses API instead of Chat Completions."""

    name = "openai_responses"

    # -- rendering -------------------------------------------------------------

    def _render_input(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Split system/developer text into ``instructions`` and render the rest
        as Responses ``input`` items."""
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.role in ("system", "developer"):
                instructions.append(message.text)
                continue
            if message.role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id or "",
                        "output": message.text,
                    }
                )
                continue
            if message.tool_calls:
                for tc in message.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        }
                    )
                if message.text:
                    items.append({"role": "assistant", "content": message.text})
                continue
            items.append({"role": message.role, "content": message.text})
        return "\n\n".join(t for t in instructions if t), items

    def _render_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        # Responses tools are flat (no nested "function" wrapper). Hosted tools
        # (web_search / file_search / code_interpreter / computer_use, 1.10) emit
        # their provider-native built-in descriptor instead of a function tool.
        from .hosted_tools import hosted_payload, is_hosted

        rendered: list[dict[str, Any]] = []
        for tool in tools:
            if is_hosted(tool):
                rendered.append(hosted_payload(tool))
                continue
            rendered.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                    or {"type": "object", "properties": {}, "additionalProperties": True},
                }
            )
        return rendered

    def _payload(self, request: ModelRequest, *, stream: bool = False) -> dict[str, Any]:
        instructions, items = self._render_input(request.messages)
        payload: dict[str, Any] = {"model": request.model, "input": items}
        if instructions:
            payload["instructions"] = instructions
        if request.previous_response_id is not None:
            payload["previous_response_id"] = request.previous_response_id
        if request.tools:
            payload["tools"] = self._render_tools(request.tools)
        if request.output_schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.output_schema_name or "output",
                    "schema": request.output_schema,
                    "strict": True,
                }
            }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = request.max_output_tokens
        if request.reasoning_effort is not None and self.capabilities(request.model).reasoning:
            effort = request.reasoning_effort
            if effort == "minimal" and "gpt-5" not in request.model:
                effort = "low"
            payload["reasoning"] = {"effort": effort}
        if stream:
            payload["stream"] = True
        return payload

    # -- parsing ---------------------------------------------------------------

    def _parse_usage(self, usage: dict[str, Any] | None) -> TokenUsage:
        if not usage:
            return TokenUsage()
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        return TokenUsage(
            input_tokens=usage.get("input_tokens", 0) or 0,
            output_tokens=usage.get("output_tokens", 0) or 0,
            cached_input_tokens=input_details.get("cached_tokens", 0) or 0,
            reasoning_tokens=output_details.get("reasoning_tokens", 0) or 0,
        )

    def _parse_response(
        self, data: dict[str, Any], request: ModelRequest, latency_ms: int
    ) -> ModelResponse:
        output = data.get("output") or []
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        for item in output:
            kind = item.get("type")
            if kind == "message":
                for part in item.get("content") or []:
                    if part.get("type") in ("output_text", "text"):
                        text_parts.append(part.get("text") or "")
            elif kind == "function_call":
                try:
                    arguments = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {"_raw": item.get("arguments")}
                tool_calls.append(
                    ToolCallRequest(
                        id=item.get("call_id") or item.get("id") or "",
                        name=item.get("name") or "",
                        arguments=arguments,
                    )
                )
        # Responses exposes a convenience aggregate; prefer it when present.
        text = data.get("output_text") or "".join(text_parts)
        if not output and "output_text" not in data:
            raise ProviderResponseError("no output in response", provider=self.name)
        usage = self._parse_usage(data.get("usage"))
        structured: dict[str, Any] | None = None
        if request.output_schema is not None and text:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None
        return ModelResponse(
            id=data.get("id") or "",
            model=data.get("model", request.model),
            text=text,
            tool_calls=tool_calls,
            structured=structured,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=usage,
            cost_usd=self.price_table.cost(request.model, usage),
            latency_ms=latency_ms,
            provider=self.name,
            raw=data,
        )

    # -- API -------------------------------------------------------------------

    async def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.monotonic()
        data = await self._post_json("/responses", self._payload(request))
        latency_ms = int((time.monotonic() - started) * 1000)
        return self._parse_response(data, request, latency_ms)

    async def stream(self, request: ModelRequest):
        # The Responses SSE event schema differs from Chat Completions; emulate
        # streaming from a single generate() rather than mis-parse the parent's
        # chat-completions stream. (Native Responses streaming is future work.)
        async for event in super(OpenAIProvider, self).stream(request):
            yield event
