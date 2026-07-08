"""Google Gemini provider (generateContent API) implemented over httpx."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from ..core.errors import ProviderResponseError
from ..core.media import encode_audio_bytes, encode_image_bytes, encode_video_bytes
from ..core.types import (
    ModelCapabilities,
    ModelEvent,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
)
from ..observability.costs import PriceTable, default_price_table
from .base import HTTPProvider, parse_sse_lines, reasoning_budget_from_effort

__all__ = ["GoogleProvider"]


def _strip_unsupported(schema: Any) -> Any:
    """Gemini's schema dialect rejects some JSON Schema keywords."""
    if isinstance(schema, dict):
        return {
            k: _strip_unsupported(v)
            for k, v in schema.items()
            if k not in ("additionalProperties", "$schema", "$defs", "title", "default")
        }
    if isinstance(schema, list):
        return [_strip_unsupported(v) for v in schema]
    return schema


class GoogleProvider(HTTPProvider):
    name = "google"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, *args: Any, price_table: PriceTable | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.price_table = price_table or default_price_table()

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key or "", "Content-Type": "application/json"}

    # -- rendering -------------------------------------------------------------

    def _render(self, request: ModelRequest) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role in ("system", "developer"):
                system_parts.append(message.text)
                continue
            role = "model" if message.role == "assistant" else "user"
            parts: list[dict[str, Any]] = []
            if message.role == "tool":
                parts.append(
                    {
                        "functionResponse": {
                            "name": message.name or "tool",
                            "response": {"output": message.text},
                        }
                    }
                )
            elif isinstance(message.content, str):
                if message.content:
                    parts.append({"text": message.content})
            else:
                for part in message.content:
                    if part.type == "text" and part.text:
                        parts.append({"text": part.text})
                    elif part.type == "image" and part.image is not None:
                        if part.image.path:
                            media_type, data = encode_image_bytes(part.image)
                            parts.append(
                                {"inlineData": {"mimeType": media_type, "data": data}}
                            )
                        elif part.image.url and (
                            part.image.url.startswith("gs://")
                            or "generativelanguage.googleapis.com" in part.image.url
                        ):
                            # Gemini fileData only accepts Google-hosted URIs
                            # (GCS gs:// or the Files API host); it does not fetch
                            # arbitrary public URLs. Other remote URLs would be
                            # rejected at request time, so they are not sent here
                            # (supply a local path to inline them instead).
                            parts.append(
                                {
                                    "fileData": {
                                        "mimeType": part.image.media_type or "image/png",
                                        "fileUri": part.image.url,
                                    }
                                }
                            )
                    elif part.type == "audio" and part.audio is not None:
                        # Gemini accepts audio as inlineData (same envelope as an
                        # image), so a local clip becomes a multimodal input part.
                        if part.audio.path:
                            media_type, data = encode_audio_bytes(part.audio)
                            parts.append({"inlineData": {"mimeType": media_type, "data": data}})
                        elif part.audio.url and (
                            part.audio.url.startswith("gs://")
                            or "generativelanguage.googleapis.com" in part.audio.url
                        ):
                            parts.append(
                                {
                                    "fileData": {
                                        "mimeType": part.audio.media_type or "audio/wav",
                                        "fileUri": part.audio.url,
                                    }
                                }
                            )
                    elif part.type == "video" and part.video is not None:
                        # Gemini accepts native video input as inlineData (a local
                        # clip) or fileData (a Google-hosted URI) — the same envelope
                        # as image/audio — so a video ``ContentPart`` is a real
                        # multimodal input, not only an analyzed-into-frames surrogate.
                        if part.video.path:
                            media_type, data = encode_video_bytes(part.video)
                            parts.append({"inlineData": {"mimeType": media_type, "data": data}})
                        elif part.video.url and (
                            part.video.url.startswith("gs://")
                            or "generativelanguage.googleapis.com" in part.video.url
                        ):
                            parts.append(
                                {
                                    "fileData": {
                                        "mimeType": part.video.media_type or "video/mp4",
                                        "fileUri": part.video.url,
                                    }
                                }
                            )
            for tc in message.tool_calls:
                parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
            contents.append({"role": role, "parts": parts or [{"text": ""}]})
        return "\n\n".join(system_parts), contents

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        system_text, contents = self._render(request)
        payload: dict[str, Any] = {"contents": contents}
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        generation: dict[str, Any] = {}
        if request.temperature is not None:
            generation["temperature"] = request.temperature
        if request.top_p is not None:
            generation["topP"] = request.top_p
        if request.max_output_tokens is not None:
            generation["maxOutputTokens"] = request.max_output_tokens
        if request.stop:
            generation["stopSequences"] = request.stop
        if request.output_schema is not None:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = _strip_unsupported(request.output_schema)
        if (
            request.reasoning_effort is not None or request.thinking_budget_tokens is not None
        ) and self.capabilities(request.model).reasoning:
            budget = reasoning_budget_from_effort(
                request.reasoning_effort, request.thinking_budget_tokens
            )
            # includeThoughts surfaces a thought summary; thinkingBudget caps the
            # thinking tokens (billed at the output rate, see _parse_usage).
            generation["thinkingConfig"] = {"thinkingBudget": budget, "includeThoughts": True}
        if generation:
            payload["generationConfig"] = generation
        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": _strip_unsupported(tool.input_schema)
                            or {"type": "object", "properties": {}},
                        }
                        for tool in request.tools
                    ]
                }
            ]
        return payload

    # -- parsing ---------------------------------------------------------------

    def _parse_usage(self, usage: dict[str, Any] | None) -> TokenUsage:
        if not usage:
            return TokenUsage()
        thoughts = usage.get("thoughtsTokenCount", 0) or 0
        # Gemini reports thinking tokens separately from candidatesTokenCount but
        # bills them at the output rate (totalTokenCount includes them). Fold them
        # into the billable output so cost tracking doesn't treat thinking as free;
        # reasoning_tokens keeps the thinking subset for telemetry.
        return TokenUsage(
            input_tokens=usage.get("promptTokenCount", 0) or 0,
            output_tokens=(usage.get("candidatesTokenCount", 0) or 0) + thoughts,
            cached_input_tokens=usage.get("cachedContentTokenCount", 0) or 0,
            reasoning_tokens=thoughts,
        )

    def _parse_response(self, data: dict[str, Any], request: ModelRequest, latency_ms: int) -> ModelResponse:
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderResponseError(
                f"no candidates in response: {json.dumps(data)[:500]}",
                provider=self.name,
                retryable=True,
            )
        candidate = candidates[0]
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        for part in (candidate.get("content") or {}).get("parts") or []:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCallRequest(name=fc.get("name") or "", arguments=fc.get("args") or {})
                )
        finish_map = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
        }
        finish = finish_map.get(candidate.get("finishReason"), "stop")
        if tool_calls:
            finish = "tool_calls"
        text = "".join(text_parts)
        structured: dict[str, Any] | None = None
        if request.output_schema is not None and text:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None
        usage = self._parse_usage(data.get("usageMetadata"))
        return ModelResponse(
            model=data.get("modelVersion", request.model),
            text=text,
            tool_calls=tool_calls,
            structured=structured,
            finish_reason=finish,  # type: ignore[arg-type]
            usage=usage,
            cost_usd=self.price_table.cost(request.model, usage),
            latency_ms=latency_ms,
            provider=self.name,
            raw=data,
        )

    # -- API ---------------------------------------------------------------------

    def _content_path(self, request: ModelRequest, action: str) -> str:
        """generateContent / streamGenerateContent path. Overridden by Vertex
        for project/region-scoped publisher endpoints."""
        suffix = "?alt=sse" if action == "streamGenerateContent" else ""
        return f"/models/{request.model}:{action}{suffix}"

    async def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.monotonic()
        data = await self._post_json(
            self._content_path(request, "generateContent"), self._payload(request)
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        return self._parse_response(data, request, latency_ms)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        started = time.monotonic()
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        usage = TokenUsage()
        model_name = request.model
        async for line in self._post_stream(
            self._content_path(request, "streamGenerateContent"), self._payload(request)
        ):
            data_str = parse_sse_lines(line)
            if data_str is None:
                continue
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            model_name = chunk.get("modelVersion", model_name)
            if chunk.get("usageMetadata"):
                usage = self._parse_usage(chunk["usageMetadata"])
            for candidate in chunk.get("candidates") or []:
                for part in (candidate.get("content") or {}).get("parts") or []:
                    if part.get("text"):
                        text_parts.append(part["text"])
                        yield ModelEvent(type="text_delta", text=part["text"])
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        tool_call = ToolCallRequest(
                            name=fc.get("name") or "", arguments=fc.get("args") or {}
                        )
                        tool_calls.append(tool_call)
                        yield ModelEvent(type="tool_call_delta", tool_call=tool_call)
        yield ModelEvent(type="usage", usage=usage)
        text = "".join(text_parts)
        structured: dict[str, Any] | None = None
        if request.output_schema is not None and text:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None
        response = ModelResponse(
            model=model_name,
            text=text,
            tool_calls=tool_calls,
            structured=structured,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=usage,
            cost_usd=self.price_table.cost(request.model, usage),
            latency_ms=int((time.monotonic() - started) * 1000),
            provider=self.name,
        )
        yield ModelEvent(type="done", response=response)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        embedding_model = model or "gemini-embedding-001"
        data = await self._post_json(
            f"/models/{embedding_model}:batchEmbedContents",
            {
                "requests": [
                    {"model": f"models/{embedding_model}", "content": {"parts": [{"text": text}]}}
                    for text in texts
                ]
            },
        )
        return [item["values"] for item in data.get("embeddings") or []]

    @staticmethod
    def _parse_models_list(data: dict[str, Any]) -> list[ModelProfile]:
        """Map a Gemini ``ListModels`` payload onto sparse profiles (``models/x``
        names are stripped to bare ids; only generation-capable models kept)."""
        out: list[ModelProfile] = []
        for item in data.get("models") or []:
            name = item.get("name") or ""
            model_id = name.split("/", 1)[1] if name.startswith("models/") else name
            if not model_id:
                continue
            methods = item.get("supportedGenerationMethods") or []
            if methods and not any(
                m in methods for m in ("generateContent", "streamGenerateContent")
            ):
                continue  # skip embedding/other-only models
            out.append(ModelProfile(name=model_id, provider="google", model=model_id))
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
            structured_output=True,
            tool_calling=True,
            vision=True,
            audio=True,
            prompt_caching="flash" in model or "pro" in model,
            reasoning="2.5" in model or "gemini-3" in model,
            max_context_tokens=1_000_000,
            max_output_tokens=65_536,
            supports_system_message=True,
        )
