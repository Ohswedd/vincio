"""Mistral provider. The Mistral chat API is OpenAI-compatible, so this
adapter specializes the OpenAI provider with Mistral's endpoint, schema
handling, and capability matrix."""

from __future__ import annotations

from typing import Any

from ..core.types import ModelCapabilities, ModelRequest
from .openai import OpenAIProvider

__all__ = ["MistralProvider"]


class MistralProvider(OpenAIProvider):
    name = "mistral"
    default_base_url = "https://api.mistral.ai/v1"
    default_embedding_model = "mistral-embed"

    def _payload(self, request: ModelRequest, *, stream: bool = False) -> dict[str, Any]:
        payload = super()._payload(request, stream=stream)
        # Mistral uses max_tokens, not max_completion_tokens.
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        # Mistral's json_schema mode uses the same response_format shape.
        payload.pop("stream_options", None)
        if "seed" in payload:
            payload["random_seed"] = payload.pop("seed")
        return payload

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(
            structured_output=True,
            tool_calling=True,
            vision="pixtral" in model,
            audio=False,
            prompt_caching=False,
            max_context_tokens=131_072,
            max_output_tokens=8_192,
            supports_system_message=True,
        )
