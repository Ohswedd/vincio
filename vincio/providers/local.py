"""Local / self-hosted provider for OpenAI-compatible servers.

Covers vLLM, Ollama (``/v1``), llama.cpp server, LM Studio, and any other
endpoint speaking the OpenAI Chat Completions dialect. No API key required
by default; cost is zero unless a price table entry is configured.
"""

from __future__ import annotations

from typing import Any

from ..core.types import ModelCapabilities, ModelRequest
from .openai import OpenAIProvider

__all__ = ["LocalProvider"]


class LocalProvider(OpenAIProvider):
    name = "local"
    default_base_url = "http://localhost:11434/v1"  # Ollama default; override for vLLM etc.
    requires_api_key = False

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, request: ModelRequest, *, stream: bool = False) -> dict[str, Any]:
        payload = super()._payload(request, stream=stream)
        # Many local servers reject stream_options / strict json_schema extras.
        payload.pop("stream_options", None)
        response_format = payload.get("response_format")
        if response_format and response_format.get("type") == "json_schema":
            response_format["json_schema"].pop("strict", None)
        return payload

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(
            structured_output=True,
            tool_calling=True,
            vision=False,
            audio=False,
            prompt_caching=False,
            max_context_tokens=32_768,
            max_output_tokens=8_192,
            supports_system_message=True,
        )
