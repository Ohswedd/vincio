"""Local / self-hosted provider for OpenAI-compatible servers.

Covers vLLM, Ollama (``/v1``), llama.cpp server, LM Studio, and any other
endpoint speaking the OpenAI Chat Completions dialect. No API key required
by default; cost is zero unless a price table entry is configured.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.errors import ConfigError
from ..core.types import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from .base import ModelProvider
from .openai import OpenAIProvider

__all__ = ["LocalProvider", "GGUFProvider"]


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


def _message_text(content: Any) -> str:
    """Flatten a Message ``content`` (str or multimodal block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", "") or ""))
        return "\n".join(p for p in parts if p)
    return str(content or "")


class GGUFProvider(ModelProvider):
    """Native in-process GGUF / llama.cpp provider with on-device embedding.

    True offline inference for air-gapped and edge deployments — a quantized
    GGUF model runs in-process via ``llama-cpp-python``, no server and no
    network, behind the same :class:`~vincio.providers.base.ModelProvider`
    interface as every hosted provider. The same model serves embeddings
    (:meth:`embed`) when loaded with ``embedding=True``, so retrieval and
    generation share one on-device model. Inject a ``llama`` object for offline
    tests; otherwise install with ``pip install "vincio[gguf]"``.

    On-device adaptation: pass ``lora_path=`` (with an optional ``lora_scale``)
    to load a quantized LoRA adapter into the base model — the native side of the
    on-device fine-tuning loop. For the dependency-free, in-process adapter that
    works against *any* provider (including this one), see
    :class:`~vincio.optimize.local_adaptation.AdaptedProvider`.
    """

    name = "gguf"
    requires_api_key = False

    def __init__(
        self,
        model_path: str | None = None,
        *,
        llama: Any = None,
        n_ctx: int = 4096,
        embedding: bool = True,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self._embedding = embedding
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self._llama = llama

    def _ensure(self) -> Any:
        if self._llama is not None:
            return self._llama
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ConfigError(
                'the GGUF/llama.cpp provider requires: pip install "vincio[gguf]" '
                "(or inject a llama=... instance)"
            ) from exc
        if not self.model_path:
            raise ConfigError("GGUFProvider needs model_path=... pointing at a .gguf file")
        kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "embedding": self._embedding,
        }
        if self.lora_path:
            kwargs["lora_path"] = self.lora_path
            kwargs["lora_scale"] = self.lora_scale
        self._llama = Llama(**kwargs)
        return self._llama

    async def generate(self, request: ModelRequest) -> ModelResponse:
        llama = self._ensure()
        messages = [
            {"role": message.role, "content": _message_text(message.content)}
            for message in request.messages
        ]
        kwargs: dict[str, Any] = {"messages": messages}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: llama.create_chat_completion(**kwargs))
        choice = (result.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "") or ""
        usage = result.get("usage") or {}
        return ModelResponse(
            model=request.model or "gguf",
            text=text,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=TokenUsage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            ),
            provider=self.name,
            raw=result,
        )

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        llama = self._ensure()
        loop = asyncio.get_running_loop()

        def _run() -> list[list[float]]:
            out: list[list[float]] = []
            for text in texts:
                vector = llama.embed(text)
                out.append([float(x) for x in vector])
            return out

        return await loop.run_in_executor(None, _run)

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(
            structured_output=False,
            tool_calling=False,
            vision=False,
            audio=False,
            prompt_caching=False,
            max_context_tokens=self.n_ctx,
            max_output_tokens=4_096,
            supports_system_message=True,
        )
