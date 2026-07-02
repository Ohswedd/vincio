"""DS4 local-inference provider — online inference against your own DeepSeek V4 box.

`DS4 <https://github.com/antirez/ds4>`_ is antirez's self-contained inference
engine for DeepSeek V4 (Flash and PRO) — C / CUDA / Metal, Metal on Apple silicon
and CUDA / ROCm on Linux, with no GGML link. A running ``ds4-server`` serves an
**OpenAI- and Anthropic-compatible** HTTP API (``/v1/chat/completions``,
``/v1/completions``, ``/v1/responses``, ``/v1/messages``, ``/v1/models``) on
``127.0.0.1:8000``, with SSE streaming, tool calling (DeepSeek DSML converted
to/from OpenAI/Anthropic JSON server-side), thinking / non-thinking generation,
and a disk-backed KV cache.

Because DS4 speaks OpenAI, the one-line path is the ``ds4`` **preset** in
:data:`~vincio.providers.openai_compat.PRESETS` (``openai_compatible("ds4")`` —
``http://127.0.0.1:8000/v1``, no API key). On top of it sits this first-class
:class:`Ds4Provider`, which subclasses
:class:`~vincio.providers.openai_compat.OpenAICompatibleProvider` (the way the
enterprise providers subclass the base) and expresses the DeepSeek-specific
capabilities the generic passthrough cannot:

* **Thinking modes → the reasoning controller.** DS4's thinking / non-thinking
  path is driven by Vincio's existing
  :class:`~vincio.agents.reasoning.ReasoningController`: ``reasoning_effort`` and
  ``thinking_budget_tokens`` map onto DS4's reasoning generation (the effort level
  is passed through on the OpenAI surface and the thinking budget is carried on the
  DeepSeek chat-template control), so test-time-compute orchestration drives a
  local DeepSeek V4 natively. Thinking is turned **off** explicitly for a plain
  request, so non-thinking generation is a first-class mode, not an accident.
* **Disk KV cache ↔ the cache-aware prompt layout.** The provider advertises
  ``prompt_caching``, and the prompt compiler's stable-prefix layout — already
  built to maximize cache hits — lines up with DS4's session KV reuse. DS4 reports
  its disk-KV hits as ``prompt_cache_hit_tokens``; :meth:`_parse_usage` folds that
  into ``cached_input_tokens`` so
  :func:`~vincio.providers.cache_strategy.cache_hit_rate` and the cost report
  account for it, and Vincio's prefix ordering *raises* DS4's hit rate.
* **The model catalog, honestly $0.** ``deepseek-v4-flash`` / ``deepseek-v4-pro``
  and the ``-q4`` quant variants register in ``model_catalog.json`` under the
  ``ds4`` provider key, priced at ``$0`` with an explicit ``self_hosted`` flag plus
  a tier that drives the energy/carbon accounting. ``ds4`` joins ``local`` /
  ``mock`` in the registry's free-providers set, so the coverage gate treats a
  self-hosted $0 as correct rather than a drift bug.
* **Residency, fail-closed.** A localhost endpoint resolves to the ``on_prem``
  region, so a residency policy that includes ``on_prem`` admits DS4 and one that
  does not refuses egress — client-side, before any request leaves the process.

Offline-first, like every provider: :class:`Ds4Provider` is testable via the
deterministic mock and injected transport (no DS4 binary in CI), and it adds **no
hard dependency** — it is plain HTTP over the existing ``HTTPProvider`` transport,
so there is no new extra. The live path simply points at a running ``ds4-server``
(``VINCIO_PROVIDER=ds4``).

DS4 also exposes the Anthropic Messages dialect at ``/v1/messages``; to reach it,
point the :class:`~vincio.providers.anthropic.AnthropicProvider` at the same box
(``build_provider("anthropic", base_url="http://127.0.0.1:8000")``). This provider
uses the OpenAI Chat Completions surface, which is the recommended path.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import ConfigError
from ..core.types import ModelCapabilities, ModelRequest, TokenUsage
from .base import reasoning_budget_from_effort
from .openai_compat import OpenAICompatibleProvider

__all__ = ["Ds4Provider"]


class Ds4Provider(OpenAICompatibleProvider):
    """First-class provider for a running ``ds4-server`` (DeepSeek V4, self-hosted).

    Behaves like :class:`OpenAICompatibleProvider` (tools, structured output,
    streaming) but defaults to the DS4 endpoint, needs no API key, wires the
    reasoning controller into DS4's thinking mode, and accounts for DS4's disk-KV
    cache. Register it via ``build_provider("ds4")`` /
    ``ContextApp(provider="ds4")`` / ``VINCIO_PROVIDER=ds4``.
    """

    name = "ds4"
    default_base_url = "http://127.0.0.1:8000/v1"
    requires_api_key = False
    # DS4 serves DeepSeek V4 generation, not embeddings — there is no canonical
    # embedding model, so :meth:`embed` refuses rather than 404-ing at call time.
    default_embedding_model = None  # type: ignore[assignment]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # OpenAICompatibleProvider requires an explicit base_url (there is no
        # canonical default for "any endpoint"); DS4 has one, so supply it.
        kwargs.setdefault("base_url", self.default_base_url)
        super().__init__(*args, **kwargs)

    def _headers(self) -> dict[str, str]:
        # A DS4 box is keyless by default; only attach a bearer token if the
        # operator put one in front of it (a reverse proxy).
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, request: ModelRequest, *, stream: bool = False) -> dict[str, Any]:
        payload = super()._payload(request, stream=stream)
        # A self-hosted server need not implement OpenAI's strict json_schema flag;
        # drop it so structured output degrades to best-effort rather than 400-ing.
        response_format = payload.get("response_format")
        if response_format and response_format.get("type") == "json_schema":
            response_format["json_schema"].pop("strict", None)
        # Thinking / non-thinking generation. The base OpenAI payload already
        # carries ``reasoning_effort`` when the model is reasoning-capable; here we
        # add DS4's DeepSeek chat-template thinking switch (explicitly on or off)
        # and carry the reasoning controller's token budget when one is set.
        if self.capabilities(request.model).reasoning:
            want_thinking = (
                request.reasoning_effort is not None
                or request.thinking_budget_tokens is not None
            )
            template_kwargs = dict(payload.get("chat_template_kwargs") or {})
            template_kwargs["thinking"] = want_thinking
            if want_thinking:
                template_kwargs["thinking_budget"] = reasoning_budget_from_effort(
                    request.reasoning_effort, request.thinking_budget_tokens
                )
            payload["chat_template_kwargs"] = template_kwargs
        return payload

    def _parse_usage(self, usage: dict[str, Any] | None) -> TokenUsage:
        parsed = super()._parse_usage(usage)
        # DS4's disk-backed KV cache reports reused prefill as
        # ``prompt_cache_hit_tokens`` (the DeepSeek convention) rather than the
        # OpenAI ``prompt_tokens_details.cached_tokens``. Fold it into
        # ``cached_input_tokens`` so the cache-hit-rate telemetry and the cost/energy
        # report see the KV reuse the stable-prefix layout earned.
        if usage and not parsed.cached_input_tokens:
            hit = usage.get("prompt_cache_hit_tokens")
            if hit:
                parsed = parsed.model_copy(update={"cached_input_tokens": int(hit)})
        return parsed

    def capabilities(self, model: str) -> ModelCapabilities:
        from .registry import default_model_registry

        profile = default_model_registry().resolve(model)
        if profile is not None:
            return profile.capabilities
        # Fallback for an unregistered DS4 model id (a custom quant): DeepSeek V4 is
        # a reasoning, tool-calling, structured-output, text-only model with a large
        # context window and disk-KV prompt caching.
        return ModelCapabilities(
            structured_output=True,
            tool_calling=True,
            vision=False,
            audio=False,
            prompt_caching=True,
            reasoning=True,
            max_context_tokens=163_840,
            max_output_tokens=65_536,
            supports_system_message=True,
        )

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise ConfigError(
            "the DS4 provider serves DeepSeek V4 generation, not embeddings "
            "(ds4-server exposes no /v1/embeddings endpoint); use a dedicated "
            "embedder for retrieval vectors — a local fastembed model, or any "
            "OpenAI-compatible embedding endpoint via openai_compatible(...)"
        )
