"""Vincio providers: provider-neutral model execution."""

from __future__ import annotations

from ..core.config import ProviderConfig
from ..core.errors import ConfigError
from .anthropic import AnthropicProvider
from .base import (
    FailoverChain,
    HTTPProvider,
    ModelProvider,
    ProviderRegistry,
    RetryingProvider,
)
from .batch import (
    AnthropicBatchBackend,
    BatchBackend,
    BatchJob,
    BatchRequest,
    BatchResult,
    BatchRunner,
    BatchRunResult,
    BatchStatus,
    InProcessBatchBackend,
    OpenAIBatchBackend,
)
from .cache_strategy import PromptCacheStrategy, cache_hit_rate
from .circuit import CircuitBreaker, CircuitState, HealthAwareFailover
from .google import GoogleProvider
from .keypool import KeyPool, RateLimiter
from .local import LocalProvider
from .mistral import MistralProvider
from .mock import MockProvider, instance_from_schema
from .openai import OpenAIProvider
from .openai_compat import (
    PRESETS,
    OpenAICompatibleProvider,
    OpenAICompatPreset,
    _preset_factory,
    openai_compatible,
)
from .openai_responses import OpenAIResponsesProvider
from .transport import CoalescingProvider, build_pooled_client

__all__ = [
    "ModelProvider",
    "HTTPProvider",
    "RetryingProvider",
    "FailoverChain",
    "CircuitBreaker",
    "CircuitState",
    "HealthAwareFailover",
    "KeyPool",
    "RateLimiter",
    "PromptCacheStrategy",
    "cache_hit_rate",
    "BatchRunner",
    "BatchBackend",
    "BatchRequest",
    "BatchResult",
    "BatchJob",
    "BatchRunResult",
    "BatchStatus",
    "InProcessBatchBackend",
    "OpenAIBatchBackend",
    "AnthropicBatchBackend",
    "CoalescingProvider",
    "build_pooled_client",
    "ProviderRegistry",
    "OpenAIProvider",
    "OpenAIResponsesProvider",
    "OpenAICompatibleProvider",
    "OpenAICompatPreset",
    "openai_compatible",
    "PRESETS",
    "AnthropicProvider",
    "GoogleProvider",
    "MistralProvider",
    "LocalProvider",
    "MockProvider",
    "instance_from_schema",
    "default_registry",
    "build_provider",
]

_registry = ProviderRegistry()
_registry.register("openai", OpenAIProvider)
_registry.register("openai_responses", OpenAIResponsesProvider)
_registry.register("anthropic", AnthropicProvider)
_registry.register("google", GoogleProvider)
_registry.register("gemini", GoogleProvider)
_registry.register("mistral", MistralProvider)
_registry.register("local", LocalProvider)
_registry.register("ollama", LocalProvider)
_registry.register("vllm", lambda **kw: LocalProvider(base_url=kw.pop("base_url", "http://localhost:8000/v1"), **kw))
_registry.register("mock", lambda **kw: MockProvider(**{k: v for k, v in kw.items() if k not in ("api_key", "base_url", "timeout_s")}))
# OpenAI-compatible passthrough: a generic adapter plus named hosted-gateway
# presets (groq, together, fireworks, openrouter, deepseek, perplexity, xai,
# nvidia). Their API keys resolve from the conventional <NAME>_API_KEY env var.
_registry.register("openai_compat", OpenAICompatibleProvider)
for _preset in PRESETS:
    _registry.register(_preset, _preset_factory(_preset))


def default_registry() -> ProviderRegistry:
    return _registry


def build_provider(
    name: str,
    config: ProviderConfig | None = None,
    *,
    with_retries: bool = True,
    **overrides,
) -> ModelProvider:
    """Construct a provider from config with retry wrapping."""
    config = config or ProviderConfig()
    kwargs: dict = {}
    if name != "mock":
        kwargs["api_key"] = overrides.pop("api_key", None) or config.resolve_api_key(name)
        base_url = overrides.pop("base_url", None) or config.base_urls.get(name)
        if base_url:
            kwargs["base_url"] = base_url
        kwargs["timeout_s"] = overrides.pop("timeout_s", None) or config.timeout_s
    kwargs.update(overrides)
    try:
        provider = _registry.create(name, **kwargs)
    except TypeError as exc:
        raise ConfigError(f"invalid arguments for provider {name!r}: {exc}") from exc
    if with_retries and config.max_retries > 0:
        return RetryingProvider(provider, max_retries=config.max_retries)
    return provider
