"""Vincio providers: provider-neutral model execution."""

from __future__ import annotations

from ..core.config import ProviderConfig
from ..core.errors import ConfigError
from .anthropic import AnthropicProvider
from .base import (
    AuthStrategy,
    FailoverChain,
    HTTPProvider,
    ModelProvider,
    ProviderRegistry,
    RetryingProvider,
    is_lifecycle_error,
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
    GoogleBatchBackend,
    InProcessBatchBackend,
    OpenAIBatchBackend,
)
from .cache_strategy import PromptCacheStrategy, cache_hit_rate
from .capabilities import (
    CapabilityVerdict,
    RequestNeeds,
    capability_check,
    requirements_for,
)
from .circuit import CircuitBreaker, CircuitState, HealthAwareFailover
from .discovery import discover_models
from .enterprise import (
    AzureKeyAuth,
    AzureOpenAIProvider,
    BearerTokenAuth,
    BedrockProvider,
    SigV4Auth,
    VertexProvider,
)
from .google import GoogleProvider
from .keypool import KeyPool, RateLimiter
from .lifecycle import LifecycleAlert, LifecycleWatcher, MigrationProposal
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
from .registry import (
    ModelRegistry,
    ModelUnknownWarning,
    default_model_registry,
    discover_entry_points,
)
from .shadow import CanaryRouter, CanaryState, ShadowObservation, ShadowProvider
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
    "GoogleBatchBackend",
    "CoalescingProvider",
    "build_pooled_client",
    "ProviderRegistry",
    "ModelRegistry",
    "ModelUnknownWarning",
    "default_model_registry",
    "discover_entry_points",
    "discover_models",
    # capability guard + rotation (1.8)
    "RequestNeeds",
    "CapabilityVerdict",
    "requirements_for",
    "capability_check",
    "is_lifecycle_error",
    "ShadowProvider",
    "ShadowObservation",
    "CanaryRouter",
    "CanaryState",
    "LifecycleWatcher",
    "LifecycleAlert",
    "MigrationProposal",
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
    "AuthStrategy",
    "BedrockProvider",
    "VertexProvider",
    "AzureOpenAIProvider",
    "SigV4Auth",
    "AzureKeyAuth",
    "BearerTokenAuth",
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
# 2.0: enterprise deployment endpoints behind the pluggable AuthStrategy —
# routed through the same registry, capability guards, swap gate, residency, and
# audit chain as every other provider. Bedrock requires (region + AWS creds),
# Vertex requires (project + access token), Azure requires (endpoint + key/AAD).
_registry.register("bedrock", BedrockProvider)
_registry.register("vertex", VertexProvider)
_registry.register("azure", AzureOpenAIProvider)
for _preset in PRESETS:
    _registry.register(_preset, _preset_factory(_preset))

# Third-party providers shipped as separate pip packages auto-register via the
# ``vincio.providers`` entry-point group (importlib.metadata). Built-ins take
# precedence: a plugin only fills a name not already registered, so an installed
# adapter can never silently shadow a core provider.
for _name, _factory in discover_entry_points("vincio.providers").items():
    if _name not in _registry.names:
        _registry.register(_name, _factory)


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
