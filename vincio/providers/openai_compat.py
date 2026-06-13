"""OpenAI-compatible passthrough provider.

Any endpoint that speaks the OpenAI Chat Completions dialect works through
:class:`OpenAICompatibleProvider` — point it at a ``base_url`` and (optionally)
pass an API key. Named presets cover the popular hosted gateways so you can
``build_provider("groq")`` without remembering base URLs::

    from vincio.providers import openai_compatible

    groq = openai_compatible("groq")                       # GROQ_API_KEY
    together = openai_compatible("together")               # TOGETHER_API_KEY
    custom = openai_compatible(base_url="https://my-gw/v1", api_key="...")

Presets only set the base URL and the conventional API-key env var; the model
name is always yours to choose. Endpoints that do not implement ``/embeddings``
(most chat-only gateways) raise at embed time, exactly like a real call would.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.errors import ConfigError
from .openai import OpenAIProvider

__all__ = ["OpenAICompatibleProvider", "OpenAICompatPreset", "PRESETS", "openai_compatible"]


@dataclass(frozen=True)
class OpenAICompatPreset:
    """A named OpenAI-compatible endpoint."""

    base_url: str
    env_key: str
    requires_api_key: bool = True
    embedding_model: str | None = None


# Stable, OpenAI-Chat-Completions-compatible hosted gateways. The conventional
# env var (``<NAME>_API_KEY``) is also what ``ProviderConfig.resolve_api_key``
# falls back to, so no extra wiring is needed to pick keys up from the env.
PRESETS: dict[str, OpenAICompatPreset] = {
    "groq": OpenAICompatPreset("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": OpenAICompatPreset(
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
        embedding_model="togethercomputer/m2-bert-80M-8k-retrieval",
    ),
    "fireworks": OpenAICompatPreset(
        "https://api.fireworks.ai/inference/v1",
        "FIREWORKS_API_KEY",
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
    ),
    "openrouter": OpenAICompatPreset("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": OpenAICompatPreset("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "perplexity": OpenAICompatPreset("https://api.perplexity.ai", "PERPLEXITY_API_KEY"),
    "xai": OpenAICompatPreset("https://api.x.ai/v1", "XAI_API_KEY"),
    "nvidia": OpenAICompatPreset(
        "https://integrate.api.nvidia.com/v1",
        "NVIDIA_API_KEY",
        embedding_model="nvidia/nv-embedqa-e5-v5",
    ),
}


class OpenAICompatibleProvider(OpenAIProvider):
    """OpenAI Chat-Completions provider for any compatible endpoint.

    Behaves exactly like :class:`OpenAIProvider` (tools, structured output,
    streaming, embeddings) but lets you set the provider ``name`` (so traces,
    costs, and failover labels read sensibly) and the default embedding model.
    A ``base_url`` is required — there is no canonical default for "any
    endpoint".
    """

    name = "openai_compat"

    def __init__(
        self,
        *args: Any,
        name: str | None = None,
        default_embedding_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        # base_url is keyword-only on HTTPProvider, and this class inherits
        # OpenAIProvider's default base URL — so a missing base_url must error
        # rather than silently point at api.openai.com.
        if not kwargs.get("base_url"):
            raise ConfigError(
                "OpenAICompatibleProvider requires base_url (or use a preset, "
                f"e.g. openai_compatible('groq'); known presets: {sorted(PRESETS)})"
            )
        super().__init__(*args, **kwargs)
        if name:
            self.name = name
        if default_embedding_model:
            self.default_embedding_model = default_embedding_model


def openai_compatible(
    preset: str | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> OpenAICompatibleProvider:
    """Construct an OpenAI-compatible provider from a preset or a raw base URL.

    With a ``preset`` the base URL and (when ``api_key`` is omitted) the API
    key are resolved from the conventional env var. Otherwise pass ``base_url``
    directly.
    """
    import os

    if preset is not None:
        if preset not in PRESETS:
            raise ConfigError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
        spec = PRESETS[preset]
        return OpenAICompatibleProvider(
            api_key=api_key or os.environ.get(spec.env_key),
            base_url=base_url or spec.base_url,
            name=preset,
            default_embedding_model=spec.embedding_model,
            **kwargs,
        )
    if not base_url:
        raise ConfigError("openai_compatible() needs a preset or a base_url")
    return OpenAICompatibleProvider(api_key=api_key, base_url=base_url, **kwargs)


def _preset_factory(preset_name: str):
    """Build a registry factory for a preset (used by the provider registry)."""
    spec = PRESETS[preset_name]

    def factory(**kwargs: Any) -> OpenAICompatibleProvider:
        kwargs.setdefault("base_url", spec.base_url)
        return OpenAICompatibleProvider(
            name=preset_name, default_embedding_model=spec.embedding_model, **kwargs
        )

    return factory
