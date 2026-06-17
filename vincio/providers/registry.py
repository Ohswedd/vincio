"""Data-driven model registry (1.7).

A versioned, hot-reloadable, config-overridable catalog keyed by *exact* model
id. Each entry is a :class:`~vincio.core.types.ModelProfile` binding
capabilities, pricing (standard + batch tiers), context window, modalities, and
GA / deprecation / retirement lifecycle dates.

The registry is the single source of truth the rest of the spine reads from:

* :meth:`ModelRegistry.capabilities` replaces per-provider substring sniffing
  (demoted to a last-resort fallback inside each provider).
* :class:`vincio.observability.costs.PriceTable` derives its prices from it, so
  an unknown model warns and emits ``model.unknown`` instead of silently
  costing ``$0``.
* (1.8) capability guards, the cost/latency router, and the lifecycle watcher
  all consult it.

It is plain data and ships in-process — no network, no hosted dependency. Third
parties extend it by shipping their own pip packages exposing the
``vincio.providers`` / ``vincio.embedders`` / ``vincio.stores`` entry-point
groups (see :func:`discover_entry_points`), or by pointing
``VINCIO_MODEL_REGISTRY`` at a JSON/YAML overlay merged over the built-ins.
"""

from __future__ import annotations

import os
import warnings
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from ..core.types import ModelCapabilities, ModelLifecycle, ModelProfile

__all__ = [
    "REGISTRY_VERSION",
    "ModelUnknownWarning",
    "ModelRegistry",
    "default_model_registry",
    "discover_entry_points",
]

# Bumped whenever the built-in catalog's data shape or contents change in a way
# consumers may want to detect (independent of the package SemVer).
REGISTRY_VERSION = "2026.06"


class ModelUnknownWarning(UserWarning):
    """Emitted once per process per unknown model id resolved against the registry."""


def _caps(**kwargs: Any) -> ModelCapabilities:
    return ModelCapabilities(**kwargs)


# ---------------------------------------------------------------------------
# Built-in catalog — plain data. Kept consistent with the prices the cost
# tracker shipped (observability/costs.py) and the capabilities each provider
# previously guessed by substring, now stated explicitly per exact id.
# Lifecycle dates are populated from provider-published data; left ``None`` when
# a provider publishes no date (honest by default — 1.8 fills them from the
# live model-list endpoints).
# ---------------------------------------------------------------------------

_OPENAI_CHAT = dict(structured_output=True, tool_calling=True, prompt_caching=True,
                     supports_system_message=True, supports_developer_message=True,
                     input_modalities=["text", "image"])
_ANTHROPIC_CHAT = dict(structured_output=True, tool_calling=True, vision=True,
                       prompt_caching=True, max_context_tokens=200_000,
                       supports_system_message=True, supports_developer_message=False,
                       input_modalities=["text", "image"])
_GOOGLE_CHAT = dict(structured_output=True, tool_calling=True, vision=True, audio=True,
                    max_context_tokens=1_000_000, max_output_tokens=65_536,
                    supports_system_message=True,
                    input_modalities=["text", "image", "audio"])
_MISTRAL_CHAT = dict(structured_output=True, tool_calling=True, max_context_tokens=131_072,
                     max_output_tokens=8_192, supports_system_message=True)


def _builtin_catalog() -> list[ModelProfile]:
    return [
        # ---- OpenAI ----
        ModelProfile(name="gpt-5.2", provider="openai", model="gpt-5.2", tier="strong",
                     capabilities=_caps(**_OPENAI_CHAT, vision=True, reasoning=True,
                                        max_context_tokens=272_000, max_output_tokens=32_768),
                     input_cost_per_mtok=1.25, output_cost_per_mtok=10.0,
                     cached_input_cost_per_mtok=0.125,
                     batch_input_cost_per_mtok=0.625, batch_output_cost_per_mtok=5.0),
        ModelProfile(name="gpt-5.2-mini", provider="openai", model="gpt-5.2-mini", tier="default",
                     capabilities=_caps(**_OPENAI_CHAT, vision=True, reasoning=True,
                                        max_context_tokens=272_000, max_output_tokens=16_384),
                     input_cost_per_mtok=0.25, output_cost_per_mtok=2.0,
                     cached_input_cost_per_mtok=0.025,
                     batch_input_cost_per_mtok=0.125, batch_output_cost_per_mtok=1.0),
        ModelProfile(name="gpt-5.2-nano", provider="openai", model="gpt-5.2-nano", tier="fast",
                     capabilities=_caps(**_OPENAI_CHAT, vision=True, reasoning=True,
                                        max_context_tokens=272_000, max_output_tokens=16_384),
                     input_cost_per_mtok=0.05, output_cost_per_mtok=0.4,
                     cached_input_cost_per_mtok=0.005,
                     batch_input_cost_per_mtok=0.025, batch_output_cost_per_mtok=0.2),
        ModelProfile(name="gpt-4o", provider="openai", model="gpt-4o", tier="default",
                     capabilities=_caps(**_OPENAI_CHAT, vision=True, reasoning=False,
                                        max_context_tokens=128_000, max_output_tokens=32_768),
                     input_cost_per_mtok=2.5, output_cost_per_mtok=10.0,
                     cached_input_cost_per_mtok=1.25,
                     batch_input_cost_per_mtok=1.25, batch_output_cost_per_mtok=5.0),
        ModelProfile(name="gpt-4o-mini", provider="openai", model="gpt-4o-mini", tier="fast",
                     capabilities=_caps(**_OPENAI_CHAT, vision=True, reasoning=False,
                                        max_context_tokens=128_000, max_output_tokens=16_384),
                     input_cost_per_mtok=0.15, output_cost_per_mtok=0.6,
                     cached_input_cost_per_mtok=0.075,
                     batch_input_cost_per_mtok=0.075, batch_output_cost_per_mtok=0.3),
        # ---- Anthropic ----
        ModelProfile(name="claude-fable-5", provider="anthropic", model="claude-fable-5",
                     tier="strong",
                     capabilities=_caps(**_ANTHROPIC_CHAT, reasoning=True, max_output_tokens=64_000),
                     input_cost_per_mtok=5.0, output_cost_per_mtok=25.0,
                     cached_input_cost_per_mtok=0.5,
                     batch_input_cost_per_mtok=2.5, batch_output_cost_per_mtok=12.5),
        ModelProfile(name="claude-opus-4-8", provider="anthropic", model="claude-opus-4-8",
                     tier="strong",
                     capabilities=_caps(**_ANTHROPIC_CHAT, reasoning=True, max_output_tokens=64_000),
                     input_cost_per_mtok=5.0, output_cost_per_mtok=25.0,
                     cached_input_cost_per_mtok=0.5,
                     batch_input_cost_per_mtok=2.5, batch_output_cost_per_mtok=12.5),
        ModelProfile(name="claude-sonnet-4-6", provider="anthropic", model="claude-sonnet-4-6",
                     tier="default",
                     capabilities=_caps(**_ANTHROPIC_CHAT, reasoning=True, max_output_tokens=64_000),
                     input_cost_per_mtok=3.0, output_cost_per_mtok=15.0,
                     cached_input_cost_per_mtok=0.3,
                     batch_input_cost_per_mtok=1.5, batch_output_cost_per_mtok=7.5),
        ModelProfile(name="claude-haiku-4-5", provider="anthropic", model="claude-haiku-4-5",
                     tier="fast",
                     capabilities=_caps(**_ANTHROPIC_CHAT, reasoning=False, max_output_tokens=32_000),
                     input_cost_per_mtok=1.0, output_cost_per_mtok=5.0,
                     cached_input_cost_per_mtok=0.1,
                     batch_input_cost_per_mtok=0.5, batch_output_cost_per_mtok=2.5),
        # ---- Google ----
        ModelProfile(name="gemini-3-pro", provider="google", model="gemini-3-pro", tier="strong",
                     capabilities=_caps(**_GOOGLE_CHAT, prompt_caching=True, reasoning=True),
                     input_cost_per_mtok=2.0, output_cost_per_mtok=12.0,
                     cached_input_cost_per_mtok=0.5),
        ModelProfile(name="gemini-3-flash", provider="google", model="gemini-3-flash",
                     tier="default",
                     capabilities=_caps(**_GOOGLE_CHAT, prompt_caching=True, reasoning=True),
                     input_cost_per_mtok=0.3, output_cost_per_mtok=2.5,
                     cached_input_cost_per_mtok=0.075),
        ModelProfile(name="gemini-2.5-pro", provider="google", model="gemini-2.5-pro", tier="strong",
                     capabilities=_caps(**_GOOGLE_CHAT, prompt_caching=True, reasoning=True),
                     input_cost_per_mtok=1.25, output_cost_per_mtok=10.0,
                     cached_input_cost_per_mtok=0.31),
        ModelProfile(name="gemini-2.5-flash", provider="google", model="gemini-2.5-flash",
                     tier="default",
                     capabilities=_caps(**_GOOGLE_CHAT, prompt_caching=True, reasoning=True),
                     input_cost_per_mtok=0.3, output_cost_per_mtok=2.5,
                     cached_input_cost_per_mtok=0.075),
        ModelProfile(name="gemini-2.5-flash-lite", provider="google",
                     model="gemini-2.5-flash-lite", tier="fast",
                     capabilities=_caps(**_GOOGLE_CHAT, prompt_caching=True, reasoning=True),
                     input_cost_per_mtok=0.1, output_cost_per_mtok=0.4,
                     cached_input_cost_per_mtok=0.025),
        ModelProfile(name="gemini-2.0-flash", provider="google", model="gemini-2.0-flash",
                     tier="fast", successor="gemini-2.5-flash",
                     capabilities=_caps(**_GOOGLE_CHAT, prompt_caching=True, reasoning=False),
                     input_cost_per_mtok=0.1, output_cost_per_mtok=0.4,
                     cached_input_cost_per_mtok=0.025),
        ModelProfile(name="gemini-2.0-flash-lite", provider="google",
                     model="gemini-2.0-flash-lite", tier="fast",
                     successor="gemini-2.5-flash-lite",
                     capabilities=_caps(**_GOOGLE_CHAT, prompt_caching=True, reasoning=False),
                     input_cost_per_mtok=0.075, output_cost_per_mtok=0.3),
        # ---- Embedding models (cost-tracking entries; capabilities minimal) ----
        ModelProfile(name="gemini-embedding-001", provider="google",
                     model="gemini-embedding-001", tier="default",
                     capabilities=_caps(max_context_tokens=2_048, max_output_tokens=0,
                                        output_modalities=["embedding"]),
                     input_cost_per_mtok=0.15),
        ModelProfile(name="text-embedding-004", provider="google", model="text-embedding-004",
                     tier="default",
                     capabilities=_caps(max_context_tokens=2_048, max_output_tokens=0,
                                        output_modalities=["embedding"]),
                     input_cost_per_mtok=0.0),
        # ---- Mistral ----
        ModelProfile(name="mistral-large-latest", provider="mistral", model="mistral-large-latest",
                     tier="strong", capabilities=_caps(**_MISTRAL_CHAT, vision=False),
                     input_cost_per_mtok=2.0, output_cost_per_mtok=6.0),
        ModelProfile(name="mistral-small-latest", provider="mistral", model="mistral-small-latest",
                     tier="fast", capabilities=_caps(**_MISTRAL_CHAT, vision=False),
                     input_cost_per_mtok=0.2, output_cost_per_mtok=0.6),
        # ---- Local / self-hosted (free) ----
        ModelProfile(name="local", provider="local", model="local", tier="default",
                     capabilities=_caps(structured_output=True, tool_calling=True,
                                        max_context_tokens=32_768, max_output_tokens=8_192)),
        # ---- Deterministic offline mock (free; capabilities mirror MockProvider) ----
        ModelProfile(name="mock", provider="mock", model="mock", tier="default",
                     capabilities=_caps(structured_output=True, tool_calling=True, vision=True,
                                        audio=True, prompt_caching=True, max_context_tokens=200_000,
                                        max_output_tokens=32_768, supports_developer_message=True,
                                        input_modalities=["text", "image", "audio"])),
    ]


class ModelRegistry:
    """A catalog of :class:`ModelProfile` keyed by exact model id.

    Lookups are exact-first, then alias, then a demoted longest-prefix fallback
    (so dated snapshots like ``gpt-4o-2024-11-20`` still resolve). A genuinely
    unknown id warns once via :class:`ModelUnknownWarning` and resolves to
    ``None`` rather than silently behaving like a known, free model.
    """

    def __init__(
        self, profiles: list[ModelProfile] | None = None, *, version: str = REGISTRY_VERSION
    ) -> None:
        self.version = version
        self._profiles: dict[str, ModelProfile] = {}
        self._aliases: dict[str, str] = {}
        self._seen_unknown: set[str] = set()
        for profile in profiles if profiles is not None else _builtin_catalog():
            self.register(profile)

    # -- mutation --------------------------------------------------------------

    def register(self, profile: ModelProfile) -> None:
        """Add or replace a profile (and its aliases)."""
        self._profiles[profile.model] = profile
        for alias in profile.aliases:
            self._aliases[alias] = profile.model

    def override(self, profiles: list[ModelProfile] | dict[str, Any]) -> None:
        """Merge user/config overrides over the built-ins (last write wins)."""
        items = profiles.values() if isinstance(profiles, dict) else profiles
        for item in items:
            profile = item if isinstance(item, ModelProfile) else ModelProfile.model_validate(item)
            self.register(profile)

    def load_file(self, path: str | Path) -> int:
        """Hot-load an overlay catalog from a JSON or YAML file; returns count merged."""
        p = Path(path)
        if not p.is_file():
            return 0
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text) or []
        else:
            import json

            data = json.loads(text)
        entries = data.get("models", data) if isinstance(data, dict) else data
        before = len(self._profiles)
        self.override(list(entries))
        return len(self._profiles) - before + (0 if len(self._profiles) > before else len(entries))

    def reload(self) -> None:
        """Reset to the built-in catalog, then re-apply the ``VINCIO_MODEL_REGISTRY`` overlay."""
        self._profiles.clear()
        self._aliases.clear()
        for profile in _builtin_catalog():
            self.register(profile)
        overlay = os.environ.get("VINCIO_MODEL_REGISTRY")
        if overlay:
            self.load_file(overlay)

    # -- lookup ----------------------------------------------------------------

    def get(self, model_id: str) -> ModelProfile | None:
        """Exact id, then alias. No substring fallback (use :meth:`resolve`)."""
        if model_id in self._profiles:
            return self._profiles[model_id]
        canonical = self._aliases.get(model_id)
        if canonical is not None:
            return self._profiles.get(canonical)
        return None

    def _prefix_match(self, model_id: str) -> ModelProfile | None:
        """Longest-prefix fallback for dated snapshots (e.g. gpt-4o-2024-11-20)."""
        best: tuple[int, ModelProfile] | None = None
        for known, profile in self._profiles.items():
            if model_id.startswith(known) and (best is None or len(known) > best[0]):
                best = (len(known), profile)
        return best[1] if best else None

    def resolve(self, model_id: str, *, warn: bool = False) -> ModelProfile | None:
        """Exact → alias → demoted longest-prefix fallback. ``None`` when truly unknown."""
        profile = self.get(model_id) or self._prefix_match(model_id)
        if profile is None and warn:
            self.note_unknown(model_id)
        return profile

    def is_known(self, model_id: str) -> bool:
        return self.resolve(model_id) is not None

    def capabilities(self, model_id: str) -> ModelCapabilities | None:
        profile = self.resolve(model_id)
        return profile.capabilities if profile is not None else None

    def lifecycle(self, model_id: str, *, as_of: Any = None) -> ModelLifecycle | None:
        profile = self.resolve(model_id)
        return profile.lifecycle(as_of=as_of) if profile is not None else None

    def successor(self, model_id: str) -> str | None:
        profile = self.resolve(model_id)
        return profile.successor if profile is not None else None

    def note_unknown(self, model_id: str) -> None:
        """Warn once per process about an unknown model id (idempotent)."""
        if model_id in self._seen_unknown:
            return
        self._seen_unknown.add(model_id)
        warnings.warn(
            f"model {model_id!r} is not in the Vincio model registry: capabilities and "
            f"pricing fall back to heuristics and it may bill $0. Register it via "
            f"ModelRegistry.register(...) or a VINCIO_MODEL_REGISTRY overlay.",
            ModelUnknownWarning,
            stacklevel=3,
        )

    # -- introspection ---------------------------------------------------------

    def models(self) -> list[str]:
        return sorted(self._profiles)

    def profiles(self) -> list[ModelProfile]:
        return list(self._profiles.values())

    def __contains__(self, model_id: str) -> bool:  # pragma: no cover - trivial
        return self.is_known(model_id)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._profiles)


_DEFAULT_REGISTRY: ModelRegistry | None = None


def default_model_registry() -> ModelRegistry:
    """Process-wide registry, seeded from the built-in catalog plus the
    ``VINCIO_MODEL_REGISTRY`` overlay (if set). Constructed lazily and cached."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = ModelRegistry()
        overlay = os.environ.get("VINCIO_MODEL_REGISTRY")
        if overlay:
            registry.load_file(overlay)
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY


def discover_entry_points(group: str) -> dict[str, Any]:
    """Discover third-party adapters advertised under an entry-point *group*.

    Used for the ``vincio.providers`` / ``vincio.embedders`` / ``vincio.stores``
    groups so adapters shipped as separate pip packages auto-register. Failures
    to import a single entry point are isolated (a broken plugin never breaks
    discovery). Returns ``{name: loaded_object}``.
    """
    found: dict[str, Any] = {}
    try:
        eps = entry_points(group=group)
    except Exception:  # pragma: no cover - importlib.metadata edge cases
        return found
    for ep in eps:
        try:
            found[ep.name] = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not break discovery
            warnings.warn(
                f"failed to load entry point {ep.name!r} in group {group!r}",
                RuntimeWarning,
                stacklevel=2,
            )
    return found
