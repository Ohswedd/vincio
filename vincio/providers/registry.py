"""Data-driven model registry.

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
* capability guards, the cost/latency router, and the lifecycle watcher
  all consult it.

It is plain data and ships in-process — no network, no hosted dependency. Third
parties extend it by shipping their own pip packages exposing the
``vincio.providers`` / ``vincio.embedders`` / ``vincio.stores`` entry-point
groups (see :func:`discover_entry_points`), or by pointing
``VINCIO_MODEL_REGISTRY`` at a JSON/YAML overlay merged over the built-ins.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import date, timedelta
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.diagnostics import note_suppressed
from ..core.types import ModelCapabilities, ModelLifecycle, ModelProfile
from ..core.utils import utcnow

__all__ = [
    "REGISTRY_VERSION",
    "CATALOG_RELEASED",
    "FRESHNESS_HORIZON_DAYS",
    "ModelUnknownWarning",
    "ModelRegistry",
    "RegistryCoverageReport",
    "default_model_registry",
    "discover_entry_points",
]

# ---------------------------------------------------------------------------
# Built-in catalog — shipped, reviewable plain data (model_catalog.json beside
# this module). It states, per exact model id, the capabilities each provider
# previously guessed by substring and the pricing the cost tracker reads, so a
# current-lineup model never falls back to a guess or silently bills $0. Loaded
# once at import; ``_builtin_catalog()`` returns fresh ``ModelProfile`` instances
# each call so callers may freely mutate/copy them.
# ---------------------------------------------------------------------------

_CATALOG_PATH = Path(__file__).with_name("model_catalog.json")


def _load_catalog_data() -> dict[str, Any]:
    with _CATALOG_PATH.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


_CATALOG_DATA = _load_catalog_data()
_CATALOG_MODELS: list[dict[str, Any]] = _CATALOG_DATA["models"]

# The catalog's data-shape/contents version (independent of the package SemVer).
REGISTRY_VERSION: str = _CATALOG_DATA["registry_version"]

# The deterministic date the freshness horizon is evaluated against — the date
# this catalog snapshot shipped, NOT the wall clock. A frozen release therefore
# reports the same freshness verdict forever (a bug-fix release never "rots"
# because months pass); only cutting a new catalog snapshot advances this date
# and can surface a price that has drifted past the horizon.
CATALOG_RELEASED: str = _CATALOG_DATA["released"]

# How long after its ``priced_as_of`` date a price is still considered fresh,
# measured against :data:`CATALOG_RELEASED`. ~6 months: long enough that a stable
# rate card stays green, short enough that a stale snapshot is caught.
FRESHNESS_HORIZON_DAYS = 180


class ModelUnknownWarning(UserWarning):
    """Emitted once per process per unknown model id resolved against the registry."""


def _builtin_catalog() -> list[ModelProfile]:
    """Fresh ``ModelProfile`` instances parsed from the shipped catalog."""
    return [ModelProfile.model_validate(m) for m in _CATALOG_MODELS]


# ---------------------------------------------------------------------------
# Coverage anchors — the frozen expectations the coverage_report() drift
# detector holds the catalog to. Editing the catalog so any of these no longer
# holds fails the registry_coverage VincioBench family rather than shipping a
# silently degraded cost report.
# ---------------------------------------------------------------------------

# Providers whose models are free / self-hosted — exempt from the pricing rules.
# ``ds4`` (a DS4 DeepSeek-V4 box the operator runs) joins ``local`` / ``mock``:
# its models legitimately bill $0, so a $0 there is the correct answer, not a
# paid model silently drifting to zero.
_FREE_PROVIDERS = frozenset({"local", "mock", "ds4"})

# First-party providers Vincio ships a native adapter for, mapped to the canonical
# GA default model the coverage report proves resolves to a non-sparse, priced,
# GA profile (so the router / cascades / energy accounting always have one).
_PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "gpt-5.2-mini",
    "anthropic": "claude-sonnet-4-6",
    "google": "gemini-2.5-flash",
    "mistral": "mistral-small-latest",
}

# One representative id per substring family each provider's capability-heuristic
# fallback branches on (vincio/providers/{openai,anthropic,google,mistral}.py).
# Every one must resolve to a non-sparse, priced profile so the current lineup
# never falls back to a guessed price or capability matrix.
_CAPABILITY_FAMILIES: dict[str, list[str]] = {
    "openai": ["gpt-5.2", "gpt-5", "gpt-4.1", "gpt-4o", "o1", "o3", "o4-mini",
               "text-embedding-3-small"],
    "anthropic": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5", "claude-fable-5",
                  "claude-3-7-sonnet", "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"],
    "google": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-pro",
               "gemini-3-flash", "gemini-embedding-001"],
    "mistral": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest",
                "codestral-latest", "pixtral-large-latest", "mistral-embed"],
}

# Canonical cheapest-capable picks the router, the cascades, and the energy
# accounting depend on. A refresh that silently reorders which model is cheapest
# would change what they select; the report re-derives each pick from the live
# catalog (blended input+output $/Mtok) and flags any that moved.
_ROUTING_ANCHORS: list[tuple[list[str], str]] = [
    (["gpt-5.2", "gpt-5.2-mini", "gpt-5.2-nano"], "gpt-5.2-nano"),
    (["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"], "claude-haiku-4-5"),
    (["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"], "gemini-2.5-flash-lite"),
    (["mistral-large-latest", "mistral-small-latest"], "mistral-small-latest"),
    (["gpt-5.2", "claude-sonnet-4-6", "gemini-2.5-flash"], "gemini-2.5-flash"),
]


def _is_sparse(profile: ModelProfile) -> bool:
    """A profile carrying only the bare-default capability matrix (no real data)."""
    return profile.capabilities == ModelCapabilities()


def _is_embedding(profile: ModelProfile) -> bool:
    return "embedding" in profile.capabilities.output_modalities


def _is_priced(profile: ModelProfile) -> bool:
    """Whether a profile carries a real (non-zero) price for what it bills on."""
    if _is_embedding(profile):
        return profile.input_cost_per_mtok > 0
    return profile.input_cost_per_mtok > 0 and profile.output_cost_per_mtok > 0


def _is_free(profile: ModelProfile) -> bool:
    """Whether a model is free by construction, so a $0 on it is correct.

    True for a self-hosted model (the ``self_hosted`` flag, e.g. a DS4 DeepSeek-V4
    box) or any model of a free/self-hosted provider (``local`` / ``mock`` /
    ``ds4``). The pricing, silent-$0, and freshness rules do not apply to these —
    there is no metered rate card to verify."""
    return profile.self_hosted or profile.provider in _FREE_PROVIDERS


class RegistryCoverageReport(BaseModel):
    """The verdict of :meth:`ModelRegistry.coverage_report` — a deterministic,
    offline drift detector over the shipped catalog.

    ``ok`` is the single gate: it holds only when every supported provider's
    default and capability-heuristic families resolve to a non-sparse, priced
    profile, every ``openai_compat`` preset's headline model is priced, no GA
    billable model silently costs $0, no price has drifted past the freshness
    horizon, and the canonical router/cascade picks are unchanged. The list
    fields pinpoint exactly what failed.
    """

    as_of: str
    released: str
    horizon_days: int
    model_count: int
    provider_count: int
    default_models_resolve: bool
    capability_families_resolve: bool
    presets_priced: bool
    no_silent_zero: bool
    no_stale_prices: bool
    no_routing_drift: bool
    coverage_complete: bool
    ok: bool
    gaps: list[str] = Field(default_factory=list)
    unpriced: list[str] = Field(default_factory=list)
    stale: list[str] = Field(default_factory=list)
    drift: list[str] = Field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Flat, JSON-friendly view for the bench gate and the CLI."""
        return {
            "ok": self.ok,
            "coverage_complete": self.coverage_complete,
            "default_models_resolve": self.default_models_resolve,
            "capability_families_resolve": self.capability_families_resolve,
            "presets_priced": self.presets_priced,
            "no_silent_zero": self.no_silent_zero,
            "no_stale_prices": self.no_stale_prices,
            "no_routing_drift": self.no_routing_drift,
            "model_count": self.model_count,
            "provider_count": self.provider_count,
            "gaps": self.gaps,
            "unpriced": self.unpriced,
            "stale": self.stale,
            "drift": self.drift,
        }


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

    def reconcile(
        self,
        profiles: list[ModelProfile],
        *,
        provider: str | None = None,
        mark_missing_deprecated: bool = False,
        as_of: Any = None,
    ) -> dict[str, list[str]]:
        """Merge live-discovered *profiles* into the catalog.

        Discovered profiles from a model-list endpoint are typically *sparse*
        (id only, bare-default capabilities), so reconciliation must never let
        that thin data shadow a richer entry:

        * a discovered id that already **resolves** (exact, alias, or longest-
          prefix — e.g. a dated snapshot ``gpt-4o-2024-11-20`` of ``gpt-4o``) is
          folded into the matched profile: lifecycle fields are filled when
          missing and the discovered id is added as an **alias**, so the rich
          shipped capabilities/pricing stand and the capability guard is not
          tricked into refusing a capable model;
        * a genuinely **new** id (no resolution) is registered as-is.

        ``mark_missing_deprecated`` flags a catalog model of ``provider`` as
        deprecated only when *no* discovered id resolves to it (so a model listed
        under its dated snapshot is correctly treated as present). Returns
        ``{"added", "updated", "deprecated_missing"}``.
        """
        from datetime import date

        today = as_of or utcnow().date()
        today_iso = today.isoformat() if isinstance(today, date) else str(today)
        added: list[str] = []
        updated: list[str] = []
        resolved_present: set[str] = set()  # catalog ids a discovered id maps to
        lifecycle_fields = ("deprecation_date", "retirement_date", "ga_date",
                            "successor", "knowledge_cutoff")
        for profile in profiles:
            exact = self.get(profile.model)
            resolved = exact or self._prefix_match(profile.model)
            if resolved is None:
                # Genuinely new model — register the sparse profile as-is.
                self.register(profile)
                added.append(profile.model)
                resolved_present.add(profile.model)
                continue
            resolved_present.add(resolved.model)
            changes: dict[str, Any] = {}
            for field in lifecycle_fields:
                discovered_value = getattr(profile, field)
                if discovered_value and not getattr(resolved, field):
                    changes[field] = discovered_value
            # Fold a not-yet-known discovered id (a snapshot/alias) into the rich
            # profile as an alias rather than registering a capability-less shadow.
            if exact is None and profile.model != resolved.model and profile.model not in resolved.aliases:
                changes["aliases"] = [*resolved.aliases, profile.model]
            if changes:
                self.register(resolved.model_copy(update=changes))
                updated.append(profile.model)

        deprecated_missing: list[str] = []
        if mark_missing_deprecated and provider is not None:
            for existing in list(self._profiles.values()):
                if (
                    existing.provider == provider
                    and existing.model not in resolved_present
                    and existing.lifecycle(as_of=today) == "ga"
                    and "embedding" not in existing.capabilities.output_modalities
                ):
                    self.register(existing.model_copy(update={"deprecation_date": today_iso}))
                    deprecated_missing.append(existing.model)
        return {"added": added, "updated": updated, "deprecated_missing": deprecated_missing}

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

    def guard_capabilities(self, model_id: str) -> ModelCapabilities | None:
        """Capabilities for the capability guard.

        Like :meth:`capabilities`, but returns ``None`` for a profile that carries
        only the bare-default capability matrix (a sparsely-discovered model whose
        real capabilities are unknown). The guard treats ``None`` as unjudgeable
        and permits the model, rather than refusing it for capabilities we never
        actually learned — so live discovery can never make the guard *block* a
        model it would previously have allowed.
        """
        profile = self.resolve(model_id)
        if profile is None:
            return None
        if profile.capabilities == ModelCapabilities():
            return None
        return profile.capabilities

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

    # -- coverage / freshness --------------------------------------------------

    def _priced_as_of(self, profile: ModelProfile) -> date | None:
        if not profile.priced_as_of:
            return None
        try:
            return date.fromisoformat(profile.priced_as_of[:10])
        except ValueError:
            return None

    def coverage_report(
        self,
        *,
        as_of: date | str | None = None,
        horizon_days: int = FRESHNESS_HORIZON_DAYS,
    ) -> RegistryCoverageReport:
        """Prove the catalog is complete, honest, fresh, and routing-stable.

        A deterministic, offline drift detector (no network, no wall clock):

        * **coverage** — every supported provider's GA default and every
          capability-heuristic family resolve to a non-sparse, priced profile,
          and every ``openai_compat`` preset's headline model is priced;
        * **no silent $0** — every GA billable model of a paid provider carries a
          real price (input, and output for generative models);
        * **freshness** — no GA price has drifted more than ``horizon_days`` past
          its ``priced_as_of``, measured against ``as_of`` (the catalog's release
          date by default, *never* the wall clock — so a frozen release is stable);
        * **no routing drift** — the canonical cheapest-capable router/cascade
          picks are unchanged.

        ``as_of`` overrides the reference date for what-if checks (e.g. "is this
        catalog still fresh a year from release?"); it never reads the clock on
        its own.
        """
        from .openai_compat import PRESETS

        if as_of is None:
            ref = date.fromisoformat(CATALOG_RELEASED)
        elif isinstance(as_of, str):
            ref = date.fromisoformat(as_of[:10])
        else:
            ref = as_of
        horizon = ref - timedelta(days=horizon_days)

        profiles = self.profiles()
        gaps: list[str] = []
        unpriced: list[str] = []
        stale: list[str] = []
        drift: list[str] = []

        def _covered(model_id: str, label: str) -> bool:
            profile = self.resolve(model_id)
            if profile is None:
                gaps.append(f"{label}: {model_id} → unresolved")
                return False
            if _is_sparse(profile):
                gaps.append(f"{label}: {model_id} → sparse (no real capabilities)")
                return False
            # A self-hosted / free model is correct at $0, so it must resolve to a
            # non-sparse profile but is not required to carry a price.
            if not _is_free(profile) and not _is_priced(profile):
                gaps.append(f"{label}: {model_id} → unpriced")
                return False
            return True

        # Coverage: provider defaults must additionally be GA.
        default_models_resolve = True
        for provider, model_id in _PROVIDER_DEFAULTS.items():
            ok = _covered(model_id, f"default[{provider}]")
            profile = self.resolve(model_id)
            if profile is not None and profile.lifecycle(as_of=ref) != "ga":
                gaps.append(f"default[{provider}]: {model_id} → not GA")
                ok = False
            default_models_resolve = default_models_resolve and ok

        capability_families_resolve = all(
            _covered(model_id, f"family[{provider}]")
            for provider, ids in _CAPABILITY_FAMILIES.items()
            for model_id in ids
        )

        presets_priced = all(
            _covered(preset.default_model, f"preset[{name}]")
            for name, preset in PRESETS.items()
            if preset.default_model
        )

        # No silent $0: every GA billable model of a paid provider must be priced.
        for profile in profiles:
            if _is_free(profile):
                continue
            if profile.lifecycle(as_of=ref) != "ga":
                continue
            if not _is_priced(profile):
                unpriced.append(profile.model)
        no_silent_zero = not unpriced

        # Freshness: every GA priced model of a paid provider must carry a
        # ``priced_as_of`` within the horizon of the reference (release) date.
        for profile in profiles:
            if _is_free(profile):
                continue
            if profile.lifecycle(as_of=ref) != "ga" or not _is_priced(profile):
                continue
            priced = self._priced_as_of(profile)
            if priced is None:
                stale.append(f"{profile.model}: missing priced_as_of")
            elif priced < horizon:
                stale.append(f"{profile.model}: priced {priced.isoformat()} < {horizon.isoformat()}")
        no_stale_prices = not stale

        # No routing drift: re-derive each canonical cheapest pick from the live
        # catalog and compare against the frozen expectation.
        for candidates, expected in _ROUTING_ANCHORS:
            resolved = {c: self.resolve(c) for c in candidates}
            if any(p is None for p in resolved.values()):
                drift.append(f"{candidates} → a candidate is unresolved")
                continue
            blended = {
                c: p.input_cost_per_mtok + p.output_cost_per_mtok
                for c, p in resolved.items()
                if p is not None
            }
            cheapest = min(candidates, key=lambda c: (blended[c], candidates.index(c)))
            if cheapest != expected:
                drift.append(f"{candidates} → cheapest is {cheapest}, expected {expected}")
        no_routing_drift = not drift

        coverage_complete = (
            default_models_resolve
            and capability_families_resolve
            and presets_priced
            and no_silent_zero
        )
        ok = coverage_complete and no_stale_prices and no_routing_drift
        return RegistryCoverageReport(
            as_of=ref.isoformat(),
            released=CATALOG_RELEASED,
            horizon_days=horizon_days,
            model_count=len(profiles),
            provider_count=len({p.provider for p in profiles}),
            default_models_resolve=default_models_resolve,
            capability_families_resolve=capability_families_resolve,
            presets_priced=presets_priced,
            no_silent_zero=no_silent_zero,
            no_stale_prices=no_stale_prices,
            no_routing_drift=no_routing_drift,
            coverage_complete=coverage_complete,
            ok=ok,
            gaps=gaps,
            unpriced=unpriced,
            stale=stale,
            drift=drift,
        )

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
    except Exception:
        note_suppressed("providers.entry_points")
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
