"""Energy & carbon accounting.

Estimates a run's **energy** (watt-hours) and **carbon** (grams CO₂-equivalent)
from the token accounting every model call already produces, the same way
:mod:`~vincio.observability.costs` estimates dollars. The estimate is mechanical
and deterministic — a per-model energy intensity (watt-hours per million tokens,
derived from the model's tier and freely overridable) scaled by a datacenter
power-overhead factor, then multiplied by a per-region grid carbon intensity
(grams CO₂e per kWh) from a built-in default table. No external service is
consulted; the numbers are reproducible offline.

The result lands on the existing cost-report surface — :class:`EnergyEstimate`
totals accrue on the :class:`~vincio.observability.costs.CostTracker`, each
attributed :class:`~vincio.observability.finops.CostEvent` carries its energy and
carbon, and :meth:`~vincio.core.app.ContextApp.energy_report` rolls them up by
the same dimensions as :meth:`~vincio.core.app.ContextApp.cost_report`. It is the
energy analogue of the dollar budget, never a separate plane.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.types import ModelProfile, TokenUsage

__all__ = [
    "EnergyProfile",
    "EnergyEstimate",
    "EnergyIntensityTable",
    "default_energy_table",
    "DEFAULT_CARBON_INTENSITY",
    "WORLD_AVERAGE_CARBON_INTENSITY",
    "DEFAULT_PUE",
]

# Datacenter power-usage-effectiveness multiplier: the energy the facility draws
# per unit the accelerator draws (cooling, power delivery, networking). 1.12 is a
# representative modern hyperscale figure; overridable per table.
DEFAULT_PUE = 1.12

# World-average grid carbon intensity (g CO₂e / kWh) — the fallback when a run's
# region is neither declared nor inferable.
WORLD_AVERAGE_CARBON_INTENSITY = 480.0

# Per-region grid carbon intensity (g CO₂e / kWh), keyed by the coarse
# jurisdiction codes the residency policy resolves (``us`` / ``eu`` / ``uk`` /
# ``ca`` / ``apac`` …) plus a few country tokens. Representative annual averages;
# an operator with a measured factor sets it via :meth:`set_region_intensity`.
DEFAULT_CARBON_INTENSITY: dict[str, float] = {
    "us": 369.0,
    "eu": 240.0,
    "uk": 207.0,
    "ca": 120.0,
    "apac": 555.0,
    "au": 510.0,
    "in": 631.0,
    "jp": 462.0,
    "de": 350.0,
    "fr": 56.0,
    # Self-hosted / on-prem grid is unknown by construction — assume the world
    # average rather than pretending it is clean.
    "on_prem": WORLD_AVERAGE_CARBON_INTENSITY,
}

# Reference accelerator energy intensity per model tier, in watt-hours per
# million tokens, split by prefill (input) and decode (output). Autoregressive
# decode dominates inference energy, so the output rate is an order of magnitude
# above the input rate; a stronger (larger) model draws more per token than a
# fast (small) one. These are deterministic reference constants, not a
# measurement of any specific deployment; override per model with
# :meth:`EnergyIntensityTable.set`.
_TIER_ENERGY: dict[str, tuple[float, float]] = {
    # tier: (wh_per_input_mtok, wh_per_output_mtok)
    "fast": (15.0, 150.0),
    "default": (60.0, 600.0),
    "strong": (250.0, 2500.0),
}


def _jurisdiction(region: str) -> str:
    """Reduce a resolved region to the coarse code the intensity table keys on.

    ``eu-west-1`` → ``eu``, ``us-east-1`` → ``us``, ``europe-west4`` → ``eu``.
    Kept self-contained (a small prefix map) so this module does not import the
    governance package, which loads later than observability.
    """
    head = region.lower().split("-")[0]
    prefixes = {
        "europe": "eu", "eu": "eu", "de": "eu", "fr": "eu", "es": "eu", "it": "eu",
        "us": "us", "northamerica": "us",
        "uk": "uk", "gb": "uk",
        "ca": "ca",
        "asia": "apac", "australia": "apac", "ap": "apac", "jp": "apac", "in": "apac",
        "me": "apac", "africa": "apac", "southamerica": "us",
    }
    return prefixes.get(head, head)


class EnergyProfile(BaseModel):
    """Per-model energy intensity, in watt-hours per million tokens."""

    wh_per_input_mtok: float = 0.0
    wh_per_output_mtok: float = 0.0

    @classmethod
    def for_tier(cls, tier: str) -> EnergyProfile:
        """The reference profile for a model tier (``fast``/``default``/``strong``)."""
        wh_in, wh_out = _TIER_ENERGY.get(tier, _TIER_ENERGY["default"])
        return cls(wh_per_input_mtok=wh_in, wh_per_output_mtok=wh_out)


class EnergyEstimate(BaseModel):
    """A run (or call)'s estimated energy and carbon, with its breakdown.

    ``energy_wh`` is facility energy (accelerator energy × PUE); ``co2e_grams`` is
    that energy at the run's regional grid intensity. The breakdown fields make
    the estimate auditable — every term is reproducible from the inputs.
    """

    energy_wh: float = 0.0
    co2e_grams: float = 0.0
    region: str = "global"
    carbon_intensity_g_per_kwh: float = WORLD_AVERAGE_CARBON_INTENSITY
    pue: float = DEFAULT_PUE
    input_wh: float = 0.0
    output_wh: float = 0.0
    model: str = ""

    def add(self, other: EnergyEstimate) -> None:
        """Accumulate another estimate's energy and carbon into this one."""
        self.energy_wh += other.energy_wh
        self.co2e_grams += other.co2e_grams
        self.input_wh += other.input_wh
        self.output_wh += other.output_wh

    @property
    def co2e_kg(self) -> float:
        """Carbon in kilograms CO₂e (the unit sustainability reports use)."""
        return self.co2e_grams / 1000.0


def _default_energy_profiles() -> dict[str, EnergyProfile]:
    """Seed per-model profiles from the registry, by tier — the energy analogue
    of :func:`~vincio.observability.costs._default_prices`."""
    from ..providers.registry import default_model_registry

    return {
        p.model: EnergyProfile.for_tier(p.tier) for p in default_model_registry().profiles()
    }


class EnergyIntensityTable(BaseModel):
    """Resolves a model + region into an energy/carbon estimate.

    The energy analogue of :class:`~vincio.observability.costs.PriceTable`: the
    model table seeds itself from the registry (by tier) and stays overridable at
    runtime via :meth:`set`; the carbon table seeds from
    :data:`DEFAULT_CARBON_INTENSITY` and is overridable via
    :meth:`set_region_intensity`.
    """

    profiles: dict[str, EnergyProfile] = Field(default_factory=_default_energy_profiles)
    carbon_intensity: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_CARBON_INTENSITY)
    )
    pue: float = DEFAULT_PUE
    # An operator-declared deployment region. When set, it pins every call's grid
    # intensity (the operator knows where their inference runs); when ``None``,
    # the region is resolved per call (e.g. from the residency policy), then falls
    # back to ``default_region``.
    region_override: str | None = None
    default_region: str = "global"
    default_carbon_intensity: float = WORLD_AVERAGE_CARBON_INTENSITY

    def set(self, model: str, profile: EnergyProfile) -> None:
        """Override the energy intensity for a model id."""
        self.profiles[model] = profile

    def set_region_intensity(self, region: str, g_per_kwh: float) -> None:
        """Override the grid carbon intensity (g CO₂e/kWh) for a region."""
        self.carbon_intensity[_jurisdiction(region)] = max(0.0, g_per_kwh)

    def lookup(self, model: str) -> EnergyProfile:
        """The energy profile for *model* (explicit override, then registry tier)."""
        if model in self.profiles:
            return self.profiles[model]
        from ..providers.registry import default_model_registry

        profile: ModelProfile | None = default_model_registry().resolve(model)
        if profile is not None:
            return EnergyProfile.for_tier(profile.tier)
        # Longest-prefix over explicit overrides, so a runtime override of a base
        # id still covers its dated snapshots.
        best: tuple[int, EnergyProfile] | None = None
        for name, prof in self.profiles.items():
            if model.startswith(name) and (best is None or len(name) > best[0]):
                best = (len(name), prof)
        if best is not None:
            return best[1]
        # Unknown model: fall back to the default-tier reference rather than zero,
        # so an unrecognized model still reports a non-zero estimate.
        return EnergyProfile.for_tier("default")

    def intensity_for(self, region: str | None) -> tuple[str, float]:
        """Resolve a region to its (region, g CO₂e/kWh) intensity.

        The exact token wins first, so a country with its own grid factor
        (``fr`` = 56) is not shadowed by its coarser jurisdiction (``eu`` = 240);
        an AWS/GCP region string (``eu-west-1``) then resolves via its
        jurisdiction; anything unknown falls back to the default.
        """
        if not region:
            return self.default_region, self.default_carbon_intensity
        key = region.lower()
        if key in self.carbon_intensity:
            return key, self.carbon_intensity[key]
        code = _jurisdiction(key)
        if code in self.carbon_intensity:
            return code, self.carbon_intensity[code]
        return self.default_region, self.default_carbon_intensity

    def estimate(
        self, model: str, usage: TokenUsage, *, region: str | None = None
    ) -> EnergyEstimate:
        """Estimate energy + carbon for a single model call.

        Cached input tokens are billed at the uncached input rate for energy —
        a prompt-cache hit saves the provider's dollar price, but the prefill
        compute (and thus the energy) is still incurred unless the serving engine
        also reuses the KV, which the dollar/KV accounting tracks separately.
        """
        profile = self.lookup(model)
        input_wh = usage.input_tokens * profile.wh_per_input_mtok / 1_000_000 * self.pue
        output_wh = usage.output_tokens * profile.wh_per_output_mtok / 1_000_000 * self.pue
        energy_wh = input_wh + output_wh
        code, g_per_kwh = self.intensity_for(region)
        co2e_grams = energy_wh / 1000.0 * g_per_kwh
        return EnergyEstimate(
            energy_wh=energy_wh,
            co2e_grams=co2e_grams,
            region=code,
            carbon_intensity_g_per_kwh=g_per_kwh,
            pue=self.pue,
            input_wh=input_wh,
            output_wh=output_wh,
            model=model,
        )


def default_energy_table() -> EnergyIntensityTable:
    return EnergyIntensityTable()
