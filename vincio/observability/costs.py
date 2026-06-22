"""Cost tracking.

Computes USD costs from token usage. Prices derive from the data-driven
:class:`~vincio.providers.registry.ModelRegistry` (the single source of truth
for capabilities, pricing, and lifecycle); the table seeds itself from the
registry and is still freely overridable at runtime via :meth:`PriceTable.set`.

Unknown models no longer silently bill ``$0``: the registry warns once and the
runtime emits a ``model.unknown`` event, so a missing price is observable
instead of hidden.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.types import ModelProfile, TokenUsage
from .energy import EnergyEstimate, EnergyIntensityTable, default_energy_table

__all__ = ["ModelPrice", "PriceTable", "CostTracker", "default_price_table"]


class ModelPrice(BaseModel):
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cached_input_per_mtok: float = 0.0
    # Batch-tier rates (typically ~half). ``None`` falls back to the standard
    # rate. Used when a cost is recorded for a batch (half-cost) execution.
    batch_input_per_mtok: float | None = None
    batch_output_per_mtok: float | None = None


def _price_from_profile(profile: ModelProfile) -> ModelPrice:
    return ModelPrice(
        input_per_mtok=profile.input_cost_per_mtok,
        output_per_mtok=profile.output_cost_per_mtok,
        cached_input_per_mtok=profile.cached_input_cost_per_mtok,
        batch_input_per_mtok=profile.batch_input_cost_per_mtok,
        batch_output_per_mtok=profile.batch_output_cost_per_mtok,
    )


def _default_prices() -> dict[str, ModelPrice]:
    """Seed the price dict from the registry so ``.prices`` reflects the catalog."""
    from ..providers.registry import default_model_registry

    return {p.model: _price_from_profile(p) for p in default_model_registry().profiles()}


class PriceTable(BaseModel):
    prices: dict[str, ModelPrice] = Field(default_factory=_default_prices)

    def set(self, model: str, price: ModelPrice) -> None:
        self.prices[model] = price

    def lookup(self, model: str) -> ModelPrice:
        from ..providers.registry import default_model_registry

        registry = default_model_registry()
        # 1. Explicit override / seeded exact entry.
        if model in self.prices:
            return self.prices[model]
        # 2. Exact registry id (alias-aware) — authoritative for known models.
        profile = registry.get(model)
        if profile is not None:
            return _price_from_profile(profile)
        # 3. Longest-prefix over the price dict FIRST, so a runtime override of a
        #    base id still covers its dated snapshots ("gpt-4o-2024-11-20" ->
        #    a user-set "gpt-4o") rather than being shadowed by the built-in.
        best: tuple[int, ModelPrice] | None = None
        for name, price in self.prices.items():
            if model.startswith(name) and (best is None or len(name) > best[0]):
                best = (len(name), price)
        if best:
            return best[1]
        # 4. Registry prefix fallback (built-in dated snapshots).
        profile = registry._prefix_match(model)
        if profile is not None:
            return _price_from_profile(profile)
        # 5. Genuinely unknown: warn (once) rather than silently bill $0.
        registry.note_unknown(model)
        return ModelPrice()

    def is_known(self, model: str) -> bool:
        """Whether *model* has a real (non-fallback) price the cost can trust."""
        if model in self.prices:
            return True
        from ..providers.registry import default_model_registry

        if default_model_registry().resolve(model) is not None:
            return True
        return any(model.startswith(name) for name in self.prices)

    def cost(self, model: str, usage: TokenUsage, *, batch: bool = False) -> float:
        price = self.lookup(model)
        uncached_input = max(0, usage.input_tokens - usage.cached_input_tokens)
        input_rate = (
            price.batch_input_per_mtok
            if batch and price.batch_input_per_mtok is not None
            else price.input_per_mtok
        )
        output_rate = (
            price.batch_output_per_mtok
            if batch and price.batch_output_per_mtok is not None
            else price.output_per_mtok
        )
        return (
            uncached_input * input_rate
            + usage.cached_input_tokens * price.cached_input_per_mtok
            + usage.output_tokens * output_rate
        ) / 1_000_000


def default_price_table() -> PriceTable:
    return PriceTable()


class CostTracker:
    """Accumulates model/embedding/tool/infra cost for a run or app."""

    def __init__(
        self,
        price_table: PriceTable | None = None,
        *,
        energy_table: EnergyIntensityTable | None = None,
    ) -> None:
        self.price_table = price_table or default_price_table()
        # Energy/carbon intensity table — the energy analogue of the price table.
        # Always present (cheap; profiles seed lazily on lookup), so the energy
        # surface is wired even before accounting is enabled.
        self.energy_table = energy_table or default_energy_table()
        self.model_cost_usd = 0.0
        self.embedding_cost_usd = 0.0
        self.tool_cost_usd = 0.0
        self.infra_cost_usd = 0.0
        self.usage = TokenUsage()
        # Estimated energy (watt-hours) and carbon (grams CO₂e) accrued across the
        # runs this tracker has accounted for. Surfaced in :meth:`summary`.
        self.energy_wh = 0.0
        self.co2e_grams = 0.0
        # Peak resident-memory footprint observed across the runs this tracker
        # has accounted for, in bytes. Surfaced in :meth:`summary`.
        self.peak_resident_bytes = 0

    def record_memory(self, resident_bytes: int) -> None:
        """Account a compiled packet's estimated resident footprint."""
        self.peak_resident_bytes = max(self.peak_resident_bytes, max(0, int(resident_bytes)))

    def record_model_call(self, model: str, usage: TokenUsage) -> float:
        cost = self.price_table.cost(model, usage)
        self.model_cost_usd += cost
        self.usage.add(usage)
        return cost

    def record_energy(
        self, model: str, usage: TokenUsage, *, region: str | None = None
    ) -> EnergyEstimate:
        """Estimate and accrue a model call's energy and carbon.

        Returns the per-call :class:`~vincio.observability.energy.EnergyEstimate`
        so the runtime can attribute it to the run and the cost event."""
        estimate = self.energy_table.estimate(model, usage, region=region)
        self.energy_wh += estimate.energy_wh
        self.co2e_grams += estimate.co2e_grams
        return estimate

    def record_embedding(self, model: str, tokens: int) -> float:
        cost = self.price_table.lookup(model).input_per_mtok * tokens / 1_000_000
        self.embedding_cost_usd += cost
        return cost

    def record_tool(self, cost_usd: float) -> None:
        self.tool_cost_usd += cost_usd

    def record_infra(self, cost_usd: float) -> None:
        self.infra_cost_usd += cost_usd

    @property
    def total_usd(self) -> float:
        return self.model_cost_usd + self.embedding_cost_usd + self.tool_cost_usd + self.infra_cost_usd

    def summary(self) -> dict[str, float | int]:
        return {
            "model_cost_usd": round(self.model_cost_usd, 8),
            "embedding_cost_usd": round(self.embedding_cost_usd, 8),
            "tool_cost_usd": round(self.tool_cost_usd, 8),
            "infra_cost_usd": round(self.infra_cost_usd, 8),
            "total_usd": round(self.total_usd, 8),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cached_input_tokens": self.usage.cached_input_tokens,
            "reasoning_tokens": self.usage.reasoning_tokens,
            "peak_resident_bytes": self.peak_resident_bytes,
            "energy_wh": round(self.energy_wh, 6),
            "co2e_grams": round(self.co2e_grams, 6),
        }
