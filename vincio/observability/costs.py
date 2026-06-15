"""Cost tracking.

Maintains a price table per model family and computes USD costs from token
usage. Prices are configurable; the built-in table covers common models and
is intentionally easy to override at runtime — provider pricing changes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.types import TokenUsage

__all__ = ["ModelPrice", "PriceTable", "CostTracker", "default_price_table"]


class ModelPrice(BaseModel):
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cached_input_per_mtok: float = 0.0


# Prices in USD per million tokens. Override via PriceTable.set() or config.
_DEFAULT_PRICES: dict[str, ModelPrice] = {
    # OpenAI
    "gpt-5.2": ModelPrice(input_per_mtok=1.25, output_per_mtok=10.0, cached_input_per_mtok=0.125),
    "gpt-5.2-mini": ModelPrice(input_per_mtok=0.25, output_per_mtok=2.0, cached_input_per_mtok=0.025),
    "gpt-5.2-nano": ModelPrice(input_per_mtok=0.05, output_per_mtok=0.4, cached_input_per_mtok=0.005),
    "gpt-4o": ModelPrice(input_per_mtok=2.5, output_per_mtok=10.0, cached_input_per_mtok=1.25),
    "gpt-4o-mini": ModelPrice(input_per_mtok=0.15, output_per_mtok=0.6, cached_input_per_mtok=0.075),
    # Anthropic
    "claude-fable-5": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0, cached_input_per_mtok=0.5),
    "claude-opus-4-8": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0, cached_input_per_mtok=0.5),
    "claude-sonnet-4-6": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0, cached_input_per_mtok=0.3),
    "claude-haiku-4-5": ModelPrice(input_per_mtok=1.0, output_per_mtok=5.0, cached_input_per_mtok=0.1),
    # Google
    "gemini-3-pro": ModelPrice(input_per_mtok=2.0, output_per_mtok=12.0, cached_input_per_mtok=0.5),
    "gemini-3-flash": ModelPrice(input_per_mtok=0.3, output_per_mtok=2.5, cached_input_per_mtok=0.075),
    # Google — current GA models (free tier bills $0; paid-tier rates shown for cost tracking)
    "gemini-2.5-pro": ModelPrice(input_per_mtok=1.25, output_per_mtok=10.0, cached_input_per_mtok=0.31),
    "gemini-2.5-flash": ModelPrice(input_per_mtok=0.3, output_per_mtok=2.5, cached_input_per_mtok=0.075),
    "gemini-2.5-flash-lite": ModelPrice(input_per_mtok=0.1, output_per_mtok=0.4, cached_input_per_mtok=0.025),
    "gemini-2.0-flash": ModelPrice(input_per_mtok=0.1, output_per_mtok=0.4, cached_input_per_mtok=0.025),
    "gemini-2.0-flash-lite": ModelPrice(input_per_mtok=0.075, output_per_mtok=0.3),
    # Embeddings: gemini-embedding-001 is the current GA model and the provider
    # default; without it here, embedding cost silently resolves to $0.
    "gemini-embedding-001": ModelPrice(input_per_mtok=0.15),
    "text-embedding-004": ModelPrice(input_per_mtok=0.0),
    # Mistral
    "mistral-large-latest": ModelPrice(input_per_mtok=2.0, output_per_mtok=6.0),
    "mistral-small-latest": ModelPrice(input_per_mtok=0.2, output_per_mtok=0.6),
    # Local/self-hosted defaults to free
    "local": ModelPrice(),
}


class PriceTable(BaseModel):
    prices: dict[str, ModelPrice] = Field(default_factory=lambda: dict(_DEFAULT_PRICES))

    def set(self, model: str, price: ModelPrice) -> None:
        self.prices[model] = price

    def lookup(self, model: str) -> ModelPrice:
        if model in self.prices:
            return self.prices[model]
        # Prefix match: "gpt-4o-2024-11-20" -> "gpt-4o"
        best: tuple[int, ModelPrice] | None = None
        for name, price in self.prices.items():
            if model.startswith(name) and (best is None or len(name) > best[0]):
                best = (len(name), price)
        if best:
            return best[1]
        return ModelPrice()

    def cost(self, model: str, usage: TokenUsage) -> float:
        price = self.lookup(model)
        uncached_input = max(0, usage.input_tokens - usage.cached_input_tokens)
        return (
            uncached_input * price.input_per_mtok
            + usage.cached_input_tokens * price.cached_input_per_mtok
            + usage.output_tokens * price.output_per_mtok
        ) / 1_000_000


def default_price_table() -> PriceTable:
    return PriceTable()


class CostTracker:
    """Accumulates model/embedding/tool/infra cost for a run or app."""

    def __init__(self, price_table: PriceTable | None = None) -> None:
        self.price_table = price_table or default_price_table()
        self.model_cost_usd = 0.0
        self.embedding_cost_usd = 0.0
        self.tool_cost_usd = 0.0
        self.infra_cost_usd = 0.0
        self.usage = TokenUsage()

    def record_model_call(self, model: str, usage: TokenUsage) -> float:
        cost = self.price_table.cost(model, usage)
        self.model_cost_usd += cost
        self.usage.add(usage)
        return cost

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
        }
