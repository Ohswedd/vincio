"""Context-aware offline search strategies.

Richer search for the evolution loop than blind random sampling: proposals
are conditioned on what already scored well (hill climbing over single-knob
mutations, simulated annealing with a cooling schedule). Every strategy is
deterministic under a seed, hard-bounded by the evaluation budget, and only
feeds candidates into the same gated promotion path — a guided search can
never bypass the safety rules.
"""

from __future__ import annotations

import math
import random
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

__all__ = [
    "SearchStrategy",
    "RandomSearch",
    "HillClimbSearch",
    "AnnealingSearch",
    "build_strategy",
    "guided_search",
]

# (config, fitness) pairs already evaluated, best first not required.
History = list[tuple[dict[str, Any], float]]


class SearchStrategy(Protocol):
    """Proposes the next batch of configs given the scored history."""

    def propose(self, history: History, *, n: int) -> list[dict[str, Any]]:
        ...  # pragma: no cover


def _config_key(config: dict[str, Any]) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in config.items()))


def _random_config(space: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    return {key: rng.choice(values) for key, values in space.items() if values}


def _mutate(
    config: dict[str, Any], space: dict[str, list[Any]], rng: random.Random, *, knobs: int = 1
) -> dict[str, Any]:
    """Change *knobs* randomly chosen keys to a different value from the grid."""
    mutable = [key for key, values in space.items() if len(values) > 1]
    if not mutable:
        return dict(config)
    neighbor = dict(config)
    for key in rng.sample(mutable, k=min(knobs, len(mutable))):
        alternatives = [value for value in space[key] if value != config.get(key)]
        if alternatives:
            neighbor[key] = rng.choice(alternatives)
    return neighbor


class _StrategyBase:
    def __init__(self, space: dict[str, list[Any]], *, seed: int = 7) -> None:
        self.space = {key: list(values) for key, values in space.items()}
        self.rng = random.Random(seed)
        self._seen: set[tuple] = set()

    def _emit(self, proposals: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        for config in proposals:
            key = _config_key(config)
            if key in self._seen:
                continue
            self._seen.add(key)
            unique.append(config)
            if len(unique) >= n:
                break
        return unique

    def _fill_random(self, batch: list[dict[str, Any]], n: int, *, attempts: int = 200) -> None:
        tries = 0
        while len(batch) < n and tries < attempts:
            tries += 1
            config = _random_config(self.space, self.rng)
            key = _config_key(config)
            if key in self._seen:
                continue
            self._seen.add(key)
            batch.append(config)


class RandomSearch(_StrategyBase):
    """Uniform random sampling (the 0.7 behaviour, as a strategy)."""

    def propose(self, history: History, *, n: int) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        self._fill_random(batch, n)
        return batch


class HillClimbSearch(_StrategyBase):
    """Mutate the best config seen so far, one knob at a time.

    The first batch is random (no gradient without a starting point); later
    batches are single-knob neighbors of the incumbent, topped up with
    random exploration when the neighborhood is exhausted.
    """

    def propose(self, history: History, *, n: int) -> list[dict[str, Any]]:
        if not history:
            batch: list[dict[str, Any]] = []
            self._fill_random(batch, n)
            return batch
        best_config, _best_fitness = max(history, key=lambda entry: entry[1])
        neighbors = [
            _mutate(best_config, self.space, self.rng, knobs=1) for _ in range(n * 4)
        ]
        batch = self._emit(neighbors, n)
        self._fill_random(batch, n)
        return batch


class AnnealingSearch(_StrategyBase):
    """Simulated annealing: walk from a current config, accepting worse
    moves with probability ``exp(Δfitness / T)`` while the temperature
    cools each batch. Early batches explore; late batches exploit."""

    def __init__(
        self,
        space: dict[str, list[Any]],
        *,
        seed: int = 7,
        temperature: float = 0.5,
        cooling: float = 0.6,
    ) -> None:
        super().__init__(space, seed=seed)
        self.temperature = temperature
        self.cooling = cooling
        self._current: dict[str, Any] | None = None
        self._current_fitness = float("-inf")

    def propose(self, history: History, *, n: int) -> list[dict[str, Any]]:
        if not history:
            batch: list[dict[str, Any]] = []
            self._fill_random(batch, n)
            return batch
        # Metropolis acceptance of the latest evaluated config.
        latest_config, latest_fitness = history[-1]
        if self._current is None or latest_fitness >= self._current_fitness:
            self._current, self._current_fitness = latest_config, latest_fitness
        else:
            delta = latest_fitness - self._current_fitness
            if self.temperature > 0 and self.rng.random() < math.exp(delta / self.temperature):
                self._current, self._current_fitness = latest_config, latest_fitness
        knobs = 2 if self.temperature > 0.2 else 1
        neighbors = [
            _mutate(self._current, self.space, self.rng, knobs=knobs) for _ in range(n * 4)
        ]
        batch = self._emit(neighbors, n)
        self._fill_random(batch, n)
        self.temperature *= self.cooling
        return batch


def build_strategy(
    kind: str, space: dict[str, list[Any]], *, seed: int = 7, **kwargs: Any
) -> SearchStrategy:
    if kind == "random":
        return RandomSearch(space, seed=seed)
    if kind == "hill_climb":
        return HillClimbSearch(space, seed=seed)
    if kind in ("anneal", "annealing"):
        return AnnealingSearch(space, seed=seed, **kwargs)
    raise ValueError(f"unknown search strategy {kind!r}; known: random, hill_climb, anneal")


# evaluate(config) -> fitness on the screening subset.
SubsetEvaluateFn = Callable[[dict[str, Any]], Awaitable[float]]


async def guided_search(
    space: dict[str, list[Any]],
    evaluate: SubsetEvaluateFn,
    *,
    strategy: str | SearchStrategy = "hill_climb",
    budget: int = 12,
    batch_size: int = 3,
    seed: int = 7,
) -> History:
    """Run a bounded strategy-guided search and return the scored history.

    ``budget`` is a hard cap on evaluations; the strategy proposes batches
    conditioned on everything scored so far. The caller feeds the history
    into the evolution loop for full-dataset verification and gated
    promotion — this function only screens.
    """
    if isinstance(strategy, str):
        strategy = build_strategy(strategy, space, seed=seed)
    history: History = []
    while len(history) < budget:
        remaining = budget - len(history)
        batch = strategy.propose(history, n=min(batch_size, remaining))
        if not batch:
            break
        for config in batch:
            score = await evaluate(config)
            history.append((config, score))
    return history
