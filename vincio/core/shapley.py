"""Exact Shapley value attribution — the shared credit-assignment kernel.

Pure and dependency-free. Two consumers share it because they ask the same
question. Causal regression attribution (:mod:`vincio.evals.attribution`)
attributes a metric delta to the components a release changed; on-policy credit
assignment (:mod:`vincio.optimize.trajectory_opt`) attributes a trajectory's
outcome reward to the steps that earned it. Both want each *player's* average
marginal contribution across all orderings, so both call the same exact
decomposition here instead of re-deriving it.

The Shapley value is the unique credit assignment that is **efficient** (the
contributions sum exactly to the grand-coalition value minus the empty-coalition
value — every point is accounted for), **symmetric** (interchangeable players get
equal credit), and shares **interaction** effects fairly rather than
double-counting them. With ``k`` players the decomposition evaluates ``2**k``
coalitions — small and offline for the handful of components or steps a real
attribution spans.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Hashable, Iterable, Iterator, Sequence
from itertools import combinations
from math import factorial
from typing import TypeVar

__all__ = [
    "coalitions",
    "shapley_from_cache",
    "shapley_values",
    "ashapley_values",
    "is_efficient",
]

P = TypeVar("P", bound=Hashable)

# A characteristic function: the value attainable by a coalition of players.
ValueFn = Callable[[frozenset[P]], float]
AsyncValueFn = Callable[[frozenset[P]], Awaitable[float]]


def coalitions(players: Iterable[P]) -> Iterator[frozenset[P]]:
    """Yield every subset of ``players`` (the powerset), smallest first."""
    names = list(players)
    for size in range(len(names) + 1):
        for subset in combinations(names, size):
            yield frozenset(subset)


def shapley_from_cache(players: Sequence[P], cache: dict[frozenset[P], float]) -> dict[P, float]:
    """Compute exact Shapley values from a fully-populated coalition→value cache.

    ``cache`` must hold a value for every subset of ``players`` (use
    :func:`coalitions` to enumerate them). Each player's value is its
    coalition-size-weighted average marginal contribution. Callers that build the
    cache themselves (with their own per-coalition setup/teardown) reuse this
    decomposition without duplicating the weighting math.
    """
    names = list(players)
    k = len(names)
    if k == 0:
        return {}
    out: dict[P, float] = {}
    for name in names:
        others = [n for n in names if n != name]
        shapley = 0.0
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(k - size - 1) / factorial(k)
            for subset in combinations(others, size):
                without = frozenset(subset)
                with_player = without | {name}
                shapley += weight * (cache[with_player] - cache[without])
        out[name] = shapley
    return out


def shapley_values(
    players: Sequence[P], value_fn: ValueFn[P]
) -> tuple[dict[P, float], dict[frozenset[P], float]]:
    """Evaluate every coalition with ``value_fn`` and return ``(shapley, cache)``.

    ``value_fn(coalition)`` is called once per distinct subset (``2**k`` total),
    memoized in the returned cache so callers can inspect the characteristic
    function or check :func:`is_efficient`.
    """
    names = list(players)
    cache: dict[frozenset[P], float] = {}
    for coalition in coalitions(names):
        if coalition not in cache:
            cache[coalition] = float(value_fn(coalition))
    return shapley_from_cache(names, cache), cache


async def ashapley_values(
    players: Sequence[P], value_fn: AsyncValueFn[P]
) -> tuple[dict[P, float], dict[frozenset[P], float]]:
    """Async counterpart of :func:`shapley_values` for an awaitable ``value_fn``
    (e.g. a coalition value computed by re-running an eval)."""
    names = list(players)
    cache: dict[frozenset[P], float] = {}
    for coalition in coalitions(names):
        if coalition not in cache:
            cache[coalition] = float(await value_fn(coalition))
    return shapley_from_cache(names, cache), cache


def is_efficient(
    players: Sequence[P],
    shapley: dict[P, float],
    cache: dict[frozenset[P], float],
    *,
    tol: float = 1e-6,
) -> bool:
    """Whether the contributions reconstruct the total value (the efficiency
    axiom): ``Σ shapley == v(grand) − v(empty)`` within ``tol``."""
    total = cache[frozenset(players)] - cache[frozenset()]
    return abs(sum(shapley.values()) - total) < tol
