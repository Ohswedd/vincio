"""Vectorized candidate scoring.

The context scorer evaluates a weighted utility over every candidate. For a
large candidate set, scoring each candidate in a Python loop — building a
validated :class:`~vincio.context.scoring.ContextScores` model per item and
recomputing the weighted sum one component at a time — dominates the compile
hot path.

This module batches that work into a single pass: the per-component scores are
laid out as columns and reduced against the signed weight vector in one
operation. When NumPy is installed the reduction is a matrix–vector product and
semantic relevance collapses to a single matrix–vector product over the cached
candidate embeddings; otherwise an equivalent pure-Python reduction runs. The
pure-Python path is the zero-dependency default and produces bit-for-bit the
same selection as the per-candidate loop — NumPy is an optional accelerator,
never a requirement.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:  # optional acceleration — the pure-Python reduction stays the default
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only when numpy is absent
    _np = None  # type: ignore[assignment]

__all__ = [
    "HAS_NUMPY",
    "weighted_totals",
    "row_normalize",
    "matrix_vector_cosine",
]

HAS_NUMPY = _np is not None


def weighted_totals(
    columns: Sequence[Sequence[float]], weights: Sequence[float]
) -> list[float]:
    """Reduce per-component score *columns* against signed *weights*.

    ``columns[k]`` is the length-``n`` vector of component ``k`` across all
    candidates and ``weights[k]`` its signed weight (negative for penalties).
    Returns the length-``n`` total-utility vector. The NumPy path is a single
    matrix–vector product; the fallback is the identical weighted sum.
    """
    if not columns:
        return []
    n = len(columns[0])
    if _np is not None:
        matrix = _np.asarray(columns, dtype=float)  # (k, n)
        weight_vec = _np.asarray(weights, dtype=float)  # (k,)
        return list(map(float, weight_vec @ matrix))  # (n,)
    totals = [0.0] * n
    for column, weight in zip(columns, weights, strict=True):
        if weight == 0.0:
            continue
        for i, value in enumerate(column):
            totals[i] += weight * value
    return totals


def row_normalize(vectors: Sequence[Sequence[float]]) -> Any | None:
    """Row-normalized matrix of *vectors* for cosine via dot product.

    Returns a NumPy array (or ``None`` when NumPy is absent, signalling the
    caller to use the pure-Python cosine path). Zero rows are left at zero so
    their cosine is zero, matching :func:`vincio.retrieval.embeddings.cosine`.
    """
    if _np is None or not vectors:
        return None
    matrix = _np.asarray(vectors, dtype=float)
    norms = _np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0
    return matrix / norms[:, None]


def matrix_vector_cosine(normalized_matrix: Any, vector: Sequence[float]) -> list[float]:
    """Cosine of every row of a row-normalized matrix against *vector*."""
    assert _np is not None  # noqa: S101 - only called on the NumPy path (the caller returns None when NumPy is absent)
    vec = _np.asarray(vector, dtype=float)
    norm = _np.linalg.norm(vec)
    if norm == 0.0:
        return [0.0] * len(normalized_matrix)
    sims = normalized_matrix @ (vec / norm)
    return [max(0.0, min(1.0, float(s))) for s in sims]
