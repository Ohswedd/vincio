"""The metrics engine — aggregate a benchmark's per-task results into its niche
primary metric.

Each adapter scores one task into a :class:`~vincio.evals.benchmarks.
BenchmarkResult` (a verifiable ``success`` plus a continuous ``score``); this
module folds a benchmark's results into the headline number its niche reports —
choice accuracy, exact-match, pass@k, faithfulness, contained-rate, needle-recall
— reusing the same direction conventions as :mod:`vincio.evals.metrics`. The
judge-based metrics (calibrated, ensemble, κ-tracked) live in
:mod:`vincio.evals.judges` / :mod:`vincio.evals.ensemble` and are wired through
the same adapters when a benchmark's scoring is a judge.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from ..benchmarks import BenchmarkResult

__all__ = [
    "pass_at_k",
    "accuracy",
    "mean_score",
    "bleu",
    "rouge_l",
    "summarize_results",
    "PRIMARY_METRIC_KIND",
]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower())


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """The unbiased ``pass@k`` estimator (Chen et al., 2021).

    Given ``num_samples`` completions of which ``num_correct`` pass, the
    probability that at least one of ``k`` randomly drawn completions passes:
    ``1 - C(n-c, k) / C(n, k)``. Reduces to the pass rate when ``k == n``.
    """
    n, c = int(num_samples), int(num_correct)
    if k <= 0 or n <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def accuracy(results: Sequence[BenchmarkResult]) -> float:
    """The fraction of tasks the adapter scored ``success`` (the headline rate)."""
    if not results:
        return 0.0
    return round(sum(1 for r in results if r.success) / len(results), 4)


def mean_score(results: Sequence[BenchmarkResult]) -> float:
    """The mean continuous ``score`` across tasks (partial-credit metrics)."""
    if not results:
        return 0.0
    return round(sum(r.score for r in results) / len(results), 4)


def bleu(prediction: str, reference: str, *, max_n: int = 4) -> float:
    """Sentence-level BLEU (up to ``max_n``-grams) with a brevity penalty.

    A deterministic, dependency-free generation-similarity metric for a custom
    benchmark that grades free-form text against a reference. Uniform n-gram
    weights and add-nothing smoothing (an empty candidate scores 0).
    """
    cand = _tokens(prediction)
    ref = _tokens(reference)
    if not cand or not ref:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(tuple(cand[i : i + n]) for i in range(len(cand) - n + 1))
        ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        overlap = sum(min(c, ref_ngrams[g]) for g, c in cand_ngrams.items())
        total = max(1, sum(cand_ngrams.values()))
        precisions.append(overlap / total)
    if min(precisions) == 0.0:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    brevity = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / len(cand))
    return round(brevity * geo_mean, 4)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 — the longest-common-subsequence overlap of two token sequences.

    A deterministic, dependency-free summarization-quality metric for a custom
    benchmark. Returns the F1 of LCS precision and recall.
    """
    cand = _tokens(prediction)
    ref = _tokens(reference)
    if not cand or not ref:
        return 0.0
    # LCS length via the classic rolling DP row (O(len(cand)·len(ref)) time, O(len) space).
    prev = [0] * (len(ref) + 1)
    for token in cand:
        curr = [0] * (len(ref) + 1)
        for j, ref_token in enumerate(ref, start=1):
            curr[j] = prev[j - 1] + 1 if token == ref_token else max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(cand), lcs / len(ref)
    return round(2 * precision * recall / (precision + recall), 4)


# Whether a benchmark's primary metric is reported as the binary success rate or
# the continuous mean score. Most niches are success-rate (accuracy, pass@1,
# exact-match, contained-rate, needle-recall); the partial-credit ones (RAG
# faithfulness) report the mean score.
_MEAN_SCORE_METRICS = frozenset({"faithfulness", "context_recall", "context_precision"})


def PRIMARY_METRIC_KIND(metric: str) -> str:  # noqa: N802 - public constant-style helper
    """``"mean_score"`` for partial-credit metrics, else ``"success_rate"``."""
    return "mean_score" if metric in _MEAN_SCORE_METRICS else "success_rate"


def summarize_results(
    results: Sequence[BenchmarkResult], *, primary_metric: str = "accuracy"
) -> dict[str, float | int | str]:
    """Fold a benchmark's results into its headline summary.

    Returns the ``success_rate`` and ``mean_score`` always, plus a ``primary`` value
    selected by the benchmark's ``primary_metric`` (the success rate, or the mean
    score for partial-credit metrics), and the task count ``n``.
    """
    kind = PRIMARY_METRIC_KIND(primary_metric)
    rate = accuracy(results)
    mean = mean_score(results)
    return {
        "primary_metric": primary_metric,
        "primary": mean if kind == "mean_score" else rate,
        "success_rate": rate,
        "mean_score": mean,
        "n": len(results),
    }
