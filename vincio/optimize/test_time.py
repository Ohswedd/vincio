"""Test-time compute search (optimize/test_time).

Parallel test-time search is the other cheap quality lever: instead of (or on
top of) thinking harder on one rollout, draw several and keep the best one a
*verifier* trusts. :class:`TestTimeSearch` runs three classic shapes —

- **best-of-N**: draw candidates, score each with a verifier, keep the best, and
  early-exit the moment a candidate clears the bar (so an easy ask costs one
  draw, a hard one spends the budget);
- **self-consistency**: draw candidates, take the majority answer, and early-exit
  the moment the lead is mathematically unbeatable by the remaining draws;
- **beam search**: expand a tree of partial tool-use trajectories, keeping the
  top-``beam_width`` by verifier score at each depth —

all scored by the platform's *existing* critics. A :class:`Verifier` is a thin
protocol; the adapters here wrap any :class:`~vincio.evals.judges.Judge` /
:class:`~vincio.evals.ensemble.JudgeEnsemble` (disagreement becomes low
confidence), any :class:`~vincio.optimize.rewards.VerifiableReward` /
:class:`~vincio.optimize.rewards.RewardModel`, or a plain callable. Nothing new
learns to judge — test-time search reuses what evaluation already proved.

Every search is bounded by a :class:`SearchBudget` — a candidate cap, an optional
cost cap, and an optional wall-clock deadline — so it composes with the same
fair-share budgets the orchestrator enforces. The decision is recorded in a
:class:`SearchResult` (every candidate, its score, why the search stopped) so the
spend is explained, not silent.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..core.utils import new_id
from ..providers.base import run_sync

__all__ = [
    "VerifierScore",
    "Verifier",
    "CallableVerifier",
    "JudgeVerifier",
    "RewardVerifier",
    "SearchCandidate",
    "SearchBudget",
    "SearchResult",
    "TestTimeSearch",
]


class VerifierScore(BaseModel):
    """A verifier's verdict on one candidate: a value, a confidence, a reason.

    ``value`` is the quality score in ``[0, 1]`` the search maximizes;
    ``confidence`` in ``[0, 1]`` is how much to trust that score (a split judge
    panel reports low confidence). ``success`` is the binary verdict when the
    verifier has one (a verifiable reward, an oracle).
    """

    value: float = 0.0
    confidence: float = 1.0
    success: bool | None = None
    source: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Verifier(Protocol):
    """Scores a candidate answer or trajectory. Reuse an existing critic via the
    adapters in this module rather than implementing this directly."""

    async def averify(self, candidate: SearchCandidate) -> VerifierScore: ...


class SearchCandidate(BaseModel):
    """One generated candidate the search scores and ranks."""

    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: new_id("cand"))
    index: int = 0
    output: Any = None
    text: str = ""
    trajectory: Any = None
    cost_usd: float = 0.0
    score: float = 0.0
    confidence: float = 1.0
    success: bool | None = None
    verified: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def answer_text(self) -> str:
        """Normalized text used for voting / display (falls back to ``output``)."""
        if self.text:
            return self.text
        if isinstance(self.output, str):
            return self.output
        return "" if self.output is None else str(self.output)


def _as_candidate(raw: Any, index: int) -> SearchCandidate:
    """Normalize whatever ``generate`` returned into a :class:`SearchCandidate`."""
    if isinstance(raw, SearchCandidate):
        if not raw.index:
            raw.index = index
        return raw
    if isinstance(raw, str):
        return SearchCandidate(index=index, output=raw, text=raw)
    # A RunResult / RunOutput-like object: pull text + cost if present.
    text = getattr(raw, "raw_text", "") or getattr(raw, "output_text", "")
    output = getattr(raw, "output", raw)
    cost = float(getattr(raw, "cost_usd", 0.0) or 0.0)
    traj = getattr(raw, "trajectory", None)
    if not text:
        text = output if isinstance(output, str) else (str(output) if output is not None else "")
    return SearchCandidate(
        index=index, output=output, text=text, trajectory=traj, cost_usd=cost
    )


# ---------------------------------------------------------------------------
# Verifier adapters over the platform's existing critics
# ---------------------------------------------------------------------------


class CallableVerifier:
    """Wrap a plain callable into a :class:`Verifier`.

    The callable receives the :class:`SearchCandidate` and returns a float, a
    ``(value, confidence)`` pair, or a :class:`VerifierScore`. Sync or async.
    """

    def __init__(
        self, fn: Callable[[SearchCandidate], Any], *, name: str = "callable"
    ) -> None:
        self._fn = fn
        self.name = name

    async def averify(self, candidate: SearchCandidate) -> VerifierScore:
        result = self._fn(candidate)
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, VerifierScore):
            return result
        if isinstance(result, tuple):
            value, confidence = result
            return VerifierScore(value=float(value), confidence=float(confidence), source=self.name)
        return VerifierScore(value=float(result), source=self.name)


class JudgeVerifier:
    """Score candidates with any :class:`~vincio.evals.judges.Judge` or
    :class:`~vincio.evals.ensemble.JudgeEnsemble`.

    ``case`` is the :class:`~vincio.evals.datasets.EvalCase` to score against
    (task input, optional reference, rubric); when omitted a minimal case is
    built from each candidate. For an ensemble the panel's disagreement becomes
    the verifier's confidence (``1 − spread``), so a split panel is trusted less.
    """

    def __init__(self, judge: Any, *, case: Any = None, name: str = "judge") -> None:
        self._judge = judge
        self._case = case
        self.name = name

    def _build_case(self, candidate: SearchCandidate) -> Any:
        from ..evals.datasets import EvalCase

        if self._case is not None:
            return self._case
        return EvalCase(id=candidate.id, input=candidate.metadata.get("input", ""))

    def _build_output(self, candidate: SearchCandidate) -> Any:
        from ..evals.metrics import RunOutput

        return RunOutput(
            output=candidate.output,
            raw_text=candidate.answer_text,
            evidence=list(candidate.metadata.get("evidence", []) or []),
            trajectory=candidate.trajectory,
            cost_usd=candidate.cost_usd,
        )

    async def averify(self, candidate: SearchCandidate) -> VerifierScore:
        case = self._build_case(candidate)
        output = self._build_output(candidate)
        if hasattr(self._judge, "averdict"):  # JudgeEnsemble — carries disagreement
            verdict = await self._judge.averdict(case, output)
            confidence = max(0.0, 1.0 - float(getattr(verdict, "spread", 0.0)))
            return VerifierScore(
                value=float(verdict.value),
                confidence=confidence,
                success=(verdict.value >= 0.5 and not getattr(verdict, "uncertain", False)),
                source=self.name,
                details={"uncertain": getattr(verdict, "uncertain", False)},
            )
        result = await self._judge.score(case, output)
        spread = float(result.details.get("spread", 0.0)) if result.details else 0.0
        return VerifierScore(
            value=float(result.value),
            confidence=max(0.0, 1.0 - spread),
            success=result.passed,
            source=self.name,
        )


class RewardVerifier:
    """Score candidates with any :class:`~vincio.optimize.rewards.VerifiableReward`
    or :class:`~vincio.optimize.rewards.RewardModel`.

    The reward's confidence ``weight`` (e.g. lowered by judge disagreement, or
    full for an oracle) becomes the verifier's confidence, so verifiable scorers
    dominate the search exactly as they dominate the reward blend.
    """

    def __init__(self, reward: Any, *, name: str = "reward") -> None:
        self._reward = reward
        self.name = name

    def _build_sample(self, candidate: SearchCandidate) -> Any:
        from .rewards import RewardSample

        meta = candidate.metadata
        return RewardSample(
            task_id=candidate.id,
            prompt=meta.get("prompt", ""),
            output=candidate.output,
            gold=meta.get("gold"),
            inputs=meta.get("inputs", {}) or {},
            trajectory=candidate.trajectory,
            verification=meta.get("verification"),
        )

    async def averify(self, candidate: SearchCandidate) -> VerifierScore:
        signal = await self._reward.aevaluate(self._build_sample(candidate))
        return VerifierScore(
            value=float(signal.value),
            confidence=float(signal.weight),
            success=signal.success,
            source=self.name,
            details={"components": dict(signal.components)},
        )


def _coerce_verifier(verifier: Any) -> Verifier | None:
    """Accept a Verifier, a Judge/JudgeEnsemble, a reward, or a callable."""
    if verifier is None:
        return None
    if hasattr(verifier, "averify"):
        return verifier
    if hasattr(verifier, "averdict") or hasattr(verifier, "score"):
        return JudgeVerifier(verifier)
    if hasattr(verifier, "aevaluate"):
        return RewardVerifier(verifier)
    if callable(verifier):
        return CallableVerifier(verifier)
    raise TypeError(f"cannot adapt {type(verifier).__name__} to a Verifier")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalize_answer(text: str) -> str:
    """Default vote key: case-folded, whitespace- and punctuation-trimmed."""
    return _WS_RE.sub(" ", text.strip().lower()).strip(" .!?\"'`")


class SearchBudget(BaseModel):
    """Bounds one search: candidate cap, optional cost cap, optional deadline.

    ``confidence_target`` is the bar a best-of-N candidate must clear to trigger
    early-exit; ``min_candidates`` is drawn before any early-exit so a single
    lucky draw cannot end the search prematurely.
    """

    max_candidates: int = 8
    min_candidates: int = 1
    max_cost_usd: float | None = None
    deadline_s: float | None = None
    confidence_target: float = 0.9


class SearchResult(BaseModel):
    """The outcome of a search: the winner, every candidate, and why it stopped."""

    model_config = {"arbitrary_types_allowed": True}

    strategy: str
    best: SearchCandidate | None = None
    candidates: list[SearchCandidate] = Field(default_factory=list)
    votes: dict[str, int] | None = None
    n_generated: int = 0
    n_scored: int = 0
    early_exit: bool = False
    stop_reason: str = ""
    cost_usd: float = 0.0

    @property
    def output(self) -> Any:
        return self.best.output if self.best is not None else None

    @property
    def confidence(self) -> float:
        return self.best.confidence if self.best is not None else 0.0


# A generator draws one candidate for attempt ``index`` (sync or async).
Generate = Callable[[int], "Any | Awaitable[Any]"]


class TestTimeSearch:
    """Verifier-guided test-time search bounded by a :class:`SearchBudget`.

    ``generate(index)`` draws one candidate (vary temperature/seed by ``index``
    for diversity); it may return a string, a ``RunResult``-like object, or a
    :class:`SearchCandidate`. ``verifier`` may be any object the adapters accept;
    it is required for :meth:`best_of_n` and :meth:`beam_search` and optional for
    :meth:`self_consistency`.
    """

    # Not a pytest test class despite the ``Test`` prefix — keep collectors away.
    __test__ = False

    def __init__(
        self,
        generate: Generate,
        *,
        verifier: Any = None,
        budget: SearchBudget | None = None,
    ) -> None:
        self._generate = generate
        self.verifier = _coerce_verifier(verifier)
        self.budget = budget or SearchBudget()

    async def _draw(self, index: int) -> SearchCandidate:
        raw = self._generate(index)
        if hasattr(raw, "__await__"):
            raw = await raw
        return _as_candidate(raw, index)

    async def _score(self, candidate: SearchCandidate) -> SearchCandidate:
        assert self.verifier is not None  # noqa: S101 - the caller scores only when a verifier is configured
        verdict = await self.verifier.averify(candidate)
        candidate.score = verdict.value
        candidate.confidence = verdict.confidence
        candidate.success = verdict.success
        candidate.verified = True
        candidate.metadata.setdefault("verifier", verdict.source)
        return candidate

    def _deadline(self) -> float | None:
        d = self.budget.deadline_s
        return (time.monotonic() + d) if d is not None else None

    async def best_of_n(self, n: int | None = None) -> SearchResult:
        """Draw up to ``n`` candidates, verify each, keep the best.

        Stops early once a verified candidate's score clears
        ``budget.confidence_target`` (after at least ``min_candidates`` draws),
        or when the cost/time budget is spent.
        """
        if self.verifier is None:
            raise ValueError("best_of_n requires a verifier")
        cap = n or self.budget.max_candidates
        deadline = self._deadline()
        candidates: list[SearchCandidate] = []
        best: SearchCandidate | None = None
        cost = 0.0
        early = False
        reason = f"exhausted {cap} candidate(s)"
        for index in range(cap):
            if deadline is not None and time.monotonic() >= deadline:
                reason = f"deadline reached after {len(candidates)} candidate(s)"
                break
            candidate = await self._score(await self._draw(index))
            candidates.append(candidate)
            cost += candidate.cost_usd
            if best is None or candidate.score > best.score:
                best = candidate
            if (
                len(candidates) >= self.budget.min_candidates
                and best.score >= self.budget.confidence_target
            ):
                early = True
                reason = (
                    f"verifier cleared the bar ({best.score:.3f} >= "
                    f"{self.budget.confidence_target:.3f}) after {len(candidates)} candidate(s)"
                )
                break
            if (
                self.budget.max_cost_usd is not None
                and cost >= self.budget.max_cost_usd
            ):
                reason = f"cost budget spent (${cost:.4f}) after {len(candidates)} candidate(s)"
                break
        return SearchResult(
            strategy="best_of_n",
            best=best,
            candidates=candidates,
            n_generated=len(candidates),
            n_scored=len(candidates),
            early_exit=early,
            stop_reason=reason,
            cost_usd=round(cost, 8),
        )

    async def self_consistency(
        self, n: int | None = None, *, normalizer: Callable[[str], str] | None = None
    ) -> SearchResult:
        """Draw up to ``n`` candidates and return the majority answer.

        Early-exits the instant the leading answer's margin cannot be overtaken
        by the remaining draws. A verifier is optional: when present it scores the
        representative of each answer cluster and breaks ties; when absent the
        vote share is the confidence.
        """
        cap = n or self.budget.max_candidates
        normalize = normalizer or _normalize_answer
        deadline = self._deadline()
        candidates: list[SearchCandidate] = []
        votes: dict[str, int] = {}
        representatives: dict[str, SearchCandidate] = {}
        cost = 0.0
        early = False
        reason = f"exhausted {cap} candidate(s)"
        for index in range(cap):
            if deadline is not None and time.monotonic() >= deadline:
                reason = f"deadline reached after {len(candidates)} candidate(s)"
                break
            candidate = await self._draw(index)
            candidates.append(candidate)
            cost += candidate.cost_usd
            key = normalize(candidate.answer_text)
            votes[key] = votes.get(key, 0) + 1
            representatives.setdefault(key, candidate)
            # Majority-lock early-exit: if the leader's margin exceeds the draws
            # still to come, no outcome can change the winner.
            ranked = sorted(votes.values(), reverse=True)
            leader = ranked[0]
            runner_up = ranked[1] if len(ranked) > 1 else 0
            remaining = cap - len(candidates)
            if (
                len(candidates) >= self.budget.min_candidates
                and leader - runner_up > remaining
            ):
                early = True
                reason = (
                    f"majority locked ({leader}/{len(candidates)}) after "
                    f"{len(candidates)} candidate(s)"
                )
                break
            if self.budget.max_cost_usd is not None and cost >= self.budget.max_cost_usd:
                reason = f"cost budget spent (${cost:.4f}) after {len(candidates)} candidate(s)"
                break

        if not candidates:
            return SearchResult(strategy="self_consistency", stop_reason="no candidates")
        # Pick the most-voted answer; tie-break by verifier score when available.
        best_key = max(votes, key=lambda k: votes[k])
        leaders = [k for k, v in votes.items() if v == votes[best_key]]
        n_scored = 0
        if len(leaders) > 1 and self.verifier is not None:
            scored = [await self._score(representatives[k]) for k in leaders]
            n_scored = len(scored)
            winner = max(scored, key=lambda c: c.score)
            best_key = normalize(winner.answer_text)
        best = representatives[best_key]
        best.confidence = votes[best_key] / len(candidates)
        best.metadata["votes"] = votes[best_key]
        best.metadata["vote_share"] = round(best.confidence, 4)
        if self.verifier is not None and not best.verified:
            best = await self._score(best)
            best.confidence = votes[best_key] / len(candidates)
            n_scored += 1
        return SearchResult(
            strategy="self_consistency",
            best=best,
            candidates=candidates,
            votes=votes,
            n_generated=len(candidates),
            n_scored=n_scored,
            early_exit=early,
            stop_reason=reason,
            cost_usd=round(cost, 8),
        )

    async def beam_search(
        self,
        *,
        root: Any,
        expand: Callable[[Any], Any],
        beam_width: int = 4,
        max_depth: int = 4,
        score: Any = None,
        state_text: Callable[[Any], str] | None = None,
    ) -> SearchResult:
        """Beam search over a tree of partial tool-use trajectories.

        ``expand(state)`` returns the successor states of a state (sync or async;
        ``[]`` marks a terminal). Each state is scored by ``score`` (a verifier or
        callable, defaulting to this search's ``verifier``); the top
        ``beam_width`` states are kept at each of ``max_depth`` rounds. The search
        is bounded by ``budget.max_candidates`` total scorings and the deadline.
        Returns the highest-scoring state reached.
        """
        verifier = _coerce_verifier(score) if score is not None else self.verifier
        if verifier is None:
            raise ValueError("beam_search requires a verifier or a score function")
        to_text = state_text or (lambda s: s if isinstance(s, str) else str(s))
        deadline = self._deadline()

        async def expand_state(state: Any) -> list[Any]:
            successors = expand(state)
            if hasattr(successors, "__await__"):
                successors = await successors
            return list(successors or [])

        async def score_state(state: Any, index: int) -> SearchCandidate:
            cand = SearchCandidate(index=index, output=state, text=to_text(state))
            verdict = await verifier.averify(cand)
            cand.score = verdict.value
            cand.confidence = verdict.confidence
            cand.success = verdict.success
            cand.verified = True
            return cand

        scored_all: list[SearchCandidate] = []
        budget_n = self.budget.max_candidates
        counter = 0
        stopped = ""

        # Seed the beam from the root's successors.
        frontier_states = await expand_state(root)
        beam: list[SearchCandidate] = []
        best: SearchCandidate | None = None
        for depth in range(max_depth):
            if not frontier_states:
                stopped = f"frontier exhausted at depth {depth}"
                break
            if deadline is not None and time.monotonic() >= deadline:
                stopped = f"deadline reached at depth {depth}"
                break
            scored: list[SearchCandidate] = []
            for state in frontier_states:
                if counter >= budget_n:
                    stopped = f"candidate budget spent ({budget_n}) at depth {depth}"
                    break
                cand = await score_state(state, counter)
                cand.metadata["depth"] = depth
                counter += 1
                scored.append(cand)
                scored_all.append(cand)
            scored.sort(key=lambda c: c.score, reverse=True)
            beam = scored[:beam_width]
            if beam and (best is None or beam[0].score > best.score):
                best = beam[0]
            if stopped:
                break
            # Expand the surviving beam into the next frontier.
            next_states: list[Any] = []
            for cand in beam:
                next_states.extend(await expand_state(cand.output))
            frontier_states = next_states
        if not stopped:
            stopped = f"reached max depth {max_depth}"
        return SearchResult(
            strategy="beam_search",
            best=best,
            candidates=scored_all,
            n_generated=counter,
            n_scored=counter,
            early_exit=False,
            stop_reason=stopped,
            cost_usd=round(sum(c.cost_usd for c in scored_all), 8),
        )

    # Sync conveniences ----------------------------------------------------

    def run_best_of_n(self, n: int | None = None) -> SearchResult:
        return run_sync(self.best_of_n(n))

    def run_self_consistency(self, n: int | None = None) -> SearchResult:
        return run_sync(self.self_consistency(n))
