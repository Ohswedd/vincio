"""Real-behavior coverage for ``vincio.optimize.test_time``.

Targets the uncovered branches of the verifier adapters and the three search
shapes: candidate normalization, the ``_coerce_verifier`` dispatch, best-of-N
deadline/cost stops, self-consistency tie-breaks and verifier scoring, beam
search expansion/deadline, and the sync conveniences. Every model interaction
uses a deterministic in-process verifier or reward (no mocks, no network).
"""

from __future__ import annotations

import time

import pytest

from vincio.evals.datasets import EvalCase
from vincio.evals.ensemble import EnsembleVerdict
from vincio.evals.judges import Judge
from vincio.evals.metrics import MetricResult, RunOutput
from vincio.optimize.rewards import RewardSample, RewardSignal, VerifiableReward
from vincio.optimize.test_time import (
    CallableVerifier,
    JudgeVerifier,
    RewardVerifier,
    SearchBudget,
    SearchCandidate,
    SearchResult,
    TestTimeSearch,
    VerifierScore,
    _as_candidate,
    _coerce_verifier,
    _normalize_answer,
)

# ---------------------------------------------------------------------------
# Test helpers: deterministic, in-process verifiers/rewards/judges (no mocks).
# ---------------------------------------------------------------------------


class _ScoreByText:
    """Verifier whose value is read from the candidate's metadata score map."""

    def __init__(self, table: dict[str, float], *, conf: float = 1.0) -> None:
        self._table = table
        self._conf = conf

    async def averify(self, candidate: SearchCandidate) -> VerifierScore:
        return VerifierScore(
            value=self._table.get(candidate.answer_text, 0.0),
            confidence=self._conf,
            source="bytext",
        )


class _SpreadJudge(Judge):
    """Plain (non-ensemble) Judge: returns a fixed value/passed plus a spread."""

    name = "spreadjudge"

    def __init__(self, value: float, *, passed: bool, spread: float) -> None:
        self._value = value
        self._passed = passed
        self._spread = spread

    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult:
        return MetricResult(
            name=self.name,
            value=self._value,
            passed=self._passed,
            details={"spread": self._spread},
        )


class _Ensemble:
    """Ensemble-shaped scorer: exposes ``averdict`` (the JudgeEnsemble protocol)."""

    def __init__(self, value: float, *, spread: float, uncertain: bool) -> None:
        self._value = value
        self._spread = spread
        self._uncertain = uncertain

    async def averdict(self, case: EvalCase, output: RunOutput) -> EnsembleVerdict:
        return EnsembleVerdict(
            value=self._value,
            disagreement={"range": self._spread},
            uncertain=self._uncertain,
        )


class _NoDetailJudge(Judge):
    """Non-ensemble Judge whose result carries no details dict entries."""

    name = "nodetail"

    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult:
        return MetricResult(name=self.name, value=0.42, passed=None)


class _FixedReward(VerifiableReward):
    name = "fixed"

    def __init__(self, value: float, *, weight: float, success: bool) -> None:
        self._value = value
        self._weight = weight
        self._success = success

    async def aevaluate(self, sample: RewardSample) -> RewardSignal:
        return RewardSignal(
            value=self._value,
            weight=self._weight,
            success=self._success,
            components={"prompt_seen": 1.0 if sample.prompt else 0.0},
        )


# ---------------------------------------------------------------------------
# SearchCandidate.answer_text  (lines 100-102)
# ---------------------------------------------------------------------------


def test_answer_text_prefers_text_field():
    c = SearchCandidate(text="hi", output="ignored")
    assert c.answer_text == "hi"


def test_answer_text_falls_back_to_str_output():
    c = SearchCandidate(output="bare")
    assert c.answer_text == "bare"


def test_answer_text_none_output_is_empty_string():
    c = SearchCandidate(output=None)
    assert c.answer_text == ""


def test_answer_text_stringifies_non_str_output():
    c = SearchCandidate(output={"a": 1})
    assert c.answer_text == str({"a": 1})


# ---------------------------------------------------------------------------
# _as_candidate  (lines 108-110, 119)
# ---------------------------------------------------------------------------


def test_as_candidate_passthrough_assigns_missing_index():
    cand = SearchCandidate(output="x")  # index defaults to 0 (falsy)
    out = _as_candidate(cand, 5)
    assert out is cand
    assert out.index == 5


def test_as_candidate_keeps_existing_index():
    cand = SearchCandidate(index=3, output="x")
    out = _as_candidate(cand, 9)
    assert out.index == 3  # truthy index is preserved


def test_as_candidate_string_sets_text_and_output():
    out = _as_candidate("hello", 2)
    assert out.text == "hello"
    assert out.output == "hello"
    assert out.index == 2


def test_as_candidate_object_without_text_derives_from_output():
    class Raw:
        output = "derived"
        cost_usd = 0.25

    out = _as_candidate(Raw(), 1)
    # No raw_text/output_text attr, output is a str -> text becomes the output.
    assert out.text == "derived"
    assert out.cost_usd == 0.25
    assert out.index == 1


def test_as_candidate_object_with_non_str_output_stringifies():
    class Raw:
        output = [1, 2, 3]

    out = _as_candidate(Raw(), 0)
    assert out.text == str([1, 2, 3])


def test_as_candidate_object_with_raw_text_used_directly():
    class Raw:
        raw_text = "explicit"
        output = "other"
        cost_usd = None  # exercises the "or 0.0" fallback

    out = _as_candidate(Raw(), 4)
    assert out.text == "explicit"
    assert out.cost_usd == 0.0


# ---------------------------------------------------------------------------
# CallableVerifier  (VerifierScore passthrough)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callable_verifier_returns_verifier_score_unchanged():
    target = VerifierScore(value=0.33, confidence=0.7, source="orig")
    v = CallableVerifier(lambda c: target)
    out = await v.averify(SearchCandidate(output="x"))
    assert out is target


@pytest.mark.asyncio
async def test_callable_verifier_tuple_sets_value_and_confidence():
    v = CallableVerifier(lambda c: (0.8, 0.25), name="pair")
    out = await v.averify(SearchCandidate(output="x"))
    assert out.value == pytest.approx(0.8)
    assert out.confidence == pytest.approx(0.25)
    assert out.source == "pair"


@pytest.mark.asyncio
async def test_callable_verifier_awaits_coroutine_result():
    async def fn(c: SearchCandidate) -> float:
        return 0.6

    out = await CallableVerifier(fn).averify(SearchCandidate(output="x"))
    assert out.value == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# JudgeVerifier  (lines 175, 201-203)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_verifier_builds_minimal_case_from_metadata():
    judge = _SpreadJudge(0.6, passed=True, spread=0.2)
    v = JudgeVerifier(judge)  # no explicit case -> _build_case path
    cand = SearchCandidate(output="ans", metadata={"input": "the question"})
    score = await v.averify(cand)
    assert score.value == 0.6
    assert score.confidence == pytest.approx(0.8)  # 1 - spread
    assert score.success is True
    assert score.source == "judge"


@pytest.mark.asyncio
async def test_judge_verifier_uses_explicit_case():
    case = EvalCase(id="fixed", input="explicit question")
    v = JudgeVerifier(_SpreadJudge(0.5, passed=False, spread=0.0), case=case)
    score = await v.averify(SearchCandidate(output="x"))
    assert score.value == 0.5
    assert score.success is False


@pytest.mark.asyncio
async def test_judge_verifier_ensemble_disagreement_lowers_confidence():
    v = JudgeVerifier(_Ensemble(0.8, spread=0.6, uncertain=True))
    score = await v.averify(SearchCandidate(output="ans", text="ans"))
    assert score.value == pytest.approx(0.8)
    assert score.confidence == pytest.approx(0.4)  # 1 - 0.6
    # uncertain panel suppresses success even though value >= 0.5.
    assert score.success is False
    assert score.details["uncertain"] is True


@pytest.mark.asyncio
async def test_judge_verifier_ensemble_confident_success():
    v = JudgeVerifier(_Ensemble(0.9, spread=0.05, uncertain=False))
    score = await v.averify(SearchCandidate(output="a"))
    assert score.success is True
    assert score.confidence == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_judge_verifier_no_details_means_full_confidence():
    v = JudgeVerifier(_NoDetailJudge())
    score = await v.averify(SearchCandidate(output="x"))
    assert score.value == pytest.approx(0.42)
    assert score.confidence == 1.0  # spread defaults to 0.0
    assert score.success is None


# ---------------------------------------------------------------------------
# RewardVerifier  (components detail)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reward_verifier_exposes_components():
    v = RewardVerifier(_FixedReward(0.9, weight=0.4, success=True))
    cand = SearchCandidate(output="o", metadata={"prompt": "p"})
    score = await v.averify(cand)
    assert score.value == pytest.approx(0.9)
    assert score.confidence == pytest.approx(0.4)
    assert score.details["components"] == {"prompt_seen": 1.0}


# ---------------------------------------------------------------------------
# _coerce_verifier dispatch  (lines 255-261)
# ---------------------------------------------------------------------------


def test_coerce_none_returns_none():
    assert _coerce_verifier(None) is None


def test_coerce_existing_verifier_is_passthrough():
    v = _ScoreByText({})
    assert _coerce_verifier(v) is v


def test_coerce_judge_with_score_wraps_in_judge_verifier():
    out = _coerce_verifier(_NoDetailJudge())
    assert isinstance(out, JudgeVerifier)


def test_coerce_reward_wraps_in_reward_verifier():
    out = _coerce_verifier(_FixedReward(0.1, weight=1.0, success=False))
    assert isinstance(out, RewardVerifier)


def test_coerce_callable_wraps_in_callable_verifier():
    out = _coerce_verifier(lambda c: 0.5)
    assert isinstance(out, CallableVerifier)


def test_coerce_unadaptable_raises_typeerror():
    with pytest.raises(TypeError, match="cannot adapt int to a Verifier"):
        _coerce_verifier(7)


# ---------------------------------------------------------------------------
# _normalize_answer
# ---------------------------------------------------------------------------


def test_normalize_collapses_whitespace_and_trims_punctuation():
    assert _normalize_answer("  Yes,   It  Is!! ") == "yes, it is"


def test_normalize_strips_trailing_punctuation_and_quotes():
    assert _normalize_answer('"Answer."') == "answer"
    assert _normalize_answer("A  B") == "a b"


# ---------------------------------------------------------------------------
# SearchResult convenience properties  (line 308)
# ---------------------------------------------------------------------------


def test_search_result_output_and_confidence_with_no_best():
    r = SearchResult(strategy="x")
    assert r.output is None
    assert r.confidence == 0.0


def test_search_result_output_and_confidence_with_best():
    best = SearchCandidate(output="win", confidence=0.66)
    r = SearchResult(strategy="x", best=best)
    assert r.output == "win"
    assert r.confidence == pytest.approx(0.66)


# ---------------------------------------------------------------------------
# best_of_n: deadline + cost stops  (lines 381-382, 402-403)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_best_of_n_requires_a_verifier():
    search = TestTimeSearch(lambda i: SearchCandidate(output=str(i)))
    with pytest.raises(ValueError, match="best_of_n requires a verifier"):
        await search.best_of_n()


@pytest.mark.asyncio
async def test_best_of_n_zero_cap_yields_empty_result():
    # max_candidates=0 (and no n override) -> the draw loop never runs.
    search = TestTimeSearch(
        lambda i: SearchCandidate(output=str(i)),
        verifier=_ScoreByText({}),
        budget=SearchBudget(max_candidates=0),
    )
    result = await search.best_of_n()
    assert result.best is None
    assert result.n_generated == 0
    assert result.stop_reason == "exhausted 0 candidate(s)"


@pytest.mark.asyncio
async def test_best_of_n_awaitable_generate():
    async def gen(i: int) -> SearchCandidate:
        return SearchCandidate(output=str(i))

    search = TestTimeSearch(
        gen,
        verifier=_ScoreByText({"0": 1.0}),
        budget=SearchBudget(max_candidates=3, confidence_target=0.9),
    )
    result = await search.best_of_n()
    assert result.best.answer_text == "0"
    assert result.early_exit is True


@pytest.mark.asyncio
async def test_best_of_n_deadline_stops_before_first_draw():
    # A deadline already in the past: the loop breaks on its very first check.
    search = TestTimeSearch(
        lambda i: SearchCandidate(output=str(i)),
        verifier=_ScoreByText({}),
        budget=SearchBudget(max_candidates=5, deadline_s=-1.0),
    )
    result = await search.best_of_n()
    assert result.n_generated == 0
    assert result.best is None
    assert "deadline reached after 0 candidate(s)" in result.stop_reason


@pytest.mark.asyncio
async def test_best_of_n_deadline_after_some_draws():
    state = {"start": None}

    def gen(i: int) -> SearchCandidate:
        # Burn time so the second iteration trips the (tiny) deadline.
        if state["start"] is None:
            state["start"] = time.monotonic()
        while time.monotonic() - state["start"] < 0.02:
            pass
        return SearchCandidate(output=str(i))

    search = TestTimeSearch(
        gen,
        verifier=_ScoreByText({}),  # scores 0.0 -> never clears bar
        budget=SearchBudget(max_candidates=10, deadline_s=0.01),
    )
    result = await search.best_of_n()
    assert result.n_generated >= 1
    assert "deadline reached" in result.stop_reason
    assert result.early_exit is False


@pytest.mark.asyncio
async def test_best_of_n_stops_when_cost_budget_spent():
    # Each candidate costs $0.10; the cost cap is $0.15 -> stop after 2 draws,
    # well before the 5-candidate cap. Scores stay below the bar.
    search = TestTimeSearch(
        lambda i: SearchCandidate(output=str(i), cost_usd=0.10),
        verifier=_ScoreByText({}),
        budget=SearchBudget(
            max_candidates=5, max_cost_usd=0.15, confidence_target=2.0
        ),
    )
    result = await search.best_of_n()
    assert result.n_generated == 2
    assert result.cost_usd == pytest.approx(0.20)
    assert "cost budget spent" in result.stop_reason
    assert result.early_exit is False


# ---------------------------------------------------------------------------
# self_consistency: deadline, cost, empty, tie-break, scoring
# (lines 434->464, 436-437, 461-462, 465, 471-474, 480-482)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_consistency_no_candidates_on_immediate_deadline():
    search = TestTimeSearch(
        lambda i: SearchCandidate(output="a"),
        budget=SearchBudget(max_candidates=5, deadline_s=-1.0),
    )
    result = await search.self_consistency()
    assert result.best is None
    assert result.stop_reason == "no candidates"
    assert result.n_generated == 0


@pytest.mark.asyncio
async def test_self_consistency_cost_budget_stops_the_draw():
    answers = ["a", "b", "c", "d", "e"]
    search = TestTimeSearch(
        lambda i: SearchCandidate(output=answers[i], cost_usd=0.10),
        budget=SearchBudget(
            max_candidates=5, min_candidates=5, max_cost_usd=0.15
        ),
    )
    result = await search.self_consistency()
    # All-distinct answers can never majority-lock; cost cap halts at 2 draws.
    assert result.n_generated == 2
    assert "cost budget spent" in result.stop_reason


@pytest.mark.asyncio
async def test_self_consistency_tie_break_uses_verifier():
    # Two answers each get 2 votes (a tie). The verifier prefers "b".
    answers = ["a", "b", "a", "b"]
    search = TestTimeSearch(
        lambda i: SearchCandidate(output=answers[i]),
        verifier=_ScoreByText({"a": 0.1, "b": 0.9}),
        budget=SearchBudget(max_candidates=4, min_candidates=4),
    )
    result = await search.self_consistency()
    assert result.best.answer_text == "b"
    assert result.votes == {"a": 2, "b": 2}
    assert result.n_scored >= 2  # both tied representatives were scored


@pytest.mark.asyncio
async def test_self_consistency_scores_unverified_winner_with_verifier():
    # Clear majority "yes" (3 of 4); winner not yet verified -> gets scored once.
    answers = ["yes", "yes", "no", "yes"]
    search = TestTimeSearch(
        lambda i: SearchCandidate(output=answers[i]),
        verifier=_ScoreByText({"yes": 0.77}),
        budget=SearchBudget(max_candidates=4, min_candidates=4),
    )
    result = await search.self_consistency()
    assert result.best.answer_text == "yes"
    assert result.best.verified is True
    assert result.best.score == pytest.approx(0.77)
    # confidence is the vote share (3/4), not the verifier value.
    assert result.best.confidence == pytest.approx(0.75)
    assert result.best.metadata["votes"] == 3
    assert result.n_scored == 1


@pytest.mark.asyncio
async def test_self_consistency_custom_normalizer():
    search = TestTimeSearch(
        lambda i: SearchCandidate(output=["YES", "yes", "no"][i]),
        budget=SearchBudget(max_candidates=3, min_candidates=3),
    )
    result = await search.self_consistency(normalizer=lambda s: s.lower())
    # "YES" and "yes" fold together -> "yes" wins 2-1.
    assert result.votes == {"yes": 2, "no": 1}
    assert result.best.confidence == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# beam_search: errors, awaitable expand, deadline, max-depth
# (lines 516, 523, 544->572, 549-550, 573)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_beam_search_requires_a_verifier():
    search = TestTimeSearch(lambda i: SearchCandidate(output=str(i)))
    with pytest.raises(ValueError, match="requires a verifier or a score function"):
        await search.beam_search(root=0, expand=lambda s: [])


@pytest.mark.asyncio
async def test_beam_search_awaitable_expand_and_score_fn():
    # expand returns a coroutine -> exercises the awaitable branch.
    async def expand(state: int):
        if state >= 3:
            return []
        return [state + 1, state + 10]

    search = TestTimeSearch(
        lambda i: SearchCandidate(output=str(i)),
        budget=SearchBudget(max_candidates=50),
    )
    result = await search.beam_search(
        root=0,
        expand=expand,
        beam_width=2,
        max_depth=5,
        score=lambda c: float(c.output),  # higher state == better
    )
    assert result.strategy == "beam_search"
    assert result.best is not None
    # The single largest reachable leaf wins (states grow by +10 each level).
    assert result.best.score == result.best.output


@pytest.mark.asyncio
async def test_beam_search_reaches_max_depth():
    # Infinite tree, but max_depth bounds it.
    def expand(state: int):
        return [state + 1]

    search = TestTimeSearch(
        lambda i: SearchCandidate(output=str(i)),
        verifier=CallableVerifier(lambda c: float(c.output)),
        budget=SearchBudget(max_candidates=100),
    )
    result = await search.beam_search(
        root=0, expand=expand, beam_width=1, max_depth=3
    )
    assert result.stop_reason == "reached max depth 3"
    assert result.n_scored == 3  # one state scored per depth at beam_width=1


@pytest.mark.asyncio
async def test_beam_search_candidate_budget_spent_mid_frontier():
    # Root expands into 5 states but the budget only allows 3 scorings, so the
    # search stops part-way through the very first frontier.
    def expand(state):
        if state == "root":
            return ["s0", "s1", "s2", "s3", "s4"]
        return ["deeper"]

    search = TestTimeSearch(
        lambda i: SearchCandidate(output=str(i)),
        verifier=CallableVerifier(lambda c: len(str(c.output))),
        budget=SearchBudget(max_candidates=3),
    )
    result = await search.beam_search(
        root="root", expand=expand, beam_width=4, max_depth=5
    )
    assert result.n_scored == 3
    assert "candidate budget spent (3) at depth 0" in result.stop_reason
    assert result.best is not None  # the partial beam still yields a best


@pytest.mark.asyncio
async def test_beam_search_zero_budget_scores_nothing():
    # budget_n=0 -> the inner loop breaks immediately, beam stays empty and no
    # best is ever set (the empty-beam branch of the best update).
    search = TestTimeSearch(
        lambda i: SearchCandidate(output=str(i)),
        verifier=CallableVerifier(lambda c: 1.0),
        budget=SearchBudget(max_candidates=0),
    )
    result = await search.beam_search(
        root="root", expand=lambda s: ["a", "b"], beam_width=2, max_depth=3
    )
    assert result.n_scored == 0
    assert result.best is None
    assert "candidate budget spent (0) at depth 0" in result.stop_reason


@pytest.mark.asyncio
async def test_beam_search_deadline_stops_at_depth():
    state = {"start": None}

    def expand(s: int):
        if state["start"] is None:
            state["start"] = time.monotonic()
        while time.monotonic() - state["start"] < 0.02:
            pass
        return [s + 1, s + 2]

    search = TestTimeSearch(
        lambda i: SearchCandidate(output=str(i)),
        verifier=CallableVerifier(lambda c: float(c.output)),
        budget=SearchBudget(max_candidates=100, deadline_s=0.01),
    )
    result = await search.beam_search(
        root=0, expand=expand, beam_width=2, max_depth=10
    )
    assert "deadline reached at depth" in result.stop_reason


@pytest.mark.asyncio
async def test_beam_search_frontier_exhausted_when_root_terminal():
    search = TestTimeSearch(
        lambda i: SearchCandidate(output=str(i)),
        verifier=CallableVerifier(lambda c: 1.0),
    )
    result = await search.beam_search(root=0, expand=lambda s: [], max_depth=4)
    assert result.stop_reason == "frontier exhausted at depth 0"
    assert result.best is None
    assert result.n_scored == 0


# ---------------------------------------------------------------------------
# Sync conveniences  (lines 588, 591)
# ---------------------------------------------------------------------------


def test_run_best_of_n_sync():
    search = TestTimeSearch(
        lambda i: SearchCandidate(output=str(i)),
        verifier=CallableVerifier(lambda c: 1.0),
        budget=SearchBudget(max_candidates=3, confidence_target=0.5),
    )
    result = search.run_best_of_n()
    assert result.strategy == "best_of_n"
    assert result.early_exit is True  # first draw clears the 0.5 bar
    assert result.best.score == 1.0


def test_run_self_consistency_sync():
    answers = ["x", "x", "y"]
    search = TestTimeSearch(
        lambda i: SearchCandidate(output=answers[i]),
        budget=SearchBudget(max_candidates=3, min_candidates=3),
    )
    result = search.run_self_consistency()
    assert result.strategy == "self_consistency"
    assert result.best.answer_text == "x"
    assert result.votes == {"x": 2, "y": 1}
