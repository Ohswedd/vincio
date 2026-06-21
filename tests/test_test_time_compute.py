"""Test-time compute & reasoning orchestration.

Covers the reasoning-trace cache, the reasoning-effort controller, the
verifier-guided test-time search (best-of-N, self-consistency, beam), the
verifier adapters over existing critics, and the runtime/app wiring.
"""

from __future__ import annotations

import pytest

from vincio import (
    ContextApp,
    ReasoningController,
    ReasoningPolicy,
    ReasoningTraceCache,
    SearchBudget,
    TaskType,
    TestTimeSearch,
)
from vincio.agents.reasoning import ReasoningDecision
from vincio.caching.reasoning import ReasoningTrace, reasoning_prefix_key
from vincio.optimize.test_time import (
    CallableVerifier,
    JudgeVerifier,
    RewardVerifier,
    SearchCandidate,
    VerifierScore,
)
from vincio.providers.mock import MockProvider

# ---------------------------------------------------------------------------
# ReasoningTraceCache
# ---------------------------------------------------------------------------


class TestReasoningTraceCache:
    def test_key_is_deterministic_and_effort_sensitive(self):
        a = reasoning_prefix_key("hash1", "gpt", "high")
        b = reasoning_prefix_key("hash1", "gpt", "high")
        c = reasoning_prefix_key("hash1", "gpt", "low")
        assert a == b
        assert a != c  # effort is part of the identity

    def test_record_and_lookup(self):
        cache = ReasoningTraceCache()
        cache.record(prefix_hash="p", model="m", effort="high", reasoning_tokens=128)
        hit = cache.lookup("p", "m", "high")
        assert hit is not None
        assert hit.reasoning_tokens == 128
        assert cache.lookup("p", "m", "low") is None
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 1

    def test_lru_entry_eviction(self):
        cache = ReasoningTraceCache(max_entries=2)
        cache.record(prefix_hash="a", model="m")
        cache.record(prefix_hash="b", model="m")
        cache.lookup("a", "m")  # touch a → b is now LRU
        cache.record(prefix_hash="c", model="m")  # evicts b
        assert cache.lookup("a", "m") is not None
        assert cache.lookup("b", "m") is None
        assert cache.lookup("c", "m") is not None
        assert len(cache) == 2

    def test_byte_budget_eviction(self):
        cache = ReasoningTraceCache(max_entries=100, max_resident_bytes=600)
        # Each entry ~ 256 overhead + text bytes; 3 entries blow a 600B ceiling.
        for i in range(3):
            cache.record(prefix_hash=f"p{i}", model="m", trace_text="x" * 100)
        assert cache.resident_bytes <= 600
        assert len(cache) < 3

    def test_keeps_one_oversized_entry(self):
        cache = ReasoningTraceCache(max_resident_bytes=10)
        cache.record(prefix_hash="big", model="m", trace_text="y" * 5000)
        assert len(cache) == 1  # a single oversized trace is still a hit

    def test_reinsert_does_not_double_count_bytes(self):
        cache = ReasoningTraceCache()
        cache.record(prefix_hash="p", model="m", trace_text="abc")
        first = cache.resident_bytes
        cache.record(prefix_hash="p", model="m", trace_text="abc")
        assert cache.resident_bytes == first
        assert len(cache) == 1

    def test_seq_is_monotonic_not_wallclock(self):
        cache = ReasoningTraceCache()
        t1 = cache.record(prefix_hash="a", model="m")
        t2 = cache.record(prefix_hash="b", model="m")
        assert t2.seq == t1.seq + 1

    def test_estimate_bytes(self):
        trace = ReasoningTrace(key="k", prefix_hash="p", model="m", trace_text="hello")
        assert trace.estimate_bytes() == 256 + len("hello")


# ---------------------------------------------------------------------------
# ReasoningController
# ---------------------------------------------------------------------------


class TestReasoningController:
    def test_easy_task_gets_minimal_effort(self):
        ctl = ReasoningController()
        d = ctl.decide(task=TaskType.CLASSIFICATION, text="cat or dog?")
        assert d.effort == "minimal"
        assert d.difficulty < 0.3

    def test_hard_task_gets_high_effort(self):
        ctl = ReasoningController()
        text = "Carefully analyze, compare and reason about why " + "the merger " * 80
        d = ctl.decide(task=TaskType.DOCUMENT_COMPARISON, text=text)
        assert d.effort == "high"
        assert d.difficulty > 0.65

    def test_hard_ceiling_caps_thinking_budget(self):
        ctl = ReasoningController(ReasoningPolicy(max_reasoning_tokens=2000))
        d = ctl.decide(difficulty=0.95)  # would map to "high" → 16384 tokens
        assert d.effort == "high"
        assert d.thinking_budget_tokens == 2000
        assert d.ceiling_capped is True

    def test_budget_fraction_caps_thinking_budget(self):
        ctl = ReasoningController(ReasoningPolicy(budget_fraction=0.5))
        d = ctl.decide(difficulty=0.95, remaining_output_tokens=1000)
        assert d.thinking_budget_tokens == 500
        assert d.budget_capped is True

    def test_low_confidence_escalates_one_level(self):
        ctl = ReasoningController()
        base = ctl.decide(difficulty=0.5)
        escalated = ctl.decide(difficulty=0.5, confidence=0.1)
        assert base.effort == "medium"
        assert escalated.effort == "high"
        assert escalated.escalated is True

    def test_warm_prefix_steps_effort_down(self):
        cache = ReasoningTraceCache()
        ctl = ReasoningController(trace_cache=cache)
        # cold: difficulty 0.5 → medium
        cold = ctl.decide(difficulty=0.5, prefix_hash="p", model="m")
        assert cold.effort == "medium"
        assert cold.warm_prefix is False
        # warm the same prefix at the chosen effort
        cache.record(prefix_hash="p", model="m", effort="medium")
        warm = ctl.decide(difficulty=0.5, prefix_hash="p", model="m")
        assert warm.warm_prefix is True
        assert warm.effort == "low"  # stepped down

    def test_effort_clamped_to_policy_band(self):
        ctl = ReasoningController(ReasoningPolicy(min_effort="low", max_effort="medium"))
        easy = ctl.decide(difficulty=0.0)
        hard = ctl.decide(difficulty=1.0)
        assert easy.effort == "low"  # floored
        assert hard.effort == "medium"  # capped

    def test_decision_is_reproducible(self):
        ctl = ReasoningController()
        a = ctl.decide(task=TaskType.GENERAL, text="explain backprop")
        b = ctl.decide(task=TaskType.GENERAL, text="explain backprop")
        assert isinstance(a, ReasoningDecision)
        assert a.model_dump() == b.model_dump()


# ---------------------------------------------------------------------------
# Verifier adapters
# ---------------------------------------------------------------------------


class TestVerifierAdapters:
    async def test_callable_verifier_float(self):
        v = CallableVerifier(lambda c: 0.7)
        score = await v.averify(SearchCandidate(output="x"))
        assert score.value == 0.7
        assert score.confidence == 1.0

    async def test_callable_verifier_tuple(self):
        v = CallableVerifier(lambda c: (0.6, 0.3))
        score = await v.averify(SearchCandidate(output="x"))
        assert score.value == 0.6
        assert score.confidence == 0.3

    async def test_callable_verifier_async(self):
        async def fn(c):
            return VerifierScore(value=0.9, confidence=0.8)

        score = await CallableVerifier(fn).averify(SearchCandidate(output="x"))
        assert score.value == 0.9

    async def test_reward_verifier_uses_weight_as_confidence(self):
        from vincio.optimize.rewards import RewardSample, RewardSignal, VerifiableReward

        class FixedReward(VerifiableReward):
            name = "fixed"

            async def aevaluate(self, sample: RewardSample) -> RewardSignal:
                return RewardSignal(value=0.8, success=True, weight=0.5, source="fixed")

        v = RewardVerifier(FixedReward())
        score = await v.averify(SearchCandidate(output="ans"))
        assert score.value == 0.8
        assert score.confidence == 0.5
        assert score.success is True

    async def test_judge_ensemble_disagreement_lowers_confidence(self):
        from vincio.evals.datasets import EvalCase
        from vincio.evals.ensemble import JudgeEnsemble
        from vincio.evals.judges import DeterministicJudge
        from vincio.evals.metrics import MetricResult, RunOutput

        def high(case: EvalCase, out: RunOutput) -> MetricResult:
            return MetricResult(name="high", value=0.9)

        def low(case: EvalCase, out: RunOutput) -> MetricResult:
            return MetricResult(name="low", value=0.1)

        ensemble = JudgeEnsemble(
            [DeterministicJudge(high, name="high"), DeterministicJudge(low, name="low")]
        )
        case = EvalCase(id="c", input="q")
        v = JudgeVerifier(ensemble, case=case)
        score = await v.averify(SearchCandidate(output="a", text="a"))
        # The panel splits 0.9 vs 0.1 → wide spread → low confidence.
        assert score.confidence < 0.5


# ---------------------------------------------------------------------------
# TestTimeSearch
# ---------------------------------------------------------------------------


class TestBestOfN:
    async def test_picks_highest_scoring_candidate(self):
        # candidate i has text "ans{i}"; verifier rewards higher index.
        def generate(i: int) -> str:
            return f"ans{i}"

        verifier = CallableVerifier(lambda c: int(c.text.replace("ans", "")) / 10)
        search = TestTimeSearch(generate, verifier=verifier, budget=SearchBudget(max_candidates=4))
        result = await search.best_of_n()
        assert result.best.text == "ans3"
        assert result.n_scored == 4

    async def test_early_exit_when_bar_cleared(self):
        def generate(i: int) -> str:
            return "good"

        verifier = CallableVerifier(lambda c: 0.95)
        search = TestTimeSearch(
            generate,
            verifier=verifier,
            budget=SearchBudget(max_candidates=8, confidence_target=0.9),
        )
        result = await search.best_of_n()
        assert result.early_exit is True
        assert result.n_generated == 1  # cleared the bar on the first draw

    async def test_min_candidates_respected_before_exit(self):
        verifier = CallableVerifier(lambda c: 0.99)
        search = TestTimeSearch(
            lambda i: "x",
            verifier=verifier,
            budget=SearchBudget(max_candidates=8, min_candidates=3, confidence_target=0.9),
        )
        result = await search.best_of_n()
        assert result.n_generated == 3

    async def test_requires_verifier(self):
        search = TestTimeSearch(lambda i: "x")
        with pytest.raises(ValueError):
            await search.best_of_n()


class TestSelfConsistency:
    async def test_majority_vote(self):
        # 3× "yes", 1× "no" → majority "yes".
        answers = ["yes", "no", "yes", "yes"]
        search = TestTimeSearch(
            lambda i: answers[i], budget=SearchBudget(max_candidates=4)
        )
        result = await search.self_consistency()
        assert result.best.answer_text == "yes"
        assert result.votes["yes"] == 3
        assert result.confidence == pytest.approx(0.75)

    async def test_majority_lock_early_exit(self):
        # First 3 all "yes": lead 3-0 with 2 draws left → unbeatable, stop early.
        answers = ["yes", "yes", "yes", "no", "no"]
        search = TestTimeSearch(
            lambda i: answers[i], budget=SearchBudget(max_candidates=5)
        )
        result = await search.self_consistency()
        assert result.early_exit is True
        assert result.n_generated == 3

    async def test_normalization_groups_answers(self):
        answers = ["Yes.", " yes ", "YES", "no"]
        search = TestTimeSearch(
            lambda i: answers[i], budget=SearchBudget(max_candidates=4)
        )
        result = await search.self_consistency()
        assert result.votes["yes"] == 3

    async def test_works_without_verifier(self):
        search = TestTimeSearch(lambda i: "a", budget=SearchBudget(max_candidates=3))
        result = await search.self_consistency()
        assert result.best is not None


class TestBeamSearch:
    async def test_finds_best_leaf(self):
        # A tiny tree: each state is an int; expand adds children +1/+2 up to depth.
        def expand(state):
            if state >= 4:
                return []
            return [state + 1, state + 2]

        # Verifier prefers larger numbers.
        verifier = CallableVerifier(lambda c: min(1.0, c.output / 6))
        search = TestTimeSearch(
            lambda i: None, verifier=verifier, budget=SearchBudget(max_candidates=50)
        )
        result = await search.beam_search(root=0, expand=expand, beam_width=2, max_depth=4)
        assert result.best is not None
        assert result.best.output >= 4
        assert result.strategy == "beam_search"

    async def test_candidate_budget_bounds_scorings(self):
        def expand(state):
            return [state + 1, state + 2, state + 3]

        verifier = CallableVerifier(lambda c: 0.5)
        search = TestTimeSearch(
            lambda i: None, verifier=verifier, budget=SearchBudget(max_candidates=5)
        )
        result = await search.beam_search(root=0, expand=expand, beam_width=3, max_depth=10)
        assert result.n_scored <= 5


# ---------------------------------------------------------------------------
# Runtime / app wiring
# ---------------------------------------------------------------------------


class TestAppWiring:
    async def test_controller_sets_reasoning_on_run(self, offline_config):
        provider = MockProvider(reasoning=True, responder=lambda r: "the answer")
        app = ContextApp(name="ttc", provider=provider, model="mock-1", config=offline_config)
        app.use_reasoning_controller()
        result = await app.arun("Analyze and compare these two complex filings in detail.")
        # The controller filled reasoning_effort (RunConfig set none) → the mock
        # emitted thinking tokens, proving effort reached the request.
        assert result.usage.reasoning_tokens > 0
        # And the paid thinking prefix was recorded for reuse.
        assert app.reasoning_controller.trace_cache.stats()["entries"] == 1

    async def test_no_controller_means_no_reasoning(self, offline_config):
        provider = MockProvider(reasoning=True, responder=lambda r: "the answer")
        app = ContextApp(name="ttc2", provider=provider, model="mock-1", config=offline_config)
        result = await app.arun("hello")
        assert result.usage.reasoning_tokens == 0  # unchanged default behavior

    async def test_explicit_config_overrides_controller(self, offline_config):
        from vincio import RunConfig

        provider = MockProvider(reasoning=True, responder=lambda r: "ok")
        app = ContextApp(name="ttc3", provider=provider, model="mock-1", config=offline_config)
        app.use_reasoning_controller(ReasoningPolicy(max_effort="minimal"))
        result = await app.arun("hard task " * 50, config=RunConfig(reasoning_effort="high"))
        # high effort → 128 thinking tokens in the mock (not the policy's minimal=8).
        assert result.usage.reasoning_tokens == 128

    async def test_app_test_time_search_best_of_n(self, offline_config):
        # Provider echoes the seed so candidates differ; verifier prefers seed 2.
        def responder(request):
            seed = request.seed if request.seed is not None else 0
            return f"candidate-{seed}"

        provider = MockProvider(responder=responder)
        app = ContextApp(name="ttc4", provider=provider, model="mock-1", config=offline_config)
        verifier = CallableVerifier(
            lambda c: 1.0 if c.text.endswith("candidate-2") else 0.2
        )
        result = await app.atest_time_search("pick one", verifier=verifier, n=4)
        assert result.best.text.endswith("candidate-2")
        assert result.strategy == "best_of_n"

    async def test_app_test_time_search_self_consistency(self, offline_config):
        provider = MockProvider(responder=lambda r: "stable answer")
        app = ContextApp(name="ttc5", provider=provider, model="mock-1", config=offline_config)
        result = await app.atest_time_search(
            "q", strategy="self_consistency", n=3
        )
        assert result.best is not None
        assert result.confidence == pytest.approx(1.0)

    def test_app_reasoning_builds_controller(self, offline_config):
        app = ContextApp(name="ttc6", provider=MockProvider(), model="mock-1", config=offline_config)
        ctl = app.reasoning()
        assert isinstance(ctl, ReasoningController)
        assert ctl.trace_cache is not None
