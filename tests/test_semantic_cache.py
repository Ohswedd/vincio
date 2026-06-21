"""Learned semantic cache & near-miss KV reuse (caching/semantic, caching/kvreuse).

Covers threshold calibration from traces, calibrated near-miss serving with
auditability and reversibility, the eval-replay safety gate, cross-request
KV-prefix reuse accounting, and the opt-in runtime wiring. All deterministic and
offline (LocalHashEmbedder + MockProvider).
"""

from __future__ import annotations

import pytest

from vincio import (
    CalibrationExample,
    ContextApp,
    KVPrefixPool,
    LearnedSemanticCache,
    SemanticCacheGate,
    SemanticCachePolicy,
    SemanticGateCase,
    ThresholdCalibrator,
)
from vincio.caching import KVReuseReport, SemanticCacheStats, kv_prefix_key, lexical_quality
from vincio.core.config import VincioConfig
from vincio.providers import MockProvider
from vincio.retrieval.embeddings import LocalHashEmbedder

REFUND_A = "what is the refund policy for orders"
REFUND_B = "what is the refund policy for returns"
REFUND_C = "what is the refund policy for orders placed today"
UNRELATED = "how do I reset my account password"


def _embedder() -> LocalHashEmbedder:
    return LocalHashEmbedder()


def _config() -> VincioConfig:
    c = VincioConfig()
    c.observability.exporter = "memory"
    return c


# =====================================================================
# Threshold calibration — the "learned" in learned semantic cache
# =====================================================================


def _examples(rows: list[tuple[float, bool]]) -> list[CalibrationExample]:
    return [CalibrationExample(similarity=s, equivalent=e) for s, e in rows]


def test_calibrator_picks_lowest_threshold_meeting_precision_target():
    # Positives at 0.99/0.95/0.92; negatives at 0.91/0.85. The lowest threshold
    # that admits no negative is 0.92 — max recall at full precision.
    cal = ThresholdCalibrator(target_precision=0.95, min_floor=0.80)
    report = cal.calibrate(
        _examples([(0.99, True), (0.95, True), (0.92, True), (0.91, False), (0.85, False)])
    )
    assert report.calibrated is True
    assert report.threshold == 0.92
    assert report.achieved_precision == 1.0
    assert report.recall == 1.0


def test_calibrator_admits_a_negative_only_if_precision_target_allows():
    # Target precision 0.66 tolerates one negative among three accepted. Lower
    # thresholds (0.50, 0.60) admit a second negative and drop precision to 0.5,
    # so the calibrator climbs to 0.85 — the lowest bar that still clears 0.66.
    cal = ThresholdCalibrator(target_precision=0.66, min_floor=0.50)
    report = cal.calibrate(_examples([(0.95, True), (0.90, True), (0.85, False), (0.60, False)]))
    assert report.calibrated is True
    # 0.85 admits {0.95,0.90,0.85} → precision 2/3 ≈ 0.667 ≥ 0.66.
    assert report.threshold == 0.85
    assert 0.66 <= report.achieved_precision <= 0.67


def test_calibrator_refuses_to_guess_when_target_unreachable():
    # Every threshold that admits a positive also admits a negative of equal or
    # higher similarity: the target is unreachable, so fall back to 1.0 (off).
    cal = ThresholdCalibrator(target_precision=0.95, min_floor=0.80)
    report = cal.calibrate(_examples([(0.90, True), (0.95, False), (0.92, False)]))
    assert report.calibrated is False
    assert report.threshold == 1.0


def test_calibrator_never_goes_below_min_floor():
    cal = ThresholdCalibrator(target_precision=0.95, min_floor=0.90)
    # A clean split at 0.70 would suffice, but the floor forbids it.
    report = cal.calibrate(_examples([(0.95, True), (0.70, True), (0.50, False)]))
    assert report.threshold >= 0.90
    assert report.floored is True


async def test_calibrate_from_pairs_embeds_and_fits():
    cache = LearnedSemanticCache(
        _embedder(), policy=SemanticCachePolicy(target_precision=0.95, min_floor=0.5)
    )
    report = await cache.calibrate_from_pairs(
        [
            (REFUND_A, REFUND_B, True),
            (REFUND_A, REFUND_C, True),
            (REFUND_A, UNRELATED, False),
        ]
    )
    assert report.calibrated is True
    # Threshold sits above the unrelated pair (~0.19) and at/below the near-miss
    # pairs (~0.73-0.87), so near-misses are served and the unrelated one is not.
    assert 0.19 < report.threshold <= 0.73
    assert cache.threshold == report.threshold


# =====================================================================
# Learned semantic cache — serving, isolation, freshness, budget
# =====================================================================


async def test_serves_near_miss_above_threshold_and_records_audit():
    cache = LearnedSemanticCache(_embedder(), policy=SemanticCachePolicy(threshold=0.6, ttl_s=None))
    await cache.store(
        REFUND_A,
        {"text": "Refunds within 30 days."},
        policy_scope="s",
        schema_ref=None,
        response_tokens=5,
    )
    hit = await cache.lookup(REFUND_B, policy_scope="s", schema_ref=None)
    assert hit is not None and hit.accepted is True
    assert hit.value == {"text": "Refunds within 30 days."}
    assert hit.matched_query == REFUND_A
    assert hit.similarity >= 0.6
    audit = cache.audit()
    assert len(audit) == 1 and audit[0].query == REFUND_B
    stats = cache.stats()
    assert isinstance(stats, SemanticCacheStats)
    assert stats.served == 1 and stats.tokens_saved == 5


async def test_rejects_near_miss_below_threshold_never_serves():
    cache = LearnedSemanticCache(_embedder(), policy=SemanticCachePolicy(threshold=0.6, ttl_s=None))
    await cache.store(REFUND_A, "answer", policy_scope="s", schema_ref=None)
    hit = await cache.lookup(UNRELATED, policy_scope="s", schema_ref=None)
    assert hit is None
    stats = cache.stats()
    assert stats.near_misses_rejected == 1
    assert stats.served == 0
    assert cache.audit() == []  # rejections are not served, so not in the audit log


async def test_scope_and_schema_isolation():
    cache = LearnedSemanticCache(_embedder(), policy=SemanticCachePolicy(threshold=0.5, ttl_s=None))
    await cache.store(REFUND_A, "scoped", policy_scope="tenant-a", schema_ref="invoice")
    # Same query, different scope → no hit.
    assert await cache.lookup(REFUND_A, policy_scope="tenant-b", schema_ref="invoice") is None
    # Same query, different schema → no hit.
    assert await cache.lookup(REFUND_A, policy_scope="tenant-a", schema_ref="other") is None
    # Matching scope + schema → hit.
    assert await cache.lookup(REFUND_A, policy_scope="tenant-a", schema_ref="invoice") is not None


async def test_ttl_expiry_with_injected_clock():
    now = {"t": 1000.0}
    cache = LearnedSemanticCache(
        _embedder(), policy=SemanticCachePolicy(threshold=0.5, ttl_s=60.0), clock=lambda: now["t"]
    )
    await cache.store(REFUND_A, "answer", policy_scope="s", schema_ref=None)
    now["t"] = 1059.0  # within TTL
    assert await cache.lookup(REFUND_A, policy_scope="s", schema_ref=None) is not None
    now["t"] = 1101.0  # past TTL
    assert await cache.lookup(REFUND_A, policy_scope="s", schema_ref=None) is None


async def test_lru_eviction_under_entry_ceiling():
    cache = LearnedSemanticCache(
        _embedder(), policy=SemanticCachePolicy(threshold=0.5, ttl_s=None, max_entries=2)
    )
    await cache.store("query alpha one", "A", policy_scope="s", schema_ref=None)
    await cache.store("query beta two", "B", policy_scope="s", schema_ref=None)
    await cache.store("query gamma three", "C", policy_scope="s", schema_ref=None)
    assert len(cache) == 2  # alpha evicted (oldest)


async def test_resident_byte_ceiling_evicts_and_keeps_one():
    cache = LearnedSemanticCache(
        _embedder(), policy=SemanticCachePolicy(threshold=0.5, ttl_s=None, max_resident_bytes=1)
    )
    await cache.store("query alpha one", "A" * 100, policy_scope="s", schema_ref=None)
    await cache.store("query beta two", "B" * 100, policy_scope="s", schema_ref=None)
    assert len(cache) == 1  # ceiling of 1 byte keeps at least the newest entry
    assert cache.resident_bytes > 0


async def test_revoke_makes_a_bad_entry_unservable():
    cache = LearnedSemanticCache(_embedder(), policy=SemanticCachePolicy(threshold=0.5, ttl_s=None))
    entry = await cache.store(REFUND_A, "answer", policy_scope="s", schema_ref=None)
    assert await cache.lookup(REFUND_B, policy_scope="s", schema_ref=None) is not None
    assert cache.revoke(entry.key) is True
    assert await cache.lookup(REFUND_B, policy_scope="s", schema_ref=None) is None
    assert cache.revoke(entry.key) is False  # already gone


async def test_clear_returns_count_and_resets_footprint():
    cache = LearnedSemanticCache(_embedder(), policy=SemanticCachePolicy(threshold=0.5, ttl_s=None))
    await cache.store(REFUND_A, "a", policy_scope="s", schema_ref=None)
    await cache.store(UNRELATED, "b", policy_scope="s", schema_ref=None)
    assert cache.clear() == 2
    assert len(cache) == 0 and cache.resident_bytes == 0


async def test_restore_same_query_replaces_without_double_counting():
    cache = LearnedSemanticCache(_embedder(), policy=SemanticCachePolicy(threshold=0.5, ttl_s=None))
    await cache.store(REFUND_A, "first", policy_scope="s", schema_ref=None)
    bytes_after_one = cache.resident_bytes
    await cache.store(REFUND_A, "third", policy_scope="s", schema_ref=None)  # same length
    assert len(cache) == 1
    assert cache.resident_bytes == bytes_after_one  # same query+scope key, swapped
    hit = await cache.lookup(REFUND_A, policy_scope="s", schema_ref=None)
    assert hit.value == "third"


# =====================================================================
# Safety gate — eval-replay no-regression check
# =====================================================================


async def test_lexical_quality_bounds():
    assert lexical_quality("a b c", "a b c") == 1.0
    assert lexical_quality("a b c", "x y z") == 0.0
    assert 0.0 < lexical_quality("a b c d", "a b") < 1.0


async def test_gate_passes_when_served_near_misses_match_reference():
    cache = LearnedSemanticCache(_embedder(), policy=SemanticCachePolicy(threshold=0.6, ttl_s=None))
    await cache.store(REFUND_A, "refunds within thirty days", policy_scope="s", schema_ref=None)
    gate = SemanticCacheGate(quality_floor=0.5)
    report = await gate.evaluate(
        cache,
        [
            SemanticGateCase(
                query=REFUND_B, reference_answer="refunds within thirty days", policy_scope="s"
            )
        ],
    )
    assert report.passed is True
    assert report.served == 1 and report.regressions == []


async def test_gate_fails_when_a_served_near_miss_regresses():
    cache = LearnedSemanticCache(_embedder(), policy=SemanticCachePolicy(threshold=0.6, ttl_s=None))
    # A drifted entry: it serves an answer unrelated to the live one for this query.
    await cache.store(REFUND_A, "completely unrelated nonsense", policy_scope="s", schema_ref=None)
    gate = SemanticCacheGate(quality_floor=0.5)
    report = await gate.evaluate(
        cache,
        [
            SemanticGateCase(
                query=REFUND_B, reference_answer="refunds within thirty days", policy_scope="s"
            )
        ],
    )
    assert report.passed is False
    assert REFUND_B in report.regressions


async def test_gate_ignores_misses_a_miss_costs_a_live_call():
    cache = LearnedSemanticCache(
        _embedder(), policy=SemanticCachePolicy(threshold=0.99, ttl_s=None)
    )
    await cache.store(REFUND_A, "x", policy_scope="s", schema_ref=None)
    gate = SemanticCacheGate(quality_floor=0.9)
    report = await gate.evaluate(
        cache, [SemanticGateCase(query=UNRELATED, reference_answer="y", policy_scope="s")]
    )
    assert report.passed is True and report.served == 0


# =====================================================================
# KV-prefix reuse accounting
# =====================================================================


def test_kv_prefix_first_sight_is_a_miss_then_reuse():
    pool = KVPrefixPool(kv_bytes_per_token=2048)
    first = pool.observe(prefix_hash="head-1", model="m", prefix_tokens=100)
    assert first.reused is False and first.kv_bytes_reused == 0 and first.family_size == 1
    second = pool.observe(prefix_hash="head-1", model="m", prefix_tokens=100)
    assert second.reused is True
    assert second.kv_bytes_reused == 100 * 2048
    assert second.family_size == 2


def test_kv_prefix_distinct_by_model():
    pool = KVPrefixPool()
    pool.observe(prefix_hash="head-1", model="m1", prefix_tokens=10)
    obs = pool.observe(prefix_hash="head-1", model="m2", prefix_tokens=10)
    assert obs.reused is False  # KV is not portable across models
    assert kv_prefix_key("head-1", "m1") != kv_prefix_key("head-1", "m2")


def test_kv_prefix_tracks_largest_prefix():
    pool = KVPrefixPool(kv_bytes_per_token=10)
    pool.observe(prefix_hash="h", model="m", prefix_tokens=5)
    obs = pool.observe(prefix_hash="h", model="m", prefix_tokens=20)
    assert obs.prefix_tokens == 20 and obs.kv_bytes_reused == 20 * 10


def test_kv_prefix_report_and_reuse_rate():
    pool = KVPrefixPool(kv_bytes_per_token=1)
    pool.observe(prefix_hash="h", model="m", prefix_tokens=10)
    pool.observe(prefix_hash="h", model="m", prefix_tokens=10)
    pool.observe(prefix_hash="h", model="m", prefix_tokens=10)
    report = pool.report()
    assert isinstance(report, KVReuseReport)
    assert report.observations == 3 and report.reuses == 2
    assert report.kv_bytes_reused == 20 and report.reuse_rate == round(2 / 3, 4)


def test_kv_prefix_eviction_under_entry_ceiling():
    pool = KVPrefixPool(max_entries=2)
    pool.observe(prefix_hash="a", model="m", prefix_tokens=1)
    pool.observe(prefix_hash="b", model="m", prefix_tokens=1)
    pool.observe(prefix_hash="c", model="m", prefix_tokens=1)
    assert len(pool) == 2


def test_kv_prefix_clear():
    pool = KVPrefixPool()
    pool.observe(prefix_hash="a", model="m", prefix_tokens=1)
    assert pool.clear() == 1 and len(pool) == 0 and pool.resident_bytes == 0


def test_kv_prefix_pool_validates_args():
    with pytest.raises(ValueError):
        KVPrefixPool(max_entries=0)
    with pytest.raises(ValueError):
        KVPrefixPool(kv_bytes_per_token=-1)


# =====================================================================
# App integration — opt-in wiring through the run path
# =====================================================================


def _app(text: str) -> ContextApp:
    return ContextApp(
        name="sc", provider=MockProvider(default_text=text), model="mock-1", config=_config()
    )


def test_use_semantic_cache_serves_near_miss_for_free():
    app = _app("LIVE ANSWER")
    app.use_semantic_cache(SemanticCachePolicy(enabled=True, threshold=0.6, ttl_s=None))
    app.run(REFUND_A)
    # A live call now would say DIFFERENT; a near-miss must still serve the cached text.
    app._provider_instance = MockProvider(default_text="DIFFERENT")
    result = app.run(REFUND_B)
    assert result.raw_text == "LIVE ANSWER"
    stats = app.semantic_cache_report()
    assert stats.served == 1
    # The served call billed nothing: it shows up as a $0 model call.
    assert result.cost_usd == 0.0


def test_semantic_cache_off_by_default():
    app = _app("X")
    assert app.semantic_cache is None
    assert app.semantic_cache_report() is None


def test_config_enables_semantic_cache():
    cfg = _config()
    cfg.cache.semantic_cache = True
    cfg.cache.semantic_threshold = 0.6
    app = ContextApp(name="sc", provider=MockProvider(default_text="A"), model="mock-1", config=cfg)
    assert isinstance(app.semantic_cache, LearnedSemanticCache)


def test_policy_change_clears_semantic_cache_via_invalidation():
    app = _app("A")
    app.use_semantic_cache(SemanticCachePolicy(enabled=True, threshold=0.6, ttl_s=None))
    app.run(REFUND_A)
    assert len(app.semantic_cache) == 1
    app.cache_invalidation.policy_changed()
    assert len(app.semantic_cache) == 0


def test_use_kv_prefix_reuse_tracks_shared_head_across_runs():
    app = _app("ANSWER")
    app.use_kv_prefix_reuse()
    app.run(REFUND_A)
    app.run(REFUND_B)  # same stable prompt head → a reuse
    report = app.kv_prefix_report()
    assert report.observations == 2
    assert report.reuses == 1
    assert report.kv_bytes_reused > 0


def test_kv_prefix_off_by_default():
    app = _app("X")
    assert app.kv_prefix_pool is None
    assert app.kv_prefix_report() is None


def test_semantic_serving_is_deterministic_across_apps():
    def served_text() -> str:
        app = _app("LIVE")
        app.use_semantic_cache(SemanticCachePolicy(enabled=True, threshold=0.6, ttl_s=None))
        app.run(REFUND_A)
        app._provider_instance = MockProvider(default_text="OTHER")
        return app.run(REFUND_C).raw_text

    assert served_text() == served_text() == "LIVE"
