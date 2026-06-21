"""Learned semantic cache & near-miss KV reuse.

Exact-match prompt caching serves a *byte-identical* request for free. This
example shows the rung above it: answering a request that is *semantically
equivalent* (not byte-identical) to a recent one straight from cache, with the
acceptance threshold **learned from traces** so a near-miss is served only when
it is safe — never below the bar.

Five steps, all offline and deterministic (driven by the mock provider and the
dependency-free local embedder):

  1. Calibrate: fit the acceptance threshold from labelled trace pairs so an
     accepted near-miss clears a precision target — the cache learns its own bar
     instead of trusting a hand-picked constant.
  2. Serve a near-miss: a semantically-equivalent request is answered from cache
     for free, even though the live provider would now answer differently.
  3. Refuse a below-bar match: an unrelated request is never served from cache —
     the calibrated bar holds.
  4. Gate against drift: the eval-replay regression gate (the cache analogue of
     the model-swap gate) passes a faithful cache and blocks a drifted one.
  5. KV-prefix reuse: a family of requests that share a stable prompt head reuse
     the head's KV footprint instead of recomputing it, and the saving is
     reported alongside the resident-memory budget.

Everything here is opt-in and additive; nothing below is required to run Vincio.
"""

from __future__ import annotations

import asyncio

from vincio import (
    ContextApp,
    LearnedSemanticCache,
    SemanticCacheGate,
    SemanticCachePolicy,
    SemanticGateCase,
    VincioConfig,
)
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import LocalHashEmbedder

SCOPE = "support:refunds"
REFUND_A = "what is the refund policy for orders"
REFUND_B = "what is the refund policy for returns"
UNRELATED = "how do I reset my account password"


def _config() -> VincioConfig:
    config = VincioConfig()
    config.observability.exporter = "memory"
    return config


async def calibrate_from_traces() -> LearnedSemanticCache:
    print("1. Calibrate — learn the acceptance threshold from labelled trace pairs")
    cache = LearnedSemanticCache(
        LocalHashEmbedder(),
        policy=SemanticCachePolicy(target_precision=0.95, min_floor=0.5, ttl_s=None),
    )
    report = await cache.calibrate_from_pairs(
        [
            (REFUND_A, REFUND_B, True),  # two phrasings of the same question
            (REFUND_A, UNRELATED, False),  # a genuinely different question
        ]
    )
    print(
        f"   fitted threshold={report.threshold:.3f}  precision={report.achieved_precision:.2f}  "
        f"calibrated={report.calibrated}"
    )
    return cache


async def serve_a_near_miss(cache: LearnedSemanticCache) -> None:
    print("\n2. Serve a near-miss — a semantically-equivalent request, answered for free")
    await cache.store(
        REFUND_A,
        {"text": "Refunds are accepted within 30 days."},
        policy_scope=SCOPE,
        response_tokens=8,
    )
    hit = await cache.lookup(REFUND_B, policy_scope=SCOPE)
    assert hit is not None
    print(f"   query:   {REFUND_B!r}")
    print(
        f"   matched: {hit.matched_query!r}  (similarity={hit.similarity:.3f} ≥ {hit.threshold:.3f})"
    )
    print(f"   served:  {hit.value['text']!r}  — no model call")


async def refuse_below_bar(cache: LearnedSemanticCache) -> None:
    print("\n3. Refuse a below-bar match — the calibrated bar holds")
    miss = await cache.lookup(UNRELATED, policy_scope=SCOPE)
    stats = cache.stats()
    print(f"   query: {UNRELATED!r} -> served={miss is not None}")
    print(f"   near-misses rejected below the bar: {stats.near_misses_rejected}")


async def gate_against_drift(cache: LearnedSemanticCache) -> None:
    print("\n4. Gate against drift — the eval-replay regression check (like a model swap)")
    gate = SemanticCacheGate(quality_floor=0.9)
    cases = [
        SemanticGateCase(
            query=REFUND_B,
            reference_answer="Refunds are accepted within 30 days.",
            policy_scope=SCOPE,
        )
    ]
    good = await gate.evaluate(cache, cases)
    print(f"   faithful cache: passed={good.passed}  ({good.reason})")

    drifted = LearnedSemanticCache(
        LocalHashEmbedder(), policy=SemanticCachePolicy(threshold=cache.threshold, ttl_s=None)
    )
    await drifted.store(REFUND_A, "completely unrelated nonsense", policy_scope=SCOPE)
    bad = await gate.evaluate(drifted, cases)
    print(f"   drifted cache:  passed={bad.passed}  ({bad.reason})")


def kv_prefix_reuse() -> None:
    print("\n5. Through the run path — near-miss served free, and KV reused across the family")
    app = ContextApp(
        name="support",
        provider=MockProvider(default_text="Refunds within 30 days."),
        config=_config(),
    )
    app.use_semantic_cache(SemanticCachePolicy(enabled=True, threshold=0.6, ttl_s=None))
    app.use_kv_prefix_reuse()
    app.run(REFUND_A)  # cold: a real call computes the stable head's KV, stores the answer
    served = app.run(REFUND_B)  # near-miss: served from cache, no provider call, billed $0
    # A distinct (non-near-miss) question still hits the provider — but, sharing the
    # same stable prompt head, it reuses the head's KV instead of recomputing it.
    app.run("what are the international shipping options for large parcels")
    sc = app.semantic_cache_report()
    kv = app.kv_prefix_report()
    print(
        f"   near-miss served through the run path: {served.raw_text!r}  (billed ${served.cost_usd:.4f})"
    )
    print(
        f"   semantic cache: served={sc.served}  tokens_saved={sc.tokens_saved}  resident_bytes={sc.resident_bytes}"
    )
    print(
        f"   kv-prefix pool: families={kv.families}  reuses={kv.reuses}  "
        f"kv_bytes_reused={kv.kv_bytes_reused}"
    )


async def main() -> None:
    cache = await calibrate_from_traces()
    await serve_a_near_miss(cache)
    await refuse_below_bar(cache)
    await gate_against_drift(cache)
    kv_prefix_reuse()
    print("\nA recent answer, reused for an equivalent request — safely, only above a learned bar.")


if __name__ == "__main__":
    asyncio.run(main())
