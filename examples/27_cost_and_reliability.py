"""Cost, reliability & scale (1.3).

What real teams hit when an LLM app meets production traffic — outages, rate
limits, runaway spend, and the need to attribute every dollar — handled
*in your application*, not a proxy hop:

  1. Batch execution at ~50% cost (same call sites, switchable sync↔batch).
  2. Circuit breaking + health-aware failover (retry → fallback → circuit-break).
  3. Health-aware key pooling with dual RPM+TPM rate limiting.
  4. Runtime model cascades: start cheap, escalate only on low confidence.
  5. Cost attribution by tenant/feature and enforced budget SLOs.
  6. Provider-aware prompt caching with cache-hit telemetry.
  7. Incremental (content-hash) and sharded indexing at scale.

Runs fully offline on the deterministic mock provider.
"""

from __future__ import annotations

import asyncio

from _shared import example_provider

from vincio import ContextApp
from vincio.core.config import ObservabilityConfig, StorageConfig, VincioConfig
from vincio.core.errors import ProviderUnavailableError
from vincio.core.types import Chunk, ModelResponse, TokenUsage
from vincio.observability.costs import ModelPrice
from vincio.providers import (
    CircuitBreaker,
    HealthAwareFailover,
    KeyPool,
    MockProvider,
)
from vincio.retrieval import LiveIndex, ShardedIndex, VectorIndex
from vincio.retrieval.embeddings import LocalHashEmbedder


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _app(name: str, provider=None, model: str = "gpt-5.2") -> ContextApp:
    p, m = (provider, model) if provider is not None else example_provider()
    app = ContextApp(
        name=name,
        config=VincioConfig(
            storage=StorageConfig(metadata="memory://"),
            observability=ObservabilityConfig(exporter="memory"),
        ),
        provider=p,
        model=m,
    )
    # A priced model so cost/budget/attribution are non-zero offline.
    app.cost_tracker.price_table.set("gpt-5.2", ModelPrice(input_per_mtok=5.0, output_per_mtok=15.0))
    app.cost_tracker.price_table.set("gpt-5.2-mini", ModelPrice(input_per_mtok=0.25, output_per_mtok=2.0))
    return app


def batch_demo() -> None:
    _section("1. Batch execution (~50% cost)")
    app = _app("batch", provider=MockProvider(default_text="summarized"))
    inputs = [f"Summarize document {i}" for i in range(6)]
    results = app.batch(inputs, discount=0.5)
    ok = sum(1 for r in results if r.error is None)
    print(f"  {ok}/{len(results)} succeeded, cost ${sum(r.cost_usd for r in results):.6f} (batch rate)")


def reliability_demo() -> None:
    _section("2. Circuit breaking + health-aware failover")

    class Outage(MockProvider):
        attempts = 0

        async def generate(self, request):
            type(self).attempts += 1
            raise ProviderUnavailableError("region down", provider="primary")

    clock = {"t": 0.0}
    primary = CircuitBreaker(Outage(), failure_threshold=0.5, min_calls=2, cooldown_s=30,
                             clock=lambda: clock["t"])
    secondary = CircuitBreaker(MockProvider(default_text="served by secondary"),
                               clock=lambda: clock["t"])
    chain = HealthAwareFailover([(primary, None), (secondary, None)])

    async def run() -> None:
        from vincio.core.types import Message, ModelRequest

        req = ModelRequest(model="gpt-5.2", messages=[Message(role="user", content="hi")])
        for i in range(4):
            resp = await chain.generate(req)
            if i == 0:
                print(f"  first call served by: {resp.text!r} (steered to healthy entry)")
        print(f"  primary breaker state: {primary.state.value}  "
              f"(dead-primary attempts: {primary.inner.attempts}, then skipped)")

    asyncio.run(run())


def keypool_demo() -> None:
    _section("3. Health-aware key pool with RPM+TPM limits")
    keys = [MockProvider(default_text=f"key{i}") for i in range(3)]
    pool = KeyPool(keys, rpm=600, tpm=200_000, breaker=True, seed=7)

    async def run() -> None:
        from vincio.core.types import Message, ModelRequest

        req = ModelRequest(model="gpt-5.2", messages=[Message(role="user", content="hi")])
        await asyncio.gather(*[pool.generate(req) for _ in range(9)])
        print(f"  dispatched {pool.dispatch_count} calls across {len(keys)} keys: "
              f"{[k.call_count for k in keys]}")

    asyncio.run(run())


def cascade_demo() -> None:
    _section("4. Runtime model cascade (cheap → strong on low confidence)")

    def responder(request):
        # The cheap model returns a truncated (low-confidence) answer; the
        # strong model finishes cleanly, so only hard runs pay for it.
        if "mini" in request.model:
            return ModelResponse(text="partial...", finish_reason="length",
                                 usage=TokenUsage(input_tokens=20, output_tokens=4))
        return ModelResponse(text="A complete, confident answer.", finish_reason="stop",
                             usage=TokenUsage(input_tokens=20, output_tokens=12))

    app = _app("cascade", provider=MockProvider(responder=responder))
    app.use_cascade(["gpt-5.2-mini", "gpt-5.2"])
    result = app.run("Explain the refund policy in detail.")
    print(f"  finished on {result.metadata['cascade']['model']} "
          f"after {result.metadata['cascade']['escalations']} escalation(s)")


def finops_demo() -> None:
    _section("5. Cost attribution + budget SLOs")
    app = _app("finops", provider=MockProvider(default_text="answered"))
    # A per-tenant daily budget that degrades to a cheaper model on breach.
    app.set_cost_budget(scope="tenant", id="acme", limit_usd=0.00005, period="day",
                        on_breach="degrade", degrade_model="gpt-5.2-mini")
    app.run("question one", tenant_id="acme", feature="chat")
    app.run("question two", tenant_id="acme", feature="search")
    app.run("question three", tenant_id="globex", feature="chat")
    print("  cost by tenant:")
    for row in app.cost_report(by="tenant").rows:
        print(f"    {row.key:8s} ${row.cost_usd:.6f}  ({row.calls} calls)")
    print("  cost by feature:")
    for row in app.cost_report(by="feature").rows:
        print(f"    {row.key:8s} ${row.cost_usd:.6f}")


def cache_demo() -> None:
    _section("6. Provider-aware prompt caching + telemetry")

    def responder(request):
        # Emulate a warm cache: most of the stable prefix is served cached.
        return ModelResponse(text="cached answer",
                             usage=TokenUsage(input_tokens=1000, cached_input_tokens=900, output_tokens=20))

    app = _app("cache", provider=MockProvider(responder=responder))
    app.enable_prompt_caching(ttl="1h")
    result = app.run("A question over a large, stable context.")
    span = next(s for s in app.tracer.exporter.traces[-1].spans if s.type == "model_call")
    print(f"  cache hit rate: {span.attributes['cache_hit_rate']:.0%} "
          f"({span.attributes['cached_input_tokens']} cached input tokens)")
    _ = result


def indexing_demo() -> None:
    _section("7. Incremental + sharded indexing at scale")

    async def run() -> None:
        # Content-hash change detection: only changed docs re-embed.
        live = LiveIndex(VectorIndex(LocalHashEmbedder()))
        chunks = [Chunk(id=f"c{i}", document_id="d1", text=f"clause {i}", index=i) for i in range(100)]
        await live.upsert(chunks)
        edited = [c.model_copy(deep=True) for c in chunks]
        edited[7].text = "clause 7 (amended)"
        stats = await live.upsert(edited)
        print(f"  re-index after editing 1 of 100 chunks: "
              f"re-embedded {stats.reembedded}, skipped {stats.unchanged}")

        # Sharded index: a corpus split across backends, queried in parallel.
        sharded = ShardedIndex([VectorIndex(LocalHashEmbedder()) for _ in range(4)])
        corpus = [Chunk(id=f"k{i}", document_id=f"doc{i % 8}", text=f"contract term {i}", index=i)
                  for i in range(200)]
        await sharded.add(corpus)
        hits = await sharded.search("contract term", top_k=5)
        print(f"  sharded across 4 backends: {len(sharded)} chunks, "
              f"top-5 query returned {len(hits)} hits")

    asyncio.run(run())


def main() -> None:
    batch_demo()
    reliability_demo()
    keypool_demo()
    cascade_demo()
    finops_demo()
    cache_demo()
    indexing_demo()


if __name__ == "__main__":
    main()
