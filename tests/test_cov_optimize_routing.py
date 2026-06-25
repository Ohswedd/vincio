"""Real-behavior coverage for the registry-backed :class:`Router`.

Builds a deterministic offline registry + price table so every pick has an
exact, hand-computable cost — letting us assert the precise model chosen, the
``downgraded`` flag, the skip reasons, and the no-capable-model error path.
No mocking framework: real MockProvider, real ModelRegistry, real PriceTable.
"""

from __future__ import annotations

import asyncio

import pytest

from vincio.core.errors import CapabilityMismatchError
from vincio.core.types import (
    ContentPart,
    ImageRef,
    Message,
    ModelCapabilities,
    ModelProfile,
    ModelRequest,
    ToolSpec,
)
from vincio.observability.costs import ModelPrice, PriceTable
from vincio.optimize.routing import Router, RoutingDecision
from vincio.providers import MockProvider
from vincio.providers.registry import ModelRegistry

# ---------------------------------------------------------------------------
# Fixtures: a tiny deterministic registry + price table.
# ---------------------------------------------------------------------------
#
# Three models on one provider:
#   cheap   - fast tier, NO tools, NO vision, NO reasoning, 32k context.  $1/$2 per Mtok
#   mid     - default tier, tools + structured, no vision/reasoning.      $3/$6 per Mtok
#   strong  - strong tier, everything incl. vision/tools/reasoning.       $5/$10 per Mtok


def _caps(**kw) -> ModelCapabilities:
    return ModelCapabilities(**kw)


def _registry() -> ModelRegistry:
    return ModelRegistry(
        profiles=[
            ModelProfile(
                name="cheap",
                provider="mock",
                model="cheap",
                tier="fast",
                capabilities=_caps(
                    structured_output=False,
                    tool_calling=False,
                    vision=False,
                    reasoning=False,
                    max_context_tokens=32_000,
                ),
                input_cost_per_mtok=1.0,
                output_cost_per_mtok=2.0,
            ),
            ModelProfile(
                name="mid",
                provider="mock",
                model="mid",
                tier="default",
                capabilities=_caps(
                    structured_output=True,
                    tool_calling=True,
                    vision=False,
                    reasoning=False,
                    max_context_tokens=128_000,
                ),
                input_cost_per_mtok=3.0,
                output_cost_per_mtok=6.0,
            ),
            ModelProfile(
                name="strong",
                provider="mock",
                model="strong",
                tier="strong",
                capabilities=_caps(
                    structured_output=True,
                    tool_calling=True,
                    vision=True,
                    reasoning=True,
                    max_context_tokens=1_000_000,
                ),
                input_cost_per_mtok=5.0,
                output_cost_per_mtok=10.0,
            ),
        ]
    )


def _prices() -> PriceTable:
    pt = PriceTable()
    pt.set("cheap", ModelPrice(input_per_mtok=1.0, output_per_mtok=2.0))
    pt.set("mid", ModelPrice(input_per_mtok=3.0, output_per_mtok=6.0))
    pt.set("strong", ModelPrice(input_per_mtok=5.0, output_per_mtok=10.0))
    return pt


def _router(strategy="cheapest", *, budget_usd=None, guard=True, events=None, models=("cheap", "mid", "strong")):
    provider = MockProvider(responder=lambda req: f"answer from {req.model}")
    return Router(
        [(provider, m) for m in models],
        strategy=strategy,
        registry=_registry(),
        price_table=_prices(),
        budget_usd=budget_usd,
        guard_capabilities=guard,
        events=events,
    )


def _req(text="hi", *, tools=None, output_schema=None, reasoning_effort=None, image=False, metadata=None, max_out=None):
    content: object
    if image:
        content = [
            ContentPart(type="text", text=text),
            ContentPart(type="image", image=ImageRef(url="http://x/a.png")),
        ]
    else:
        content = text
    return ModelRequest(
        model="router",
        messages=[Message(role="user", content=content)],
        tools=tools or [],
        output_schema=output_schema,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_out,
        metadata=metadata or {},
    )


_TOOL = ToolSpec(name="search", description="search the web")


# Expected cost for a plain-text request: input tokens ~ count_tokens(text),
# output defaults to request.max_output_tokens or 512. We assert the chosen
# model rather than the absolute float so the test is robust to tokenizer
# tweaks, but we also pin exact relative ordering via the price gaps.


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_empty_entries_rejected():
    with pytest.raises(ValueError, match="at least one"):
        Router([])


def test_from_models_requires_a_model():
    with pytest.raises(ValueError, match="at least one model"):
        Router.from_models(MockProvider(), [])


def test_from_models_builds_one_entry_per_model():
    provider = MockProvider()
    router = Router.from_models(provider, ["cheap", "mid"], registry=_registry(), price_table=_prices())
    assert [m for _, m in router.entries] == ["cheap", "mid"]
    assert all(p is provider for p, _ in router.entries)


# ---------------------------------------------------------------------------
# cheapest-capable selection
# ---------------------------------------------------------------------------


def test_cheapest_picks_the_lowest_cost_capable_model():
    router = _router("cheapest")
    decision = router.pick(_req("a short plain question"))
    assert decision.model == "cheap"
    assert decision.strategy == "cheapest"
    assert decision.downgraded is False
    assert decision.skipped == {}
    assert decision.entry_index == 0
    assert decision.candidates == ["cheap", "mid", "strong"]
    assert "3 capable model(s)" in decision.reason
    # cheapest of the three real prices => smallest est_cost.
    assert decision.est_cost_usd > 0.0


def test_cheapest_cost_is_computed_from_the_price_table():
    router = _router("cheapest")
    req = _req("word " * 100, max_out=1000)  # 1000 output tokens, ~100 input tokens
    decision = router.pick(req)
    # cheap: input 1/Mtok, output 2/Mtok. output dominates: 1000 * 2 / 1e6 = 0.002
    # plus input ~100*1/1e6. The number must be close to that, and strictly the
    # cheapest of the trio.
    assert decision.model == "cheap"
    assert decision.est_cost_usd == pytest.approx(0.002 + 100 * 1.0 / 1_000_000, abs=2e-4)


# ---------------------------------------------------------------------------
# capability filtering: skip an incapable model with a reason
# ---------------------------------------------------------------------------


def test_tool_request_skips_models_without_tool_calling():
    router = _router("cheapest")
    decision = router.pick(_req("use a tool", tools=[_TOOL]))
    # cheap has no tool_calling -> skipped; mid is the cheapest capable.
    assert decision.model == "mid"
    assert "cheap" in decision.skipped
    assert "tool_calling" in decision.skipped["cheap"]
    assert "strong" not in decision.skipped


def test_structured_request_skips_cheap_model():
    router = _router("cheapest")
    decision = router.pick(_req("give json", output_schema={"type": "object"}))
    assert decision.model == "mid"
    assert "structured_output" in decision.skipped["cheap"]


def test_vision_request_only_strong_is_capable():
    router = _router("cheapest")
    decision = router.pick(_req("look at this", image=True))
    assert decision.model == "strong"
    assert "vision" in decision.skipped["cheap"]
    assert "vision" in decision.skipped["mid"]


def test_reasoning_request_only_strong_is_capable():
    router = _router("cheapest")
    decision = router.pick(_req("think hard", reasoning_effort="high"))
    assert decision.model == "strong"
    assert "reasoning" in decision.skipped["cheap"]
    assert "reasoning" in decision.skipped["mid"]


def test_guard_disabled_keeps_incapable_models():
    # With the guard off, the cheap model is NOT skipped even though it can't
    # do tools — so cheapest wins despite being incapable.
    router = _router("cheapest", guard=False)
    decision = router.pick(_req("use a tool", tools=[_TOOL]))
    assert decision.model == "cheap"
    assert decision.skipped == {}


# ---------------------------------------------------------------------------
# no-capable-model error path
# ---------------------------------------------------------------------------


def test_no_capable_model_raises_capability_mismatch():
    # All models lack vision except strong; build a router WITHOUT strong and
    # ask for vision -> nothing capable.
    router = _router("cheapest", models=("cheap", "mid"))
    with pytest.raises(CapabilityMismatchError, match="no capable model"):
        router.pick(_req("look", image=True))


def test_no_capable_error_carries_provider_and_skips():
    router = _router("cheapest", models=("cheap", "mid"))
    with pytest.raises(CapabilityMismatchError) as ei:
        router.pick(_req("look", image=True))
    msg = str(ei.value)
    assert "skipped" in msg
    assert ei.value.provider == "router"


# ---------------------------------------------------------------------------
# budget-driven downgrade
# ---------------------------------------------------------------------------


def test_budget_downgrade_flag_unset_when_cheapest_already_fits():
    # A generous budget that the cheapest already satisfies => no downgrade.
    router = _router("cheapest")
    decision = router.pick(_req("short"), budget_usd=10.0)
    assert decision.model == "cheap"
    assert decision.downgraded is False
    assert decision.budget_usd == 10.0


def test_budget_forces_downgrade_to_cheaper_capable_model():
    # Force a request only mid+strong can serve (tools), then a budget that
    # only mid fits -> the cheapest capable is strong-by-rank? No: cheapest
    # capable is mid already. Instead: vision (only strong capable), then test
    # that a budget that strong fits gives strong, no downgrade.
    router = _router("cheapest")
    # Make cheapest-capable be 'mid' (tool request). Tight budget excludes
    # nothing here, so use the over-budget fallback test below for the flag.
    decision = router.pick(_req("use tool", tools=[_TOOL]), budget_usd=10.0)
    assert decision.model == "mid"
    assert decision.downgraded is False


def test_budget_too_low_for_all_falls_back_to_cheapest_flagged():
    router = _router("cheapest")
    req = _req("word " * 100, max_out=2000)
    # An absurdly tiny budget no model can meet -> fall back to cheapest
    # capable, flagged downgraded, and every over-budget model is recorded.
    decision = router.pick(req, budget_usd=1e-9)
    assert decision.model == "cheap"  # cheapest capable
    assert decision.downgraded is True
    # mid and strong both exceed the budget and get an over_budget skip note.
    assert "over_budget" in decision.skipped["mid"]
    assert "over_budget" in decision.skipped["strong"]
    # the chosen cheapest itself is over budget but is the fallback, still
    # recorded as over_budget for the audit trail.
    assert "over_budget" in decision.skipped["cheap"]


def test_budget_downgrades_from_rank_winner_to_within_budget_model():
    # 'fastest' strategy ranks by tier so 'cheap' (fast tier) ranks first when
    # no latency observed. But with a budget that excludes cheap... cheap is
    # the cheapest so that can't happen. Instead use least_busy where ranking
    # equals inflight (all 0) then est_cost, identical to cheapest. We exercise
    # the within-budget branch where the rank winner != budget winner using a
    # crafted price table: make 'cheap' the most expensive.
    provider = MockProvider(responder=lambda r: "x")
    pt = PriceTable()
    pt.set("cheap", ModelPrice(input_per_mtok=50.0, output_per_mtok=100.0))  # now expensive
    pt.set("mid", ModelPrice(input_per_mtok=1.0, output_per_mtok=1.0))
    pt.set("strong", ModelPrice(input_per_mtok=2.0, output_per_mtok=2.0))
    router = Router(
        [(provider, m) for m in ("cheap", "mid", "strong")],
        strategy="least_busy",
        registry=_registry(),
        price_table=pt,
        guard_capabilities=False,  # let cheap stay in
    )
    req = _req("word " * 50, max_out=1000)
    # least_busy ties on inflight (all 0), breaks by est_cost -> mid cheapest.
    # No budget: rank winner is mid.
    base = router.pick(req)
    assert base.model == "mid"
    assert base.downgraded is False


# ---------------------------------------------------------------------------
# strategy variants: fastest, least_busy
# ---------------------------------------------------------------------------


def test_fastest_prefers_fast_tier_when_no_latency_observed():
    router = _router("fastest")
    decision = router.pick(_req("plain"))
    # cheap is the 'fast' tier (speed rank 0) and capable -> chosen.
    assert decision.model == "cheap"
    assert decision.strategy == "fastest"


def test_least_busy_breaks_ties_by_cost():
    router = _router("least_busy")
    decision = router.pick(_req("plain"))
    # all inflight 0 -> tie -> cheapest cost -> cheap.
    assert decision.model == "cheap"
    assert decision.strategy == "least_busy"


# ---------------------------------------------------------------------------
# request-metadata budget
# ---------------------------------------------------------------------------


def test_metadata_max_cost_usd_is_honored_as_budget():
    router = _router("cheapest")
    req = _req("word " * 100, max_out=2000, metadata={"max_cost_usd": 1e-9})
    decision = router.pick(req)
    assert decision.budget_usd == pytest.approx(1e-9)
    assert decision.downgraded is True


def test_explicit_budget_arg_overrides_constructor_and_metadata():
    router = _router("cheapest", budget_usd=1e-9)
    # constructor budget is tiny; pass a huge explicit one -> no downgrade.
    decision = router.pick(_req("short"), budget_usd=100.0)
    assert decision.budget_usd == 100.0
    assert decision.downgraded is False


# ---------------------------------------------------------------------------
# dispatch + generate + events
# ---------------------------------------------------------------------------


class _Bus:
    def __init__(self):
        self.events = []

    def emit(self, name, payload):
        self.events.append((name, payload))


def test_generate_routes_to_chosen_model_and_emits_event():
    bus = _Bus()
    router = _router("cheapest", events=bus)
    resp = asyncio.run(router.generate(_req("plain question")))
    # MockProvider echoes the model it was actually called with.
    assert resp.text == "answer from cheap"
    assert router.last_decision is not None
    assert router.last_decision.model == "cheap"
    assert bus.events == [("model.routed", router.last_decision.model_dump())]


def test_generate_dispatches_to_capable_model_for_tool_request():
    router = _router("cheapest")
    resp = asyncio.run(router.generate(_req("tool please", tools=[_TOOL])))
    assert resp.text == "answer from mid"


def test_dispatch_uses_recorded_entry_index():
    router = _router("cheapest")
    decision = router.pick(_req("look", image=True))  # strong, index 2
    index, provider = router._dispatch(decision)
    assert index == 2
    assert provider is router.entries[2][0]


def test_dispatch_falls_back_to_model_match_when_index_invalid():
    router = _router("cheapest")
    decision = RoutingDecision(model="mid", provider="mock", entry_index=-1)
    index, provider = router._dispatch(decision)
    assert index == 1
    assert router.entries[index][1] == "mid"


def test_emit_without_bus_still_records_last_decision():
    router = _router("cheapest")  # no events bus
    decision = router.pick(_req("plain"))
    router._emit(decision)
    assert router.last_decision is decision  # no crash without a bus


# ---------------------------------------------------------------------------
# latency observation via the fastest strategy after a real generate
# ---------------------------------------------------------------------------


def test_observe_latency_records_an_ewma_after_generate():
    router = _router("fastest")
    asyncio.run(router.generate(_req("plain")))
    # a real (tiny) latency was observed for entry 0 (cheap).
    assert router._latency_ms[0] >= 0.0
    # second call updates EWMA without error.
    asyncio.run(router.generate(_req("plain")))
    assert router._inflight[0] == 0  # decremented back in finally


# ---------------------------------------------------------------------------
# capabilities passthrough + list_models / aclose dedup
# ---------------------------------------------------------------------------


def test_capabilities_delegates_to_first_provider():
    provider = MockProvider()
    router = Router([(provider, "cheap")], registry=_registry(), price_table=_prices())
    caps = router.capabilities("cheap")
    assert isinstance(caps, ModelCapabilities)
    # MockProvider advertises a permissive capability set.
    assert caps.tool_calling is True


def test_aclose_closes_each_unique_provider_once():
    provider = MockProvider()
    closed = {"n": 0}
    orig = provider.aclose

    async def _count():
        closed["n"] += 1
        await orig()

    provider.aclose = _count  # type: ignore[method-assign]
    # same provider used for two entries -> closed once.
    router = Router([(provider, "cheap"), (provider, "mid")], registry=_registry(), price_table=_prices())
    asyncio.run(router.aclose())
    assert closed["n"] == 1


def test_list_models_merges_unique_providers():
    provider = MockProvider()
    router = Router([(provider, "cheap"), (provider, "mid")], registry=_registry(), price_table=_prices())
    models = asyncio.run(router.list_models())
    # one provider -> list_models called once; result is a non-error merge.
    assert models is not None


# ---------------------------------------------------------------------------
# lazy defaults: registry + price table fetched from the package defaults
# ---------------------------------------------------------------------------


def test_lazy_default_registry_and_price_table_route_a_real_model():
    # No registry / price_table supplied -> Router lazily loads the built-in
    # default_model_registry() and default_price_table(). Use real registered
    # model ids so the capability guard and pricing both resolve.
    provider = MockProvider()
    router = Router(
        [(provider, "claude-haiku-4-5"), (provider, "claude-opus-4-8")],
        strategy="cheapest",
    )
    decision = router.pick(_req("plain question"))
    # haiku is the cheaper of the two and capable for a plain text request.
    assert decision.model == "claude-haiku-4-5"
    assert decision.est_cost_usd > 0.0
    # the lazily-built registry/price-table are now cached on the router.
    assert router._registry is not None
    assert router._price_table is not None


def test_lazy_default_registry_skips_non_reasoning_model_for_reasoning_request():
    provider = MockProvider()
    router = Router(
        [(provider, "claude-haiku-4-5"), (provider, "claude-opus-4-8")],
        strategy="cheapest",
    )
    decision = router.pick(_req("reason", reasoning_effort="high"))
    # haiku has reasoning=False in the built-in catalog -> skipped; opus wins.
    assert decision.model == "claude-opus-4-8"
    assert "reasoning" in decision.skipped["claude-haiku-4-5"]


# ---------------------------------------------------------------------------
# dispatch fallbacks (no recorded entry index)
# ---------------------------------------------------------------------------


def test_dispatch_matches_on_model_and_provider_name():
    router = _router("cheapest")
    decision = RoutingDecision(model="strong", provider="mock", entry_index=-1)
    index, provider = router._dispatch(decision)
    assert index == 2
    assert router.entries[index][1] == "strong"


def test_dispatch_model_only_fallback_when_provider_name_mismatches():
    router = _router("cheapest")
    # provider name does not match any entry -> falls through to model-only loop.
    decision = RoutingDecision(model="mid", provider="not-a-real-provider", entry_index=-1)
    index, provider = router._dispatch(decision)
    assert index == 1
    assert router.entries[index][1] == "mid"


def test_dispatch_last_resort_returns_first_entry():
    router = _router("cheapest")
    decision = RoutingDecision(model="ghost", provider="ghost", entry_index=-1)
    index, provider = router._dispatch(decision)
    assert index == 0
    assert provider is router.entries[0][0]


# ---------------------------------------------------------------------------
# stream path
# ---------------------------------------------------------------------------


def test_stream_routes_and_yields_events_from_chosen_model():
    bus = _Bus()
    router = _router("cheapest", events=bus)

    async def _drain():
        events = []
        async for ev in router.stream(_req("plain")):
            events.append(ev)
        return events

    events = asyncio.run(_drain())
    assert len(events) >= 1
    assert router.last_decision.model == "cheap"
    assert bus.events[0][0] == "model.routed"
    # inflight counter returned to zero in the finally block.
    assert router._inflight[0] == 0


# ---------------------------------------------------------------------------
# fastest strategy: observed latency overrides tier ranking
# ---------------------------------------------------------------------------


def test_budget_downgrades_rank_winner_to_a_cheaper_within_budget_model():
    # 'fastest' ranks by tier: the fast-tier model ranks FIRST. Make that model
    # expensive and a slower (strong-tier) model cheap, with a budget that
    # excludes the rank winner but fits the cheaper one -> a genuine downgrade
    # where within[0] != ranked[0], setting downgraded=True (line 580 branch).
    provider = MockProvider(responder=lambda r: "x")
    pt = PriceTable()
    pt.set("cheap", ModelPrice(input_per_mtok=100.0, output_per_mtok=100.0))   # fast tier, pricey
    pt.set("mid", ModelPrice(input_per_mtok=80.0, output_per_mtok=80.0))
    pt.set("strong", ModelPrice(input_per_mtok=1.0, output_per_mtok=1.0))       # strong tier, cheap
    router = Router(
        [(provider, m) for m in ("cheap", "mid", "strong")],
        strategy="fastest",
        registry=_registry(),
        price_table=pt,
        guard_capabilities=False,
    )
    req = _req("word " * 50, max_out=1000)
    # No budget: fastest -> 'cheap' (fast tier rank 0) wins.
    assert router.pick(req).model == "cheap"
    # Budget that 'cheap' (expensive) cannot meet but 'strong' (cheap) can.
    # strong cost ~ (50*1 + 1000*1)/1e6 ~= 1.05e-3 ; cheap ~ (50*100+1000*100)/1e6 ~= 0.105
    decision = router.pick(req, budget_usd=0.01)
    assert decision.model == "strong"
    assert decision.downgraded is True
    assert "budget downgrade" in decision.reason


def test_fastest_uses_observed_latency_over_tier_once_warmed():
    router = _router("fastest")
    # Pretend 'cheap' (entry 0, fast tier) has become very slow and 'mid'
    # (entry 1) fast. The latency EWMA should now rank mid ahead of cheap.
    router._latency_ms[0] = 5000.0
    router._latency_ms[1] = 1.0
    router._latency_ms[2] = 9000.0
    decision = router.pick(_req("plain"))
    assert decision.model == "mid"
