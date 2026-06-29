"""5.1 — model pricing & capability registry coverage, freshness, and honesty.

The data-driven :class:`ModelRegistry` is the single source of truth the cost
``PriceTable``, the capability guard, the cost/latency router, the model
cascades, and the energy accounting read from. These tests prove the shipped
``model_catalog.json`` is complete, honest (no silent $0), fresh (against a
deterministic horizon), and routing-stable — and that each gate actually bites.
"""

from __future__ import annotations

import warnings
from datetime import date

import pytest

from vincio.core.types import ModelCapabilities, ModelProfile
from vincio.providers.openai_compat import PRESETS
from vincio.providers.registry import (
    CATALOG_RELEASED,
    FRESHNESS_HORIZON_DAYS,
    ModelRegistry,
    RegistryCoverageReport,
    default_model_registry,
)


@pytest.fixture
def reg() -> ModelRegistry:
    return ModelRegistry()


# -- the shipped catalog passes every gate -----------------------------------


def test_shipped_catalog_is_fully_covered(reg: ModelRegistry):
    report = reg.coverage_report()
    assert isinstance(report, RegistryCoverageReport)
    assert report.ok, (report.gaps, report.unpriced, report.stale, report.drift)
    assert report.coverage_complete
    assert report.default_models_resolve
    assert report.capability_families_resolve
    assert report.presets_priced
    assert report.no_silent_zero
    assert report.no_stale_prices
    assert report.no_routing_drift
    assert report.gaps == [] and report.unpriced == [] and report.stale == [] and report.drift == []


def test_coverage_is_deterministic_against_release_date_not_clock(reg: ModelRegistry):
    # as_of defaults to the catalog's release date, so two calls agree regardless
    # of when the test runs — the freshness verdict never reads the wall clock.
    first = reg.coverage_report()
    second = reg.coverage_report()
    assert first.as_of == CATALOG_RELEASED == second.as_of
    assert first.model_dump() == second.model_dump()


def test_every_provider_default_resolves_priced_and_ga(reg: ModelRegistry):
    for provider, model_id in (
        ("openai", "gpt-5.2-mini"),
        ("anthropic", "claude-sonnet-4-6"),
        ("google", "gemini-2.5-flash"),
        ("mistral", "mistral-small-latest"),
    ):
        profile = reg.resolve(model_id)
        assert profile is not None and profile.provider == provider
        assert profile.input_cost_per_mtok > 0 and profile.output_cost_per_mtok > 0
        assert profile.lifecycle(as_of=date.fromisoformat(CATALOG_RELEASED)) == "ga"


def test_every_preset_headline_model_is_priced(reg: ModelRegistry):
    priced = [p for p in PRESETS.values() if p.default_model]
    assert priced, "presets should declare default models"
    for preset in priced:
        profile = reg.resolve(preset.default_model)
        assert profile is not None, preset.default_model
        assert profile.input_cost_per_mtok > 0, preset.default_model


def test_priced_as_of_on_every_billable_paid_ga_model(reg: ModelRegistry):
    as_of = date.fromisoformat(CATALOG_RELEASED)
    for profile in reg.profiles():
        if profile.provider in ("local", "mock"):
            continue
        if profile.lifecycle(as_of=as_of) != "ga":
            continue
        billable = profile.input_cost_per_mtok > 0 or profile.output_cost_per_mtok > 0
        if billable:
            assert profile.priced_as_of, f"{profile.model} has a price but no priced_as_of"


# -- the gates are real, not vacuous -----------------------------------------


def test_freshness_gate_fires_past_the_horizon(reg: ModelRegistry):
    fresh = reg.coverage_report(as_of=CATALOG_RELEASED)
    assert fresh.no_stale_prices
    # A year past release exceeds the 180-day horizon for prices stamped at release.
    stale = reg.coverage_report(as_of="2027-06-29")
    assert stale.no_stale_prices is False
    assert stale.stale and not stale.ok


def test_freshness_horizon_boundary():
    reg = ModelRegistry()
    # The reference date is exactly horizon days after the release date: a price
    # stamped at release sits right on the boundary and is still fresh.
    boundary = date.fromisoformat(CATALOG_RELEASED).toordinal() + FRESHNESS_HORIZON_DAYS
    on_edge = reg.coverage_report(as_of=date.fromordinal(boundary))
    assert on_edge.no_stale_prices
    just_past = reg.coverage_report(as_of=date.fromordinal(boundary + 1))
    assert just_past.no_stale_prices is False


def test_no_silent_zero_gate_fires_on_unpriced_chat_model(reg: ModelRegistry):
    reg.register(ModelProfile(name="ghost", provider="openai", model="ghost-chat",
                              capabilities=ModelCapabilities(tool_calling=True)))
    report = reg.coverage_report()
    assert report.no_silent_zero is False
    assert "ghost-chat" in report.unpriced
    assert not report.ok


def test_free_and_deprecated_models_are_exempt_from_silent_zero(reg: ModelRegistry):
    # local/mock are free; text-embedding-004 is a deprecated free embedder. None
    # should be flagged as a silent $0.
    report = reg.coverage_report()
    assert "local" not in report.unpriced and "mock" not in report.unpriced
    assert "text-embedding-004" not in report.unpriced


def test_no_routing_drift_gate_fires_when_cheapest_changes(reg: ModelRegistry):
    nano = reg.resolve("gpt-5.2-nano")
    reg.register(nano.model_copy(update={"input_cost_per_mtok": 99.0}))
    report = reg.coverage_report()
    assert report.no_routing_drift is False
    assert report.drift and not report.ok


def test_routing_anchors_match_bench_cost_router_picks():
    # The no-routing-drift anchors must agree with what the live Router picks, so
    # the gate protects the real selection (mirrors benchmarks bench_cost).
    from vincio.core.types import Message, ModelRequest
    from vincio.optimize.routing import Router
    from vincio.providers import MockProvider

    req = ModelRequest(model="x", messages=[Message(role="user", content="route this")])
    router = Router.from_models(
        MockProvider(default_text="x"), ["gpt-5.2", "gpt-5.2-mini", "gpt-5.2-nano"],
        strategy="cheapest",
    )
    assert router.pick(req).model == "gpt-5.2-nano"


# -- summary + introspection --------------------------------------------------


def test_summary_is_flat_and_json_friendly(reg: ModelRegistry):
    summary = reg.coverage_report().summary()
    assert summary["ok"] is True
    assert set(summary) >= {
        "ok", "coverage_complete", "no_silent_zero", "no_stale_prices",
        "no_routing_drift", "model_count", "provider_count",
    }
    assert summary["model_count"] >= 45


def test_default_registry_singleton_covers(reg: ModelRegistry):
    # The process-wide registry the cost/energy/router all read from is covered.
    assert default_model_registry().coverage_report().ok


# -- review-only sync emits a candidate overlay, never mutates the catalog -----


def test_registry_sync_emits_candidate_overlay_without_mutating(reg: ModelRegistry):
    from vincio.providers.base import run_sync

    class _FakeProvider:
        name = "fake"

        async def list_models(self) -> list[ModelProfile]:
            # one genuinely new id, one dated snapshot of a known model.
            return [
                ModelProfile(name="brand-new", provider="fake", model="brand-new-7b"),
                ModelProfile(name="snap", provider="openai", model="gpt-4o-2099-01-01"),
            ]

    live = run_sync(_FakeProvider().list_models())
    before = reg.resolve("brand-new-7b")
    overlay = [p.model_dump(exclude_none=True) for p in live if reg.resolve(p.model) is None]
    # The genuinely new id is a candidate; the dated snapshot folds into gpt-4o.
    assert any(m["model"] == "brand-new-7b" for m in overlay)
    assert reg.resolve("gpt-4o-2099-01-01").model == "gpt-4o"
    # Computing the overlay never registered anything (review-only).
    assert before is None and reg.resolve("brand-new-7b") is None


def test_registry_coverage_cli_exit_codes():
    from vincio.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["registry", "coverage", "--json"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert args.fn(args) == 0
