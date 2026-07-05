"""The model pricing & capability registry — an honest cost report by construction.

The data-driven ``ModelRegistry`` is the single source of truth the cost
``PriceTable``, the capability guard, the cost/latency router, model cascades, and
energy/carbon accounting all read from. This tour teaches how the registry is kept
*complete and held honest* by a gate, fully offline on the shipped catalog:

  * the real lineup is priced, so no GA model silently bills $0;
  * freshness is judged against the catalog's release date, never the wall clock;
  * ``coverage_report()`` is a drift detector; and it *bites* when the catalog breaks.
"""

from __future__ import annotations

from vincio.core.types import ModelProfile, TokenUsage
from vincio.observability.costs import default_price_table
from vincio.providers.openai_compat import PRESETS
from vincio.providers.registry import (
    CATALOG_RELEASED,
    FRESHNESS_HORIZON_DAYS,
    ModelRegistry,
    default_model_registry,
)


def main() -> None:
    # 1. The real lineup is priced. Ask the PriceTable what 1M tokens costs across
    #    providers: a model the providers branch on must resolve to a real price,
    #    never silently to $0 (the failure mode this registry exists to kill).
    pt = default_price_table()
    one_m_in, one_m_out = TokenUsage(input_tokens=1_000_000), TokenUsage(output_tokens=1_000_000)
    print("1. Real lineup priced (cost of 1M tokens)")
    for model, usage, label in [
        ("o3", one_m_in, "input"),
        ("gpt-4.1", one_m_out, "output"),
        ("claude-3-5-sonnet", one_m_in, "input"),
        ("deepseek-chat", one_m_out, "output"),
        ("llama-3.3-70b-versatile", one_m_in, "input"),  # the groq preset's headline model
    ]:
        print(f"   {model:28s} ${pt.cost(model, usage):>7.4f} / 1M {label}")
    # Every openai_compat preset prices its headline model, not $0.
    unpriced = [n for n, p in PRESETS.items()
                if p.default_model and default_model_registry().resolve(p.default_model) is None]
    print(f"   openai_compat presets with an UNpriced headline model: {unpriced or 'none'}")

    # 2. Freshness is deterministic. The horizon is measured from the catalog's
    #    *release* date, so a frozen release reports the same verdict forever — only
    #    a genuinely newer snapshot (pass as_of=) can surface a stale price.
    reg = ModelRegistry()
    fresh = reg.coverage_report()
    stale = reg.coverage_report(as_of="2027-06-29")
    print(f"\n2. Freshness vs release date ({CATALOG_RELEASED}, horizon {FRESHNESS_HORIZON_DAYS}d)")
    print(f"   at release: no_stale_prices={fresh.no_stale_prices}; "
          f"a year on: {len(stale.stale)} prices now past the horizon")

    # 3. coverage_report() is the drift detector: one call proves every default and
    #    heuristic family resolves to a priced, GA profile and that no router/cascade
    #    pick moved. Run it in CI to catch a catalog edit that quietly changed billing.
    report = default_model_registry().coverage_report()
    print(f"\n3. coverage_report() — {report.model_count} models / {report.provider_count} providers")
    for label, passed in [
        ("defaults resolve, priced & GA", report.default_models_resolve),
        ("no GA billable model bills $0", report.no_silent_zero),
        ("no price past freshness horizon", report.no_stale_prices),
        ("no routing/cascade/energy drift", report.no_routing_drift),
    ]:
        print(f"   [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"   overall ok={report.ok}")

    # 4. The gate bites. Plant a GA chat model at $0, or make the cheapest model
    #    pricey, and the matching invariant flips to fail — the pinpointed reason
    #    is in the report, so a bad catalog edit can't merge green.
    silent = ModelRegistry()
    silent.register(ModelProfile(name="ghost", provider="openai", model="ghost-pro"))
    drifted = ModelRegistry()
    nano = drifted.resolve("gpt-5.2-nano")
    drifted.register(nano.model_copy(update={"input_cost_per_mtok": 99.0}))
    print("\n4. The gate bites")
    print(f"   planted $0 GA model → no_silent_zero={silent.coverage_report().no_silent_zero}")
    drift = drifted.coverage_report()
    print(f"   made cheapest model pricey → no_routing_drift={drift.no_routing_drift} ({drift.drift[0]})")

    # 5. `vincio registry sync` is review-only: it diffs a provider's live list into
    #    a candidate overlay (new ids a human must price; dated snapshots fold in as
    #    aliases) and never mutates the catalog — pricing stays a reviewed decision.
    reg = default_model_registry()
    live = [
        ModelProfile(name="new", provider="acme", model="acme-llm-9b"),
        ModelProfile(name="snap", provider="openai", model="gpt-4o-2099-01-01"),
    ]
    candidates = [p.model for p in live if reg.resolve(p.model) is None]
    print("\n5. registry sync — review-only overlay")
    print(f"   needs a human-set price: {candidates}; catalog unchanged "
          f"(acme-llm-9b resolves? {reg.resolve('acme-llm-9b') is not None})")

    print("\nDone — the registry prices the real lineup and a gate keeps it honest, "
          "fresh, and routing-stable, fully offline.")


if __name__ == "__main__":
    main()
