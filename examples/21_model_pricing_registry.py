"""The model pricing & capability registry: an honest cost report by construction.

The data-driven `ModelRegistry` is the single source of truth the cost `PriceTable`,
the capability guard, the cost/latency router, the model cascades, and the
energy/carbon accounting all read from. This program walks 5.1 — the registry made
*complete and held* — fully offline on the shipped catalog:

  * the catalog ships as reviewable data (`vincio/providers/model_catalog.json`) and
    prices the real current lineup of every provider Vincio supports, so a model the
    providers branch on (OpenAI o-series / gpt-4.1, the openai_compat presets, …) no
    longer resolves to nothing and silently bills $0;
  * `priced_as_of` stamps each profile, and an `as_of`-deterministic freshness horizon
    is evaluated against the catalog's *release* date — never the wall clock — so a
    frozen release reports the same verdict forever and only a genuinely stale snapshot
    fails the gate;
  * `registry.coverage_report()` is a drift detector proving every provider default and
    capability-heuristic family resolves to a non-sparse, priced profile, that no GA
    billable model silently costs $0, and that the canonical router/cascade picks are
    unchanged by a refresh;
  * `vincio registry sync` is review-only: it diffs a provider's live model list into a
    candidate overlay for a human to price and merge — it never mutates the catalog.

Everything below is deterministic and offline; none of it touches a price feed or a
network.
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


def banner(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def section_priced_lineup() -> None:
    banner("1. the real lineup is priced — no more silent $0")
    pt = default_price_table()
    one_m_in = TokenUsage(input_tokens=1_000_000)
    one_m_out = TokenUsage(output_tokens=1_000_000)
    for model, usage, label in [
        ("o3", one_m_in, "input"),
        ("gpt-4.1", one_m_out, "output"),
        ("text-embedding-3-small", one_m_in, "input"),
        ("claude-3-5-sonnet", one_m_in, "input"),
        ("mistral-medium-latest", one_m_out, "output"),
        ("deepseek-chat", one_m_out, "output"),
        ("llama-3.3-70b-versatile", one_m_in, "input"),  # the groq preset's headline model
    ]:
        cost = pt.cost(model, usage)
        print(f"   {model:32s} ${cost:>7.4f} / 1M {label} tokens")
    print("   (every openai_compat preset prices its headline model instead of $0)")
    for name, preset in PRESETS.items():
        if preset.default_model:
            profile = default_model_registry().resolve(preset.default_model)
            print(f"     {name:12s} → {preset.default_model:42s} "
                  f"${profile.input_cost_per_mtok}/{profile.output_cost_per_mtok}")


def section_freshness_is_deterministic() -> None:
    banner("2. freshness is evaluated against the release date, not the clock")
    reg = ModelRegistry()
    print(f"   catalog released:   {CATALOG_RELEASED}")
    print(f"   freshness horizon:  {FRESHNESS_HORIZON_DAYS} days")
    at_release = reg.coverage_report()
    print(f"   fresh at release:   no_stale_prices={at_release.no_stale_prices} "
          f"(as_of={at_release.as_of})")
    a_year_out = reg.coverage_report(as_of="2027-06-29")
    print(f"   a year later:       no_stale_prices={a_year_out.no_stale_prices} "
          f"({len(a_year_out.stale)} prices now past the horizon)")
    print("   → a frozen release never rots; only a new snapshot can surface a stale price")


def section_coverage_report() -> None:
    banner("3. coverage_report() — the drift detector")
    report = default_model_registry().coverage_report()
    print(f"   {report.model_count} models across {report.provider_count} providers")
    for label, passed in [
        ("every provider default resolves, priced & GA", report.default_models_resolve),
        ("every capability-heuristic family resolves", report.capability_families_resolve),
        ("every openai_compat preset is priced", report.presets_priced),
        ("no GA billable model silently bills $0", report.no_silent_zero),
        ("no price drifted past the freshness horizon", report.no_stale_prices),
        ("no routing/cascade/energy pick changed", report.no_routing_drift),
    ]:
        print(f"   [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"   overall: {'OK' if report.ok else 'FAIL'}")


def section_gate_bites() -> None:
    banner("4. the gates bite — a broken catalog fails coverage")
    silent = ModelRegistry()
    silent.register(ModelProfile(name="ghost", provider="openai", model="ghost-pro"))
    rep = silent.coverage_report()
    print(f"   plant a GA chat model at $0 → no_silent_zero={rep.no_silent_zero}, "
          f"unpriced={rep.unpriced}")

    drifted = ModelRegistry()
    nano = drifted.resolve("gpt-5.2-nano")
    drifted.register(nano.model_copy(update={"input_cost_per_mtok": 99.0}))
    rep = drifted.coverage_report()
    print(f"   make the cheapest model pricey → no_routing_drift={rep.no_routing_drift}")
    print(f"     {rep.drift[0]}")


def section_review_only_sync() -> None:
    banner("5. vincio registry sync — review-only candidate overlay")
    reg = default_model_registry()

    # A provider's live model list (here a deterministic stand-in) is diffed into a
    # candidate overlay: genuinely new ids need a human-set price; dated snapshots of
    # known models fold in as aliases. Nothing is registered — the catalog is unchanged.
    live = [
        ModelProfile(name="new", provider="acme", model="acme-llm-9b"),
        ModelProfile(name="snap", provider="openai", model="gpt-4o-2099-01-01"),
    ]
    candidates = [p.model for p in live if reg.resolve(p.model) is None]
    folds_in = [p.model for p in live if reg.resolve(p.model) is not None]
    print(f"   candidate (needs price + capabilities): {candidates}")
    print(f"   folds in as alias of a known model:     "
          f"{[(m, reg.resolve(m).model) for m in folds_in]}")
    print(f"   catalog still unchanged:                acme-llm-9b resolves? "
          f"{reg.resolve('acme-llm-9b') is not None}")


def main() -> None:
    section_priced_lineup()
    section_freshness_is_deterministic()
    section_coverage_report()
    section_gate_bites()
    section_review_only_sync()
    print("\nDone — the registry prices the real lineup and is held honest, fresh, "
          "and routing-stable by a gate, fully offline.")


if __name__ == "__main__":
    main()
