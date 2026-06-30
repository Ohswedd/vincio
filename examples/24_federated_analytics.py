"""Cross-org / federated analytics — the data plane across a trust boundary.

A single organization's analytics plane is the capstone of example 20. But an
analytical question often spans **more than one organization's** data — total
revenue across a partnership, a benchmark over a cohort of independent operators —
and the answer must be computed **without pooling the raw rows into a shared
warehouse**. This program walks the federated reach, fully offline with two
in-process orgs and a coordinator:

  * `FederatedQuery` — the *shape* of one governed metric run everywhere (the
    measures, dimensions, the columns it touches, the residency posture, the
    budget), bound by its content digest into a negotiated `Contract`;
  * `app.federated_data_engagement` — the facade that negotiates the contract,
    choreographs a contract-governed `Saga` (one step per org, each running that
    org's governed query plane **locally**), and returns only the aggregated,
    cell-cited `MetricResult` — never the raw rows;
  * reconciliation — `SUM`/`COUNT` add across orgs, `MIN`/`MAX` take the extremum,
    group by group, into one `FederatedFinding` per metric;
  * `FederatedNarrative` — the sealed, signed, hash-chained narrative whose every
    finding **re-derives from each org's content-hashed source**: `verify()`
    re-executes each org's aggregate and re-derives every reconciled value;
  * governance at the boundary — residency egress refusal, the consent ledger's
    analytics purpose, the differential-privacy budget, and a k-anonymity
    contributor floor all apply exactly as they would to a local query.

Everything below is opt-in and additive; none of it touches a database, a network,
or a shared warehouse — the raw rows never leave the org that holds them.
"""

from __future__ import annotations

from _shared import example_provider

from vincio import ContextApp, FederatedNarrative, FederatedQuery
from vincio.core.errors import DataError, ResidencyViolationError
from vincio.data import DerivedColumn, Dimension, Measure
from vincio.governance.consent import Purpose

COLS = ["region", "price", "qty", "account"]
# Each org's private ledger. The "account" column is raw, per-row data that must
# NEVER cross the trust boundary — only the aggregated metric does.
ACME_ROWS = [
    {"region": "NA", "price": 10.0, "qty": 3, "account": "acme-cust-7781"},
    {"region": "EU", "price": 8.0, "qty": 5, "account": "acme-cust-2294"},
    {"region": "NA", "price": 11.0, "qty": 6, "account": "acme-cust-5510"},
]
GLOBEX_ROWS = [
    {"region": "EU", "price": 9.0, "qty": 4, "account": "globex-cust-3140"},
    {"region": "APAC", "price": 7.0, "qty": 2, "account": "globex-cust-8862"},
    {"region": "NA", "price": 12.0, "qty": 1, "account": "globex-cust-1207"},
]

# The shared governed vocabulary every org computes the metric by — defined once,
# identically, so the federated metric means the same thing everywhere.
DERIVED = [DerivedColumn(name="revenue", expression="price * qty", unit="USD")]
MEASURES = [
    Measure(name="total_revenue", agg="sum", expression="revenue", unit="USD"),
    Measure(name="order_count", agg="count"),
]
DIMENSIONS = [Dimension(name="region")]


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def build_org(name: str, rows: list[dict], region: str) -> ContextApp:
    """An organization with its *own* data, layer, audit chain, and signer."""
    provider, _ = example_provider()
    app = ContextApp(name=name, provider=provider)
    app.register_dataset(rows, columns=COLS, name="sales", source=f"{name}-crm")
    app.semantic_layer("sales", derived=DERIVED, measures=MEASURES, dimensions=DIMENSIONS)
    return app


def build_coordinator() -> ContextApp:
    """The query requester. It holds the layer *definitions* but no data."""
    provider, _ = example_provider()
    app = ContextApp(name="coordinator", provider=provider)
    app.semantic_layer(
        "sales", derived=DERIVED, measures=MEASURES, dimensions=DIMENSIONS, validate=False
    )
    return app


def run_federated(coordinator: ContextApp, acme: ContextApp, globex: ContextApp):
    banner("1. one governed metric, run across two orgs, no shared warehouse")
    query = FederatedQuery.of(
        ["total_revenue", "order_count"],
        table="sales",
        by=["region"],
        columns_touched=["region", "price", "qty"],
        min_members=2,
    )
    fed = coordinator.federated_data_engagement(query=query)
    fed.add_member("acme", acme, region="us-east-1")
    fed.add_member("globex", globex, region="eu-west-1")
    findings = fed.run()  # negotiate → dispatch (choreographed) → reconcile
    narrative = fed.seal()
    print("   stages threaded:", " → ".join(narrative.stage_names))
    for f in findings:
        print(f"   {f.metric:<14} {f.group_label:<10} = {f.value!r:<8} ({f.op} of {f.members})")
    return fed, query, narrative


def section_rows_never_cross(narrative: FederatedNarrative) -> None:
    banner("2. the raw rows never cross — only the aggregate does")
    import json

    blob = json.dumps(narrative.to_wire())
    leaked = [r["account"] for r in (*ACME_ROWS, *GLOBEX_ROWS) if r["account"] in blob]
    print("   per-row account ids in the sealed narrative:", leaked or "none")
    print("   only aggregated group rows crossed:", not leaked)


def section_offline_and_data_bound(coordinator: ContextApp, fed) -> None:
    banner("3. the narrative verifies offline, and every finding re-derives from each source")
    v = fed.verify(coordinator.contract_signer)
    print("   valid:", v.valid, "| chain intact:", v.intact, "| data-bound:", v.data_bound)
    restored = FederatedNarrative.from_wire(fed.narrative.to_wire())
    print("   round-trips and verifies:", restored.verify(coordinator.contract_signer).valid)


def section_governance(coordinator: ContextApp, acme: ContextApp, globex: ContextApp) -> None:
    banner("4. governance crosses the boundary intact — refusals apply as for a local query")

    # Residency-aware egress refusal: an EU-only posture refuses the US org.
    eu_only = FederatedQuery.of("total_revenue", table="sales", residency=["eu"], min_members=1)
    fed = coordinator.federated_data_engagement(query=eu_only)
    fed.add_member("acme", acme, region="us-east-1")
    fed.add_member("globex", globex, region="eu-west-1")
    try:
        fed.run()
    except ResidencyViolationError as exc:
        print("   residency egress refused:   ", str(exc)[:64], "…")

    # Consent: an org without ANALYTICS consent is refused.
    strict = build_org("strict", ACME_ROWS, "eu-west-1")
    strict.use_consent_ledger(default_allow=False)  # default-deny, no grant
    fed2 = coordinator.federated_data_engagement(
        query=FederatedQuery.of("total_revenue", table="sales", min_members=1)
    )
    fed2.add_member("strict", strict, region="eu-west-1")
    try:
        fed2.run()
    except DataError as exc:
        print("   consent refused:            ", str(exc)[:64], "…")
    # …granting it lets the same org contribute.
    strict.consent_ledger.grant("strict", [Purpose.ANALYTICS])
    fed3 = coordinator.federated_data_engagement(
        query=FederatedQuery.of("total_revenue", table="sales", min_members=1)
    )
    fed3.add_member("strict", strict, region="eu-west-1")
    print("   consent granted → contributes:", fed3.run()[0].value is not None)

    # The k-anonymity contributor floor refuses a round that would single out one org.
    floor = FederatedQuery.of("total_revenue", table="sales", min_members=3)
    fed4 = coordinator.federated_data_engagement(query=floor)
    fed4.add_member("acme", acme, region="us-east-1")
    fed4.add_member("globex", globex, region="eu-west-1")
    try:
        fed4.run()
    except DataError as exc:
        print("   contributor floor refused:  ", str(exc)[:64], "…")


def section_tamper(coordinator: ContextApp, acme: ContextApp, globex: ContextApp) -> None:
    banner("5. a tamper anywhere — chain, signature, or reconciliation — is caught")
    fed = coordinator.federated_data_engagement(
        query=FederatedQuery.of("total_revenue", table="sales", by=["region"], min_members=2)
    )
    fed.add_member("acme", acme, region="us-east-1")
    fed.add_member("globex", globex, region="eu-west-1")
    fed.run()
    narrative = fed.seal()
    # A re-ordered stage breaks the hash chain.
    reordered = FederatedNarrative.from_wire(narrative.to_wire())
    reordered.stages[1], reordered.stages[2] = reordered.stages[2], reordered.stages[1]
    print("   re-ordered stage caught:    ", not reordered.verify().valid)
    # A tampered reconciliation no longer re-derives from the orgs' sources.
    fed.findings[0].value = 999_999.0
    print(
        "   tampered reconciliation caught:",
        fed.verify(coordinator.contract_signer).data_bound is False,
    )


def main() -> None:
    coordinator = build_coordinator()
    acme = build_org("acme", ACME_ROWS, "us-east-1")
    globex = build_org("globex", GLOBEX_ROWS, "eu-west-1")
    fed, _query, narrative = run_federated(coordinator, acme, globex)
    section_rows_never_cross(narrative)
    section_offline_and_data_bound(coordinator, fed)
    section_governance(coordinator, acme, globex)
    section_tamper(coordinator, acme, globex)
    print(
        "\nDone — a governed metric computed across organizations into one signed, "
        "data-bound, offline-verifiable narrative, with the raw rows never leaving home."
    )


if __name__ == "__main__":
    main()
