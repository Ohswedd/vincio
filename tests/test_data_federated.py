"""Cross-org / federated analytics — composition, governance, conformance.

The cross-org twin of the single-org data engagement. These tests prove the facade
runs one governed metric across organizations over the existing cross-org fabric:
negotiated as a `Contract`, choreographed as a `Saga` whose steps run each org's
governed query plane **locally** and return only the aggregated, cell-cited
`MetricResult` — never the raw rows — reconciled into one hash-linked, signed,
offline-verifiable `FederatedNarrative` whose every finding **re-derives from each
org's content-hashed source**; that residency egress refusal, the consent ledger,
the differential-privacy budget, and the k-anonymity contributor floor each refuse
a non-compliant round exactly as a local query's governance would; and that a
tamper introduced anywhere — a re-ordered stage, an edited digest, a forged
signature, a tampered source, or an edited reconciliation — is caught from the
bytes alone.
"""

from __future__ import annotations

import json

import pytest

from vincio import (
    ContextApp,
    FederatedDataEngagement,
    FederatedFinding,
    FederatedNarrative,
    FederatedQuery,
    FederatedStage,
    VincioConfig,
)
from vincio.core.errors import DataError, ResidencyViolationError
from vincio.data import DerivedColumn, Dimension, Measure
from vincio.governance.consent import Purpose
from vincio.governance.privacy import PrivacyBudgetError
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner

COLS = ["region", "price", "qty", "account"]
ACME_ROWS = [
    {"region": "NA", "price": 10.0, "qty": 3, "account": "acme-7781"},
    {"region": "EU", "price": 8.0, "qty": 5, "account": "acme-2294"},
    {"region": "NA", "price": 11.0, "qty": 6, "account": "acme-5510"},
]
GLOBEX_ROWS = [
    {"region": "EU", "price": 9.0, "qty": 4, "account": "globex-3140"},
    {"region": "APAC", "price": 7.0, "qty": 2, "account": "globex-8862"},
    {"region": "NA", "price": 12.0, "qty": 1, "account": "globex-1207"},
]

DERIVED = [DerivedColumn(name="revenue", expression="price * qty")]
MEASURES = [
    Measure(name="total_revenue", agg="sum", expression="revenue"),
    Measure(name="order_count", agg="count"),
    Measure(name="max_price", agg="max", expression="price"),
]
DIMENSIONS = [Dimension(name="region")]


def _app(name: str) -> ContextApp:
    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), config=cfg)


def _org(name: str, rows: list[dict]) -> ContextApp:
    app = _app(name)
    app.register_dataset(rows, columns=COLS, name="sales", source=f"{name}-crm")
    app.semantic_layer("sales", derived=DERIVED, measures=MEASURES, dimensions=DIMENSIONS)
    return app


def _coordinator() -> ContextApp:
    app = _app("coordinator")
    app.semantic_layer(
        "sales", derived=DERIVED, measures=MEASURES, dimensions=DIMENSIONS, validate=False
    )
    return app


def _federation(
    *,
    metrics="total_revenue",
    by=("region",),
    min_members=2,
    **query_kwargs,
) -> tuple[ContextApp, FederatedDataEngagement, FederatedQuery]:
    coord = _coordinator()
    query = FederatedQuery.of(
        metrics, table="sales", by=list(by), min_members=min_members, **query_kwargs
    )
    fed = coord.federated_data_engagement(query=query)
    fed.add_member("acme", _org("acme", ACME_ROWS), region="us-east-1")
    fed.add_member("globex", _org("globex", GLOBEX_ROWS), region="eu-west-1")
    return coord, fed, query


# -- construction & factory ---------------------------------------------------


def test_factory_returns_facade():
    coord = _coordinator()
    q = FederatedQuery.of("total_revenue", table="sales")
    fed = coord.federated_data_engagement(query=q, coordinator="ctrl")
    assert isinstance(fed, FederatedDataEngagement)
    assert fed.coordinator == "ctrl"
    assert fed.query is q


def test_coordinator_defaults_to_app_name():
    assert _coordinator().federated_data_engagement().coordinator == "coordinator"


def test_add_member_defaults_table_to_query():
    coord, fed, _ = _federation()
    assert all(m.table == "sales" for m in fed.members)


# -- composition: the lifecycle threads a verifiable narrative -----------------


def test_run_threads_every_stage_in_order():
    coord, fed, _ = _federation(metrics=["total_revenue", "order_count"])
    fed.run()
    narrative = fed.seal()
    # negotiate, choreograph, one query stage per member, reconcile
    assert narrative.stage_names == ["negotiate", "choreograph", "query", "query", "reconcile"]
    assert [s.member for s in narrative.stages if s.stage == "query"] == ["acme", "globex"]


def test_reconciliation_is_exact_sum_and_count():
    coord, fed, _ = _federation(metrics=["total_revenue", "order_count"])
    findings = fed.run()
    rev = {f.group["region"]: f.value for f in findings if f.metric == "total_revenue"}
    cnt = {f.group["region"]: f.value for f in findings if f.metric == "order_count"}
    # NA: acme 30+66 + globex 12 = 108 ; EU: 40 + 36 = 76 ; APAC: 14 (globex only)
    assert rev == {"NA": 108.0, "EU": 76.0, "APAC": 14.0}
    assert cnt == {"NA": 3, "EU": 2, "APAC": 1}


def test_reconciliation_extremum_for_max():
    coord, fed, _ = _federation(metrics="max_price")
    findings = fed.run()
    by_region = {f.group["region"]: f.value for f in findings}
    assert by_region["NA"] == 12.0  # max(10, 11) acme vs max(12) globex
    assert findings[0].op == "max"


def test_ungrouped_total_reconciles():
    coord, fed, _ = _federation(metrics="total_revenue", by=())
    findings = fed.run()
    assert len(findings) == 1
    # acme: 30+40+66 = 136 ; globex: 36+14+12 = 62
    assert findings[0].value == 136.0 + 62.0
    assert findings[0].group_label == "*"


def test_findings_record_per_member_contributions():
    coord, fed, _ = _federation(metrics="total_revenue")
    fed.run()
    finding = next(f for f in fed.findings if f.group.get("region") == "NA")
    orgs = {c.org for c in finding.contributions}
    assert orgs == {"acme", "globex"}
    assert all(c.source_hash for c in finding.contributions)


def test_finding_lookup_on_narrative():
    coord, fed, _ = _federation(metrics="total_revenue")
    fed.run()
    nar = fed.seal()
    assert nar.finding("total_revenue", region="EU").value == 76.0
    assert nar.finding("total_revenue", region="ZZ") is None


# -- the raw rows never cross the trust boundary ------------------------------


def test_raw_rows_never_cross_into_the_narrative():
    coord, fed, _ = _federation(metrics=["total_revenue", "order_count"])
    fed.run()
    blob = json.dumps(fed.seal().to_wire())
    # No per-row account id (raw, un-aggregated data) appears anywhere in the sealed
    # narrative — only group-by aggregates crossed.
    for row in (*ACME_ROWS, *GLOBEX_ROWS):
        assert row["account"] not in blob


def test_raw_rows_never_cross_into_the_saga_journal():
    coord, fed, _ = _federation(metrics="total_revenue")
    fed.run()
    journal_blob = json.dumps(fed.delivery.journal.model_dump(mode="json"))
    for row in (*ACME_ROWS, *GLOBEX_ROWS):
        assert row["account"] not in journal_blob


# -- offline verification, data-binding, audit, wire --------------------------


def test_narrative_seals_signs_and_verifies_offline():
    coord, fed, _ = _federation()
    fed.run()
    narrative = fed.seal()
    v = narrative.verify(coord.contract_signer)
    assert v.valid and v.intact and v.head_ok and v.hash_ok and v.signatures_ok
    assert v.signed_by == ["coordinator"]


def test_engagement_is_data_bound_against_live_members():
    coord, fed, _ = _federation()
    fed.run()
    v = fed.verify(coord.contract_signer)
    assert v.valid and v.digests_ok and v.data_bound is True


def test_data_binding_unchecked_without_members():
    coord, fed, _ = _federation()
    fed.run()
    narrative = fed.seal()
    # the narrative's own verify is offline-only; data_bound stays None there
    assert narrative.verify(coord.contract_signer).data_bound is None


def test_seal_lands_on_audit_chain():
    coord, fed, _ = _federation()
    fed.run()
    narrative = fed.seal()
    assert narrative.audit_id is not None
    assert len(coord.audit.query(action="federated_data_engagement")) == 1
    assert coord.audit.verify_chain()


def test_governance_decisions_are_audited():
    coord, fed, _ = _federation()
    fed.run()
    decisions = coord.audit.query(action="federated_query_governance")
    assert len(decisions) == 2  # one allow per approved member
    assert all(e.decision == "allow" for e in decisions)


def test_narrative_round_trips_through_wire():
    coord, fed, _ = _federation(metrics=["total_revenue", "order_count"])
    fed.run()
    narrative = fed.seal()
    restored = FederatedNarrative.from_wire(narrative.to_wire())
    assert restored.verify(coord.contract_signer).valid
    assert restored.stage_names == narrative.stage_names
    assert [f.value for f in restored.findings] == [f.value for f in narrative.findings]
    stage = FederatedStage.from_wire(narrative.stages[0].to_wire())
    assert stage.entry_hash == narrative.stages[0].entry_hash


def test_member_local_query_audited_on_its_own_chain():
    coord, fed, _ = _federation()
    fed.run()
    # Each org's governed metric ran on its OWN query plane, audited on its chain.
    acme = next(m for m in fed.members if m.org == "acme")
    assert len(acme.app.audit.query(action="metric_query")) >= 1


# -- governance preservation: refusals at the boundary ------------------------


def test_residency_egress_refusal():
    coord = _coordinator()
    q = FederatedQuery.of("total_revenue", table="sales", residency=["eu"], min_members=1)
    fed = coord.federated_data_engagement(query=q)
    fed.add_member("acme", _org("acme", ACME_ROWS), region="us-east-1")
    fed.add_member("globex", _org("globex", GLOBEX_ROWS), region="eu-west-1")
    with pytest.raises(ResidencyViolationError):
        fed.run()
    # the refusal is audited on the coordinator's chain
    denials = coord.audit.query(action="federated_query_governance")
    assert any(e.decision == "deny" for e in denials)


def test_residency_allows_in_jurisdiction_member():
    coord = _coordinator()
    q = FederatedQuery.of("total_revenue", table="sales", residency=["eu"], min_members=1)
    fed = coord.federated_data_engagement(query=q)
    fed.add_member("globex", _org("globex", GLOBEX_ROWS), region="eu-west-1")
    findings = fed.run()  # eu-west-1 is in the 'eu' jurisdiction
    assert findings and findings[0].value is not None


def test_consent_refusal_without_analytics_purpose():
    from vincio.governance.consent import ConsentLedger

    coord = _coordinator()
    strict = _org("strict", ACME_ROWS)
    # A fresh, storeless default-deny ledger (no cross-test grants); the data subject
    # has no ANALYTICS consent, so the contribution is refused.
    strict.use_consent_ledger(ConsentLedger(default_allow=False))
    q = FederatedQuery.of("total_revenue", table="sales", min_members=1)
    fed = coord.federated_data_engagement(query=q)
    fed.add_member("strict", strict, subject="eu-data-subjects")
    with pytest.raises(DataError, match="ANALYTICS consent"):
        fed.run()


def test_consent_granted_allows_contribution():
    from vincio.governance.consent import ConsentLedger

    coord = _coordinator()
    strict = _org("strict", ACME_ROWS)
    ledger = strict.use_consent_ledger(ConsentLedger(default_allow=False))
    ledger.grant("eu-data-subjects", [Purpose.ANALYTICS])
    q = FederatedQuery.of("total_revenue", table="sales", min_members=1)
    fed = coord.federated_data_engagement(query=q)
    fed.add_member("strict", strict, subject="eu-data-subjects")
    assert fed.run()[0].value is not None


def test_differential_privacy_budget_refusal():
    coord = _coordinator()
    acme = _org("acme", ACME_ROWS)
    acme.use_privacy_accountant()
    acme.set_privacy_budget(subject_id="acme", epsilon=0.0001, on_breach="refuse")
    q = FederatedQuery.of("total_revenue", table="sales", min_members=1)
    fed = coord.federated_data_engagement(query=q)
    fed.add_member("acme", acme)
    with pytest.raises(PrivacyBudgetError):
        fed.run()


def test_min_members_contributor_floor_refusal():
    coord, fed, _ = _federation(min_members=3)  # only 2 members
    with pytest.raises(DataError, match="contributor floor"):
        fed.run()
    assert any(e.decision == "deny" for e in coord.audit.query(action="federated_query_governance"))


def test_layer_mismatch_is_refused():
    coord = _coordinator()
    acme = _org("acme", ACME_ROWS)
    # globex computes total_revenue a DIFFERENT way (revenue = price only) — a
    # different layer digest, so the federated metric is not computed one way.
    globex = _app("globex")
    globex.register_dataset(GLOBEX_ROWS, columns=COLS, name="sales")
    globex.semantic_layer(
        "sales",
        derived=[DerivedColumn(name="revenue", expression="price")],
        measures=[Measure(name="total_revenue", agg="sum", expression="revenue")],
        dimensions=DIMENSIONS,
    )
    q = FederatedQuery.of("total_revenue", table="sales", min_members=2)
    fed = coord.federated_data_engagement(query=q)
    fed.add_member("acme", acme)
    fed.add_member("globex", globex)
    with pytest.raises(DataError, match="layer"):
        fed.run()


# -- non-decomposable measures are refused at construction --------------------


@pytest.mark.parametrize("agg", ["avg"])
def test_non_decomposable_aggregation_refused(agg):
    coord = _app("c")
    coord.register_dataset(ACME_ROWS, columns=COLS, name="sales")
    coord.semantic_layer(
        "sales",
        derived=DERIVED,
        measures=[Measure(name="avg_rev", agg=agg, expression="revenue")],
        dimensions=DIMENSIONS,
    )
    q = FederatedQuery.of("avg_rev", table="sales", min_members=1)
    fed = coord.federated_data_engagement(query=q)
    fed.add_member("c", coord)
    with pytest.raises(DataError, match="not partition-decomposable"):
        fed.negotiate()


def test_ratio_measure_refused():
    coord = _app("c")
    coord.register_dataset(ACME_ROWS, columns=COLS, name="sales")
    coord.semantic_layer(
        "sales",
        derived=DERIVED,
        measures=[
            Measure(name="total_revenue", agg="sum", expression="revenue"),
            Measure(name="order_count", agg="count"),
            Measure(name="aov", numerator="total_revenue", denominator="order_count"),
        ],
        dimensions=DIMENSIONS,
    )
    q = FederatedQuery.of("aov", table="sales", min_members=1)
    fed = coord.federated_data_engagement(query=q)
    fed.add_member("c", coord)
    with pytest.raises(DataError, match="ratio"):
        fed.negotiate()


def test_unknown_metric_refused():
    coord, fed, _ = _federation(metrics="not_a_metric")
    with pytest.raises(DataError, match="not a governed measure"):
        fed.negotiate()


# -- tamper detection: a change anywhere is caught ----------------------------


def test_reordered_stage_breaks_the_chain():
    coord, fed, _ = _federation(metrics=["total_revenue", "order_count"])
    fed.run()
    narrative = fed.seal()
    forged = FederatedNarrative.from_wire(narrative.to_wire())
    forged.stages[1], forged.stages[2] = forged.stages[2], forged.stages[1]
    v = forged.verify()
    assert not v.valid and not v.intact


def test_edited_digest_is_caught_and_pinpointed():
    coord, fed, _ = _federation()
    fed.run()
    narrative = fed.seal()
    forged = FederatedNarrative.from_wire(narrative.to_wire())
    forged.stages[0].digest = "deadbeef"
    v = forged.verify()
    assert not v.valid and v.broken_at == 0


def test_forged_signature_fails_authentication():
    coord, fed, _ = _federation()
    fed.run()
    narrative = fed.seal()
    assert narrative.verify(coord.contract_signer).valid
    assert not narrative.verify(HMACSigner("stranger-secret", key_id="stranger")).valid


def test_tampered_reconciliation_breaks_data_binding():
    coord, fed, _ = _federation()
    fed.run()
    fed.seal()
    fed.findings[0].value = 999_999.0  # tamper the reconciled answer
    v = fed.verify(coord.contract_signer)
    assert v.data_bound is False and not v.valid


def test_tampered_source_breaks_data_binding():
    coord, fed, _ = _federation()
    fed.run()
    fed.seal()
    # Re-register one member's source with different bytes: the aggregate no longer
    # re-derives, so data-binding fails even though the chain is intact.
    acme = next(m for m in fed.members if m.org == "acme")
    acme.app.register_dataset(
        [{**r, "qty": r["qty"] + 100} for r in ACME_ROWS], columns=COLS, name="sales"
    )
    v = fed.verify(coord.contract_signer)
    assert v.intact and v.digests_ok
    assert v.data_bound is False and not v.valid


def test_require_valid_raises_on_tamper():
    coord, fed, _ = _federation()
    fed.run()
    narrative = fed.seal()
    narrative.require_valid(coord.contract_signer)  # ok
    narrative.stages[0].digest = "deadbeef"
    with pytest.raises(DataError):
        narrative.require_valid(coord.contract_signer, artifacts=list(fed._artifacts))


# -- contract binding ---------------------------------------------------------


def test_query_shape_is_bound_into_the_contract_scope():
    coord, fed, query = _federation()
    contract = fed.negotiate()
    assert contract.terms.scope == query.scope()
    assert query.digest() in contract.terms.scope
    assert contract.verify(coord.contract_signer).valid


def test_no_agreement_raises_by_default(monkeypatch):
    coord, fed, query = _federation()
    # Force a no-deal by an impossible buyer/seller overlap.
    from vincio.negotiation import buyer_position, seller_position

    with pytest.raises(DataError):
        fed.negotiate(
            buyer=buyer_position(max_price_usd=0.01, max_sla_seconds=5.0),
            seller=seller_position(min_price_usd=5.0, ideal_price_usd=9.0),
        )


# -- escape hatches & edge cases ----------------------------------------------


def test_record_stage_escape_hatch():
    coord, fed, _ = _federation()
    fed.run()
    fed.record_stage("note", {"comment": "reviewed"}, note="custom")
    narrative = fed.seal()
    assert "note" in narrative.stage_names
    assert narrative.verify(coord.contract_signer).valid


def test_reconcile_without_dispatch_raises():
    coord, fed, _ = _federation()
    fed.negotiate()
    with pytest.raises(DataError, match="nothing to reconcile"):
        fed.reconcile()


def test_dispatch_without_members_raises():
    coord = _coordinator()
    fed = coord.federated_data_engagement(query=FederatedQuery.of("total_revenue", table="sales"))
    with pytest.raises(DataError, match="at least one member"):
        fed.dispatch()


def test_engagement_without_query_raises():
    coord = _coordinator()
    fed = coord.federated_data_engagement()
    fed.add_member("acme", _org("acme", ACME_ROWS))
    with pytest.raises(DataError, match="no federated query"):
        fed.run()


def test_stages_property_is_a_copy():
    coord, fed, _ = _federation()
    fed.run()
    snapshot = fed.stages
    snapshot[0].digest = "mutated"
    assert fed.seal().verify(coord.contract_signer).valid


def test_print_summary_runs(capsys):
    coord, fed, _ = _federation()
    fed.run()
    fed.seal().print_summary()
    out = capsys.readouterr().out
    assert "Federated engagement" in out
    assert "total_revenue" in out


# -- surface ------------------------------------------------------------------


def test_public_surface_exports_present():
    import vincio

    for sym in (
        "FederatedQuery",
        "FederatedMember",
        "FederatedContribution",
        "FederatedFinding",
        "FederatedStage",
        "FederatedSignature",
        "FederatedVerification",
        "FederatedNarrative",
        "FederatedDataEngagement",
    ):
        assert sym in vincio.__all__
        assert hasattr(vincio, sym)


def test_finding_helpers():
    f = FederatedFinding(metric="m", op="sum", group={"region": "NA"}, value=3, members=["a", "b"])
    assert f.group_label == "region=NA"
    assert FederatedFinding(metric="m", op="sum").group_label == "*"
