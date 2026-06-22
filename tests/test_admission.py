"""Cross-org reputation-gated admission & progressive exposure.

Mapping a counterparty's earned standing — an imported ``PortableReputation`` or a
local ``ReputationLedger`` — to a bounded, offline-verifiable admission posture: a
maximum contract value (the exposure ceiling), a required escrow fraction, and an
SLA-strictness factor. A thin or low-trust standing is admitted on conservative terms
rather than refused, its ceiling ramps toward parity as settled, corroborated history
accrues, and a regression walks it back; every decision binds the standing it read and
the terms it set onto a content hash that recomputes from the bytes alone, and folds
into the existing negotiation / contracting path.
"""

from __future__ import annotations

import pytest

from vincio import (
    AdmissionConfig,
    AdmissionDecision,
    AdmissionPolicy,
    ContextApp,
    admit,
    attest_reputation,
    combine_attestations,
    settle_contract,
)
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms, buyer_position, seller_position
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement import Standing
from vincio.settlement.admission import ADMISSION_ACTION

ACME = HMACSigner("acme-key", key_id="acme")
GLOBEX = HMACSigner("globex-key", key_id="globex")


def _app(name: str = "buyer") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def _contract(price: float = 0.10, *, seller: str = "vendor", buyer: str = "acme") -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope="work", price_usd=price)
    ).seal()


def _records(seller: str = "vendor", *, settled: int = 2, breached: int = 0):
    """Build settled / breached records with ``seller`` as the delivering party."""
    out = []
    for _ in range(settled):
        out.append(settle_contract(_contract(seller=seller), cost_usd=0.05))
    for _ in range(breached):
        out.append(settle_contract(_contract(0.04, seller=seller), cost_usd=0.09))
    return out


def _ledger_app(member: str = "vendor", *, passed: int = 0, failed: int = 0) -> ContextApp:
    app = _app()
    app.use_reputation_ledger()
    for _ in range(passed):
        app.reputation_ledger.record_outcome(member, passed=True, round_id="r")
    for _ in range(failed):
        app.reputation_ledger.record_outcome(member, passed=False, round_id="r")
    return app


# -- config validation --------------------------------------------------------


def test_default_config_is_coherent():
    AdmissionConfig().validate_coherent()


@pytest.mark.parametrize(
    "field, value",
    [
        ("parity_exposure_usd", 0.0),
        ("parity_exposure_usd", -1.0),
        ("floor_fraction", -0.1),
        ("floor_fraction", 1.1),
        ("full_trust_evidence", 0.0),
        ("ramp_floor", -0.1),
        ("ramp_floor", 1.5),
        ("max_escrow_fraction", -0.1),
        ("max_escrow_fraction", 1.1),
        ("min_sla_factor", 0.0),
        ("min_sla_factor", 1.1),
    ],
)
def test_incoherent_config_raises(field, value):
    cfg = AdmissionConfig(**{field: value})
    with pytest.raises(SettlementError):
        cfg.validate_coherent()


def test_policy_validates_its_config_on_construction():
    with pytest.raises(SettlementError):
        AdmissionPolicy(AdmissionConfig(parity_exposure_usd=0.0))


# -- the graduated-exposure map ----------------------------------------------


def test_ramp_climbs_from_floor_to_one():
    cfg = AdmissionConfig(ramp_floor=0.2, full_trust_evidence=10.0)
    assert cfg.ramp_progress(0.0) == pytest.approx(0.2)
    assert cfg.ramp_progress(5.0) == pytest.approx(0.6)
    assert cfg.ramp_progress(10.0) == pytest.approx(1.0)


def test_ramp_saturates_at_parity_beyond_threshold():
    cfg = AdmissionConfig(ramp_floor=0.2, full_trust_evidence=10.0)
    # Evidence past the threshold cannot push the ramp past 1.
    assert cfg.ramp_progress(100.0) == pytest.approx(1.0)


def test_exposure_fraction_is_bounded_in_band():
    cfg = AdmissionConfig(floor_fraction=0.1)
    # Worst case (zero reputation, zero history) sits at the floor, never below.
    assert cfg.exposure_fraction(0.0, 0.0) == pytest.approx(0.1)
    # Best case (full reputation, saturated history) reaches parity, never above.
    assert cfg.exposure_fraction(1.0, 1000.0) == pytest.approx(1.0)


def test_exposure_fraction_monotonic_in_reputation():
    cfg = AdmissionConfig()
    low = cfg.exposure_fraction(0.2, 10.0)
    high = cfg.exposure_fraction(0.9, 10.0)
    assert high > low


def test_exposure_fraction_monotonic_in_evidence():
    cfg = AdmissionConfig()
    thin = cfg.exposure_fraction(0.8, 1.0)
    thick = cfg.exposure_fraction(0.8, 10.0)
    assert thick > thin


def test_terms_track_exposure():
    cfg = AdmissionConfig(parity_exposure_usd=1000.0, max_escrow_fraction=0.5, min_sla_factor=0.5)
    terms = cfg.terms_for(1.0, 1000.0)  # at parity
    assert terms["exposure_fraction"] == pytest.approx(1.0)
    assert terms["max_contract_value_usd"] == pytest.approx(1000.0)
    assert terms["escrow_fraction"] == pytest.approx(0.0)
    assert terms["sla_factor"] == pytest.approx(1.0)


def test_escrow_is_inverse_to_exposure():
    cfg = AdmissionConfig(max_escrow_fraction=0.5)
    thin = cfg.terms_for(0.1, 0.0)
    thick = cfg.terms_for(0.95, 100.0)
    assert thin["escrow_fraction"] > thick["escrow_fraction"]
    assert thin["sla_factor"] < thick["sla_factor"]


# -- deciding -----------------------------------------------------------------


def test_brand_new_counterparty_is_admitted_conservatively_not_refused():
    from vincio.settlement import AttestationConfig

    decision = admit("stranger")
    assert decision.max_contract_value_usd > 0.0  # admitted, never a hard gate
    assert decision.exposure_fraction >= AdmissionConfig().floor_fraction
    assert not decision.at_parity
    assert decision.verify().valid
    # The recorded standing is the benefit-of-the-doubt prior — no evidence, no issuers.
    prior_rep = round(AttestationConfig().reputation_of(0.0, 0.0), 9)
    assert decision.standing.reputation == pytest.approx(prior_rep)
    assert decision.standing.evidence == 0.0
    assert decision.standing.issuers == []


def test_higher_standing_earns_a_higher_ceiling():
    good = admit("good", ledger=_ledger_app("good", passed=12).reputation_ledger)
    bad = admit("bad", ledger=_ledger_app("bad", failed=12).reputation_ledger)
    assert good.max_contract_value_usd > bad.max_contract_value_usd


def test_regression_walks_the_ceiling_back():
    app = _ledger_app("vendor", passed=12)
    before = app.admit("vendor")
    for _ in range(20):
        app.reputation_ledger.record_outcome("vendor", passed=False, round_id="r")
    after = app.admit("vendor")
    assert after.max_contract_value_usd < before.max_contract_value_usd


def test_ceiling_never_exceeds_parity():
    app = _ledger_app("vendor", passed=200)
    decision = app.admit("vendor")
    assert decision.max_contract_value_usd <= AdmissionConfig().parity_exposure_usd + 1e-6


def test_decision_records_the_standing_it_read():
    ledger = _ledger_app("vendor", passed=4, failed=1).reputation_ledger
    decision = admit("vendor", ledger=ledger)
    assert decision.standing.evidence == pytest.approx(5.0)
    assert decision.standing.reputation == pytest.approx(ledger.reputation("vendor"))
    assert decision.standing.weight == pytest.approx(ledger.weight("vendor"))
    assert decision.standing.issuers == []  # first-hand, not attested


def test_custom_config_sets_the_parity_ceiling():
    decision = admit("vendor", config=AdmissionConfig(parity_exposure_usd=50_000.0))
    assert decision.max_contract_value_usd <= 50_000.0
    assert decision.config.parity_exposure_usd == 50_000.0


# -- offline verification -----------------------------------------------------


def test_sealed_decision_verifies_offline():
    decision = admit("vendor", ledger=_ledger_app("vendor", passed=6).reputation_ledger)
    result = decision.verify()
    assert result.valid
    assert result.hash_ok
    assert result.terms_sound


def test_unsealed_decision_is_invalid():
    decision = AdmissionDecision(
        subject="vendor", standing=Standing(reputation=0.7), config=AdmissionConfig()
    )
    result = decision.verify()
    assert not result.valid
    assert not result.hash_ok
    assert "not sealed" in (result.reason or "")


def test_tampered_ceiling_caught_even_after_reseal():
    decision = admit("vendor", ledger=_ledger_app("vendor", passed=6).reputation_ledger)
    decision.max_contract_value_usd = 999_999.0
    decision.seal()  # recompute the hash to match the tampered ceiling
    result = decision.verify()
    assert result.hash_ok  # the hash matches the tampered fields
    assert not result.terms_sound  # but the terms no longer re-derive
    assert not result.valid


def test_tampered_standing_without_reseal_breaks_the_hash():
    decision = admit("vendor", ledger=_ledger_app("vendor", passed=6).reputation_ledger)
    decision.standing.reputation = 0.99  # lie about the standing, do not reseal
    result = decision.verify()
    assert not result.hash_ok
    assert not result.valid


def test_require_valid_raises_on_tamper():
    decision = admit("vendor", ledger=_ledger_app("vendor", passed=6).reputation_ledger)
    decision.escrow_fraction = 0.0
    decision.seal()
    with pytest.raises(SettlementError):
        decision.require_valid()


def test_require_valid_returns_self_when_sound():
    decision = admit("vendor", ledger=_ledger_app("vendor", passed=6).reputation_ledger)
    assert decision.require_valid() is decision


def test_wire_roundtrip_preserves_verification():
    decision = admit("vendor", ledger=_ledger_app("vendor", passed=6, failed=2).reputation_ledger)
    restored = AdmissionDecision.from_wire(decision.to_wire())
    assert restored.content_hash == decision.content_hash
    assert restored.verify().valid
    assert restored.max_contract_value_usd == decision.max_contract_value_usd


def test_same_standing_and_policy_hash_identically():
    standing = Standing(weight=0.7, reputation=0.8, evidence=5.0, issuers=["acme"])
    a = AdmissionPolicy().decide("vendor", standing)
    b = AdmissionPolicy().decide("vendor", standing)
    assert a.content_hash == b.content_hash


# -- folding into negotiation -------------------------------------------------


def test_bound_position_clamps_price_reserve_to_ceiling():
    decision = admit("vendor")  # conservative new-counterparty ceiling
    position = buyer_position(max_price_usd=5_000.0, ideal_price_usd=10.0, max_sla_seconds=10.0)
    bounded = decision.bound_position(position)
    price = next(i for i in bounded.issues if i.name == "price_usd")
    assert price.reserve == pytest.approx(decision.max_contract_value_usd)


def test_bound_position_tightens_sla_and_stays_coherent():
    decision = admit("vendor")
    position = buyer_position(max_price_usd=5_000.0, ideal_price_usd=10.0, max_sla_seconds=10.0)
    bounded = decision.bound_position(position).validate_coherent()
    sla = next(i for i in bounded.issues if i.name == "sla_seconds")
    assert sla.reserve == pytest.approx(round(10.0 * decision.sla_factor, 9))


def test_bound_position_does_not_mutate_the_original():
    decision = admit("vendor")
    position = buyer_position(max_price_usd=5_000.0, ideal_price_usd=10.0, max_sla_seconds=10.0)
    decision.bound_position(position)
    price = next(i for i in position.issues if i.name == "price_usd")
    assert price.reserve == 5_000.0  # untouched


def test_bound_position_leaves_a_position_within_ceiling_unchanged():
    decision = admit("vendor", ledger=_ledger_app("vendor", passed=200).reputation_ledger)
    # A position whose price walk-away is already under the ceiling is not raised.
    position = buyer_position(max_price_usd=5.0, ideal_price_usd=1.0, max_sla_seconds=100.0)
    bounded = decision.bound_position(position)
    price = next(i for i in bounded.issues if i.name == "price_usd")
    assert price.reserve == 5.0


def test_admit_then_negotiate_converges_within_the_ceiling():
    app = _ledger_app("vendor", passed=0, failed=8)  # low standing → tight ceiling
    decision = app.admit("vendor", config=AdmissionConfig(parity_exposure_usd=1.0))
    buyer = decision.bound_position(
        buyer_position(max_price_usd=10.0, ideal_price_usd=0.01, max_sla_seconds=5.0)
    )
    result = app.negotiate(
        "transcribe",
        buyer=buyer,
        seller=seller_position(min_price_usd=0.01, ideal_price_usd=0.05),
        seller_id="vendor",
    )
    if result.agreed:
        assert result.contract.terms.price_usd <= decision.max_contract_value_usd + 1e-9


# -- folding into contracting -------------------------------------------------


def test_apply_to_terms_caps_price_and_stamps_metadata():
    decision = admit("vendor")
    terms = ContractTerms(scope="work", price_usd=10_000.0, sla_seconds=10.0)
    capped = decision.apply_to_terms(terms)
    assert capped.price_usd == pytest.approx(decision.max_contract_value_usd)
    assert capped.sla_seconds == pytest.approx(round(10.0 * decision.sla_factor, 9))
    stamp = capped.metadata["admission"]
    assert stamp["decision_id"] == decision.id
    assert stamp["escrow_fraction"] == pytest.approx(decision.escrow_fraction)


def test_apply_to_terms_keeps_the_contract_hash_independent_of_the_stamp():
    decision = admit("vendor")
    terms = ContractTerms(scope="work", price_usd=1.0, sla_seconds=5.0)
    capped = decision.apply_to_terms(terms)
    contract = Contract(buyer="acme", seller="vendor", terms=capped).seal()
    stamped_hash = contract.content_hash
    # The admission stamp lives in metadata, which is dropped from the canonical hash —
    # so a contract minted from the capped terms hashes exactly as the bare terms would.
    bare = ContractTerms(scope="work", price_usd=capped.price_usd, sla_seconds=capped.sla_seconds)
    bare_hash = Contract(
        buyer="acme", seller="vendor", terms=bare, agreed_at=contract.agreed_at
    ).compute_hash()
    assert stamped_hash == bare_hash
    assert contract.terms.metadata["admission"]["subject"] == "vendor"


def test_apply_to_terms_does_not_mutate_the_original():
    decision = admit("vendor")
    terms = ContractTerms(scope="work", price_usd=10_000.0)
    decision.apply_to_terms(terms)
    assert terms.price_usd == 10_000.0
    assert "admission" not in terms.metadata


# -- portable reputation source ----------------------------------------------


def test_admit_reads_corroborating_issuers_from_a_portable_prior():
    acme_att = attest_reputation(_records("vendor", settled=4), "vendor", issuer="acme").sign(ACME)
    globex_att = attest_reputation(_records("vendor", settled=3), "vendor", issuer="globex").sign(
        GLOBEX
    )
    prior = combine_attestations([acme_att, globex_att])
    decision = admit("vendor", reputation=prior)
    assert decision.standing.issuers == ["acme", "globex"]
    assert decision.standing.evidence == pytest.approx(7.0)
    assert decision.verify().valid


def test_thin_attestation_admits_more_conservatively_than_corroborated_one():
    thin = combine_attestations(
        [attest_reputation(_records("vendor", settled=1), "vendor", issuer="acme").sign(ACME)]
    )
    thick = combine_attestations(
        [
            attest_reputation(_records("vendor", settled=8), "vendor", issuer="acme").sign(ACME),
            attest_reputation(_records("vendor", settled=8), "vendor", issuer="globex").sign(
                GLOBEX
            ),
        ]
    )
    thin_d = admit("vendor", reputation=thin)
    thick_d = admit("vendor", reputation=thick)
    assert thick_d.max_contract_value_usd > thin_d.max_contract_value_usd


# -- app surface --------------------------------------------------------------


def test_app_admit_records_on_the_audit_chain():
    app = _ledger_app("vendor", passed=6)
    decision = app.admit("vendor")
    assert decision.audit_id is not None
    entries = app.audit.query(action=ADMISSION_ACTION)
    assert len(entries) == 1
    assert app.audit.verify_chain()
    assert decision.verify().valid


def test_app_admit_can_skip_the_audit_record():
    app = _ledger_app("vendor", passed=6)
    decision = app.admit("vendor", record_audit=False)
    assert decision.audit_id is None
    assert app.audit.query(action=ADMISSION_ACTION) == []


def test_app_admit_prefers_imported_reputation_over_local_ledger():
    app = _app()
    app.use_reputation_ledger()
    # No local history for the vendor; an imported, corroborated prior decides.
    att = attest_reputation(_records("vendor", settled=8), "vendor", issuer="acme").sign(ACME)
    app.import_reputation([att])
    decision = app.admit("vendor")
    assert decision.standing.issuers == ["acme"]
    assert decision.standing.evidence == pytest.approx(8.0)


def test_local_first_hand_evidence_wins_over_attested_standing():
    # Other orgs attest a glowing standing, but the importer has lived through breaches.
    glowing = attest_reputation(_records("vendor", settled=10), "vendor", issuer="acme").sign(ACME)
    app = _app()
    app.use_reputation_ledger()
    for _ in range(15):
        app.reputation_ledger.record_outcome("vendor", passed=False, round_id="r")
    prior = app.import_reputation([glowing])
    decision = admit("vendor", reputation=prior)
    # The decision reads the local regression, not the attested standing — issuers empty.
    assert decision.standing.issuers == []
    assert decision.standing.reputation == pytest.approx(app.reputation_ledger.reputation("vendor"))
    # A regression walks the ceiling well below what the glowing attestation would set.
    attested_only = admit("vendor", reputation=combine_attestations([glowing]))
    assert decision.max_contract_value_usd < attested_only.max_contract_value_usd


def test_app_admit_falls_back_to_local_ledger_when_no_import():
    app = _ledger_app("vendor", passed=5)
    decision = app.admit("vendor")
    assert decision.standing.evidence == pytest.approx(5.0)
    assert decision.standing.issuers == []


def test_app_admit_with_no_reputation_source_uses_the_prior():
    app = _app()  # no ledger, no import
    decision = app.admit("stranger")
    assert decision.max_contract_value_usd > 0.0
    assert not decision.at_parity


def test_app_admit_decision_marked_parity_or_graduated_in_audit():
    app = _ledger_app("vendor", passed=300)
    app.admit("vendor")
    entry = app.audit.query(action=ADMISSION_ACTION)[-1]
    assert entry.decision in {"parity", "graduated"}


# -- determinism --------------------------------------------------------------


def test_admit_is_deterministic_for_the_same_standing():
    ledger = _ledger_app("vendor", passed=6, failed=2).reputation_ledger
    a = admit("vendor", ledger=ledger)
    b = admit("vendor", ledger=ledger)
    assert a.decision_facts() == b.decision_facts()
    assert a.content_hash == b.content_hash
