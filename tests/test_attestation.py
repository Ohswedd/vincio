"""Cross-org reputation attestation & portability.

Issuing a signed, offline-verifiable attestation over a counterparty's earned
standing — derived from an org's own settlement records and arbitration
resolutions — and combining several issuers' attestations into a bounded,
evidence-weighted prior that weights the next negotiation: a tampered score or a
forged issuer is caught, a self-attestation is refused, and an unknown counterparty
is weighted by what its past counterparties attest under the same ``[floor, 1]``
rule a local reputation is.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from vincio import (
    AttestationRevocation,
    ContextApp,
    PortableReputation,
    ReputationAttestation,
    attest_reputation,
    combine_attestations,
    revoke_attestation,
    settle_contract,
)
from vincio.core.errors import SettlementError
from vincio.core.utils import utcnow
from vincio.negotiation import (
    Contract,
    ContractTerms,
    buyer_position,
    select_offer,
    seller_position,
)
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement import AttestationConfig, SettlementBook
from vincio.settlement.attestation import ATTESTATION_ACTION, REVOCATION_ACTION
from vincio.settlement.attestation import attest_reputation as _attest

ACME = HMACSigner("acme-key", key_id="acme")
GLOBEX = HMACSigner("globex-key", key_id="globex")


def _app(name: str = "issuer") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def _contract(price: float = 0.10, *, seller: str = "vendor", buyer: str = "acme") -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope="work", price_usd=price)
    ).seal()


def _records(seller: str = "vendor", *, settled: int = 2, breached: int = 0):
    """Build settled / breached records with ``seller`` as the delivering party."""
    out = []
    for _ in range(settled):
        out.append(settle_contract(_contract(seller=seller), cost_usd=0.05))  # within price
    for _ in range(breached):
        out.append(settle_contract(_contract(0.04, seller=seller), cost_usd=0.09))  # overrun
    return out


# -- issuing ------------------------------------------------------------------


def test_attest_counts_settled_and_breached_as_seller():
    att = attest_reputation(_records(settled=3, breached=1), "vendor", issuer="acme")
    assert att.settled == 3
    assert att.breached == 1
    assert att.dissents == 0
    assert att.successes == 3
    assert att.failures == 1
    assert att.evidence == 4
    assert att.settlements == 4


def test_attest_reputation_is_posterior_mean():
    att = attest_reputation(_records(settled=2, breached=0), "vendor", issuer="acme")
    cfg = AttestationConfig()
    assert att.reputation == round(cfg.reputation_of(2, 0), 9)


def test_attest_ignores_records_where_subject_is_not_seller():
    recs = _records("vendor", settled=2) + _records("other", settled=5)
    att = attest_reputation(recs, "vendor", issuer="acme")
    assert att.settled == 2


def test_attest_empty_history_raises():
    with pytest.raises(SettlementError):
        attest_reputation(_records("other", settled=2), "vendor", issuer="acme")


def test_attest_skips_a_tampered_own_record():
    recs = _records("vendor", settled=2)
    recs[0].balance_usd = 999.0  # tamper without resealing
    att = attest_reputation(recs, "vendor", issuer="acme")
    assert att.settled == 1  # the tampered record is not counted


def test_attest_binds_source_hashes():
    recs = _records("vendor", settled=2)
    att = attest_reputation(recs, "vendor", issuer="acme")
    assert att.source_hashes == sorted(r.content_hash for r in recs)


def test_attest_counts_arbitration_dissents_as_failures():
    app = _app()
    app.use_reputation_ledger()
    c = _contract()
    acme_rec = settle_contract(c, cost_usd=0.08).sign(ACME, party="acme")
    vendor_agrees = settle_contract(c, cost_usd=0.08).sign(GLOBEX, party="vendor")
    liar = settle_contract(c, cost_usd=0.05).sign(GLOBEX, party="vendor")
    resolution = app.arbitrate([acme_rec, vendor_agrees, liar])
    assert resolution.dissenters == ["vendor"]
    att = attest_reputation(
        _records("vendor", settled=1), "vendor", issuer="acme", resolutions=[resolution]
    )
    assert att.dissents == 1
    assert att.failures == 1


def test_attest_ignores_a_tampered_resolution_dissent():
    app = _app()
    app.use_reputation_ledger()
    c = _contract()
    acme_rec = settle_contract(c, cost_usd=0.08).sign(ACME, party="acme")
    vendor_agrees = settle_contract(c, cost_usd=0.08).sign(GLOBEX, party="vendor")
    liar = settle_contract(c, cost_usd=0.05).sign(GLOBEX, party="vendor")
    resolution = app.arbitrate([acme_rec, vendor_agrees, liar])
    assert resolution.dissenters == ["vendor"]
    resolution.upheld_balance_usd = 999.0  # tamper the upheld figure and re-seal
    resolution.seal()
    assert not resolution.verify().decision_sound
    att = attest_reputation(
        _records("vendor", settled=1), "vendor", issuer="acme", resolutions=[resolution]
    )
    assert att.dissents == 0  # a tampered resolution cannot inflate dissents


# -- offline verification -----------------------------------------------------


def test_signed_attestation_verifies_offline():
    att = attest_reputation(_records(), "vendor", issuer="acme").sign(ACME)
    result = att.verify(ACME)
    assert result.valid
    assert result.hash_ok
    assert result.evidence_sound
    assert result.signed_by == ["acme"]


def test_tampered_score_breaks_the_hash():
    att = attest_reputation(_records(), "vendor", issuer="acme").sign(ACME)
    att.reputation = 0.99
    result = att.verify(ACME)
    assert not result.hash_ok
    assert not result.valid


def test_tampered_score_after_reseal_caught_by_evidence():
    att = attest_reputation(_records(settled=2), "vendor", issuer="acme")
    att.reputation = 0.99
    att.seal()  # recompute the hash to match the tampered score
    result = att.verify()
    assert result.hash_ok
    assert not result.evidence_sound  # but the score no longer re-derives
    assert not result.valid


def test_forged_signature_is_caught():
    att = attest_reputation(_records(), "vendor", issuer="acme").sign(ACME)
    att.signatures[0].signature = "deadbeef"
    assert not att.verify(ACME).signatures_ok


def test_verify_binding_only_with_empty_require():
    att = attest_reputation(_records(), "vendor", issuer="acme")
    result = att.verify(require=[])
    assert result.hash_ok
    assert result.evidence_sound


def test_require_valid_raises_on_tamper():
    att = attest_reputation(_records(), "vendor", issuer="acme").sign(ACME)
    att.reputation = 0.99
    with pytest.raises(SettlementError):
        att.require_valid(ACME)


def test_wire_roundtrip_preserves_verification():
    att = attest_reputation(_records(), "vendor", issuer="acme").sign(ACME)
    back = ReputationAttestation.from_wire(att.to_wire())
    assert back.verify(ACME).valid
    assert back.content_hash == att.content_hash


def test_two_issuers_same_evidence_distinct_attestations():
    recs = _records(settled=2)
    a = attest_reputation(recs, "vendor", issuer="acme")
    b = attest_reputation(recs, "vendor", issuer="globex")
    # The issuer is bound, so the same evidence from two issuers is two artifacts.
    assert a.content_hash != b.content_hash
    assert a.reputation == b.reputation


# -- combining ----------------------------------------------------------------


def test_combine_pools_evidence_across_issuers():
    a = attest_reputation(_records(settled=2), "vendor", issuer="acme").sign(ACME)
    b = attest_reputation(_records(settled=2), "vendor", issuer="globex").sign(GLOBEX)
    prior = combine_attestations([a, b])
    standing = prior.standing("vendor")
    assert standing.successes == 4  # 2 + 2 pooled
    assert standing.failures == 0
    assert standing.attestations == 2
    assert standing.issuers == ["acme", "globex"]


def test_combine_weight_is_bounded():
    # A regressor with many breaches is discounted, never zeroed.
    bad = attest_reputation(_records(settled=0, breached=6), "vendor", issuer="acme").sign(ACME)
    prior = combine_attestations([bad])
    weight = prior.weight("vendor")
    assert 0.1 <= weight < 1.0  # floor holds; discounted, not zeroed


def test_combine_more_positive_evidence_raises_weight():
    thin = combine_attestations(
        [attest_reputation(_records(settled=1), "vendor", issuer="acme").sign(ACME)]
    )
    thick = combine_attestations(
        [
            attest_reputation(_records(settled=8), "vendor", issuer="acme").sign(ACME),
            attest_reputation(_records(settled=8), "vendor", issuer="globex").sign(GLOBEX),
        ]
    )
    assert thick.weight("vendor") > thin.weight("vendor")


def test_unknown_counterparty_gets_prior_weight():
    prior = combine_attestations([])
    cfg = AttestationConfig()
    assert prior.weight("stranger") == cfg.weight_of(cfg.reputation_of(0, 0))


def test_self_attestation_is_refused():
    self_att = attest_reputation(_records("vendor"), "vendor", issuer="vendor").sign(GLOBEX)
    prior = combine_attestations([self_att])
    verdict = prior.verdict_for("vendor", "vendor")
    assert not verdict.counted
    assert "self-attestation" in verdict.reason
    assert prior.standing("vendor") is None  # nothing counted


def test_self_attestation_allowed_when_opted_in():
    self_att = attest_reputation(_records("vendor"), "vendor", issuer="vendor").sign(GLOBEX)
    prior = combine_attestations([self_att], allow_self=True)
    assert prior.verdict_for("vendor", "vendor").counted


def test_tampered_attestation_is_pinpointed_not_dropped():
    good = attest_reputation(_records(settled=2), "vendor", issuer="acme").sign(ACME)
    bad = attest_reputation(_records(settled=2), "vendor", issuer="globex").sign(GLOBEX)
    bad.reputation = 0.99  # tamper without resealing
    prior = combine_attestations([good, bad])
    assert len(prior.refused) == 1
    assert prior.refused[0].issuer == "globex"
    assert "tampered" in prior.refused[0].reason
    assert prior.standing("vendor").attestations == 1  # only the good one counted


def test_forged_attestation_refused_with_verifier():
    good = attest_reputation(_records(settled=2), "vendor", issuer="acme").sign(ACME)
    forged = attest_reputation(_records(settled=2), "vendor", issuer="globex").sign(GLOBEX)
    forged.signatures[0].signature = "deadbeef"
    prior = combine_attestations([good, forged], verify_with=ACME)
    assert any("forged" in (v.reason or "") for v in prior.refused)


def test_issuer_cannot_stack_attestations():
    small = attest_reputation(_records(settled=2), "vendor", issuer="acme").sign(ACME)
    big = attest_reputation(_records(settled=6), "vendor", issuer="acme").sign(ACME)
    prior = combine_attestations([small, big])
    standing = prior.standing("vendor")
    assert standing.successes == 6  # only the larger attestation from acme is counted
    assert standing.attestations == 1
    assert any("superseded" in (v.reason or "") for v in prior.excluded)


def test_per_issuer_cap_bounds_one_issuer_mass():
    whale = attest_reputation(_records(settled=100), "vendor", issuer="acme").sign(ACME)
    cfg = AttestationConfig(per_issuer_cap=5.0)
    prior = combine_attestations([whale], config=cfg)
    assert prior.standing("vendor").evidence <= 5.0 + 1e-9


def test_two_importers_compute_the_same_standing():
    atts = [
        attest_reputation(_records(settled=3, breached=1), "vendor", issuer="acme").sign(ACME),
        attest_reputation(_records(settled=2), "vendor", issuer="globex").sign(GLOBEX),
    ]
    a = combine_attestations(atts)
    b = combine_attestations(list(reversed(atts)))
    assert a.weight("vendor") == b.weight("vendor")
    assert a.standing("vendor").successes == b.standing("vendor").successes


def test_combine_subject_filter():
    atts = [
        attest_reputation(_records("vendor"), "vendor", issuer="acme").sign(ACME),
        attest_reputation(_records("other"), "other", issuer="acme").sign(ACME),
    ]
    prior = combine_attestations(atts, subject="vendor")
    assert prior.subjects() == ["vendor"]


# -- base ledger fallthrough --------------------------------------------------


def test_local_evidence_wins_over_imported_prior():
    app = _app()
    ledger = app.use_reputation_ledger()
    # Local history: vendor regressed hard locally.
    for _ in range(5):
        ledger.record_outcome("vendor", passed=False, round_id="c")
    good = attest_reputation(_records(settled=8), "vendor", issuer="acme").sign(ACME)
    prior = combine_attestations([good], base=ledger)
    # The importer trusts what it lived through over what others attest.
    assert prior.weight("vendor") == ledger.weight("vendor")


def test_imported_prior_used_for_unknown_counterparty():
    app = _app()
    ledger = app.use_reputation_ledger()  # empty — vendor is unknown locally
    good = attest_reputation(_records(settled=8), "vendor", issuer="acme").sign(ACME)
    prior = combine_attestations([good], base=ledger)
    assert prior.weight("vendor") == prior.config.weight_of(prior.reputation("vendor"))


# -- book & app surface -------------------------------------------------------


def test_settlement_book_attest_signs_as_owner():
    signer = HMACSigner("book-key", key_id="acme")
    book = SettlementBook("acme", signer=signer)
    book.settle(_contract(seller="vendor"), cost_usd=0.05)
    book.settle(_contract(seller="vendor"), cost_usd=0.05)
    att = book.attest("vendor")
    assert att.issuer == "acme"
    assert att.signed_by == ["acme"]
    assert att.verify(signer).valid


def test_app_attest_reputation_signs_as_issuer_for_a_differently_owned_book():
    # A book owned by a different org with no signer: the app must sign as the issuer
    # (the book owner) so the attestation verifies against its own require=[issuer].
    app = _app("app-host")
    book = SettlementBook("org-a")  # owner != app.name, no signer
    book.settle(_contract(buyer="org-a", seller="vendor"), cost_usd=0.05, party="org-a")
    book.settle(_contract(buyer="org-a", seller="vendor"), cost_usd=0.05, party="org-a")
    att = app.attest_reputation("vendor", book=book)
    assert att.issuer == "org-a"
    assert att.signed_by == ["org-a"]
    assert att.verify(app.contract_signer).valid


def test_app_attest_reputation_audits_and_verifies():
    app = _app("acme")
    app.use_settlement_book()
    app.settle(_contract(seller="vendor"), cost_usd=0.05)
    app.settle(_contract(seller="vendor"), cost_usd=0.05)
    att = app.attest_reputation("vendor")
    assert att.verify(app.contract_signer).valid
    assert app.audit.query(action=ATTESTATION_ACTION)
    assert app.audit.verify_chain()


def test_app_import_reputation_weights_negotiation():
    # Two issuers attest a regressor; a buyer with no local history imports the prior.
    a_app = _app("acme")
    a_app.use_settlement_book()
    for _ in range(4):
        a_app.settle(_contract(0.04, seller="vendor"), cost_usd=0.09)  # breaches
    att_a = a_app.attest_reputation("vendor")

    buyer = _app("buyer")
    buyer.use_reputation_ledger()
    prior = buyer.import_reputation([att_a])
    assert buyer._reputation_view() is prior
    assert prior.weight("vendor") < 1.0


def test_import_reputation_select_offer_prefers_reliable_seller():
    # A reliable seller and a regressor offer the same terms; the prior breaks the tie.
    good_app = _app("acme")
    good_app.use_settlement_book()
    for _ in range(5):
        good_app.settle(_contract(seller="reliable"), cost_usd=0.05)
    bad_app = _app("acme2")
    bad_app.use_settlement_book()
    for _ in range(5):
        bad_app.settle(_contract(0.04, seller="flaky"), cost_usd=0.09)
    prior = combine_attestations(
        [good_app.attest_reputation("reliable"), bad_app.attest_reputation("flaky")]
    )

    pos = buyer_position(max_price_usd=0.10, max_sla_seconds=5.0)
    app = _app("buyer")
    reliable = app.negotiate(
        "work",
        buyer=pos,
        seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.12),
        buyer_id="buyer",
        seller_id="reliable",
    )
    flaky = app.negotiate(
        "work",
        buyer=pos,
        seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.12),
        buyer_id="buyer",
        seller_id="flaky",
    )
    chosen = select_offer([reliable, flaky], pos, reputation=prior)
    assert chosen is not None and chosen.seller == "reliable"


def test_import_reputation_refuses_self_attestation_in_prior():
    app = _app("vendor")
    app.use_settlement_book(owner="vendor")
    # vendor's own book where it sold to others, attesting itself.
    app.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05)
    self_att = app.attest_reputation("vendor")  # issuer == subject
    importer = _app("buyer")
    prior = importer.import_reputation([self_att])
    assert prior.standing("vendor") is None


def test_attest_reputation_uses_imported_alias():
    # The module-level function and the re-exported name are the same object.
    assert _attest is attest_reputation


def test_portable_reputation_weight_protocol_duck_types():
    prior = combine_attestations(
        [attest_reputation(_records(settled=3), "vendor", issuer="acme").sign(ACME)]
    )
    assert isinstance(prior, PortableReputation)
    assert isinstance(prior.weight("vendor"), float)
    assert isinstance(prior.reputation("vendor"), float)


# -- freshness: validity window & decay ---------------------------------------


def _aged(att: ReputationAttestation, days: float, signer: HMACSigner = ACME):
    """Backdate an attestation's issuance by ``days`` and re-seal / re-sign it."""
    att.issued_at = utcnow() - timedelta(days=days)
    att.seal()
    return att.sign(signer)


def test_horizon_attestation_still_verifies_offline():
    att = attest_reputation(_records(), "vendor", issuer="acme", horizon_days=30).sign(ACME)
    assert att.verify(ACME).valid
    assert att.horizon_days == 30
    assert att.expires_at is not None


def test_horizon_is_bound_into_the_hash():
    plain = attest_reputation(_records(settled=2), "vendor", issuer="acme")
    windowed = attest_reputation(_records(settled=2), "vendor", issuer="acme", horizon_days=30)
    # The validity window is a signed claim, so it changes the content hash.
    assert plain.content_hash != windowed.content_hash


def test_no_horizon_hash_is_unchanged_from_before_freshness():
    # A no-horizon attestation must hash exactly as it did pre-3.30 (horizon excluded
    # from the bound facts when None), so an already-issued attestation stays verifiable.
    att = attest_reputation(_records(settled=2), "vendor", issuer="acme")
    facts = att.attestation_facts()
    assert "horizon_days" not in facts


def test_no_horizon_attestation_never_stale():
    att = attest_reputation(_records(), "vendor", issuer="acme").sign(ACME)
    assert not att.is_stale(utcnow() + timedelta(days=3650))


def test_is_stale_past_the_window():
    att = _aged(attest_reputation(_records(), "vendor", issuer="acme", horizon_days=30), 40)
    assert att.is_stale(utcnow())
    assert not att.is_stale(att.issued_at + timedelta(days=10))


def test_combine_excludes_a_stale_attestation_pinpointed():
    fresh = attest_reputation(_records(settled=2), "vendor", issuer="globex", horizon_days=30)
    fresh.sign(GLOBEX)
    old = _aged(
        attest_reputation(_records(settled=2), "vendor", issuer="acme", horizon_days=30), 90
    )
    prior = combine_attestations([fresh, old], as_of=utcnow())
    assert len(prior.stale) == 1
    assert prior.stale[0].issuer == "acme"
    assert "stale" in prior.stale[0].reason
    # Only the fresh attestation's evidence is pooled.
    assert prior.standing("vendor").attestations == 1
    assert prior.standing("vendor").issuers == ["globex"]


def test_no_as_of_clock_means_no_expiry():
    old = _aged(attest_reputation(_records(settled=2), "vendor", issuer="acme", horizon_days=1), 90)
    prior = combine_attestations([old])  # no as_of → point-in-time, nothing expires
    assert prior.standing("vendor").attestations == 1
    assert not prior.stale


def test_half_life_decays_evidence_by_age():
    cfg = AttestationConfig(half_life_days=30)
    old = _aged(attest_reputation(_records(settled=8), "vendor", issuer="acme"), 30)
    prior = combine_attestations([old], config=cfg, as_of=utcnow())
    # One half-life halves the mass: 8 successes → ~4.
    assert abs(prior.standing("vendor").successes - 4.0) < 1e-6


def test_decay_lowers_weight_versus_fresh():
    cfg = AttestationConfig(half_life_days=30)
    fresh = attest_reputation(_records(settled=8), "vendor", issuer="acme").sign(ACME)
    old = _aged(attest_reputation(_records("other", settled=8), "other", issuer="acme"), 120)
    now = utcnow()
    fresh_prior = combine_attestations([fresh], config=cfg, as_of=now)
    old_prior = combine_attestations([old], config=cfg, as_of=now)
    assert old_prior.weight("other") < fresh_prior.weight("vendor")
    assert old_prior.weight("other") >= cfg.weight_floor  # decays toward prior, not below floor


def test_decay_is_deterministic_across_importers():
    cfg = AttestationConfig(half_life_days=45)
    a = _aged(attest_reputation(_records(settled=4), "vendor", issuer="acme"), 20)
    b = _aged(attest_reputation(_records(settled=3), "vendor", issuer="globex"), 60, GLOBEX)
    now = utcnow()
    one = combine_attestations([a, b], config=cfg, as_of=now)
    two = combine_attestations([b, a], config=cfg, as_of=now)
    assert one.standing("vendor").successes == two.standing("vendor").successes
    assert one.weight("vendor") == two.weight("vendor")


def test_naive_issued_at_does_not_crash_as_of_combination():
    # A cross-org attestation deserialized from a tz-naive ISO timestamp must still
    # combine against an as-of clock rather than raising on naive/aware comparison.
    att = attest_reputation(_records(settled=2), "vendor", issuer="acme", horizon_days=30)
    att.issued_at = att.issued_at.replace(tzinfo=None) - timedelta(days=90)
    att.seal().sign(ACME)
    prior = combine_attestations([att], as_of=utcnow())
    assert prior.standing("vendor") is None  # stale, excluded — no TypeError
    assert len(prior.stale) == 1


def test_invalid_horizon_raises():
    with pytest.raises(SettlementError):
        attest_reputation(_records(), "vendor", issuer="acme", horizon_days=0)


def test_invalid_half_life_raises():
    with pytest.raises(SettlementError):
        AttestationConfig(half_life_days=-1).validate_coherent()


# -- revocation ---------------------------------------------------------------


def test_revoke_builds_signs_and_verifies():
    att = attest_reputation(_records(), "vendor", issuer="acme").sign(ACME)
    rev = revoke_attestation(att, reason="vendor regressed").sign(ACME)
    assert rev.issuer == "acme"
    assert rev.subject == "vendor"
    assert rev.attestation_hash == att.content_hash
    assert rev.verify(ACME).valid
    assert rev.revokes(att)


def test_revocation_excludes_attestation_pinpointed():
    att = attest_reputation(_records(settled=2), "vendor", issuer="acme").sign(ACME)
    rev = revoke_attestation(att).sign(ACME)
    prior = combine_attestations([att], revocations=[rev])
    assert prior.standing("vendor") is None
    assert len(prior.revoked) == 1
    assert prior.revoked[0].revoked is True
    assert "revoked" in prior.revoked[0].reason


def test_revocation_keeps_other_issuers_evidence():
    bad = attest_reputation(_records(settled=2), "vendor", issuer="acme").sign(ACME)
    good = attest_reputation(_records(settled=2), "vendor", issuer="globex").sign(GLOBEX)
    rev = revoke_attestation(bad).sign(ACME)
    prior = combine_attestations([bad, good], revocations=[rev])
    assert prior.standing("vendor").attestations == 1
    assert prior.standing("vendor").issuers == ["globex"]


def test_supersession_names_the_replacement():
    old = attest_reputation(_records(settled=2), "vendor", issuer="acme").sign(ACME)
    new = attest_reputation(_records(settled=6), "vendor", issuer="acme").sign(ACME)
    rev = revoke_attestation(old, replacement=new, reason="updated").sign(ACME)
    assert rev.is_supersession
    assert rev.replacement_hash == new.content_hash
    prior = combine_attestations([old, new], revocations=[rev])
    # The old one is revoked; the new one is counted.
    assert prior.standing("vendor").successes == 6
    assert "superseded" in prior.revoked[0].reason


def test_forged_revocation_is_ignored_with_verifier():
    att = attest_reputation(_records(settled=2), "vendor", issuer="acme").sign(ACME)
    rev = revoke_attestation(att).sign(ACME)
    rev.signatures[0].signature = "deadbeef"  # forge
    prior = combine_attestations([att], revocations=[rev], verify_with=ACME)
    assert prior.standing("vendor") is not None  # forged revocation not honored
    assert not prior.revoked


def test_cross_issuer_revocation_cannot_cancel_anothers_attestation():
    att = attest_reputation(_records(settled=2), "vendor", issuer="acme").sign(ACME)
    # globex tries to revoke acme's attestation.
    rogue = AttestationRevocation(
        issuer="globex", subject="vendor", attestation_hash=att.content_hash
    ).sign(GLOBEX)
    # The issuer mismatch alone protects acme's claim — a revocation withdraws only
    # an attestation issued by the same party, regardless of how it is signed.
    prior = combine_attestations([att], revocations=[rogue])
    assert prior.standing("vendor") is not None
    assert not prior.revoked


def test_tampered_revocation_hash_is_ignored():
    att = attest_reputation(_records(settled=2), "vendor", issuer="acme").sign(ACME)
    rev = revoke_attestation(att).sign(ACME)
    rev.subject = "someone-else"  # tamper without resealing → hash mismatch
    prior = combine_attestations([att], revocations=[rev])
    assert prior.standing("vendor") is not None
    assert not prior.revoked


def test_revoke_by_hash_string_offline():
    att = attest_reputation(_records(settled=2), "vendor", issuer="acme").sign(ACME)
    rev = revoke_attestation(att.content_hash, subject="vendor", issuer="acme").sign(ACME)
    prior = combine_attestations([att], revocations=[rev])
    assert prior.standing("vendor") is None


def test_revoke_wrong_issuer_raises():
    att = attest_reputation(_records(), "vendor", issuer="acme")
    with pytest.raises(SettlementError):
        revoke_attestation(att, issuer="globex")


def test_revocation_wire_roundtrip_preserves_verification():
    att = attest_reputation(_records(), "vendor", issuer="acme").sign(ACME)
    rev = revoke_attestation(att, reason="r").sign(ACME)
    back = AttestationRevocation.from_wire(rev.to_wire())
    assert back.verify(ACME).valid
    assert back.content_hash == rev.content_hash


def test_revocation_require_valid_raises_on_tamper():
    att = attest_reputation(_records(), "vendor", issuer="acme").sign(ACME)
    rev = revoke_attestation(att).sign(ACME)
    rev.attestation_hash = "0" * 64  # tamper without resealing
    with pytest.raises(SettlementError):
        rev.require_valid(ACME)


# -- book & app surface for revocation & freshness ----------------------------


def test_settlement_book_revoke_signs_as_owner():
    signer = HMACSigner("book-key", key_id="acme")
    book = SettlementBook("acme", signer=signer)
    book.settle(_contract(seller="vendor"), cost_usd=0.05)
    book.settle(_contract(seller="vendor"), cost_usd=0.05)
    att = book.attest("vendor")
    rev = book.revoke(att, reason="regressed")
    assert rev.issuer == "acme"
    assert rev.signed_by == ["acme"]
    assert rev.verify(signer).valid
    assert rev.revokes(att)


def test_settlement_book_revoke_rejects_foreign_attestation():
    book = SettlementBook("acme", signer=HMACSigner("k", key_id="acme"))
    foreign = attest_reputation(_records(), "vendor", issuer="globex")
    with pytest.raises(SettlementError):
        book.revoke(foreign)


def test_app_revoke_attestation_audits_and_verifies():
    app = _app("acme")
    app.use_settlement_book()
    app.settle(_contract(seller="vendor"), cost_usd=0.05)
    app.settle(_contract(seller="vendor"), cost_usd=0.05)
    att = app.attest_reputation("vendor")
    rev = app.revoke_attestation(att, reason="regressed")
    assert rev.verify(app.contract_signer).valid
    assert app.audit.query(action=REVOCATION_ACTION)
    assert app.audit.verify_chain()


def test_app_import_reputation_honors_revocation():
    issuer = _app("acme")
    issuer.use_settlement_book()
    for _ in range(4):
        issuer.settle(_contract(seller="vendor"), cost_usd=0.05)
    att = issuer.attest_reputation("vendor")
    rev = issuer.revoke_attestation(att, reason="vendor regressed")

    buyer = _app("buyer")
    buyer.use_reputation_ledger()
    prior = buyer.import_reputation([att], revocations=[rev])
    assert prior.standing("vendor") is None
    assert len(prior.revoked) == 1


def test_app_import_reputation_decays_stale_with_as_of():
    issuer = _app("acme")
    issuer.use_settlement_book()
    for _ in range(4):
        issuer.settle(_contract(seller="vendor"), cost_usd=0.05)
    att = issuer.attest_reputation("vendor")
    att.horizon_days = 30
    att.issued_at = utcnow() - timedelta(days=90)
    att.seal()
    att.sign(issuer.contract_signer, party="acme")

    buyer = _app("buyer")
    buyer.use_reputation_ledger()
    prior = buyer.import_reputation([att], as_of=utcnow())
    assert prior.standing("vendor") is None  # stale past its 30-day window
    assert len(prior.stale) == 1
