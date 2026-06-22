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

import pytest

from vincio import (
    ContextApp,
    PortableReputation,
    ReputationAttestation,
    attest_reputation,
    combine_attestations,
    settle_contract,
)
from vincio.core.errors import SettlementError
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
from vincio.settlement.attestation import ATTESTATION_ACTION
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
    att = attest_reputation(_records("vendor", settled=1), "vendor", issuer="acme",
                            resolutions=[resolution])
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
    att = attest_reputation(_records("vendor", settled=1), "vendor", issuer="acme",
                            resolutions=[resolution])
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
        "work", buyer=pos, seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.12),
        buyer_id="buyer", seller_id="reliable",
    )
    flaky = app.negotiate(
        "work", buyer=pos, seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.12),
        buyer_id="buyer", seller_id="flaky",
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
