"""Coverage-hardening tests for :mod:`vincio.settlement.attestation`.

These target the uncovered error paths, freshness/decay edges, revocation
withdrawal, the transitive trust web (``build_trust_model`` / ``TrustModel`` /
``_trust_multiplier``), and the ``combine_attestations`` pinpointing branches —
all through the real API with the deterministic ``MockProvider`` and a real
``HMACSigner`` (never mock/patch).
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from vincio import (
    AttestationRevocation,
    ContextApp,
    ReputationAttestation,
    attest_reputation,
    combine_attestations,
    revoke_attestation,
    settle_contract,
)
from vincio.core.errors import SettlementError
from vincio.core.utils import utcnow
from vincio.negotiation import Contract, ContractTerms
from vincio.optimize.reputation import ReputationLedger
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement.attestation import (
    AttestationConfig,
    IssuerTrust,
    TrustConfig,
    TrustModel,
    _admissible,
    _has_local_evidence,
    _supersedes,
    _trust_multiplier,
    build_trust_model,
)

ACME = HMACSigner("acme-key", key_id="acme")
GLOBEX = HMACSigner("globex-key", key_id="globex")
INITECH = HMACSigner("initech-key", key_id="initech")


def _app(name: str = "issuer") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def _contract(price: float = 0.10, *, seller: str = "vendor", buyer: str = "acme") -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope="work", price_usd=price)
    ).seal()


def _records(seller: str = "vendor", *, settled: int = 2, breached: int = 0):
    out = []
    for _ in range(settled):
        out.append(settle_contract(_contract(seller=seller), cost_usd=0.05))
    for _ in range(breached):
        out.append(settle_contract(_contract(0.04, seller=seller), cost_usd=0.09))
    return out


def _att(
    issuer: str,
    subject: str,
    *,
    settled: int = 2,
    breached: int = 0,
    signer: HMACSigner | None = None,
) -> ReputationAttestation:
    att = attest_reputation(_records(subject, settled=settled, breached=breached), subject, issuer=issuer)
    if signer is not None:
        att.sign(signer, party=issuer)
    return att


# -- AttestationConfig.validate_coherent error paths --------------------------


def test_config_rejects_non_positive_prior():
    with pytest.raises(SettlementError, match="prior pseudo-counts must be positive"):
        AttestationConfig(prior_success=0.0).validate_coherent()
    with pytest.raises(SettlementError, match="prior pseudo-counts must be positive"):
        AttestationConfig(prior_failure=-1.0).validate_coherent()


def test_config_rejects_inverted_weight_band():
    with pytest.raises(SettlementError, match="weight_floor . weight_ceiling"):
        AttestationConfig(weight_floor=0.9, weight_ceiling=0.2).validate_coherent()


def test_config_rejects_non_positive_per_issuer_cap():
    with pytest.raises(SettlementError, match="per_issuer_cap must be positive"):
        AttestationConfig(per_issuer_cap=0.0).validate_coherent()


def test_config_rejects_non_positive_half_life():
    with pytest.raises(SettlementError, match="half_life_days must be positive"):
        AttestationConfig(half_life_days=-3.0).validate_coherent()


# -- TrustConfig validation, clamp, and trust mapping -------------------------


def test_trust_config_rejects_negative_depth():
    with pytest.raises(SettlementError, match="max_depth must be non-negative"):
        TrustConfig(max_depth=-1).validate_coherent()


def test_trust_config_rejects_bad_hop_decay():
    with pytest.raises(SettlementError, match="hop_decay must satisfy"):
        TrustConfig(hop_decay=0.0).validate_coherent()
    with pytest.raises(SettlementError, match="hop_decay must satisfy"):
        TrustConfig(hop_decay=1.5).validate_coherent()


def test_trust_config_rejects_inverted_trust_band():
    with pytest.raises(SettlementError, match="trust weights must satisfy"):
        TrustConfig(trust_floor=0.8, trust_ceiling=0.3).validate_coherent()


def test_trust_config_clamp_and_map():
    cfg = TrustConfig(trust_floor=0.2, trust_ceiling=0.9)
    # clamp_trust pins out-of-band values to the edges.
    assert cfg.clamp_trust(1.5) == 0.9
    assert cfg.clamp_trust(-1.0) == 0.2
    assert cfg.clamp_trust(0.5) == 0.5
    # trust_of is the same monotonic floor+span*r map a weight uses.
    assert cfg.trust_of(0.0) == 0.2
    assert cfg.trust_of(1.0) == 0.9
    assert cfg.trust_of(0.5) == pytest.approx(0.55, abs=1e-9)
    # out-of-range reputation is clamped before mapping.
    assert cfg.trust_of(2.0) == 0.9


# -- attestation sign/seal/verify edges ---------------------------------------


def test_sign_seals_unsealed_attestation():
    att = ReputationAttestation(issuer="acme", subject="vendor", settled=2)
    att.reputation = round(AttestationConfig().reputation_of(2, 0), 9)
    assert att.content_hash == ""
    att.sign(ACME)  # no prior seal — sign must seal first
    assert att.content_hash == att.compute_hash()
    assert att.verify(ACME).valid


def test_evidence_unsound_when_count_negative():
    att = _att("acme", "vendor", settled=2).sign(ACME)
    att.settled = -1  # negative count, then re-seal so the hash matches
    att.seal()
    assert att._evidence_sound() is False


def test_require_valid_returns_self_on_success():
    att = _att("acme", "vendor").sign(ACME)
    assert att.require_valid(ACME) is att


def test_revocation_require_valid_returns_self_on_success():
    att = _att("acme", "vendor").sign(ACME)
    rev = _rev(att, ACME)
    assert rev.require_valid(ACME) is rev


def test_counted_and_admitted_properties():
    counted_att = _att("acme", "vendor").sign(ACME)
    self_att = _att("dup", "dup").sign(HMACSigner("dup-key", key_id="dup"))
    pr = combine_attestations([counted_att, self_att], verifier=None)
    assert [v.issuer for v in pr.counted] == ["acme"]
    # both verified as artifacts (admissible) even though the self-attestation
    # is excluded from the pool.
    assert {v.issuer for v in pr.admitted} == {"acme", "dup"}


def test_evidence_unsound_when_prior_non_positive():
    att = _att("acme", "vendor", settled=2).sign(ACME)
    att.prior_success = 0.0  # corrupt the prior, then re-seal so the hash matches
    att.seal()
    result = att.verify(ACME)
    assert result.evidence_sound is False
    assert result.reason == "attested reputation does not re-derive from the evidence counts"


def test_verify_reason_no_verifier_but_signature_required():
    att = _att("acme", "vendor").sign(ACME)
    result = att.verify(None)  # binding-only, but issuer signature still required
    assert result.hash_ok is True
    assert result.valid is False
    assert result.reason == "no verifier supplied — signature present but not authenticated"


def test_require_valid_raises_on_missing_signature():
    att = _att("acme", "vendor").sign(ACME)
    with pytest.raises(SettlementError, match="failed verification"):
        att.require_valid(ACME, require=["globex"])  # globex never signed


# -- revocation verify reasons & require_valid --------------------------------


def _rev(att: ReputationAttestation, signer: HMACSigner | None = None) -> AttestationRevocation:
    rev = revoke_attestation(att)
    if signer is not None:
        rev.sign(signer)
    return rev


def test_revocation_verify_reason_no_verifier():
    att = _att("acme", "vendor").sign(ACME)
    rev = _rev(att, ACME)
    result = rev.verify(None)
    assert result.hash_ok is True
    assert result.valid is False
    assert result.reason == "no verifier supplied — signature present but not authenticated"


def test_revocation_verify_reason_missing_signature():
    att = _att("acme", "vendor").sign(ACME)
    rev = _rev(att, ACME)
    result = rev.verify(ACME, require=["globex"])
    assert result.signatures_ok is False
    assert "missing/invalid signatures for ['globex']" in result.reason


def test_revocation_require_valid_raises_on_tamper():
    att = _att("acme", "vendor").sign(ACME)
    rev = _rev(att, ACME)
    rev.attestation_hash = "0" * 32  # tamper without resealing -> hash mismatch
    with pytest.raises(SettlementError, match="revocation .* failed verification"):
        rev.require_valid(ACME)


# -- revoke_attestation argument paths ----------------------------------------


def test_revoke_by_hash_string_offline():
    att = _att("acme", "vendor").sign(ACME)
    rev = revoke_attestation(att.content_hash, issuer="acme", subject="vendor")
    assert rev.attestation_hash == att.content_hash
    assert rev.revokes(att)


def test_revoke_empty_hash_string_raises():
    with pytest.raises(SettlementError, match="must name the content hash"):
        revoke_attestation("", issuer="acme", subject="vendor")


def test_revoke_issuer_mismatch_raises():
    att = _att("acme", "vendor").sign(ACME)
    with pytest.raises(SettlementError, match="cannot revoke an attestation issued by"):
        revoke_attestation(att, issuer="globex")


def test_revoke_subject_mismatch_raises():
    att = _att("acme", "vendor").sign(ACME)
    with pytest.raises(SettlementError, match="does not match the attestation's subject"):
        revoke_attestation(att, subject="other")


def test_revoke_with_replacement_attestation_object_is_supersession():
    old = _att("acme", "vendor", settled=1).sign(ACME)
    new = _att("acme", "vendor", settled=4).sign(ACME)
    rev = revoke_attestation(old, replacement=new)
    assert rev.is_supersession
    assert rev.replacement_hash == new.content_hash


def test_revoke_with_replacement_hash_string():
    old = _att("acme", "vendor", settled=1).sign(ACME)
    rev = revoke_attestation(old, replacement="deadbeef" * 4)
    assert rev.replacement_hash == "deadbeef" * 4


def test_revokes_false_for_different_issuer():
    att = _att("acme", "vendor").sign(ACME)
    rev = AttestationRevocation(
        issuer="globex", subject="vendor", attestation_hash=att.content_hash
    ).seal()
    assert rev.revokes(att) is False  # issuer mismatch — cannot cancel another's claim


# -- attest_reputation: verifier skips forged record, dissent hashes ----------


def test_attest_skips_forged_signed_record():
    good = settle_contract(_contract(seller="vendor"), cost_usd=0.05).sign(GLOBEX, party="vendor")
    forged = settle_contract(_contract(seller="vendor"), cost_usd=0.05)
    # Sign forged, then mutate the signature bytes so it no longer verifies.
    forged.sign(GLOBEX, party="vendor")
    forged.signatures[0].signature = "tampered-sig"
    att = attest_reputation([good, forged], "vendor", issuer="acme", verifier=GLOBEX)
    assert att.settled == 1  # only the genuinely-signed record counted


def test_attest_horizon_must_be_positive():
    with pytest.raises(SettlementError, match="horizon_days must be positive"):
        attest_reputation(_records("vendor"), "vendor", issuer="acme", horizon_days=0.0)


def test_attest_binds_resolution_source_hash():
    app = _app()
    c = _contract()
    acme_rec = settle_contract(c, cost_usd=0.08).sign(ACME, party="acme")
    vendor_ok = settle_contract(c, cost_usd=0.08).sign(GLOBEX, party="vendor")
    liar = settle_contract(c, cost_usd=0.05).sign(GLOBEX, party="vendor")
    resolution = app.arbitrate([acme_rec, vendor_ok, liar])
    assert resolution.dissenters == ["vendor"]
    att = attest_reputation(
        _records("vendor", settled=1), "vendor", issuer="acme", resolutions=[resolution]
    )
    assert att.dissents == 1
    assert resolution.content_hash in att.source_hashes


def test_attest_dissent_resolution_without_content_hash():
    # upheld + dissenter matches the subject + decision sound, but no content hash:
    # the dissent is counted but no source hash is bound (the 1216->1208 branch).
    class _Sound:
        decision_sound = True

    class _NoHashResolution:
        upheld = True
        dissenters = ["vendor"]
        content_hash = ""

        def verify(self, verifier=None):
            return _Sound()

    att = attest_reputation(
        _records("vendor", settled=1), "vendor", issuer="acme", resolutions=[_NoHashResolution()]
    )
    assert att.dissents == 1
    # the empty resolution hash is not bound as a source.
    assert "" not in att.source_hashes


def test_attest_skips_resolution_when_subject_not_dissenter():
    class _OtherDissenter:
        upheld = True
        dissenters = ["someone-else"]
        content_hash = "y" * 32

    att = attest_reputation(
        _records("vendor", settled=2), "vendor", issuer="acme", resolutions=[_OtherDissenter()]
    )
    assert att.dissents == 0


def test_attest_ignores_non_upheld_resolution():
    # An unresolved (not upheld) resolution contributes no dissent and is skipped.
    class _FakeResolution:
        upheld = False
        dissenters = ["vendor"]
        content_hash = "x" * 32

    att = attest_reputation(
        _records("vendor", settled=2), "vendor", issuer="acme", resolutions=[_FakeResolution()]
    )
    assert att.dissents == 0


# -- _has_local_evidence branches ---------------------------------------------


def test_has_local_evidence_via_snapshot_rounds():
    ledger = ReputationLedger()
    ledger.record_outcome("seen", passed=True)
    assert _has_local_evidence(ledger, "seen") is True
    assert _has_local_evidence(ledger, "never") is False


def test_has_local_evidence_via_members_when_no_snapshot():
    class _MembersOnly:
        def members(self):
            return ["known"]

    assert _has_local_evidence(_MembersOnly(), "known") is True
    assert _has_local_evidence(_MembersOnly(), "stranger") is False


def test_has_local_evidence_false_for_bare_object():
    assert _has_local_evidence(object(), "anyone") is False


def test_has_local_evidence_members_raises_returns_false():
    class _BadMembers:
        def members(self):
            raise RuntimeError("boom")

    assert _has_local_evidence(_BadMembers(), "m") is False


def test_has_local_evidence_snapshot_raises_falls_through_to_members():
    class _Weird:
        def snapshot(self, member_id):
            raise RuntimeError("boom")

        def members(self):
            return ["m"]

    assert _has_local_evidence(_Weird(), "m") is True


# -- PortableReputation reads -------------------------------------------------


def test_issuers_for_unknown_subject_is_empty():
    pr = combine_attestations([_att("acme", "vendor").sign(ACME)], verifier=ACME)
    assert pr.issuers_for("nobody") == []
    assert pr.issuers_for("vendor") == ["acme"]


def test_reputation_for_unknown_returns_prior_mean():
    pr = combine_attestations([_att("acme", "vendor").sign(ACME)], verifier=ACME)
    cfg = AttestationConfig()
    assert pr.reputation("stranger") == round(cfg.reputation_of(0.0, 0.0), 9)


def test_weight_falls_back_when_base_weight_raises():
    # base has evidence for the member but weight() blows up -> falls back to prior.
    class _BrokenBase:
        def snapshot(self, member_id):
            class _S:
                rounds = 5

            return _S()

        def weight(self, member_id):
            raise RuntimeError("nope")

    pr = combine_attestations(
        [_att("acme", "vendor").sign(ACME)], verifier=ACME, base=_BrokenBase()
    )
    assert pr.weight("vendor") == pr.config.weight_of(pr.reputation("vendor"))


def test_verdict_for_and_standings_listing():
    pr = combine_attestations(
        [_att("acme", "vendor").sign(ACME), _att("globex", "supplier").sign(GLOBEX)],
        verifier=None,
    )
    v = pr.verdict_for("acme", "vendor")
    assert v is not None and v.counted
    assert pr.verdict_for("acme", "nope") is None
    subjects = [s.subject for s in pr.standings()]
    assert subjects == ["supplier", "vendor"]  # sorted by subject


# -- TrustModel reads ---------------------------------------------------------


def test_trust_model_trust_in_floor_for_unknown():
    model = TrustModel({}, TrustConfig(trust_floor=0.15))
    assert model.trust_in("ghost") == 0.15
    assert model.weight("ghost") == 0.15  # weight aliases trust_in
    assert model.assessment("ghost") is None
    assert model.issuers() == []
    assert model.direct_issuers() == []
    assert model.transitive_issuers() == []
    assert model.assessments() == []


def test_trust_model_partitions_direct_and_transitive():
    direct = IssuerTrust(issuer="d", trust=0.8, depth=0, direct=True)
    trans = IssuerTrust(issuer="t", trust=0.4, depth=1, vouched_by=["d"])
    model = TrustModel({"d": direct, "t": trans}, TrustConfig())
    assert model.trust_in("d") == 0.8
    assert model.direct_issuers() == ["d"]
    assert model.transitive_issuers() == ["t"]
    assert trans.transitive is True
    assert direct.transitive is False


# -- _trust_multiplier --------------------------------------------------------


def test_trust_multiplier_none_is_full_pull():
    assert _trust_multiplier(None, "anyone") == 1.0


def test_trust_multiplier_via_callable_clamped():
    assert _trust_multiplier(lambda i: 0.3, "x") == 0.3
    assert _trust_multiplier(lambda i: 5.0, "x") == 1.0  # clamped up to 1
    assert _trust_multiplier(lambda i: -2.0, "x") == 0.0  # clamped down to 0


def test_trust_multiplier_swallows_resolver_error():
    class _Boom:
        def trust_in(self, issuer):
            raise RuntimeError("explode")

    assert _trust_multiplier(_Boom(), "x") == 1.0  # a miss must not break weighting


# -- _admissible --------------------------------------------------------------


def test_admissible_rejects_tampered_hash():
    att = _att("acme", "vendor").sign(ACME)
    att.settled = 99  # mutate without resealing -> hash mismatch
    assert _admissible(att, None) is False


def test_admissible_rejects_forged_signature():
    att = _att("acme", "vendor").sign(ACME)
    att.signatures[0].signature = "garbage"
    assert _admissible(att, ACME) is False


def test_admissible_true_for_genuine():
    att = _att("acme", "vendor").sign(ACME)
    assert _admissible(att, ACME) is True


# -- build_trust_model: direct + transitive + sybil resistance ----------------


def _shared_signer(*names: str) -> HMACSigner:
    return HMACSigner("shared-key", key_id="shared")


def test_build_trust_model_direct_hop_zero():
    base = ReputationLedger()
    for _ in range(4):
        base.record_outcome("acme", passed=True)  # importer trusts acme first-hand
    atts = [_att("acme", "vendor", signer=ACME)]
    model = build_trust_model(atts, base=base, config=TrustConfig(max_depth=0))
    a = model.assessment("acme")
    assert a is not None
    assert a.direct is True
    assert a.depth == 0
    assert a.trust > TrustConfig().trust_floor


def test_build_trust_model_transitive_one_hop():
    # importer trusts acme directly; acme attests globex (vouches for it as a
    # counterparty); globex inherits decayed transitive trust at hop 1.
    base = ReputationLedger()
    for _ in range(6):
        base.record_outcome("acme", passed=True)
    signer = _shared_signer()
    acme_att = _att("acme", "vendor", settled=3, signer=signer)
    acme_vouches_globex = _att("acme", "globex", settled=5, signer=signer)
    globex_att = _att("globex", "supplier", settled=2, signer=signer)
    model = build_trust_model(
        [acme_att, acme_vouches_globex, globex_att],
        base=base,
        config=TrustConfig(max_depth=1, hop_decay=0.5),
        verifier=signer,
    )
    g = model.assessment("globex")
    assert g is not None
    assert g.depth == 1
    assert g.direct is False
    assert g.vouched_by == ["acme"]
    assert g.reputation is not None
    # transitive trust is decayed and stays within the band.
    assert TrustConfig().trust_floor <= g.trust <= 1.0


def test_build_trust_model_unknown_issuer_falls_to_floor():
    base = ReputationLedger()
    base.record_outcome("acme", passed=True)
    signer = _shared_signer()
    # globex is attested by nobody trusted -> never reached -> floor.
    atts = [_att("acme", "vendor", signer=signer), _att("globex", "x", signer=signer)]
    model = build_trust_model(atts, base=base, config=TrustConfig(max_depth=1), verifier=signer)
    assert model.trust_in("globex") == TrustConfig().trust_floor


def test_build_trust_model_sybil_cluster_stays_at_floor():
    # importer knows nobody; a clutch of mutually-vouching unknowns is never reached.
    base = ReputationLedger()
    signer = _shared_signer()
    atts = [
        _att("sybilA", "sybilB", signer=signer),
        _att("sybilB", "sybilA", signer=signer),
    ]
    model = build_trust_model(atts, base=base, config=TrustConfig(max_depth=3), verifier=signer)
    assert model.trust_in("sybilA") == TrustConfig().trust_floor
    assert model.trust_in("sybilB") == TrustConfig().trust_floor
    assert model.direct_issuers() == []


def test_build_trust_model_no_base_reaches_nobody():
    # with no base ledger there is no hop-0 root, so every issuer stays at the floor.
    signer = _shared_signer()
    atts = [_att("acme", "vendor", signer=signer), _att("globex", "x", signer=signer)]
    model = build_trust_model(atts, base=None, config=TrustConfig(max_depth=2), verifier=signer)
    assert model.issuers() == []
    assert model.trust_in("acme") == TrustConfig().trust_floor


def test_build_trust_model_self_voucher_does_not_bootstrap():
    # acme is trusted directly and attests itself; the self-vouch must not be used
    # to lend acme transitive trust (the att.issuer != issuer guard).
    base = ReputationLedger()
    for _ in range(4):
        base.record_outcome("acme", passed=True)
    signer = _shared_signer()
    acme_self = _att("acme", "acme", settled=9, signer=signer)
    acme_vouches_globex = _att("acme", "globex", settled=3, signer=signer)
    # globex must itself issue an attestation to be assessed as an issuer.
    globex_att = _att("globex", "supplier", settled=2, signer=signer)
    model = build_trust_model(
        [acme_self, acme_vouches_globex, globex_att],
        base=base,
        config=TrustConfig(max_depth=1),
        verifier=signer,
    )
    acme = model.assessment("acme")
    assert acme is not None and acme.depth == 0  # stays the direct first-hand trust
    globex = model.assessment("globex")
    assert globex is not None and globex.depth == 1


def test_build_trust_model_base_weight_raises_leaves_unreached():
    class _BadWeight:
        def members(self):
            return ["acme"]

        def weight(self, member_id):
            raise RuntimeError("kaboom")

    atts = [_att("acme", "vendor", signer=ACME)]
    model = build_trust_model(atts, base=_BadWeight(), config=TrustConfig(), verifier=ACME)
    assert model.assessment("acme") is None  # weight blew up -> not trusted directly


# -- combine_attestations: trust_config, revocation, freshness, supersede -----


def test_combine_with_trust_config_builds_model_and_pinpoints():
    base = ReputationLedger()
    for _ in range(5):
        base.record_outcome("acme", passed=True)
    signer = _shared_signer()
    acme_att = _att("acme", "vendor", settled=4, signer=signer)
    globex_att = _att("globex", "vendor", settled=4, signer=signer)
    pr = combine_attestations(
        [acme_att, globex_att],
        verifier=signer,
        base=base,
        trust_config=TrustConfig(max_depth=0),
    )
    standing = pr.standing("vendor")
    assert standing is not None
    # acme is trusted directly (full-ish pull), globex unknown (floored) -> acme's
    # trust multiplier strictly exceeds globex's.
    assert standing.issuer_trust["acme"] > standing.issuer_trust["globex"]
    assert pr.trust_in("acme") > pr.trust_in("globex")


def test_combine_excludes_revoked_attestation_pinpointed():
    att = _att("acme", "vendor", settled=3).sign(ACME)
    rev = revoke_attestation(att, reason="regressed").sign(ACME)
    pr = combine_attestations([att], verifier=ACME, revocations=[rev])
    assert pr.standing("vendor") is None  # nothing counted
    [revoked] = pr.revoked
    assert revoked.revoked is True
    assert "withdrawn by its issuer" in revoked.reason
    assert "regressed" in revoked.reason


def test_combine_ignores_cross_issuer_revocation():
    att = _att("acme", "vendor", settled=3).sign(ACME)
    # globex tries to revoke acme's attestation by hash; issuer mismatch -> ignored.
    forged_rev = AttestationRevocation(
        issuer="globex", subject="vendor", attestation_hash=att.content_hash
    ).seal()
    forged_rev.sign(GLOBEX)
    pr = combine_attestations([att], verifier=None, revocations=[forged_rev])
    assert pr.standing("vendor") is not None  # acme's attestation still counts
    assert pr.revoked == []


def test_combine_revocation_subject_filter_skips_other_subjects():
    att = _att("acme", "vendor", settled=2).sign(ACME)
    other_rev = revoke_attestation(att, subject="vendor").sign(ACME)
    other_rev.subject = "different"  # belongs to another subject than the filter
    other_rev.seal()
    other_rev.sign(ACME)
    pr = combine_attestations(
        [att], subject="vendor", verifier=ACME, revocations=[other_rev]
    )
    assert pr.standing("vendor") is not None  # revocation for other subject ignored


def test_combine_supersession_revocation_reason():
    old = _att("acme", "vendor", settled=2).sign(ACME)
    new = _att("acme", "other", settled=5).sign(ACME)
    rev = revoke_attestation(old, replacement=new).sign(ACME)
    pr = combine_attestations([old], verifier=ACME, revocations=[rev])
    [revoked] = pr.revoked
    assert "superseded by its issuer" in revoked.reason


def test_combine_excludes_stale_attestation_pinpointed():
    now = utcnow()
    att = attest_reputation(
        _records("vendor", settled=2), "vendor", issuer="acme", horizon_days=10.0
    )
    att.issued_at = now - timedelta(days=40)
    att.seal().sign(ACME)
    pr = combine_attestations([att], verifier=ACME, as_of=now)
    assert pr.standing("vendor") is None
    [stale] = pr.stale
    assert stale.stale is True
    assert "validity window" in stale.reason


def test_combine_half_life_decays_evidence_under_clock():
    now = utcnow()
    fresh = attest_reputation(_records("vendor", settled=8), "vendor", issuer="acme").sign(ACME)
    aged = attest_reputation(_records("vendor", settled=8), "vendor", issuer="acme")
    aged.issued_at = now - timedelta(days=30)
    aged.seal().sign(ACME)
    cfg = AttestationConfig(half_life_days=10.0)
    pr_fresh = combine_attestations([fresh], config=cfg, verifier=ACME, as_of=now)
    pr_aged = combine_attestations([aged], config=cfg, verifier=ACME, as_of=now)
    # the aged attestation contributes less pooled evidence than the fresh one.
    assert pr_aged.standing("vendor").successes < pr_fresh.standing("vendor").successes


def test_combine_refuses_evidence_unsound_attestation():
    att = _att("acme", "vendor", settled=4).sign(ACME)
    att.reputation = 0.999  # inflate the score, then re-seal so the hash recomputes
    att.seal()
    pr = combine_attestations([att], verifier=None)
    [refused] = pr.refused
    assert refused.admissible is False
    assert "does not re-derive from the evidence" in refused.reason
    assert pr.standing("vendor") is None


def test_combine_trust_via_callable_scales_mass():
    # a plain issuer->float callable scales each issuer's contributed mass.
    acme_att = _att("acme", "vendor", settled=4).sign(ACME)
    globex_att = _att("globex", "vendor", settled=4).sign(GLOBEX)
    trust = {"acme": 1.0, "globex": 0.2}
    pr = combine_attestations(
        [acme_att, globex_att], verifier=None, trust=lambda i: trust[i]
    )
    standing = pr.standing("vendor")
    assert standing.issuer_trust["acme"] == 1.0
    assert standing.issuer_trust["globex"] == 0.2
    # globex's discounted mass means acme contributes the bulk of the evidence.
    assert standing.successes < acme_att.successes + globex_att.successes


def test_combine_pinpoints_superseded_attestation():
    small = _att("acme", "vendor", settled=1).sign(ACME)
    big = _att("acme", "vendor", settled=5).sign(ACME)
    pr = combine_attestations([small, big], verifier=ACME)
    # only the larger counts; the smaller is excluded with a superseded reason.
    counted = [v for v in pr.verdicts if v.counted]
    assert len(counted) == 1
    assert counted[0].attestation_id == big.id
    superseded = [v for v in pr.excluded if v.reason and "superseded" in v.reason]
    assert len(superseded) == 1
    assert superseded[0].attestation_id == small.id


# -- _supersedes tie-breaks ---------------------------------------------------


def test_supersedes_prefers_more_evidence():
    a = _att("acme", "vendor", settled=5)
    b = _att("acme", "vendor", settled=2)
    assert _supersedes(a, b) is True
    assert _supersedes(b, a) is False


def test_supersedes_breaks_evidence_tie_by_issue_time():
    earlier = _att("acme", "vendor", settled=3)
    later = _att("acme", "vendor", settled=3)
    earlier.issued_at = utcnow() - timedelta(hours=2)
    later.issued_at = utcnow()
    assert _supersedes(later, earlier) is True
    assert _supersedes(earlier, later) is False


def test_supersedes_breaks_full_tie_by_id():
    a = _att("acme", "vendor", settled=3)
    b = _att("acme", "vendor", settled=3)
    a.issued_at = b.issued_at  # identical evidence and time -> deterministic id break
    hi, lo = (a, b) if a.id > b.id else (b, a)
    assert _supersedes(hi, lo) is True
    assert _supersedes(lo, hi) is False
