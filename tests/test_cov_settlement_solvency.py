"""Coverage-hardening tests for :mod:`vincio.settlement.solvency`.

These exercise the error paths and edge branches of the proof-of-solvency stack: tampered /
forged / wrong-poster refusals, the mutually-exclusive attested-vs-completed liability figures,
the Merkle inclusion-proof binding, completeness re-derivation invariants, root non-equivocation,
and snapshot-monotonicity walks with creditor-signed discharges. Every assertion pins a concrete
value, a re-derived verdict, or a raised ``SettlementError`` with its message — no mocks, all real
offline artifacts signed with deterministic HMAC keys.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from vincio.core.errors import SettlementError
from vincio.security.audit import HMACSigner
from vincio.settlement.custody import attest_custody
from vincio.settlement.solvency import (
    CompletenessProof,
    Discharge,
    EquivocationProof,
    HistoryConsistencyProof,
    InclusionProof,
    LiabilityAttestation,
    LiabilityLine,
    MerkleStep,
    OmissionBreach,
    RootCommitment,
    SolvencyProof,
    attest_liabilities,
    check_completeness,
    check_history_consistency,
    check_root_consistency,
    discharge_liability,
    prove_equivocation,
    prove_solvency,
)

ATTESTOR = HMACSigner("attestor-key", key_id="attestor")
CUSTODIAN = HMACSigner("custodian-key", key_id="custodian")
CREDITOR = HMACSigner("creditor-key", key_id="creditor")
FORGER = HMACSigner("forger-key", key_id="forger")

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _at(days: int) -> datetime:
    return _T0 + timedelta(days=days)


# == coercion error paths =====================================================


def test_attest_liabilities_rejects_negative_obligation() -> None:
    with pytest.raises(SettlementError, match="cannot net a real debt against a fictitious"):
        attest_liabilities("vendor", {"acme": -5.0})


def test_attest_liabilities_rejects_unknown_item_shape() -> None:
    with pytest.raises(SettlementError, match="must be a number, a mapping, or"):
        attest_liabilities("vendor", [object()])


def test_attest_liabilities_accepts_tuple_and_dict_items() -> None:
    att = attest_liabilities(
        "vendor", [("acme", 10.0), {"creditor": "globex", "amount_usd": 5.0}]
    )
    assert att.liabilities_usd == 15.0
    assert att.creditors == ["acme", "globex"]


def test_attest_liabilities_single_number_makes_one_unnamed_line() -> None:
    att = attest_liabilities("vendor", 40.0)
    assert [line.creditor for line in att.liabilities] == ["liabilities"]
    assert att.liabilities_usd == 40.0


# == link_to / linked-history validation ======================================


def test_link_to_seals_unsealed_prior() -> None:
    prior = LiabilityAttestation(attestor="vendor", poster="vendor", as_of=_at(0))
    later = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(1))
    assert not prior.content_hash  # prior was never sealed
    later.link_to(prior)
    # link_to had to seal the prior to pin its hash; that hash now flows into the successor.
    assert prior.content_hash
    assert later.prior_hash == prior.content_hash


def test_link_to_rejects_different_counterparty() -> None:
    prior = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    later = attest_liabilities("other", {"acme": 10.0}, as_of=_at(1))
    with pytest.raises(SettlementError, match="a history is one counterparty's sequence"):
        later.link_to(prior)


def test_link_to_rejects_back_dated_successor() -> None:
    prior = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(5))
    later = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(2))
    with pytest.raises(SettlementError, match="strictly later"):
        later.link_to(prior)


def test_attest_liabilities_prior_kwarg_links_chain() -> None:
    first = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    second = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(1), prior=first)
    assert second.has_prior is True
    assert second.prior_root == first.liabilities_root


def test_prior_link_sound_rejects_partial_link() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(1))
    # A prior hash with no prior root / instant is a malformed half-link.
    att.prior_hash = "deadbeef"
    att.seal()
    result = att.verify()
    assert result.liabilities_sound is False
    assert "does not re-derive" in (result.reason or "")


# == verify reasons & require_valid ===========================================


def test_verify_reason_unsealed() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    att.content_hash = ""  # un-seal
    result = att.verify()
    assert result.valid is False
    assert result.reason == "attestation is not sealed (no content hash)"


def test_verify_reason_tampered_total() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    att.liabilities_usd = 999.0
    att.content_hash = att.compute_hash()  # re-seal the lie
    result = att.verify()
    assert result.hash_ok is True
    assert result.liabilities_sound is False
    assert "does not re-derive" in (result.reason or "")


def test_verify_reason_hash_mismatch() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    att.poster = "tampered"  # change a hashed field without re-sealing
    result = att.verify()
    assert result.hash_ok is False
    assert result.reason == "content hash does not match the attestation facts"


def test_verify_missing_required_signature() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor").sign(ATTESTOR)
    result = att.verify(ATTESTOR, require=["someone-else"])
    assert result.signatures_ok is False
    assert "missing/invalid signatures" in (result.reason or "")


def test_verify_signature_forged() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor").sign(ATTESTOR)
    att.signatures[0].signature = FORGER.sign(att.content_hash)
    result = att.verify(ATTESTOR)
    assert result.signatures_ok is False
    assert result.reason == "signature mismatch"


def test_require_valid_raises_with_reason() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    att.liabilities_usd = 1.0
    att.content_hash = att.compute_hash()
    with pytest.raises(SettlementError, match="failed verification"):
        att.require_valid()


def test_sign_refuses_non_attestor_party() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor")
    with pytest.raises(SettlementError, match="signed by its attestor"):
        att.sign(ATTESTOR, party="impostor")


def test_negative_line_makes_unsound() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    att.liabilities = [LiabilityLine(creditor="acme", amount_usd=-1.0)]
    att.liabilities_usd = -1.0
    att.content_hash = att.compute_hash()
    assert att.verify().liabilities_sound is False


# == inclusion proofs =========================================================


def test_inclusion_proof_unknown_creditor_raises() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    with pytest.raises(SettlementError, match="not among the attested liabilities"):
        att.inclusion_proof("nobody")


def test_inclusion_proof_ambiguous_creditor_raises() -> None:
    att = attest_liabilities(
        "vendor", [LiabilityLine(creditor="acme", amount_usd=5.0), ("acme", 5.0)]
    )
    with pytest.raises(SettlementError, match="appears in 2 line items"):
        att.inclusion_proof("acme")


def test_inclusion_proof_verifies_and_binds() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0, "globex": 20.0, "initech": 30.0})
    att.sign(ATTESTOR)
    proof = att.inclusion_proof("globex")
    result = proof.verify(att, ATTESTOR)
    assert result.valid is True
    assert result.path_ok is True
    assert result.bound_ok is True
    assert ATTESTOR.key_id in result.signed_by or "vendor" in result.signed_by


def test_inclusion_proof_path_only_without_attestation() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0, "globex": 20.0})
    proof = att.inclusion_proof("acme")
    # No attestation supplied: only the Merkle path is checked, bound_ok defaults True.
    result = proof.verify()
    assert result.path_ok is True
    assert result.bound_ok is True
    assert result.valid is True


def test_inclusion_proof_tampered_amount_breaks_path() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0, "globex": 20.0})
    proof = att.inclusion_proof("acme")
    proof.amount_usd = 999.0  # leaf no longer hashes to the committed root
    result = proof.verify()
    assert result.path_ok is False
    assert "does not reconstruct" in (result.reason or "")


def test_inclusion_proof_root_lifted_from_other_attestation() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0, "globex": 20.0})
    other = attest_liabilities("vendor", {"acme": 99.0, "globex": 1.0})
    proof = att.inclusion_proof("acme")
    # Point the proof at a *different* attestation: the cited root no longer matches.
    result = proof.verify(other)
    assert result.bound_ok is False
    assert "does not bind the supplied attestation" in (result.reason or "")


def test_inclusion_proof_leaf_not_in_attestation() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0, "globex": 20.0})
    proof = att.inclusion_proof("acme")
    # Forge a self-consistent path/root for a leaf the attestation never committed.
    forged = MerkleStep(sibling="x", sibling_on_right=True)
    proof.creditor = "ghost"
    proof.path = [forged]
    proof.liabilities_root = proof._reconstructed_root()
    result = proof.verify(att)
    assert result.path_ok is True
    assert result.bound_ok is False
    assert "does not bind" in (result.reason or "") or "not a leaf" in (result.reason or "")


def test_inclusion_proof_require_valid_raises() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    proof = att.inclusion_proof("acme")
    proof.amount_usd = 5.0
    with pytest.raises(SettlementError, match="failed verification"):
        proof.require_valid()


def test_inclusion_proofs_one_per_line() -> None:
    att = attest_liabilities("vendor", {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
    proofs = att.inclusion_proofs()
    assert [p.creditor for p in proofs] == ["a", "b", "c", "d"]
    assert all(p.verify(att).valid for p in proofs)


def test_inclusion_proof_index_out_of_range_fails_path() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    proof = att.inclusion_proof("acme")
    proof.leaf_index = 7  # beyond leaf_count
    assert proof.verify().path_ok is False


# == completeness =============================================================


def test_check_completeness_refuses_tampered_attestation() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    att.liabilities_usd = 50.0
    att.content_hash = att.compute_hash()
    with pytest.raises(SettlementError, match="is tampered"):
        check_completeness(att, {"acme": 10.0})


def test_check_completeness_refuses_forged_signature() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor").sign(ATTESTOR)
    att.signatures[0].signature = FORGER.sign(att.content_hash)
    with pytest.raises(SettlementError, match="invalid attestor signature"):
        check_completeness(att, {"acme": 10.0}, verifier=ATTESTOR)


def test_check_completeness_pinpoints_omitted_creditor() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    proof = check_completeness(att, {"acme": 10.0, "ghost": 25.0})
    assert proof.complete is False
    assert proof.omitted_creditors == ["ghost"]
    breach = proof.breaches[0]
    assert breach.omitted is True
    assert breach.understatement_usd == 25.0
    assert proof.completed_usd == 35.0
    assert proof.understated_usd == 25.0
    assert proof.status == "incomplete"


def test_check_completeness_understated_not_omitted() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    proof = check_completeness(att, {"acme": 30.0})
    breach = proof.breaches[0]
    assert breach.omitted is False
    assert breach.understatement_usd == 20.0
    assert proof.completed_usd == 30.0


def test_check_completeness_complete_when_claims_covered() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0, "globex": 5.0})
    proof = check_completeness(att, {"acme": 10.0})
    assert proof.complete is True
    assert proof.completed_usd == 15.0
    assert proof.understated_usd == 0.0


def test_completeness_claims_coercion_records_and_negative() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    # SettlementRecord-free coercion: dict + LiabilityLine + tuple all merge by creditor.
    proof = check_completeness(
        att, [LiabilityLine(creditor="acme", amount_usd=6.0), ("acme", 6.0)]
    )
    # acme claims 6+6 = 12 vs attested 10 -> understated by 2.
    assert proof.breaches[0].understatement_usd == 2.0


def test_completeness_negative_claim_raises() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    with pytest.raises(SettlementError, match="fictitious negative claim"):
        check_completeness(att, {"acme": -1.0})


def test_completeness_bad_claim_item_raises() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    with pytest.raises(SettlementError, match="claims must be a mapping"):
        check_completeness(att, [object()])


def test_completeness_verify_reasons() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    proof = check_completeness(att, {"acme": 30.0})
    assert proof.verify().valid is True

    # Hash mismatch: mutate a hashed field without re-sealing.
    tampered = proof.model_copy(deep=True)
    tampered.attested_usd = 999.0
    r = tampered.verify()
    assert r.hash_ok is False
    assert r.reason == "content hash does not match the completeness facts"

    # Completed-total tampering caught even after re-sealing.
    forged = proof.model_copy(deep=True)
    forged.completed_usd = forged.attested_usd  # drop the understatement
    forged.content_hash = forged.compute_hash()
    r2 = forged.verify()
    assert r2.completeness_sound is False
    assert r2.reason == "completed total or omission breach does not re-derive"


def test_completeness_unsealed_reason() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    proof = check_completeness(att, {"acme": 30.0})
    proof.content_hash = ""
    r = proof.verify()
    assert r.reason == "completeness check is not sealed (no content hash)"


def test_completeness_sign_and_verify_with_verifier() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    proof = check_completeness(att, {"acme": 30.0}).sign(CREDITOR, party="acme")
    assert proof.verify(CREDITOR, require=["acme"]).valid is True
    # Forge the signature -> mismatch.
    proof.signatures[0].signature = FORGER.sign(proof.content_hash)
    assert proof.verify(CREDITOR).signatures_ok is False


def test_completeness_require_complete_raises() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    proof = check_completeness(att, {"acme": 10.0, "ghost": 5.0})
    with pytest.raises(SettlementError, match="omits or under-states"):
        proof.require_complete()


def test_completeness_require_complete_passes_when_complete() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    proof = check_completeness(att, {"acme": 10.0})
    assert proof.require_complete() is proof


def test_completeness_require_valid_raises() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    proof = check_completeness(att, {"acme": 30.0})
    proof.completed_usd = proof.attested_usd
    proof.content_hash = proof.compute_hash()
    with pytest.raises(SettlementError, match="failed verification"):
        proof.require_valid()


def test_completeness_sound_rejects_phantom_breach() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0})
    proof = check_completeness(att, {"acme": 10.0})  # complete, no breaches
    # Inject a "breach" with zero understatement — not a real omission.
    proof.breaches = [
        OmissionBreach(
            poster="vendor",
            creditor="acme",
            attested_usd=10.0,
            claimed_usd=10.0,
            understatement_usd=0.0,
        )
    ]
    proof.content_hash = proof.compute_hash()
    assert proof.verify().completeness_sound is False


# == proof-of-solvency ========================================================


def _custody(poster: str, reserves: float) -> object:
    return attest_custody(poster, reserves, custodian="custodian")


def test_prove_solvency_solvent_margin() -> None:
    cust = _custody("vendor", 100.0)
    liab = attest_liabilities("vendor", {"acme": 60.0})
    proof = prove_solvency(cust, liab)
    assert proof.solvent is True
    assert proof.margin_usd == 40.0
    assert proof.solvency_adjusted_held == 40.0
    assert proof.breach is None
    assert proof.status == "solvent"


def test_prove_solvency_insolvent_breach() -> None:
    cust = _custody("vendor", 40.0)
    liab = attest_liabilities("vendor", {"acme": 100.0})
    proof = prove_solvency(cust, liab)
    assert proof.insolvent is True
    assert proof.margin_usd == -60.0
    assert proof.solvency_adjusted_held == 0.0
    assert proof.breach is not None
    assert proof.breach.shortfall_usd == 60.0
    with pytest.raises(SettlementError, match="insolvent by"):
        proof.require_solvent()


def test_prove_solvency_require_solvent_passes() -> None:
    proof = prove_solvency(_custody("vendor", 100.0), attest_liabilities("vendor", {"a": 1.0}))
    assert proof.require_solvent() is proof


def test_prove_solvency_needs_explicit_poster_on_mismatch() -> None:
    cust = _custody("vendor", 100.0)
    liab = attest_liabilities("other", {"acme": 60.0})
    with pytest.raises(SettlementError, match="needs an explicit poster"):
        prove_solvency(cust, liab)


def test_prove_solvency_refuses_wrong_poster_custody() -> None:
    cust = _custody("someone-else", 100.0)
    liab = attest_liabilities("vendor", {"acme": 60.0})
    with pytest.raises(SettlementError, match="not the poster"):
        prove_solvency(cust, liab, poster="vendor")


def test_prove_solvency_refuses_tampered_liability() -> None:
    cust = _custody("vendor", 100.0)
    liab = attest_liabilities("vendor", {"acme": 60.0})
    liab.liabilities_usd = 5.0
    liab.content_hash = liab.compute_hash()
    with pytest.raises(SettlementError, match="refusing to read it as proof-of-liabilities"):
        prove_solvency(cust, liab)


def test_prove_solvency_refuses_tampered_custody() -> None:
    cust = _custody("vendor", 100.0)
    cust.reserves_usd = 5.0
    cust.content_hash = cust.compute_hash()
    liab = attest_liabilities("vendor", {"acme": 60.0})
    with pytest.raises(SettlementError, match="refusing to read it as proof-of-reserves"):
        prove_solvency(cust, liab)


def test_prove_solvency_refuses_forged_liability_signature() -> None:
    cust = _custody("vendor", 100.0)
    liab = attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor").sign(ATTESTOR)
    liab.signatures[0].signature = FORGER.sign(liab.content_hash)
    with pytest.raises(SettlementError, match="invalid attestor signature"):
        prove_solvency(cust, liab, verifier=ATTESTOR)


def test_prove_solvency_with_completeness_uses_completed_total() -> None:
    cust = _custody("vendor", 100.0)
    liab = attest_liabilities("vendor", {"acme": 60.0})
    comp = check_completeness(liab, {"acme": 60.0, "ghost": 50.0})
    proof = prove_solvency(cust, liab, completeness=comp)
    assert proof.completeness_adjusted is True
    assert proof.attested_liabilities_usd == 60.0
    assert proof.liabilities_usd == 110.0  # completed total
    assert proof.understated_usd == 50.0
    assert proof.insolvent is True


def test_prove_solvency_completeness_wrong_poster_refused() -> None:
    cust = _custody("vendor", 100.0)
    liab = attest_liabilities("vendor", {"acme": 60.0})
    other = attest_liabilities("other", {"acme": 60.0})
    comp = check_completeness(other, {"acme": 60.0})
    with pytest.raises(SettlementError, match="not the poster"):
        prove_solvency(cust, liab, completeness=comp, poster="vendor")


def test_prove_solvency_completeness_unrelated_attestation_refused() -> None:
    cust = _custody("vendor", 100.0)
    liab = attest_liabilities("vendor", {"acme": 60.0})
    other_liab = attest_liabilities("vendor", {"acme": 99.0})
    comp = check_completeness(other_liab, {"acme": 99.0})  # binds the *other* attestation
    with pytest.raises(SettlementError, match="does not bind the liability attestation"):
        prove_solvency(cust, liab, completeness=comp)


def test_prove_solvency_completeness_tampered_refused() -> None:
    cust = _custody("vendor", 100.0)
    liab = attest_liabilities("vendor", {"acme": 60.0})
    comp = check_completeness(liab, {"acme": 90.0})
    comp.completed_usd = comp.attested_usd  # drop the completion, re-seal
    comp.content_hash = comp.compute_hash()
    with pytest.raises(SettlementError, match="refusing to read it as a completed"):
        prove_solvency(cust, liab, completeness=comp)


def test_solvency_verify_reasons() -> None:
    proof = prove_solvency(_custody("vendor", 100.0), attest_liabilities("vendor", {"a": 60.0}))
    assert proof.verify().valid is True

    unsealed = proof.model_copy(deep=True)
    unsealed.content_hash = ""
    assert unsealed.verify().reason == "proof is not sealed (no content hash)"

    mismatch = proof.model_copy(deep=True)
    mismatch.poster = "x"
    assert mismatch.verify().reason == "content hash does not match the solvency facts"

    flipped = proof.model_copy(deep=True)
    flipped.margin_usd = -1.0  # margin no longer = reserves - liabilities
    flipped.content_hash = flipped.compute_hash()
    assert flipped.verify().reason == "solvency margin or insolvency breach does not re-derive"


def test_solvency_margin_sound_rejects_understated_completed() -> None:
    proof = prove_solvency(_custody("vendor", 100.0), attest_liabilities("vendor", {"a": 60.0}))
    forged = proof.model_copy(deep=True)
    # liabilities below the attestor's figure is impossible (completion only raises).
    forged.liabilities_usd = 50.0
    forged.attested_liabilities_usd = 60.0
    forged.margin_usd = 50.0
    forged.content_hash = forged.compute_hash()
    assert forged.verify().margin_sound is False


def test_solvency_margin_sound_rejects_flipped_breach() -> None:
    proof = prove_solvency(_custody("vendor", 40.0), attest_liabilities("vendor", {"a": 100.0}))
    assert proof.breach is not None
    forged = proof.model_copy(deep=True)
    forged.breach = None  # drop the breach but keep the negative margin
    forged.content_hash = forged.compute_hash()
    r = forged.verify()
    assert r.margin_sound is False


def test_solvency_breach_tampered_shortfall_caught() -> None:
    proof = prove_solvency(_custody("vendor", 40.0), attest_liabilities("vendor", {"a": 100.0}))
    forged = proof.model_copy(deep=True)
    forged.breach.shortfall_usd = 1.0  # lie about the magnitude
    forged.content_hash = forged.compute_hash()
    assert forged.verify().margin_sound is False


def test_solvency_sign_and_require_valid() -> None:
    proof = prove_solvency(_custody("vendor", 100.0), attest_liabilities("vendor", {"a": 60.0}))
    proof.sign(CUSTODIAN, party="custodian")
    assert proof.verify(CUSTODIAN, require=["custodian"]).valid is True
    assert proof.require_valid(CUSTODIAN, require=["custodian"]) is proof


def test_solvency_require_valid_raises() -> None:
    proof = prove_solvency(_custody("vendor", 100.0), attest_liabilities("vendor", {"a": 60.0}))
    proof.margin_usd = 5.0
    proof.content_hash = proof.compute_hash()
    with pytest.raises(SettlementError, match="failed verification"):
        proof.require_valid()


# == root commitment & equivocation ===========================================


def test_root_commitment_verify_unsigned_and_signed() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor")
    bare = att.root_commitment()
    assert bare.verify().committed is True
    assert bare.signed_by == []

    signed_att = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor").sign(ATTESTOR)
    rc = signed_att.root_commitment()
    assert rc.verify(ATTESTOR).signed_ok is True
    assert rc.signed_by == ["auditor"]


def test_root_commitment_uncommitted_reason() -> None:
    rc = RootCommitment(poster="vendor", attestor="auditor")
    r = rc.verify()
    assert r.committed is False
    assert r.reason == "commitment pins no root or content hash"


def test_root_commitment_forged_signature_refused() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor").sign(ATTESTOR)
    rc = att.root_commitment()
    rc.signature.signature = FORGER.sign(rc.liability_hash)
    r = rc.verify(ATTESTOR)
    assert r.signed_ok is False
    assert r.reason == "embedded attestor signature does not verify"


def test_root_commitment_conflicts_with() -> None:
    a = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor", as_of=_at(0))
    b = attest_liabilities("vendor", {"acme": 99.0}, attestor="auditor", as_of=_at(0))
    assert a.root_commitment().conflicts_with(b.root_commitment()) is True
    # Different instant -> distinct snapshots, not a conflict.
    c = attest_liabilities("vendor", {"acme": 99.0}, attestor="auditor", as_of=_at(1))
    assert a.root_commitment().conflicts_with(c.root_commitment()) is False


def _two_conflicting(as_of: datetime) -> tuple[LiabilityAttestation, LiabilityAttestation]:
    a = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor", as_of=as_of)
    b = attest_liabilities("vendor", {"acme": 99.0}, attestor="auditor", as_of=as_of)
    a.sign(ATTESTOR)
    b.sign(ATTESTOR)
    return a, b


def test_prove_equivocation_canonical_and_verifies() -> None:
    a, b = _two_conflicting(_at(0))
    proof = prove_equivocation(a, b, verifier=ATTESTOR, first_creditor="acme", second_creditor="x")
    # Canonical content-hash order: same conflict whichever way supplied.
    proof2 = prove_equivocation(b, a, verifier=ATTESTOR)
    assert proof.first_hash == proof2.first_hash
    assert proof.liabilities_gap_usd == 89.0
    assert set(proof.creditors) == {"acme", "x"}
    assert proof.verify(ATTESTOR).valid is True


def test_prove_equivocation_rejects_same_root() -> None:
    a = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor", as_of=_at(0)).sign(ATTESTOR)
    b = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor", as_of=_at(0)).sign(ATTESTOR)
    with pytest.raises(SettlementError, match="committed the same"):
        prove_equivocation(a, b)


def test_prove_equivocation_rejects_different_instant() -> None:
    a = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor", as_of=_at(0)).sign(ATTESTOR)
    b = attest_liabilities("vendor", {"acme": 99.0}, attestor="auditor", as_of=_at(1)).sign(ATTESTOR)
    with pytest.raises(SettlementError, match="different instants"):
        prove_equivocation(a, b)


def test_prove_equivocation_rejects_different_poster() -> None:
    a = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor", as_of=_at(0)).sign(ATTESTOR)
    b = attest_liabilities("other", {"acme": 99.0}, attestor="auditor", as_of=_at(0)).sign(ATTESTOR)
    with pytest.raises(SettlementError, match="different posters/attestors"):
        prove_equivocation(a, b)


def test_prove_equivocation_refuses_tampered() -> None:
    a, b = _two_conflicting(_at(0))
    a.liabilities_usd = 5.0
    a.content_hash = a.compute_hash()
    with pytest.raises(SettlementError, match="refusing\n?\\s*to found an equivocation"):
        prove_equivocation(a, b)


def test_prove_equivocation_refuses_unsigned_attestor() -> None:
    # No signatures: with a verifier the attestor isn't in signed_by.
    a = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor", as_of=_at(0))
    b = attest_liabilities("vendor", {"acme": 99.0}, attestor="auditor", as_of=_at(0))
    with pytest.raises(SettlementError, match="not signed by its attestor"):
        prove_equivocation(a, b, verifier=ATTESTOR)


def test_prove_equivocation_refuses_forged_signature() -> None:
    a, b = _two_conflicting(_at(0))
    a.signatures[0].signature = FORGER.sign(a.content_hash)
    with pytest.raises(SettlementError, match="invalid attestor signature"):
        prove_equivocation(a, b, verifier=ATTESTOR)


def test_equivocation_verify_reasons() -> None:
    a, b = _two_conflicting(_at(0))
    proof = prove_equivocation(a, b)

    unsealed = proof.model_copy(deep=True)
    unsealed.content_hash = ""
    assert unsealed.verify().reason == "equivocation proof is not sealed (no content hash)"

    mismatch = proof.model_copy(deep=True)
    mismatch.poster = "x"
    assert mismatch.verify().reason == "content hash does not match the equivocation facts"


def test_equivocation_require_valid_raises() -> None:
    a, b = _two_conflicting(_at(0))
    proof = prove_equivocation(a, b)
    proof.poster = "tampered"
    with pytest.raises(SettlementError, match="failed verification"):
        proof.require_valid()


def test_equivocation_reporter_signature() -> None:
    a, b = _two_conflicting(_at(0))
    # Reporter signs with the same HMAC key the verifier checks against, so its provenance
    # signature verifies alongside the embedded attestor signatures.
    proof = prove_equivocation(a, b, verifier=ATTESTOR).sign(ATTESTOR, party="acme")
    assert proof.verify(ATTESTOR, require=["acme"]).valid is True
    assert proof.verify(ATTESTOR, require=["nobody"]).signatures_ok is False


# == check_root_consistency ===================================================


def test_check_root_consistency_detects_equivocation() -> None:
    a, b = _two_conflicting(_at(0))
    report = check_root_consistency([("acme", a), ("globex", b)], verifier=ATTESTOR)
    assert report.consistent is False
    assert report.checked == 2
    assert report.keys == 1
    assert report.equivocating_posters == ["vendor"]
    with pytest.raises(SettlementError, match="signed conflicting"):
        report.require_consistent()


def test_check_root_consistency_consistent_when_agreeing() -> None:
    a = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor", as_of=_at(0)).sign(ATTESTOR)
    b = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor", as_of=_at(0)).sign(ATTESTOR)
    report = check_root_consistency([a, b], verifier=ATTESTOR)
    assert report.consistent is True
    assert report.equivocations == []
    assert report.require_consistent() is report


def test_check_root_consistency_excludes_tampered() -> None:
    a, b = _two_conflicting(_at(0))
    a.liabilities_usd = 1.0
    a.content_hash = a.compute_hash()  # now tampered -> inadmissible
    report = check_root_consistency([a, b], verifier=ATTESTOR)
    assert report.checked == 1  # only b admissible
    assert report.consistent is True


def test_check_root_consistency_bad_item_raises() -> None:
    with pytest.raises(SettlementError, match="items must be LiabilityAttestation"):
        check_root_consistency([object()])


def test_check_root_consistency_dict_without_attestation_raises() -> None:
    with pytest.raises(SettlementError, match="need an 'attestation'"):
        check_root_consistency([{"creditor": "acme"}])


def test_check_root_consistency_pair_without_attestation_raises() -> None:
    with pytest.raises(SettlementError, match="need a"):
        check_root_consistency([("acme", object())])


def test_check_root_consistency_accepts_dict_views() -> None:
    a, b = _two_conflicting(_at(0))
    report = check_root_consistency(
        [{"creditor": "acme", "attestation": a}, {"creditor": "globex", "attestation": b}],
        verifier=ATTESTOR,
    )
    assert report.consistent is False
    assert set(report.equivocations[0].creditors) == {"acme", "globex"}


# == discharges ===============================================================


def test_discharge_liability_negative_raises() -> None:
    with pytest.raises(SettlementError, match="releases a negative amount"):
        discharge_liability("vendor", "acme", -1.0)


def test_discharge_status_partial_and_empty() -> None:
    assert discharge_liability("vendor", "acme", 5.0).status == "partial"
    assert discharge_liability("vendor", "acme", 0.0).status == "empty"


def test_discharge_sign_refuses_non_creditor() -> None:
    d = discharge_liability("vendor", "acme", 5.0)
    with pytest.raises(SettlementError, match="signed by its creditor"):
        d.sign(CREDITOR, party="vendor")


def test_discharge_verify_reasons() -> None:
    d = discharge_liability("vendor", "acme", 5.0).sign(CREDITOR, party="acme")
    assert d.verify(CREDITOR, require=["acme"]).valid is True

    unsealed = d.model_copy(deep=True)
    unsealed.content_hash = ""
    assert unsealed.verify().reason == "discharge is not sealed (no content hash)"

    mismatch = d.model_copy(deep=True)
    mismatch.amount_usd = 99.0
    assert mismatch.verify().reason == "content hash does not match the discharge facts"

    forged = d.model_copy(deep=True)
    forged.signatures[0].signature = FORGER.sign(forged.content_hash)
    assert forged.verify(CREDITOR).reason == "signature mismatch"

    missing = d.model_copy(deep=True)
    assert missing.verify(CREDITOR, require=["other"]).signatures_ok is False


def test_discharge_negative_amount_unsound() -> None:
    d = discharge_liability("vendor", "acme", 5.0)
    d.amount_usd = -1.0
    d.content_hash = d.compute_hash()
    r = d.verify()
    assert r.sound is False
    assert r.reason == "discharge releases a negative amount"


def test_discharge_require_valid_raises() -> None:
    d = discharge_liability("vendor", "acme", 5.0)
    d.amount_usd = -1.0
    d.content_hash = d.compute_hash()
    with pytest.raises(SettlementError, match="failed verification"):
        d.require_valid()


def test_coerce_discharges_bad_item_raises() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(1))
    with pytest.raises(SettlementError, match="discharges must be Discharge or dict"):
        check_history_consistency([s0, s1], discharges=[object()])


# == check_history_consistency ================================================


def test_history_single_snapshot_yields_no_proof() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    report = check_history_consistency([s0])
    assert report.chains == 0
    assert report.consistent is True
    assert report.checked == 1


def test_history_monotone_clean() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0, "globex": 5.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 12.0, "globex": 5.0}, as_of=_at(1))
    report = check_history_consistency([s0, s1])
    assert report.consistent is True
    assert report.chains == 1
    proof = report.proofs[0]
    assert proof.monotone is True
    assert proof.snapshot_count == 2
    assert proof.status == "consistent"


def test_history_unexplained_drop_breaches() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 4.0}, as_of=_at(1))
    report = check_history_consistency([s0, s1])
    assert report.consistent is False
    assert report.breaching_posters == ["vendor"]
    proof = report.proofs[0]
    breach = proof.breaches[0]
    assert breach.creditor == "acme"
    assert breach.dropped_usd == 6.0
    assert breach.discharged_usd == 0.0
    assert breach.unexplained_usd == 6.0
    assert proof.unexplained_usd == 6.0
    with pytest.raises(SettlementError, match="not monotone"):
        proof.require_monotone()
    with pytest.raises(SettlementError, match="dropped an obligation"):
        report.require_consistent()


def test_history_drop_explained_by_discharge() -> None:
    # No verifier: snapshots are admissible unsigned and the discharge's hash/soundness still
    # gate it, so a legitimate in-window release explains the drop.
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 4.0}, as_of=_at(2))
    d = discharge_liability("vendor", "acme", 6.0, as_of=_at(1))
    report = check_history_consistency([s0, s1], discharges=d)
    assert report.consistent is True
    proof = report.proofs[0]
    assert proof.breaches == []
    # Only the consumed discharge is embedded.
    assert len(proof.discharges) == 1


def test_history_partial_discharge_leaves_residue() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 2.0}, as_of=_at(2))
    d = discharge_liability("vendor", "acme", 5.0, as_of=_at(1))
    report = check_history_consistency([s0, s1], discharges=[d])
    breach = report.proofs[0].breaches[0]
    assert breach.dropped_usd == 8.0
    assert breach.discharged_usd == 5.0
    assert breach.unexplained_usd == 3.0


def test_history_forged_discharge_does_not_explain() -> None:
    # Snapshots self-attested and signed under party "vendor" so they pass the verifier gate;
    # the discharge is forged so it cannot explain the drop.
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0)).sign(CREDITOR, party="vendor")
    s1 = attest_liabilities("vendor", {"acme": 4.0}, as_of=_at(2)).sign(CREDITOR, party="vendor")
    d = discharge_liability("vendor", "acme", 6.0, as_of=_at(1)).sign(CREDITOR, party="acme")
    d.signatures[0].signature = FORGER.sign(d.content_hash)
    report = check_history_consistency([s0, s1], discharges=[d], verifier=CREDITOR)
    # Forged signature -> discharge ignored -> drop unexplained.
    assert report.consistent is False
    assert report.proofs[0].breaches[0].unexplained_usd == 6.0


def test_history_out_of_window_discharge_ignored() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 4.0}, as_of=_at(2))
    # Discharge dated *before* the earlier snapshot -> outside (lo, hi] window.
    d = discharge_liability("vendor", "acme", 6.0, as_of=_at(-1))
    report = check_history_consistency([s0, s1], discharges=[d])
    assert report.consistent is False


def test_history_chain_linked_flag() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 12.0}, as_of=_at(1), prior=s0)
    report = check_history_consistency([s0, s1])
    proof = report.proofs[0]
    assert proof.linked is True
    assert proof.require_linked() is proof


def test_history_require_linked_raises_when_unlinked() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 12.0}, as_of=_at(1))  # no prior=
    report = check_history_consistency([s0, s1])
    proof = report.proofs[0]
    assert proof.linked is False
    with pytest.raises(SettlementError, match="not a contiguous hash-linked chain"):
        proof.require_linked()


def test_history_excludes_tampered_snapshot() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 4.0}, as_of=_at(1))
    s0.liabilities_usd = 1.0
    s0.content_hash = s0.compute_hash()  # tampered -> inadmissible
    report = check_history_consistency([s0, s1])
    # Only one admissible snapshot -> no chain to walk.
    assert report.checked == 1
    assert report.chains == 0


def test_history_same_instant_dedup_keeps_one() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s0b = attest_liabilities("vendor", {"acme": 99.0}, as_of=_at(0))  # same instant
    s1 = attest_liabilities("vendor", {"acme": 12.0}, as_of=_at(1))
    report = check_history_consistency([s0, s0b, s1])
    # Two distinct instants -> a walkable chain of length 2.
    assert report.chains == 1
    assert report.proofs[0].snapshot_count == 2


def test_history_proof_verify_and_reasons() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0)).sign(ATTESTOR)
    s1 = attest_liabilities("vendor", {"acme": 4.0}, as_of=_at(1)).sign(ATTESTOR)
    report = check_history_consistency([s0, s1], verifier=ATTESTOR)
    proof = report.proofs[0]
    assert proof.verify(ATTESTOR).valid is True

    unsealed = proof.model_copy(deep=True)
    unsealed.content_hash = ""
    assert unsealed.verify().reason == "history proof is not sealed (no content hash)"

    mismatch = proof.model_copy(deep=True)
    mismatch.poster = "x"
    assert mismatch.verify().reason == "content hash does not match the history facts"


def test_history_proof_chain_link_status_must_rederive() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 12.0}, as_of=_at(1))  # unlinked
    report = check_history_consistency([s0, s1])
    proof = report.proofs[0]
    forged = proof.model_copy(deep=True)
    forged.chain_linked = True  # claim linked when snapshots are not
    forged.content_hash = forged.compute_hash()
    r = forged.verify()
    assert r.chain_ok is False
    assert r.reason == "the recorded chain-link status does not re-derive from the snapshots"


def test_history_proof_dropped_breach_resurfaces() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 4.0}, as_of=_at(1))
    report = check_history_consistency([s0, s1])
    proof = report.proofs[0]
    forged = proof.model_copy(deep=True)
    forged.breaches = []  # drop the recorded breach
    forged.content_hash = forged.compute_hash()
    r = forged.verify()
    assert r.monotone_sound is False
    assert r.reason == "the monotonicity breaches do not re-derive from the snapshots/discharges"


def test_history_proof_require_valid_raises() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 4.0}, as_of=_at(1))
    proof = check_history_consistency([s0, s1]).proofs[0]
    proof.poster = "tampered"
    with pytest.raises(SettlementError, match="failed verification"):
        proof.require_valid()


def test_history_proof_require_monotone_passes_when_clean() -> None:
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0))
    s1 = attest_liabilities("vendor", {"acme": 12.0}, as_of=_at(1))
    proof = check_history_consistency([s0, s1]).proofs[0]
    assert proof.require_monotone() is proof


def test_history_back_dated_snapshot_fails_verify() -> None:
    # Build a proof then mutate a snapshot's as_of to break strict ordering.
    s0 = attest_liabilities("vendor", {"acme": 10.0}, as_of=_at(0)).sign(ATTESTOR)
    s1 = attest_liabilities("vendor", {"acme": 12.0}, as_of=_at(1)).sign(ATTESTOR)
    proof = HistoryConsistencyProof(
        poster="vendor",
        attestor="vendor",
        snapshots=[s0, s1],
        head_hash=s1.content_hash,
        span_from=s0.as_of,
        span_to=s1.as_of,
    ).seal()
    assert proof.verify().snapshots_ok is True
    # Now equalise instants so the strict-increase check fails.
    proof.snapshots[1].as_of = proof.snapshots[0].as_of
    proof.snapshots[1].content_hash = proof.snapshots[1].compute_hash()
    r = proof.verify()
    assert r.snapshots_ok is False


# == wire round-trips =========================================================


def test_wire_round_trips_preserve_verdicts() -> None:
    att = attest_liabilities("vendor", {"acme": 10.0}, attestor="auditor").sign(ATTESTOR)
    restored_att = LiabilityAttestation.from_wire(att.to_wire())
    assert restored_att.verify(ATTESTOR).valid is True

    comp = check_completeness(att, {"acme": 30.0})
    assert CompletenessProof.from_wire(comp.to_wire()).verify().valid is True

    proof = prove_solvency(_custody("vendor", 100.0), attest_liabilities("vendor", {"a": 60.0}))
    assert SolvencyProof.from_wire(proof.to_wire()).verify().valid is True

    a, b = _two_conflicting(_at(0))
    eq = prove_equivocation(a, b, verifier=ATTESTOR)
    assert EquivocationProof.from_wire(eq.to_wire()).verify(ATTESTOR).valid is True

    ip = att.inclusion_proof("acme")
    assert InclusionProof.from_wire(ip.to_wire()).verify(att).valid is True

    d = discharge_liability("vendor", "acme", 5.0, as_of=_at(1)).sign(CREDITOR, party="acme")
    assert Discharge.from_wire(d.to_wire()).verify(CREDITOR).valid is True
