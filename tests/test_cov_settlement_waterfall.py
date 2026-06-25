"""Targeted coverage for vincio.settlement.waterfall.

Drives the uncovered error/edge branches of the seniority schedule and the insolvency
waterfall through the real API: re-sign idempotence, signature-mismatch reasons,
distribution-soundness rejection of every tampered field, set-off binding refusals, and the
resolve_insolvency refusals for an invalid signature / wrong-poster / zero-net set-off.

Everything is deterministic and offline (HMAC signing, no provider model calls needed for the
math itself; a MockProvider-backed app exercises the book path).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from vincio import (
    attest_custody,
    attest_liabilities,
    build_seniority_schedule,
    build_set_off_statement,
    resolve_insolvency,
)
from vincio.core.errors import SettlementError
from vincio.security.audit import HMACSigner
from vincio.settlement.setoff import SetOffStatement
from vincio.settlement.waterfall import (
    CreditorRecovery,
    InsolvencyResolution,
    SenioritySchedule,
    SeniorityTranche,
    _apply_set_off,
)

FABRIC = "fabric-secret"
VERIFIER = HMACSigner(FABRIC, key_id="any")
FORGER = HMACSigner("other-secret", key_id="any")

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _signer(party: str) -> HMACSigner:
    return HMACSigner(FABRIC, key_id=party)


def _reserves(amount: float = 60.0, *, poster: str = "vendor") -> object:
    return attest_custody(poster, {"omnibus": amount}, custodian="custodian")


def _owed(mapping: dict[str, float] | None = None, *, poster: str = "vendor") -> object:
    return attest_liabilities(
        poster, mapping or {"bank": 50.0, "acme": 30.0, "globex": 20.0}, attestor="auditor"
    )


def _signed_setoff(statement: SetOffStatement) -> SetOffStatement:
    statement.sign(_signer(statement.poster), party=statement.poster)
    statement.sign(_signer(statement.creditor), party=statement.creditor)
    return statement


# == SenioritySchedule signing & verification ================================


def test_resign_replaces_signature_without_resealing():
    # Covers the sign() branch where content_hash is already set (seal is skipped) and the
    # re-sign-replaces-prior invariant.
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]])
    hash_before = schedule.content_hash
    assert hash_before
    schedule.sign(_signer("vendor"), party="vendor")
    schedule.sign(_signer("vendor"), party="vendor")  # second sign for same party
    assert schedule.content_hash == hash_before  # not re-sealed
    assert schedule.signed_by == ["vendor"]  # prior replaced, not accumulated


def test_sign_seals_unsealed_schedule():
    # An unsealed schedule (no content_hash) is sealed by sign() before the signature is taken.
    schedule = SenioritySchedule(poster="vendor", tranches=[SeniorityTranche(creditors=["bank"])])
    assert schedule.content_hash == ""
    schedule.sign(_signer("vendor"), party="vendor")
    assert schedule.content_hash  # now sealed
    assert schedule.verify(VERIFIER, require=["vendor"]).valid


def test_schedule_audit_details_and_roundtrip():
    schedule = build_seniority_schedule(
        "vendor", [SeniorityTranche(rank=0, creditors=["bank"], label="secured")]
    )
    schedule.sign(_signer("vendor"), party="vendor")
    details = schedule.audit_details()
    assert details["poster"] == "vendor"
    assert details["ranks"] == [0]
    assert details["signed_by"] == ["vendor"]
    restored = SenioritySchedule.from_wire(schedule.to_wire())
    assert restored.content_hash == schedule.content_hash
    assert restored.rank_of("bank") == 0


def test_coercion_rejects_unrecognized_item():
    with pytest.raises(SettlementError, match="must be SeniorityTranche"):
        build_seniority_schedule("vendor", [12345])


def test_coercion_single_string_is_one_creditor_tranche():
    schedule = build_seniority_schedule("vendor", ["bank", "acme"])
    assert schedule.rank_of("bank") == 0
    assert schedule.rank_of("acme") == 1


def test_coercion_dict_tranche():
    schedule = build_seniority_schedule(
        "vendor", [{"rank": 3, "creditors": ["bank"], "label": "junior"}]
    )
    assert schedule.rank_of("bank") == 3
    assert schedule.tranches[0].label == "junior"


def test_verify_reports_missing_required_signature():
    # require= names a party with no verified signature -> signatures_ok False, specific reason.
    schedule = build_seniority_schedule("vendor", [["bank"]])
    schedule.sign(_signer("vendor"), party="vendor")
    result = schedule.verify(VERIFIER, require=["bank"])
    assert result.valid is False
    assert result.signatures_ok is False
    assert result.reason is not None
    assert "missing/invalid signatures for ['bank']" in result.reason


def test_verify_reports_signature_mismatch_with_forger():
    # A signature that does not verify against the content hash -> "signature mismatch".
    schedule = build_seniority_schedule("vendor", [["bank"]])
    schedule.sign(FORGER, party="vendor")
    result = schedule.verify(VERIFIER)
    assert result.valid is False
    assert result.signatures_ok is False
    assert result.signed_by == []  # forger never lands as verified
    assert result.reason == "signature mismatch"


def test_schedule_require_valid_raises_on_malformed():
    # A creditor placed in two tranches makes the schedule not well-formed; require_valid raises
    # SettlementError carrying the malformed reason.
    schedule = SenioritySchedule(
        poster="vendor",
        tranches=[
            SeniorityTranche(rank=0, creditors=["bank"]),
            SeniorityTranche(rank=1, creditors=["bank"]),
        ],
    ).seal()
    with pytest.raises(SettlementError, match="failed verification"):
        schedule.require_valid()


# == CreditorRecovery / WaterfallTranche derived state =======================


def test_creditor_recovery_set_off_flag_true():
    r = CreditorRecovery(creditor="acme", claim_usd=10.0, set_off_usd=4.0)
    assert r.set_off is True


def test_creditor_recovery_set_off_flag_false_below_tolerance():
    r = CreditorRecovery(creditor="acme", claim_usd=10.0, set_off_usd=0.0)
    assert r.set_off is False


# == InsolvencyResolution: pari-passu and seniority math =====================


def test_partly_funded_tranche_splits_pari_passu():
    # 60 reserves, 100 owed, one tranche -> each claim recovers 60% pro rata.
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    assert res.insolvent
    bank = res.recovery_of("bank")
    acme = res.recovery_of("acme")
    assert bank is not None and acme is not None
    assert bank.recovery_usd == pytest.approx(36.0)
    assert acme.recovery_usd == pytest.approx(24.0)
    assert res.distributed_usd == pytest.approx(60.0)
    assert res.shortfall_usd == pytest.approx(40.0)


def test_senior_tranche_paid_in_full_before_junior():
    # 60 reserves: senior bank (50) paid in full, junior acme (30) gets the remaining 10.
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]])
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}), schedule)
    bank = res.recovery_of("bank")
    acme = res.recovery_of("acme")
    assert bank is not None and acme is not None
    assert bank.recovery_usd == pytest.approx(50.0)
    assert bank.made_whole
    assert acme.recovery_usd == pytest.approx(10.0)
    assert acme.shortfall_usd == pytest.approx(20.0)
    assert res.shortfall_bearers == ["acme"]


def test_surplus_when_reserves_exceed_liabilities():
    res = resolve_insolvency(_reserves(200.0), _owed({"bank": 50.0, "acme": 30.0}))
    assert res.solvent
    assert res.fully_recovered
    assert res.surplus_usd == pytest.approx(120.0)
    assert res.shortfall_bearers == []
    assert res.status == "solvent"


# == InsolvencyResolution re-sign + verify reasons ===========================


def test_resolution_sign_seals_unsealed():
    res = resolve_insolvency(_reserves(60.0), _owed())
    res.content_hash = ""  # un-seal so sign() must seal it first
    res.sign(_signer("auditor"), party="auditor")
    assert res.content_hash
    assert res.verify(VERIFIER, require=["auditor"]).valid


def test_resolution_audit_details_and_roundtrip_and_recovery_rate():
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    details = res.audit_details()
    assert details["status"] == "resolved"
    assert details["distributed_usd"] == pytest.approx(60.0)
    assert details["shortfall_bearers"] == ["acme", "bank"]
    assert res.recovery_rate == pytest.approx(0.6)  # 60 distributed of 100 owed
    restored = InsolvencyResolution.from_wire(res.to_wire())
    assert restored.content_hash == res.content_hash
    assert restored.recovery_of("bank").recovery_usd == pytest.approx(36.0)


def test_resolution_resign_skips_reseal():
    res = resolve_insolvency(_reserves(60.0), _owed())
    hash_before = res.content_hash
    res.sign(_signer("auditor"), party="auditor")
    res.sign(_signer("auditor"), party="auditor")
    assert res.content_hash == hash_before
    assert res.signed_by == ["auditor"]


def test_resolution_verify_missing_required_signature():
    res = resolve_insolvency(_reserves(60.0), _owed())
    res.sign(_signer("auditor"), party="auditor")
    result = res.verify(VERIFIER, require=["bank"])
    assert result.valid is False
    assert result.reason is not None
    assert "missing/invalid signatures for ['bank']" in result.reason


def test_resolution_verify_signature_mismatch():
    res = resolve_insolvency(_reserves(60.0), _owed())
    res.sign(FORGER, party="auditor")
    result = res.verify(VERIFIER)
    assert result.valid is False
    assert result.signatures_ok is False
    assert result.reason == "signature mismatch"


def test_resolution_verify_unsealed_reports_not_sealed():
    res = resolve_insolvency(_reserves(60.0), _owed())
    res.content_hash = ""  # un-seal
    result = res.verify()
    assert result.valid is False
    assert result.hash_ok is False
    assert result.reason == "resolution is not sealed (no content hash)"


def test_resolution_verify_tampered_hash_mismatch():
    res = resolve_insolvency(_reserves(60.0), _owed())
    res.as_of = res.as_of.replace(year=2030)  # change a bound fact without resealing
    result = res.verify()
    assert result.valid is False
    assert result.hash_ok is False
    assert result.reason == "content hash does not match the resolution facts"


# == distribution-soundness: every tampered figure rejected ==================


def _tampered(res: InsolvencyResolution, **changes: object) -> InsolvencyResolution:
    """Apply field changes then re-seal so hash_ok passes but the math is wrong."""
    for field, value in changes.items():
        setattr(res, field, value)
    return res.seal()


def test_distribution_unsound_on_negative_reserves():
    res = resolve_insolvency(_reserves(60.0), _owed())
    _tampered(res, reserves_usd=-1.0)
    result = res.verify()
    assert result.distribution_sound is False
    assert result.reason == "the seniority waterfall does not re-derive from the recorded recoveries"


def test_distribution_unsound_on_overstated_distributed():
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    _tampered(res, distributed_usd=res.distributed_usd + 5.0)
    assert res.verify().distribution_sound is False


def test_distribution_unsound_on_understated_shortfall():
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    _tampered(res, shortfall_usd=0.0)
    assert res.verify().distribution_sound is False


def test_distribution_unsound_on_wrong_surplus():
    res = resolve_insolvency(_reserves(200.0), _owed({"bank": 50.0}))
    _tampered(res, surplus_usd=res.surplus_usd + 10.0)
    assert res.verify().distribution_sound is False


def test_distribution_unsound_on_forged_liability_total():
    # liabilities_usd no longer equals the sum of recorded claims.
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}))
    _tampered(res, liabilities_usd=res.liabilities_usd + 1.0)
    assert res.verify().distribution_sound is False


def test_distribution_unsound_on_overstated_recovery():
    # An over-stated per-creditor recovery is caught by _recoveries_match.
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    res.recoveries[0].recovery_usd += 5.0
    res.seal()
    assert res.verify().distribution_sound is False


def test_distribution_unsound_on_reranked_creditor():
    # Re-rank a creditor away from what the math implies -> _recoveries_match (rank) fails.
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    res.recoveries[0].rank = 99
    res.seal()
    assert res.verify().distribution_sound is False


def test_distribution_unsound_on_dropped_recovery():
    # Removing a recovery makes length mismatch / liability-total mismatch -> unsound.
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    res.recoveries.pop()
    res.seal()
    assert res.verify().distribution_sound is False


def test_distribution_unsound_on_tampered_tranche_summary():
    # The recoveries re-derive but a tranche roll-up was edited -> _tranches_match fails.
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    res.tranches[0].paid_usd += 7.0
    res.seal()
    assert res.verify().distribution_sound is False


def test_distribution_unsound_on_attested_below_completed_floor():
    # gross floor (no set-off) below attested figure -> unsound.
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}))
    _tampered(res, attested_liabilities_usd=res.liabilities_usd + 100.0)
    assert res.verify().distribution_sound is False


# == schedule binding on verify ==============================================


def test_verify_schedule_bound_rank_mismatch():
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]])
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}), schedule)
    # A schedule that ranks acme senior instead -> a creditor's rank no longer matches.
    other = build_seniority_schedule("vendor", [["acme"], ["bank"]])
    result = res.verify(schedule=other)
    assert result.schedule_bound is False
    assert result.reason is not None
    assert "rank does not match" in result.reason or "does not bind" in result.reason


def test_verify_schedule_bound_hash_mismatch():
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]])
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}), schedule)
    res.schedule_hash = "deadbeef"
    res.seal()
    result = res.verify(schedule=schedule)
    assert result.schedule_bound is False
    assert result.reason == "resolution does not bind the supplied schedule's hash"


def test_verify_schedule_bound_invalid_schedule():
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]])
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}), schedule)
    schedule.content_hash = ""  # un-seal the supplied schedule
    result = res.verify(schedule=schedule)
    assert result.schedule_bound is False
    assert result.reason is not None
    assert "bound schedule failed verification" in result.reason


# == set-off binding on verify ===============================================


def _setoff_resolution() -> tuple[InsolvencyResolution, SetOffStatement]:
    statement = _signed_setoff(build_set_off_statement("vendor", "acme", 30.0, 12.0))
    res = resolve_insolvency(
        _reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}), set_off=[statement], verifier=VERIFIER
    )
    return res, statement


def test_set_off_folds_net_claim_before_waterfall():
    res, statement = _setoff_resolution()
    acme = res.recovery_of("acme")
    assert acme is not None
    assert acme.gross_claim_usd == pytest.approx(30.0)
    assert acme.set_off_usd == pytest.approx(12.0)
    assert acme.claim_usd == pytest.approx(18.0)  # net = max(0, 30 - 12)
    assert res.set_off is True
    # A correctly-bound resolution verifies with its statement supplied.
    assert res.verify(VERIFIER, set_off=[statement]).valid


def test_creditor_in_debit_recovers_nothing():
    # acme owes the estate more than it is owed -> net claim floors at zero, it recovers nothing.
    statement = _signed_setoff(build_set_off_statement("vendor", "acme", 30.0, 50.0))
    res = resolve_insolvency(
        _reserves(100.0),
        _owed({"bank": 50.0, "acme": 30.0}),
        set_off=[statement],
        verifier=VERIFIER,
    )
    acme = res.recovery_of("acme")
    assert acme is not None
    assert acme.claim_usd == pytest.approx(0.0)  # floored
    assert acme.recovery_usd == pytest.approx(0.0)
    assert acme.set_off_usd == pytest.approx(30.0)  # capped at the gross
    assert res.recovery_of("bank").recovery_usd == pytest.approx(50.0)  # made whole
    assert res.verify(VERIFIER, set_off=[statement]).valid


def test_set_off_resolution_rejects_restated_net_liabilities():
    # With a set-off bound, restating the net liabilities_usd (and a matching claim so the
    # sum-of-claims check is satisfied) still fails distribution-soundness: the bound net no longer
    # reconciles against gross - set_off / the re-derived waterfall.
    res, statement = _setoff_resolution()
    res.liabilities_usd = res.liabilities_usd + 5.0
    res.recoveries[0].claim_usd = res.recoveries[0].claim_usd + 5.0
    res.seal()
    assert res.verify().distribution_sound is False


def test_verify_set_off_wrong_statement_set():
    res, _ = _setoff_resolution()
    other = _signed_setoff(build_set_off_statement("vendor", "acme", 30.0, 5.0))
    result = res.verify(VERIFIER, set_off=[other])
    assert result.set_off_bound is False
    assert result.reason == "resolution does not bind exactly the supplied set-off statements"


def test_verify_set_off_creditor_has_no_recovery():
    # Bind the right hash but supply a statement whose creditor was never in the resolution.
    res, statement = _setoff_resolution()
    rogue = _signed_setoff(build_set_off_statement("vendor", "ghost", 0.0, 0.0))
    # Force the resolution to "bind" the rogue's hash so the exact-set check passes.
    res.set_off_hashes = sorted([rogue.content_hash])
    res.seal()
    result = res.verify(VERIFIER, set_off=[rogue])
    assert result.set_off_bound is False
    assert result.reason is not None
    assert "has no recovery to set off" in result.reason


def test_verify_set_off_gross_mismatch():
    res, statement = _setoff_resolution()
    # Same poster/creditor/hash binding, but gross owed disagrees with the recorded claim.
    bad = _signed_setoff(build_set_off_statement("vendor", "acme", 999.0, 12.0))
    res.set_off_hashes = sorted([bad.content_hash])
    res.seal()
    result = res.verify(VERIFIER, set_off=[bad])
    assert result.set_off_bound is False
    assert result.reason == "a set-off statement's gross does not match the recorded claim"


def test_verify_set_off_netted_amount_mismatch():
    res, statement = _setoff_resolution()
    # Gross matches the recorded 30, but the netted amount differs from the recorded 12.
    bad = _signed_setoff(build_set_off_statement("vendor", "acme", 30.0, 25.0))
    res.set_off_hashes = sorted([bad.content_hash])
    res.seal()
    result = res.verify(VERIFIER, set_off=[bad])
    assert result.set_off_bound is False
    assert result.reason == "a set-off statement nets a different amount than recorded"


def test_verify_set_off_statement_fails_verification():
    res, statement = _setoff_resolution()
    forged = build_set_off_statement("vendor", "acme", 30.0, 12.0)
    forged.sign(FORGER, party="vendor")  # only one, forged signature
    res.set_off_hashes = sorted([forged.content_hash])
    res.seal()
    result = res.verify(VERIFIER, set_off=[forged])
    assert result.set_off_bound is False
    assert result.reason is not None
    assert "bound set-off statement failed verification" in result.reason


def test_set_off_sound_rejects_inflated_set_off_total():
    res, statement = _setoff_resolution()
    _tampered(res, set_off_usd=res.set_off_usd + 5.0)
    assert res.verify().distribution_sound is False


def test_set_off_sound_rejects_tampered_gross_total():
    res, statement = _setoff_resolution()
    _tampered(res, gross_liabilities_usd=res.gross_liabilities_usd + 5.0)
    assert res.verify().distribution_sound is False


def test_set_off_sound_rejects_negative_gross():
    res, statement = _setoff_resolution()
    res.recovery_of("acme").gross_claim_usd = -1.0
    res.seal()
    assert res.verify().distribution_sound is False


def test_set_off_sound_rejects_inconsistent_net_total():
    # gross and per-creditor set-off totals reconcile, but liabilities_usd (the net) is restated
    # so it no longer equals gross - set_off -> the final reconcile check in _set_off_sound fails.
    res, statement = _setoff_resolution()
    res.liabilities_usd = res.liabilities_usd + 3.0
    # keep the recorded claims summing to the new total so the *other* sum-check also sees a
    # mismatch is avoided: instead nudge one recorded claim to match the inflated total.
    res.recoveries[0].claim_usd = res.recoveries[0].claim_usd + 3.0
    res.seal()
    assert res.verify().distribution_sound is False


def test_distribution_unsound_on_negative_attested():
    res = resolve_insolvency(_reserves(60.0), _owed())
    _tampered(res, attested_liabilities_usd=-5.0)
    assert res.verify().distribution_sound is False


def test_set_off_sound_rejects_broken_net_identity():
    # claim_usd no longer equals max(0, gross - set_off).
    res, statement = _setoff_resolution()
    res.recovery_of("acme").set_off_usd = 0.0  # claim still 18 but gross 30 -> identity broken
    res.seal()
    assert res.verify().distribution_sound is False


# == require_valid raises ====================================================


def test_resolution_require_valid_raises_on_tamper():
    res = resolve_insolvency(_reserves(60.0), _owed())
    res.recoveries[0].recovery_usd += 10.0
    res.seal()
    with pytest.raises(SettlementError, match="failed verification"):
        res.require_valid()


def test_require_fully_recovered_raises_on_shortfall():
    res = resolve_insolvency(_reserves(40.0), _owed({"bank": 50.0, "acme": 30.0}))
    with pytest.raises(SettlementError, match="unrecovered"):
        res.require_fully_recovered()


def test_require_fully_recovered_returns_self_when_solvent():
    res = resolve_insolvency(_reserves(200.0), _owed({"bank": 50.0}))
    assert res.require_fully_recovered() is res


# == resolve_insolvency refusal paths ========================================


def test_resolve_refuses_set_off_for_wrong_poster():
    statement = _signed_setoff(build_set_off_statement("other", "acme", 30.0, 12.0))
    with pytest.raises(SettlementError, match="not the poster"):
        resolve_insolvency(
            _reserves(60.0),
            _owed({"bank": 50.0, "acme": 30.0}),
            set_off=[statement],
            verifier=VERIFIER,
        )


def test_resolve_refuses_set_off_double_netting_same_creditor():
    s1 = _signed_setoff(build_set_off_statement("vendor", "acme", 30.0, 5.0))
    s2 = _signed_setoff(build_set_off_statement("vendor", "acme", 30.0, 7.0))
    with pytest.raises(SettlementError, match="more than one statement"):
        resolve_insolvency(
            _reserves(60.0),
            _owed({"bank": 50.0, "acme": 30.0}),
            set_off=[s1, s2],
            verifier=VERIFIER,
        )


def test_resolve_refuses_overstated_set_off():
    statement = _signed_setoff(build_set_off_statement("vendor", "acme", 999.0, 12.0))
    with pytest.raises(SettlementError, match="over-stated set-off"):
        resolve_insolvency(
            _reserves(60.0),
            _owed({"bank": 50.0, "acme": 30.0}),
            set_off=[statement],
            verifier=VERIFIER,
        )


def test_apply_set_off_records_zero_net_statement():
    # owing 0 -> nothing nets, but the statement is still recorded (auditable) and the claim
    # is unchanged. Exercises the `applied <= tolerance` continue branch directly.
    claims = {"acme": 30.0, "bank": 50.0}
    statement = _signed_setoff(build_set_off_statement("vendor", "acme", 30.0, 0.0))
    netted, by_creditor, hashes = _apply_set_off(
        claims, [statement], "vendor", verifier=VERIFIER
    )
    assert netted == {"acme": 30.0, "bank": 50.0}  # unchanged
    assert by_creditor == {}  # nothing netted
    assert hashes == [statement.content_hash]  # still recorded


def test_resolve_refuses_schedule_with_invalid_signature():
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]])
    schedule.sign(FORGER, party="vendor")  # signature won't verify under VERIFIER
    with pytest.raises(SettlementError, match="invalid signature"):
        resolve_insolvency(
            _reserves(60.0),
            _owed({"bank": 50.0, "acme": 30.0}),
            schedule,
            verifier=VERIFIER,
        )


def test_resolve_refuses_malformed_schedule():
    schedule = SenioritySchedule(
        poster="vendor",
        tranches=[
            SeniorityTranche(rank=0, creditors=["bank"]),
            SeniorityTranche(rank=0, creditors=["acme"]),  # duplicate rank
        ],
    ).seal()
    with pytest.raises(SettlementError, match="is invalid"):
        resolve_insolvency(_reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}), schedule)


def test_resolve_refuses_solvency_proof_with_invalid_signature():
    from vincio.settlement.solvency import prove_solvency

    custody = _reserves(60.0)
    liab = _owed({"bank": 50.0, "acme": 30.0})
    proof = prove_solvency(custody, liab, poster="vendor")
    proof.sign(FORGER, party="vendor")  # invalid signature under VERIFIER
    with pytest.raises(SettlementError, match="invalid signature"):
        resolve_insolvency(custody, liab, solvency=proof, verifier=VERIFIER)


def test_resolve_refuses_tampered_solvency_proof():
    from vincio.settlement.solvency import prove_solvency

    custody = _reserves(60.0)
    liab = _owed({"bank": 50.0, "acme": 30.0})
    proof = prove_solvency(custody, liab, poster="vendor")
    proof.reserves_usd = proof.reserves_usd + 100.0  # break margin without resealing
    proof.seal()  # re-seal so the hash matches but margin is now unsound vs the attestation
    with pytest.raises(SettlementError, match="does not bind the supplied|tampered"):
        resolve_insolvency(custody, liab, solvency=proof)


def test_resolve_refuses_schedule_for_wrong_poster():
    schedule = build_seniority_schedule("someone-else", [["bank"], ["acme"]])
    with pytest.raises(SettlementError, match="not the poster"):
        resolve_insolvency(_reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}), schedule)


def test_resolution_verify_no_verifier_lists_signers():
    # With no verifier every recorded signer is taken at face value (the verifier-is-None branch).
    res = resolve_insolvency(_reserves(200.0), _owed({"bank": 50.0}))
    res.sign(_signer("auditor"), party="auditor")
    result = res.verify()  # no verifier
    assert result.valid is True
    assert result.signed_by == ["auditor"]


def test_resolution_require_valid_returns_self_when_valid():
    res = resolve_insolvency(_reserves(200.0), _owed({"bank": 50.0}))
    assert res.require_valid() is res


def test_verify_set_off_wrong_poster():
    # A mutually-signed statement that binds correctly by hash but is for another poster.
    res, statement = _setoff_resolution()
    alien = _signed_setoff(build_set_off_statement("other-co", "acme", 30.0, 12.0))
    res.set_off_hashes = sorted([alien.content_hash])
    res.seal()
    result = res.verify(VERIFIER, set_off=[alien])
    assert result.set_off_bound is False
    assert result.reason == "a set-off statement is for a different poster"


def test_verify_schedule_rank_mismatch_message():
    # Bind the real schedule's hash but feed verify a *different* well-formed schedule of the
    # same hash-length whose ranks differ -> the "rank does not match" branch specifically.
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]])
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}), schedule)
    swapped = build_seniority_schedule("vendor", [["acme"], ["bank"]])
    res.schedule_hash = swapped.content_hash  # make same_schedule true
    res.seal()
    result = res.verify(schedule=swapped)
    assert result.schedule_bound is False
    assert result.reason == "a creditor's rank does not match the supplied schedule"


def test_build_raises_on_duplicate_rank_dicts():
    with pytest.raises(SettlementError, match="malformed"):
        build_seniority_schedule(
            "vendor",
            [{"rank": 0, "creditors": ["bank"]}, {"rank": 0, "creditors": ["acme"]}],
        )


def test_schedule_verify_unsealed_reason():
    schedule = SenioritySchedule(poster="vendor", tranches=[SeniorityTranche(creditors=["bank"])])
    result = schedule.verify()
    assert result.valid is False
    assert result.reason == "schedule is not sealed (no content hash)"


def test_schedule_verify_hash_mismatch_reason():
    schedule = build_seniority_schedule("vendor", [["bank"]])
    schedule.poster = "tampered"  # mutate a bound fact without resealing
    result = schedule.verify()
    assert result.valid is False
    assert result.hash_ok is False
    assert result.reason == "content hash does not match the schedule facts"


def test_schedule_verify_rejects_blank_creditor():
    # A blank creditor name makes the tranche not well-formed (the `if not creditor` branch).
    schedule = SenioritySchedule(
        poster="vendor", tranches=[SeniorityTranche(rank=0, creditors=[""])]
    ).seal()
    result = schedule.verify()
    assert result.well_formed is False
    assert result.valid is False


def test_schedule_verify_signed_without_verifier_trusts_signers():
    # With no verifier the recorded signer is taken at face value (verifier-is-None append branch).
    schedule = build_seniority_schedule("vendor", [["bank"]])
    schedule.sign(FORGER, party="vendor")  # signature is bogus, but no verifier checks it
    result = schedule.verify()  # no verifier
    assert result.valid is True
    assert result.signed_by == ["vendor"]


def test_schedule_verify_malformed_reason():
    schedule = SenioritySchedule(
        poster="vendor",
        tranches=[
            SeniorityTranche(rank=0, creditors=["bank"]),
            SeniorityTranche(rank=1, creditors=["bank"]),  # creditor ranked twice
        ],
    ).seal()
    result = schedule.verify()
    assert result.well_formed is False
    assert result.reason is not None
    assert "malformed" in result.reason


# == _recoveries_match / _tranches_match distinct branches ===================


def test_recoveries_match_length_mismatch():
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    expected = list(res.recoveries)[:-1]  # one short
    assert res._recoveries_match(expected) is False


def test_recoveries_match_unknown_creditor():
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    expected = [
        CreditorRecovery(creditor="ghost", rank=0, claim_usd=r.claim_usd, recovery_usd=r.recovery_usd)
        for r in res.recoveries
    ]
    assert res._recoveries_match(expected) is False  # creditor not in recorded map -> None


def test_recoveries_match_rank_mismatch():
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    expected = [r.model_copy(update={"rank": r.rank + 5}) for r in res.recoveries]
    assert res._recoveries_match(expected) is False


def test_recoveries_match_claim_mismatch():
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    expected = [r.model_copy(update={"claim_usd": r.claim_usd + 1.0}) for r in res.recoveries]
    assert res._recoveries_match(expected) is False


def test_recoveries_match_shortfall_mismatch():
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    expected = [r.model_copy(update={"shortfall_usd": r.shortfall_usd + 1.0}) for r in res.recoveries]
    assert res._recoveries_match(expected) is False


def test_recoveries_match_true_on_self():
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 60.0, "acme": 40.0}))
    assert res._recoveries_match(list(res.recoveries)) is True


def test_tranches_match_length_mismatch():
    res = resolve_insolvency(
        _reserves(60.0),
        _owed({"bank": 50.0, "acme": 30.0}),
        build_seniority_schedule("vendor", [["bank"], ["acme"]]),
    )
    assert res._tranches_match(list(res.tranches)[:-1]) is False


def test_tranches_match_unknown_rank():
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}))
    expected = [t.model_copy(update={"rank": t.rank + 99}) for t in res.tranches]
    assert res._tranches_match(expected) is False  # rank not in have map -> None


def test_tranches_match_claim_mismatch():
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}))
    expected = [t.model_copy(update={"claim_usd": t.claim_usd + 1.0}) for t in res.tranches]
    assert res._tranches_match(expected) is False


def test_tranches_match_paid_mismatch():
    res = resolve_insolvency(_reserves(60.0), _owed({"bank": 50.0, "acme": 30.0}))
    expected = [t.model_copy(update={"paid_usd": t.paid_usd + 1.0}) for t in res.tranches]
    assert res._tranches_match(expected) is False


def test_schedule_require_valid_returns_self_when_valid():
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]])
    assert schedule.require_valid() is schedule


def test_resolve_refuses_unrelated_solvency_proof():
    from vincio.settlement.solvency import prove_solvency

    custody = _reserves(60.0)
    liab = _owed({"bank": 50.0, "acme": 30.0})
    proof = prove_solvency(custody, liab, poster="vendor")
    # Resolve with a *different* liability attestation than the proof binds.
    other_liab = _owed({"bank": 50.0, "acme": 30.0, "globex": 5.0})
    with pytest.raises(SettlementError, match="does not bind the supplied"):
        resolve_insolvency(custody, other_liab, solvency=proof)
