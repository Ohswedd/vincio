"""Cross-org insolvency resolution & liability seniority waterfall (3.42).

A ``SolvencyProof`` *flags* an insolvency; this resolves it into who-gets-what — a signed
seniority schedule ranks the obligations into priority tranches, and ``resolve_insolvency``
distributes the proven reserves across them by seniority then pari-passu within a tranche, into a
content-bound, offline-verifiable resolution.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from vincio import (
    ContextApp,
    InsolvencyResolution,
    SenioritySchedule,
    SeniorityTranche,
    attest_custody,
    attest_liabilities,
    build_seniority_schedule,
    check_completeness,
    prove_solvency,
    resolve_insolvency,
)
from vincio.core.errors import SettlementError
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement.waterfall import INSOLVENCY_ACTION, SENIORITY_ACTION

# Cross-party signing convention: every party signs with the shared fabric secret, distinguished
# only by key_id (its identity), so one verifier checks every party's signature alike.
FABRIC = "fabric-secret"
AUDITOR = HMACSigner(FABRIC, key_id="auditor")
CUSTODIAN = HMACSigner(FABRIC, key_id="custodian")
VENDOR = HMACSigner(FABRIC, key_id="vendor")
FORGER = HMACSigner("forger-secret", key_id="auditor")

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _app(name: str = "auditor") -> ContextApp:
    app = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    app.use_settlement_book(owner=name)
    return app


def _reserves(amount: float = 60.0, *, poster: str = "vendor") -> object:
    return attest_custody(poster, {"omnibus": amount}, custodian="custodian")


def _owed(mapping: dict[str, float] | None = None, *, poster: str = "vendor") -> object:
    return attest_liabilities(
        poster, mapping or {"bank": 50.0, "acme": 30.0, "globex": 20.0}, attestor="auditor"
    )


# == seniority schedule =======================================================

# -- construction & coercion --------------------------------------------------


def test_positional_lists_become_ranked_tranches():
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]])
    assert schedule.rank_of("bank") == 0
    assert schedule.rank_of("acme") == 1
    assert schedule.rank_of("globex") == 1
    assert schedule.ranks == [0, 1]


def test_single_creditor_string_is_a_one_creditor_tranche():
    schedule = build_seniority_schedule("vendor", ["bank", "acme"])
    assert schedule.rank_of("bank") == 0
    assert schedule.rank_of("acme") == 1


def test_explicit_tranche_objects_keep_their_rank_and_label():
    schedule = build_seniority_schedule(
        "vendor",
        [
            SeniorityTranche(rank=5, creditors=["bank"], label="secured"),
            SeniorityTranche(rank=10, creditors=["acme"], label="subordinated"),
        ],
    )
    assert schedule.rank_of("bank") == 5
    assert schedule.rank_of("acme") == 10
    assert schedule.residual_rank == 11


def test_dict_tranches_validate():
    schedule = build_seniority_schedule(
        "vendor", [{"rank": 0, "creditors": ["bank"], "label": "senior"}]
    )
    assert schedule.rank_of("bank") == 0


def test_unrecognized_tranche_item_is_refused():
    with pytest.raises(SettlementError):
        build_seniority_schedule("vendor", [123])


def test_unlisted_creditor_falls_to_residual_rank():
    schedule = build_seniority_schedule("vendor", [["bank"]])
    assert schedule.residual_rank == 1
    assert schedule.rank_of("acme") == 1  # not listed -> most junior


def test_empty_schedule_residual_rank_is_zero():
    schedule = build_seniority_schedule("vendor", [])
    assert schedule.residual_rank == 0
    assert schedule.rank_of("anyone") == 0


# -- well-formedness ----------------------------------------------------------


def test_creditor_in_two_tranches_is_refused():
    with pytest.raises(SettlementError):
        build_seniority_schedule("vendor", [["bank"], ["bank", "acme"]])


def test_duplicate_rank_is_refused():
    with pytest.raises(SettlementError):
        build_seniority_schedule(
            "vendor",
            [
                SeniorityTranche(rank=0, creditors=["bank"]),
                SeniorityTranche(rank=0, creditors=["acme"]),
            ],
        )


def test_blank_creditor_is_not_well_formed():
    schedule = SenioritySchedule(poster="vendor", tranches=[SeniorityTranche(creditors=[""])])
    schedule.seal()
    assert not schedule.verify().well_formed


# -- hashing, signing, verification -------------------------------------------


def test_schedule_is_sealed_and_verifies():
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]])
    assert schedule.content_hash
    assert schedule.verify().valid


def test_unsealed_schedule_is_invalid():
    schedule = SenioritySchedule(poster="vendor", tranches=[SeniorityTranche(creditors=["a"])])
    assert not schedule.verify().valid


def test_listing_order_does_not_change_the_hash():
    a = build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]], as_of=T0)
    b = build_seniority_schedule("vendor", [["bank"], ["globex", "acme"]], as_of=T0)
    assert a.content_hash == b.content_hash


def test_tampered_rank_caught_even_after_reseal():
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]], as_of=T0)
    original = schedule.content_hash
    schedule.tranches[1].rank = 0  # promote acme to senior
    # Without re-sealing the hash no longer matches.
    assert not schedule.verify().hash_ok
    schedule.seal()  # re-seal the lie -> now malformed (two rank-0 tranches)
    assert schedule.content_hash != original
    assert not schedule.verify().well_formed


def test_schedule_signature_and_forged_signature():
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]])
    schedule.sign(VENDOR, party="vendor")
    assert schedule.verify(VENDOR).valid
    assert schedule.verify(VENDOR, require=["vendor"]).valid
    schedule.sign(FORGER, party="vendor")  # wrong key
    assert not schedule.verify(VENDOR).signatures_ok


def test_re_signing_replaces_prior_signature():
    schedule = build_seniority_schedule("vendor", [["bank"]])
    schedule.sign(VENDOR, party="vendor")
    schedule.sign(VENDOR, party="vendor")
    assert schedule.signed_by == ["vendor"]


def test_require_valid_raises_on_tamper():
    schedule = build_seniority_schedule("vendor", [["bank"]])
    schedule.content_hash = "deadbeef"
    with pytest.raises(SettlementError):
        schedule.require_valid()


def test_schedule_wire_roundtrip():
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]]).sign(VENDOR, party="vendor")
    back = SenioritySchedule.from_wire(schedule.to_wire())
    assert back.verify(VENDOR).valid
    assert back.rank_of("bank") == 0


def test_schedule_audit_details_json_safe():
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme"]])
    details = schedule.audit_details()
    assert details["poster"] == "vendor"
    assert details["tranches"] == 2


# == insolvency waterfall =====================================================

# -- the distribution ---------------------------------------------------------


def test_senior_tranche_paid_in_full_before_junior():
    # reserves 60, bank(50) senior, acme(30)+globex(20) junior -> bank full, junior gets the
    # remaining 10 pari-passu (20%): acme 6, globex 4.
    res = resolve_insolvency(
        _reserves(60.0), _owed(), build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]])
    )
    rec = {r.creditor: r for r in res.recoveries}
    assert rec["bank"].recovery_usd == 50.0 and rec["bank"].shortfall_usd == 0.0
    assert rec["acme"].recovery_usd == 6.0 and rec["acme"].shortfall_usd == 24.0
    assert rec["globex"].recovery_usd == 4.0
    assert res.distributed_usd == 60.0
    assert res.shortfall_usd == 40.0
    assert res.insolvent


def test_no_schedule_is_pure_pari_passu():
    res = resolve_insolvency(_reserves(60.0), _owed())  # 60 across 100 -> 60% each
    rec = {r.creditor: r.recovery_usd for r in res.recoveries}
    assert rec == {"bank": 30.0, "acme": 18.0, "globex": 12.0}
    assert all(r.rank == 0 for r in res.recoveries)


def test_solvent_reserves_make_every_creditor_whole():
    res = resolve_insolvency(_reserves(120.0), _owed())  # 120 covers 100
    assert res.solvent
    assert res.fully_recovered
    assert res.shortfall_usd == 0.0
    assert res.surplus_usd == 20.0
    assert all(r.made_whole for r in res.recoveries)
    assert res.status == "solvent"


def test_exactly_solvent_has_no_shortfall():
    res = resolve_insolvency(_reserves(100.0), _owed())
    assert res.solvent
    assert res.distributed_usd == 100.0
    assert res.surplus_usd == 0.0


def test_zero_reserves_pays_nothing():
    res = resolve_insolvency(_reserves(0.0), _owed())
    assert res.distributed_usd == 0.0
    assert res.shortfall_usd == 100.0
    assert all(r.recovery_usd == 0.0 for r in res.recoveries)


def test_shortfall_bearers_ordered_by_seniority():
    res = resolve_insolvency(
        _reserves(60.0), _owed(), build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]])
    )
    # bank made whole; junior tranche bears the shortfall, ordered by rank then creditor.
    assert res.shortfall_bearers == ["acme", "globex"]


def test_recovery_rate_and_recovery_of():
    res = resolve_insolvency(_reserves(60.0), _owed())
    assert abs(res.recovery_rate - 0.6) <= 1e-9
    bank = res.recovery_of("bank")
    assert bank is not None and abs(bank.recovery_rate - 0.6) <= 1e-9
    assert res.recovery_of("nobody") is None


def test_tranche_summaries_roll_up():
    res = resolve_insolvency(
        _reserves(60.0), _owed(), build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]])
    )
    by_rank = {t.rank: t for t in res.tranches}
    assert by_rank[0].paid_usd == 50.0 and by_rank[0].coverage == 1.0
    assert by_rank[1].claim_usd == 50.0 and by_rank[1].paid_usd == 10.0
    assert abs(by_rank[1].coverage - 0.2) <= 1e-9


def test_duplicate_creditor_lines_are_summed():
    owed = attest_liabilities(
        "vendor",
        [("acme", 30.0), ("acme", 20.0), ("bank", 50.0)],
        attestor="auditor",
    )
    res = resolve_insolvency(_reserves(100.0), owed)
    assert res.recovery_of("acme").claim_usd == 50.0


# -- completeness folding -----------------------------------------------------


def test_completeness_adds_omitted_creditor_to_the_waterfall():
    owed = attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor")  # omits globex
    check = check_completeness(owed, {"acme": 60.0, "globex": 40.0})
    res = resolve_insolvency(_reserves(50.0), owed, completeness=check)
    assert res.liabilities_usd == 100.0  # completed, not the attested 60
    assert res.recovery_of("globex") is not None  # the omitted creditor is paid
    assert res.distributed_usd == 50.0
    assert res.completeness_hash == check.content_hash


# == verification =============================================================


def test_resolution_verifies_offline():
    res = resolve_insolvency(_reserves(60.0), _owed())
    assert res.verify().valid
    assert res.verify().distribution_sound


def test_over_stated_recovery_caught_even_after_reseal():
    res = resolve_insolvency(_reserves(60.0), _owed())
    res.recoveries[0].recovery_usd += 100.0  # inflate one recovery
    res.seal()  # recompute the hash to match the lie
    result = res.verify()
    assert result.hash_ok
    assert not result.distribution_sound  # the waterfall no longer re-derives


def test_reordered_rank_caught():
    res = resolve_insolvency(
        _reserves(60.0), _owed(), build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]])
    )
    # Promote acme to the senior tranche without redoing the distribution.
    for r in res.recoveries:
        if r.creditor == "acme":
            r.rank = 0
    res.seal()
    assert not res.verify().distribution_sound


def test_tampered_liability_total_caught():
    res = resolve_insolvency(_reserves(60.0), _owed())
    res.liabilities_usd = 1.0  # no longer equals the sum of claims
    res.seal()
    assert not res.verify().distribution_sound


def test_schedule_binding_detects_rank_mismatch():
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]])
    res = resolve_insolvency(_reserves(60.0), _owed(), schedule)
    assert res.verify(schedule=schedule).valid
    # A different schedule (acme promoted) does not bind.
    other = build_seniority_schedule("vendor", [["bank", "acme"], ["globex"]])
    assert not res.verify(schedule=other).schedule_bound


def test_resolution_signature_and_require():
    res = resolve_insolvency(_reserves(60.0), _owed())
    res.sign(AUDITOR, party="auditor")
    assert res.verify(AUDITOR).valid
    assert res.verify(AUDITOR, require=["auditor"]).valid
    assert not res.verify(AUDITOR, require=["someone-else"]).signatures_ok


def test_resolution_wire_roundtrip_preserves_verification():
    res = resolve_insolvency(
        _reserves(60.0), _owed(), build_seniority_schedule("vendor", [["bank"], ["acme"]])
    ).sign(AUDITOR, party="auditor")
    back = InsolvencyResolution.from_wire(res.to_wire())
    assert back.verify(AUDITOR).valid
    assert back.shortfall_bearers == res.shortfall_bearers


def test_require_fully_recovered():
    solvent = resolve_insolvency(_reserves(120.0), _owed())
    assert solvent.require_fully_recovered() is solvent
    insolvent = resolve_insolvency(_reserves(60.0), _owed())
    with pytest.raises(SettlementError):
        insolvent.require_fully_recovered()


def test_resolution_require_valid_raises_on_tamper():
    res = resolve_insolvency(_reserves(60.0), _owed())
    res.content_hash = "deadbeef"
    with pytest.raises(SettlementError):
        res.require_valid()


def test_two_folders_compute_the_same_hash():
    cust = attest_custody("vendor", {"omnibus": 60.0}, custodian="custodian", as_of=T0)
    owed = attest_liabilities("vendor", {"bank": 50.0, "acme": 50.0}, attestor="auditor", as_of=T0)
    a = resolve_insolvency(cust, owed, as_of=T0)
    b = resolve_insolvency(cust, owed, as_of=T0)
    assert a.content_hash == b.content_hash


def test_audit_details_json_safe():
    res = resolve_insolvency(_reserves(60.0), _owed())
    details = res.audit_details()
    assert details["status"] == "resolved"
    assert details["shortfall_usd"] == 40.0


# == refusals =================================================================


def test_tampered_reserves_refused():
    cust = _reserves(60.0)
    cust.reserves_usd = 9_999.0  # no longer re-derives
    cust.seal()
    with pytest.raises(SettlementError):
        resolve_insolvency(cust, _owed())


def test_tampered_liabilities_refused():
    owed = _owed()
    owed.liabilities_usd = 1.0
    owed.seal()
    with pytest.raises(SettlementError):
        resolve_insolvency(_reserves(60.0), owed)


def test_mismatched_posters_require_explicit_poster():
    cust = attest_custody("vendor", {"omnibus": 60.0})
    owed = attest_liabilities("globex", {"acme": 30.0})
    with pytest.raises(SettlementError):
        resolve_insolvency(cust, owed)


def test_schedule_for_wrong_poster_refused():
    schedule = build_seniority_schedule("globex", [["bank"]])  # not vendor
    with pytest.raises(SettlementError):
        resolve_insolvency(_reserves(60.0), _owed(), schedule)


def test_forged_attestor_signature_refused_with_verifier():
    owed = _owed()
    owed.sign(FORGER, party="auditor")  # wrong key
    with pytest.raises(SettlementError):
        resolve_insolvency(_reserves(60.0), owed, verifier=AUDITOR)


# -- pre-built solvency proof binding -----------------------------------------


def test_prebuilt_solvency_proof_is_reused():
    cust, owed = _reserves(60.0), _owed()
    proof = prove_solvency(cust, owed)
    res = resolve_insolvency(cust, owed, solvency=proof)
    assert res.solvency_hash == proof.content_hash
    assert res.distributed_usd == 60.0


def test_unrelated_solvency_proof_refused():
    cust, owed = _reserves(60.0), _owed()
    other_proof = prove_solvency(_reserves(99.0), _owed({"x": 5.0}))
    with pytest.raises(SettlementError):
        resolve_insolvency(cust, owed, solvency=other_proof)


def test_tampered_solvency_proof_refused():
    cust, owed = _reserves(60.0), _owed()
    proof = prove_solvency(cust, owed)
    proof.reserves_usd = 9_999.0
    proof.seal()
    with pytest.raises(SettlementError):
        resolve_insolvency(cust, owed, solvency=proof)


def test_solvency_plus_completeness_reproves_against_completed_set():
    # A pre-built proof passed alongside completeness must not bypass completeness verification:
    # the call re-proves, so the omitted creditor still enters the waterfall and a tampered
    # completeness is refused.
    owed = attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor")  # omits globex
    check = check_completeness(owed, {"acme": 60.0, "globex": 40.0})
    proof = prove_solvency(_reserves(50.0), owed, completeness=check)
    res = resolve_insolvency(_reserves(50.0), owed, solvency=proof, completeness=check)
    assert res.recovery_of("globex") is not None  # the omitted creditor is paid
    assert res.liabilities_usd == 100.0
    tampered = check_completeness(owed, {"acme": 60.0, "globex": 40.0})
    tampered.completed_usd = 1.0
    tampered.seal()
    with pytest.raises(SettlementError):
        resolve_insolvency(_reserves(50.0), owed, solvency=proof, completeness=tampered)


# == app & book wiring ========================================================


def test_app_build_seniority_schedule_signs_and_audits():
    app = _app()
    schedule = app.build_seniority_schedule("vendor", [["bank"], ["acme"]])
    assert schedule.verify(app.contract_signer).valid
    assert "auditor" in schedule.signed_by
    assert len(app.audit.query(action=SENIORITY_ACTION)) == 1
    assert app.audit.query(action=SENIORITY_ACTION)[0].decision == "ranked"


def test_app_self_ranked_when_poster_is_app():
    app = _app(name="vendor")
    app.build_seniority_schedule("vendor", [["bank"]])
    assert app.audit.query(action=SENIORITY_ACTION)[0].decision == "self_ranked"


def test_app_resolve_insolvency_signs_audits_and_dings_reputation():
    app = _app()
    app.use_reputation_ledger()
    reserves = app.attest_custody("vendor", {"omnibus": 60.0})
    owed = app.attest_liabilities("vendor", {"bank": 50.0, "acme": 50.0})
    schedule = app.build_seniority_schedule("vendor", [["bank"], ["acme"]])
    res = app.resolve_insolvency(reserves, owed, schedule, verifier=app.contract_signer)
    assert res.verify(app.contract_signer).valid
    assert res.insolvent
    assert len(app.audit.query(action=INSOLVENCY_ACTION)) == 1
    assert app.audit.query(action=INSOLVENCY_ACTION)[0].decision == "resolved"
    assert app.audit.verify_chain()
    # A resolved insolvency dings the poster below an unseen member's prior.
    assert app.reputation_ledger.weight("vendor") < app.reputation_ledger.weight("never-seen")


def test_app_solvent_resolution_does_not_ding_reputation():
    app = _app()
    app.use_reputation_ledger()
    reserves = app.attest_custody("vendor", {"omnibus": 200.0})
    owed = app.attest_liabilities("vendor", {"bank": 50.0, "acme": 50.0})
    res = app.resolve_insolvency(reserves, owed)
    assert res.solvent
    # A solvent resolution records no failure: the poster keeps an unseen member's prior weight.
    assert app.reputation_ledger.weight("vendor") == app.reputation_ledger.weight("never-seen")


def test_book_resolve_insolvency_path():
    app = _app()
    book = app.settlement_book
    reserves = attest_custody("vendor", {"omnibus": 60.0})
    owed = attest_liabilities("vendor", {"bank": 50.0, "acme": 50.0}, attestor="auditor")
    schedule = book.build_seniority_schedule("vendor", [["bank"], ["acme"]])
    res = book.resolve_insolvency(reserves, owed, schedule)
    assert res.verify(app.contract_signer).valid
    assert res.recovery_of("bank").made_whole
    assert not res.recovery_of("acme").made_whole
