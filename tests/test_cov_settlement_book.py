"""Real-behavior coverage for the settlement book and engine (vincio.settlement.book).

Every test drives the real reconciliation/hash-chain/verify code with concrete
contracts, records, signers, and reputation ledgers — no mocking. The targets are the
uncovered branches: the un-metered / zero-term reconciliation lines, the neutral-book
counterparty attribution, the signature-mismatch and head-hash verify paths, the
collateral / liability / solvency / arbitration / netting / attestation wrappers, the
best-effort persistence-failure swallows, and the error paths.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from vincio import ContextApp
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import AuditLog, HMACSigner
from vincio.settlement import (
    Reconciliation,
    SettlementBook,
    SettlementRecord,
    settle_contract,
)
from vincio.settlement.book import (
    BookVerification,
    SettlementReport,
    SettlementRow,
)
from vincio.settlement.collateral import COLLATERAL_ACTION
from vincio.settlement.custody import CUSTODY_ACTION
from vincio.settlement.setoff import SETOFF_ACTION
from vincio.settlement.solvency import (
    COMPLETENESS_ACTION,
    DISCHARGE_ACTION,
    LIABILITY_ACTION,
    SOLVENCY_ACTION,
)
from vincio.settlement.waterfall import SENIORITY_ACTION
from vincio.storage.base import InMemoryMetadataStore


def _app(name: str = "acme") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def _contract(buyer: str = "acme", seller: str = "vendor", **terms) -> Contract:
    defaults = {"scope": "work", "price_usd": 0.10, "sla_seconds": 5.0, "quality_floor": 0.8}
    defaults.update(terms)
    return Contract(buyer=buyer, seller=seller, terms=ContractTerms(**defaults)).seal()


def _signer(key: str = "k0") -> HMACSigner:
    return HMACSigner(key, key_id=key)


# -- settle_contract: the reconciliation-line branches ------------------------


def test_unmetered_settlement_marks_lines_not_metered():
    # No cost/latency/quality supplied -> delivered is None, lines note "not metered".
    record = settle_contract(_contract(price_usd=0.10, sla_seconds=5.0, quality_floor=0.8))
    by_dim = {line.dimension: line for line in record.lines}
    assert by_dim["price"].delivered is None
    assert by_dim["price"].note == "not metered"
    assert by_dim["sla"].delivered is None and by_dim["sla"].delta is None
    assert by_dim["quality"].note == "not metered"
    assert record.delivered_cost_usd is None
    # An unmetered delivery counts no units and no events.
    assert record.metered_units == 0.0 and record.metered_events == 0
    # The balance is the whole price (nothing was delivered against it).
    assert record.balance_usd == pytest.approx(0.10)


def test_zero_term_dimensions_produce_no_lines():
    # A contract that constrains nothing produces an empty reconciliation.
    record = settle_contract(_contract(price_usd=0.0, sla_seconds=0.0, quality_floor=0.0))
    assert record.lines == []
    assert record.amount_owed_usd == 0.0
    assert record.fulfilled is True


def test_partial_metric_marks_one_event_one_unit():
    # Only quality supplied (cost/latency None): still a metered event of one unit.
    record = settle_contract(_contract(price_usd=0.10, quality_floor=0.8), quality=0.95)
    assert record.metered_units == 1.0 and record.metered_events == 1
    by_dim = {line.dimension: line for line in record.lines}
    assert by_dim["quality"].delivered == pytest.approx(0.95)
    assert by_dim["quality"].within is True
    # Price was never metered -> its delta stays None even though an event fired.
    assert by_dim["price"].delivered is None


def test_overrun_balance_is_negative():
    record = settle_contract(_contract(price_usd=0.10), cost_usd=0.30, quality=0.9, latency_ms=100)
    assert record.balance_usd == pytest.approx(-0.20)  # 0.10 owed - 0.30 delivered


# -- report roll-up & report properties ---------------------------------------


def test_report_for_unknown_counterparty_is_a_zeroed_row():
    book = SettlementBook("acme")
    book.settle(_contract(seller="vendor"), cost_usd=0.05, latency_ms=100, quality=0.9)
    report = book.report(counterparty="ghost")
    assert len(report.rows) == 1
    row = report.rows[0]
    assert row.counterparty == "ghost"
    assert row.settlements == 0 and row.total_owed_usd == 0.0


def test_report_net_balance_property_sums_rows():
    report = SettlementReport(
        owner="acme",
        rows=[
            SettlementRow(counterparty="a", net_balance_usd=0.05),
            SettlementRow(counterparty="b", net_balance_usd=-0.02, breached=3),
        ],
    )
    assert report.net_balance_usd == pytest.approx(0.03)
    assert report.breached == 3


# -- append edge cases --------------------------------------------------------


def test_append_seals_an_unsealed_record():
    book = SettlementBook("acme")
    # A bare record with no content_hash: append must seal it before linking.
    record = SettlementRecord(contract_id="c1", buyer="acme", seller="vendor", price_usd=0.10)
    assert record.content_hash == ""
    book.append(record)
    assert record.content_hash != ""  # sealed on append
    assert record.seq == 0 and record.prev_hash == ""
    assert book.head_hash == record.entry_hash
    assert book.verify().intact


def test_resolve_party_none_when_owner_is_neither_side():
    # A neutral book (owner not buyer or seller) cannot sign as a party.
    book = SettlementBook("observer", signer=_signer())
    record = book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05)
    assert record.signed_by == []  # nothing signed: owner is on neither side


# -- counterparty attribution -------------------------------------------------


def test_counterparty_when_owner_is_seller():
    book = SettlementBook("vendor")
    book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05)
    assert book.counterparties() == ["acme"]
    assert book.records_with("acme")[0].seller == "vendor"


def test_neutral_book_attributes_to_seller():
    book = SettlementBook("coord")
    book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05)
    # Owner is neither party -> counterparty is the seller it pays out to.
    assert book.counterparties() == ["vendor"]


def test_record_by_id_finds_and_misses():
    book = SettlementBook("acme")
    rec = book.settle(_contract(seller="vendor"), cost_usd=0.05)
    assert book.record_by_id(rec.id) is rec
    assert book.record_by_id("nope") is None


# -- verification paths -------------------------------------------------------


def test_verify_detects_reconciliation_hash_tamper():
    book = SettlementBook("acme")
    book.settle(_contract(), cost_usd=0.05)
    book.settle(_contract(), cost_usd=0.05)
    book.records[1].balance_usd = 99.0  # mutate an economic fact
    verdict = book.verify()
    assert not verdict.intact
    assert verdict.broken_at == 1
    assert verdict.reason == "reconciliation hash mismatch"


def test_verify_detects_entry_chain_break():
    book = SettlementBook("acme")
    book.settle(_contract(), cost_usd=0.05)
    book.settle(_contract(), cost_usd=0.05)
    # Re-seal so content_hash matches but break the chain link.
    book.records[1].prev_hash = "tampered-link"
    verdict = book.verify()
    assert not verdict.intact
    assert verdict.broken_at == 1
    assert verdict.reason == "entry chain broken"


def test_verify_detects_head_hash_mismatch():
    book = SettlementBook("acme")
    book.settle(_contract(), cost_usd=0.05)
    book.head_hash = "wrong-head"  # head no longer matches the chain tail
    verdict = book.verify()
    assert not verdict.intact
    assert verdict.broken_at is None
    assert verdict.reason == "head hash does not match chain"


def test_verify_detects_forged_signature():
    signer = _signer("real")
    book = SettlementBook("acme", signer=signer)
    book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05)
    # A different key cannot reproduce the recorded signature.
    forged = _signer("forged")
    verdict = book.verify(verifier=forged)
    assert not verdict.intact
    assert verdict.broken_at == 0
    assert verdict.reason == "signature mismatch"


def test_verify_passes_with_correct_signer():
    signer = _signer("real")
    book = SettlementBook("acme", signer=signer)
    book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05)
    verdict = book.verify(verifier=signer)
    assert verdict.intact
    assert verdict.entries == 1


def test_require_intact_returns_self_when_clean():
    book = SettlementBook("acme")
    book.settle(_contract(), cost_usd=0.05)
    assert book.require_intact() is book


def test_require_intact_raises_with_broken_at_detail():
    book = SettlementBook("acme")
    book.settle(_contract(), cost_usd=0.05)
    book.records[0].balance_usd = 7.0
    with pytest.raises(SettlementError, match="failed verification"):
        book.require_intact()


# -- reconcile_with & arbitrate -----------------------------------------------


def test_reconcile_with_agrees_on_matching_figures():
    book = SettlementBook("acme")
    c = _contract(seller="vendor")
    book.settle(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    theirs = settle_contract(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    result = book.reconcile_with(theirs)
    assert isinstance(result, Reconciliation)
    assert result.agrees


def test_reconcile_with_missing_contract_raises():
    book = SettlementBook("acme")
    book.settle(_contract(seller="vendor"), cost_usd=0.05)
    other = settle_contract(_contract(seller="rival"), cost_usd=0.05)
    with pytest.raises(SettlementError, match="no settlement for contract"):
        book.reconcile_with(other)


def test_arbitrate_upholds_agreeing_records():
    signer = _signer("acme")
    book = SettlementBook("acme", signer=signer)
    c = _contract(buyer="acme", seller="vendor")
    ours = book.settle(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    theirs = settle_contract(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    theirs.sign(_signer("vendor"), party="vendor")
    resolution = book.arbitrate(theirs, contract_id=c.id, sign=False)
    assert resolution.contract_id == c.id
    assert resolution.upheld_balance_usd == pytest.approx(ours.balance_usd)


def test_arbitrate_infers_contract_id_from_counterparty_records():
    book = SettlementBook("acme", signer=_signer("acme"))
    c = _contract(buyer="acme", seller="vendor")
    book.settle(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    theirs = settle_contract(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    # contract_id omitted -> inferred from the single counterparty record's contract.
    resolution = book.arbitrate(theirs, sign=False)
    assert resolution.contract_id == c.id


# -- netting ------------------------------------------------------------------


def test_net_folds_records_into_positions():
    book = SettlementBook("acme")
    # acme buys from vendor (owes) on two settled records.
    book.settle(_contract(buyer="acme", seller="vendor", price_usd=0.10), cost_usd=0.05)
    netting = book.net(sign=False)
    assert netting.owner == "acme"
    assert any(p.party == "vendor" for p in netting.positions)


def test_net_signs_when_signer_attached():
    signer = _signer("acme")
    book = SettlementBook("acme", signer=signer)
    book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05)
    netting = book.net()  # sign defaults True
    assert netting.signatures  # signed as the owner


# -- attestation / revocation -------------------------------------------------


def test_attest_summarizes_subject_standing():
    signer = _signer("acme")
    book = SettlementBook("acme", signer=signer)
    book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05, latency_ms=100, quality=0.9)
    attestation = book.attest("vendor")
    assert attestation.subject == "vendor"
    assert attestation.issuer == "acme"
    assert attestation.signatures  # signed as the issuer


def test_attest_without_history_raises():
    book = SettlementBook("acme")
    with pytest.raises(SettlementError):
        book.attest("never-traded-with")


def test_attest_without_signer_is_unsigned():
    book = SettlementBook("acme")  # no signer
    book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05, latency_ms=100, quality=0.9)
    attestation = book.attest("vendor")
    assert attestation.issuer == "acme"
    assert attestation.signatures == []  # nothing to sign with


def test_revoke_own_attestation():
    signer = _signer("acme")
    book = SettlementBook("acme", signer=signer)
    book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05, latency_ms=100, quality=0.9)
    attestation = book.attest("vendor")
    revocation = book.revoke(attestation, reason="superseded")
    assert revocation.issuer == "acme"
    assert revocation.signatures


def test_revoke_foreign_attestation_raises():
    signer = _signer("acme")
    book = SettlementBook("acme", signer=signer)
    book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05, latency_ms=100, quality=0.9)
    attestation = book.attest("vendor")
    # A different book cannot revoke acme's attestation.
    other = SettlementBook("rival", signer=_signer("rival"))
    with pytest.raises(SettlementError, match="cannot revoke"):
        other.revoke(attestation)


# -- collateral / escrow wrappers ---------------------------------------------


def test_post_and_settle_escrow_signs_and_releases():
    signer = _signer("vendor")
    book = SettlementBook("vendor", signer=signer)
    c = _contract(buyer="acme", seller="vendor", price_usd=0.10)
    escrow = book.post_escrow(c, fraction=0.5)
    assert escrow.amount_usd == pytest.approx(0.05)
    assert escrow.signatures  # signed as vendor (the poster/seller side)
    record = settle_contract(c, cost_usd=0.05, latency_ms=100, quality=0.9)  # fulfilled
    resolved = book.settle_escrow(escrow, record)
    assert resolved.state == "released"


def test_settle_with_escrow_in_one_call():
    signer = _signer("vendor")
    book = SettlementBook("vendor", signer=signer)
    c = _contract(buyer="acme", seller="vendor", price_usd=0.10)
    escrow = book.post_escrow(c, fraction=0.5, sign=False)
    record = book.settle(
        c, cost_usd=0.20, quality=0.2, escrow=escrow
    )  # a breach -> escrow forfeits
    assert record.status == "breached"
    assert escrow.state == "forfeited"


# -- custody / liability / solvency wrappers ----------------------------------


def test_attest_custody_self_signs():
    signer = _signer("acme")
    book = SettlementBook("acme", signer=signer)
    attestation = book.attest_custody("acme", {"reserve": 100.0})
    assert attestation.reserves_usd == pytest.approx(100.0)
    assert attestation.self_custody is True
    assert attestation.signatures  # owner == custodian -> signed


def test_attest_custody_for_other_poster_not_signed():
    signer = _signer("acme")
    book = SettlementBook("acme", signer=signer)
    # Owner attests a *different* poster's reserves as its custodian; signs as custodian.
    attestation = book.attest_custody("ward", 50.0)
    assert attestation.poster == "ward"
    assert attestation.custodian == "acme"
    assert attestation.signatures  # owner IS the custodian here


def test_attest_liabilities_self_attested():
    signer = _signer("acme")
    book = SettlementBook("acme", signer=signer)
    liab = book.attest_liabilities("acme", {"vendor": 40.0})
    assert liab.liabilities_usd == pytest.approx(40.0)
    assert liab.self_attested is True
    assert liab.signatures


def test_prove_solvency_solvent_margin():
    signer = _signer("acme")
    book = SettlementBook("acme", signer=signer)
    custody = book.attest_custody("acme", 100.0, sign=False)
    liab = book.attest_liabilities("acme", {"vendor": 60.0}, sign=False)
    proof = book.prove_solvency(custody, liab)
    assert proof.solvent is True
    assert proof.margin_usd == pytest.approx(40.0)
    assert proof.signatures


def test_prove_solvency_insolvent_breach():
    signer = _signer("acme")
    book = SettlementBook("acme", signer=signer)
    custody = book.attest_custody("acme", 30.0, sign=False)
    liab = book.attest_liabilities("acme", {"vendor": 80.0}, sign=False)
    proof = book.prove_solvency(custody, liab)
    assert proof.insolvent is True
    assert proof.margin_usd == pytest.approx(-50.0)


def test_inclusion_proof_for_named_creditor():
    book = SettlementBook("acme")
    liab = book.attest_liabilities("acme", {"vendor": 40.0, "supplier": 10.0}, sign=False)
    proof = book.inclusion_proof(liab, "vendor")
    assert proof.creditor == "vendor"
    assert proof.amount_usd == pytest.approx(40.0)


def test_claims_against_sums_owner_seller_records():
    book = SettlementBook("acme")
    # acme as SELLER to buyer "bob" -> bob owes acme.
    book.settle(_contract(buyer="bob", seller="acme", price_usd=0.10), cost_usd=0.05)
    book.settle(_contract(buyer="bob", seller="acme", price_usd=0.20), cost_usd=0.10)
    claims = book.claims_against("bob")
    assert claims == {"acme": pytest.approx(0.30)}
    # No claims against a buyer acme never sold to.
    assert book.claims_against("stranger") == {}


def test_discharge_liability_as_creditor():
    signer = _signer("acme")
    book = SettlementBook("acme", signer=signer)
    # Owner is the creditor releasing what "debtor" owes it.
    discharge = book.discharge_liability("debtor", 25.0, note="paid")
    assert discharge.creditor == "acme"
    assert discharge.amount_usd == pytest.approx(25.0)
    assert discharge.signatures


# -- reputation closing on insolvency -----------------------------------------


def test_resolve_insolvency_dings_reputation_on_shortfall():
    rep_app = _app()
    rep_app.use_reputation_ledger()
    signer = _signer("acme")
    book = SettlementBook(
        "acme", signer=signer, reputation=rep_app.reputation_ledger
    )
    before = rep_app.reputation_ledger.reputation("debtor")
    custody = book.attest_custody("debtor", 10.0, sign=False)
    liab = book.attest_liabilities("debtor", {"acme": 100.0}, sign=False)
    resolution = book.resolve_insolvency(custody, liab)
    assert resolution.solvent is False
    assert resolution.shortfall_usd > 0
    after = rep_app.reputation_ledger.reputation("debtor")
    assert after < before  # an unmet creditor debits the poster


def test_resolve_insolvency_solvent_does_not_ding():
    rep_app = _app()
    rep_app.use_reputation_ledger()
    book = SettlementBook("acme", reputation=rep_app.reputation_ledger)
    before = rep_app.reputation_ledger.reputation("debtor")
    custody = book.attest_custody("debtor", 100.0, sign=False)
    liab = book.attest_liabilities("debtor", {"acme": 40.0}, sign=False)
    resolution = book.resolve_insolvency(custody, liab)
    assert resolution.solvent is True
    assert rep_app.reputation_ledger.reputation("debtor") == before  # untouched


# -- audit wiring -------------------------------------------------------------


def test_settle_records_on_audit_chain():
    audit = AuditLog(directory=None)  # in-memory, no disk writes
    book = SettlementBook("acme", audit=audit)
    record = book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05)
    assert audit.query(action="settlement")
    assert record.audit_id is not None
    assert audit.verify_chain()


# -- persistence --------------------------------------------------------------


def test_book_persists_and_reloads_with_records():
    store = InMemoryMetadataStore()
    book = SettlementBook("acme", store=store, book_id="ledger-1")
    book.settle(_contract(seller="v1"), cost_usd=0.05)
    book.settle(_contract(seller="v2"), cost_usd=0.05)
    fresh = SettlementBook("acme", store=store, book_id="ledger-1")
    assert len(fresh.records) == 2
    assert fresh.head_hash == book.head_hash
    assert fresh.verify().intact


def test_load_record_restores_state_and_created_at():
    book = SettlementBook("acme")
    book.settle(_contract(seller="vendor"), cost_usd=0.05)
    projection = book.to_record()
    restored = SettlementBook("acme").load_record(projection)
    assert restored.id == book.id
    assert restored.head_hash == book.head_hash
    assert len(restored.records) == 1
    # created_at came back as an aware datetime parsed from the iso string.
    assert restored.created_at.tzinfo is not None


def test_checkpoint_swallows_a_failing_store():
    class _BoomStore:
        def get(self, kind, key):  # noqa: D401, ANN001 - test double
            return None

        def save(self, kind, record):  # noqa: ANN001
            raise RuntimeError("disk full")

    # A store whose save() blows up must not break settling (best-effort persistence).
    book = SettlementBook("acme", store=_BoomStore())
    record = book.settle(_contract(seller="vendor"), cost_usd=0.05)
    assert record.status == "settled"
    assert len(book.records) == 1


def test_load_swallows_a_failing_store():
    class _GetBoomStore:
        def get(self, kind, key):  # noqa: ANN001
            raise RuntimeError("no such kind")

        def save(self, kind, record):  # noqa: ANN001
            return None

    # A store whose get() raises on construction is simply treated as empty.
    book = SettlementBook("acme", store=_GetBoomStore())
    assert book.records == []


# -- event emission -----------------------------------------------------------


def test_settle_emits_event():
    seen = []

    class _Bus:
        def emit(self, name, payload):  # noqa: ANN001
            seen.append((name, payload))

    book = SettlementBook("acme", events=_Bus())
    record = book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05)
    assert seen[0][0] == "settlement.recorded"
    assert seen[0][1]["settlement_id"] == record.id
    assert seen[0][1]["seller"] == "vendor"


def test_settle_swallows_a_failing_event_bus():
    class _BoomBus:
        def emit(self, name, payload):  # noqa: ANN001
            raise RuntimeError("broker down")

    # Event delivery is best-effort: a failing bus must not break settling.
    book = SettlementBook("acme", events=_BoomBus())
    record = book.settle(_contract(seller="vendor"), cost_usd=0.05)
    assert record.status == "settled"


# -- BookVerification model ---------------------------------------------------


def test_book_verification_defaults():
    v = BookVerification(intact=True, entries=3)
    assert v.broken_at is None and v.reason is None


# -- audit wiring on the collateral / liability / dispute helpers --------------
#
# Each wrapper has an `_audit_*` helper that only fires when an audit is attached.
# These drive those branches with an in-memory AuditLog and assert the verdict landed.


def _audited_book(owner: str = "acme", key: str = "acme"):
    audit = AuditLog(directory=None)
    return SettlementBook(owner, signer=_signer(key), audit=audit), audit


def test_post_collateral_pool_audits():
    book, audit = _audited_book("vendor", "vendor")
    c1 = _contract(buyer="acme", seller="vendor", price_usd=0.10)
    c2 = _contract(buyer="acme", seller="vendor", price_usd=0.20)
    pool = book.post_collateral_pool([c1, c2], fraction=0.1)
    assert pool.status == "posted"
    assert audit.query(action=COLLATERAL_ACTION)
    assert pool.audit_id is not None


def test_draw_pool_forfeits_on_breach_and_audits():
    book, audit = _audited_book("vendor", "vendor")
    c = _contract(buyer="acme", seller="vendor", price_usd=0.10)
    pool = book.post_collateral_pool([c], fraction=0.5, sign=False)
    record = settle_contract(c, cost_usd=0.40, quality=0.1)  # breach
    drawn = book.draw_pool(pool, record)  # the book wrapper returns the pool itself
    assert drawn is pool
    assert pool.drawn_usd > 0  # the pool drew the forfeiture
    assert any(c.forfeited_usd > 0 for c in pool.contracts)
    # The draw is on the audit log under the collateral action (post + draw).
    assert len(audit.query(action=COLLATERAL_ACTION)) >= 2


def test_settle_with_pool_draws_in_one_call():
    book = SettlementBook("vendor", signer=_signer("vendor"))
    c = _contract(buyer="acme", seller="vendor", price_usd=0.10)
    pool = book.post_collateral_pool([c], fraction=0.5, sign=False)
    record = book.settle(c, cost_usd=0.05, latency_ms=100, quality=0.9, pool=pool)
    assert record.status == "settled"
    assert pool.drawn_usd == 0.0  # a clean delivery forfeits nothing


def test_attest_custody_audits():
    book, audit = _audited_book()
    att = book.attest_custody("acme", {"reserve": 100.0})
    assert att.audit_id is not None
    assert audit.query(action=CUSTODY_ACTION)


def test_attest_liabilities_audits():
    book, audit = _audited_book()
    liab = book.attest_liabilities("acme", {"vendor": 40.0})
    assert liab.audit_id is not None
    assert audit.query(action=LIABILITY_ACTION)


def test_check_completeness_uses_book_claims_when_omitted():
    book, audit = _audited_book("acme", "acme")
    # acme is owed 0.10 by buyer "debtor" (acme is the seller).
    book.settle(_contract(buyer="debtor", seller="acme", price_usd=0.10), cost_usd=0.05)
    # A liability attestation by debtor that OMITS what it owes acme.
    liab = book.attest_liabilities("debtor", {"other": 5.0}, sign=False)
    proof = book.check_completeness(liab)  # claims derived from acme's own records
    assert proof.complete is False  # the omission is caught
    assert proof.completed_usd >= 0.10
    assert audit.query(action=COMPLETENESS_ACTION)


def test_prove_solvency_audits():
    book, audit = _audited_book()
    custody = book.attest_custody("acme", 100.0, sign=False)
    liab = book.attest_liabilities("acme", {"vendor": 60.0}, sign=False)
    proof = book.prove_solvency(custody, liab)
    assert proof.audit_id is not None
    assert audit.query(action=SOLVENCY_ACTION)


def test_discharge_liability_audits():
    book, audit = _audited_book()
    discharge = book.discharge_liability("debtor", 25.0)
    assert discharge.audit_id is not None
    assert audit.query(action=DISCHARGE_ACTION)


def test_build_set_off_statement_explicit_figures_audits():
    book, audit = _audited_book("acme", "acme")
    st = book.build_set_off_statement("vendor", "acme", owed_usd=30.0, owing_usd=12.0)
    assert st.net_usd == pytest.approx(18.0)
    assert st.direction == "poster_owes"
    assert audit.query(action=SETOFF_ACTION)


def test_build_set_off_statement_from_liabilities():
    book = SettlementBook("acme", signer=_signer("acme"))
    # acme (creditor) has a record where vendor (poster) is the seller? No: owing is
    # records where poster is seller and creditor is buyer. Here vendor sold to acme.
    book.settle(_contract(buyer="acme", seller="vendor", price_usd=0.10), cost_usd=0.05)
    liab = book.attest_liabilities("vendor", {"acme": 30.0}, sign=False)
    st = book.build_set_off_statement("vendor", "acme", liabilities=liab)
    # owed_usd read from the attestation (30.0); owing from acme-buyer records (0.10).
    assert st.owed_usd == pytest.approx(30.0)
    assert st.owing_usd == pytest.approx(0.10)


def test_build_seniority_schedule_audits():
    book, audit = _audited_book("vendor", "vendor")
    schedule = book.build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]])
    assert schedule.poster == "vendor"
    assert audit.query(action=SENIORITY_ACTION)
    # The owner ranking its own creditors is a self-ranking.
    assert audit.query(action=SENIORITY_ACTION)[0].decision == "self_ranked"


def test_resolve_insolvency_with_schedule_distributes_by_seniority():
    book = SettlementBook("trustee", signer=_signer("trustee"))
    custody = book.attest_custody("debtor", 50.0, sign=False)
    liab = book.attest_liabilities("debtor", {"bank": 40.0, "acme": 40.0}, sign=False)
    schedule = book.build_seniority_schedule("debtor", [["bank"], ["acme"]], sign=False)
    resolution = book.resolve_insolvency(custody, liab, schedule)
    # 50 reserves: senior bank fully paid (40), junior acme gets the remaining 10.
    by_creditor = {r.creditor: r.recovery_usd for r in resolution.recoveries}
    assert by_creditor["bank"] == pytest.approx(40.0)
    assert by_creditor["acme"] == pytest.approx(10.0)
    assert resolution.solvent is False


def test_check_root_consistency_surfaces_equivocation_and_dings():
    rep_app = _app()
    rep_app.use_reputation_ledger()
    attestor = _signer("attestor")
    book = SettlementBook(
        "acme", signer=_signer("acme"), audit=AuditLog(directory=None),
        reputation=rep_app.reputation_ledger,
    )
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    a = book.attest_liabilities("vendor", {"acme": 60.0}, attestor="attestor", as_of=as_of, sign=False)
    a.sign(attestor, party="attestor")
    b = book.attest_liabilities("vendor", {"globex": 40.0}, attestor="attestor", as_of=as_of, sign=False)
    b.sign(attestor, party="attestor")
    before = rep_app.reputation_ledger.reputation("vendor")
    report = book.check_root_consistency([("acme", a), ("globex", b)], verifier=attestor)
    assert not report.consistent
    assert report.equivocating_posters == ["vendor"]
    assert rep_app.reputation_ledger.reputation("vendor") < before  # poster dinged once


def test_check_root_consistency_honest_set_does_not_ding():
    rep_app = _app()
    rep_app.use_reputation_ledger()
    book = SettlementBook("acme", reputation=rep_app.reputation_ledger)
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    a = book.attest_liabilities("vendor", {"acme": 60.0}, attestor="vendor", as_of=as_of, sign=False)
    same = book.attest_liabilities("vendor", {"acme": 60.0}, attestor="vendor", as_of=as_of, sign=False)
    before = rep_app.reputation_ledger.reputation("vendor")
    report = book.check_root_consistency([("acme", a), ("globex", same)])
    assert report.consistent
    assert rep_app.reputation_ledger.reputation("vendor") == before


def test_check_history_consistency_catches_unexplained_drop():
    rep_app = _app()
    rep_app.use_reputation_ledger()
    book = SettlementBook(
        "acme", signer=_signer("acme"), audit=AuditLog(directory=None),
        reputation=rep_app.reputation_ledger,
    )
    early = datetime(2026, 1, 1, tzinfo=UTC)
    late = datetime(2026, 2, 1, tzinfo=UTC)
    first = book.attest_liabilities("vendor", {"acme": 80.0}, attestor="vendor", as_of=early, sign=False)
    # A LATER snapshot where the debt to acme silently shrinks with no discharge.
    second = book.attest_liabilities(
        "vendor", {"acme": 20.0}, attestor="vendor", as_of=late, prior=first, sign=False
    )
    before = rep_app.reputation_ledger.reputation("vendor")
    report = book.check_history_consistency([first, second])
    assert not report.consistent
    assert rep_app.reputation_ledger.reputation("vendor") < before


def test_guard_collateral_audits_and_signs():
    book = SettlementBook("acme", signer=_signer("acme"), audit=AuditLog(directory=None))
    c = _contract(buyer="acme", seller="vendor", price_usd=0.10)
    pool = book.post_collateral_pool([c], fraction=0.5, poster="vendor", sign=False)
    ledger = book.guard_collateral([pool], poster="vendor", held=100.0)
    assert ledger.signatures  # signed as the owner
    assert ledger.audit_id is not None


def test_settle_saga_on_book_signs_each_record():
    from vincio.choreography import Saga, StepOutcome

    app = _app("coord")
    book = SettlementBook("coord", signer=_signer("coord"))
    c_res = Contract(
        buyer="coord", seller="wh", terms=ContractTerms(scope="reserve", price_usd=0.20)
    ).seal()
    c_chg = Contract(
        buyer="coord", seller="pay", terms=ContractTerms(scope="charge", price_usd=0.10)
    ).seal()
    saga = (
        Saga(name="fulfil")
        .step("reserve", participant="wh", action="reserve", contract=c_res)
        .step("charge", participant="pay", action="charge", contract=c_chg)
    )
    parts = {
        "wh": {"reserve": lambda p: StepOutcome(ok=True, cost_usd=0.15)},
        "pay": {"charge": lambda p: StepOutcome(ok=True, cost_usd=0.08)},
    }
    result = app.choreograph(saga, participants=parts)
    records = book.settle_saga(result, contracts={c_res.id: c_res, c_chg.id: c_chg})
    assert len(records) == 2
    assert all("coord" in r.signed_by for r in records)  # signed as the buyer side
    assert book.verify().intact


def test_settle_saga_book_missing_contract_raises():
    from vincio.choreography import Saga, StepOutcome

    app = _app("coord")
    book = SettlementBook("coord")
    c = Contract(buyer="coord", seller="wh", terms=ContractTerms(price_usd=0.10)).seal()
    saga = Saga(name="s").step("a", participant="wh", action="do", contract=c)
    parts = {"wh": {"do": lambda p: StepOutcome(ok=True, cost_usd=0.05)}}
    result = app.choreograph(saga, participants=parts)
    with pytest.raises(SettlementError, match="no matching contract"):
        book.settle_saga(result, contracts={})  # the terms were never supplied


# -- remaining gap closers ----------------------------------------------------


def test_settle_escrow_in_one_call_is_audited():
    audit = AuditLog(directory=None)
    book = SettlementBook("vendor", signer=_signer("vendor"), audit=audit)
    c = _contract(buyer="acme", seller="vendor", price_usd=0.10)
    escrow = book.post_escrow(c, fraction=0.5)
    book.settle(c, cost_usd=0.05, latency_ms=100, quality=0.9, escrow=escrow)
    # The escrow post and its resolution both land on the audit chain.
    assert len(audit.query(action="escrow")) >= 2
    assert escrow.audit_id is not None


def test_resolve_insolvency_is_audited():
    audit = AuditLog(directory=None)
    book = SettlementBook("trustee", signer=_signer("trustee"), audit=audit)
    custody = book.attest_custody("debtor", 10.0, sign=False)
    liab = book.attest_liabilities("debtor", {"acme": 100.0}, sign=False)
    resolution = book.resolve_insolvency(custody, liab)
    assert resolution.audit_id is not None
    assert audit.query(action="insolvency_resolution")


def test_resolve_insolvency_reputation_disabled_does_not_ding():
    rep_app = _app()
    rep_app.use_reputation_ledger()
    book = SettlementBook("acme", reputation=rep_app.reputation_ledger)
    before = rep_app.reputation_ledger.reputation("debtor")
    custody = book.attest_custody("debtor", 10.0, sign=False)
    liab = book.attest_liabilities("debtor", {"acme": 100.0}, sign=False)
    resolution = book.resolve_insolvency(custody, liab, record_reputation=False)
    assert resolution.solvent is False
    assert rep_app.reputation_ledger.reputation("debtor") == before  # flag off -> no ding


def test_check_root_consistency_without_audit_or_reputation():
    # Equivocation surfaced on a bare book (no audit, no reputation): the helpers
    # take their early-return paths but the report still pinpoints the poster.
    book = SettlementBook("acme")
    attestor = _signer("attestor")
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    a = book.attest_liabilities("vendor", {"acme": 60.0}, attestor="attestor", as_of=as_of, sign=False)
    a.sign(attestor, party="attestor")
    b = book.attest_liabilities(
        "vendor", {"globex": 40.0}, attestor="attestor", as_of=as_of, sign=False
    )
    b.sign(attestor, party="attestor")
    report = book.check_root_consistency([("acme", a), ("globex", b)], verifier=attestor)
    assert report.equivocating_posters == ["vendor"]


def test_check_history_consistency_reputation_disabled():
    rep_app = _app()
    rep_app.use_reputation_ledger()
    book = SettlementBook("acme", reputation=rep_app.reputation_ledger)
    early = datetime(2026, 1, 1, tzinfo=UTC)
    late = datetime(2026, 2, 1, tzinfo=UTC)
    first = book.attest_liabilities("vendor", {"acme": 80.0}, attestor="vendor", as_of=early, sign=False)
    second = book.attest_liabilities(
        "vendor", {"acme": 20.0}, attestor="vendor", as_of=late, prior=first, sign=False
    )
    before = rep_app.reputation_ledger.reputation("vendor")
    report = book.check_history_consistency([first, second], record_reputation=False)
    assert not report.consistent
    assert rep_app.reputation_ledger.reputation("vendor") == before  # flag off -> no ding


def test_revoke_without_signer_is_unsigned():
    # A signer-less book can still revoke its own attestation; the revocation is unsigned.
    issuer = SettlementBook("acme", signer=_signer("acme"))
    issuer.settle(
        _contract(buyer="acme", seller="vendor"), cost_usd=0.05, latency_ms=100, quality=0.9
    )
    attestation = issuer.attest("vendor")
    bare = SettlementBook("acme")  # same owner, no signer
    revocation = bare.revoke(attestation)
    assert revocation.issuer == "acme"
    assert revocation.signatures == []


def test_net_without_signer_is_unsigned():
    book = SettlementBook("acme")  # no signer
    book.settle(_contract(buyer="acme", seller="vendor"), cost_usd=0.05)
    netting = book.net()
    assert netting.signatures == []


def test_settle_records_reputation_on_fulfilment():
    rep_app = _app()
    rep_app.use_reputation_ledger()
    book = SettlementBook("acme", reputation=rep_app.reputation_ledger)
    # Start from a known-bad standing, then a clean delivery should credit the seller up.
    rep_app.reputation_ledger.record_outcome("vendor", passed=False, round_id="seed")
    low = rep_app.reputation_ledger.reputation("vendor")
    book.settle(
        _contract(buyer="acme", seller="vendor", price_usd=0.10),
        cost_usd=0.05,
        latency_ms=100,
        quality=0.95,  # a clean delivery credits the seller
    )
    assert rep_app.reputation_ledger.reputation("vendor") > low


def test_check_completeness_on_unaudited_book():
    book = SettlementBook("acme", signer=_signer("acme"))  # no audit attached
    book.settle(_contract(buyer="debtor", seller="acme", price_usd=0.10), cost_usd=0.05)
    liab = book.attest_liabilities("debtor", {"other": 5.0}, sign=False)
    proof = book.check_completeness(liab)
    assert proof.complete is False
    assert proof.audit_id is None  # no audit chain to land on


def test_arbitrate_signs_when_signer_attached():
    signer = _signer("acme")
    book = SettlementBook("acme", signer=signer)
    c = _contract(buyer="acme", seller="vendor")
    book.settle(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    theirs = settle_contract(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    theirs.sign(_signer("vendor"), party="vendor")
    resolution = book.arbitrate(theirs, contract_id=c.id)  # sign defaults True
    assert resolution.signatures  # signed as the arbiter


def test_arbitrate_with_no_inferable_target_uses_whole_book():
    book = SettlementBook("acme", signer=_signer("acme"))
    c1 = _contract(buyer="acme", seller="vendor", scope="a")
    c2 = _contract(buyer="acme", seller="rival", scope="b")
    book.settle(c1, cost_usd=0.05, latency_ms=100, quality=0.9)
    book.settle(c2, cost_usd=0.05, latency_ms=100, quality=0.9)
    t1 = settle_contract(c1, cost_usd=0.05, latency_ms=100, quality=0.9)
    t2 = settle_contract(c2, cost_usd=0.05, latency_ms=100, quality=0.9)
    # Two different contract ids -> no single target inferred; the whole book is pooled
    # and arbitration refuses to settle records spanning several contracts.
    with pytest.raises(SettlementError, match="spanning several contracts"):
        book.arbitrate(t1, t2, sign=False)


def test_pool_party_resolves_none_when_owner_not_a_party():
    # A signer-bearing book whose owner is not a pool party: signing is skipped because
    # _resolve_pool_party returns None.
    book = SettlementBook("outsider", signer=_signer("outsider"))
    c = _contract(buyer="acme", seller="vendor", price_usd=0.10)
    pool = book.post_collateral_pool([c], fraction=0.5, poster="vendor")
    assert pool.signatures == []  # owner is on neither side of the pool


def test_pool_with_explicit_party_signs_as_that_party():
    # An explicit party= is honored verbatim by _resolve_pool_party.
    book = SettlementBook("acme", signer=_signer("acme"))
    c = _contract(buyer="acme", seller="vendor", price_usd=0.10)
    pool = book.post_collateral_pool([c], fraction=0.5, poster="vendor", party="vendor")
    assert "vendor" in {s.party for s in pool.signatures}


def test_report_over_all_counterparties_aggregates_rows():
    book = SettlementBook("acme")
    book.settle(_contract(seller="v1", price_usd=0.10), cost_usd=0.08, latency_ms=100, quality=0.9)
    book.settle(_contract(seller="v1", price_usd=0.10), cost_usd=0.20, quality=0.2)  # breach
    book.settle(_contract(seller="v2", price_usd=0.30), cost_usd=0.10, latency_ms=100, quality=0.9)
    report = book.report()  # no counterparty -> every party, populated rows
    by_party = {r.counterparty: r for r in report.rows}
    assert by_party["v1"].settlements == 2
    assert by_party["v1"].settled == 1 and by_party["v1"].breached == 1
    assert by_party["v1"].total_owed_usd == pytest.approx(0.20)
    assert by_party["v1"].total_delivered_usd == pytest.approx(0.28)
    assert by_party["v2"].net_balance_usd == pytest.approx(0.20)  # 0.30 - 0.10


def test_settle_with_explicit_party_overrides_default():
    book = SettlementBook("acme", signer=_signer("acme"))
    # Force signing as a named party even though resolution would pick the owner.
    record = book.settle(
        _contract(buyer="acme", seller="vendor"), cost_usd=0.05, party="acme"
    )
    assert "acme" in record.signed_by


def test_signerless_wrappers_skip_signing():
    # A book with no signer runs every wrapper's body but never signs.
    book = SettlementBook("acme")  # no signer
    custody = book.attest_custody("acme", 100.0)
    liab = book.attest_liabilities("acme", {"v": 40.0})
    proof = book.prove_solvency(custody, liab)
    assert custody.signatures == [] and liab.signatures == []
    assert proof.signatures == []
    schedule = book.build_seniority_schedule("acme", [["v"]])
    assert schedule.signatures == []
    discharge = book.discharge_liability("debtor", 5.0)
    assert discharge.signatures == []
    c = _contract(buyer="acme", seller="vendor", price_usd=0.10)
    pool = book.post_collateral_pool([c], fraction=0.5, poster="vendor")
    ledger = book.guard_collateral([pool], poster="vendor", held=100.0)
    assert pool.signatures == [] and ledger.signatures == []
    st = book.build_set_off_statement("vendor", "acme", owed_usd=10.0, owing_usd=4.0)
    assert st.signatures == []


def test_load_record_keeps_created_at_when_not_a_string():
    # When the projection's created_at is already a datetime (not iso str), it is left
    # as the book's existing value (the str-parse branch is skipped).
    book = SettlementBook("acme")
    book.settle(_contract(seller="vendor"), cost_usd=0.05)
    projection = book.to_record()
    projection["created_at"] = None  # not a str -> the isoformat parse branch is skipped
    original = book.created_at
    book.load_record(projection)
    assert book.created_at == original  # unchanged


def test_settle_reputation_disabled_does_not_record():
    rep_app = _app()
    rep_app.use_reputation_ledger()
    book = SettlementBook("acme", reputation=rep_app.reputation_ledger)
    before = rep_app.reputation_ledger.reputation("vendor")
    book.settle(
        _contract(buyer="acme", seller="vendor"),
        cost_usd=0.50,
        quality=0.1,  # a breach that would normally debit
        record_reputation=False,
    )
    assert rep_app.reputation_ledger.reputation("vendor") == before
