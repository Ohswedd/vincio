"""Agent-to-agent settlement & metering: metering, reconciliation, the book."""

from __future__ import annotations

import pytest

from vincio import ContextApp
from vincio.choreography import Saga, StepOutcome
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement import (
    Meter,
    Reconciliation,
    SettlementBook,
    SettlementRecord,
    UsageEvent,
    reconcile,
    settle_contract,
    settle_saga,
)
from vincio.storage.base import InMemoryMetadataStore


def _app(name: str = "acme") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def _contract(buyer="acme", seller="vendor", **terms) -> Contract:
    defaults = {"scope": "work", "price_usd": 0.10, "sla_seconds": 5.0, "quality_floor": 0.8}
    defaults.update(terms)
    return Contract(buyer=buyer, seller=seller, terms=ContractTerms(**defaults)).seal()


# -- metering -----------------------------------------------------------------


def test_meter_reading_is_sum_of_events():
    meter = Meter("c1", run_id="run-1")
    meter.accrue(units=500, cost_usd=0.04, latency_ms=1200, quality=0.95, step="a")
    meter.accrue(units=500, cost_usd=0.03, latency_ms=900, quality=0.90, step="b")
    reading = meter.reading()
    assert reading.events == 2
    assert reading.units == 1000.0
    assert reading.cost_usd == 0.07  # summed
    assert reading.latency_ms == 2100.0  # summed
    assert reading.quality == 0.90  # minimum (weakest link)
    assert reading.per_step == {"a": 0.04, "b": 0.03}
    assert reading.run_id == "run-1"
    assert reading.metered is True


def test_empty_reading_is_unmetered():
    reading = Meter("c1").reading()
    assert reading.events == 0
    assert reading.metered is False
    assert reading.cost_usd is None


def test_meter_rejects_negative_units():
    with pytest.raises(SettlementError):
        Meter("c1").accrue(units=-1, cost_usd=0.01)


def test_meter_requires_contract_id():
    with pytest.raises(SettlementError):
        Meter("")


def test_accrue_event_must_match_contract():
    meter = Meter("c1")
    foreign = UsageEvent(contract_id="other", cost_usd=0.01)
    with pytest.raises(SettlementError):
        meter.accrue_event(foreign)


def test_usage_event_wire_roundtrip():
    event = UsageEvent(contract_id="c1", cost_usd=0.05, step="x")
    assert UsageEvent.from_wire(event.to_wire()).cost_usd == 0.05


# -- settle_contract ----------------------------------------------------------


def test_settle_fulfilled_contract():
    c = _contract(price_usd=0.10, sla_seconds=5.0, quality_floor=0.8)
    record = settle_contract(c, cost_usd=0.08, latency_ms=1200, quality=0.9)
    assert record.status == "settled"
    assert record.fulfilled is True
    assert record.amount_owed_usd == 0.10
    assert record.balance_usd == pytest.approx(0.02)
    assert record.credit_usd == pytest.approx(0.02)
    assert record.overrun_usd == 0.0
    assert not record.breaches
    # one line per constrained dimension, all within
    dims = {line.dimension: line for line in record.lines}
    assert set(dims) == {"price", "sla", "quality"}
    assert all(line.within for line in record.lines)


def test_settle_breached_contract_records_breaches():
    c = _contract(price_usd=0.05, sla_seconds=1.0, quality_floor=0.9)
    record = settle_contract(c, cost_usd=0.08, latency_ms=2000, quality=0.7)
    assert record.status == "breached"
    assert record.fulfilled is False
    assert record.balance_usd == pytest.approx(-0.03)
    assert record.overrun_usd == pytest.approx(0.03)
    kinds = {b.split(":", 1)[0] for b in record.breaches}
    assert kinds == {"price", "sla", "quality"}
    assert all(not line.within for line in record.lines)


def test_unmetered_dimension_is_not_a_breach():
    c = _contract(price_usd=0.10, sla_seconds=5.0, quality_floor=0.8)
    record = settle_contract(c, cost_usd=0.05)  # latency/quality not metered
    assert record.status == "settled"
    sla = next(line for line in record.lines if line.dimension == "sla")
    assert sla.within is True and sla.delivered is None and sla.note == "not metered"


def test_settle_from_reading():
    c = _contract(price_usd=0.10)
    meter = Meter(c.id)
    meter.accrue(units=1, cost_usd=0.03)
    meter.accrue(units=1, cost_usd=0.04)
    record = settle_contract(c, reading=meter.reading())
    assert record.delivered_cost_usd == 0.07
    assert record.metered_events == 2
    assert record.metered_units == 2.0


# -- signing & offline verification -------------------------------------------


def test_record_signs_and_verifies_offline():
    c = _contract()
    signer = HMACSigner("k", key_id="acme")
    record = settle_contract(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    record.sign(signer, party="acme").sign(signer, party="vendor")
    assert record.fully_signed
    verdict = record.verify(signer)
    assert verdict.valid and verdict.hash_ok and verdict.signatures_ok
    assert set(verdict.signed_by) == {"acme", "vendor"}


def test_record_tamper_is_detected():
    c = _contract()
    signer = HMACSigner("k")
    record = settle_contract(c, cost_usd=0.05).sign(signer, party="acme")
    record.balance_usd = 999.0  # tamper an economic fact
    assert record.verify(signer, require=[]).hash_ok is False
    assert record.verify(signer).valid is False


def test_sign_rejects_non_party():
    c = _contract(buyer="acme", seller="vendor")
    with pytest.raises(SettlementError):
        settle_contract(c, cost_usd=0.05).sign(HMACSigner("k"), party="intruder")


def test_require_valid_raises_on_bad_record():
    c = _contract()
    record = settle_contract(c, cost_usd=0.05)  # unsigned
    with pytest.raises(SettlementError):
        record.require_valid(HMACSigner("k"))


def test_record_wire_roundtrip():
    c = _contract()
    record = settle_contract(c, cost_usd=0.05, quality=0.9).sign(HMACSigner("k"), party="acme")
    back = SettlementRecord.from_wire(record.to_wire())
    assert back.content_hash == record.content_hash
    assert back.signed_by == ["acme"]


# -- reconciliation -----------------------------------------------------------


def test_two_parties_records_reconcile():
    c = _contract()
    buyer_rec = settle_contract(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    seller_rec = settle_contract(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    # same economic facts → same reconciliation hash → both co-sign it
    assert buyer_rec.content_hash == seller_rec.content_hash
    verdict = reconcile(buyer_rec, seller_rec)
    assert isinstance(verdict, Reconciliation)
    assert verdict.agrees and verdict.hashes_match


def test_disagreeing_records_flag_a_dispute():
    c = _contract()
    buyer_rec = settle_contract(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    seller_rec = settle_contract(c, cost_usd=0.08, latency_ms=1000, quality=0.9)
    verdict = reconcile(buyer_rec, seller_rec)
    assert not verdict.agrees
    assert any("delivered_cost_usd" in d for d in verdict.discrepancies)


def test_reconcile_different_contracts_raises():
    a = settle_contract(_contract(), cost_usd=0.05)
    b = settle_contract(_contract(), cost_usd=0.05)  # different id
    with pytest.raises(SettlementError):
        reconcile(a, b)


# -- the settlement book ------------------------------------------------------


def test_book_is_hash_chained_and_verifies():
    book = SettlementBook("acme", signer=HMACSigner("k", key_id="acme"))
    book.settle(_contract(seller="v1"), cost_usd=0.05, latency_ms=1000, quality=0.9)
    book.settle(_contract(seller="v2"), cost_usd=0.05, latency_ms=1000, quality=0.9)
    assert book.verify().intact
    assert book.verify(HMACSigner("k", key_id="acme")).intact
    assert len(book.records) == 2
    assert book.records[0].seq == 0 and book.records[1].prev_hash == book.records[0].entry_hash


def test_book_tamper_breaks_the_chain():
    book = SettlementBook("acme")
    book.settle(_contract(), cost_usd=0.05)
    book.settle(_contract(), cost_usd=0.05)
    book.records[0].balance_usd = 1.0  # tamper an economic fact
    verdict = book.verify()
    assert not verdict.intact
    assert verdict.broken_at == 0


def test_book_persists_and_reloads_intact():
    store = InMemoryMetadataStore()
    # A stable book_id keeps one durable ledger across restarts (resume contract).
    book = SettlementBook("acme", store=store, book_id="acme-ledger")
    book.settle(_contract(seller="v1"), cost_usd=0.05)
    book.settle(_contract(seller="v2"), cost_usd=0.05)
    fresh = SettlementBook("acme", store=store, book_id="acme-ledger")
    assert fresh.verify().intact
    assert len(fresh.records) == 2


def test_book_require_intact_raises_on_tamper():
    book = SettlementBook("acme")
    book.settle(_contract(), cost_usd=0.05)
    book.records[0].seq = 99
    with pytest.raises(SettlementError):
        book.require_intact()


def test_book_report_rolls_up_per_counterparty():
    book = SettlementBook("acme")
    book.settle(_contract(seller="v1", price_usd=0.10), cost_usd=0.08, latency_ms=1000, quality=0.9)
    book.settle(_contract(seller="v1", price_usd=0.10), cost_usd=0.20, quality=0.5)  # breach
    report = book.report()
    row = report.rows[0]
    assert row.counterparty == "v1"
    assert row.settlements == 2 and row.settled == 1 and row.breached == 1
    assert row.total_owed_usd == pytest.approx(0.20)
    assert report.breached == 1


def test_book_reconcile_with_counterparty_record():
    book = SettlementBook("acme")
    c = _contract(seller="vendor")
    book.settle(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    theirs = settle_contract(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    assert book.reconcile_with(theirs).agrees
    with pytest.raises(SettlementError):
        book.reconcile_with(settle_contract(_contract(), cost_usd=0.05))


# -- reputation closing -------------------------------------------------------


def test_settlement_closes_reputation_loop():
    book_ledger_app = _app()
    book_ledger_app.use_reputation_ledger()
    book = SettlementBook("acme", reputation=book_ledger_app.reputation_ledger)
    c = _contract(seller="vendor", price_usd=0.10)
    book.settle(c, cost_usd=0.05, latency_ms=1000, quality=0.9)  # fulfilled
    good = book_ledger_app.reputation_ledger.reputation("vendor")
    book.settle(c, cost_usd=0.50, quality=0.2)  # breach
    after = book_ledger_app.reputation_ledger.reputation("vendor")
    assert after < good  # a breach debits the seller


# -- app surface --------------------------------------------------------------


def test_app_settle_signs_audits_and_books():
    app = _app()
    app.use_settlement_book()
    c = _contract(buyer="acme", seller="vendor")
    record = app.settle(c, cost_usd=0.05, latency_ms=1000, quality=0.9)
    assert record.status == "settled"
    assert "acme" in record.signed_by  # signed as this app's side
    assert record.verify(app.contract_signer, require=["acme"]).valid
    assert app.settlement_book.verify().intact
    assert app.audit.query(action="settlement")
    assert app.audit.verify_chain()


def test_app_settle_without_book_uses_transient():
    app = _app()
    c = _contract()
    record = app.settle(c, cost_usd=0.05)
    assert record.status == "settled"
    assert app.settlement_report().rows == []  # no attached book


def test_app_meter_and_settle():
    app = _app()
    app.use_settlement_book()
    c = _contract(seller="vendor", price_usd=0.10)
    meter = app.meter(c)
    meter.accrue(units=1, cost_usd=0.04)
    meter.accrue(units=1, cost_usd=0.03)
    record = app.settle(c, reading=meter.reading())
    assert record.delivered_cost_usd == 0.07
    assert app.settlement_report("vendor").rows[0].total_delivered_usd == 0.07


async def test_app_settle_saga_closes_every_contract():
    app = _app("coord")
    app.use_settlement_book()
    app.use_reputation_ledger()
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
        "wh": {"reserve": lambda p: StepOutcome(ok=True, cost_usd=0.15, output={"r": 1})},
        "pay": {"charge": lambda p: StepOutcome(ok=True, cost_usd=0.08, output={"c": 1})},
    }
    result = await app.achoreograph(saga, participants=parts)
    assert result.status == "completed"
    records = app.settle_saga(result, contracts={c_res.id: c_res, c_chg.id: c_chg})
    assert len(records) == 2
    assert all(r.status == "settled" for r in records)
    assert all(r.saga_id == result.saga_id for r in records)
    assert app.settlement_book.verify().intact


def test_settle_saga_missing_contract_raises():
    journal_app = _app("coord")
    c = Contract(buyer="coord", seller="wh", terms=ContractTerms(price_usd=0.10)).seal()
    saga = Saga(name="s").step("a", participant="wh", action="do", contract=c)
    parts = {"wh": {"do": lambda p: StepOutcome(ok=True, cost_usd=0.05)}}
    result = journal_app.choreograph(saga, participants=parts)
    with pytest.raises(SettlementError):
        settle_saga(result, contracts={})  # missing the contract terms


def test_settlement_report_empty_without_book():
    app = _app()
    report = app.settlement_report()
    assert report.rows == [] and report.owner == "acme"
