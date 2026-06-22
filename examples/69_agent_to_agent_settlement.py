"""Agent-to-agent settlement & metering — closing the books on contracted work.

Vincio lets agents negotiate a contract and run durable, compensating sagas across
organizations. This example adds the next rung: **closing the books** on that work —
a metered, auditable settlement record reconciling delivered work against the
negotiated contract, the way a run closes its cost report. It is never a payment
rail, only a verifiable ledger of what was owed and delivered.

Six steps, all offline and deterministic:

  1. Meter: usage accrues against the agreed price as work completes; the reading
     is a total-preserving roll-up of the events.
  2. Settle: the delivered work reconciles against the agreed price/SLA/quality
     into a typed settlement record, signed and offline-verifiable.
  3. Breach: an overrun or shortfall reconciles to a breached settlement, with the
     breaching dimensions pinpointed — a settled outcome, not an error.
  4. Reconcile: two orgs' independently-produced records tie out (and a
     disagreement is flagged as a dispute) — the cross-org reconciliation.
  5. Book: the hash-chained settlement ledger verifies offline; a tampered record
     is caught, and the report rolls the books up per counterparty.
  6. Close the loop: settling a whole saga closes every contract from its durable
     journal, and a breach debits the seller's reputation for the next negotiation.

Everything here is opt-in and additive; this is a library capability inside your
process, never a hosted marketplace or a payment processor.
"""

from __future__ import annotations

import asyncio

from vincio import ContextApp, reconcile, settle_contract
from vincio.choreography import Saga, StepOutcome
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider


def a_contract(buyer: str, seller: str, **terms: float) -> Contract:
    base = {"scope": "transcribe 1k calls", "price_usd": 0.10, "sla_seconds": 5.0, "quality_floor": 0.8}
    base.update(terms)  # type: ignore[arg-type]
    return Contract(buyer=buyer, seller=seller, terms=ContractTerms(**base)).seal()  # type: ignore[arg-type]


async def main() -> None:
    app = ContextApp(name="acme", provider=MockProvider(default_text="ok"))
    app.use_reputation_ledger()
    app.use_settlement_book()

    contract = a_contract("acme", "vendor", price_usd=0.10)

    # 1. Meter the delivery: usage accrues against the agreed price as work runs.
    meter = app.meter(contract)
    meter.accrue(units=500, cost_usd=0.04, latency_ms=1200, quality=0.95, step="batch-1")
    meter.accrue(units=500, cost_usd=0.03, latency_ms=900, quality=0.92, step="batch-2")
    reading = meter.reading()
    print(
        f"1. Metered {reading.units:g} units over {reading.events} events: "
        f"cost=${reading.cost_usd:.2f} latency={reading.latency_ms:.0f}ms quality={reading.quality:.2f}"
    )

    # 2. Settle: reconcile the delivery against the agreed terms, signed.
    record = app.settle(contract, reading=reading)
    print(
        f"2. Settled status={record.status}; owed=${record.amount_owed_usd:.2f} "
        f"balance=${record.balance_usd:+.2f}; verifies offline="
        f"{record.verify(app.contract_signer, require=['acme']).valid}"
    )

    # 3. A breach reconciles to a breached settlement — not an error.
    breach_c = a_contract("acme", "vendor", price_usd=0.05, quality_floor=0.9)
    breached = app.settle(breach_c, cost_usd=0.08, quality=0.6)
    print(
        f"3. Breach status={breached.status}; overrun=${breached.overrun_usd:.2f}, "
        f"breaches={breached.breaches}"
    )

    # 4. Reconcile across the boundary: the seller's record ties out with ours.
    seller_view = settle_contract(contract, reading=reading)  # the vendor's own books
    verdict = reconcile(record, seller_view)
    dispute = reconcile(record, settle_contract(contract, cost_usd=0.20))
    print(
        f"4. Reconciliation agrees={verdict.agrees} (hashes match={verdict.hashes_match}); "
        f"a disagreement is flagged: agrees={dispute.agrees}, "
        f"discrepancies={dispute.discrepancies[:1]}"
    )

    # 5. The book is a hash-chained ledger; verify it and catch a tamper.
    print(f"5. Book verifies offline={app.settlement_book.verify().intact}; ", end="")
    tampered = app.settlement_book.to_record()
    tampered["records"][0]["balance_usd"] = 999.0
    from vincio.settlement import SettlementBook

    forged = SettlementBook("acme").load_record(tampered)
    print(f"a tampered record is caught={not forged.verify().intact}")
    app.settlement_report().print_summary()

    # 6. Settle a whole saga, closing every contract — and the reputation loop.
    c_res = Contract(buyer="acme", seller="wh", terms=ContractTerms(scope="reserve", price_usd=0.20)).seal()
    c_chg = Contract(buyer="acme", seller="pay", terms=ContractTerms(scope="charge", price_usd=0.10)).seal()
    saga = (
        Saga(name="fulfil")
        .step("reserve", participant="wh", action="reserve", contract=c_res)
        .step("charge", participant="pay", action="charge", contract=c_chg)
    )
    parts = {
        "wh": {"reserve": lambda p: StepOutcome(ok=True, cost_usd=0.15, output={"r": 1})},
        "pay": {"charge": lambda p: StepOutcome(ok=True, cost_usd=0.08, output={"c": 1})},
    }
    saga_result = await app.achoreograph(saga, participants=parts)
    records = app.settle_saga(saga_result, contracts={c_res.id: c_res, c_chg.id: c_chg})
    print(
        f"6. Saga settled {len(records)} contracts "
        f"({', '.join(f'{r.seller}:{r.status}' for r in records)}); "
        f"vendor reputation after a breach={app.reputation_report('vendor').rows[0].reputation:.3f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
