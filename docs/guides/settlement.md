# Guide: agent-to-agent settlement & metering

Vincio already lets agents negotiate a `Contract` and run durable, compensating
sagas across organizations. This guide covers the next rung: **closing the books**
on that work — a metered, auditable settlement record for work delivered under a
negotiated contract, so a cross-org engagement reconciles the way a run closes its
cost report. It is never a payment rail, only a verifiable ledger of what was owed
and delivered: usage accrues against the agreed price as the work completes, a
signed settlement record reconciles delivery against the contract terms and
verifies offline from the bytes alone, and a settled overrun or shortfall feeds the
reputation that weights the next negotiation.

This is additive (`vincio.settlement`); it changes nothing about how a contract is
negotiated or a saga runs, and works fully offline against deterministic figures.

## Metering delivery

A `Meter` accrues the **usage** of work delivered under one contract — each unit a
`UsageEvent` attributed to the contract and the run. The reading is a deterministic,
total-preserving roll-up: cost and latency are summed, quality is the minimum
observed (the figure to hold against a *floor*), and the totals are exactly the sum
of the events — no double-count, no drop.

```python
from vincio import ContextApp

app = ContextApp(name="acme")
deal = app.negotiate("transcribe 1k support calls", buyer=buyer, seller=seller)

meter = app.meter(deal.contract)
meter.accrue(units=500, cost_usd=0.04, latency_ms=1200, quality=0.95, step="batch-1")
meter.accrue(units=500, cost_usd=0.03, latency_ms=900,  quality=0.92, step="batch-2")

reading = meter.reading()
print(reading.cost_usd, reading.latency_ms, reading.quality)  # 0.07  2100.0  0.92
```

Metering is pure accumulation — it records what was delivered, the way the cost
report attributes spend. Enforcing a cap stays where it already lives: a contract's
`to_budget()` and `check()`.

## A signed, offline-verifiable settlement

`app.settle(...)` reconciles the delivered work against the contract's agreed
price / SLA / quality into a typed `SettlementRecord`, signs it as your side of the
contract, and records the verdict on the audit chain. Pass a metered `reading` or
explicit delivered figures.

```python
record = app.settle(deal.contract, reading=reading)

print(record.status)           # "settled" | "breached"
print(record.amount_owed_usd)  # the agreed price for the scope
print(record.balance_usd)      # price − delivered cost (+credit / −overrun)
print(record.breaches)         # the breaching dimensions, if any

record.verify(app.contract_signer)   # offline-verifiable from the bytes alone
```

The record binds the **economic facts** — contract, parties, agreed terms,
delivered metrics, balance — into one reconciliation hash that both parties sign.
`verify()` recomputes it from the stored fields, so a tampered figure or a forged
signature is caught without the live parties:

```python
record.balance_usd = 999.0                      # tamper
assert not record.verify(app.contract_signer).valid
```

A breach is **not** an error — it reconciles to a record with `status="breached"`
and the breaching dimensions on `.breaches` (an overrun on price/SLA, a shortfall on
quality). Settling is closing the books, not raising an alarm.

## Reconciling across the boundary

Because both sides compute the *same* deterministic reconciliation hash from the
same contract and delivery, the buyer's record and the seller's record co-sign one
hash when their books agree. `reconcile` ties two independently-produced records out
and pinpoints any discrepancy as a dispute:

```python
from vincio import reconcile

# Each org settles its own view of the delivery on its own book.
ours   = app.settle(deal.contract, reading=reading)
theirs = counterparty_record   # received over the wire (record.to_wire / from_wire)

verdict = reconcile(ours, theirs)
if verdict.agrees:
    ...                         # the books tie out; both hashes match
else:
    print(verdict.discrepancies)  # e.g. ["delivered_cost_usd: 0.15 != 0.18"]
```

`record.to_wire()` / `SettlementRecord.from_wire(...)` move a record across the wire;
reconciliation is purely offline.

## The settlement book

Attach a `SettlementBook` to keep a durable, **hash-chained** ledger of an org's
settlements — the settlement analogue of the saga journal. Each record links to the
previous by an entry hash, so `verify()` recomputes the whole book offline and
pinpoints any edited record, while `report()` rolls the books up per counterparty.

```python
app.use_settlement_book()

app.settle(deal.contract, reading=reading)
app.settle(other_contract, cost_usd=0.08, quality=0.7)   # a breach

assert app.settlement_book.verify().intact      # the whole ledger recomputes
app.settlement_report().print_summary()
#   vendor: owed=$0.15 delivered=$0.15 balance=$+0.00 (1 settled / 1 breached)
```

## Settling a whole saga

`app.settle_saga(...)` closes the books on every contract a cross-org saga ran
under, straight from its durable journal: it meters each contracted step and
reconciles the per-step delivery against the matching contract, appending one signed
record per contract to the book.

```python
result = app.choreograph(saga, participants=participants)   # steps carry contracts

records = app.settle_saga(
    result,
    contracts={c.id: c for c in (reserve_deal, charge_deal)},
)
for r in records:
    print(r.seller, r.status, r.balance_usd)
```

## Closing the reputation loop

When a reputation ledger is attached, closing the books also closes the loop: a
settlement that fulfils its terms credits the seller, and one that overruns or falls
short debits it — so reliability earned in *delivery* weights the next negotiation,
where a discounted counterparty's offers are weighted down without being singled
out.

```python
app.use_reputation_ledger()
app.use_settlement_book()

app.settle(deal.contract, cost_usd=0.20, quality=0.6)   # a shortfall
print(app.reputation_report(deal.contract.seller).rows[0].reputation)  # debited
```

## What it is not

This is a library capability inside your process, not a payment rail or a hosted
marketplace. There is no money movement, no escrow service, no managed clearing
house — a settlement is a typed, signed record you hold and verify yourself, and the
book is a hash-chained ledger each organization keeps on its own. Vincio gives you a
verifiable reconciliation of what was owed and delivered; how an obligation is paid
is yours.
