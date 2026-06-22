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

## Netting & multilateral clearing

Once an org keeps many bilateral books, it is often both a buyer and a seller across
a web of contracts. **Netting** folds those bilateral balances into a single minimal
set of net obligations, so the books close once. `net_settlements` (over loose
records) and `net_books` (over whole books) read only the existing signed,
hash-chained records and produce a content-bound `NettingSet`:

```python
from vincio import net_settlements, settle_contract

# A cycle: acme owes vendor, vendor owes data, data owes acme.
fleet = [
    settle_contract(acme_vendor, cost_usd=0.08),
    settle_contract(vendor_data, cost_usd=0.05),
    settle_contract(data_acme, cost_usd=0.03),
]
netting = app.clear_settlements(records=fleet)   # signs as the clearer, audits

print(netting.gross_edges, "→", netting.cleared_transfers)   # 3 → 2
for o in netting.obligations:
    print(o.debtor, "→", o.creditor, o.amount_usd)
```

Each settled contract is a directed obligation — the buyer owes the seller the
agreed price for the scope (the payable cap; a breach is surfaced by the
settlement's own status and the reputation loop, it does not change what is owed).
The directed payables aggregate per pair, collapse to one `BilateralNet` per
counterparty, and clear to the minimal set of `NetObligation` transfers — at most
`N − 1` for `N` parties, net-debtors paying net-creditors, deterministically.

The `NettingSet` is offline-verifiable the way a record is. `verify()` recomputes
the netting hash from the bytes alone, confirms the net positions balance to zero,
and confirms the cleared transfers reproduce every org's position:

```python
verdict = netting.verify(app.contract_signer)
assert verdict.valid and verdict.positions_balanced and verdict.conserves
```

Netting reads only signed records and asserts nothing it cannot recompute. A
tampered source record (its reconciliation hash no longer recomputes) is refused
outright, and two books that disagree on the same contract are pinpointed as a
`NettingDispute` — excluded from the clearing, never silently absorbed:

```python
netting.require_clean()   # raises if any contract was disputed
```

`book.net()` nets a single org's own book into its position against each
counterparty — the same calculation, one ledger.

## Resolving a dispute

Netting *pinpoints* a disagreement as a `NettingDispute`, and `reconcile` pinpoints
one between two records — but neither *resolves* it. **Arbitration** does: each party
submits its signed `SettlementRecord`s for the disputed contract and the
deterministic `arbitrate` decides which figure stands, producing a content-bound,
offline-verifiable `Resolution`.

```python
from vincio import arbitrate

# The buyer and the seller each submit their signed record for the contract.
resolution = app.arbitrate([buyer_record, seller_record])   # signs, audits
print(resolution.status)                                    # "upheld" | "unresolved"
```

The decision rests on nothing it cannot recompute. A reconciliation hash that **both**
the buyer and the seller signed — each on their own record, the two co-signing one
figure exactly as `reconcile` describes — is mutually corroborated and **stands**; a
unilateral claim contradicting it is **rejected** and its claimant pinpointed; a single
uncontested figure stands on its own. When neither side's figure is corroborated the
dispute is honestly left **unresolved** rather than decided by fiat:

```python
if resolution.upheld:
    print(resolution.upheld_balance_usd, "corroborated by", resolution.corroborated_by)
    for claim in resolution.rejected_claims:
        print("rejected:", claim.settlement_id, claim.reason)
else:
    resolution.require_resolved()   # raises: no figure was corroborated
```

Unlike netting, which *refuses* to clear over a tampered book, arbitration is the
venue where a bad claim is adjudicated: a tampered or forged claim is marked
**inadmissible** and pinpointed (`resolution.inadmissible_claims`), never silently
dropped and never crashing the resolution. The `Resolution` is offline-verifiable the
way a record is — `verify()` recomputes the resolution hash from the bytes alone and
re-derives the whole decision from the recorded claims, so a flipped verdict is caught
even after re-sealing, and two arbiters reading the same records compute the same
co-signable hash:

```python
verdict = resolution.verify(app.contract_signer)
assert verdict.valid and verdict.hash_ok and verdict.decision_sound
```

Resolving a dispute also **closes the reputation loop**: the party whose claim did not
stand is debited, so a bad-faith revision weights the next negotiation. `book.arbitrate()`
resolves one org's own record against a counterparty's submitted claims — the dispute
counterpart of `book.reconcile_with()`.

## Porting reputation across orgs

Settlement, netting, and arbitration all close the reputation loop — but the standing
they earn lives inside one org's own ledger. A *new* counterparty has no way to trust it
without a hosted reputation bureau. `app.attest_reputation()` (or `book.attest()`) issues
a signed, offline-verifiable `ReputationAttestation` over a counterparty's earned
standing, derived only from an org's own signed records and arbitration resolutions — a
fulfilled delivery a success, a breach or a dissent a failure. It reads only what it can
recompute (a tampered own record is skipped, the exact source hashes bound):

```python
# Each org attests the vendor's standing from its own settlement book.
acme_attestation = acme.attest_reputation("vendor")       # signed by acme, audited
globex_attestation = globex.attest_reputation("vendor")   # signed by globex

# Offline-verifiable: the hash recomputes and the reputation re-derives from the
# evidence counts, so a tampered score is caught even after re-sealing.
assert acme_attestation.verify(acme.contract_signer).valid
```

A prospective counterparty combines several issuers' attestations into a bounded,
evidence-weighted prior with `app.import_reputation()` (or `combine_attestations()`).
Because a Beta-Bernoulli posterior is conjugate, combining is *pooling the evidence* —
never a single self-asserted number: an issuer that vouches for itself is refused, an
issuer cannot stack its own pull (only its largest attestation counts), and a tampered or
forged attestation is pinpointed (`prior.refused`) and excluded:

```python
buyer.use_reputation_ledger()
prior = buyer.import_reputation([acme_attestation, globex_attestation])

# The prior exposes weight(member_id), so it drops into the existing negotiation path:
# a never-before-seen vendor is weighted by what its past counterparties attest, under
# the same bounded [floor, 1] rule a local reputation is — discounted, never zeroed.
result = buyer.negotiate("transcribe calls", buyer=..., seller=..., seller_id="vendor")
```

The imported prior weights an *unknown* counterparty; one the buyer has lived through
keeps its own earned `ReputationLedger` standing (passed as the `base`), so portability
only fills the gap where there is no local history — `prior.standing("vendor")` shows the
pooled evidence and `prior.weight("vendor")` the bounded weight it maps to.

## Keeping a ported reputation current

An attestation is a *point-in-time* claim, but standing changes — so the portable prior
is **time-aware and revocable**, reflecting current standing rather than a frozen
snapshot, without a hosted revocation service.

**Freshness.** An issuer declares a validity window when it attests
(`horizon_days=`), bound into the attestation's signed hash. Against an as-of clock, a
combination *excludes* a stale attestation (past its window) and *decays* an older one
within its window by an importer-set half-life — its evidence mass halved each
`half_life_days`, its attested ratio preserved — so an old attestation eases out of the
pooled prior rather than anchoring it forever:

```python
from datetime import timedelta
from vincio.core.utils import utcnow
from vincio.settlement import AttestationConfig

att = acme.attest_reputation("vendor", horizon_days=90)  # issuer's validity window
cfg = AttestationConfig(half_life_days=30)               # importer's decay policy
prior = buyer.import_reputation([att], config=cfg, as_of=utcnow())
# A stale attestation lands in prior.stale (pinpointed); a fresh one decays by age.
```

**Revocation.** An issuer withdraws a claim it can no longer stand behind by signing a
content-bound `AttestationRevocation` (`app.revoke_attestation()` / `book.revoke()`) that
names the attestation **by its hash**. The importer passes revocations to the combination,
which excludes the withdrawn claim — pinpointed in `prior.revoked`, never silently
honored. Only the issuer can withdraw its *own* attestation: a forged revocation, or one
naming another org's attestation, cannot cancel a claim.

```python
revocation = acme.revoke_attestation(att, reason="vendor regressed")  # signed, audited
prior = buyer.import_reputation([att], revocations=[revocation])
assert prior.standing("vendor") is None        # the withdrawn claim is excluded
assert prior.revoked[0].issuer == "acme"       # pinpointed, not silently dropped
```

Freshness and revocation read only the existing signed artifacts, assert nothing they
cannot recompute, and fold into the *same* bounded `[floor, 1]` weighting — never a
central revocation service or a hosted bulletin board.

## What it is not

This is a library capability inside your process, not a payment rail or a hosted
marketplace. There is no money movement, no escrow service, no managed clearing
house, no arbitration service or court of record, no reputation bureau — a settlement
is a typed, signed record you hold and verify yourself, the book is a hash-chained
ledger each organization keeps on its own, netting is a clearing *calculation* over
those books, not a clearing *house*, arbitration is a deterministic adjudication over
the parties' own signed records, not a third party that rules by fiat, and a reputation
attestation is one org's signed, verifiable claim that combines into an evidence-weighted
prior, not a central score. Vincio gives you a verifiable reconciliation of what was owed
and delivered, a verifiable netting of it across a fleet, a verifiable resolution when two
books disagree, and a portable, verifiable attestation of earned standing; how an
obligation is paid is yours.
