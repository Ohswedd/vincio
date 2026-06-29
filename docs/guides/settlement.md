# Guide: agent-to-agent settlement & metering

Vincio already lets agents negotiate a `Contract` and run durable, compensating
sagas across organizations. This guide covers the next rung: **closing the books**
on that work, a metered, auditable settlement record for work delivered under a
negotiated contract, so a cross-org engagement reconciles the way a run closes its
cost report. It is never a payment rail, only a verifiable ledger of what was owed
and delivered: usage accrues against the agreed price as the work completes, a
signed settlement record reconciles delivery against the contract terms and
verifies offline from the bytes alone, and a settled overrun or shortfall feeds the
reputation that weights the next negotiation.

This is additive (`vincio.settlement`); it changes nothing about how a contract is
negotiated or a saga runs, and works fully offline against deterministic figures.

## Metering delivery

A `Meter` accrues the **usage** of work delivered under one contract, each unit a
`UsageEvent` attributed to the contract and the run. The reading is a deterministic,
total-preserving roll-up: cost and latency are summed, quality is the minimum
observed (the figure to hold against a *floor*), and the totals are exactly the sum
of the events, no double-count, no drop.

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

Metering is pure accumulation, it records what was delivered, the way the cost
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

The record binds the **economic facts**, contract, parties, agreed terms,
delivered metrics, balance, into one reconciliation hash that both parties sign.
`verify()` recomputes it from the stored fields, so a tampered figure or a forged
signature is caught without the live parties:

```python
record.balance_usd = 999.0                      # tamper
assert not record.verify(app.contract_signer).valid
```

A breach is **not** an error; it reconciles to a record with `status="breached"`
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
settlements, the settlement analogue of the saga journal. Each record links to the
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
short debits it, so reliability earned in *delivery* weights the next negotiation,
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

Each settled contract is a directed obligation, the buyer owes the seller the
agreed price for the scope (the payable cap; a breach is surfaced by the
settlement's own status and the reputation loop, it does not change what is owed).
The directed payables aggregate per pair, collapse to one `BilateralNet` per
counterparty, and clear to the minimal set of `NetObligation` transfers, at most
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
`NettingDispute`, excluded from the clearing, never silently absorbed:

```python
netting.require_clean()   # raises if any contract was disputed
```

`book.net()` nets a single org's own book into its position against each
counterparty, the same calculation, one ledger.

## Resolving a dispute

Netting *pinpoints* a disagreement as a `NettingDispute`, and `reconcile` pinpoints
one between two records, but neither *resolves* it. **Arbitration** does: each party
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
the buyer and the seller signed, each on their own record, the two co-signing one
figure exactly as `reconcile` describes, is mutually corroborated and **stands**; a
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
way a record is, `verify()` recomputes the resolution hash from the bytes alone and
re-derives the whole decision from the recorded claims, so a flipped verdict is caught
even after re-sealing, and two arbiters reading the same records compute the same
co-signable hash:

```python
verdict = resolution.verify(app.contract_signer)
assert verdict.valid and verdict.hash_ok and verdict.decision_sound
```

Resolving a dispute also **closes the reputation loop**: the party whose claim did not
stand is debited, so a bad-faith revision weights the next negotiation. `book.arbitrate()`
resolves one org's own record against a counterparty's submitted claims, the dispute
counterpart of `book.reconcile_with()`.

## Porting reputation across orgs

Settlement, netting, and arbitration all close the reputation loop, but the standing
they earn lives inside one org's own ledger. A *new* counterparty has no way to trust it
without a hosted reputation bureau. `app.attest_reputation()` (or `book.attest()`) issues
a signed, offline-verifiable `ReputationAttestation` over a counterparty's earned
standing, derived only from an org's own signed records and arbitration resolutions, a
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
Because a Beta-Bernoulli posterior is conjugate, combining is *pooling the evidence*,
never a single self-asserted number: an issuer that vouches for itself is refused, an
issuer cannot stack its own pull (only its largest attestation counts), and a tampered or
forged attestation is pinpointed (`prior.refused`) and excluded:

```python
buyer.use_reputation_ledger()
prior = buyer.import_reputation([acme_attestation, globex_attestation])

# The prior exposes weight(member_id), so it drops into the existing negotiation path:
# a never-before-seen vendor is weighted by what its past counterparties attest, under
# the same bounded [floor, 1] rule a local reputation is, discounted, never zeroed.
result = buyer.negotiate("transcribe calls", buyer=..., seller=..., seller_id="vendor")
```

The imported prior weights an *unknown* counterparty; one the buyer has lived through
keeps its own earned `ReputationLedger` standing (passed as the `base`), so portability
only fills the gap where there is no local history, `prior.standing("vendor")` shows the
pooled evidence and `prior.weight("vendor")` the bounded weight it maps to.

## Keeping a ported reputation current

An attestation is a *point-in-time* claim, but standing changes, so the portable prior
is **time-aware and revocable**, reflecting current standing rather than a frozen
snapshot, without a hosted revocation service.

**Freshness.** An issuer declares a validity window when it attests
(`horizon_days=`), bound into the attestation's signed hash. Against an as-of clock, a
combination *excludes* a stale attestation (past its window) and *decays* an older one
within its window by an importer-set half-life, its evidence mass halved each
`half_life_days`, its attested ratio preserved, so an old attestation eases out of the
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
which excludes the withdrawn claim, pinpointed in `prior.revoked`, never silently
honored. Only the issuer can withdraw its *own* attestation: a forged revocation, or one
naming another org's attestation, cannot cancel a claim.

```python
revocation = acme.revoke_attestation(att, reason="vendor regressed")  # signed, audited
prior = buyer.import_reputation([att], revocations=[revocation])
assert prior.standing("vendor") is None        # the withdrawn claim is excluded
assert prior.revoked[0].issuer == "acme"       # pinpointed, not silently dropped
```

Freshness and revocation read only the existing signed artifacts, assert nothing they
cannot recompute, and fold into the *same* bounded `[floor, 1]` weighting, never a
central revocation service or a hosted bulletin board.

## Gossiping reputation across the fabric

Attestations are portable, current, and revocable, but an importer still has to be
*handed* the right bundle: it has no way to **discover** who has attested a
counterparty, or to learn that an issuer has since revoked one, without a hosted
registry. A bounded, **pull-based** exchange of those signed artifacts over the A2A
fabric closes that gap, the discovery analogue for reputation.

**Pull, never push.** An org exposes its book as a queryable peer with
`app.serve_attestations()`, returning an `A2AServer` whose Agent Card advertises an
`attestation-exchange` skill. Answering a query for a subject, the peer returns a
`ReputationBundle` of its *own* signed artifacts, the current attestation it can
issue from its `SettlementBook` records, plus any revocations it has signed, and
nothing else:

```python
acme.use_settlement_book()
# ... acme settles work delivered by "vendor" ...
acme_peer = acme.serve_attestations()      # a queryable peer, pull-only
```

**Bounded, governed gather.** An importer with no local history pulls a *bounded* set
of peers with `app.gather_reputation()` (a remote `AttestationExchange` under the
hood). Each peer is governed through an `AgentDirectory` allow-list (every resolution
audited), every fetched artifact is *independently verified from the bytes*, the
artifacts are deduplicated by content hash, and the result folds into the **same**
`combine_attestations`, so gossip changes only *where the evidence comes from*:

```python
buyer.use_reputation_ledger()
result = await buyer.agather_reputation(
    "vendor",
    peers={"acme": acme_peer, "globex": globex_peer},
    directory=buyer.agent_directory(allow=["acme", "globex"]),  # governed
    max_peers=8,                                                # bounded fan-out
)
result.weight("vendor")          # drops into the negotiation path, like a handed prior
result.standing("vendor").issuers  # which peers corroborated the standing
```

A directory-denied peer is **skipped and pinpointed** (`result.visit_for(peer)`), a
forged or tampered artifact a peer serves is **refused** (nothing is trusted that does
not verify), a revocation a peer gossips **excludes** the withdrawn claim
(`result.reputation.revoked`), and every peer visited and artifact fetched lands on the
hash-chained audit log (`reputation_peer` / `reputation_fetch`). The whole exchange runs
byte-for-byte the same against deterministic in-process peers as over the live fabric.

## Weighing evidence by trust in the issuer

Pooling every counted issuer's evidence with **equal pull** lets a clutch of unknown
peers out-evidence a few you've lived through, and an adversary can spin up *Sybil*
issuers that all vouch the same way. The trust kernel scales each issuer's contributed
evidence *mass* by your **own trust in that issuer**, a bounded, transitive
web-of-trust rooted in your local `ReputationLedger`, so corroboration from a trusted
peer counts for more than volume from a stranger. It is **opt-in**: pass a `trust_config`
(or an explicit `trust` source) to `combine_attestations` / `app.import_reputation` /
`app.gather_reputation`; with neither, pooling is unchanged.

```python
from vincio import TrustConfig

prior = buyer.import_reputation(
    [trusted_issuer_att, unknown_issuer_att],   # equal evidence each
    trust_config=TrustConfig(),                 # turn on issuer-trust weighting
)
standing = prior.standing("vendor")
standing.issuer_trust   # {"acme": 0.93, "stranger": 0.10}, pinpointed multipliers
```

An issuer you know **first-hand** counts at its earned weight (hop 0); trust then
**composes at most a bounded hop** outward, a trusted issuer lends weight to the issuers
*it* attests, attenuated by a per-hop decay, under a hard depth bound, so a long
unverifiable chain cannot manufacture standing. An **unknown** issuer is *floored*
(`trust_floor`), never zeroed or singled out. Because trust is lent only *outward from a
trusted root*, a cluster of mutually-vouching unknown issuers is never reached and every
member stays at the floor, **pull follows earned trust, not issuer count**, so a Sybil
clutch cannot outvote a few corroborating trusted peers. Build and inspect the model
directly with `build_trust_model(...)` (each issuer's `IssuerTrust` records its
`trust` / `depth` / `vouched_by`), and read the applied multipliers back from
`AttestationVerdict.trust` and `SubjectStanding.issuer_trust`, bounded `[floor, 1]`,
reversible, and pinpointed at every step.

## Gating admission by earned standing

A portable, current, discoverable, trust-weighted standing still only ever *softened*
a negotiation; nothing **acted** on a too-thin or too-low standing to bound how much a
new counterparty was trusted with up front. An `AdmissionPolicy` (`app.admit`) maps the
standing the fabric already earns, an imported `PortableReputation` or a local
`ReputationLedger`, to a bounded, offline-verifiable `AdmissionDecision`: a maximum
contract value (the exposure ceiling), a required escrow/collateral fraction, and an
SLA-strictness factor.

```python
from vincio import AdmissionConfig

policy = AdmissionConfig(parity_exposure_usd=1000.0)   # the ceiling at full trust
decision = buyer.admit("vendor", config=policy)
decision.max_contract_value_usd   # the exposure ceiling this standing earns
decision.escrow_fraction          # collateral asked at this trust level
decision.verify().valid           # offline-verifiable, terms re-derive from the bytes
```

Exposure is the product of two bounded signals, *how good* the standing is (its
posterior-mean reputation) and *how much corroborated, settled history* stands behind it,
ramped from a floor to parity, lifted off a `floor_fraction`. So a thin or low-trust
standing is admitted on **conservative terms rather than refused** (discounted exposure,
never a hard gate, never singled out), and as the counterparty accrues settled deliveries
its ceiling **ramps** deterministically toward parity, a regression walking it back,
exposure unlocked the way a credit line builds. Local first-hand evidence wins over what
others attest, exactly as the negotiation `weight` resolves it, so a regression you lived
through walks the ceiling back even when other orgs still attest a high standing.

The decision folds into the **existing** path without a new code path through it:

```python
from vincio.negotiation import buyer_position, seller_position

bounded = decision.bound_position(           # clamp the buyer to the exposure ceiling
    buyer_position(max_price_usd=1e6, ideal_price_usd=0.01, max_sla_seconds=5.0)
)
result = buyer.negotiate("transcribe", buyer=bounded, seller=seller_position(...),
                         seller_id="vendor")
# the bargain can only converge within the admitted exposure
```

`apply_to_terms(terms)` caps a contract's price and stamps the escrow posture into the
terms' (unhashed) metadata, so a contract minted from the capped terms stays
offline-verifiable while carrying the collateral the deal must post. Every decision binds
the standing it read and the terms it set onto a content hash that `verify` recomputes
from the bytes, a tampered ceiling is caught even after re-sealing, and `app.admit`
records it on the hash-chained audit log.

## Backing the collateral with an escrow

Admission sets a required escrow fraction on a thin or low-trust counterparty's contract,
but on its own that fraction is only a *number stamped on the terms*; nothing **holds** it,
releases it on a clean delivery, or forfeits a slice on a breach. An `Escrow`
(`app.post_escrow` / `post_escrow`) makes the posted collateral a verifiable, offline
escrow **bound to the contract**, the escrow analogue of a `SettlementRecord`.

```python
decision = buyer.admit("vendor")
escrow = buyer.post_escrow(contract, decision=decision)   # hold the required collateral
escrow.amount_usd          # = escrow_fraction × the contract price
escrow.verify().valid      # offline-verifiable, the amount re-derives from the posture
```

The collateral source is, in order: an explicit `amount` (a flat stake), an explicit
`fraction` of the contract price, an `AdmissionDecision`'s `escrow_fraction` (`decision=`),
or, with none of those, the admission posture `apply_to_terms` already stamped onto the
contract's terms. The poster (the counterparty backing its delivery) defaults to the
seller and the beneficiary to the buyer.

Settling the contract resolves the escrow deterministically, releasing the whole stake on
a fulfilled delivery and forfeiting a **bounded, pinpointed slice proportional to the
shortfall** on a breach (never the whole stake, never punitive), the remainder released:

```python
record = buyer.settle(contract, cost_usd=140.0, escrow=escrow)   # a 40% cost overrun
escrow.state            # "forfeited"
escrow.forfeited_usd    # a slice proportional to the shortfall the settlement measured
escrow.released_usd     # the remainder, back to the poster
escrow.breaches         # ["price"], pinpointed to the term the delivery missed
```

The outcome is driven by the **same** `SettlementRecord` verdict the books already close
on, so the collateral closes the same loop the settlement does, never judging the delivery
a second time. Pass `escrow=` to `app.settle` to resolve it in the same call, or
`app.settle_escrow(escrow, record)` to resolve it against a record you already have.
`EscrowConfig(max_forfeit_fraction=0.8)` caps a single breach's forfeiture below the whole
stake when you want a residual always released. Every post, release, and forfeiture binds
the contract, the amount, and the verdict onto a content hash that `Escrow.verify`
recomputes, a tampered amount or forfeiture is caught even after re-sealing, and lands on
the hash-chained audit log, so an escrow's whole lifecycle is reconstructable offline.

## Pooling collateral across many contracts

An escrow backs *one* contract. A counterparty running many concurrent contracts would have
to lock **separate** collateral per deal, even though its breaches and clean deliveries
across those contracts net out, so capital is stranded contract-by-contract the way bilateral
settlements were stranded book-by-book before netting folded them. A `CollateralPool`
(`app.post_collateral_pool` / `post_collateral_pool`) is a **bounded margin account** a
counterparty posts once that backs many contracts, the collateral analogue of a
`NettingSet`.

```python
# One stake backs three concurrent contracts, allocated per-contract by required collateral.
pool = buyer.post_collateral_pool([c1, c2, c3], decisions=decision)
pool.posted_usd                  # the single stake (defaults to the total required)
[c.allocated_usd for c in pool.contracts]   # shares proportional to each required collateral
pool.verify().valid              # offline-verifiable, allocations re-derive, balance reconciles
```

Each contract's required collateral re-derives from its admission posture exactly as a
standalone escrow's does, a matching `AdmissionDecision` in `decisions=` (a dict keyed by
contract id, or one decision applied to all), a uniform `fraction=`, or the posture stamped
onto each contract's terms. The poster defaults to the seller every contract shares.

Settling a contract draws against the **shared stake** instead of a per-deal escrow:

```python
buyer.settle(c1, cost_usd=60.0, pool=pool)    # clean, frees its requirement back to available
buyer.settle(c2, cost_usd=300.0, pool=pool)   # a breach, draws a bounded slice from the stake
pool.contract(c2.id).forfeited_usd            # proportional to the shortfall, never the whole stake
pool.available_usd                            # capital the clean delivery freed for the next deal
```

A clean delivery frees its committed capital back to the available balance (reused, not
stranded), and a breach draws a bounded, pinpointed slice, the **same** proportional forfeiture
the `SettlementRecord` verdict measures, from the shared stake, releasing the rest. A pool
committed below the collateral its still-open contracts require surfaces a bounded, pinpointed
**top-up** obligation rather than silently over-committing:

```python
pool.back(c4, decision=decision)   # backing a new deal can over-commit the pool
pool.topup_usd                     # the bounded amount the pool is short by
pool.top_up(pool.topup_usd)        # add capital; the obligation clears
```

Every post, draw, release, and top-up binds the pool, the contracts, and the balances onto a
content hash that `CollateralPool.verify` recomputes, the allocations re-derive and the
balance reconciles (posted minus drawn), so a tampered allocation or balance is caught even
after re-sealing, and lands on the hash-chained audit log, so a pool's whole lifecycle is
reconstructable offline.

## Guarding against rehypothecation

A pool only ever re-allocates capital *within itself*. When a counterparty pledges the **same**
stake across more than one pool, or re-pledges collateral a beneficiary already has a claim
on, nothing bounds the **re-use**: the same capital is double-counted, over-stating what
actually backs each deal, the collateral analogue of a settlement record double-counted before
netting deduplicated it. A `CollateralLedger` (`app.guard_collateral` / `guard_collateral`) is
the **rehypothecation guard**: it folds a counterparty's pools into one view and reconciles
what they collectively pledge against the capital it actually holds.

```python
# vendor backs two pools, but contract A is re-pledged across both, the same stake, twice.
pool_one = buyer.post_collateral_pool([c_a, c_b], decisions=decision)
pool_two = buyer.post_collateral_pool([c_a], decisions=decision)
ledger = buyer.guard_collateral([pool_one, pool_two])

ledger.pledged_usd      # what the pools collectively pledge
ledger.held_usd         # the capital the poster actually holds (defaults to the distinct pledge)
ledger.reuse_usd        # what the pledges exceed the holdings by, the over-commitment
ledger.over_committed   # the same capital pledged twice
ledger.breaches[0]      # the ReuseBreach: contract A, the pools, the double-pledged excess
```

Pass `held=` when you know the counterparty's true custody balance (the ground-truth figure the
guard bounds the pledges by); it defaults to the gross pledge minus the provably double-pledged
capital, so a re-pledged contract surfaces as an over-commitment while genuinely separately-
funded pools do not. When a stake backs deals for more than one beneficiary and the held capital
is scarce, each beneficiary's claim is bounded to its deterministic **pari-passu** share, so a
forfeiture cannot pay one beneficiary out of capital another has first claim on:

```python
ledger = buyer.guard_collateral([multi_beneficiary_pool], held=20.0)
claim = ledger.claim("globex")
claim.claim_usd        # the capital pledged to globex across the poster's pools
claim.secured_usd      # its bounded, proportional share of the held capital
claim.unsecured_usd    # what the over-commitment leaves it exposed to
ledger.require_within_bounds()   # raises if the poster has over-committed its capital
```

The ledger reads only the signed, content-bound pools and asserts nothing it cannot recompute: a
tampered pool (its content hash no longer recomputes) is **refused** at fold time, and
`CollateralLedger.verify` re-derives the re-use bound and the beneficiary apportionment from the
bytes alone (a tampered figure is caught even after re-sealing). `app.guard_collateral` signs the
ledger as the org and records the guard on the hash-chained audit log, so two folders reading the
same pools compute the same co-signable hash.

## Proving the reserves

The `held` figure the guard bounds pledges against is the one input it **trusts**: it is
*asserted*, not proven, so a counterparty over-stating its real reserves still passes, the way a
self-asserted reputation score passed before attestation made standing verifiable. A
`CustodyAttestation` (`app.attest_custody` / `attest_custody`) is the **proof-of-reserves**: a
custodian (or the poster's own signed reserve record) attests the capital actually held, itemized
into reserve accounts whose total re-derives on every verify, and `guard_collateral(custody=)`
reads it as the held figure instead of the asserted default.

```python
# a custodian attests the vendor's reserves, itemized per account; the total re-derives on verify
proof = custodian.attest_custody("vendor", {"omnibus": 40.0, "escrow": 10.0})  # 50 proven
proof.verify(custodian.contract_signer).valid   # signed, content-bound, offline-verifiable

ledger = buyer.guard_collateral([pool], custody=proof)   # held = proven reserves
ledger.reserves_proven   # True, the held figure is evidence-backed, not asserted
ledger.under_reserved    # True when the proven reserves fall below what the pools pledge
ledger.reserve_breach    # the UnderReservedBreach: custodian, attestation hash, shortfall
ledger.require_reserved()  # raises if the proven reserves cannot cover the pledges
```

`held=` and `custody=` are mutually exclusive, the held figure has one source. The attestation
reads only signed, content-bound artifacts: a tampered reserve figure, a forged custodian (with
`verify_with`), or an attestation that vouches for a **different poster** than the pools' is
**refused**, and the under-reserved breach re-derives from the bytes alone (a fabricated or hidden
breach is caught even after re-sealing). `app.attest_custody` signs it as the custodian and
records the issuance on the hash-chained audit log (action `custody_attestation`). An asserted
`held=` figure can over-commit but never *under-reserves*, because nothing proves it, only a
custody attestation can. This proves the reserves *exist*, not that they exceed every liability
the counterparty owes elsewhere, that second half is the proof-of-solvency below.

## Proving solvency

Proven reserves are only one side of the ledger. A counterparty solvent against *one* buyer's
pledges may be deeply **under-water** once *every* obligation it owes is counted, and could prove
the same reserves against many buyers while quietly insolvent across all of them. A
`LiabilityAttestation` (`app.attest_liabilities` / `attest_liabilities`) makes the liability side
evidence-backed too, a counterparty (or its auditor) attests the total obligations it owes,
itemized into creditors whose total re-derives on every verify, and `prove_solvency`
(`app.prove_solvency`) folds it against the proven reserves into a `SolvencyProof`: a bounded
solvency margin (`reserves − liabilities`) the guard reads as the held figure.

```python
# the vendor proves $80 of reserves but owes $50 to other creditors
reserves = custodian.attest_custody("vendor", {"omnibus": 80.0})       # 80 held
owed = auditor.attest_liabilities("vendor", {"globex": 35.0, "initech": 15.0})  # 50 owed
proof = auditor.prove_solvency(reserves, owed)   # margin = 30 free
proof.solvent              # True, reserves cover the liabilities
proof.margin_usd           # 30.0 (reserves − liabilities)
proof.solvency_adjusted_held  # 30.0, the unencumbered capital, max(0, margin)

ledger = buyer.guard_collateral([pool], solvency=proof)  # held = solvency-adjusted margin
ledger.solvency_adjusted   # True, pledges bounded against capital not already owed elsewhere
ledger.insolvent           # True when the proven liabilities exceed the proven reserves
ledger.under_reserved      # True when the unencumbered capital falls below the pledges
proof.require_solvent()    # raises if the liabilities exceed the reserves
```

`held=`, `custody=`, and `solvency=` are mutually exclusive, the held figure has one source.
`prove_solvency` reads only signed, content-bound artifacts: a tampered liability figure, a forged
issuer (with `verifier`/`verify_with`), or a custody / liability pair for **different posters** is
**refused**, and the solvency margin and the `InsolvencyBreach` re-derive from the bytes alone (a
flipped solvency verdict re-sealed to match is caught). `app.attest_liabilities` signs the
attestation as the attestor (action `liability_attestation`) and `app.prove_solvency` signs and
records the proof (action `solvency_proof`, decision `solvent` / `insolvent`) on the hash-chained
audit log. The proof bounds the guard's held figure by the counterparty's whole obligation set,
not one buyer's view.

## Proving the liabilities are complete

A solvency proof folds the attestor's liability *total*, but that total is still one number: a
counterparty could **under-state** what it owes by quietly omitting a creditor and still attest a
sound, re-deriving total over the creditors it *did* list. The second half of a proof-of-liabilities
makes the total provably **complete**. A `LiabilityAttestation` now commits its line items into a
Merkle root (bound in its signed hash, the total *and* the root re-derive on every verify), so each
creditor gets an offline-verifiable `InclusionProof` (`app.inclusion_proof` / `attestation.inclusion_proof`)
that its claim is a leaf of that root, a poster cannot drop a creditor without the omitted party
detecting it. `check_completeness` (`app.check_completeness`) folds creditors' own proven claims
against the attestation, pinpointing every omitted or under-stated claim and raising the attested
figure to a **completed** total `prove_solvency` reads.

```python
owed = auditor.attest_liabilities("vendor", {"acme": 60.0})   # quietly omits globex

acme_proof = auditor.inclusion_proof(owed, "acme")            # acme proves its claim is counted
acme_proof.verify(owed).valid                                  # True, a leaf of the signed root

# globex folds its own $40 claim, it is not in the attestation, so the check is incomplete
check = globex.check_completeness(owed, {"globex": 40.0})
check.complete                 # False
check.omitted_creditors        # ["globex"]
check.completed_usd            # 100.0, attested 60 + the proven 40 omission

reserves = custodian.attest_custody("vendor", {"omnibus": 80.0})  # 80 held
proof = auditor.prove_solvency(reserves, owed, completeness=check)  # margin uses 100, not 60
proof.insolvent                # True, the hidden $40 tips 80 − 100 into a shortfall
proof.understated_usd          # 40.0, how far the completed total exceeds the attestor's figure
check.require_complete()       # raises: globex is omitted
```

The inclusion and completeness proofs read only signed, content-bound artifacts: a tampered leaf or
a forged root fails to reconstruct the committed root, an under-stated completed total is caught from
the bytes alone (the completed total never sits below the folded claims, and the breaches re-derive),
and a completeness check for a **different** attestation or poster is **refused** at fold time.
`app.check_completeness` signs the check and records it (action `liability_completeness`, decision
`complete` / `incomplete`) on the hash-chained audit log. When `claims` is omitted, the book derives
them from its owner's own settled records against the attestation's poster, what the creditor
delivered against and is owed.

## Catching equivocation across creditors

Completeness catches an omission only when the *omitted* creditor folds its own claim. But a
counterparty issues its attestation **per relationship**, so it can **equivocate**: sign a *smaller*
liability root for one creditor and a different one for another, each creditor's `InclusionProof`
verifying against the root *it* was shown while the totals disagree. The creditors compare the signed
roots to catch it. `attestation.root_commitment()` produces a signed, privacy-preserving
`RootCommitment`, the root and `as_of` the attestor signed, **without** the line items, and
`check_root_consistency` (`app.check_root_consistency` / `book.check_root_consistency`) groups a set
of held attestations by their `(poster, attestor, as_of)` key and folds any two conflicting roots
into a non-repudiable `EquivocationProof`.

```python
as_of = datetime(2026, 1, 1, tzinfo=UTC)
to_acme = auditor.attest_liabilities("vendor", {"acme": 60.0}, as_of=as_of)    # root shown to acme
to_globex = auditor.attest_liabilities("vendor", {"globex": 40.0}, as_of=as_of)  # a different root

# The creditors compare the signed commitments, root + as-of only, no line items.
to_acme.root_commitment().conflicts_with(to_globex.root_commitment())   # True

# An auditor verifying both folds the conflict into a non-repudiable proof.
report = auditor.check_root_consistency(
    [("acme", to_acme), ("globex", to_globex)], verify_with=auditor.contract_signer
)
report.consistent              # False
report.equivocating_posters    # ["vendor"]
report.equivocations[0].liabilities_gap_usd   # 20.0, the two signed totals disagree
report.require_consistent()    # raises: vendor signed two roots for one instant
```

The comparison and the proof read only signed, content-bound artifacts: each embedded attestation
re-derives its root from the bytes (a mislabeled root cannot survive), and with the attestor's
verifier a **forged conflicting root is refused** and excluded from a scan, so it cannot manufacture
a false accusation against an honest poster. `app.check_root_consistency` records each equivocation
(action `liability_equivocation`, decision `equivocation`) on the hash-chained audit log and dings
the equivocating poster on the bound reputation ledger. Non-equivocation is defined for one `as_of`:
two roots a poster signed *as of the same instant* are a contradiction, while two roots for
*different* instants are distinct snapshots a later one legitimately supersedes.

## Walking a liability history over time

Non-equivocation is scoped to one `as_of`. But a counterparty can still issue a *later* snapshot that
quietly **drops** a past obligation, a debt committed at `T` simply absent from the root it signs at
`T'`, each snapshot internally sound. Equivocation is conflict *across creditors*; this is
consistency *across time*. Link each snapshot to its predecessor with `prior=` (so the attestations
form a hash-linked sequence each `as_of` strictly succeeding the last) and `check_history_consistency`
(`app.check_history_consistency` / `book.check_history_consistency`) walks them, pinpointing any debt
that vanished without a signed, creditor-issued `Discharge` explaining it.

```python
t1 = datetime(2026, 1, 1, tzinfo=UTC)
t2 = datetime(2026, 2, 1, tzinfo=UTC)
s1 = auditor.attest_liabilities("vendor", {"acme": 100.0}, as_of=t1)
s2 = auditor.attest_liabilities("vendor", {"acme": 30.0}, as_of=t2, prior=s1)  # acme dropped $70

report = auditor.check_history_consistency([s1, s2])
report.consistent              # False, acme's $70 vanished between snapshots
report.proofs[0].breaches[0].unexplained_usd   # 70.0
report.require_consistent()    # raises: vendor dropped an obligation without a discharge

# A signed, creditor-issued discharge legitimizes the drop (acme was paid $70):
settled = acme.discharge_liability("vendor", 70.0, as_of=t2)
auditor.check_history_consistency([s1, s2], discharges=[settled]).consistent   # True
```

The discharge is the *creditor's* to issue, so a poster cannot forge its own, with the verifier a
forged or poster-signed release does not explain a drop, and a release dated outside the transition
window (or already consumed by another drop) does not apply. A back-dated link (a snapshot claiming to
follow a *later* one) is refused, a tampered or unsigned snapshot is excluded as inadmissible, and a
dropped `MonotonicityBreach` is caught by re-derivation. `app.check_history_consistency` records each
inconsistent history (action `liability_history`, decision `consistent` / `inconsistent`) on the
hash-chained audit log and dings the breaching poster on the reputation ledger; `require_monotone`
raises on any unexplained drop and `require_linked` additionally demands a contiguous hash-linked
chain.

## Resolving an insolvency

A solvency proof *flags* an insolvency when proven liabilities exceed proven reserves, but it says
nothing about **which** creditors the scarce capital pays, or in what order, so every creditor is left
to assume it is made whole. Rank the obligations into a signed `SenioritySchedule`
(`app.build_seniority_schedule` / `build_seniority_schedule`), the simplest spec is a list of
creditor-name lists where position is priority (rank `0` most senior), and `resolve_insolvency`
(`app.resolve_insolvency` / `book.resolve_insolvency`) distributes the proven reserves across the
obligations **by seniority then pari-passu within a tranche**, into an `InsolvencyResolution` that
pinpoints each creditor's bounded recovery and the shortfall it bears.

```python
reserves = vendor.attest_custody("vendor", {"omnibus": 60.0})       # $60 held
owed = auditor.attest_liabilities(
    "vendor", {"bank": 50.0, "acme": 30.0, "globex": 20.0}          # $100 owed -> $40 short
)
schedule = bank.build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]])

resolution = auditor.resolve_insolvency(reserves, owed, schedule)
resolution.status                       # "resolved" (some creditor bears a shortfall)
resolution.recovery_of("bank").recovery_usd     # 50.0, senior tranche paid in full first
resolution.recovery_of("acme").recovery_usd     # 6.0, junior tranche pari-passu (20%)
resolution.shortfall_bearers            # ["acme", "globex"], ordered by seniority
resolution.require_fully_recovered()    # raises: $40 unrecovered
```

With no schedule the whole set is one pari-passu tranche (exactly the rehypothecation guard's
apportionment); a counterparty whose reserves cover every obligation is `solvent` and makes each
creditor whole. `resolve_insolvency` reuses `prove_solvency`, so a tampered, forged, or wrong-poster
attestation (or a malformed/wrong-poster schedule) is refused; pass `completeness=` to distribute
against the *completed* liability set. `InsolvencyResolution.verify` re-derives the entire distribution
from the recorded claims, ranks, and reserves, an over-stated recovery or a re-ordered tranche is
caught from the bytes alone, and passing the `schedule` binds each creditor's rank to the one it
signed. `app.resolve_insolvency` records the resolution (action `insolvency_resolution`, decision
`solvent` / `resolved`) on the hash-chained audit log and dings a poster that could not make its
creditors whole on the reputation ledger.

## Setting off mutual obligations

The waterfall pays a creditor on its **gross** claim, but a creditor of an insolvent estate is often
*also* a debtor of it, still owing the other side across a web of contracts. Real insolvency law
resolves this first with **set-off** (close-out netting): mutual obligations collapse to a single net
claim *before* any distribution. State the obligations running both ways between a poster and one
creditor as a `SetOffStatement` (`app.build_set_off_statement` / `build_set_off_statement`, or
`set_off_from_records` to derive it straight from the existing `LiabilityAttestation` and settlement
records), have **both** parties co-sign it, and pass `set_off=` to `resolve_insolvency` to net each
creditor to its true exposure before the reserves are distributed.

```python
owed = auditor.attest_liabilities("vendor", {"bank": 50.0, "acme": 30.0})   # $80 gross
# acme owes the vendor $12 back: its $30 claim nets to $18. Both parties co-sign.
statement = build_set_off_statement("vendor", "acme", 30.0, 12.0)
statement.sign(vendor_signer, party="vendor").sign(acme_signer, party="acme")

resolution = auditor.resolve_insolvency(reserves, owed, set_off=[statement])
resolution.gross_liabilities_usd        # 80.0, before set-off
resolution.liabilities_usd              # 68.0, acme netted from 30 to 18
resolution.recovery_of("acme").set_off_usd      # 12.0 netted out of acme's gross claim
```

A creditor **in debit** (owing the estate at least as much as it is owed) nets to a zero claim and
recovers nothing, and the estate's distributable claims shrink to the true net exposure. The set-off
is applied after `completeness` (so it nets the *completed* gross), reconciled against that gross, an
over-stated set-off claiming a different gross than the attestation is refused, and the statements
must be mutually-signed: a one-sided close-out is refused at fold time. `InsolvencyResolution.verify`
re-derives every net claim from the recorded gross and the applied set-off (an inflated set-off is
caught even after re-sealing), and passing `set_off=` to `verify` binds the statements by hash.
`app.build_set_off_statement` records the statement (action `liability_set_off`, decision
`poster_owes` / `creditor_in_debit` / `eliminated`) on the hash-chained audit log.

## Composing the whole fabric as one engagement

Every section above is a *primitive*, signed, content-bound, offline-verifiable on its own.
The capstone composes them. `app.cross_org_engagement(...)` returns a `CrossOrgEngagement`: a
purely-compositional facade that threads the whole pipeline behind one governed, audited
call-path and seals it into a single, hash-linked `EngagementNarrative`.

```python
app = ContextApp(name="acme", provider=MockProvider(default_text="ok"))
eng = app.cross_org_engagement(buyer="acme", seller="vendor", scope="transcribe 1k calls")

# Each lifecycle method delegates to the SAME app.* primitive you could call directly,
# captures the artifact it produced, and records it as a stage in the narrative.
contract = eng.negotiate(
    buyer=buyer_position(max_price_usd=0.12, max_sla_seconds=5.0),
    seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.10),
)
eng.choreograph(saga, participants=parts, directory=directory)  # discover + deliver
eng.settle_saga(contracts={contract.id: contract})              # meter + settle
eng.net()                                                       # multilateral clearing
reserves = eng.attest_custody("vendor", {"omnibus": 120.0})
owed = eng.attest_liabilities("vendor", {"acme": 40.0, "globex": 30.0})
eng.prove_solvency(reserves, owed)

narrative = eng.seal()                       # one content-bound, signed narrative
narrative.verify(app.contract_signer).valid  # True, recomputes the whole chain offline
```

The narrative is the proof the fabric is a *system*, not a pile of primitives. Each
`EngagementStage` binds the verb, the captured artifact's own content hash, and a digest of
its bytes into a link that chains to the previous one. `narrative.verify()` recomputes the
whole chain from the bytes alone, a re-ordered stage, an edited digest, a broken link, or a
forged signature is caught, and `eng.verify(app.contract_signer)` additionally re-digests
the live artifacts against the bound digests, so a tamper to any *underlying* artifact is
caught too. The facade adds no new economic logic: every primitive stays unchanged and usable
on its own (the methods above are exactly the `app.*` calls documented in the preceding
sections); the engagement only captures and **narrates** them. Sealing lands the engagement on
the hash-chained audit log under `cross_org_engagement`, one continuous signed narrative from
the first offer to the final distribution.

With this capstone, the cross-org settlement & credit surface is **feature-complete and
frozen** under the [stability policy](../reference/stability.md): no further cross-org
*primitive* is scheduled, and subsequent cross-org work is bug-fix and standards-tracking only.

## What it is not

This is a library capability inside your process, not a payment rail or a hosted
marketplace. There is no money movement, no escrow service, no managed clearing
house, no arbitration service or court of record, no reputation bureau, no underwriting
service, a settlement is a typed, signed record you hold and verify yourself, the book
is a hash-chained ledger each organization keeps on its own, netting is a clearing
*calculation* over those books, not a clearing *house*, arbitration is a deterministic
adjudication over the parties' own signed records, not a third party that rules by fiat, a
reputation attestation is one org's signed, verifiable claim that combines into an
evidence-weighted prior, not a central score, the attestation exchange is a bounded pull of
those signed artifacts from peers you govern, not a hosted reputation registry or a
push-based gossip bus, the trust kernel is a bounded, transitive weighting computed
in-process from your own ledger, not a central trust authority or a Sybil-detection service,
an admission decision is a mechanical, reconstructable exposure number computed from the
standing you already hold, not a hosted underwriting service, an escrow is a verifiable
record of collateral posted against a contract and settled deterministically against the
delivery verdict, not a custodian's ledger entry, an escrow service, or money in motion, and a
collateral pool is a verifiable margin account that allocates one posted stake across many
contracts and draws it deterministically, not a hosted clearing house, a margin custodian, or
an omnibus account, a collateral ledger is a verifiable re-use bound that folds a
counterparty's pools and reconciles what they pledge against what it holds, not a hosted
custodian or a rehypothecation registry, a custody attestation is one party's signed,
verifiable proof-of-reserves that the guard reads as the held figure, not a hosted proof-of-reserves
auditor or a trusted third party, and a solvency proof is a verifiable fold of a counterparty's
proven reserves against its proven liabilities that the guard reads as a solvency-adjusted held
figure, not a hosted solvency auditor, and an inclusion proof and a completeness check are one
party's signed, verifiable proof that a creditor's claim is counted in the attested liabilities and
that the attested total omits nothing a creditor can prove, not a hosted attestation registry or a
transparency log, and an equivocation proof is a non-repudiable fold of two conflicting liability
roots one counterparty signed for the same instant, surfaced by creditors comparing the signed roots
themselves, not a hosted transparency log or a trusted third party, and a history-consistency proof is
a non-repudiable walk of a counterparty's hash-linked liability snapshots that pinpoints a debt
dropped between them without a signed, creditor-issued discharge, not a hosted transparency log or a
trusted third party, and an insolvency resolution is a verifiable distribution of a counterparty's
proven reserves across the creditors it owes by seniority then pari-passu within a tranche, not a
hosted receiver, a bankruptcy court, or a trusted third party, and a set-off statement is a
mutually-signed, verifiable close-out of the obligations running both ways between a poster and one
creditor that the waterfall nets before distributing, not a hosted clearing house or a trusted third
party, and a cross-org engagement is a verifiable, hash-linked narrative that composes those
primitives and proves they chain end-to-end from the bytes alone, not a hosted orchestration service
or a managed control plane. Vincio gives you a verifiable
reconciliation of what was owed and delivered, a verifiable netting of it across a fleet, a
verifiable resolution when two books disagree, a portable, verifiable attestation of earned
standing, a verifiable way to discover it across the fabric, a verifiable way to weigh it by
your own earned trust, a verifiable way to bound a counterparty's exposure to what its
standing justifies, a verifiable way to back that exposure with posted collateral, a
verifiable way to pool that collateral across many concurrent deals, a verifiable way to
bound its re-use across them, a verifiable proof that the capital backing it actually
exists, a verifiable proof that it exceeds everything the counterparty owes, a verifiable
proof that the liabilities counted against it are complete, a verifiable proof that the
counterparty signed one liability total per instant, not different ones to different creditors, a
verifiable proof that its liabilities are monotone over time, not a debt quietly dropped between
snapshots, a verifiable resolution of an insolvency into who-gets-what by seniority, not a debt
left to assume it is made whole, and a verifiable close-out of mutual obligations so a creditor
recovers only its net exposure, not its gross claim while it still owes the estate the other side;
how an obligation is paid is yours.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Example: 12_cross_org_economy.py](../../examples/12_cross_org_economy.py)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#governance)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
