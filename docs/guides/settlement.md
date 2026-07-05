# Guide: agent-to-agent settlement & metering

Vincio lets agents negotiate a `Contract` and run durable, compensating sagas
across organizations. This guide covers the last rung: **closing the books** on
that work — a metered, signed, offline-verifiable ledger of what was owed and
delivered, and the credit machinery around it (netting, arbitration, portable
reputation, admission, collateral, proof-of-reserves/solvency, and insolvency
resolution). A cross-org engagement reconciles the way a run closes its cost
report.

It is never a payment rail. There is no money movement, no escrow *service*, no
clearing *house*, no court, no reputation bureau: every primitive here is a
typed, signed artifact **you** hold and verify yourself, and "how an obligation
is paid" is out of scope. It is additive (`vincio.settlement`), changes nothing
about how a contract is negotiated or a saga runs, and works fully offline
against deterministic figures.

## Shared invariants

Every primitive below is one of a family, so the same guarantees hold throughout
and are **stated once here** rather than repeated per section:

- **Signed & content-bound.** Each artifact binds its economic facts into a
  content hash and is signed as your side of the deal (`app.contract_signer`, or
  an `Ed25519Signer` for third-party verifiability).
- **`verify()` recomputes from the bytes.** It re-derives the hash *and* the whole
  decision from the stored fields, so a tampered figure or a forged signature is
  caught even after re-sealing — no live counterparty needed. A verdict is a typed
  object (e.g. `.valid`, `.intact`, `.positions_balanced`).
- **Reads only what it can recompute.** A source artifact whose own hash no longer
  recomputes is *refused* at fold time (netting, guards, proofs), never silently
  absorbed; a disputed one is *pinpointed*, not dropped.
- **Audited.** Every `app.*` verb records its verdict on the hash-chained audit
  log under a stable action name (called out per section).
- **Reputation-coupled & bounded.** A breach, dissent, or breach-of-good-faith
  dings the bound `ReputationLedger`; reputation only ever weights within
  `[floor, 1]` — discounted, never zeroed, never singled out.
- **One source of truth per quantity.** Where several inputs could set a figure
  (e.g. the collateral guard's `held`), they are **mutually exclusive** and the
  rule is stated where it applies.

## The pipeline at a glance

```
deliver ─▶ meter ─▶ settle ─▶ reconcile ─▶ book ─▶ net a fleet
                       │                              │
                       └── reputation loop ◀──────────┘
   port standing:  attest ─▶ import/gather ─▶ trust-weight
   trust a new party:  admit ─▶ escrow / pool ─▶ rehypothecation guard
   prove the capital:  reserves ─▶ solvency ─▶ completeness ─▶ (non-)equivocation ─▶ history
   when it fails:  resolve insolvency (seniority ▸ set-off ▸ pari-passu)
   compose it all:  cross_org_engagement ─▶ one signed EngagementNarrative
```

| Stage | Verb(s) | Produces |
|---|---|---|
| Meter delivery | `app.meter` | `MeterReading` |
| Settle a contract | `app.settle` · `settle_contract` · `app.settle_saga` | `SettlementRecord` |
| Reconcile two views | `reconcile` | `Reconciliation` |
| Keep a ledger | `app.use_settlement_book` · `app.settlement_report` | `SettlementBook` |
| Net a fleet | `app.clear_settlements` · `net_settlements` · `net_books` | `NettingSet` |
| Arbitrate a dispute | `app.arbitrate` · `book.arbitrate` | `Resolution` |
| Attest & port standing | `app.attest_reputation` · `app.import_reputation` · `combine_attestations` | `PortableReputation` |
| Keep it current | `app.revoke_attestation` · `AttestationConfig(half_life_days=)` | `AttestationRevocation` |
| Discover it | `app.serve_attestations` · `app.gather_reputation` | `ReputationBundle` |
| Weight by trust | `build_trust_model` · `TrustConfig` | `TrustModel` |
| Bound exposure | `app.admit` · `AdmissionPolicy` | `AdmissionDecision` |
| Back a deal | `app.post_escrow` · `app.post_collateral_pool` | `Escrow` / `CollateralPool` |
| Bound re-use | `app.guard_collateral` | `CollateralLedger` |
| Prove reserves | `app.attest_custody` | `CustodyAttestation` |
| Prove solvency | `app.attest_liabilities` · `app.prove_solvency` | `SolvencyProof` |
| Prove completeness | `app.inclusion_proof` · `app.check_completeness` | `CompletenessProof` |
| Catch equivocation | `app.check_root_consistency` | `EquivocationProof` |
| Check history | `app.check_history_consistency` · `discharge_liability` | `HistoryConsistencyProof` |
| Resolve insolvency | `app.resolve_insolvency` · `build_seniority_schedule` · `build_set_off_statement` | `InsolvencyResolution` |
| Compose all | `app.cross_org_engagement` | `EngagementNarrative` |

## Metering & settlement

### Metering delivery

A `Meter` accrues the **usage** of work delivered under one contract as
`UsageEvent`s. The reading is a deterministic, total-preserving roll-up — cost
and latency **summed**, quality the **minimum** observed (the figure to hold
against a floor) — exactly the sum of the events, no double-count, no drop.

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

Metering is pure accumulation; *enforcing* a cap stays where it already lives —
the contract's `to_budget()` and `check()`.

### A settlement record

`app.settle(...)` reconciles delivery against the contract's agreed
price / SLA / quality into a `SettlementRecord` (pass a `reading=` or explicit
`cost_usd=` / `quality=`). It records the verdict under audit action `settle`.

```python
record = app.settle(deal.contract, reading=reading)
record.status           # "settled" | "breached"
record.amount_owed_usd  # the agreed price for the scope
record.balance_usd      # price − delivered cost (+credit / −overrun)
record.breaches         # the breaching dimensions, if any
record.verify(app.contract_signer)   # offline-verifiable from the bytes
```

A breach is **not an error** — it settles to `status="breached"` with the
breaching dimensions on `.breaches`. Settling is closing the books, not raising
an alarm.

### Reconciling across the boundary

Both sides compute the *same* deterministic reconciliation hash, so their records
co-sign one hash when the books agree. `reconcile` ties two independently-produced
records out and pinpoints any discrepancy. Move a record with
`record.to_wire()` / `SettlementRecord.from_wire(...)`; reconciliation itself is
purely offline.

```python
from vincio import reconcile

ours   = app.settle(deal.contract, reading=reading)
theirs = counterparty_record            # received over the wire

verdict = reconcile(ours, theirs)
verdict.agrees                          # both hashes match
verdict.discrepancies                   # e.g. ["delivered_cost_usd: 0.15 != 0.18"]
```

### The settlement book

A `SettlementBook` is a durable, **hash-chained** ledger of an org's settlements —
the settlement analogue of the saga journal. `verify()` recomputes the whole book;
`report()` rolls it up per counterparty.

```python
app.use_settlement_book()
app.settle(deal.contract, reading=reading)
app.settle(other_contract, cost_usd=0.08, quality=0.7)   # a breach

assert app.settlement_book.verify().intact
app.settlement_report().print_summary()
#   vendor: owed=$0.15 delivered=$0.15 balance=$+0.00 (1 settled / 1 breached)
```

`app.settle_saga(result, contracts={c.id: c for c in (...)})` closes every
contract a cross-org saga ran under straight from its journal — metering each
contracted step, reconciling per-step delivery, appending one signed record per
contract.

### The reputation loop

With a ledger attached, closing the books closes the loop: a fulfilling
settlement credits the seller, an overrun or shortfall debits it, so reliability
earned in *delivery* weights the next negotiation.

```python
app.use_reputation_ledger()
app.use_settlement_book()
app.settle(deal.contract, cost_usd=0.20, quality=0.6)    # a shortfall
app.reputation_report(deal.contract.seller).rows[0].reputation   # debited
```

## Netting & multilateral clearing

Once an org is both buyer and seller across a web of contracts, **netting** folds
the bilateral balances into a minimal set of net obligations. `net_settlements`
(over loose records), `net_books` (over whole books), and `app.clear_settlements`
read only existing signed records and produce a content-bound `NettingSet`.

```python
from vincio import settle_contract

# A cycle: acme owes vendor, vendor owes data, data owes acme.
fleet = [
    settle_contract(acme_vendor, cost_usd=0.08),
    settle_contract(vendor_data, cost_usd=0.05),
    settle_contract(data_acme,   cost_usd=0.03),
]
netting = app.clear_settlements(records=fleet)   # signs as the clearer, audits

print(netting.gross_edges, "→", netting.cleared_transfers)   # 3 → 2
for o in netting.obligations:
    print(o.debtor, "→", o.creditor, o.amount_usd)
```

Each settled contract is a directed obligation — the buyer owes the seller the
agreed price for the scope (the payable cap; a breach is surfaced by the
settlement's own status and the reputation loop, it does not change what is
owed). Payables aggregate per pair into one `BilateralNet` per counterparty and
clear to the minimal `NetObligation` set — **at most `N − 1` transfers for `N`
parties**, net-debtors paying net-creditors, deterministically.

`verify()` recomputes the netting hash, confirms net positions balance to zero,
and confirms the transfers reproduce every org's position:

```python
verdict = netting.verify(app.contract_signer)
assert verdict.valid and verdict.positions_balanced and verdict.conserves
```

Two books that disagree on the same contract are pinpointed as a `NettingDispute`
and *excluded* from the clearing — `netting.require_clean()` raises if any
contract was disputed. `book.net()` nets a single org's own book into its
position against each counterparty.

## Resolving a dispute

Netting and `reconcile` *pinpoint* a disagreement; **arbitration** resolves it.
Each party submits its signed record and the deterministic `arbitrate` decides
which figure stands, producing a content-bound `Resolution` (audit action
`arbitrate`).

```python
from vincio import arbitrate

resolution = app.arbitrate([buyer_record, seller_record])
resolution.status              # "upheld" | "unresolved"
```

The decision rests only on what it can recompute. A reconciliation hash **both**
parties signed (each on their own record) is mutually corroborated and **stands**;
a unilateral claim contradicting it is **rejected** and its claimant pinpointed; a
single uncontested figure stands alone; when nothing is corroborated the dispute
is honestly left **unresolved** rather than decided by fiat. Unlike netting
(which *refuses* to clear over a tampered book), arbitration is the venue where a
bad claim is *adjudicated*: a tampered or forged claim is marked **inadmissible**
and pinpointed, never crashing the resolution.

```python
if resolution.upheld:
    print(resolution.upheld_balance_usd, "corroborated by", resolution.corroborated_by)
    for claim in resolution.rejected_claims:
        print("rejected:", claim.settlement_id, claim.reason)
else:
    resolution.require_resolved()   # raises: no figure was corroborated

verdict = resolution.verify(app.contract_signer)
assert verdict.valid and verdict.hash_ok and verdict.decision_sound
```

The party whose claim did not stand is debited, so bad-faith revision weights the
next negotiation. `book.arbitrate()` resolves one org's own record against a
counterparty's submitted claims.

## Portable reputation

Settlement, netting, and arbitration earn standing, but it lives inside one org's
ledger — a *new* counterparty has no way to trust it. These primitives make
standing portable, current, discoverable, and Sybil-resistant.

### Attest & import

`app.attest_reputation("vendor")` (or `book.attest()`) issues a signed
`ReputationAttestation` over a counterparty's earned standing, derived only from
an org's own signed records and resolutions (a fulfilled delivery a success, a
breach or dissent a failure; audit action `attest_reputation`). A prospective
counterparty pools several issuers' attestations with `app.import_reputation(...)`
(or `combine_attestations()`) into an evidence-weighted `PortableReputation`.

```python
acme_att   = acme.attest_reputation("vendor")     # signed by acme, audited
globex_att = globex.attest_reputation("vendor")   # signed by globex

buyer.use_reputation_ledger()
prior = buyer.import_reputation([acme_att, globex_att])
result = buyer.negotiate("transcribe calls", buyer=..., seller=..., seller_id="vendor")
```

Because a Beta-Bernoulli posterior is conjugate, combining is **pooling the
evidence**, never a single self-asserted number: a self-vouch is refused, an
issuer cannot stack its own pull (only its largest attestation counts), and a
tampered attestation is pinpointed in `prior.refused` and excluded. The prior
weights an *unknown* counterparty; one the buyer has lived through keeps its own
`ReputationLedger` standing (passed as `base=`), so portability only fills the
gap. Read `prior.standing("vendor")` (pooled evidence) and `prior.weight("vendor")`
(the bounded weight it maps to).

### Keeping it current

An attestation is a point-in-time claim, so the prior is time-aware and revocable
without a hosted service:

- **Freshness.** The issuer declares a validity window (`horizon_days=`, bound
  into the signed hash). Against an as-of clock, a combination *excludes* a stale
  attestation (into `prior.stale`) and *decays* an older one by the importer's
  `AttestationConfig(half_life_days=)` — its evidence mass halved each half-life,
  its attested ratio preserved.
- **Revocation.** The issuer withdraws a claim by signing a content-bound
  `AttestationRevocation` naming it **by hash** (`app.revoke_attestation` /
  `book.revoke`). The combination excludes it (into `prior.revoked`). Only the
  issuer can withdraw its *own* attestation; a forged or mis-targeted revocation
  cannot cancel a claim.

```python
from vincio.core.utils import utcnow
from vincio.settlement import AttestationConfig

att = acme.attest_reputation("vendor", horizon_days=90)
cfg = AttestationConfig(half_life_days=30)
prior = buyer.import_reputation([att], config=cfg, as_of=utcnow())

rev = acme.revoke_attestation(att, reason="vendor regressed")   # signed, audited
prior = buyer.import_reputation([att], revocations=[rev])
assert prior.standing("vendor") is None        # the withdrawn claim is excluded
assert prior.revoked[0].issuer == "acme"       # pinpointed, not silently dropped
```

### Discovering it (gossip)

An importer still has to be *handed* the right bundle. A bounded, **pull-based**
exchange over A2A closes that gap. An org exposes its book as a queryable peer
with `app.serve_attestations()` (an `A2AServer` advertising an
`attestation-exchange` skill), answering a subject query with a `ReputationBundle`
of its *own* signed artifacts and nothing else.

```python
acme.use_settlement_book()
acme_peer = acme.serve_attestations()      # pull-only

buyer.use_reputation_ledger()
result = await buyer.agather_reputation(
    "vendor",
    peers={"acme": acme_peer, "globex": globex_peer},
    directory=buyer.agent_directory(allow=["acme", "globex"]),  # governed
    max_peers=8,                                                # bounded fan-out
)
result.weight("vendor")            # drops into the negotiation path
result.standing("vendor").issuers  # which peers corroborated
```

Each peer is governed through an `AgentDirectory` allow-list, every fetched
artifact is independently verified from the bytes, artifacts are deduped by hash,
and the result folds into the *same* `combine_attestations` — gossip changes only
*where the evidence comes from*. A denied peer is skipped and pinpointed
(`result.visit_for(peer)`), a forged artifact is refused, a gossiped revocation
excludes the claim, and every peer/fetch lands on the audit log
(`reputation_peer` / `reputation_fetch`).

### Weighting by trust in the issuer

Equal-pull pooling lets a clutch of unknown peers out-evidence a few you've lived
through — and lets a Sybil spin up issuers that all vouch alike. The **opt-in**
trust kernel scales each issuer's contributed evidence *mass* by your own trust in
that issuer: a bounded, transitive web-of-trust rooted in your local
`ReputationLedger`. Pass `trust_config=` (or an explicit `trust=`) to
`combine_attestations` / `import_reputation` / `gather_reputation`; with neither,
pooling is unchanged.

```python
from vincio import TrustConfig

prior = buyer.import_reputation(
    [trusted_issuer_att, unknown_issuer_att],   # equal evidence each
    trust_config=TrustConfig(),
)
prior.standing("vendor").issuer_trust   # {"acme": 0.93, "stranger": 0.10}
```

A first-hand issuer counts at its earned weight (hop 0); trust then composes **at
most a bounded hop** outward under a per-hop decay, so a long unverifiable chain
can't manufacture standing. An unknown issuer is *floored* (`trust_floor`), never
zeroed. Because trust is lent only *outward from a trusted root*, a cluster of
mutually-vouching unknowns is never reached — **pull follows earned trust, not
issuer count**, so a Sybil clutch can't outvote a few trusted peers. Build and
inspect the model with `build_trust_model(...)` (`IssuerTrust.trust` / `.depth` /
`.vouched_by`); the applied multipliers read back from `AttestationVerdict.trust`
and `SubjectStanding.issuer_trust`.

## Admission & collateral

### Gating admission by standing

Standing only ever *softened* a negotiation; nothing **acted** on a too-thin or
too-low one. An `AdmissionPolicy` (`app.admit`) maps a `PortableReputation` or a
local `ReputationLedger` to a bounded `AdmissionDecision`: an exposure ceiling, a
required escrow fraction, and an SLA-strictness factor (audited).

```python
from vincio import AdmissionConfig

policy = AdmissionConfig(parity_exposure_usd=1000.0)   # the ceiling at full trust
decision = buyer.admit("vendor", config=policy)
decision.max_contract_value_usd   # the exposure ceiling this standing earns
decision.escrow_fraction          # collateral asked at this trust level
decision.verify().valid           # terms re-derive from the bytes
```

Exposure is the product of two bounded signals — *how good* the standing is
(posterior mean) and *how much corroborated, settled history* backs it — ramped
from a `floor_fraction` toward parity. So a thin standing is **admitted on
conservative terms rather than refused**, and its ceiling ramps deterministically
toward parity as it accrues settled deliveries (a regression walking it back).
Local first-hand evidence wins over what others attest.

The decision folds into the **existing** path, no new code path:

```python
from vincio.negotiation import buyer_position, seller_position

bounded = decision.bound_position(           # clamp the buyer to the exposure ceiling
    buyer_position(max_price_usd=1e6, ideal_price_usd=0.01, max_sla_seconds=5.0)
)
result = buyer.negotiate("transcribe", buyer=bounded, seller=seller_position(...),
                         seller_id="vendor")   # can only converge within the ceiling
```

`apply_to_terms(terms)` caps a contract's price and stamps the escrow posture into
the terms' (unhashed) metadata, so a contract minted from capped terms stays
verifiable while carrying the collateral it must post.

### Backing it with escrow

The stamped escrow fraction is only a number until something **holds** it. An
`Escrow` (`app.post_escrow`) makes posted collateral a verifiable escrow bound to
the contract — the escrow analogue of a `SettlementRecord`.

```python
escrow = buyer.post_escrow(contract, decision=decision)   # hold the required collateral
escrow.amount_usd          # = escrow_fraction × the contract price
escrow.verify().valid

record = buyer.settle(contract, cost_usd=140.0, escrow=escrow)   # a 40% cost overrun
escrow.state            # "forfeited"
escrow.forfeited_usd    # a slice proportional to the shortfall, never the whole stake
escrow.released_usd     # the remainder, back to the poster
escrow.breaches         # ["price"], pinpointed
```

The collateral source is, in order: an explicit `amount`, an explicit `fraction`,
an `AdmissionDecision`'s `escrow_fraction` (`decision=`), or the posture stamped
on the terms. Resolution is driven by the **same** `SettlementRecord` verdict — a
fulfilled delivery releases the whole stake, a breach forfeits a **bounded,
pinpointed slice proportional to the shortfall** (never punitive), the rest
released. Pass `escrow=` to `app.settle` to resolve in one call, or
`app.settle_escrow(escrow, record)` against a record you already have;
`EscrowConfig(max_forfeit_fraction=0.8)` caps a single breach's forfeiture.

### Pooling collateral across contracts

An escrow backs *one* contract, stranding capital per-deal. A `CollateralPool`
(`app.post_collateral_pool`) is a **bounded margin account** posted once that
backs many contracts — the collateral analogue of a `NettingSet`.

```python
pool = buyer.post_collateral_pool([c1, c2, c3], decisions=decision)
pool.posted_usd                              # the single stake
[c.allocated_usd for c in pool.contracts]    # shares proportional to each requirement
pool.verify().valid

buyer.settle(c1, cost_usd=60.0,  pool=pool)  # clean → frees its requirement back to available
buyer.settle(c2, cost_usd=300.0, pool=pool)  # breach → draws a bounded slice from the stake
pool.contract(c2.id).forfeited_usd           # proportional to the shortfall
pool.available_usd                           # capital the clean delivery freed
```

Each requirement re-derives from its admission posture (a per-contract-id
`decisions=` dict, a uniform `fraction=`, or the stamped posture). A clean
delivery frees committed capital (reused, not stranded); a breach draws the
**same** proportional forfeiture the settlement measures. A pool committed below
what its open contracts require surfaces a bounded, pinpointed **top-up**
obligation rather than over-committing:

```python
pool.back(c4, decision=decision)   # backing a new deal can over-commit
pool.topup_usd                     # the bounded shortfall
pool.top_up(pool.topup_usd)        # add capital; the obligation clears
```

### Guarding against rehypothecation

A pool re-allocates only *within itself*. Pledging the **same** stake across more
than one pool double-counts what backs each deal. A `CollateralLedger`
(`app.guard_collateral`) folds a counterparty's pools into one view and reconciles
what they collectively pledge against what it actually holds (audited).

```python
pool_one = buyer.post_collateral_pool([c_a, c_b], decisions=decision)
pool_two = buyer.post_collateral_pool([c_a],      decisions=decision)   # c_a re-pledged
ledger = buyer.guard_collateral([pool_one, pool_two])

ledger.pledged_usd      # what the pools collectively pledge
ledger.held_usd         # capital actually held (defaults to the distinct pledge)
ledger.reuse_usd        # the over-commitment
ledger.over_committed   # the same capital pledged twice
ledger.breaches[0]      # the ReuseBreach: contract A, the pools, the excess
```

Pass `held=` when you know the true custody balance; it defaults to the gross
pledge minus the provably double-pledged capital, so genuinely separately-funded
pools don't flag. When a scarce stake backs deals for several beneficiaries, each
claim is bounded to its deterministic **pari-passu** share, so a forfeiture can't
pay one beneficiary out of capital another has first claim on:

```python
ledger = buyer.guard_collateral([multi_beneficiary_pool], held=20.0)
claim = ledger.claim("globex")
claim.claim_usd / claim.secured_usd / claim.unsecured_usd   # pledged / secured share / exposed
ledger.require_within_bounds()   # raises if over-committed
```

## Proving the capital

The `held` figure the guard trusts is *asserted*, not proven — until these
proofs make it evidence-backed. **`held=`, `custody=`, and `solvency=` are
mutually exclusive**: the held figure has exactly one source, and an asserted
`held=` can over-commit but can never *under-reserve*, because nothing proves it.

### Proof of reserves

A `CustodyAttestation` (`app.attest_custody`) attests the capital actually held,
itemized into reserve accounts whose total re-derives on verify;
`guard_collateral(custody=)` reads it instead of the asserted default (audit
action `custody_attestation`).

```python
proof = custodian.attest_custody("vendor", {"omnibus": 40.0, "escrow": 10.0})  # 50 proven
proof.verify(custodian.contract_signer).valid

ledger = buyer.guard_collateral([pool], custody=proof)   # held = proven reserves
ledger.reserves_proven    # evidence-backed, not asserted
ledger.under_reserved     # True when reserves fall below what the pools pledge
ledger.reserve_breach     # the UnderReservedBreach: custodian, attestation hash, shortfall
ledger.require_reserved()
```

An attestation that vouches for a **different poster** than the pools' is refused.

### Proof of solvency

Proven reserves are one side of the ledger; a counterparty solvent against *one*
buyer may be under-water once *every* obligation is counted. A
`LiabilityAttestation` (`app.attest_liabilities`) attests the total owed, itemized
per creditor; `app.prove_solvency` folds it against the reserves into a
`SolvencyProof` — a bounded margin `reserves − liabilities` the guard reads
(audit actions `liability_attestation`, `solvency_proof`).

```python
reserves = custodian.attest_custody("vendor", {"omnibus": 80.0})               # 80 held
owed     = auditor.attest_liabilities("vendor", {"globex": 35.0, "initech": 15.0})  # 50 owed
proof    = auditor.prove_solvency(reserves, owed)
proof.solvent                 # reserves cover liabilities
proof.margin_usd              # 30.0
proof.solvency_adjusted_held  # 30.0 = max(0, margin), the unencumbered capital

ledger = buyer.guard_collateral([pool], solvency=proof)   # held = solvency-adjusted margin
ledger.solvency_adjusted / ledger.insolvent
```

A custody/liability pair for **different posters** is refused; the margin and the
`InsolvencyBreach` re-derive from the bytes.

### Proof of complete liabilities

A solvency proof folds the attestor's *total*, which a counterparty could
**under-state** by omitting a creditor. So a `LiabilityAttestation` commits its
line items into a **Merkle root** (bound in the signed hash): each creditor gets an
`InclusionProof` (`app.inclusion_proof`) that its claim is a leaf, and
`check_completeness` (`app.check_completeness`) folds creditors' own proven claims,
pinpointing omissions and raising the total to a **completed** figure
`prove_solvency` reads (audit action `liability_completeness`).

```python
owed = auditor.attest_liabilities("vendor", {"acme": 60.0})   # quietly omits globex

auditor.inclusion_proof(owed, "acme").verify(owed).valid       # a leaf of the signed root

check = globex.check_completeness(owed, {"globex": 40.0})      # globex folds its own claim
check.complete            # False
check.omitted_creditors   # ["globex"]
check.completed_usd       # 100.0 (attested 60 + proven 40)

proof = auditor.prove_solvency(reserves, owed, completeness=check)   # margin uses 100, not 60
proof.insolvent           # the hidden $40 tips it into a shortfall
proof.understated_usd     # 40.0
check.require_complete()
```

A tampered leaf or forged root fails to reconstruct the committed root; a check
for a *different* attestation or poster is refused. When `claims` is omitted, the
book derives them from its owner's own settled records.

### Non-equivocation across creditors

Completeness catches an omission only when the omitted creditor folds its claim.
But a per-relationship attestor can **equivocate**: sign a *smaller* liability root
for one creditor, a different one for another, each `InclusionProof` valid against
the root *it* was shown. Creditors compare the signed roots.
`attestation.root_commitment()` produces a privacy-preserving `RootCommitment`
(root + `as_of`, **no** line items), and `check_root_consistency`
(`app.check_root_consistency`) groups held attestations by `(poster, attestor,
as_of)` and folds any two conflicting roots into a non-repudiable
`EquivocationProof` (audit action `liability_equivocation`; also dings the poster).

```python
as_of = datetime(2026, 1, 1, tzinfo=UTC)
to_acme   = auditor.attest_liabilities("vendor", {"acme": 60.0},   as_of=as_of)
to_globex = auditor.attest_liabilities("vendor", {"globex": 40.0}, as_of=as_of)

to_acme.root_commitment().conflicts_with(to_globex.root_commitment())   # True

report = auditor.check_root_consistency(
    [("acme", to_acme), ("globex", to_globex)], verify_with=auditor.contract_signer
)
report.consistent                              # False
report.equivocating_posters                    # ["vendor"]
report.equivocations[0].liabilities_gap_usd    # 20.0
report.require_consistent()
```

With the attestor's verifier a **forged conflicting root is refused**, so it can't
manufacture a false accusation. Non-equivocation is defined *for one `as_of`*: two
roots signed as of the same instant contradict; two for different instants are
distinct snapshots a later one legitimately supersedes.

### Liability history over time

Non-equivocation is scoped to one `as_of`; a *later* snapshot could quietly
**drop** a past obligation, each snapshot internally sound. Link snapshots with
`prior=` (a hash-linked sequence, each `as_of` strictly succeeding the last) and
`check_history_consistency` (`app.check_history_consistency`) walks them,
pinpointing any debt that vanished without a signed, **creditor-issued**
`Discharge` (audit action `liability_history`; dings the poster).

```python
t1 = datetime(2026, 1, 1, tzinfo=UTC)
t2 = datetime(2026, 2, 1, tzinfo=UTC)
s1 = auditor.attest_liabilities("vendor", {"acme": 100.0}, as_of=t1)
s2 = auditor.attest_liabilities("vendor", {"acme": 30.0},  as_of=t2, prior=s1)  # acme dropped $70

report = auditor.check_history_consistency([s1, s2])
report.consistent                                # False
report.proofs[0].breaches[0].unexplained_usd     # 70.0

settled = acme.discharge_liability("vendor", 70.0, as_of=t2)   # creditor's to issue
auditor.check_history_consistency([s1, s2], discharges=[settled]).consistent   # True
```

The discharge is the *creditor's* — a poster can't forge its own. A back-dated
link is refused, a tampered snapshot is inadmissible, and a dropped
`MonotonicityBreach` is caught by re-derivation. `require_monotone` raises on any
unexplained drop; `require_linked` additionally demands a contiguous chain.

## Resolving an insolvency

A solvency proof *flags* insolvency but says nothing about **who** the scarce
capital pays. Rank obligations into a signed `SenioritySchedule`
(`app.build_seniority_schedule`, position = priority, rank `0` most senior) and
`resolve_insolvency` distributes the proven reserves **by seniority then
pari-passu within a tranche** into an `InsolvencyResolution` (audit action
`insolvency_resolution`; dings a poster that can't make creditors whole).

```python
reserves = vendor.attest_custody("vendor", {"omnibus": 60.0})                      # $60 held
owed     = auditor.attest_liabilities("vendor", {"bank": 50.0, "acme": 30.0, "globex": 20.0})  # $100 owed
schedule = bank.build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]])

res = auditor.resolve_insolvency(reserves, owed, schedule)
res.status                             # "resolved" (a creditor bears a shortfall)
res.recovery_of("bank").recovery_usd   # 50.0, senior tranche paid in full first
res.recovery_of("acme").recovery_usd   # 6.0, junior tranche pari-passu (20%)
res.shortfall_bearers                  # ["acme", "globex"]
res.require_fully_recovered()
```

With no schedule the whole set is one pari-passu tranche. `resolve_insolvency`
reuses `prove_solvency` (so a tampered/wrong-poster attestation or schedule is
refused); pass `completeness=` to distribute against the *completed* set.
`verify` re-derives the whole distribution — an over-stated recovery or re-ordered
tranche is caught, and passing the `schedule` binds each rank to the one signed.

### Set-off before the waterfall

A creditor of an insolvent estate is often *also* a debtor of it. Real insolvency
law nets mutual obligations first (**close-out netting**). State the obligations
running both ways as a `SetOffStatement` (`app.build_set_off_statement`, or
`set_off_from_records` to derive it from existing liabilities + settlement
records), have **both** parties co-sign, and pass `set_off=` to net each creditor
to its true exposure before distributing (audit action `liability_set_off`).

```python
owed = auditor.attest_liabilities("vendor", {"bank": 50.0, "acme": 30.0})   # $80 gross
# acme owes the vendor $12 back: its $30 claim nets to $18. Both parties co-sign.
statement = build_set_off_statement("vendor", "acme", 30.0, 12.0)
statement.sign(vendor_signer, party="vendor").sign(acme_signer, party="acme")

res = auditor.resolve_insolvency(reserves, owed, set_off=[statement])
res.gross_liabilities_usd             # 80.0, before set-off
res.liabilities_usd                   # 68.0, acme netted 30 → 18
res.recovery_of("acme").set_off_usd   # 12.0 netted out
```

A creditor **in debit** (owing at least as much as owed) nets to a zero claim.
Set-off is applied after `completeness` (nets the *completed* gross), reconciled
against that gross, and a **one-sided** close-out is refused at fold time.
`verify` re-derives every net claim; passing `set_off=` to `verify` binds the
statements by hash.

## The engagement capstone

Every primitive above is signed, content-bound, and offline-verifiable on its own.
`app.cross_org_engagement(...)` returns a `CrossOrgEngagement`: a purely
compositional facade that threads the whole pipeline behind one governed call-path
and seals it into a single hash-linked `EngagementNarrative`. Each lifecycle
method delegates to the **same** `app.*` primitive you could call directly,
captures the artifact it produced, and records it as a stage.

```python
from vincio.providers import MockProvider

app = ContextApp(name="acme", provider=MockProvider(default_text="ok"))
eng = app.cross_org_engagement(buyer="acme", seller="vendor", scope="transcribe 1k calls")

contract = eng.negotiate(
    buyer=buyer_position(max_price_usd=0.12, max_sla_seconds=5.0),
    seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.10),
)
eng.choreograph(saga, participants=parts, directory=directory)   # discover + deliver
eng.settle_saga(contracts={contract.id: contract})              # meter + settle
eng.net()                                                       # multilateral clearing
reserves = eng.attest_custody("vendor", {"omnibus": 120.0})
owed     = eng.attest_liabilities("vendor", {"acme": 40.0, "globex": 30.0})
eng.prove_solvency(reserves, owed)

narrative = eng.seal()                       # one content-bound, signed narrative
narrative.verify(app.contract_signer).valid  # recomputes the whole chain offline
```

Each `EngagementStage` binds the verb, the captured artifact's own content hash,
and a digest of its bytes into a link that chains to the previous one.
`narrative.verify()` recomputes the whole chain (a re-ordered stage, edited digest,
broken link, or forged signature is caught); `eng.verify(app.contract_signer)`
additionally re-digests the *live* artifacts against the bound digests, so a tamper
to any underlying artifact is caught too. The facade adds no economic logic —
every primitive stays unchanged and usable on its own. Sealing lands on the audit
log under `cross_org_engagement`, one continuous signed narrative from the first
offer to the final distribution.

With this capstone the cross-org settlement & credit surface is
**feature-complete and frozen** under the
[stability policy](../reference/stability.md): no further cross-org *primitive* is
scheduled, and subsequent work is bug-fix and standards-tracking only.

## What it is not

A library capability inside your process, not a payment rail or a hosted service.
Concretely, each primitive is a *calculation over signed artifacts you hold*, not
an operated institution:

- a **settlement** is a signed record, not money in motion;
- a **book** is a hash-chained ledger each org keeps itself, not a custodian;
- **netting** is a clearing *calculation* over those books, not a clearing *house*;
- **arbitration** is deterministic adjudication over the parties' own signed
  records, not a court that rules by fiat;
- a **reputation attestation / exchange / trust kernel** is your own signed,
  verifiable evidence pooled and weighted in-process, not a reputation bureau,
  registry, gossip bus, or Sybil-detection service;
- an **admission decision** is a mechanical exposure number from standing you
  already hold, not an underwriting service;
- an **escrow / pool / collateral ledger** is a verifiable record of collateral
  posted, allocated, and reconciled against a delivery verdict, not a custodian,
  margin account, or rehypothecation registry;
- a **custody / solvency / completeness / equivocation / history proof** is one
  party's signed, verifiable fold over reserves and liabilities, not a hosted
  auditor or transparency log;
- an **insolvency resolution / set-off** is a verifiable distribution by seniority
  then pari-passu, netting mutual obligations first, not a receiver or bankruptcy
  court;
- a **cross-org engagement** is a verifiable, hash-linked narrative proving the
  primitives chain end-to-end, not an orchestration service or control plane.

Vincio gives you a verifiable reconciliation of what was owed and delivered, and a
verifiable proof at every rung above it. *How* an obligation is paid is yours.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Example: 12_cross_org_economy.py](../../examples/12_cross_org_economy.py)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#governance)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
