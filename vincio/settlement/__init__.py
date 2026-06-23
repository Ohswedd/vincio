"""Agent-to-agent settlement & metering over the negotiated contract.

With cross-org sagas dispatching contracted work across organizations, the next
reach is **closing the books** on it: a metered, auditable settlement record for
work delivered under a negotiated :class:`~vincio.negotiation.Contract`, so a
cross-org engagement reconciles the way a run closes its cost report — never a
payment rail, only a verifiable ledger of what was owed and delivered.

* A :class:`Meter` accrues the **usage** of delivered work against the agreed
  price as a saga's steps complete — each unit a :class:`UsageEvent` attributed to
  the contract and the run — and rolls it up into a deterministic, total-preserving
  :class:`MeterReading`.
* A :class:`SettlementRecord` reconciles that delivery against the agreed
  price / SLA / quality into a typed, **signed, offline-verifiable** record: both
  parties sign one reconciliation hash, :meth:`SettlementRecord.verify` checks it
  from the bytes alone, and :func:`reconcile` ties two orgs' records out — a
  dispute is pinpointed, not merely flagged.
* A :class:`SettlementBook` is an org's durable, **hash-chained** ledger of those
  records — the settlement analogue of the
  :class:`~vincio.choreography.SagaJournal` — that :meth:`SettlementBook.verify`
  recomputes offline and :meth:`SettlementBook.report` rolls up per counterparty.
  Closing the books also **closes the reputation loop**: a settled overrun or
  shortfall debits the seller, so reliability earned in delivery weights the next
  negotiation.

:func:`settle_contract` settles one contract; :func:`settle_saga` settles every
contract a cross-org saga ran under, straight from its durable journal.

Once an org keeps many bilateral books, :func:`net_settlements` / :func:`net_books`
fold the whole fleet's balances into a single, content-bound :class:`NettingSet` —
each org's many positions collapsed to the **minimal set** of net obligations,
offline-verifiable the way a record is — so an org that is both buyer and seller
across a web of contracts closes its books once. When a disagreement is pinpointed
(a :class:`NettingDispute`, or two records that do not reconcile), :func:`arbitrate`
adjudicates it: each party submits its signed records and a deterministic
:class:`Resolution` decides which figure stands — a reconciliation hash both parties
co-signed is upheld, a contradicting unilateral claim is rejected and pinpointed —
content-bound and offline-verifiable the way a record is.

The standing all of this earns lives inside one org's own ledger; making it
**portable** is the last reach. :func:`attest_reputation` (or
:meth:`SettlementBook.attest`) issues a signed, offline-verifiable
:class:`ReputationAttestation` over a counterparty's earned standing — derived from
an org's own :class:`SettlementBook` and arbitration :class:`Resolution`\\ s — that a
prospective counterparty verifies from the bytes alone (a tampered score or a forged
issuer is caught) and :func:`combine_attestations` folds, across several issuers,
into a bounded, evidence-weighted :class:`PortableReputation` prior that weights the
next negotiation under the same ``[floor, 1]`` rule a local reputation does — never a
hosted reputation bureau. Because standing changes, the portable prior is
**time-aware and revocable**: an attestation carries an issuer ``horizon_days``
validity window, an issuer signs a content-bound :class:`AttestationRevocation` to
withdraw or supersede one (:meth:`SettlementBook.revoke`), and an as-of-aware
:func:`combine_attestations` decays a stale attestation out of the prior and excludes
a revoked one — pinpointed, never silently honored. And because pooling every issuer's
evidence with equal pull lets a *Sybil* cluster manufacture standing, a
:class:`TrustModel` (:func:`build_trust_model`) weighs each issuer's evidence by the
importer's **own trust in that issuer** — a bounded, transitive web-of-trust rooted in
its local ledger — so corroboration from a trusted peer counts for more than volume
from an unknown one. And because that weighted standing is still only *consulted* as a
soft weight on a negotiation, an :class:`AdmissionPolicy` (:func:`admit` /
:meth:`~vincio.core.app.ContextApp.admit`) finally **acts** on it: it maps a
counterparty's :class:`PortableReputation` / :class:`~vincio.optimize.reputation.ReputationLedger`
standing to a bounded, offline-verifiable :class:`AdmissionDecision` — a maximum
contract value, a required escrow fraction, and an SLA-strictness factor — admitting a
thin or low-trust standing on *conservative* terms rather than refusing it, and ramping
its ceiling toward parity as settled, corroborated history accrues (a regression walking
it back). The decision binds the standing it read and the terms it set onto a content
hash and folds into the existing negotiation / contracting path
(:meth:`AdmissionDecision.bound_position` / :meth:`AdmissionDecision.apply_to_terms`).
And because that required escrow fraction was still only a *number stamped on the terms* —
nothing **held** it, released it on a clean delivery, or forfeited a slice on a breach —
an :class:`Escrow` (:func:`post_escrow` / :meth:`~vincio.core.app.ContextApp.post_escrow`)
finally backs it: it binds the admission-required collateral to a *specific*
:class:`~vincio.negotiation.Contract` and counterparty into a signed, offline-verifiable
artifact (the escrow analogue of a :class:`SettlementRecord`), and settling the contract
**releases** the whole stake on a fulfilled delivery or **forfeits** a bounded, pinpointed
slice proportional to the shortfall on a breach (:func:`settle_escrow` /
:meth:`SettlementBook.settle` with an attached escrow) — driven by the same settlement
verdict the books already close on, so the collateral closes the same loop the settlement
does. And because a counterparty running many concurrent contracts would have to lock
*separate* collateral per deal even though its breaches and clean deliveries net out — capital
stranded contract-by-contract the way bilateral settlements were stranded book-by-book before
netting — a :class:`CollateralPool` (:func:`post_collateral_pool` /
:meth:`~vincio.core.app.ContextApp.post_collateral_pool`) pools it: a single posted stake
backs many contracts at a deterministic, offline-verifiable allocation (proportional to each
contract's admission-required collateral — the collateral analogue of a :class:`NettingSet`),
settling a contract **draws** a bounded forfeiture from the shared stake and **releases** the
rest back to the available balance (:func:`draw_pool` / :meth:`SettlementBook.settle` with an
attached pool), and a pool committed below the collateral its open contracts require surfaces a
bounded, pinpointed **top-up** obligation rather than silently over-committing. And because a
pool only ever re-allocates capital *within itself* — nothing bounds a counterparty that
pledges the *same* stake across more than one pool, double-counting what actually backs each
deal — a :class:`CollateralLedger` (:func:`guard_collateral` /
:meth:`~vincio.core.app.ContextApp.guard_collateral`) is the **rehypothecation guard**: it
folds a counterparty's pools into one view, reconciles what they collectively pledge against
the capital it actually holds, surfaces the same capital pledged twice as a bounded, pinpointed
:class:`ReuseBreach`, and bounds each beneficiary's claim to its deterministic pari-passu share
so a forfeiture cannot pay one beneficiary out of capital another has first claim on — reading
only the existing signed, content-bound pools (a tampered one is refused) and landing the guard
on the hash-chained audit log. And because the guard's ``held`` figure was the one input it
*trusted* — asserted, not proven — a :class:`CustodyAttestation` (:func:`attest_custody` /
:meth:`~vincio.core.app.ContextApp.attest_custody`) makes it **evidence-backed**: a custodian
(or the poster's own signed reserve record) issues a signed, content-bound **proof-of-reserves**
over the capital actually held, which :func:`guard_collateral` reads (``custody=``) as the held
figure — surfacing an :class:`UnderReservedBreach` when the proven reserves fall below the
pledges and refusing a tampered figure, a forged custodian, or an attestation for a different
poster. And because proven reserves are only one side of the ledger — a counterparty solvent
against one buyer's pledges may be under-water once *every* obligation it owes is counted — a
:class:`LiabilityAttestation` (:func:`attest_liabilities`) makes the liability side
evidence-backed too, and :func:`prove_solvency` folds the two proofs into a signed,
offline-verifiable :class:`SolvencyProof` (reserves − liabilities) that :func:`guard_collateral`
reads (``solvency=``) as a *solvency-adjusted* held figure — bounding a pledge against capital
not already owed elsewhere and pinpointing an :class:`InsolvencyBreach` when the proven
liabilities exceed the reserves. Everything is dependency-free, deterministic, and offline —
never a hosted marketplace, a clearing house, an arbitration service, a reputation service, an
underwriting service, an escrow custodian, a margin custodian, a rehypothecation registry, a
proof-of-reserves auditor, a solvency auditor, or a payment processor, only a mechanical,
verifiable reconciliation.
"""

from __future__ import annotations

from .admission import (
    AdmissionConfig,
    AdmissionDecision,
    AdmissionPolicy,
    AdmissionVerification,
    Standing,
    admit,
)
from .arbitration import (
    ClaimVerdict,
    Resolution,
    ResolutionStatus,
    ResolutionVerification,
    arbitrate,
)
from .attestation import (
    AttestationConfig,
    AttestationRevocation,
    AttestationVerdict,
    AttestationVerification,
    IssuerTrust,
    PortableReputation,
    ReputationAttestation,
    RevocationVerification,
    SubjectStanding,
    TrustConfig,
    TrustModel,
    attest_reputation,
    build_trust_model,
    combine_attestations,
    revoke_attestation,
)
from .book import (
    BookVerification,
    SettlementBook,
    SettlementReport,
    SettlementRow,
    settle_contract,
    settle_saga,
)
from .collateral import (
    CollateralPool,
    CollateralPoolVerification,
    PooledContract,
    PooledContractState,
    PoolStatus,
    draw_pool,
    post_collateral_pool,
)
from .custody import (
    CustodyAttestation,
    CustodyAttestationVerification,
    ReserveLine,
    attest_custody,
)
from .escrow import (
    Escrow,
    EscrowConfig,
    EscrowSignature,
    EscrowState,
    EscrowVerification,
    post_escrow,
    settle_escrow,
)
from .exchange import (
    AttestationExchange,
    GatheredReputation,
    PeerVisit,
    ReputationBundle,
    attestation_a2a_server,
    gather_reputation,
)
from .meter import Meter, MeterReading, UsageEvent
from .netting import (
    BilateralNet,
    GrossObligation,
    NetObligation,
    NetPosition,
    NettingDispute,
    NettingSet,
    NettingVerification,
    net_books,
    net_settlements,
)
from .record import (
    Reconciliation,
    SettlementLine,
    SettlementRecord,
    SettlementSignature,
    SettlementStatus,
    SettlementVerification,
    reconcile,
)
from .rehypothecation import (
    BeneficiaryClaim,
    CollateralLedger,
    CollateralLedgerVerification,
    LedgerContract,
    LedgerPool,
    ReuseBreach,
    UnderReservedBreach,
    guard_collateral,
)
from .solvency import (
    CompletenessProof,
    CompletenessVerification,
    Discharge,
    DischargeVerification,
    EquivocationProof,
    EquivocationProofVerification,
    HistoryConsistencyProof,
    HistoryConsistencyProofVerification,
    HistoryConsistencyReport,
    InclusionProof,
    InclusionProofVerification,
    InsolvencyBreach,
    LiabilityAttestation,
    LiabilityAttestationVerification,
    LiabilityLine,
    MerkleStep,
    MonotonicityBreach,
    OmissionBreach,
    RootCommitment,
    RootCommitmentVerification,
    RootConsistencyReport,
    SolvencyProof,
    SolvencyProofVerification,
    attest_liabilities,
    check_completeness,
    check_history_consistency,
    check_root_consistency,
    discharge_liability,
    prove_equivocation,
    prove_solvency,
)

__all__ = [
    # metering primitive
    "Meter",
    "MeterReading",
    "UsageEvent",
    # settlement record & reconciliation
    "SettlementRecord",
    "SettlementLine",
    "SettlementSignature",
    "SettlementVerification",
    "SettlementStatus",
    "Reconciliation",
    "reconcile",
    # durable book & engine
    "SettlementBook",
    "SettlementReport",
    "SettlementRow",
    "BookVerification",
    "settle_contract",
    "settle_saga",
    # multilateral netting & clearing
    "NettingSet",
    "NetPosition",
    "NetObligation",
    "BilateralNet",
    "GrossObligation",
    "NettingDispute",
    "NettingVerification",
    "net_settlements",
    "net_books",
    # dispute resolution & arbitration
    "Resolution",
    "ResolutionStatus",
    "ResolutionVerification",
    "ClaimVerdict",
    "arbitrate",
    # reputation attestation & portability
    "ReputationAttestation",
    "AttestationConfig",
    "AttestationVerification",
    "AttestationVerdict",
    "AttestationRevocation",
    "RevocationVerification",
    "SubjectStanding",
    "PortableReputation",
    "attest_reputation",
    "revoke_attestation",
    "combine_attestations",
    # transitive trust & Sybil-resistant weighting
    "TrustConfig",
    "TrustModel",
    "IssuerTrust",
    "build_trust_model",
    # reputation-gated admission & progressive exposure
    "AdmissionConfig",
    "AdmissionDecision",
    "AdmissionVerification",
    "AdmissionPolicy",
    "Standing",
    "admit",
    # collateralized settlement & escrow
    "Escrow",
    "EscrowConfig",
    "EscrowSignature",
    "EscrowState",
    "EscrowVerification",
    "post_escrow",
    "settle_escrow",
    # collateral pooling & cross-contract margin
    "CollateralPool",
    "CollateralPoolVerification",
    "PooledContract",
    "PooledContractState",
    "PoolStatus",
    "post_collateral_pool",
    "draw_pool",
    # collateral rehypothecation guards & re-use bounds
    "CollateralLedger",
    "CollateralLedgerVerification",
    "LedgerPool",
    "LedgerContract",
    "ReuseBreach",
    "BeneficiaryClaim",
    "UnderReservedBreach",
    "guard_collateral",
    # collateral custody attestation & proof-of-reserves
    "CustodyAttestation",
    "CustodyAttestationVerification",
    "ReserveLine",
    "attest_custody",
    # custody liability attestation & proof-of-solvency
    "LiabilityAttestation",
    "LiabilityAttestationVerification",
    "LiabilityLine",
    "InsolvencyBreach",
    "SolvencyProof",
    "SolvencyProofVerification",
    "attest_liabilities",
    "prove_solvency",
    # liability inclusion proofs & completeness
    "MerkleStep",
    "InclusionProof",
    "InclusionProofVerification",
    "OmissionBreach",
    "CompletenessProof",
    "CompletenessVerification",
    "check_completeness",
    # liability non-equivocation & root consistency
    "RootCommitment",
    "RootCommitmentVerification",
    "EquivocationProof",
    "EquivocationProofVerification",
    "RootConsistencyReport",
    "prove_equivocation",
    "check_root_consistency",
    "Discharge",
    "DischargeVerification",
    "discharge_liability",
    "MonotonicityBreach",
    "HistoryConsistencyProof",
    "HistoryConsistencyProofVerification",
    "HistoryConsistencyReport",
    "check_history_consistency",
    # reputation gossip & attestation exchange
    "ReputationBundle",
    "PeerVisit",
    "GatheredReputation",
    "AttestationExchange",
    "attestation_a2a_server",
    "gather_reputation",
]
