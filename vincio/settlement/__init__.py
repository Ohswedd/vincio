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
Everything is dependency-free, deterministic, and
offline — never a hosted marketplace, a clearing house, an arbitration service, a
reputation service, an underwriting service, or a payment processor, only a mechanical,
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
    # reputation gossip & attestation exchange
    "ReputationBundle",
    "PeerVisit",
    "GatheredReputation",
    "AttestationExchange",
    "attestation_a2a_server",
    "gather_reputation",
]
