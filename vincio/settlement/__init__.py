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
across a web of contracts closes its books once. Everything is dependency-free,
deterministic, and offline — never a hosted marketplace, a clearing house, or a
payment processor, only a mechanical, verifiable reconciliation.
"""

from __future__ import annotations

from .book import (
    BookVerification,
    SettlementBook,
    SettlementReport,
    SettlementRow,
    settle_contract,
    settle_saga,
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
]
