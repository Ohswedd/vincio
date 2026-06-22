"""Bounded agent negotiation & typed contracting over the A2A fabric.

Vincio governs a fabric of agents over A2A and the MCP registry behind an
allow-list, scores per-member reliability with a Beta-Bernoulli reputation
ledger, and discounts an unreliable member's pull on a federated round. This
package adds the next rung: **bounded negotiation and contracting** between agents
in a multi-org crew.

* A :class:`Negotiation` runs a typed, terminating offer/counter-offer bargain
  between a buyer and a seller party — the negotiation analogue of a bounded crew
  round. It returns a :class:`NegotiationResult` carrying a signed
  :class:`Contract` on agreement, or a partial result when its round/deadline
  budget runs out.
* A :class:`Contract` is a typed agreement over **price / SLA / scope / quality**
  that both parties sign, that :meth:`Contract.verify` checks **offline** from the
  bytes alone, and that :meth:`Contract.to_budget` / :meth:`Contract.check` enforce
  like any other budget.
* The counterparty's standing in a
  :class:`~vincio.optimize.reputation.ReputationLedger` weights its offers, so a
  repeatedly-regressing agent is **discounted without being singled out**, and
  :func:`select_offer` picks the reputation-weighted best deal among competing
  sellers.

A negotiation runs fully offline against local deterministic parties, or over the
A2A agent fabric against a remote counterparty (see :mod:`vincio.negotiation.fabric`).
Every outcome lands on the hash-chained audit log, so a contract is a mechanical,
verifiable artifact — never a hosted marketplace.
"""

from __future__ import annotations

from .contract import (
    Contract,
    ContractFulfillment,
    ContractSignature,
    ContractTerms,
    ContractVerification,
)
from .engine import (
    IssuePreference,
    LocalParty,
    Negotiation,
    NegotiationBudget,
    NegotiationPosition,
    NegotiationResult,
    Offer,
    Party,
    Role,
    buyer_position,
    select_offer,
    seller_position,
)
from .fabric import A2ANegotiator, negotiation_a2a_server

__all__ = [
    # contract artifact
    "Contract",
    "ContractTerms",
    "ContractSignature",
    "ContractVerification",
    "ContractFulfillment",
    # negotiation engine
    "Negotiation",
    "NegotiationResult",
    "NegotiationBudget",
    "NegotiationPosition",
    "IssuePreference",
    "Offer",
    "Party",
    "LocalParty",
    "Role",
    "select_offer",
    "buyer_position",
    "seller_position",
    # A2A fabric binding
    "A2ANegotiator",
    "negotiation_a2a_server",
]
