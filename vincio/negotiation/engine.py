"""Bounded, terminating, reputation-weighted agent negotiation.

A :class:`Negotiation` runs a typed **alternating-offers** bargain between a buyer
agent and a seller agent and converges on a :class:`~vincio.negotiation.contract.Contract`
— or returns a partial result when its round/deadline budget runs out. It is the
negotiation analogue of a bounded crew round: **termination is guaranteed**, and
the worst case is "no deal", never an unbounded loop.

The model is first-principles, not a wrapper around any one library:

* Each party holds a private :class:`NegotiationPosition` — for each issue an
  *ideal* value and a *reservation* (walk-away) value, with a weight. Utility is
  the weighted, normalized distance between an offer and the party's ideal,
  ``∈ [0, 1]`` (``1`` at the ideal, ``0`` at the reservation).
* Concession is **time-dependent** (a Faratin-style polynomial tactic): at round
  ``t`` of a ``T``-round deadline a party offers terms whose utility-to-itself is
  ``1 − (t/T)^(1/e)`` down to its reservation, where the concession exponent
  ``e`` tunes a tough (``e < 1``, "boulware") or generous (``e > 1``, "conceder")
  curve. Concession is monotone toward the reservation, so the bargain converges.
* Acceptance is **rational and termination-preserving** (an ``AC_next`` rule): a
  party accepts the opponent's offer the moment it is at least as good as the offer
  it would make next. Because both sides concede monotonically and accept rationally,
  the bargain ends in a deal within ``T`` rounds when the parties' acceptable regions
  overlap, and ends in a clean "no agreement" at ``T`` when they do not.

**Reputation weights the deal.** When a party evaluates the *opponent's* offer it
discounts that offer's utility by the opponent's standing in a
:class:`~vincio.optimize.reputation.ReputationLedger` (any object exposing
``weight(member_id) -> float``). The weight lives in ``[floor, 1]``, so a
repeatedly-regressing counterparty's offers are **discounted — never zeroed, never
singled out**: the discounted party can still close a deal by conceding more
(a risk premium), and a reformed party recovers. With several competing sellers,
:func:`select_offer` picks the deal that maximizes the buyer's reputation-weighted
utility, so reliability — not just price — decides the winner.

Everything is deterministic and offline; a local party computes its offers from
its position with no model call, and a remote party (see
:mod:`vincio.negotiation.fabric`) exchanges the same typed offers over the A2A
agent fabric.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..core.errors import NegotiationError
from ..core.utils import utcnow
from .contract import Contract, ContractTerms

__all__ = [
    "Role",
    "IssuePreference",
    "NegotiationPosition",
    "Offer",
    "NegotiationBudget",
    "Party",
    "LocalParty",
    "NegotiationResult",
    "Negotiation",
    "select_offer",
    "buyer_position",
    "seller_position",
]

Role = Literal["buyer", "seller"]

# The audit action a negotiation outcome is recorded under.
NEGOTIATION_ACTION = "negotiation"
CONTRACT_ACTION = "contract_signed"


class IssuePreference(BaseModel):
    """A party's preference over one numeric issue.

    ``ideal`` is the value the party most wants; ``reserve`` is the worst value it
    will still accept. The direction is implied: if ``ideal > reserve`` the party
    prefers higher values (a seller's price), if ``ideal < reserve`` it prefers
    lower (a buyer's price). ``weight`` is the issue's relative importance in the
    party's utility.
    """

    name: str
    ideal: float
    reserve: float
    weight: float = 1.0

    def utility(self, value: float) -> float:
        """This party's utility for ``value`` — ``1`` at the ideal, ``0`` at the
        reservation, and **negative beyond the reservation**.

        The upper bound is capped at the ideal (``1``); the lower bound is left
        unclamped on purpose, so a value worse than the party's walk-away point is
        genuinely penalized (utility ``< 0``) rather than looking merely "at
        reservation". That is what lets a bargain with no overlapping acceptable
        region terminate in a clean no-deal instead of a false agreement.
        """
        span = self.ideal - self.reserve
        if abs(span) < 1e-12:
            # A degenerate issue (ideal == reserve): satisfied iff exactly met,
            # but treat any value as fully acceptable to avoid dividing by zero.
            return 1.0
        u = (value - self.reserve) / span
        return min(1.0, u)

    def value_at(self, level: float) -> float:
        """The issue value whose utility-to-this-party equals ``level``."""
        level = min(1.0, max(0.0, level))
        return self.reserve + (self.ideal - self.reserve) * level


class NegotiationPosition(BaseModel):
    """A party's private stance: per-issue preferences and a concession curve.

    ``concession`` is the time-dependent tactic's exponent ``e``: ``e < 1`` is
    tough (concede late, "boulware"), ``e == 1`` is linear, ``e > 1`` is generous
    (concede early, "conceder"). ``min_utility`` is the party's floor — it will
    never offer or accept below this utility-to-itself, so the reservation is a
    hard walk-away.
    """

    role: Role
    issues: list[IssuePreference] = Field(default_factory=list)
    concession: float = 1.0
    min_utility: float = 0.0

    def validate_coherent(self) -> NegotiationPosition:
        """Raise :class:`NegotiationError` unless the position is coherent."""
        if self.concession <= 0.0:
            raise NegotiationError(
                f"concession exponent must be positive; got {self.concession}"
            )
        if not 0.0 <= self.min_utility <= 1.0:
            raise NegotiationError(
                f"min_utility must be in [0, 1]; got {self.min_utility}"
            )
        if not self.issues:
            raise NegotiationError("a negotiation position needs at least one issue")
        total_weight = sum(max(0.0, i.weight) for i in self.issues)
        if total_weight <= 0.0:
            raise NegotiationError("issue weights must sum to a positive value")
        return self

    def _weights(self) -> list[float]:
        return [max(0.0, i.weight) for i in self.issues]

    def utility(self, terms: ContractTerms) -> float:
        """The party's overall utility for ``terms``, weighted across issues."""
        weights = self._weights()
        total = sum(weights) or 1.0
        score = 0.0
        for issue, w in zip(self.issues, weights, strict=True):
            score += w * issue.utility(_term_value(terms, issue.name))
        return score / total

    def concession_level(self, round_index: int, max_rounds: int) -> float:
        """Target utility-to-self at ``round_index`` of ``max_rounds`` (Faratin).

        Starts at ``1.0`` (the ideal) and concedes monotonically toward
        ``min_utility`` as the deadline approaches, shaped by ``concession``.
        """
        if max_rounds <= 0:
            return self.min_utility
        t = min(1.0, max(0.0, round_index / max_rounds))
        concede = t ** (1.0 / self.concession)
        level = 1.0 - (1.0 - self.min_utility) * concede
        return min(1.0, max(self.min_utility, level))

    def offer_terms(self, scope: str, round_index: int, max_rounds: int) -> ContractTerms:
        """The terms this party proposes at ``round_index`` (its concession point)."""
        level = self.concession_level(round_index, max_rounds)
        values = {issue.name: issue.value_at(level) for issue in self.issues}
        return _terms_from_values(scope, values)


def _term_value(terms: ContractTerms, name: str) -> float:
    return float(getattr(terms, name, 0.0))


def _terms_from_values(scope: str, values: dict[str, float]) -> ContractTerms:
    return ContractTerms(
        scope=scope,
        price_usd=round(values.get("price_usd", 0.0), 9),
        sla_seconds=round(values.get("sla_seconds", 0.0), 9),
        quality_floor=round(values.get("quality_floor", 0.0), 9),
    )


class Offer(BaseModel):
    """One move in a negotiation: a proposal, an acceptance, or a walk-away.

    An offer with ``accept=True`` accepts the opponent's *previous* terms (carried
    here for the record); ``walk_away=True`` ends the bargain with no deal. A plain
    offer proposes ``terms`` for the named ``scope`` at ``round_index``.
    """

    party: str
    role: Role
    terms: ContractTerms
    round_index: int = 0
    accept: bool = False
    walk_away: bool = False
    utility: float | None = None  # the proposer's utility for these terms (record)
    rationale: str = ""

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange over the A2A fabric."""
        return {
            "party": self.party,
            "role": self.role,
            "terms": self.terms.model_dump(mode="json"),
            "round_index": self.round_index,
            "accept": self.accept,
            "walk_away": self.walk_away,
            "utility": self.utility,
            "rationale": self.rationale,
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> Offer:
        return cls(
            party=data.get("party", ""),
            role=data.get("role", "seller"),
            terms=ContractTerms.model_validate(data.get("terms") or {}),
            round_index=int(data.get("round_index", 0)),
            accept=bool(data.get("accept", False)),
            walk_away=bool(data.get("walk_away", False)),
            utility=data.get("utility"),
            rationale=data.get("rationale", ""),
        )


class NegotiationBudget(BaseModel):
    """The guaranteed-termination budget for a negotiation.

    ``max_rounds`` bounds the number of offer exchanges (the deadline ``T`` the
    concession curve is shaped against); ``deadline_s`` is an optional wall-clock
    cap that returns a partial result the moment it is hit. One of them always
    fires, so a negotiation cannot run forever.
    """

    max_rounds: int = 8
    deadline_s: float | None = None

    def validate_coherent(self) -> NegotiationBudget:
        if self.max_rounds <= 0:
            raise NegotiationError(f"max_rounds must be positive; got {self.max_rounds}")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise NegotiationError(f"deadline_s must be positive; got {self.deadline_s}")
        return self


@runtime_checkable
class Party(Protocol):
    """A negotiating agent: opens with an offer and responds to the opponent's.

    The two methods are async so a remote A2A counterparty satisfies the same
    contract as a local, deterministic one. ``respond`` returns its counter-offer,
    an acceptance, or a walk-away.
    """

    member_id: str
    role: Role

    async def open(self, scope: str, budget: NegotiationBudget) -> Offer: ...

    async def respond(
        self, scope: str, incoming: Offer, round_index: int, budget: NegotiationBudget
    ) -> Offer: ...


class LocalParty:
    """A deterministic, offline negotiating party driven by a position.

    Computes its offers from its :class:`NegotiationPosition`'s concession curve,
    and accepts the opponent's offer (``AC_next``) the moment it is at least as
    good as the party's own next offer — discounting the opponent's offer by the
    opponent's reputation weight when a ledger is supplied.
    """

    def __init__(
        self,
        member_id: str,
        position: NegotiationPosition,
        *,
        reputation: Any | None = None,
    ) -> None:
        self.member_id = member_id
        self.position = position.validate_coherent()
        self.role: Role = position.role
        self.reputation = reputation

    def _opponent_weight(self, opponent_id: str) -> float:
        if self.reputation is None or not opponent_id:
            return 1.0
        try:
            return float(self.reputation.weight(opponent_id))
        except Exception:  # noqa: BLE001 - a ledger miss should not break a bargain
            return 1.0

    def discounted_utility(self, incoming: Offer) -> float:
        """The party's utility for ``incoming`` after the reputation risk discount."""
        raw = self.position.utility(incoming.terms)
        return raw * self._opponent_weight(incoming.party)

    async def open(self, scope: str, budget: NegotiationBudget) -> Offer:
        terms = self.position.offer_terms(scope, 0, budget.max_rounds)
        return Offer(
            party=self.member_id,
            role=self.role,
            terms=terms,
            round_index=0,
            utility=round(self.position.utility(terms), 9),
        )

    async def respond(
        self, scope: str, incoming: Offer, round_index: int, budget: NegotiationBudget
    ) -> Offer:
        my_next = self.position.offer_terms(scope, round_index, budget.max_rounds)
        my_next_u = self.position.utility(my_next)
        incoming_u = self.discounted_utility(incoming)
        # AC_next: accept once the opponent's (reputation-discounted) offer is at
        # least as good as what we would counter with, provided it clears our floor.
        if incoming_u >= my_next_u and incoming_u >= self.position.min_utility:
            return Offer(
                party=self.member_id,
                role=self.role,
                terms=incoming.terms,
                round_index=round_index,
                accept=True,
                utility=round(incoming_u, 9),
                rationale="offer meets or beats our next concession",
            )
        # If even conceding to our reservation cannot beat the opponent's standing
        # discount, and we are out of rounds, walk away cleanly.
        if round_index >= budget.max_rounds and incoming_u < self.position.min_utility:
            return Offer(
                party=self.member_id,
                role=self.role,
                terms=my_next,
                round_index=round_index,
                walk_away=True,
                utility=round(my_next_u, 9),
                rationale="deadline reached below reservation",
            )
        return Offer(
            party=self.member_id,
            role=self.role,
            terms=my_next,
            round_index=round_index,
            utility=round(my_next_u, 9),
        )


class NegotiationResult(BaseModel):
    """The outcome of a bounded negotiation — a deal, or a partial no-deal.

    ``status`` is ``"agreement"`` when a contract was reached, ``"no_agreement"``
    when the round/deadline budget ran out, or ``"walk_away"`` when a party left.
    The partial result always carries the full ``offers`` trace and the last offer
    from each side, so a deadline outcome is inspectable, not a bare failure.
    """

    status: Literal["agreement", "no_agreement", "walk_away"]
    contract: Contract | None = None
    rounds: int = 0
    buyer: str = ""
    seller: str = ""
    offers: list[Offer] = Field(default_factory=list)
    deadline_hit: bool = False
    reason: str = ""

    @property
    def agreed(self) -> bool:
        """Whether a contract was reached."""
        return self.status == "agreement" and self.contract is not None

    @property
    def last_buyer_offer(self) -> Offer | None:
        return next((o for o in reversed(self.offers) if o.role == "buyer"), None)

    @property
    def last_seller_offer(self) -> Offer | None:
        return next((o for o in reversed(self.offers) if o.role == "seller"), None)


class Negotiation:
    """Drives a bounded alternating-offers bargain between a buyer and a seller.

    The engine is dumb orchestration: it alternates ``open`` / ``respond`` between
    the two :class:`Party` objects (each of which owns its strategy and reputation
    view), enforces the :class:`NegotiationBudget`, and on agreement mints a
    :class:`~vincio.negotiation.contract.Contract` — signed by both parties when a
    signer is given, and recorded on the audit chain and event bus. Termination is
    guaranteed by the budget; a deadline returns a partial
    :class:`NegotiationResult`.
    """

    def __init__(
        self,
        buyer: Party,
        seller: Party,
        *,
        budget: NegotiationBudget | None = None,
        signer: Any | None = None,
        audit: Any | None = None,
        events: Any | None = None,
        clock: Any | None = None,
    ) -> None:
        if buyer is None or seller is None:  # pragma: no cover - defensive
            raise NegotiationError("a negotiation needs both a buyer and a seller party")
        if getattr(buyer, "role", "buyer") != "buyer" or getattr(seller, "role", "seller") != "seller":
            raise NegotiationError("buyer must have role 'buyer' and seller role 'seller'")
        self.buyer = buyer
        self.seller = seller
        self.budget = (budget or NegotiationBudget()).validate_coherent()
        self.signer = signer
        self.audit = audit
        self.events = events
        self._clock = clock or time.monotonic

    async def run(self, scope: str) -> NegotiationResult:
        """Run the bargain to a deal or a bounded no-deal and return the result."""
        offers: list[Offer] = []
        start = self._clock()
        max_rounds = self.budget.max_rounds
        deadline = self.budget.deadline_s

        def timed_out() -> bool:
            return deadline is not None and (self._clock() - start) >= deadline

        # Round 0: the buyer opens.
        current = await self.buyer.open(scope, self.budget)
        offers.append(current)
        last_proposer: Party = self.buyer
        result: NegotiationResult | None = None

        for round_index in range(1, max_rounds + 1):
            if timed_out():
                result = self._finish_no_agreement(
                    offers, round_index - 1, deadline_hit=True, reason="wall-clock deadline reached"
                )
                break
            responder = self.seller if last_proposer is self.buyer else self.buyer
            response = await responder.respond(scope, current, round_index, self.budget)
            offers.append(response)
            if response.accept:
                result = self._finish_agreement(current, offers, round_index)
                break
            if response.walk_away:
                result = self._finish_walk_away(offers, round_index, responder)
                break
            current = response
            last_proposer = responder
        if result is None:
            result = self._finish_no_agreement(
                offers, max_rounds, deadline_hit=False, reason="round budget exhausted"
            )
        self._record(result)
        return result

    def _finish_agreement(
        self, accepted: Offer, offers: list[Offer], round_index: int
    ) -> NegotiationResult:
        contract = Contract(
            buyer=self.buyer.member_id,
            seller=self.seller.member_id,
            terms=accepted.terms,
            rounds=round_index,
            agreed_at=utcnow(),
        ).seal()
        if self.signer is not None:
            contract.sign(self.signer, party=self.buyer.member_id)
            contract.sign(self.signer, party=self.seller.member_id)
        return NegotiationResult(
            status="agreement",
            contract=contract,
            rounds=round_index,
            buyer=self.buyer.member_id,
            seller=self.seller.member_id,
            offers=offers,
            reason="offer accepted",
        )

    def _finish_no_agreement(
        self, offers: list[Offer], round_index: int, *, deadline_hit: bool, reason: str
    ) -> NegotiationResult:
        return NegotiationResult(
            status="no_agreement",
            rounds=round_index,
            buyer=self.buyer.member_id,
            seller=self.seller.member_id,
            offers=offers,
            deadline_hit=deadline_hit,
            reason=reason,
        )

    def _finish_walk_away(
        self, offers: list[Offer], round_index: int, responder: Party
    ) -> NegotiationResult:
        return NegotiationResult(
            status="walk_away",
            rounds=round_index,
            buyer=self.buyer.member_id,
            seller=self.seller.member_id,
            offers=offers,
            reason=f"{responder.member_id} walked away",
        )

    def _record(self, result: NegotiationResult) -> None:
        if self.audit is not None:
            self.audit.record(
                NEGOTIATION_ACTION,
                resource=f"{result.buyer}->{result.seller}",
                decision="agreement" if result.agreed else result.status,
                details={
                    "buyer": result.buyer,
                    "seller": result.seller,
                    "status": result.status,
                    "rounds": result.rounds,
                    "deadline_hit": result.deadline_hit,
                    "reason": result.reason,
                },
            )
            if result.contract is not None:
                entry = self.audit.record(
                    CONTRACT_ACTION,
                    resource=result.contract.id,
                    decision="signed",
                    details=result.contract.audit_details(),
                )
                result.contract.audit_id = getattr(entry, "id", None)
        if self.events is not None:
            try:
                self.events.emit("negotiation.completed", _result_event(result))
            except Exception:  # noqa: BLE001 - event delivery is best-effort
                pass


def _result_event(result: NegotiationResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "buyer": result.buyer,
        "seller": result.seller,
        "rounds": result.rounds,
        "contract_id": result.contract.id if result.contract else None,
    }


def select_offer(
    results: Iterable[NegotiationResult],
    buyer_position: NegotiationPosition,
    *,
    reputation: Any | None = None,
) -> NegotiationResult | None:
    """Pick the best deal among competing sellers by reputation-weighted utility.

    For each agreed result, scores the contract by the buyer's utility for its
    terms discounted by the seller's reputation weight (``[floor, 1]``), and
    returns the highest-scoring agreement — so a reliable seller wins a close
    price race, and an unreliable one is discounted without being singled out.
    Returns ``None`` when no result reached agreement.
    """
    best: NegotiationResult | None = None
    best_score = float("-inf")
    for result in results:
        if not result.agreed or result.contract is None:
            continue
        utility = buyer_position.utility(result.contract.terms)
        weight = 1.0
        if reputation is not None:
            try:
                weight = float(reputation.weight(result.seller))
            except Exception:  # noqa: BLE001
                weight = 1.0
        score = utility * weight
        if score > best_score:
            best_score = score
            best = result
    return best


def buyer_position(
    *,
    max_price_usd: float,
    ideal_price_usd: float = 0.0,
    max_sla_seconds: float,
    ideal_sla_seconds: float = 0.0,
    min_quality: float = 0.0,
    ideal_quality: float = 1.0,
    weights: Sequence[float] | None = None,
    concession: float = 1.0,
    min_utility: float = 0.0,
) -> NegotiationPosition:
    """Build a buyer position: wants low price, fast SLA, high quality.

    ``max_*`` are the buyer's reservation (walk-away) values; ``ideal_*`` are what
    it most wants. ``weights`` orders ``(price, sla, quality)``.
    """
    w = list(weights or (1.0, 1.0, 1.0))
    return NegotiationPosition(
        role="buyer",
        issues=[
            IssuePreference(name="price_usd", ideal=ideal_price_usd, reserve=max_price_usd, weight=w[0]),
            IssuePreference(name="sla_seconds", ideal=ideal_sla_seconds, reserve=max_sla_seconds, weight=w[1]),
            IssuePreference(name="quality_floor", ideal=ideal_quality, reserve=min_quality, weight=w[2]),
        ],
        concession=concession,
        min_utility=min_utility,
    )


def seller_position(
    *,
    min_price_usd: float,
    ideal_price_usd: float,
    min_sla_seconds: float = 0.0,
    ideal_sla_seconds: float = 0.0,
    max_quality: float = 1.0,
    ideal_quality: float = 0.0,
    weights: Sequence[float] | None = None,
    concession: float = 1.0,
    min_utility: float = 0.0,
) -> NegotiationPosition:
    """Build a seller position: wants high price, a loose SLA, a low quality floor.

    ``min_*`` / ``max_*`` are the seller's reservation (walk-away) values;
    ``ideal_*`` are what it most wants. ``weights`` orders ``(price, sla, quality)``.
    """
    w = list(weights or (1.0, 1.0, 1.0))
    return NegotiationPosition(
        role="seller",
        issues=[
            IssuePreference(name="price_usd", ideal=ideal_price_usd, reserve=min_price_usd, weight=w[0]),
            IssuePreference(name="sla_seconds", ideal=ideal_sla_seconds, reserve=min_sla_seconds, weight=w[1]),
            IssuePreference(name="quality_floor", ideal=ideal_quality, reserve=max_quality, weight=w[2]),
        ],
        concession=concession,
        min_utility=min_utility,
    )
