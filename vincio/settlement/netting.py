"""Cross-org settlement netting & multilateral clearing.

With bilateral settlements signed, reconciled, and reputation-closing, the next
reach is **netting** them: folding a fleet's many bilateral
:class:`~vincio.settlement.book.SettlementBook` balances into a single minimal set
of net obligations, so an org that is both a buyer and a seller across a web of
contracts closes its books once. It is a library-side clearing *calculation* — the
cross-org analogue of the cost report's roll-up — never a hosted clearing house or
a payment rail.

* **Multilateral netting.** Each settled contract is a directed obligation — the
  buyer owes the seller the agreed price for the scope (the payable cap; a breach
  is surfaced by the settlement's own status and the reputation loop, it does not
  alter what is contractually owed). :func:`net_settlements` folds those directed
  obligations across the whole fleet: an org's many positions against many
  counterparties collapse to one :class:`BilateralNet` figure per counterparty, and
  the :class:`NetPosition` per org (what it is owed minus what it owes) is cleared
  to the **minimal set** of :class:`NetObligation` transfers — at most ``N - 1`` of
  them, deterministically, net-debtors paying net-creditors.
* **Still offline-verifiable.** The resulting :class:`NettingSet` is content-bound
  the way a :class:`~vincio.settlement.record.SettlementRecord` is: a netting hash
  binds the fleet, the exact source records read (by their reconciliation hashes),
  the net positions, and the cleared obligations, and the same
  :class:`~vincio.security.audit.ChainSigner` co-signs *that* hash.
  :meth:`NettingSet.verify` recomputes it from the bytes alone — the hash matches,
  the positions balance to zero, and the cleared obligations reproduce every org's
  net position — so a cleared balance verifies without a central ledger.
* **Same discipline.** Netting reads only the existing signed, hash-chained books;
  it asserts nothing it cannot recompute. A tampered source record (its
  reconciliation hash no longer recomputes) is refused outright, and two books that
  disagree on the *same* contract are pinpointed as a :class:`NettingDispute` —
  excluded from the clearing, never silently absorbed.

:func:`net_books` nets a fleet of books straight from their records;
:meth:`~vincio.settlement.book.SettlementBook.net` nets one org's own book into its
positions against each counterparty. Everything is dependency-free, deterministic,
and offline.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .record import SettlementRecord, SettlementSignature, _resolve_verifier

if TYPE_CHECKING:
    from ..security.audit import ChainSigner
    from .book import SettlementBook

__all__ = [
    "GrossObligation",
    "BilateralNet",
    "NetPosition",
    "NetObligation",
    "NettingDispute",
    "NettingVerification",
    "NettingSet",
    "net_settlements",
    "net_books",
]

# The audit action a netting set is recorded under.
NETTING_ACTION = "netting"

_TOLERANCE = 1e-6


def _r(value: float) -> float:
    """Round a money figure so float drift never breaks the clearing or its hash."""
    return round(float(value), 9)


class GrossObligation(BaseModel):
    """A directed gross obligation: ``debtor`` owes ``creditor`` ``amount_usd``.

    The fleet's settled contracts aggregated per ordered ``(debtor, creditor)``
    pair before any netting — the buyer-owes-seller payables, summed across however
    many contracts ran between the two. ``contracts`` lists the contract ids that
    contributed, so a gross figure is traceable back to its settlements.
    """

    debtor: str
    creditor: str
    amount_usd: float = 0.0
    settlements: int = 0
    contracts: list[str] = Field(default_factory=list)


class BilateralNet(BaseModel):
    """The net figure between two parties — one per counterparty pair.

    The two directed gross obligations between ``party_low`` and ``party_high``
    (named in sorted order so the pair is canonical) collapsed to a single signed
    figure: ``net_debtor`` owes ``net_creditor`` ``net_amount_usd`` once the
    opposing flows cancel. When the two directions are equal the pair nets to zero
    and ``net_debtor`` / ``net_creditor`` are empty.
    """

    party_low: str
    party_high: str
    low_owes_high: float = 0.0
    high_owes_low: float = 0.0
    net_debtor: str = ""
    net_creditor: str = ""
    net_amount_usd: float = 0.0


class NetPosition(BaseModel):
    """An org's net position across the whole fleet.

    ``owed_usd`` is everything this party owes (its payables out), ``due_usd`` is
    everything owed to it (its receivables in), and ``net_usd`` is ``due - owed`` —
    positive for a net creditor, negative for a net debtor. The fleet's net
    positions sum to zero (every payable is someone's receivable); that
    conservation is what :meth:`NettingSet.verify` recomputes.
    """

    party: str
    owed_usd: float = 0.0
    due_usd: float = 0.0
    net_usd: float = 0.0

    @property
    def is_creditor(self) -> bool:
        """The fleet owes this party more than it owes the fleet."""
        return self.net_usd > _TOLERANCE

    @property
    def is_debtor(self) -> bool:
        """This party owes the fleet more than the fleet owes it."""
        return self.net_usd < -_TOLERANCE


class NetObligation(BaseModel):
    """One cleared transfer in the minimal set: ``debtor`` pays ``creditor``.

    The output of multilateral clearing — a single, directed, positive transfer
    that, together with the others, settles every org's net position. The set has
    at most ``N - 1`` of these for ``N`` parties with a non-zero position.
    """

    debtor: str
    creditor: str
    amount_usd: float = 0.0


class NettingDispute(BaseModel):
    """Two books disagree on the same contract — pinpointed, not absorbed.

    Produced when the fleet carries more than one settlement record for a contract
    and they do not share a reconciliation hash (their economic facts differ). The
    contract is **excluded** from the clearing and reported here with the conflicting
    hashes, so a disagreement is a named dispute, never silently netted away.
    """

    contract_id: str
    parties: list[str] = Field(default_factory=list)
    hashes: list[str] = Field(default_factory=list)
    reason: str = "settlement records disagree on the same contract"


class NettingVerification(BaseModel):
    """The (non-raising) outcome of verifying a netting set offline."""

    valid: bool
    hash_ok: bool
    positions_balanced: bool
    conserves: bool
    signatures_ok: bool
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class NettingSet(BaseModel):
    """A content-bound, offline-verifiable multilateral clearing of a fleet's books.

    Produced by :func:`net_settlements` / :func:`net_books` (or
    :meth:`~vincio.settlement.book.SettlementBook.net`) from the signed settlement
    records of one or more orgs. It carries the gross obligations, the per-pair
    :class:`BilateralNet`, each org's :class:`NetPosition`, and the minimal cleared
    set of :class:`NetObligation` transfers, plus any :class:`NettingDispute` it
    refused to net.

    The netting hash (:meth:`compute_hash`) binds the economic facts — the fleet,
    the exact source records read (by their reconciliation hashes), the positions,
    the cleared obligations, and the disputes — so :meth:`verify` recomputes the
    whole clearing from the bytes alone: the hash matches, the positions sum to
    zero, and the cleared transfers reproduce every position. A
    :class:`~vincio.security.audit.ChainSigner` co-signs that hash, exactly as a
    settlement record's parties co-sign theirs.
    """

    id: str = Field(default_factory=lambda: new_id("netting"))
    owner: str = ""
    fleet: list[str] = Field(default_factory=list)
    settlements: int = 0
    source_hashes: list[str] = Field(default_factory=list)

    gross: list[GrossObligation] = Field(default_factory=list)
    bilateral: list[BilateralNet] = Field(default_factory=list)
    positions: list[NetPosition] = Field(default_factory=list)
    obligations: list[NetObligation] = Field(default_factory=list)
    disputes: list[NettingDispute] = Field(default_factory=list)

    total_gross_usd: float = 0.0
    total_cleared_usd: float = 0.0

    netted_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived figures ----------------------------------------------------

    @property
    def gross_edges(self) -> int:
        """The number of directed gross obligations before clearing."""
        return len(self.gross)

    @property
    def cleared_transfers(self) -> int:
        """The number of transfers in the minimal cleared set."""
        return len(self.obligations)

    @property
    def reduction(self) -> int:
        """How many obligations clearing eliminated (gross edges minus transfers)."""
        return self.gross_edges - self.cleared_transfers

    @property
    def clean(self) -> bool:
        """No disputes — every contract in the fleet could be netted."""
        return not self.disputes

    def position(self, party: str) -> NetPosition | None:
        """This party's net position, or ``None`` if it is not in the fleet."""
        return next((p for p in self.positions if p.party == party), None)

    def obligations_for(self, party: str) -> list[NetObligation]:
        """The cleared transfers this party pays or receives."""
        return [o for o in self.obligations if party in (o.debtor, o.creditor)]

    # -- hashing & signing --------------------------------------------------

    def netting_facts(self) -> dict[str, Any]:
        """The economic facts the netting hash binds (and a signer signs)."""
        return {
            "fleet": sorted(self.fleet),
            "source_hashes": sorted(self.source_hashes),
            "positions": [
                {"party": p.party, "net_usd": _r(p.net_usd)}
                for p in sorted(self.positions, key=lambda p: p.party)
            ],
            "obligations": [
                {
                    "debtor": o.debtor,
                    "creditor": o.creditor,
                    "amount_usd": _r(o.amount_usd),
                }
                for o in sorted(
                    self.obligations, key=lambda o: (o.debtor, o.creditor)
                )
            ],
            "disputes": sorted(d.contract_id for d in self.disputes),
        }

    def compute_hash(self) -> str:
        """The netting hash binding the cleared economic facts (position-independent).

        Deliberately excludes the set id and the timestamp: those are local
        metadata, not economic terms. Two clearers that read the same source records
        therefore compute the *same* hash and can co-sign it, while a tampered figure
        changes it.
        """
        return stable_hash(self.netting_facts(), length=32)

    def seal(self) -> NettingSet:
        """Stamp the netting hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> NettingSet:
        """Add ``party``'s signature over the netting hash (sealing first).

        A netting set is signed by whoever computed the clearing — a coordinator or
        any fleet member that independently recomputes it. Re-signing for the same
        party replaces its prior signature, so a set cannot accumulate stale
        signatures for one identity.
        """
        if not self.content_hash:
            self.seal()
        sig = SettlementSignature(
            party=party,
            signature=signer.sign(self.content_hash),
            key_id=getattr(signer, "key_id", ""),
        )
        self.signatures = [s for s in self.signatures if s.party != party]
        self.signatures.append(sig)
        return self

    # -- verification -------------------------------------------------------

    def _positions_balanced(self) -> bool:
        """Every payable is a receivable: the net positions sum to zero."""
        return abs(sum(p.net_usd for p in self.positions)) <= _TOLERANCE

    def _conserves(self) -> bool:
        """The cleared transfers reproduce every org's net position exactly."""
        known = {p.party for p in self.positions}
        flow: dict[str, float] = {p.party: 0.0 for p in self.positions}
        for o in self.obligations:
            if o.amount_usd < -_TOLERANCE:
                return False  # a transfer must be a positive, directed payment
            if o.debtor not in known or o.creditor not in known:
                return False  # a transfer to a party with no net position is unconserved
            flow[o.creditor] += o.amount_usd
            flow[o.debtor] -= o.amount_usd
        return all(abs(flow[p.party] - p.net_usd) <= _TOLERANCE for p in self.positions)

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> NettingVerification:
        """Verify the clearing offline: hash, conservation, and signatures.

        Recomputes the netting hash from the stored fields, checks that the net
        positions balance to zero and that the cleared obligations reproduce every
        position, and — with a ``verifier`` — that each signature checks. ``require``
        names parties that must have a verified signature (none by default: a netting
        set's authenticity is that anyone can recompute it and co-sign the same
        hash). A tampered figure breaks the hash and, almost always, conservation too.
        """
        expected = self.compute_hash()
        hash_ok = bool(self.content_hash) and self.content_hash == expected
        positions_balanced = self._positions_balanced()
        conserves = self._conserves()
        if not hash_ok:
            return NettingVerification(
                valid=False,
                hash_ok=False,
                positions_balanced=positions_balanced,
                conserves=conserves,
                signatures_ok=False,
                reason="content hash mismatch",
            )
        verified: list[str] = []
        signatures_ok = True
        for sig in self.signatures:
            if verifier is not None:
                if verifier.verify(self.content_hash, sig.signature):
                    verified.append(sig.party)
                else:
                    signatures_ok = False
            else:
                verified.append(sig.party)
        required = require or []
        missing = [p for p in required if p not in verified]
        if missing:
            signatures_ok = False
        valid = hash_ok and positions_balanced and conserves and signatures_ok
        reason = None
        if not positions_balanced:
            reason = "net positions do not balance to zero"
        elif not conserves:
            reason = "cleared obligations do not reproduce the net positions"
        elif not signatures_ok:
            reason = (
                f"missing/invalid signatures for {missing}" if missing else "signature mismatch"
            )
        return NettingVerification(
            valid=valid,
            hash_ok=hash_ok,
            positions_balanced=positions_balanced,
            conserves=conserves,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> NettingSet:
        """Verify and raise :class:`SettlementError` if the set is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"netting set {self.id} failed verification: {result.reason}",
                details={"netting_id": self.id, "reason": result.reason},
            )
        return self

    def require_clean(self) -> NettingSet:
        """Raise :class:`SettlementError` if any contract was disputed.

        The strict-mode counterpart to inspecting :attr:`disputes`: a fleet whose
        books disagree on a contract cannot be cleared without resolving the dispute
        first, and this pinpoints exactly which contracts and which hashes conflict.
        """
        if self.disputes:
            ids = [d.contract_id for d in self.disputes]
            raise SettlementError(
                f"netting set {self.id} has {len(ids)} disputed contract(s): {ids}",
                details={"netting_id": self.id, "disputes": ids},
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the clearing for the audit chain."""
        return to_jsonable(
            {
                "netting_id": self.id,
                "owner": self.owner,
                "fleet": self.fleet,
                "settlements": self.settlements,
                "gross_edges": self.gross_edges,
                "cleared_transfers": self.cleared_transfers,
                "total_gross_usd": self.total_gross_usd,
                "total_cleared_usd": self.total_cleared_usd,
                "disputes": [d.contract_id for d in self.disputes],
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> NettingSet:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the cleared obligations and the reduction they achieved."""
        title = f"Netting set ({self.owner})" if self.owner else "Netting set"
        print(
            f"{title}: {self.settlements} settlements, {self.gross_edges} gross "
            f"obligations → {self.cleared_transfers} cleared transfers"
        )
        for o in self.obligations:
            print(f"  {o.debtor} → {o.creditor}: ${o.amount_usd:.4f}")
        for d in self.disputes:
            print(f"  ! dispute on contract {d.contract_id}")


# -- the netting calculation --------------------------------------------------


def _representatives(
    records: list[SettlementRecord],
    *,
    verifier: ChainSigner | None,
) -> tuple[list[SettlementRecord], list[NettingDispute]]:
    """One canonical record per contract; disagreements pinpointed as disputes.

    The same bilateral settlement appears in *both* parties' books co-signing one
    reconciliation hash, so the fleet is deduplicated by contract: records that
    share a hash collapse to one, and records that do not are a dispute (excluded).
    A record whose reconciliation hash no longer recomputes — a tampered economic
    figure — is refused outright; with a ``verifier`` a forged signature is too.
    """
    by_contract: dict[str, list[SettlementRecord]] = {}
    for record in records:
        if record.content_hash != record.compute_hash():
            raise SettlementError(
                f"settlement {record.id} for contract {record.contract_id!r} is "
                f"tampered (reconciliation hash does not recompute); refusing to net it",
                details={"settlement_id": record.id, "contract_id": record.contract_id},
            )
        if verifier is not None and record.signatures:
            check = record.verify(verifier, require=[])
            if not check.signatures_ok:
                raise SettlementError(
                    f"settlement {record.id} for contract {record.contract_id!r} has an "
                    f"invalid signature; refusing to net it",
                    details={"settlement_id": record.id, "contract_id": record.contract_id},
                )
        by_contract.setdefault(record.contract_id, []).append(record)

    reps: list[SettlementRecord] = []
    disputes: list[NettingDispute] = []
    for contract_id in sorted(by_contract):
        group = by_contract[contract_id]
        hashes = sorted({r.content_hash for r in group})
        if len(hashes) > 1:
            parties = sorted({p for r in group for p in (r.buyer, r.seller)})
            disputes.append(
                NettingDispute(contract_id=contract_id, parties=parties, hashes=hashes)
            )
            continue
        reps.append(group[0])
    return reps, disputes


def _gross_obligations(reps: list[SettlementRecord]) -> list[GrossObligation]:
    """Aggregate the buyer-owes-seller payables per directed pair, in sorted order."""
    agg: dict[tuple[str, str], GrossObligation] = {}
    for record in reps:
        amount = _r(record.amount_owed_usd)
        if amount <= 0 or record.buyer == record.seller:
            continue
        key = (record.buyer, record.seller)
        gross = agg.get(key)
        if gross is None:
            gross = GrossObligation(debtor=record.buyer, creditor=record.seller)
            agg[key] = gross
        gross.amount_usd = _r(gross.amount_usd + amount)
        gross.settlements += 1
        gross.contracts.append(record.contract_id)
    for gross in agg.values():
        gross.contracts.sort()
    return [agg[k] for k in sorted(agg)]


def _bilateral_nets(gross: list[GrossObligation]) -> list[BilateralNet]:
    """Collapse the two directed flows between each pair into one net figure."""
    directed = {(g.debtor, g.creditor): g.amount_usd for g in gross}
    pairs = sorted({tuple(sorted((g.debtor, g.creditor))) for g in gross})
    nets: list[BilateralNet] = []
    for low, high in pairs:
        low_owes_high = _r(directed.get((low, high), 0.0))
        high_owes_low = _r(directed.get((high, low), 0.0))
        diff = _r(low_owes_high - high_owes_low)
        if diff > _TOLERANCE:
            debtor, creditor, amount = low, high, diff
        elif diff < -_TOLERANCE:
            debtor, creditor, amount = high, low, _r(-diff)
        else:
            debtor, creditor, amount = "", "", 0.0
        nets.append(
            BilateralNet(
                party_low=low,
                party_high=high,
                low_owes_high=low_owes_high,
                high_owes_low=high_owes_low,
                net_debtor=debtor,
                net_creditor=creditor,
                net_amount_usd=amount,
            )
        )
    return nets


def _net_positions(
    gross: list[GrossObligation], fleet: list[str]
) -> list[NetPosition]:
    """Each org's payables out, receivables in, and the resulting net position."""
    owed: dict[str, float] = {p: 0.0 for p in fleet}
    due: dict[str, float] = {p: 0.0 for p in fleet}
    for g in gross:
        owed[g.debtor] = owed.get(g.debtor, 0.0) + g.amount_usd
        due[g.creditor] = due.get(g.creditor, 0.0) + g.amount_usd
    positions: list[NetPosition] = []
    for party in fleet:
        o = _r(owed.get(party, 0.0))
        d = _r(due.get(party, 0.0))
        positions.append(
            NetPosition(party=party, owed_usd=o, due_usd=d, net_usd=_r(d - o))
        )
    return positions


def _clear(positions: list[NetPosition]) -> list[NetObligation]:
    """Multilateral clearing: the minimal set of debtor→creditor transfers.

    A deterministic greedy debt simplification — repeatedly match the largest net
    debtor with the largest net creditor and transfer the smaller of the two
    outstanding amounts — which settles every position in at most ``N - 1`` transfers
    (one fewer than a naive bilateral payout) and is conserved by construction. Ties
    are broken by party name so the cleared set is reproducible byte-for-byte.
    """
    debtors = sorted(
        ((p.party, -p.net_usd) for p in positions if p.net_usd < -_TOLERANCE),
        key=lambda x: (-x[1], x[0]),
    )
    creditors = sorted(
        ((p.party, p.net_usd) for p in positions if p.net_usd > _TOLERANCE),
        key=lambda x: (-x[1], x[0]),
    )
    debt = {party: amount for party, amount in debtors}
    credit = {party: amount for party, amount in creditors}
    # Each side pairs the current-amounts dict with a lazy max-heap: an entry is
    # (-amount, party, amount), and heap order on (-amount, party) is exactly the
    # min(...) selection key the greedy used, so the transfer sequence — and the
    # content-bound cleared set — is byte-for-byte unchanged. A partial fill
    # re-pushes the party at its new amount; the superseded entry stays behind
    # and is skipped as stale on pop (its amount no longer matches the dict).
    debt_heap = [(-amount, party, amount) for party, amount in debtors]
    credit_heap = [(-amount, party, amount) for party, amount in creditors]
    heapq.heapify(debt_heap)
    heapq.heapify(credit_heap)

    def _pop_live(
        heap: list[tuple[float, str, float]], current: dict[str, float]
    ) -> str:
        while True:
            _, party, amount = heapq.heappop(heap)
            if current.get(party) == amount:
                return party

    obligations: list[NetObligation] = []
    # Re-pick the extremes each round so the transfer count stays minimal.
    while debt and credit:
        d_party = _pop_live(debt_heap, debt)
        c_party = _pop_live(credit_heap, credit)
        transfer = _r(min(debt[d_party], credit[c_party]))
        if transfer <= _TOLERANCE:
            break
        obligations.append(
            NetObligation(debtor=d_party, creditor=c_party, amount_usd=transfer)
        )
        debt[d_party] = _r(debt[d_party] - transfer)
        credit[c_party] = _r(credit[c_party] - transfer)
        if debt[d_party] <= _TOLERANCE:
            del debt[d_party]
        else:
            heapq.heappush(debt_heap, (-debt[d_party], d_party, debt[d_party]))
        if credit[c_party] <= _TOLERANCE:
            del credit[c_party]
        else:
            heapq.heappush(credit_heap, (-credit[c_party], c_party, credit[c_party]))
    obligations.sort(key=lambda o: (o.debtor, o.creditor))
    return obligations


def net_settlements(
    records: Iterable[SettlementRecord],
    *,
    owner: str = "",
    fleet: list[str] | None = None,
    verifier: ChainSigner | None = None,
    verify_with: ChainSigner | None = None,
) -> NettingSet:
    """Fold a fleet's settled contracts into a minimal cleared set of obligations.

    Reads the signed :class:`~vincio.settlement.record.SettlementRecord`\\ s of one or
    more orgs, deduplicates the same bilateral settlement seen from both sides,
    pinpoints any contract two books disagree on as a :class:`NettingDispute`
    (excluded from the clearing), aggregates the buyer-owes-seller payables into
    directed gross obligations, computes each org's :class:`NetPosition`, and clears
    them to the minimal set of :class:`NetObligation` transfers. A tampered source
    record is refused; with ``verifier`` a forged signature is too. The returned
    :class:`NettingSet` is sealed but unsigned — sign it with the clearer's key.

    ``fleet`` overrides the party set (defaults to every party seen in an included
    obligation); ``owner`` labels the set with the clearing org. ``verify_with`` is
    a deprecated alias for ``verifier`` (since 7.5, removed in 8.0).
    """
    verifier = _resolve_verifier(verifier, verify_with, "net_settlements")
    record_list = list(records)
    reps, disputes = _representatives(record_list, verifier=verifier)
    gross = _gross_obligations(reps)
    parties = (
        sorted(fleet)
        if fleet is not None
        else sorted({p for g in gross for p in (g.debtor, g.creditor)})
    )
    positions = _net_positions(gross, parties)
    bilateral = _bilateral_nets(gross)
    obligations = _clear(positions)
    source_hashes = sorted({r.content_hash for r in reps})
    netting = NettingSet(
        owner=owner,
        fleet=parties,
        settlements=len(reps),
        source_hashes=source_hashes,
        gross=gross,
        bilateral=bilateral,
        positions=positions,
        obligations=obligations,
        disputes=disputes,
        total_gross_usd=_r(sum(g.amount_usd for g in gross)),
        total_cleared_usd=_r(sum(o.amount_usd for o in obligations)),
    )
    return netting.seal()


def net_books(
    books: Iterable[SettlementBook],
    *,
    owner: str = "",
    verifier: ChainSigner | None = None,
    require_intact: bool = False,
    verify_with: ChainSigner | None = None,
) -> NettingSet:
    """Net a fleet of :class:`~vincio.settlement.book.SettlementBook`\\ s into one set.

    The convenience over :func:`net_settlements` for whole books: gathers every
    record across the fleet and clears them. With ``require_intact`` each book's
    hash chain is verified before its records are read (``verifier`` checks the
    signatures too), so the clearing is built only on intact, signed ledgers.
    ``verify_with`` is a deprecated alias for ``verifier`` (since 7.5, removed
    in 8.0).
    """
    verifier = _resolve_verifier(verifier, verify_with, "net_books")
    book_list = list(books)
    records: list[SettlementRecord] = []
    for book in book_list:
        if require_intact:
            book.require_intact(verifier)
        records.extend(book.records)
    return net_settlements(records, owner=owner, verifier=verifier)
