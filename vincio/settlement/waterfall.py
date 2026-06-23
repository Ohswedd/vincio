"""Cross-org insolvency resolution & liability seniority waterfall.

A :class:`~vincio.settlement.solvency.SolvencyProof` surfaces an
:class:`~vincio.settlement.solvency.InsolvencyBreach` when a counterparty's proven
liabilities exceed its proven reserves, and :func:`~vincio.settlement.solvency.check_history_consistency`
pinpoints a debt that silently vanished — but when the reserves genuinely *cannot* cover every
obligation, nothing yet says **which** creditors the available capital pays, and in what order.
Today an insolvency is *flagged*, not *resolved*: every creditor is left to assume it is made
whole. The rehypothecation guard already apportions a scarce stake across beneficiaries
*pari passu* (:func:`~vincio.settlement.rehypothecation.guard_collateral`); the liability side
needs the same, plus the **seniority** real obligations carry. This module is that reach: it
distributes the proven reserves across the attested liabilities by seniority then pari-passu
within a tranche, into a bounded, offline-verifiable resolution of who-gets-what.

* **Signed seniority ordering.** A :class:`SenioritySchedule` (:func:`build_seniority_schedule`)
  ranks a poster's obligations into priority tranches — rank ``0`` most senior — and is
  content-bound and signed with the same :class:`~vincio.security.audit.ChainSigner` a contract
  uses, by the counterparty itself or by its creditors (an inter-creditor agreement). So the
  order capital is paid in is itself an auditable, non-repudiable artifact, not an after-the-fact
  assertion: :meth:`SenioritySchedule.verify` recomputes the content hash and refuses a re-ordered
  or malformed schedule (a creditor in two tranches, a duplicate rank) from the bytes alone.
* **Insolvency waterfall.** :func:`resolve_insolvency` folds a proven
  :class:`~vincio.settlement.custody.CustodyAttestation` against a proven
  :class:`~vincio.settlement.solvency.LiabilityAttestation` (reusing :func:`~vincio.settlement.solvency.prove_solvency`
  for every tamper/forgery/wrong-poster refusal) and distributes the proven reserves across the
  attested obligations **by seniority, then pari-passu within a tranche** — a senior tranche is
  paid in full before any capital reaches a junior one, and a partly-funded tranche splits what is
  left proportionally to each claim. The result is a content-bound :class:`InsolvencyResolution`
  pinpointing each creditor's bounded :class:`CreditorRecovery` (what it recovers and the
  shortfall it bears), so an insolvency is *resolved* into who-gets-what rather than merely
  flagged. With no schedule the whole liability set is one tranche — pure pari-passu, exactly the
  apportionment the rehypothecation guard already performs.
* **Auditable & offline.** The schedule and the waterfall read only signed, content-bound
  artifacts (the existing :class:`~vincio.settlement.solvency.LiabilityAttestation` and
  :class:`~vincio.settlement.custody.CustodyAttestation`). :meth:`InsolvencyResolution.verify`
  re-derives the *entire* distribution from the recorded per-creditor claims, ranks, and reserves
  — so an over-stated recovery, a re-ordered tranche, or a junior creditor paid ahead of a senior
  one is caught from the bytes alone, even after re-sealing — and binds the seniority schedule by
  hash (passing the schedule additionally checks each creditor's rank matches the one it signed).
  The :class:`~vincio.settlement.book.SettlementBook` / :class:`~vincio.core.app.ContextApp` path
  signs each schedule and resolution, lands them on the hash-chained audit log, and folds an
  unresolved insolvency into the reputation path (the poster that could not make its creditors
  whole is dinged). Never a hosted receiver, a bankruptcy court, or a trusted third party — a
  mechanical, reconstructable resolution over the obligations and reserves the fabric already
  attests.

The resolution folds into the *existing* solvency path: :func:`~vincio.settlement.solvency.attest_liabilities`
attests the obligations, :func:`~vincio.settlement.custody.attest_custody` proves the reserves,
:func:`build_seniority_schedule` ranks them, and :func:`resolve_insolvency` distributes the
reserves across the obligations in one call (:meth:`~vincio.core.app.ContextApp.build_seniority_schedule`
/ :meth:`~vincio.core.app.ContextApp.resolve_insolvency`). Everything is dependency-free,
deterministic, and offline.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .custody import CustodyAttestation
from .record import SettlementSignature
from .setoff import SetOffStatement
from .solvency import CompletenessProof, LiabilityAttestation, SolvencyProof, prove_solvency

if TYPE_CHECKING:
    from ..security.audit import ChainSigner

__all__ = [
    "SeniorityTranche",
    "SeniorityVerification",
    "SenioritySchedule",
    "build_seniority_schedule",
    "CreditorRecovery",
    "WaterfallTranche",
    "InsolvencyResolutionVerification",
    "InsolvencyResolution",
    "resolve_insolvency",
]

# The audit action a seniority schedule is recorded under; the decision field carries who signed
# the ranking (``self_ranked`` when the poster signs its own, otherwise ``ranked``).
SENIORITY_ACTION = "seniority_schedule"

# The audit action an insolvency resolution is recorded under; the decision field carries whether
# every creditor was made whole (``solvent``) or some bear a shortfall (``resolved``).
INSOLVENCY_ACTION = "insolvency_resolution"

_TOLERANCE = 1e-6


def _r6(value: float) -> float:
    """Round a money figure to six places so float drift never breaks a balance."""
    return round(float(value), 6)


def _rate(recovery: float, claim: float) -> float:
    """The recovery rate (recovered / claimed), 1.0 for a zero claim, rounded for stability."""
    if claim <= _TOLERANCE:
        return 1.0
    return round(min(1.0, max(0.0, recovery / claim)), 9)


# -- seniority schedule -------------------------------------------------------


class SeniorityTranche(BaseModel):
    """One priority rank of a :class:`SenioritySchedule` — the creditors paid at that level.

    ``rank`` is the priority (``0`` most senior; capital reaches a tranche only once every
    lower-ranked one is paid in full); ``creditors`` are the obligations sharing that rank
    (paid pari-passu among themselves when the tranche is only partly funded); ``label`` is
    free-text provenance (e.g. ``"secured"`` / ``"senior unsecured"`` / ``"subordinated"``)
    and is bound into the schedule hash like every other field.
    """

    rank: int = 0
    creditors: list[str] = Field(default_factory=list)
    label: str = ""

    def facts(self) -> dict[str, Any]:
        """The per-tranche facts the schedule's content hash binds (creditors sorted)."""
        return {
            "rank": self.rank,
            "label": self.label,
            "creditors": sorted(self.creditors),
        }


class SeniorityVerification(BaseModel):
    """The (non-raising) outcome of verifying a seniority schedule offline.

    A schedule is **valid** when its content hash recomputes (``hash_ok``), the tranches are
    well-formed — distinct ranks, no creditor in two tranches, no blank creditor (``well_formed``)
    — and, with a ``verifier``, every signature checks (``signatures_ok``). A re-ordered or
    malformed ranking is caught from the bytes alone, even after re-sealing.
    """

    valid: bool
    hash_ok: bool
    well_formed: bool
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class SenioritySchedule(BaseModel):
    """A signed, offline-verifiable ranking of a poster's obligations into priority tranches.

    Produced by :func:`build_seniority_schedule` (or
    :meth:`~vincio.settlement.book.SettlementBook.build_seniority_schedule` /
    :meth:`~vincio.core.app.ContextApp.build_seniority_schedule`): it ranks the creditors a
    ``poster`` owes into :class:`SeniorityTranche`\\ s — rank ``0`` most senior — so the order an
    insolvency waterfall pays capital out in is an auditable, non-repudiable artifact rather than
    an after-the-fact assertion. It binds the poster, the tranches, and the instant onto a content
    hash, so the ranking is a mechanical number anyone recomputes.

    A schedule may be signed by the poster (its own declared subordination) or by its creditors
    (an inter-creditor agreement) — the signature records *who agreed to the order*; validity rests
    on the content hash and the well-formedness of the tranches. :meth:`verify` recomputes the hash
    and refuses a malformed ranking (a creditor in two tranches, a duplicate rank) from the bytes
    alone. A creditor not listed in any tranche falls to the :attr:`residual_rank` — the most
    junior level — so the schedule need not enumerate every obligation up front.
    """

    id: str = Field(default_factory=lambda: new_id("seniority"))
    poster: str
    tranches: list[SeniorityTranche] = Field(default_factory=list)
    as_of: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    def _sorted_tranches(self) -> list[SeniorityTranche]:
        """The tranches in canonical (rank-ascending) order — most senior first."""
        return sorted(self.tranches, key=lambda t: t.rank)

    @property
    def ranks(self) -> list[int]:
        """The distinct priority ranks the schedule defines, ascending."""
        return sorted({t.rank for t in self.tranches})

    @property
    def residual_rank(self) -> int:
        """The rank an unlisted creditor falls to — one below the most junior listed tranche.

        A creditor the schedule does not name is treated as the most junior obligation (paid only
        after every listed tranche), so an incomplete schedule never silently promotes an omitted
        creditor. ``0`` when the schedule lists no tranche (every creditor is then one pari-passu
        tranche).
        """
        return (max((t.rank for t in self.tranches), default=-1)) + 1

    @property
    def creditors(self) -> list[str]:
        """Every creditor the schedule ranks, sorted."""
        return sorted(c for t in self.tranches for c in t.creditors)

    def ranking(self) -> dict[str, int]:
        """The ``creditor -> rank`` map the waterfall reads (a listed creditor's tranche rank)."""
        return {c: t.rank for t in self.tranches for c in t.creditors}

    def rank_of(self, creditor: str) -> int:
        """The rank ``creditor`` is paid at — its tranche's, or the residual rank if unlisted."""
        return self.ranking().get(creditor, self.residual_rank)

    # -- hashing ------------------------------------------------------------

    def schedule_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the poster, the tranches, and the instant.

        Excludes the id, signatures, and audit linkage (local metadata, not the ranking), so the
        same poster's same ranking as of the same instant hashes identically wherever it is
        recomputed. Tranches are sorted by rank (and creditors within each) so the order they were
        listed in never changes the hash.
        """
        return {
            "poster": self.poster,
            "as_of": self.as_of.isoformat(),
            "tranches": [t.facts() for t in self._sorted_tranches()],
        }

    def compute_hash(self) -> str:
        """The content hash binding the poster, the tranches, and the instant."""
        return stable_hash(self.schedule_facts(), length=32)

    def seal(self) -> SenioritySchedule:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> SenioritySchedule:
        """Add ``party``'s signature over the content hash (sealing first).

        A schedule is signed by whoever agreed to the ranking — the poster declaring its own
        subordination, or a creditor party to an inter-creditor agreement. Re-signing for the same
        party replaces its prior signature, so a schedule cannot accumulate stale signatures for
        one identity.
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

    def _well_formed(self) -> bool:
        """Distinct ranks, every creditor in exactly one tranche, no blank creditor."""
        ranks = [t.rank for t in self.tranches]
        if len(ranks) != len(set(ranks)):
            return False
        seen: set[str] = set()
        for tranche in self.tranches:
            for creditor in tranche.creditors:
                if not creditor:
                    return False
                if creditor in seen:
                    return False
                seen.add(creditor)
        return True

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> SeniorityVerification:
        """Verify the schedule offline: the hash recomputes and the tranches are well-formed.

        Recomputes the content hash and checks the tranches are well-formed (distinct ranks, no
        creditor ranked twice) — so a re-ordered or malformed ranking is caught even when the hash
        was recomputed to match. ``verifier`` additionally checks each signature against the content
        hash; ``require`` names parties that must have a verified signature (defaults to none).
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        well_formed = self._well_formed()
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
        missing = [p for p in (require or []) if p not in verified]
        if missing:
            signatures_ok = False
        valid = hash_ok and well_formed and signatures_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "schedule is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the schedule facts"
            elif not well_formed:
                reason = "tranches are malformed (a duplicate rank or a creditor ranked twice)"
            elif missing:
                reason = f"missing/invalid signatures for {missing}"
            else:
                reason = "signature mismatch"
        return SeniorityVerification(
            valid=valid,
            hash_ok=hash_ok,
            well_formed=well_formed,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> SenioritySchedule:
        """Verify and raise :class:`SettlementError` if the schedule is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"seniority schedule {self.id} failed verification: {result.reason}",
                details={"schedule_id": self.id, "reason": result.reason},
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the schedule for the audit chain."""
        return to_jsonable(
            {
                "schedule_id": self.id,
                "poster": self.poster,
                "tranches": len(self.tranches),
                "ranks": self.ranks,
                "creditors": len(self.creditors),
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> SenioritySchedule:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the priority tranches and the creditors at each rank."""
        print(f"Seniority schedule ({self.poster}): {len(self.tranches)} tranche(s)")
        for tranche in self._sorted_tranches():
            label = f" {tranche.label}" if tranche.label else ""
            print(f"  rank {tranche.rank}{label}: {', '.join(sorted(tranche.creditors))}")


def _coerce_tranches(tranches: Any) -> list[SeniorityTranche]:
    """Normalize a tranches spec into ranked :class:`SeniorityTranche`\\ s.

    Accepts an iterable whose items are :class:`SeniorityTranche`, a list/tuple of creditor
    strings (its **position** becomes the rank — earlier is more senior), a single creditor
    string (a one-creditor tranche at its position), or a ``{"rank", "creditors", "label"}`` dict.
    A positional rank is only assigned when the item carries none, so explicit ranks and positional
    ones never silently collide. Raises :class:`SettlementError` on an unrecognized item.
    """
    coerced: list[SeniorityTranche] = []
    for position, item in enumerate(tranches):
        if isinstance(item, SeniorityTranche):
            coerced.append(item)
        elif isinstance(item, dict):
            tranche = SeniorityTranche.model_validate(item)
            coerced.append(tranche)
        elif isinstance(item, str):
            coerced.append(SeniorityTranche(rank=position, creditors=[item]))
        elif isinstance(item, (list, tuple)):
            creditors = [str(c) for c in item]
            coerced.append(SeniorityTranche(rank=position, creditors=creditors))
        else:
            raise SettlementError(
                "build_seniority_schedule tranches must be SeniorityTranche, a list of creditor "
                f"names, a creditor name, or a {{rank, creditors}} dict; got {item!r}",
                details={"item": repr(item)},
            )
    return coerced


def build_seniority_schedule(
    poster: str,
    tranches: Any,
    *,
    as_of: datetime | None = None,
) -> SenioritySchedule:
    """Rank a poster's obligations into a sealed, unsigned :class:`SenioritySchedule`.

    The seniority analogue of :func:`~vincio.settlement.solvency.attest_liabilities`: it ranks the
    creditors ``poster`` owes into priority tranches an :func:`resolve_insolvency` waterfall pays
    out in. ``tranches`` is an ordered spec — its simplest form is a list of creditor-name lists
    where **position is priority** (the first list most senior, e.g. ``[["bank"], ["acme",
    "globex"]]``) — or :class:`SeniorityTranche` items / ``{"rank", "creditors", "label"}`` dicts
    for explicit ranks and labels.

    Returns a sealed, unsigned schedule — sign it with the poster's or a creditor's key (or let a
    :class:`~vincio.settlement.book.SettlementBook` do it on
    :meth:`~vincio.settlement.book.SettlementBook.build_seniority_schedule`). Raises
    :class:`SettlementError` when the tranches are malformed (a creditor ranked twice, a duplicate
    rank).
    """
    coerced = _coerce_tranches(tranches)
    schedule = SenioritySchedule(poster=poster, tranches=coerced, as_of=as_of or utcnow())
    schedule.seal()
    if not schedule._well_formed():
        raise SettlementError(
            f"seniority schedule for {poster!r} is malformed: a creditor is ranked in more than "
            "one tranche, or two tranches share a rank",
            details={"poster": poster, "ranks": [t.rank for t in coerced]},
        )
    return schedule


# -- insolvency waterfall -----------------------------------------------------


class CreditorRecovery(BaseModel):
    """One creditor's outcome in an :class:`InsolvencyResolution` waterfall.

    Pinpoints what a creditor recovers from the scarce reserves and the shortfall it bears:
    ``claim_usd`` is the obligation owed, ``rank`` the seniority tranche it was paid at,
    ``recovery_usd`` what the waterfall distributes to it (full when its tranche was funded,
    a pari-passu share when only partly funded, ``0`` when nothing reached its tranche),
    ``shortfall_usd`` the residue (``claim − recovery``), and ``recovery_rate`` the fraction
    recovered (``1.0`` when made whole).
    """

    creditor: str
    rank: int = 0
    label: str = ""
    claim_usd: float = 0.0
    recovery_usd: float = 0.0
    shortfall_usd: float = 0.0
    recovery_rate: float = 1.0
    # Close-out set-off (3.43): the creditor's gross claim before set-off and the amount its own
    # obligation back cancelled, so ``claim_usd`` (the net distributed over) re-derives as
    # ``max(0, gross_claim_usd − set_off_usd)``. Both equal ``claim_usd`` / ``0`` when no set-off
    # applies, and are bound into the resolution hash only when a set-off was folded.
    gross_claim_usd: float = 0.0
    set_off_usd: float = 0.0

    @property
    def made_whole(self) -> bool:
        """Whether the creditor recovered its full claim (no shortfall)."""
        return self.shortfall_usd <= _TOLERANCE

    @property
    def set_off(self) -> bool:
        """Whether the creditor's gross claim was reduced by a close-out set-off."""
        return self.set_off_usd > _TOLERANCE


class WaterfallTranche(BaseModel):
    """The per-tranche distribution summary of an :class:`InsolvencyResolution`.

    A roll-up of one priority rank: ``claim_usd`` is the total owed at the rank, ``paid_usd`` what
    the waterfall distributed to it, ``coverage`` the fraction paid (``1.0`` when funded in full,
    the pari-passu fraction when partly funded, ``0.0`` when nothing reached it), and ``creditors``
    the per-creditor :class:`CreditorRecovery`\\ s at that rank.
    """

    rank: int = 0
    label: str = ""
    claim_usd: float = 0.0
    paid_usd: float = 0.0
    coverage: float = 1.0
    creditors: list[CreditorRecovery] = Field(default_factory=list)


def _distribute(
    claims: dict[str, float],
    ranking: dict[str, int],
    residual_rank: int,
    reserves_usd: float,
    labels: dict[int, str] | None = None,
) -> tuple[list[WaterfallTranche], list[CreditorRecovery]]:
    """Distribute ``reserves_usd`` across ``claims`` by seniority, then pari-passu within a rank.

    The deterministic core of the waterfall — used identically at build time and on every
    :meth:`InsolvencyResolution.verify`, so the recorded distribution re-derives from the recorded
    claims, ranks, and reserves alone. Each creditor's rank is ``ranking`` (or ``residual_rank``
    if unlisted). Tranches are paid in ascending rank order: a tranche is funded in full before any
    capital reaches a more junior one, and a partly-funded tranche splits what is left
    proportionally to each claim (pari passu). Returns the per-tranche summaries and the flat,
    ``(rank, creditor)``-ordered recoveries.
    """
    labels = labels or {}
    by_rank: dict[int, dict[str, float]] = {}
    for creditor in sorted(claims):
        rank = ranking.get(creditor, residual_rank)
        by_rank.setdefault(rank, {})[creditor] = _r6(claims[creditor])

    available = _r6(max(0.0, reserves_usd))
    tranches: list[WaterfallTranche] = []
    recoveries: list[CreditorRecovery] = []
    for rank in sorted(by_rank):
        members = by_rank[rank]
        tranche_claim = _r6(sum(members.values()))
        pay = _r6(min(available, tranche_claim))
        fully_paid = tranche_claim <= _TOLERANCE or pay + _TOLERANCE >= tranche_claim
        coverage = 1.0 if fully_paid else round(max(0.0, pay / tranche_claim), 9)
        tranche_recoveries: list[CreditorRecovery] = []
        paid_in_tranche = 0.0
        for creditor in sorted(members):
            claim = members[creditor]
            if fully_paid:
                recovery = claim
            else:
                recovery = min(_r6(claim * coverage), claim)
            recovery = _r6(recovery)
            paid_in_tranche = _r6(paid_in_tranche + recovery)
            shortfall = _r6(max(0.0, claim - recovery))
            tranche_recoveries.append(
                CreditorRecovery(
                    creditor=creditor,
                    rank=rank,
                    label=labels.get(rank, ""),
                    claim_usd=claim,
                    recovery_usd=recovery,
                    shortfall_usd=shortfall,
                    recovery_rate=_rate(recovery, claim),
                )
            )
        available = _r6(max(0.0, available - paid_in_tranche))
        tranches.append(
            WaterfallTranche(
                rank=rank,
                label=labels.get(rank, ""),
                claim_usd=tranche_claim,
                paid_usd=paid_in_tranche,
                coverage=coverage,
                creditors=tranche_recoveries,
            )
        )
        recoveries.extend(tranche_recoveries)
    return tranches, recoveries


class InsolvencyResolutionVerification(BaseModel):
    """The (non-raising) outcome of verifying an insolvency resolution offline.

    A resolution is **valid** when its content hash recomputes (``hash_ok``), the entire
    distribution re-derives from the recorded per-creditor claims, ranks, and reserves
    (``distribution_sound`` — so an over-stated recovery, a re-ordered tranche, or a junior
    creditor paid ahead of a senior one is caught), and — with a ``verifier`` — every signature
    checks (``signatures_ok``). When the seniority schedule is supplied, ``schedule_bound`` also
    holds when each creditor's rank matches the one the schedule signed; when the set-off statements
    are supplied, ``set_off_bound`` holds when the resolution binds exactly those statements and each
    nets its creditor's claim as the statement commits.
    """

    valid: bool
    hash_ok: bool
    distribution_sound: bool
    schedule_bound: bool = True
    set_off_bound: bool = True
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class InsolvencyResolution(BaseModel):
    """A signed, offline-verifiable resolution distributing reserves across ranked liabilities.

    Produced by :func:`resolve_insolvency` (or
    :meth:`~vincio.settlement.book.SettlementBook.resolve_insolvency` /
    :meth:`~vincio.core.app.ContextApp.resolve_insolvency`): it distributes a poster's proven
    reserves (a :class:`~vincio.settlement.custody.CustodyAttestation`) across its proven
    obligations (a :class:`~vincio.settlement.solvency.LiabilityAttestation`) **by seniority then
    pari-passu within a tranche** (a :class:`SenioritySchedule`), pinpointing each creditor's
    bounded :class:`CreditorRecovery`. It binds both attestation hashes, the schedule hash, the
    proven figures, and the distribution onto a content hash, so the resolution is a mechanical
    number anyone recomputes — the who-gets-what an :class:`~vincio.settlement.solvency.InsolvencyBreach`
    only flagged.

    The resolution is **solvent** when the reserves cover every obligation (no shortfall);
    otherwise the creditors with a shortfall are :attr:`shortfall_bearers`, ordered by seniority.
    When a close-out set-off was folded (:func:`~vincio.settlement.resolve_insolvency` with
    ``set_off=``), each creditor's gross claim is reduced to its net liability *before* the
    distribution, the netted statements are bound by hash, and every :class:`CreditorRecovery`
    carries its ``gross_claim_usd`` and ``set_off_usd`` so the net it was distributed over
    re-derives. :meth:`verify` re-derives the entire waterfall from the recorded claims, ranks, and
    reserves — so an over-stated recovery or a re-ordered tranche is caught from the bytes alone —
    and binds the seniority schedule and the set-off statements by hash.
    """

    id: str = Field(default_factory=lambda: new_id("insolvency"))
    poster: str
    custodian: str = ""
    attestor: str = ""
    custody_hash: str = ""
    liability_hash: str = ""
    completeness_hash: str = ""
    solvency_hash: str = ""
    schedule_hash: str = ""
    # Close-out set-off (3.43): the statements whose mutual obligations were netted before the
    # waterfall, sorted. Bound into the content hash only when non-empty, so a resolution with no
    # set-off hashes exactly as before — the close-out is additive.
    set_off_hashes: list[str] = Field(default_factory=list)

    reserves_usd: float = 0.0
    liabilities_usd: float = 0.0
    gross_liabilities_usd: float = 0.0
    attested_liabilities_usd: float = 0.0
    set_off_usd: float = 0.0
    distributed_usd: float = 0.0
    shortfall_usd: float = 0.0
    surplus_usd: float = 0.0
    residual_rank: int = 0

    tranches: list[WaterfallTranche] = Field(default_factory=list)
    recoveries: list[CreditorRecovery] = Field(default_factory=list)

    as_of: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def solvent(self) -> bool:
        """Whether the reserves made every creditor whole (no shortfall borne)."""
        return self.shortfall_usd <= _TOLERANCE

    @property
    def insolvent(self) -> bool:
        """Whether some creditor bears a shortfall (the reserves could not cover the obligations)."""
        return not self.solvent

    @property
    def status(self) -> str:
        """``solvent`` (every creditor made whole) or ``resolved`` (a distributed shortfall)."""
        return "solvent" if self.solvent else "resolved"

    @property
    def fully_recovered(self) -> bool:
        """Whether every creditor recovered its full claim — the headline of a solvent resolution."""
        return self.solvent

    @property
    def shortfall_bearers(self) -> list[str]:
        """The creditors that bear a shortfall, ordered by seniority then creditor (most senior first)."""
        return [
            r.creditor
            for r in sorted(self.recoveries, key=lambda r: (r.rank, r.creditor))
            if r.shortfall_usd > _TOLERANCE
        ]

    @property
    def recovery_rate(self) -> float:
        """The overall fraction of the (net) obligations the reserves recovered."""
        return _rate(self.distributed_usd, self.liabilities_usd)

    @property
    def set_off(self) -> bool:
        """Whether a close-out set-off reduced the obligations before the distribution."""
        return bool(self.set_off_hashes)

    def recovery_of(self, creditor: str) -> CreditorRecovery | None:
        """The :class:`CreditorRecovery` for ``creditor``, or ``None`` if it is not a creditor."""
        return next((r for r in self.recoveries if r.creditor == creditor), None)

    def _recorded_claims(self) -> dict[str, float]:
        """The per-creditor claims the recorded recoveries carry."""
        return {r.creditor: _r6(r.claim_usd) for r in self.recoveries}

    def _recorded_ranking(self) -> dict[str, int]:
        """The per-creditor ranks the recorded recoveries carry."""
        return {r.creditor: r.rank for r in self.recoveries}

    # -- hashing ------------------------------------------------------------

    def resolution_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the proofs, the figures, and the distribution.

        Binds both attestation hashes, the schedule hash, the proven reserves and liabilities, and
        every per-creditor recovery (with its rank and claim, sorted by ``(rank, creditor)``), so a
        re-ordered tranche or an over-stated recovery is caught even after re-sealing. Excludes the
        id, signatures, and audit linkage (local metadata), so two folders compute the same hash.
        """
        facts: dict[str, Any] = {
            "poster": self.poster,
            "custodian": self.custodian,
            "attestor": self.attestor,
            "custody_hash": self.custody_hash,
            "liability_hash": self.liability_hash,
            "completeness_hash": self.completeness_hash,
            "solvency_hash": self.solvency_hash,
            "schedule_hash": self.schedule_hash,
            "reserves_usd": _r6(self.reserves_usd),
            "liabilities_usd": _r6(self.liabilities_usd),
            "attested_liabilities_usd": _r6(self.attested_liabilities_usd),
            "distributed_usd": _r6(self.distributed_usd),
            "shortfall_usd": _r6(self.shortfall_usd),
            "surplus_usd": _r6(self.surplus_usd),
            "residual_rank": self.residual_rank,
            "as_of": self.as_of.isoformat(),
            "recoveries": [
                {
                    "creditor": r.creditor,
                    "rank": r.rank,
                    "claim_usd": _r6(r.claim_usd),
                    "recovery_usd": _r6(r.recovery_usd),
                    "shortfall_usd": _r6(r.shortfall_usd),
                }
                for r in sorted(self.recoveries, key=lambda r: (r.rank, r.creditor))
            ],
        }
        # The close-out set-off is bound only when one was folded, so a resolution with no set-off
        # hashes exactly as before (backward-compatible). The per-creditor gross and netted amounts
        # are bound for every creditor the set-off touched, so verify re-derives each net claim.
        if self.set_off_hashes:
            facts["set_off"] = {
                "hashes": sorted(self.set_off_hashes),
                "gross_liabilities_usd": _r6(self.gross_liabilities_usd),
                "set_off_usd": _r6(self.set_off_usd),
                "by_creditor": [
                    {
                        "creditor": r.creditor,
                        "gross_claim_usd": _r6(r.gross_claim_usd),
                        "set_off_usd": _r6(r.set_off_usd),
                    }
                    for r in sorted(self.recoveries, key=lambda r: (r.rank, r.creditor))
                    if r.set_off_usd > _TOLERANCE
                ],
            }
        return facts

    def compute_hash(self) -> str:
        """The content hash binding the folded proofs and the seniority waterfall."""
        return stable_hash(self.resolution_facts(), length=32)

    def seal(self) -> InsolvencyResolution:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> InsolvencyResolution:
        """Add ``party``'s signature over the content hash (sealing first).

        A resolution is signed by whoever folded it — a creditor, an administrator, or the poster
        proving how it would distribute its reserves. Re-signing for the same party replaces its
        prior signature, so a resolution cannot accumulate stale signatures for one identity.
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

    def _set_off_sound(self) -> bool:
        """When a close-out set-off was folded, every net claim re-derives from the gross.

        With no set-off bound this is vacuously true. Otherwise each recovery's distributed claim
        must equal its gross minus the netted amount floored at zero, both figures non-negative, and
        the bound gross / set-off totals must equal the sums and reconcile the net liabilities — so a
        tampered gross, an inflated set-off, or a re-stated net is caught from the bytes alone.
        """
        if not self.set_off_hashes:
            return True
        gross_total = 0.0
        set_off_total = 0.0
        for r in self.recoveries:
            gross = _r6(r.gross_claim_usd)
            netted = _r6(r.set_off_usd)
            if gross < -_TOLERANCE or netted < -_TOLERANCE:
                return False
            if abs(r.claim_usd - _r6(max(0.0, gross - netted))) > _TOLERANCE:
                return False
            gross_total = _r6(gross_total + gross)
            set_off_total = _r6(set_off_total + netted)
        if abs(self.gross_liabilities_usd - gross_total) > _TOLERANCE:
            return False
        if abs(self.set_off_usd - set_off_total) > _TOLERANCE:
            return False
        if abs(self.liabilities_usd - _r6(gross_total - set_off_total)) > _TOLERANCE:
            return False
        return True

    def _distribution_sound(self) -> bool:
        """The recorded distribution re-derives from the recorded claims, ranks, and reserves."""
        if self.reserves_usd < -_TOLERANCE or self.liabilities_usd < -_TOLERANCE:
            return False
        if self.attested_liabilities_usd < -_TOLERANCE:
            return False
        if not self._set_off_sound():
            return False
        # The completed liabilities can only raise the attestor's figure, never lower it. With a
        # close-out set-off the bound *gross* total carries that floor (the net is the gross minus
        # the set-off and can legitimately fall below the attested figure).
        gross_floor = self.gross_liabilities_usd if self.set_off_hashes else self.liabilities_usd
        if gross_floor < _r6(self.attested_liabilities_usd) - _TOLERANCE:
            return False
        claims = self._recorded_claims()
        ranking = self._recorded_ranking()
        # The bound liability total must equal the sum of the per-creditor claims the waterfall
        # distributed over — so a forged total that does not match the recoveries is caught.
        if abs(self.liabilities_usd - _r6(sum(claims.values()))) > _TOLERANCE:
            return False
        expected_tranches, expected = _distribute(
            claims, ranking, self.residual_rank, self.reserves_usd
        )
        if not self._recoveries_match(expected):
            return False
        if not self._tranches_match(expected_tranches):
            return False
        distributed = _r6(sum(r.recovery_usd for r in expected))
        if abs(self.distributed_usd - distributed) > _TOLERANCE:
            return False
        if abs(self.shortfall_usd - _r6(max(0.0, self.liabilities_usd - distributed))) > _TOLERANCE:
            return False
        if abs(self.surplus_usd - _r6(max(0.0, self.reserves_usd - distributed))) > _TOLERANCE:
            return False
        return True

    def _recoveries_match(self, expected: list[CreditorRecovery]) -> bool:
        """Whether the recorded recoveries equal the re-derived ones (order-independent)."""
        if len(expected) != len(self.recoveries):
            return False
        have = {r.creditor: r for r in self.recoveries}
        for want in expected:
            got = have.get(want.creditor)
            if got is None or got.rank != want.rank:
                return False
            if abs(got.claim_usd - want.claim_usd) > _TOLERANCE:
                return False
            if abs(got.recovery_usd - want.recovery_usd) > _TOLERANCE:
                return False
            if abs(got.shortfall_usd - want.shortfall_usd) > _TOLERANCE:
                return False
        return True

    def _tranches_match(self, expected: list[WaterfallTranche]) -> bool:
        """Whether the recorded tranche summaries equal the re-derived ones (order-independent)."""
        if len(expected) != len(self.tranches):
            return False
        have = {t.rank: t for t in self.tranches}
        for want in expected:
            got = have.get(want.rank)
            if got is None:
                return False
            if abs(got.claim_usd - want.claim_usd) > _TOLERANCE:
                return False
            if abs(got.paid_usd - want.paid_usd) > _TOLERANCE:
                return False
        return True

    def verify(
        self,
        verifier: ChainSigner | None = None,
        schedule: SenioritySchedule | None = None,
        set_off: list[SetOffStatement] | None = None,
        *,
        require: list[str] | None = None,
    ) -> InsolvencyResolutionVerification:
        """Verify the resolution offline: the hash recomputes and the waterfall re-derives.

        Recomputes the content hash and re-derives the entire distribution from the recorded
        per-creditor claims, ranks, and reserves — so an over-stated recovery, a re-ordered
        tranche, or a junior creditor paid ahead of a senior one is caught even when the hash was
        recomputed to match. ``verifier`` additionally checks each signature; ``require`` names
        parties that must have a verified signature (defaults to none). Passing the
        ``schedule`` binds it: each creditor's recorded rank must match the rank the schedule
        signed, and the bound :attr:`schedule_hash` must match the schedule's content hash — so a
        resolution cannot quietly re-rank a creditor away from the order its creditors agreed to.
        Passing the ``set_off`` statements binds them: the resolution must bind exactly those
        statements (by hash), each must verify (mutually-signed with a ``verifier``), and each must
        net its creditor's gross claim by the amount it commits — so a resolution cannot quietly
        net a creditor against a set-off it never agreed to.
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        distribution_sound = self._distribution_sound()
        schedule_bound = True
        schedule_reason: str | None = None
        if schedule is not None:
            sched_result = schedule.verify(verifier)
            same_schedule = self.schedule_hash == schedule.content_hash
            ranks_match = all(r.rank == schedule.rank_of(r.creditor) for r in self.recoveries)
            schedule_bound = sched_result.valid and same_schedule and ranks_match
            if not schedule_bound:
                if not sched_result.valid:
                    schedule_reason = f"bound schedule failed verification: {sched_result.reason}"
                elif not same_schedule:
                    schedule_reason = "resolution does not bind the supplied schedule's hash"
                else:
                    schedule_reason = "a creditor's rank does not match the supplied schedule"
        set_off_bound = True
        set_off_reason: str | None = None
        if set_off is not None:
            set_off_bound, set_off_reason = self._verify_set_off(set_off, verifier)
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
        missing = [p for p in (require or []) if p not in verified]
        if missing:
            signatures_ok = False
        valid = (
            hash_ok and distribution_sound and schedule_bound and set_off_bound and signatures_ok
        )
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "resolution is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the resolution facts"
            elif not distribution_sound:
                reason = "the seniority waterfall does not re-derive from the recorded recoveries"
            elif not schedule_bound:
                reason = schedule_reason
            elif not set_off_bound:
                reason = set_off_reason
            elif missing:
                reason = f"missing/invalid signatures for {missing}"
            else:
                reason = "signature mismatch"
        return InsolvencyResolutionVerification(
            valid=valid,
            hash_ok=hash_ok,
            distribution_sound=distribution_sound,
            schedule_bound=schedule_bound,
            set_off_bound=set_off_bound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def _verify_set_off(
        self, statements: list[SetOffStatement], verifier: ChainSigner | None
    ) -> tuple[bool, str | None]:
        """Whether the resolution binds exactly these statements and nets each as they commit."""
        bound = sorted(self.set_off_hashes)
        supplied = sorted(s.content_hash for s in statements)
        if bound != supplied:
            return False, "resolution does not bind exactly the supplied set-off statements"
        recoveries = {r.creditor: r for r in self.recoveries}
        for statement in statements:
            result = statement.verify(verifier, require_mutual=True)
            if not result.valid:
                return False, f"bound set-off statement failed verification: {result.reason}"
            if statement.poster != self.poster:
                return False, "a set-off statement is for a different poster"
            recovery = recoveries.get(statement.creditor)
            if recovery is None:
                return False, f"creditor {statement.creditor!r} has no recovery to set off"
            if abs(recovery.gross_claim_usd - statement.owed_usd) > _TOLERANCE:
                return False, "a set-off statement's gross does not match the recorded claim"
            if abs(recovery.set_off_usd - statement.set_off_usd) > _TOLERANCE:
                return False, "a set-off statement nets a different amount than recorded"
        return True, None

    def require_valid(
        self,
        verifier: ChainSigner | None = None,
        schedule: SenioritySchedule | None = None,
        set_off: list[SetOffStatement] | None = None,
        *,
        require: list[str] | None = None,
    ) -> InsolvencyResolution:
        """Verify and raise :class:`SettlementError` if the resolution is not valid."""
        result = self.verify(verifier, schedule, set_off, require=require)
        if not result.valid:
            raise SettlementError(
                f"insolvency resolution {self.id} failed verification: {result.reason}",
                details={"resolution_id": self.id, "reason": result.reason},
            )
        return self

    def require_fully_recovered(self) -> InsolvencyResolution:
        """Raise :class:`SettlementError` if any creditor bears a shortfall.

        The strict-mode counterpart to inspecting :attr:`fully_recovered`: a resolution whose
        reserves could not make every creditor whole is the resolved insolvency itself, and this
        pinpoints the creditors that bear the shortfall and how much was distributed.
        """
        if not self.solvent:
            raise SettlementError(
                f"insolvency resolution {self.id} leaves ${self.shortfall_usd:,.2f} unrecovered: "
                f"{self.shortfall_bearers} bear a shortfall after distributing "
                f"${self.distributed_usd:,.2f} of ${self.liabilities_usd:,.2f} owed",
                details={
                    "resolution_id": self.id,
                    "poster": self.poster,
                    "shortfall_usd": self.shortfall_usd,
                    "shortfall_bearers": self.shortfall_bearers,
                },
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the resolution for the audit chain."""
        return to_jsonable(
            {
                "resolution_id": self.id,
                "poster": self.poster,
                "custodian": self.custodian,
                "attestor": self.attestor,
                "status": self.status,
                "reserves_usd": _r6(self.reserves_usd),
                "liabilities_usd": _r6(self.liabilities_usd),
                "gross_liabilities_usd": _r6(self.gross_liabilities_usd),
                "set_off_usd": _r6(self.set_off_usd),
                "distributed_usd": _r6(self.distributed_usd),
                "shortfall_usd": _r6(self.shortfall_usd),
                "recovery_rate": self.recovery_rate,
                "tranches": len(self.tranches),
                "shortfall_bearers": self.shortfall_bearers,
                "schedule_hash": self.schedule_hash,
                "set_off_hashes": sorted(self.set_off_hashes),
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> InsolvencyResolution:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the reserves distributed across the ranked creditors and any shortfalls."""
        print(
            f"Insolvency resolution ({self.poster}): distributed ${self.distributed_usd:,.2f} of "
            f"${self.liabilities_usd:,.2f} owed from ${self.reserves_usd:,.2f} reserves "
            f"— {self.status}"
        )
        for tranche in sorted(self.tranches, key=lambda t: t.rank):
            label = f" {tranche.label}" if tranche.label else ""
            print(
                f"  rank {tranche.rank}{label}: paid ${tranche.paid_usd:,.2f} of "
                f"${tranche.claim_usd:,.2f} ({tranche.coverage:.0%})"
            )
            for r in sorted(tranche.creditors, key=lambda r: r.creditor):
                mark = "" if r.made_whole else f" (short ${r.shortfall_usd:,.2f})"
                print(f"    {r.creditor}: ${r.recovery_usd:,.2f} of ${r.claim_usd:,.2f}{mark}")


def _claim_map(liabilities: LiabilityAttestation) -> dict[str, float]:
    """The per-creditor obligation map an attestation commits, summing duplicate line items."""
    owed: dict[str, float] = {}
    for line in liabilities.liabilities:
        owed[line.creditor] = _r6(owed.get(line.creditor, 0.0) + line.amount_usd)
    return owed


def _apply_set_off(
    claims: dict[str, float],
    statements: list[SetOffStatement],
    poster: str,
    *,
    verifier: ChainSigner | None,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Reduce each creditor's gross claim to its net liability via the close-out statements.

    The close-out pass run *before* the waterfall: for every set-off statement, the creditor's
    proven liability is reduced by what it owes the estate back, floored at zero, so a creditor in
    debit recovers nothing and the distributable claims shrink to the true net exposure. Each
    statement must be mutually-signed and well-formed (a one-sided or tampered close-out is refused),
    for *this* poster, and reconcile against the gross the attestation commits — an over-stated
    set-off claiming a different gross than the attested claim is refused — and one creditor cannot
    be set off twice. Returns the netted claims, the per-creditor amount netted out, and the bound
    statement hashes (sorted).
    """
    netted = dict(claims)
    set_off_by_creditor: dict[str, float] = {}
    hashes: list[str] = []
    seen: set[str] = set()
    for statement in statements:
        statement.require_valid(verifier, require_mutual=True)
        if statement.poster != poster:
            raise SettlementError(
                f"set-off statement {statement.id} is for poster {statement.poster!r}, not the "
                f"poster {poster!r} the resolution distributes; refusing it",
                details={"setoff_id": statement.id, "poster": poster},
            )
        creditor = statement.creditor
        if creditor in seen:
            raise SettlementError(
                f"creditor {creditor!r} is set off in more than one statement for {poster!r}; "
                "a creditor's mutual obligations must net into a single close-out",
                details={"poster": poster, "creditor": creditor},
            )
        seen.add(creditor)
        gross = _r6(netted.get(creditor, 0.0))
        if abs(statement.owed_usd - gross) > _TOLERANCE:
            raise SettlementError(
                f"set-off statement {statement.id} states {poster!r} owes {creditor!r} "
                f"${statement.owed_usd:,.2f}, but the proven liabilities show ${gross:,.2f}; "
                "refusing an over-stated set-off",
                details={
                    "setoff_id": statement.id,
                    "creditor": creditor,
                    "stated_owed_usd": statement.owed_usd,
                    "proven_owed_usd": gross,
                },
            )
        applied = _r6(min(gross, statement.owing_usd))
        if applied <= _TOLERANCE:
            # A statement that nets nothing (the creditor owes the estate nothing) is recorded so
            # the close-out is auditable, but leaves the claim unchanged.
            hashes.append(statement.content_hash)
            continue
        netted[creditor] = _r6(max(0.0, gross - statement.owing_usd))
        set_off_by_creditor[creditor] = applied
        hashes.append(statement.content_hash)
    return netted, set_off_by_creditor, sorted(hashes)


def resolve_insolvency(
    custody: CustodyAttestation,
    liabilities: LiabilityAttestation,
    schedule: SenioritySchedule | None = None,
    *,
    poster: str | None = None,
    completeness: CompletenessProof | None = None,
    solvency: SolvencyProof | None = None,
    set_off: list[SetOffStatement] | None = None,
    as_of: datetime | None = None,
    verifier: ChainSigner | None = None,
) -> InsolvencyResolution:
    """Distribute a poster's proven reserves across its ranked liabilities into a resolution.

    The reach once :func:`~vincio.settlement.solvency.prove_solvency` *flags* an insolvency:
    resolving it into who-gets-what. Folds a poster's
    :class:`~vincio.settlement.custody.CustodyAttestation` (proven reserves) against its
    :class:`~vincio.settlement.solvency.LiabilityAttestation` (proven obligations) — reusing
    :func:`~vincio.settlement.solvency.prove_solvency` for every tamper, forgery, and wrong-poster
    refusal — and distributes the reserves across the obligations **by seniority then pari-passu
    within a tranche**, ranked by ``schedule``. With no ``schedule`` the whole liability set is one
    tranche (pure pari-passu, exactly the rehypothecation guard's apportionment).

    Pass ``completeness`` (a :class:`~vincio.settlement.solvency.CompletenessProof` over this
    attestation) to distribute against the **completed** liability set — every creditor a proof
    shows the attestor omitted is added at its proven claim, so the waterfall pays the obligations
    creditors can prove, not only the ones the attestor listed.

    Pass ``set_off`` — a list of mutually-signed :class:`~vincio.settlement.setoff.SetOffStatement`\\
    s for this poster — to **close-out net** the obligations *before* distributing: each creditor's
    gross claim is reduced by what it owes the estate back (floored at zero), so a creditor in debit
    recovers nothing and the distributable claims shrink to the true net exposure. The set-off is
    applied after completeness (so it nets the *completed* gross), reconciled against that gross (an
    over-stated set-off is refused), and bound into the resolution by hash. Returns a sealed,
    unsigned :class:`InsolvencyResolution` whose per-creditor :class:`CreditorRecovery` is bounded
    and whose :meth:`~InsolvencyResolution.verify` re-derives the whole distribution from the bytes.

    ``poster`` is the counterparty both attestations are about (defaults to the one they share).
    Raises :class:`SettlementError` when an attestation is tampered, forged, or for the wrong
    poster, when the ``schedule`` is malformed or for a different poster, or when a ``set_off``
    statement is one-sided, tampered, for a different poster, or over-states the gross it nets.
    """
    # A pre-built proof is honored only when no completeness is folded here: folding completeness
    # raises the claims (and adds omitted creditors) and is verified by prove_solvency, so a
    # completeness check must flow through prove_solvency rather than past a pre-built proof
    # unverified. Otherwise re-prove from the attestations (every tamper/forgery/wrong-poster
    # refusal, and the completeness verification, comes for free).
    if solvency is not None and completeness is None:
        proof = solvency
        # A pre-built proof must verify and bind *these* attestations for *this* poster, so a
        # resolution can never distribute against a proof folded from a different reserve or
        # liability claim than the one it embeds.
        result = proof.verify(verifier)
        if not result.hash_ok or not result.margin_sound:
            raise SettlementError(
                f"solvency proof {proof.id} is tampered ({result.reason}); refusing to resolve an "
                "insolvency on it",
                details={"proof_id": proof.id, "reason": result.reason},
            )
        if verifier is not None and proof.signatures and not result.signatures_ok:
            raise SettlementError(
                f"solvency proof {proof.id} has an invalid signature; refusing to resolve an "
                "insolvency on it",
                details={"proof_id": proof.id},
            )
        if (
            proof.custody_hash != custody.content_hash
            or proof.liability_hash != liabilities.content_hash
        ):
            raise SettlementError(
                f"solvency proof {proof.id} does not bind the supplied custody/liability "
                "attestations; refusing to resolve an unrelated proof",
                details={"proof_id": proof.id},
            )
    else:
        proof = prove_solvency(
            custody,
            liabilities,
            poster=poster,
            completeness=completeness,
            as_of=as_of,
            verifier=verifier,
        )
    resolved_poster = proof.poster

    claims = _claim_map(liabilities)
    if completeness is not None:
        # The completeness check was already verified by prove_solvency above; fold each proven
        # omission in at the larger of the attested or claimed figure, so an omitted creditor is
        # paid and an under-stated one is topped up to what it can prove.
        for breach in completeness.breaches:
            claims[breach.creditor] = _r6(max(claims.get(breach.creditor, 0.0), breach.claimed_usd))

    # The gross claims (post-completeness) the close-out nets against; retained so each recovery can
    # carry its gross and the amount set off, and verify can re-derive the net.
    gross_claims = dict(claims)
    set_off_by_creditor: dict[str, float] = {}
    set_off_hashes: list[str] = []
    if set_off:
        claims, set_off_by_creditor, set_off_hashes = _apply_set_off(
            claims, list(set_off), resolved_poster, verifier=verifier
        )

    schedule_hash = ""
    ranking: dict[str, int] = {}
    residual_rank = 0
    labels: dict[int, str] = {}
    if schedule is not None:
        sched_result = schedule.verify(verifier)
        if not sched_result.hash_ok or not sched_result.well_formed:
            raise SettlementError(
                f"seniority schedule {schedule.id} is invalid ({sched_result.reason}); refusing to "
                "resolve an insolvency on it",
                details={"schedule_id": schedule.id, "reason": sched_result.reason},
            )
        if verifier is not None and schedule.signatures and not sched_result.signatures_ok:
            raise SettlementError(
                f"seniority schedule {schedule.id} has an invalid signature; refusing to resolve "
                "an insolvency on it",
                details={"schedule_id": schedule.id},
            )
        if schedule.poster != resolved_poster:
            raise SettlementError(
                f"seniority schedule {schedule.id} ranks {schedule.poster!r}, not the poster "
                f"{resolved_poster!r} the resolution distributes; refusing it",
                details={
                    "schedule_id": schedule.id,
                    "ranks": schedule.poster,
                    "poster": resolved_poster,
                },
            )
        ranking = schedule.ranking()
        residual_rank = schedule.residual_rank
        schedule_hash = schedule.content_hash
        labels = {t.rank: t.label for t in schedule.tranches if t.label}

    reserves_usd = _r6(proof.reserves_usd)
    gross_liabilities_usd = _r6(sum(gross_claims.values()))
    liabilities_usd = _r6(sum(claims.values()))
    tranches, recoveries = _distribute(claims, ranking, residual_rank, reserves_usd, labels)
    # Annotate each recovery with the gross it carried and the amount its own obligation cancelled,
    # so verify re-derives the net claim it was distributed over (both equal the claim / 0 when no
    # set-off touched the creditor).
    for recovery in recoveries:
        gross = _r6(gross_claims.get(recovery.creditor, recovery.claim_usd))
        recovery.gross_claim_usd = gross
        recovery.set_off_usd = _r6(set_off_by_creditor.get(recovery.creditor, 0.0))
    distributed = _r6(sum(r.recovery_usd for r in recoveries))
    shortfall = _r6(max(0.0, liabilities_usd - distributed))
    surplus = _r6(max(0.0, reserves_usd - distributed))

    resolution = InsolvencyResolution(
        poster=resolved_poster,
        custodian=proof.custodian,
        attestor=proof.attestor,
        custody_hash=proof.custody_hash,
        liability_hash=proof.liability_hash,
        completeness_hash=proof.completeness_hash,
        solvency_hash=proof.content_hash,
        schedule_hash=schedule_hash,
        set_off_hashes=set_off_hashes,
        reserves_usd=reserves_usd,
        liabilities_usd=liabilities_usd,
        gross_liabilities_usd=gross_liabilities_usd,
        attested_liabilities_usd=_r6(proof.attested_liabilities_usd),
        set_off_usd=_r6(gross_liabilities_usd - liabilities_usd),
        distributed_usd=distributed,
        shortfall_usd=shortfall,
        surplus_usd=surplus,
        residual_rank=residual_rank,
        tranches=tranches,
        recoveries=recoveries,
        as_of=as_of or utcnow(),
    )
    return resolution.seal()
