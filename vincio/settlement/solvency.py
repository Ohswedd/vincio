"""Cross-org custody liability attestation & proof-of-solvency.

A :class:`~vincio.settlement.custody.CustodyAttestation` now proves the capital a
counterparty *holds*, so :func:`~vincio.settlement.rehypothecation.guard_collateral`
bounds its pledges against a **proven** reserve figure rather than a self-reported one.
But reserves are only one side of the ledger. A counterparty solvent against *one*
buyer's pledges may be deeply **under-water** once *every* obligation it owes is counted:
the guard sees the reserves and this buyer's pledges, never the counterparty's other
liabilities. A counterparty could prove the same reserves against many buyers while
quietly insolvent across all of them — the canonical gap the proof-of-reserves literature
closes next with a **proof-of-solvency** (reserves ≥ total liabilities). This module is
that reach: it makes the *liability* side evidence-backed too, and folds the two proofs
into a bounded, offline-verifiable solvency margin the guard bounds pledges against.

* **Signed liability attestation.** A counterparty (or its custodian) issues a
  :class:`LiabilityAttestation` over the total obligations it owes — itemized into one
  :class:`LiabilityLine` per creditor, so the attested ``liabilities_usd`` total re-derives
  from the components the way a reserve attestation's total does. It is the liability
  analogue of the proof-of-reserves: content-bound and signed with the same
  :class:`~vincio.security.audit.ChainSigner` a contract uses, so the claim is a mechanical,
  reconstructable artifact, never a counterparty's say-so.
* **Proof-of-solvency.** :func:`prove_solvency` folds a
  :class:`~vincio.settlement.custody.CustodyAttestation` against a
  :class:`LiabilityAttestation` into a :class:`SolvencyProof` — a bounded, offline-verifiable
  solvency margin (``reserves − liabilities``). When the proven reserves cannot cover the
  proven liabilities the shortfall surfaces as a pinpointed :class:`InsolvencyBreach` rather
  than passing on a one-sided reserve claim, and the proof exposes a *solvency-adjusted* held
  figure (the unencumbered capital, ``max(0, reserves − liabilities)``) that
  :func:`~vincio.settlement.rehypothecation.guard_collateral` reads (``solvency=``) as the
  held figure — so a pledge is bounded against capital **not already owed elsewhere**.
* **Inclusion proofs & completeness.** The liability *total* is still the attestor's single
  number: a counterparty could **under-state** what it owes by quietly omitting a creditor and
  still attest a sound, re-deriving total over the creditors it *did* list. So the attestation
  commits its line items into a Merkle root bound in the signed hash, and each creditor gets an
  offline-verifiable :class:`InclusionProof` that its claim is a leaf of that root
  (:meth:`LiabilityAttestation.inclusion_proof`) — a poster cannot drop a creditor without the
  omitted party detecting it. :func:`check_completeness` folds a set of creditor claims (what
  each can prove it is owed, e.g. from its own settled records) against the attestation into a
  signed :class:`CompletenessProof`, pinpointing every omitted or under-stated claim as an
  :class:`OmissionBreach` and raising the attested figure to a **completed** total that
  :func:`prove_solvency` reads (``completeness=``) — so the solvency margin is bounded by the
  obligations creditors can prove, not only the ones the attestor chose to list.
* **Root consistency & non-equivocation.** Completeness catches an omission only when the omitted
  creditor folds its *own* claim. A counterparty issues its liability attestation per relationship,
  so it can instead **equivocate** — sign a *smaller* root for one creditor and a different one for
  another, each creditor's :class:`InclusionProof` verifying against the root *it* was shown while
  the totals disagree across the set. :meth:`LiabilityAttestation.root_commitment` produces the
  signed :class:`RootCommitment` creditors compare over the attestation exchange — the root and the
  ``as_of`` the attestor signed, **without** the line items — and :func:`check_root_consistency`
  groups a set of held attestations by their ``(poster, attestor, as_of)`` key and folds any two
  conflicting roots into an :class:`EquivocationProof`: a content-bound, offline-verifiable breach
  pinning the poster, the two signed roots, and the creditors each was shown, so a counterparty
  signing inconsistent totals is caught with non-repudiable evidence rather than merely suspected.
* **History consistency & snapshot monotonicity.** Non-equivocation is scoped to one ``as_of``:
  a counterparty can still issue a *later* snapshot that quietly **drops** a past obligation — a
  debt committed at ``T`` simply absent from the root it signs at ``T'`` — each snapshot internally
  sound, nothing tying one attestation to its predecessor. A :class:`LiabilityAttestation` now
  carries an optional commitment to the prior snapshot's root (:meth:`LiabilityAttestation.link_to`,
  ``attest_liabilities(..., prior=)``), bound into the signed hash, so a poster's attestations form a
  hash-linked sequence a creditor walks, each ``as_of`` strictly succeeding the last (a back-dated
  link is caught from the bytes). :func:`check_history_consistency` walks a poster's snapshots in
  order and folds them into a signed :class:`HistoryConsistencyProof` that re-derives each
  per-creditor obligation from the embedded snapshots: an obligation that **shrinks** between
  snapshots is legitimate only when a signed, creditor-issued :class:`Discharge` (``discharge_liability``)
  evidences the release, and any unexplained drop surfaces as a pinpointed :class:`MonotonicityBreach`
  — so a debt cannot silently vanish between snapshots.
* **Auditable & offline.** Both attestations read only signed, content-bound artifacts:
  :meth:`LiabilityAttestation.verify` recomputes the content hash and re-derives the
  liability total from the line items, so a tampered figure is caught even after re-sealing,
  and a forged issuer signature is caught with the verifier. :func:`prove_solvency` refuses a
  tampered attestation, a forged issuer, or a custody / liability pair for *different* posters,
  and :meth:`SolvencyProof.verify` re-derives the margin and the insolvency breach from the
  bytes alone. The :class:`~vincio.settlement.book.SettlementBook` /
  :class:`~vincio.core.app.ContextApp` path lands each issuance and proof on the hash-chained
  audit log. Never a hosted solvency auditor or a trusted third party — a verifiable
  proof-of-solvency over the obligations and reserves the fabric already attests.

The proof folds into the *existing* guard path: :func:`attest_liabilities` builds a liability
proof, :func:`prove_solvency` folds it against a reserve proof, and
:func:`~vincio.settlement.rehypothecation.guard_collateral` reads the result as the held figure
in one call (:meth:`~vincio.core.app.ContextApp.attest_liabilities` /
:meth:`~vincio.core.app.ContextApp.prove_solvency`). Everything is dependency-free,
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

if TYPE_CHECKING:
    from ..security.audit import ChainSigner

__all__ = [
    "LiabilityLine",
    "LiabilityAttestationVerification",
    "LiabilityAttestation",
    "attest_liabilities",
    "MerkleStep",
    "InclusionProofVerification",
    "InclusionProof",
    "OmissionBreach",
    "CompletenessVerification",
    "CompletenessProof",
    "check_completeness",
    "InsolvencyBreach",
    "SolvencyProofVerification",
    "SolvencyProof",
    "prove_solvency",
    "RootCommitmentVerification",
    "RootCommitment",
    "EquivocationProofVerification",
    "EquivocationProof",
    "RootConsistencyReport",
    "prove_equivocation",
    "check_root_consistency",
    "DischargeVerification",
    "Discharge",
    "discharge_liability",
    "MonotonicityBreach",
    "HistoryConsistencyProofVerification",
    "HistoryConsistencyProof",
    "HistoryConsistencyReport",
    "check_history_consistency",
]

# The audit action a liability attestation is recorded under; the decision field carries
# whether the attestation is self-attested (``self_attested`` / ``attested``).
LIABILITY_ACTION = "liability_attestation"

# The audit action a completeness check is recorded under; the decision field carries whether
# the attested liabilities are provably complete (``complete`` / ``incomplete``).
COMPLETENESS_ACTION = "liability_completeness"

# The audit action a solvency proof is recorded under; the decision field carries whether the
# counterparty is solvent (``solvent`` / ``insolvent``).
SOLVENCY_ACTION = "solvency_proof"

# The audit action a root-consistency check is recorded under; the decision field carries whether
# the poster's signed liability roots are mutually consistent (``consistent`` / ``equivocation``).
EQUIVOCATION_ACTION = "liability_equivocation"

# The audit action a history-consistency check is recorded under; the decision field carries whether
# the poster's snapshot history is monotone (``consistent`` / ``inconsistent``).
HISTORY_ACTION = "liability_history"

# The audit action a liability discharge is recorded under; the decision field carries whether the
# discharge is partial (``partial``) or releases nothing (``empty``).
DISCHARGE_ACTION = "liability_discharge"

_TOLERANCE = 1e-6


def _r6(value: float) -> float:
    """Round a money figure to six places so float drift never breaks a balance."""
    return round(float(value), 6)


# -- Merkle commitment over the liability line items --------------------------
#
# A proof-of-liabilities is complete only if each creditor can prove its own claim is one of
# the leaves the attested total was summed over — otherwise a poster could quietly omit a
# creditor and still attest a sound, re-deriving total over the creditors it *did* list. The
# attestation commits its line items into a binary Merkle tree and binds the root into the
# signed content hash, so each creditor gets a compact :class:`InclusionProof` of membership
# that verifies against that one signed root. Leaf and interior hashes are domain-separated
# (distinct tags) so an interior node can never be presented as a leaf — the classic Merkle
# second-preimage guard — and the leaf binds the creditor's *position*, so a reordering cannot
# substitute one creditor's claim for another's.

_LEAF_TAG = "vincio.liability.leaf"
_NODE_TAG = "vincio.liability.node"

# The root of an empty liability set (no line items). Distinct from any leaf or interior node
# so a zero-creditor attestation still commits to a well-defined, domain-separated root.
_EMPTY_ROOT = stable_hash({"tag": _NODE_TAG, "empty": True}, length=32)


def _leaf_hash(index: int, creditor: str, amount_usd: float, note: str) -> str:
    """The domain-separated hash of one liability leaf, binding its sorted position."""
    return stable_hash(
        {
            "tag": _LEAF_TAG,
            "index": index,
            "creditor": creditor,
            "amount_usd": _r6(amount_usd),
            "note": note,
        },
        length=32,
    )


def _node_hash(left: str, right: str) -> str:
    """The domain-separated hash of one interior Merkle node from its two children."""
    return stable_hash({"tag": _NODE_TAG, "left": left, "right": right}, length=32)


def _merkle_levels(leaves: list[str]) -> list[list[str]]:
    """Build the Merkle tree bottom-up, returning every level (leaves first, root last).

    An odd node at a level is paired with itself (the standard duplicate-last rule), so the
    tree is well-defined for any leaf count. An empty leaf set commits to :data:`_EMPTY_ROOT`.
    """
    if not leaves:
        return [[_EMPTY_ROOT]]
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        current = levels[-1]
        nxt: list[str] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1] if i + 1 < len(current) else current[i]
            nxt.append(_node_hash(left, right))
        levels.append(nxt)
    return levels


def _merkle_root(leaves: list[str]) -> str:
    """The Merkle root committing to ``leaves`` (the empty root for no leaves)."""
    return _merkle_levels(leaves)[-1][0]


def _merkle_path(leaves: list[str], index: int) -> list[tuple[str, bool]]:
    """The authentication path for the leaf at ``index`` — ``(sibling, sibling_on_right)`` steps.

    Walking the path from the leaf up, combining the running hash with each sibling on the
    recorded side, reconstructs the root. ``sibling_on_right`` is ``True`` when the sibling is
    the *right* child (the running hash is the left), so the verifier folds the pair in the
    same order the tree was built.
    """
    levels = _merkle_levels(leaves)
    steps: list[tuple[str, bool]] = []
    idx = index
    for level in levels[:-1]:  # every level except the root
        if idx % 2 == 0:
            sibling = level[idx + 1] if idx + 1 < len(level) else level[idx]
            steps.append((sibling, True))
        else:
            steps.append((level[idx - 1], False))
        idx //= 2
    return steps


def _merkle_apply(leaf: str, steps: list[tuple[str, bool]]) -> str:
    """Fold a leaf up through its authentication ``steps`` to reconstruct the root."""
    current = leaf
    for sibling, sibling_on_right in steps:
        current = _node_hash(current, sibling) if sibling_on_right else _node_hash(sibling, current)
    return current


# -- liability attestation ----------------------------------------------------


class LiabilityLine(BaseModel):
    """One obligation owed, backing the poster's attested total liabilities.

    A proof-of-liabilities is itemized into its components — each obligation owed to a
    ``creditor`` for ``amount_usd`` — so the attested total re-derives from the line items
    and a tampered total is caught even after re-sealing, the way a
    :class:`~vincio.settlement.custody.ReserveLine`'s holding does for reserves. The ``note``
    is free-text provenance (the obligation's reference) and is bound into the attestation
    hash like every other field.
    """

    creditor: str
    amount_usd: float = 0.0
    note: str = ""

    def facts(self) -> dict[str, Any]:
        """The per-line facts the attestation's content hash binds."""
        return {
            "creditor": self.creditor,
            "amount_usd": _r6(self.amount_usd),
            "note": self.note,
        }


class LiabilityAttestationVerification(BaseModel):
    """The (non-raising) outcome of verifying a liability attestation offline.

    An attestation is **valid** when its content hash recomputes (``hash_ok``), the attested
    liability total re-derives from the line items and every obligation is non-negative
    (``liabilities_sound``), and — with a ``verifier`` — every signature checks
    (``signatures_ok``). A tampered liability figure or a forged issuer is caught from the
    bytes alone, even after re-sealing.
    """

    valid: bool
    hash_ok: bool
    liabilities_sound: bool
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class LiabilityAttestation(BaseModel):
    """A signed, offline-verifiable proof-of-liabilities over a poster's total obligations.

    Produced by :func:`attest_liabilities` (or
    :meth:`~vincio.settlement.book.SettlementBook.attest_liabilities` /
    :meth:`~vincio.core.app.ContextApp.attest_liabilities`): an ``attestor`` attests the total
    obligations a ``poster`` owes, itemized into :class:`LiabilityLine`\\ s whose amounts sum
    to ``liabilities_usd``. It binds the attestor, the poster, the liability line items, and
    the total onto a content hash, so the claim is a mechanical number anyone recomputes —
    the liability analogue of a :class:`~vincio.settlement.custody.CustodyAttestation`.

    When ``attestor == poster`` the attestation is **self-attested** — the poster's own signed
    liability record, still content-bound and non-repudiable rather than a bare number;
    otherwise an independent attestor (e.g. an auditor or custodian) vouches for the
    obligations. :func:`prove_solvency` folds ``liabilities_usd`` against a proven reserve
    figure into a solvency margin. :meth:`verify` re-derives the total from the line items and
    checks the attestor signature from the bytes alone.
    """

    id: str = Field(default_factory=lambda: new_id("liability"))
    attestor: str
    poster: str
    liabilities: list[LiabilityLine] = Field(default_factory=list)
    liabilities_usd: float = 0.0
    liabilities_root: str = ""

    as_of: datetime = Field(default_factory=utcnow)
    # -- linked liability history (optional predecessor commitment) ---------
    # When set, these pin the immediately preceding snapshot this attestation succeeds, so a
    # poster's attestations form a hash-linked sequence a creditor can walk. They are bound into
    # the signed content hash *only when present* (see :meth:`attestation_facts`), so an
    # attestation with no predecessor hashes exactly as before — the link is additive.
    prior_hash: str = ""
    prior_root: str = ""
    prior_as_of: datetime | None = None
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def self_attested(self) -> bool:
        """Whether the poster attests its own liabilities (attestor is the poster)."""
        return self.attestor == self.poster

    @property
    def creditors(self) -> list[str]:
        """The creditors the obligations are owed to, sorted."""
        return sorted(line.creditor for line in self.liabilities)

    @property
    def has_prior(self) -> bool:
        """Whether this attestation commits to a predecessor snapshot (a linked history)."""
        return bool(self.prior_hash)

    def _sorted_lines(self) -> list[LiabilityLine]:
        """The line items in the canonical (creditor-sorted) order the commitment uses."""
        return sorted(self.liabilities, key=lambda line: line.creditor)

    def _liabilities_total(self) -> float:
        """The liability total re-derived from the line items."""
        return _r6(sum(line.amount_usd for line in self.liabilities))

    # -- Merkle commitment --------------------------------------------------

    def _liability_leaves(self) -> list[str]:
        """The Merkle leaf hashes, one per line item, in canonical creditor-sorted order."""
        return [
            _leaf_hash(index, line.creditor, line.amount_usd, line.note)
            for index, line in enumerate(self._sorted_lines())
        ]

    def compute_root(self) -> str:
        """The Merkle root committing to the line items (the empty root for no lines).

        Each creditor's claim is a leaf of this root, so an :class:`InclusionProof` proves a
        claim is part of the attested total against a single signed value — the second half of
        a proof-of-liabilities, where the total is provably **complete**, not merely internally
        consistent.
        """
        return _merkle_root(self._liability_leaves())

    # -- hashing ------------------------------------------------------------

    def attestation_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the attestor, poster, lines, and total.

        Excludes the id, signatures, and audit linkage (local metadata, not the proof), so the
        same attestor attesting the same obligations for the same poster as of the same instant
        hashes identically wherever it is recomputed. Lines are sorted by creditor so the order
        they were listed in never changes the hash.
        """
        facts: dict[str, Any] = {
            "attestor": self.attestor,
            "poster": self.poster,
            "liabilities_usd": _r6(self.liabilities_usd),
            "liabilities_root": self.liabilities_root,
            "as_of": self.as_of.isoformat(),
            "liabilities": [line.facts() for line in self._sorted_lines()],
        }
        # The predecessor commitment is bound into the hash only when present, so an attestation
        # with no prior snapshot hashes identically to one issued before linked history existed —
        # the link is purely additive and never perturbs a standalone attestation's content hash.
        if self.has_prior:
            facts["prior"] = {
                "prior_hash": self.prior_hash,
                "prior_root": self.prior_root,
                "prior_as_of": self.prior_as_of.isoformat() if self.prior_as_of else "",
            }
        return facts

    def compute_hash(self) -> str:
        """The content hash binding the attestor, the poster, the liabilities, and the root."""
        return stable_hash(self.attestation_facts(), length=32)

    def seal(self) -> LiabilityAttestation:
        """Stamp the Merkle root and the content hash from the current fields (idempotent)."""
        self.liabilities_root = self.compute_root()
        self.content_hash = self.compute_hash()
        return self

    # -- linked liability history -------------------------------------------

    def link_to(self, prior: LiabilityAttestation) -> LiabilityAttestation:
        """Commit this attestation to its predecessor ``prior``, forming a hash-linked history.

        Pins ``prior``'s content hash, root, and ``as_of`` into this attestation (re-sealing so
        they bind the content hash), so a creditor walking the chain knows it has the contiguous
        sequence and :func:`check_history_consistency` can verify each ``as_of`` strictly succeeds
        the last. The predecessor must be the **same** ``(poster, attestor)`` and strictly earlier
        in time; a different counterparty or a back-dated predecessor is refused. Seals ``prior``
        first if needed. Returns ``self``.
        """
        if not prior.content_hash or not prior.liabilities_root:
            prior.seal()
        if prior.poster != self.poster or prior.attestor != self.attestor:
            raise SettlementError(
                f"cannot link liability attestation for {self.poster!r}/{self.attestor!r} to a "
                f"predecessor for {prior.poster!r}/{prior.attestor!r}; a history is one "
                "counterparty's sequence",
                details={
                    "poster": self.poster,
                    "attestor": self.attestor,
                    "prior_poster": prior.poster,
                    "prior_attestor": prior.attestor,
                },
            )
        if self.as_of <= prior.as_of:
            raise SettlementError(
                f"cannot link a snapshot as of {self.as_of.isoformat()} to a predecessor as of "
                f"{prior.as_of.isoformat()}; a successor must be strictly later (no back-dating)",
                details={"as_of": self.as_of.isoformat(), "prior_as_of": prior.as_of.isoformat()},
            )
        self.prior_hash = prior.content_hash
        self.prior_root = prior.liabilities_root
        self.prior_as_of = prior.as_of
        return self.seal()

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str | None = None) -> LiabilityAttestation:
        """Add the attestor's signature over the content hash (sealing first).

        A liability attestation is the *attestor's* claim, so only the attestor signs it
        (``party`` defaults to the attestor; passing a different party is refused). Re-signing
        replaces the prior signature, so an attestation cannot accumulate stale signatures.
        """
        resolved = party or self.attestor
        if resolved != self.attestor:
            raise SettlementError(
                f"a liability attestation is signed by its attestor {self.attestor!r}, "
                f"not {resolved!r}",
                details={"attestation_id": self.id, "attestor": self.attestor, "party": resolved},
            )
        if not self.content_hash:
            self.seal()
        sig = SettlementSignature(
            party=resolved,
            signature=signer.sign(self.content_hash),
            key_id=getattr(signer, "key_id", ""),
        )
        self.signatures = [s for s in self.signatures if s.party != resolved]
        self.signatures.append(sig)
        return self

    # -- verification -------------------------------------------------------

    def _liabilities_sound(self) -> bool:
        """The total and Merkle root re-derive from the line items and no obligation is negative.

        Re-deriving the root as well as the total means a tampered, dropped, or reordered line
        item is caught from the bytes alone, even after re-sealing — so the commitment each
        :class:`InclusionProof` verifies against is exactly the one the attestor signed.
        """
        if any(line.amount_usd < -_TOLERANCE for line in self.liabilities):
            return False
        if abs(self.liabilities_usd - self._liabilities_total()) > _TOLERANCE:
            return False
        if not self._prior_link_sound():
            return False
        return self.liabilities_root == self.compute_root()

    def _prior_link_sound(self) -> bool:
        """The predecessor commitment, when present, is well-formed and strictly in the past.

        A linked snapshot must succeed its predecessor *in time*: ``as_of`` strictly later than
        ``prior_as_of``. A back-dated link (a snapshot claiming to follow a *later* one) is caught
        from the bytes alone, so a poster cannot re-order its own history. A partial link (a prior
        hash without a prior root or instant, or vice versa) is malformed and refused.
        """
        present = (bool(self.prior_hash), bool(self.prior_root), self.prior_as_of is not None)
        if not any(present):
            return True
        if not all(present):
            return False
        return self.prior_as_of is None or self.as_of > self.prior_as_of

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> LiabilityAttestationVerification:
        """Verify the attestation offline: the hash recomputes and the liabilities re-derive.

        Recomputes the content hash and re-derives the liability total from the line items
        (checking every obligation is non-negative) — so a tampered liability figure is caught
        even when the hash was recomputed to match. ``verifier`` additionally checks each
        signature against the content hash; ``require`` names parties that must have a verified
        signature (defaults to none — pass ``[attestor]`` to demand the attestor's signature).
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        liabilities_sound = self._liabilities_sound()
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
        valid = hash_ok and liabilities_sound and signatures_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "attestation is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the attestation facts"
            elif not liabilities_sound:
                reason = "liability total or Merkle root does not re-derive from the line items"
            elif missing:
                reason = f"missing/invalid signatures for {missing}"
            else:
                reason = "signature mismatch"
        return LiabilityAttestationVerification(
            valid=valid,
            hash_ok=hash_ok,
            liabilities_sound=liabilities_sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> LiabilityAttestation:
        """Verify and raise :class:`SettlementError` if the attestation is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"liability attestation {self.id} failed verification: {result.reason}",
                details={"attestation_id": self.id, "reason": result.reason},
            )
        return self

    # -- inclusion proofs ---------------------------------------------------

    def inclusion_proof(self, creditor: str) -> InclusionProof:
        """Build an offline-verifiable :class:`InclusionProof` for one creditor's claim.

        The proof shows ``creditor``'s obligation is a leaf of the attestation's signed Merkle
        root, so the creditor — verifying the proof against the same signed attestation — knows
        its claim was counted in the attested total. A poster cannot drop a creditor without
        that creditor's :meth:`check_completeness` detecting the omission. Seals the attestation
        first if needed. Raises :class:`SettlementError` if the creditor is not among the line
        items, or appears in more than one (an ambiguous claim — fold the lines first).
        """
        if not self.content_hash or not self.liabilities_root:
            self.seal()
        lines = self._sorted_lines()
        matches = [(index, line) for index, line in enumerate(lines) if line.creditor == creditor]
        if not matches:
            raise SettlementError(
                f"creditor {creditor!r} is not among the attested liabilities of {self.poster!r}; "
                "no inclusion proof can be built",
                details={"attestation_id": self.id, "creditor": creditor},
            )
        if len(matches) > 1:
            raise SettlementError(
                f"creditor {creditor!r} appears in {len(matches)} line items of attestation "
                f"{self.id}; fold them into one before proving inclusion",
                details={"attestation_id": self.id, "creditor": creditor, "lines": len(matches)},
            )
        index, line = matches[0]
        leaves = self._liability_leaves()
        path = [MerkleStep(sibling=s, sibling_on_right=r) for s, r in _merkle_path(leaves, index)]
        return InclusionProof(
            attestation_id=self.id,
            poster=self.poster,
            attestor=self.attestor,
            liability_hash=self.content_hash,
            liabilities_root=self.liabilities_root,
            liabilities_usd=_r6(self.liabilities_usd),
            leaf_count=len(leaves),
            leaf_index=index,
            creditor=line.creditor,
            amount_usd=_r6(line.amount_usd),
            note=line.note,
            path=path,
            as_of=self.as_of,
        )

    def inclusion_proofs(self) -> list[InclusionProof]:
        """An :class:`InclusionProof` for every line item, in canonical creditor-sorted order.

        Builds one proof per leaf (so a creditor appearing in more than one line item still
        gets a proof for each), the way a custodian publishes a per-creditor liabilities tree.
        """
        if not self.content_hash or not self.liabilities_root:
            self.seal()
        lines = self._sorted_lines()
        leaves = self._liability_leaves()
        proofs: list[InclusionProof] = []
        for index, line in enumerate(lines):
            path = [
                MerkleStep(sibling=s, sibling_on_right=r) for s, r in _merkle_path(leaves, index)
            ]
            proofs.append(
                InclusionProof(
                    attestation_id=self.id,
                    poster=self.poster,
                    attestor=self.attestor,
                    liability_hash=self.content_hash,
                    liabilities_root=self.liabilities_root,
                    liabilities_usd=_r6(self.liabilities_usd),
                    leaf_count=len(leaves),
                    leaf_index=index,
                    creditor=line.creditor,
                    amount_usd=_r6(line.amount_usd),
                    note=line.note,
                    path=path,
                    as_of=self.as_of,
                )
            )
        return proofs

    # -- root commitment ----------------------------------------------------

    def root_commitment(self) -> RootCommitment:
        """A compact, signed digest of this attestation's root for cross-creditor comparison.

        The privacy-preserving artifact a creditor shares over the attestation exchange to
        compare the ``liabilities_root`` (and ``as_of``) a poster signed *for it* against the root
        the poster signed for *another* creditor — **without** revealing its line items. It
        carries the signed ``content_hash`` and the attestor's signature, so a peer confirms the
        attestor authored this root (:meth:`RootCommitment.verify`) but learns nothing of the
        obligations behind it. Two commitments a poster signed for the same ``as_of`` with
        **different** roots are a detected equivocation (:meth:`RootCommitment.conflicts_with`)
        that :func:`prove_equivocation` turns into a non-repudiable :class:`EquivocationProof` from
        the two full attestations. Seals the attestation first if needed.
        """
        if not self.content_hash or not self.liabilities_root:
            self.seal()
        attestor_sig = next((s for s in self.signatures if s.party == self.attestor), None)
        return RootCommitment(
            poster=self.poster,
            attestor=self.attestor,
            as_of=self.as_of,
            liabilities_root=self.liabilities_root,
            liabilities_usd=_r6(self.liabilities_usd),
            liability_hash=self.content_hash,
            signature=attestor_sig,
        )

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the attestation for the audit chain."""
        return to_jsonable(
            {
                "attestation_id": self.id,
                "attestor": self.attestor,
                "poster": self.poster,
                "self_attested": self.self_attested,
                "liabilities_usd": _r6(self.liabilities_usd),
                "liabilities_root": self.liabilities_root,
                "creditors": len(self.liabilities),
                "prior_hash": self.prior_hash,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> LiabilityAttestation:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the attested liabilities and the creditors."""
        kind = "self-attested" if self.self_attested else f"attested by {self.attestor}"
        print(
            f"Liability attestation ({self.poster}): ${self.liabilities_usd:,.2f} owed "
            f"across {len(self.liabilities)} creditor(s) — {kind}"
        )
        for line in sorted(self.liabilities, key=lambda line: line.creditor):
            note = f" ({line.note})" if line.note else ""
            print(f"  {line.creditor}: ${line.amount_usd:,.2f}{note}")


def _coerce_liabilities(liabilities: Any) -> list[LiabilityLine]:
    """Normalize a liabilities spec into :class:`LiabilityLine`\\ s.

    Accepts a single number (one unnamed obligation), a mapping of ``creditor -> amount``, or
    an iterable of :class:`LiabilityLine` / ``(creditor, amount)`` pairs / dicts. Raises
    :class:`SettlementError` on a negative obligation so a proof-of-liabilities can never net a
    real debt against a fictitious negative balance.
    """
    lines: list[LiabilityLine] = []
    if isinstance(liabilities, (int, float)):
        lines = [LiabilityLine(creditor="liabilities", amount_usd=float(liabilities))]
    elif isinstance(liabilities, dict):
        lines = [
            LiabilityLine(creditor=str(k), amount_usd=float(v)) for k, v in liabilities.items()
        ]
    else:
        for item in liabilities:
            if isinstance(item, LiabilityLine):
                lines.append(item)
            elif isinstance(item, dict):
                lines.append(LiabilityLine.model_validate(item))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                lines.append(LiabilityLine(creditor=str(item[0]), amount_usd=float(item[1])))
            else:
                raise SettlementError(
                    "attest_liabilities liabilities must be a number, a mapping, or "
                    f"LiabilityLine / (creditor, amount) items; got {item!r}",
                    details={"item": repr(item)},
                )
    for line in lines:
        if line.amount_usd < 0.0:
            raise SettlementError(
                f"liability to {line.creditor!r} owes a negative amount {line.amount_usd}; a "
                "proof-of-liabilities cannot net a real debt against a fictitious balance",
                details={"creditor": line.creditor, "amount_usd": line.amount_usd},
            )
    return lines


def attest_liabilities(
    poster: str,
    liabilities: Any,
    *,
    attestor: str | None = None,
    as_of: datetime | None = None,
    prior: LiabilityAttestation | None = None,
) -> LiabilityAttestation:
    """Attest a poster's total obligations into an (unsigned) :class:`LiabilityAttestation`.

    The proof-of-liabilities analogue of
    :func:`~vincio.settlement.custody.attest_custody`: ``attestor`` (defaulting to the
    ``poster`` itself — self-attested) vouches for the total obligations ``poster`` owes.
    ``liabilities`` is a single number (one unnamed obligation), a mapping of ``creditor ->
    amount``, or an iterable of :class:`LiabilityLine` / ``(creditor, amount)`` items; the
    attested ``liabilities_usd`` is their sum, re-derived on every verify.

    Pass ``prior`` (the immediately preceding snapshot) to link this attestation into a
    hash-linked history (:meth:`LiabilityAttestation.link_to`): its content hash, root, and
    ``as_of`` are bound into this attestation's signed hash, so :func:`check_history_consistency`
    can walk the sequence and catch a debt dropped between snapshots. ``prior`` must be the same
    ``(poster, attestor)`` and strictly earlier in time.

    Returns a sealed, unsigned attestation — sign it with the attestor's key (or let a
    :class:`~vincio.settlement.book.SettlementBook` do it on
    :meth:`~vincio.settlement.book.SettlementBook.attest_liabilities`). Raises
    :class:`SettlementError` when an obligation is negative or the ``prior`` link is invalid.
    """
    lines = _coerce_liabilities(liabilities)
    attestation = LiabilityAttestation(
        attestor=attestor or poster,
        poster=poster,
        liabilities=lines,
        liabilities_usd=_r6(sum(line.amount_usd for line in lines)),
        as_of=as_of or utcnow(),
    )
    attestation.seal()
    if prior is not None:
        attestation.link_to(prior)
    return attestation


# -- liability inclusion proofs ----------------------------------------------


class MerkleStep(BaseModel):
    """One step of an :class:`InclusionProof`'s authentication path.

    ``sibling`` is the hash of the co-node at this level; ``sibling_on_right`` is ``True`` when
    that sibling is the *right* child (the running hash is the left), so a verifier folds the
    pair in the order the tree was built.
    """

    sibling: str
    sibling_on_right: bool


class InclusionProofVerification(BaseModel):
    """The (non-raising) outcome of verifying a liability inclusion proof offline.

    A proof is **valid** when its authentication path reconstructs the committed root from the
    creditor's leaf (``path_ok``) and — when checked against the attestation it cites — that
    root is the one the attestor signed and the cited leaf is really one of its line items
    (``bound_ok``). A tampered leaf, a forged path, or a root that does not belong to the signed
    attestation is caught from the bytes alone.
    """

    valid: bool
    path_ok: bool
    bound_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class InclusionProof(BaseModel):
    """An offline-verifiable proof that one creditor's claim is in a liability attestation.

    Produced by :meth:`LiabilityAttestation.inclusion_proof` (or
    :meth:`~vincio.settlement.book.SettlementBook.inclusion_proof` /
    :meth:`~vincio.core.app.ContextApp.inclusion_proof`): it carries the creditor's obligation
    (``creditor`` / ``amount_usd`` / ``note``) and the Merkle authentication ``path`` from that
    leaf up to the attestation's committed ``liabilities_root`` — the root bound into the
    attestation's signed ``content_hash`` (pinned here as ``liability_hash``). A creditor
    verifies the proof against the signed attestation to confirm its claim was **counted** in
    the attested total, so a poster cannot quietly drop it.

    :meth:`verify` reconstructs the root from the leaf and the path; passing the attestation
    additionally checks the root and content hash match the signed one and the cited leaf is a
    real line item — so a tampered leaf or a forged root is caught from the bytes alone.
    """

    attestation_id: str
    poster: str
    attestor: str
    liability_hash: str
    liabilities_root: str
    liabilities_usd: float = 0.0

    leaf_count: int = 0
    leaf_index: int = 0
    creditor: str = ""
    amount_usd: float = 0.0
    note: str = ""
    path: list[MerkleStep] = Field(default_factory=list)
    as_of: datetime = Field(default_factory=utcnow)

    def _leaf(self) -> str:
        """The Merkle leaf this proof claims, recomputed from the creditor's claim."""
        return _leaf_hash(self.leaf_index, self.creditor, self.amount_usd, self.note)

    def _reconstructed_root(self) -> str:
        """The root the authentication path reconstructs from the recomputed leaf."""
        steps = [(step.sibling, step.sibling_on_right) for step in self.path]
        return _merkle_apply(self._leaf(), steps)

    def verify(
        self,
        attestation: LiabilityAttestation | None = None,
        verifier: ChainSigner | None = None,
    ) -> InclusionProofVerification:
        """Verify the proof offline: the path reconstructs the committed root.

        Recomputes the creditor's leaf and folds it up the authentication path, checking it
        reconstructs :attr:`liabilities_root`. When ``attestation`` is supplied the proof is
        additionally bound to it: the cited root and content hash must match the attestation's,
        the attestation itself must verify (with ``verifier`` its attestor signature too), and
        the cited leaf must be one the attestation actually commits to — so a root lifted from a
        different attestation, or a leaf the attestation never listed, is refused.
        """
        path_ok = (
            bool(self.liabilities_root)
            and 0 <= self.leaf_index < max(self.leaf_count, 1)
            and self._reconstructed_root() == self.liabilities_root
        )
        bound_ok = True
        signed_by: list[str] = []
        reason: str | None = None
        if attestation is not None:
            att_result = attestation.verify(verifier)
            signed_by = att_result.signed_by
            same_commitment = (
                attestation.poster == self.poster
                and attestation.attestor == self.attestor
                and attestation.content_hash == self.liability_hash
                and attestation.liabilities_root == self.liabilities_root
            )
            leaf_present = self._leaf() in attestation._liability_leaves()
            bound_ok = att_result.valid and same_commitment and leaf_present
            if not bound_ok:
                if not att_result.valid:
                    reason = f"bound attestation failed verification: {att_result.reason}"
                elif not same_commitment:
                    reason = "proof does not bind the supplied attestation's signed root"
                else:
                    reason = "cited claim is not a leaf of the attestation"
        valid = path_ok and bound_ok
        if not valid and reason is None:
            reason = (
                "authentication path does not reconstruct the committed root"
                if not path_ok
                else "inclusion proof is not bound to a valid attestation"
            )
        return InclusionProofVerification(
            valid=valid, path_ok=path_ok, bound_ok=bound_ok, signed_by=signed_by, reason=reason
        )

    def require_valid(
        self,
        attestation: LiabilityAttestation | None = None,
        verifier: ChainSigner | None = None,
    ) -> InclusionProof:
        """Verify and raise :class:`SettlementError` if the inclusion proof is not valid."""
        result = self.verify(attestation, verifier)
        if not result.valid:
            raise SettlementError(
                f"inclusion proof for {self.creditor!r} failed verification: {result.reason}",
                details={"attestation_id": self.attestation_id, "reason": result.reason},
            )
        return self

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> InclusionProof:
        return cls.model_validate(data)


# -- liability completeness ---------------------------------------------------


class OmissionBreach(BaseModel):
    """A creditor's proven claim the attested liabilities omit or under-state.

    Surfaced by :func:`check_completeness` when a creditor can prove it is owed more than the
    attestation attributes to it: ``claimed_usd`` is what the creditor proves (from its own
    settled records), ``attested_usd`` is what the attestation lists for it (``0`` when the
    creditor is omitted entirely — ``omitted`` is then ``True``), and ``understatement_usd`` is
    the gap (``claimed − attested``). The attestor could attest a sound, re-deriving total over
    the creditors it *did* list while quietly leaving this one out; the omission is what the
    completeness check pinpoints.
    """

    poster: str
    attestor: str = ""
    creditor: str
    attested_usd: float = 0.0
    claimed_usd: float = 0.0
    understatement_usd: float = 0.0
    omitted: bool = False


class CompletenessVerification(BaseModel):
    """The (non-raising) outcome of verifying a completeness check offline.

    A check is **valid** when its content hash recomputes (``hash_ok``), the completed liability
    total re-derives from the attested figure and the per-creditor understatements and each
    omission breach re-derives from the claimed and attested amounts (``completeness_sound``),
    and — with a ``verifier`` — every signature checks (``signatures_ok``). A hidden omission or
    a tampered completed total is caught from the bytes alone, even after re-sealing.
    """

    valid: bool
    hash_ok: bool
    completeness_sound: bool
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class CompletenessProof(BaseModel):
    """A signed, offline-verifiable completeness check over a liability attestation.

    Produced by :func:`check_completeness` (or
    :meth:`~vincio.settlement.book.SettlementBook.check_completeness` /
    :meth:`~vincio.core.app.ContextApp.check_completeness`): it folds a set of creditor claims
    — what each creditor can prove it is owed, e.g. from its own settled records — against a
    :class:`LiabilityAttestation`, pinpointing every claim the attestation omits or under-states
    as an :class:`OmissionBreach`. It binds the attestation (``liability_hash`` /
    ``liabilities_root``), the attestor's figure (``attested_usd``), the folded claims
    (``claimed_usd``), and the **completed** liability total (``completed_usd`` — the attested
    figure raised by every proven understatement) onto a content hash, so the check is a
    mechanical number anyone recomputes.

    The check is **complete** when no claim is omitted or under-stated. :func:`prove_solvency`
    reads ``completed_usd`` (``completeness=``) instead of the attestor's figure, so the solvency
    margin is bounded by the obligations creditors can *prove*, not only the ones the attestor
    chose to list. :meth:`verify` re-derives the completed total and the breaches from the bytes
    alone.
    """

    id: str = Field(default_factory=lambda: new_id("completeness"))
    poster: str
    attestor: str = ""
    liability_hash: str = ""
    liabilities_root: str = ""

    attested_usd: float = 0.0
    claimed_usd: float = 0.0
    completed_usd: float = 0.0
    breaches: list[OmissionBreach] = Field(default_factory=list)

    as_of: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def complete(self) -> bool:
        """Whether every folded claim is included in the attestation at its proven amount."""
        return not self.breaches

    @property
    def understated_usd(self) -> float:
        """How far the attested total falls below the completed total (``0`` when complete)."""
        return _r6(max(0.0, self.completed_usd - self.attested_usd))

    @property
    def status(self) -> str:
        """``complete`` (no omitted/under-stated claim) or ``incomplete`` (a proven omission)."""
        return "complete" if self.complete else "incomplete"

    @property
    def omitted_creditors(self) -> list[str]:
        """The creditors a proven claim shows are omitted or under-stated, sorted."""
        return sorted(breach.creditor for breach in self.breaches)

    def _expected_breaches(self) -> list[OmissionBreach]:
        """The omission breaches re-derived from the recorded claimed/attested figures."""
        derived: list[OmissionBreach] = []
        for breach in self.breaches:
            understatement = _r6(max(0.0, breach.claimed_usd - breach.attested_usd))
            derived.append(
                OmissionBreach(
                    poster=self.poster,
                    attestor=self.attestor,
                    creditor=breach.creditor,
                    attested_usd=_r6(breach.attested_usd),
                    claimed_usd=_r6(breach.claimed_usd),
                    understatement_usd=understatement,
                    omitted=breach.attested_usd <= _TOLERANCE,
                )
            )
        return derived

    # -- hashing ------------------------------------------------------------

    def completeness_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the attestation, the figures, and the breaches.

        Excludes the id, signatures, and audit linkage (local metadata), so the same claims
        folded against the same attestation hash identically wherever they are recomputed. The
        breaches are bound in (sorted by creditor) so a hidden omission is caught even after
        re-sealing.
        """
        return {
            "poster": self.poster,
            "attestor": self.attestor,
            "liability_hash": self.liability_hash,
            "liabilities_root": self.liabilities_root,
            "attested_usd": _r6(self.attested_usd),
            "claimed_usd": _r6(self.claimed_usd),
            "completed_usd": _r6(self.completed_usd),
            "as_of": self.as_of.isoformat(),
            "breaches": [
                {
                    "creditor": breach.creditor,
                    "attested_usd": _r6(breach.attested_usd),
                    "claimed_usd": _r6(breach.claimed_usd),
                    "understatement_usd": _r6(breach.understatement_usd),
                    "omitted": breach.omitted,
                }
                for breach in sorted(self.breaches, key=lambda b: b.creditor)
            ],
        }

    def compute_hash(self) -> str:
        """The content hash binding the attestation, the figures, and the omission breaches."""
        return stable_hash(self.completeness_facts(), length=32)

    def seal(self) -> CompletenessProof:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> CompletenessProof:
        """Add ``party``'s signature over the content hash (sealing first).

        A completeness check is signed by the creditor (or coordinator) that folded its claims,
        the party that vouches the attested liabilities omit a debt it can prove. Re-signing for
        the same party replaces its prior signature, so the check cannot accumulate stale ones.
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

    def _completeness_sound(self) -> bool:
        """The completed total and every breach re-derive from the recorded figures.

        Beyond re-deriving each breach and the completed total, enforces the structural
        invariant ``completed_usd >= claimed_usd``: the completed total absorbs every folded
        claim (each capped at what the attestation already lists for that creditor), so it can
        never sit below the claims it was built from. A forged check that drops an omission
        breach while keeping the claims it folded is caught by this alone.
        """
        if self.attested_usd < -_TOLERANCE or self.claimed_usd < -_TOLERANCE:
            return False
        if self.completed_usd < _r6(self.claimed_usd) - _TOLERANCE:
            return False
        expected = self._expected_breaches()
        if len(expected) != len(self.breaches):
            return False
        for got, want in zip(
            sorted(self.breaches, key=lambda b: b.creditor),
            sorted(expected, key=lambda b: b.creditor),
            strict=True,
        ):
            if got.creditor != want.creditor:
                return False
            if abs(got.understatement_usd - want.understatement_usd) > _TOLERANCE:
                return False
            if got.omitted != want.omitted:
                return False
            if want.understatement_usd <= _TOLERANCE:
                # A recorded "breach" with no understatement is not a real omission.
                return False
        expected_completed = _r6(self.attested_usd + sum(b.understatement_usd for b in expected))
        return abs(self.completed_usd - expected_completed) <= _TOLERANCE

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> CompletenessVerification:
        """Verify the check offline: the hash recomputes and the completed total re-derives.

        Recomputes the content hash and re-derives the completed liability total and every
        omission breach from the recorded claimed and attested figures — so a hidden omission
        or a tampered completed total is caught even when the hash was recomputed to match.
        ``verifier`` additionally checks each signature; ``require`` names parties that must have
        a verified signature (defaults to none).
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        completeness_sound = self._completeness_sound()
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
        valid = hash_ok and completeness_sound and signatures_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "completeness check is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the completeness facts"
            elif not completeness_sound:
                reason = "completed total or omission breach does not re-derive"
            elif missing:
                reason = f"missing/invalid signatures for {missing}"
            else:
                reason = "signature mismatch"
        return CompletenessVerification(
            valid=valid,
            hash_ok=hash_ok,
            completeness_sound=completeness_sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> CompletenessProof:
        """Verify and raise :class:`SettlementError` if the completeness check is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"completeness check {self.id} failed verification: {result.reason}",
                details={"check_id": self.id, "reason": result.reason},
            )
        return self

    def require_complete(self) -> CompletenessProof:
        """Raise :class:`SettlementError` if any folded claim is omitted or under-stated.

        The strict-mode counterpart to inspecting :attr:`complete`: a counterparty whose
        attested liabilities omit a debt a creditor can prove cannot be taken at its attested
        figure, and this pinpoints the omitted creditors and the understatement.
        """
        if self.breaches:
            raise SettlementError(
                f"completeness check {self.id} is incomplete: {self.poster!r} omits or "
                f"under-states ${self.understated_usd:,.2f} owed to {self.omitted_creditors}",
                details={
                    "check_id": self.id,
                    "poster": self.poster,
                    "understated_usd": self.understated_usd,
                    "omitted_creditors": self.omitted_creditors,
                },
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the completeness check for the audit chain."""
        return to_jsonable(
            {
                "check_id": self.id,
                "poster": self.poster,
                "attestor": self.attestor,
                "status": self.status,
                "attested_usd": _r6(self.attested_usd),
                "claimed_usd": _r6(self.claimed_usd),
                "completed_usd": _r6(self.completed_usd),
                "understated_usd": self.understated_usd,
                "omitted_creditors": self.omitted_creditors,
                "liability_hash": self.liability_hash,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> CompletenessProof:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the attested vs completed liabilities and any omitted creditors."""
        print(
            f"Completeness check ({self.poster}): attested ${self.attested_usd:,.2f}, "
            f"completed ${self.completed_usd:,.2f} against ${self.claimed_usd:,.2f} claimed "
            f"— {self.status}"
        )
        for breach in sorted(self.breaches, key=lambda b: b.creditor):
            kind = "omitted" if breach.omitted else "under-stated"
            print(
                f"  ! {breach.creditor}: {kind} — attested ${breach.attested_usd:,.2f} vs "
                f"claimed ${breach.claimed_usd:,.2f} (short ${breach.understatement_usd:,.2f})"
            )


def _coerce_claims(claims: Any) -> dict[str, float]:
    """Normalize a creditor-claims spec into a ``creditor -> proven amount`` mapping.

    Accepts a mapping of ``creditor -> amount``, or an iterable of :class:`LiabilityLine` /
    :class:`~vincio.settlement.record.SettlementRecord` / ``(creditor, amount)`` pairs / dicts.
    A :class:`~vincio.settlement.record.SettlementRecord` is read from the creditor's point of
    view — the seller is the creditor and ``amount_owed_usd`` is what it is owed. Amounts for
    the same creditor are summed. Raises :class:`SettlementError` on a negative claim.
    """
    from .record import SettlementRecord

    raw: list[tuple[str, float]] = []
    if isinstance(claims, dict):
        raw = [(str(creditor), float(amount)) for creditor, amount in claims.items()]
    else:
        for item in claims:
            if isinstance(item, LiabilityLine):
                raw.append((item.creditor, float(item.amount_usd)))
            elif isinstance(item, SettlementRecord):
                raw.append((item.seller, float(item.amount_owed_usd)))
            elif isinstance(item, dict):
                line = LiabilityLine.model_validate(item)
                raw.append((line.creditor, float(line.amount_usd)))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                raw.append((str(item[0]), float(item[1])))
            else:
                raise SettlementError(
                    "check_completeness claims must be a mapping, or LiabilityLine / "
                    f"SettlementRecord / (creditor, amount) items; got {item!r}",
                    details={"item": repr(item)},
                )
    merged: dict[str, float] = {}
    for creditor, amount in raw:
        if amount < 0.0:
            raise SettlementError(
                f"creditor {creditor!r} claims a negative amount {amount}; a completeness "
                "check cannot prove a debt against a fictitious negative claim",
                details={"creditor": creditor, "amount_usd": amount},
            )
        merged[creditor] = _r6(merged.get(creditor, 0.0) + amount)
    return merged


def check_completeness(
    liabilities: LiabilityAttestation,
    claims: Any,
    *,
    verifier: ChainSigner | None = None,
    as_of: datetime | None = None,
) -> CompletenessProof:
    """Fold a set of creditor claims against a liability attestation into a completeness check.

    The second half of a proof-of-liabilities: a liability attestation proves its total
    re-derives from the creditors it *lists*, but not that it lists *every* creditor. This folds
    what creditors can prove they are owed (``claims`` — a ``creditor -> amount`` mapping, or
    :class:`LiabilityLine` / :class:`~vincio.settlement.record.SettlementRecord` /
    ``(creditor, amount)`` items) against the attestation, pinpointing every claim it omits or
    under-states as an :class:`OmissionBreach` and raising the attested total to the
    **completed** total (the attested figure plus every proven understatement).

    Refuses a tampered attestation (its total or Merkle root no longer re-deriving) the way
    :func:`prove_solvency` does, and with ``verifier`` a forged attestor signature too. Returns
    a sealed, unsigned :class:`CompletenessProof` whose :attr:`~CompletenessProof.completed_usd`
    :func:`prove_solvency` reads (``completeness=``) so the solvency margin is bounded by the
    obligations creditors can prove, not only the ones the attestor listed.
    """
    result = liabilities.verify(verifier)
    if not result.hash_ok or not result.liabilities_sound:
        raise SettlementError(
            f"liability attestation {liabilities.id} is tampered ({result.reason}); refusing to "
            "check it for completeness",
            details={"attestation_id": liabilities.id, "reason": result.reason},
        )
    if verifier is not None and liabilities.signatures and not result.signatures_ok:
        raise SettlementError(
            f"liability attestation {liabilities.id} has an invalid attestor signature; refusing "
            "to check it for completeness",
            details={"attestation_id": liabilities.id},
        )

    attested_map: dict[str, float] = {}
    for line in liabilities.liabilities:
        attested_map[line.creditor] = _r6(attested_map.get(line.creditor, 0.0) + line.amount_usd)

    claim_map = _coerce_claims(claims)
    breaches: list[OmissionBreach] = []
    claimed_total = 0.0
    for creditor in sorted(claim_map):
        claimed = _r6(claim_map[creditor])
        claimed_total = _r6(claimed_total + claimed)
        attested = _r6(attested_map.get(creditor, 0.0))
        understatement = _r6(max(0.0, claimed - attested))
        if understatement > _TOLERANCE:
            breaches.append(
                OmissionBreach(
                    poster=liabilities.poster,
                    attestor=liabilities.attestor,
                    creditor=creditor,
                    attested_usd=attested,
                    claimed_usd=claimed,
                    understatement_usd=understatement,
                    omitted=attested <= _TOLERANCE,
                )
            )
    completed = _r6(liabilities.liabilities_usd + sum(b.understatement_usd for b in breaches))
    proof = CompletenessProof(
        poster=liabilities.poster,
        attestor=liabilities.attestor,
        liability_hash=liabilities.content_hash,
        liabilities_root=liabilities.liabilities_root,
        attested_usd=_r6(liabilities.liabilities_usd),
        claimed_usd=_r6(claimed_total),
        completed_usd=completed,
        breaches=breaches,
        as_of=as_of or utcnow(),
    )
    return proof.seal()


# -- proof-of-solvency --------------------------------------------------------


class InsolvencyBreach(BaseModel):
    """A proven shortfall: the obligations owed exceed the reserves actually held.

    Surfaced by :func:`prove_solvency` when the proven reserves cannot cover the proven
    liabilities: the ``poster`` holds ``reserves_usd`` (the custody attestation pinned by
    ``custody_hash``) but owes ``liabilities_usd`` (the liability attestation pinned by
    ``liability_hash``, attested by ``attestor``), so it is insolvent by ``shortfall_usd``
    (``liabilities − reserves``). The reserve proof alone could not catch this: it proves the
    capital exists, not that it exceeds every obligation the counterparty owes elsewhere.
    """

    poster: str
    custodian: str = ""
    attestor: str = ""
    custody_hash: str = ""
    liability_hash: str = ""
    reserves_usd: float = 0.0
    liabilities_usd: float = 0.0
    shortfall_usd: float = 0.0


class SolvencyProofVerification(BaseModel):
    """The (non-raising) outcome of verifying a solvency proof offline.

    A proof is **valid** when its content hash recomputes (``hash_ok``), the solvency margin
    and the solvency-adjusted held figure re-derive from the proven reserves and liabilities
    and the insolvency breach re-derives from the margin (``margin_sound``), and — with a
    ``verifier`` — every signature checks (``signatures_ok``). A tampered margin or a flipped
    solvency verdict is caught from the bytes alone, even after re-sealing.
    """

    valid: bool
    hash_ok: bool
    margin_sound: bool
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class SolvencyProof(BaseModel):
    """A signed, offline-verifiable proof-of-solvency over a poster's reserves and liabilities.

    Produced by :func:`prove_solvency` (or
    :meth:`~vincio.settlement.book.SettlementBook.prove_solvency` /
    :meth:`~vincio.core.app.ContextApp.prove_solvency`): it folds a proven
    :class:`~vincio.settlement.custody.CustodyAttestation` (``reserves_usd``, by ``custodian``)
    against a proven :class:`LiabilityAttestation` (``liabilities_usd``, by ``attestor``) for
    the same ``poster`` into a solvency ``margin_usd`` (``reserves − liabilities``). It binds
    both attestation hashes, the proven figures, and the margin onto a content hash, so the
    proof is a mechanical number anyone recomputes.

    The proof is **solvent** when ``margin_usd >= 0``; otherwise the shortfall surfaces as a
    pinpointed :class:`InsolvencyBreach`. :attr:`solvency_adjusted_held` is the unencumbered
    capital — ``max(0, margin)`` — that :func:`~vincio.settlement.rehypothecation.guard_collateral`
    reads (``solvency=``) as the held figure, so a pledge is bounded against capital **not
    already owed elsewhere**. :meth:`verify` re-derives the margin and the breach from the
    bytes alone.
    """

    id: str = Field(default_factory=lambda: new_id("solvency"))
    poster: str
    custodian: str = ""
    attestor: str = ""
    custody_hash: str = ""
    liability_hash: str = ""
    completeness_hash: str = ""

    reserves_usd: float = 0.0
    liabilities_usd: float = 0.0
    attested_liabilities_usd: float = 0.0
    margin_usd: float = 0.0

    as_of: datetime = Field(default_factory=utcnow)
    breach: InsolvencyBreach | None = None
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def solvent(self) -> bool:
        """Whether the proven reserves cover the proven liabilities (margin ≥ 0)."""
        return self.margin_usd >= -_TOLERANCE

    @property
    def insolvent(self) -> bool:
        """Whether the proven liabilities exceed the proven reserves (a shortfall)."""
        return not self.solvent

    @property
    def solvency_adjusted_held(self) -> float:
        """The unencumbered capital the guard bounds pledges against (``max(0, margin)``).

        The capital left once every proven obligation is covered — what the counterparty can
        back a *new* pledge with. Floored at zero, since an insolvent counterparty has no free
        capital to pledge (its shortfall is pinpointed separately as an
        :class:`InsolvencyBreach`).
        """
        return _r6(max(0.0, self.margin_usd))

    @property
    def status(self) -> str:
        """``solvent`` (reserves cover liabilities) or ``insolvent`` (a proven shortfall)."""
        return "solvent" if self.solvent else "insolvent"

    @property
    def completeness_adjusted(self) -> bool:
        """Whether the margin counts a completed liability total, not just the attestor's figure.

        ``True`` when :func:`prove_solvency` folded a :class:`CompletenessProof` in (``completeness=``),
        so :attr:`liabilities_usd` is the *completed* total — the attested figure raised by every
        omission a creditor proved — and the margin is bounded against obligations the attestor
        did not all list.
        """
        return bool(self.completeness_hash)

    @property
    def understated_usd(self) -> float:
        """How far the completed liabilities exceed the attestor's figure (``0`` when complete)."""
        return _r6(max(0.0, self.liabilities_usd - self.attested_liabilities_usd))

    def _derive_breach(self) -> InsolvencyBreach | None:
        """The insolvency breach when the proven liabilities exceed the proven reserves."""
        shortfall = _r6(max(0.0, self.liabilities_usd - self.reserves_usd))
        if shortfall <= _TOLERANCE:
            return None
        return InsolvencyBreach(
            poster=self.poster,
            custodian=self.custodian,
            attestor=self.attestor,
            custody_hash=self.custody_hash,
            liability_hash=self.liability_hash,
            reserves_usd=_r6(self.reserves_usd),
            liabilities_usd=_r6(self.liabilities_usd),
            shortfall_usd=shortfall,
        )

    # -- hashing ------------------------------------------------------------

    def solvency_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the parties, the proofs, and the margin.

        Excludes the id, signatures, and audit linkage (local metadata), so the same reserve
        and liability proofs folded for the same poster hash identically wherever they are
        recomputed — the way two parties co-sign one reconciliation hash. The insolvency breach
        is bound in (when present) so a flipped verdict is caught even after re-sealing.
        """
        return {
            "poster": self.poster,
            "custodian": self.custodian,
            "attestor": self.attestor,
            "custody_hash": self.custody_hash,
            "liability_hash": self.liability_hash,
            "completeness_hash": self.completeness_hash,
            "reserves_usd": _r6(self.reserves_usd),
            "liabilities_usd": _r6(self.liabilities_usd),
            "attested_liabilities_usd": _r6(self.attested_liabilities_usd),
            "margin_usd": _r6(self.margin_usd),
            "solvency_adjusted_held": self.solvency_adjusted_held,
            "as_of": self.as_of.isoformat(),
            "breach": (
                {
                    "poster": self.breach.poster,
                    "custodian": self.breach.custodian,
                    "attestor": self.breach.attestor,
                    "custody_hash": self.breach.custody_hash,
                    "liability_hash": self.breach.liability_hash,
                    "reserves_usd": _r6(self.breach.reserves_usd),
                    "liabilities_usd": _r6(self.breach.liabilities_usd),
                    "shortfall_usd": _r6(self.breach.shortfall_usd),
                }
                if self.breach is not None
                else None
            ),
        }

    def compute_hash(self) -> str:
        """The content hash binding the folded proofs and the solvency margin."""
        return stable_hash(self.solvency_facts(), length=32)

    def seal(self) -> SolvencyProof:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> SolvencyProof:
        """Add ``party``'s signature over the content hash (sealing first).

        A proof is signed by whoever folded it — the poster proving its own solvency, or a
        counterparty that independently re-folds the two attestations it was handed. Re-signing
        for the same party replaces its prior signature, so a proof cannot accumulate stale
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

    def _margin_sound(self) -> bool:
        """The margin re-derives from the proven figures and the breach re-derives from it."""
        if self.reserves_usd < -_TOLERANCE or self.liabilities_usd < -_TOLERANCE:
            return False
        # The completed liabilities can only raise the attestor's figure, never lower it — so a
        # forged "completion" that understates the debt is caught from the bytes alone.
        if self.attested_liabilities_usd < -_TOLERANCE:
            return False
        if self.liabilities_usd < _r6(self.attested_liabilities_usd) - _TOLERANCE:
            return False
        if abs(self.margin_usd - _r6(self.reserves_usd - self.liabilities_usd)) > _TOLERANCE:
            return False
        expected = self._derive_breach()
        if (expected is None) != (self.breach is None):
            return False
        if expected is not None and self.breach is not None:
            if (
                self.breach.poster != expected.poster
                or self.breach.custodian != expected.custodian
                or self.breach.attestor != expected.attestor
                or self.breach.custody_hash != expected.custody_hash
                or self.breach.liability_hash != expected.liability_hash
                or abs(self.breach.reserves_usd - expected.reserves_usd) > _TOLERANCE
                or abs(self.breach.liabilities_usd - expected.liabilities_usd) > _TOLERANCE
                or abs(self.breach.shortfall_usd - expected.shortfall_usd) > _TOLERANCE
            ):
                return False
        return True

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> SolvencyProofVerification:
        """Verify the proof offline: the hash recomputes and the margin re-derives.

        Recomputes the content hash and re-derives the solvency margin from the proven
        reserves and liabilities and the insolvency breach from the margin — so a tampered
        margin or a flipped solvency verdict is caught even when the hash was recomputed to
        match. ``verifier`` additionally checks each signature against the content hash;
        ``require`` names parties that must have a verified signature (defaults to none).
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        margin_sound = self._margin_sound()
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
        valid = hash_ok and margin_sound and signatures_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "proof is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the solvency facts"
            elif not margin_sound:
                reason = "solvency margin or insolvency breach does not re-derive"
            elif missing:
                reason = f"missing/invalid signatures for {missing}"
            else:
                reason = "signature mismatch"
        return SolvencyProofVerification(
            valid=valid,
            hash_ok=hash_ok,
            margin_sound=margin_sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> SolvencyProof:
        """Verify and raise :class:`SettlementError` if the proof is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"solvency proof {self.id} failed verification: {result.reason}",
                details={"proof_id": self.id, "reason": result.reason},
            )
        return self

    def require_solvent(self) -> SolvencyProof:
        """Raise :class:`SettlementError` if the proven liabilities exceed the reserves.

        The strict-mode counterpart to inspecting :attr:`solvent`: a counterparty whose
        liability attestation proves more debt than its custody attestation proves reserves
        cannot be admitted to a new deal without resolving the insolvency first, and this
        pinpoints the custodian, the attestor, and the shortfall.
        """
        if self.breach is not None:
            raise SettlementError(
                f"solvency proof {self.id} is insolvent by ${self.breach.shortfall_usd:,.2f}: "
                f"{self.poster!r} owes ${self.liabilities_usd:,.2f} against "
                f"${self.reserves_usd:,.2f} proven reserves",
                details={
                    "proof_id": self.id,
                    "poster": self.poster,
                    "reserves_usd": self.reserves_usd,
                    "liabilities_usd": self.liabilities_usd,
                    "shortfall_usd": self.breach.shortfall_usd,
                },
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the proof for the audit chain."""
        return to_jsonable(
            {
                "proof_id": self.id,
                "poster": self.poster,
                "custodian": self.custodian,
                "attestor": self.attestor,
                "status": self.status,
                "reserves_usd": _r6(self.reserves_usd),
                "liabilities_usd": _r6(self.liabilities_usd),
                "attested_liabilities_usd": _r6(self.attested_liabilities_usd),
                "completeness_adjusted": self.completeness_adjusted,
                "understated_usd": self.understated_usd,
                "margin_usd": _r6(self.margin_usd),
                "solvency_adjusted_held": self.solvency_adjusted_held,
                "shortfall_usd": (_r6(self.breach.shortfall_usd) if self.breach else 0.0),
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> SolvencyProof:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the proven reserves, liabilities, and the solvency margin."""
        print(
            f"Solvency proof ({self.poster}): ${self.reserves_usd:,.2f} reserves − "
            f"${self.liabilities_usd:,.2f} liabilities = ${self.margin_usd:,.2f} margin "
            f"({self.solvency_adjusted_held:,.2f} free) — {self.status}"
        )
        if self.completeness_adjusted and self.understated_usd > _TOLERANCE:
            print(
                f"  completeness-adjusted: attestor listed ${self.attested_liabilities_usd:,.2f}, "
                f"creditors proved ${self.understated_usd:,.2f} more"
            )
        if self.breach is not None:
            print(
                f"  ! insolvent by ${self.breach.shortfall_usd:,.2f}: proven liabilities "
                f"exceed the reserves"
            )


def _proven_reserves(
    custody: CustodyAttestation, *, poster: str, verifier: ChainSigner | None
) -> float:
    """Read a custody attestation's proven reserves, refusing a tampered or mismatched one.

    Reads only what it can recompute: an attestation whose content hash no longer recomputes
    or whose total no longer re-derives — a tampered reserve figure — is refused outright, and
    with a ``verifier`` a forged custodian signature is too. An attestation for a *different*
    poster cannot stand in for this poster's reserves and is refused. Returns ``reserves_usd``.
    """
    result = custody.verify(verifier)
    if not result.hash_ok or not result.reserves_sound:
        raise SettlementError(
            f"custody attestation {custody.id} is tampered ({result.reason}); refusing to read "
            "it as proof-of-reserves",
            details={"attestation_id": custody.id, "reason": result.reason},
        )
    if verifier is not None and custody.signatures and not result.signatures_ok:
        raise SettlementError(
            f"custody attestation {custody.id} has an invalid custodian signature; refusing to "
            "read it as proof-of-reserves",
            details={"attestation_id": custody.id},
        )
    if custody.poster != poster:
        raise SettlementError(
            f"custody attestation {custody.id} attests reserves for {custody.poster!r}, not the "
            f"poster {poster!r} the solvency proof folds; refusing it",
            details={"attestation_id": custody.id, "attests": custody.poster, "poster": poster},
        )
    return _r6(custody.reserves_usd)


def _proven_liabilities(
    liabilities: LiabilityAttestation, *, poster: str, verifier: ChainSigner | None
) -> float:
    """Read a liability attestation's proven total, refusing a tampered or mismatched one.

    The liability analogue of :func:`_proven_reserves`: an attestation whose hash no longer
    recomputes or whose total no longer re-derives is refused, with a ``verifier`` a forged
    attestor signature is too, and an attestation for a *different* poster is refused. Returns
    ``liabilities_usd``.
    """
    result = liabilities.verify(verifier)
    if not result.hash_ok or not result.liabilities_sound:
        raise SettlementError(
            f"liability attestation {liabilities.id} is tampered ({result.reason}); refusing to "
            "read it as proof-of-liabilities",
            details={"attestation_id": liabilities.id, "reason": result.reason},
        )
    if verifier is not None and liabilities.signatures and not result.signatures_ok:
        raise SettlementError(
            f"liability attestation {liabilities.id} has an invalid attestor signature; refusing "
            "to read it as proof-of-liabilities",
            details={"attestation_id": liabilities.id},
        )
    if liabilities.poster != poster:
        raise SettlementError(
            f"liability attestation {liabilities.id} attests liabilities for "
            f"{liabilities.poster!r}, not the poster {poster!r} the solvency proof folds; "
            "refusing it",
            details={
                "attestation_id": liabilities.id,
                "attests": liabilities.poster,
                "poster": poster,
            },
        )
    return _r6(liabilities.liabilities_usd)


def _completed_liabilities(
    completeness: CompletenessProof,
    liabilities: LiabilityAttestation,
    *,
    poster: str,
    verifier: ChainSigner | None,
) -> float:
    """Read a completeness check's completed total, refusing a tampered or mismatched one.

    The completeness analogue of :func:`_proven_liabilities`: a check whose hash no longer
    recomputes or whose completed total no longer re-derives is refused, with a ``verifier`` a
    forged signature is too, and a check that does not bind *this* poster's *this* liability
    attestation (same content hash and root) is refused — so a completion lifted from a
    different attestation cannot stand in. Returns the completed liability total.
    """
    result = completeness.verify(verifier)
    if not result.hash_ok or not result.completeness_sound:
        raise SettlementError(
            f"completeness check {completeness.id} is tampered ({result.reason}); refusing to "
            "read it as a completed liability total",
            details={"check_id": completeness.id, "reason": result.reason},
        )
    if verifier is not None and completeness.signatures and not result.signatures_ok:
        raise SettlementError(
            f"completeness check {completeness.id} has an invalid signature; refusing to read it "
            "as a completed liability total",
            details={"check_id": completeness.id},
        )
    if completeness.poster != poster:
        raise SettlementError(
            f"completeness check {completeness.id} is for {completeness.poster!r}, not the poster "
            f"{poster!r} the solvency proof folds; refusing it",
            details={"check_id": completeness.id, "checks": completeness.poster, "poster": poster},
        )
    if (
        completeness.liability_hash != liabilities.content_hash
        or completeness.liabilities_root != liabilities.liabilities_root
    ):
        raise SettlementError(
            f"completeness check {completeness.id} does not bind the liability attestation "
            f"{liabilities.id}; refusing to fold an unrelated completion",
            details={
                "check_id": completeness.id,
                "check_liability_hash": completeness.liability_hash,
                "attestation_hash": liabilities.content_hash,
            },
        )
    return _r6(completeness.completed_usd)


def prove_solvency(
    custody: CustodyAttestation,
    liabilities: LiabilityAttestation,
    *,
    poster: str | None = None,
    completeness: CompletenessProof | None = None,
    as_of: datetime | None = None,
    verifier: ChainSigner | None = None,
) -> SolvencyProof:
    """Fold a reserve proof against a liability proof into a proof-of-solvency.

    The proof-of-solvency the literature pairs with a proof-of-reserves: ``reserves ≥
    liabilities``. Reads a poster's :class:`~vincio.settlement.custody.CustodyAttestation`
    (proven reserves) and its :class:`LiabilityAttestation` (proven obligations), refusing
    either if its content hash no longer recomputes or its total no longer re-derives (a forged
    signature too, with ``verifier``), and reconciles them into a bounded solvency
    ``margin_usd`` (``reserves − liabilities``). When the liabilities exceed the reserves the
    shortfall is pinpointed as an :class:`InsolvencyBreach`. Returns a sealed, unsigned
    :class:`SolvencyProof` whose :attr:`~SolvencyProof.solvency_adjusted_held` the guard reads
    (``solvency=``) as the held figure — a pledge bounded against capital not already owed.

    Pass ``completeness`` (a :class:`CompletenessProof` over *this* attestation) to bound the
    margin against the **completed** liability total — the attestor's figure raised by every
    obligation a creditor proved it omitted — rather than the attestor's single number. A
    completeness check that is tampered, forged, or for a different poster / attestation is
    refused; ``attested_liabilities_usd`` records the attestor's original figure beside it.

    ``poster`` is the counterparty both attestations are about (defaults to the poster they
    share; an explicit poster is required when they differ, and both must attest it). Raises
    :class:`SettlementError` when the two attestations attest different posters and none is
    given, or when either is tampered, forged, or for the wrong poster.
    """
    resolved_poster = poster
    if resolved_poster is None:
        if custody.poster != liabilities.poster:
            raise SettlementError(
                "prove_solvency needs an explicit poster: the custody attestation attests "
                f"{custody.poster!r} but the liability attestation attests "
                f"{liabilities.poster!r}",
                details={"custody_poster": custody.poster, "liability_poster": liabilities.poster},
            )
        resolved_poster = custody.poster
    reserves_usd = _proven_reserves(custody, poster=resolved_poster, verifier=verifier)
    attested_usd = _proven_liabilities(liabilities, poster=resolved_poster, verifier=verifier)
    liabilities_usd = attested_usd
    completeness_hash = ""
    if completeness is not None:
        liabilities_usd = _completed_liabilities(
            completeness, liabilities, poster=resolved_poster, verifier=verifier
        )
        completeness_hash = completeness.content_hash
    proof = SolvencyProof(
        poster=resolved_poster,
        custodian=custody.custodian,
        attestor=liabilities.attestor,
        custody_hash=custody.content_hash,
        liability_hash=liabilities.content_hash,
        completeness_hash=completeness_hash,
        reserves_usd=reserves_usd,
        liabilities_usd=liabilities_usd,
        attested_liabilities_usd=attested_usd,
        margin_usd=_r6(reserves_usd - liabilities_usd),
        as_of=as_of or utcnow(),
    )
    proof.breach = proof._derive_breach()
    return proof.seal()


# -- root consistency & non-equivocation --------------------------------------


class RootCommitmentVerification(BaseModel):
    """The (non-raising) outcome of verifying a liability root commitment offline.

    A commitment is **valid** when it pins a non-empty root and content hash (``committed``) and —
    with a ``verifier`` — the attestor's signature over that content hash checks (``signed_ok``),
    so the root is attributable to the attestor. A commitment whose embedded signature is forged
    (does not verify against the attestor's key) is caught from the bytes alone.
    """

    valid: bool
    committed: bool
    signed_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class RootCommitment(BaseModel):
    """A compact, signed digest of one liability attestation's root, for cross-creditor compare.

    Built by :meth:`LiabilityAttestation.root_commitment`: the privacy-preserving artifact a
    creditor shares over the attestation exchange to compare the ``liabilities_root`` (and
    ``as_of``) a poster signed *for it* against the root the poster signed for *another* creditor,
    **without** revealing its line items. It carries the attestation's signed ``liability_hash``
    (the content hash, which already binds the root and the ``as_of``) and the attestor's
    ``signature`` over it, so a peer confirms the attestor authored this root
    (:meth:`verify`) but learns nothing of the obligations behind it.

    A commitment is a **detection** aid, not the proof: two commitments a poster signed for the
    same ``as_of`` with **different** roots are a detected equivocation (:meth:`conflicts_with`).
    The non-repudiable :class:`EquivocationProof` is then substantiated by :func:`prove_equivocation`
    from the two *full* attestations, which re-derive their roots from the bytes — closing the gap
    that a commitment alone, lacking the line items, cannot recompute its own hash.
    """

    poster: str
    attestor: str
    as_of: datetime = Field(default_factory=utcnow)
    liabilities_root: str = ""
    liabilities_usd: float = 0.0
    liability_hash: str = ""
    signature: SettlementSignature | None = None

    @property
    def consistency_key(self) -> tuple[str, str, str]:
        """The ``(poster, attestor, as_of)`` key two roots must share to be an equivocation.

        The ``as_of`` is the snapshot instant: two roots a poster signed *as of the same instant*
        are a contradiction, while two roots for *different* instants are distinct snapshots (a
        later one legitimately supersedes an earlier one) and the staleness horizon governs them.
        """
        return (self.poster, self.attestor, self.as_of.isoformat())

    @property
    def signed_by(self) -> list[str]:
        """The party that signed the underlying attestation (the attestor), if any."""
        return [self.signature.party] if self.signature is not None else []

    def conflicts_with(self, other: RootCommitment) -> bool:
        """Whether ``other`` is the same poster's root for the same ``as_of`` but a *different* root.

        The detection predicate: a poster equivocates when it signs two different
        ``liabilities_root`` values for the same ``(poster, attestor, as_of)`` — each creditor's
        inclusion proof verifies against the root *it* was shown while the totals disagree across
        the set. Confirm both commitments verify (the attestor really signed each) before treating
        the conflict as proven, then build the :class:`EquivocationProof` from the full attestations.
        """
        return (
            bool(self.liabilities_root)
            and bool(other.liabilities_root)
            and self.consistency_key == other.consistency_key
            and self.liabilities_root != other.liabilities_root
        )

    def verify(self, verifier: ChainSigner | None = None) -> RootCommitmentVerification:
        """Verify the commitment offline: it pins a root and the attestor's signature checks.

        ``committed`` holds when the commitment pins a non-empty root and content hash;
        ``verifier`` additionally checks the embedded attestor signature against the content hash,
        so a forged signature (one not produced with the attestor's key) is refused. Without a
        ``verifier`` the signature is taken as presented (the attestor named, unverified).
        """
        committed = bool(self.liabilities_root) and bool(self.liability_hash)
        signed_ok = True
        signed_by: list[str] = []
        if self.signature is not None:
            if verifier is not None:
                if verifier.verify(self.liability_hash, self.signature.signature):
                    signed_by = [self.signature.party]
                else:
                    signed_ok = False
            else:
                signed_by = [self.signature.party]
        valid = committed and signed_ok
        reason: str | None = None
        if not valid:
            reason = (
                "commitment pins no root or content hash"
                if not committed
                else "embedded attestor signature does not verify"
            )
        return RootCommitmentVerification(
            valid=valid,
            committed=committed,
            signed_ok=signed_ok,
            signed_by=signed_by,
            reason=reason,
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> RootCommitment:
        return cls.model_validate(data)


class EquivocationProofVerification(BaseModel):
    """The (non-raising) outcome of verifying a liability equivocation proof offline.

    A proof is **valid** when its content hash recomputes (``hash_ok``), the two embedded
    attestations each re-derive their root and total from the bytes — and, with a ``verifier``,
    carry the attestor's signature (``attestations_ok`` / ``attestor_signed``) — and they share
    one ``(poster, attestor, as_of)`` key while committing **different** roots (``conflict_ok``).
    A forged conflicting root (one the attestor never signed) is refused with the verifier, and a
    fabricated pairing of unrelated or identical attestations fails the conflict check.
    """

    valid: bool
    hash_ok: bool
    attestations_ok: bool
    conflict_ok: bool
    attestor_signed: bool = False
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class EquivocationProof(BaseModel):
    """A signed, offline-verifiable proof that a poster signed two conflicting liability roots.

    Produced by :func:`prove_equivocation` (or
    :meth:`~vincio.settlement.book.SettlementBook.check_root_consistency` /
    :meth:`~vincio.core.app.ContextApp.check_root_consistency`): it folds the two *full*
    :class:`LiabilityAttestation`\\ s a poster signed for the same ``(poster, attestor, as_of)``
    with **different** ``liabilities_root`` values — a smaller total shown to one creditor, a
    different one to another — into a pinpointed breach. Completeness (:func:`check_completeness`)
    catches an omission only when the omitted creditor folds its *own* claim; equivocation hides
    the omission by showing each creditor a root on which its own claim *is* present, and this is
    what surfaces it: the two contradictory statements, each signed by the attestor.

    Both attestations are embedded whole, so :meth:`verify` re-derives each root from its line
    items (a mislabeled root cannot survive) and — with a ``verifier`` — checks the attestor's
    signature on each (a forged conflicting root is refused, the forger lacking the attestor's
    key). The two are stored in canonical content-hash order, so the same conflict yields the same
    proof whichever way the inputs were supplied. The proof itself may be signed by the creditor /
    coordinator that lodged it (provenance); its validity rests on the two embedded attestor
    signatures, not the reporter's.
    """

    id: str = Field(default_factory=lambda: new_id("equivocation"))
    poster: str
    attestor: str
    as_of: datetime = Field(default_factory=utcnow)

    first: LiabilityAttestation
    second: LiabilityAttestation
    first_root: str = ""
    second_root: str = ""
    first_hash: str = ""
    second_hash: str = ""
    first_creditor: str = ""
    second_creditor: str = ""

    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def roots(self) -> list[str]:
        """The two conflicting roots the poster signed, in canonical order."""
        return [self.first_root, self.second_root]

    @property
    def creditors(self) -> list[str]:
        """The creditors each conflicting attestation was shown to (blank when unrecorded)."""
        return [c for c in (self.first_creditor, self.second_creditor) if c]

    @property
    def liabilities_gap_usd(self) -> float:
        """The absolute gap between the two totals the poster signed for the same instant."""
        return _r6(abs(self.first.liabilities_usd - self.second.liabilities_usd))

    # -- hashing ------------------------------------------------------------

    def equivocation_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the poster, the instant, and the two roots.

        Binds each conflicting attestation by its signed content hash and root (in canonical
        order) plus the creditor each was shown to, so the same conflict hashes identically
        wherever it is recomputed. Excludes the id, signatures, and audit linkage (local metadata).
        """
        return {
            "poster": self.poster,
            "attestor": self.attestor,
            "as_of": self.as_of.isoformat(),
            "first_hash": self.first_hash,
            "second_hash": self.second_hash,
            "first_root": self.first_root,
            "second_root": self.second_root,
            "first_creditor": self.first_creditor,
            "second_creditor": self.second_creditor,
        }

    def compute_hash(self) -> str:
        """The content hash binding the poster, the instant, and the two conflicting roots."""
        return stable_hash(self.equivocation_facts(), length=32)

    def seal(self) -> EquivocationProof:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed the proof, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> EquivocationProof:
        """Add ``party``'s signature over the content hash (sealing first).

        The proof is signed by the creditor or coordinator that detected and lodged the
        equivocation — provenance of *who reported it*. Re-signing for the same party replaces its
        prior signature. The non-repudiable evidence is the two embedded attestor signatures; the
        reporter's signature is not required for :meth:`verify` to hold.
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

    def _attestation_ok(
        self, attestation: LiabilityAttestation, verifier: ChainSigner | None
    ) -> tuple[bool, bool]:
        """Whether an embedded attestation re-derives, and (with a verifier) is attestor-signed."""
        result = attestation.verify(verifier)
        sound = result.hash_ok and result.liabilities_sound
        attestor_signed = False
        if verifier is not None:
            if attestation.signatures and not result.signatures_ok:
                sound = False
            attestor_signed = attestation.attestor in result.signed_by
        return sound, attestor_signed

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> EquivocationProofVerification:
        """Verify the proof offline: the two embedded attestations conflict and re-derive.

        Recomputes the content hash, re-derives each embedded attestation's root and total from
        its line items, and confirms the two share one ``(poster, attestor, as_of)`` key while
        committing different roots. With a ``verifier`` it additionally checks the attestor signed
        each attestation (so a forged conflicting root is refused) and any reporter signatures on
        the proof; ``require`` names reporter parties that must have a verified signature.
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()

        first_sound, first_signed = self._attestation_ok(self.first, verifier)
        second_sound, second_signed = self._attestation_ok(self.second, verifier)
        attestor_signed = first_signed and second_signed
        attestations_ok = first_sound and second_sound
        if verifier is not None:
            attestations_ok = attestations_ok and attestor_signed

        same_key = (
            self.first.poster == self.second.poster == self.poster
            and self.first.attestor == self.second.attestor == self.attestor
            and self.first.as_of == self.second.as_of == self.as_of
        )
        recorded_ok = (
            self.first_hash == self.first.content_hash
            and self.second_hash == self.second.content_hash
            and self.first_root == self.first.liabilities_root
            and self.second_root == self.second.liabilities_root
        )
        conflict = (
            self.first.content_hash != self.second.content_hash
            and self.first.liabilities_root != self.second.liabilities_root
        )
        conflict_ok = same_key and recorded_ok and conflict

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

        valid = hash_ok and attestations_ok and conflict_ok and signatures_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "equivocation proof is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the equivocation facts"
            elif not attestations_ok:
                reason = "an embedded attestation does not re-derive or is not attestor-signed"
            elif not conflict_ok:
                reason = "the two attestations do not conflict on one (poster, attestor, as_of) key"
            elif missing:
                reason = f"missing/invalid reporter signatures for {missing}"
            else:
                reason = "reporter signature mismatch"
        return EquivocationProofVerification(
            valid=valid,
            hash_ok=hash_ok,
            attestations_ok=attestations_ok,
            conflict_ok=conflict_ok,
            attestor_signed=attestor_signed,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> EquivocationProof:
        """Verify and raise :class:`SettlementError` if the equivocation proof is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"equivocation proof {self.id} failed verification: {result.reason}",
                details={"proof_id": self.id, "reason": result.reason},
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the equivocation for the audit chain."""
        return to_jsonable(
            {
                "proof_id": self.id,
                "poster": self.poster,
                "attestor": self.attestor,
                "as_of": self.as_of.isoformat(),
                "first_hash": self.first_hash,
                "second_hash": self.second_hash,
                "first_root": self.first_root,
                "second_root": self.second_root,
                "first_creditor": self.first_creditor,
                "second_creditor": self.second_creditor,
                "liabilities_gap_usd": self.liabilities_gap_usd,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> EquivocationProof:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the poster, the instant, and the two conflicting roots."""
        print(
            f"Equivocation proof ({self.poster}, attested by {self.attestor}) as of "
            f"{self.as_of.isoformat()}: two signed roots disagree by "
            f"${self.liabilities_gap_usd:,.2f}"
        )
        left = f" shown to {self.first_creditor}" if self.first_creditor else ""
        right = f" shown to {self.second_creditor}" if self.second_creditor else ""
        print(f"  root A {self.first_root[:16]}… (${self.first.liabilities_usd:,.2f}){left}")
        print(f"  root B {self.second_root[:16]}… (${self.second.liabilities_usd:,.2f}){right}")


class RootConsistencyReport(BaseModel):
    """The outcome of comparing a set of liability roots for cross-creditor non-equivocation.

    Produced by :func:`check_root_consistency` (or
    :meth:`~vincio.settlement.book.SettlementBook.check_root_consistency` /
    :meth:`~vincio.core.app.ContextApp.check_root_consistency`): it groups the attestations a set
    of creditors hold by their ``(poster, attestor, as_of)`` key and surfaces every poster that
    signed **different** roots for one key as an :class:`EquivocationProof`. ``consistent`` holds
    when no poster equivocated; ``checked`` is how many attestations were considered (tampered or
    unsigned ones are excluded as inadmissible evidence) and ``keys`` is how many distinct
    snapshots they spanned.
    """

    consistent: bool
    checked: int = 0
    keys: int = 0
    equivocations: list[EquivocationProof] = Field(default_factory=list)

    @property
    def equivocating_posters(self) -> list[str]:
        """The posters a proven equivocation shows signed inconsistent roots, sorted and unique."""
        return sorted({proof.poster for proof in self.equivocations})

    def require_consistent(self) -> RootConsistencyReport:
        """Raise :class:`SettlementError` if any poster signed inconsistent liability roots.

        The strict-mode counterpart to inspecting :attr:`consistent`: a counterparty that signed
        two different liability totals for the same instant cannot be taken at either, and this
        pinpoints the equivocating posters.
        """
        if self.equivocations:
            raise SettlementError(
                f"liability roots are inconsistent: {self.equivocating_posters} signed conflicting "
                "roots for the same instant",
                details={
                    "equivocating_posters": self.equivocating_posters,
                    "equivocations": len(self.equivocations),
                },
            )
        return self

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> RootConsistencyReport:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print whether the roots are consistent and any equivocating posters."""
        status = "consistent" if self.consistent else "EQUIVOCATION"
        print(
            f"Root-consistency check: {self.checked} attestation(s) across {self.keys} "
            f"snapshot(s) — {status}"
        )
        for proof in self.equivocations:
            proof.print_summary()


def prove_equivocation(
    first: LiabilityAttestation,
    second: LiabilityAttestation,
    *,
    verifier: ChainSigner | None = None,
    first_creditor: str = "",
    second_creditor: str = "",
) -> EquivocationProof:
    """Fold two conflicting liability attestations into a non-repudiable :class:`EquivocationProof`.

    The two must be the same poster's attestations (same ``attestor`` too) for the **same**
    ``as_of`` but commit **different** ``liabilities_root`` values — the signature of an
    equivocation, where a counterparty signs a smaller total for one creditor and a different one
    for another. Refuses a tampered attestation (its total or root no longer re-deriving), and with
    ``verifier`` an attestation not signed by its attestor (so a forged conflicting root cannot
    found an accusation). Raises :class:`SettlementError` when the two are *not* an equivocation —
    different posters / instants (distinct snapshots), or the same root (no conflict).

    ``first_creditor`` / ``second_creditor`` record which creditor each attestation was shown to
    (provenance). The result is canonicalized in content-hash order, so the same conflict yields the
    same proof whichever way the inputs are supplied.
    """
    pairs = [(first, first_creditor), (second, second_creditor)]
    for attestation, _ in pairs:
        if not attestation.content_hash or not attestation.liabilities_root:
            attestation.seal()
        result = attestation.verify(verifier)
        if not result.hash_ok or not result.liabilities_sound:
            raise SettlementError(
                f"liability attestation {attestation.id} is tampered ({result.reason}); refusing "
                "to found an equivocation proof on it",
                details={"attestation_id": attestation.id, "reason": result.reason},
            )
        if verifier is not None:
            if attestation.signatures and not result.signatures_ok:
                raise SettlementError(
                    f"liability attestation {attestation.id} has an invalid attestor signature; "
                    "refusing to found an equivocation proof on it",
                    details={"attestation_id": attestation.id},
                )
            if attestation.attestor not in result.signed_by:
                raise SettlementError(
                    f"liability attestation {attestation.id} is not signed by its attestor "
                    f"{attestation.attestor!r}; a forged conflicting root cannot found an "
                    "equivocation proof",
                    details={"attestation_id": attestation.id, "attestor": attestation.attestor},
                )
    if first.poster != second.poster or first.attestor != second.attestor:
        raise SettlementError(
            "the two attestations are not an equivocation: they attest different posters/attestors "
            f"({first.poster!r}/{first.attestor!r} vs {second.poster!r}/{second.attestor!r})",
            details={
                "first": [first.poster, first.attestor],
                "second": [second.poster, second.attestor],
            },
        )
    if first.as_of != second.as_of:
        raise SettlementError(
            "the two attestations are not an equivocation: they are for different instants "
            f"({first.as_of.isoformat()} vs {second.as_of.isoformat()}) — distinct snapshots, not "
            "a contradiction",
            details={
                "first_as_of": first.as_of.isoformat(),
                "second_as_of": second.as_of.isoformat(),
            },
        )
    if first.liabilities_root == second.liabilities_root:
        raise SettlementError(
            f"the two attestations are not an equivocation: {first.poster!r} committed the same "
            "root for the same instant",
            details={"poster": first.poster, "root": first.liabilities_root},
        )
    (att_a, cred_a), (att_b, cred_b) = sorted(pairs, key=lambda pair: pair[0].content_hash)
    proof = EquivocationProof(
        poster=att_a.poster,
        attestor=att_a.attestor,
        as_of=att_a.as_of,
        first=att_a,
        second=att_b,
        first_root=att_a.liabilities_root,
        second_root=att_b.liabilities_root,
        first_hash=att_a.content_hash,
        second_hash=att_b.content_hash,
        first_creditor=cred_a,
        second_creditor=cred_b,
    )
    return proof.seal()


def _coerce_attestation_views(attestations: Any) -> list[tuple[str, LiabilityAttestation]]:
    """Normalize a set of attestations (optionally creditor-labelled) into ``(creditor, att)``.

    Accepts an iterable of :class:`LiabilityAttestation`, ``(creditor, attestation)`` pairs, or
    ``{"creditor": ..., "attestation": ...}`` dicts — so a caller can record which creditor was
    shown each root. A bare attestation carries no creditor label.
    """
    views: list[tuple[str, LiabilityAttestation]] = []
    for item in attestations:
        if isinstance(item, LiabilityAttestation):
            views.append(("", item))
        elif isinstance(item, dict):
            attestation = item.get("attestation")
            if not isinstance(attestation, LiabilityAttestation):
                raise SettlementError(
                    "check_root_consistency dict items need an 'attestation' "
                    f"LiabilityAttestation; got {attestation!r}",
                    details={"item": repr(item)},
                )
            views.append((str(item.get("creditor", "")), attestation))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            creditor, attestation = item[0], item[1]
            if not isinstance(attestation, LiabilityAttestation):
                raise SettlementError(
                    "check_root_consistency (creditor, attestation) items need a "
                    f"LiabilityAttestation; got {attestation!r}",
                    details={"item": repr(item)},
                )
            views.append((str(creditor), attestation))
        else:
            raise SettlementError(
                "check_root_consistency items must be LiabilityAttestation, "
                f"(creditor, attestation), or {{creditor, attestation}}; got {item!r}",
                details={"item": repr(item)},
            )
    return views


def check_root_consistency(
    attestations: Any, *, verifier: ChainSigner | None = None
) -> RootConsistencyReport:
    """Compare a set of liability attestations for cross-creditor root non-equivocation.

    The set is what a group of creditors hold — each the attestation a poster signed for *it*.
    Groups them by ``(poster, attestor, as_of)`` and surfaces every poster that signed **different**
    roots for one key as an :class:`EquivocationProof`. Completeness (:func:`check_completeness`)
    catches an omission only when the omitted creditor folds its own claim; this catches the
    counterparty that equivocates across the set — showing each creditor a root on which its own
    claim *is* present while the totals disagree.

    Only attestations that re-derive are considered evidence (a tampered one is excluded); with a
    ``verifier`` only those carrying a valid attestor signature are (a forged or unsigned root is
    excluded, so it can never manufacture a false accusation). Each conflicting pair is folded by
    :func:`prove_equivocation`, so every returned proof verifies from the bytes alone. ``attestations``
    items may be bare attestations or ``(creditor, attestation)`` pairs to record provenance.
    """
    views = _coerce_attestation_views(attestations)
    admissible: list[tuple[str, LiabilityAttestation]] = []
    for creditor, attestation in views:
        if not attestation.content_hash or not attestation.liabilities_root:
            attestation.seal()
        result = attestation.verify(verifier)
        if not result.hash_ok or not result.liabilities_sound:
            continue
        if verifier is not None:
            if attestation.signatures and not result.signatures_ok:
                continue
            if attestation.attestor not in result.signed_by:
                continue
        admissible.append((creditor, attestation))

    groups: dict[tuple[str, str, str], list[tuple[str, LiabilityAttestation]]] = {}
    for creditor, attestation in admissible:
        key = (attestation.poster, attestation.attestor, attestation.as_of.isoformat())
        groups.setdefault(key, []).append((creditor, attestation))

    equivocations: list[EquivocationProof] = []
    for members in groups.values():
        representatives: dict[str, tuple[str, LiabilityAttestation]] = {}
        for creditor, attestation in members:
            representatives.setdefault(attestation.liabilities_root, (creditor, attestation))
        if len(representatives) < 2:
            continue
        ordered = [representatives[root] for root in sorted(representatives)]
        for (cred_a, att_a), (cred_b, att_b) in zip(ordered, ordered[1:], strict=False):
            equivocations.append(
                prove_equivocation(
                    att_a,
                    att_b,
                    verifier=verifier,
                    first_creditor=cred_a,
                    second_creditor=cred_b,
                )
            )

    return RootConsistencyReport(
        consistent=not equivocations,
        checked=len(admissible),
        keys=len(groups),
        equivocations=equivocations,
    )


# -- liability history consistency & snapshot monotonicity --------------------
#
# Non-equivocation catches a counterparty signing *different* roots for the same instant. But it
# is scoped to one ``as_of``: a counterparty can still issue a *later* snapshot that quietly
# **drops** a past obligation, each snapshot internally sound. This closes that escape hatch with
# a hash-linked history and a monotone-consistency proof — a debt committed at one snapshot must
# persist into the next (or be released by a signed, creditor-issued discharge), so it cannot
# silently vanish between snapshots.


def _creditor_map(attestation: LiabilityAttestation) -> dict[str, float]:
    """The per-creditor obligation map an attestation commits, summing duplicate line items."""
    owed: dict[str, float] = {}
    for line in attestation.liabilities:
        owed[line.creditor] = _r6(owed.get(line.creditor, 0.0) + line.amount_usd)
    return owed


class DischargeVerification(BaseModel):
    """The (non-raising) outcome of verifying a liability discharge offline.

    A discharge is **valid** when its content hash recomputes (``hash_ok``), the released amount is
    non-negative (``sound``), and — with a ``verifier`` — the creditor's signature checks
    (``signatures_ok``). A discharge is the *creditor's* release of what a poster owes it, so a
    legitimate one carries the creditor's signature; a poster cannot forge its own discharge.
    """

    valid: bool
    hash_ok: bool
    sound: bool
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class Discharge(BaseModel):
    """A signed, content-bound release of part of what a poster owes one creditor.

    Produced by :func:`discharge_liability` (or
    :meth:`~vincio.settlement.book.SettlementBook.discharge_liability` /
    :meth:`~vincio.core.app.ContextApp.discharge_liability`): the **creditor** signs that ``poster``
    has legitimately settled (or it has forgiven) ``amount_usd`` of the obligation owed to it, as of
    ``as_of``. It is the evidence that lets a liability *shrink* between two snapshots without
    tripping a :class:`MonotonicityBreach`: :func:`check_history_consistency` applies a discharge to
    the transition whose window contains its ``as_of``, so the drop it covers is explained rather
    than treated as a debt that silently vanished.

    Because the release is the creditor's to make, only the creditor signs it (``party`` defaults to
    the creditor; signing as anyone else is refused), so a poster cannot manufacture a discharge to
    paper over a drop. :meth:`verify` recomputes the content hash and checks the creditor signature
    from the bytes alone.
    """

    id: str = Field(default_factory=lambda: new_id("discharge"))
    poster: str
    creditor: str
    amount_usd: float = 0.0
    as_of: datetime = Field(default_factory=utcnow)
    note: str = ""

    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def status(self) -> str:
        """``partial`` (a positive release) or ``empty`` (nothing released)."""
        return "partial" if self.amount_usd > _TOLERANCE else "empty"

    # -- hashing ------------------------------------------------------------

    def discharge_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the poster, the creditor, the amount, the instant."""
        return {
            "poster": self.poster,
            "creditor": self.creditor,
            "amount_usd": _r6(self.amount_usd),
            "as_of": self.as_of.isoformat(),
            "note": self.note,
        }

    def compute_hash(self) -> str:
        """The content hash binding the release."""
        return stable_hash(self.discharge_facts(), length=32)

    def seal(self) -> Discharge:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str | None = None) -> Discharge:
        """Add the creditor's signature over the content hash (sealing first).

        A discharge is the creditor's release, so only the creditor signs it (``party`` defaults to
        the creditor; passing a different party is refused). Re-signing replaces the prior
        signature, so a discharge cannot accumulate stale ones.
        """
        resolved = party or self.creditor
        if resolved != self.creditor:
            raise SettlementError(
                f"a liability discharge is signed by its creditor {self.creditor!r}, not "
                f"{resolved!r}",
                details={"discharge_id": self.id, "creditor": self.creditor, "party": resolved},
            )
        if not self.content_hash:
            self.seal()
        sig = SettlementSignature(
            party=resolved,
            signature=signer.sign(self.content_hash),
            key_id=getattr(signer, "key_id", ""),
        )
        self.signatures = [s for s in self.signatures if s.party != resolved]
        self.signatures.append(sig)
        return self

    # -- verification -------------------------------------------------------

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> DischargeVerification:
        """Verify the discharge offline: the hash recomputes and the creditor signature checks.

        ``verifier`` checks each signature against the content hash; ``require`` names parties that
        must have a verified signature (defaults to none — pass ``[creditor]`` to demand the
        creditor's signature). A negative release is unsound and refused.
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        sound = self.amount_usd >= -_TOLERANCE
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
        valid = hash_ok and sound and signatures_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "discharge is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the discharge facts"
            elif not sound:
                reason = "discharge releases a negative amount"
            elif missing:
                reason = f"missing/invalid signatures for {missing}"
            else:
                reason = "signature mismatch"
        return DischargeVerification(
            valid=valid,
            hash_ok=hash_ok,
            sound=sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> Discharge:
        """Verify and raise :class:`SettlementError` if the discharge is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"discharge {self.id} failed verification: {result.reason}",
                details={"discharge_id": self.id, "reason": result.reason},
            )
        return self

    def _applies_to(
        self, poster: str, creditor: str, lo: datetime, hi: datetime, verifier: ChainSigner | None
    ) -> bool:
        """Whether this discharge explains a drop for ``creditor`` in the window ``(lo, hi]``.

        A discharge applies to the transition between two snapshots when it is for the same poster
        and creditor and its ``as_of`` falls in the half-open window after the earlier snapshot up
        to and including the later one — so each release is consumed by exactly one transition. It
        is only credited when it verifies and (with a ``verifier``) carries the creditor's
        signature, so a forged or poster-signed discharge cannot explain a drop.
        """
        if self.poster != poster or self.creditor != creditor:
            return False
        if not (lo < self.as_of <= hi):
            return False
        result = self.verify(verifier)
        if not result.hash_ok or not result.sound:
            return False
        if verifier is not None and self.creditor not in result.signed_by:
            return False
        return True

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the discharge for the audit chain."""
        return to_jsonable(
            {
                "discharge_id": self.id,
                "poster": self.poster,
                "creditor": self.creditor,
                "amount_usd": _r6(self.amount_usd),
                "as_of": self.as_of.isoformat(),
                "status": self.status,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> Discharge:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the released obligation."""
        note = f" ({self.note})" if self.note else ""
        print(
            f"Discharge ({self.poster} → {self.creditor}): ${self.amount_usd:,.2f} released as of "
            f"{self.as_of.isoformat()}{note}"
        )


def discharge_liability(
    poster: str,
    creditor: str,
    amount_usd: float,
    *,
    as_of: datetime | None = None,
    note: str = "",
) -> Discharge:
    """Build an (unsigned) :class:`Discharge` releasing part of what ``poster`` owes ``creditor``.

    The evidence a creditor issues when an obligation is legitimately settled or forgiven, so the
    matching drop in the poster's next liability snapshot is *explained* rather than treated as a
    debt that silently vanished. Sign it with the creditor's key (or let a
    :class:`~vincio.settlement.book.SettlementBook` do it on
    :meth:`~vincio.settlement.book.SettlementBook.discharge_liability`). Raises
    :class:`SettlementError` on a negative release.
    """
    if amount_usd < 0.0:
        raise SettlementError(
            f"discharge of {poster!r} → {creditor!r} releases a negative amount {amount_usd}",
            details={"poster": poster, "creditor": creditor, "amount_usd": amount_usd},
        )
    return Discharge(
        poster=poster,
        creditor=creditor,
        amount_usd=_r6(amount_usd),
        as_of=as_of or utcnow(),
        note=note,
    ).seal()


class MonotonicityBreach(BaseModel):
    """A creditor's obligation that shrank between two snapshots without a backing discharge.

    Surfaced by :func:`check_history_consistency` when a poster's later snapshot owes a creditor
    *less* than an earlier one and no signed :class:`Discharge` covers the difference:
    ``prior_usd`` is what the earlier snapshot (``prior_hash``) committed, ``next_usd`` what the
    later one (``next_hash``) commits, ``dropped_usd`` the reduction, ``discharged_usd`` how much a
    creditor-signed discharge legitimately released, and ``unexplained_usd`` the residue
    (``dropped − discharged``) — the debt that silently vanished. Non-equivocation cannot catch
    this: each snapshot is internally sound and they are for *different* instants.
    """

    poster: str
    attestor: str = ""
    creditor: str
    prior_hash: str = ""
    next_hash: str = ""
    prior_as_of: datetime | None = None
    next_as_of: datetime | None = None
    prior_usd: float = 0.0
    next_usd: float = 0.0
    dropped_usd: float = 0.0
    discharged_usd: float = 0.0
    unexplained_usd: float = 0.0


class HistoryConsistencyProofVerification(BaseModel):
    """The (non-raising) outcome of verifying a liability history-consistency proof offline.

    A proof is **valid** when its content hash recomputes (``hash_ok``), every embedded snapshot
    re-derives its root and total and they share one ``(poster, attestor)`` in strictly increasing
    ``as_of`` order — and, with a ``verifier``, each is attestor-signed (``snapshots_ok`` /
    ``attestor_signed``), the recorded chain-link status re-derives (``chain_ok``), and the
    monotonicity breaches re-derive from the embedded snapshots and discharges (``monotone_sound``).
    A forged or back-dated snapshot, a dropped breach, or a poster-forged discharge is caught from
    the bytes alone.
    """

    valid: bool
    hash_ok: bool
    snapshots_ok: bool
    chain_ok: bool
    monotone_sound: bool
    attestor_signed: bool = False
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class HistoryConsistencyProof(BaseModel):
    """A signed, offline-verifiable proof a poster's liability history is monotone over time.

    Produced by :func:`check_history_consistency` (or
    :meth:`~vincio.settlement.book.SettlementBook.check_history_consistency` /
    :meth:`~vincio.core.app.ContextApp.check_history_consistency`): it embeds one poster's
    :class:`LiabilityAttestation` snapshots in ``as_of`` order along with the :class:`Discharge`\\ s
    that explain any legitimate reductions, and folds them into a per-creditor walk. Every creditor
    obligation must **persist** from each snapshot into the next unless a signed, creditor-issued
    discharge released it; an unexplained drop is pinpointed as a :class:`MonotonicityBreach`.

    Both the snapshots and the discharges are embedded whole, so :meth:`verify` re-derives each
    snapshot's root from its line items (a mislabeled or back-dated snapshot cannot survive), checks
    they form a strictly increasing ``as_of`` sequence for one ``(poster, attestor)``, and re-derives
    every breach from the bytes (a forged or poster-signed discharge does not count, so a hidden drop
    resurfaces). ``chain_linked`` records whether every successor also commits to its predecessor's
    root (:meth:`LiabilityAttestation.link_to`) — a contiguous, tamper-evident chain — though
    monotonicity is checked on the sorted sequence regardless, so an unlinked legacy history is still
    walked. The proof may be signed by the creditor / auditor that lodged it (provenance).
    """

    id: str = Field(default_factory=lambda: new_id("history"))
    poster: str
    attestor: str = ""

    snapshots: list[LiabilityAttestation] = Field(default_factory=list)
    discharges: list[Discharge] = Field(default_factory=list)
    breaches: list[MonotonicityBreach] = Field(default_factory=list)

    chain_linked: bool = False
    head_hash: str = ""
    span_from: datetime | None = None
    span_to: datetime | None = None

    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def monotone(self) -> bool:
        """Whether every committed obligation persisted (no unexplained drop between snapshots)."""
        return not self.breaches

    @property
    def consistent(self) -> bool:
        """Whether the history is monotone — the headline (a debt never silently vanished)."""
        return self.monotone

    @property
    def linked(self) -> bool:
        """Whether every successor commits to its predecessor's root (a contiguous chain)."""
        return self.chain_linked

    @property
    def snapshot_count(self) -> int:
        """How many snapshots the history spans."""
        return len(self.snapshots)

    @property
    def status(self) -> str:
        """``consistent`` (monotone history) or ``inconsistent`` (an unexplained drop)."""
        return "consistent" if self.consistent else "inconsistent"

    @property
    def breaching_creditors(self) -> list[str]:
        """The creditors whose obligation an unexplained drop shows vanished, sorted and unique."""
        return sorted({breach.creditor for breach in self.breaches})

    @property
    def unexplained_usd(self) -> float:
        """The total obligation that dropped between snapshots without a backing discharge."""
        return _r6(sum(breach.unexplained_usd for breach in self.breaches))

    def _ordered_snapshots(self) -> list[LiabilityAttestation]:
        """The embedded snapshots in canonical (as_of, content-hash) order."""
        return sorted(self.snapshots, key=lambda att: (att.as_of, att.content_hash))

    # -- hashing ------------------------------------------------------------

    def history_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the poster, the snapshots, discharges, and breaches.

        Binds each snapshot by its signed content hash (in ``as_of`` order), each applied discharge
        by its content hash, and every breach, so a re-ordered, swapped, or dropped element is
        caught even after re-sealing. Excludes the id, signatures, and audit linkage (local
        metadata).
        """
        return {
            "poster": self.poster,
            "attestor": self.attestor,
            "chain_linked": self.chain_linked,
            "head_hash": self.head_hash,
            "span_from": self.span_from.isoformat() if self.span_from else "",
            "span_to": self.span_to.isoformat() if self.span_to else "",
            "snapshots": [att.content_hash for att in self._ordered_snapshots()],
            "discharges": sorted(d.content_hash for d in self.discharges),
            "breaches": [
                {
                    "creditor": b.creditor,
                    "prior_hash": b.prior_hash,
                    "next_hash": b.next_hash,
                    "prior_usd": _r6(b.prior_usd),
                    "next_usd": _r6(b.next_usd),
                    "dropped_usd": _r6(b.dropped_usd),
                    "discharged_usd": _r6(b.discharged_usd),
                    "unexplained_usd": _r6(b.unexplained_usd),
                }
                for b in sorted(self.breaches, key=lambda b: (b.next_hash, b.creditor))
            ],
        }

    def compute_hash(self) -> str:
        """The content hash binding the poster, the snapshots, the discharges, and the breaches."""
        return stable_hash(self.history_facts(), length=32)

    def seal(self) -> HistoryConsistencyProof:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed the proof, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> HistoryConsistencyProof:
        """Add ``party``'s signature over the content hash (sealing first).

        Signed by the creditor or auditor that walked the history and lodged the finding —
        provenance of *who reported it*. The non-repudiable evidence is the embedded attestor
        signatures on each snapshot and the creditor signatures on each discharge; the reporter's
        signature is not required for :meth:`verify` to hold. Re-signing replaces the prior one.
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

    def _snapshots_ok(self, verifier: ChainSigner | None) -> tuple[bool, bool]:
        """Whether the embedded snapshots re-derive and form a strict one-poster sequence."""
        ordered = self._ordered_snapshots()
        if not ordered:
            return False, False
        attestor_signed = True
        for att in ordered:
            result = att.verify(verifier)
            if not result.hash_ok or not result.liabilities_sound:
                return False, False
            if att.poster != self.poster or att.attestor != self.attestor:
                return False, False
            if verifier is not None:
                if att.signatures and not result.signatures_ok:
                    return False, False
                if att.attestor not in result.signed_by:
                    attestor_signed = False
        # Strictly increasing instants — no two snapshots share an ``as_of`` (that is the domain of
        # non-equivocation, not history) and none is back-dated.
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if later.as_of <= earlier.as_of:
                return False, attestor_signed
        head_ok = self.head_hash == ordered[-1].content_hash
        span_ok = self.span_from == ordered[0].as_of and self.span_to == ordered[-1].as_of
        return (head_ok and span_ok), attestor_signed

    def _recompute_chain_linked(self) -> bool:
        """Whether every successor commits to its predecessor's hash, root, and instant."""
        ordered = self._ordered_snapshots()
        if len(ordered) < 2:
            return False
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if (
                later.prior_hash != earlier.content_hash
                or later.prior_root != earlier.liabilities_root
                or later.prior_as_of != earlier.as_of
            ):
                return False
        return True

    def _walk(self, verifier: ChainSigner | None) -> tuple[list[MonotonicityBreach], set[str]]:
        """Walk the snapshots, deriving each transition's breaches and the discharges they consume.

        For every consecutive pair, each creditor's obligation that fell is covered by the signed,
        in-window discharges available for it (each consumed by exactly one transition, so a single
        release cannot explain two drops); the residue is a :class:`MonotonicityBreach`. Returns the
        breaches and the set of discharge content hashes that explained a drop.
        """
        ordered = self._ordered_snapshots()
        consumed: set[str] = set()
        breaches: list[MonotonicityBreach] = []
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            prior_map = _creditor_map(earlier)
            next_map = _creditor_map(later)
            for creditor in sorted(prior_map):
                prior_usd = prior_map[creditor]
                next_usd = _r6(next_map.get(creditor, 0.0))
                dropped = _r6(max(0.0, prior_usd - next_usd))
                if dropped <= _TOLERANCE:
                    continue
                discharged = 0.0
                for discharge in self.discharges:
                    if discharge.content_hash in consumed:
                        continue
                    if discharge._applies_to(
                        self.poster, creditor, earlier.as_of, later.as_of, verifier
                    ):
                        discharged = _r6(discharged + discharge.amount_usd)
                        consumed.add(discharge.content_hash)
                unexplained = _r6(max(0.0, dropped - discharged))
                if unexplained > _TOLERANCE:
                    breaches.append(
                        MonotonicityBreach(
                            poster=self.poster,
                            attestor=self.attestor,
                            creditor=creditor,
                            prior_hash=earlier.content_hash,
                            next_hash=later.content_hash,
                            prior_as_of=earlier.as_of,
                            next_as_of=later.as_of,
                            prior_usd=prior_usd,
                            next_usd=next_usd,
                            dropped_usd=dropped,
                            discharged_usd=discharged,
                            unexplained_usd=unexplained,
                        )
                    )
        return breaches, consumed

    def _recompute_breaches(self, verifier: ChainSigner | None) -> list[MonotonicityBreach]:
        """Re-derive the monotonicity breaches from the embedded snapshots and discharges."""
        return self._walk(verifier)[0]

    @staticmethod
    def _breach_key(breach: MonotonicityBreach) -> tuple[str, str, str]:
        return (breach.next_hash, breach.creditor, breach.prior_hash)

    def _monotone_sound(self, verifier: ChainSigner | None) -> bool:
        """The recorded breaches re-derive exactly from the embedded snapshots and discharges."""
        expected = self._recompute_breaches(verifier)
        if len(expected) != len(self.breaches):
            return False
        for got, want in zip(
            sorted(self.breaches, key=self._breach_key),
            sorted(expected, key=self._breach_key),
            strict=True,
        ):
            if self._breach_key(got) != self._breach_key(want):
                return False
            if abs(got.unexplained_usd - want.unexplained_usd) > _TOLERANCE:
                return False
            if abs(got.dropped_usd - want.dropped_usd) > _TOLERANCE:
                return False
            if abs(got.discharged_usd - want.discharged_usd) > _TOLERANCE:
                return False
        return True

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> HistoryConsistencyProofVerification:
        """Verify the proof offline: the snapshots form a monotone history.

        Recomputes the content hash, re-derives each embedded snapshot's root and total, confirms
        they form a strictly increasing ``as_of`` sequence for one ``(poster, attestor)``, re-derives
        the chain-link status and the monotonicity breaches from the embedded snapshots and
        discharges, and (with a ``verifier``) checks each snapshot is attestor-signed and any reporter
        signatures on the proof. ``require`` names reporter parties that must have a verified
        signature. A forged or back-dated snapshot, a poster-forged discharge, or a dropped breach is
        caught from the bytes alone.
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        snapshots_ok, attestor_signed = self._snapshots_ok(verifier)
        if verifier is not None:
            snapshots_ok = snapshots_ok and attestor_signed
        chain_ok = self.chain_linked == self._recompute_chain_linked()
        monotone_sound = snapshots_ok and self._monotone_sound(verifier)

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

        valid = hash_ok and snapshots_ok and chain_ok and monotone_sound and signatures_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "history proof is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the history facts"
            elif not snapshots_ok:
                reason = "an embedded snapshot does not re-derive, is mis-ordered, or is unsigned"
            elif not chain_ok:
                reason = "the recorded chain-link status does not re-derive from the snapshots"
            elif not monotone_sound:
                reason = "the monotonicity breaches do not re-derive from the snapshots/discharges"
            elif missing:
                reason = f"missing/invalid reporter signatures for {missing}"
            else:
                reason = "reporter signature mismatch"
        return HistoryConsistencyProofVerification(
            valid=valid,
            hash_ok=hash_ok,
            snapshots_ok=snapshots_ok,
            chain_ok=chain_ok,
            monotone_sound=monotone_sound,
            attestor_signed=attestor_signed,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> HistoryConsistencyProof:
        """Verify and raise :class:`SettlementError` if the history proof is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"history consistency proof {self.id} failed verification: {result.reason}",
                details={"proof_id": self.id, "reason": result.reason},
            )
        return self

    def require_monotone(self) -> HistoryConsistencyProof:
        """Raise :class:`SettlementError` if any obligation dropped without a backing discharge.

        The strict-mode counterpart to inspecting :attr:`monotone`: a counterparty that let a debt
        vanish between snapshots cannot be taken at its latest figure, and this pinpoints the
        creditors whose obligation silently dropped and by how much.
        """
        if self.breaches:
            raise SettlementError(
                f"liability history for {self.poster!r} is not monotone: "
                f"${self.unexplained_usd:,.2f} owed to {self.breaching_creditors} dropped between "
                "snapshots without a backing discharge",
                details={
                    "proof_id": self.id,
                    "poster": self.poster,
                    "unexplained_usd": self.unexplained_usd,
                    "breaching_creditors": self.breaching_creditors,
                },
            )
        return self

    def require_linked(self) -> HistoryConsistencyProof:
        """Raise :class:`SettlementError` if the snapshots are not a contiguous hash-linked chain.

        Stricter than :meth:`require_monotone`: demands every successor commit to its predecessor's
        root, so a creditor knows it holds the *complete* sequence with no snapshot spliced out.
        """
        if not self.chain_linked:
            raise SettlementError(
                f"liability history for {self.poster!r} is not a contiguous hash-linked chain; a "
                "snapshot may be missing or unlinked",
                details={"proof_id": self.id, "poster": self.poster},
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the history check for the audit chain."""
        return to_jsonable(
            {
                "proof_id": self.id,
                "poster": self.poster,
                "attestor": self.attestor,
                "status": self.status,
                "snapshots": self.snapshot_count,
                "chain_linked": self.chain_linked,
                "span_from": self.span_from.isoformat() if self.span_from else None,
                "span_to": self.span_to.isoformat() if self.span_to else None,
                "discharges": len(self.discharges),
                "unexplained_usd": self.unexplained_usd,
                "breaching_creditors": self.breaching_creditors,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> HistoryConsistencyProof:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the snapshot span and any unexplained drops."""
        link = "linked" if self.chain_linked else "unlinked"
        print(
            f"History consistency ({self.poster}, attested by {self.attestor}): "
            f"{self.snapshot_count} snapshot(s), {link} — {self.status}"
        )
        for breach in sorted(self.breaches, key=lambda b: b.creditor):
            print(
                f"  ! {breach.creditor}: dropped ${breach.dropped_usd:,.2f} "
                f"(${breach.prior_usd:,.2f} → ${breach.next_usd:,.2f}), discharged "
                f"${breach.discharged_usd:,.2f}, unexplained ${breach.unexplained_usd:,.2f}"
            )


class HistoryConsistencyReport(BaseModel):
    """The outcome of walking a set of liability snapshots for cross-time monotonicity.

    Produced by :func:`check_history_consistency` (or
    :meth:`~vincio.settlement.book.SettlementBook.check_history_consistency` /
    :meth:`~vincio.core.app.ContextApp.check_history_consistency`): it groups the attestations by
    their ``(poster, attestor)`` key, walks each poster's snapshots in ``as_of`` order, and folds
    every chain into a :class:`HistoryConsistencyProof`. ``consistent`` holds when no poster let an
    obligation silently drop; ``checked`` is how many snapshots were admissible (tampered or unsigned
    ones are excluded as inadmissible evidence) and ``chains`` is how many distinct posters' histories
    were walked.
    """

    consistent: bool
    checked: int = 0
    chains: int = 0
    proofs: list[HistoryConsistencyProof] = Field(default_factory=list)

    @property
    def breaching_posters(self) -> list[str]:
        """The posters whose history dropped an obligation without a discharge, sorted and unique."""
        return sorted({proof.poster for proof in self.proofs if not proof.monotone})

    def require_consistent(self) -> HistoryConsistencyReport:
        """Raise :class:`SettlementError` if any poster's liability history is not monotone."""
        if self.breaching_posters:
            raise SettlementError(
                f"liability history is inconsistent: {self.breaching_posters} dropped an obligation "
                "between snapshots without a backing discharge",
                details={"breaching_posters": self.breaching_posters},
            )
        return self

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> HistoryConsistencyReport:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print whether the histories are monotone and any breaching posters."""
        status = "consistent" if self.consistent else "INCONSISTENT"
        print(
            f"History-consistency check: {self.checked} snapshot(s) across {self.chains} "
            f"chain(s) — {status}"
        )
        for proof in self.proofs:
            if not proof.monotone:
                proof.print_summary()


def _coerce_discharges(discharges: Any) -> list[Discharge]:
    """Normalize a discharges spec into sealed :class:`Discharge`\\ s.

    Accepts ``None`` (no discharges), a single :class:`Discharge`, or an iterable of
    :class:`Discharge` / dict items. A dict is validated into a :class:`Discharge`. Each is sealed
    if needed so its content hash is available for application and de-duplication.
    """
    if discharges is None:
        return []
    if isinstance(discharges, Discharge):
        items: list[Any] = [discharges]
    else:
        items = list(discharges)
    coerced: list[Discharge] = []
    for item in items:
        if isinstance(item, Discharge):
            discharge = item
        elif isinstance(item, dict):
            discharge = Discharge.model_validate(item)
        else:
            raise SettlementError(
                f"check_history_consistency discharges must be Discharge or dict items; got {item!r}",
                details={"item": repr(item)},
            )
        if not discharge.content_hash:
            discharge.seal()
        coerced.append(discharge)
    return coerced


def check_history_consistency(
    attestations: Any,
    *,
    discharges: Any | None = None,
    verifier: ChainSigner | None = None,
) -> HistoryConsistencyReport:
    """Walk a set of liability snapshots for cross-time monotonicity (no debt silently dropped).

    The companion to :func:`check_root_consistency`: where non-equivocation catches a counterparty
    signing *different* roots for the **same** instant, this catches one issuing a *later* snapshot
    that quietly **drops** a past obligation. Groups the ``attestations`` by ``(poster, attestor)``,
    sorts each poster's snapshots by ``as_of``, and walks them: every creditor's obligation must
    persist from each snapshot into the next unless a signed, creditor-issued :class:`Discharge`
    (``discharges``) released it. An unexplained drop is pinpointed as a :class:`MonotonicityBreach`
    in the chain's :class:`HistoryConsistencyProof`.

    Only snapshots that re-derive are considered evidence (a tampered one is excluded); with a
    ``verifier`` only those carrying a valid attestor signature are, and only discharges carrying a
    valid creditor signature explain a drop (a poster-forged discharge cannot). ``attestations`` items
    may be bare attestations or ``(creditor, attestation)`` pairs (the creditor label is ignored — a
    history is one observer's sequence). A poster with fewer than two distinct snapshot instants has
    no history to walk and yields no proof.
    """
    views = _coerce_attestation_views(attestations)
    coerced_discharges = _coerce_discharges(discharges)

    admissible: list[LiabilityAttestation] = []
    for _creditor, attestation in views:
        if not attestation.content_hash or not attestation.liabilities_root:
            attestation.seal()
        result = attestation.verify(verifier)
        if not result.hash_ok or not result.liabilities_sound:
            continue
        if verifier is not None:
            if attestation.signatures and not result.signatures_ok:
                continue
            if attestation.attestor not in result.signed_by:
                continue
        admissible.append(attestation)

    groups: dict[tuple[str, str], list[LiabilityAttestation]] = {}
    for attestation in admissible:
        groups.setdefault((attestation.poster, attestation.attestor), []).append(attestation)

    proofs: list[HistoryConsistencyProof] = []
    for (poster, attestor), members in groups.items():
        # One snapshot per instant: a history is a *linear* sequence over time. Two snapshots for the
        # same ``as_of`` are the domain of non-equivocation (:func:`check_root_consistency`), not a
        # cross-time transition; keep the first in canonical order so the walk stays strictly
        # increasing and the proof always verifies.
        by_instant: dict[datetime, LiabilityAttestation] = {}
        for att in sorted(members, key=lambda att: (att.as_of, att.content_hash)):
            by_instant.setdefault(att.as_of, att)
        ordered = [by_instant[instant] for instant in sorted(by_instant)]
        # A history needs at least two distinct instants to compare; a single snapshot has no
        # cross-time transition to walk.
        if len(ordered) < 2:
            continue
        relevant = [d for d in coerced_discharges if d.poster == poster]
        proof = HistoryConsistencyProof(
            poster=poster,
            attestor=attestor,
            snapshots=ordered,
            discharges=relevant,
            chain_linked=False,
            head_hash=ordered[-1].content_hash,
            span_from=ordered[0].as_of,
            span_to=ordered[-1].as_of,
        )
        proof.chain_linked = proof._recompute_chain_linked()
        # Walk once to derive the breaches and which discharges actually explained a drop; embed only
        # those, so the proof carries exactly the evidence it rests on (an unrelated discharge does
        # not bloat the content hash). Trimming to the consumed set leaves the breaches unchanged.
        breaches, consumed = proof._walk(verifier)
        proof.discharges = [d for d in relevant if d.content_hash in consumed]
        proof.breaches = breaches
        proofs.append(proof.seal())

    return HistoryConsistencyReport(
        consistent=all(proof.monotone for proof in proofs),
        checked=len(admissible),
        chains=len(proofs),
        proofs=proofs,
    )
