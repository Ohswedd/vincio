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
        return {
            "attestor": self.attestor,
            "poster": self.poster,
            "liabilities_usd": _r6(self.liabilities_usd),
            "liabilities_root": self.liabilities_root,
            "as_of": self.as_of.isoformat(),
            "liabilities": [line.facts() for line in self._sorted_lines()],
        }

    def compute_hash(self) -> str:
        """The content hash binding the attestor, the poster, the liabilities, and the root."""
        return stable_hash(self.attestation_facts(), length=32)

    def seal(self) -> LiabilityAttestation:
        """Stamp the Merkle root and the content hash from the current fields (idempotent)."""
        self.liabilities_root = self.compute_root()
        self.content_hash = self.compute_hash()
        return self

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
        return self.liabilities_root == self.compute_root()

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
) -> LiabilityAttestation:
    """Attest a poster's total obligations into an (unsigned) :class:`LiabilityAttestation`.

    The proof-of-liabilities analogue of
    :func:`~vincio.settlement.custody.attest_custody`: ``attestor`` (defaulting to the
    ``poster`` itself — self-attested) vouches for the total obligations ``poster`` owes.
    ``liabilities`` is a single number (one unnamed obligation), a mapping of ``creditor ->
    amount``, or an iterable of :class:`LiabilityLine` / ``(creditor, amount)`` items; the
    attested ``liabilities_usd`` is their sum, re-derived on every verify.

    Returns a sealed, unsigned attestation — sign it with the attestor's key (or let a
    :class:`~vincio.settlement.book.SettlementBook` do it on
    :meth:`~vincio.settlement.book.SettlementBook.attest_liabilities`). Raises
    :class:`SettlementError` when an obligation is negative.
    """
    lines = _coerce_liabilities(liabilities)
    attestation = LiabilityAttestation(
        attestor=attestor or poster,
        poster=poster,
        liabilities=lines,
        liabilities_usd=_r6(sum(line.amount_usd for line in lines)),
        as_of=as_of or utcnow(),
    )
    return attestation.seal()


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
    """A proven shortfall — the obligations owed exceed the reserves actually held.

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
