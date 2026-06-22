"""Federated / cross-org self-improvement.

The platform already learns from its own traffic three ways — the on-policy RLVR
loop (:mod:`vincio.optimize.trajectory_opt`), the distillation flywheel
(:mod:`vincio.optimize.distill`), and on-device local adaptation
(:mod:`vincio.optimize.local_adaptation`) — but always **within one trust
boundary**. The rung this module adds is **sharing what was learned across
organizations without sharing the raw traffic**: a fleet of members each fits a
local adapter on its own private data, contributes a *privacy-preserving,
numeric* summary of where that adaptation lives — never a prompt, never a
response — and a secure aggregation merges those summaries into a shared
candidate that any member can adopt behind the very same no-regression and canary
gates a local promotion clears. The fleet improves together while each member's
data stays put.

The artifact that travels is a **subspace**, not an example memory. A member's
local adapter (:class:`~vincio.optimize.local_adaptation.LocalAdapter`) separates
*where it fires* — a low-rank geometry over its prompt embeddings — from *what it
answers* — the grounded response text. Only the geometry is federated, and only
as an aggregate **scatter matrix** (a second-moment sufficient statistic), so no
single prompt or response is recoverable from a contribution. Three pieces, all
offline-first, deterministic, and gated by the platform's existing discipline:

* :class:`Contribution` is a member's numeric, raw-text-free update: the
  ``d × d`` weighted scatter of its local prompt-embedding subspace, **clipped**
  to a sensitivity bound, optionally perturbed by a **differential-privacy**
  Gaussian mechanism, and **masked** for secure aggregation so an individual
  update is indistinguishable from noise on the wire. It carries a consent
  attestation and a residency tag, never raw traffic. :class:`ContributionBuilder`
  (and :meth:`FederatedImprovement.build_contribution`) produce one from a
  member's local data behind the consent ledger and the residency posture.
* :class:`SecureAggregator` sums the masked contributions — the pairwise masks
  cancel across the exact participant set, so the aggregator recovers the fleet
  scatter **without ever seeing an unmasked individual update** — refuses a round
  with fewer than ``min_contributors`` members (round-level k-anonymity), and
  extracts the consensus :class:`FederatedSubspace` by a deterministic federated
  PCA (top eigenvectors of the aggregate scatter).
* :func:`refit_with_subspace` re-fits a member's **own** local adapter against the
  shared subspace: the geometry is the fleet's, the codes and the grounded
  ``targets`` are the member's own local data, so adoption imports the fleet's
  learned structure without importing anyone's text. The refit adapter clears the
  existing :class:`~vincio.optimize.local_adaptation.AdapterGate` (no-regression),
  is versioned in the existing
  :class:`~vincio.optimize.local_adaptation.AdapterRegistry`, applied with
  :meth:`~vincio.core.app.ContextApp.use_local_adapter`, and rolled back on
  regression.

:class:`FederatedImprovement` (``app.federated_improvement`` / ``app.adopt_federated``)
wires these into one gated round: aggregate the fleet's contributions, refit the
adopting member's adapter against the shared subspace, gate it against the member's
base on a held-out set, and **adopt or roll back** — every decision on the shared
audit chain and event bus, under the same safety gates a hosted fine-tune job
clears. Nothing but numeric, masked, bounded-sensitivity aggregates ever crosses a
trust boundary; this is a library capability inside your process, never a hosted
federation service.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import OptimizationError
from .local_adaptation import (
    AdapterGate,
    AdapterRegistry,
    LocalAdapter,
    _message_text,
    _project,
    _unit,
)
from .self_improvement import CanaryVerdict

if TYPE_CHECKING:
    from ..core.app import ContextApp
    from ..evals.datasets import Dataset
    from ..evals.reports import EvalReport
    from ..providers.base import ModelProvider
    from ..retrieval.embeddings import Embedder
    from .distill import TrainingSet

__all__ = [
    "FederatedError",
    "PrivacyConfig",
    "PrivacyAccounting",
    "Contribution",
    "ContributionBuilder",
    "FederatedSubspace",
    "SecureAggregator",
    "refit_with_subspace",
    "FederatedPolicy",
    "FederatedEvent",
    "FederatedRoundResult",
    "FederatedImprovement",
]


class FederatedError(OptimizationError):
    """A federated round could not proceed safely.

    Raised on too few contributors, a base-model or embedding-dimension mismatch
    between members, a denied training consent, or a residency violation.
    Inherits :class:`~vincio.core.errors.OptimizationError`'s stable ``.code`` so
    it carries the same remediation surface as every other optimization failure.
    """


# ---------------------------------------------------------------------------
# Dense symmetric linear algebra (pure-Python, deterministic)
# ---------------------------------------------------------------------------


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _frobenius(matrix: list[list[float]]) -> float:
    return math.sqrt(sum(v * v for row in matrix for v in row))


def _scale(matrix: list[list[float]], factor: float) -> list[list[float]]:
    return [[v * factor for v in row] for row in matrix]


def _add_into(target: list[list[float]], other: list[list[float]]) -> None:
    for i, row in enumerate(other):
        target_row = target[i]
        for j, v in enumerate(row):
            target_row[j] += v


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix]


def _scatter(vectors: list[list[float]], weights: list[float], dim: int) -> list[list[float]]:
    """Weighted second-moment ``Σ wᵢ·(vᵢ ⊗ vᵢ)`` of unit vectors.

    The ``d × d`` sufficient statistic of a member's prompt-embedding subspace:
    it captures *which directions the member's traffic concentrates along and how
    strongly*, with no way to recover an individual ``vᵢ`` (let alone the prompt
    it came from). This is what makes the contribution an aggregate, not an
    example store.
    """
    matrix = _zeros(dim, dim)
    for vector, weight in zip(vectors, weights, strict=True):
        unit = _unit(vector)
        for i in range(dim):
            ui = unit[i] * weight
            if ui == 0.0:
                continue
            row = matrix[i]
            for j in range(dim):
                row[j] += ui * unit[j]
    return matrix


def _top_eigenvectors(
    matrix: list[list[float]], rank: int, *, iterations: int = 128, tol: float = 1e-9
) -> tuple[list[list[float]], list[float]]:
    """Top-``rank`` eigenvectors/eigenvalues of a symmetric PSD matrix.

    Deterministic power iteration with deflation: a fixed, index-dependent start
    vector (never accidentally orthogonal to the leading eigenvector) is iterated
    to convergence, recorded, and deflated out before the next direction. For the
    small ``d`` and ``rank`` a federated subspace uses this converges in a handful
    of iterations and is fully reproducible — the same aggregate scatter always
    yields the same consensus basis.
    """
    dim = len(matrix)
    work = [list(row) for row in matrix]
    basis: list[list[float]] = []
    values: list[float] = []
    for k in range(min(rank, dim)):
        # A deterministic, non-degenerate start: a smooth ramp perturbed by the
        # deflation index so successive extractions don't share a seed vector.
        vector = _unit([1.0 + ((i * 31 + k * 7) % 13) / 13.0 for i in range(dim)])
        value = 0.0
        for _ in range(iterations):
            nxt = _matvec(work, vector)
            norm = math.sqrt(sum(v * v for v in nxt))
            if norm <= tol:
                vector = [0.0] * dim
                value = 0.0
                break
            nxt = [v / norm for v in nxt]
            value = norm
            if sum(abs(a - b) for a, b in zip(nxt, vector, strict=True)) <= tol:
                vector = nxt
                break
            vector = nxt
        if value <= tol or not any(vector):
            break
        basis.append(vector)
        values.append(round(value, 9))
        # Deflate: remove the captured rank-1 component so the next iteration
        # finds the next-strongest orthogonal direction.
        for i in range(dim):
            vi = value * vector[i]
            row = work[i]
            for j in range(dim):
                row[j] -= vi * vector[j]
    return basis, values


# ---------------------------------------------------------------------------
# Privacy configuration & accounting
# ---------------------------------------------------------------------------


class PrivacyConfig(BaseModel):
    """How a contribution is made privacy-preserving before it leaves a member.

    Three composable protections, all bounding what the aggregator (or a network
    observer) can learn about one member:

    * ``clip_norm`` bounds each contribution's Frobenius norm, capping a single
      member's sensitivity (its maximum influence on the merged result) — the
      precondition for any differential-privacy guarantee and a brake on a
      poisoned outlier.
    * ``dp_epsilon`` (when set) adds calibrated Gaussian noise via the analyze-Gauss
      mechanism so the merged scatter is ``(ε, δ)``-differentially private with
      respect to any single example; ``None`` disables noise (the exact,
      deterministic default — clipping and masking still apply).
    * ``secure_aggregation`` masks each contribution with pairwise masks that
      cancel across the participant set, so the aggregator only ever sees the
      sum, never an individual update.

    ``min_contributors`` is the round-level k-anonymity floor: the aggregator
    refuses to publish a subspace distilled from fewer than this many members.
    ``seed`` keeps the (otherwise random) DP noise and secure-aggregation masks
    reproducible for offline testing; a real deployment derives the mask seeds
    from a key-agreement protocol instead.
    """

    clip_norm: float = 1.0
    dp_epsilon: float | None = None
    dp_delta: float = 1e-5
    secure_aggregation: bool = True
    min_contributors: int = 2
    seed: int = 0

    def noise_sigma(self) -> float:
        """Gaussian-mechanism standard deviation for the configured ``ε``/``δ``.

        Zero when ``dp_epsilon`` is unset. Otherwise the standard analyze-Gauss
        scale ``σ = clip_norm·√(2·ln(1.25/δ)) / ε`` — sensitivity over privacy
        budget — so a tighter ``ε`` injects more noise.
        """
        if self.dp_epsilon is None or self.dp_epsilon <= 0.0:
            return 0.0
        return self.clip_norm * math.sqrt(2.0 * math.log(1.25 / self.dp_delta)) / self.dp_epsilon


class PrivacyAccounting(BaseModel):
    """The privacy posture actually applied to a merged subspace — for audit."""

    secure_aggregation: bool = True
    clip_norm: float = 1.0
    dp_epsilon: float | None = None
    dp_delta: float = 1e-5
    noise_sigma: float = 0.0
    min_contributors: int = 2
    contributor_count: int = 0


# ---------------------------------------------------------------------------
# The contribution (numeric, raw-text-free)
# ---------------------------------------------------------------------------


def _mask_matrix(seed_text: str, dim: int, sign: float) -> list[list[float]]:
    """A deterministic ``d × d`` pseudo-random mask from a shared seed.

    Two members holding the same pairwise seed generate the identical mask; one
    adds ``+mask`` and the other ``-mask`` so the pair cancels in the aggregate.
    The seed stands in for a key-agreement-derived shared secret.
    """
    rng = random.Random(seed_text)
    return [[sign * (rng.random() - 0.5) for _ in range(dim)] for _ in range(dim)]


class Contribution(BaseModel):
    """One member's privacy-preserving federated update — numeric, no raw traffic.

    The only thing that leaves a member: the ``d × d`` weighted ``scatter`` of its
    local prompt-embedding subspace (clipped, optionally DP-noised, optionally
    masked), the example ``n_examples`` *count*, a ``training_set_hash``, and a
    consent + residency attestation. There is no prompt and no response anywhere
    in this object — the federated artifact is geometry, not content. ``masked``
    records whether secure-aggregation masks are folded into ``scatter`` (a masked
    contribution is indistinguishable from noise until summed with its peers).
    """

    member_id: str
    round_id: str = "round"
    base_model: str = ""
    embed_dim: int = 0
    scatter: list[list[float]] = Field(default_factory=list)
    n_examples: int = 0
    local_rank: int = 0
    training_set_hash: str = ""
    clipped: bool = False
    clipped_scale: float = 1.0
    dp_epsilon: float | None = None
    masked: bool = False
    # The reliability weight folded into ``scatter`` before masking (1.0 = none).
    # A reputation-discounted member releases a proportionally smaller-pull update;
    # because the scale is applied before the secure-aggregation masks, the masks
    # still cancel exactly in the aggregate. See :mod:`vincio.optimize.reputation`.
    reputation_weight: float = 1.0
    # Governance attestation that travels with the numeric update.
    consent_basis: str | None = None
    residency: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)

    @property
    def digest(self) -> str:
        """Content address over the numeric payload (member + round + scatter)."""
        payload = json.dumps(
            {
                "member_id": self.member_id,
                "round_id": self.round_id,
                "base_model": self.base_model,
                "scatter": self.scatter,
                "masked": self.masked,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ContributionBuilder:
    """Build a :class:`Contribution` from a member's local data — never its text.

    Embeds the member's prompts, forms the weighted subspace scatter, **clips** it
    to the configured sensitivity bound, optionally adds the **DP** Gaussian
    mechanism's noise, and optionally folds in **secure-aggregation** masks against
    the named ``participants`` — leaving a numeric object that carries no prompt or
    response. The same embedder a member fits its local adapter with must build the
    scatter, so the federated geometry lines up across the fleet.
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        privacy: PrivacyConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.privacy = privacy or PrivacyConfig()

    async def build(
        self,
        training_set: TrainingSet,
        base_model: str,
        *,
        member_id: str,
        participants: list[str] | None = None,
        round_id: str = "round",
        consent_basis: str | None = None,
        residency: str = "",
        min_examples: int = 1,
        reputation_weight: float = 1.0,
    ) -> Contribution:
        """Build this member's numeric contribution from a grounded training set.

        ``reputation_weight`` (``1.0`` by default) is the reliability weight the
        federated round assigns this member from its track record; it scales the
        contribution's signal *before* the secure-aggregation masks are folded in,
        so a discounted member pulls the consensus less while the masks still
        cancel exactly. See :mod:`vincio.optimize.reputation`.
        """
        examples = list(training_set.examples)
        if len(examples) < min_examples:
            raise FederatedError(
                f"federated contribution needs at least {min_examples} grounded examples; "
                f"got {len(examples)}"
            )
        embedder = self.embedder
        if embedder is None:
            from ..retrieval.embeddings import LocalHashEmbedder

            embedder = LocalHashEmbedder()
        from ..retrieval.embeddings import embed_texts

        prompts = [_message_text(e.messages, "user") for e in examples]
        supports = [float(e.support) for e in examples]
        vectors = await embed_texts(embedder, prompts)
        embed_dim = len(vectors[0]) if vectors else int(getattr(embedder, "dim", 0))
        scatter = _scatter(vectors, supports, embed_dim)

        # Estimate the member's local effective rank (for fleet-coverage reporting),
        # bounded by the example count.
        local_rank = min(len(examples), embed_dim)

        # 1. Clip the contribution's Frobenius norm — bound member sensitivity.
        privacy = self.privacy
        norm = _frobenius(scatter)
        clipped_scale = 1.0
        clipped = False
        if privacy.clip_norm > 0.0 and norm > privacy.clip_norm:
            clipped_scale = privacy.clip_norm / norm
            scatter = _scale(scatter, clipped_scale)
            clipped = True

        # 1b. Reliability weighting — scale the *signal* by the member's reputation
        #     weight before any noise or mask is added. A discounted member's pull on
        #     the consensus shrinks proportionally while the DP noise stays calibrated
        #     to the clip (so its privacy is unchanged) and the pairwise masks, added
        #     last, still cancel exactly in the aggregate.
        if reputation_weight != 1.0:
            scatter = _scale(scatter, reputation_weight)

        # 2. Differential-privacy Gaussian mechanism (opt-in, seeded for tests).
        sigma = privacy.noise_sigma()
        if sigma > 0.0:
            rng = random.Random(f"{privacy.seed}:dp:{member_id}:{round_id}")
            for row in scatter:
                for j in range(len(row)):
                    row[j] += rng.gauss(0.0, sigma)

        # 3. Secure-aggregation masks against every other participant (cancel in
        #    the aggregate sum); a masked contribution looks like noise alone.
        masked = False
        if privacy.secure_aggregation and participants:
            others = [m for m in participants if m != member_id]
            if others:
                for other in others:
                    lo, hi = sorted((member_id, other))
                    sign = 1.0 if member_id == lo else -1.0
                    seed_text = f"{privacy.seed}:mask:{round_id}:{lo}:{hi}"
                    _add_into(scatter, _mask_matrix(seed_text, embed_dim, sign))
                masked = True

        return Contribution(
            member_id=member_id,
            round_id=round_id,
            base_model=base_model,
            embed_dim=embed_dim,
            scatter=scatter,
            n_examples=len(examples),
            local_rank=local_rank,
            training_set_hash=hashlib.sha256(
                training_set.to_jsonl().encode("utf-8")
            ).hexdigest(),
            clipped=clipped,
            clipped_scale=round(clipped_scale, 9),
            dp_epsilon=privacy.dp_epsilon,
            masked=masked,
            reputation_weight=round(reputation_weight, 9),
            consent_basis=consent_basis,
            residency=residency,
            provenance={
                "embedder": type(embedder).__name__,
                "grounded_fraction": training_set.grounded_fraction,
            },
        )


# ---------------------------------------------------------------------------
# The merged subspace
# ---------------------------------------------------------------------------


class FederatedSubspace(BaseModel):
    """The fleet-consensus low-rank subspace distilled from a secure aggregation.

    The output of a round: an ``r × d`` orthonormal ``basis`` (the directions the
    fleet's traffic concentrates along, recovered as the top eigenvectors of the
    aggregate scatter) and their ``energy`` (eigenvalues, the consensus support per
    direction). A member adopts it by re-fitting its **own** local adapter against
    this geometry — see :func:`refit_with_subspace`. The subspace carries the
    privacy posture it was produced under but no member's data.
    """

    round_id: str = "round"
    base_model: str = ""
    embed_dim: int = 0
    rank: int = 0
    basis: list[list[float]] = Field(default_factory=list)
    energy: list[float] = Field(default_factory=list)
    contributor_count: int = 0
    privacy: PrivacyAccounting = Field(default_factory=PrivacyAccounting)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @property
    def digest(self) -> str:
        """Content address over the consensus geometry — two equal rounds match."""
        payload = json.dumps(
            {
                "base_model": self.base_model,
                "embed_dim": self.embed_dim,
                "basis": self.basis,
                "round_id": self.round_id,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SecureAggregator:
    """Merge masked contributions into a :class:`FederatedSubspace` — never seeing one.

    Sums the contributions coordinate-wise: when secure aggregation is on, the
    pairwise masks cancel across the exact participant set, so the aggregator
    recovers the true fleet scatter *without* any single member's unmasked update
    being observable. Refuses a round below ``min_contributors`` (round-level
    k-anonymity), enforces a single ``base_model`` and embedding dimension across
    members, optionally enforces a residency allow-list, and extracts the consensus
    basis by deterministic federated PCA (:func:`_top_eigenvectors`).

    Bind a :class:`~vincio.optimize.reputation.ReputationLedger` (or pass explicit
    ``weights`` to :meth:`aggregate`) to **reliability-weight** the merge: a
    member's contribution is scaled by its earned reputation, so a repeatedly
    regressing or adversarial member is discounted without being singled out.
    Because the masks only cancel at unit weight, a masked contribution must
    already carry the assigned weight (folded in at build); the aggregator enforces
    that and applies the weight directly only on the unmasked path.
    """

    def __init__(
        self,
        *,
        privacy: PrivacyConfig | None = None,
        rank: int = 8,
        allowed_regions: list[str] | None = None,
        reputation: Any | None = None,
    ) -> None:
        self.privacy = privacy or PrivacyConfig()
        self.rank = rank
        self.allowed_regions = list(allowed_regions) if allowed_regions else []
        self.reputation = reputation

    def aggregate(
        self,
        contributions: list[Contribution],
        *,
        round_id: str = "round",
        weights: dict[str, float] | None = None,
    ) -> FederatedSubspace:
        """Securely merge ``contributions`` into the fleet-consensus subspace.

        When a :class:`~vincio.optimize.reputation.ReputationLedger` is bound (or
        ``weights`` are passed explicitly), each member's contribution is weighted
        by its reputation before the consensus is distilled.
        """
        members = {c.member_id for c in contributions}
        if len(members) < self.privacy.min_contributors:
            raise FederatedError(
                f"federated round needs at least {self.privacy.min_contributors} "
                f"distinct contributors; got {len(members)}"
            )
        base_models = {c.base_model for c in contributions}
        if len(base_models) != 1:
            raise FederatedError(
                f"contributions span multiple base models {sorted(base_models)}; "
                "a federated round merges one base model"
            )
        dims = {c.embed_dim for c in contributions}
        if len(dims) != 1:
            raise FederatedError(
                f"contributions span multiple embedding dimensions {sorted(dims)}; "
                "all members must use the same embedder"
            )
        if self.allowed_regions:
            violating = sorted(
                {c.residency for c in contributions if c.residency and c.residency not in self.allowed_regions}
            )
            if violating:
                raise FederatedError(
                    f"contributions from non-allowed residency regions {violating}; "
                    f"allowed: {sorted(self.allowed_regions)}"
                )
        base_model = next(iter(base_models))
        dim = next(iter(dims))

        # Resolve the per-member reliability weights: an explicit ``weights`` map
        # wins, else a bound reputation ledger assigns them, else every member is
        # weighted equally (the unweighted round, byte-identical to before).
        applied_weights = self._resolve_weights(members, weights, round_id=round_id)

        # Sum the (masked) contributions — masks cancel across participants, so this
        # is the true fleet scatter and no individual update was ever unmasked. A
        # reputation weight is folded in here: on the unmasked path the aggregator
        # scales the contribution directly; on the masked path the weight must
        # already be carried (folded in before masking, or the masks would not
        # cancel), so the aggregator only enforces consistency.
        total = _zeros(dim, dim)
        for contribution in contributions:
            assigned = applied_weights.get(contribution.member_id, 1.0)
            scatter = self._weighted_scatter(contribution, assigned)
            _add_into(total, scatter)
        # Symmetrize to absorb any floating-point drift before the eigensolve.
        for i in range(dim):
            for j in range(i + 1, dim):
                avg = 0.5 * (total[i][j] + total[j][i])
                total[i][j] = total[j][i] = avg

        basis, energy = _top_eigenvectors(total, self.rank)
        accounting = PrivacyAccounting(
            secure_aggregation=any(c.masked for c in contributions),
            clip_norm=self.privacy.clip_norm,
            dp_epsilon=self.privacy.dp_epsilon,
            dp_delta=self.privacy.dp_delta,
            noise_sigma=round(self.privacy.noise_sigma(), 9),
            min_contributors=self.privacy.min_contributors,
            contributor_count=len(members),
        )
        return FederatedSubspace(
            round_id=round_id,
            base_model=base_model,
            embed_dim=dim,
            rank=len(basis),
            basis=basis,
            energy=energy,
            contributor_count=len(members),
            privacy=accounting,
            provenance={
                "members": sorted(members),
                "total_examples": sum(c.n_examples for c in contributions),
                "max_local_rank": max((c.local_rank for c in contributions), default=0),
                "reputation_weighted": any(w != 1.0 for w in applied_weights.values()),
                "reputation_weights": {m: applied_weights[m] for m in sorted(applied_weights)},
            },
        )

    def _resolve_weights(
        self,
        members: set[str],
        weights: dict[str, float] | None,
        *,
        round_id: str,
    ) -> dict[str, float]:
        """The per-member weight map for this round (explicit > ledger > equal)."""
        if weights is not None:
            return {m: float(weights.get(m, 1.0)) for m in members}
        if self.reputation is not None:
            assignment = self.reputation.assign(members, round_id=round_id)
            return {m: assignment.get(m) for m in members}
        return {m: 1.0 for m in members}

    @staticmethod
    def _weighted_scatter(contribution: Contribution, assigned: float) -> list[list[float]]:
        """Apply ``assigned`` to ``contribution.scatter``, mask-safe.

        If the contribution already carries the assigned weight (the production
        path — the member folded it in before masking) the scatter is summed as-is.
        Otherwise the weight is applied here, which is only safe for an unmasked
        contribution; re-weighting a masked one would break mask cancellation, so
        that case is refused with a clear error.
        """
        carried = contribution.reputation_weight
        if abs(carried - assigned) <= 1e-9:
            return contribution.scatter
        if contribution.masked:
            raise FederatedError(
                f"cannot re-weight masked contribution from {contribution.member_id!r} "
                f"(carries weight {carried:g}, ledger assigns {assigned:g}); a masked "
                "contribution must be built with its reputation weight so the secure-"
                "aggregation masks still cancel — rebuild it against the current ledger"
            )
        factor = assigned / carried if carried != 0.0 else assigned
        return _scale(contribution.scatter, factor)


# ---------------------------------------------------------------------------
# Adoption: refit a member's own adapter against the shared subspace
# ---------------------------------------------------------------------------


async def refit_with_subspace(
    subspace: FederatedSubspace,
    training_set: TrainingSet,
    *,
    embedder: Embedder | None = None,
    gate: float = 0.85,
    scale: float = 1.0,
    name: str = "federated-adapter",
    min_examples: int = 1,
    metadata: dict[str, Any] | None = None,
) -> LocalAdapter:
    """Re-fit a member's **own** local adapter against the shared fleet subspace.

    The geometry (``basis``) is the fleet's consensus; the codes and the grounded
    ``targets`` are the member's own local data — so adoption imports the fleet's
    learned structure without importing anyone's text. The result is an ordinary
    :class:`~vincio.optimize.local_adaptation.LocalAdapter`: it applies through
    :class:`~vincio.optimize.local_adaptation.AdaptedProvider`, clears the existing
    :class:`~vincio.optimize.local_adaptation.AdapterGate`, and versions in the
    existing :class:`~vincio.optimize.local_adaptation.AdapterRegistry` unchanged.
    """
    examples = list(training_set.examples)
    if len(examples) < min_examples:
        raise FederatedError(
            f"federated adoption needs at least {min_examples} grounded examples; "
            f"got {len(examples)}"
        )
    if not subspace.basis:
        raise FederatedError("federated subspace is empty; cannot refit an adapter against it")
    if embedder is None:
        from ..retrieval.embeddings import LocalHashEmbedder

        embedder = LocalHashEmbedder(dim=subspace.embed_dim or 256)
    from ..retrieval.embeddings import embed_texts

    prompts = [_message_text(e.messages, "user") for e in examples]
    targets = [_message_text(e.messages, "assistant") for e in examples]
    supports = [float(e.support) for e in examples]
    vectors = await embed_texts(embedder, prompts)
    embed_dim = len(vectors[0]) if vectors else subspace.embed_dim
    if embed_dim != subspace.embed_dim:
        raise FederatedError(
            f"member embedding dim {embed_dim} != subspace dim {subspace.embed_dim}; "
            "adopt the subspace with the same embedder family the fleet used"
        )
    codes = [_project(_unit(v), subspace.basis) for v in vectors]
    return LocalAdapter(
        name=name,
        base_model=subspace.base_model,
        rank=subspace.rank,
        embed_dim=embed_dim,
        gate=gate,
        scale=scale,
        basis=[list(row) for row in subspace.basis],
        codes=codes,
        targets=targets,
        supports=supports,
        n_examples=len(examples),
        training_set_hash=hashlib.sha256(training_set.to_jsonl().encode("utf-8")).hexdigest(),
        provenance={
            "federated_round": subspace.round_id,
            "federated_subspace_digest": subspace.digest,
            "contributor_count": subspace.contributor_count,
            "embedder": type(embedder).__name__,
        },
        metadata={**(metadata or {}), "federated": True},
    )


# ---------------------------------------------------------------------------
# The policy, events & the gated round
# ---------------------------------------------------------------------------


class FederatedPolicy(BaseModel):
    """The opt-in contract for one gated federated-improvement round.

    Composes the :class:`PrivacyConfig` that bounds what crosses a trust boundary
    with the adapter shape (low-rank ``rank``, acceptance ``gate``/``scale``) and
    the no-regression gate (``metric``/``regression_threshold`` — default ``0.0``
    enforces at-least-as-good) that promotion clears. ``require_consent`` gates a
    contribution behind the consent ledger's TRAINING purpose; ``allowed_regions``
    enforces the residency posture at the aggregator. ``dry_run`` aggregates and
    gates without adopting; ``keep_versions`` bounds the registry footprint.
    """

    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    rank: int = 8
    gate: float = 0.85
    scale: float = 1.0
    min_examples: int = 4
    name: str = "federated-adapter"
    round_id: str = "round"
    # Governance.
    require_consent: bool = False
    consent_subject: str | None = None
    allowed_regions: list[str] | None = None
    # No-regression gate (shared with the local-adaptation / model-swap gate).
    metric: str = "lexical_overlap"
    regression_threshold: float = 0.0
    require_significance: bool = True
    min_samples: int = 4
    alpha: float = 0.05
    gates: dict[str, str] | None = None
    # Reputation (opt-in: active only when a ReputationLedger is bound). When
    # ``record_reputation`` is set, the round records its gate verdict back to the
    # ledger for every contributor, so a member's reliability accrues across rounds.
    record_reputation: bool = True
    # Eval & lifecycle.
    concurrency: int = 4
    keep_versions: int = 10
    dry_run: bool = False


FederatedPhase = str


class FederatedEvent(BaseModel):
    """One event in a federated round — stamped on the audit chain & event bus."""

    phase: FederatedPhase = "observe"
    action: str = ""
    reason: str = ""
    contributor_count: int = 0
    subspace_rank: int | None = None
    subspace_digest: str | None = None
    adapter_version: int | None = None
    adapter_digest: str | None = None
    verdict: CanaryVerdict | None = None
    rolled_back_to: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class FederatedRoundResult(BaseModel):
    """The outcome of one gated federated-improvement round."""

    adopted: bool = False
    round_id: str = "round"
    base_model: str = ""
    adapter_name: str = ""
    adapter_version: int | None = None
    adapter_digest: str | None = None
    subspace_digest: str | None = None
    subspace_rank: int = 0
    contributor_count: int = 0
    verdict: CanaryVerdict | None = None
    rolled_back_to: int | None = None
    privacy: PrivacyAccounting | None = None
    reputation_weights: dict[str, float] | None = None
    reason: str = ""


class FederatedImprovement:
    """Drive one gated, privacy-preserving federated round for the adopting member.

    The cross-org analogue of
    :class:`~vincio.optimize.local_adaptation.ContinualAdaptation`: securely
    aggregate the fleet's :class:`Contribution`\\ s into a :class:`FederatedSubspace`,
    re-fit *this* member's own adapter against that shared geometry, gate it against
    the member's base on a held-out set, and **adopt or roll back** — every version
    in the :class:`~vincio.optimize.local_adaptation.AdapterRegistry`, every decision
    on the shared audit chain and event bus, promotion held by the same
    no-regression discipline a hosted fine-tune job clears. Only numeric, masked,
    bounded-sensitivity aggregates ever cross a trust boundary.

    Bind a held-out ``dataset`` to enable gating (without one the round aggregates
    and refits but refuses to adopt, since it cannot prove no regression). The
    adopting member always refits over its **own** local data (``runs`` /
    ``training_set`` / its captured traces) — it never sees another member's data.

    Bind a :class:`~vincio.optimize.reputation.ReputationLedger` (via
    :meth:`~vincio.core.app.ContextApp.use_reputation_ledger`, or the ``reputation``
    argument) to **reliability-weight** the round: each member's contribution is
    discounted by its earned track record against the no-regression gate, and the
    round records its own verdict back to the ledger. The discount is bounded and
    reversible — a weight only ever lowers a member's pull, and adoption still
    clears the same gate — so a bad reputation can never bypass the quality bar.
    """

    def __init__(
        self,
        app: ContextApp,
        policy: FederatedPolicy | None = None,
        *,
        dataset: Dataset | None = None,
        registry: AdapterRegistry | None = None,
        embedder: Embedder | None = None,
        base_model: str | None = None,
        reputation: Any | None = None,
    ) -> None:
        self.app = app
        self.policy = policy or FederatedPolicy()
        self.dataset = dataset
        self.registry = registry or AdapterRegistry()
        self.embedder = embedder or app.embedder
        self.base_model = base_model or app.model
        # The cross-fleet reputation ledger (opt-in): when bound, contributions are
        # weighted by each member's earned reliability and the round records its
        # gate verdict back. Defaults to the app's attached ledger, if any.
        self.reputation = reputation if reputation is not None else getattr(
            app, "reputation_ledger", None
        )
        self.events: list[FederatedEvent] = []
        self.result = FederatedRoundResult(
            round_id=self.policy.round_id,
            base_model=self.base_model,
            adapter_name=self.policy.name,
        )

    # -- contribution (this member's private update) ------------------------

    async def build_contribution(
        self,
        *,
        member_id: str,
        participants: list[str] | None = None,
        runs: list[Any] | None = None,
        training_set: TrainingSet | None = None,
        consent_subject: str | None = None,
        residency: str | None = None,
    ) -> Contribution:
        """Build this member's numeric, raw-text-free contribution to the round.

        Curates a grounded training set (from ``runs``, a prebuilt
        ``training_set``, or the app's captured traces), enforces the consent
        ledger's TRAINING purpose when ``require_consent`` is set, stamps the
        residency tag from the app's posture, and returns a :class:`Contribution`
        that carries only the numeric subspace scatter — never a prompt or a
        response.
        """
        policy = self.policy
        subject = consent_subject if consent_subject is not None else policy.consent_subject
        consent_basis = self._check_consent(subject)
        region = residency if residency is not None else self._residency_tag()
        corpus = self._build_training_set(runs, training_set)

        # Differential-privacy budget gate: when an accountant is attached and the
        # policy configures the Gaussian mechanism, this contribution composes the
        # subject's cross-round privacy budget. A contribution that would exceed it
        # is refused; a budget set to down-weight is honoured by releasing a *more
        # private* contribution — the Gaussian mechanism's ``ε`` is scaled down by
        # the same factor, so its noise rises (more noise at a fixed clip → the
        # released noise multiplier ``z = σ/Δ`` rises to ``z/downweight``) and the
        # discounted cost the accountant recorded matches the geometry released.
        # (Scaling the clip alone would scale ``σ`` with it and leave ``z`` — and the
        # true privacy cost — unchanged, an under-count.)
        privacy = policy.privacy
        spend = self._charge_privacy(subject, member_id=member_id)
        if spend is not None and spend.downweight < 1.0 and privacy.dp_epsilon is not None:
            privacy = privacy.model_copy(
                update={"dp_epsilon": privacy.dp_epsilon * spend.downweight}
            )

        # Reliability weighting: a member discounted by its track record releases a
        # proportionally smaller-pull update. The weight is public (derived from the
        # gate verdicts on the audit chain), folded into the signal before masking so
        # the secure-aggregation masks still cancel exactly.
        reputation_weight = (
            self.reputation.weight(member_id) if self.reputation is not None else 1.0
        )

        builder = ContributionBuilder(embedder=self.embedder, privacy=privacy)
        contribution = await builder.build(
            corpus,
            self.base_model,
            member_id=member_id,
            participants=participants,
            round_id=policy.round_id,
            consent_basis=consent_basis,
            residency=region,
            min_examples=1,
            reputation_weight=reputation_weight,
        )
        self.app.audit.record(
            "federated_contribution",
            decision="allow",
            resource=member_id,
            details={
                "round_id": policy.round_id,
                "digest": contribution.digest,
                "n_examples": contribution.n_examples,
                "masked": contribution.masked,
                "dp_epsilon": contribution.dp_epsilon,
                "reputation_weight": contribution.reputation_weight,
                "residency": region,
                "consent_basis": consent_basis,
                "privacy_spent_epsilon": spend.cumulative_epsilon if spend else None,
                "privacy_downweight": spend.downweight if spend else None,
            },
        )
        self.app.events.emit("federated.contribute", contribution.model_dump(exclude={"scatter"}))
        return contribution

    def _check_consent(self, subject: str | None) -> str | None:
        if not self.policy.require_consent:
            return None
        ledger = getattr(self.app, "consent_ledger", None)
        if ledger is None:
            raise FederatedError(
                "federated policy requires consent but no consent ledger is attached; "
                "call app.use_consent_ledger(...) first"
            )
        if not subject:
            raise FederatedError(
                "federated policy requires consent but no consent_subject was provided"
            )
        from ..governance.consent import Purpose

        decision = ledger.check(subject, Purpose.TRAINING)
        if not decision.allowed:
            raise FederatedError(
                f"training consent denied for subject {subject!r}: {decision.reason}"
            )
        return decision.lawful_basis

    def _charge_privacy(self, subject: str | None, *, member_id: str) -> Any:
        """Compose this contribution onto the subject's DP budget.

        Returns the recorded :class:`~vincio.governance.privacy.PrivacySpend`, or
        ``None`` when no accountant is attached or the policy configures no
        Gaussian mechanism (nothing differentially private to account). Raises
        :class:`~vincio.governance.privacy.PrivacyBudgetError` when the budget
        refuses the contribution.
        """
        accountant = getattr(self.app, "privacy_accountant", None)
        privacy = self.policy.privacy
        if (
            accountant is None
            or not subject
            or privacy.dp_epsilon is None
            or privacy.dp_epsilon <= 0.0
            or privacy.clip_norm <= 0.0
        ):
            return None
        from ..governance.privacy import PrivacyMechanism

        # Noise relative to the clipped L2 sensitivity: z = σ/Δ, independent of the
        # clip norm itself (σ = clip·√(2 ln(1.25/δ))/ε, so z = √(2 ln(1.25/δ))/ε).
        noise_multiplier = privacy.noise_sigma() / privacy.clip_norm
        mechanism = PrivacyMechanism(
            label="federated_contribution", noise_multiplier=noise_multiplier
        )
        return accountant.charge(
            subject,
            mechanism,
            operation="federated_contribution",
            round_id=self.policy.round_id,
            delta=privacy.dp_delta,
            details={"member_id": member_id, "round_id": self.policy.round_id},
        )

    def _residency_tag(self) -> str:
        regions = getattr(self.app.config.governance, "allowed_regions", []) or []
        return regions[0] if regions else ""

    # -- training-set assembly (this member's own data) --------------------

    def _build_training_set(
        self, runs: list[Any] | None, training_set: TrainingSet | None
    ) -> TrainingSet:
        if training_set is not None:
            return training_set
        from .distill import export_training_set, export_training_set_from_runs

        system = self.app.prompt_spec.role or self.app.prompt_spec.objective
        if runs is not None:
            return export_training_set_from_runs(runs, name=self.policy.name, system=system)
        traces: list[Any] = []
        exporter = self.app.tracer.exporter
        if hasattr(exporter, "load_all"):
            traces = exporter.load_all(limit=500)
        elif hasattr(exporter, "traces"):
            traces = list(exporter.traces)[-500:]
        return export_training_set(traces, name=self.policy.name, system=system)

    # -- evaluation (provider swap, memory-write-back disabled) ------------

    @staticmethod
    def _unwrap(provider: ModelProvider) -> ModelProvider:
        from .local_adaptation import AdaptedProvider

        return provider.base if isinstance(provider, AdaptedProvider) else provider

    async def _eval_report(self, provider: ModelProvider) -> EvalReport:
        from ..evals.runners import EvalRunner

        assert self.dataset is not None
        app = self.app
        original_provider = app._provider_instance
        original_write_back = app.config.memory.write_back
        app._provider_instance = provider
        app.config.memory.write_back = []
        try:
            metrics = [self.policy.metric]
            if self.policy.gates:
                metrics += [m for m in self.policy.gates if m not in metrics]
            runner = EvalRunner(app, metrics=metrics, concurrency=self.policy.concurrency)
            return await runner.arun(self.dataset, name=f"federated:{self.policy.name}")
        finally:
            app._provider_instance = original_provider
            app.config.memory.write_back = original_write_back

    # -- the streaming round ------------------------------------------------

    async def astream(
        self,
        *,
        contributions: list[Contribution],
        runs: list[Any] | None = None,
        training_set: TrainingSet | None = None,
        apply: bool = True,
    ) -> AsyncIterator[FederatedEvent]:
        """Run one gated federated round, yielding each phase as it lands.

        Sequence: ``observe → aggregate → refit → gate → adopt / rollback``. On a
        pass the refit adapter is registered, made the active head, and (with
        ``apply``) installed on the app via
        :meth:`~vincio.core.app.ContextApp.use_local_adapter`; on a fail it is
        refused and the registry head stays on the last known-good version.
        """
        from .local_adaptation import AdaptedProvider

        policy = self.policy
        base = self._unwrap(self.app._base_provider())
        self.result.contributor_count = len({c.member_id for c in contributions})
        yield self._emit(
            FederatedEvent(
                phase="observe",
                reason="federated round started",
                contributor_count=self.result.contributor_count,
            )
        )

        aggregator = SecureAggregator(
            privacy=policy.privacy,
            rank=policy.rank,
            allowed_regions=policy.allowed_regions,
            reputation=self.reputation,
        )
        subspace = aggregator.aggregate(contributions, round_id=policy.round_id)
        self.result.subspace_digest = subspace.digest
        self.result.subspace_rank = subspace.rank
        self.result.privacy = subspace.privacy
        applied_weights = subspace.provenance.get("reputation_weights") or {}
        if subspace.provenance.get("reputation_weighted"):
            self.result.reputation_weights = applied_weights
        yield self._emit(
            FederatedEvent(
                phase="aggregate",
                action="merged",
                contributor_count=subspace.contributor_count,
                subspace_rank=subspace.rank,
                subspace_digest=subspace.digest,
                reason=(
                    f"merged rank-{subspace.rank} subspace from "
                    f"{subspace.contributor_count} contributors"
                ),
                details={
                    "secure_aggregation": subspace.privacy.secure_aggregation,
                    "dp_epsilon": subspace.privacy.dp_epsilon,
                    "clip_norm": subspace.privacy.clip_norm,
                    "min_contributors": subspace.privacy.min_contributors,
                    "reputation_weighted": bool(subspace.provenance.get("reputation_weighted")),
                    "reputation_weights": applied_weights,
                },
            )
        )

        corpus = self._build_training_set(runs, training_set)
        if len(corpus) < policy.min_examples:
            self.result.reason = (
                f"only {len(corpus)} local grounded examples (< {policy.min_examples}); "
                "refusing to refit an adapter"
            )
            yield self._emit(
                FederatedEvent(phase="exhausted", action="skipped", reason=self.result.reason)
            )
            return

        adapter = await refit_with_subspace(
            subspace,
            corpus,
            embedder=self.embedder,
            gate=policy.gate,
            scale=policy.scale,
            name=policy.name,
            min_examples=policy.min_examples,
        )
        self.result.adapter_digest = adapter.digest
        yield self._emit(
            FederatedEvent(
                phase="refit",
                action="fit",
                subspace_digest=subspace.digest,
                adapter_digest=adapter.digest,
                reason=(
                    f"refit rank-{adapter.rank} adapter from {len(corpus)} local examples "
                    f"against the fleet subspace"
                ),
                details={"rank": adapter.rank, "size_bytes": adapter.size_bytes},
            )
        )

        if self.dataset is None:
            self.result.reason = "no held-out dataset bound; cannot gate (not adopting)"
            yield self._emit(
                FederatedEvent(phase="gate", action="skipped", reason=self.result.reason)
            )
            return

        adapted = AdaptedProvider(base, adapter, embedder=self.embedder)
        base_report = await self._eval_report(base)
        adapted_report = await self._eval_report(adapted)
        gate = AdapterGate(
            metric=policy.metric,
            regression_threshold=policy.regression_threshold,
            require_significance=policy.require_significance,
            min_samples=policy.min_samples,
            alpha=policy.alpha,
        )
        verdict = gate.evaluate(base_report, adapted_report)
        self.result.verdict = verdict

        # Safety/schema overlay: a failing gate blocks adoption regardless of metric.
        if verdict.passed and policy.gates:
            from ..evals.reports import evaluate_gates

            outcomes = evaluate_gates(adapted_report, policy.gates)
            failed = [k for k, v in outcomes.items() if not v["passed"]]
            if failed:
                verdict.passed = False
                verdict.reason = f"federated adapter safety gates failed: {failed}"

        yield self._emit(
            FederatedEvent(
                phase="gate",
                action="passed" if verdict.passed else "failed",
                verdict=verdict,
                subspace_digest=subspace.digest,
                reason=verdict.reason,
            )
        )

        # Reputation accrual: the round's gate verdict is each contributor's track
        # record for this round — a pass credits every contributor, a regression
        # debits them — composed onto the bound ledger and stamped on the audit
        # chain, so the next round's weights reflect how this one fared.
        self._record_reputation(contributions, verdict, weights=applied_weights)

        if not verdict.passed:
            current = self.registry.active(policy.name)
            self.result.adopted = False
            self.result.rolled_back_to = current.version if current else None
            self.result.reason = (
                f"federated adapter not adopted: {verdict.reason}"
                + (f"; kept v{current.version}" if current else "")
            )
            yield self._emit(
                FederatedEvent(
                    phase="rollback",
                    action="refused",
                    verdict=verdict,
                    rolled_back_to=self.result.rolled_back_to,
                    reason=self.result.reason,
                )
            )
            return

        if policy.dry_run:
            self.result.reason = f"dry run: federated adapter would be adopted ({verdict.reason})"
            yield self._emit(
                FederatedEvent(
                    phase="adopt", action="dry_run", verdict=verdict, reason=self.result.reason
                )
            )
            return

        stored = self.registry.register(adapter)
        self.registry.prune(policy.name, policy.keep_versions)
        if apply:
            self.app.use_local_adapter(stored)
        self.result.adopted = True
        self.result.adapter_version = stored.version
        self.result.reason = f"adopted federated adapter v{stored.version}: {verdict.reason}"
        yield self._emit(
            FederatedEvent(
                phase="adopt",
                action="adopted",
                adapter_version=stored.version,
                adapter_digest=stored.digest,
                subspace_digest=subspace.digest,
                verdict=verdict,
                reason=self.result.reason,
            )
        )

    async def aadopt(
        self,
        *,
        contributions: list[Contribution],
        runs: list[Any] | None = None,
        training_set: TrainingSet | None = None,
        apply: bool = True,
    ) -> FederatedRoundResult:
        """Run one round to completion and return its :class:`FederatedRoundResult`."""
        async for _ in self.astream(
            contributions=contributions, runs=runs, training_set=training_set, apply=apply
        ):
            pass
        return self.result

    def adopt(
        self,
        *,
        contributions: list[Contribution],
        runs: list[Any] | None = None,
        training_set: TrainingSet | None = None,
        apply: bool = True,
    ) -> FederatedRoundResult:
        """Sync wrapper over :meth:`aadopt`."""
        from ..providers.base import run_sync

        return run_sync(
            self.aadopt(
                contributions=contributions, runs=runs, training_set=training_set, apply=apply
            )
        )

    # -- internals ----------------------------------------------------------

    def _record_reputation(
        self,
        contributions: list[Contribution],
        verdict: CanaryVerdict,
        *,
        weights: dict[str, float],
    ) -> None:
        """Compose this round's gate verdict onto every contributor's reputation."""
        if self.reputation is None or not self.policy.record_reputation:
            return
        from .reputation import ReputationWeights

        self.reputation.record_round(
            (c.member_id for c in contributions),
            passed=verdict.passed,
            round_id=self.policy.round_id,
            weights=ReputationWeights(round_id=self.policy.round_id, weights=weights),
            details={"delta": verdict.delta, "metric": self.policy.metric},
        )

    def _emit(self, event: FederatedEvent) -> FederatedEvent:
        self.events.append(event)
        self.app.audit.record(
            "federated_improvement",
            decision="allow" if event.action not in ("skipped", "refused") else "deny",
            resource=self.policy.name,
            details={
                "phase": event.phase,
                "action": event.action,
                "reason": event.reason,
                "round_id": self.policy.round_id,
                "contributor_count": event.contributor_count,
                "subspace_digest": event.subspace_digest,
                "adapter_version": event.adapter_version,
                "adapter_digest": event.adapter_digest,
                "rolled_back_to": event.rolled_back_to,
            },
        )
        self.app.events.emit(f"federated.{event.phase}", event.model_dump())
        return event
