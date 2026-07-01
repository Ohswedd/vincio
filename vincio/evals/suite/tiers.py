"""Provenance tiers — the honesty contract of the open evaluation plane.

Every benchmark run carries a **provenance tier** that says, structurally, *how
real* the number is. The platform's "from fabricated/static to only live" design
rule is here made an enforced contract: the engine computes the tier a run is
*allowed* to claim from its actual inputs — the dataset's provenance and whether
the solver is a live model — and **refuses** to let a lower tier print a higher
tier's label.

============  =========================================================  ============  =========
Tier          What it is                                                 Reproducible  Gates CI?
============  =========================================================  ============  =========
``S`` Static  a small, bundled, *fabricated* fixture that exercises the  yes — byte-   yes
              adapter + metric end to end                                identical
``R`` Recorded a hash-pinned slice of the *real* public dataset replayed yes — from    yes
              against recorded model outputs                             the pin
``L`` Live    the full public dataset run against a live model+provider  no            no
============  =========================================================  ============  =========

A Tier-S mechanism check can never masquerade as a Tier-L score: the tier is a
property of the *execution*, not a label a caller asserts. This mechanizes the
discipline the README's *A note on claims* section enforces by hand.
"""

from __future__ import annotations

from enum import StrEnum

from ...core.errors import TierViolationError

__all__ = [
    "ProvenanceTier",
    "resolve_tier",
]


class ProvenanceTier(StrEnum):
    """How real a benchmark number is — ordered ``STATIC < RECORDED < LIVE``.

    The string value is the one-letter code (``"S"`` / ``"R"`` / ``"L"``) so a
    tier round-trips through JSON, a CLI flag, and a report cell unchanged.
    """

    STATIC = "S"
    RECORDED = "R"
    LIVE = "L"

    # -- properties -----------------------------------------------------------

    @property
    def code(self) -> str:
        """The one-letter tier code (``"S"`` / ``"R"`` / ``"L"``)."""
        return self.value

    @property
    def label(self) -> str:
        """The human-readable tier name (``"Static"`` / ``"Recorded"`` / ``"Live"``)."""
        return self.name.capitalize()

    @property
    def rank(self) -> int:
        """The realness rank used for ordering: ``STATIC=0 < RECORDED=1 < LIVE=2``."""
        return _RANK[self]

    @property
    def reproducible(self) -> bool:
        """Whether a run at this tier is offline-reproducible.

        Static and Recorded are reproducible (byte-identical / from the hash pin);
        Live is not — it only exists from a real key against a live provider.
        """
        return self is not ProvenanceTier.LIVE

    @property
    def gates_ci(self) -> bool:
        """Whether a run at this tier may gate CI.

        Only the reproducible tiers gate; a Live score is *reported, never gated*.
        """
        return self.reproducible

    # -- ordering -------------------------------------------------------------

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ProvenanceTier):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, ProvenanceTier):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, ProvenanceTier):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, ProvenanceTier):
            return NotImplemented
        return self.rank >= other.rank

    @classmethod
    def parse(cls, value: str | ProvenanceTier) -> ProvenanceTier:
        """Coerce a tier from its code (``"S"``), name (``"static"``), or itself."""
        if isinstance(value, cls):
            return value
        text = str(value).strip()
        if not text:
            raise TierViolationError("empty provenance tier")
        upper = text.upper()
        for tier in cls:
            if upper in (tier.value, tier.name):
                return tier
        raise TierViolationError(
            f"unknown provenance tier {value!r}; known: "
            f"{', '.join(t.value + ' (' + t.label + ')' for t in cls)}"
        )


_RANK: dict[ProvenanceTier, int] = {
    ProvenanceTier.STATIC: 0,
    ProvenanceTier.RECORDED: 1,
    ProvenanceTier.LIVE: 2,
}


def resolve_tier(
    requested: ProvenanceTier | str | None,
    *,
    dataset_ceiling: ProvenanceTier,
    solver_live: bool,
) -> ProvenanceTier:
    """Resolve the tier a run may honestly claim, or refuse.

    ``dataset_ceiling`` is the highest tier the dataset's provenance supports (a
    fabricated fixture caps at :attr:`~ProvenanceTier.STATIC`; a hash-pinned real
    slice at :attr:`~ProvenanceTier.RECORDED`; a full real dataset at
    :attr:`~ProvenanceTier.LIVE`). ``solver_live`` is whether the solver is a live
    model (a replay solver caps at :attr:`~ProvenanceTier.RECORDED`).

    The **achievable ceiling** is the lower of the two. When ``requested`` is
    ``None`` the achievable ceiling is returned. When ``requested`` exceeds the
    ceiling — a lower tier trying to print a higher tier's label — a
    :class:`~vincio.core.errors.TierViolationError` is raised. A request at or
    below the ceiling is honored verbatim (claiming a *more* conservative tier is
    always allowed).
    """
    solver_ceiling = ProvenanceTier.LIVE if solver_live else ProvenanceTier.RECORDED
    ceiling = min(dataset_ceiling, solver_ceiling, key=lambda t: t.rank)
    if requested is None:
        return ceiling
    want = ProvenanceTier.parse(requested)
    if want.rank > ceiling.rank:
        reason = (
            "the dataset is fabricated/static"
            if dataset_ceiling.rank < want.rank
            else "the solver replays recorded outputs (no live model)"
        )
        raise TierViolationError(
            f"cannot report tier {want.value} ({want.label}): the run's inputs only "
            f"support tier {ceiling.value} ({ceiling.label}) — {reason}. A lower tier "
            f"may not print a higher tier's label."
        )
    return want
