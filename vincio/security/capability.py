"""Capability-secure execution: information-flow labels, taint propagation,
unforgeable capability tokens, and a machine-checkable containment invariant.

Detection (PII / secrets / injection / RAG-poisoning) is necessary but not
sufficient: an attacker only needs one missed instruction inside a retrieved
document or a tool result to escalate to an unauthorized side effect. This
module provides *containment that holds even when detection misses*, by
separating the control plane (what the user authorized) from the data plane
(bytes that arrived from untrusted sources):

* :class:`TrustLabel` promotes provenance to a typed information-flow label
  (``trusted`` / ``untrusted`` / ``quarantined``). The label forms a lattice;
  any value derived from untrusted data is *tainted* by the least-trusted of
  its inputs — taint never decreases by combination.
* :class:`TaintedValue` carries a value together with its label and the
  provenance sources it was derived from, and propagates the label through
  ``map`` / ``derive`` so taint follows the data end-to-end.
* :class:`CapabilityToken` is an unforgeable, HMAC-signed grant minted by a
  :class:`CapabilityBroker` from the *user's* request — never from model
  output. A side-effecting tool call must present a capability whose authority
  traces back to the user; a value carrying an ``untrusted`` taint cannot mint
  one.
* :class:`ContainmentMonitor` records every capability exercise and its taint,
  and :func:`verify_containment` checks the invariant
  ``untrusted ⇒ no unapproved capability`` over a whole run, so containment is
  provable after the fact rather than assumed.

Everything here is deterministic and offline; it never depends on a model
judgment.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from ..core.types import TrustLevel
from ..core.utils import new_id, stable_hash, utcnow

__all__ = [
    "TrustLabel",
    "TaintedValue",
    "CapabilityToken",
    "CapabilityVerification",
    "CapabilityBroker",
    "ContainmentEvent",
    "ContainmentReport",
    "ContainmentMonitor",
    "verify_containment",
    "SIDE_EFFECTING",
]

# Tool side-effect classes that require an authority (a user-minted capability
# or a human approval) before an untrusted-tainted argument may flow into them.
# Read-only and pure tools carry no escalation risk and are exempt.
SIDE_EFFECTING: frozenset[str] = frozenset({"write", "external"})

T = TypeVar("T")

_TRUST_RANK: dict[str, int] = {"trusted": 0, "untrusted": 1, "quarantined": 2}


class TrustLabel(StrEnum):
    """A typed information-flow label on a value or context candidate.

    ``trusted`` content (the system prompt, the developer config, the user's
    own request) may instruct the model and authorize capabilities.
    ``untrusted`` content (a retrieved document, a tool result, any external
    byte) is data, never instruction. ``quarantined`` is untrusted content a
    detector additionally flagged as actively hostile (injection or poisoning).

    The three labels form a lattice ordered by restrictiveness
    (``trusted`` < ``untrusted`` < ``quarantined``); :meth:`merge` returns the
    least-trusted of its inputs, so a value derived from any untrusted source is
    tainted — taint propagates monotonically and never decreases.
    """

    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    QUARANTINED = "quarantined"

    @property
    def rank(self) -> int:
        """Lattice rank: higher is less trusted (more restrictive)."""
        return _TRUST_RANK[self.value]

    @property
    def is_tainted(self) -> bool:
        """True for any non-``trusted`` label (i.e. untrusted-derived data)."""
        return self is not TrustLabel.TRUSTED

    @property
    def is_quarantined(self) -> bool:
        """True only for the actively-flagged ``quarantined`` label."""
        return self is TrustLabel.QUARANTINED

    @property
    def may_instruct(self) -> bool:
        """Whether content with this label may carry instructions to the model."""
        return self is TrustLabel.TRUSTED

    def merge(self, other: TrustLabel) -> TrustLabel:
        """Lattice join of two labels: the least-trusted (highest rank).

        Named ``merge`` (not ``join``) so it never shadows ``str.join`` on the
        underlying :class:`~enum.StrEnum`.
        """
        return self if self.rank >= other.rank else other

    @classmethod
    def from_trust_level(cls, level: TrustLevel | str) -> TrustLabel:
        """Derive an information-flow label from a :class:`TrustLevel`.

        System / developer / user provenance maps to ``trusted``; every
        ``untrusted_*`` provenance maps to ``untrusted``. ``quarantined`` is
        never inferred from provenance alone — a detector assigns it explicitly.
        """
        level = TrustLevel(level)
        return cls.TRUSTED if level.allowed_to_instruct_model else cls.UNTRUSTED

    @classmethod
    def combine(cls, labels: Iterable[TrustLabel]) -> TrustLabel:
        """Join an iterable of labels; an empty iterable is ``trusted``."""
        result = cls.TRUSTED
        for label in labels:
            result = result.merge(label)
        return result


@dataclass(frozen=True, slots=True)
class TaintedValue(Generic[T]):
    """A value carried together with its :class:`TrustLabel` and provenance.

    Operations on the wrapped value propagate the label by construction:
    :meth:`map` keeps the label, and :meth:`derive` joins the labels of every
    parent, so a result computed from any untrusted input is itself tainted.
    Use :meth:`unwrap` to read the value back out — an explicit, auditable point
    where code takes responsibility for a (possibly tainted) value.
    """

    value: T
    label: TrustLabel = field(default_factory=lambda: TrustLabel.TRUSTED)
    sources: tuple[str, ...] = ()

    @property
    def is_tainted(self) -> bool:
        """Whether this value carries any untrusted-derived taint."""
        return self.label.is_tainted

    @classmethod
    def trusted(cls, value: T, *, source: str = "user") -> TaintedValue[T]:
        """Wrap a value known to originate from a trusted source."""
        return cls(value=value, label=TrustLabel.TRUSTED, sources=(source,))

    @classmethod
    def untrusted(
        cls, value: T, *, source: str = "external", quarantined: bool = False
    ) -> TaintedValue[T]:
        """Wrap a value that arrived from an untrusted (or flagged) source."""
        label = TrustLabel.QUARANTINED if quarantined else TrustLabel.UNTRUSTED
        return cls(value=value, label=label, sources=(source,))

    def map(self, fn: Callable[[T], Any]) -> TaintedValue[Any]:
        """Apply ``fn`` to the value, carrying the label and sources forward."""
        return TaintedValue(value=fn(self.value), label=self.label, sources=self.sources)

    @classmethod
    def derive(
        cls, value: Any, parents: Iterable[TaintedValue[Any]], *, source: str | None = None
    ) -> TaintedValue[Any]:
        """Build a value from several tainted parents, joining their labels.

        The result is tainted by the least-trusted parent and records the union
        of their provenance sources, so derivation can never launder taint.
        """
        parents = list(parents)
        label = TrustLabel.combine(p.label for p in parents)
        merged: list[str] = []
        for parent in parents:
            for src in parent.sources:
                if src not in merged:
                    merged.append(src)
        if source is not None and source not in merged:
            merged.append(source)
        return cls(value=value, label=label, sources=tuple(merged))

    def unwrap(self) -> T:
        """Return the wrapped value (the explicit taint-acknowledgement point)."""
        return self.value


# ---------------------------------------------------------------------------
# Capability tokens
# ---------------------------------------------------------------------------


class CapabilityToken(BaseModel):
    """An unforgeable, capability-scoped grant minted from the user's request.

    A token authorizes one :attr:`capability` (a tool name or an abstract
    authority) for a specific principal, optionally narrowed by
    :attr:`constraints` (exact-match argument bounds), until :attr:`expires_at`.
    The :attr:`signature` is an HMAC over the canonical payload keyed by the
    minting :class:`CapabilityBroker`'s secret, so a value derived from model
    output or untrusted data cannot manufacture a valid token. The token's
    authority always traces to ``origin="user_request"``.
    """

    id: str = Field(default_factory=lambda: new_id("cap"))
    capability: str
    principal_user: str | None = None
    principal_tenant: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    origin: str = "user_request"
    issued_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    nonce: str = Field(default_factory=lambda: new_id("nonce"))
    signature: str = ""

    def signing_payload(self) -> dict[str, Any]:
        """The canonical, signature-excluded payload the broker signs."""
        return {
            "id": self.id,
            "capability": self.capability,
            "principal_user": self.principal_user,
            "principal_tenant": self.principal_tenant,
            "constraints": self.constraints,
            "origin": self.origin,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "nonce": self.nonce,
        }

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """Whether the token's validity window has elapsed."""
        if self.expires_at is None:
            return False
        return (now or utcnow()) >= self.expires_at


class CapabilityVerification(BaseModel):
    """The explainable result of verifying a :class:`CapabilityToken`."""

    valid: bool
    reason: str = ""
    capability: str = ""


class CapabilityBroker:
    """Mints and verifies :class:`CapabilityToken`\\ s from the user's authority.

    The broker holds a secret key; only it can produce a signature a later
    :meth:`verify` will accept. Mint a capability at the trusted boundary — the
    point where the *user's* request is known — then require it on any
    side-effecting tool call whose arguments were derived from untrusted data.
    Because minting needs the secret, an injected instruction inside a document
    or tool result has no path to a usable capability: the containment property
    rests on key secrecy, not on detecting the attack.
    """

    def __init__(self, secret: str | bytes | None = None, *, default_ttl_s: float = 300.0) -> None:
        if secret is None:
            # A per-process random secret: tokens are unforgeable within the run
            # even when no durable key is configured. Pass a stable secret to
            # mint tokens that verify across processes.
            secret = new_id("secret")
        self._secret = secret.encode("utf-8") if isinstance(secret, str) else secret
        self.default_ttl_s = default_ttl_s

    def _sign(self, token: CapabilityToken) -> str:
        message = stable_hash(token.signing_payload(), length=64).encode("utf-8")
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    def mint(
        self,
        capability: str,
        *,
        principal_user: str | None = None,
        principal_tenant: str | None = None,
        constraints: dict[str, Any] | None = None,
        ttl_s: float | None = None,
        now: datetime | None = None,
    ) -> CapabilityToken:
        """Mint a signed capability for ``capability`` from the user's request.

        ``constraints`` pins argument values the capability is limited to (an
        exact-match allow-list per key), so a token to ``send_email`` can be
        scoped to one recipient. The token expires after ``ttl_s`` seconds.
        """
        issued = now or utcnow()
        ttl = self.default_ttl_s if ttl_s is None else ttl_s
        token = CapabilityToken(
            capability=capability,
            principal_user=principal_user,
            principal_tenant=principal_tenant,
            constraints=dict(constraints or {}),
            issued_at=issued,
            expires_at=issued + timedelta(seconds=ttl) if ttl else None,
        )
        return token.model_copy(update={"signature": self._sign(token)})

    def verify(
        self,
        token: CapabilityToken | None,
        *,
        capability: str,
        principal_user: str | None = None,
        principal_tenant: str | None = None,
        arguments: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> CapabilityVerification:
        """Verify a token authorizes ``capability`` for this call.

        Checks, in order: presence, an authentic signature (HMAC, constant-time
        compared), the validity window, the capability name, the principal (when
        the token pins one), and that ``arguments`` satisfy every pinned
        constraint. Any failure returns ``valid=False`` with a reason; nothing
        about the call can make a forged or mismatched token pass.
        """
        if token is None:
            return CapabilityVerification(valid=False, reason="no capability presented")
        expected = self._sign(token)
        if not token.signature or not hmac.compare_digest(token.signature, expected):
            return CapabilityVerification(
                valid=False, reason="capability signature invalid (forged or wrong key)",
                capability=token.capability,
            )
        if token.is_expired(now=now):
            return CapabilityVerification(
                valid=False, reason="capability expired", capability=token.capability
            )
        if token.capability != capability:
            return CapabilityVerification(
                valid=False,
                reason=f"capability is for {token.capability!r}, not {capability!r}",
                capability=token.capability,
            )
        if token.principal_user is not None and principal_user != token.principal_user:
            return CapabilityVerification(
                valid=False, reason="capability bound to a different user",
                capability=token.capability,
            )
        if token.principal_tenant is not None and principal_tenant != token.principal_tenant:
            return CapabilityVerification(
                valid=False, reason="capability bound to a different tenant",
                capability=token.capability,
            )
        for key, allowed in token.constraints.items():
            actual = (arguments or {}).get(key)
            permitted = allowed if isinstance(allowed, (list, tuple, set)) else (allowed,)
            if actual not in permitted:
                return CapabilityVerification(
                    valid=False,
                    reason=f"argument {key!r}={actual!r} outside capability constraint",
                    capability=token.capability,
                )
        return CapabilityVerification(valid=True, reason="ok", capability=token.capability)


# ---------------------------------------------------------------------------
# Containment invariant
# ---------------------------------------------------------------------------


class ContainmentEvent(BaseModel):
    """One capability-exercise decision recorded for the containment proof.

    Captures the taint of the call's arguments, the side-effect class, the
    :attr:`authority` that permitted it (``capability`` / ``approval`` /
    ``trusted`` / ``none``), and whether it was ``blocked``. An *escalation* is
    an untrusted-tainted side-effecting call that executed on authority
    ``none`` — exactly what containment must make impossible.
    """

    capability: str
    taint: TrustLabel = Field(default_factory=lambda: TrustLabel.TRUSTED)
    side_effects: str = "read"
    authority: str = "none"  # capability | approval | trusted | none
    blocked: bool = False
    detail: str = ""

    @property
    def is_side_effecting(self) -> bool:
        """Whether this call belongs to a side-effecting tool class."""
        return self.side_effects in SIDE_EFFECTING

    @property
    def is_escalation(self) -> bool:
        """An executed, untrusted-tainted side effect with no real authority."""
        return (
            not self.blocked
            and self.is_side_effecting
            and self.taint.is_tainted
            and self.authority not in ("capability", "approval")
        )


class ContainmentReport(BaseModel):
    """The verdict of checking the containment invariant over a run.

    :attr:`held` is true exactly when no :class:`ContainmentEvent` was an
    escalation — i.e. ``untrusted ⇒ no unapproved capability`` held for every
    recorded decision. :attr:`escalation_rate` is escalations over
    untrusted-tainted side-effecting attempts (the adversarial denominator).
    """

    held: bool
    total_events: int = 0
    side_effecting: int = 0
    untrusted_side_effecting: int = 0
    blocked: int = 0
    escalations: list[ContainmentEvent] = Field(default_factory=list)

    @property
    def escalation_rate(self) -> float:
        """Escalations as a fraction of untrusted side-effecting attempts."""
        if self.untrusted_side_effecting == 0:
            return 0.0
        return round(len(self.escalations) / self.untrusted_side_effecting, 6)


def verify_containment(events: Iterable[ContainmentEvent]) -> ContainmentReport:
    """Check ``untrusted ⇒ no unapproved capability`` over recorded events.

    A pure function over a run's :class:`ContainmentEvent` log: it returns a
    :class:`ContainmentReport` whose :attr:`~ContainmentReport.held` is true iff
    no untrusted-tainted side effect executed without a capability or approval.
    Pairs with the formal-verification goal — the invariant is machine-checkable
    from the trace, not asserted by inspection.
    """
    events = list(events)
    side_effecting = [e for e in events if e.is_side_effecting]
    untrusted_se = [e for e in side_effecting if e.taint.is_tainted]
    escalations = [e for e in events if e.is_escalation]
    return ContainmentReport(
        held=not escalations,
        total_events=len(events),
        side_effecting=len(side_effecting),
        untrusted_side_effecting=len(untrusted_se),
        blocked=sum(1 for e in events if e.blocked),
        escalations=escalations,
    )


class ContainmentMonitor:
    """Records capability exercises so containment can be proven after a run.

    The :class:`DualPlaneExecutor` (and any capability-gated call site) appends
    a :class:`ContainmentEvent` per decision; :meth:`report` folds the log
    through :func:`verify_containment`. Hold a single monitor for a run to get a
    whole-run containment verdict, or inspect :attr:`events` directly.
    """

    def __init__(self) -> None:
        self.events: list[ContainmentEvent] = []

    def record(
        self,
        capability: str,
        *,
        taint: TrustLabel | None = None,
        side_effects: str = "read",
        authority: str = "none",
        blocked: bool = False,
        detail: str = "",
    ) -> ContainmentEvent:
        """Append one capability-exercise decision to the log."""
        event = ContainmentEvent(
            capability=capability,
            taint=taint or TrustLabel.TRUSTED,
            side_effects=side_effects,
            authority=authority,
            blocked=blocked,
            detail=detail,
        )
        self.events.append(event)
        return event

    def report(self) -> ContainmentReport:
        """Verify the containment invariant over everything recorded so far."""
        return verify_containment(self.events)

    @property
    def held(self) -> bool:
        """Whether containment has held for every recorded decision so far."""
        return self.report().held
