"""Cross-org reputation gossip & attestation exchange over the A2A fabric.

Attestations are portable, time-aware, and revocable — but an importer still has
to be *handed* the right bundle out of band: it has no way to **discover** who has
attested a counterparty, or to learn that an issuer has since revoked one, without
a hosted registry. This module adds the missing rung: a bounded, **pull-based**
exchange of the existing signed artifacts over the A2A fabric — the discovery
analogue for reputation — so an importer assembles a *current* prior from what its
peers hold, never from a central bulletin board.

* **Pull, never push.** An importer queries a peer for the attestations and
  revocations it holds about a subject (:func:`attestation_a2a_server` /
  :class:`AttestationExchange`), and the peer returns only its **own** signed
  artifacts: the current attestation it can issue from its own
  :class:`~vincio.settlement.book.SettlementBook` records, plus any revocations it
  has signed. Nothing is trusted that does not :meth:`verify` from the bytes alone,
  exactly as a directly-handed bundle is — the peer never pushes, and an importer
  never trusts a peer's word over a signature.
* **Bounded fan-out.** :func:`gather_reputation` visits a **bounded** set of peers,
  each governed through an :class:`~vincio.registry.AgentDirectory`'s allow-list
  (every resolution audited), deduplicates the gathered artifacts by content hash,
  and folds them straight into
  :func:`~vincio.settlement.attestation.combine_attestations` under the *same*
  freshness, revocation, and ``[floor, 1]`` discipline — so gossip changes *where
  the evidence comes from*, never *how it is weighed*.
* **Auditable & offline.** Every peer visited and every artifact fetched lands on
  the hash-chained audit log, and the whole exchange runs byte-for-byte the same
  against deterministic in-process peers as over the live fabric (the same
  :func:`~vincio.a2a.connect_a2a_in_process` the negotiation and choreography
  fabrics use).

:meth:`~vincio.core.app.ContextApp.serve_attestations` exposes an org's book as a
queryable peer; :meth:`~vincio.core.app.ContextApp.gather_reputation` assembles a
current prior from a bounded set of peers. Everything is dependency-free,
deterministic, and offline — never a hosted reputation registry or a push-based
gossip bus.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import SettlementError
from .attestation import (
    AttestationConfig,
    AttestationRevocation,
    PortableReputation,
    ReputationAttestation,
    TrustConfig,
    combine_attestations,
)
from .record import _resolve_verifier

if TYPE_CHECKING:
    from ..a2a.server import A2AServer
    from ..security.audit import ChainSigner

__all__ = [
    "ATTESTATION_EXCHANGE_SKILL_ID",
    "EXCHANGE_PEER_ACTION",
    "EXCHANGE_FETCH_ACTION",
    "ReputationBundle",
    "PeerVisit",
    "GatheredReputation",
    "attestation_a2a_server",
    "AttestationExchange",
    "gather_reputation",
]

# The A2A skill id a queryable attestation peer advertises on its Agent Card.
ATTESTATION_EXCHANGE_SKILL_ID = "attestation-exchange"
# The audit action one peer visit is recorded under on the importer's chain.
EXCHANGE_PEER_ACTION = "reputation_peer"
# The audit action one fetched, verified artifact is recorded under.
EXCHANGE_FETCH_ACTION = "reputation_fetch"

_ENVELOPE_KEY = "vincio_attestation_exchange"


# -- the wire bundle ----------------------------------------------------------


class ReputationBundle(BaseModel):
    """The signed artifacts a peer holds about one subject, its reply to a query.

    A peer answering a gossip query returns a bundle of its **own** signed
    :class:`~vincio.settlement.attestation.ReputationAttestation`\\ s and
    :class:`~vincio.settlement.attestation.AttestationRevocation`\\ s about the
    queried subject — nothing it cannot stand behind with a signature. The bundle is
    transport-neutral: it serializes to a small JSON envelope on the A2A text
    channel (:meth:`to_wire`) and an importer reconstructs and *independently
    verifies* every artifact from the bytes (:meth:`from_wire`), so a peer's reply is
    trusted exactly as far as it verifies — never on the peer's word.
    """

    subject: str
    attestations: list[ReputationAttestation] = Field(default_factory=list)
    revocations: list[AttestationRevocation] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether the peer holds nothing signed about the subject."""
        return not self.attestations and not self.revocations

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for the A2A channel."""
        return {
            "subject": self.subject,
            "attestations": [a.to_wire() for a in self.attestations],
            "revocations": [r.to_wire() for r in self.revocations],
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> ReputationBundle:
        return cls(
            subject=str(data.get("subject", "")),
            attestations=[
                ReputationAttestation.from_wire(a) for a in data.get("attestations", []) or []
            ],
            revocations=[
                AttestationRevocation.from_wire(r) for r in data.get("revocations", []) or []
            ],
        )


def _encode_query(subject: str) -> str:
    return json.dumps({_ENVELOPE_KEY: {"kind": "gather", "subject": subject}})


def _decode_query(text: str) -> str:
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise SettlementError(f"malformed attestation-exchange query: {exc}") from exc
    env = data.get(_ENVELOPE_KEY) if isinstance(data, dict) else None
    if not isinstance(env, dict):
        raise SettlementError("A2A message is not an attestation-exchange query")
    return str(env.get("subject", ""))


def _decode_bundle(text: str) -> ReputationBundle:
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise SettlementError(f"malformed attestation-exchange reply: {exc}") from exc
    env = data.get(_ENVELOPE_KEY) if isinstance(data, dict) else None
    if not isinstance(env, dict):
        raise SettlementError("A2A reply is not an attestation-exchange bundle")
    return ReputationBundle.from_wire(env)


# -- the peer (server) side ---------------------------------------------------


def attestation_a2a_server(
    book: Any,
    *,
    revocations: list[AttestationRevocation] | None = None,
    attestations: list[ReputationAttestation] | None = None,
    config: AttestationConfig | None = None,
    name: str | None = None,
    url: str = "",
    description: str = "",
    tracer: Any | None = None,
    token_validator: Any | None = None,
    audit: Any | None = None,
) -> A2AServer:
    """Expose an org's settlement book as a queryable attestation peer over A2A.

    The returned :class:`~vincio.a2a.A2AServer` answers a gossip query for a subject
    by returning a :class:`ReputationBundle` of this org's **own** signed artifacts:
    the *current* attestation it can issue from its
    :class:`~vincio.settlement.book.SettlementBook` records (:meth:`book.attest`,
    signed as the book's owner) when it has admissible history with the subject, plus
    any ``revocations`` it has signed about that subject. Pass an explicit
    ``attestations`` list to serve a fixed signed snapshot instead of re-issuing from
    the book — only those naming the queried subject are returned. The org governs
    and audits the query on its own chain; an importer reaches it with an
    :class:`AttestationExchange`. **Pull, never push:** the peer only ever answers a
    query, and only with artifacts it signed.

    A subject the book has no admissible history for yields an attestation-free bundle
    (its held revocations, if any) rather than an error — the peer simply holds no
    attestation about it.
    """
    from ..a2a.protocol import AgentCard, AgentSkill
    from ..a2a.server import A2AServer

    owner = getattr(book, "owner", "") or "attestor"
    held_revocations = list(revocations or [])
    held_attestations = attestations
    card = AgentCard(
        name=str(name or owner),
        description=description or "A Vincio attestation peer exposed over A2A.",
        url=url,
        skills=[
            AgentSkill(
                id=ATTESTATION_EXCHANGE_SKILL_ID,
                name="attestation-exchange",
                description=(
                    "Pull-based exchange of signed reputation attestations and "
                    "revocations about a subject."
                ),
                tags=["reputation", "attestation", "gossip"],
            )
        ],
    )

    def _bundle_for(subject: str) -> ReputationBundle:
        out_attestations: list[ReputationAttestation] = []
        if held_attestations is not None:
            out_attestations = [a for a in held_attestations if a.subject == subject]
        else:
            try:
                out_attestations = [book.attest(subject, config=config, sign=True)]
            except SettlementError:
                out_attestations = []  # no admissible history — the peer holds nothing
        out_revocations = [
            r for r in held_revocations if r.subject == subject and r.issuer == owner
        ]
        return ReputationBundle(
            subject=subject, attestations=out_attestations, revocations=out_revocations
        )

    async def executor(text: str, task: Any) -> dict[str, Any]:
        subject = _decode_query(text)
        bundle = _bundle_for(subject)
        return {
            "state": "completed",
            "output": json.dumps({_ENVELOPE_KEY: bundle.to_wire()}),
        }

    return A2AServer(
        card, executor, tracer=tracer, token_validator=token_validator, audit=audit
    )


# -- the importer (client) side ----------------------------------------------


class AttestationExchange:
    """A peer reached over A2A that an importer pulls signed artifacts from.

    Wraps an :class:`~vincio.a2a.A2AClient`: :meth:`fetch` sends a typed gossip query
    for a subject over A2A and parses the peer's :class:`ReputationBundle` reply, so
    :func:`gather_reputation` pulls a cross-org peer exactly as it would an in-process
    one. The exchange itself does **not** trust the bundle — it returns the artifacts
    verbatim, and the gather verifies each from the bytes before counting it.
    """

    def __init__(self, client: Any, *, peer_id: str = "") -> None:
        self.client = client
        self.peer_id = peer_id

    async def fetch(self, subject: str) -> ReputationBundle:
        """Query the peer for its signed artifacts about ``subject``."""
        task = await self.client.send(_encode_query(subject))
        if task.status.state in ("submitted", "working"):
            task = await self.client.poll_task(task.id)
        if task.status.state != "completed":
            raise SettlementError(
                f"attestation peer {self.peer_id or '?'} ended in {task.status.state}",
                details={"peer_id": self.peer_id, "state": task.status.state},
            )
        return _decode_bundle(_task_output(task))


class PeerVisit(BaseModel):
    """One peer's contribution to a gather — visited, governed, and counted.

    Every peer the gather considers becomes a visit, whether or not it contributed,
    so the :class:`GatheredReputation` carries a complete, auditable picture of where
    the evidence came from. ``allowed`` records the directory's governance verdict,
    ``reachable`` whether the peer answered, and ``attestations`` / ``revocations``
    how many *fresh* (deduplicated) artifacts it added; a skipped or unreachable peer
    carries a ``reason``.
    """

    peer: str
    allowed: bool = True
    reachable: bool = True
    attestations: int = 0
    revocations: int = 0
    duplicates: int = 0
    reason: str | None = None

    @property
    def contributed(self) -> bool:
        """Whether the peer added at least one fresh artifact."""
        return self.attestations > 0 or self.revocations > 0


class GatheredReputation:
    """A current prior assembled by pulling signed artifacts from a set of peers.

    Produced by :func:`gather_reputation` (or
    :meth:`~vincio.core.app.ContextApp.gather_reputation`). It carries the
    deduplicated artifacts gathered across the visited peers, the per-peer
    :class:`PeerVisit` record of where they came from, and the
    :class:`~vincio.settlement.attestation.PortableReputation` they fold into — so the
    same ``weight(member_id)`` that drops into the negotiation / discovery path is
    available here, now sourced from gossip rather than a handed bundle. Gossip
    changes *where the evidence comes from*; the combination weighs it exactly as
    :func:`~vincio.settlement.attestation.combine_attestations` always has.
    """

    def __init__(
        self,
        subject: str,
        visits: list[PeerVisit],
        attestations: list[ReputationAttestation],
        revocations: list[AttestationRevocation],
        reputation: PortableReputation,
        *,
        duplicates: int = 0,
    ) -> None:
        self.subject = subject
        self.visits = visits
        self.attestations = attestations
        self.revocations = revocations
        self.reputation = reputation
        self.duplicates = duplicates

    # -- reads --------------------------------------------------------------

    @property
    def peers_visited(self) -> int:
        """How many peers the gather considered (allowed or not)."""
        return len(self.visits)

    @property
    def peers_reachable(self) -> int:
        """How many visited peers answered the query."""
        return sum(1 for v in self.visits if v.reachable and v.allowed)

    @property
    def peers_contributing(self) -> list[str]:
        """The peers that added at least one fresh artifact, sorted."""
        return sorted(v.peer for v in self.visits if v.contributed)

    @property
    def attestations_gathered(self) -> int:
        """The number of distinct attestations folded into the prior."""
        return len(self.attestations)

    @property
    def revocations_gathered(self) -> int:
        """The number of distinct revocations folded into the prior."""
        return len(self.revocations)

    def weight(self, member_id: str) -> float:
        """The aggregation weight for ``member_id`` — the negotiation drop-in.

        Delegates to the assembled
        :class:`~vincio.settlement.attestation.PortableReputation`, so a
        :class:`GatheredReputation` weights an offer exactly where a handed prior or a
        local ledger would.
        """
        return self.reputation.weight(member_id)

    def standing(self, member_id: str) -> Any | None:
        """The pooled standing for ``member_id`` in the assembled prior, or ``None``."""
        return self.reputation.standing(member_id)

    def visit_for(self, peer: str) -> PeerVisit | None:
        """The visit record for ``peer``, or ``None`` if it was not considered."""
        return next((v for v in self.visits if v.peer == peer), None)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print where the evidence came from and the resulting standing."""
        print(
            f"Gathered reputation for {self.subject!r}: "
            f"{self.attestations_gathered} attestation(s) / "
            f"{self.revocations_gathered} revocation(s) from "
            f"{self.peers_reachable}/{self.peers_visited} peer(s) "
            f"({self.duplicates} duplicate(s) deduped)"
        )
        standing = self.reputation.standing(self.subject)
        if standing is not None:
            print(
                f"  {self.subject}: reputation={standing.reputation:.3f} "
                f"weight={standing.weight:.3f} (issuers={standing.issuers})"
            )
        for visit in self.visits:
            if not visit.contributed:
                print(f"  ! {visit.peer}: {visit.reason or 'no fresh artifacts'}")


def _as_exchange(conn: Any, *, peer_id: str) -> AttestationExchange:
    """Coerce a connection into an :class:`AttestationExchange`.

    Accepts an already-built :class:`AttestationExchange`, an in-process
    :class:`~vincio.a2a.A2AServer` (wrapped with
    :func:`~vincio.a2a.connect_a2a_in_process`), or an
    :class:`~vincio.a2a.A2AClient` — so the same gather drives in-process and remote
    peers identically.
    """
    if isinstance(conn, AttestationExchange):
        if not conn.peer_id:
            conn.peer_id = peer_id
        return conn
    from ..a2a import connect_a2a_in_process
    from ..a2a.server import A2AServer

    if isinstance(conn, A2AServer):
        return AttestationExchange(connect_a2a_in_process(conn), peer_id=peer_id)
    # Anything with the A2A client surface (send / poll_task) is used directly.
    if hasattr(conn, "send") and hasattr(conn, "poll_task"):
        return AttestationExchange(conn, peer_id=peer_id)
    raise SettlementError(
        f"peer {peer_id!r} is not an AttestationExchange, A2AServer, or A2AClient",
        details={"peer_id": peer_id, "type": type(conn).__name__},
    )


def _verifies(artifact: Any, verifier: ChainSigner | None) -> bool:
    """Whether a fetched artifact verifies from the bytes — the trust gate.

    Its hash must recompute, and — with a ``verifier`` and a present signature —
    the issuer's signature must check; an artifact that fails either is refused
    rather than trusted on the peer's word. The downstream
    :func:`~vincio.settlement.attestation.combine_attestations` re-checks the same
    invariants, so this is the gather's first, not its only, line of defense.
    """
    check = artifact.verify(verifier, require=[])
    if not check.hash_ok:
        return False
    if verifier is not None and artifact.signatures and not check.signatures_ok:
        return False
    return True


def _peer_items(peers: Any) -> list[tuple[str, Any]]:
    """Normalize the ``peers`` argument into a deterministic ``(id, conn)`` list."""
    if isinstance(peers, dict):
        return sorted(peers.items(), key=lambda kv: kv[0])
    items: list[tuple[str, Any]] = []
    for entry in peers or []:
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            items.append((str(entry[0]), entry[1]))
        else:
            peer_id = getattr(entry, "peer_id", "") or getattr(entry, "org_id", "")
            items.append((str(peer_id), entry))
    return sorted(items, key=lambda kv: kv[0])


async def gather_reputation(
    subject: str,
    *,
    peers: Any,
    directory: Any | None = None,
    principal: Any | None = None,
    config: AttestationConfig | None = None,
    verifier: ChainSigner | None = None,
    base: Any | None = None,
    allow_self: bool = False,
    held_attestations: list[ReputationAttestation] | None = None,
    held_revocations: list[AttestationRevocation] | None = None,
    as_of: Any | None = None,
    trust: Any | None = None,
    trust_config: TrustConfig | None = None,
    max_peers: int | None = None,
    audit: Any | None = None,
    record_audit: bool = True,
    verify_with: ChainSigner | None = None,
) -> GatheredReputation:
    """Pull signed attestations and revocations from a bounded set of peers.

    Visits the peers in deterministic order, **governing** each through ``directory``
    (an :class:`~vincio.registry.AgentDirectory`; a peer the allow-list denies is
    skipped and pinpointed, its resolution audited) and **bounding** the fan-out to
    ``max_peers`` allowed peers. From each reachable peer it fetches a
    :class:`ReputationBundle`, **independently verifies** every artifact from the
    bytes (the hash recomputes, an attestation's reputation re-derives, and — with
    ``verifier`` — the issuer's signature checks), **deduplicates** by content hash
    across peers, and records the visit. The gathered artifacts (plus any
    ``held_attestations`` / ``held_revocations`` the importer already has) fold
    straight into :func:`~vincio.settlement.attestation.combine_attestations` under
    the same ``config`` prior, ``revocations``, ``as_of`` freshness, and ``[floor, 1]``
    discipline — so gossip changes only *where the evidence comes from*.

    Pass a ``trust`` source or a ``trust_config`` to weigh each gathered issuer's
    evidence by the importer's **own trust in that issuer** (rooted in ``base``,
    composed transitively over the gathered attestations) — so a cluster of unknown
    peers gossiping the same way cannot out-evidence a few the importer trusts. With
    neither, every reachable peer's evidence pools with equal pull, as before.

    ``peers`` maps a peer id to a connection (an :class:`AttestationExchange`, an
    in-process :class:`~vincio.a2a.A2AServer`, or an
    :class:`~vincio.a2a.A2AClient`), or is an iterable of ``(id, connection)`` pairs.
    Every peer visited and every artifact fetched lands on ``audit`` (when
    ``record_audit``). Returns a :class:`GatheredReputation` exposing
    ``weight(member_id)`` for the negotiation path. ``verify_with`` is a deprecated
    alias for ``verifier`` (since 7.5, removed in 8.0).
    """
    verifier = _resolve_verifier(verifier, verify_with, "gather_reputation")
    cfg = (config or AttestationConfig()).validate_coherent()
    items = _peer_items(peers)

    visits: list[PeerVisit] = []
    seen_attestations: dict[str, ReputationAttestation] = {}
    seen_revocations: dict[str, AttestationRevocation] = {}
    duplicates = 0
    allowed_visited = 0

    def _record(visit: PeerVisit) -> None:
        visits.append(visit)
        _audit_peer(audit, record_audit, subject, visit)

    for peer_id, conn in items:
        if max_peers is not None and allowed_visited >= max_peers:
            _record(
                PeerVisit(peer=peer_id, allowed=True, reachable=False, reason="fan-out bound reached")
            )
            continue

        # Governance: a directory's allow-list decides (and audits) reachability.
        if directory is not None:
            resolution = directory.try_resolve(peer_id, principal=principal)
            if not resolution.allowed:
                _record(
                    PeerVisit(
                        peer=peer_id,
                        allowed=False,
                        reachable=False,
                        reason=resolution.decision.reason or "not allow-listed",
                    )
                )
                continue
        allowed_visited += 1

        try:
            exchange = _as_exchange(conn, peer_id=peer_id)
            bundle = await exchange.fetch(subject)
        except Exception as exc:  # noqa: BLE001 - a dead peer is skipped, not fatal
            _record(PeerVisit(peer=peer_id, allowed=True, reachable=False, reason=str(exc)))
            continue

        fresh_atts = 0
        fresh_revs = 0
        dupes = 0
        for att in bundle.attestations:
            # Verify from the bytes before counting — never on the peer's word.
            if not _verifies(att, verifier):
                continue
            key = att.content_hash
            if not key or key in seen_attestations:
                dupes += 1
                continue
            seen_attestations[key] = att
            fresh_atts += 1
            _audit_fetch(audit, record_audit, subject, peer_id, "attestation", att)
        for rev in bundle.revocations:
            if not _verifies(rev, verifier):
                continue
            key = rev.content_hash
            if not key or key in seen_revocations:
                dupes += 1
                continue
            seen_revocations[key] = rev
            fresh_revs += 1
            _audit_fetch(audit, record_audit, subject, peer_id, "revocation", rev)

        duplicates += dupes
        _record(
            PeerVisit(
                peer=peer_id,
                allowed=True,
                reachable=True,
                attestations=fresh_atts,
                revocations=fresh_revs,
                duplicates=dupes,
                reason=None if (fresh_atts or fresh_revs) else "no fresh artifacts",
            )
        )

    # Fold in any artifacts the importer already holds out of band.
    for att in held_attestations or []:
        if att.content_hash and att.content_hash not in seen_attestations:
            seen_attestations[att.content_hash] = att
    for rev in held_revocations or []:
        if rev.content_hash and rev.content_hash not in seen_revocations:
            seen_revocations[rev.content_hash] = rev

    attestations = list(seen_attestations.values())
    revocations = list(seen_revocations.values())
    reputation = combine_attestations(
        attestations,
        subject=subject,
        config=cfg,
        verifier=verifier,
        base=base,
        allow_self=allow_self,
        revocations=revocations,
        as_of=as_of,
        trust=trust,
        trust_config=trust_config,
    )
    return GatheredReputation(
        subject, visits, attestations, revocations, reputation, duplicates=duplicates
    )


def _audit_peer(audit: Any, record_audit: bool, subject: str, visit: PeerVisit) -> None:
    if not record_audit or audit is None:
        return
    audit.record(
        EXCHANGE_PEER_ACTION,
        resource=subject,
        decision="allow" if visit.allowed and visit.reachable else "skip",
        details={
            "peer": visit.peer,
            "attestations": visit.attestations,
            "revocations": visit.revocations,
            "duplicates": visit.duplicates,
            "reason": visit.reason,
        },
    )


def _audit_fetch(
    audit: Any, record_audit: bool, subject: str, peer: str, kind: str, artifact: Any
) -> None:
    if not record_audit or audit is None:
        return
    audit.record(
        EXCHANGE_FETCH_ACTION,
        resource=subject,
        decision="fetched",
        details={
            "peer": peer,
            "kind": kind,
            "issuer": getattr(artifact, "issuer", ""),
            "content_hash": getattr(artifact, "content_hash", ""),
        },
    )


def _task_output(task: Any) -> str:
    for artifact in getattr(task, "artifacts", []) or []:
        text = "\n".join(p.text for p in artifact.parts if getattr(p, "kind", "text") == "text")
        if text:
            return text
    message = getattr(task.status, "message", None)
    if message is not None:
        return message.text
    return ""
