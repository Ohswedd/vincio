"""Run a bounded negotiation over the A2A agent fabric.

The :class:`~vincio.negotiation.engine.Negotiation` engine alternates typed offers
between two :class:`~vincio.negotiation.engine.Party` objects. This module lets one
of those parties live in **another organization, reached over A2A** — so a
multi-org crew negotiates a contract over the same governed, audited fabric that
already carries A2A delegation, with no new transport.

* :func:`negotiation_a2a_server` exposes a local
  :class:`~vincio.negotiation.engine.Party` as an :class:`~vincio.a2a.A2AServer`.
  Each inbound ``message/send`` carries a typed offer envelope; the server runs the
  local party's strategy and returns its counter / acceptance / walk-away. The
  party's Agent Card advertises a ``negotiate`` skill.
* :class:`A2ANegotiator` is the client side: a
  :class:`~vincio.negotiation.engine.Party` backed by an
  :class:`~vincio.a2a.A2AClient`. Its ``open`` / ``respond`` serialize the current
  offer, send it over A2A, and parse the remote party's reply — so the local engine
  drives a remote counterparty exactly as it would a local one.

Offers travel as a small JSON envelope on the A2A text channel, so nothing about
the existing A2A server/transport changes. The whole path is offline-testable
in-process via :func:`~vincio.a2a.connect_a2a_in_process`.
"""

from __future__ import annotations

import json
from typing import Any

from ..a2a.protocol import AgentCard, AgentSkill
from ..a2a.server import A2AServer
from ..core.errors import NegotiationError
from .engine import NegotiationBudget, Offer, Party, Role

__all__ = ["NEGOTIATION_SKILL_ID", "negotiation_a2a_server", "A2ANegotiator"]

NEGOTIATION_SKILL_ID = "negotiate"


def _encode_envelope(
    *, scope: str, offer: Offer | None, round_index: int, budget: NegotiationBudget, kind: str
) -> str:
    return json.dumps(
        {
            "vincio_negotiation": {
                "kind": kind,
                "scope": scope,
                "round_index": round_index,
                "max_rounds": budget.max_rounds,
                "offer": offer.to_wire() if offer is not None else None,
            }
        }
    )


def _decode_envelope(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise NegotiationError(f"malformed negotiation envelope: {exc}") from exc
    env = data.get("vincio_negotiation") if isinstance(data, dict) else None
    if not isinstance(env, dict):
        raise NegotiationError("A2A message is not a negotiation envelope")
    return env


def negotiation_a2a_server(
    party: Party,
    *,
    name: str | None = None,
    url: str = "",
    description: str = "",
    tracer: Any | None = None,
    token_validator: Any | None = None,
    audit: Any | None = None,
) -> A2AServer:
    """Expose a local negotiating :class:`Party` over A2A.

    The returned :class:`~vincio.a2a.A2AServer` answers a typed offer envelope by
    running ``party.open`` (round 0) or ``party.respond`` (later rounds) and
    returning the party's reply offer — so a remote engine can bargain against this
    party over the fabric, bounded and audited like any other A2A task.
    """
    card = AgentCard(
        name=str(name or getattr(party, "member_id", "negotiator")),
        description=description or "A Vincio negotiating agent exposed over A2A.",
        url=url,
        skills=[
            AgentSkill(
                id=NEGOTIATION_SKILL_ID,
                name="negotiate",
                description="Bounded offer/counter negotiation over typed price/SLA/scope terms.",
                tags=["negotiation", "contracting"],
            )
        ],
    )

    async def executor(text: str, task: Any) -> dict[str, Any]:
        env = _decode_envelope(text)
        scope = str(env.get("scope", ""))
        round_index = int(env.get("round_index", 0))
        budget = NegotiationBudget(max_rounds=int(env.get("max_rounds", 8)))
        offer_wire = env.get("offer")
        if env.get("kind") == "open" or offer_wire is None:
            reply = await party.open(scope, budget)
        else:
            incoming = Offer.from_wire(offer_wire)
            reply = await party.respond(scope, incoming, round_index, budget)
        return {"state": "completed", "output": json.dumps({"vincio_negotiation": reply.to_wire()})}

    return A2AServer(card, executor, tracer=tracer, token_validator=token_validator, audit=audit)


class A2ANegotiator:
    """A negotiating :class:`Party` whose moves are made by a remote A2A agent.

    Wraps an :class:`~vincio.a2a.A2AClient`: each ``open`` / ``respond`` sends the
    current typed offer over A2A and parses the remote party's reply, so the local
    :class:`~vincio.negotiation.engine.Negotiation` drives a cross-org counterparty
    exactly as it would a local :class:`~vincio.negotiation.engine.LocalParty`.
    """

    def __init__(self, client: Any, *, member_id: str, role: Role = "seller") -> None:
        self.client = client
        self.member_id = member_id
        self.role: Role = role

    async def _exchange(
        self, *, scope: str, offer: Offer | None, round_index: int, budget: NegotiationBudget, kind: str
    ) -> Offer:
        text = _encode_envelope(
            scope=scope, offer=offer, round_index=round_index, budget=budget, kind=kind
        )
        task = await self.client.send(text)
        if task.status.state in ("submitted", "working"):
            task = await self.client.poll_task(task.id)
        if task.status.state != "completed":
            raise NegotiationError(
                f"remote negotiator {self.member_id} ended in {task.status.state}",
                details={"member_id": self.member_id, "state": task.status.state},
            )
        env = _decode_envelope(_task_output(task))
        reply = Offer.from_wire(env)
        # Trust the remote's identity to the directory-resolved member id, not a
        # self-asserted one on the wire, so reputation lookups cannot be spoofed.
        reply.party = self.member_id
        reply.role = self.role
        return reply

    async def open(self, scope: str, budget: NegotiationBudget) -> Offer:
        return await self._exchange(
            scope=scope, offer=None, round_index=0, budget=budget, kind="open"
        )

    async def respond(
        self, scope: str, incoming: Offer, round_index: int, budget: NegotiationBudget
    ) -> Offer:
        return await self._exchange(
            scope=scope, offer=incoming, round_index=round_index, budget=budget, kind="respond"
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
