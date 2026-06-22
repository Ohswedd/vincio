"""Dispatch a choreography step to a remote org over the A2A agent fabric.

The :class:`~vincio.choreography.engine.Choreography` engine drives a saga by
calling a :class:`~vincio.choreography.engine.Participant`'s ``perform`` /
``compensate``. This module lets a participant live in **another organization,
reached over A2A** — so a cross-org saga dispatches its steps over the same
governed, audited fabric that already carries A2A delegation and negotiation, with
no new transport.

* :func:`choreography_a2a_server` exposes a local org's capability handlers as an
  :class:`~vincio.a2a.A2AServer`. Each inbound ``message/send`` carries a typed
  step envelope; the server runs the named handler and returns its outcome. The
  org governs and audits its own execution on its own chain — its self-governance.
  The Agent Card advertises a ``choreograph`` skill.
* :class:`RemoteParticipant` is the client side: a
  :class:`~vincio.choreography.engine.Participant` backed by an
  :class:`~vincio.a2a.A2AClient`. Its ``perform`` / ``compensate`` serialize the
  step request, send it over A2A, and parse the remote org's outcome — so the
  coordinator drives a cross-org participant exactly as it would a local one.

Step requests and outcomes travel as a small JSON envelope on the A2A text
channel, so nothing about the existing A2A server/transport changes. The whole
path is offline-testable in-process via
:func:`~vincio.a2a.connect_a2a_in_process`.
"""

from __future__ import annotations

import json
from typing import Any

from ..a2a.protocol import AgentCard, AgentSkill
from ..a2a.server import A2AServer
from ..core.errors import ChoreographyError
from .engine import LocalParticipant
from .saga import StepOutcome, StepRequest

__all__ = ["CHOREOGRAPHY_SKILL_ID", "choreography_a2a_server", "RemoteParticipant"]

CHOREOGRAPHY_SKILL_ID = "choreograph"


def _encode_request(request: StepRequest) -> str:
    return json.dumps({"vincio_choreography": request.to_wire()})


def _decode_request(text: str) -> StepRequest:
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ChoreographyError(f"malformed choreography envelope: {exc}") from exc
    env = data.get("vincio_choreography") if isinstance(data, dict) else None
    if not isinstance(env, dict):
        raise ChoreographyError("A2A message is not a choreography envelope")
    return StepRequest.from_wire(env)


def _decode_outcome(text: str) -> StepOutcome:
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ChoreographyError(f"malformed choreography reply: {exc}") from exc
    env = data.get("vincio_choreography") if isinstance(data, dict) else None
    if not isinstance(env, dict):
        raise ChoreographyError("A2A reply is not a choreography outcome")
    return StepOutcome.from_wire(env)


def choreography_a2a_server(
    handlers: dict[str, Any] | LocalParticipant,
    *,
    org_id: str = "participant",
    name: str | None = None,
    url: str = "",
    description: str = "",
    tracer: Any | None = None,
    token_validator: Any | None = None,
    audit: Any | None = None,
) -> A2AServer:
    """Expose a local org's choreography handlers over A2A.

    ``handlers`` is a mapping of action name to a callable (wrapped in a
    :class:`~vincio.choreography.engine.LocalParticipant`) or an already-built
    participant. The returned :class:`~vincio.a2a.A2AServer` answers a typed step
    envelope by running the named handler and returning its
    :class:`~vincio.choreography.saga.StepOutcome`, recording the step on this org's
    own audit chain — so a remote coordinator dispatches to this org over the
    fabric, governed and audited locally.
    """
    participant = (
        handlers
        if isinstance(handlers, LocalParticipant)
        else LocalParticipant(org_id, handlers, audit=audit)
    )
    card = AgentCard(
        name=str(name or participant.org_id),
        description=description or "A Vincio choreography participant exposed over A2A.",
        url=url,
        skills=[
            AgentSkill(
                id=CHOREOGRAPHY_SKILL_ID,
                name="choreograph",
                description="Perform or compensate a typed cross-org saga step.",
                tags=["choreography", "saga", "contracting"],
            )
        ],
    )

    async def executor(text: str, task: Any) -> dict[str, Any]:
        request = _decode_request(text)
        if request.kind == "compensation":
            outcome = await participant.compensate(request)
        else:
            outcome = await participant.perform(request)
        return {
            "state": "completed",
            "output": json.dumps({"vincio_choreography": outcome.to_wire()}),
        }

    return A2AServer(
        card, executor, tracer=tracer, token_validator=token_validator, audit=audit
    )


class RemoteParticipant:
    """A choreography :class:`Participant` whose steps run in a remote A2A org.

    Wraps an :class:`~vincio.a2a.A2AClient`: each ``perform`` / ``compensate`` sends
    the typed step request over A2A and parses the remote org's outcome, so the
    local :class:`~vincio.choreography.engine.Choreography` drives a cross-org
    participant exactly as it would a local
    :class:`~vincio.choreography.engine.LocalParticipant`.
    """

    def __init__(self, client: Any, *, org_id: str) -> None:
        self.client = client
        self.org_id = org_id

    async def _exchange(self, request: StepRequest) -> StepOutcome:
        # Pin the request's saga identity; the remote answers for this org only.
        task = await self.client.send(_encode_request(request))
        if task.status.state in ("submitted", "working"):
            task = await self.client.poll_task(task.id)
        if task.status.state != "completed":
            raise ChoreographyError(
                f"remote participant {self.org_id} ended in {task.status.state}",
                details={"org_id": self.org_id, "state": task.status.state},
            )
        return _decode_outcome(_task_output(task))

    async def perform(self, request: StepRequest) -> StepOutcome:
        return await self._exchange(request)

    async def compensate(self, request: StepRequest) -> StepOutcome:
        return await self._exchange(request)


def _task_output(task: Any) -> str:
    for artifact in getattr(task, "artifacts", []) or []:
        text = "\n".join(p.text for p in artifact.parts if getattr(p, "kind", "text") == "text")
        if text:
            return text
    message = getattr(task.status, "message", None)
    if message is not None:
        return message.text
    return ""
