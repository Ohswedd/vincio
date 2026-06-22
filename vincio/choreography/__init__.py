"""Durable, compensating cross-org workflow choreography over the A2A fabric.

With agents that discover, negotiate, and contract across organizations, the next
reach is the **durable work** they coordinate: a long-running, compensating
workflow that spans more than one organization's agent fabric. This package adds
that rung — the choreography analogue of the in-process durable graph, now
crossing trust boundaries.

* A :class:`Saga` defines an ordered, compensating cross-org workflow: each
  :class:`SagaStep` dispatches to a named participant org with an optional undo.
* A :class:`Choreography` drives it — coordinator-driven dispatch with **per-org
  self-governance**: the coordinator sends a typed
  :class:`StepRequest` under a negotiated
  :class:`~vincio.negotiation.Contract` and audits the handoff on its own chain;
  each :class:`Participant` runs and audits the step on its own chain. There is no
  shared control plane, only the typed contract and the audited handoffs.
* The :class:`SagaJournal` is checkpointed to the metadata store after every step,
  so a saga **survives a restart** — a fresh process resumes it by ``saga_id`` and
  never re-runs a completed step — and is **hash-chained**, so
  :meth:`SagaJournal.verify` recomputes it offline and catches any tampered record.
* A forward step that fails — the participant returns ``ok=False``, raises, or
  **breaches its contract** — triggers deterministic compensation of the completed
  steps in reverse order, so a half-completed cross-org transaction unwinds
  cleanly.

A choreography runs fully offline against in-process :class:`LocalParticipant` s,
or over the A2A agent fabric against a :class:`RemoteParticipant` (see
:mod:`vincio.choreography.fabric`). Every step lands on a hash-chained audit log —
the coordinator's and each participant's — so a cross-org transaction is a
mechanical, verifiable artifact, never a hosted control plane.
"""

from __future__ import annotations

from .engine import Choreography, LocalParticipant, Participant
from .fabric import RemoteParticipant, choreography_a2a_server
from .saga import (
    JournalVerification,
    Saga,
    SagaContext,
    SagaJournal,
    SagaResult,
    SagaStep,
    StepOutcome,
    StepRecord,
    StepRequest,
)

__all__ = [
    # saga definition & durable state
    "Saga",
    "SagaStep",
    "SagaContext",
    "SagaJournal",
    "JournalVerification",
    "SagaResult",
    "StepRequest",
    "StepOutcome",
    "StepRecord",
    # engine & participants
    "Choreography",
    "Participant",
    "LocalParticipant",
    # A2A fabric binding
    "RemoteParticipant",
    "choreography_a2a_server",
]
