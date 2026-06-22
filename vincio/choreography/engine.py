"""The cross-org choreography engine: a durable, compensating saga orchestrator.

A :class:`Choreography` drives a :class:`~vincio.choreography.saga.Saga` across
more than one organization's agent fabric. It is coordinator-driven dispatch with
**per-org self-governance**: the coordinator only sends a typed
:class:`~vincio.choreography.saga.StepRequest` under a negotiated contract and
audits the handoff on its *own* chain; each participant runs and audits the step
on *its* chain. There is no shared control plane — only the typed contract and the
audited handoffs cross a trust boundary.

The engine gives the three guarantees the in-process durable graph gives within a
single boundary, now spanning several:

* **Durable & resumable.** The :class:`~vincio.choreography.saga.SagaJournal` is
  checkpointed to the metadata store after every step. A fresh process reloads it
  by ``saga_id`` and continues from the cursor — completed steps are never re-run,
  so a saga survives a restart the way a durable graph does.
* **Compensating saga.** A forward step that fails — the participant returns
  ``ok=False``, raises, or breaches its contract — triggers deterministic
  compensation of the already-completed steps in reverse order, so a
  half-completed cross-org transaction unwinds cleanly.
* **Bounded.** A failure is terminal (it compensates, it does not loop), and
  ``interrupt_after`` cooperatively pauses a long saga into a resumable state.

A :class:`Participant` is the binding to one org: :class:`LocalParticipant` runs an
org's capabilities in-process (offline tests, same-org steps), and
:class:`~vincio.choreography.fabric.RemoteParticipant` reaches a remote org over
the A2A fabric. The engine drives both through the same protocol.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from ..core.errors import ChoreographyError, CompensationError
from .discovery import BIND_ACTION, CapabilityBinder, StepBinding
from .saga import (
    SAGA_STORE_KIND,
    STEP_DISPATCH_ACTION,
    Saga,
    SagaContext,
    SagaJournal,
    SagaResult,
    SagaStep,
    StepOutcome,
    StepRecord,
    StepRequest,
    StepStatus,
)

__all__ = ["Participant", "LocalParticipant", "Choreography"]

HandlerFn = Callable[..., Any]


@runtime_checkable
class Participant(Protocol):
    """One organization in a choreography: it performs steps and compensates them.

    The two methods are async so a remote A2A counterparty satisfies the same
    contract as a local, in-process one. ``perform`` runs a forward action;
    ``compensate`` runs the undo named by the saga step. Each returns a
    :class:`~vincio.choreography.saga.StepOutcome`.
    """

    org_id: str

    async def perform(self, request: StepRequest) -> StepOutcome: ...

    async def compensate(self, request: StepRequest) -> StepOutcome: ...


class LocalParticipant:
    """An in-process participant: a registry of an org's named capabilities.

    ``handlers`` maps an action name to a callable invoked with the step's payload
    dict. A handler may return a plain dict (its output), a
    :class:`~vincio.choreography.saga.StepOutcome` (to also declare delivered
    cost / latency / quality for contract enforcement), or ``None``; raising marks
    the step failed. When an ``audit`` log is supplied, every step the org runs is
    recorded on the org's *own* hash-chained chain — its self-governance.
    """

    def __init__(
        self,
        org_id: str,
        handlers: dict[str, HandlerFn],
        *,
        audit: Any | None = None,
        name: str | None = None,
    ) -> None:
        self.org_id = org_id
        self.handlers = dict(handlers)
        self.audit = audit
        self.name = name or org_id

    async def _run(self, request: StepRequest) -> StepOutcome:
        handler = self.handlers.get(request.action)
        if handler is None:
            outcome = StepOutcome(
                ok=False, error=f"org {self.org_id!r} has no action {request.action!r}"
            )
        else:
            try:
                result = await _call(handler, request.payload)
                outcome = _coerce_outcome(result)
            except Exception as exc:  # noqa: BLE001 - rendered as a failed outcome
                outcome = StepOutcome(ok=False, error=f"{type(exc).__name__}: {exc}")
        if self.audit is not None:
            self.audit.record(
                STEP_DISPATCH_ACTION,
                resource=request.saga_id,
                decision=("ok" if outcome.ok else "error"),
                details={
                    "step": request.step,
                    "action": request.action,
                    "kind": request.kind,
                    "contract_id": request.contract_id,
                    "ok": outcome.ok,
                    "error": outcome.error,
                },
            )
        return outcome

    async def perform(self, request: StepRequest) -> StepOutcome:
        return await self._run(request)

    async def compensate(self, request: StepRequest) -> StepOutcome:
        return await self._run(request)


def _coerce_outcome(result: Any) -> StepOutcome:
    if isinstance(result, StepOutcome):
        return result
    if result is None:
        return StepOutcome(ok=True, output={})
    if isinstance(result, dict):
        return StepOutcome(ok=True, output=result)
    return StepOutcome(ok=True, output={"result": result})


async def _call(fn: HandlerFn, payload: dict[str, Any]) -> Any:
    if inspect.iscoroutinefunction(fn):
        return await fn(payload)
    out = fn(payload)
    if inspect.isawaitable(out):
        return await out
    return out


class Choreography:
    """Drives a :class:`~vincio.choreography.saga.Saga` across organizations.

    The engine is dumb, deterministic orchestration: it dispatches each step's
    typed request to the named :class:`Participant`, enforces the step's contract
    on the delivered outcome, checkpoints the durable journal after every move, and
    on a failure compensates the completed steps in reverse order. Termination is
    guaranteed (a failure compensates, it never loops); a restart resumes from the
    journal by ``saga_id``.

    ``participants`` maps an org id to a :class:`Participant` — or, as a
    convenience, to a ``dict`` of handler callables, which is wrapped in a
    :class:`LocalParticipant`. ``store`` is the metadata store the journal is
    checkpointed to (durability); ``audit`` / ``events`` record the coordinator's
    handoffs and completion; ``signer`` signs each journal record for third-party
    verifiability.

    ``binder`` enables **run-time discovery**: a step that declares a ``capability``
    instead of a fixed ``participant`` is resolved against the binder's governed
    :class:`~vincio.choreography.discovery.CapabilityBinder` at dispatch time, which
    ranks the allowed candidates by reputation and prior settlement fit and picks
    the best. The chosen org must be present in ``participants`` (that is *how* the
    coordinator reaches it); discovery changes *who* runs a step, never how it is
    governed, contract-enforced, compensated, or audited.
    """

    def __init__(
        self,
        saga: Saga,
        participants: dict[str, Any],
        *,
        coordinator: str = "coordinator",
        store: Any | None = None,
        audit: Any | None = None,
        events: Any | None = None,
        signer: Any | None = None,
        binder: CapabilityBinder | None = None,
        clock: Callable[[], float] | None = None,
        raise_on_compensation_failure: bool = False,
    ) -> None:
        self.saga = saga.validate_coherent()
        self.participants = {
            org: self._coerce_participant(org, p) for org, p in participants.items()
        }
        self.binder = binder
        # Static steps must name a registered participant up front; discovered
        # steps bind at dispatch time and so require a binder, not a pre-registered
        # org (the resolved org is checked against ``participants`` when it is bound).
        static_orgs = {s.participant for s in self.saga.steps if not s.is_discovered}
        missing = sorted(static_orgs - set(self.participants))
        if missing:
            raise ChoreographyError(
                f"saga {self.saga.name!r} dispatches to unregistered participant(s) {missing}",
                details={"saga": self.saga.name, "missing": missing},
            )
        if any(s.is_discovered for s in self.saga.steps) and self.binder is None:
            raise ChoreographyError(
                f"saga {self.saga.name!r} declares capability steps but no binder was "
                f"supplied to resolve them (pass binder= / directory=)",
                details={
                    "saga": self.saga.name,
                    "discovered_steps": [s.name for s in self.saga.steps if s.is_discovered],
                },
            )
        self.coordinator = coordinator
        self.store = store
        self.audit = audit
        self.events = events
        self.signer = signer
        self._clock = clock or time.monotonic
        self.raise_on_compensation_failure = raise_on_compensation_failure

    @staticmethod
    def _coerce_participant(org: str, spec: Any) -> Participant:
        if isinstance(spec, dict):
            return LocalParticipant(org, spec)
        if isinstance(spec, Participant):
            return spec
        raise ChoreographyError(
            f"participant for {org!r} must be a Participant or a dict of handlers"
        )

    def _resolve(self, org: str) -> Participant:
        participant = self.participants.get(org)
        if participant is None:
            raise ChoreographyError(
                f"saga {self.saga.name!r} dispatches to unregistered participant {org!r}",
                details={"saga": self.saga.name, "participant": org},
            )
        return participant

    # -- public API ----------------------------------------------------------------

    async def arun(
        self,
        input: dict[str, Any] | None = None,
        *,
        saga_id: str | None = None,
        interrupt_after: int | None = None,
    ) -> SagaResult:
        """Run the saga to completion, a clean unwind, or a cooperative pause.

        ``interrupt_after`` stops the forward pass after that many steps with
        ``status="interrupted"`` (the journal is persisted), so a long cross-org
        saga can be paused and continued later — or in another process — with
        :meth:`aresume`.
        """
        journal = SagaJournal(
            name=self.saga.name,
            coordinator=self.coordinator,
            input=dict(input or {}),
        )
        if saga_id:
            journal.id = saga_id
        self._checkpoint(journal)
        return await self._drive(journal, interrupt_after=interrupt_after)

    def run(self, input: dict[str, Any] | None = None, **kwargs: Any) -> SagaResult:
        """Synchronous wrapper around :meth:`arun`."""
        from ..providers.base import run_sync

        return run_sync(self.arun(input, **kwargs))

    async def aresume(
        self, saga_id: str, *, interrupt_after: int | None = None
    ) -> SagaResult:
        """Resume a paused, running, or compensating saga from the durable store.

        Loads the journal by ``saga_id`` and continues from the cursor — completed
        steps keep their outputs and are not re-run; a saga interrupted mid-rollback
        finishes compensating, and one left ``failed`` (a compensation could not
        complete) retries its outstanding compensations. A cleanly-finished saga
        (``completed`` / ``compensated``) is returned unchanged, so resume is
        idempotent.
        """
        journal = self._load(saga_id)
        if journal is None:
            raise ChoreographyError(
                f"no saga {saga_id!r} in the durable store to resume",
                details={"saga_id": saga_id},
            )
        # A cleanly-finished saga is terminal and idempotent; a "failed" one
        # (compensation left incomplete) is resumable so its outstanding
        # compensations retry once the participant is reachable again.
        if journal.status in ("completed", "compensated"):
            return SagaResult.from_journal(journal)
        return await self._drive(journal, interrupt_after=interrupt_after)

    def resume(self, saga_id: str, **kwargs: Any) -> SagaResult:
        """Synchronous wrapper around :meth:`aresume`."""
        from ..providers.base import run_sync

        return run_sync(self.aresume(saga_id, **kwargs))

    # -- execution -----------------------------------------------------------------

    async def _drive(
        self, journal: SagaJournal, *, interrupt_after: int | None
    ) -> SagaResult:
        started = self._clock()
        # A fresh / interrupted saga runs forward (which drives its own
        # compensation on a failure); a saga resumed while compensating — or one
        # left "failed" with compensations still outstanding — re-enters
        # compensation directly. The branches are exclusive so a failing forward
        # pass never double-compensates.
        if journal.status in ("pending", "running", "interrupted"):
            await self._forward(journal, interrupt_after=interrupt_after)
        elif journal.status in ("compensating", "failed"):
            await self._compensate(journal)
        duration_ms = int((self._clock() - started) * 1000)
        return SagaResult.from_journal(journal, duration_ms=duration_ms)

    async def _forward(self, journal: SagaJournal, *, interrupt_after: int | None) -> None:
        journal.status = "running"
        self._checkpoint(journal)
        run_this_segment = 0
        while journal.cursor < len(self.saga.steps):
            if interrupt_after is not None and run_this_segment >= interrupt_after:
                journal.status = "interrupted"
                self._checkpoint(journal)
                return
            step = self.saga.steps[journal.cursor]
            record = await self._perform_step(journal, step)
            run_this_segment += 1
            if record.status == "failed":
                journal.status = "compensating"
                self._checkpoint(journal)
                await self._compensate(journal)
                return
            journal.cursor += 1
            self._checkpoint(journal)
        journal.status = "completed"
        self._checkpoint(journal)
        self._emit(journal)

    def _bind_step(
        self, journal: SagaJournal, step: SagaStep
    ) -> tuple[str, Participant, StepBinding | None]:
        """Resolve which org runs ``step`` — static wiring or run-time discovery.

        A discovered step is bound at dispatch time from the governed directory,
        ranked by reputation and prior settlement fit; the decision is recorded on
        the coordinator's audit chain (``choreography_bind``) and carried on the
        journal. The resolved org must have a participant binding to be reachable.
        """
        if not step.is_discovered:
            return step.participant, self._resolve(step.participant), None
        assert self.binder is not None  # guaranteed by __init__ validation
        binding = self.binder.bind(step, available=set(self.participants))
        self._audit_bind(journal, binding)
        return binding.org, self._resolve(binding.org), binding

    async def _perform_step(self, journal: SagaJournal, step: SagaStep) -> StepRecord:
        org, participant, binding = self._bind_step(journal, step)
        payload = self._build_payload(journal, step)
        request = StepRequest(
            saga_id=journal.id,
            step=step.name,
            action=step.action,
            kind="forward",
            payload=payload,
            scope=step.scope,
            contract_id=step.contract_id,
            capability=step.capability,
        )
        outcome, attempts = await self._dispatch(participant.perform, request, step)
        fulfilled, breaches = self._check_contract(step, outcome)
        status: StepStatus = "completed" if (outcome.ok and fulfilled) else "failed"
        error = outcome.error
        if outcome.ok and not fulfilled:
            error = "contract breach: " + "; ".join(breaches)
        record = StepRecord(
            seq=len(journal.records),
            step=step.name,
            org=org,
            action=step.action,
            kind="forward",
            status=status,
            attempts=attempts,
            contract_id=step.contract_id,
            capability=step.capability or None,
            binding=binding,
            output=outcome.output,
            cost_usd=outcome.cost_usd,
            latency_ms=outcome.latency_ms,
            quality=outcome.quality,
            fulfilled=fulfilled if step.contract is not None else None,
            breaches=breaches,
            error=error,
        )
        journal.append(record, signer=self.signer)
        self._audit(journal, record)
        return record

    async def _compensate(self, journal: SagaJournal) -> None:
        journal.status = "compensating"
        self._checkpoint(journal)
        already = journal.compensated_steps()
        # Reverse completion order: the most recent completed step unwinds first.
        pending = [
            r for r in reversed(journal.completed_forward()) if r.step not in already
        ]
        for forward in pending:
            step = self.saga.by_name(forward.step)
            if step is None or step.compensation is None:
                continue
            # Compensate against the org that actually ran the forward step — the
            # one recorded on the journal — so a discovered step unwinds at the
            # counterparty it was bound to, never a freshly re-resolved one.
            participant = self._resolve(forward.org)
            request = StepRequest(
                saga_id=journal.id,
                step=step.name,
                action=step.compensation,
                kind="compensation",
                payload={"forward_output": dict(forward.output), **dict(step.payload)},
                scope=step.scope,
                contract_id=step.contract_id,
                capability=step.capability,
            )
            outcome, attempts = await self._dispatch(
                participant.compensate, request, step
            )
            status: StepStatus = "compensated" if outcome.ok else "compensation_failed"
            record = StepRecord(
                seq=len(journal.records),
                step=step.name,
                org=forward.org,
                action=step.compensation,
                kind="compensation",
                status=status,
                attempts=attempts,
                contract_id=step.contract_id,
                capability=step.capability or None,
                output=outcome.output,
                error=outcome.error,
            )
            journal.append(record, signer=self.signer)
            self._audit(journal, record)
            self._checkpoint(journal)
        # A step is outstanding only if it completed forward, declares a
        # compensation, and has no successful compensation on record — so a
        # compensation retried successfully on a resume clears the residue
        # rather than being shadowed by its earlier failed attempt.
        compensated = journal.compensated_steps()
        failures = [
            r.step
            for r in journal.completed_forward()
            if r.step not in compensated
            and (step := self.saga.by_name(r.step)) is not None
            and step.compensation is not None
        ]
        journal.status = "failed" if failures else "compensated"
        self._checkpoint(journal)
        self._emit(journal)
        if failures and self.raise_on_compensation_failure:
            raise CompensationError(
                f"saga {journal.id} could not unwind cleanly: {failures}",
                failures=failures,
                details={"saga_id": journal.id, "failures": failures},
            )

    async def _dispatch(
        self,
        fn: Callable[[StepRequest], Awaitable[StepOutcome]],
        request: StepRequest,
        step: SagaStep,
    ) -> tuple[StepOutcome, int]:
        last: StepOutcome = StepOutcome(ok=False, error="not dispatched")
        for attempt in range(step.retries + 1):
            try:
                last = await fn(request)
            except Exception as exc:  # noqa: BLE001 - a raising binding is a failed step
                last = StepOutcome(ok=False, error=f"{type(exc).__name__}: {exc}")
            if last.ok:
                return last, attempt + 1
            if attempt < step.retries and step.retry_delay_s > 0:
                await asyncio.sleep(step.retry_delay_s * (2**attempt))
        return last, step.retries + 1

    def _build_payload(self, journal: SagaJournal, step: SagaStep) -> dict[str, Any]:
        if step.build is None:
            return dict(step.payload)
        ctx = SagaContext(input=dict(journal.input), outputs=journal.context())
        built = step.build(ctx)
        if not isinstance(built, dict):
            raise ChoreographyError(
                f"step {step.name!r} build must return a dict; got {type(built).__name__}"
            )
        return {**dict(step.payload), **built}

    @staticmethod
    def _check_contract(step: SagaStep, outcome: StepOutcome) -> tuple[bool, list[str]]:
        if step.contract is None or not outcome.ok:
            return True, []
        fulfillment = step.contract.check(
            cost_usd=outcome.cost_usd,
            latency_ms=outcome.latency_ms,
            quality=outcome.quality,
        )
        return fulfillment.fulfilled, list(fulfillment.breaches)

    # -- durability / observability ------------------------------------------------

    def _checkpoint(self, journal: SagaJournal) -> None:
        if self.store is not None:
            self.store.save(SAGA_STORE_KIND, journal.to_record())

    def _load(self, saga_id: str) -> SagaJournal | None:
        if self.store is None:
            return None
        record = self.store.get(SAGA_STORE_KIND, saga_id)
        return SagaJournal.from_record(record) if record else None

    def _audit_bind(self, journal: SagaJournal, binding: StepBinding) -> None:
        """Record a run-time binding decision on the coordinator's chain."""
        if self.audit is None:
            return
        self.audit.record(
            BIND_ACTION,
            resource=journal.id,
            decision=binding.org,
            details=binding.audit_details(),
        )

    def _audit(self, journal: SagaJournal, record: StepRecord) -> None:
        if self.audit is None:
            return
        self.audit.record(
            STEP_DISPATCH_ACTION,
            resource=journal.id,
            decision=record.status,
            details={
                "step": record.step,
                "org": record.org,
                "action": record.action,
                "kind": record.kind,
                "contract_id": record.contract_id,
                "fulfilled": record.fulfilled,
                "breaches": record.breaches,
                "entry_hash": record.entry_hash,
            },
        )

    def _emit(self, journal: SagaJournal) -> None:
        if self.events is None:
            return
        try:
            self.events.emit(
                "choreography.completed",
                {
                    "saga_id": journal.id,
                    "name": journal.name,
                    "status": journal.status,
                    "steps": len(journal.completed_forward()),
                    "compensated": len(journal.compensated_steps()),
                },
            )
        except Exception:  # noqa: BLE001 - event delivery is best-effort
            pass
