"""The durable, content-bound state of a cross-org saga.

A :class:`Saga` is the *definition* — an ordered list of :class:`SagaStep` s, each
dispatched to a named participant org with an optional compensating step. The
:class:`SagaJournal` is the *durable record* of one run of that definition: a
hash-chained log of every forward and compensating step's outcome that survives a
restart and verifies **offline** from the bytes alone, exactly like the audit
chain and the negotiated :class:`~vincio.negotiation.Contract` it runs under.

Three guarantees, all dependency-free and deterministic:

* **Typed handoffs.** A :class:`StepRequest` is the only thing that crosses a
  trust boundary — a typed envelope (saga / step / action / payload, bound to a
  contract) the remote org answers with a :class:`StepOutcome`. No shared control
  plane, only the audited handoff.
* **Durable & resumable.** Every step appends a :class:`StepRecord` to the
  :class:`SagaJournal`, which the engine checkpoints to the metadata store after
  each move. A fresh process reloads the journal by ``saga_id`` and continues from
  the cursor — completed steps are never re-run.
* **Tamper-evident.** Each record links to the previous by a content hash (and,
  with a signer, carries a signature), so :meth:`SagaJournal.verify` recomputes
  the chain and pinpoints any edited, inserted, or dropped record without the live
  coordinator.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .discovery import StepBinding

__all__ = [
    "StepRequest",
    "StepOutcome",
    "StepRecord",
    "SagaStep",
    "Saga",
    "SagaContext",
    "SagaJournal",
    "JournalVerification",
    "SagaResult",
    "SAGA_STORE_KIND",
    "STEP_DISPATCH_ACTION",
]

# The metadata-store record kind a saga journal is persisted under (generic
# ``records`` table on backed stores; no RECORD_KINDS registration needed).
SAGA_STORE_KIND = "choreography_sagas"

# The audit action a coordinator records a dispatched handoff under.
STEP_DISPATCH_ACTION = "choreography_step"

StepKind = Literal["forward", "compensation"]
StepStatus = Literal[
    "completed", "failed", "compensated", "compensation_failed", "skipped"
]
SagaStatus = Literal[
    "pending", "running", "interrupted", "completed", "compensating", "compensated", "failed"
]


class StepRequest(BaseModel):
    """The typed envelope dispatched to a participant for one step — the handoff.

    The only artifact that crosses a trust boundary. ``kind`` distinguishes a
    forward step from its compensation; ``payload`` is the step's input (built from
    prior steps' outputs); ``contract_id`` binds the handoff to the negotiated
    :class:`~vincio.negotiation.Contract` that governs it.
    """

    saga_id: str
    step: str
    action: str
    kind: StepKind = "forward"
    payload: dict[str, Any] = Field(default_factory=dict)
    scope: str = ""
    contract_id: str | None = None
    capability: str = ""

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for dispatch over the A2A fabric."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> StepRequest:
        return cls.model_validate(data)


class StepOutcome(BaseModel):
    """A participant's result for one dispatched step.

    ``ok`` is the participant's own verdict; ``cost_usd`` / ``latency_ms`` /
    ``quality`` are the delivered metrics the coordinator checks against the step's
    contract (a breach turns a returned success into a saga failure). A handler
    that raises is rendered as ``ok=False`` with the exception on ``error``.
    """

    ok: bool = True
    output: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float | None = None
    latency_ms: float | None = None
    quality: float | None = None
    error: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for return over the A2A fabric."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> StepOutcome:
        return cls.model_validate(data)


class SagaContext(BaseModel):
    """What a step's ``build`` callable sees: the saga input and prior outputs.

    ``input`` is the run input; ``outputs`` maps each completed forward step's name
    to its recorded output. ``ctx["step"]`` is shorthand for that step's output, so
    a later step can derive its payload from what ran before it.
    """

    input: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def output_of(self, step: str) -> dict[str, Any]:
        return dict(self.outputs.get(step, {}))

    def __getitem__(self, step: str) -> dict[str, Any]:
        return self.output_of(step)


class SagaStep(BaseModel):
    """One step of a :class:`Saga`: a forward action and its compensation.

    A step names **who** runs it in exactly one of two ways. A statically-wired
    step sets ``participant`` to a fixed org id. A **discovered** step instead sets
    ``capability`` to the capability it needs, and the engine resolves the
    participant from the governed agent directory at dispatch time — preferring the
    allowed candidate whose reputation and prior settlement record best fit the
    step's contract (see :mod:`vincio.choreography.discovery`). Either way ``action``
    is the capability it runs forward and ``compensation`` is the capability that
    undoes it on rollback (``None`` means the step needs no undo).

    ``build`` optionally derives the request payload from the accumulated outputs of
    prior steps; ``contract`` binds the step to a negotiated agreement whose
    price / SLA / quality the coordinator enforces on the delivered outcome (and
    which a discovered step's binding ranks candidates against).
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    participant: str = ""
    action: str
    capability: str = ""
    compensation: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    build: Any = None  # Callable[[SagaContext], dict] | None
    scope: str = ""
    contract: Any = None  # vincio.negotiation.Contract | None
    retries: int = 0
    retry_delay_s: float = 0.0

    @model_validator(mode="after")
    def _exactly_one_binding(self) -> SagaStep:
        """A step is statically wired (``participant``) xor discovered (``capability``)."""
        if bool(self.participant) == bool(self.capability):
            from ..core.errors import ChoreographyError

            raise ChoreographyError(
                f"saga step {self.name!r} must declare exactly one of "
                f"participant= (static) or capability= (discovered)"
            )
        return self

    @property
    def is_discovered(self) -> bool:
        """Whether this step binds its participant at run time from a capability."""
        return bool(self.capability)

    @property
    def contract_id(self) -> str | None:
        return getattr(self.contract, "id", None)


class Saga(BaseModel):
    """A cross-org compensating workflow: an ordered list of steps.

    Steps run in declaration order. A failure or contract breach on any step
    triggers deterministic compensation of the already-completed steps in reverse
    order, so a half-completed cross-org transaction unwinds cleanly. The
    definition is rebuilt in code on a restart; only the :class:`SagaJournal`
    (outputs and progress) is persisted, so dynamic ``build`` callables resume
    transparently.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    steps: list[SagaStep] = Field(default_factory=list)

    def step(
        self,
        name: str,
        *,
        action: str,
        participant: str | None = None,
        capability: str | None = None,
        compensation: str | None = None,
        payload: dict[str, Any] | None = None,
        build: Any = None,
        scope: str = "",
        contract: Any = None,
        retries: int = 0,
        retry_delay_s: float = 0.0,
    ) -> Saga:
        """Append a step and return ``self`` for chaining.

        Pass exactly one of ``participant`` (a fixed org id — static wiring) or
        ``capability`` (the capability to resolve from the governed directory at
        dispatch time — run-time discovery).
        """
        if any(s.name == name for s in self.steps):
            from ..core.errors import ChoreographyError

            raise ChoreographyError(f"duplicate saga step {name!r}")
        self.steps.append(
            SagaStep(
                name=name,
                participant=participant or "",
                action=action,
                capability=capability or "",
                compensation=compensation,
                payload=payload or {},
                build=build,
                scope=scope,
                contract=contract,
                retries=retries,
                retry_delay_s=retry_delay_s,
            )
        )
        return self

    def by_name(self, name: str) -> SagaStep | None:
        return next((s for s in self.steps if s.name == name), None)

    def validate_coherent(self) -> Saga:
        """Raise :class:`ChoreographyError` unless the saga can run."""
        from ..core.errors import ChoreographyError

        if not self.steps:
            raise ChoreographyError(f"saga {self.name!r} has no steps")
        seen: set[str] = set()
        for s in self.steps:
            if s.name in seen:
                raise ChoreographyError(f"duplicate saga step {s.name!r}")
            seen.add(s.name)
        return self


class StepRecord(BaseModel):
    """One immutable, hash-chained entry in a :class:`SagaJournal`.

    A forward step records its outcome and (when a contract governs it) the
    fulfilment verdict; a compensation records whether the undo ran. ``prev_hash``
    links to the previous record and ``entry_hash`` seals this one, so the journal
    is a tamper-evident chain like the audit log.
    """

    seq: int
    step: str
    org: str
    action: str
    kind: StepKind
    status: StepStatus
    attempts: int = 1
    contract_id: str | None = None
    capability: str | None = None
    binding: StepBinding | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float | None = None
    latency_ms: float | None = None
    quality: float | None = None
    fulfilled: bool | None = None
    breaches: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime = Field(default_factory=utcnow)
    prev_hash: str = ""
    entry_hash: str = ""
    signature: str = ""
    key_id: str = ""

    def compute_hash(self) -> str:
        """The content hash binding this record to the one before it."""
        return stable_hash(
            {
                "seq": self.seq,
                "step": self.step,
                "org": self.org,
                "action": self.action,
                "kind": self.kind,
                "status": self.status,
                "contract_id": self.contract_id,
                "capability": self.capability,
                "output": to_jsonable(self.output),
                "cost_usd": self.cost_usd,
                "latency_ms": self.latency_ms,
                "quality": self.quality,
                "fulfilled": self.fulfilled,
                "breaches": self.breaches,
                "error": self.error,
                "prev_hash": self.prev_hash,
            },
            length=32,
        )


class JournalVerification(BaseModel):
    """The (non-raising) outcome of verifying a saga journal offline."""

    intact: bool
    entries: int
    broken_at: int | None = None
    reason: str | None = None


class SagaJournal(BaseModel):
    """The durable, resumable, offline-verifiable record of one saga run.

    ``id`` is the saga id the metadata store keys on; ``cursor`` is the index of
    the next forward step to attempt; ``records`` is the hash-chained log of every
    move. :meth:`append` links and seals a record; :meth:`verify` recomputes the
    chain; :meth:`context` reconstructs prior steps' outputs so a later step's
    payload (and a resumed run) sees what ran before it.
    """

    id: str = Field(default_factory=lambda: new_id("saga"))
    name: str = ""
    coordinator: str = ""
    status: SagaStatus = "pending"
    cursor: int = 0
    input: dict[str, Any] = Field(default_factory=dict)
    records: list[StepRecord] = Field(default_factory=list)
    head_hash: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def saga_id(self) -> str:
        return self.id

    def append(self, record: StepRecord, *, signer: Any | None = None) -> StepRecord:
        """Link ``record`` to the chain head, seal it, and advance the head."""
        record.prev_hash = self.head_hash
        record.entry_hash = record.compute_hash()
        if signer is not None:
            record.signature = signer.sign(record.entry_hash)
            record.key_id = getattr(signer, "key_id", "")
        self.records.append(record)
        self.head_hash = record.entry_hash
        self.updated_at = utcnow()
        return record

    def verify(self, verifier: Any | None = None) -> JournalVerification:
        """Recompute the hash chain (and, with a verifier, every signature).

        Detects any record edited, inserted, or dropped after the fact — the chain
        breaks at the first record whose hash does not recompute or whose link to
        its predecessor is wrong, pinpointed by ``broken_at``.
        """
        previous = ""
        for record in self.records:
            if record.prev_hash != previous or record.entry_hash != record.compute_hash():
                return JournalVerification(
                    intact=False,
                    entries=len(self.records),
                    broken_at=record.seq,
                    reason="hash chain broken",
                )
            if verifier is not None and record.signature:
                if not verifier.verify(record.entry_hash, record.signature):
                    return JournalVerification(
                        intact=False,
                        entries=len(self.records),
                        broken_at=record.seq,
                        reason="signature mismatch",
                    )
            previous = record.entry_hash
        if self.head_hash != previous:
            return JournalVerification(
                intact=False,
                entries=len(self.records),
                reason="head hash does not match chain",
            )
        return JournalVerification(intact=True, entries=len(self.records))

    def forward_records(self) -> list[StepRecord]:
        return [r for r in self.records if r.kind == "forward"]

    def completed_forward(self) -> list[StepRecord]:
        """Forward steps that completed, in execution order (compensation source)."""
        return [r for r in self.records if r.kind == "forward" and r.status == "completed"]

    def compensated_steps(self) -> set[str]:
        return {
            r.step
            for r in self.records
            if r.kind == "compensation" and r.status == "compensated"
        }

    def context(self) -> dict[str, dict[str, Any]]:
        """The accumulated outputs of completed forward steps, keyed by step name."""
        return {r.step: dict(r.output) for r in self.completed_forward()}

    def bindings(self) -> dict[str, StepBinding]:
        """The run-time binding decision for each discovered step, keyed by step.

        Empty for a fully statically-wired saga; one
        :class:`~vincio.choreography.discovery.StepBinding` per discovered forward
        step otherwise, so a resolved-at-run-time choreography records *who* was
        bound and why, beside the journal it ran.
        """
        return {
            r.step: r.binding
            for r in self.records
            if r.kind == "forward" and r.binding is not None
        }

    def to_record(self) -> dict[str, Any]:
        """A JSON-safe projection for the metadata store (keyed by ``id``)."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> SagaJournal:
        return cls.model_validate(data)


class SagaResult(BaseModel):
    """The outcome of a cross-org saga run — completion, a clean unwind, or a pause.

    ``status`` is ``"completed"`` when every forward step ran, ``"compensated"``
    when a failure unwound the completed steps cleanly, ``"failed"`` when a
    compensation itself could not complete, or ``"interrupted"`` when the run was
    paused (and :meth:`~vincio.choreography.Choreography.resume` can continue it).
    The full :class:`SagaJournal` is always carried, so any outcome is inspectable
    and offline-verifiable.
    """

    saga_id: str
    name: str = ""
    status: SagaStatus
    journal: SagaJournal
    completed_steps: list[str] = Field(default_factory=list)
    compensated_steps: list[str] = Field(default_factory=list)
    failed_step: str | None = None
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        """Whether every forward step completed."""
        return self.status == "completed"

    @property
    def unwound(self) -> bool:
        """Whether a failure was compensated cleanly (no residue)."""
        return self.status == "compensated"

    @property
    def output(self) -> dict[str, Any]:
        """The output of the last completed forward step (empty if none)."""
        completed = self.journal.completed_forward()
        return dict(completed[-1].output) if completed else {}

    def output_of(self, step: str) -> dict[str, Any]:
        """The recorded output of a named completed forward step."""
        for record in self.journal.completed_forward():
            if record.step == step:
                return dict(record.output)
        return {}

    @property
    def bindings(self) -> dict[str, StepBinding]:
        """The run-time binding decision for each discovered step (see the journal)."""
        return self.journal.bindings()

    @classmethod
    def from_journal(cls, journal: SagaJournal, *, duration_ms: int = 0) -> SagaResult:
        failed = next(
            (r.step for r in journal.forward_records() if r.status == "failed"), None
        )
        return cls(
            saga_id=journal.id,
            name=journal.name,
            status=journal.status,
            journal=journal,
            completed_steps=[r.step for r in journal.completed_forward()],
            compensated_steps=[
                r.step
                for r in journal.records
                if r.kind == "compensation" and r.status == "compensated"
            ],
            failed_step=failed,
            duration_ms=duration_ms,
        )
