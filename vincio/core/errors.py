"""Vincio error hierarchy.

Every error raised by Vincio derives from :class:`VincioError` so applications
can catch the full family with one except clause. Subsystem errors carry
structured details for tracing and programmatic handling.
"""

from __future__ import annotations

from typing import Any

from .error_catalog import docs_url_for, remediation_for

__all__ = [
    "VincioError",
    "ConfigError",
    "ProviderError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "ProviderResponseError",
    "CircuitOpenError",
    "BatchError",
    "FineTuneError",
    "CapabilityMismatchError",
    "ModelRetiredError",
    "PromptError",
    "PromptLintError",
    "PromptBudgetError",
    "ContextError",
    "ContextCompileError",
    "BudgetExceededError",
    "InputError",
    "DocumentError",
    "LoaderError",
    "RetrievalError",
    "IndexError_",
    "MemoryEngineError",
    "MemoryPolicyError",
    "MemoryConflictError",
    "ToolError",
    "ToolNotFoundError",
    "ToolPermissionError",
    "ToolValidationError",
    "ToolTimeoutError",
    "ToolApprovalRequiredError",
    "SandboxError",
    "AgentEngineError",
    "AgentStepError",
    "AgentBudgetExhaustedError",
    "AgentMaxStepsError",
    "GraphError",
    "CheckpointConflictError",
    "WorkflowError",
    "WorkflowStepError",
    "OutputError",
    "OutputParseError",
    "OutputSchemaError",
    "OutputRepairForbiddenError",
    "CitationValidationError",
    "GenerationError",
    "DocumentContractError",
    "MediaGenerationError",
    "EvalError",
    "DatasetError",
    "GateFailedError",
    "OptimizationError",
    "CacheError",
    "SecurityError",
    "AccessDeniedError",
    "TenantIsolationError",
    "InjectionDetectedError",
    "ContainmentError",
    "PIIPolicyError",
    "EgressBlockedError",
    "StorageError",
    "ServerError",
    "AuthenticationError",
    "GovernanceError",
    "ResidencyViolationError",
    "ErasureError",
    "ObservabilityError",
    "ReplayDivergenceError",
    "EnergyBudgetError",
    "EdgeError",
    "NegotiationError",
    "ContractError",
    "ChoreographyError",
    "CompensationError",
    "SettlementError",
]


class VincioError(Exception):
    """Base class for all Vincio errors.

    Every error carries a stable :attr:`code`, plus an actionable
    :attr:`remediation` hint and a :attr:`docs_url` deep link resolved from the
    :mod:`~vincio.core.error_catalog`. Catch the whole family with one
    ``except VincioError``, branch on ``.code`` for programmatic handling, and
    surface ``.remediation`` to users. Message *strings* are not part of the
    stable API; the ``.code`` values and the catalog are.
    """

    code: str = "VINCIO_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        hint: str | None = None,
        docs_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}
        self._hint = hint
        self._docs_url = docs_url

    @property
    def remediation(self) -> str | None:
        """Actionable next step: the instance override, else the catalog hint."""
        if self._hint is not None:
            return self._hint
        code = self.code
        return remediation_for(code) if isinstance(code, str) else None

    @property
    def docs_url(self) -> str | None:
        """Deep link to this error's reference entry (override or catalog)."""
        if self._docs_url is not None:
            return self._docs_url
        code = self.code
        return docs_url_for(code) if isinstance(code, str) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "remediation": self.remediation,
            "docs_url": self.docs_url,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.message!r})"


# --- configuration ---------------------------------------------------------


class ConfigError(VincioError):
    code = "CONFIG_ERROR"


# --- providers --------------------------------------------------------------


class ProviderError(VincioError):
    """Base error for model provider failures."""

    code = "PROVIDER_ERROR"

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.provider = provider
        self.model = model
        self.retryable = retryable


class ProviderAuthError(ProviderError):
    code = "PROVIDER_AUTH"


class ProviderRateLimitError(ProviderError):
    code = "PROVIDER_RATE_LIMIT"

    def __init__(self, message: str, *, retry_after_s: float | None = None, **kw: Any) -> None:
        kw.setdefault("retryable", True)
        super().__init__(message, **kw)
        self.retry_after_s = retry_after_s


class ProviderTimeoutError(ProviderError):
    code = "PROVIDER_TIMEOUT"

    def __init__(self, message: str, **kw: Any) -> None:
        kw.setdefault("retryable", True)
        super().__init__(message, **kw)


class ProviderUnavailableError(ProviderError):
    code = "PROVIDER_UNAVAILABLE"

    def __init__(self, message: str, **kw: Any) -> None:
        kw.setdefault("retryable", True)
        super().__init__(message, **kw)


class ProviderResponseError(ProviderError):
    code = "PROVIDER_RESPONSE"


class CircuitOpenError(ProviderUnavailableError):
    """Raised by an open :class:`~vincio.providers.CircuitBreaker`.

    Fails fast (non-retryable) so a failover chain skips the unhealthy entry
    immediately instead of waiting on a call that is expected to fail.
    """

    code = "CIRCUIT_OPEN"

    def __init__(self, message: str, **kw: Any) -> None:
        kw.setdefault("retryable", False)
        super().__init__(message, **kw)


class BatchError(ProviderError):
    """A provider Batch API submission/poll/reconciliation failure."""

    code = "BATCH_ERROR"


class FineTuneError(ProviderError):
    """A provider fine-tuning job submission/poll failure.

    Raised when a distillation fine-tune job cannot be submitted, polls to a
    failed/cancelled terminal state, or exceeds its wait budget — so the
    flywheel surfaces "the student was not trained" rather than silently
    promoting the untrained base model.
    """

    code = "FINETUNE_ERROR"


class CapabilityMismatchError(ProviderError):
    """A model cannot serve the request (missing vision/tools/context/etc.).

    Raised by the capability guard when a substitution would route a
    request to a model that structurally cannot fulfil it. Non-retryable: the
    fix is to escalate to a capable model, not to retry the same one.
    """

    code = "CAPABILITY_MISMATCH"

    def __init__(self, message: str, *, missing: list[str] | None = None, **kw: Any) -> None:
        kw.setdefault("retryable", False)
        super().__init__(message, **kw)
        self.missing = list(missing or [])


class ModelRetiredError(ProviderError):
    """A pinned model is past its registry retirement date.

    Terminal and lifecycle-classified, distinct from a transient availability
    error: a retired-model failure surfaces "rotate now" rather than being
    buried in "all providers failed".
    """

    code = "MODEL_RETIRED"

    def __init__(self, message: str, **kw: Any) -> None:
        kw.setdefault("retryable", False)
        super().__init__(message, **kw)


# --- prompt engine ----------------------------------------------------------


class PromptError(VincioError):
    code = "PROMPT_ERROR"


class PromptLintError(PromptError):
    code = "PROMPT_LINT"

    def __init__(self, message: str, *, findings: list[Any] | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.findings = findings or []


class PromptBudgetError(PromptError):
    code = "PROMPT_BUDGET"


# --- context compiler -------------------------------------------------------


class ContextError(VincioError):
    code = "CONTEXT_ERROR"


class ContextCompileError(ContextError):
    code = "CONTEXT_COMPILE"


class BudgetExceededError(ContextError):
    code = "BUDGET_EXCEEDED"

    def __init__(
        self, message: str, *, used: int | float = 0, limit: int | float = 0, **kw: Any
    ) -> None:
        super().__init__(message, **kw)
        self.used = used
        self.limit = limit


# --- input ------------------------------------------------------------------


class InputError(VincioError):
    code = "INPUT_ERROR"


# --- documents --------------------------------------------------------------


class DocumentError(VincioError):
    code = "DOCUMENT_ERROR"


class LoaderError(DocumentError):
    code = "LOADER_ERROR"


# --- retrieval --------------------------------------------------------------


class RetrievalError(VincioError):
    code = "RETRIEVAL_ERROR"


class IndexError_(RetrievalError):
    """Index failure (named with a trailing underscore to avoid shadowing builtins)."""

    code = "INDEX_ERROR"


# --- memory -----------------------------------------------------------------


class MemoryEngineError(VincioError):
    code = "MEMORY_ERROR"


class MemoryPolicyError(MemoryEngineError):
    code = "MEMORY_POLICY"


class MemoryConflictError(MemoryEngineError):
    code = "MEMORY_CONFLICT"


# --- tools ------------------------------------------------------------------


class ToolError(VincioError):
    code = "TOOL_ERROR"

    def __init__(self, message: str, *, tool: str | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.tool = tool


class ToolNotFoundError(ToolError):
    code = "TOOL_NOT_FOUND"


class ToolPermissionError(ToolError):
    code = "TOOL_PERMISSION"


class ToolValidationError(ToolError):
    code = "TOOL_VALIDATION"


class ToolTimeoutError(ToolError):
    code = "TOOL_TIMEOUT"


class ToolApprovalRequiredError(ToolError):
    code = "TOOL_APPROVAL_REQUIRED"


class SandboxError(ToolError):
    """Isolation/sandbox failure: backend unavailable or isolation too weak."""

    code = "SANDBOX_ERROR"


# --- agents -----------------------------------------------------------------


class AgentEngineError(VincioError):
    code = "AGENT_ERROR"


class AgentStepError(AgentEngineError):
    code = "AGENT_STEP"

    def __init__(self, message: str, *, step_id: str | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.step_id = step_id


class AgentBudgetExhaustedError(AgentEngineError):
    code = "AGENT_BUDGET_EXHAUSTED"


class AgentMaxStepsError(AgentEngineError):
    code = "AGENT_MAX_STEPS"


class GraphError(AgentEngineError):
    """Stateful-graph definition or execution error."""

    code = "GRAPH_ERROR"


class CheckpointConflictError(GraphError):
    """A distributed super-step commit lost the optimistic-concurrency race.

    Raised when a checkpoint write's expected version no longer matches the
    thread head — another worker advanced the thread first. The losing worker
    aborts instead of double-executing the step; the winning worker's
    checkpoint stands. Non-fatal at the orchestration layer: re-acquire the
    lease and resume from the new head.
    """

    code = "CHECKPOINT_CONFLICT"

    def __init__(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        expected_version: int | None = None,
        actual_version: int | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(message, **kw)
        self.thread_id = thread_id
        self.expected_version = expected_version
        self.actual_version = actual_version


# --- workflows ---------------------------------------------------------------


class WorkflowError(VincioError):
    code = "WORKFLOW_ERROR"


class WorkflowStepError(WorkflowError):
    code = "WORKFLOW_STEP"

    def __init__(self, message: str, *, step: str | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.step = step


# --- output ------------------------------------------------------------------


class OutputError(VincioError):
    code = "OUTPUT_ERROR"


class OutputParseError(OutputError):
    code = "OUTPUT_PARSE"


class OutputSchemaError(OutputError):
    code = "OUTPUT_SCHEMA"

    def __init__(self, message: str, *, errors: list[Any] | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.errors = errors or []


class OutputRepairForbiddenError(OutputError):
    code = "OUTPUT_REPAIR_FORBIDDEN"


class CitationValidationError(OutputError):
    code = "CITATION_INVALID"


# --- generation (documents & media out) --------------------------------


class GenerationError(VincioError):
    """A document/media generation failure (rendering, contract, provider)."""

    code = "GENERATION_ERROR"


class DocumentContractError(GenerationError):
    """A rendered document violates its :class:`DocumentContract` and the
    formatting-only repair could not bring it into compliance."""

    code = "DOCUMENT_CONTRACT"

    def __init__(self, message: str, *, violations: list[Any] | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.violations = violations or []


class MediaGenerationError(GenerationError):
    """An image-, video-, or speech-generation/editing provider call failed."""

    code = "MEDIA_GENERATION"


# --- evals -------------------------------------------------------------------


class EvalError(VincioError):
    code = "EVAL_ERROR"


class DatasetError(EvalError):
    code = "DATASET_ERROR"


class GateFailedError(EvalError):
    code = "GATE_FAILED"

    def __init__(self, message: str, *, failures: list[Any] | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.failures = failures or []


# --- optimization ------------------------------------------------------------


class OptimizationError(VincioError):
    code = "OPTIMIZATION_ERROR"


# --- caching -----------------------------------------------------------------


class CacheError(VincioError):
    code = "CACHE_ERROR"


# --- security ----------------------------------------------------------------


class SecurityError(VincioError):
    code = "SECURITY_ERROR"


class AccessDeniedError(SecurityError):
    code = "ACCESS_DENIED"


class TenantIsolationError(SecurityError):
    code = "TENANT_ISOLATION"


class InjectionDetectedError(SecurityError):
    code = "INJECTION_DETECTED"


class ContainmentError(SecurityError):
    """A side-effecting capability was exercised on untrusted authority.

    Raised by the capability-secure execution path when an argument derived from
    untrusted data (a retrieved document, a tool result) would flow into a
    write/external tool without a user-minted :class:`~vincio.security.CapabilityToken`
    or a human approval. Containment refuses the call rather than letting an
    injected instruction escalate to an unauthorized side effect.
    """

    code = "CONTAINMENT_BLOCKED"


class PIIPolicyError(SecurityError):
    code = "PII_POLICY"


class EgressBlockedError(SecurityError):
    """The always-on egress DLP scan blocked an outbound provider request
    because it carried credentials, secrets, or sensitive identifiers."""

    code = "EGRESS_BLOCKED"


# --- governance & compliance -------------------------------------------------


class GovernanceError(VincioError):
    """Enterprise governance / compliance failure (cards, BOM, lineage)."""

    code = "GOVERNANCE_ERROR"


class ResidencyViolationError(GovernanceError):
    """A run would route to a provider region the residency policy forbids.

    Raised (or surfaced as a blocking :class:`~vincio.security.PolicyViolation`)
    when in-jurisdiction processing is required and the resolved provider/model
    is not pinned to an allowed region.
    """

    code = "RESIDENCY_VIOLATION"

    def __init__(
        self,
        message: str,
        *,
        region: str | None = None,
        allowed: list[str] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(message, **kw)
        self.region = region
        self.allowed = allowed or []


class ErasureError(GovernanceError):
    """A right-to-erasure-by-source operation could not complete atomically."""

    code = "ERASURE_ERROR"


class GovernanceVerificationError(GovernanceError):
    """A formal governance invariant did not hold across its state space.

    Raised by :meth:`~vincio.core.app.ContextApp.verify_governance` (with
    ``raise_on_violation=True``) when the deterministic verifier finds a
    counterexample to one of the machine-checked invariants — containment,
    residency, budget, or erasure. The offending counterexample(s) are carried on
    :attr:`counterexamples` so the violation is debuggable, not just flagged.
    """

    code = "GOVERNANCE_INVARIANT_VIOLATED"

    def __init__(self, message: str, *, counterexamples: list[Any] | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.counterexamples = counterexamples or []


# --- storage -----------------------------------------------------------------


class StorageError(VincioError):
    code = "STORAGE_ERROR"


# --- server ------------------------------------------------------------------


class ServerError(VincioError):
    code = "SERVER_ERROR"


class AuthenticationError(ServerError):
    code = "AUTHENTICATION_ERROR"


# --- observability -----------------------------------------------------------


class ObservabilityError(VincioError):
    code = "OBSERVABILITY_ERROR"


class ReplayDivergenceError(ObservabilityError):
    """A recorded run no longer replays: live code asked for an edge (a model
    call, tool output, or retrieval) that is not in the recording, or a recording
    could not be loaded/verified."""

    code = "REPLAY_DIVERGENCE"


class EnergyBudgetError(ObservabilityError):
    """An energy/carbon budget was misconfigured — e.g. set with neither an
    energy (``limit_wh``) nor a carbon (``limit_co2e_grams``) ceiling."""

    code = "ENERGY_BUDGET_INVALID"


# --- edge / WASM runtime -----------------------------------------------------


class EdgeError(VincioError):
    """An edge / WASM in-process runtime failure.

    Raised by :class:`~vincio.edge.runtime.EdgeRuntime` when a request carries
    neither a task nor an objective, or (with ``strict=True``) when the compiled
    context cannot be held inside the :class:`~vincio.edge.profile.EdgeProfile`'s
    resident-memory and token bounds.
    """

    code = "EDGE_ERROR"


# --- agent negotiation & contracting -----------------------------------------


class NegotiationError(VincioError):
    """A bounded agent negotiation could not proceed.

    Raised on an incoherent :class:`~vincio.negotiation.NegotiationPosition`
    (a reservation strictly worse for the party than its own ideal, a
    non-positive concession exponent) or a :class:`~vincio.negotiation.Negotiation`
    that cannot run (no buyer/seller party, a non-positive round budget). A
    negotiation that simply reaches its round/deadline budget without a deal does
    **not** raise — it returns a partial :class:`~vincio.negotiation.NegotiationResult`
    with ``status="no_agreement"``; termination is a guarantee, not an error.
    """

    code = "NEGOTIATION_ERROR"


class ContractError(NegotiationError):
    """A negotiated contract failed verification or was breached.

    Raised when a :class:`~vincio.negotiation.Contract`'s content hash does not
    recompute, a required signature is missing or does not verify, or (with
    ``raise_on_breach=True``) when delivered work breaches the agreed
    price / SLA / quality terms the orchestrator enforces like any other budget.
    The breaching terms are carried on :attr:`breaches` so the violation is
    debuggable, not just flagged.
    """

    code = "CONTRACT_VIOLATION"

    def __init__(self, message: str, *, breaches: list[Any] | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.breaches = breaches or []


class ChoreographyError(VincioError):
    """A cross-org workflow choreography could not proceed.

    Raised when a :class:`~vincio.choreography.Saga` cannot run or resume — a step
    that names a participant with no registered binding, a duplicate or empty step
    set, or a :meth:`~vincio.choreography.Choreography.resume` for a ``saga_id``
    that is not in the durable store. A saga whose forward step fails does **not**
    raise — it compensates the completed steps and returns a
    :class:`~vincio.choreography.SagaResult` with ``status="compensated"``; a clean
    unwind is an outcome, not an error.
    """

    code = "CHOREOGRAPHY_ERROR"


class CompensationError(ChoreographyError):
    """A saga could not unwind cleanly: one or more compensations failed.

    Raised (only with ``raise_on_compensation_failure=True``) when a compensating
    step itself fails, so a half-completed cross-org transaction is left partially
    unwound and needs operator attention. The steps whose compensation failed are
    carried on :attr:`failures` so the residue is pinpointed, not just flagged; the
    saga's :class:`~vincio.choreography.SagaResult` ends with ``status="failed"``.
    """

    code = "COMPENSATION_FAILED"

    def __init__(self, message: str, *, failures: list[Any] | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.failures = failures or []


class SettlementError(VincioError):
    """A settlement or metering operation could not proceed.

    Raised when a settlement cannot be built or verified — a meter accruing a
    negative quantity, a settlement signed by a party that is neither the buyer nor
    the seller, a :class:`~vincio.settlement.SettlementRecord` or
    :class:`~vincio.settlement.SettlementBook` that fails offline verification, a
    saga settled without the contract terms its steps ran under, or a reconciliation
    of records for two different contracts. A settlement whose delivered work simply
    breaches the agreed terms does **not** raise — it reconciles to a record with
    ``status="breached"`` and the breaching dimensions on
    :attr:`~vincio.settlement.SettlementRecord.breaches`; a breach is a settled
    outcome that debits the seller's reputation, not an error.
    """

    code = "SETTLEMENT_ERROR"
