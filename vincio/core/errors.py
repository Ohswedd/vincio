"""Vincio error hierarchy.

Every error raised by Vincio derives from :class:`VincioError` so applications
can catch the full family with one except clause. Subsystem errors carry
structured details for tracing and programmatic handling.
"""

from __future__ import annotations

from typing import Any

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
    "PIIPolicyError",
    "EgressBlockedError",
    "StorageError",
    "ServerError",
    "AuthenticationError",
    "GovernanceError",
    "ResidencyViolationError",
    "ErasureError",
]


class VincioError(Exception):
    """Base class for all Vincio errors."""

    code: str = "VINCIO_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}

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
    """An image-generation/editing or speech-synthesis provider call failed."""

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


# --- storage -----------------------------------------------------------------


class StorageError(VincioError):
    code = "STORAGE_ERROR"


# --- server ------------------------------------------------------------------


class ServerError(VincioError):
    code = "SERVER_ERROR"


class AuthenticationError(ServerError):
    code = "AUTHENTICATION_ERROR"
