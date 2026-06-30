"""Verified reasoning & neuro-symbolic certificates.

Three planes of deterministic, offline verification:

* **Proof-carrying answers** — a :class:`Certificate` records what a set of
  deterministic kernels checked about an answer, a :class:`VerifiedAnswer` pairs
  the result with it, and ``app.verify_reasoning`` attaches and checks it (driving
  the bounded self-correction loop, refusing to emit a refuted answer). Kernels:
  :class:`ArithmeticVerifier`, :class:`UnitVerifier`, :class:`TemporalVerifier`,
  :class:`ConstraintVerifier`, :class:`SchemaVerifier`, :class:`CitationVerifier`.
  The **statistical** kernels — :class:`TrendVerifier`, :class:`CorrelationVerifier`,
  :class:`IntervalVerifier`, :class:`ForecastVerifier` — certify the analytical
  claims a data answer makes (a trend, a correlation, a confidence interval, a
  forecast) by recomputing them from the cited cells, refuting a spurious one.
* **Runtime verification & shielding** — a :class:`BehaviorSpec` states a property
  over an agent's trajectory, a :class:`RuntimeMonitor` checks it step-by-step, and
  a :class:`Shield` blocks or repairs a violating action before it executes.
* **Verified tool use & synthesized programs** — a :class:`ToolContract` declares
  pre/post-conditions the runtime enforces, and :func:`synthesize` emits a
  :class:`SynthesizedProgram` whose properties are proven before it runs.

Optional SMT / CAS backends sit behind ``vincio[verify]`` (:mod:`vincio.verify.smt`).
"""

from .certificates import (
    Certificate,
    CertificateStatus,
    Check,
    CheckStatus,
    CompositeVerifier,
    ReasoningVerifier,
    VerificationContext,
    VerifiedAnswer,
    build_certificate,
    canonical_subject,
    derive_status,
)
from .kernels import (
    ArithmeticVerifier,
    CitationVerifier,
    Constraint,
    ConstraintVerifier,
    SchemaVerifier,
    TemporalVerifier,
    UnitVerifier,
    default_verifiers,
    safe_eval_arithmetic,
)
from .programs import (
    ProgramOp,
    ProgramProperty,
    ProgramSpec,
    SynthesizedProgram,
    ToolClause,
    ToolContract,
    synthesize,
)
from .runtime import (
    BehaviorEvent,
    BehaviorSpec,
    EventPattern,
    MonitorVerdict,
    RuntimeMonitor,
    Shield,
    ShieldDecision,
    ShieldMode,
    Violation,
)
from .statistical import (
    CellRef,
    CitedSeries,
    CorrelationClaim,
    CorrelationVerifier,
    ForecastClaim,
    ForecastVerifier,
    IntervalClaim,
    IntervalVerifier,
    StatisticalClaim,
    TrendClaim,
    TrendVerifier,
    statistical_verifiers,
)

__all__ = [
    # proof-carrying answers
    "Certificate",
    "CertificateStatus",
    "Check",
    "CheckStatus",
    "CompositeVerifier",
    "ReasoningVerifier",
    "VerificationContext",
    "VerifiedAnswer",
    "build_certificate",
    "canonical_subject",
    "derive_status",
    # kernels
    "ArithmeticVerifier",
    "UnitVerifier",
    "TemporalVerifier",
    "ConstraintVerifier",
    "SchemaVerifier",
    "CitationVerifier",
    "Constraint",
    "default_verifiers",
    "safe_eval_arithmetic",
    # statistical / forecasting & causal-inference kernels
    "CellRef",
    "CitedSeries",
    "StatisticalClaim",
    "TrendClaim",
    "CorrelationClaim",
    "IntervalClaim",
    "ForecastClaim",
    "TrendVerifier",
    "CorrelationVerifier",
    "IntervalVerifier",
    "ForecastVerifier",
    "statistical_verifiers",
    # runtime verification & shielding
    "BehaviorEvent",
    "BehaviorSpec",
    "EventPattern",
    "MonitorVerdict",
    "RuntimeMonitor",
    "Shield",
    "ShieldDecision",
    "ShieldMode",
    "Violation",
    # verified tool use & synthesized programs
    "ToolClause",
    "ToolContract",
    "ProgramOp",
    "ProgramProperty",
    "ProgramSpec",
    "SynthesizedProgram",
    "synthesize",
]
