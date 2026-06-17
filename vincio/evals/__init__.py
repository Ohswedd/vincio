"""Vincio evaluation engine."""

from .annotation import AnnotationItem, AnnotationQueue, cohens_kappa
from .datasets import Dataset, EvalCase, dataset_from_traces
from .drift import DriftMonitor, DriftReport
from .experiments import Experiment, ExperimentRun, ExperimentTracker, ab_test
from .guardrails import metric_guardrail
from .judges import (
    DeterministicJudge,
    EmbeddingJudge,
    GEvalJudge,
    HybridJudge,
    Judge,
    ModelJudge,
)
from .metrics import METRICS, Metric, MetricResult, RunOutput, register_metric
from .online import OnlineEvaluator
from .redteam import (
    BUILTIN_PROBES,
    ProbeResult,
    RedTeamProbe,
    RedTeamReport,
    RedTeamSuite,
)
from .replay import ReplayCase, ReplayResult, ReplayRunner
from .reports import CaseResult, EvalReport, GateSpec, evaluate_gates
from .runners import EvalRunner, EvalTarget
from .simulator import Persona, SimulatedConversation, Simulator
from .swap import (
    SwapGate,
    SwapRegressionReport,
    SwapVerdict,
    behavioral_shapes,
    model_swap_regression,
)
from .synthetic import SyntheticGenerator
from .trajectory import (
    TRAJECTORY_METRICS,
    Trajectory,
    TrajectoryStep,
    trajectory_from_agent_state,
    trajectory_from_crew_result,
    trajectory_from_trace,
)

__all__ = [
    "Dataset",
    "EvalCase",
    "dataset_from_traces",
    "ExperimentRun",
    "ExperimentTracker",
    "Experiment",
    "ab_test",
    "DeterministicJudge",
    "EmbeddingJudge",
    "GEvalJudge",
    "HybridJudge",
    "Judge",
    "ModelJudge",
    "METRICS",
    "Metric",
    "MetricResult",
    "RunOutput",
    "register_metric",
    "metric_guardrail",
    "BUILTIN_PROBES",
    "ProbeResult",
    "RedTeamProbe",
    "RedTeamReport",
    "RedTeamSuite",
    "CaseResult",
    "EvalReport",
    "GateSpec",
    "evaluate_gates",
    "EvalRunner",
    "EvalTarget",
    "ReplayRunner",
    "ReplayResult",
    "ReplayCase",
    # 1.8 — model-swap regression & the swap gate
    "SwapGate",
    "SwapVerdict",
    "SwapRegressionReport",
    "model_swap_regression",
    "behavioral_shapes",
    "SyntheticGenerator",
    # 1.2 — agentic evaluation & continuous quality
    "Trajectory",
    "TrajectoryStep",
    "trajectory_from_agent_state",
    "trajectory_from_crew_result",
    "trajectory_from_trace",
    "TRAJECTORY_METRICS",
    "Simulator",
    "Persona",
    "SimulatedConversation",
    "OnlineEvaluator",
    "DriftMonitor",
    "DriftReport",
    "AnnotationQueue",
    "AnnotationItem",
    "cohens_kappa",
]
