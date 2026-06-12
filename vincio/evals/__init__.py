"""Vincio evaluation engine."""

from .datasets import Dataset, EvalCase, dataset_from_traces
from .experiments import ExperimentRun, ExperimentTracker, ab_test
from .judges import (
    DeterministicJudge,
    EmbeddingJudge,
    GEvalJudge,
    HybridJudge,
    Judge,
    ModelJudge,
)
from .metrics import METRICS, Metric, MetricResult, RunOutput, register_metric
from .redteam import (
    BUILTIN_PROBES,
    ProbeResult,
    RedTeamProbe,
    RedTeamReport,
    RedTeamSuite,
)
from .reports import CaseResult, EvalReport, GateSpec, evaluate_gates
from .runners import EvalRunner, EvalTarget
from .synthetic import SyntheticGenerator

__all__ = [
    "Dataset",
    "EvalCase",
    "dataset_from_traces",
    "ExperimentRun",
    "ExperimentTracker",
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
    "SyntheticGenerator",
]
