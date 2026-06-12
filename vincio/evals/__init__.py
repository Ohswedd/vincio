"""Vincio evaluation engine."""

from .datasets import Dataset, EvalCase
from .judges import DeterministicJudge, EmbeddingJudge, HybridJudge, Judge, ModelJudge
from .metrics import METRICS, Metric, MetricResult, RunOutput, register_metric
from .reports import CaseResult, EvalReport, GateSpec, evaluate_gates
from .runners import EvalRunner, EvalTarget

__all__ = [
    "Dataset",
    "EvalCase",
    "DeterministicJudge",
    "EmbeddingJudge",
    "HybridJudge",
    "Judge",
    "ModelJudge",
    "METRICS",
    "Metric",
    "MetricResult",
    "RunOutput",
    "register_metric",
    "CaseResult",
    "EvalReport",
    "GateSpec",
    "evaluate_gates",
    "EvalRunner",
    "EvalTarget",
]
