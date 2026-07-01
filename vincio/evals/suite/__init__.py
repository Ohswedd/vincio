"""The open evaluation plane — one pluggable harness for the standard public
model benchmarks, grouped by niche, scored by reusable metrics, reported the same
way for every model and every model version, with a provenance tier on every
number. In-process and offline-reproducible; never a hosted leaderboard.
"""

from .adapters import (
    ARCAdapter,
    CEvalAdapter,
    CMMLUAdapter,
    GPQAAdapter,
    GSM8KAdapter,
    HellaSwagAdapter,
    HumanEvalAdapter,
    IFEvalAdapter,
    MATHAdapter,
    MBPPAdapter,
    MMLUAdapter,
    PromptInjectionAdapter,
    RAGFaithfulnessAdapter,
    RULERAdapter,
    TruthfulQAAdapter,
)
from .datasets import BenchmarkDataset
from .engine import BenchmarkSuite
from .metrics import bleu, pass_at_k, rouge_l
from .registry import (
    NICHES,
    BenchmarkRegistry,
    BenchmarkSpec,
    available_suite_benchmarks,
    default_benchmark_registry,
    register_benchmark,
)
from .report import Leaderboard, LeaderboardRow, SuiteReport
from .results import BenchmarkRun, ItemResult, SuiteRun
from .store import RunStore
from .tiers import ProvenanceTier, resolve_tier
from .viz import (
    SuiteChart,
    confusion_matrix_chart,
    heatmap_chart,
    leaderboard_chart,
    radar_chart,
    trend_chart,
)

__all__ = [
    "ProvenanceTier",
    "resolve_tier",
    "BenchmarkSpec",
    "BenchmarkRegistry",
    "register_benchmark",
    "default_benchmark_registry",
    "available_suite_benchmarks",
    "NICHES",
    "BenchmarkDataset",
    "BenchmarkSuite",
    "SuiteRun",
    "BenchmarkRun",
    "ItemResult",
    "pass_at_k",
    "bleu",
    "rouge_l",
    # reporting & leaderboard
    "SuiteReport",
    "Leaderboard",
    "LeaderboardRow",
    "RunStore",
    # visualization
    "SuiteChart",
    "leaderboard_chart",
    "radar_chart",
    "heatmap_chart",
    "confusion_matrix_chart",
    "trend_chart",
    # niche adapters for the standard public benchmarks
    "MMLUAdapter",
    "GPQAAdapter",
    "ARCAdapter",
    "HellaSwagAdapter",
    "CEvalAdapter",
    "CMMLUAdapter",
    "TruthfulQAAdapter",
    "GSM8KAdapter",
    "MATHAdapter",
    "HumanEvalAdapter",
    "MBPPAdapter",
    "IFEvalAdapter",
    "PromptInjectionAdapter",
    "RAGFaithfulnessAdapter",
    "RULERAdapter",
]
