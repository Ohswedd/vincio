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
from .feature_bench import (
    Contender,
    FeatureContest,
    FeatureMeasurement,
    FeatureRegistry,
    FeatureRun,
    FeatureSuite,
    FeatureSuiteRun,
    available_feature_contests,
    default_feature_registry,
    register_feature_contest,
)
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
from .track_report import render_feature_report, render_uplift_report
from .tracks import BenchmarkTrack
from .uplift import (
    UpliftBenchmark,
    UpliftRegistry,
    UpliftResult,
    UpliftRun,
    UpliftSuite,
    available_uplift_benchmarks,
    default_uplift_registry,
    register_uplift_benchmark,
)
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
    "BenchmarkTrack",
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
    # track 2 — uplift (model through Vincio vs direct)
    "UpliftSuite",
    "UpliftBenchmark",
    "UpliftResult",
    "UpliftRun",
    "UpliftRegistry",
    "default_uplift_registry",
    "register_uplift_benchmark",
    "available_uplift_benchmarks",
    # track 3 — feature (Vincio feature vs a competitor library)
    "FeatureSuite",
    "FeatureContest",
    "FeatureRun",
    "FeatureSuiteRun",
    "FeatureMeasurement",
    "Contender",
    "FeatureRegistry",
    "default_feature_registry",
    "register_feature_contest",
    "available_feature_contests",
    # reporting & leaderboard
    "SuiteReport",
    "Leaderboard",
    "LeaderboardRow",
    "RunStore",
    "render_uplift_report",
    "render_feature_report",
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
