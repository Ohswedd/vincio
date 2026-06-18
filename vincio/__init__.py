"""Vincio: the context engineering platform for AI applications.

Compiles prompts, memory, retrieval, tools, schemas, and policies into
optimized, validated, observable, provider-neutral context packets.
"""

from .agents import (
    AgentRole,
    Blackboard,
    Crew,
    DistributedCheckpointer,
    ResearchAgent,
    ResearchBudget,
    ResearchReport,
    Send,
    StateGraph,
    WorkerPoolBackend,
    compose,
)
from .context.llmlingua import LLMLinguaCompressor
from .core.app import ContextApp, RunHandle
from .core.config import VincioConfig, load_config
from .core.errors import VincioError
from .core.types import (
    Budget,
    Constraint,
    EvidenceItem,
    Example,
    Instruction,
    MemoryItem,
    MemoryScope,
    MemoryType,
    Objective,
    PolicySet,
    RunConfig,
    RunResult,
    RunStreamEvent,
    TaskType,
    UserInput,
)
from .evals.benchmarks import BenchmarkAdapter, load_benchmark
from .evals.datasets import Dataset, GoldenRegressionSuite
from .evals.environment import (
    Environment,
    EnvironmentSimulator,
    ToolEnvironment,
    make_retail_environment,
)
from .evals.retrieval_eval import RetrievalEvaluator, RetrievalGoldenSet, retrieval_regression
from .evals.swap import SwapGate, SwapVerdict, model_swap_regression
from .generation import (
    CitationContract,
    CitedReportBuilder,
    DocumentArtifact,
    DocumentBuilder,
    DocumentContract,
    ImageGenRequest,
    ImageProvider,
    MockImageProvider,
    MockSpeechProvider,
    SpeechProvider,
    SpeechRequest,
    generate_redline,
)
from .governance import (
    AIBOM,
    AnnexIVBuilder,
    ComplianceFramework,
    ComplianceReport,
    ErasureResult,
    FertilityTracker,
    FRIAGenerator,
    LineageRecord,
    ModelCard,
    ProvenanceManifest,
    ResidencyPolicy,
    RiskTierClassifier,
    SystemCard,
)
from .memory.engine import MemoryEngine, ScopedMemory
from .notebook import enable_rich_reprs
from .observability.exporters import AlertSink, PrometheusExporter
from .observability.finops import AlertManager, AlertRule, BudgetManager, CostBudget, CostLedger
from .observability.redaction import ContentCapturePolicy
from .observability.store import IndexedTraceStore
from .observability.viewer import serve_viewer
from .optimize.controller import ContinuousImprovementController, ControllerDecision
from .optimize.distill import BootstrapFinetune, TrainingSet, provider_trainer
from .optimize.judge_calibration import JudgeCalibrator
from .optimize.loop import ExperimentProposer, ImprovementLoop, LoopResult
from .optimize.reflective import ReflectiveOptimizer
from .optimize.routing import GuardedBanditRouter, ModelCascade, Router
from .output.routing import SchemaRouter
from .output.schemas import OutputContract, OutputSchema
from .packs import Pack, available_packs, load_pack
from .prompts.signatures import InputField, OutputField, Predict, Signature, signature
from .prompts.templates import PromptSpec
from .providers import (
    BatchRunner,
    CanaryRouter,
    CircuitBreaker,
    GGUFProvider,
    HealthAwareFailover,
    KeyPool,
    LifecycleWatcher,
    ModelRegistry,
    OpenAIFineTuneBackend,
    ShadowProvider,
    default_model_registry,
    make_finetune_backend,
)
from .realtime import RealtimeSession
from .registry import AgentDirectory
from .retrieval import FastEmbedEmbedder, MatryoshkaEmbedder, ShardedIndex, TwoStageIndex
from .security.access import AllowListGate
from .security.poisoning import PoisoningDetector
from .security.rails import Rail
from .stability import (
    API_VERSION,
    StabilityLevel,
    VincioDeprecationWarning,
    VincioExperimentalWarning,
    deprecated,
    experimental,
    stability_of,
)
from .workflows.engine import Workflow

__version__ = "2.2.0"

__all__ = [
    "ContextApp",
    "RunHandle",
    "AgentRole",
    "Blackboard",
    "Crew",
    "ResearchAgent",
    "ResearchBudget",
    "ResearchReport",
    "StateGraph",
    "compose",
    "VincioConfig",
    "load_config",
    "VincioError",
    "Budget",
    "Constraint",
    "EvidenceItem",
    "Example",
    "Instruction",
    "MemoryItem",
    "MemoryScope",
    "MemoryType",
    "MemoryEngine",
    "ScopedMemory",
    "Objective",
    "PolicySet",
    "RunConfig",
    "RunResult",
    "RunStreamEvent",
    "TaskType",
    "UserInput",
    "Dataset",
    "ImprovementLoop",
    "LoopResult",
    "ContinuousImprovementController",
    "ControllerDecision",
    "ExperimentProposer",
    "GoldenRegressionSuite",
    "GuardedBanditRouter",
    "ReflectiveOptimizer",
    "TrainingSet",
    "BootstrapFinetune",
    "LLMLinguaCompressor",
    "JudgeCalibrator",
    "OutputContract",
    "OutputSchema",
    "SchemaRouter",
    "PromptSpec",
    "Signature",
    "InputField",
    "OutputField",
    "signature",
    "Predict",
    "Rail",
    "Workflow",
    "Pack",
    "load_pack",
    "available_packs",
    "enable_rich_reprs",
    "BatchRunner",
    "CircuitBreaker",
    "HealthAwareFailover",
    "KeyPool",
    "ModelRegistry",
    "default_model_registry",
    "ModelCascade",
    # 1.8 — provider/model rotation & swap regression
    "Router",
    "SwapGate",
    "SwapVerdict",
    "model_swap_regression",
    "ShadowProvider",
    "CanaryRouter",
    "LifecycleWatcher",
    "CostLedger",
    "CostBudget",
    "BudgetManager",
    "ShardedIndex",
    "MatryoshkaEmbedder",
    "RealtimeSession",
    "ModelCard",
    "SystemCard",
    "ComplianceReport",
    "ComplianceFramework",
    "AIBOM",
    "ResidencyPolicy",
    "LineageRecord",
    "ErasureResult",
    "ProvenanceManifest",
    "FertilityTracker",
    "PoisoningDetector",
    # 1.9 — documents & images flow OUT
    "DocumentBuilder",
    "DocumentContract",
    "DocumentArtifact",
    "CitedReportBuilder",
    "CitationContract",
    "generate_redline",
    "ImageProvider",
    "ImageGenRequest",
    "MockImageProvider",
    "SpeechProvider",
    "SpeechRequest",
    "MockSpeechProvider",
    "RiskTierClassifier",
    "AnnexIVBuilder",
    "FRIAGenerator",
    # 2.1 — scale out & train for real
    "WorkerPoolBackend",
    "DistributedCheckpointer",
    "Send",
    "provider_trainer",
    "OpenAIFineTuneBackend",
    "make_finetune_backend",
    "GGUFProvider",
    "IndexedTraceStore",
    "AlertManager",
    "AlertRule",
    "AlertSink",
    "PrometheusExporter",
    "ContentCapturePolicy",
    "serve_viewer",
    "TwoStageIndex",
    "FastEmbedEmbedder",
    # 2.2 — environment eval, benchmarks, retrieval eval, agent fabric
    "Environment",
    "ToolEnvironment",
    "EnvironmentSimulator",
    "make_retail_environment",
    "BenchmarkAdapter",
    "load_benchmark",
    "RetrievalEvaluator",
    "RetrievalGoldenSet",
    "retrieval_regression",
    "AgentDirectory",
    "AllowListGate",
    "API_VERSION",
    "StabilityLevel",
    "VincioDeprecationWarning",
    "VincioExperimentalWarning",
    "deprecated",
    "experimental",
    "stability_of",
    "__version__",
]
