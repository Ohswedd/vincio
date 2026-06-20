"""Vincio: the context engineering platform for AI applications.

Compiles prompts, memory, retrieval, tools, schemas, and policies into
optimized, validated, observable, provider-neutral context packets.
"""

from .agents import (
    AgentRole,
    Blackboard,
    CostAwareSelector,
    Crew,
    DistributedCheckpointer,
    HTNDomain,
    PlanRepairer,
    ResearchAgent,
    ResearchBudget,
    ResearchReport,
    ScheduleResult,
    Send,
    StateGraph,
    SubgraphScheduler,
    SubgraphTask,
    TimerService,
    WorkerPoolBackend,
    compose,
    sleep_for,
    sleep_until,
    wait_for_event,
)
from .assistant import ApprovalRecord, Assistant, AssistantTurn
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
from .evals.adaptive import AdaptiveSampler
from .evals.attribution import CausalAttributor, attribute_regression
from .evals.benchmarks import BenchmarkAdapter, load_benchmark
from .evals.datasets import Dataset, GoldenRegressionSuite
from .evals.ensemble import JudgeEnsemble
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
    ConsentLedger,
    ErasureProof,
    ErasureResult,
    FertilityTracker,
    FRIAGenerator,
    LawfulBasis,
    LineageRecord,
    ModelCard,
    ProvenanceManifest,
    Purpose,
    ResidencyPolicy,
    RiskTierClassifier,
    SystemCard,
    verify_erasure_proof,
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
from .optimize.rewards import RewardModel, VerifiableReward
from .optimize.routing import GuardedBanditRouter, ModelCascade, Router
from .optimize.self_improvement import (
    CanarySpec,
    DeployResult,
    SelfImprovementController,
    SelfImprovementPolicy,
)
from .optimize.trajectory_opt import (
    LearningResult,
    TrajectoryAdvantage,
    TrajectoryOptimizer,
)
from .output.routing import SchemaRouter
from .output.schemas import OutputContract, OutputSchema
from .packs import Pack, available_packs, load_pack
from .plugins import PluginInfo, discover_plugins, installed_plugins, load_plugins
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
from .realtime import RealtimeSession, VoiceAgent
from .registry import AgentDirectory, BundleRecord, CommunityRegistry
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

__version__ = "3.7.0"

__all__ = [
    "ContextApp",
    "RunHandle",
    "Assistant",
    "AssistantTurn",
    "ApprovalRecord",
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
    "SelfImprovementPolicy",
    "SelfImprovementController",
    "CanarySpec",
    "DeployResult",
    "GoldenRegressionSuite",
    "GuardedBanditRouter",
    "ReflectiveOptimizer",
    "TrainingSet",
    "BootstrapFinetune",
    "RewardModel",
    "VerifiableReward",
    "TrajectoryAdvantage",
    "TrajectoryOptimizer",
    "LearningResult",
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
    "PluginInfo",
    "discover_plugins",
    "installed_plugins",
    "load_plugins",
    "enable_rich_reprs",
    "BatchRunner",
    "CircuitBreaker",
    "HealthAwareFailover",
    "KeyPool",
    "ModelRegistry",
    "default_model_registry",
    "ModelCascade",
    # provider/model rotation & swap regression
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
    "VoiceAgent",
    "ModelCard",
    "SystemCard",
    "ComplianceReport",
    "ComplianceFramework",
    "AIBOM",
    "ResidencyPolicy",
    "LineageRecord",
    "ErasureResult",
    "ErasureProof",
    "verify_erasure_proof",
    "ConsentLedger",
    "Purpose",
    "LawfulBasis",
    "ProvenanceManifest",
    "FertilityTracker",
    "PoisoningDetector",
    # documents & images flow OUT
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
    # orchestrator & planner depth
    "HTNDomain",
    "PlanRepairer",
    "CostAwareSelector",
    "SubgraphScheduler",
    "SubgraphTask",
    "ScheduleResult",
    "TimerService",
    "sleep_until",
    "sleep_for",
    "wait_for_event",
    # scale out & train for real
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
    # environment eval, benchmarks, retrieval eval, agent fabric
    "Environment",
    "ToolEnvironment",
    "EnvironmentSimulator",
    "make_retail_environment",
    "BenchmarkAdapter",
    "load_benchmark",
    # evaluation & quality frontier
    "JudgeEnsemble",
    "CausalAttributor",
    "attribute_regression",
    "AdaptiveSampler",
    "RetrievalEvaluator",
    "RetrievalGoldenSet",
    "retrieval_regression",
    "AgentDirectory",
    "CommunityRegistry",
    "BundleRecord",
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
