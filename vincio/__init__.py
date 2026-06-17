"""Vincio: the context engineering platform for AI applications.

Compiles prompts, memory, retrieval, tools, schemas, and policies into
optimized, validated, observable, provider-neutral context packets.
"""

from .agents import (
    AgentRole,
    Blackboard,
    Crew,
    ResearchAgent,
    ResearchBudget,
    ResearchReport,
    StateGraph,
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
from .evals.datasets import Dataset, GoldenRegressionSuite
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
from .observability.finops import BudgetManager, CostBudget, CostLedger
from .optimize.controller import ContinuousImprovementController, ControllerDecision
from .optimize.distill import BootstrapFinetune, TrainingSet
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
    HealthAwareFailover,
    KeyPool,
    LifecycleWatcher,
    ModelRegistry,
    ShadowProvider,
    default_model_registry,
)
from .realtime import RealtimeSession
from .retrieval import MatryoshkaEmbedder, ShardedIndex
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

__version__ = "1.10.0"

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
    "API_VERSION",
    "StabilityLevel",
    "VincioDeprecationWarning",
    "VincioExperimentalWarning",
    "deprecated",
    "experimental",
    "stability_of",
    "__version__",
]
