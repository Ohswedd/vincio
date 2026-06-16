"""Vincio: the context engineering platform for AI applications.

Compiles prompts, memory, retrieval, tools, schemas, and policies into
optimized, validated, observable, provider-neutral context packets.
"""

from .agents import AgentRole, Blackboard, Crew, StateGraph, compose
from .context.llmlingua import LLMLinguaCompressor
from .core.app import ContextApp
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
from .evals.datasets import Dataset
from .governance import (
    AIBOM,
    ComplianceFramework,
    ComplianceReport,
    ErasureResult,
    FertilityTracker,
    LineageRecord,
    ModelCard,
    ProvenanceManifest,
    ResidencyPolicy,
    SystemCard,
)
from .memory.engine import MemoryEngine, ScopedMemory
from .notebook import enable_rich_reprs
from .observability.finops import BudgetManager, CostBudget, CostLedger
from .optimize.distill import BootstrapFinetune, TrainingSet
from .optimize.judge_calibration import JudgeCalibrator
from .optimize.loop import ImprovementLoop, LoopResult
from .optimize.reflective import ReflectiveOptimizer
from .optimize.routing import ModelCascade
from .output.routing import SchemaRouter
from .output.schemas import OutputContract, OutputSchema
from .packs import Pack, available_packs, load_pack
from .prompts.signatures import InputField, OutputField, Predict, Signature, signature
from .prompts.templates import PromptSpec
from .providers import BatchRunner, CircuitBreaker, HealthAwareFailover, KeyPool
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

__version__ = "1.6.1"

__all__ = [
    "ContextApp",
    "AgentRole",
    "Blackboard",
    "Crew",
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
    "ModelCascade",
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
    "API_VERSION",
    "StabilityLevel",
    "VincioDeprecationWarning",
    "VincioExperimentalWarning",
    "deprecated",
    "experimental",
    "stability_of",
    "__version__",
]
