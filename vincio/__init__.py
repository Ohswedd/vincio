"""Vincio: the context engineering platform for AI applications.

Compiles prompts, memory, retrieval, tools, schemas, and policies into
optimized, validated, observable, provider-neutral context packets.
"""

from .agents import AgentRole, Blackboard, Crew, StateGraph, compose
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
from .memory.engine import MemoryEngine, ScopedMemory
from .optimize.loop import ImprovementLoop, LoopResult
from .output.routing import SchemaRouter
from .output.schemas import OutputContract, OutputSchema
from .prompts.signatures import InputField, OutputField, Predict, Signature, signature
from .prompts.templates import PromptSpec
from .security.rails import Rail
from .workflows.engine import Workflow

__version__ = "0.8.0"

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
    "__version__",
]
