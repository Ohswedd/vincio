"""Vincio: the context engineering platform for AI applications.

Compiles prompts, memory, retrieval, tools, schemas, and policies into
optimized, validated, observable, provider-neutral context packets.
"""

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
    Objective,
    PolicySet,
    RunConfig,
    RunResult,
    TaskType,
    UserInput,
)
from .evals.datasets import Dataset
from .output.schemas import OutputContract, OutputSchema
from .prompts.templates import PromptSpec
from .workflows.engine import Workflow

__version__ = "0.1.0"

__all__ = [
    "ContextApp",
    "VincioConfig",
    "load_config",
    "VincioError",
    "Budget",
    "Constraint",
    "EvidenceItem",
    "Example",
    "Instruction",
    "MemoryItem",
    "Objective",
    "PolicySet",
    "RunConfig",
    "RunResult",
    "TaskType",
    "UserInput",
    "Dataset",
    "OutputContract",
    "OutputSchema",
    "PromptSpec",
    "Workflow",
    "__version__",
]
