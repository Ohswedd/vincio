"""Vincio agent engine: bounded executors, planners, handoffs, crews,
durable stateful graphs, declarative composition, and runtime backends."""

from .backends import (
    LangGraphBackend,
    OpenAIAgentsBackend,
    RayBackend,
    RuntimeBackend,
    TemporalBackend,
    WorkerPoolBackend,
)
from .blackboard import Blackboard, BlackboardEntry
from .compose import Composable, NodeEvent, branch, compose, parallel
from .crew import AgentRole, Crew, CrewEvent, CrewMemberReport, CrewResult, DelegationRecord
from .dag import StepDAG
from .distributed import (
    DistributedCheckpointer,
    GraphCoordinator,
    InMemoryGraphCoordinator,
    RedisGraphCoordinator,
)
from .executor import AgentEvent, AgentExecutor
from .graph import (
    END,
    START,
    Checkpoint,
    Checkpointer,
    CompiledGraph,
    GraphEvent,
    GraphInterrupt,
    GraphResult,
    Send,
    StateGraph,
    interrupt,
)
from .handoffs import HandoffRecord, HandoffRouter
from .planner import Planner, PlanningMode
from .research import ResearchAgent, ResearchBudget, ResearchReport
from .state import AgentError, AgentState, AgentStep, AgentStepType, TerminationReason

__all__ = [
    "StepDAG",
    "AgentExecutor",
    "AgentEvent",
    "HandoffRecord",
    "HandoffRouter",
    "Planner",
    "PlanningMode",
    "ResearchAgent",
    "ResearchBudget",
    "ResearchReport",
    "AgentError",
    "AgentState",
    "AgentStep",
    "AgentStepType",
    "TerminationReason",
    # multi-agent teams
    "Blackboard",
    "BlackboardEntry",
    "AgentRole",
    "Crew",
    "CrewResult",
    "CrewMemberReport",
    "CrewEvent",
    "DelegationRecord",
    # durable stateful graphs
    "StateGraph",
    "CompiledGraph",
    "Checkpoint",
    "Checkpointer",
    "GraphEvent",
    "GraphInterrupt",
    "GraphResult",
    "START",
    "END",
    "interrupt",
    # declarative composition
    "compose",
    "parallel",
    "branch",
    "Composable",
    "NodeEvent",
    # runtime backends
    "RuntimeBackend",
    "LangGraphBackend",
    "OpenAIAgentsBackend",
    # distributed durable execution
    "Send",
    "WorkerPoolBackend",
    "RayBackend",
    "TemporalBackend",
    "DistributedCheckpointer",
    "GraphCoordinator",
    "InMemoryGraphCoordinator",
    "RedisGraphCoordinator",
]
