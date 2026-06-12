"""Vincio agent engine: bounded executors, planners, handoffs, crews,
durable stateful graphs, declarative composition, and runtime backends."""

from .backends import LangGraphBackend, OpenAIAgentsBackend, RuntimeBackend
from .blackboard import Blackboard, BlackboardEntry
from .compose import Composable, NodeEvent, branch, compose, parallel
from .crew import AgentRole, Crew, CrewMemberReport, CrewResult, DelegationRecord
from .dag import StepDAG
from .executor import AgentExecutor
from .graph import (
    END,
    START,
    Checkpoint,
    Checkpointer,
    CompiledGraph,
    GraphEvent,
    GraphInterrupt,
    GraphResult,
    StateGraph,
    interrupt,
)
from .handoffs import HandoffRecord, HandoffRouter
from .planner import Planner, PlanningMode
from .state import AgentError, AgentState, AgentStep, AgentStepType, TerminationReason

__all__ = [
    "StepDAG",
    "AgentExecutor",
    "HandoffRecord",
    "HandoffRouter",
    "Planner",
    "PlanningMode",
    "AgentError",
    "AgentState",
    "AgentStep",
    "AgentStepType",
    "TerminationReason",
    # 0.6: multi-agent teams
    "Blackboard",
    "BlackboardEntry",
    "AgentRole",
    "Crew",
    "CrewResult",
    "CrewMemberReport",
    "DelegationRecord",
    # 0.6: durable stateful graphs
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
    # 0.6: declarative composition
    "compose",
    "parallel",
    "branch",
    "Composable",
    "NodeEvent",
    # 0.6: runtime backends
    "RuntimeBackend",
    "LangGraphBackend",
    "OpenAIAgentsBackend",
]
