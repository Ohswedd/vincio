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
from .hierarchical import (
    HTNDomain,
    HTNMethod,
    HTNOperator,
    HTNPlanNode,
    dag_from_plan_node,
)
from .planner import Planner, PlanningMode
from .repair import PlanRepairer
from .research import ResearchAgent, ResearchBudget, ResearchReport
from .scheduling import ScheduleResult, SubgraphOutcome, SubgraphScheduler, SubgraphTask
from .selection import ActionCandidate, CostAwareSelector, SelectionDecision
from .state import (
    AgentError,
    AgentState,
    AgentStep,
    AgentStepType,
    PlanRepair,
    RepairAction,
    RepairTrigger,
    TerminationReason,
)
from .timers import (
    DurableTimer,
    PendingTimer,
    TimerService,
    deliver_event,
    due_timers,
    pending_timers,
    resume_due_timers,
    sleep_for,
    sleep_until,
    wait_for_event,
)

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
    # hierarchical (HTN) planning
    "HTNDomain",
    "HTNMethod",
    "HTNOperator",
    "HTNPlanNode",
    "dag_from_plan_node",
    # plan repair & cost-aware selection
    "PlanRepairer",
    "PlanRepair",
    "RepairAction",
    "RepairTrigger",
    "CostAwareSelector",
    "SelectionDecision",
    "ActionCandidate",
    # parallel sub-graph scheduling
    "SubgraphScheduler",
    "SubgraphTask",
    "SubgraphOutcome",
    "ScheduleResult",
    # durable timers & scheduled steps
    "DurableTimer",
    "PendingTimer",
    "TimerService",
    "sleep_until",
    "sleep_for",
    "wait_for_event",
    "pending_timers",
    "due_timers",
    "resume_due_timers",
    "deliver_event",
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
