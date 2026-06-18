"""Stateful-environment evaluation harness.

The conversational :class:`~vincio.evals.simulator.Simulator` scores *what an
agent says*. An :class:`Environment` scores *what an agent does to a mutable
world*: the agent takes actions, the world's state changes, and a task-success
**oracle** verifies the **end state** — not turn-by-turn plausibility. This is
the shape the agentic leaderboards (τ-bench, WebArena) judge on, and it turns
agentic eval from post-hoc trajectory scoring into a closed-loop signal: the
verifiable success the oracle returns feeds the same optimizer/Pareto loop that
tunes prompts, routing, and budgets.

An environment exposes four operations — ``reset`` / ``step`` / ``observe`` /
``verify`` — and reference environments run **deterministically in-process** (no
network, no randomness), so a run is reproducible and CI-golden. The
:class:`EnvironmentSimulator` drives an agent *policy* through the world and
projects the interaction onto the same :class:`~vincio.evals.trajectory.Trajectory`
the trajectory metrics already score, with ``success`` set from the oracle.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..providers.base import run_sync
from .trajectory import Trajectory, TrajectoryStep

__all__ = [
    "EnvObservation",
    "EnvAction",
    "EnvToolResult",
    "EnvStepResult",
    "StateCheck",
    "TaskCheck",
    "TaskVerification",
    "EnvTask",
    "Environment",
    "ToolEnvironment",
    "EnvironmentResult",
    "EnvironmentSimulator",
    "AgentPolicy",
    "scripted_policy",
    "task_success",
    "make_retail_environment",
    "make_counter_environment",
]


# An agent under test: given the current observation, return the next action.
# May be sync or async. Returning an action with ``kind="finish"`` ends the run.
AgentPolicy = Callable[["EnvObservation"], "EnvAction | Awaitable[EnvAction]"]


class EnvObservation(BaseModel):
    """What the agent sees before choosing its next action."""

    text: str = ""
    state: dict[str, Any] = Field(default_factory=dict)  # public view of the world
    available_tools: list[str] = Field(default_factory=list)
    step: int = 0
    done: bool = False


class EnvAction(BaseModel):
    """An action the agent takes against the environment."""

    kind: Literal["tool", "message", "finish"] = "tool"
    tool: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    text: str = ""  # free-text message or final answer


class EnvToolResult(BaseModel):
    """The outcome a tool handler returns."""

    ok: bool = True
    text: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EnvStepResult(BaseModel):
    """The result of applying one action: a new observation plus a reward."""

    observation: EnvObservation
    reward: float = 0.0
    done: bool = False
    ok: bool = True
    error: str | None = None


class TaskCheck(BaseModel):
    """One evaluated end-state condition."""

    name: str
    passed: bool
    detail: str = ""


class TaskVerification(BaseModel):
    """The task-success oracle's verdict over the final world state."""

    passed: bool
    score: float = 0.0  # fraction of checks satisfied
    checks: list[TaskCheck] = Field(default_factory=list)
    reason: str = ""


# Comparison operators a declarative end-state check may use. Kept declarative
# (not arbitrary callables) so a task — and therefore a benchmark task set — is
# serializable and hashable for reproducible pinning.
CheckOp = Literal["eq", "ne", "in", "contains", "gte", "lte", "truthy", "falsy"]


def _resolve_path(state: Any, path: str) -> tuple[bool, Any]:
    """Walk a dotted path (``orders.O1.status``) into nested dicts/lists.

    Returns ``(found, value)``; list segments accept integer indices.
    """
    node: Any = state
    for part in path.split("."):
        if isinstance(node, dict):
            if part not in node:
                return False, None
            node = node[part]
        elif isinstance(node, (list, tuple)):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return False, None
        else:
            return False, None
    return True, node


class StateCheck(BaseModel):
    """A declarative end-state assertion over a path into the world state."""

    name: str
    path: str
    op: CheckOp = "eq"
    value: Any = None

    def evaluate(self, state: dict[str, Any]) -> TaskCheck:
        found, actual = _resolve_path(state, self.path)
        if not found and self.op not in ("falsy",):
            return TaskCheck(name=self.name, passed=False, detail=f"{self.path} not present")
        passed = self._apply(actual)
        detail = f"{self.path}={actual!r} {self.op} {self.value!r}" if self.op not in (
            "truthy", "falsy"
        ) else f"{self.path}={actual!r} {self.op}"
        return TaskCheck(name=self.name, passed=passed, detail=detail)

    def _apply(self, actual: Any) -> bool:
        op = self.op
        if op == "eq":
            return actual == self.value
        if op == "ne":
            return actual != self.value
        if op == "in":
            return actual in self.value if isinstance(self.value, (list, tuple, set, str)) else False
        if op == "contains":
            try:
                return self.value in actual
            except TypeError:
                return False
        if op == "gte":
            return isinstance(actual, (int, float)) and actual >= self.value
        if op == "lte":
            return isinstance(actual, (int, float)) and actual <= self.value
        if op == "truthy":
            return bool(actual)
        if op == "falsy":
            return not actual
        return False  # pragma: no cover - exhaustive above


class EnvTask(BaseModel):
    """A goal in an environment: the instruction plus its end-state checks."""

    id: str
    instruction: str
    checks: list[StateCheck] = Field(default_factory=list)
    max_steps: int = 12


@runtime_checkable
class Environment(Protocol):
    """The stateful-environment contract: ``reset`` / ``step`` / ``observe`` / ``verify``.

    Implementations own a mutable world. ``reset`` returns the world to a known
    initial state and yields the first observation; ``step`` applies an action
    and returns the consequence; ``observe`` returns the current observation
    without mutating; ``verify`` runs the task-success oracle over the *current*
    (typically final) state.
    """

    task: EnvTask

    def reset(self) -> EnvObservation: ...

    def step(self, action: EnvAction) -> EnvStepResult: ...

    def observe(self) -> EnvObservation: ...

    def verify(self) -> TaskVerification: ...


# A tool handler mutates the (mutable) world state in place and returns an
# outcome. Deterministic by construction — no I/O, no randomness.
EnvToolHandler = Callable[[dict[str, Any], dict[str, Any]], EnvToolResult]


class ToolEnvironment:
    """A deterministic, in-process environment whose world is a dict mutated by tools.

    Subclass-free: pass an ``initial_state``, a ``tools`` map, and a
    :class:`EnvTask`. The oracle (``verify``) runs the task's declarative
    :class:`StateCheck`\\ s against the live state, so success is **verifiable
    end-state**, not a plausibility judgement.
    """

    def __init__(
        self,
        *,
        name: str,
        initial_state: dict[str, Any],
        tools: dict[str, EnvToolHandler],
        task: EnvTask,
        instructions: str = "",
    ) -> None:
        self.name = name
        self._initial_state = initial_state
        self._tools = tools
        self.task = task
        self.instructions = instructions or task.instruction
        self.state: dict[str, Any] = {}
        self._step = 0
        self.reset()

    # -- Environment protocol -------------------------------------------------

    def reset(self) -> EnvObservation:
        # Deep copy via round-trip so reset is total and independent of prior runs.
        import copy

        self.state = copy.deepcopy(self._initial_state)
        self._step = 0
        return self.observe()

    def observe(self) -> EnvObservation:
        return EnvObservation(
            text=self.instructions if self._step == 0 else f"step {self._step}",
            state=self._public_state(),
            available_tools=sorted(self._tools),
            step=self._step,
            done=False,
        )

    def step(self, action: EnvAction) -> EnvStepResult:
        self._step += 1
        if action.kind != "tool" or not action.tool:
            # A message/finish action does not mutate the world.
            return EnvStepResult(observation=self.observe(), reward=0.0, done=action.kind == "finish")
        handler = self._tools.get(action.tool)
        if handler is None:
            return EnvStepResult(
                observation=self.observe(),
                reward=0.0,
                ok=False,
                error=f"unknown tool {action.tool!r}",
            )
        result = handler(self.state, dict(action.arguments))
        obs = self.observe()
        obs.text = result.text or obs.text
        return EnvStepResult(observation=obs, reward=1.0 if result.ok else 0.0, ok=result.ok, error=result.error)

    def verify(self) -> TaskVerification:
        checks = [c.evaluate(self.state) for c in self.task.checks]
        passed = bool(checks) and all(c.passed for c in checks)
        score = (sum(1 for c in checks if c.passed) / len(checks)) if checks else 0.0
        failed = [c.name for c in checks if not c.passed]
        reason = "all checks passed" if passed else f"failed checks: {failed}"
        return TaskVerification(passed=passed, score=round(score, 4), checks=checks, reason=reason)

    # -- helpers --------------------------------------------------------------

    def _public_state(self) -> dict[str, Any]:
        """The agent-visible projection of the world (override to redact)."""
        return self.state


class EnvironmentResult(BaseModel):
    """The outcome of driving a policy through an environment."""

    task_id: str
    trajectory: Trajectory
    verification: TaskVerification
    steps_taken: int = 0
    terminated: bool = True
    reward: float = 0.0

    @property
    def success(self) -> bool:
        """The task-success oracle: True iff the end-state checks all pass."""
        return self.verification.passed


class EnvironmentSimulator:
    """Drive an agent *policy* through an :class:`Environment` to a verified end state.

    The interaction is projected onto a :class:`Trajectory` whose ``success`` is
    the oracle's verdict, so the trajectory metrics in :mod:`vincio.evals.metrics`
    (``tool_call_accuracy``, ``goal_accuracy`` …) and the optimizer score the
    *verifiable end-state* directly.
    """

    def __init__(self, *, max_steps: int | None = None) -> None:
        self.max_steps = max_steps

    async def arun(
        self, env: Environment, policy: AgentPolicy, *, max_steps: int | None = None
    ) -> EnvironmentResult:
        limit = max_steps or self.max_steps or env.task.max_steps
        obs = env.reset()
        steps: list[TrajectoryStep] = []
        reward_total = 0.0
        tool_steps = 0
        final_text = ""
        for _ in range(limit):
            action = policy(obs)
            if hasattr(action, "__await__"):
                action = await action  # type: ignore[assignment]
            if action.kind == "finish":
                final_text = action.text
                break
            if action.kind == "message":
                final_text = action.text or final_text
                steps.append(TrajectoryStep(type="think", name="message", instruction=action.text))
                obs = env.observe()
                continue
            result = env.step(action)
            tool_steps += 1
            steps.append(
                TrajectoryStep(
                    type="tool",
                    name=action.tool,
                    tool_name=action.tool,
                    tool_arguments=dict(action.arguments),
                    status="done" if result.ok else "failed",
                    error=result.error,
                )
            )
            reward_total += result.reward
            obs = result.observation
            if result.done:
                break
        verification = env.verify()
        steps.append(
            TrajectoryStep(
                type="finalize",
                name="finalize",
                status="done" if verification.passed else "failed",
            )
        )
        trajectory = Trajectory(
            objective=env.task.instruction,
            steps=steps,
            final_answer=final_text or verification.reason,
            raw_text=final_text,
            terminated=True,
            termination_reason="objective_complete" if verification.passed else "incomplete",
            success=verification.passed,
            source="environment",
            usage={
                "steps": float(len(steps)),
                "tool_calls": float(tool_steps),
                "reward": round(reward_total, 4),
            },
        )
        return EnvironmentResult(
            task_id=env.task.id,
            trajectory=trajectory,
            verification=verification,
            steps_taken=tool_steps,
            terminated=True,
            reward=round(reward_total, 4),
        )

    def run(self, env: Environment, policy: AgentPolicy, *, max_steps: int | None = None) -> EnvironmentResult:
        """Synchronous wrapper; accepts a sync or async ``policy``."""

        async def _async_policy(obs: EnvObservation) -> EnvAction:
            action = policy(obs)
            if hasattr(action, "__await__"):
                return await action  # type: ignore[return-value]
            return action

        return run_sync(self.arun(env, _async_policy, max_steps=max_steps))


def scripted_policy(actions: list[EnvAction]) -> AgentPolicy:
    """A deterministic policy that replays a fixed action list, then finishes.

    Useful for recorded-fixture replay (benchmark adapters) and CI-golden tests.
    """
    queue = list(actions)

    def policy(_obs: EnvObservation) -> EnvAction:
        if queue:
            return queue.pop(0)
        return EnvAction(kind="finish")

    return policy


def task_success(result: EnvironmentResult) -> bool:
    """The task-success oracle as a free function (``result.verification.passed``)."""
    return result.verification.passed


# ---------------------------------------------------------------------------
# Reference environments (deterministic, in-process)
# ---------------------------------------------------------------------------


_RETAIL_SEED: dict[str, Any] = {
    "orders": {
        "O1001": {"item": "wireless-mouse", "status": "delivered", "refunded": False, "address": "1 Main St"},
        "O1002": {"item": "usb-c-cable", "status": "processing", "refunded": False, "address": "2 Oak Ave"},
    },
    "users": {"u-amir": {"name": "Amir", "orders": ["O1001", "O1002"]}},
}


def _retail_tools() -> dict[str, EnvToolHandler]:
    def get_order(state: dict[str, Any], args: dict[str, Any]) -> EnvToolResult:
        order = state["orders"].get(args.get("order_id"))
        if order is None:
            return EnvToolResult(ok=False, error="order not found")
        return EnvToolResult(text=f"order {args['order_id']}: {order['status']}", data=dict(order))

    def cancel_order(state: dict[str, Any], args: dict[str, Any]) -> EnvToolResult:
        order = state["orders"].get(args.get("order_id"))
        if order is None:
            return EnvToolResult(ok=False, error="order not found")
        if order["status"] == "cancelled":
            return EnvToolResult(text="already cancelled", data=dict(order))
        order["status"] = "cancelled"
        return EnvToolResult(text="order cancelled", data=dict(order))

    def refund_order(state: dict[str, Any], args: dict[str, Any]) -> EnvToolResult:
        order = state["orders"].get(args.get("order_id"))
        if order is None:
            return EnvToolResult(ok=False, error="order not found")
        # A refund is only valid once the order is cancelled or delivered — the
        # policy the agent must respect; refunding a processing order is wrong.
        if order["status"] not in ("cancelled", "delivered"):
            return EnvToolResult(ok=False, error="cannot refund an order that is not cancelled or delivered")
        order["refunded"] = True
        return EnvToolResult(text="refund issued", data=dict(order))

    def update_address(state: dict[str, Any], args: dict[str, Any]) -> EnvToolResult:
        order = state["orders"].get(args.get("order_id"))
        if order is None:
            return EnvToolResult(ok=False, error="order not found")
        order["address"] = str(args.get("address", ""))
        return EnvToolResult(text="address updated", data=dict(order))

    return {
        "get_order": get_order,
        "cancel_order": cancel_order,
        "refund_order": refund_order,
        "update_address": update_address,
    }


_RETAIL_TASKS: dict[str, EnvTask] = {
    "cancel_refund": EnvTask(
        id="cancel_refund",
        instruction=(
            "Customer Amir wants to cancel order O1002 and be refunded. Cancel the "
            "order, then issue the refund. Do not touch any other order."
        ),
        checks=[
            StateCheck(name="o1002_cancelled", path="orders.O1002.status", op="eq", value="cancelled"),
            StateCheck(name="o1002_refunded", path="orders.O1002.refunded", op="truthy"),
            StateCheck(name="o1001_untouched", path="orders.O1001.status", op="eq", value="delivered"),
        ],
    ),
    "update_shipping": EnvTask(
        id="update_shipping",
        instruction="Update the shipping address for order O1002 to '9 New Rd'.",
        checks=[
            StateCheck(name="address_updated", path="orders.O1002.address", op="eq", value="9 New Rd"),
            StateCheck(name="not_cancelled", path="orders.O1002.status", op="ne", value="cancelled"),
        ],
    ),
}


def make_retail_environment(task_id: str = "cancel_refund") -> ToolEnvironment:
    """A τ-bench-style retail world: orders mutated by tools, verified by end state.

    Tasks (``cancel_refund``, ``update_shipping``) require the agent to make the
    *correct* mutations under a policy (e.g. a refund is only valid on a
    cancelled/delivered order), then the oracle checks the resulting state.
    """
    if task_id not in _RETAIL_TASKS:
        raise ValueError(f"unknown retail task {task_id!r}; known: {sorted(_RETAIL_TASKS)}")
    return ToolEnvironment(
        name="retail",
        initial_state=_RETAIL_SEED,
        tools=_retail_tools(),
        task=_RETAIL_TASKS[task_id],
    )


def make_counter_environment(target: int = 3) -> ToolEnvironment:
    """A minimal world (a single counter) for harness determinism tests."""

    def increment(state: dict[str, Any], _args: dict[str, Any]) -> EnvToolResult:
        state["count"] = int(state.get("count", 0)) + 1
        return EnvToolResult(text=f"count={state['count']}", data={"count": state["count"]})

    def reset_counter(state: dict[str, Any], _args: dict[str, Any]) -> EnvToolResult:
        state["count"] = 0
        return EnvToolResult(text="count=0", data={"count": 0})

    return ToolEnvironment(
        name="counter",
        initial_state={"count": 0},
        tools={"increment": increment, "reset": reset_counter},
        task=EnvTask(
            id=f"count_to_{target}",
            instruction=f"Increment the counter until it reaches {target}.",
            checks=[StateCheck(name="reached_target", path="count", op="eq", value=target)],
        ),
    )
