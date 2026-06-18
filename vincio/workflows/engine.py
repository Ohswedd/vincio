"""Deterministic workflow engine.

Features: DAG execution, retries with backoff, timeouts, compensation,
branching (``when`` conditions), parallel steps, human approval gates with
pause/resume (a gate with no ``approval_fn`` pauses the run; answer it with
``workflow.resume(result, approvals={...})``), edit-and-resume on the saved
context, typed inputs/outputs, trace spans.

Example::

    workflow = Workflow("contract_review")
    workflow.step("ingest", ingest_documents)
    workflow.step("retrieve", retrieve_clauses, depends_on=["ingest"])
    workflow.step("analyze", analyze_risk, depends_on=["retrieve"])
    workflow.step("validate", validate_report, depends_on=["analyze"])
    result = await workflow.arun({"files": ["msa.pdf"]})
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from ..core.concurrency import gather_bounded
from ..core.errors import WorkflowError, WorkflowStepError
from ..observability.traces import Tracer
from ..providers.base import run_sync

__all__ = ["StepResult", "WorkflowContext", "WorkflowResult", "Workflow"]

StepFn = Callable[..., Any]
ConditionFn = Callable[["WorkflowContext"], bool]
CompensationFn = Callable[["WorkflowContext"], Any]
ApprovalFn = Callable[[str, "WorkflowContext"], Awaitable[bool]]


class StepResult(BaseModel):
    name: str
    # pending | running | done | failed | skipped | compensated | waiting_approval
    status: str = "pending"
    output: Any = None
    error: str | None = None
    attempts: int = 0
    duration_ms: int = 0


class WorkflowContext(BaseModel):
    """State threaded through the workflow: initial input + step outputs."""

    input: Any = None
    results: dict[str, StepResult] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)

    def output_of(self, step: str) -> Any:
        result = self.results.get(step)
        return result.output if result else None

    def __getitem__(self, step: str) -> Any:
        return self.output_of(step)


class WorkflowResult(BaseModel):
    workflow: str
    status: str  # succeeded | failed | partial | paused
    context: WorkflowContext
    duration_ms: int = 0
    failed_steps: list[str] = Field(default_factory=list)
    compensated_steps: list[str] = Field(default_factory=list)
    pending_approvals: list[str] = Field(default_factory=list)

    @property
    def output(self) -> Any:
        """Output of the terminal step(s): single value or dict by name."""
        done = {name: r.output for name, r in self.context.results.items() if r.status == "done"}
        if not done:
            return None
        return done[next(reversed(done))] if len(done) == 1 else done


class _StepDef(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    fn: Any
    depends_on: list[str] = Field(default_factory=list)
    retries: int = 0
    retry_delay_s: float = 0.5
    timeout_s: float | None = None
    when: Any = None  # ConditionFn
    compensation: Any = None  # CompensationFn
    approval: bool = False
    map_over: Any = None  # str (prior step) | Callable[[WorkflowContext], list] for fan-out
    map_limit: int = 8


class Workflow:
    def __init__(
        self,
        name: str,
        *,
        tracer: Tracer | None = None,
        approval_fn: ApprovalFn | None = None,
    ) -> None:
        self.name = name
        self.tracer = tracer or Tracer()
        self.approval_fn = approval_fn
        self._steps: dict[str, _StepDef] = {}

    def step(
        self,
        name: str,
        fn: StepFn,
        *,
        depends_on: list[str] | None = None,
        retries: int = 0,
        retry_delay_s: float = 0.5,
        timeout_s: float | None = None,
        when: ConditionFn | None = None,
        compensation: CompensationFn | None = None,
        approval: bool = False,
    ) -> Workflow:
        """Register a step. ``fn`` receives the WorkflowContext (or, when its
        signature names match prior steps, those steps' outputs)."""
        if name in self._steps:
            raise WorkflowError(f"duplicate step {name!r}")
        for dep in depends_on or []:
            if dep not in self._steps:
                raise WorkflowError(f"step {name!r} depends on unknown step {dep!r}")
        self._steps[name] = _StepDef(
            name=name,
            fn=fn,
            depends_on=list(depends_on or []),
            retries=retries,
            retry_delay_s=retry_delay_s,
            timeout_s=timeout_s,
            when=when,
            compensation=compensation,
            approval=approval,
        )
        return self

    def map_step(
        self,
        name: str,
        fn: StepFn,
        *,
        over: str | Callable[[WorkflowContext], list[Any]],
        depends_on: list[str] | None = None,
        limit: int = 8,
        retries: int = 0,
        retry_delay_s: float = 0.5,
        timeout_s: float | None = None,
        when: ConditionFn | None = None,
        compensation: CompensationFn | None = None,
    ) -> Workflow:
        """Register a map-reduce fan-out step.

        ``over`` names a prior step whose output is a list (or a callable that
        returns the items from the context); ``fn`` is applied to **each item**
        concurrently with bounded parallelism (``limit``), and the step's output
        is the ordered list of per-item results. Pair it with a downstream step
        that reduces the list. Unlike a static parallel level, the fan-out width
        is data-dependent — discovered at run time, not declared at build time.
        """
        dep = list(depends_on or [])
        if isinstance(over, str) and over not in self._steps:
            raise WorkflowError(f"map step {name!r} maps over unknown step {over!r}")
        if isinstance(over, str) and over not in dep:
            dep.append(over)
        if name in self._steps:
            raise WorkflowError(f"duplicate step {name!r}")
        for d in dep:
            if d not in self._steps:
                raise WorkflowError(f"step {name!r} depends on unknown step {d!r}")
        self._steps[name] = _StepDef(
            name=name,
            fn=fn,
            depends_on=dep,
            retries=retries,
            retry_delay_s=retry_delay_s,
            timeout_s=timeout_s,
            when=when,
            compensation=compensation,
            map_over=over,
            map_limit=limit,
        )
        return self

    # -- invocation helpers -----------------------------------------------------------

    @staticmethod
    def _build_args(step: _StepDef, context: WorkflowContext) -> tuple[list[Any], dict[str, Any]]:
        signature = inspect.signature(step.fn)
        parameters = list(signature.parameters.values())
        if len(parameters) == 1 and parameters[0].name in ("context", "ctx"):
            return [context], {}
        kwargs: dict[str, Any] = {}
        for parameter in parameters:
            if parameter.name in ("context", "ctx"):
                kwargs[parameter.name] = context
            elif parameter.name == "input":
                kwargs[parameter.name] = context.input
            elif parameter.name in context.results:
                kwargs[parameter.name] = context.results[parameter.name].output
            elif parameter.default is not inspect.Parameter.empty:
                continue
            else:
                raise WorkflowStepError(
                    f"cannot bind parameter {parameter.name!r}; it matches no prior step, "
                    "'input', or 'context'",
                    step=step.name,
                )
        return [], kwargs

    @staticmethod
    async def _call(fn: Any, *args: Any) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(*args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args))

    def _map_items(self, step: _StepDef, context: WorkflowContext) -> list[Any]:
        over = step.map_over
        items = over(context) if callable(over) else context.output_of(over)
        return list(items or [])

    async def _invoke(self, step: _StepDef, context: WorkflowContext) -> Any:
        if step.map_over is not None:
            items = self._map_items(step, context)
            coroutine: Awaitable[Any] = gather_bounded(
                [self._call(step.fn, item) for item in items], limit=step.map_limit
            )
            if step.timeout_s is not None:
                return await asyncio.wait_for(coroutine, timeout=step.timeout_s)
            return await coroutine
        args, kwargs = self._build_args(step, context)
        if inspect.iscoroutinefunction(step.fn):
            coroutine = step.fn(*args, **kwargs)
        else:
            loop = asyncio.get_running_loop()
            coroutine = loop.run_in_executor(None, lambda: step.fn(*args, **kwargs))
        if step.timeout_s is not None:
            return await asyncio.wait_for(coroutine, timeout=step.timeout_s)
        return await coroutine

    async def _run_step(
        self,
        step: _StepDef,
        context: WorkflowContext,
        approvals: dict[str, bool] | None = None,
    ) -> StepResult:
        result = context.results[step.name]
        if step.when is not None and not step.when(context):
            result.status = "skipped"
            return result
        if step.approval:
            if self.approval_fn is not None:
                # A configured approver is always consulted; an approvals map
                # never bypasses it.
                approved = await self.approval_fn(step.name, context)
            elif approvals is not None and step.name in approvals:
                approved = approvals[step.name]
            else:
                # First-class interrupt: pause here; resume with an approvals map.
                result.status = "waiting_approval"
                return result
            if not approved:
                result.status = "failed"
                result.error = "approval denied"
                return result
        started = time.monotonic()
        result.status = "running"
        last_error: str | None = None
        for attempt in range(step.retries + 1):
            result.attempts = attempt + 1
            try:
                with self.tracer.span(step.name, type="workflow_step") as span:
                    span.set(workflow=self.name, attempt=attempt + 1)
                    output = await self._invoke(step, context)
                result.output = output
                result.status = "done"
                result.error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                result.error = last_error
                if attempt < step.retries:
                    await asyncio.sleep(step.retry_delay_s * (2**attempt))
        else:
            pass
        if result.status != "done":
            result.status = "failed"
            result.error = last_error
        result.duration_ms = int((time.monotonic() - started) * 1000)
        return result

    async def _compensate(self, context: WorkflowContext, completed: list[str]) -> list[str]:
        """Run compensation handlers in reverse completion order."""
        compensated: list[str] = []
        for name in reversed(completed):
            step = self._steps[name]
            if step.compensation is None:
                continue
            try:
                output = step.compensation(context)
                if inspect.isawaitable(output):
                    await output
                context.results[name].status = "compensated"
                compensated.append(name)
            except Exception:  # noqa: BLE001 - compensation is best-effort
                continue
        return compensated

    # -- execution ---------------------------------------------------------------------

    def _levels(self) -> list[list[_StepDef]]:
        in_degree = {name: len(s.depends_on) for name, s in self._steps.items()}
        dependents: dict[str, list[str]] = {name: [] for name in self._steps}
        for step in self._steps.values():
            for dep in step.depends_on:
                dependents[dep].append(step.name)
        current = [self._steps[n] for n, d in in_degree.items() if d == 0]
        levels: list[list[_StepDef]] = []
        seen: set[str] = set()
        while current:
            levels.append(current)
            seen.update(s.name for s in current)
            next_level: list[_StepDef] = []
            for step in current:
                for dependent in dependents[step.name]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_level.append(self._steps[dependent])
            current = next_level
        if len(seen) != len(self._steps):
            raise WorkflowError("workflow graph contains a cycle")
        return levels

    async def arun(
        self,
        input: Any = None,
        *,
        compensate_on_failure: bool = True,
        context: WorkflowContext | None = None,
        approvals: dict[str, bool] | None = None,
    ) -> WorkflowResult:
        """Run the workflow. Pass a prior result's ``context`` (or call
        :meth:`aresume`) to continue a paused/failed run: done steps keep
        their outputs and are not re-executed; ``approvals`` answers steps
        that paused at an approval gate."""
        if not self._steps:
            raise WorkflowError(f"workflow {self.name!r} has no steps")
        if approvals:
            gated = {n for n, s in self._steps.items() if s.approval}
            unknown = set(approvals) - gated
            if unknown:
                raise WorkflowError(
                    f"approvals reference steps without an approval gate: {sorted(unknown)}"
                )
        context = context or WorkflowContext(input=input)
        for name in self._steps:
            existing = context.results.get(name)
            # Only finished work survives a resume; anything else re-runs fresh
            # (a compensated/failed step must not leak its stale output).
            if existing is None or existing.status != "done":
                context.results[name] = StepResult(name=name)
        started = time.monotonic()
        # Steps completed in earlier segments still compensate (in order) on failure.
        completed: list[str] = [n for n in self._steps if context.results[n].status == "done"]
        failed: list[str] = []
        waiting: list[str] = []
        with self.tracer.trace(run_id=None, workflow=self.name):
            for level in self._levels():
                runnable: list[_StepDef] = []
                for step in level:
                    if context.results[step.name].status == "done":
                        continue  # resumed: keep prior output, don't re-run
                    upstream = [context.results[d] for d in step.depends_on]
                    if any(u.status in ("failed", "skipped") for u in upstream):
                        # Skipped-on-condition upstream is fine only if optional;
                        # failed upstream always skips dependents.
                        if any(u.status == "failed" for u in upstream):
                            context.results[step.name].status = "skipped"
                            context.results[step.name].error = "upstream failure"
                            continue
                        if any(u.status == "skipped" for u in upstream):
                            context.results[step.name].status = "skipped"
                            context.results[step.name].error = "upstream skipped"
                            continue
                    runnable.append(step)
                if runnable:
                    results = await asyncio.gather(
                        *(self._run_step(step, context, approvals) for step in runnable)
                    )
                    for step, result in zip(runnable, results, strict=False):
                        if result.status == "done":
                            completed.append(step.name)
                        elif result.status == "failed":
                            failed.append(step.name)
                        elif result.status == "waiting_approval":
                            waiting.append(step.name)
                if failed or waiting:
                    break
        compensated: list[str] = []
        if failed:
            # A failure is terminal even when a sibling paused at a gate in the
            # same level: never report a compensated run as resumable-paused.
            for name in waiting:
                context.results[name] = StepResult(name=name)
            waiting = []
            if compensate_on_failure:
                compensated = await self._compensate(context, completed)
        if waiting:
            status = "paused"
        else:
            status = "succeeded" if not failed else ("partial" if completed else "failed")
        return WorkflowResult(
            workflow=self.name,
            status=status,
            context=context,
            duration_ms=int((time.monotonic() - started) * 1000),
            failed_steps=failed,
            compensated_steps=compensated,
            pending_approvals=waiting,
        )

    async def aresume(
        self,
        previous: WorkflowResult,
        *,
        approvals: dict[str, bool] | None = None,
        compensate_on_failure: bool = True,
    ) -> WorkflowResult:
        """Resume a paused (or failed) run. Done steps are not re-executed.
        Edit-and-resume: mutate ``previous.context`` (input, ``data``, or a
        step's recorded output) before calling to steer the continuation."""
        return await self.arun(
            previous.context.input,
            context=previous.context,
            approvals=approvals,
            compensate_on_failure=compensate_on_failure,
        )

    def resume(self, previous: WorkflowResult, **kwargs: Any) -> WorkflowResult:
        return run_sync(self.aresume(previous, **kwargs))

    def run(self, input: Any = None, **kwargs: Any) -> WorkflowResult:
        return run_sync(self.arun(input, **kwargs))
