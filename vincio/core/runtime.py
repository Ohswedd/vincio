"""Vincio runtime core: the 17-step run flow.

1. receive input        7. retrieve evidence      13. parse + validate output
2. normalize            8. score + compress       14. evaluate output
3. detect objective     9. plan tools             15. write trace
4. resolve policy      10. compile Context IR     16. write memory updates
5. memory candidates   11. render model request   17. return result

The flow is async-first (0.2): memory recall, retrieval, and file ingestion
run concurrently; tool calls within a model round fan out under a bounded
worker pool; cancellation propagates through every stage; and
``execute_stream`` runs the same pipeline with token streaming, incremental
partial-JSON parsing, and per-event trace spans.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..context.compiler import CompiledContext
from ..context.ir import OutputContractRef
from ..core.errors import VincioError
from ..core.types import (
    Budget,
    EvidenceItem,
    Message,
    ModelRequest,
    ModelResponse,
    RunConfig,
    RunResult,
    RunStatus,
    RunStreamEvent,
    TokenUsage,
    ToolCall,
    ToolResult,
    UserInput,
)
from ..core.utils import new_id, utcnow
from ..evals.datasets import EvalCase
from ..evals.metrics import METRICS, RunOutput
from ..output.constrained import DecodingMode, negotiate_decoding, to_strict_json_schema
from ..output.correction import SelfCorrector
from ..output.schemas import OutputContract
from ..output.streaming import StreamingValidator
from ..output.validators import OutputValidator
from ..providers.cache_strategy import cache_hit_rate
from ..retrieval.chunking import extract_entities
from .concurrency import gather_bounded

if TYPE_CHECKING:  # pragma: no cover
    from ..prompts.compiler import CompiledPrompt
    from ..providers.base import ModelProvider
    from .app import ContextApp

__all__ = ["VincioRuntime"]


@dataclass
class _PreparedRun:
    """Everything steps 1-11 produce, shared by both execution paths."""

    routed: Any
    compiled_context: CompiledContext
    compiled_prompt: CompiledPrompt
    provider: ModelProvider
    model: str
    tool_specs: list[Any]
    messages: list[Message]
    contract: OutputContract
    decoding: DecodingMode = DecodingMode.NONE
    request_kwargs: dict[str, Any] = field(default_factory=dict)
    cache_info: dict[str, Any] = field(default_factory=dict)
    cascade_active: bool = False  # this run routes through app.cascade (cheap→strong)


class VincioRuntime:
    def __init__(self, app: ContextApp) -> None:
        self.app = app

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------

    async def execute(self, user_input: UserInput, run_config: RunConfig | None = None) -> RunResult:
        app = self.app
        run_config = run_config or RunConfig()
        run_id = new_id("run")
        started = time.monotonic()
        budget = run_config.budget or app.budget
        policies = run_config.policies or app.policies
        result = RunResult(run_id=run_id, status=RunStatus.RUNNING)
        cancelled = False

        with app.tracer.trace(
            run_id=run_id,
            session_id=user_input.session_id,
            user_id=user_input.user_id,
            tenant_id=user_input.tenant_id,
            input=(user_input.text or "")[:500],
        ) as trace:
            result.trace_id = trace.id
            try:
                # Budget-enforced deadline; cancellation propagates into
                # every concurrent subtask (retrieval, tools, model call).
                async with asyncio.timeout(budget.max_latency_ms / 1000):
                    await self._execute_inner(
                        user_input, run_config, budget, policies, result, run_id
                    )
            except VincioError as exc:
                result.status = RunStatus.FAILED
                result.error = exc.message
            except TimeoutError:
                result.status = RunStatus.FAILED
                result.error = f"run exceeded max_latency_ms budget ({budget.max_latency_ms} ms)"
            except asyncio.CancelledError:
                result.status = RunStatus.CANCELLED
                result.error = "run cancelled"
                cancelled = True
                trace.attributes["cancelled"] = True
            result.latency_ms = int((time.monotonic() - started) * 1000)
            trace.attributes["output"] = (result.raw_text or "")[:500]
            if app.config.observability.training_capture:
                self._capture_training_artifacts(trace, result, user_input)
            for metric_name, score in (result.eval_scores or {}).items():
                trace.add_score(metric_name, score)

        # 15. persist run + packet (trace export happens in the tracer).
        self._persist_run(result, run_id, user_input)
        app.events.emit(
            "run.completed", {"run_id": run_id, "status": result.status.value}, trace_id=result.trace_id
        )
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def execute_stream(
        self, user_input: UserInput, run_config: RunConfig | None = None
    ) -> AsyncIterator[RunStreamEvent]:
        """The same 17-step flow with end-to-end streaming.

        Yields :class:`RunStreamEvent` items: pipeline ``stage`` markers,
        ``text_delta`` chunks as the provider streams, incremental
        ``partial_output`` parses for structured output, tool activity, and
        a terminal ``done`` event carrying the full :class:`RunResult`.
        """
        app = self.app
        run_config = run_config or RunConfig()
        run_id = new_id("run")
        started = time.monotonic()
        budget = run_config.budget or app.budget
        policies = run_config.policies or app.policies
        result = RunResult(run_id=run_id, status=RunStatus.RUNNING)

        with app.tracer.trace(
            run_id=run_id,
            session_id=user_input.session_id,
            user_id=user_input.user_id,
            tenant_id=user_input.tenant_id,
            input=(user_input.text or "")[:500],
            stream=True,
        ) as trace:
            result.trace_id = trace.id
            try:
                prepared = await self._prepare(
                    user_input, run_config, budget, policies, result, run_id
                )
                if prepared is not None:
                    yield RunStreamEvent(
                        type="stage",
                        stage="context_compiled",
                        data={
                            "packet_id": result.context_packet_id,
                            "token_count": prepared.compiled_context.token_count,
                            "evidence": len(prepared.compiled_context.ir.evidence),
                        },
                    )
                    response: ModelResponse | None = None
                    async for item in self._model_tool_loop_stream(
                        prepared, budget, result, user_input, run_id
                    ):
                        if isinstance(item, RunStreamEvent):
                            yield item
                        else:
                            response = item
                    assert response is not None
                    await self._finalize(
                        prepared, response, result, run_id, user_input, policies
                    )
                    yield RunStreamEvent(
                        type="stage",
                        stage="validated",
                        data={"valid": result.validation.get("valid")} if result.validation else {},
                    )
            except VincioError as exc:
                result.status = RunStatus.FAILED
                result.error = exc.message
                yield RunStreamEvent(type="error", error=result.error)
            result.latency_ms = int((time.monotonic() - started) * 1000)

        self._persist_run(result, run_id, user_input)
        app.events.emit(
            "run.completed", {"run_id": run_id, "status": result.status.value}, trace_id=result.trace_id
        )
        yield RunStreamEvent(type="done", result=result, usage=result.usage)

    async def execute_batch(
        self,
        inputs: list[str | UserInput],
        *,
        run_config: RunConfig | None = None,
        backend: Any | None = None,
        discount: float = 0.5,
        timeout_s: float | None = None,
    ) -> list[RunResult]:
        """Run a set of inputs through a provider Batch API (1.3).

        Prepares every input (steps 1-11) under its own trace, submits the
        model requests as one batch (at the discounted rate), then validates and
        finalizes each result by custom id. Latency-tolerant work — evals, bulk
        extraction, synthetic data — at ~half the cost, same RunResult contract.
        """
        from ..providers.batch import BatchRequest, BatchRunner

        app = self.app
        run_config = run_config or RunConfig()
        budget = run_config.budget or app.budget
        policies = run_config.policies or app.policies

        prepared_runs: list[tuple[Any, RunResult, str, UserInput, Any]] = []
        batch_requests: list[BatchRequest] = []
        for user_input in inputs:
            if isinstance(user_input, str):
                user_input = UserInput(text=user_input)
            else:
                user_input = user_input.model_copy(deep=True)
            run_id = new_id("run")
            result = RunResult(run_id=run_id, status=RunStatus.RUNNING)
            trace = app.tracer.new_trace(
                run_id=run_id,
                session_id=user_input.session_id,
                user_id=user_input.user_id,
                tenant_id=user_input.tenant_id,
                input=(user_input.text or "")[:500],
                batch=True,
            )
            result.trace_id = trace.id
            with app.tracer.bind(trace):
                # Batch runs are the queue-to-batch target, so they are exempt
                # from the interactive cost cap (they pay the discounted rate).
                prepared = await self._prepare(
                    user_input, run_config, budget, policies, result, run_id, enforce_budget=False
                )
            if prepared is None:
                app.tracer.export(trace)
                self._persist_run(result, run_id, user_input)
                prepared_runs.append((None, result, run_id, user_input, trace))
                continue
            request = ModelRequest(
                model=prepared.model,
                messages=prepared.messages,
                tools=prepared.tool_specs,
                **prepared.request_kwargs,
            )
            batch_requests.append(BatchRequest(custom_id=run_id, request=request))
            prepared_runs.append((prepared, result, run_id, user_input, trace))

        by_id: dict[str, Any] = {}
        if batch_requests:
            runner = BatchRunner(
                backend or app.resolve_provider(run_config),
                price_table=app.cost_tracker.price_table,
                tracer=app.tracer,
                discount=discount,
                poll_interval_s=0.0,
                timeout_s=timeout_s,
            )
            run_result = await runner.run(batch_requests, timeout_s=timeout_s)
            by_id = run_result.by_id()

        results: list[RunResult] = []
        for prepared, result, run_id, user_input, trace in prepared_runs:
            if prepared is None:
                results.append(result)
                continue
            with app.tracer.bind(trace):
                batch_result = by_id.get(run_id)
                if batch_result is None or not batch_result.ok:
                    result.status = RunStatus.FAILED
                    result.error = (
                        batch_result.error if batch_result else "missing from batch output"
                    )
                else:
                    response = batch_result.response
                    # Discounted cost (already applied by the runner) flows into
                    # the run, the app cost tracker, and the attribution ledger.
                    app.cost_tracker.usage.add(response.usage)
                    app.cost_tracker.model_cost_usd += response.cost_usd
                    result.usage.add(response.usage)
                    result.cost_usd += response.cost_usd
                    app.cost_ledger.record_model_call(
                        model=prepared.model,
                        usage=response.usage,
                        cost_usd=response.cost_usd,
                        provider=response.provider or "",
                        tenant_id=user_input.tenant_id,
                        user_id=user_input.user_id,
                        feature=user_input.feature,
                        run_id=run_id,
                        trace_id=result.trace_id,
                        batch=True,
                    )
                    await self._finalize(prepared, response, result, run_id, user_input, policies)
            app.tracer.export(trace)
            self._persist_run(result, run_id, user_input)
            app.events.emit(
                "run.completed",
                {"run_id": run_id, "status": result.status.value, "batch": True},
                trace_id=result.trace_id,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # steps 1-11: prepare (shared)
    # ------------------------------------------------------------------

    async def _prepare(
        self,
        user_input: UserInput,
        run_config: RunConfig,
        budget: Budget,
        policies: Any,
        result: RunResult,
        run_id: str,
        *,
        enforce_budget: bool = True,
    ) -> _PreparedRun | None:
        app = self.app

        # 2-3. normalize input, detect objective/task type.
        with app.tracer.span("input", type="input") as span:
            routed = app.input_router.route(
                user_input,
                objective=app.objective,
                has_sources=bool(app.sources),
            )
            span.set(
                task_type=routed.task.task_type.value,
                language=routed.language,
                injection_risk=routed.injection.risk if routed.injection else 0.0,
                ambiguous=routed.ambiguity.ambiguous if routed.ambiguity else False,
            )

        # 4. resolve policy and permissions.
        with app.tracer.span("policy", type="security") as span:
            check = app.policy_engine.check_input(routed.input.text or "")
            span.set(allowed=check.allowed, violations=[v.policy for v in check.violations])
            if not check.allowed:
                result.status = RunStatus.DENIED
                result.error = "; ".join(v.message for v in check.blocking)
                app.audit.record(
                    "run", run_id=run_id, user_id=user_input.user_id,
                    tenant_id=user_input.tenant_id, decision="deny",
                    details={"violations": [v.policy for v in check.violations]},
                )
                return None
            if check.transformed_text is not None:
                routed.input.text = check.transformed_text

        # 4c. cost-budget SLO enforcement (1.3): per-tenant/feature budgets are
        # enforced on the same audit path as every other policy decision. A hard
        # cap (or queue-to-batch) denies the interactive run; degrade-to-cheaper
        # swaps in a cheaper model for this run.
        budget_model_override: str | None = None
        if enforce_budget and app.budget_manager.budgets:
            decision = app.budget_manager.check(
                tenant_id=routed.input.tenant_id,
                user_id=routed.input.user_id,
                feature=routed.input.feature,
            )
            if decision.action != "allow":
                app.audit.record(
                    "cost_budget",
                    run_id=run_id,
                    user_id=user_input.user_id,
                    tenant_id=user_input.tenant_id,
                    trace_id=result.trace_id,
                    resource=decision.scope,
                    decision="deny" if not decision.allowed else "degrade",
                    details={
                        "action": decision.action,
                        "spent_usd": decision.spent_usd,
                        "limit_usd": decision.limit_usd,
                        "reason": decision.reason,
                    },
                )
                app.events.emit(
                    "cost.budget_exceeded",
                    {"action": decision.action, "scope": decision.scope, "reason": decision.reason},
                    trace_id=result.trace_id,
                )
                result.metadata["budget"] = decision.model_dump()
                if not decision.allowed:
                    result.status = RunStatus.DENIED
                    hint = (
                        " — resubmit via app.batch() for the discounted batch rate"
                        if decision.action == "queue_to_batch"
                        else ""
                    )
                    result.error = f"cost budget exceeded: {decision.reason}{hint}"
                    return None
                budget_model_override = decision.model_override

        # 4b. multi-schema routing: pick the output contract for this task.
        contract = app.output_contract
        if app.schema_router is not None:
            route = app.schema_router.route(
                routed.input.text or routed.objective.text,
                task_type=routed.task.task_type.value,
            )
            if route is not None:
                contract = OutputContract.from_schema(
                    route.schema_obj,
                    require_citations=app.output_contract.require_citations,
                    validators=app.output_contract.validators,
                    repair_policy=app.output_contract.repair_policy,
                )

        # 5-7. memory recall, file ingestion, and retrieval run concurrently —
        # they are independent reads. Each opens its own span; contextvars
        # keep nesting correct per task.
        async def memory_candidates() -> list[Any]:
            if not (app.memory_enabled and app.memory is not None):
                return []
            with app.tracer.span("memory", type="memory") as span:
                # Hybrid recall, utility-scored against the task (objective +
                # entities) before anything enters the packet.
                task_entities = extract_entities(
                    f"{routed.objective.text} {routed.input.text or ''}"
                )
                memory_results = await app.memory.asearch(
                    routed.input.text or routed.objective.text,
                    user_id=routed.input.user_id,
                    tenant_id=routed.input.tenant_id,
                    session_id=routed.input.session_id,
                    task_entities=task_entities or None,
                    top_k=app.config.memory.max_items_per_run,
                )
                items = [r.item for r in memory_results]
                span.set(candidates=len(items))
                return items

        async def file_evidence() -> list[EvidenceItem]:
            if not user_input.files:
                return []
            return await app.ingest_files([f.path for f in user_input.files])

        async def retrieved_evidence() -> list[EvidenceItem]:
            if app.retrieval is None:
                return []
            with app.tracer.span("retrieval", type="retrieval") as span:
                where = app.tenant_filter(routed.input.tenant_id)
                retrieval = await app.retrieval.retrieve(
                    routed.input.text or routed.objective.text,
                    top_k=run_config.retrieval_top_k or app.config.retrieval.top_k,
                    where=where,
                    objective=routed.objective.text,
                )
                # Untrusted-content screen on retrieved evidence.
                screened: list[EvidenceItem] = []
                for item in retrieval.evidence:
                    verdict = app.policy_engine.check_untrusted_content(
                        item.text or "", source=item.source_id
                    )
                    if verdict.allowed:
                        screened.append(item)
                    else:
                        span.add_event("evidence_blocked", id=item.id)
                span.set(
                    evidence=len(screened),
                    subqueries=len(retrieval.plan.subqueries) if retrieval.plan else 0,
                    latency_ms=retrieval.latency_ms,
                )
                return screened

        memory_items, ingested, retrieved = await gather_bounded(
            (memory_candidates(), file_evidence(), retrieved_evidence()),
            limit=app.config.performance.max_concurrency,
        )
        evidence: list[EvidenceItem] = list(app.pending_evidence)
        evidence.extend(ingested)
        evidence.extend(retrieved)
        # Agent Skills: inject the always-on index plus any task-relevant skill
        # bodies as scored evidence (progressive disclosure). The compiler
        # budgets and cites them like any other context.
        if app.skill_library is not None and len(app.skill_library):
            evidence.extend(
                app.skill_library.evidence_for(
                    f"{routed.objective.text} {routed.input.text or ''}"
                )
            )

        # 9/12a. tool loop happens with the model below; collect tool specs.
        tool_specs = app.tool_registry.specs(app.enabled_tools) if app.enabled_tools else []

        # 8+10. score, compress, compile Context IR + packet.
        with app.tracer.span("context_compile", type="context_compile") as span:
            compiled_context = await app.context_compiler.compile(
                objective=routed.objective,
                user_input=routed.input,
                instructions=app.instructions,
                constraints=app.constraints,
                examples=app.prompt_spec.examples,
                evidence=evidence,
                memory=memory_items,
                tool_specs=tool_specs,
                output_contract=OutputContractRef(
                    schema_ref=contract.schema_name if contract.schema_def else None,
                    schema_def=contract.schema_def,
                    format=contract.format,
                ),
                budget=budget,
                policies=policies,
                trace_parent_id=result.trace_id,
            )
            packet = compiled_context.packet
            result.context_packet_id = packet.id
            result.excluded_context = compiled_context.excluded_report
            span.set(
                packet_id=packet.id,
                token_count=compiled_context.token_count,
                evidence_kept=len(compiled_context.ir.evidence),
                excluded=len(compiled_context.excluded_report),
                cached=compiled_context.from_cache,
            )
            app.store.save(
                "context_packets",
                {
                    "id": packet.id,
                    "run_id": run_id,
                    "spec_hash": packet.spec_hash,
                    "token_count": packet.token_count,
                    "json": packet.model_dump(mode="json"),
                    "created_at": utcnow().isoformat(),
                },
            )

        # 11. render provider request via the prompt compiler.
        provider = app.resolve_provider(run_config)
        # A runtime cascade applies only when the caller did not pin a model and
        # a budget did not force a degrade; it sets the cheapest rung as the
        # baseline so streaming and non-streaming runs start on the same model
        # (escalation itself happens in the non-streaming model loop).
        cascade_active = (
            app.cascade is not None
            and budget_model_override is None
            and run_config.model is None
        )
        if cascade_active:
            model = app.cascade.first().model
        else:
            model = budget_model_override or run_config.model or app.model
        capabilities = provider.capabilities(model)
        decoding = negotiate_decoding(capabilities, contract.schema_def)
        with app.tracer.span("prompt_render", type="prompt_render") as span:
            compiled_prompt = app.prompt_compiler.compile(
                app.prompt_spec,
                user_task=compiled_context.ir.input.text or routed.objective.text,
                variables=app.prompt_variables,
                memory_items=compiled_context.ir.memory_as_items(),
                evidence_items=compiled_context.ir.evidence_as_items(),
                provider_enforces_schema=decoding == DecodingMode.NATIVE,
            )
            span.set(
                prompt_id=compiled_prompt.prompt_id,
                rendered_hash=compiled_prompt.rendered_hash,
                cacheability=round(compiled_prompt.cacheability, 4),
                tokens=compiled_prompt.token_count,
                lint=[f.code for f in compiled_prompt.lint_findings],
                decoding=decoding.value,
                reasoning=(
                    (run_config.reasoning_effort or "default")
                    if (
                        run_config.reasoning_effort is not None
                        or run_config.thinking_budget_tokens is not None
                    )
                    and capabilities.reasoning
                    else "off"
                ),
                schema=contract.schema_name if contract.schema_def else None,
            )

        messages = list(compiled_prompt.messages)
        # Provider-aware prompt caching (1.3): attach a TTL to the stable prefix
        # for caching-capable providers; auto-cache providers rely on ordering.
        cache_info: dict[str, Any] = {}
        if app.prompt_cache is not None:
            messages, cache_info = app.prompt_cache.apply(
                messages, capabilities=capabilities, model=model
            )
        request_kwargs: dict[str, Any] = {}
        if decoding == DecodingMode.NATIVE and contract.schema_def is not None:
            # Strict-sanitized schema rides the provider's constrained
            # decoder; validation still runs against the original schema.
            request_kwargs["output_schema"] = to_strict_json_schema(contract.schema_def)
            request_kwargs["output_schema_name"] = contract.schema_name
        if run_config.temperature is not None:
            request_kwargs["temperature"] = run_config.temperature
        if run_config.seed is not None:
            request_kwargs["seed"] = run_config.seed
        # Unified reasoning control: provider-neutral effort / thinking budget.
        # Providers that don't expose reasoning ignore these fields.
        if run_config.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = run_config.reasoning_effort
        if run_config.thinking_budget_tokens is not None:
            request_kwargs["thinking_budget_tokens"] = run_config.thinking_budget_tokens
        request_kwargs["max_output_tokens"] = budget.max_output_tokens

        return _PreparedRun(
            routed=routed,
            compiled_context=compiled_context,
            compiled_prompt=compiled_prompt,
            provider=provider,
            model=model,
            tool_specs=tool_specs,
            messages=messages,
            contract=contract,
            decoding=decoding,
            request_kwargs=request_kwargs,
            cache_info=cache_info,
            cascade_active=cascade_active,
        )

    # ------------------------------------------------------------------
    # step 12: model + tool loop
    # ------------------------------------------------------------------

    async def _run_tool_round(
        self,
        tool_calls: list[Any],
        prepared: _PreparedRun,
        budget: Budget,
        result: RunResult,
        user_input: UserInput,
        run_id: str,
    ) -> list[tuple[Any, ToolResult]]:
        """Execute one round of tool calls concurrently (bounded fan-out).

        Results come back in call order, so the transcript is deterministic
        regardless of completion order.
        """
        app = self.app
        remaining = budget.max_tool_calls - len(result.tool_results)
        allowed_calls = tool_calls[: max(0, remaining)]
        if not allowed_calls:
            return []

        async def run_one(tool_call: Any) -> ToolResult:
            try:
                return await app.tool_runtime.execute(
                    ToolCall(tool_name=tool_call.name, arguments=tool_call.arguments),
                    principal=app.principal_for(prepared.routed.input),
                )
            except VincioError as exc:
                return ToolResult(
                    call_id=tool_call.id, tool_name=tool_call.name, status="error", error=exc.message
                )

        tool_results = await gather_bounded(
            (run_one(tool_call) for tool_call in allowed_calls),
            limit=app.config.performance.tool_parallelism,
        )
        rounds: list[tuple[Any, ToolResult]] = []
        for tool_call, tool_result in zip(allowed_calls, tool_results, strict=True):
            result.tool_results.append(tool_result)
            app.audit.record(
                "tool_call", run_id=run_id, user_id=user_input.user_id,
                tenant_id=user_input.tenant_id, resource=tool_call.name,
                decision=tool_result.status,
            )
            rounds.append((tool_call, tool_result))
        return rounds

    @staticmethod
    def _tool_message(tool_call: Any, tool_result: ToolResult) -> Message:
        return Message(
            role="tool",
            content=json.dumps(tool_result.output, default=str)
            if tool_result.status == "ok"
            else f"error: {tool_result.error}",
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )

    def _confidence(self, response: ModelResponse, *, expects_schema: bool) -> float:
        """Runtime confidence for cascade escalation (custom signal or default)."""
        fn = self.app._cascade_confidence
        if fn is not None:
            try:
                return float(fn(response))
            except Exception:  # noqa: BLE001 - a bad signal must not break the run
                pass
        from ..optimize.routing import response_confidence

        return response_confidence(response, expects_schema=expects_schema)

    def _cascade_provider(self, provider_name: str | None) -> ModelProvider | None:
        if not provider_name:
            return None
        return self.app.resolve_provider(RunConfig(provider=provider_name))

    def _attribute_cost(
        self,
        *,
        model: str,
        response: ModelResponse,
        cost: float,
        result: RunResult,
        user_input: UserInput,
        run_id: str,
        span: Any,
        batch: bool = False,
    ) -> None:
        """Record an attributed cost event and run anomaly detection."""
        app = self.app
        event = app.cost_ledger.record_model_call(
            model=model,
            usage=response.usage,
            cost_usd=cost,
            provider=response.provider or "",
            tenant_id=user_input.tenant_id,
            user_id=user_input.user_id,
            feature=user_input.feature,
            run_id=run_id,
            trace_id=result.trace_id,
            batch=batch,
        )
        app.budget_manager.observe(event)
        if user_input.tenant_id or user_input.feature or user_input.user_id:
            span.set(
                tenant_id=user_input.tenant_id,
                feature=user_input.feature,
                user_id=user_input.user_id,
            )

    async def _call_model(
        self,
        prepared: _PreparedRun,
        messages: list[Message],
        model: str,
        result: RunResult,
        user_input: UserInput,
        run_id: str,
        *,
        provider: ModelProvider | None = None,
    ) -> ModelResponse:
        """One model call: cache-checked, cost-tracked, attributed, traced."""
        app = self.app
        provider = provider or prepared.provider
        request = ModelRequest(
            model=model, messages=messages, tools=prepared.tool_specs, **prepared.request_kwargs
        )
        cached = app.response_cache.get(request) if app.response_cache else None
        with app.tracer.span("model", type="model_call") as span:
            span.set(model=model, request_hash=request.hash, cached=cached is not None)
            if prepared.cache_info.get("applied"):
                span.set(
                    cache_strategy_ttl=prepared.cache_info.get("ttl"),
                    cache_breakpoints=prepared.cache_info.get("breakpoints"),
                )
            if cached is not None:
                # A response-cache hit served the answer without an API call, so
                # it is free; track the tokens but bill nothing.
                response = cached
                app.cost_tracker.usage.add(response.usage)
                spent = 0.0
            else:
                response = await provider.generate(request)
                if app.response_cache is not None and not response.tool_calls:
                    app.response_cache.set(
                        request, response, prompt_version=prepared.compiled_prompt.prompt_spec_hash
                    )
                cost = app.cost_tracker.record_model_call(model, response.usage)
                spent = cost if cost else response.cost_usd
            result.usage.add(response.usage)
            result.cost_usd += spent
            self._attribute_cost(
                model=model, response=response, cost=spent, result=result,
                user_input=user_input, run_id=run_id, span=span,
            )
            span.set(
                finish=response.finish_reason,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                reasoning_tokens=response.usage.reasoning_tokens,
                cached_input_tokens=response.usage.cached_input_tokens,
                cache_hit_rate=round(
                    cache_hit_rate(response.usage.input_tokens, response.usage.cached_input_tokens), 4
                ),
                cost_usd=round(result.cost_usd, 8),
                response_text=response.text[:500],
            )
        return response

    async def _model_tool_loop(
        self,
        prepared: _PreparedRun,
        budget: Budget,
        result: RunResult,
        user_input: UserInput,
        run_id: str,
    ) -> ModelResponse:
        """Model + tool loop with optional confidence-based cascade escalation.

        Runs start on ``prepared.model`` — which ``_prepare`` already set to the
        cascade's cheapest rung when one is active. A terminal answer below the
        rung's confidence threshold is regenerated on the next stronger rung
        (clean: the prior answer is never appended to the transcript), bounded by
        the cascade's escalation cap. Tool rounds use the current rung and are
        bounded by ``max_tool_calls``. Streaming runs reuse this loop and replay
        the accepted answer as deltas (see ``_model_tool_loop_stream``).
        """
        cascade = self.app.cascade if prepared.cascade_active else None
        expects_schema = prepared.contract.schema_def is not None
        rung = cascade.first() if cascade is not None else None
        model = prepared.model  # already the cascade's first rung when active
        provider = self._cascade_provider(rung.provider) if rung is not None else None
        response: ModelResponse | None = None
        tool_rounds = 0
        escalations = 0
        while True:
            response = await self._call_model(
                prepared, prepared.messages, model, result, user_input, run_id, provider=provider
            )
            if response.tool_calls and prepared.tool_specs and tool_rounds < budget.max_tool_calls:
                tool_rounds += 1
                prepared.messages.append(
                    Message(role="assistant", content=response.text or "", tool_calls=response.tool_calls)
                )
                executed = await self._run_tool_round(
                    response.tool_calls, prepared, budget, result, user_input, run_id
                )
                for tool_call, tool_result in executed:
                    prepared.messages.append(self._tool_message(tool_call, tool_result))
                continue
            # Terminal answer: consider a cascade escalation.
            if cascade is not None and escalations < cascade.escalation_cap:
                confidence = self._confidence(response, expects_schema=expects_schema)
                nxt = cascade.next_rung(model, confidence)
                if nxt is not None:
                    escalations += 1
                    model = nxt.model
                    provider = self._cascade_provider(nxt.provider)
                    continue
            break
        assert response is not None
        if cascade is not None:
            result.metadata.setdefault("cascade", {})["model"] = model
            result.metadata["cascade"]["escalations"] = escalations
        return response

    async def _model_tool_loop_stream(
        self,
        prepared: _PreparedRun,
        budget: Budget,
        result: RunResult,
        user_input: UserInput,
        run_id: str,
    ) -> AsyncIterator[RunStreamEvent | ModelResponse]:
        """Streaming model loop: yields RunStreamEvents and, finally, the
        terminal ModelResponse (the caller filters by type)."""
        app = self.app
        messages = prepared.messages
        expects_json = (
            prepared.contract.schema_def is not None or prepared.contract.format == "json"
        )
        min_parse_chars = app.config.performance.partial_parse_min_chars
        response: ModelResponse | None = None

        # Cascade runs buffer each rung and stream the accepted answer: run the
        # non-streaming cascade loop (which escalates on low confidence) to
        # produce the final response, then replay its text as deltas so the
        # consumer sees only the final, escalated answer — never a discarded
        # cheap attempt.
        if prepared.cascade_active:
            response = await self._model_tool_loop(prepared, budget, result, user_input, run_id)
            chunk = 16
            for start in range(0, len(response.text), chunk):
                yield RunStreamEvent(type="text_delta", text=response.text[start : start + chunk])
            if response.structured is not None:
                yield RunStreamEvent(
                    type="partial_output",
                    partial_output=response.structured,
                    output_complete=True,
                    valid_prefix=True,
                )
            yield response
            return

        for _round in range(budget.max_tool_calls + 1):
            # Fresh per model round: deltas accumulate per response.
            stream_validator = (
                StreamingValidator(
                    prepared.contract.output_schema(),
                    repair_policy=prepared.contract.repair_policy,
                    min_interval_chars=min_parse_chars,
                )
                if expects_json
                else None
            )
            request = ModelRequest(
                model=prepared.model,
                messages=messages,
                tools=prepared.tool_specs,
                **prepared.request_kwargs,
            )
            cached = app.response_cache.get(request) if app.response_cache else None
            with app.tracer.span("model", type="model_call") as span:
                span.set(
                    model=prepared.model,
                    request_hash=request.hash,
                    cached=cached is not None,
                    stream=True,
                )
                if cached is not None:
                    response = cached
                    if response.text:
                        yield RunStreamEvent(type="text_delta", text=response.text)
                else:
                    started = time.monotonic()
                    first_token_ms: int | None = None
                    accumulated: list[str] = []
                    response = None
                    async for event in prepared.provider.stream(request):
                        if event.type == "text_delta" and event.text:
                            if first_token_ms is None:
                                first_token_ms = int((time.monotonic() - started) * 1000)
                                span.set(ttft_ms=first_token_ms)
                                span.add_event("first_token", ttft_ms=first_token_ms)
                            accumulated.append(event.text)
                            yield RunStreamEvent(type="text_delta", text=event.text)
                            if stream_validator is not None:
                                # Streaming validation: parse the balanced
                                # partial and prefix-check it against the
                                # schema; definite mismatches surface
                                # mid-stream.
                                partial_event = stream_validator.feed(event.text)
                                if partial_event is not None:
                                    if not partial_event.valid_prefix:
                                        span.add_event(
                                            "stream_invalid_prefix",
                                            errors=partial_event.errors[:4],
                                        )
                                    yield RunStreamEvent(
                                        type="partial_output",
                                        partial_output=partial_event.data,
                                        output_complete=partial_event.complete,
                                        valid_prefix=partial_event.valid_prefix,
                                        validation_errors=partial_event.errors,
                                    )
                        elif event.type == "usage" and event.usage is not None:
                            yield RunStreamEvent(type="usage", usage=event.usage)
                        elif event.type == "done" and event.response is not None:
                            response = event.response
                    if response is None:
                        response = ModelResponse(
                            model=prepared.model, text="".join(accumulated)
                        )
                    if app.response_cache is not None and not response.tool_calls:
                        app.response_cache.set(
                            request, response, prompt_version=prepared.compiled_prompt.prompt_spec_hash
                        )
                if cached is not None:
                    # A response-cache hit is free; track tokens, bill nothing.
                    app.cost_tracker.usage.add(response.usage)
                    spent = 0.0
                else:
                    cost = app.cost_tracker.record_model_call(prepared.model, response.usage)
                    spent = cost if cost else response.cost_usd
                result.usage.add(response.usage)
                result.cost_usd += spent
                self._attribute_cost(
                    model=prepared.model, response=response, cost=spent, result=result,
                    user_input=user_input, run_id=run_id, span=span,
                )
                span.set(
                    finish=response.finish_reason,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    reasoning_tokens=response.usage.reasoning_tokens,
                    cached_input_tokens=response.usage.cached_input_tokens,
                    cache_hit_rate=round(
                        cache_hit_rate(
                            response.usage.input_tokens, response.usage.cached_input_tokens
                        ),
                        4,
                    ),
                    cost_usd=round(result.cost_usd, 8),
                    response_text=response.text[:500],
                )
            if not response.tool_calls or not prepared.tool_specs:
                break
            messages.append(
                Message(role="assistant", content=response.text or "", tool_calls=response.tool_calls)
            )
            for tool_call in response.tool_calls:
                yield RunStreamEvent(type="tool_call", tool_name=tool_call.name)
            executed = await self._run_tool_round(
                response.tool_calls, prepared, budget, result, user_input, run_id
            )
            for tool_call, tool_result in executed:
                messages.append(self._tool_message(tool_call, tool_result))
                yield RunStreamEvent(
                    type="tool_result", tool_name=tool_call.name, tool_result=tool_result
                )
        assert response is not None
        yield response

    # ------------------------------------------------------------------
    # steps 13-16: finalize (shared)
    # ------------------------------------------------------------------

    async def _finalize(
        self,
        prepared: _PreparedRun,
        response: ModelResponse,
        result: RunResult,
        run_id: str,
        user_input: UserInput,
        policies: Any,
    ) -> None:
        app = self.app
        compiled_context = prepared.compiled_context
        routed = prepared.routed
        result.raw_text = response.text

        # 13. parse and validate output.
        contract = prepared.contract
        evidence_ids = {e.id for e in compiled_context.ir.evidence} | {
            e.citation_ref for e in compiled_context.ir.evidence
        } | {entry.get("id") for entry in compiled_context.ir.evidence_ledger if entry.get("id")}
        with app.tracer.span("output_validation", type="output_validation") as span:
            validator = OutputValidator(
                contract,
                semantic_validators=app.semantic_validators,
                policy_engine=app.policy_engine,
                repairer=app.repairer,
            )
            report = await validator.validate(
                response.text,
                structured=response.structured,
                evidence_ids=evidence_ids if evidence_ids else None,
            )

            # 13b. bounded self-correction (validate → critique → repair).
            correction_cycles = 0
            if not report.valid and app.self_correction is not None:
                corrector = SelfCorrector(
                    validator,
                    provider=prepared.provider,
                    model=prepared.model,
                    **app.self_correction,
                )
                correction = await corrector.correct(
                    response.text,
                    structured=response.structured,
                    evidence_ids=evidence_ids if evidence_ids else None,
                    initial_report=report,
                )
                result.cost_usd += correction.cost_usd
                # Attribute the correction's model spend too, so the cost ledger
                # stays consistent with result.cost_usd (1.3).
                if correction.cost_usd:
                    app.cost_ledger.record_model_call(
                        model=prepared.model,
                        usage=TokenUsage(),
                        cost_usd=correction.cost_usd,
                        provider=getattr(prepared.provider, "name", ""),
                        tenant_id=user_input.tenant_id,
                        user_id=user_input.user_id,
                        feature=user_input.feature,
                        run_id=run_id,
                        trace_id=result.trace_id,
                    )
                correction_cycles = correction.cycles
                span.add_event(
                    "self_correction",
                    cycles=correction.cycles,
                    cost_usd=round(correction.cost_usd, 8),
                    stopped=correction.stopped_reason,
                    valid=correction.valid,
                )
                if correction.report is not None:
                    report = correction.report
                if correction.valid:
                    result.raw_text = correction.raw_text

            # Every repair and failure is a trace event (and an audit entry below).
            for action in report.repair_actions:
                span.add_event("repair", action=action)
            for step in report.steps:
                if not step.passed:
                    span.add_event("validation_failed", step=step.name, detail=step.detail[:200])
            span.set(
                valid=report.valid,
                steps=[s.name for s in report.steps if not s.passed],
                repairs=report.repair_actions,
                decoding=prepared.decoding.value,
                schema=contract.schema_name if contract.schema_def else None,
            )
            result.validation = report.model_dump(mode="json", exclude={"output", "raw_text"})
            result.citations = report.citations
            if report.valid:
                result.output = report.output
            else:
                result.output = response.structured or response.text
                result.status = RunStatus.FAILED
                result.error = "output validation failed: " + "; ".join(report.errors)
            if report.repair_actions or correction_cycles or not report.valid:
                app.audit.record(
                    "output_validation",
                    run_id=run_id,
                    user_id=user_input.user_id,
                    tenant_id=user_input.tenant_id,
                    trace_id=result.trace_id,
                    decision="repair" if report.valid else "deny",
                    details={
                        "schema": contract.schema_name if contract.schema_def else None,
                        "errors": report.errors[:8],
                        "repairs": report.repair_actions[:8],
                        "correction_cycles": correction_cycles,
                    },
                )

        result.evidence = compiled_context.ir.evidence

        # 14. evaluate output with attached evaluators.
        if app.evaluators:
            with app.tracer.span("evaluate", type="eval") as span:
                run_output = RunOutput(
                    output=result.output,
                    raw_text=result.raw_text,
                    evidence=result.evidence,
                    citations=result.citations,
                    usage=result.usage,
                    cost_usd=result.cost_usd,
                    schema_valid=report.valid if contract.schema_def else None,
                )
                case = EvalCase(id=run_id, input=routed.input.text or "")
                for evaluator_name in app.evaluators:
                    metric = METRICS.get(evaluator_name)
                    if metric is None:
                        continue
                    metric_result = metric(case, run_output)
                    result.eval_scores[metric_result.name] = metric_result.value
                span.set(**result.eval_scores)
                for metric_name, score in result.eval_scores.items():
                    span.add_score(metric_name, score)

        # 16. memory writes.
        if (
            app.memory_enabled
            and app.memory is not None
            and policies.allow_memory_writes
            and app.config.memory.write_policy != "off"
            and result.status != RunStatus.FAILED
        ):
            with app.tracer.span("memory_write", type="memory_write") as span:
                write_back = app.config.memory.write_back
                written: list[Any] = []
                if "input" in write_back:
                    written += await app.memory.ingest(
                        routed.input.text or "",
                        owner_id=routed.input.user_id,
                        source_trace_id=result.trace_id,
                    )
                # Confirmed evidence (cited in the output) and successful
                # tool results flow back as candidate memories with
                # provenance; recall utility-scores them before reuse.
                if "evidence" in write_back and result.citations:
                    cited_refs = set(result.citations)
                    cited = [
                        item
                        for item in result.evidence
                        if item.id in cited_refs or item.citation_ref in cited_refs
                    ]
                    written += app.memory.write_back(
                        evidence=cited[:5],
                        owner_id=routed.input.user_id,
                        session_id=routed.input.session_id,
                        source_trace_id=result.trace_id,
                    )
                if "tools" in write_back and result.tool_results:
                    written += app.memory.write_back(
                        tool_results=[t for t in result.tool_results if t.status == "ok"][:5],
                        owner_id=routed.input.user_id,
                        session_id=routed.input.session_id,
                        source_trace_id=result.trace_id,
                    )
                # Auto-memory from runs (0.8): verifiable output claims that
                # the cited evidence supports become candidate memories.
                if "facts" in write_back and result.raw_text:
                    from ..memory.facts import extract_grounded_facts

                    cited_refs = set(result.citations)
                    grounding = [
                        item
                        for item in result.evidence
                        if item.id in cited_refs or item.citation_ref in cited_refs
                    ] or result.evidence
                    facts = extract_grounded_facts(
                        result.raw_text,
                        grounding,
                        min_support=app.config.memory.fact_min_support,
                        max_facts=app.config.memory.max_facts_per_run,
                    )
                    if facts:
                        written += app.memory.write_back(
                            facts=facts,
                            owner_id=routed.input.user_id,
                            session_id=routed.input.session_id,
                            source_trace_id=result.trace_id,
                        )
                span.set(written=len(written))
                if written:
                    app.audit.record(
                        "memory_write", run_id=run_id, user_id=user_input.user_id,
                        tenant_id=user_input.tenant_id,
                        details={
                            "count": len(written),
                            "origins": sorted(
                                {str(w.metadata.get("origin", "input")) for w in written}
                            ),
                        },
                    )

        app.audit.record(
            "run", run_id=run_id, user_id=user_input.user_id, tenant_id=user_input.tenant_id,
            decision="allow",
            details={
                "sources": [e.source_id for e in result.evidence][:32],
                "tools": [t.tool_name for t in result.tool_results],
                "cost_usd": result.cost_usd,
            },
        )
        if result.status == RunStatus.RUNNING:
            result.status = RunStatus.SUCCEEDED

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _execute_inner(
        self,
        user_input: UserInput,
        run_config: RunConfig,
        budget: Budget,
        policies: Any,
        result: RunResult,
        run_id: str,
    ) -> None:
        prepared = await self._prepare(user_input, run_config, budget, policies, result, run_id)
        if prepared is None:
            return
        response = await self._model_tool_loop(prepared, budget, result, user_input, run_id)
        await self._finalize(prepared, response, result, run_id, user_input, policies)

    def _capture_training_artifacts(
        self, trace: Any, result: RunResult, user_input: UserInput
    ) -> None:
        """Record the full output and cited evidence on the trace for the
        distillation flywheel (opt-in via ``observability.training_capture``).

        The span's ``output`` stays truncated for cost; these untruncated
        attributes let ``export_training_set`` build faithful, grounded training
        examples and run the grounding check against the evidence the answer
        actually cited.
        """
        trace.attributes["input_full"] = user_input.text or ""
        trace.attributes["output_full"] = result.raw_text or ""
        cited = set(result.citations or [])
        evidence = [
            {
                "id": item.id,
                "source_id": item.source_id,
                "text": item.text,
                "citation_ref": item.citation_ref,
            }
            for item in result.evidence
            if not cited or item.id in cited or item.citation_ref in cited
        ] or [
            {"id": item.id, "source_id": item.source_id, "text": item.text,
             "citation_ref": item.citation_ref}
            for item in result.evidence
        ]
        if evidence:
            trace.attributes["evidence"] = evidence

    def _persist_run(self, result: RunResult, run_id: str, user_input: UserInput) -> None:
        self.app.store.save(
            "runs",
            {
                "id": run_id,
                "app_id": self.app.name,
                "user_id": user_input.user_id,
                "tenant_id": user_input.tenant_id,
                "objective": (user_input.text or "")[:300],
                "status": result.status.value,
                "started_at": utcnow().isoformat(),
                "cost_usd": result.cost_usd,
                "latency_ms": result.latency_ms,
                "trace_id": result.trace_id,
            },
        )
