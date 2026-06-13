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
        model = run_config.model or app.model
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
                schema=contract.schema_name if contract.schema_def else None,
            )

        messages = list(compiled_prompt.messages)
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

    async def _model_tool_loop(
        self,
        prepared: _PreparedRun,
        budget: Budget,
        result: RunResult,
        user_input: UserInput,
        run_id: str,
    ) -> ModelResponse:
        app = self.app
        messages = prepared.messages
        response: ModelResponse | None = None
        for _round in range(budget.max_tool_calls + 1):
            request = ModelRequest(
                model=prepared.model,
                messages=messages,
                tools=prepared.tool_specs,
                **prepared.request_kwargs,
            )
            cached = app.response_cache.get(request) if app.response_cache else None
            with app.tracer.span("model", type="model_call") as span:
                span.set(model=prepared.model, request_hash=request.hash, cached=cached is not None)
                if cached is not None:
                    response = cached
                else:
                    response = await prepared.provider.generate(request)
                    if app.response_cache is not None and not response.tool_calls:
                        app.response_cache.set(
                            request, response, prompt_version=prepared.compiled_prompt.prompt_spec_hash
                        )
                cost = app.cost_tracker.record_model_call(prepared.model, response.usage)
                result.usage.add(response.usage)
                result.cost_usd += cost if cost else response.cost_usd
                span.set(
                    finish=response.finish_reason,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cost_usd=round(result.cost_usd, 8),
                    response_text=response.text[:500],
                )
            if not response.tool_calls or not prepared.tool_specs:
                break
            # Tool execution round (concurrent fan-out).
            messages.append(
                Message(role="assistant", content=response.text or "", tool_calls=response.tool_calls)
            )
            executed = await self._run_tool_round(
                response.tool_calls, prepared, budget, result, user_input, run_id
            )
            for tool_call, tool_result in executed:
                messages.append(self._tool_message(tool_call, tool_result))
        assert response is not None
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
                cost = app.cost_tracker.record_model_call(prepared.model, response.usage)
                result.usage.add(response.usage)
                result.cost_usd += cost if cost else response.cost_usd
                span.set(
                    finish=response.finish_reason,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
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
