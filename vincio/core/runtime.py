"""Vincio runtime core: the 17-step run flow.

1. receive input        7. retrieve evidence      13. parse + validate output
2. normalize            8. score + compress       14. evaluate output
3. detect objective     9. plan tools             15. write trace
4. resolve policy      10. compile Context IR     16. write memory updates
5. memory candidates   11. render model request   17. return result
6. plan retrieval      12. execute model/tools
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from ..context.ir import OutputContractRef
from ..core.errors import VincioError
from ..core.types import (
    Budget,
    EvidenceItem,
    Message,
    ModelRequest,
    RunConfig,
    RunResult,
    RunStatus,
    ToolCall,
    UserInput,
)
from ..core.utils import new_id, utcnow
from ..evals.datasets import EvalCase
from ..evals.metrics import METRICS, RunOutput
from ..output.validators import OutputValidator

if TYPE_CHECKING:  # pragma: no cover
    from .app import ContextApp

__all__ = ["VincioRuntime"]


class VincioRuntime:
    def __init__(self, app: ContextApp) -> None:
        self.app = app

    async def execute(self, user_input: UserInput, run_config: RunConfig | None = None) -> RunResult:
        app = self.app
        run_config = run_config or RunConfig()
        run_id = new_id("run")
        started = time.monotonic()
        budget = run_config.budget or app.budget
        policies = run_config.policies or app.policies
        result = RunResult(run_id=run_id, status=RunStatus.RUNNING)

        with app.tracer.trace(
            run_id=run_id,
            user_id=user_input.user_id,
            tenant_id=user_input.tenant_id,
            input=(user_input.text or "")[:500],
        ) as trace:
            result.trace_id = trace.id
            try:
                await self._execute_inner(
                    user_input, run_config, budget, policies, result, run_id
                )
            except VincioError as exc:
                result.status = RunStatus.FAILED
                result.error = exc.message
            result.latency_ms = int((time.monotonic() - started) * 1000)

        # 15. persist run + packet (trace export happens in the tracer).
        app.store.save(
            "runs",
            {
                "id": run_id,
                "app_id": app.name,
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
        app.events.emit("run.completed", {"run_id": run_id, "status": result.status.value}, trace_id=result.trace_id)
        return result

    async def _execute_inner(
        self,
        user_input: UserInput,
        run_config: RunConfig,
        budget: Budget,
        policies: Any,
        result: RunResult,
        run_id: str,
    ) -> None:
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
                return
            if check.transformed_text is not None:
                routed.input.text = check.transformed_text

        # 5. memory candidates.
        memory_items = []
        if app.memory_enabled and app.memory is not None:
            with app.tracer.span("memory", type="memory") as span:
                memory_results = app.memory.search(
                    routed.input.text or routed.objective.text,
                    user_id=routed.input.user_id,
                    tenant_id=routed.input.tenant_id,
                    session_id=routed.input.session_id,
                    top_k=app.config.memory.max_items_per_run,
                )
                memory_items = [r.item for r in memory_results]
                span.set(candidates=len(memory_items))

        # 6-7. retrieval planning + evidence retrieval.
        evidence: list[EvidenceItem] = list(app.pending_evidence)
        if user_input.files:
            evidence.extend(await app.ingest_files([f.path for f in user_input.files]))
        if app.retrieval is not None:
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
                evidence.extend(screened)
                span.set(
                    evidence=len(evidence),
                    subqueries=len(retrieval.plan.subqueries) if retrieval.plan else 0,
                    latency_ms=retrieval.latency_ms,
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
                    schema_ref=app.output_contract.schema_name if app.output_contract.schema_def else None,
                    schema_def=app.output_contract.schema_def,
                    format=app.output_contract.format,
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
        with app.tracer.span("prompt_render", type="prompt_render") as span:
            compiled_prompt = app.prompt_compiler.compile(
                app.prompt_spec,
                user_task=compiled_context.ir.input.text or routed.objective.text,
                variables=app.prompt_variables,
                memory_items=compiled_context.ir.memory_as_items(),
                evidence_items=compiled_context.ir.evidence_as_items(),
                provider_enforces_schema=capabilities.structured_output
                and app.output_contract.schema_def is not None,
            )
            span.set(
                prompt_id=compiled_prompt.prompt_id,
                rendered_hash=compiled_prompt.rendered_hash,
                cacheability=round(compiled_prompt.cacheability, 4),
                tokens=compiled_prompt.token_count,
                lint=[f.code for f in compiled_prompt.lint_findings],
            )

        messages = list(compiled_prompt.messages)
        request_kwargs: dict[str, Any] = {}
        if app.output_contract.schema_def is not None and capabilities.structured_output:
            request_kwargs["output_schema"] = app.output_contract.schema_def
            request_kwargs["output_schema_name"] = app.output_contract.schema_name
        if run_config.temperature is not None:
            request_kwargs["temperature"] = run_config.temperature
        if run_config.seed is not None:
            request_kwargs["seed"] = run_config.seed
        request_kwargs["max_output_tokens"] = budget.max_output_tokens

        # 12. execute model (with bounded tool loop when tools are enabled).
        response = None
        for _round in range(budget.max_tool_calls + 1):
            request = ModelRequest(
                model=model, messages=messages, tools=tool_specs, **request_kwargs
            )
            cached = app.response_cache.get(request) if app.response_cache else None
            with app.tracer.span("model", type="model_call") as span:
                span.set(model=model, request_hash=request.hash, cached=cached is not None)
                if cached is not None:
                    response = cached
                else:
                    response = await provider.generate(request)
                    if app.response_cache is not None and not response.tool_calls:
                        app.response_cache.set(request, response, prompt_version=compiled_prompt.prompt_spec_hash)
                cost = app.cost_tracker.record_model_call(model, response.usage)
                result.usage.add(response.usage)
                result.cost_usd += cost if cost else response.cost_usd
                span.set(
                    finish=response.finish_reason,
                    output_tokens=response.usage.output_tokens,
                    cost_usd=round(result.cost_usd, 8),
                    response_text=response.text[:500],
                )
            if not response.tool_calls or not tool_specs:
                break
            # Tool execution round.
            messages.append(
                Message(role="assistant", content=response.text or "", tool_calls=response.tool_calls)
            )
            for tool_call in response.tool_calls:
                if len(result.tool_results) >= budget.max_tool_calls:
                    break
                try:
                    tool_result = await app.tool_runtime.execute(
                        ToolCall(tool_name=tool_call.name, arguments=tool_call.arguments),
                        principal=app.principal_for(routed.input),
                    )
                except VincioError as exc:
                    from ..core.types import ToolResult

                    tool_result = ToolResult(
                        call_id=tool_call.id, tool_name=tool_call.name, status="error", error=exc.message
                    )
                result.tool_results.append(tool_result)
                app.audit.record(
                    "tool_call", run_id=run_id, user_id=user_input.user_id,
                    tenant_id=user_input.tenant_id, resource=tool_call.name,
                    decision=tool_result.status,
                )
                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(tool_result.output, default=str)
                        if tool_result.status == "ok"
                        else f"error: {tool_result.error}",
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    )
                )

        assert response is not None
        result.raw_text = response.text

        # 13. parse and validate output.
        evidence_ids = {e.id for e in compiled_context.ir.evidence} | {
            e.citation_ref for e in compiled_context.ir.evidence
        } | {entry.get("id") for entry in compiled_context.ir.evidence_ledger if entry.get("id")}
        with app.tracer.span("output_validation", type="output_validation") as span:
            validator = OutputValidator(
                app.output_contract,
                semantic_validators=app.semantic_validators,
                policy_engine=app.policy_engine,
                repairer=app.repairer,
            )
            report = await validator.validate(
                response.text,
                structured=response.structured,
                evidence_ids=evidence_ids if evidence_ids else None,
            )
            span.set(valid=report.valid, steps=[s.name for s in report.steps if not s.passed], repairs=report.repair_actions)
            result.validation = report.model_dump(mode="json", exclude={"output", "raw_text"})
            result.citations = report.citations
            if report.valid:
                result.output = report.output
            else:
                result.output = response.structured or response.text
                result.status = RunStatus.FAILED
                result.error = "output validation failed: " + "; ".join(report.errors)

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
                    schema_valid=report.valid if app.output_contract.schema_def else None,
                )
                case = EvalCase(id=run_id, input=routed.input.text or "")
                for evaluator_name in app.evaluators:
                    metric = METRICS.get(evaluator_name)
                    if metric is None:
                        continue
                    metric_result = metric(case, run_output)
                    result.eval_scores[metric_result.name] = metric_result.value
                span.set(**result.eval_scores)

        # 16. memory writes.
        if (
            app.memory_enabled
            and app.memory is not None
            and policies.allow_memory_writes
            and app.config.memory.write_policy != "off"
            and result.status != RunStatus.FAILED
        ):
            with app.tracer.span("memory_write", type="memory_write") as span:
                written = await app.memory.ingest(
                    routed.input.text or "",
                    owner_id=routed.input.user_id,
                    source_trace_id=result.trace_id,
                )
                span.set(written=len(written))
                if written:
                    app.audit.record(
                        "memory_write", run_id=run_id, user_id=user_input.user_id,
                        tenant_id=user_input.tenant_id,
                        details={"count": len(written)},
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
