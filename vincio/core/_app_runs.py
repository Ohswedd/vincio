"""Evaluator registration, structured output, task, execution, agent, and workflow verbs — a private mixin of
:class:`~vincio.core.app.ContextApp`.

Extracted verbatim from ``vincio/core/app.py`` (v7.5 structure line): method
source, decorators, comments, and docstrings are unchanged. ``ContextApp``
composes this class, so every method here remains an ``app.*`` verb; the
``self: ContextApp`` annotations keep attribute access type-checked against
the composed app. The standing hygiene lints (:mod:`vincio._error_contract`,
:mod:`vincio._observable_failure`, :mod:`vincio._assert_robustness`)
deliberately keep ``vincio/core/_app_*.py`` in scope despite the private
filename, so the verb surface stays guarded after the split.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..agents.blackboard import Blackboard
from ..agents.crew import AgentRole, Crew
from ..agents.executor import AgentExecutor
from ..agents.graph import Checkpointer, StateGraph
from ..agents.planner import Planner
from ..evals.datasets import Dataset
from ..evals.online import OnlineEvaluator
from ..output.routing import SchemaRouter
from ..output.schemas import OutputContract, OutputSchema
from ..output.validators import SemanticValidator
from ..prompts.signatures import Predict, Signature
from ..prompts.templates import PromptSpec
from ..providers.base import run_sync
from ..stability import experimental
from ..workflows.engine import Workflow
from ._app_support import RunHandle, _AgentHandle
from .errors import (
    AgentEngineError,
    ConfigError,
    InputError,
)
from .types import (
    EvidenceItem,
    FileRef,
    Objective,
    RunConfig,
    RunResult,
    RunStreamEvent,
    TaskType,
    UserInput,
)

if TYPE_CHECKING:
    from ..assistant import Assistant
    from .app import ContextApp
    from .types import Objective


class _RunVerbs:
    """Evaluator registration, structured output, task, execution, agent, and workflow verbs. Mixed into :class:`~vincio.core.app.ContextApp`."""

    if TYPE_CHECKING:
        # ContextApp state this mixin's verbs assign. mypy would otherwise
        # attribute the unannotated ``self.X = ...`` assignments to this class
        # and clash with ContextApp.__init__; the declarations (type-checking
        # only, no runtime effect) keep the split typing identical to the
        # monolith's.
        context_governor: Any | None
        kv_prefix_pool: Any | None
        objective: Objective | None
        output_contract: OutputContract
        prompt_spec: PromptSpec
        reasoning_controller: Any | None
        reasoning_engine: Any | None
        schema_router: SchemaRouter | None
        self_correction: dict[str, Any] | None
        semantic_cache: Any | None


    # -- evaluators / optimizers ----------------------------------------------------------------------

    def add_evaluator(self: ContextApp, name: str | Callable) -> ContextApp:  # type: ignore[misc]
        """Register a metric (by name or callable) that scores every run."""
        if callable(name):
            from ..evals.metrics import METRICS

            # Resolve the name once: computing it twice around the insertion
            # would disagree for a callable without __name__ (the second
            # len(METRICS) is one larger), registering one key but recording
            # a different one in self.evaluators.
            fn = name
            name = getattr(fn, "__name__", f"custom_{len(METRICS)}")
            METRICS[name] = fn
        self.evaluators.append(name)
        return self

    def add_validator(  # type: ignore[misc]
        self: ContextApp, name: str, validator: SemanticValidator, *, blocking: bool = True
    ) -> ContextApp:
        """Register a semantic output validator (blocking by default)."""
        from ..output.schemas import ValidatorSpec

        self.semantic_validators[name] = validator
        self.output_contract.validators.append(ValidatorSpec(name=name, blocking=blocking))
        return self

    def add_optimizer(self: ContextApp, name: str) -> ContextApp:  # type: ignore[misc]
        """Register an optimization dimension the improvement loop may tune."""
        known = {"context_budget", "prompt_format", "retrieval_config", "model_routing"}
        if name not in known:
            raise ConfigError(f"unknown optimizer {name!r}; known: {sorted(known)}")
        if name not in self.optimizers:
            self.optimizers.append(name)
        return self

    def add_online_evaluator(  # type: ignore[misc]
        self: ContextApp, metric: str | Callable, *, sample_rate: float = 1.0, name: str | None = None
    ) -> ContextApp:
        """Score a sampled fraction of live runs with ``metric`` after each run
        completes, writing the score as a time series on the metadata store
        (no traffic mirrored anywhere). Scoring runs off the hot path; sampling
        bounds the overhead. The same metric object can gate releases offline
        and act as a runtime guardrail::

            app.add_online_evaluator("answer_relevance", sample_rate=0.1)
            app.add_online_evaluator("goal_accuracy", sample_rate=0.2)
        """
        self.online_evaluators.append(
            OnlineEvaluator(
                metric, name=name, sample_rate=sample_rate, store=self.store, app_name=self.name
            )
        )
        return self

    def add_metric_rail(  # type: ignore[misc]
        self: ContextApp,
        metric: str | Callable,
        *,
        threshold: float,
        direction: str = "output",
        action: str = "block",
        name: str | None = None,
        **params: Any,
    ) -> ContextApp:
        """Use an eval metric as a runtime guardrail. The same metric that gates
        releases offline blocks (or warns on) generations at run time::

            app.add_metric_rail("toxicity", threshold=0.0)
            app.add_metric_rail("answer_relevance", threshold=0.3, action="warn")
        """
        from ..evals.guardrails import metric_guardrail

        metric_name = metric if isinstance(metric, str) else getattr(metric, "__name__", "metric")
        predicate_name = name or f"{metric_name}_guard"
        self.register_rail_predicate(
            predicate_name, metric_guardrail(metric, threshold=threshold, name=predicate_name)
        )
        self.add_rail(
            name=predicate_name,
            kind="custom",
            direction=direction,
            action=action,
            predicate=predicate_name,
            params=params,
        )
        return self

    def experiment(  # type: ignore[misc]
        self: ContextApp,
        name: str,
        *,
        variants: dict[str, dict[str, Any]] | None = None,
        dataset: Dataset | str | None = None,
        metrics: list[str] | None = None,
    ) -> Any:
        """A production-style A/B over prompt/model/config variants of this app,
        compared on eval metrics *and* cost with significance tests. Returns an
        :class:`~vincio.evals.experiments.Experiment` handle; if ``variants`` and
        ``dataset`` are given, every variant is evaluated first::

            exp = app.experiment(
                "prompt_ab",
                variants={"baseline": {}, "concise": {"prompt": concise_spec}},
                dataset=golden, metrics=["goal_accuracy", "cost"],
            )
            exp.compare(); exp.significance("goal_accuracy"); exp.cost()
        """
        from ..evals.experiments import Experiment

        handle = Experiment(self, name, metrics=metrics)
        if variants and dataset is not None:
            for variant, config in variants.items():
                config = config or {}
                handle.run_variant(
                    variant,
                    dataset,
                    model=config.get("model"),
                    prompt=config.get("prompt"),
                    apply=config.get("apply"),
                    params=config.get("params"),
                )
        return handle

    # -- structured output -------------------------------------------------

    def add_output_schema(  # type: ignore[misc]
        self: ContextApp,
        schema: type[BaseModel] | OutputSchema | dict[str, Any],
        *,
        name: str | None = None,
        task_types: list[str] | None = None,
        keywords: list[str] | None = None,
        when: Callable[[str], bool] | None = None,
        priority: int = 100,
    ) -> ContextApp:
        """Register an alternative output schema, routed by task or content.

        The first call creates the schema router; the app's base schema (if
        any) stays the default when no route matches::

            app.add_output_schema(BugReport, keywords=["bug", "crash"])
            app.add_output_schema(BillingIssue, keywords=["invoice", "refund"])
        """
        if self.schema_router is None:
            self.schema_router = SchemaRouter(default=self.output_contract.output_schema())
        self.schema_router.add(
            schema,
            name=name,
            task_types=task_types,
            keywords=keywords,
            when=when,
            priority=priority,
        )
        return self

    def enable_self_correction(  # type: ignore[misc]
        self: ContextApp, *, max_cycles: int = 2, max_cost_usd: float = 0.05, temperature: float = 0.0
    ) -> ContextApp:
        """Turn on bounded validate → critique → repair cycles for failed
        outputs. Structure-only: the critique and repair prompt forbid
        changing factual content, and all validators re-run each cycle."""
        self.self_correction = {
            "max_cycles": max_cycles,
            "max_cost_usd": max_cost_usd,
            "temperature": temperature,
        }
        return self

    def predictor(  # type: ignore[misc]
        self: ContextApp,
        sig: type[Signature],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        prompt_spec: PromptSpec | None = None,
    ) -> Predict:
        """A :class:`~vincio.prompts.signatures.Predict` bound to the app's
        provider and model: ``app.predictor(Triage)(ticket="...")``."""
        return Predict(
            sig,
            provider=self.resolve_provider(),
            model=model or self.model,
            temperature=temperature,
            prompt_spec=prompt_spec,
        )

    # -- task decorator ----------------------------------------------------------------------

    def task(self: ContextApp, cls: type) -> type:  # type: ignore[misc]
        """Configure the app from a task class::

        @app.task
        class Triage:
            objective = "Classify support tickets"
            labels = ["bug", "billing", "feature", "other"]
        """
        objective = getattr(cls, "objective", None)
        labels = getattr(cls, "labels", None)
        rules = list(getattr(cls, "rules", []))
        update: dict[str, Any] = {}
        if objective:
            self.objective = Objective(
                text=objective, task_type=TaskType.CLASSIFICATION if labels else TaskType.GENERAL
            )
            update["objective"] = objective
        if labels:
            update["rules"] = [
                *rules,
                f"Answer with exactly one of these labels: {', '.join(labels)}.",
            ]
            if self.output_contract.schema_def is None:
                schema = OutputSchema.from_json_schema(
                    {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "enum": list(labels)},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reason": {"type": "string"},
                        },
                        "required": ["label", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                    name=cls.__name__,
                )
                self.output_contract = OutputContract.from_schema(schema)
        elif rules:
            update["rules"] = rules
        self.prompt_spec = self.prompt_spec.model_copy(update=update)
        return cls

    # -- execution -------------------------------------------------------------------------------------------

    @staticmethod
    def _coerce_input(
        user_input: str | UserInput,
        *,
        files: list[str] | None,
        tenant_id: str | None,
        user_id: str | None,
        session_id: str | None,
        feature: str | None,
    ) -> UserInput:
        """Normalize the run entry points' input into a fresh UserInput."""
        if isinstance(user_input, str):
            normalized = UserInput(text=user_input)
        else:
            normalized = user_input.model_copy(deep=True)
        if files:
            normalized.files.extend(FileRef(path=f) for f in files)
        if tenant_id is not None:
            normalized.tenant_id = tenant_id
        if user_id is not None:
            normalized.user_id = user_id
        if session_id is not None:
            normalized.session_id = session_id
        if feature is not None:
            normalized.feature = feature
        return normalized

    async def arun(  # type: ignore[misc]
        self: ContextApp,
        user_input: str | UserInput,
        *,
        files: list[str] | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        feature: str | None = None,
        config: RunConfig | None = None,
    ) -> RunResult:
        """Run the full context-engineering pipeline asynchronously → :class:`RunResult`."""
        user_input = self._coerce_input(
            user_input,
            files=files,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            feature=feature,
        )
        engine = getattr(self, "reasoning_engine", None)
        internal = bool(config and config.metadata.get("_universal_reasoning_internal"))
        if engine is not None and not internal:
            result = (await engine.arun(user_input, config=config)).result
        else:
            result = await self._runtime.execute(user_input, config)
        if self.online_evaluators:
            self._spawn_online(result, user_input)
        return result

    def submit(  # type: ignore[misc]
        self: ContextApp,
        user_input: str | UserInput,
        *,
        files: list[str] | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        feature: str | None = None,
        config: RunConfig | None = None,
    ) -> RunHandle:
        """Start a run in the background and return a :class:`RunHandle`.

        ``handle.cancel()`` propagates cooperative cancellation into the run's
        bounded-concurrency groups (retrieval, tools, model); ``await handle``
        (or ``await handle.result()``) yields the :class:`RunResult`. A cancelled
        run is still fully recorded on its trace and audit chain — cancellation
        is identical to the non-streaming path's. Must be called from within a
        running event loop::

            handle = app.submit("Summarize the filing")
            handle.cancel()  # cooperative — the partial run is still recorded
        """
        normalized = self._coerce_input(
            user_input,
            files=files,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            feature=feature,
        )

        async def _run() -> RunResult:
            result = await self._runtime.execute(normalized, config)
            if self.online_evaluators:
                self._spawn_online(result, normalized)
            return result

        task = asyncio.ensure_future(_run())
        return RunHandle(task)

    def run(self: ContextApp, user_input: str | UserInput, **kwargs: Any) -> RunResult:  # type: ignore[misc]
        """Run the full context-engineering pipeline synchronously → :class:`RunResult`."""
        return run_sync(self._run_and_flush(user_input, **kwargs))

    async def _run_and_flush(self: ContextApp, user_input: str | UserInput, **kwargs: Any) -> RunResult:  # type: ignore[misc]
        result = await self.arun(user_input, **kwargs)
        await self.aflush_online()
        return result

    async def astream(  # type: ignore[misc]
        self: ContextApp,
        user_input: str | UserInput,
        *,
        files: list[str] | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        feature: str | None = None,
        config: RunConfig | None = None,
    ) -> AsyncIterator[RunStreamEvent]:
        """Run the full pipeline with end-to-end streaming.

        Yields :class:`RunStreamEvent` items — pipeline stages, model text
        deltas, incremental partial-JSON output, tool activity — ending with
        a ``done`` event that carries the final :class:`RunResult`::

            async for event in app.astream("Summarize the refund policy"):
                if event.type == "text_delta":
                    print(event.text, end="", flush=True)
                elif event.type == "done":
                    result = event.result
        """
        user_input = self._coerce_input(
            user_input,
            files=files,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            feature=feature,
        )
        config = config or RunConfig()
        config = config.model_copy(update={"stream": True})
        async for event in self._runtime.execute_stream(user_input, config):
            yield event

    def stream(self: ContextApp, user_input: str | UserInput, **kwargs: Any) -> Iterator[RunStreamEvent]:  # type: ignore[misc]
        """Synchronous streaming convenience: collects the async event
        stream and yields the events in order (like provider.stream_sync)."""

        async def collect() -> list[RunStreamEvent]:
            return [event async for event in self.astream(user_input, **kwargs)]

        yield from run_sync(collect())

    # -- agents -------------------------------------------------------------------------------------------------

    def _build_executor(  # type: ignore[misc]
        self: ContextApp,
        *,
        tools: list[str | Callable] | None = None,
        planner: str = "dag",
        max_steps: int = 8,
        model: str | None = None,
        system_prompt_extra: str = "",
        restrict_tools: bool = False,
        domain: Any | None = None,
        cost_aware_models: list[str] | None = None,
    ) -> AgentExecutor:
        tool_names: list[str] = []
        for tool in tools or []:
            self.add_tool(tool)
            tool_names.append(
                tool if isinstance(tool, str) else getattr(tool, "__name__", str(tool))
            )
        planner_mode = {
            "dag": "static",
            "static": "static",
            "dynamic": "dynamic",
            "react": "react",
            "direct": "direct",
            "plan_and_execute": "plan_and_execute",
            "hierarchical": "hierarchical",
            "htn": "hierarchical",
        }.get(planner, "static")
        provider = self.resolve_provider()
        agent_model = model or self.model
        llm_planning = planner_mode in ("dynamic", "plan_and_execute", "hierarchical")
        planner_obj = Planner(
            mode=planner_mode,  # type: ignore[arg-type]
            provider=provider if llm_planning else None,
            model=agent_model if llm_planning else None,
            max_steps=max_steps,
            domain=domain,
        )
        # Cost-aware action selection over the candidate models, reading the
        # data-driven model registry's pricing and the live budget.
        selector = None
        if cost_aware_models:
            from ..agents.selection import CostAwareSelector

            selector = CostAwareSelector(cost_aware_models, events=self.events)
        retrieve_fn = None
        if self.retrieval is not None:
            engine = self.retrieval

            async def retrieve_fn(query: str) -> list[EvidenceItem]:
                result = await engine.retrieve(query, top_k=self.config.retrieval.top_k)
                return result.evidence

        from ..output.validators import OutputValidator

        validator = None
        if self.output_contract.schema_def is not None or self.output_contract.require_citations:
            validator = OutputValidator(
                self.output_contract,
                semantic_validators=self.semantic_validators,
                policy_engine=self.policy_engine,
                repairer=self.repairer,
            )
        system_prompt = self.prompt_compiler.compile(
            self.prompt_spec, variables=self.prompt_variables
        ).system_text
        if system_prompt_extra:
            system_prompt = (
                f"{system_prompt}\n\n{system_prompt_extra}"
                if system_prompt
                else system_prompt_extra
            )
        # restrict_tools (crew members): least privilege — only the tools named
        # for this executor, never the app-wide enabled set.
        enabled = tool_names if restrict_tools else self.enabled_tools
        return AgentExecutor(
            provider,
            model=agent_model,
            planner=planner_obj,
            tool_runtime=self.tool_runtime if enabled else None,
            tool_specs=self.tool_registry.specs(enabled) if enabled else [],
            retrieve_fn=retrieve_fn,
            output_validator=validator,
            tracer=self.tracer,
            cost_tracker=self.cost_tracker,
            cost_ledger=self.cost_ledger,
            events=self.events,
            selector=selector,
            system_prompt=system_prompt,
        )

    def agent(  # type: ignore[misc]
        self: ContextApp,
        *,
        name: str | None = None,
        tools: list[str | Callable] | None = None,
        planner: str = "dag",
        max_steps: int = 8,
        evaluator: str | None = None,
        model: str | None = None,
        domain: Any | None = None,
        cost_aware_models: list[str] | None = None,
    ) -> _AgentHandle:
        """Build a bounded agent over the app's tools, memory, and retrieval.

        ``planner`` selects the planning shape (``dag`` / ``dynamic`` / ``react``
        / ``direct`` / ``plan_and_execute`` / ``hierarchical``). Pass an
        :class:`~vincio.agents.HTNDomain` as ``domain`` to drive deterministic
        hierarchical decomposition, and ``cost_aware_models`` (cheapest→strongest)
        to enable cost-aware action selection. In-place plan repair is on by
        default. Tool failures, validation contradictions, and budget shocks are
        repaired in place rather than restarting the run.
        """
        if evaluator is not None:
            self.add_evaluator(evaluator)
        executor = self._build_executor(
            tools=tools,
            planner=planner,
            max_steps=max_steps,
            model=model,
            domain=domain,
            cost_aware_models=cost_aware_models,
        )
        return _AgentHandle(self, executor, max_steps)

    def assistant(  # type: ignore[misc]
        self: ContextApp,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
        memory_writeback: bool = True,
        auto_approve: list[str] | None = None,
        on_approval: Any | None = None,
        feature: str | None = "assistant",
    ) -> Assistant:
        """Open a conversational, session-aware :class:`~vincio.assistant.Assistant`.

        A thin multi-turn layer over this app: every turn is still a full
        :meth:`run` (retrieval, grounding, validation, rails, budget, trace,
        audit all apply), threaded under one ``session_id`` with session-scoped
        memory write-back and an approval surface for write tools::

            chat = app.assistant(user_id="u-1")
            print(chat.send("How do I reset my password?").text)
            print(chat.send("And change my email?").text)   # remembers the thread

        Write tools are denied by default and surfaced as pending approvals;
        ``auto_approve=[...]`` pre-allows trusted tools, or pass ``on_approval``
        for an interactive decision.
        """
        from ..assistant import Assistant

        return Assistant(
            self,
            user_id=user_id,
            tenant_id=tenant_id,
            session_id=session_id,
            memory_writeback=memory_writeback,
            auto_approve=auto_approve,
            on_approval=on_approval,
            feature=feature,
        )

    def research(  # type: ignore[misc]
        self: ContextApp, question: str, *, objective: str = "", web: bool | dict = False, **kwargs: Any
    ):
        """Run the deep-research loop: search → read → reflect → verify →
        synthesize, emitting a cited, budget-bounded, eval-scored report.

        Composes the query-understanding planners, the retrieval engine, the
        grounded-fact extractor, and the cited-report builder into one
        :class:`~vincio.agents.research.ResearchAgent`. Reads from the sources
        you index (``app.add_source(...)``); pass ``web=True`` to make it
        **web-backed** — the open web is searched for the question and the
        governed browsing tools are enabled, so the report is grounded in fresh
        pages that each re-derive offline from their snapshots::

            report = app.research("What shipped in Python 3.13?", web=True)
            report.answer, report.metrics["citation_coverage"], report.sources
        """
        from ..agents.research import ResearchAgent

        if web:
            self._enable_research_web(question, web)
        return ResearchAgent(self, **kwargs).run(question, objective=objective)

    async def aresearch(  # type: ignore[misc]
        self: ContextApp, question: str, *, objective: str = "", web: bool | dict = False, **kwargs: Any
    ):
        """Async :meth:`research`."""
        from ..agents.research import ResearchAgent

        if web:
            self._enable_research_web(question, web)
        return await ResearchAgent(self, **kwargs).arun(question, objective=objective)

    def _enable_research_web(self: ContextApp, question: str, web: bool | dict) -> None:  # type: ignore[misc]
        """Seed a web-search source for *question* and enable the browsing tools,
        so the research agent reads fresh pages and can iterate with the model."""
        from ..connectors import connect

        options = web if isinstance(web, dict) else {}
        if getattr(self, "web_browser", None) is None:
            self.use_web_search(
                preset=options.get("preset", "research"),
                backend=options.get("backend"),
                client=options.get("client"),
            )
        connector = connect(
            "websearch",
            queries=[question, *options.get("queries", [])],
            backend=options.get("backend"),
            client=options.get("client"),
            max_results=options.get("max_results", 4),
        )
        self.add_source("web", connector=connector, retrieval=options.get("retrieval", "hybrid"))

    def reasoning(self: ContextApp, policy: Any | None = None, **kwargs: Any):  # type: ignore[misc]
        """Build a :class:`~vincio.agents.reasoning.ReasoningController`.

        The controller sets the thinking effort and a hard-ceilinged thinking
        budget per step from the task classification and the live budget, and
        steps effort down when a thinking prefix is already warm in its
        :class:`~vincio.caching.ReasoningTraceCache` (created here unless one is
        passed as ``trace_cache=``). Pass it to :meth:`use_reasoning_controller`
        to have the runtime apply it on every run, or call ``.decide(...)``
        directly::

            ctl = app.reasoning()
            d = ctl.decide(task=routed.task, text=question, remaining_output_tokens=4096)
            app.run(question, config=RunConfig(reasoning_effort=d.effort))
        """
        from ..agents.reasoning import ReasoningController, ReasoningPolicy
        from ..caching import ReasoningTraceCache

        if policy is None:
            policy = ReasoningPolicy()
        trace_cache = kwargs.pop("trace_cache", None) or ReasoningTraceCache()
        return ReasoningController(policy, trace_cache=trace_cache, **kwargs)

    def use_reasoning_controller(self: ContextApp, controller: Any | None = None, **kwargs: Any) -> ContextApp:  # type: ignore[misc]
        """Install a reasoning controller so the runtime sets effort per run.

        With a controller installed, a run that does not pin ``reasoning_effort``
        on its :class:`~vincio.core.types.RunConfig` has the effort and thinking
        budget chosen by the controller (only for reasoning-capable models), and
        each paid thinking prefix is recorded so a re-ask reuses it. ``controller``
        may be a :class:`~vincio.agents.reasoning.ReasoningController`, a
        :class:`~vincio.agents.reasoning.ReasoningPolicy`, or ``None`` (default
        policy). Returns ``self`` for chaining."""
        from ..agents.reasoning import ReasoningController

        if not isinstance(controller, ReasoningController):
            controller = self.reasoning(controller, **kwargs)
        self.reasoning_controller = controller
        return self

    @experimental(since="7.10")
    def use_reasoning_engine(  # type: ignore[misc]
        self: ContextApp,
        engine: Any | None = None,
        *,
        policy: Any | None = None,
        web: bool | dict[str, Any] = False,
        **kwargs: Any,
    ) -> ContextApp:
        """Install adaptive universal reasoning on ordinary ``run`` / ``arun``.

        Easy requests retain the normal single-pass path. Hard, mathematical,
        logical, tool-dependent or freshness-sensitive requests receive a
        provider-independent plan, bounded candidate/verification/correction
        passes, and (when enabled) governed web evidence. Native reasoning is
        used when available but is never required. Non-English and uncertain-
        language requests are semantically classified by the configured model,
        so routing follows model language support without a finite locale list;
        deterministic policy remains authoritative. Pass ``web=True`` to enable
        Vincio's universal web tools with their default policy, or a dict of
        :meth:`use_web_search` options::

            app.use_reasoning_engine(web=True)
            result = app.run("Compare the latest two Python releases")
            result.metadata["universal_reasoning"]

        The engine records strategy and verifier receipts, never model scratch
        work or hidden chain-of-thought. The API is experimental in 7.10.
        """
        from ..agents.universal_reasoning import (
            UniversalReasoningEngine,
            UniversalReasoningPolicy,
        )

        if web and getattr(self, "web_browser", None) is None:
            options = web if isinstance(web, dict) else {}
            self.use_web_search(**options)
        if engine is None:
            if policy is not None and kwargs:
                raise InputError("pass either policy or policy keyword fields, not both")
            if policy is None:
                policy = UniversalReasoningPolicy(**kwargs)
            engine = UniversalReasoningEngine(self, policy=policy)
        elif kwargs or policy is not None:
            raise InputError("policy/kwargs apply only when building the universal reasoning engine")
        self.reasoning_engine = engine
        return self

    @experimental(since="7.10")
    async def areason(  # type: ignore[misc]
        self: ContextApp,
        user_input: str | UserInput,
        *,
        config: RunConfig | None = None,
        policy: Any | None = None,
    ) -> Any:
        """Run universal reasoning once and return its full reasoning receipt.

        Unlike :meth:`use_reasoning_engine`, this explicit verb does not change
        subsequent ``app.run`` calls. The returned
        :class:`~vincio.agents.universal_reasoning.UniversalReasoningResult`
        contains the normal validated :class:`RunResult` as ``.result`` plus the
        adaptive assessment, plan, pass receipts and content-bound web evidence.
        """
        from ..agents.universal_reasoning import UniversalReasoningEngine

        engine = UniversalReasoningEngine(self, policy=policy)
        normalized = self._coerce_input(
            user_input,
            files=None,
            tenant_id=None,
            user_id=None,
            session_id=None,
            feature=None,
        )
        outcome = await engine.arun(normalized, config=config)
        if self.online_evaluators:
            self._spawn_online(outcome.result, normalized)
        return outcome

    @experimental(since="7.10")
    def reason(self: ContextApp, user_input: str | UserInput, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Synchronous :meth:`areason`."""
        return run_sync(self._reason_and_flush(user_input, **kwargs))

    async def _reason_and_flush(  # type: ignore[misc]
        self: ContextApp, user_input: str | UserInput, **kwargs: Any
    ) -> Any:
        outcome = await self.areason(user_input, **kwargs)
        await self.aflush_online()
        return outcome

    def use_context_governor(  # type: ignore[misc]
        self: ContextApp,
        governor: Any | None = None,
        *,
        evidence_store: Any | None = None,
        blob_store: Any | None = None,
        **kwargs: Any,
    ) -> ContextApp:
        """Install a long-horizon :class:`~vincio.context.ContextGovernor`.

        For million-token, multi-day, multi-session runs: the governor holds a
        :class:`~vincio.context.ContextBudget` (live tokens, resident bytes,
        KV-cache footprint), decays stale spans within the run, and compacts the
        coldest ones into the memory OS — paging the full text back on demand —
        so the live context stays bounded as the horizon grows. ``governor`` may
        be a :class:`~vincio.context.ContextGovernor`, a
        :class:`~vincio.context.ContextBudget` (wrapped in a default governor that
        writes compaction summaries into this app's memory engine), or ``None``
        (an unbounded-budget governor). Feed each run's packet with
        :meth:`govern_packet`. Returns ``self`` for chaining::

            app.use_context_governor(ContextBudget(max_tokens=8000))
            for turn in conversation:
                result = app.run(turn)
                app.govern_packet(result)          # admits result.evidence
            report = app.context_budget_report()

        By default a compacted span's full text pages back from a process-local
        store, which a fresh worker or a restart cannot read. Pass a ``blob_store``
        (any :class:`~vincio.storage.base.BlobStore`) and the compactor backs cold
        spans with a content-addressed
        :class:`~vincio.context.evidence_store.BlobEvidenceStore`, so a multi-day
        run survives a restart and a multi-process run pages the same cold text
        back across workers — the cross-process path slim packets use. Pass a ready
        :class:`~vincio.context.evidence_store.EvidenceStore` as ``evidence_store``
        to supply your own. These apply only when building the default governor;
        a fully-built ``governor`` carries its own store::

            from vincio.storage.base import FileBlobStore
            app.use_context_governor(
                ContextBudget(max_tokens=8000), blob_store=FileBlobStore("spans/")
            )
        """
        from ..context.evidence_store import BlobEvidenceStore
        from ..context.longhorizon import (
            ContextBudget,
            ContextCompactor,
            ContextGovernor,
        )

        if isinstance(governor, ContextGovernor):
            if evidence_store is not None or blob_store is not None:
                raise InputError(
                    "evidence_store / blob_store apply only when building a governor; "
                    "the passed ContextGovernor already carries its own compactor store"
                )
            self.context_governor = governor
            return self
        budget = governor if isinstance(governor, ContextBudget) else ContextBudget(**kwargs)
        store = evidence_store
        if store is None and blob_store is not None:
            store = BlobEvidenceStore(blob_store)
        compactor = ContextCompactor(
            memory=getattr(self, "memory", None), owner_id=self.name, store=store
        )
        self.context_governor = ContextGovernor(budget, compactor=compactor)
        return self

    def govern_packet(self: ContextApp, source: Any) -> Any:  # type: ignore[misc]
        """Admit a run's evidence into the installed long-horizon governor.

        The multi-session hook: after each :meth:`run`, pass the
        :class:`~vincio.core.types.RunResult` (or a
        :class:`~vincio.context.ContextPacket` from :meth:`compile`) here so the
        long-horizon footprint stays bounded across the conversation. Returns the
        governor's :class:`~vincio.context.ContextBudgetReport`, or ``None`` if no
        governor is installed."""
        if self.context_governor is None or source is None:
            return None
        if hasattr(source, "evidence_items"):  # a ContextPacket
            self.context_governor.admit_packet(source)
        elif hasattr(source, "evidence"):  # a RunResult
            self.context_governor.admit_evidence(source.evidence)
        return self.context_governor.report()

    def context_budget_report(self: ContextApp) -> Any:  # type: ignore[misc]
        """The installed governor's live context-budget report (or ``None``).

        The residency analogue of :meth:`cost_report`: live tokens, resident
        bytes, KV-cache footprint, compactions, and intra-run decay for the
        long-horizon run."""
        if self.context_governor is None:
            return None
        return self.context_governor.report()

    def use_semantic_cache(self: ContextApp, cache: Any | None = None, **kwargs: Any) -> ContextApp:  # type: ignore[misc]
        """Install a learned semantic cache so near-misses are served from cache.

        With a cache installed, a run whose request misses the exact-match
        response cache is checked against recent answers in the same scope (model
        + stable prompt head) and schema; a semantically-equivalent one is served
        for free **only above the calibrated acceptance threshold** — never below
        the floor. ``cache`` may be a
        :class:`~vincio.caching.LearnedSemanticCache`, a
        :class:`~vincio.caching.SemanticCachePolicy`, or ``None`` (a policy built
        from this app's ``cache`` config). The cache shares the app embedder.
        Calibrate it from traces before trusting near-misses, and gate it with a
        :class:`~vincio.caching.SemanticCacheGate`. Returns ``self`` for
        chaining::

            app.use_semantic_cache()
            app.semantic_cache.calibrate(examples)   # fit the threshold
            report = app.semantic_cache_report()
        """
        from ..caching import LearnedSemanticCache, SemanticCachePolicy

        if isinstance(cache, LearnedSemanticCache):
            self.semantic_cache = cache
        else:
            if isinstance(cache, SemanticCachePolicy):
                policy = cache
            else:
                cfg = self.config.cache
                policy = SemanticCachePolicy(
                    enabled=True,
                    threshold=cfg.semantic_threshold,
                    target_precision=cfg.semantic_cache_target_precision,
                    min_floor=cfg.semantic_cache_min_floor,
                    ttl_s=float(cfg.ttl_s),
                    max_entries=cfg.semantic_cache_max_entries,
                    max_resident_bytes=cfg.semantic_cache_max_resident_bytes,
                    **kwargs,
                )
            self.semantic_cache = LearnedSemanticCache(self.embedder, policy=policy)
        self.cache_invalidation.register_semantic(self.semantic_cache)
        return self

    def semantic_cache_report(self: ContextApp) -> Any:  # type: ignore[misc]
        """The installed semantic cache's stats (or ``None``).

        Hit-rate, near-misses rejected, output tokens saved, the calibrated
        threshold in force, and the cache's resident footprint — the savings the
        cache realized, alongside the $0-billed calls it produced in the cost
        report."""
        if self.semantic_cache is None:
            return None
        return self.semantic_cache.stats()

    def use_kv_prefix_reuse(self: ContextApp, pool: Any | None = None, **kwargs: Any) -> ContextApp:  # type: ignore[misc]
        """Install a KV-prefix pool so cross-request stable-prefix reuse is tracked.

        With a pool installed, each run's compiled stable prefix is recorded; a
        later request that shares the same head (same ``prompt_spec_hash`` on the
        same model) is reported as a reuse, with the serving-engine KV bytes the
        shared head avoids recomputing. ``pool`` may be a
        :class:`~vincio.caching.KVPrefixPool` or ``None`` (one built from this
        app's ``cache`` config). Returns ``self`` for chaining::

            app.use_kv_prefix_reuse()
            for q in questions:
                app.run(q)
            report = app.kv_prefix_report()
        """
        from ..caching import KVPrefixPool

        if isinstance(pool, KVPrefixPool):
            self.kv_prefix_pool = pool
        else:
            cfg = self.config.cache
            self.kv_prefix_pool = KVPrefixPool(
                kv_bytes_per_token=cfg.kv_bytes_per_token,
                max_entries=cfg.kv_prefix_max_entries,
                max_resident_bytes=cfg.kv_prefix_max_resident_bytes,
                **kwargs,
            )
        return self

    def kv_prefix_report(self: ContextApp) -> Any:  # type: ignore[misc]
        """The installed KV-prefix pool's reuse report (or ``None``).

        Distinct stable heads tracked, total requests seen, how many reused a
        warm head, and the cumulative serving-engine KV those reuses avoided
        recomputing — the cross-request analogue of the prompt-cache hit rate."""
        if self.kv_prefix_pool is None:
            return None
        return self.kv_prefix_pool.report()

    async def atest_time_search(  # type: ignore[misc]
        self: ContextApp,
        user_input: str | UserInput,
        *,
        verifier: Any = None,
        strategy: str = "best_of_n",
        n: int = 4,
        config: RunConfig | None = None,
        vary: str = "seed",
        generate: Any | None = None,
        budget: Any | None = None,
        **budget_kwargs: Any,
    ):
        """Run verifier-guided test-time search over this app.

        Draws ``n`` candidates by re-running the app with a varied ``seed`` (or
        ``temperature``), scores them with ``verifier`` (any
        :class:`~vincio.evals.judges.Judge` / ensemble, any
        :class:`~vincio.optimize.rewards.VerifiableReward` / ``RewardModel``, or a
        callable), and returns a
        :class:`~vincio.optimize.test_time.SearchResult`. ``strategy`` is
        ``"best_of_n"`` (verifier required) or ``"self_consistency"`` (verifier
        optional). Pass a custom ``generate(index)`` to search over something
        other than a plain re-run::

            from vincio.optimize import RewardVerifier
            res = await app.atest_time_search(q, verifier=reward, n=8)
            res.output, res.confidence, res.stop_reason
        """
        from ..optimize.test_time import SearchBudget, TestTimeSearch

        if generate is None:
            base = config

            def generate(index: int):  # noqa: ANN202 — local closure
                update: dict[str, Any] = {"seed": index}
                if vary == "temperature":
                    update = {"temperature": round(0.2 * index, 4)}
                cfg = base.model_copy(update=update) if base is not None else RunConfig(**update)
                return self.arun(user_input, config=cfg)

        if budget is None:
            budget = SearchBudget(max_candidates=n, **budget_kwargs)
        search = TestTimeSearch(generate, verifier=verifier, budget=budget)
        if strategy == "best_of_n":
            return await search.best_of_n(n)
        if strategy == "self_consistency":
            return await search.self_consistency(n)
        raise InputError(
            f"unknown test-time search strategy {strategy!r}; "
            f"expected 'best_of_n' or 'self_consistency'"
        )

    def test_time_search(self: ContextApp, user_input: str | UserInput, **kwargs: Any):  # type: ignore[misc]
        """Synchronous :meth:`atest_time_search`."""
        return run_sync(self.atest_time_search(user_input, **kwargs))

    def crew(  # type: ignore[misc]
        self: ContextApp,
        name: str = "crew",
        *,
        members: list[AgentRole | dict[str, Any]],
        process: str = "sequential",
        tools: list[str | Callable] | None = None,
        planner: str = "direct",
        max_steps: int = 8,
        max_rounds: int = 4,
        model: str | None = None,
    ) -> Crew:
        """Build a multi-agent crew over a shared blackboard.

        ``members`` are :class:`AgentRole` objects or dicts with the role
        fields (``name``, ``description``, ``goal``, ``keywords``,
        ``budget_fraction``) plus optional per-member ``tools`` / ``planner``
        / ``model`` / ``max_steps`` overrides. The hierarchical process uses
        the app's provider as the crew manager (deterministic fallback when
        offline)::

            crew = app.crew(members=[
                {"name": "researcher", "goal": "gather evidence", "keywords": ["find"]},
                {"name": "writer", "goal": "draft the report"},
            ])
            result = crew.run("Summarize Q3 refund trends")
        """
        crew = Crew(
            name,
            process=process,  # type: ignore[arg-type]
            blackboard=Blackboard(event_bus=self.events),
            tracer=self.tracer,
            manager_provider=self.resolve_provider() if process == "hierarchical" else None,
            manager_model=model or self.model,
            max_rounds=max_rounds,
            cost_tracker=self.cost_tracker,
            cost_ledger=self.cost_ledger,
        )
        role_fields = set(AgentRole.model_fields)
        override_fields = {"tools", "planner", "model", "max_steps"}
        for spec in members:
            overrides: dict[str, Any] = {}
            if isinstance(spec, dict):
                unknown = set(spec) - role_fields - override_fields
                if unknown:
                    raise AgentEngineError(
                        f"unknown crew member fields {sorted(unknown)}; "
                        f"expected {sorted(role_fields | override_fields)}"
                    )
                overrides = spec
                role = AgentRole(**{k: v for k, v in spec.items() if k in role_fields})
            else:
                role = spec
            executor = self._build_executor(
                tools=overrides.get("tools", tools),
                planner=overrides.get("planner", planner),
                max_steps=overrides.get("max_steps", max_steps),
                model=overrides.get("model", model),
                system_prompt_extra=f"You are {role.name}. {role.description}".strip(),
                restrict_tools=True,
            )
            crew.add(role, executor)
        return crew

    def graph(  # type: ignore[misc]
        self: ContextApp,
        name: str = "graph",
        *,
        state_schema: type[BaseModel] | None = None,
        reducers: dict[str, Callable[[Any, Any], Any]] | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> StateGraph:
        """A durable :class:`StateGraph` bound to the app's tracer and
        metadata store: checkpoints persist wherever the app's runs do, so
        threads survive restarts when the store is SQLite/Postgres."""
        graph = StateGraph(name, state_schema=state_schema, reducers=reducers, defaults=defaults)
        graph.default_tracer = self.tracer
        graph.default_checkpointer = Checkpointer(self.store)
        return graph

    # -- workflows ------------------------------------------------------------------------------------------------

    def workflow(self: ContextApp, name: str) -> Workflow:  # type: ignore[misc]
        """Create a deterministic :class:`Workflow` builder bound to this app's tracer."""
        return Workflow(name, tracer=self.tracer)
