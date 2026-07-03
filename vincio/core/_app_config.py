"""Configuration, rails, cost & reliability, and provider-rotation verbs — a private mixin of
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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..providers import build_provider
from ..providers.base import ModelProvider, run_sync
from ..providers.cache_strategy import PromptCacheStrategy
from ..security.policy import PolicyEngine
from ..security.rails import Rail
from .errors import (
    ConfigError,
)
from .types import (
    Example,
    Instruction,
    Objective,
    RunConfig,
    RunResult,
    UserInput,
)

if TYPE_CHECKING:
    from ..optimize.routing import ModelCascade
    from ..prompts.templates import PromptSpec
    from .app import ContextApp
    from .types import Instruction, Objective


class _ConfigVerbs:
    """Configuration, rails, cost & reliability, and provider-rotation verbs. Mixed into :class:`~vincio.core.app.ContextApp`."""

    if TYPE_CHECKING:
        # ContextApp state this mixin's verbs assign. mypy would otherwise
        # attribute the unannotated ``self.X = ...`` assignments to this class
        # and clash with ContextApp.__init__; the declarations (type-checking
        # only, no runtime effect) keep the split typing identical to the
        # monolith's.
        _cascade_confidence: Callable[[Any], float] | None
        _provider_instance: ModelProvider | None
        cascade: ModelCascade | None
        energy_accounting_enabled: bool
        instructions: list[Instruction]
        model: str
        objective: Objective | None
        policy_engine: PolicyEngine
        prompt_cache: PromptCacheStrategy | None
        prompt_spec: PromptSpec


    # -- public configuration API ----------------------------------------------

    def configure(  # type: ignore[misc]
        self: ContextApp,
        *,
        objective: str | None = None,
        role: str | None = None,
        rules: list[str] | None = None,
        soft_rules: list[str] | None = None,
        definitions: dict[str, str] | None = None,
        examples: list[Example] | None = None,
        citation_policy: str | None = None,
        insufficient_evidence_behavior: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> ContextApp:
        """Configure the prompt and run defaults declaratively (objective, role, rules, …)."""
        update: dict[str, Any] = {}
        if objective is not None:
            update["objective"] = objective
            self.objective = Objective(text=objective)
        if role is not None:
            update["role"] = role
        if rules is not None:
            update["rules"] = rules
            self.instructions = [Instruction(text=r) for r in rules]
        if soft_rules is not None:
            update["soft_rules"] = soft_rules
        if definitions is not None:
            update["definitions"] = definitions
        if examples is not None:
            update["examples"] = examples
        if citation_policy is not None:
            update["citation_policy"] = citation_policy
        if insufficient_evidence_behavior is not None:
            update["insufficient_evidence_behavior"] = insufficient_evidence_behavior
        self.prompt_spec = self.prompt_spec.model_copy(update=update)
        if variables:
            self.prompt_variables.update(variables)
        return self

    def set_policy(self: ContextApp, name: str, value: Any) -> ContextApp:  # type: ignore[misc]
        """Set a run policy (e.g. ``answer_only_from_sources``)."""
        self.policies.set(name, value)
        if name == "answer_only_from_sources" and value:
            self.policies.require_citations = True
            self.output_contract.require_citations = True
            if not self.prompt_spec.citation_policy:
                self.prompt_spec = self.prompt_spec.model_copy(
                    update={
                        "rules": [
                            *self.prompt_spec.rules,
                            "Use only the provided sources to answer.",
                        ],
                        "citation_policy": "Cite evidence IDs in square brackets for every claim.",
                        "insufficient_evidence_behavior": "If the sources do not contain the answer, say so explicitly.",
                    }
                )
        if name == "require_citations":
            self.output_contract.require_citations = bool(value)
        self.policy_engine = PolicyEngine(
            self.policies,
            pii_detector=self._pii_detector,
            rails=self.rail_engine,
            egress_dlp=self.config.security.egress_dlp,
        )
        self.events.emit("policy.changed", {"policy": name})
        return self

    # -- rails ------------------------------------------------------------------

    def add_rail(self: ContextApp, rail: Rail | None = None, **kwargs: Any) -> ContextApp:  # type: ignore[misc]
        """Add a programmable input/output rail (topic, format, safety, custom).

        Rails are evaluated by the deterministic policy engine before and
        after every generation::

            app.add_rail(name="no_competitors", kind="topic", direction="output",
                         blocked_topics=["acme corp"])
            app.add_rail(name="no_secrets", kind="safety", detectors=["secrets", "pii"])
        """
        self.rail_engine.add(rail if rail is not None else Rail(**kwargs))
        return self

    def register_rail_predicate(  # type: ignore[misc]
        self: ContextApp, name: str, predicate: Callable[[str, dict[str, Any]], Any]
    ) -> ContextApp:
        """Register a custom rail predicate: ``(text, params) -> falsy | message``."""
        self.rail_engine.register(name, predicate)
        return self

    def edge_runtime(self: ContextApp, profile: Any | None = None) -> Any:  # type: ignore[misc]
        """Build a bounded, in-process edge runtime that shares this app's rails.

        Returns an :class:`~vincio.edge.runtime.EdgeRuntime` — the dependency-free
        compile/score/rail/pack core packaged for a constrained or browser/WASM
        target — seeded with the app's configured rails so the edge path enforces
        the same deterministic safety the server does. ``profile`` defaults to the
        bounded edge-worker :class:`~vincio.edge.profile.EdgeProfile`. The runtime
        holds no provider, store, or tracer; it runs the identical context
        engineering at the edge, offline::

            edge = app.edge_runtime()
            result = edge.run("Summarize the renewal terms")
        """
        from ..edge import EdgeRuntime

        return EdgeRuntime(profile, rails=list(self.rail_engine.rails))

    # -- cost & reliability -----------------------------------------------

    def enable_prompt_caching(  # type: ignore[misc]
        self: ContextApp, *, ttl: str = "5m", min_prefix_tokens: int = 1024
    ) -> ContextApp:
        """Turn on provider-aware prompt caching (default on).

        For providers with explicit breakpoints (Anthropic) the compiler's
        stable prefix gets a ``cache_control`` breakpoint with the chosen
        ``ttl`` ("5m" or "1h") when it is at least ``min_prefix_tokens`` long;
        for auto-cache providers (OpenAI/Gemini) the stable→volatile ordering
        already maximizes hits. Cache-hit rate is recorded on every model
        span::

            app.enable_prompt_caching(ttl="1h")  # long-lived stable context
        """
        self.prompt_cache = PromptCacheStrategy(
            enabled=True,
            ttl=ttl,  # type: ignore[arg-type]
            min_prefix_tokens=min_prefix_tokens,
        )
        return self

    def use_cascade(  # type: ignore[misc]
        self: ContextApp,
        models: list[str] | None = None,
        *,
        rungs: list[Any] | None = None,
        min_confidence: float = 0.5,
        max_escalations: int | None = None,
        confidence: Callable[[Any], float] | None = None,
    ) -> ContextApp:
        """Route runs through a cheap→strong model cascade at run time.

        A run starts on the cheapest model and escalates to the next only when a
        response's confidence falls below the rung threshold (default: a clean,
        schema-valid stop is confident; a truncated/filtered/unparseable answer
        is not). Pass a custom ``confidence`` callable ``(ModelResponse) -> float``
        to drive escalation from your own metric. The offline routing optimizer
        keeps tuning the thresholds. An explicit per-run ``config.model`` or a
        budget degrade overrides the cascade; streaming runs (``astream``) buffer
        each rung and stream the accepted (escalated) answer, never a discarded
        cheap attempt::

            app.use_cascade(["gpt-5.2-mini", "gpt-5.2"])
            app.use_cascade(rungs=[{"model": "haiku", "min_confidence": 0.6}, {"model": "opus"}])
        """
        from ..optimize.routing import CascadeRung, ModelCascade

        if rungs is not None:
            parsed: list[CascadeRung] = []
            for rung in rungs:
                if isinstance(rung, CascadeRung):
                    parsed.append(rung)
                elif isinstance(rung, dict):
                    parsed.append(CascadeRung(**rung))
                else:
                    parsed.append(CascadeRung(model=str(rung)))
            self.cascade = ModelCascade(rungs=parsed, max_escalations=max_escalations)
        elif models:
            self.cascade = ModelCascade.from_models(
                list(models), min_confidence=min_confidence, max_escalations=max_escalations
            )
        else:
            raise ConfigError("use_cascade requires models=[...] or rungs=[...]")
        # Residency: every rung the cascade may escalate into must be an
        # allowed region — closes the gap where an escalation egressed below the
        # run's residency choke point.
        for rung in self.cascade.rungs:
            self._enforce_model_residency(rung.model, provider=rung.provider)
        self._cascade_confidence = confidence
        return self

    # -- provider/model rotation & swap regression ------------------------

    def _base_provider(self: ContextApp) -> ModelProvider:  # type: ignore[misc]
        """The raw model provider (the current instance, or one built from config)
        — the inner provider that rotation wrappers compose over."""
        if self._provider_instance is not None:
            return self._provider_instance
        return build_provider(self._provider_name, self.config.provider)

    def _pinned_models(self: ContextApp) -> list[str]:  # type: ignore[misc]
        """Every model id this app currently pins (default model + cascade rungs)."""
        models: set[str] = set()
        if self.model:
            models.add(self.model)
        cascade = getattr(self, "cascade", None)
        if cascade is not None:
            models.update(rung.model for rung in cascade.rungs)
        return sorted(m for m in models if m)

    def use_router(  # type: ignore[misc]
        self: ContextApp,
        models: list[str],
        *,
        strategy: str = "cheapest",
        budget_usd: float | None = None,
        guard_capabilities: bool = True,
        provider: ModelProvider | None = None,
    ) -> ContextApp:
        """Route each run to the cheapest / fastest / least-busy *capable* model.

        A registry-backed :class:`~vincio.optimize.routing.Router` becomes the
        app's provider: before every call it filters ``models`` to those that can
        serve the request (capability guard) and picks by ``strategy``, optionally
        **downgrading** to honor a per-request ``budget_usd``. Each pick is emitted
        as a ``model.routed`` event on the app's bus::

            app.use_router(["gpt-5.2-nano", "gpt-5.2-mini", "gpt-5.2"], strategy="cheapest")
        """
        from ..optimize.routing import Router

        if not models:
            raise ConfigError("use_router requires at least one model")
        for candidate in models:  # residency: every routable model must be allowed
            self._enforce_model_residency(candidate)
        base = provider or self._base_provider()
        self._provider_instance = Router(
            [(base, m) for m in models],
            strategy=strategy,  # type: ignore[arg-type]
            budget_usd=budget_usd,
            guard_capabilities=guard_capabilities,
            events=self.events,
        )
        self.model = models[0]
        return self

    def shadow(  # type: ignore[misc]
        self: ContextApp,
        candidate_model: str,
        *,
        candidate_provider: ModelProvider | None = None,
        block: bool = False,
    ) -> Any:
        """Serve the primary model but dual-dispatch ``candidate_model`` for an
        offline diff. Returns the :class:`~vincio.providers.shadow.ShadowProvider`
        (read ``.observations`` / ``.diff()``); it also becomes the app's provider
        so every run is shadowed until removed."""
        from ..providers.shadow import ShadowProvider

        # Residency: the shadow dual-dispatches the request to the candidate, so
        # the candidate model's region must be allowed before any egress.
        self._enforce_model_residency(
            candidate_model, provider=getattr(candidate_provider, "name", None)
        )
        primary = self._base_provider()
        candidate = candidate_provider or primary
        shadow = ShadowProvider(
            primary,
            candidate,
            candidate_model=candidate_model,
            block=block,
            price_table=self.cost_tracker.price_table,
            events=self.events,
        )
        self._provider_instance = shadow
        return shadow

    def canary(  # type: ignore[misc]
        self: ContextApp,
        candidate_model: str,
        *,
        percent: float = 5.0,
        candidate_provider: ModelProvider | None = None,
        score_fn: Callable[[Any], float] | None = None,
        min_samples: int = 20,
        regression_threshold: float = 0.05,
        prompt_name: str | None = None,
    ) -> Any:
        """Ramp ``percent``% of live traffic onto ``candidate_model`` with online
        scoring and auto-rollback to the primary (and prompt-registry head) on
        regression. Returns the :class:`~vincio.providers.shadow.CanaryRouter`,
        which also becomes the app's provider."""
        from ..providers.shadow import CanaryRouter

        # Residency: the canary routes live traffic to the candidate model.
        self._enforce_model_residency(
            candidate_model, provider=getattr(candidate_provider, "name", None)
        )
        primary = self._base_provider()
        candidate = candidate_provider or primary
        canary = CanaryRouter(
            primary,
            candidate,
            percent=percent,
            candidate_model=candidate_model,
            score_fn=score_fn,
            min_samples=min_samples,
            regression_threshold=regression_threshold,
            prompt_registry=getattr(self, "prompt_registry", None),
            prompt_name=prompt_name,
            events=self.events,
        )
        self._provider_instance = canary
        return canary

    async def agate_swap(  # type: ignore[misc]
        self: ContextApp,
        candidate_model: str,
        *,
        baseline_model: str | None = None,
        dataset: Any = None,
        traces: list[Any] | None = None,
        metrics: list[str] | None = None,
        quality_metric: str = "lexical_overlap",
        gates: dict[str, str] | None = None,
        alpha: float = 0.05,
        repeats: int = 1,
        flake_quarantine: bool = True,
        pin_tools: bool = True,
    ) -> Any:
        """Gate a model swap on replayed golden traces + an eval/cost/latency/
        behavioral diff with statistical backing. Returns a
        :class:`~vincio.evals.swap.SwapVerdict`."""
        from ..evals.swap import SwapGate

        gate = SwapGate(
            self,
            metrics=metrics,
            quality_metric=quality_metric,
            gates=gates,
            alpha=alpha,
            repeats=repeats,
            flake_quarantine=flake_quarantine,
        )
        return await gate.evaluate(
            candidate_model=candidate_model,
            baseline_model=baseline_model,
            dataset=dataset,
            traces=traces,
            pin_tools=pin_tools,
        )

    def gate_swap(self: ContextApp, candidate_model: str, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Synchronous :meth:`agate_swap`."""
        from ..providers.base import run_sync

        return run_sync(self.agate_swap(candidate_model, **kwargs))

    async def aswap_regression(  # type: ignore[misc]
        self: ContextApp,
        dataset: Any,
        *,
        candidate_model: str,
        baseline_model: str | None = None,
        metrics: list[str] | None = None,
        quality_metric: str = "lexical_overlap",
        alpha: float = 0.05,
        repeats: int = 1,
        flake_quarantine: bool = True,
    ) -> Any:
        """Swap only the model on a fixed dataset and return a statistically
        grounded :class:`~vincio.evals.swap.SwapRegressionReport`."""
        from ..evals.swap import model_swap_regression

        return await model_swap_regression(
            self,
            dataset,
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            metrics=metrics,
            quality_metric=quality_metric,
            alpha=alpha,
            repeats=repeats,
            flake_quarantine=flake_quarantine,
        )

    def swap_regression(self: ContextApp, dataset: Any, *, candidate_model: str, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Synchronous :meth:`aswap_regression`."""
        from ..providers.base import run_sync

        return run_sync(self.aswap_regression(dataset, candidate_model=candidate_model, **kwargs))

    def watch_lifecycle(  # type: ignore[misc]
        self: ContextApp,
        models: list[str] | None = None,
        *,
        as_of: Any = None,
        warn_within_days: int = 90,
        propose: bool = True,
    ) -> dict[str, Any]:
        """Scan pinned models for sunset and (optionally) propose migrations off
        deprecated/retired/nearing-retirement ones. Returns ``{"alerts",
        "proposals"}``; defaults to the app's pinned models."""
        from ..providers.lifecycle import LifecycleWatcher

        watcher = LifecycleWatcher(warn_within_days=warn_within_days, events=self.events)
        targets = list(models) if models else self._pinned_models()
        alerts = watcher.scan(targets, as_of=as_of)
        proposals = watcher.propose_all(targets, as_of=as_of) if propose else []
        return {"alerts": alerts, "proposals": proposals}

    def set_cost_budget(  # type: ignore[misc]
        self: ContextApp,
        *,
        limit_usd: float,
        scope: str = "tenant",
        id: str | None = None,
        period: str = "day",
        on_breach: str = "cap",
        degrade_model: str | None = None,
        anomaly_factor: float | None = None,
    ) -> ContextApp:
        """Enforce a per-tenant/feature/user cost budget.

        When the scope's spend over ``period`` reaches ``limit_usd``, ``on_breach``
        decides the action: ``"cap"`` denies the run, ``"degrade"`` swaps in
        ``degrade_model`` (a cheaper model), and ``"queue_to_batch"`` denies the
        interactive run and points the caller at :meth:`batch`. Set
        ``anomaly_factor`` to raise a ``cost.anomaly`` event on a spend spike::

            app.set_cost_budget(scope="tenant", id="acme", limit_usd=10.0, period="day")
            app.set_cost_budget(scope="feature", id="chat", limit_usd=5.0,
                                 on_breach="degrade", degrade_model="gpt-5.2-mini")
        """
        from ..observability.finops import CostBudget

        self.budget_manager.add(
            CostBudget(
                scope=scope,  # type: ignore[arg-type]
                id=id,
                limit_usd=limit_usd,
                period=period,  # type: ignore[arg-type]
                on_breach=on_breach,  # type: ignore[arg-type]
                degrade_model=degrade_model,
                anomaly_factor=anomaly_factor,
            )
        )
        return self

    def cost_report(self: ContextApp, *, by: str = "tenant", since: Any | None = None):  # type: ignore[misc]
        """Roll up attributed model cost by ``tenant``/``feature``/``user``/
        ``model``/``provider``/``run`` (returns a :class:`CostReport`)."""
        return self.cost_ledger.report(by, since=since)  # type: ignore[arg-type]

    def use_energy_accounting(  # type: ignore[misc]
        self: ContextApp,
        *,
        region: str | None = None,
        pue: float | None = None,
        carbon_intensity: dict[str, float] | None = None,
    ) -> ContextApp:
        """Turn on per-run energy & carbon accounting (opt-in).

        Once enabled, every run accrues an estimated energy (watt-hours) and
        carbon (grams CO₂e) figure — mechanical and deterministic, from the run's
        token accounting against a per-model intensity (by tier) and a per-region
        grid factor — onto the cost-report surface
        (:meth:`energy_report`, ``result.energy_wh`` / ``result.co2e_grams``) and
        the hash-chained audit log. No external service is consulted. The energy
        analogue of the dollar cost report::

            app.use_energy_accounting(region="eu")
            result = app.run("summarize this")
            print(result.energy_wh, result.co2e_grams)
            app.energy_report(by="model").print_summary()

        ``region`` overrides the default region used when a call's region cannot
        be resolved; ``pue`` overrides the datacenter power-overhead factor;
        ``carbon_intensity`` merges operator-measured grid factors (g CO₂e/kWh,
        keyed by region) over the built-in defaults.
        """
        table = self.cost_tracker.energy_table
        if region is not None:
            table.region_override = region
        if pue is not None:
            table.pue = max(0.0, pue)
        if carbon_intensity:
            for reg, g in carbon_intensity.items():
                table.set_region_intensity(reg, g)
        self.energy_accounting_enabled = True
        return self

    def set_energy_budget(  # type: ignore[misc]
        self: ContextApp,
        *,
        scope: str = "global",
        id: str | None = None,
        limit_wh: float | None = None,
        limit_co2e_grams: float | None = None,
        period: str = "day",
    ) -> ContextApp:
        """Set an energy/carbon budget, refused on breach like a cost cap.

        The sustainability analogue of :meth:`set_cost_budget`. Give an energy
        ceiling (``limit_wh``), a carbon ceiling (``limit_co2e_grams``), or both;
        when a scope's accrued energy or carbon over ``period`` reaches a ceiling,
        the run is refused on the same audit path as a cost cap. Enables energy
        accounting on first use::

            app.set_energy_budget(scope="tenant", id="acme", limit_co2e_grams=500.0)
            app.set_energy_budget(limit_wh=1000.0, period="hour")
        """
        from ..core.errors import EnergyBudgetError
        from ..observability.finops import EnergyBudget

        if limit_wh is None and limit_co2e_grams is None:
            raise EnergyBudgetError(
                "an energy budget needs at least one of limit_wh / limit_co2e_grams"
            )
        if not self.energy_accounting_enabled:
            self.use_energy_accounting()
        self.budget_manager.add_energy_budget(
            EnergyBudget(
                scope=scope,  # type: ignore[arg-type]
                id=id,
                limit_wh=limit_wh,
                limit_co2e_grams=limit_co2e_grams,
                period=period,  # type: ignore[arg-type]
            )
        )
        return self

    def energy_report(self: ContextApp, *, by: str = "tenant", since: Any | None = None):  # type: ignore[misc]
        """Roll up estimated energy + carbon by ``tenant``/``feature``/``user``/
        ``model``/``provider``/``run`` (returns an :class:`EnergyReport`).

        The energy analogue of :meth:`cost_report`, on the same surface and from
        the same attributed events."""
        return self.cost_ledger.energy_report(by, since=since)  # type: ignore[arg-type]

    def batch(  # type: ignore[misc]
        self: ContextApp,
        inputs: list[str | UserInput],
        *,
        backend: Any | None = None,
        config: RunConfig | None = None,
        discount: float = 0.5,
        timeout_s: float | None = None,
    ) -> list[RunResult]:
        """Run a set of inputs through a provider Batch API at ~half the cost.

        Latency-tolerant work — evals, bulk extraction, synthetic data — with the
        same :class:`RunResult` contract as :meth:`run`. ``backend`` is a
        :class:`~vincio.providers.BatchBackend` (or a provider; defaults to the
        app's provider run in-process)::

            results = app.batch(["summarize doc A", "summarize doc B"])
        """
        return run_sync(
            self.abatch(
                inputs, backend=backend, config=config, discount=discount, timeout_s=timeout_s
            )
        )

    async def abatch(  # type: ignore[misc]
        self: ContextApp,
        inputs: list[str | UserInput],
        *,
        backend: Any | None = None,
        config: RunConfig | None = None,
        discount: float = 0.5,
        timeout_s: float | None = None,
    ) -> list[RunResult]:
        """Async :meth:`batch`."""
        return await self._runtime.execute_batch(
            inputs, run_config=config, backend=backend, discount=discount, timeout_s=timeout_s
        )
