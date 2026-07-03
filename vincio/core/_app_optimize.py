"""Evaluation, online-eval, closed-loop, learning, assurance, and training verbs — a private mixin of
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
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..evals.datasets import Dataset, EvalCase
from ..evals.metrics import RunOutput
from ..evals.runners import EvalRunner
from ..providers.base import run_sync
from .diagnostics import note_suppressed
from .errors import (
    ConfigError,
)
from .types import (
    RunResult,
    UserInput,
)

if TYPE_CHECKING:
    from ..optimize.routing import ModelCascade
    from ..prompts.templates import PromptSpec
    from ..providers.base import ModelProvider
    from .app import ContextApp

logger = logging.getLogger("vincio.app")


class _OptimizeVerbs:
    """Evaluation, online-eval, closed-loop, learning, assurance, and training verbs. Mixed into :class:`~vincio.core.app.ContextApp`."""

    if TYPE_CHECKING:
        # ContextApp state this mixin's verbs assign. mypy would otherwise
        # attribute the unannotated ``self.X = ...`` assignments to this class
        # and clash with ContextApp.__init__; the declarations (type-checking
        # only, no runtime effect) keep the split typing identical to the
        # monolith's.
        _provider_instance: ModelProvider | None
        cascade: ModelCascade | None
        local_adapter: Any | None
        model: str
        prompt_spec: PromptSpec


    # -- evaluation -------------------------------------------------------------------------------

    async def eval_target(self: ContextApp, case: EvalCase) -> RunOutput:  # type: ignore[misc]
        """EvalRunner adapter: run one case through the app."""
        result = await self.arun(case.input_text)
        return self._run_output_from_result(result)

    @staticmethod
    def _run_output_from_result(result: RunResult) -> RunOutput:
        """Project a RunResult onto a RunOutput, carrying a lightweight trajectory
        built from the run's tool results so trajectory metrics can score it."""
        from ..evals.trajectory import Trajectory, TrajectoryStep

        steps = [
            TrajectoryStep(type="tool", name=tr.tool_name, tool_name=tr.tool_name, status=tr.status)
            for tr in result.tool_results
        ]
        trajectory = Trajectory(
            steps=steps,
            final_answer=result.output,
            raw_text=result.raw_text,
            terminated=True,
            termination_reason=result.status.value
            if hasattr(result.status, "value")
            else str(result.status),
            success=result.error is None,
            source="run",
            usage={
                "steps": float(len(steps)),
                "tool_calls": float(len(steps)),
                "cost_usd": float(result.cost_usd),
            },
        )
        return RunOutput(
            output=result.output,
            raw_text=result.raw_text,
            evidence=result.evidence,
            citations=result.citations,
            usage=result.usage,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            schema_valid=result.validation.get("valid") if result.validation else None,
            error=result.error,
            trace_id=result.trace_id,
            trajectory=trajectory if steps else None,
            metadata={"input": result.metadata.get("input", "")},
        )

    # -- online / continuous evaluation --------------------------------

    def _spawn_online(self: ContextApp, result: RunResult, user_input: UserInput) -> None:  # type: ignore[misc]
        """Schedule online scoring off the hot path; run inline if no loop."""
        coro = self._score_online(result, user_input)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            run_sync(coro)
            return
        task = loop.create_task(coro)
        self._online_tasks.add(task)
        task.add_done_callback(self._online_tasks.discard)

    async def _score_online(self: ContextApp, result: RunResult, user_input: UserInput) -> None:  # type: ignore[misc]
        run_output = self._run_output_from_result(result)
        run_output.metadata.setdefault("input", user_input.text or "")
        case = EvalCase(id=result.trace_id or result.run_id, input=user_input.text or "")
        for evaluator in self.online_evaluators:
            try:
                metric_result = evaluator.observe(
                    run_output, case=case, run_id=result.trace_id or result.run_id
                )
            except Exception:  # noqa: BLE001 - online eval must never break a run
                logger.exception("online evaluator %s failed", evaluator.name)
                continue
            if metric_result is not None:
                self.events.emit(
                    "eval.online",
                    {
                        "metric": evaluator.name,
                        "value": metric_result.value,
                        "run_id": result.trace_id,
                    },
                )

    async def aflush_online(self: ContextApp) -> None:  # type: ignore[misc]
        """Await any in-flight online evaluations (for tests and shutdown)."""
        if self._online_tasks:
            await asyncio.gather(*list(self._online_tasks), return_exceptions=True)

    def evaluate(  # type: ignore[misc]
        self: ContextApp,
        dataset: Dataset | str,
        *,
        metrics: list[str] | None = None,
        concurrency: int = 8,
        gates: dict[str, str] | None = None,
        judges: list[Any] | None = None,
    ):
        """Evaluate the app over a dataset and return an :class:`EvalReport`."""
        runner = EvalRunner(
            self,
            metrics=metrics or (self.evaluators or None),
            concurrency=concurrency,
            gates=gates,
            judges=judges,
        )
        return runner.run(dataset)

    def benchmark_suite(  # type: ignore[misc]
        self: ContextApp,
        benchmarks: str | list[str] = "all",
        *,
        tier: str = "static",
        sample: int | None = None,
        datasets: dict[str, Any] | None = None,
        concurrency: int = 8,
        model: str | None = None,
        solver_mode: str | None = None,
        store: Any | None = None,
        version: str = "",
        record: bool = True,
    ):
        """Run the open evaluation plane over this app and return a ``SuiteRun``.

        The pluggable harness for the **standard public model benchmarks** (MMLU,
        GPQA, GSM8K, HumanEval, IFEval, TruthfulQA, RULER, …) grouped by niche and
        reported the same way for every model and version — distinct from
        :meth:`evaluate`, which scores this app over a golden ``Dataset``. Every
        number carries a **provenance tier**: the default ``"static"`` replays the
        bundled fabricated fixtures fully offline (reproducible, gates CI);
        ``"recorded"`` / ``"live"`` need a per-benchmark
        :class:`~vincio.evals.suite.BenchmarkDataset` in ``datasets`` (and, for
        ``"live"``, this app drives the model). The engine **refuses** to let a
        lower tier print a higher tier's label, and runs each long-context
        benchmark twice — with and without the context governor — so the uplift is
        measured::

            run = app.benchmark_suite("knowledge", tier="static")
            run.overall(); run.niche_scores(); run.determinism_digest
            from vincio.evals.suite import SuiteReport
            SuiteReport(run).save("report.md")

        ``benchmarks`` is an id (``"knowledge.mmlu"``), a niche (``"knowledge"``),
        ``"all"``, or a list. Pass a :class:`~vincio.evals.suite.RunStore` as
        ``store`` to persist the run (``version`` tags the model version for
        :meth:`~vincio.evals.suite.RunStore.model_version_diff`). Returns a
        :class:`~vincio.evals.suite.SuiteRun`.
        """
        from ..evals.suite import BenchmarkSuite

        runner = BenchmarkSuite(concurrency=concurrency)
        run = run_sync(
            runner.arun(
                benchmarks, target=self, model=model, tier=tier, sample=sample,
                datasets=datasets, solver_mode=solver_mode,
            )
        )
        if store is not None:
            store.save(run, version=version)
        if record and self.audit is not None:
            self.audit.record(
                "benchmark_suite",
                decision="allow",
                details={
                    "run_id": run.run_id, "tier": run.tier.value,
                    "benchmarks": len(run.runs), "overall": run.overall(),
                    "gated": run.gated,
                },
            )
        return run

    # -- closed loop ---------------------------------------------------------

    def improvement_loop(self: ContextApp, **kwargs: Any):  # type: ignore[misc]
        """The trace → dataset → eval → optimize → promote loop on this app.

        Returns an :class:`~vincio.optimize.ImprovementLoop` bound to this
        app's tracer, store, and prompt::

            loop = app.improvement_loop(gates={"groundedness": ">= 0.8"})
            result = loop.run(min_feedback_score=0.5)
        """
        from ..optimize.loop import ImprovementLoop

        return ImprovementLoop(self, **kwargs)

    # -- reflective optimization & the data flywheel -------------------------

    def _evaluate_variant_fn(self: ContextApp, metrics: list[str], *, concurrency: int = 4):  # type: ignore[misc]
        """Build a memory-write-free variant evaluator for the optimizers.

        Candidate evaluations must never mutate user memory or hand later
        candidates different recall state than earlier ones saw, so the prompt
        spec, compiler options, and ``memory.write_back`` are saved, neutralized,
        and restored around every evaluation — the same discipline the
        improvement loop uses.
        """
        from ..evals.runners import EvalRunner

        async def evaluate_variant(variant, ds):
            original_spec = self.prompt_spec
            original_options = self.prompt_compiler.options
            original_write_back = self.config.memory.write_back
            self.prompt_spec = variant.spec
            self.prompt_compiler.options = variant.compiler_options
            self.config.memory.write_back = []
            try:
                runner = EvalRunner(self, metrics=metrics, concurrency=concurrency)
                return await runner.arun(ds, name=variant.name)
            finally:
                self.prompt_spec = original_spec
                self.prompt_compiler.options = original_options
                self.config.memory.write_back = original_write_back

        return evaluate_variant

    def reflective_optimize(  # type: ignore[misc]
        self: ContextApp,
        dataset: Dataset,
        *,
        strategy: str = "reflective",
        metrics: list[str] | None = None,
        budget: int = 12,
        minibatch_size: int = 8,
        seed: int = 7,
        weights: Any | None = None,
        gates: dict[str, str] | None = None,
        objectives: Any | None = None,
        concurrency: int = 4,
        reflector: str = "heuristic",
        apply: bool = False,
    ):
        """Run the GEPA-style reflective optimizer against ``dataset``.

        Instead of blind variant search, the optimizer reads the eval report's
        failures, reflects on why the prompt lost, and proposes targeted edits,
        evolving a Pareto frontier under a hard ``budget`` of rollouts —
        deterministic under ``seed``. ``strategy="mipro"`` switches to joint
        instruction+example proposal. With ``apply=True`` a promoted winner is
        installed on the app::

            result = app.reflective_optimize(dataset, gates={"groundedness": ">= 0.8"})
            result.promoted, result.reason, result.frontier.front

        ``reflector="llm"`` uses the real provider-backed :class:`LLMReflector`
        wired to this app's own provider — it reads the actual failing cases,
        clusters them into failure modes, and proposes targeted edits, falling
        back to the deterministic heuristic reflector offline. ``"heuristic"``
        (the default) is the fully reproducible, air-gapped floor.
        """
        from ..optimize.loop import DEFAULT_LOOP_METRICS
        from ..optimize.reflective import LLMReflector, ReflectiveOptimizer

        metric_list = metrics or (self.evaluators or DEFAULT_LOOP_METRICS)
        reflector_impl = None
        if reflector == "llm":
            reflector_impl = LLMReflector(self._base_provider(), self.model)
        optimizer = ReflectiveOptimizer(
            self._evaluate_variant_fn(metric_list, concurrency=concurrency),
            weights=weights,
            gates=gates,
            objectives=objectives,
            reflector=reflector_impl,
        )
        result = run_sync(
            optimizer.optimize(
                self.prompt_spec,
                dataset,
                strategy=strategy,  # type: ignore[arg-type]
                budget=budget,
                minibatch_size=minibatch_size,
                seed=seed,
            )
        )
        if apply and result.promoted and result.best is not None:
            winner = result.best.payload
            self.prompt_spec = winner.spec
            self.prompt_compiler.options = winner.compiler_options
            self.events.emit("optimize.reflective", {"reason": result.reason})
        return result

    def self_improvement(self: ContextApp, policy: Any | None = None, **kwargs: Any):  # type: ignore[misc]
        """The unified, declarative self-improvement contract.

        One :class:`~vincio.optimize.self_improvement.SelfImprovementPolicy`
        composes scheduling, autonomous proposal, online updates, meta-optimization
        (learned fitness weights + successive-halving), active-learning label
        acquisition, and canary-gated promotion/rollback. Returns a
        :class:`~vincio.optimize.self_improvement.SelfImprovementController` whose
        :meth:`~vincio.optimize.self_improvement.SelfImprovementController.astream`
        emits the cycle as ``observe → proposal → meta → label → canary →
        promote/rollback`` events::

            from vincio.optimize import SelfImprovementPolicy
            ctl = app.self_improvement(SelfImprovementPolicy(), dataset=golden)
            async for ev in ctl.astream():
                print(ev.phase, ev.reason)

        Every promotion passes the same significance + safety + golden
        non-regression gates the loop always used; every decision lands on the
        shared audit chain and event bus."""
        from ..optimize.self_improvement import SelfImprovementController, SelfImprovementPolicy

        if policy is None:
            policy = SelfImprovementPolicy()
        return SelfImprovementController(self, policy, **kwargs)

    def deploy(self: ContextApp, candidate: Any, *, dataset: Any = None, **kwargs: Any):  # type: ignore[misc]
        """Canary-gate a prompt/policy candidate and deploy it only if it clears.

        Two modes: an **offline** gated comparison against the live prompt on a
        canary ``dataset=``, or a **live-traffic** canary that ramps a fraction of
        ``live_inputs=`` onto the candidate (scored by ``score_fn=``) with
        auto-rollback. On a pass it is pushed to the prompt registry, tagged,
        applied live, and audited (``deploy``); on a fail it is refused and rolled
        back to the last known-good version. Returns a
        :class:`~vincio.optimize.self_improvement.DeployResult`. This is the
        canary-driven promotion surface for prompt and policy candidates."""
        from ..optimize.self_improvement import deploy_candidate
        from ..providers.base import run_sync

        return run_sync(deploy_candidate(self, candidate, dataset=dataset, **kwargs))

    def learn(  # type: ignore[misc]
        self: ContextApp,
        tasks: list[Any],
        *,
        reward: Any,
        policy: Any | None = None,
        learning_rate: float = 0.5,
        kl_max: float = 0.5,
        iterations: int = 3,
        group_normalize: bool = True,
        min_reward_improvement: float = 0.0,
        flywheel: Any | None = None,
        held_out: Any | None = None,
        teacher: str | None = None,
        student: str | None = None,
    ):
        """On-policy reinforcement from verifiable rewards (RLVR).

        Closes the learning loop on a *policy*, not just a prompt. Each
        :class:`~vincio.optimize.trajectory_opt.LearningTask` carries a group of
        candidate outcomes; a :class:`~vincio.optimize.rewards.RewardModel` scores
        them from the verifiable signals the platform already computes (the
        task-success oracle, the benchmark scorers, calibrated judge ensembles),
        and a GRPO-style update improves the policy behind the same safety
        discipline prompt optimization uses — advantage normalization, a
        KL-to-reference clamp, and a monotonic no-regression gate::

            from vincio.optimize import LearningTask, OracleReward, RewardModel
            result = app.learn(tasks, reward=RewardModel([OracleReward()]))
            result.promoted, result.reward_delta, result.kl_to_reference

        The result's verdict is the same
        :class:`~vincio.optimize.self_improvement.CanaryVerdict` a prompt deploy
        produces, and the decision lands on the shared audit chain and event bus.
        On a promotion the on-policy winners are exported as a grounded
        :class:`~vincio.optimize.distill.TrainingSet`; pass a configured
        ``flywheel`` (with ``held_out`` / ``teacher`` / ``student``) to emit a
        fine-tune job through the existing distillation flywheel in the same call.
        Returns a :class:`~vincio.optimize.trajectory_opt.LearningResult`.
        """
        from ..optimize.trajectory_opt import TrajectoryOptimizer

        optimizer = TrajectoryOptimizer(
            reward,
            policy=policy,
            learning_rate=learning_rate,
            kl_max=kl_max,
            iterations=iterations,
            group_normalize=group_normalize,
            min_reward_improvement=min_reward_improvement,
        )
        result = run_sync(
            optimizer.alearn(
                tasks, flywheel=flywheel, held_out=held_out, teacher=teacher, student=student
            )
        )
        self.audit.record(
            "learn",
            decision="allow" if result.promoted else "deny",
            resource=self.name,
            details={
                "promoted": result.promoted,
                "reason": result.reason,
                "baseline_reward": result.baseline_reward,
                "policy_reward": result.policy_reward,
                "reward_delta": result.reward_delta,
                "kl_to_reference": result.kl_to_reference,
                "kl_within_bound": result.kl_within_bound,
                "tasks": result.tasks,
            },
        )
        self.events.emit(
            "learn.promoted" if result.promoted else "learn.rejected",
            {
                "reward_delta": result.reward_delta,
                "kl_to_reference": result.kl_to_reference,
                "reason": result.reason,
            },
        )
        return result

    def cultivate(  # type: ignore[misc]
        self: ContextApp,
        curriculum: Any,
        *,
        library: Any | None = None,
        held_out: list[Any] | None = None,
        cycles: int = 3,
        rails: Any | None = None,
        governance: Any | None = None,
        search: Any | None = None,
        min_capability_gain: float = 0.0,
        prune: bool = True,
        record: bool = True,
    ) -> Any:
        """Grow capability open-endedly: propose → attempt → verify → distill → promote.

        Closes the open-ended-learning loop on a *skill library*, not just a
        prompt or a policy. ``curriculum`` is an
        :class:`~vincio.cultivate.AutoCurriculum` (or a list of
        :class:`~vincio.cultivate.CurriculumTask`); each cycle proposes the tasks
        at the **frontier of current competence** — gating every objective through
        this app's rails and its :meth:`verify_governance` invariants, so an unsafe
        or out-of-policy task is refused and never attempted — attempts each with a
        library-composing test-time search, verifies the result against the
        task-success oracle, distills a winning trajectory into a verified,
        content-addressed :class:`~vincio.cultivate.LearnedSkill`, and promotes it
        only through the **same no-regression gate** a prompt deploy clears
        (capability on a held-out frontier set must not fall). A skill that stops
        paying its way is demoted, never silently kept::

            from vincio.cultivate import AutoCurriculum, CurriculumTask
            result = app.cultivate(AutoCurriculum(tasks))
            result.capability_after >= result.capability_before  # monotone
            result.stayed_in_policy  # no refused objective was attempted

        The decision lands on the shared audit chain (``skill_cultivation``) and
        event bus (``cultivation.completed``). Returns a content-bound
        :class:`~vincio.cultivate.CultivationResult` whose ``verify`` re-derives the
        monotonicity and stay-in-policy verdicts from the bytes, with the grown
        :class:`~vincio.cultivate.LearnedSkillLibrary` on ``result.library``.
        """
        from ..cultivate import Cultivator

        cultivator = Cultivator(
            self,
            curriculum=curriculum,
            library=library,
            held_out=held_out,
            rails=rails if rails is not None else self.rail_engine,
            governance=governance,
            search=search,
            min_capability_gain=min_capability_gain,
            prune=prune,
            record=record,
        )
        return cultivator.run(cycles=cycles)
    # -- continuous assurance & production certification ----------------

    def assurance_case(  # type: ignore[misc]
        self: ContextApp,
        statement: str,
        *,
        context: str = "",
        subclaims: list[Any] | None = None,
        evidence: list[Any] | None = None,
        subject: str | None = None,
        sign: bool = True,
        signer: Any | None = None,
        record: bool = True,
    ) -> Any:
        """Assemble the platform's evidence into one continuously-checkable safety argument.

        Builds a content-bound :class:`~vincio.assurance.AssuranceCase`: a top
        :class:`~vincio.assurance.Claim` (*this app is fit for purpose X under
        context Y*) decomposed into ``subclaims``, each discharged by
        :class:`~vincio.assurance.Evidence` the platform **already emits** — an eval
        gate verdict, a :meth:`verify_governance` proof, a reasoning
        :class:`~vincio.verify.Certificate`, an audit-chain segment, an
        identity/delegation chain, or an AI-BOM — bound by hash so the whole case
        :meth:`~vincio.assurance.AssuranceCase.verify`\\s offline and a missing,
        stale, or falsified piece of evidence is pinpointed::

            from vincio.assurance import Claim, Evidence
            case = app.assurance_case(
                "The assistant is fit for production",
                context="EU deployment",
                subclaims=[Claim(id="governance", statement="Controls hold",
                                 evidence=[Evidence.from_governance(app.verify_governance())])],
            )
            report = case.check()  # re-derives the verdict from the bytes

        Re-check the case on every change (a model swap, a prompt edit, a dependency
        bump) with :meth:`~vincio.assurance.AssuranceCase.check` and gate the build
        with :func:`~vincio.assurance.assurance_regression_gate`. The case is signed
        with the app's identity unless ``sign`` is off, and (when ``record``) the
        verdict lands on the hash-chained audit log as an ``assurance_case``
        decision. Returns the sealed :class:`~vincio.assurance.AssuranceCase`.
        """
        from ..assurance import AssuranceCase, Claim

        goal = Claim(
            id="goal",
            statement=statement,
            context=context,
            subclaims=list(subclaims or []),
            evidence=list(evidence or []),
        )
        case = AssuranceCase(subject=subject or self.name, goal=goal).seal()
        chain_signer = self._resolve_contract_signer(signer, sign)
        if chain_signer is not None:
            case.sign(chain_signer)
        if record and self.audit is not None:
            report = case.check()
            self.audit.record(
                "assurance_case",
                resource=case.case_hash,
                decision="allow" if report.holds else "deny",
                details={
                    "statement": statement,
                    "holds": report.holds,
                    "claims": len(report.root.walk()),
                    "failing_claims": report.failing_claims,
                    "missing": report.missing,
                    "stale": report.stale,
                    "falsified": report.falsified,
                },
            )
        return case

    def certify(  # type: ignore[misc]
        self: ContextApp,
        case: Any,
        *,
        residual_risks: list[str] | None = None,
        provenance: dict[str, Any] | None = None,
        aibom: bool = True,
        sign: bool = True,
        signer: Any | None = None,
        record: bool = True,
        as_of: Any | None = None,
    ) -> Any:
        """Emit a portable, offline-verifiable production-certification report.

        Checks the :class:`~vincio.assurance.AssuranceCase`, records the residual
        risks (any undischarged claim, plus any passed in), stamps the build
        provenance (the ``vincio`` version and, unless ``aibom`` is off, a CycloneDX
        AI-BOM of the live configuration), and signs the report with the app's
        identity. Returns a :class:`~vincio.assurance.CertificationReport` a
        downstream operator or auditor checks **from the bytes**::

            report = app.certify(case)
            assert report.verify()                 # re-runs the case's own check
            assert report.certified                # the case holds

        :meth:`~vincio.assurance.CertificationReport.verify` recomputes the report
        hash, re-verifies the embedded case, and re-runs its evidence check, so a
        report certifying a case that does not hold is caught. The verdict lands on
        the hash-chained audit log as an ``assurance_certification`` decision unless
        ``record`` is off.
        """
        from ..assurance import certify as _certify

        prov: dict[str, Any] = dict(provenance or {})
        if aibom and "sbom" not in prov:
            try:
                from ..governance.aibom import generate_aibom

                bom = generate_aibom(self)
                prov.setdefault("vincio_version", bom.vincio_version)
                prov["sbom"] = bom.to_cyclonedx()
            except Exception:
                note_suppressed("assurance.certify.sbom")
        prov.setdefault("slsa", "SLSA build provenance attested by the release pipeline")
        chain_signer = self._resolve_contract_signer(signer, sign)
        report = _certify(
            case,
            signer=chain_signer,
            residual_risks=residual_risks,
            provenance=prov,
            as_of=as_of,
        )
        if record and self.audit is not None:
            self.audit.record(
                "assurance_certification",
                resource=report.report_hash,
                decision="allow" if report.certified else "deny",
                details={
                    "statement": report.statement,
                    "certified": report.certified,
                    "residual_risks": report.residual_risks,
                    "case_hash": case.case_hash,
                },
            )
        return report

    def use_bandit_router(  # type: ignore[misc]
        self: ContextApp, models: list[str], *, bandit: str = "epsilon_greedy", **kwargs: Any
    ) -> ContextApp:
        """Route live traffic through a guarded online bandit over ``models``.

        Wires an :class:`~vincio.optimize.routing.GuardedBanditRouter` over the
        app's base provider: the bandit learns which model pays off, never
        explores on safety-/high-risk-tagged traffic, persists arm stats to the
        app's store, and auto-freezes / rolls back on regression. The router
        becomes the app's provider, so it nests inside the existing
        circuit-breaker / key-pool / failover stack.
        """
        from ..optimize.routing import GuardedBanditRouter

        base = self._base_provider()
        entries = [(base, m) for m in models]
        self._provider_instance = GuardedBanditRouter(
            entries,
            bandit=bandit,
            store=self.store,
            app_name=self.name,
            events=self.events,
            **kwargs,
        )
        if models:
            self.model = models[0]
        return self

    def enable_training_capture(self: ContextApp, enabled: bool = True) -> ContextApp:  # type: ignore[misc]
        """Record the full output and cited evidence on every trace, so
        :meth:`export_training_set` can curate faithful, grounded fine-tuning
        data. Off by default (the span output stays truncated for cost)::

            app.enable_training_capture()  # then run production traffic
        """
        self.config.observability.training_capture = enabled
        return self

    def export_training_set(  # type: ignore[misc]
        self: ContextApp,
        *,
        name: str = "distilled",
        runs: list[Any] | None = None,
        traces: list[Any] | None = None,
        limit: int = 500,
        min_feedback_score: float | None = None,
        require_grounding: bool = True,
        min_support: float = 0.5,
        max_examples: int | None = None,
        path: str | None = None,
        format: str = "openai",
    ):
        """Curate runs or captured traces into a grounded fine-tuning :class:`TrainingSet`.

        Two faithful sources, both grounding-checked, deduped, and
        provenance-stamped, emitting provider-ready JSONL (nothing ungrounded is
        exported):

        - ``runs=[...]`` — :class:`RunResult` objects (the natural output of
          :meth:`run`). These carry the **full** output and cited evidence, so
          the export is faithful with **no opt-in capture** required — the
          recommended path::

              results = [app.run(q) for q in prompts]
              ts = app.export_training_set(runs=results, path="train.jsonl")

        - traces (default) — reuses the traces production runs already write,
          feedback-filtered (``min_feedback_score``). Faithful only when
          :meth:`enable_training_capture` recorded the full artifacts; otherwise
          the span output is truncated.

        With ``path`` the JSONL is written for ``format`` ("openai"/"anthropic").
        """
        from ..optimize.distill import export_training_set, export_training_set_from_runs

        system = self.prompt_spec.role or self.prompt_spec.objective
        if runs is not None:
            training_set = export_training_set_from_runs(
                runs,
                name=name,
                system=system,
                require_grounding=require_grounding,
                min_support=min_support,
                max_examples=max_examples,
            )
        else:
            if traces is None:
                exporter = self.tracer.exporter
                if hasattr(exporter, "load_all"):
                    traces = exporter.load_all(limit=limit)
                elif hasattr(exporter, "traces"):
                    traces = list(exporter.traces)[-limit:]
                else:
                    traces = []
            training_set = export_training_set(
                traces,
                name=name,
                system=system,
                min_feedback_score=min_feedback_score,
                require_grounding=require_grounding,
                min_support=min_support,
                max_examples=max_examples,
            )
        if path is not None:
            training_set.save(path, format=format)  # type: ignore[arg-type]
            self.events.emit(
                "distill.exported", {"path": path, "examples": len(training_set), "format": format}
            )
        return training_set

    def distill(  # type: ignore[misc]
        self: ContextApp,
        training_set: Any,
        dataset: Dataset,
        *,
        teacher: str,
        student: str,
        trainer: Any | None = None,
        quality_metric: str = "lexical_overlap",
        min_quality_ratio: float = 0.97,
        gates: dict[str, str] | None = None,
        concurrency: int = 4,
        apply: bool = True,
    ):
        """Teacher → student distillation, gated on holding quality.

        Evaluates teacher and student on the held-out ``dataset`` and promotes
        the (optionally fine-tuned) student into a cheap→strong runtime cascade
        only when it preserves ``min_quality_ratio`` of the teacher's quality at
        strictly lower cost, with no safety/schema regression. With
        ``apply=True`` a promoted cascade is installed via :meth:`use_cascade`::

            ts = app.export_training_set(min_feedback_score=0.5)
            result = app.distill(ts, held_out, teacher="gpt-5.2", student="gpt-5.2-mini")
            result.promoted, result.cost_savings
        """
        from ..optimize.distill import BootstrapFinetune

        async def evaluate_model(model, ds):
            from ..evals.runners import EvalRunner

            original_model = self.model
            original_write_back = self.config.memory.write_back
            self.model = model
            self.config.memory.write_back = []
            try:
                runner = EvalRunner(
                    self,
                    metrics=[quality_metric, "cost", "safety", "schema_validity"],
                    concurrency=concurrency,
                )
                return await runner.arun(ds, name=f"distill:{model}")
            finally:
                self.model = original_model
                self.config.memory.write_back = original_write_back

        loop = BootstrapFinetune(
            evaluate_model,
            quality_metric=quality_metric,
            min_quality_ratio=min_quality_ratio,
            gates=gates,
            trainer=trainer,
        )
        result = run_sync(loop.distill(training_set, dataset, teacher=teacher, student=student))
        if apply and result.promoted and result.cascade is not None:
            self.cascade = result.cascade
            self.events.emit(
                "distill.promoted",
                {"student": result.trained_student, "cost_savings": result.cost_savings},
            )
        return result

    def use_local_adapter(self: ContextApp, adapter: Any | None) -> ContextApp:  # type: ignore[misc]
        """Apply (or remove) an on-device LoRA-class adapter on the base provider.

        Wraps the app's base provider in an
        :class:`~vincio.optimize.local_adaptation.AdaptedProvider` so an
        in-distribution request is answered the way the locally-fit
        :class:`~vincio.optimize.local_adaptation.LocalAdapter` learned, while
        everything else falls through to the base model unchanged — the run never
        leaves the process. The wrapper reports the base provider's name and
        capabilities, so residency, provenance, and the rotation stack are
        unaffected. Pass ``None`` to unload the adapter and restore the base model
        (the one-call reversibility path). Returns ``self`` for chaining::

            adapter = app.adapt_locally(golden, runs=results).verdict  # gated fit
            app.use_local_adapter(registry.active("local-adapter"))
        """
        from ..optimize.local_adaptation import AdaptedProvider

        base = self._base_provider()
        if isinstance(base, AdaptedProvider):
            base = base.base
        if adapter is None:
            self._provider_instance = base
            self.local_adapter = None
            return self
        self._provider_instance = AdaptedProvider(base, adapter, embedder=self.embedder)
        self.local_adapter = adapter
        return self

    def local_adaptation(self: ContextApp, policy: Any | None = None, **kwargs: Any):  # type: ignore[misc]
        """The continual on-device adaptation loop, as a streaming controller.

        Returns a
        :class:`~vincio.optimize.local_adaptation.ContinualAdaptation` driven by a
        :class:`~vincio.optimize.local_adaptation.LocalAdaptationPolicy`: gather
        the flywheel's promoted grounded dataset, fit a new
        :class:`~vincio.optimize.local_adaptation.LocalAdapter` version on-device,
        gate it against the current base on a held-out set, and promote or roll
        back — every version registered and reversible, every decision on the
        shared audit chain and event bus, all in-process::

            ctl = app.local_adaptation(dataset=golden)
            async for ev in ctl.astream(runs=results):
                print(ev.phase, ev.reason)

        Promotion clears the same no-regression discipline a hosted fine-tune job
        does (:class:`~vincio.optimize.local_adaptation.AdapterGate`)."""
        from ..optimize.local_adaptation import ContinualAdaptation

        return ContinualAdaptation(self, policy, **kwargs)

    def adapt_locally(  # type: ignore[misc]
        self: ContextApp,
        dataset: Any,
        *,
        runs: list[Any] | None = None,
        training_set: Any | None = None,
        policy: Any | None = None,
        registry: Any | None = None,
        base_model: str | None = None,
        apply: bool = True,
    ):
        """Fit, gate, and (on a pass) install an on-device adapter — one call.

        The one-shot form of :meth:`local_adaptation`. Curates a grounded training
        set (from ``runs``, a prebuilt ``training_set``, or the app's captured
        traces), fits a LoRA-class adapter on-device, gates it against the base on
        the held-out ``dataset`` (no-regression — the adapted model must be
        at-least-as-good), and on a pass registers it, makes it the active head,
        and (with ``apply``) applies it via :meth:`use_local_adapter`. Returns an
        :class:`~vincio.optimize.local_adaptation.AdaptationResult`::

            results = [app.run(q) for q in prompts]
            result = app.adapt_locally(golden, runs=results)
            result.promoted, result.verdict.delta
        """
        controller = self.local_adaptation(
            policy, dataset=dataset, registry=registry, base_model=base_model
        )
        return controller.adapt(runs=runs, training_set=training_set, apply=apply)

    def federated_improvement(self: ContextApp, policy: Any | None = None, **kwargs: Any):  # type: ignore[misc]
        """The cross-org federated-improvement round, as a streaming controller.

        Returns a
        :class:`~vincio.optimize.federated.FederatedImprovement` driven by a
        :class:`~vincio.optimize.federated.FederatedPolicy`: securely aggregate a
        fleet's privacy-preserving :class:`~vincio.optimize.federated.Contribution`\\ s
        into a shared :class:`~vincio.optimize.federated.FederatedSubspace`, re-fit
        *this* member's own on-device adapter against that geometry, gate it against
        the member's base on a held-out set, and adopt or roll back — every version
        in the :class:`~vincio.optimize.local_adaptation.AdapterRegistry`, every
        decision on the shared audit chain and event bus, all in-process::

            ctl = app.federated_improvement(dataset=golden)
            mine = await ctl.build_contribution(member_id="org-a", participants=fleet)
            async for ev in ctl.astream(contributions=[mine, *peer_updates]):
                print(ev.phase, ev.reason)

        Only numeric, masked, bounded-sensitivity aggregates cross a trust
        boundary; adoption clears the same no-regression discipline a hosted
        fine-tune job does."""
        from ..optimize.federated import FederatedImprovement

        return FederatedImprovement(self, policy, **kwargs)

    def contribute_federated(  # type: ignore[misc]
        self: ContextApp,
        *,
        member_id: str,
        participants: list[str] | None = None,
        runs: list[Any] | None = None,
        training_set: Any | None = None,
        policy: Any | None = None,
        consent_subject: str | None = None,
        residency: str | None = None,
    ):
        """Build this member's privacy-preserving contribution to a federated round.

        Curates a grounded training set from this app's own data (``runs``, a
        prebuilt ``training_set``, or captured traces), then returns a
        :class:`~vincio.optimize.federated.Contribution` carrying **only** the
        numeric subspace scatter — clipped, optionally DP-noised, and masked for
        secure aggregation — never a prompt or a response. Enforces the consent
        ledger's TRAINING purpose when the policy requires it and stamps the app's
        residency tag::

            mine = app.contribute_federated(member_id="org-a", participants=fleet)
        """
        controller = self.federated_improvement(policy)
        return run_sync(
            controller.build_contribution(
                member_id=member_id,
                participants=participants,
                runs=runs,
                training_set=training_set,
                consent_subject=consent_subject,
                residency=residency,
            )
        )

    def adopt_federated(  # type: ignore[misc]
        self: ContextApp,
        dataset: Any,
        contributions: list[Any],
        *,
        runs: list[Any] | None = None,
        training_set: Any | None = None,
        policy: Any | None = None,
        registry: Any | None = None,
        base_model: str | None = None,
        apply: bool = True,
    ):
        """Aggregate a fleet's contributions, refit, gate, and adopt — one call.

        The one-shot form of :meth:`federated_improvement`. Securely merges the
        fleet's :class:`~vincio.optimize.federated.Contribution`\\ s into a shared
        subspace, re-fits this member's adapter against it over the member's **own**
        local data (from ``runs``, a ``training_set``, or captured traces), gates it
        against the base on the held-out ``dataset`` (no-regression — at-least-as-good),
        and on a pass registers, makes active, and (with ``apply``) applies it.
        Returns a :class:`~vincio.optimize.federated.FederatedRoundResult`::

            result = app.adopt_federated(golden, [mine, *peer_updates])
            result.adopted, result.verdict.delta, result.privacy.secure_aggregation
        """
        controller = self.federated_improvement(
            policy, dataset=dataset, registry=registry, base_model=base_model
        )
        return controller.adopt(
            contributions=contributions, runs=runs, training_set=training_set, apply=apply
        )

    def use_semantic_context_scoring(  # type: ignore[misc]
        self: ContextApp, enabled: bool = True, *, mmr_lambda: float | None = None
    ) -> ContextApp:
        """Score and select context by embedding cosine instead of lexical overlap.

        When enabled, the context compiler scores relevance, novelty, dedup, and
        conflict by cosine over the app embedder's cached vectors, blends the
        reranker's verdict into relevance, and selects evidence by maximal
        marginal relevance (``mmr_lambda`` trades relevance against diversity).
        Only meaningful with a real semantic embedder configured
        (``retrieval.embedder``); the default hash embedder is not semantic, so
        leave it off unless you've set one::

            app = ContextApp(config={"retrieval": {"embedder": "voyage"}})
            app.use_semantic_context_scoring()
        """
        self.context_compiler.options.semantic_scoring = enabled
        if mmr_lambda is not None:
            self.context_compiler.options.mmr_lambda = mmr_lambda
        return self

    def use_learned_compression(self: ContextApp, compressor: Any | None = None) -> ContextApp:  # type: ignore[misc]
        """Install a learned token-importance compressor on the compiler.

        Replaces the default extractive compressor with a learned one (default:
        :class:`~vincio.context.LLMLinguaCompressor`) for the inline
        budget-overflow compression step. Prefer :meth:`gate_compression` to
        adopt one only after it passes the faithfulness gate::

            app.use_learned_compression()  # opt-in, ungated
        """
        from ..context.llmlingua import LLMLinguaCompressor

        self.context_compiler.compressor = compressor or LLMLinguaCompressor()
        return self

    def gate_compression(  # type: ignore[misc]
        self: ContextApp,
        dataset: Dataset,
        *,
        compressor: Any | None = None,
        metrics: list[str] | None = None,
        min_faithfulness: float = 0.9,
        min_quality_ratio: float = 0.98,
        concurrency: int = 4,
    ):
        """Adopt a learned compressor only if it preserves cited facts and quality.

        Runs the dataset with the baseline and the learned compressor, compares
        faithfulness, quality, and token usage, and installs the learned
        compressor only when it shrinks the prompt without losing the cited-fact
        set or regressing quality — returning the :class:`CompressionTuningResult`
        with the decision::

            result = app.gate_compression(golden)
            result.adopted, result.token_savings, result.learned_faithfulness
        """
        from ..context.compression import extractive_compress
        from ..context.llmlingua import LLMLinguaCompressor
        from ..evals.runners import EvalRunner
        from ..optimize.compression_tuning import CompressionTuner

        learned = compressor or LLMLinguaCompressor()
        metric_list = metrics or ["lexical_overlap", "faithfulness", "input_tokens"]

        async def evaluate(compressor_choice, ds):
            original = self.context_compiler.compressor
            original_write_back = self.config.memory.write_back
            self.context_compiler.compressor = compressor_choice or extractive_compress
            self.config.memory.write_back = []
            try:
                runner = EvalRunner(self, metrics=metric_list, concurrency=concurrency)
                return await runner.arun(ds)
            finally:
                self.context_compiler.compressor = original
                self.config.memory.write_back = original_write_back

        tuner = CompressionTuner(
            evaluate, min_faithfulness=min_faithfulness, min_quality_ratio=min_quality_ratio
        )
        result, chosen = run_sync(tuner.tune(learned, dataset))
        if chosen is not None:
            self.context_compiler.compressor = chosen
            self.events.emit("compression.adopted", {"token_savings": result.token_savings})
        return result

    def calibrate_judge(self: ContextApp, judge: Any, samples: list[Any], *, budget: int = 4):  # type: ignore[misc]
        """Reflectively tune an LLM judge's evaluation steps for κ agreement.

        Proposes alternative evaluation procedures, scores each against the
        labelled ``samples`` (``(case, output, human_score)``), and installs the
        procedure that best agrees with people — only when it strictly beats the
        incumbent — leaving the judge calibrated for CI gating::

            result = app.calibrate_judge(geval, labelled_samples)
            result.adopted, result.kappa_before, result.kappa_after
        """
        from ..optimize.judge_calibration import JudgeCalibrator

        return JudgeCalibrator(judge).calibrate(samples, budget=budget)

    def use_learned_budgets(self: ContextApp, source: Any) -> ContextApp:  # type: ignore[misc]
        """Install eval-tuned per-task budget allocations.

        ``source`` is a :class:`~vincio.optimize.LearnedAllocations`, a path
        to one saved as JSON, or a plain ``{task_type: {block: fraction}}``
        mapping. Tasks without a learned table keep the fixed defaults.
        """
        from ..context.budgeting import BudgetAllocator
        from ..optimize.budget_learning import LearnedAllocations

        if isinstance(source, (str, Path)):
            source = LearnedAllocations.load(source)
        if isinstance(source, LearnedAllocations):
            learned = source.allocations
        else:
            learned = {str(key): dict(value) for key, value in dict(source).items()}
        self.context_compiler.allocator = BudgetAllocator(learned=learned)
        return self

    def use_pack(  # type: ignore[misc]
        self: ContextApp, pack: Any, *, set_schema: bool = True, merge_rules: bool = False
    ) -> ContextApp:
        """Apply a domain pack: prompt config + schema + policies +
        evaluators + rails.

        ``pack`` is a pack name (``"support"``, ``"engineering"``, ``"finance"``,
        ``"legal"``) or a :class:`~vincio.packs.Pack`. Packs are opt-in, ship in
        the package, and configure the app through its public API, so you can
        layer your own settings on top::

            app = ContextApp(name="helpdesk").use_pack("support")
        """
        from ..packs import Pack, load_pack

        if isinstance(pack, str):
            pack = load_pack(pack)
        if not isinstance(pack, Pack):
            raise ConfigError(f"use_pack expects a pack name or Pack, got {type(pack).__name__}")
        pack.apply(self, set_schema=set_schema, merge_rules=merge_rules)
        self.events.emit("pack.applied", {"pack": pack.name})
        return self
