"""Reflective optimization: GEPA-style prompt evolution.

Where the evolution loop mutates configs blindly, the reflective optimizer
*reads the eval report's failures*, reflects in natural language on **why** a
prompt lost, and proposes targeted edits — then verifies each child on the
same gated, Pareto-aware machinery the rest of the optimizer uses.

The win GEPA reports — beating reinforcement learning with far fewer rollouts
— comes from spending the evaluation budget on *informed* proposals: a child
is screened on a minibatch and only earns a full-dataset rollout when it beats
its parent, so most of the budget goes to candidates the reflection already
has reason to believe in. Everything here is deterministic under a seed and
hard-bounded by the evaluation budget.

Two strategies share the same selection and gating path:

* ``"reflective"`` (GEPA) — iterative, failure-driven single edits evolving a
  Pareto frontier.
* ``"mipro"`` (MIPROv2-style) — joint instruction + few-shot proposal: a batch
  of (instruction-rewrite × example-subset) candidates screened together.

The result is a drop-in :class:`OptimizationResult`, so it slots straight into
:class:`~vincio.optimize.loop.ImprovementLoop` promotion — pushed to the
registry, eval-linked, audited — with no new stores.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..evals.datasets import Dataset
from ..evals.reports import EvalReport
from ..prompts.compiler import CompilerOptions
from ..prompts.optimizers import PromptVariant
from ..prompts.templates import PromptSpec

if TYPE_CHECKING:
    from ..providers.base import ModelProvider
from .pareto import (
    ObjectiveSpec,
    ParetoFrontier,
    ParetoPoint,
    dominates,
    objective_vector,
    objectives_from_weights,
)
from .search import (
    Candidate,
    FitnessWeights,
    OptimizationResult,
    _promotion_safe,
    apply_significance_gate,
    fitness,
)

__all__ = [
    "ProposedEdit",
    "Reflection",
    "Reflector",
    "HeuristicReflector",
    "LLMReflector",
    "MIPROProposer",
    "ReflectiveResult",
    "ReflectiveOptimizer",
    "apply_edits",
    "cluster_failures",
]

# evaluate_variant(variant, dataset) -> EvalReport (the app supplies this).
VariantEvaluateFn = Callable[[PromptVariant, Dataset], Awaitable[EvalReport]]

_EDIT_FIELDS = {
    "objective",
    "rules",
    "soft_rules",
    "citation_policy",
    "insufficient_evidence_behavior",
    "output_instructions",
    "reasoning_mode",
    "safety_policies",
    "examples",
    "format",
    "max_examples",
}


class ProposedEdit(BaseModel):
    """One targeted change to a prompt, justified by an observed failure."""

    field: str  # a PromptSpec field, or a compiler option ("format" / "max_examples")
    op: Literal["set", "append", "prepend", "reduce_examples"] = "append"
    value: Any = None
    rationale: str = ""


class Reflection(BaseModel):
    """A natural-language diagnosis of why a candidate lost, plus proposed edits."""

    parent: str
    failures_observed: int = 0
    diagnosis: str = ""
    signals: dict[str, float] = Field(default_factory=dict)
    edits: list[ProposedEdit] = Field(default_factory=list)


class Reflector:
    """Proposes targeted prompt edits from an eval report. Subclass to plug in
    a different reflection strategy (e.g. a hosted model)."""

    def reflect(
        self,
        spec: PromptSpec,
        report: EvalReport | None,
        *,
        objectives: list[ObjectiveSpec],
        dataset: Dataset | None = None,
    ) -> Reflection:  # pragma: no cover - interface
        raise NotImplementedError


def _signal(report: EvalReport | None, metric: str, default: float) -> float:
    if report is None:
        return default
    values = report.metric_values(metric)
    return sum(values) / len(values) if values else default


def _invoke_reflect(
    reflector: Reflector,
    spec: PromptSpec,
    report: EvalReport | None,
    *,
    objectives: list[ObjectiveSpec],
    dataset: Dataset | None,
) -> Reflection:
    """Call ``reflect`` with the dataset when the reflector accepts it.

    The ``dataset`` keyword is additive: a custom :class:`Reflector`
    written against the older 4-argument signature still works — we retry
    without it on the resulting ``TypeError``.
    """
    try:
        return reflector.reflect(spec, report, objectives=objectives, dataset=dataset)
    except TypeError:
        return reflector.reflect(spec, report, objectives=objectives)


class HeuristicReflector(Reflector):
    """Deterministic, offline reflection.

    Reads the report's per-metric means and failing cases and maps each
    weakness to a concrete, minimal prompt edit — the same moves a careful
    prompt engineer makes when a metric sags. No model call, fully
    reproducible. Edits are emitted in a fixed priority order and de-duplicated
    against what the spec already says, so a single reflection proposes the one
    or two highest-leverage changes rather than a scattershot rewrite.
    """

    def __init__(
        self,
        *,
        groundedness_floor: float = 0.85,
        schema_floor: float = 0.98,
        accuracy_floor: float = 0.8,
        safety_floor: float = 0.99,
        cost_ceiling: float | None = None,
        max_edits: int = 2,
    ) -> None:
        self.groundedness_floor = groundedness_floor
        self.schema_floor = schema_floor
        self.accuracy_floor = accuracy_floor
        self.safety_floor = safety_floor
        self.cost_ceiling = cost_ceiling
        self.max_edits = max_edits

    def reflect(
        self,
        spec: PromptSpec,
        report: EvalReport | None,
        *,
        objectives: list[ObjectiveSpec],
        dataset: Dataset | None = None,
    ) -> Reflection:
        accuracy_metric = next(
            (o.metric for o in objectives if o.name in ("accuracy", "goal")),
            "lexical_overlap",
        )
        grounded = _signal(report, "groundedness", 1.0)
        schema = _signal(report, "schema_validity", 1.0)
        accuracy = _signal(report, accuracy_metric, 1.0)
        safety = _signal(report, "safety", 1.0)
        toxicity = _signal(report, "toxicity", 0.0)
        cost = _signal(report, "cost", 0.0)
        failures = report.failures(metric=accuracy_metric, threshold=0.5) if report else []
        signals = {
            "groundedness": round(grounded, 4),
            "schema_validity": round(schema, 4),
            "accuracy": round(accuracy, 4),
            "safety": round(safety, 4),
            "cost": round(cost, 6),
        }

        candidates: list[tuple[float, ProposedEdit, str]] = []  # (priority, edit, diagnosis)

        # Safety first: never trade it away.
        if safety < self.safety_floor or toxicity > 0.05:
            policy = "Refuse unsafe, biased, or policy-violating requests and never produce harmful content."
            if policy not in spec.safety_policies:
                candidates.append(
                    (
                        0.0,
                        ProposedEdit(
                            field="safety_policies",
                            op="append",
                            value=policy,
                            rationale=f"safety {safety:.2f} below floor / toxicity {toxicity:.2f}",
                        ),
                        "outputs drifted unsafe; added an explicit refusal policy",
                    )
                )

        # Groundedness: tighten the citation contract and the evidence stance.
        if grounded < self.groundedness_floor:
            if not spec.citation_policy:
                candidates.append(
                    (
                        1.0,
                        ProposedEdit(
                            field="citation_policy",
                            op="set",
                            value="Cite the evidence ID in square brackets for every claim you make.",
                            rationale=f"groundedness {grounded:.2f} below floor",
                        ),
                        "answers were under-cited; required bracketed evidence IDs",
                    )
                )
            elif spec.reasoning_mode != "evidence_first":
                candidates.append(
                    (
                        1.1,
                        ProposedEdit(
                            field="reasoning_mode",
                            op="set",
                            value="evidence_first",
                            rationale=f"groundedness {grounded:.2f} below floor; force evidence-first reasoning",
                        ),
                        "claims ran ahead of evidence; switched to evidence-first reasoning",
                    )
                )
            elif not spec.insufficient_evidence_behavior:
                candidates.append(
                    (
                        1.2,
                        ProposedEdit(
                            field="insufficient_evidence_behavior",
                            op="set",
                            value="If the evidence does not support an answer, say so instead of guessing.",
                            rationale="ground unsupported answers in an explicit abstention rule",
                        ),
                        "model guessed without support; added an abstention rule",
                    )
                )

        # Schema validity: make the output contract louder.
        if schema < self.schema_floor:
            rule = "Return only the requested structured output, with no prose before or after it."
            if rule not in spec.rules and rule not in spec.output_instructions:
                candidates.append(
                    (
                        2.0,
                        ProposedEdit(
                            field="output_instructions",
                            op="append",
                            value=rule,
                            rationale=f"schema_validity {schema:.2f} below floor",
                        ),
                        "structured output was malformed; reinforced the output contract",
                    )
                )

        # Accuracy: add deliberate reasoning when answers miss.
        if accuracy < self.accuracy_floor and failures:
            if spec.reasoning_mode == "direct":
                candidates.append(
                    (
                        3.0,
                        ProposedEdit(
                            field="reasoning_mode",
                            op="set",
                            value="plan",
                            rationale=f"accuracy {accuracy:.2f} below floor on {len(failures)} cases",
                        ),
                        f"{len(failures)} cases answered wrong; added a plan-then-answer step",
                    )
                )
            else:
                focus = "Answer the exact question asked; do not add unrequested information."
                if focus not in spec.rules:
                    candidates.append(
                        (
                            3.1,
                            ProposedEdit(
                                field="rules",
                                op="append",
                                value=focus,
                                rationale=f"accuracy {accuracy:.2f} below floor on {len(failures)} cases",
                            ),
                            "answers drifted off-question; added a focus rule",
                        )
                    )

        # Cost/latency: trim few-shot weight when the budget is the problem.
        if self.cost_ceiling is not None and cost > self.cost_ceiling and len(spec.examples) > 1:
            candidates.append(
                (
                    4.0,
                    ProposedEdit(
                        field="examples",
                        op="reduce_examples",
                        value=max(1, len(spec.examples) // 2),
                        rationale=f"cost {cost:.4f} over ceiling {self.cost_ceiling:.4f}",
                    ),
                    "prompt was overlong for its accuracy; halved the few-shot examples",
                )
            )

        candidates.sort(key=lambda item: item[0])
        chosen = candidates[: self.max_edits]
        diagnosis = "; ".join(item[2] for item in chosen) or "no actionable weakness detected"
        return Reflection(
            parent=spec.name,
            failures_observed=len(failures),
            diagnosis=diagnosis,
            signals=signals,
            edits=[item[1] for item in chosen],
        )


_REFLECT_SYSTEM = """\
You are a prompt-optimization analyst. You are given the failing evaluation cases \
of a prompt, already clustered into failure modes, plus the prompt's current \
fields. Diagnose *why* the prompt loses on these cases, then propose the one or \
two highest-leverage, minimal edits that would fix the dominant failure mode \
without regressing the others.

Return ONLY a JSON object of the form:
{"diagnosis": "<one sentence>", "edits": [{"field": "<field>", "op": "<op>", \
"value": <value>, "rationale": "<why>"}]}

Allowed fields: objective, rules, soft_rules, citation_policy, \
insufficient_evidence_behavior, output_instructions, reasoning_mode, \
safety_policies, examples, format, max_examples.
Allowed ops: set, append, prepend, reduce_examples.
reasoning_mode values: direct, plan, evidence_first, react.
Do not invent fields or ops outside the lists."""


def cluster_failures(
    report: EvalReport | None,
    dataset: Dataset | None,
    *,
    accuracy_metric: str = "lexical_overlap",
    max_cases: int = 12,
) -> list[dict[str, Any]]:
    """Group failing cases into deterministic failure modes for reflection.

    Each failing case is assigned to the metric on which it falls furthest below
    a nominal floor (groundedness / schema_validity / accuracy / safety / errors),
    so the model reflects on *modes* — "answers were under-cited", "output was
    malformed" — rather than a flat list. Joins the report's cases with the
    dataset's :class:`EvalCase`\\ s by id to surface the input, expected answer,
    and grounding evidence the case actually saw.
    """
    if report is None:
        return []
    floors = {
        "groundedness": ("below", 0.85),
        "schema_validity": ("below", 0.98),
        accuracy_metric: ("below", 0.8),
        "safety": ("below", 0.99),
        "toxicity": ("above", 0.05),
    }
    by_id = {case.id: case for case in (dataset.cases if dataset else [])}
    modes: dict[str, list[dict[str, Any]]] = {}
    for case in report.cases:
        if case.failed:
            mode = "error"
            severity = 1.0
        else:
            worst_mode = None
            worst_gap = 0.0
            for metric, (direction, floor) in floors.items():
                value = case.metrics.get(metric)
                if value is None:
                    continue
                gap = (floor - value) if direction == "below" else (value - floor)
                if gap > worst_gap:
                    worst_gap = gap
                    worst_mode = metric
            if worst_mode is None:
                continue  # case passed every floor — not a failure
            mode, severity = worst_mode, worst_gap
        eval_case = by_id.get(case.case_id)
        modes.setdefault(mode, []).append(
            {
                "case_id": case.case_id,
                "severity": round(severity, 4),
                "input": (eval_case.input_text[:280] if eval_case else ""),
                "expected": (str(eval_case.expected)[:280] if eval_case and eval_case.expected else ""),
                "output": case.output_text[:280],
                "error": case.error,
            }
        )
    clusters: list[dict[str, Any]] = [
        {"mode": mode, "count": len(cases),
         "cases": sorted(cases, key=lambda c: c["severity"], reverse=True)[:3]}
        for mode, cases in modes.items()
    ]
    clusters.sort(key=lambda c: int(c["count"]), reverse=True)
    # Bound the total sample cases handed to the model.
    budget = max_cases
    for cluster in clusters:
        sample: list[dict[str, Any]] = cluster["cases"]
        cluster["cases"] = sample[: max(1, budget)]
        budget -= len(cluster["cases"])
        if budget <= 0:
            budget = 1
    return clusters


class LLMReflector(Reflector):
    """Provider-backed reflection (GEPA proper) with a deterministic fallback.

    Wired to the app's own provider, the reflector reads the *actual* failing
    cases — input, model output, expected answer, and the evidence that grounded
    them — clusters them into failure modes (:func:`cluster_failures`), and asks
    the model to diagnose the dominant mode and propose targeted edits. The edits
    are validated against the same allowed field/op set as every other reflector,
    so the model can only move knobs the optimizer already understands.

    Offline, on any provider error, or on an unparseable / empty proposal, it
    falls back to :class:`HeuristicReflector`, so air-gapped and test runs stay
    reproducible. An explicit ``propose`` callable overrides the provider path
    entirely (deterministic injection); supplying neither a provider nor a
    callable makes the reflector a thin pass-through to the heuristic floor.
    """

    def __init__(
        self,
        provider: ModelProvider | None = None,
        model: str | None = None,
        *,
        propose: Callable[[PromptSpec, EvalReport | None], list[dict[str, Any]]] | None = None,
        fallback: Reflector | None = None,
        max_edits: int = 3,
        max_cases: int = 12,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self._propose = propose
        self._fallback = fallback or HeuristicReflector()
        self.max_edits = max_edits
        self.max_cases = max_cases
        self.temperature = temperature

    def reflect(
        self,
        spec: PromptSpec,
        report: EvalReport | None,
        *,
        objectives: list[ObjectiveSpec],
        dataset: Dataset | None = None,
    ) -> Reflection:
        accuracy_metric = next(
            (o.metric for o in objectives if o.name in ("accuracy", "goal")),
            "lexical_overlap",
        )
        diagnosis = "model-proposed edits"
        raw: list[dict[str, Any]] | None = None
        if self._propose is not None:
            try:
                raw = self._propose(spec, report)
            except Exception:  # noqa: BLE001 - any proposer error → deterministic floor
                raw = None
        elif self.provider is not None:
            proposal = self._reflect_via_provider(spec, report, dataset, accuracy_metric)
            if proposal is not None:
                raw, diagnosis = proposal
        else:
            return self._fallback.reflect(spec, report, objectives=objectives, dataset=dataset)

        edits = self._validate(raw)
        if not edits:
            return self._fallback.reflect(spec, report, objectives=objectives, dataset=dataset)
        return Reflection(
            parent=spec.name,
            failures_observed=len(report.failures(metric=accuracy_metric)) if report else 0,
            diagnosis=diagnosis,
            signals={
                "groundedness": round(_signal(report, "groundedness", 1.0), 4),
                "schema_validity": round(_signal(report, "schema_validity", 1.0), 4),
                "accuracy": round(_signal(report, accuracy_metric, 1.0), 4),
            },
            edits=edits[: self.max_edits],
        )

    def _validate(self, raw: list[dict[str, Any]] | None) -> list[ProposedEdit]:
        edits: list[ProposedEdit] = []
        for item in raw or []:
            try:
                edit = ProposedEdit.model_validate(item)
            except (ValueError, TypeError):
                continue
            if edit.field in _EDIT_FIELDS:
                edits.append(edit)
        return edits

    def _reflect_via_provider(
        self,
        spec: PromptSpec,
        report: EvalReport | None,
        dataset: Dataset | None,
        accuracy_metric: str,
    ) -> tuple[list[dict[str, Any]], str] | None:
        from ..core.types import Message, ModelRequest
        from ..output.parsers import extract_json
        from ..providers.base import run_sync

        clusters = cluster_failures(
            report, dataset, accuracy_metric=accuracy_metric, max_cases=self.max_cases
        )
        if not clusters:
            return None
        user = self._build_user_prompt(spec, clusters)
        request = ModelRequest(
            model=self.model or "",
            messages=[
                Message(
                    role="system",
                    content=f"{_REFLECT_SYSTEM}\nPropose at most {self.max_edits} edits.",
                ),
                Message(role="user", content=user),
            ],
            temperature=self.temperature,
            max_output_tokens=1024,
        )
        try:
            response = run_sync(self.provider.generate(request))  # type: ignore[union-attr]
            parsed = extract_json(response.text or "")
        except Exception:  # noqa: BLE001 - provider/parse failure → caller falls back
            return None
        if isinstance(parsed, dict):
            edits = parsed.get("edits")
            diagnosis = str(parsed.get("diagnosis") or "model-proposed edits")
        elif isinstance(parsed, list):
            edits, diagnosis = parsed, "model-proposed edits"
        else:
            return None
        if not isinstance(edits, list):
            return None
        return edits, diagnosis

    @staticmethod
    def _build_user_prompt(spec: PromptSpec, clusters: list[dict[str, Any]]) -> str:
        import json

        lines = ["Current prompt fields:"]
        lines.append(f"- objective: {spec.objective!r}")
        lines.append(f"- reasoning_mode: {spec.reasoning_mode!r}")
        lines.append(f"- citation_policy: {spec.citation_policy!r}")
        lines.append(f"- rules: {list(spec.rules)}")
        lines.append(f"- examples: {len(spec.examples)} few-shot example(s)")
        lines.append("\nFailure modes (most common first):")
        for cluster in clusters:
            lines.append(f"\n## {cluster['mode']} — {cluster['count']} case(s)")
            for case in cluster["cases"]:
                lines.append(json.dumps(case, ensure_ascii=False))
        return "\n".join(lines)


def apply_edits(
    spec: PromptSpec,
    options: CompilerOptions,
    edits: list[ProposedEdit],
) -> tuple[PromptSpec, CompilerOptions]:
    """Apply reflection edits, returning a new (spec, compiler options) pair.

    Edits are additive and field-scoped: list fields append/prepend, scalar
    fields set, and ``reduce_examples`` shrinks the few-shot set. Unknown
    fields are ignored, so a malformed proposal can never corrupt the spec.
    """
    spec_update: dict[str, Any] = {}
    opt_update: dict[str, Any] = {}
    for edit in edits:
        field = edit.field
        if field in ("format", "max_examples"):
            opt_update[field] = edit.value
            continue
        if field == "examples" and edit.op == "reduce_examples":
            keep = int(edit.value) if edit.value is not None else max(1, len(spec.examples) // 2)
            current = spec_update.get("examples", list(spec.examples))
            spec_update["examples"] = current[:keep]
            opt_update["max_examples"] = max(1, keep)
            continue
        if field not in _EDIT_FIELDS:
            continue
        current_value = spec_update.get(field, getattr(spec, field, None))
        if field in ("rules", "soft_rules", "safety_policies"):
            base = list(current_value or [])
            value = edit.value if isinstance(edit.value, list) else [edit.value]
            value = [str(v) for v in value if v is not None]
            if edit.op == "prepend":
                spec_update[field] = [*value, *base]
            else:  # append / set onto a list field appends
                spec_update[field] = [*base, *[v for v in value if v not in base]]
        elif field == "output_instructions":
            existing = current_value or ""
            text = str(edit.value or "")
            if edit.op == "set" or not existing:
                spec_update[field] = text
            elif text not in existing:
                spec_update[field] = f"{existing} {text}".strip()
        elif field == "reasoning_mode":
            # Setting the mode also injects its preamble rule, so the edit
            # actually changes the rendered prompt (mirroring how variant
            # generation applies a reasoning mode).
            from ..prompts.optimizers import REASONING_PREAMBLES

            spec_update[field] = edit.value
            preamble = REASONING_PREAMBLES.get(str(edit.value), "")
            if preamble:
                base = list(spec_update.get("rules", list(spec.rules)))
                if preamble not in base:
                    spec_update["rules"] = [*base, preamble]
        else:  # scalar string fields: set
            spec_update[field] = edit.value
    new_spec = spec.model_copy(update=spec_update) if spec_update else spec
    new_options = options.model_copy(update=opt_update) if opt_update else options
    return new_spec, new_options


class MIPROProposer:
    """MIPROv2-style joint instruction + few-shot proposal.

    Rather than evolving one edit at a time, it proposes a small batch of
    *combinations* — each pairs an instruction rewrite (a focusing rule, an
    evidence-first stance, a structured-output reminder) with a few-shot subset
    size — so instruction and demonstration are searched together, not in
    sequence. The batch is screened and verified by the same frontier pipeline
    as the reflective strategy.
    """

    INSTRUCTIONS: list[tuple[str, ProposedEdit | None]] = [
        ("baseline", None),
        (
            "evidence_first",
            ProposedEdit(field="reasoning_mode", op="set", value="evidence_first"),
        ),
        (
            "plan",
            ProposedEdit(field="reasoning_mode", op="set", value="plan"),
        ),
        (
            "focus",
            ProposedEdit(
                field="rules",
                op="append",
                value="Answer the exact question asked; do not add unrequested information.",
            ),
        ),
    ]

    def propose(
        self,
        spec: PromptSpec,
        options: CompilerOptions,
        *,
        max_candidates: int,
        rng: random.Random,
    ) -> list[PromptVariant]:
        example_counts = sorted({0, min(2, len(spec.examples)), len(spec.examples)})
        variants: list[PromptVariant] = []
        seen: set[str] = set()
        for label, edit in self.INSTRUCTIONS:
            for n_examples in example_counts:
                edits = [e for e in (edit,) if e is not None]
                edits.append(
                    ProposedEdit(field="examples", op="reduce_examples", value=max(0, n_examples) or 1)
                )
                child_spec, child_opts = apply_edits(spec, options, edits)
                child_spec = child_spec.model_copy(update={"examples": spec.examples[:n_examples]})
                child_opts = child_opts.model_copy(update={"max_examples": max(1, n_examples)})
                name = f"mipro:{label}-ex{n_examples}"
                if name in seen:
                    continue
                seen.add(name)
                variants.append(
                    PromptVariant(
                        name=name,
                        spec=child_spec,
                        compiler_options=child_opts,
                        dimensions={"instruction": label, "examples": n_examples},
                    )
                )
        rng.shuffle(variants)
        return variants[:max_candidates]


class ReflectiveResult(OptimizationResult):
    """An :class:`OptimizationResult` enriched with the reflection trace and the
    evolved Pareto frontier. The base fields keep it a drop-in for the
    improvement loop; the extra fields explain *how* the winner was found."""

    strategy: str = "reflective"
    rounds: int = 0
    evaluations: int = 0
    reflections: list[Reflection] = Field(default_factory=list)
    frontier: ParetoFrontier | None = None


class ReflectiveOptimizer:
    """GEPA-style reflective prompt optimizer.

    Mirrors :class:`~vincio.optimize.prompt_search.PromptOptimizer`'s
    constructor (an ``evaluate_variant`` coroutine plus weights/gates), so it is
    interchangeable in the improvement loop. ``optimize`` returns a
    :class:`ReflectiveResult` whose ``best.payload`` is a ready-to-promote
    :class:`~vincio.prompts.optimizers.PromptVariant`.
    """

    def __init__(
        self,
        evaluate_variant: VariantEvaluateFn,
        *,
        weights: FitnessWeights | None = None,
        gates: dict[str, str] | None = None,
        max_cost_per_case: float | None = None,
        objectives: list[ObjectiveSpec] | None = None,
        reflector: Reflector | None = None,
        constraints: dict[str, float] | None = None,
        prefer: str | None = None,
    ) -> None:
        self.evaluate_variant = evaluate_variant
        self.weights = weights or FitnessWeights()
        self.gates = gates
        self.max_cost_per_case = max_cost_per_case
        # Frontier objectives default to the axes the weights actually care about,
        # so a zero-weighted axis (e.g. latency) can't let measurement jitter flip
        # the knee-point pick. An explicit ``objectives`` list overrides this.
        self.objectives = objectives or objectives_from_weights(self.weights)
        self.reflector = reflector or HeuristicReflector(cost_ceiling=max_cost_per_case)
        self.constraints = constraints
        self.prefer = prefer

    async def optimize(
        self,
        spec: PromptSpec,
        dataset: Dataset,
        *,
        strategy: Literal["reflective", "mipro"] = "reflective",
        budget: int = 12,
        minibatch_size: int = 8,
        seed: int = 7,
        min_dataset_coverage: int = 4,
        screen_tolerance: float = 1e-6,
    ) -> ReflectiveResult:
        if len(dataset) < min_dataset_coverage:
            return ReflectiveResult(
                baseline_fitness=float("nan"),
                strategy=strategy,
                reason=(
                    f"dataset too small ({len(dataset)} cases < {min_dataset_coverage}); "
                    "refusing to optimize"
                ),
            )
        rng = random.Random(seed)
        base_options = CompilerOptions(format="markdown", max_examples=max(1, len(spec.examples)))

        async def eval_variant(variant: PromptVariant, ds: Dataset) -> EvalReport:
            return await self.evaluate_variant(variant, ds)

        # Baseline: one full-dataset rollout anchors the frontier and the gate.
        baseline_variant = PromptVariant(
            name=f"{spec.name}:baseline", spec=spec, compiler_options=base_options, dimensions={}
        )
        baseline = Candidate(name=baseline_variant.name, payload=baseline_variant)
        baseline.full_report = await eval_variant(baseline_variant, dataset)
        baseline.full_fitness = fitness(baseline.full_report, self.weights)
        evaluations = 1

        points: list[ParetoPoint] = [
            ParetoPoint(
                name=baseline.name,
                objectives=objective_vector(baseline.full_report, self.objectives),
                candidate=baseline,
            )
        ]
        candidates: list[Candidate] = []
        reflections: list[Reflection] = []
        history: list[dict[str, Any]] = [
            {"phase": "baseline", "name": baseline.name, "fitness": baseline.full_fitness}
        ]
        rounds = 0

        if strategy == "mipro":
            evaluations, rounds = await self._run_mipro(
                spec, base_options, dataset, baseline, points, candidates, history, rng,
                budget=budget, minibatch_size=minibatch_size, seed=seed,
                screen_tolerance=screen_tolerance, evaluations=evaluations,
            )
        else:
            evaluations, rounds = await self._run_reflective(
                base_options, dataset, baseline, points, candidates, reflections, history, rng,
                budget=budget, minibatch_size=minibatch_size, seed=seed,
                screen_tolerance=screen_tolerance, evaluations=evaluations,
            )

        frontier = ParetoFrontier.build(points, specs=self.objectives)
        result = ReflectiveResult(
            baseline_fitness=baseline.full_fitness,
            baseline=baseline,
            candidates=candidates,
            history=history,
            strategy=strategy,
            rounds=rounds,
            evaluations=evaluations,
            reflections=reflections,
            frontier=frontier,
        )
        return self._select(result, frontier, baseline)

    # -- strategies -----------------------------------------------------------------

    async def _run_reflective(
        self, base_options, dataset, baseline, points, candidates, reflections, history, rng,
        *, budget, minibatch_size, seed, screen_tolerance, evaluations,
    ) -> tuple[int, int]:
        rounds = 0
        stale = 0
        while evaluations < budget and stale < 4:
            frontier = ParetoFrontier.build([p.model_copy() for p in points], specs=self.objectives)
            front = frontier.front or points
            parent_point = front[rounds % len(front)]
            parent = parent_point.candidate or baseline
            rounds += 1

            reflection = _invoke_reflect(
                self.reflector, parent.payload.spec, parent.full_report,
                objectives=self.objectives, dataset=dataset,
            )
            reflections.append(reflection)
            if not reflection.edits:
                stale += 1
                continue
            stale = 0
            child_spec, child_opts = apply_edits(
                parent.payload.spec, parent.payload.compiler_options, reflection.edits
            )
            child_variant = PromptVariant(
                name=f"reflective:r{rounds}",
                spec=child_spec.model_copy(update={"name": f"reflective_r{rounds}"}),
                compiler_options=child_opts,
                dimensions={"parent": parent.name, "edits": [e.field for e in reflection.edits]},
            )
            child = Candidate(name=child_variant.name, params=child_variant.dimensions, payload=child_variant)
            candidates.append(child)

            minibatch = dataset.sample(minibatch_size, seed=seed + rounds)
            child.subset_report = await self.evaluate_variant(child_variant, minibatch)
            child.subset_fitness = fitness(child.subset_report, self.weights)
            evaluations += 1
            history.append(
                {"phase": "reflect", "name": child.name, "parent": parent.name,
                 "diagnosis": reflection.diagnosis, "fitness": child.subset_fitness}
            )

            # Sample-efficiency gate: only spend a full rollout on a child that
            # already beats its parent on the screening minibatch.
            if child.subset_fitness < (parent.full_fitness or float("-inf")) - screen_tolerance:
                history.append({"phase": "skip_full", "name": child.name, "fitness": child.subset_fitness})
                continue
            if evaluations >= budget:
                break
            child.full_report = await self.evaluate_variant(child_variant, dataset)
            child.full_fitness = fitness(child.full_report, self.weights)
            evaluations += 1
            points.append(
                ParetoPoint(
                    name=child.name,
                    objectives=objective_vector(child.full_report, self.objectives),
                    candidate=child,
                )
            )
            history.append({"phase": "full", "name": child.name, "fitness": child.full_fitness})
        return evaluations, rounds

    async def _run_mipro(
        self, spec, base_options, dataset, baseline, points, candidates, history, rng,
        *, budget, minibatch_size, seed, screen_tolerance, evaluations,
    ) -> tuple[int, int]:
        proposer = MIPROProposer()
        remaining = max(0, budget - evaluations)
        # Reserve roughly half the budget for full-dataset verification.
        max_candidates = max(1, remaining - max(1, remaining // 2))
        variants = proposer.propose(spec, base_options, max_candidates=max_candidates, rng=rng)
        minibatch = dataset.sample(minibatch_size, seed=seed)
        screened: list[Candidate] = []
        for variant in variants:
            if evaluations >= budget:
                break
            child = Candidate(name=variant.name, params=variant.dimensions, payload=variant)
            child.subset_report = await self.evaluate_variant(variant, minibatch)
            child.subset_fitness = fitness(child.subset_report, self.weights)
            evaluations += 1
            candidates.append(child)
            screened.append(child)
            history.append({"phase": "screen", "name": child.name, "fitness": child.subset_fitness})
        screened.sort(key=lambda c: c.subset_fitness or float("-inf"), reverse=True)
        for child in screened:
            if evaluations >= budget:
                break
            if (child.subset_fitness or float("-inf")) < (baseline.full_fitness or float("-inf")) - screen_tolerance:
                continue
            child.full_report = await self.evaluate_variant(child.payload, dataset)
            child.full_fitness = fitness(child.full_report, self.weights)
            evaluations += 1
            points.append(
                ParetoPoint(
                    name=child.name,
                    objectives=objective_vector(child.full_report, self.objectives),
                    candidate=child,
                )
            )
            history.append({"phase": "full", "name": child.name, "fitness": child.full_fitness})
        return evaluations, len(variants)

    # -- selection + gate -----------------------------------------------------------

    def _select(
        self, result: ReflectiveResult, frontier: ParetoFrontier, baseline: Candidate
    ) -> ReflectiveResult:
        pick = frontier.select(constraints=self.constraints, prefer=self.prefer)
        if pick is None:
            result.reason = "no frontier point satisfies the constraints"
            return result
        if pick.name == baseline.name:
            result.reason = "baseline is the selected frontier point; no reflective gain"
            return result
        candidate = pick.candidate
        result.best = candidate
        baseline_objectives = objective_vector(baseline.full_report, self.objectives)
        improves = dominates(pick.objectives, baseline_objectives, self.objectives) or (
            (candidate.full_fitness or float("-inf")) > (baseline.full_fitness or 0.0)
        )
        if not improves:
            result.best = None
            result.reason = "selected point neither dominates the baseline nor improves fitness"
            return result
        safe, reason = _promotion_safe(
            candidate.full_report,
            baseline.full_report,
            gates=self.gates,
            max_cost_per_case=self.max_cost_per_case,
        )
        if not safe:
            result.promoted = False
            result.reason = reason
            return result
        # Significance gate: the reflective path gets the same statistical
        # backing as the evolution path — block a significant primary-metric
        # regression and record the verdict on the result/audit.
        blocked, sig_reason = apply_significance_gate(
            result,
            baseline_report=baseline.full_report,
            candidate_report=candidate.full_report,
            accuracy_metric=self.weights.accuracy_metric,
        )
        if blocked:
            result.best = None
            result.promoted = False
            result.reason = sig_reason or "primary metric significantly regressed"
            return result
        result.promoted = True
        sig = result.significance
        detail = f" (p={sig['p_value']}, effect={sig['effect_size']})" if sig is not None else ""
        result.reason = (
            f"reflective promotion: {pick.name} "
            f"(fitness {baseline.full_fitness:.4f} → {candidate.full_fitness:.4f}, "
            f"{len(frontier.front)} non-dominated points, {result.evaluations} rollouts){detail}"
        )
        return result
