"""Distillation / fine-tune data flywheel (1.4).

The one lever the rest of the field is missing: turn the traces production runs
already write into *cheaper inference*. Two pieces, both grounded and gated:

* :func:`export_training_set` curates traces into provider-ready fine-tuning
  **JSONL** — feedback-filtered, grounding-checked against the cited evidence,
  deduped, and carrying full provenance on every example. The flywheel never
  trains on hallucinations: an example whose answer the evidence does not
  support is dropped, not exported.
* :class:`BootstrapFinetune` runs the teacher → student loop. A cheaper student
  (optionally fine-tuned on the teacher's grounded traces via an injected
  trainer) is promoted into a runtime :class:`~vincio.optimize.routing.ModelCascade`
  **only** when it holds quality on the eval suite within tolerance while
  costing less — the same gated, eval-driven discipline as every other promotion.

Everything reuses what 0.8 shipped — ``dataset_from_traces`` and the grounded-
fact extractor — and the routing cascade from 1.3. No new stores, no SDKs;
JSONL is emitted with the standard library only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..core.types import EvidenceItem
from ..evals.datasets import Dataset
from ..evals.reports import EvalReport
from ..memory.facts import extract_grounded_facts
from .routing import ModelCascade
from .search import _promotion_safe

if TYPE_CHECKING:
    from ..observability.spans import Trace
    from ..providers.finetune import FineTuneBackend
    from ..providers.registry import ModelRegistry
    from ..retrieval.embeddings import Embedder

__all__ = [
    "TrainingExample",
    "TrainingSet",
    "export_training_set",
    "export_training_set_from_runs",
    "semantic_dedupe",
    "provider_trainer",
    "DistillationResult",
    "BootstrapFinetune",
]

TrainingFormat = Literal["openai", "anthropic"]


class TrainingExample(BaseModel):
    """One supervised fine-tuning example with grounding provenance.

    ``messages`` is the provider-neutral chat transcript (system / user /
    assistant). The grounding fields record *why* the example was admitted, so
    an exported set is auditable: which evidence supported the answer and how
    strongly.
    """

    messages: list[dict[str, str]]
    support: float = 1.0  # max evidence support for the assistant turn, 0..1
    grounded: bool = True
    evidence_ids: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @property
    def system(self) -> str:
        return next((m["content"] for m in self.messages if m.get("role") == "system"), "")

    def _dialog(self) -> list[dict[str, str]]:
        return [m for m in self.messages if m.get("role") != "system"]

    def to_record(self, fmt: TrainingFormat) -> dict[str, Any]:
        """Render one provider fine-tuning record.

        OpenAI expects a flat ``messages`` array including the system turn;
        Anthropic expects a top-level ``system`` string plus user/assistant
        ``messages``. Both are emitted from the same neutral transcript.
        """
        if fmt == "anthropic":
            record: dict[str, Any] = {"messages": self._dialog()}
            if self.system:
                record["system"] = self.system
            return record
        return {"messages": list(self.messages)}


class TrainingSet(BaseModel):
    """A curated, grounded fine-tuning corpus."""

    name: str = "training_set"
    examples: list[TrainingExample] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)

    def to_jsonl(self, *, format: TrainingFormat = "openai") -> str:
        return "\n".join(
            json.dumps(example.to_record(format), ensure_ascii=False) for example in self.examples
        )

    def save(self, path: str | Path, *, format: TrainingFormat = "openai") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = self.to_jsonl(format=format)
        path.write_text(body + ("\n" if body else ""), encoding="utf-8")
        return path

    @property
    def grounded_fraction(self) -> float:
        if not self.examples:
            return 1.0
        return round(sum(1 for e in self.examples if e.grounded) / len(self.examples), 4)


# evidence_for(trace) -> evidence items the run cited (for grounding checks).
EvidenceResolver = Callable[["Trace"], list[EvidenceItem]]


def _evidence_from_trace(trace: Trace) -> list[EvidenceItem]:
    """Best-effort evidence recovery from a trace's recorded attributes.

    Runs that persisted their cited evidence under ``attributes['evidence']``
    (a list of ``{id, text, ...}`` dicts) get a real grounding check; runs that
    did not return an empty list, which the caller treats per ``require_grounding``.
    """
    raw = trace.attributes.get("evidence")
    if not isinstance(raw, list):
        return []
    items: list[EvidenceItem] = []
    for entry in raw:
        if isinstance(entry, EvidenceItem):
            items.append(entry)
        elif isinstance(entry, dict):
            try:
                items.append(EvidenceItem.model_validate(entry))
            except (ValueError, TypeError):
                continue
    return items


def _grounded_example(
    input_text: str,
    output_text: str,
    evidence: list[EvidenceItem],
    *,
    system: str,
    require_grounding: bool,
    min_support: float,
    provenance: dict[str, Any],
) -> TrainingExample | None:
    """Build one grounded training example, or ``None`` if it fails the gate.

    Grounding reuses the deterministic extractor 0.8 uses for auto-memory: the
    output must contain at least one claim the supplied evidence supports at
    ``min_support``. With ``require_grounding`` off, an example whose evidence
    does not support it is kept but flagged ``grounded=False``.
    """
    support = 1.0
    evidence_ids: list[str] = []
    grounded = True
    if evidence:
        facts = extract_grounded_facts(str(output_text), evidence, min_support=min_support)
        grounded = bool(facts)
        support = max((f.support for f in facts), default=0.0)
        evidence_ids = sorted({eid for f in facts for eid in f.evidence_ids})
    elif require_grounding:
        grounded = False
        support = 0.0
    if require_grounding and not grounded:
        return None
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(input_text)})
    messages.append({"role": "assistant", "content": str(output_text)})
    return TrainingExample(
        messages=messages,
        support=round(support, 4),
        grounded=grounded,
        evidence_ids=evidence_ids,
        provenance=provenance,
    )


def export_training_set(
    traces: list[Trace],
    *,
    name: str = "distilled",
    system: str = "",
    min_feedback_score: float | None = None,
    only_ok: bool = True,
    require_grounding: bool = True,
    min_support: float = 0.5,
    dedupe: bool = True,
    max_examples: int | None = None,
    max_example_chars: int | None = None,
    evidence_for: EvidenceResolver | None = None,
) -> TrainingSet:
    """Curate captured traces into a grounded fine-tuning :class:`TrainingSet`.

    A trace is admitted only when it (1) succeeded (``only_ok``), (2) clears
    ``min_feedback_score`` if set, (3) carries both an input and an output, and
    (4) when ``require_grounding`` is on, has at least one output claim the
    cited evidence supports at ``min_support`` (via the same deterministic
    extractor 0.8 uses for auto-memory). Duplicates (same input + output) are
    collapsed. Every surviving example keeps trace/run/session provenance and
    its measured support — so a reviewer can audit exactly what the student
    will learn from.

    Note: trace spans truncate the recorded output, so this path is faithful
    only when ``training_capture`` recorded the full artifacts (``output_full`` /
    ``evidence``). For a flag-free, always-faithful export, build from
    :class:`~vincio.core.types.RunResult` objects with
    :func:`export_training_set_from_runs` instead.
    """
    resolver = evidence_for or _evidence_from_trace
    examples: list[TrainingExample] = []
    seen: set[str] = set()
    considered = 0
    dropped_ungrounded = 0
    dropped_truncated = 0
    for trace in {t.id: t for t in traces}.values():
        if only_ok and trace.status != "ok":
            continue
        # Prefer the untruncated artifacts recorded when training_capture is on;
        # fall back to the (truncated) span attributes otherwise.
        input_text = trace.attributes.get("input_full") or trace.attributes.get("input")
        output_text = trace.attributes.get("output_full") or trace.attributes.get("output")
        if not input_text or not output_text:
            continue
        if min_feedback_score is not None:
            scores = [f.score for f in trace.feedback if f.score is not None]
            if not scores or sum(scores) / len(scores) < min_feedback_score:
                continue
        considered += 1
        if dedupe:
            key = hashlib.sha256(f"{input_text}\x00{output_text}".encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)

        example = _grounded_example(
            input_text,
            output_text,
            resolver(trace),
            system=str(trace.attributes.get("system") or system),
            require_grounding=require_grounding,
            min_support=min_support,
            provenance={
                "trace_id": trace.id,
                "run_id": trace.run_id,
                "session_id": trace.session_id,
            },
        )
        if example is None:
            dropped_ungrounded += 1
            continue
        if max_example_chars is not None and _example_chars(example) > max_example_chars:
            dropped_truncated += 1
            continue
        examples.append(example)
        if max_examples is not None and len(examples) >= max_examples:
            break

    return TrainingSet(
        name=name,
        examples=examples,
        metadata={
            "source": "traces",
            "traces": len(traces),
            "considered": considered,
            "dropped_ungrounded": dropped_ungrounded,
            "dropped_truncated": dropped_truncated,
            "require_grounding": require_grounding,
            "min_support": min_support,
        },
    )


def export_training_set_from_runs(
    runs: list[Any],
    *,
    name: str = "distilled",
    system: str = "",
    only_ok: bool = True,
    require_grounding: bool = True,
    min_support: float = 0.5,
    dedupe: bool = True,
    max_examples: int | None = None,
    max_example_chars: int | None = None,
) -> TrainingSet:
    """Curate :class:`~vincio.core.types.RunResult` objects into a grounded
    fine-tuning :class:`TrainingSet` — faithful by construction, no opt-in.

    Unlike the trace path, a ``RunResult`` already carries the **full**
    untruncated output (``raw_text``) and the **full** cited evidence
    (``evidence`` / ``citations``), and the runtime stamps the original input on
    ``metadata['input']`` — so grounding-checked, faithful training data needs no
    ``training_capture`` flag. The natural usage is to keep the results you run::

        results = [app.run(q) for q in prompts]
        ts = app.export_training_set(runs=results)

    A run is admitted only when it succeeded (``only_ok``), carries an input and
    an output, and — with ``require_grounding`` on — has at least one output
    claim its evidence supports at ``min_support``. Feedback filtering is not
    applied here (feedback attaches to traces, not results); use the trace path
    when you need it.
    """
    examples: list[TrainingExample] = []
    seen: set[str] = set()
    considered = 0
    dropped_ungrounded = 0
    dropped_truncated = 0
    for run in runs:
        status = getattr(run, "status", None)
        if only_ok and str(getattr(status, "value", status)) != "succeeded":
            continue
        metadata = getattr(run, "metadata", {}) or {}
        input_text = metadata.get("input")
        output_text = getattr(run, "raw_text", "") or (
            getattr(run, "output", "") if isinstance(getattr(run, "output", None), str) else ""
        )
        if not input_text or not output_text:
            continue
        considered += 1
        if dedupe:
            key = hashlib.sha256(f"{input_text}\x00{output_text}".encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)

        evidence = [e for e in (getattr(run, "evidence", None) or []) if isinstance(e, EvidenceItem)]
        example = _grounded_example(
            input_text,
            output_text,
            evidence,
            system=system,
            require_grounding=require_grounding,
            min_support=min_support,
            provenance={
                "run_id": getattr(run, "run_id", None),
                "trace_id": getattr(run, "trace_id", None),
            },
        )
        if example is None:
            dropped_ungrounded += 1
            continue
        if max_example_chars is not None and _example_chars(example) > max_example_chars:
            dropped_truncated += 1
            continue
        examples.append(example)
        if max_examples is not None and len(examples) >= max_examples:
            break

    return TrainingSet(
        name=name,
        examples=examples,
        metadata={
            "source": "runs",
            "runs": len(runs),
            "considered": considered,
            "dropped_ungrounded": dropped_ungrounded,
            "dropped_truncated": dropped_truncated,
            "require_grounding": require_grounding,
            "min_support": min_support,
        },
    )


def _example_chars(example: TrainingExample) -> int:
    return sum(len(str(m.get("content", ""))) for m in example.messages)


def _example_text(example: TrainingExample) -> str:
    return "\n".join(str(m.get("content", "")) for m in example.messages)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def semantic_dedupe(
    training_set: TrainingSet,
    *,
    embedder: Embedder | None = None,
    threshold: float = 0.92,
) -> TrainingSet:
    """Drop near-duplicate examples by embedding similarity (2.1).

    Exact (input+output) duplicates are already collapsed at export time; this
    catches *paraphrases* — runs that say the same thing differently — so the
    fine-tuning corpus stays diverse and the student does not overfit a handful
    of high-traffic prompts. Greedy: keep an example unless its embedding is at
    least ``threshold`` cosine-similar to one already kept. Uses the offline
    hash embedder by default (no network, deterministic); pass any
    :class:`~vincio.retrieval.embeddings.Embedder` for semantic quality.
    """
    if len(training_set.examples) <= 1:
        return training_set
    if embedder is None:
        from ..retrieval.embeddings import LocalHashEmbedder

        embedder = LocalHashEmbedder()
    from ..retrieval.embeddings import embed_texts

    vectors = await embed_texts(embedder, [_example_text(e) for e in training_set.examples])
    kept: list[TrainingExample] = []
    kept_vectors: list[list[float]] = []
    for example, vector in zip(training_set.examples, vectors, strict=True):
        if any(_cosine(vector, other) >= threshold for other in kept_vectors):
            continue
        kept.append(example)
        kept_vectors.append(vector)
    metadata = dict(training_set.metadata)
    metadata["semantic_dropped"] = len(training_set.examples) - len(kept)
    metadata["dedupe_threshold"] = threshold
    return TrainingSet(name=training_set.name, examples=kept, metadata=metadata)


# trainer(training_set, base_model) -> the student model name to evaluate.
# In production this submits a fine-tune job and returns the resulting model id;
# offline it returns the base model unchanged (a faithful no-op). 2.1 ships
# :func:`provider_trainer` as the executed implementation.
StudentTrainer = Callable[[TrainingSet, str], Awaitable[str]]
# evaluate_model(model, dataset) -> EvalReport on a held-out set.
ModelEvaluateFn = Callable[[str, Dataset], Awaitable[EvalReport]]


def _register_student(
    registry: ModelRegistry,
    model_id: str,
    *,
    base_model: str,
    inherit_from: str | None,
    pricing: dict[str, float] | None,
    provider: str,
) -> None:
    """Register the trained student so the routing cascade can serve it.

    The student inherits its capabilities and (by default) its per-Mtok pricing
    from the base model it was fine-tuned from — the cheaper model — so its cost
    advantage over the teacher is measurable at the swap gate. ``pricing``
    overrides the inherited rates when the fine-tuned variant is priced
    differently.
    """
    from ..core.types import ModelCapabilities, ModelProfile

    base = registry.resolve(inherit_from or base_model)
    rates = pricing or {}
    profile = ModelProfile(
        name=model_id,
        provider=base.provider if base else provider,
        model=model_id,
        capabilities=base.capabilities if base else ModelCapabilities(),
        tier="fast",  # a distilled student is the cheap rung of the cascade
        input_cost_per_mtok=rates.get(
            "input", base.input_cost_per_mtok if base else 0.0
        ),
        output_cost_per_mtok=rates.get(
            "output", base.output_cost_per_mtok if base else 0.0
        ),
    )
    registry.register(profile)


def provider_trainer(
    backend: FineTuneBackend,
    *,
    registry: ModelRegistry | None = None,
    inherit_from: str | None = None,
    pricing: dict[str, float] | None = None,
    suffix: str | None = "vincio",
    fmt: TrainingFormat = "openai",
    poll_interval_s: float = 5.0,
    max_polls: int = 240,
) -> StudentTrainer:
    """Build an *executed* :data:`StudentTrainer` over a fine-tune backend (2.1).

    The returned async callable renders the grounded :class:`TrainingSet` to
    provider JSONL, submits and polls a real fine-tune job through ``backend``
    (:class:`~vincio.providers.finetune.FineTuneBackend`), registers the
    resulting model in ``registry`` (capabilities/pricing inherited from the
    base model, ``pricing`` overriding), and returns the trained model id — the
    piece that turns the flywheel from "grounded JSONL plus a gate" into an
    actual cheaper model. Plug it straight into :class:`BootstrapFinetune`::

        trainer = provider_trainer(OpenAIFineTuneBackend(provider), registry=reg)
        loop = BootstrapFinetune(evaluate_model=ev, trainer=trainer)

    Offline, point the backend at a cassette and set ``poll_interval_s=0`` for a
    deterministic, instantaneous job lifecycle.
    """
    from ..providers.finetune import run_finetune
    from ..providers.registry import default_model_registry

    reg = registry or default_model_registry()

    async def trainer(training_set: TrainingSet, base_model: str) -> str:
        jsonl = training_set.to_jsonl(format=fmt)
        job = await run_finetune(
            backend,
            jsonl,
            base_model,
            suffix=suffix,
            poll_interval_s=poll_interval_s,
            max_polls=max_polls,
        )
        model_id = job.fine_tuned_model or base_model
        _register_student(
            reg,
            model_id,
            base_model=base_model,
            inherit_from=inherit_from,
            pricing=pricing,
            provider=backend.name,
        )
        return model_id

    return trainer


class DistillationResult(BaseModel):
    """Outcome of one teacher → student distillation cycle."""

    teacher: str
    student: str
    trained_student: str = ""
    training_examples: int = 0
    teacher_quality: float = 0.0
    student_quality: float = 0.0
    quality_ratio: float = 0.0
    teacher_cost: float = 0.0
    student_cost: float = 0.0
    cost_savings: float = 0.0  # fraction cheaper than the teacher
    promoted: bool = False
    reason: str = ""
    cascade: ModelCascade | None = None
    # 2.1: when a swap gate backs the promotion, its significance verdict.
    swap_passed: bool | None = None
    swap_reason: str = ""


class BootstrapFinetune:
    """Teacher → student distillation with a gated quality hold.

    Given a grounded :class:`TrainingSet` and a held-out eval dataset, the loop
    (optionally) fine-tunes the student on the teacher's traces, evaluates both
    models on the suite, and promotes the student into a cheap→strong
    :class:`~vincio.optimize.routing.ModelCascade` **only** if it preserves at
    least ``min_quality_ratio`` of the teacher's quality, costs strictly less,
    and regresses neither safety nor schema validity. A failing student returns
    a ``promoted=False`` result with the reason — never a silent downgrade.
    """

    def __init__(
        self,
        evaluate_model: ModelEvaluateFn,
        *,
        quality_metric: str = "lexical_overlap",
        min_quality_ratio: float = 0.97,
        gates: dict[str, str] | None = None,
        trainer: StudentTrainer | None = None,
        swap_gate: Any = None,
        dedupe_embedder: Embedder | None = None,
    ) -> None:
        self.evaluate_model = evaluate_model
        self.quality_metric = quality_metric
        self.min_quality_ratio = min_quality_ratio
        self.gates = gates
        self.trainer = trainer
        # 2.1: an optional SwapGate adds a significance-backed final check, so the
        # student promotes only if it provably does not regress — the same gate a
        # model rotation clears.
        self.swap_gate = swap_gate
        self.dedupe_embedder = dedupe_embedder

    async def distill(
        self,
        training_set: TrainingSet,
        dataset: Dataset,
        *,
        teacher: str,
        student: str,
        min_dataset_coverage: int = 4,
        semantic_dedupe_threshold: float | None = None,
    ) -> DistillationResult:
        if semantic_dedupe_threshold is not None:
            training_set = await semantic_dedupe(
                training_set, embedder=self.dedupe_embedder, threshold=semantic_dedupe_threshold
            )
        result = DistillationResult(
            teacher=teacher, student=student, training_examples=len(training_set)
        )
        if len(dataset) < min_dataset_coverage:
            result.reason = (
                f"held-out set too small ({len(dataset)} cases < {min_dataset_coverage}); "
                "refusing to distill"
            )
            return result
        if not training_set.examples:
            result.reason = "no grounded training examples to distill from"
            return result

        trained_student = student
        if self.trainer is not None:
            trained_student = await self.trainer(training_set, student)
        result.trained_student = trained_student
        if trained_student == teacher:
            result.reason = "student and teacher are the same model; nothing to distill"
            return result

        teacher_report = await self.evaluate_model(teacher, dataset)
        student_report = await self.evaluate_model(trained_student, dataset)

        teacher_quality = _mean(teacher_report, self.quality_metric)
        student_quality = _mean(student_report, self.quality_metric)
        teacher_cost = _mean(teacher_report, "cost")
        student_cost = _mean(student_report, "cost")
        ratio = (student_quality / teacher_quality) if teacher_quality > 0 else 0.0

        result.teacher_quality = round(teacher_quality, 4)
        result.student_quality = round(student_quality, 4)
        result.quality_ratio = round(ratio, 4)
        result.teacher_cost = round(teacher_cost, 6)
        result.student_cost = round(student_cost, 6)
        result.cost_savings = round(1.0 - student_cost / teacher_cost, 4) if teacher_cost > 0 else 0.0

        if ratio < self.min_quality_ratio:
            result.reason = (
                f"student holds only {ratio:.1%} of teacher quality "
                f"(< {self.min_quality_ratio:.0%}); not promoting"
            )
            return result
        # The flywheel's whole point is a cheaper student, so a cost win must be
        # *verifiable*: refuse to promote when there is no cost signal (e.g.
        # unpriced models report 0) rather than claim a saving we cannot measure.
        if teacher_cost <= 0:
            result.reason = (
                "no cost signal on the teacher (price the models so the student's "
                "cost saving can be verified before promotion)"
            )
            return result
        if student_cost >= teacher_cost:
            result.reason = "student is not cheaper than the teacher; distillation has no payoff"
            return result
        safe, reason = _promotion_safe(
            student_report, teacher_report, gates=self.gates, max_cost_per_case=None
        )
        if not safe:
            result.reason = reason
            return result

        # 2.1: a significance-backed swap gate is the same final check a model
        # rotation clears — promote the trained student only if it provably does
        # not regress against the teacher on a replayed/scored diff.
        if self.swap_gate is not None:
            verdict = await self.swap_gate.evaluate(
                candidate_model=trained_student, baseline_model=teacher, dataset=dataset
            )
            result.swap_passed = bool(getattr(verdict, "passed", False))
            result.swap_reason = str(getattr(verdict, "reason", ""))
            if not result.swap_passed:
                result.reason = f"swap gate blocked promotion: {result.swap_reason}"
                return result

        # Promote: student is the cheap rung, teacher the strong fallback.
        result.cascade = ModelCascade.from_models([trained_student, teacher])
        result.promoted = True
        result.reason = (
            f"student {trained_student} holds {ratio:.1%} of teacher quality at "
            f"{result.cost_savings:.0%} lower cost; promoted to the cascade"
        )
        return result


def _mean(report: EvalReport, metric: str) -> float:
    values = report.metric_values(metric)
    return sum(values) / len(values) if values else 0.0
