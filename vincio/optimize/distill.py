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

__all__ = [
    "TrainingExample",
    "TrainingSet",
    "export_training_set",
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
    """
    resolver = evidence_for or _evidence_from_trace
    examples: list[TrainingExample] = []
    seen: set[str] = set()
    considered = 0
    dropped_ungrounded = 0
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

        evidence = resolver(trace)
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
            dropped_ungrounded += 1
            continue

        messages: list[dict[str, str]] = []
        sys_prompt = str(trace.attributes.get("system") or system)
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": str(input_text)})
        messages.append({"role": "assistant", "content": str(output_text)})
        examples.append(
            TrainingExample(
                messages=messages,
                support=round(support, 4),
                grounded=grounded,
                evidence_ids=evidence_ids,
                provenance={
                    "trace_id": trace.id,
                    "run_id": trace.run_id,
                    "session_id": trace.session_id,
                },
            )
        )
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
            "require_grounding": require_grounding,
            "min_support": min_support,
        },
    )


# trainer(training_set, base_model) -> the student model name to evaluate.
# In production this submits a fine-tune job and returns the resulting model id;
# offline it returns the base model unchanged (a faithful no-op).
StudentTrainer = Callable[[TrainingSet, str], Awaitable[str]]
# evaluate_model(model, dataset) -> EvalReport on a held-out set.
ModelEvaluateFn = Callable[[str, Dataset], Awaitable[EvalReport]]


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
        quality_metric: str = "semantic_similarity",
        min_quality_ratio: float = 0.97,
        gates: dict[str, str] | None = None,
        trainer: StudentTrainer | None = None,
    ) -> None:
        self.evaluate_model = evaluate_model
        self.quality_metric = quality_metric
        self.min_quality_ratio = min_quality_ratio
        self.gates = gates
        self.trainer = trainer

    async def distill(
        self,
        training_set: TrainingSet,
        dataset: Dataset,
        *,
        teacher: str,
        student: str,
        min_dataset_coverage: int = 4,
    ) -> DistillationResult:
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
