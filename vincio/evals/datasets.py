"""Eval datasets: JSONL golden datasets with tags and rubrics.

``dataset_from_traces`` turns captured traces into a dataset in one call —
the bridge from observability to evaluation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import DatasetError

if TYPE_CHECKING:
    from ..observability.spans import Trace
    from .reports import EvalReport

__all__ = [
    "EvalCase",
    "Dataset",
    "dataset_from_traces",
    "GoldenGateResult",
    "GoldenRegressionSuite",
]


class EvalCase(BaseModel):
    id: str
    input: str | dict[str, Any]
    context: dict[str, Any] = Field(default_factory=dict)
    expected: Any = None
    rubric: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    difficulty: str = "medium"  # easy | medium | hard
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def input_text(self) -> str:
        if isinstance(self.input, str):
            return self.input
        return str(self.input.get("text") or self.input.get("input") or json.dumps(self.input))


class Dataset(BaseModel):
    name: str = "dataset"
    cases: list[EvalCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None) -> Dataset:
        path = Path(path)
        if not path.is_file():
            raise DatasetError(f"dataset file not found: {path}")
        cases: list[EvalCase] = []
        with path.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                record.setdefault("id", f"case_{line_number:04d}")
                try:
                    cases.append(EvalCase.model_validate(record))
                except ValueError as exc:
                    raise DatasetError(f"{path}:{line_number}: invalid case: {exc}") from exc
        return cls(name=name or path.stem, cases=cases)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for case in self.cases:
                fh.write(json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def filter(
        self,
        *,
        tags: list[str] | None = None,
        difficulty: str | None = None,
        ids: list[str] | None = None,
    ) -> Dataset:
        cases = [
            case
            for case in self.cases
            if (tags is None or any(tag in case.tags for tag in tags))
            and (difficulty is None or case.difficulty == difficulty)
            and (ids is None or case.id in ids)
        ]
        return Dataset(name=self.name, cases=cases, metadata=self.metadata)

    def sample(self, n: int, *, seed: int = 42) -> Dataset:
        if n >= len(self.cases):
            return self
        rng = random.Random(seed)
        return Dataset(name=f"{self.name}_sample{n}", cases=rng.sample(self.cases, n))

    def split(self, fraction: float = 0.8, *, seed: int = 42) -> tuple[Dataset, Dataset]:
        rng = random.Random(seed)
        shuffled = list(self.cases)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * fraction)
        return (
            Dataset(name=f"{self.name}_train", cases=shuffled[:cut]),
            Dataset(name=f"{self.name}_held", cases=shuffled[cut:]),
        )


class GoldenGateResult(BaseModel):
    """Verdict of replaying a candidate against the growing golden suite."""

    passed: bool = True
    checked: int = 0
    regressed: list[str] = Field(default_factory=list)  # case ids that fell below their floor
    missing: list[str] = Field(default_factory=list)  # cases the candidate did not run
    details: dict[str, Any] = Field(default_factory=dict)


class GoldenRegressionSuite:
    """A held-out, *growing* golden regression set with per-case provenance.

    Every time a promotion fixes a previously-failing case, that case is recorded
    here with the metric and floor it must keep clearing and *which* fix added it
    (``fixed_by``). Before any later promotion lands, the candidate is replayed
    against the whole suite and gated: a sequential auto-promotion can never
    silently undo a prior fix, because regressing any recorded case fails the
    gate. Backed by a JSONL file (no hosted service); the provenance rides each
    case's ``metadata`` so the suite is itself a reproducible artifact.
    """

    def __init__(self, path: str | Path = ".vincio/golden_regression.jsonl",
                 *, name: str = "golden_regression") -> None:
        self.path = Path(path)
        self.name = name
        self.dataset = Dataset.load(self.path, name=name) if self.path.is_file() else Dataset(name=name)

    def __len__(self) -> int:
        return len(self.dataset)

    def case_ids(self) -> list[str]:
        return [c.id for c in self.dataset.cases]

    def add(
        self,
        case: EvalCase,
        *,
        fixed_by: str,
        guard_metric: str = "lexical_overlap",
        guard_threshold: float = 0.5,
    ) -> EvalCase:
        """Record a case the latest fix must keep passing (idempotent on id)."""
        from ..core.utils import utcnow

        recorded = case.model_copy(deep=True)
        recorded.metadata = {
            **recorded.metadata,
            "fixed_by": fixed_by,
            "guard_metric": guard_metric,
            "guard_threshold": guard_threshold,
            "added_at": utcnow().isoformat(),
        }
        if "regression_guard" not in recorded.tags:
            recorded.tags = [*recorded.tags, "regression_guard"]
        existing = {c.id: i for i, c in enumerate(self.dataset.cases)}
        if recorded.id in existing:
            self.dataset.cases[existing[recorded.id]] = recorded
        else:
            self.dataset.cases.append(recorded)
        self.dataset.save(self.path)
        return recorded

    def add_from_report(
        self,
        report: EvalReport,
        dataset: Dataset,
        *,
        fixed_by: str,
        guard_metric: str = "lexical_overlap",
        guard_threshold: float = 0.5,
    ) -> list[str]:
        """Record every case the candidate now passes above the floor, drawn from
        the dataset it was evaluated on (so input + expected ride along)."""
        by_id = {c.id: c for c in dataset.cases}
        added: list[str] = []
        for case in report.cases:
            if case.metrics.get(guard_metric, 0.0) >= guard_threshold and case.case_id in by_id:
                self.add(by_id[case.case_id], fixed_by=fixed_by,
                         guard_metric=guard_metric, guard_threshold=guard_threshold)
                added.append(case.case_id)
        return added

    def as_dataset(self) -> Dataset:
        return Dataset(name=self.name, cases=list(self.dataset.cases), metadata=dict(self.dataset.metadata))

    def gate(self, report: EvalReport) -> GoldenGateResult:
        """Check a candidate's report against every recorded case's floor.

        ``report`` must be the candidate evaluated on :meth:`as_dataset`. A case
        regresses when its guard metric falls below the floor it was recorded
        with; an absent case counts as missing (treated as a regression, since
        the guard cannot be verified)."""
        from .metrics import LOWER_IS_BETTER

        by_id = {c.case_id: c for c in report.cases}
        result = GoldenGateResult(checked=len(self.dataset.cases))
        for case in self.dataset.cases:
            metric = case.metadata.get("guard_metric", "lexical_overlap")
            threshold = float(case.metadata.get("guard_threshold", 0.5))
            scored = by_id.get(case.id)
            if scored is None or metric not in scored.metrics:
                result.missing.append(case.id)
                continue
            value = scored.metrics[metric]
            ok = value <= threshold if metric in LOWER_IS_BETTER else value >= threshold
            if not ok:
                result.regressed.append(case.id)
        result.passed = not result.regressed and not result.missing
        result.details = {"regressed": len(result.regressed), "missing": len(result.missing)}
        return result


def dataset_from_traces(
    traces: list[Trace],
    *,
    name: str = "traces",
    include_outputs: bool = True,
    min_feedback_score: float | None = None,
    only_ok: bool = True,
    group_by_session: bool = False,
) -> Dataset:
    """Curate captured traces into an eval dataset (one command, full provenance).

    Each trace with a recorded input becomes a case; the trace's recorded
    output becomes the reference answer when ``include_outputs`` is set.
    ``min_feedback_score`` keeps only traces whose mean feedback score reaches
    the bar — the usual way to bootstrap a golden set from production runs
    users approved of. Trace/run/session ids ride along in case metadata.

    With ``group_by_session=True`` the traces of one session/thread are stitched
    into a single **multi-turn** golden case (``context['messages']`` holds the
    whole thread), so the conversational and outcome metrics score the session
    end to end rather than turn by turn.
    """
    if group_by_session:
        return _multi_turn_dataset_from_traces(
            traces, name=name, min_feedback_score=min_feedback_score, only_ok=only_ok
        )
    cases: list[EvalCase] = []
    latest = {trace.id: trace for trace in traces}
    for trace in latest.values():
        if only_ok and trace.status != "ok":
            continue
        input_text = trace.attributes.get("input")
        if not input_text:
            continue
        if min_feedback_score is not None:
            feedback_scores = [f.score for f in trace.feedback if f.score is not None]
            if not feedback_scores or sum(feedback_scores) / len(feedback_scores) < min_feedback_score:
                continue
        expected = trace.attributes.get("output") if include_outputs else None
        cases.append(
            EvalCase(
                id=trace.run_id or trace.id,
                input=str(input_text),
                expected=expected or None,
                tags=["from_trace"],
                metadata={
                    "trace_id": trace.id,
                    "run_id": trace.run_id,
                    "session_id": trace.session_id,
                    "scores": dict(trace.scores),
                },
            )
        )
    return Dataset(name=name, cases=cases, metadata={"source": "traces", "traces": len(traces)})


def _multi_turn_dataset_from_traces(
    traces: list[Trace],
    *,
    name: str,
    min_feedback_score: float | None,
    only_ok: bool,
) -> Dataset:
    """Stitch traces sharing a session/thread into multi-turn golden cases."""
    by_session: dict[str, list[Trace]] = {}
    for trace in {t.id: t for t in traces}.values():
        if only_ok and trace.status != "ok":
            continue
        if not trace.attributes.get("input"):
            continue
        key = trace.session_id or trace.thread_id or trace.id
        by_session.setdefault(key, []).append(trace)

    cases: list[EvalCase] = []
    for session_id, session_traces in by_session.items():
        ordered = sorted(session_traces, key=lambda t: t.start_time)
        if min_feedback_score is not None:
            scores = [f.score for t in ordered for f in t.feedback if f.score is not None]
            if not scores or sum(scores) / len(scores) < min_feedback_score:
                continue
        messages: list[dict[str, Any]] = []
        last_output: Any = None
        for trace in ordered:
            user_text = str(trace.attributes.get("input"))
            messages.append({"role": "user", "content": user_text})
            output = trace.attributes.get("output")
            if output is not None:
                messages.append({"role": "assistant", "content": str(output)})
                last_output = output
        cases.append(
            EvalCase(
                id=session_id,
                input=messages[0]["content"] if messages else "",
                context={"messages": messages},
                expected=last_output or None,
                tags=["from_trace", "multi_turn"],
                metadata={"session_id": session_id, "turns": len(ordered)},
            )
        )
    return Dataset(
        name=name,
        cases=cases,
        metadata={"source": "traces", "traces": len(traces), "sessions": len(cases), "multi_turn": True},
    )
