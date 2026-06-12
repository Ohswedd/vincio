"""Eval reports: summary, distributions, failures, baseline diff,
regression gates."""

from __future__ import annotations

import json
import operator
import re
import statistics
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import GateFailedError
from ..core.utils import utcnow

__all__ = ["CaseResult", "GateSpec", "EvalReport", "evaluate_gates"]


class CaseResult(BaseModel):
    case_id: str
    metrics: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    output_text: str = ""
    error: str | None = None
    trace_id: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0
    tags: list[str] = Field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.error is not None


_GATE_RE = re.compile(r"^\s*(==|>=|<=|>|<|!=)\s*([\d.]+)\s*$")
_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


class GateSpec(BaseModel):
    metric: str
    expression: str  # e.g. ">= 0.91"
    aggregate: str = "mean"  # mean | min | max | p95

    def check(self, values: list[float]) -> tuple[bool, float]:
        if not values:
            return False, float("nan")
        if self.aggregate == "mean":
            aggregated = sum(values) / len(values)
        elif self.aggregate == "min":
            aggregated = min(values)
        elif self.aggregate == "max":
            aggregated = max(values)
        elif self.aggregate.startswith("p"):
            percentile = float(self.aggregate[1:])
            ordered = sorted(values)
            index = min(len(ordered) - 1, max(0, round(percentile / 100 * (len(ordered) - 1))))
            aggregated = ordered[index]
        else:
            aggregated = sum(values) / len(values)
        match = _GATE_RE.match(self.expression)
        if not match:
            raise ValueError(f"invalid gate expression {self.expression!r}")
        op, threshold = _OPS[match.group(1)], float(match.group(2))
        return op(aggregated, threshold), aggregated


class EvalReport(BaseModel):
    name: str = "eval"
    dataset: str = ""
    created_at: Any = Field(default_factory=utcnow)
    cases: list[CaseResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    gates: dict[str, Any] = Field(default_factory=dict)  # gate -> {passed, value}

    # -- aggregation -----------------------------------------------------------------

    def metric_values(self, metric: str) -> list[float]:
        return [c.metrics[metric] for c in self.cases if metric in c.metrics]

    def summary(self) -> dict[str, dict[str, float]]:
        names: set[str] = set()
        for case in self.cases:
            names.update(case.metrics)
        table: dict[str, dict[str, float]] = {}
        for name in sorted(names):
            values = self.metric_values(name)
            if not values:
                continue
            table[name] = {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "p50": round(statistics.median(values), 4),
                "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        return table

    def distribution(self, metric: str, *, bins: int = 10) -> dict[str, int]:
        values = self.metric_values(metric)
        if not values:
            return {}
        low, high = min(values), max(values)
        if high == low:
            return {f"{low:.2f}": len(values)}
        width = (high - low) / bins
        histogram: dict[str, int] = {}
        for value in values:
            bucket = min(bins - 1, int((value - low) / width))
            label = f"{low + bucket * width:.2f}–{low + (bucket + 1) * width:.2f}"
            histogram[label] = histogram.get(label, 0) + 1
        return histogram

    def failures(self, *, metric: str | None = None, threshold: float = 0.5) -> list[CaseResult]:
        if metric is None:
            return [c for c in self.cases if c.failed]
        return [c for c in self.cases if c.metrics.get(metric, 1.0) < threshold or c.failed]

    @property
    def total_cost_usd(self) -> float:
        return round(sum(c.cost_usd for c in self.cases), 6)

    # -- baseline diff ------------------------------------------------------------------

    def diff(self, baseline: EvalReport) -> dict[str, Any]:
        current = self.summary()
        base = baseline.summary()
        changes: dict[str, Any] = {}
        for metric in sorted(set(current) | set(base)):
            mean_now = current.get(metric, {}).get("mean")
            mean_base = base.get(metric, {}).get("mean")
            if mean_now is None or mean_base is None:
                changes[metric] = {"current": mean_now, "baseline": mean_base, "delta": None}
            else:
                changes[metric] = {
                    "current": mean_now,
                    "baseline": mean_base,
                    "delta": round(mean_now - mean_base, 4),
                }
        regressed_cases = []
        base_by_id = {c.case_id: c for c in baseline.cases}
        for case in self.cases:
            base_case = base_by_id.get(case.case_id)
            if base_case is None:
                continue
            for metric, value in case.metrics.items():
                base_value = base_case.metrics.get(metric)
                if base_value is not None and value < base_value - 1e-9 and metric not in ("cost", "latency", "input_tokens", "output_tokens", "retries", "unsupported_claim_rate"):
                    regressed_cases.append({"case_id": case.case_id, "metric": metric, "from": base_value, "to": value})
        return {"metrics": changes, "regressed_cases": regressed_cases}

    # -- rendering ----------------------------------------------------------------------

    def to_markdown(self) -> str:
        lines = [f"# Eval report: {self.name}", "", f"Dataset: `{self.dataset}` · cases: {len(self.cases)} · total cost: ${self.total_cost_usd}", ""]
        lines.append("| metric | mean | min | max | p50 | n |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for metric, stats in self.summary().items():
            lines.append(
                f"| {metric} | {stats['mean']} | {stats['min']} | {stats['max']} | {stats['p50']} | {stats['n']} |"
            )
        if self.gates:
            lines.extend(["", "## Gates", "", "| gate | value | passed |", "|---|---:|---|"])
            for gate, info in self.gates.items():
                lines.append(f"| {gate} | {info.get('value')} | {'✅' if info.get('passed') else '❌'} |")
        errors = [c for c in self.cases if c.failed]
        if errors:
            lines.extend(["", "## Errored cases", ""])
            for case in errors[:20]:
                lines.append(f"- `{case.case_id}`: {case.error}")
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.to_markdown())

    # -- persistence ---------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def evaluate_gates(
    report: EvalReport,
    gates: dict[str, str],
    *,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Check regression gates like {"accuracy": ">= 0.91", "p95_latency_ms": "<= 8000"}.

    A gate key may be ``metric`` or ``p95_metric`` / ``min_metric`` etc.
    """
    outcomes: dict[str, Any] = {}
    failures: list[str] = []
    for key, expression in gates.items():
        aggregate = "mean"
        metric = key
        for prefix in ("p95_", "p99_", "min_", "max_"):
            if key.startswith(prefix):
                aggregate = prefix.rstrip("_")
                metric = key[len(prefix):]
                break
        spec = GateSpec(metric=metric, expression=expression, aggregate=aggregate)
        passed, value = spec.check(report.metric_values(metric))
        outcomes[key] = {"passed": passed, "value": round(value, 6) if value == value else None, "expression": expression}
        if not passed:
            failures.append(f"{key} {expression} (got {value})")
    report.gates = outcomes
    if failures and raise_on_failure:
        raise GateFailedError("eval gates failed: " + "; ".join(failures), failures=failures)
    return outcomes
