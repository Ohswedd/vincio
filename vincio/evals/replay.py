"""Trace-replay executor.

``trace_replay_plan`` (observability) only *extracts* a replay; nothing executed
it. ``ReplayRunner`` is the missing executor: it re-runs captured trace inputs
through a target :class:`~vincio.core.app.ContextApp` and diffs

* **outputs** — recorded answer vs replayed answer (text similarity),
* **trajectory** — recorded vs replayed span tree, via ``trace_diff``,
* **cost** and **latency** — recorded vs replayed, via ``EvalReport.diff``,

so behavioral regression becomes a reproducible primitive instead of a stub.
Recorded tool outputs can be *pinned* so the replay is deterministic even when
tools are non-deterministic. Surfaced from the CLI as ``vincio trace replay
--against <app>``; it is the building block the ``SwapGate`` composes.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field

from ..core.types import ToolResult
from ..observability.exporters import TraceExporter
from ..observability.spans import Trace
from ..observability.traces import trace_diff, trace_replay_plan
from .reports import CaseResult, EvalReport

__all__ = ["ReplayCase", "ReplayResult", "ReplayRunner"]


class ReplayCase(BaseModel):
    trace_id: str
    input: str
    baseline_text: str
    candidate_text: str
    output_similarity: float  # 1.0 == identical answer
    output_match: bool
    baseline_cost_usd: float
    candidate_cost_usd: float
    cost_delta_usd: float
    baseline_latency_ms: int
    candidate_latency_ms: int
    status_match: bool
    trajectory: dict[str, Any] = Field(default_factory=dict)


class ReplayResult(BaseModel):
    cases: list[ReplayCase] = Field(default_factory=list)
    # The output/cost/latency diff, reusing EvalReport.diff (candidate vs baseline).
    report_diff: dict[str, Any] = Field(default_factory=dict)

    @property
    def output_match_rate(self) -> float:
        return (
            sum(1 for c in self.cases if c.output_match) / len(self.cases) if self.cases else 1.0
        )

    @property
    def mean_output_similarity(self) -> float:
        return (
            sum(c.output_similarity for c in self.cases) / len(self.cases) if self.cases else 1.0
        )

    @property
    def total_cost_delta_usd(self) -> float:
        return round(sum(c.cost_delta_usd for c in self.cases), 8)

    def summary(self) -> dict[str, Any]:
        return {
            "cases": len(self.cases),
            "output_match_rate": round(self.output_match_rate, 4),
            "mean_output_similarity": round(self.mean_output_similarity, 4),
            "total_cost_delta_usd": self.total_cost_delta_usd,
            "trajectory_changed": sum(
                1 for c in self.cases
                if c.trajectory.get("spans_only_in_a") or c.trajectory.get("spans_only_in_b")
                or c.trajectory.get("status_changes")
            ),
        }


class _CaptureExporter:
    """Wraps the target tracer's exporter to capture replayed traces by id
    (and still forward to the original exporter, so replay never loses data)."""

    def __init__(self, inner: TraceExporter) -> None:
        self._inner = inner
        self.captured: dict[str, Trace] = {}

    def export(self, trace: Trace) -> None:
        self.captured[trace.id] = trace
        self._inner.export(trace)


def _recorded_output(trace: Trace) -> str:
    model_calls = [s for s in trace.spans if s.type == "model_call"]
    if model_calls:
        text = model_calls[-1].attributes.get("response_text")
        if text:
            return str(text)
    return str(trace.attributes.get("output") or "")


def _recorded_cost(trace: Trace) -> float:
    # The cumulative cost is stamped on each model span; the last carries the total.
    model_calls = [s for s in trace.spans if s.type == "model_call"]
    for span in reversed(model_calls):
        cost = span.attributes.get("cost_usd")
        if cost is not None:
            return float(cost)
    return 0.0


class ReplayRunner:
    """Re-run captured trace inputs through a target app and diff the results."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def _pinned_tool_outputs(self, trace: Trace) -> dict[tuple[str, str], Any]:
        plan = trace_replay_plan(trace)
        pinned: dict[tuple[str, str], Any] = {}
        for call in plan["tool_calls"]:
            tool = call.get("tool")
            if tool is None:
                continue
            key = (str(tool), json.dumps(call.get("arguments"), sort_keys=True, default=str))
            pinned[key] = call.get("output")
        return pinned

    async def replay(
        self,
        traces: list[Trace],
        *,
        pin_tools: bool = False,
    ) -> ReplayResult:
        """Replay *traces* through the target app and return a structured diff.

        With ``pin_tools`` the target app's tool runtime returns the recorded
        outputs for matching calls, making the replay deterministic regardless of
        tool side effects (the swap-regression use-case).
        """
        app = self.app
        cases: list[ReplayCase] = []
        baseline_cases: list[CaseResult] = []
        candidate_cases: list[CaseResult] = []

        capture = _CaptureExporter(app.tracer.exporter)
        original_exporter = app.tracer.exporter
        original_execute = app.tool_runtime.execute
        app.tracer.exporter = capture
        try:
            for trace in traces:
                plan = trace_replay_plan(trace)
                user_input = str(plan.get("input") or "")
                pinned = self._pinned_tool_outputs(trace) if pin_tools else {}

                if pin_tools and pinned:
                    async def _pinned_execute(call: Any, *a: Any, _pinned: dict = pinned, **k: Any) -> ToolResult:
                        key = (str(call.tool_name), json.dumps(call.arguments, sort_keys=True, default=str))
                        if key in _pinned:
                            return ToolResult(
                                call_id=getattr(call, "call_id", "") or "",
                                tool_name=call.tool_name, status="ok", output=_pinned[key],
                            )
                        return await original_execute(call, *a, **k)

                    app.tool_runtime.execute = _pinned_execute
                else:
                    app.tool_runtime.execute = original_execute

                result = await app.arun(user_input)
                candidate_trace = capture.captured.get(result.trace_id)

                baseline_text = _recorded_output(trace)
                candidate_text = result.raw_text or ""
                similarity = (
                    1.0 if baseline_text == candidate_text
                    else SequenceMatcher(None, baseline_text, candidate_text).ratio()
                )
                baseline_cost = _recorded_cost(trace)
                trajectory = (
                    trace_diff(trace, candidate_trace) if candidate_trace is not None else {}
                )
                cases.append(
                    ReplayCase(
                        trace_id=trace.id,
                        input=user_input,
                        baseline_text=baseline_text,
                        candidate_text=candidate_text,
                        output_similarity=round(similarity, 4),
                        output_match=baseline_text == candidate_text,
                        baseline_cost_usd=round(baseline_cost, 8),
                        candidate_cost_usd=round(result.cost_usd, 8),
                        cost_delta_usd=round(result.cost_usd - baseline_cost, 8),
                        baseline_latency_ms=trace.duration_ms,
                        candidate_latency_ms=result.latency_ms,
                        status_match=(trace.status in ("ok", "running"))
                        == (result.status.value == "succeeded"),
                        trajectory=trajectory,
                    )
                )
                baseline_cases.append(
                    CaseResult(
                        case_id=trace.id,
                        metrics={"output_match": 1.0, "cost": baseline_cost,
                                 "latency": float(trace.duration_ms)},
                    )
                )
                candidate_cases.append(
                    CaseResult(
                        case_id=trace.id,
                        metrics={"output_match": similarity, "cost": result.cost_usd,
                                 "latency": float(result.latency_ms)},
                    )
                )
        finally:
            app.tracer.exporter = original_exporter
            app.tool_runtime.execute = original_execute

        baseline_report = EvalReport(name="replay_baseline", cases=baseline_cases)
        candidate_report = EvalReport(name="replay_candidate", cases=candidate_cases)
        report_diff = candidate_report.diff(baseline_report) if cases else {}
        return ReplayResult(cases=cases, report_diff=report_diff)
