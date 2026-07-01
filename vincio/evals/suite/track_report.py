"""Rendering for the uplift and feature tracks — one run to Markdown, every number
carrying its provenance tier, the way :class:`~vincio.evals.suite.report.SuiteReport`
renders the model track.
"""

from __future__ import annotations

from .feature_bench import FeatureSuiteRun
from .uplift import UpliftRun

__all__ = ["render_uplift_report", "render_feature_report"]

_TIER_LEGEND = (
    "Tiers — **L** Live (the real thing ran end to end), **R** Recorded (a pinned "
    "replay), **S/Mockup** (offline, reproducible, gates CI). A lower tier can never "
    "print a higher tier's label."
)


def render_uplift_report(run: UpliftRun) -> str:
    """Render an :class:`UpliftRun` to Markdown: per-benchmark direct → Vincio deltas."""
    lines = [
        f"# Uplift report — {run.model}",
        "",
        f"Track **Uplift** · tier **{run.tier.code} ({run.tier.label})** · "
        f"{len(run.results)} benchmarks · overall **{run.overall_direct():.3f} → "
        f"{run.overall_vincio():.3f}** ({run.overall_delta():+.3f}) · "
        f"{'reproducible, gates CI' if run.gated else 'reported, not gated'}.",
        "",
        _TIER_LEGEND,
        "",
        "| Benchmark | Metric | Tier | Direct | Through Vincio | Δ |",
        "|---|---|:--:|---:|---:|---:|",
    ]
    for r in sorted(run.results, key=lambda r: r.benchmark_id):
        arrow = "▲" if r.improved else ("▼" if r.regressed else "=")
        lines.append(
            f"| `{r.benchmark_id}` | {r.primary_metric} | {r.tier.code} | "
            f"{r.direct:.3f} | **{r.vincio:.3f}** | {arrow} {r.delta:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def render_feature_report(run: FeatureSuiteRun) -> str:
    """Render a :class:`FeatureSuiteRun` to Markdown: per-contest contender measurements."""
    lines = [
        "# Feature report — Vincio vs the alternatives",
        "",
        f"Track **Feature** · suite tier **{run.tier.code} ({run.tier.label})** · "
        f"{len(run.runs)} contests · "
        f"{'reproducible, gates CI' if run.gated else 'reported, not gated'}.",
        "",
        _TIER_LEGEND,
        "",
    ]
    for r in sorted(run.runs, key=lambda x: x.contest_id):
        better = "higher is better" if r.higher_is_better else "lower is better"
        unit = f" {r.unit}" if r.unit else ""
        lines.append(f"## `{r.contest_id}` — {r.title}  ·  tier {r.tier.code}")
        lines.append("")
        lines.append(f"Primary: **{r.primary_metric}** ({better}). Winner: **{r.winner or 'n/a'}**.")
        lines.append("")
        lines.append(f"| Contender | Kind | {r.primary_metric} | Latency | Notes |")
        lines.append("|---|---|---:|---:|---|")
        for m in r.measurements:
            if not m.available:
                lines.append(f"| {m.contender} | {m.kind} | — | — | {m.note or 'skipped'} |")
                continue
            latency = f"{m.latency_ms:.2f} ms" if m.latency_ms is not None else "—"
            mark = " ⭐" if m.contender == r.winner else ""
            lines.append(
                f"| {m.contender}{mark} | {m.kind} | {m.primary:g}{unit} | {latency} | "
                f"{_extra_metrics(m.metrics, r.primary_metric)} |"
            )
        if r.verdict:
            lines.append("")
            lines.append(f"> {r.verdict}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _extra_metrics(metrics: dict[str, float], primary: str) -> str:
    extras = {k: v for k, v in sorted(metrics.items()) if k != primary}
    return ", ".join(f"{k}={v:g}" for k, v in extras.items())
