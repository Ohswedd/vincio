"""Visualization — leaderboard, radar, heatmap, confusion-matrix, and trend charts.

Each builder returns a :class:`SuiteChart` whose default rendering is a
deterministic **Vega-Lite v5** spec (sorted-key JSON, data embedded inline, no
network) — the same offline-first default as the charts subsystem. A rasterized
**PNG** is available behind ``vincio[eval-viz]`` (matplotlib). The charts read
straight off a :class:`~vincio.evals.suite.results.SuiteRun` or a
:class:`~vincio.evals.suite.report.Leaderboard`, so a run visualizes with no extra
plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ...core.errors import EvalSuiteError
from .registry import NICHES
from .report import Leaderboard
from .results import SuiteRun

__all__ = [
    "SuiteChart",
    "leaderboard_chart",
    "radar_chart",
    "heatmap_chart",
    "confusion_matrix_chart",
    "trend_chart",
]


class SuiteChart(BaseModel):
    """A chart artifact — a deterministic Vega-Lite spec with embedded data.

    ``kind`` names the chart; ``spec`` is the Vega-Lite v5 dict. :meth:`to_json`
    serializes it deterministically; :meth:`to_png` rasterizes it (requires
    ``vincio[eval-viz]``).
    """

    kind: str
    title: str = ""
    spec: dict[str, Any] = Field(default_factory=dict)
    rows: list[dict[str, Any]] = Field(default_factory=list)

    def to_vega_lite(self) -> dict[str, Any]:
        return self.spec

    def to_json(self) -> str:
        """Deterministic Vega-Lite JSON (sorted keys, stable separators)."""
        return json.dumps(self.spec, sort_keys=True, separators=(",", ":"), default=str)

    def to_png(self, *, width: int = 720, height: int = 440, dpi: int = 100) -> bytes:
        """Rasterize to PNG (requires ``vincio[eval-viz]`` / matplotlib)."""
        return _render_png(self, width=width, height=height, dpi=dpi)

    def save(self, path: str | Path) -> str:
        """Write the chart: ``.png`` rasterizes (needs the extra), else Vega-Lite JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".png":
            target.write_bytes(self.to_png())
        else:
            target.write_text(self.to_json(), encoding="utf-8")
        return str(target)


_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"


def _bar_spec(values: list[dict[str, Any]], *, x: str, y: str, title: str,
              x_type: str = "nominal", color: str | None = None) -> dict[str, Any]:
    encoding: dict[str, Any] = {
        "x": {"field": x, "type": x_type, "sort": None},
        "y": {"field": y, "type": "quantitative"},
    }
    if color:
        encoding["color"] = {"field": color, "type": "nominal"}
        encoding["xOffset"] = {"field": color, "type": "nominal"}
    return {"$schema": _SCHEMA, "title": title, "data": {"values": values},
            "mark": {"type": "bar", "tooltip": True}, "encoding": encoding}


def leaderboard_chart(leaderboard: Leaderboard, *, title: str = "Leaderboard") -> SuiteChart:
    """A ranked bar of each model's overall primary score."""
    values = [{"model": r.model, "overall": round(r.overall, 4), "tier": r.tier.code}
              for r in leaderboard.rows]
    spec = _bar_spec(values, x="model", y="overall", title=title, color="tier")
    return SuiteChart(kind="leaderboard", title=title, spec=spec, rows=values)


def radar_chart(run: SuiteRun, *, title: str = "Niche breakdown") -> SuiteChart:
    """A per-niche breakdown of a run's mean primary score.

    The dependency-free Vega-Lite rendering is a faithful bar across the niche
    axes (Vega-Lite has no native radial mark); :meth:`SuiteChart.to_png` draws a
    true radar with matplotlib when ``vincio[eval-viz]`` is installed.
    """
    scores = run.niche_scores()
    values = [{"niche": NICHES.get(n, n), "score": round(scores[n], 4)} for n in NICHES if n in scores]
    spec = _bar_spec(values, x="niche", y="score", title=title)
    return SuiteChart(kind="radar", title=title, spec=spec, rows=values)


def heatmap_chart(leaderboard: Leaderboard, *, title: str = "Model × benchmark") -> SuiteChart:
    """A model × benchmark heatmap of primary scores (Vega-Lite ``rect`` mark)."""
    values: list[dict[str, Any]] = []
    for row in leaderboard.rows:
        for benchmark in leaderboard.benchmarks:
            score = row.benchmark_scores.get(benchmark)
            if score is not None:
                values.append({"model": row.model, "benchmark": benchmark, "score": round(score, 4)})
    spec = {
        "$schema": _SCHEMA, "title": title, "data": {"values": values},
        "mark": {"type": "rect", "tooltip": True},
        "encoding": {
            "x": {"field": "benchmark", "type": "nominal"},
            "y": {"field": "model", "type": "nominal"},
            "color": {"field": "score", "type": "quantitative", "scale": {"scheme": "blues"}},
        },
    }
    return SuiteChart(kind="heatmap", title=title, spec=spec, rows=values)


def confusion_matrix_chart(
    matrix: list[list[int]], labels: list[str], *, title: str = "Confusion matrix"
) -> SuiteChart:
    """A confusion-matrix heatmap from a counts matrix and its class labels."""
    if any(len(row) != len(labels) for row in matrix) or len(matrix) != len(labels):
        raise EvalSuiteError("confusion matrix must be square and match the label count")
    values = [
        {"actual": labels[i], "predicted": labels[j], "count": int(matrix[i][j])}
        for i in range(len(labels))
        for j in range(len(labels))
    ]
    spec = {
        "$schema": _SCHEMA, "title": title, "data": {"values": values},
        "mark": {"type": "rect", "tooltip": True},
        "encoding": {
            "x": {"field": "predicted", "type": "nominal"},
            "y": {"field": "actual", "type": "nominal"},
            "color": {"field": "count", "type": "quantitative", "scale": {"scheme": "blues"}},
        },
    }
    return SuiteChart(kind="confusion_matrix", title=title, spec=spec, rows=values)


def trend_chart(
    history: list[dict[str, Any]],
    *,
    x: str = "version",
    y: str = "overall",
    title: str = "Historical trend",
) -> SuiteChart:
    """A line of a metric across run versions / dates (model-version history)."""
    values = [{x: str(point.get(x, "")), y: round(float(point.get(y, 0.0)), 4)} for point in history]
    spec = {
        "$schema": _SCHEMA, "title": title, "data": {"values": values},
        "mark": {"type": "line", "point": True, "tooltip": True},
        "encoding": {
            "x": {"field": x, "type": "ordinal"},
            "y": {"field": y, "type": "quantitative"},
        },
    }
    return SuiteChart(kind="trend", title=title, spec=spec, rows=values)


# ---------------------------------------------------------------------------
# Optional matplotlib rasterizer (vincio[eval-viz])
# ---------------------------------------------------------------------------


def _render_png(chart: SuiteChart, *, width: int, height: int, dpi: int) -> bytes:
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise EvalSuiteError(
            'chart rasterization requires matplotlib: pip install "vincio[eval-viz]"'
        ) from exc
    matplotlib.use("Agg")  # pragma: no cover - rendering path
    import io  # pragma: no cover

    import matplotlib.pyplot as plt  # pragma: no cover

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)  # pragma: no cover
    try:  # pragma: no cover - rendering path
        _draw(chart, ax)
        ax.set_title(chart.title)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", metadata={"Software": None})
        return buffer.getvalue()
    finally:  # pragma: no cover
        plt.close(fig)


def _draw(chart: SuiteChart, ax: Any) -> None:  # pragma: no cover - rendering path
    import math

    rows = chart.rows
    if chart.kind == "leaderboard":
        ax.bar([r["model"] for r in rows], [r["overall"] for r in rows])
        ax.set_ylabel("overall")
    elif chart.kind == "radar":
        labels = [r["niche"] for r in rows]
        scores = [r["score"] for r in rows]
        angles = [n / len(labels) * 2 * math.pi for n in range(len(labels))] + [0.0]
        ax.remove()
        ax = ax.figure.add_subplot(111, polar=True)
        ax.plot(angles, scores + scores[:1])
        ax.fill(angles, scores + scores[:1], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
    elif chart.kind in ("heatmap", "confusion_matrix"):
        xs = sorted({r[list(r)[1]] for r in rows})
        ys = sorted({r[list(r)[0]] for r in rows})
        value_key = "score" if chart.kind == "heatmap" else "count"
        x_key, y_key = list(rows[0])[1], list(rows[0])[0]
        grid = [[next((r[value_key] for r in rows if r[x_key] == x and r[y_key] == y), 0)
                 for x in xs] for y in ys]
        ax.imshow(grid, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(xs)), xs, rotation=45, ha="right")
        ax.set_yticks(range(len(ys)), ys)
    elif chart.kind == "trend":
        keys = list(rows[0]) if rows else ["version", "overall"]
        ax.plot([r[keys[0]] for r in rows], [r[keys[1]] for r in rows], marker="o")
        ax.set_ylabel(keys[1])
