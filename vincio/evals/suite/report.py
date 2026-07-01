"""Reporting & the leaderboard — one run rendered every way, every number tiered.

A :class:`SuiteReport` renders one :class:`~vincio.evals.suite.results.SuiteRun`
to Markdown, HTML, JSON, CSV, and PDF, each **citing the exact scored items** and
carrying the **provenance tier** on every number — the honesty culture made
visible. A :class:`Leaderboard` ranks several models (or model *versions*) over a
shared benchmark set the same way.

The Markdown / HTML / PDF renderers compose the generation subsystem's document
IR (``DocumentModel`` → ``render``); CSV is emitted with the standard library;
JSON is the validated result model. The Markdown/HTML/JSON/CSV bytes are a pure
function of the run (no wall-clock in the body), so a Tier-S/R report diffs
cleanly across machines.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ...core.errors import EvalSuiteError
from .registry import NICHES
from .results import SuiteRun
from .tiers import ProvenanceTier

__all__ = ["Leaderboard", "LeaderboardRow", "SuiteReport"]


_FORMATS = ("markdown", "html", "json", "csv", "pdf")
_TIER_LEGEND = (
    "Every number carries its provenance tier — "
    "**S** Static (fabricated fixture, reproducible, gates CI), "
    "**R** Recorded (hash-pinned real slice replayed, reproducible, gates CI), "
    "**L** Live (full dataset against a live model, reported, never gated). "
    "A lower tier can never print a higher tier's label."
)


def _tier_cell(tier: ProvenanceTier) -> str:
    return tier.code


# ---------------------------------------------------------------------------
# Leaderboard — rank several models / model versions over a shared benchmark set
# ---------------------------------------------------------------------------


class LeaderboardRow(BaseModel):
    """One model's place on the leaderboard."""

    rank: int = 0
    model: str
    run_id: str = ""
    tier: ProvenanceTier = ProvenanceTier.STATIC
    overall: float = 0.0
    niche_scores: dict[str, float] = Field(default_factory=dict)
    benchmark_scores: dict[str, float] = Field(default_factory=dict)


class Leaderboard(BaseModel):
    """A ranked comparison of models over a shared benchmark set.

    Built from one or more :class:`SuiteRun`s (one per model / model version). Rows
    are ranked by overall mean primary score; the per-niche and per-benchmark
    columns let a reader see where a model wins or loses, not just the headline.
    A leaderboard mixing tiers carries each row's tier so a Live number is never
    silently compared against a reproducible one.
    """

    title: str = "Leaderboard"
    rows: list[LeaderboardRow] = Field(default_factory=list)
    benchmarks: list[str] = Field(default_factory=list)
    niches: list[str] = Field(default_factory=list)

    @classmethod
    def from_runs(cls, runs: list[SuiteRun], *, title: str = "Leaderboard") -> Leaderboard:
        benchmarks: list[str] = sorted({r.benchmark_id for run in runs for r in run.runs})
        niches: list[str] = [n for n in NICHES if any(
            r.niche == n for run in runs for r in run.runs
        )]
        rows = [
            LeaderboardRow(
                model=run.model, run_id=run.run_id, tier=run.tier, overall=run.overall(),
                niche_scores=run.niche_scores(),
                benchmark_scores={r.benchmark_id: r.primary for r in run.runs},
            )
            for run in runs
        ]
        rows.sort(key=lambda r: (-r.overall, r.model))
        for index, row in enumerate(rows, start=1):
            row.rank = index
        return cls(title=title, rows=rows, benchmarks=benchmarks, niches=niches)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", _TIER_LEGEND, "",
                 "| Rank | Model | Tier | Overall | " + " | ".join(NICHES[n] for n in self.niches) + " |",
                 "|---:|---|:--:|---:|" + "|".join("---:" for _ in self.niches) + "|"]
        for row in self.rows:
            cells = " | ".join(f"{row.niche_scores.get(n, float('nan')):.3f}"
                               if n in row.niche_scores else "—" for n in self.niches)
            lines.append(
                f"| {row.rank} | {row.model} | {row.tier.code} | {row.overall:.3f} | {cells} |"
            )
        return "\n".join(lines) + "\n"

    def model_dump_json_indented(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, default=str)


# ---------------------------------------------------------------------------
# SuiteReport — one run rendered to every format, citing the scored items
# ---------------------------------------------------------------------------


class SuiteReport:
    """Render one :class:`SuiteRun` to Markdown / HTML / JSON / CSV / PDF.

    Each rendering carries the run's tier, the per-benchmark scores, and the exact
    scored items (the failing tasks are cited by id), so a number is always
    traceable to what produced it.
    """

    def __init__(self, run: SuiteRun, *, title: str | None = None, cite_failures: int = 10) -> None:
        self.run = run
        self.title = title or f"Evaluation report — {run.model}"
        self.cite_failures = cite_failures

    # -- the document IR ------------------------------------------------------

    def to_document(self) -> Any:
        from ...documents.parsers import TableData
        from ...generation import DocumentModel

        run = self.run
        doc = DocumentModel(title=self.title)
        env = run.environment
        doc = doc.paragraph(
            f"Model `{run.model}` · tier **{run.tier.code} ({run.tier.label})** · "
            f"{len(run.runs)} benchmarks · overall primary **{run.overall():.3f}** · "
            f"Vincio {env.get('vincio_version', '?')} · "
            f"{'reproducible, gates CI' if run.gated else 'reported, not gated'}."
        )
        doc = doc.paragraph(_TIER_LEGEND)

        # Headline table: one row per benchmark, every cell tier-tagged.
        summary_rows = [
            [r.niche, r.benchmark_id, r.tier.code, r.primary_metric,
             f"{r.primary:.3f}", str(r.n)]
            for r in sorted(run.runs, key=lambda r: r.benchmark_id)
        ]
        doc = doc.heading("Results", level=2)
        doc = doc.add_table(TableData(
            columns=["Niche", "Benchmark", "Tier", "Metric", "Score", "N"], rows=summary_rows
        ))

        # Per-niche detail with cited failing items.
        for niche, runs in run.by_niche().items():
            doc = doc.heading(NICHES.get(niche, niche), level=2)
            for r in runs:
                line = f"`{r.benchmark_id}` — {r.primary_metric} **{r.primary:.3f}** (tier {r.tier.code}, n={r.n})"
                if r.governed is not None:
                    line += (f" · governor uplift **{r.governed['uplift']:+.3f}** "
                             f"(base {r.governed['base']:.3f} → governed {r.governed['governed']:.3f})")
                doc = doc.paragraph(line)
                failures = r.failures[: self.cite_failures]
                if failures:
                    doc = doc.bullet_list([
                        f"{f.task_id}: scored {f.score:.2f} (tier {f.tier.code})" for f in failures
                    ])
        return doc

    # -- renderers ------------------------------------------------------------

    def to_markdown(self) -> str:
        from ...generation import render

        return render(self.to_document(), "markdown").text

    def to_html(self) -> str:
        from ...generation import render

        return render(self.to_document(), "html").text

    def to_json(self) -> str:
        # Exclude the wall-clock ``created_at`` so the JSON body — like the
        # Markdown / HTML / CSV bodies — is a pure function of the run and a
        # Tier-S/R report diffs cleanly across machines.
        return json.dumps(
            self.run.model_dump(mode="json", exclude={"created_at"}), indent=2, default=str
        )

    def to_csv(self) -> str:
        """A flat CSV: one row per scored benchmark, every number tier-tagged."""
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["model", "niche", "benchmark", "tier", "metric",
                         "primary", "success_rate", "mean_score", "n", "task_set_hash"])
        for r in sorted(self.run.runs, key=lambda r: r.benchmark_id):
            writer.writerow([self.run.model, r.niche, r.benchmark_id, r.tier.code, r.primary_metric,
                             f"{r.primary:.4f}", f"{r.success_rate:.4f}", f"{r.mean_score:.4f}",
                             r.n, r.task_set_hash])
        return buffer.getvalue()

    def to_pdf(self) -> bytes:
        """Render to PDF (requires ``vincio[eval-pdf]`` / reportlab)."""
        from ...core.errors import GenerationError
        from ...generation import render

        try:
            return render(self.to_document(), "pdf").content
        except GenerationError as exc:  # the reportlab backend is absent
            raise EvalSuiteError(
                'PDF reports require reportlab: pip install "vincio[eval-pdf]"'
            ) from exc

    def render(self, fmt: str = "markdown") -> str | bytes:
        """Render to ``fmt`` (``markdown`` / ``html`` / ``json`` / ``csv`` / ``pdf``)."""
        if fmt not in _FORMATS:
            raise EvalSuiteError(f"unknown report format {fmt!r}; known: {list(_FORMATS)}")
        if fmt == "pdf":
            return self.to_pdf()
        return {"markdown": self.to_markdown, "html": self.to_html,
                "json": self.to_json, "csv": self.to_csv}[fmt]()

    def save(self, path: str | Path, *, format: str | None = None) -> str:
        """Render and write to ``path``; the format is inferred from the suffix."""
        target = Path(path)
        fmt = format or {
            ".md": "markdown", ".markdown": "markdown", ".html": "html",
            ".htm": "html", ".json": "json", ".csv": "csv", ".pdf": "pdf",
        }.get(target.suffix.lower(), "markdown")
        content = self.render(fmt)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
        return str(target)
