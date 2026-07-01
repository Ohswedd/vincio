"""Storage — run history and model-version comparison.

A :class:`RunStore` persists every :class:`~vincio.evals.suite.results.SuiteRun`
so a later run can be compared against an earlier one and a model version diffed
against its predecessor — the in-process equivalent of a leaderboard's history,
over **your own store**. The default backend is the standard-library ``sqlite3``
(no extra); a Postgres DSN (``postgresql://…``) is honored behind
``vincio[eval-store]``. Nothing here is a hosted service.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ...core.errors import EvalSuiteError
from .results import SuiteRun

__all__ = ["RunStore"]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS suite_runs (
    run_id      TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    version     TEXT NOT NULL DEFAULT '',
    tier        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    overall     REAL NOT NULL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_suite_runs_model ON suite_runs(model, created_at);
"""


class RunStore:
    """Persist and query :class:`SuiteRun`s over SQLite (default) or Postgres.

    ``dsn`` is a SQLite file path (the default ``.vincio/eval_runs.db``) or a
    ``postgresql://…`` DSN (requires ``vincio[eval-store]``). Construction creates
    the schema if absent.
    """

    def __init__(self, dsn: str | Path = ".vincio/eval_runs.db") -> None:
        self.dsn = str(dsn)
        self._is_postgres = self.dsn.startswith(("postgres://", "postgresql://"))
        if self._is_postgres:
            self._conn = self._connect_postgres()
        else:
            path = Path(self.dsn)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.dsn)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _connect_postgres(self) -> Any:  # pragma: no cover - optional dependency
        try:
            import psycopg
        except ImportError as exc:
            raise EvalSuiteError(
                'a Postgres run store requires psycopg: pip install "vincio[eval-store]"'
            ) from exc
        conn = psycopg.connect(self.dsn)
        with conn.cursor() as cur:
            cur.execute(_SCHEMA.replace("AUTOINCREMENT", ""))
        conn.commit()
        return conn

    # -- persistence ----------------------------------------------------------

    def save(self, run: SuiteRun, *, version: str = "") -> str:
        """Persist a run (idempotent on ``run_id``). ``version`` tags the model
        version so :meth:`model_version_diff` can compare across versions."""
        version = version or str(run.metadata.get("version", ""))
        payload = json.dumps(run.model_dump(mode="json"), default=str)
        row = (run.run_id, run.model, version, run.tier.value,
               str(run.created_at), run.overall(), payload)
        if self._is_postgres:  # pragma: no cover - optional dependency
            with self._conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO suite_runs (run_id, model, version, tier, created_at, overall, payload) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (run_id) DO UPDATE SET "
                    "payload = EXCLUDED.payload, overall = EXCLUDED.overall, version = EXCLUDED.version",
                    row,
                )
            self._conn.commit()
        else:
            self._conn.execute(
                "INSERT OR REPLACE INTO suite_runs "
                "(run_id, model, version, tier, created_at, overall, payload) VALUES (?,?,?,?,?,?,?)",
                row,
            )
            self._conn.commit()
        return run.run_id

    def get(self, run_id: str) -> SuiteRun:
        """Load a persisted run by id."""
        rows = self._query("SELECT payload FROM suite_runs WHERE run_id = ?", (run_id,))
        if not rows:
            raise EvalSuiteError(f"no run {run_id!r} in the store")
        return SuiteRun.model_validate(json.loads(rows[0][0]))

    def list_runs(self, *, model: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Run summaries (newest first), optionally filtered by model."""
        if model is not None:
            sql = ("SELECT run_id, model, version, tier, created_at, overall FROM suite_runs "
                   "WHERE model = ? ORDER BY created_at DESC, run_id DESC LIMIT ?")
            params: tuple[Any, ...] = (model, limit)
        else:
            sql = ("SELECT run_id, model, version, tier, created_at, overall FROM suite_runs "
                   "ORDER BY created_at DESC, run_id DESC LIMIT ?")
            params = (limit,)
        return [
            {"run_id": r[0], "model": r[1], "version": r[2], "tier": r[3],
             "created_at": r[4], "overall": r[5]}
            for r in self._query(sql, params)
        ]

    # -- comparison -----------------------------------------------------------

    def compare_runs(self, run_a_id: str, run_b_id: str) -> dict[str, Any]:
        """Per-benchmark delta between two runs (``b`` − ``a``).

        Returns the overall delta, a per-benchmark ``{from, to, delta}`` table, and
        the regressions / improvements, so a model swap or a rerun is judged by
        what moved, not just the headline.
        """
        run_a, run_b = self.get(run_a_id), self.get(run_b_id)
        return _diff_runs(run_a, run_b)

    def model_version_diff(
        self, model: str, *, version_a: str | None = None, version_b: str | None = None
    ) -> dict[str, Any]:
        """Diff two versions of ``model`` (default: the two most recent runs).

        With ``version_a`` / ``version_b`` the named versions are compared; without
        them, the latest run is diffed against the one before it — the usual
        "did this version regress?" question.
        """
        runs = self.list_runs(model=model, limit=100)
        if version_a is not None and version_b is not None:
            a = self._latest_for_version(model, version_a)
            b = self._latest_for_version(model, version_b)
        else:
            if len(runs) < 2:
                raise EvalSuiteError(f"need at least two runs of {model!r} to diff (have {len(runs)})")
            b = self.get(runs[0]["run_id"])
            a = self.get(runs[1]["run_id"])
        diff = _diff_runs(a, b)
        diff["model"] = model
        return diff

    def history(self, model: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """A model's run history oldest-first — feed straight to ``trend_chart``."""
        runs = self.list_runs(model=model, limit=limit)
        runs.reverse()
        return [{"version": r["version"] or r["run_id"][:10], "overall": r["overall"],
                 "created_at": r["created_at"], "tier": r["tier"]} for r in runs]

    def close(self) -> None:
        self._conn.close()

    # -- internals ------------------------------------------------------------

    def _latest_for_version(self, model: str, version: str) -> SuiteRun:
        rows = self._query(
            "SELECT run_id FROM suite_runs WHERE model = ? AND version = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (model, version),
        )
        if not rows:
            raise EvalSuiteError(f"no run for {model!r} version {version!r}")
        return self.get(rows[0][0])

    def _query(self, sql: str, params: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        if self._is_postgres:  # pragma: no cover - optional dependency
            with self._conn.cursor() as cur:
                cur.execute(sql.replace("?", "%s"), params)
                return list(cur.fetchall())
        cur = self._conn.execute(sql, params)
        return list(cur.fetchall())


def _diff_runs(run_a: SuiteRun, run_b: SuiteRun) -> dict[str, Any]:
    a_scores = {r.benchmark_id: r.primary for r in run_a.runs}
    b_scores = {r.benchmark_id: r.primary for r in run_b.runs}
    benchmarks = sorted(set(a_scores) | set(b_scores))
    table: dict[str, dict[str, float | None]] = {}
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        before, after = a_scores.get(benchmark), b_scores.get(benchmark)
        delta = round(after - before, 4) if before is not None and after is not None else None
        table[benchmark] = {"from": before, "to": after, "delta": delta}
        if delta is not None and delta < -1e-9:
            regressions.append({"benchmark": benchmark, "from": before, "to": after, "delta": delta})
        elif delta is not None and delta > 1e-9:
            improvements.append({"benchmark": benchmark, "from": before, "to": after, "delta": delta})
    return {
        "run_a": run_a.run_id, "run_b": run_b.run_id,
        "model_a": run_a.model, "model_b": run_b.model,
        "overall_from": run_a.overall(), "overall_to": run_b.overall(),
        "overall_delta": round(run_b.overall() - run_a.overall(), 4),
        "benchmarks": table,
        "regressions": sorted(regressions, key=lambda d: d["delta"]),
        "improvements": sorted(improvements, key=lambda d: -d["delta"]),
    }
