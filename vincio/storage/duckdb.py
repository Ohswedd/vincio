"""DuckDB analytics: columnar queries over eval results, runs,
and traces. Requires ``pip install "vincio[duckdb]"``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.errors import StorageError
from ..evals.reports import EvalReport

__all__ = ["DuckDBAnalytics"]


class DuckDBAnalytics:
    def __init__(self, path: str | Path = ".vincio/analytics.duckdb") -> None:
        try:
            import duckdb
        except ImportError as exc:
            raise StorageError(
                'DuckDB analytics requires: pip install "vincio[duckdb]"'
            ) from exc
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_metrics (
              report_name VARCHAR, dataset VARCHAR, case_id VARCHAR,
              metric VARCHAR, value DOUBLE, created_at TIMESTAMP, tags VARCHAR
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id VARCHAR, app_id VARCHAR, tenant_id VARCHAR, status VARCHAR,
              cost_usd DOUBLE, latency_ms BIGINT, input_tokens BIGINT,
              output_tokens BIGINT, started_at TIMESTAMP
            )
            """
        )

    def close(self) -> None:
        self._conn.close()

    def ingest_report(self, report: EvalReport) -> int:
        rows = []
        for case in report.cases:
            for metric, value in case.metrics.items():
                rows.append(
                    (report.name, report.dataset, case.case_id, metric, float(value),
                     report.created_at, ",".join(case.tags))
                )
        if rows:
            self._conn.executemany("INSERT INTO eval_metrics VALUES (?,?,?,?,?,?,?)", rows)
        return len(rows)

    def ingest_run(self, run: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
            (
                run.get("id"), run.get("app_id"), run.get("tenant_id"), run.get("status"),
                run.get("cost_usd", 0.0), run.get("latency_ms", 0),
                run.get("input_tokens", 0), run.get("output_tokens", 0),
                run.get("started_at"),
            ),
        )

    def query(self, sql: str, params: list[Any] | None = None) -> list[tuple]:
        return self._conn.execute(sql, params or []).fetchall()

    def metric_trend(self, metric: str, *, dataset: str | None = None) -> list[tuple]:
        """Mean metric value per report over time."""
        sql = (
            "SELECT report_name, AVG(value) AS mean_value, COUNT(*) AS n, MIN(created_at) AS at "
            "FROM eval_metrics WHERE metric = ?"
        )
        params: list[Any] = [metric]
        if dataset:
            sql += " AND dataset = ?"
            params.append(dataset)
        sql += " GROUP BY report_name ORDER BY at"
        return self.query(sql, params)

    def cost_by_app(self) -> list[tuple]:
        return self.query(
            "SELECT app_id, COUNT(*) AS runs, SUM(cost_usd) AS total_cost, "
            "AVG(latency_ms) AS avg_latency FROM runs GROUP BY app_id ORDER BY total_cost DESC"
        )
