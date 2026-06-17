"""Indexed trace/cost store for the served observability plane (2.1).

The zero-dependency JSONL exporter is perfect for a single run and a static
HTML export, but answering "p95 latency for tenant X over the last hour" or
"cost by model today" against it means an O(n) scan of every line. The served
plane (``vincio.observability.viewer.serve_viewer``) needs indexed reads, so
this module persists traces into a SQLite database with:

* a ``traces`` table indexed by ``(tenant, start)``, ``(status, start)`` and
  ``session`` for fast filtered tails and search;
* ``cost_buckets`` **pre-aggregates** — per ``(bucket-size, period, dimension,
  key)`` rollups of calls / cost / duration / errors — so dashboards read a
  handful of summary rows instead of scanning raw traces; and
* a ``purge`` retention sweep so the store stays bounded.

It implements the :class:`~vincio.observability.exporters.TraceExporter`
protocol (``export(trace)``), so you add it alongside the static JSONL exporter
— the zero-dep path stays, this one powers the served dashboards. Percentiles
are computed over the indexed window (bounded by ``percentile_window``) rather
than a full scan. No new dependency: the standard-library ``sqlite3`` only.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..core.utils import to_jsonable
from .spans import Trace

__all__ = [
    "IndexedTraceStore",
    "Percentiles",
    "RollupBucket",
    "CostSlice",
]

# Pre-aggregate bucket widths (seconds) and the dimensions rolled up per trace.
_BUCKETS = {"1h": 3600, "1d": 86400}
_DIMENSIONS = ("global", "tenant", "model")


class Percentiles(BaseModel):
    """p50/p95/p99 of a metric over a window."""

    metric: str
    count: int = 0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    max: float = 0.0


class RollupBucket(BaseModel):
    period_start: float  # epoch seconds at the bucket's start
    calls: int = 0
    cost_usd: float = 0.0
    errors: int = 0
    duration_ms_avg: float = 0.0


class CostSlice(BaseModel):
    key: str
    cost_usd: float = 0.0
    calls: int = 0
    errors: int = 0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    return float(ordered[low] * (high - rank) + ordered[high] * (rank - low))


def _trace_cost(trace: Trace) -> float:
    """Best-effort total cost of a trace.

    Prefer a run-level total span; otherwise sum the model-call spans, never
    double-counting the two.
    """
    for span in trace.spans:
        if span.type == "run" and "cost_usd" in span.attributes:
            return float(span.attributes.get("cost_usd") or 0.0)
    model_cost = sum(
        float(span.attributes.get("cost_usd") or 0.0)
        for span in trace.spans
        if span.type == "model_call"
    )
    attr = float(trace.attributes.get("cost_usd") or 0.0)
    return max(model_cost, attr)


def _trace_model(trace: Trace) -> str:
    for span in trace.spans:
        if span.type == "model_call":
            model = span.attributes.get("model") or span.attributes.get("gen_ai.request.model")
            if model:
                return str(model)
    return str(trace.attributes.get("model") or "")


class IndexedTraceStore:
    """SQLite-backed, indexed trace + cost store with pre-aggregated rollups."""

    def __init__(
        self, path: str | Path = ".vincio/observability.db", *, percentile_window: int = 5000
    ) -> None:
        self.path = Path(path)
        self.percentile_window = percentile_window
        self._lock = threading.Lock()
        if self.path.parent and str(self.path.parent):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    app_name TEXT,
                    tenant_id TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    status TEXT,
                    model TEXT,
                    start_ts REAL,
                    end_ts REAL,
                    duration_ms INTEGER,
                    cost_usd REAL,
                    is_error INTEGER,
                    json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_traces_start ON traces(start_ts);
                CREATE INDEX IF NOT EXISTS idx_traces_tenant ON traces(tenant_id, start_ts);
                CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status, start_ts);
                CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
                CREATE INDEX IF NOT EXISTS idx_traces_model ON traces(model, start_ts);

                CREATE TABLE IF NOT EXISTS cost_buckets (
                    bucket TEXT,
                    period_start REAL,
                    dimension TEXT,
                    key TEXT,
                    calls INTEGER,
                    cost_usd REAL,
                    duration_ms_sum REAL,
                    errors INTEGER,
                    PRIMARY KEY (bucket, period_start, dimension, key)
                );
                CREATE INDEX IF NOT EXISTS idx_buckets_lookup
                    ON cost_buckets(dimension, key, bucket, period_start);
                """
            )
            self._conn.commit()

    # -- write path ------------------------------------------------------------

    def export(self, trace: Trace) -> None:
        """Persist a trace and fold it into the pre-aggregates (TraceExporter)."""
        start_ts = trace.start_time.timestamp()
        end_ts = trace.end_time.timestamp() if trace.end_time else None
        cost = _trace_cost(trace)
        model = _trace_model(trace)
        is_error = 1 if trace.status == "error" else 0
        body = json.dumps(to_jsonable(trace.model_dump(mode="json")), ensure_ascii=False)
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM traces WHERE id = ?", (trace.id,)
            ).fetchone()
            self._conn.execute(
                """
                INSERT INTO traces
                  (id, app_name, tenant_id, user_id, session_id, status, model,
                   start_ts, end_ts, duration_ms, cost_usd, is_error, json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  status=excluded.status, end_ts=excluded.end_ts,
                  duration_ms=excluded.duration_ms, cost_usd=excluded.cost_usd,
                  model=excluded.model, is_error=excluded.is_error, json=excluded.json
                """,
                (
                    trace.id, trace.app_name, trace.tenant_id, trace.user_id, trace.session_id,
                    trace.status, model, start_ts, end_ts, trace.duration_ms, cost, is_error, body,
                ),
            )
            # Only fold a trace into the rollups once (re-exports update the row,
            # not the aggregates) so counts/sums stay honest.
            if existing is None:
                self._fold_buckets(trace, start_ts=start_ts, cost=cost, model=model, is_error=is_error)
            self._conn.commit()

    record = export

    def _fold_buckets(
        self, trace: Trace, *, start_ts: float, cost: float, model: str, is_error: int
    ) -> None:
        keys = {"global": "∅", "tenant": trace.tenant_id or "∅", "model": model or "∅"}
        for bucket, width in _BUCKETS.items():
            period_start = math.floor(start_ts / width) * width
            for dimension in _DIMENSIONS:
                self._conn.execute(
                    """
                    INSERT INTO cost_buckets
                      (bucket, period_start, dimension, key, calls, cost_usd, duration_ms_sum, errors)
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    ON CONFLICT(bucket, period_start, dimension, key) DO UPDATE SET
                      calls = calls + 1,
                      cost_usd = cost_usd + excluded.cost_usd,
                      duration_ms_sum = duration_ms_sum + excluded.duration_ms_sum,
                      errors = errors + excluded.errors
                    """,
                    (bucket, period_start, dimension, keys[dimension], cost, trace.duration_ms, is_error),
                )

    # -- read path -------------------------------------------------------------

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT json FROM traces WHERE id = ?", (trace_id,)
            ).fetchone()
        return Trace.model_validate(json.loads(row["json"])) if row else None

    def query(
        self,
        *,
        tenant_id: str | None = None,
        model: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[Trace]:
        """Indexed, filtered, most-recent-first trace search."""
        clauses: list[str] = []
        params: list[Any] = []
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if since is not None:
            clauses.append("start_ts >= ?")
            params.append(since.timestamp())
        if until is not None:
            clauses.append("start_ts <= ?")
            params.append(until.timestamp())
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT json FROM traces{where} ORDER BY start_ts DESC LIMIT ?", params
            ).fetchall()
        return [Trace.model_validate(json.loads(r["json"])) for r in rows]

    def tail(self, limit: int = 50, **filters: Any) -> list[Trace]:
        """The most recent traces (live tail), newest first."""
        return self.query(limit=limit, **filters)

    def percentiles(
        self,
        metric: str = "latency",
        *,
        since: datetime | None = None,
        tenant_id: str | None = None,
    ) -> Percentiles:
        """p50/p95/p99 of ``latency`` (ms) or ``cost`` (usd) over the window."""
        column = "duration_ms" if metric == "latency" else "cost_usd"
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("start_ts >= ?")
            params.append(since.timestamp())
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(self.percentile_window))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {column} AS v FROM traces{where} ORDER BY start_ts DESC LIMIT ?", params
            ).fetchall()
        values = [float(r["v"] or 0.0) for r in rows]
        return Percentiles(
            metric=metric,
            count=len(values),
            p50=round(_percentile(values, 0.50), 4),
            p95=round(_percentile(values, 0.95), 4),
            p99=round(_percentile(values, 0.99), 4),
            max=round(max(values), 4) if values else 0.0,
        )

    def cost_by_dimension(
        self, dimension: str = "tenant", *, since: datetime | None = None, limit: int = 20
    ) -> list[CostSlice]:
        """Cost rolled up by ``tenant`` / ``model`` / ``user`` / ``session``."""
        column = {
            "tenant": "tenant_id",
            "model": "model",
            "user": "user_id",
            "session": "session_id",
        }.get(dimension, "tenant_id")
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("start_ts >= ?")
            params.append(since.timestamp())
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT COALESCE({column}, '∅') AS key,
                       SUM(cost_usd) AS cost, COUNT(*) AS calls, SUM(is_error) AS errors
                FROM traces{where}
                GROUP BY key ORDER BY cost DESC LIMIT ?
                """,
                params,
            ).fetchall()
        return [
            CostSlice(
                key=str(r["key"]),
                cost_usd=round(float(r["cost"] or 0.0), 8),
                calls=int(r["calls"]),
                errors=int(r["errors"] or 0),
            )
            for r in rows
        ]

    def rollup(
        self,
        bucket: str = "1h",
        *,
        dimension: str = "global",
        key: str = "∅",
        since: datetime | None = None,
        limit: int = 168,
    ) -> list[RollupBucket]:
        """Time-bucketed series from the pre-aggregates (no raw-trace scan)."""
        if bucket not in _BUCKETS:
            raise ValueError(f"unknown bucket {bucket!r}; use one of {sorted(_BUCKETS)}")
        clauses = ["bucket = ?", "dimension = ?", "key = ?"]
        params: list[Any] = [bucket, dimension, key]
        if since is not None:
            clauses.append("period_start >= ?")
            params.append(math.floor(since.timestamp() / _BUCKETS[bucket]) * _BUCKETS[bucket])
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT period_start, calls, cost_usd, duration_ms_sum, errors
                FROM cost_buckets
                WHERE {' AND '.join(clauses)}
                ORDER BY period_start DESC LIMIT ?
                """,
                params,
            ).fetchall()
        out = [
            RollupBucket(
                period_start=float(r["period_start"]),
                calls=int(r["calls"]),
                cost_usd=round(float(r["cost_usd"] or 0.0), 8),
                errors=int(r["errors"] or 0),
                duration_ms_avg=round((r["duration_ms_sum"] or 0.0) / max(1, r["calls"]), 2),
            )
            for r in rows
        ]
        out.reverse()  # chronological
        return out

    def purge(self, *, before: datetime) -> int:
        """Retention: delete traces (and stale buckets) older than ``before``."""
        cutoff = before.timestamp()
        with self._lock:
            cur = self._conn.execute("DELETE FROM traces WHERE start_ts < ?", (cutoff,))
            deleted = cur.rowcount
            self._conn.execute("DELETE FROM cost_buckets WHERE period_start < ?", (cutoff,))
            self._conn.commit()
        return int(deleted)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) AS n FROM traces").fetchone()["n"]
            errors = self._conn.execute(
                "SELECT COUNT(*) AS n FROM traces WHERE is_error = 1"
            ).fetchone()["n"]
            cost = self._conn.execute("SELECT SUM(cost_usd) AS c FROM traces").fetchone()["c"]
        return {
            "traces": int(total),
            "errors": int(errors),
            "error_rate": round(errors / total, 4) if total else 0.0,
            "total_cost_usd": round(float(cost or 0.0), 8),
        }

    def count(self) -> int:
        return int(self.stats()["traces"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __len__(self) -> int:
        return self.count()
