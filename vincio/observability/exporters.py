"""Trace exporters: JSONL, in-memory, console, fan-out."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Protocol

from ..core.utils import to_jsonable
from .spans import Trace

__all__ = [
    "TraceExporter",
    "InMemoryExporter",
    "JSONLExporter",
    "ConsoleExporter",
    "MultiExporter",
    "NullExporter",
]

logger = logging.getLogger("vincio.observability")


class TraceExporter(Protocol):
    def export(self, trace: Trace) -> None:  # pragma: no cover - protocol
        ...


class NullExporter:
    def export(self, trace: Trace) -> None:
        return None


class InMemoryExporter:
    """Keeps traces in memory — used by tests and trace queries."""

    def __init__(self, max_traces: int = 1000) -> None:
        self.max_traces = max_traces
        self.traces: list[Trace] = []
        self._lock = threading.Lock()

    def export(self, trace: Trace) -> None:
        with self._lock:
            self.traces.append(trace)
            if len(self.traces) > self.max_traces:
                self.traces = self.traces[-self.max_traces :]

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            for trace in reversed(self.traces):
                if trace.id == trace_id:
                    return trace
        return None

    def clear(self) -> None:
        with self._lock:
            self.traces.clear()


class JSONLExporter:
    """Appends one JSON object per trace to ``<dir>/traces.jsonl``."""

    def __init__(self, directory: str | Path = ".vincio/traces") -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self.directory / "traces.jsonl"

    def export(self, trace: Trace) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            line = json.dumps(to_jsonable(trace.model_dump(mode="json")), ensure_ascii=False)
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            logger.exception("failed to export trace %s", trace.id)

    def load(self, trace_id: str) -> Trace | None:
        if not self.path.is_file():
            return None
        found: Trace | None = None
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("id") == trace_id:
                    found = Trace.model_validate(data)  # keep last occurrence
        return found

    def load_all(self, limit: int | None = None) -> list[Trace]:
        """All traces, latest record per trace id (re-exports act as updates)."""
        latest: dict[str, Trace] = {}
        if not self.path.is_file():
            return []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = Trace.model_validate(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
                latest.pop(trace.id, None)  # re-insert so updates keep file order
                latest[trace.id] = trace
        traces = list(latest.values())
        if limit is not None:
            traces = traces[-limit:]
        return traces


class ConsoleExporter:
    """Logs a one-line summary per trace — convenient during development."""

    def export(self, trace: Trace) -> None:
        logger.info(
            "trace %s app=%s status=%s spans=%d duration_ms=%d",
            trace.id,
            trace.app_name,
            trace.status,
            len(trace.spans),
            trace.duration_ms,
        )


class MultiExporter:
    def __init__(self, exporters: list[TraceExporter]) -> None:
        self.exporters = exporters

    def export(self, trace: Trace) -> None:
        for exporter in self.exporters:
            try:
                exporter.export(trace)
            except Exception:  # noqa: BLE001 - exporters must not break runs
                logger.exception("exporter %s failed", type(exporter).__name__)
