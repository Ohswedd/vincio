"""Trace exporters: JSONL, in-memory, console, fan-out."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from ..core.utils import new_id, to_jsonable, utcnow
from .spans import Trace

__all__ = [
    "TraceExporter",
    "InMemoryExporter",
    "JSONLExporter",
    "ConsoleExporter",
    "MultiExporter",
    "NullExporter",
    # 2.1: tail-based sampling + alert sinks
    "TailSamplingExporter",
    "Alert",
    "AlertSink",
    "MemoryAlertSink",
    "WebhookAlertSink",
    "SlackAlertSink",
    "PagerDutyAlertSink",
    "PrometheusExporter",
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


class TailSamplingExporter:
    """Tail-based, error-prioritized sampling in front of an inner exporter (2.1).

    The served plane should keep what matters and drop the noise: this wrapper
    **always** keeps error traces (and any trace slower than ``keep_slow_ms``),
    and keeps successful traces at ``sample_rate``. Sampling is *deterministic*
    on the trace id (a hash, not a coin flip), so a decision is reproducible and
    a parent and its children sample identically — never a torn trace. Set
    ``sample_rate=1.0`` to keep everything (the default is to thin the head).
    """

    def __init__(
        self,
        inner: TraceExporter,
        *,
        sample_rate: float = 0.1,
        keep_errors: bool = True,
        keep_slow_ms: int | None = None,
    ) -> None:
        self.inner = inner
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.keep_errors = keep_errors
        self.keep_slow_ms = keep_slow_ms
        self.kept = 0
        self.dropped = 0

    def _keep(self, trace: Trace) -> bool:
        if self.keep_errors and trace.status == "error":
            return True
        if self.keep_slow_ms is not None and trace.duration_ms >= self.keep_slow_ms:
            return True
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        bucket = int(hashlib.sha256(trace.id.encode()).hexdigest(), 16) % 10_000
        return bucket < int(self.sample_rate * 10_000)

    def export(self, trace: Trace) -> None:
        if self._keep(trace):
            self.kept += 1
            self.inner.export(trace)
        else:
            self.dropped += 1


# ---------------------------------------------------------------------------
# Alerting sinks (2.1)
# ---------------------------------------------------------------------------


class Alert(BaseModel):
    """A fired alert from the observability rule engine."""

    id: str = Field(default_factory=lambda: new_id("alert"))
    rule: str
    severity: Literal["info", "warning", "critical"] = "warning"
    message: str = ""
    value: float = 0.0
    threshold: float = 0.0
    dimension: str = ""
    key: str = ""
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    def to_payload(self) -> dict[str, Any]:
        return to_jsonable(self.model_dump(mode="json"))


class AlertSink(Protocol):
    def send(self, alert: Alert) -> None:  # pragma: no cover - protocol
        ...


class MemoryAlertSink:
    """Collects alerts in memory — for tests and in-process inspection."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.alerts.append(alert)


def _post_json(url: str, payload: dict[str, Any], *, client: Any | None, timeout_s: float) -> None:
    """POST JSON best-effort; alert delivery must never break a run."""
    try:
        if client is not None:
            client.post(url, json=payload, timeout=timeout_s)
            return
        import httpx

        with httpx.Client(timeout=timeout_s) as http:
            http.post(url, json=payload)
    except Exception:  # noqa: BLE001 - delivery failures are logged, not raised
        logger.warning("alert delivery to %s failed", url, exc_info=True)


class WebhookAlertSink:
    """Delivers alerts as a JSON POST to any webhook endpoint."""

    def __init__(self, url: str, *, client: Any | None = None, timeout_s: float = 5.0) -> None:
        self.url = url
        self._client = client
        self.timeout_s = timeout_s

    def _payload(self, alert: Alert) -> dict[str, Any]:
        return alert.to_payload()

    def send(self, alert: Alert) -> None:
        _post_json(self.url, self._payload(alert), client=self._client, timeout_s=self.timeout_s)


class SlackAlertSink(WebhookAlertSink):
    """Posts a formatted message to a Slack incoming webhook."""

    _EMOJI = {"info": ":information_source:", "warning": ":warning:", "critical": ":rotating_light:"}

    def _payload(self, alert: Alert) -> dict[str, Any]:
        emoji = self._EMOJI.get(alert.severity, ":warning:")
        scope = f" [{alert.dimension}:{alert.key}]" if alert.key else ""
        text = (
            f"{emoji} *{alert.rule}*{scope} — {alert.message} "
            f"(value={alert.value:g}, threshold={alert.threshold:g})"
        )
        return {"text": text}


class PagerDutyAlertSink(WebhookAlertSink):
    """Triggers a PagerDuty Events API v2 incident."""

    _SEVERITY = {"info": "info", "warning": "warning", "critical": "critical"}

    def __init__(
        self,
        routing_key: str,
        *,
        url: str = "https://events.pagerduty.com/v2/enqueue",
        client: Any | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        super().__init__(url, client=client, timeout_s=timeout_s)
        self.routing_key = routing_key

    def _payload(self, alert: Alert) -> dict[str, Any]:
        return {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "dedup_key": f"{alert.rule}:{alert.dimension}:{alert.key}",
            "payload": {
                "summary": f"{alert.rule}: {alert.message}",
                "severity": self._SEVERITY.get(alert.severity, "warning"),
                "source": "vincio",
                "custom_details": alert.to_payload(),
            },
        }


class PrometheusExporter:
    """Scrape-friendly Prometheus metrics for the served plane (2.1).

    Prometheus pulls, so this is a registry the ``/metrics`` endpoint renders in
    the text exposition format — and an :class:`AlertSink` that increments a
    fired-alert counter labelled by rule/severity. Observe traces with
    :meth:`observe_trace` to track run counts, errors, and a cost total.
    """

    def __init__(self, *, namespace: str = "vincio") -> None:
        self.namespace = namespace
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._lock = threading.Lock()

    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        """Increment a counter metric by ``value`` (default 1)."""
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._counters[key] += value

    _inc = inc  # internal alias

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._gauges[key] = value

    def observe_trace(self, trace: Trace) -> None:
        self._inc("runs_total", {"status": trace.status})
        if trace.status == "error":
            self._inc("run_errors_total")

    def export(self, trace: Trace) -> None:  # TraceExporter compatible
        self.observe_trace(trace)

    def send(self, alert: Alert) -> None:  # AlertSink compatible
        self._inc("alerts_total", {"rule": alert.rule, "severity": alert.severity})

    def render(self) -> str:
        """Render the Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
        for (name, labels), value in sorted(counters.items()):
            lines.append(f"# TYPE {self.namespace}_{name} counter")
            lines.append(f"{self.namespace}_{name}{_render_labels(labels)} {value:g}")
        for (name, labels), value in sorted(gauges.items()):
            lines.append(f"# TYPE {self.namespace}_{name} gauge")
            lines.append(f"{self.namespace}_{name}{_render_labels(labels)} {value:g}")
        return "\n".join(lines) + ("\n" if lines else "")


def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in labels)
    return "{" + inner + "}"
