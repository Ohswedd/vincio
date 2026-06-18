"""Served observability & alerting plane: indexed trace/cost store with
rollups + retention, tail-based sampling, the alert rule engine + sinks, and the
served viewer app. All offline and deterministic."""

from __future__ import annotations

import json
from datetime import timedelta

from vincio.core.events import EventBus
from vincio.core.utils import utcnow
from vincio.observability import (
    AlertManager,
    AlertRule,
    IndexedTraceStore,
    MemoryAlertSink,
    PrometheusExporter,
    Span,
    TailSamplingExporter,
    Trace,
    ViewerApp,
    WebhookAlertSink,
)


def _trace(
    tid: str, *, status: str = "ok", duration_ms: int = 100, cost: float = 0.001,
    tenant: str | None = "acme", model: str = "gpt-5.2-mini", session: str | None = None,
    minutes_ago: float = 0.0,
) -> Trace:
    start = utcnow() - timedelta(minutes=minutes_ago)
    end = start + timedelta(milliseconds=duration_ms)
    t = Trace(
        id=tid, app_name="app", tenant_id=tenant, session_id=session, status=status,
        start_time=start, end_time=end,
    )
    t.spans.append(
        Span(name="run", type="run", trace_id=tid, attributes={"cost_usd": cost, "model": model})
    )
    return t


# ---------------------------------------------------------------------------
# Indexed store
# ---------------------------------------------------------------------------


class TestIndexedTraceStore:
    def test_record_query_and_get(self, tmp_path):
        store = IndexedTraceStore(tmp_path / "obs.db")
        store.export(_trace("t1", tenant="acme"))
        store.export(_trace("t2", tenant="globex", status="error"))
        assert store.count() == 2
        assert store.get("t1").tenant_id == "acme"
        acme = store.query(tenant_id="acme")
        assert {t.id for t in acme} == {"t1"}
        errors = store.query(status="error")
        assert {t.id for t in errors} == {"t2"}

    def test_percentiles(self, tmp_path):
        store = IndexedTraceStore(tmp_path / "obs.db")
        for i, dur in enumerate([10, 20, 30, 40, 100]):
            store.export(_trace(f"t{i}", duration_ms=dur))
        pct = store.percentiles("latency")
        assert pct.count == 5
        assert pct.p50 == 30
        assert pct.max == 100
        assert pct.p95 >= pct.p50

    def test_cost_by_dimension(self, tmp_path):
        store = IndexedTraceStore(tmp_path / "obs.db")
        store.export(_trace("t1", tenant="acme", cost=0.01))
        store.export(_trace("t2", tenant="acme", cost=0.02))
        store.export(_trace("t3", tenant="globex", cost=0.005))
        rows = store.cost_by_dimension("tenant")
        assert rows[0].key == "acme"
        assert round(rows[0].cost_usd, 6) == 0.03
        assert rows[0].calls == 2

    def test_rollup_preaggregates(self, tmp_path):
        store = IndexedTraceStore(tmp_path / "obs.db")
        store.export(_trace("t1", cost=0.01))
        store.export(_trace("t2", cost=0.02, status="error"))
        series = store.rollup("1h", dimension="global", key="∅")
        assert series and series[-1].calls == 2
        assert round(series[-1].cost_usd, 6) == 0.03
        assert series[-1].errors == 1

    def test_reexport_updates_row_without_double_counting(self, tmp_path):
        store = IndexedTraceStore(tmp_path / "obs.db")
        store.export(_trace("t1", status="running", cost=0.0))
        store.export(_trace("t1", status="ok", cost=0.01))  # same id, updated
        assert store.count() == 1
        series = store.rollup("1h")
        assert series[-1].calls == 1  # folded once

    def test_purge_retention(self, tmp_path):
        store = IndexedTraceStore(tmp_path / "obs.db")
        store.export(_trace("old", minutes_ago=120))
        store.export(_trace("new", minutes_ago=1))
        removed = store.purge(before=utcnow() - timedelta(minutes=60))
        assert removed == 1
        assert store.count() == 1
        assert store.get("new") is not None


# ---------------------------------------------------------------------------
# Tail-based sampling
# ---------------------------------------------------------------------------


class TestTailSampling:
    def test_keeps_errors_drops_sampled_success(self):
        store = IndexedTraceStore(":memory:")
        sampler = TailSamplingExporter(store, sample_rate=0.0, keep_errors=True)
        sampler.export(_trace("err", status="error"))
        sampler.export(_trace("ok1", status="ok"))
        sampler.export(_trace("ok2", status="ok"))
        assert store.count() == 1  # only the error kept
        assert sampler.kept == 1 and sampler.dropped == 2

    def test_keeps_slow_traces(self):
        store = IndexedTraceStore(":memory:")
        sampler = TailSamplingExporter(store, sample_rate=0.0, keep_slow_ms=500)
        sampler.export(_trace("slow", duration_ms=900))
        sampler.export(_trace("fast", duration_ms=10))
        assert store.count() == 1

    def test_sampling_is_deterministic(self):
        a = TailSamplingExporter(IndexedTraceStore(":memory:"), sample_rate=0.5)
        b = TailSamplingExporter(IndexedTraceStore(":memory:"), sample_rate=0.5)
        decisions_a = [a._keep(_trace(f"t{i}")) for i in range(50)]
        decisions_b = [b._keep(_trace(f"t{i}")) for i in range(50)]
        assert decisions_a == decisions_b  # same ids → same decisions


# ---------------------------------------------------------------------------
# Alert rule engine + sinks
# ---------------------------------------------------------------------------


class TestAlertEngine:
    def test_threshold_rule_fires(self):
        sink = MemoryAlertSink()
        mgr = AlertManager(sinks=[sink])
        mgr.add_rule(AlertRule(name="hot", metric="latency", kind="threshold", threshold=500))
        assert mgr.observe("latency", 200) == []
        fired = mgr.observe("latency", 900)
        assert len(fired) == 1 and fired[0].rule == "hot"
        assert len(sink.alerts) == 1

    def test_ewma_anomaly_rule(self):
        mgr = AlertManager()
        sink = mgr.add_sink(MemoryAlertSink())
        mgr.add_rule(AlertRule(name="spike", metric="cost", kind="ewma", factor=3.0, min_samples=5))
        for _ in range(8):
            mgr.observe("cost", 0.001)  # stable baseline
        fired = mgr.observe("cost", 0.5)  # large spike
        assert fired and "σ" in fired[0].message
        assert len(sink.alerts) == 1

    def test_burn_rate_rule(self):
        mgr = AlertManager()
        sink = mgr.add_sink(MemoryAlertSink())
        # SLO 99% -> budget 1%; error_rate 0.2 -> burn 20x >= 14.4x fast-burn page.
        mgr.add_rule(
            AlertRule(name="burn", metric="error_rate", kind="burn_rate", threshold=14.4, slo_target=0.99)
        )
        assert mgr.observe("error_rate", 0.05) == []  # burn 5x, below
        fired = mgr.observe("error_rate", 0.2)
        assert fired and fired[0].value >= 14.4
        assert sink.alerts

    def test_subscribe_translates_cost_events(self):
        bus = EventBus()
        sink = MemoryAlertSink()
        mgr = AlertManager(sinks=[sink])
        mgr.subscribe(bus)
        bus.emit("cost.anomaly", {"scope": "tenant:acme", "cost_usd": 5.0, "mean_usd": 0.1, "factor": 50})
        bus.emit("cost.budget_exceeded", {"scope": "tenant:acme", "reason": "daily cap reached"})
        assert {a.rule for a in sink.alerts} == {"cost.anomaly", "cost.budget_exceeded"}
        assert any(a.severity == "critical" for a in sink.alerts)

    def test_check_store_evaluates_percentiles(self, tmp_path):
        store = IndexedTraceStore(tmp_path / "obs.db")
        for i in range(5):
            store.export(_trace(f"e{i}", status="error", duration_ms=2000))
        mgr = AlertManager(sinks=[MemoryAlertSink()])
        mgr.add_rule(AlertRule(name="slow", metric="latency", kind="threshold", threshold=1000))
        mgr.add_rule(AlertRule(name="errs", metric="error_rate", kind="threshold", threshold=0.5))
        fired = mgr.check_store(store)
        assert {a.rule for a in fired} == {"slow", "errs"}

    def test_sink_failures_do_not_raise(self):
        class _Boom:
            def send(self, alert):
                raise RuntimeError("down")

        mgr = AlertManager(sinks=[_Boom(), MemoryAlertSink()])
        mgr.add_rule(AlertRule(name="x", metric="value", kind="threshold", threshold=0))
        fired = mgr.observe("value", 1.0)  # must not raise
        assert len(fired) == 1


class TestAlertSinks:
    def test_webhook_posts_payload(self):
        import httpx

        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        mgr = AlertManager(sinks=[WebhookAlertSink("https://hook.example/alert", client=client)])
        mgr.add_rule(AlertRule(name="hot", metric="latency", kind="threshold", threshold=100))
        mgr.observe("latency", 200)
        assert captured["url"] == "https://hook.example/alert"
        assert captured["body"]["rule"] == "hot"

    def test_prometheus_exporter_renders_metrics(self):
        prom = PrometheusExporter()
        prom.observe_trace(_trace("t1", status="ok"))
        prom.observe_trace(_trace("t2", status="error"))
        from vincio.observability import Alert

        prom.send(Alert(rule="hot", severity="critical"))
        text = prom.render()
        assert "vincio_runs_total" in text
        assert 'vincio_alerts_total{rule="hot",severity="critical"} 1' in text


# ---------------------------------------------------------------------------
# Served viewer app
# ---------------------------------------------------------------------------


class TestViewerApp:
    def _store(self, tmp_path):
        store = IndexedTraceStore(tmp_path / "obs.db")
        store.export(_trace("t1", tenant="acme", cost=0.01))
        store.export(_trace("t2", tenant="globex", status="error", cost=0.02))
        return store

    def test_dashboard_renders_html(self, tmp_path):
        app = ViewerApp(self._store(tmp_path))
        status, ctype, body = app.handle("/")
        assert status == 200 and "text/html" in ctype
        assert "Vincio observability" in body and "Live tail" in body

    def test_healthz(self, tmp_path):
        app = ViewerApp(self._store(tmp_path))
        assert app.handle("/healthz") == (200, "text/plain; charset=utf-8", "ok")

    def test_api_stats_json(self, tmp_path):
        app = ViewerApp(self._store(tmp_path))
        status, ctype, body = app.handle("/api/stats")
        assert status == 200 and ctype == "application/json"
        data = json.loads(body)
        assert data["stats"]["traces"] == 2
        assert "latency" in data and "cost_by_tenant" in data

    def test_api_traces_filtered(self, tmp_path):
        app = ViewerApp(self._store(tmp_path))
        _, _, body = app.handle("/api/traces", {"status": "error"})
        rows = json.loads(body)
        assert [r["id"] for r in rows] == ["t2"]

    def test_api_trace_detail_and_404(self, tmp_path):
        app = ViewerApp(self._store(tmp_path))
        status, ctype, body = app.handle("/api/trace", {"id": "t1"})
        assert status == 200 and json.loads(body)["id"] == "t1"
        assert app.handle("/trace", {"id": "missing"})[0] == 404

    def test_unknown_route_404(self, tmp_path):
        app = ViewerApp(self._store(tmp_path))
        assert app.handle("/nope")[0] == 404
