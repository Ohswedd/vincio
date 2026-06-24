"""Real behavioural coverage for vincio.server.app FastAPI routes.

Every test drives the real FastAPI application through the in-process
TestClient against ContextApps backed by the deterministic MockProvider —
no network, no API key, no mocking. Assertions check exact status codes,
exact JSON payloads, tenant-scoping invariants, auth accept/reject, SSE
event framing, and error responses on the uncovered branches.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from vincio import ContextApp  # noqa: E402
from vincio.core.config import VincioConfig  # noqa: E402
from vincio.providers import MockProvider  # noqa: E402
from vincio.server import create_app  # noqa: E402
from vincio.server.auth import issue_jwt  # noqa: E402


def _config(tmp_path, **server_overrides) -> VincioConfig:
    config = VincioConfig()
    # keep everything offline + in-process
    config.storage.metadata = f"sqlite:///{tmp_path}/vincio.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    for key, value in server_overrides.items():
        setattr(config.server, key, value)
    return config


def _app(tmp_path, *, name="demo", with_memory=False) -> ContextApp:
    config = _config(tmp_path)
    app = ContextApp(name=name, provider=MockProvider(default_text="hello world"), config=config)
    if with_memory:
        app.add_memory()
    return app


def _api(tmp_path, *, apps=None, **server_overrides):
    config = _config(tmp_path, **server_overrides)
    return create_app(config, apps=apps or {})


# ---------------------------------------------------------------------------
# Health / readiness / metrics
# ---------------------------------------------------------------------------


class TestHealthAndReadiness:
    def test_health_lists_sorted_apps(self, tmp_path):
        apps = {"zeta": _app(tmp_path, name="zeta"), "alpha": _app(tmp_path, name="alpha")}
        with TestClient(_api(tmp_path, apps=apps)) as client:
            body = client.get("/v1/health").json()
        assert body == {"status": "ok", "apps": ["alpha", "zeta"]}

    def test_ready_503_when_no_apps_registered(self, tmp_path):
        # readiness requires at least one registered app even after startup
        with TestClient(_api(tmp_path, apps={})) as client:
            resp = client.get("/v1/health/ready")
        assert resp.status_code == 503
        assert resp.json() == {"status": "not_ready"}

    def test_ready_200_with_app(self, tmp_path):
        apps = {"demo": _app(tmp_path)}
        with TestClient(_api(tmp_path, apps=apps)) as client:
            resp = client.get("/v1/health/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ready", "apps": ["demo"]}

    def test_metrics_reports_app_gauge(self, tmp_path):
        apps = {"demo": _app(tmp_path)}
        with TestClient(_api(tmp_path, apps=apps)) as client:
            client.get("/v1/health")  # generate a counted request
            resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert "vincio_apps 1" in resp.text
        assert "vincio_requests_total" in resp.text


# ---------------------------------------------------------------------------
# Authentication: accept / reject across api-key and JWT
# ---------------------------------------------------------------------------


class TestAuthentication:
    def test_missing_credentials_rejected_401(self, tmp_path):
        apps = {"demo": _app(tmp_path)}
        with TestClient(_api(tmp_path, apps=apps, api_keys=["sekret"])) as client:
            resp = client.post("/v1/apps/demo/run", json={"input": "hi"})
        assert resp.status_code == 401
        assert "credentials" in resp.json()["detail"]

    def test_valid_api_key_via_header_accepted(self, tmp_path):
        apps = {"demo": _app(tmp_path)}
        with TestClient(_api(tmp_path, apps=apps, api_keys=["sekret"])) as client:
            resp = client.post(
                "/v1/apps/demo/run",
                json={"input": "hi"},
                headers={"x-api-key": "sekret"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "succeeded"

    def test_wrong_api_key_rejected_401(self, tmp_path):
        apps = {"demo": _app(tmp_path)}
        with TestClient(_api(tmp_path, apps=apps, api_keys=["sekret"])) as client:
            resp = client.post(
                "/v1/apps/demo/run",
                json={"input": "hi"},
                headers={"x-api-key": "wrong"},
            )
        assert resp.status_code == 401

    def test_jwt_bearer_token_scopes_tenant(self, tmp_path):
        token = issue_jwt("topsecret", subject="u-1", tenant_id="acme")
        apps = {"demo": _app(tmp_path)}
        api = _api(tmp_path, apps=apps, jwt_secret="topsecret")
        with TestClient(api) as client:
            resp = client.post(
                "/v1/apps/demo/run",
                json={"input": "hi"},
                headers={"authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200

    def test_tampered_jwt_rejected_401(self, tmp_path):
        token = issue_jwt("topsecret", subject="u-1")
        apps = {"demo": _app(tmp_path)}
        api = _api(tmp_path, apps=apps, jwt_secret="topsecret")
        with TestClient(api) as client:
            resp = client.post(
                "/v1/apps/demo/run",
                json={"input": "hi"},
                headers={"authorization": f"Bearer {token}x"},
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Run route: success, tenant scoping, unknown app, evidence shaping
# ---------------------------------------------------------------------------


class TestRunApp:
    def test_run_unknown_app_404(self, tmp_path):
        with TestClient(_api(tmp_path, apps={"demo": _app(tmp_path)})) as client:
            resp = client.post("/v1/apps/ghost/run", json={"input": "hi"})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "unknown app 'ghost'"

    def test_run_returns_text_and_evidence_list(self, tmp_path):
        with TestClient(_api(tmp_path, apps={"demo": _app(tmp_path)})) as client:
            resp = client.post("/v1/apps/demo/run", json={"input": "summarize"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "succeeded"
        assert isinstance(body["evidence"], list)  # evidence reshaped to id/source_id/...

    def test_tenant_scoped_jwt_rejects_conflicting_body_tenant(self, tmp_path):
        token = issue_jwt("s", subject="u", tenant_id="acme")
        api = _api(tmp_path, apps={"demo": _app(tmp_path)}, jwt_secret="s")
        with TestClient(api) as client:
            resp = client.post(
                "/v1/apps/demo/run",
                json={"input": "hi", "tenant_id": "other"},
                headers={"authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "token is scoped to another tenant"

    def test_tenant_scoped_jwt_allows_matching_body_tenant(self, tmp_path):
        token = issue_jwt("s", subject="u", tenant_id="acme")
        api = _api(tmp_path, apps={"demo": _app(tmp_path)}, jwt_secret="s")
        with TestClient(api) as client:
            resp = client.post(
                "/v1/apps/demo/run",
                json={"input": "hi", "tenant_id": "acme"},
                headers={"authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Streaming (SSE) routes
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_stream_emits_sse_events_with_done(self, tmp_path):
        with TestClient(_api(tmp_path, apps={"demo": _app(tmp_path)})) as client:
            resp = client.post("/v1/apps/demo/stream", json={"input": "go"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        frames = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
        assert frames, "expected at least one SSE data frame"
        types = [json.loads(f[len("data: "):])["type"] for f in frames]
        assert "done" in types
        # the done frame carries the final result, not the raw evidence
        done = next(json.loads(f[len("data: "):]) for f in frames if json.loads(f[len("data: "):])["type"] == "done")
        assert "result" in done

    def test_stream_unknown_app_404(self, tmp_path):
        with TestClient(_api(tmp_path, apps={"demo": _app(tmp_path)})) as client:
            resp = client.post("/v1/apps/ghost/stream", json={"input": "go"})
        assert resp.status_code == 404

    def test_agui_emits_sse(self, tmp_path):
        with TestClient(_api(tmp_path, apps={"demo": _app(tmp_path)})) as client:
            resp = client.post("/v1/apps/demo/agui", json={"input": "go"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "data:" in resp.text


# ---------------------------------------------------------------------------
# Runs / traces lookups + tenant isolation
# ---------------------------------------------------------------------------


class TestRunsAndTraces:
    def test_get_run_not_found_404(self, tmp_path):
        with TestClient(_api(tmp_path, apps={"demo": _app(tmp_path)})) as client:
            resp = client.get("/v1/runs/does-not-exist")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "run 'does-not-exist' not found"

    def test_get_run_returns_persisted_record(self, tmp_path):
        app = _app(tmp_path)
        app.store.save("runs", {"id": "r-1", "app_id": "demo", "tenant_id": None, "status": "ok"})
        with TestClient(_api(tmp_path, apps={"demo": app})) as client:
            resp = client.get("/v1/runs/r-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "r-1"

    def test_get_run_cross_tenant_forbidden_403(self, tmp_path):
        app = _app(tmp_path)
        app.store.save("runs", {"id": "r-2", "tenant_id": "acme", "status": "ok"})
        token = issue_jwt("s", subject="u", tenant_id="other")
        api = _api(tmp_path, apps={"demo": app}, jwt_secret="s")
        with TestClient(api) as client:
            resp = client.get("/v1/runs/r-2", headers={"authorization": f"Bearer {token}"})
        assert resp.status_code == 403
        assert resp.json()["detail"] == "run belongs to another tenant"

    def test_get_trace_not_found_404(self, tmp_path):
        with TestClient(_api(tmp_path, apps={"demo": _app(tmp_path)})) as client:
            resp = client.get("/v1/traces/missing")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "trace 'missing' not found"

    def test_get_trace_returns_recorded_trace(self, tmp_path):
        app = _app(tmp_path)
        from vincio.observability.spans import Trace

        trace = Trace(id="t-1", name="run", tenant_id=None)
        app.tracer.exporter.export(trace)
        with TestClient(_api(tmp_path, apps={"demo": app})) as client:
            resp = client.get("/v1/traces/t-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "t-1"

    def test_get_trace_cross_tenant_forbidden_403(self, tmp_path):
        app = _app(tmp_path)
        from vincio.observability.spans import Trace

        app.tracer.exporter.export(Trace(id="t-2", name="run", tenant_id="acme"))
        token = issue_jwt("s", subject="u", tenant_id="other")
        api = _api(tmp_path, apps={"demo": app}, jwt_secret="s")
        with TestClient(api) as client:
            resp = client.get("/v1/traces/t-2", headers={"authorization": f"Bearer {token}"})
        assert resp.status_code == 403
        assert resp.json()["detail"] == "trace belongs to another tenant"


# ---------------------------------------------------------------------------
# Document indexing
# ---------------------------------------------------------------------------


class TestIndexing:
    def test_add_documents_returns_count(self, tmp_path):
        with TestClient(_api(tmp_path, apps={"demo": _app(tmp_path)})) as client:
            resp = client.post(
                "/v1/indexes/demo/documents",
                json=[{"text": "alpha"}, {"text": "beta", "title": "B"}],
            )
        assert resp.status_code == 200
        assert resp.json() == {"indexed": 2}

    def test_add_documents_unknown_index_404(self, tmp_path):
        with TestClient(_api(tmp_path, apps={"demo": _app(tmp_path)})) as client:
            resp = client.post("/v1/indexes/ghost/documents", json=[{"text": "x"}])
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Memory routes — both the "memory disabled" 400 branch and live behaviour
# ---------------------------------------------------------------------------


class TestMemoryDisabled:
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("get", "/v1/memory/search?app_id=demo&q=x", None),
            ("post", "/v1/memory/write?app_id=demo", {"content": "fact"}),
            ("get", "/v1/memory/export?app_id=demo", None),
            ("get", "/v1/memory/stats?app_id=demo", None),
            ("delete", "/v1/memory/m1?app_id=demo", None),
        ],
    )
    def test_memory_route_400_when_disabled(self, tmp_path, method, path, body):
        # app built without add_memory() -> app.memory is None -> 400 each route
        with TestClient(_api(tmp_path, apps={"demo": _app(tmp_path)})) as client:
            resp = getattr(client, method)(path, **({"json": body} if body else {}))
        assert resp.status_code == 400
        assert resp.json()["detail"] == "memory is not enabled for this app"

    def test_memory_consolidate_400_when_disabled(self, tmp_path):
        with TestClient(_api(tmp_path, apps={"demo": _app(tmp_path)})) as client:
            resp = client.post(
                "/v1/memory/consolidate?app_id=demo",
                json={"session_id": "s1"},
            )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "memory is not enabled for this app"


class TestMemoryEnabled:
    def test_write_then_search_then_stats(self, tmp_path):
        app = _app(tmp_path, with_memory=True)
        with TestClient(_api(tmp_path, apps={"demo": app})) as client:
            written = client.post(
                "/v1/memory/write?app_id=demo",
                json={"content": "Alice prefers dark mode", "owner_id": "u1", "confidence": 0.9},
            )
            assert written.status_code == 200
            assert written.json()["status"] in {"active", "validated", "candidate"}

            found = client.get("/v1/memory/search?app_id=demo&q=dark+mode&user_id=u1")
            assert found.status_code == 200
            assert any("dark mode" in r["content"] for r in found.json())

            stats = client.get("/v1/memory/stats?app_id=demo")
            assert stats.status_code == 200
            assert stats.json()["total"] >= 1

    def test_export_returns_owner_records(self, tmp_path):
        app = _app(tmp_path, with_memory=True)
        app.memory.write_fact("owned note", owner_id="u1", confidence=0.9)
        app.memory.write_fact("other note", owner_id="u2", confidence=0.9)
        with TestClient(_api(tmp_path, apps={"demo": app})) as client:
            resp = client.get("/v1/memory/export?app_id=demo&owner_id=u1")
        assert resp.status_code == 200
        contents = [r["content"] for r in resp.json()]
        assert "owned note" in contents
        assert "other note" not in contents  # scoped strictly to the owner

    def test_export_falls_back_to_auth_subject(self, tmp_path):
        # no owner_id query param -> owner resolves to the auth subject
        # ("anonymous" with no auth configured), which is non-None so no 422.
        app = _app(tmp_path, with_memory=True)
        app.memory.write_fact("anon note", owner_id="anonymous", confidence=0.9)
        with TestClient(_api(tmp_path, apps={"demo": app})) as client:
            resp = client.get("/v1/memory/export?app_id=demo")
        assert resp.status_code == 200
        assert any(r["content"] == "anon note" for r in resp.json())

    def test_forget_existing_and_missing(self, tmp_path):
        app = _app(tmp_path, with_memory=True)
        item = app.memory.write_fact("temporary", owner_id="u1", confidence=0.9)
        with TestClient(_api(tmp_path, apps={"demo": app})) as client:
            ok = client.delete(f"/v1/memory/{item.id}?app_id=demo")
            assert ok.status_code == 200
            assert ok.json() == {"id": item.id, "status": "deleted"}

            again = client.delete(f"/v1/memory/{item.id}?app_id=demo")
            assert again.status_code == 404
            assert again.json()["detail"] == f"memory not found: {item.id}"

    def test_consolidate_returns_promoted_ids(self, tmp_path):
        app = _app(tmp_path, with_memory=True)
        with TestClient(_api(tmp_path, apps={"demo": app})) as client:
            resp = client.post(
                "/v1/memory/consolidate?app_id=demo",
                json={"session_id": "s1", "user_id": "u1"},
            )
        assert resp.status_code == 200
        assert "promoted_ids" in resp.json()
        assert "items" not in resp.json()


# ---------------------------------------------------------------------------
# Evals route
# ---------------------------------------------------------------------------


class TestEvals:
    def test_eval_run_executes_dataset(self, tmp_path):
        dataset = tmp_path / "ds.jsonl"
        dataset.write_text(
            json.dumps({"input": "q1", "expected": "hello world"}) + "\n",
            encoding="utf-8",
        )
        app = _app(tmp_path)
        with TestClient(_api(tmp_path, apps={"demo": app})) as client:
            resp = client.post(
                "/v1/evals/run",
                json={"app_id": "demo", "dataset_path": str(dataset), "concurrency": 1},
            )
        assert resp.status_code == 200
        # the report dumps to JSON listing the case(s) it actually ran
        assert len(resp.json()["cases"]) == 1
        assert resp.json()["cases"][0]["error"] is None

    def test_eval_run_unknown_app_404(self, tmp_path):
        with TestClient(_api(tmp_path, apps={"demo": _app(tmp_path)})) as client:
            resp = client.post(
                "/v1/evals/run",
                json={"app_id": "ghost", "dataset_path": str(tmp_path / "x.jsonl")},
            )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# CORS middleware wiring
# ---------------------------------------------------------------------------


class TestConfigAndRateLimit:
    def test_config_loaded_from_path_string(self, tmp_path):
        # create_app(config=str) -> load_config(path) branch
        cfg_path = tmp_path / "vincio.yaml"
        cfg_path.write_text("server:\n  api_keys: [from-file]\n", encoding="utf-8")
        api = create_app(str(cfg_path), apps={"demo": _app(tmp_path)})
        with TestClient(api) as client:
            denied = client.post("/v1/apps/demo/run", json={"input": "hi"})
            assert denied.status_code == 401  # config from file demands the key
            ok = client.post(
                "/v1/apps/demo/run",
                json={"input": "hi"},
                headers={"x-api-key": "from-file"},
            )
            assert ok.status_code == 200

    def test_inmemory_rate_limiter_blocks_third_request(self, tmp_path):
        # rate_limit_per_min>0 + no redis_url -> InMemoryRateLimiter + middleware
        api = _api(tmp_path, apps={"demo": _app(tmp_path)}, rate_limit_per_min=2)
        with TestClient(api) as client:
            assert client.get("/v1/health").status_code == 200
            assert client.get("/v1/health").status_code == 200
            blocked = client.get("/v1/health")
            assert blocked.status_code == 429
            assert blocked.json()["detail"] == "rate limit exceeded"
            assert int(blocked.headers["Retry-After"]) >= 0


class TestMemoryWriteRejection:
    def test_low_confidence_write_rejected_422(self, tmp_path):
        # confidence below the engine's min_confidence -> MemoryPolicyError -> 422
        app = _app(tmp_path, with_memory=True)
        with TestClient(_api(tmp_path, apps={"demo": app})) as client:
            resp = client.post(
                "/v1/memory/write?app_id=demo",
                json={"content": "weak fact", "owner_id": "u1", "confidence": 0.01},
            )
        assert resp.status_code == 422
        assert "rejected" in resp.json()["detail"]


class TestJsonlTrace:
    def test_trace_loaded_from_jsonl_exporter(self, tmp_path):
        # exporter="jsonl" -> get_trace uses the JSONLExporter.load branch
        config = _config(tmp_path)
        config.observability.exporter = "jsonl"
        config.observability.traces_dir = str(tmp_path / "traces")
        app = ContextApp(name="demo", provider=MockProvider(default_text="ok"), config=config)
        from vincio.observability.spans import Trace

        trace = Trace(id="jt-1", name="run", tenant_id=None)
        app.tracer.exporter.export(trace)
        with TestClient(_api(tmp_path, apps={"demo": app})) as client:
            resp = client.get("/v1/traces/jt-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "jt-1"


class TestCors:
    def test_cors_headers_present_when_configured(self, tmp_path):
        apps = {"demo": _app(tmp_path)}
        api = _api(tmp_path, apps=apps, cors_origins=["https://ui.example.com"])
        with TestClient(api) as client:
            resp = client.get(
                "/v1/health",
                headers={"Origin": "https://ui.example.com"},
            )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "https://ui.example.com"
