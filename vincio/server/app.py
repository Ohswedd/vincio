"""FastAPI server.

Endpoints::

    POST /v1/apps/{app_id}/run
    POST /v1/apps/{app_id}/stream
    POST /v1/apps/{app_id}/agui
    POST /v1/evals/run
    GET  /v1/runs/{run_id}
    GET  /v1/traces/{trace_id}
    POST /v1/indexes/{index_id}/documents
    GET  /v1/memory/search
    POST /v1/memory/write
    POST /v1/memory/consolidate
    GET  /v1/memory/export
    GET  /v1/memory/stats
    DELETE /v1/memory/{memory_id}

Usage::

    from vincio.server import create_app
    api = create_app(config="vincio.yaml", apps={"contract_review": app})
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from ..core.app import ContextApp
from ..core.config import VincioConfig, load_config
from ..core.errors import (
    AccessDeniedError,
    AuthenticationError,
    ConfigError,
    VincioError,
)
from ..core.types import RunConfig
from ..evals.runners import EvalRunner
from ..observability.exporters import JSONLExporter
from .auth import AuthContext, Authenticator

__all__ = ["create_app", "RunRequest", "MemoryWriteRequest", "MemoryConsolidateRequest", "EvalRunRequest"]


class RunRequest(BaseModel):
    input: str
    files: list[str] = Field(default_factory=list)
    tenant_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    model: str | None = None
    temperature: float | None = None


class MemoryWriteRequest(BaseModel):
    content: str
    scope: str = "user"
    owner_id: str | None = None
    type: str = "fact"
    confidence: float = 0.8


class MemoryConsolidateRequest(BaseModel):
    session_id: str
    user_id: str | None = None


class EvalRunRequest(BaseModel):
    app_id: str
    dataset_path: str
    metrics: list[str] | None = None
    concurrency: int = 8
    gates: dict[str, str] | None = None


class DocumentUpload(BaseModel):
    text: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def create_app(
    config: VincioConfig | str | None = None,
    *,
    apps: dict[str, ContextApp] | None = None,
):
    """Build the FastAPI application serving the given ContextApps."""
    try:
        from contextlib import asynccontextmanager

        from fastapi import Depends, FastAPI, Header, HTTPException, Request
        from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    except ImportError as exc:
        raise ConfigError(
            'server mode requires: pip install "vincio[server]"'
        ) from exc

    if isinstance(config, str):
        config = load_config(config)
    config = config or load_config()
    registry: dict[str, ContextApp] = dict(apps or {})
    authenticator = Authenticator(
        api_keys=config.server.api_keys, jwt_secret=config.server.jwt_secret
    )

    from .. import __version__
    from ..observability.exporters import PrometheusExporter

    metrics = PrometheusExporter()

    # 2.1: shared rate-limit state — Redis-backed when configured (coherent
    # across uvicorn workers), process-local otherwise.
    from ..storage.shared_state import RateLimiter

    rate_limiter: RateLimiter | None = None
    rate_limit = int(config.server.rate_limit_per_min or 0)
    if rate_limit > 0:
        if config.server.redis_url:
            from ..storage.redis import RedisRateLimiter

            rate_limiter = RedisRateLimiter(config.server.redis_url)
        else:
            from ..storage.shared_state import InMemoryRateLimiter

            rate_limiter = InMemoryRateLimiter()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 2.1: graceful startup/shutdown. Readiness flips true after startup and
        # false on shutdown so load balancers drain in flight before exit.
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False

    api = FastAPI(
        title="Vincio",
        version=__version__,
        description="Context engineering platform API",
        lifespan=lifespan,
    )

    if rate_limiter is not None:

        @api.middleware("http")
        async def _rate_limit(request: Request, call_next):
            caller = (
                request.headers.get("x-api-key")
                or request.headers.get("authorization")
                or (request.client.host if request.client else "anonymous")
            )
            decision = rate_limiter.check(caller, limit=rate_limit, window_s=60.0)
            metrics.set_gauge("rate_limit_remaining", decision.remaining)
            if not decision.allowed:
                metrics.inc("rate_limited_total")
                return JSONResponse(
                    status_code=429,
                    content={"detail": "rate limit exceeded"},
                    headers={"Retry-After": str(int(decision.retry_after_s))},
                )
            return await call_next(request)

    @api.middleware("http")
    async def _count_requests(request: Request, call_next):
        response = await call_next(request)
        metrics.inc("requests_total", {"status": str(response.status_code)})
        return response

    if config.server.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        api.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def auth(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> AuthContext:
        try:
            return authenticator.authenticate(authorization, x_api_key)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=exc.message) from exc

    def get_app(app_id: str) -> ContextApp:
        if app_id not in registry:
            raise HTTPException(status_code=404, detail=f"unknown app {app_id!r}")
        return registry[app_id]

    def scope_tenant(request_tenant: str | None, context: AuthContext) -> str | None:
        # Tenant-scoped tokens override the request body.
        if context.tenant_id is not None:
            if request_tenant is not None and request_tenant != context.tenant_id:
                raise HTTPException(status_code=403, detail="token is scoped to another tenant")
            return context.tenant_id
        return request_tenant

    @api.get("/v1/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "apps": sorted(registry)}

    @api.get("/v1/health/ready")
    def ready():
        # Readiness (vs liveness): 503 until startup completes and at least one
        # app is registered, so a load balancer waits before routing traffic.
        if getattr(api.state, "ready", False) and registry:
            return {"status": "ready", "apps": sorted(registry)}
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    @api.get("/v1/metrics")
    def prometheus_metrics():
        # Scrape-friendly Prometheus exposition of request/rate-limit counters.
        metrics.set_gauge("apps", len(registry))
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @api.post("/v1/apps/{app_id}/run")
    async def run_app(app_id: str, request: RunRequest, context: AuthContext = Depends(auth)):
        app = get_app(app_id)
        tenant_id = scope_tenant(request.tenant_id, context)
        run_config = RunConfig(model=request.model, temperature=request.temperature)
        try:
            result = await app.arun(
                request.input,
                files=request.files or None,
                tenant_id=tenant_id,
                user_id=request.user_id or context.subject,
                session_id=request.session_id,
                config=run_config,
            )
        except AccessDeniedError as exc:
            raise HTTPException(status_code=403, detail=exc.message) from exc
        except VincioError as exc:
            raise HTTPException(status_code=422, detail=exc.message) from exc
        payload = result.model_dump(mode="json", exclude={"evidence"})
        payload["evidence"] = [
            {"id": e.id, "source_id": e.source_id, "text": e.text, "relevance": e.relevance}
            for e in result.evidence
        ]
        return payload

    @api.post("/v1/apps/{app_id}/stream")
    async def stream_app(app_id: str, request: RunRequest, context: AuthContext = Depends(auth)):
        app = get_app(app_id)
        tenant_id = scope_tenant(request.tenant_id, context)

        async def event_stream():
            # End-to-end streaming (0.2): events arrive as the pipeline and
            # the provider produce them — stage markers, real token deltas,
            # partial structured output, tool activity, then the result.
            async for event in app.astream(
                request.input,
                files=request.files or None,
                tenant_id=tenant_id,
                user_id=request.user_id or context.subject,
                session_id=request.session_id,
                config=RunConfig(model=request.model, temperature=request.temperature, stream=True),
            ):
                if event.type == "done" and event.result is not None:
                    final = event.result.model_dump(mode="json", exclude={"evidence", "raw_text"})
                    yield f"data: {json.dumps({'type': 'done', 'result': final}, default=str)}\n\n"
                else:
                    payload = event.model_dump(
                        mode="json", exclude_none=True, exclude={"result"}, exclude_defaults=True
                    )
                    payload["type"] = event.type
                    yield f"data: {json.dumps(payload, default=str)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @api.post("/v1/apps/{app_id}/agui")
    async def agui_app(app_id: str, request: RunRequest, context: AuthContext = Depends(auth)):
        # Generative UI (2.2): the same astream run, translated into AG-UI events
        # so an interactive frontend renders text/tool/state deltas live. The UI
        # inherits the run's provenance, budget metering, and audit — one run.
        app = get_app(app_id)
        tenant_id = scope_tenant(request.tenant_id, context)
        from .agui import run_stream_to_agui

        async def event_stream():
            stream = app.astream(
                request.input,
                files=request.files or None,
                tenant_id=tenant_id,
                user_id=request.user_id or context.subject,
                session_id=request.session_id,
                config=RunConfig(model=request.model, temperature=request.temperature, stream=True),
            )
            async for ui_event in run_stream_to_agui(stream, run_id=request.session_id or None):
                yield ui_event.to_sse()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @api.post("/v1/evals/run")
    async def run_eval(request: EvalRunRequest, context: AuthContext = Depends(auth)):
        app = get_app(request.app_id)
        runner = EvalRunner(
            app, metrics=request.metrics, concurrency=request.concurrency, gates=request.gates
        )
        report = await runner.arun(request.dataset_path)
        return report.model_dump(mode="json")

    @api.get("/v1/runs/{run_id}")
    def get_run(run_id: str, context: AuthContext = Depends(auth)):
        for app in registry.values():
            record = app.store.get("runs", run_id)
            if record is not None:
                if context.tenant_id and record.get("tenant_id") not in (None, context.tenant_id):
                    raise HTTPException(status_code=403, detail="run belongs to another tenant")
                return record
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")

    @api.get("/v1/traces/{trace_id}")
    def get_trace(trace_id: str, context: AuthContext = Depends(auth)):
        for app in registry.values():
            exporter = app.tracer.exporter
            trace = None
            if hasattr(exporter, "get"):
                trace = exporter.get(trace_id)
            elif isinstance(exporter, JSONLExporter):
                trace = exporter.load(trace_id)
            if trace is not None:
                if context.tenant_id and trace.tenant_id not in (None, context.tenant_id):
                    raise HTTPException(status_code=403, detail="trace belongs to another tenant")
                return trace.model_dump(mode="json")
        raise HTTPException(status_code=404, detail=f"trace {trace_id!r} not found")

    @api.post("/v1/indexes/{index_id}/documents")
    async def add_documents(
        index_id: str, documents: list[DocumentUpload], context: AuthContext = Depends(auth)
    ):
        # index_id == app_id with an indexable source space.
        app = get_app(index_id)
        from ..core.types import Document

        docs = [
            Document(text=d.text, title=d.title, metadata=d.metadata, tenant_id=context.tenant_id)
            for d in documents
        ]
        app.add_source(f"api_{index_id}", documents=docs)
        return {"indexed": len(docs)}

    @api.get("/v1/memory/search")
    def memory_search(
        q: str,
        app_id: str,
        user_id: str | None = None,
        top_k: int = 8,
        context: AuthContext = Depends(auth),
    ):
        app = get_app(app_id)
        if app.memory is None:
            raise HTTPException(status_code=400, detail="memory is not enabled for this app")
        results = app.memory.search(
            q, user_id=user_id or context.subject, tenant_id=context.tenant_id, top_k=top_k
        )
        return [
            {"id": r.item.id, "content": r.item.content, "score": r.score, "scope": r.item.scope.value}
            for r in results
        ]

    @api.post("/v1/memory/write")
    def memory_write(request: MemoryWriteRequest, app_id: str, context: AuthContext = Depends(auth)):
        app = get_app(app_id)
        if app.memory is None:
            raise HTTPException(status_code=400, detail="memory is not enabled for this app")
        try:
            item = app.memory.write_fact(
                request.content,
                scope=request.scope,
                owner_id=request.owner_id or context.subject,
                type=request.type,
                confidence=request.confidence,
            )
        except VincioError as exc:
            raise HTTPException(status_code=422, detail=exc.message) from exc
        return {"id": item.id, "status": item.status}

    @api.post("/v1/memory/consolidate")
    async def memory_consolidate(
        request: MemoryConsolidateRequest, app_id: str, context: AuthContext = Depends(auth)
    ):
        app = get_app(app_id)
        if app.memory is None:
            raise HTTPException(status_code=400, detail="memory is not enabled for this app")
        report = await app.memory.consolidate(
            request.session_id, user_id=request.user_id or context.subject
        )
        return report.model_dump(mode="json", exclude={"items"}) | {
            "promoted_ids": [item.id for item in report.items]
        }

    @api.get("/v1/memory/export")
    def memory_export(app_id: str, owner_id: str | None = None, context: AuthContext = Depends(auth)):
        app = get_app(app_id)
        if app.memory is None:
            raise HTTPException(status_code=400, detail="memory is not enabled for this app")
        owner = owner_id or context.subject
        if owner is None:
            raise HTTPException(status_code=422, detail="owner_id is required")
        return app.memory.export_owner_data(owner)

    @api.get("/v1/memory/stats")
    def memory_stats(app_id: str, context: AuthContext = Depends(auth)):
        app = get_app(app_id)
        if app.memory is None:
            raise HTTPException(status_code=400, detail="memory is not enabled for this app")
        return app.memory.stats()

    @api.delete("/v1/memory/{memory_id}")
    def memory_forget(memory_id: str, app_id: str, context: AuthContext = Depends(auth)):
        app = get_app(app_id)
        if app.memory is None:
            raise HTTPException(status_code=400, detail="memory is not enabled for this app")
        if not app.memory.forget(memory_id):
            raise HTTPException(status_code=404, detail=f"memory not found: {memory_id}")
        return {"id": memory_id, "status": "deleted"}

    api.state.registry = registry
    api.state.config = config
    return api
