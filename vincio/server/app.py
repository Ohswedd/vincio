"""FastAPI server.

Endpoints::

    POST /v1/apps/{app_id}/run
    POST /v1/apps/{app_id}/stream
    POST /v1/evals/run
    GET  /v1/runs/{run_id}
    GET  /v1/traces/{trace_id}
    POST /v1/indexes/{index_id}/documents
    GET  /v1/memory/search
    POST /v1/memory/write

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

__all__ = ["create_app", "RunRequest", "MemoryWriteRequest", "EvalRunRequest"]


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
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse
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

    api = FastAPI(title="Vincio", version="0.1.0", description="Context engineering platform API")

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
            # Full pipeline run; final result streamed as SSE events so
            # clients get a single consistent protocol.
            result = await app.arun(
                request.input,
                files=request.files or None,
                tenant_id=tenant_id,
                user_id=request.user_id or context.subject,
                session_id=request.session_id,
                config=RunConfig(model=request.model, temperature=request.temperature, stream=True),
            )
            text = result.raw_text or (result.output if isinstance(result.output, str) else "")
            for start in range(0, len(text), 256):
                yield f"data: {json.dumps({'type': 'text_delta', 'text': text[start:start+256]})}\n\n"
            final = result.model_dump(mode="json", exclude={"evidence", "raw_text"})
            yield f"data: {json.dumps({'type': 'done', 'result': final}, default=str)}\n\n"

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

    api.state.registry = registry
    api.state.config = config
    return api
