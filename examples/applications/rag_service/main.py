"""HTTP shell for the grounded document-QA microservice.

Thin FastAPI layer over ``core.py``: it owns request/response typing, status
codes, and error mapping, and delegates all Vincio logic to ``core.answer``.
Run it offline with::

    cd examples/applications/rag_service
    uvicorn main:app --reload

The service answers strictly from a small bundled knowledge base and returns the
supporting citations, the per-call cost, and a trace id for every response.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import core
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("rag_service")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Build and index the app once at startup so the first request is fast."""
    core.get_app()
    yield


app = FastAPI(
    title="Vincio Grounded RAG Service",
    description="Grounded document QA with citations, cost, and trace ids.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    """A question to answer against the knowledge base."""

    question: str = Field(..., min_length=1, examples=["what is the refund window?"])


class AskResponse(BaseModel):
    """A grounded answer plus its provenance and accounting."""

    answer: str
    citations: list[str]
    cost_usd: float
    trace_id: str
    groundedness: float | None = None


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe — cheap, no model call."""
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question, grounded in the bundled knowledge base."""
    try:
        result = core.answer(request.question)
    except ValueError as exc:  # empty/invalid question
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # provider / retrieval failure
        # Log the detail server-side; never echo internal exception text to the client.
        log.exception("answer failed")
        raise HTTPException(status_code=502, detail="answer failed") from exc
    return AskResponse(**result)
