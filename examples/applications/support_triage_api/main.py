"""HTTP shell for the support-triage engine.

This is the *only* file that imports FastAPI. It is a thin adapter: every
endpoint validates its request with a Pydantic model and delegates to a plain
function in ``core.py`` (which has no web-framework dependency and runs offline).

Run it::

    uvicorn main:app --reload

Then::

    curl -s localhost:8000/health
    curl -s localhost:8000/triage \
      -H 'content-type: application/json' \
      -d '{"ticket": "I was double charged", "user_id": "u1"}'
"""

from __future__ import annotations

import logging

import core
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("support_triage")

app = FastAPI(
    title="Vincio Support Triage API",
    description="Classify support tickets into a typed verdict with approval-gated escalation.",
    version="1.0.0",
)


# --- request / response contracts -----------------------------------------
class TriageRequest(BaseModel):
    ticket: str = Field(..., description="raw support ticket text", min_length=1)
    user_id: str = Field(..., description="stable id of the reporting user", min_length=1)


class PendingApproval(BaseModel):
    tool: str
    reason: str
    arguments: dict
    requires_human_approval: bool


class TriageResponse(BaseModel):
    category: str
    priority: str
    summary: str
    pending_approvals: list[PendingApproval] = []
    prior_tickets: int = 0
    trace_id: str
    cost_usd: float


class Health(BaseModel):
    status: str


# --- endpoints -------------------------------------------------------------
@app.get("/health", response_model=Health)
def health() -> Health:
    """Liveness probe."""
    return Health(status="ok")


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    """Triage one ticket: typed verdict + any pending (approval-gated) escalations."""
    try:
        result = core.triage(req.ticket, req.user_id)
    except ValueError as exc:  # bad input from the caller
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # provider / model / runtime failure
        log.exception("triage failed")  # detail stays server-side
        raise HTTPException(status_code=502, detail="triage failed") from exc
    return TriageResponse(**result)
