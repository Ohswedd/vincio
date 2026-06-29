"""HTTP shell for the structured-extraction service.

A thin FastAPI wrapper around :mod:`core`. All of the Vincio logic lives in
``core.py`` (which has no web dependency at all); this file only handles
request/response shapes, status codes, and error translation.

Run it offline::

    uvicorn main:app --reload

Point it at a real model by exporting ``VINCIO_PROVIDER`` first; the service
code does not change.
"""

from __future__ import annotations

import logging

from core import Invoice, extract
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("extraction_service")

app = FastAPI(
    title="Vincio Invoice Extraction Service",
    description="Turn raw invoice text into a validated, structured invoice.",
    version="1.0.0",
)


class ExtractRequest(BaseModel):
    """Request body for POST /extract."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw invoice text to extract structured fields from.",
        examples=["Invoice from Acme Corp, total 1200.50 USD for widgets and gadgets"],
    )


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe — returns ``{"status": "ok"}`` with no model call."""
    return HealthResponse(status="ok")


@app.post("/extract", response_model=Invoice)
def extract_invoice(req: ExtractRequest) -> Invoice:
    """Extract a structured :class:`Invoice` from the supplied text.

    Returns 422 for empty/invalid input and 502 when extraction fails even
    after the bounded self-correction loop.
    """
    try:
        data = extract(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        log.exception("extraction failed")  # detail stays server-side
        raise HTTPException(status_code=502, detail="extraction failed") from exc
    return Invoice(**data)
