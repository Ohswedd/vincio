"""Pluggable detector backends.

The PII, injection, and secret detectors are deterministic by default. A
:class:`DetectorBackend` lets a caller plug an ML model (NER, a classifier,
a hosted detector) *alongside* the deterministic rules — never replacing them,
so detection with no backend configured is byte-for-byte unchanged.

A backend returns :class:`DetectorSpan` findings; each detector merges them with
its own findings using its existing overlap/blend rules.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

__all__ = ["DetectorSpan", "DetectorBackend"]


class DetectorSpan(BaseModel):
    """A backend finding: a labeled span over the input text."""

    start: int
    end: int
    label: str  # e.g. "email", "api_key", "injection"
    score: float = 1.0  # confidence / risk in [0, 1]
    text: str = ""


@runtime_checkable
class DetectorBackend(Protocol):
    """A model-based detector. ``detect`` is synchronous and side-effect-free."""

    def detect(self, text: str) -> list[DetectorSpan]:  # pragma: no cover - protocol
        ...
