"""Small shared utilities: id generation, stable hashing, time, JSON helpers."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "new_id",
    "stable_hash",
    "utcnow",
    "to_jsonable",
    "json_dumps",
    "slugify",
]


def new_id(prefix: str) -> str:
    """Generate a collision-resistant id like ``run_3f2a...``."""
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_jsonable(value: Any) -> Any:
    """Convert arbitrary values (pydantic models, datetimes, sets...) to JSON-safe data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(to_jsonable(v) for v in value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, BaseException):
        return {"error": type(value).__name__, "message": str(value)}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return to_jsonable(dump(mode="json"))
    return str(value)


def json_dumps(value: Any, *, indent: int | None = None, sort_keys: bool = False) -> str:
    return json.dumps(to_jsonable(value), indent=indent, sort_keys=sort_keys, ensure_ascii=False)


def stable_hash(value: Any, *, length: int = 16) -> str:
    """Deterministic content hash for prompts, packets, cache keys, versions."""
    payload = json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-") or "item"
