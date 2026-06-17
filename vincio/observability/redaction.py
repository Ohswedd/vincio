"""Content-capture controls for telemetry export (2.1).

Prompts and completions are the most sensitive thing a run touches, and the
served observability plane must not widen your data-exposure surface. So raw
prompt/completion content is **off by default** in exports: a
:class:`ContentCapturePolicy` decides, at the export boundary, whether content
attributes ride along at all — and when you opt in, it truncates and
PII-redacts them first, before anything reaches an OTel event, a JSONL file, or
the served viewer.

The default policy (``capture=False``) drops the content-bearing attribute keys
entirely while leaving structural telemetry (model, token counts, latency,
cost, scores) intact. Opt in with ``capture=True`` and content is redacted via
the security PII detector and clipped to ``max_chars``.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

__all__ = ["ContentCapturePolicy"]

# Attribute keys that carry raw prompt/completion/tool content (vs. structural
# metadata like model/tokens/cost which always export).
_CONTENT_KEYS = (
    "input",
    "output",
    "input_full",
    "output_full",
    "prompt",
    "completion",
    "messages",
    "content",
    "raw_text",
    "text",
    "answer",
    "gen_ai.prompt",
    "gen_ai.completion",
)


class ContentCapturePolicy(BaseModel):
    """Gate + redact prompt/completion content at the telemetry export boundary."""

    capture: bool = False
    max_chars: int = 2000
    redact_pii: bool = True
    content_keys: tuple[str, ...] = _CONTENT_KEYS

    def apply(self, value: Any) -> str | None:
        """Return the export-safe form of a content value, or ``None`` to drop it."""
        if not self.capture:
            return None
        text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
        if self.redact_pii:
            text = _redact(text)
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "…[truncated]"
        return text

    def scrub_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """Copy ``attributes`` with content keys gated/redacted/truncated."""
        scrubbed: dict[str, Any] = {}
        keys = set(self.content_keys)
        for key, value in attributes.items():
            if key in keys:
                result = self.apply(value)
                if result is not None:
                    scrubbed[key] = result
                # else: dropped — content capture is off or value vanished
            else:
                scrubbed[key] = value
        return scrubbed


def _redact(text: str) -> str:
    try:
        from ..security.pii import redact

        return redact(text)
    except Exception:  # noqa: BLE001 - redaction must never break export
        return text
