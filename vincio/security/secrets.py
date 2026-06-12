"""Secret scanning and safe handling (API keys/secrets).

Scans text and structured data for credentials before they reach prompts,
traces, or memory. Also provides ``SecretString`` for keys held in memory —
its repr never leaks the value into logs or traces.
"""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel

from .pii import PIIDetector, PIIMatch

__all__ = ["SecretString", "SecretFinding", "SecretScanner"]


class SecretString:
    """Wrapper that hides a secret from repr/str/traces."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretString('****')"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretString):
            return self._value == other._value
        return False

    def __hash__(self) -> int:
        return hash(self._value)


class SecretFinding(BaseModel):
    path: str  # location: "text", "metadata.api_key", "files[0].name", ...
    kind: str  # api_key | secret | high_entropy
    value_preview: str
    confidence: float


_ENTROPY_CANDIDATE_RE = re.compile(r"\b[A-Za-z0-9+/_=-]{24,}\b")


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for char in value:
        counts[char] = counts.get(char, 0) + 1
    total = len(value)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _preview(value: str) -> str:
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}…{value[-2:]}"


class SecretScanner:
    """Finds credentials in strings and nested structures."""

    def __init__(self, *, entropy_threshold: float = 4.4) -> None:
        self.entropy_threshold = entropy_threshold
        self._pii = PIIDetector(enabled_types={"api_key", "secret"})

    def scan_text(self, text: str, *, path: str = "text") -> list[SecretFinding]:
        findings: list[SecretFinding] = []
        matches: list[PIIMatch] = self._pii.detect(text or "")
        for match in matches:
            findings.append(
                SecretFinding(
                    path=path,
                    kind=match.type,
                    value_preview=_preview(match.value),
                    confidence=match.confidence,
                )
            )
        for candidate in _ENTROPY_CANDIDATE_RE.finditer(text or ""):
            value = candidate.group(0)
            if any(f.value_preview == _preview(value) for f in findings):
                continue
            entropy = _shannon_entropy(value)
            if entropy >= self.entropy_threshold and not value.isdigit():
                findings.append(
                    SecretFinding(
                        path=path,
                        kind="high_entropy",
                        value_preview=_preview(value),
                        confidence=min(0.9, 0.5 + (entropy - self.entropy_threshold) / 4),
                    )
                )
        return findings

    _SECRET_KEY_RE = re.compile(
        r"(?i)\b(?:password|passwd|secret|token|credential|api[_-]?key|private[_-]?key|auth)\b"
    )

    def scan(self, data: Any, *, path: str = "$") -> list[SecretFinding]:
        findings: list[SecretFinding] = []
        if isinstance(data, str):
            findings.extend(self.scan_text(data, path=path))
        elif isinstance(data, dict):
            for key, value in data.items():
                child_path = f"{path}.{key}"
                # A secret-named key with a non-trivial string value is a
                # finding even when the value alone looks innocuous.
                if (
                    isinstance(value, str)
                    and len(value) >= 8
                    and self._SECRET_KEY_RE.search(str(key))
                ):
                    findings.append(
                        SecretFinding(
                            path=child_path,
                            kind="secret",
                            value_preview=_preview(value),
                            confidence=0.85,
                        )
                    )
                findings.extend(self.scan(value, path=child_path))
        elif isinstance(data, (list, tuple)):
            for index, value in enumerate(data):
                findings.extend(self.scan(value, path=f"{path}[{index}]"))
        return findings

    def redact_text(self, text: str) -> str:
        result = text
        for match in sorted(self._pii.detect(text or ""), key=lambda m: m.start, reverse=True):
            result = result[: match.start] + "[REDACTED:secret]" + result[match.end :]
        for candidate in sorted(
            _ENTROPY_CANDIDATE_RE.finditer(result or ""), key=lambda m: m.start(), reverse=True
        ):
            if _shannon_entropy(candidate.group(0)) >= self.entropy_threshold and not candidate.group(0).isdigit():
                result = result[: candidate.start()] + "[REDACTED:secret]" + result[candidate.end() :]
        return result
