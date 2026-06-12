"""PII detection and redaction.

Deterministic regex + heuristic detectors for: email, phone, person names,
addresses, government IDs, financial data, health data markers, and API
keys/secrets. Detection returns spans with type and confidence so policies
can decide between blocking, redaction, and pass-through.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

__all__ = ["PIIType", "PIIMatch", "PIIDetector", "redact"]

PIIType = Literal[
    "email",
    "phone",
    "person_name",
    "address",
    "government_id",
    "credit_card",
    "iban",
    "health",
    "api_key",
    "secret",
    "ip_address",
]


class PIIMatch(BaseModel):
    type: PIIType
    value: str
    start: int
    end: int
    confidence: float


_PATTERNS: list[tuple[PIIType, re.Pattern[str], float]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), 0.99),
    (
        "phone",
        re.compile(
            r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3}[\s.-]\d{3,4}(?:[\s.-]\d{2,4})?(?!\d)"
        ),
        0.7,
    ),
    (
        "government_id",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN format
        0.95,
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        0.5,  # validated by Luhn below
    ),
    (
        "iban",
        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
        0.85,
    ),
    (
        "ip_address",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
        0.9,
    ),
    (
        "api_key",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{16,}|sk-ant-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|xox[bpas]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35})\b"
        ),
        0.98,
    ),
    (
        "secret",
        re.compile(
            r"(?i)\b(?:password|passwd|secret|token|api[_-]?key|private[_-]?key)\b\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?"
        ),
        0.9,
    ),
    (
        "health",
        re.compile(
            r"(?i)\b(?:diagnos(?:is|ed)|prescri(?:ption|bed)|icd-10|medical record|patient id|blood type|hiv|diabetes|chemotherapy)\b"
        ),
        0.6,
    ),
    (
        "address",
        re.compile(
            r"(?i)\b\d{1,5}\s+[A-Za-z0-9.\s]{2,40}\b(?:street|st\.|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|via|strasse|straße)\b"
        ),
        0.75,
    ),
]

# Names: Honorific + capitalized words, or "FirstName LastName" near name cues.
_NAME_HONORIFIC_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
)
_NAME_CUE_RE = re.compile(
    r"(?i)\b(?:my name is|i am|contact|attn:?|signed,?)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)"
)


def _luhn_valid(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if len(digits) < 13:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


class PIIDetector:
    def __init__(self, *, enabled_types: set[PIIType] | None = None) -> None:
        self.enabled_types = enabled_types

    def _enabled(self, pii_type: PIIType) -> bool:
        return self.enabled_types is None or pii_type in self.enabled_types

    def detect(self, text: str) -> list[PIIMatch]:
        matches: list[PIIMatch] = []
        if not text:
            return matches
        for pii_type, pattern, confidence in _PATTERNS:
            if not self._enabled(pii_type):
                continue
            for match in pattern.finditer(text):
                value = match.group(0)
                if pii_type == "credit_card":
                    if not _luhn_valid(value):
                        continue
                    confidence = 0.95
                if pii_type == "phone":
                    if sum(c.isdigit() for c in value) < 7:
                        continue
                    # Don't report IPs or dotted versions as phone numbers.
                    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value.strip()):
                        continue
                matches.append(
                    PIIMatch(
                        type=pii_type,
                        value=value.strip(),
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                    )
                )
        if self._enabled("person_name"):
            for pattern, confidence in ((_NAME_HONORIFIC_RE, 0.85), (_NAME_CUE_RE, 0.7)):
                for match in pattern.finditer(text):
                    value = match.group(1) if match.groups() else match.group(0)
                    matches.append(
                        PIIMatch(
                            type="person_name",
                            value=value,
                            start=match.start(),
                            end=match.end(),
                            confidence=confidence,
                        )
                    )
        # Drop overlapping lower-confidence matches.
        matches.sort(key=lambda m: (m.start, -m.confidence))
        kept: list[PIIMatch] = []
        for match in matches:
            if kept and match.start < kept[-1].end and match.type == kept[-1].type:
                continue
            kept.append(match)
        return kept

    def contains_pii(self, text: str, *, min_confidence: float = 0.7) -> bool:
        return any(m.confidence >= min_confidence for m in self.detect(text))


def redact(
    text: str,
    matches: list[PIIMatch] | None = None,
    *,
    detector: PIIDetector | None = None,
    placeholder: str = "[REDACTED:{type}]",
    min_confidence: float = 0.7,
) -> str:
    """Replace detected PII spans with typed placeholders."""
    if matches is None:
        matches = (detector or PIIDetector()).detect(text)
    result = text
    for match in sorted(matches, key=lambda m: m.start, reverse=True):
        if match.confidence < min_confidence:
            continue
        result = result[: match.start] + placeholder.format(type=match.type) + result[match.end :]
    return result
