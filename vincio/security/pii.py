"""PII detection and redaction.

Deterministic regex + heuristic detectors for: email, phone, person names,
addresses, government IDs, financial data, health data markers, and API
keys/secrets. Detection returns spans with type and confidence so policies
can decide between blocking, redaction, and pass-through.

The built-in patterns are English/US-centric. Non-English **locale packs**
(1.6, :mod:`vincio.security.locales`) extend detection with national-ID and
locale phone formats (e.g. France NIR, Germany Steuer-ID, India Aadhaar/PAN,
Singapore NRIC) without changing the built-in path: pass ``locales=`` to
:class:`PIIDetector`. Locale matches carry a ``locale`` tag on the
:class:`PIIMatch`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from .backends import DetectorBackend

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .locales import LocalePack

__all__ = ["PIIType", "PIIMatch", "PIIDetector", "redact"]

# The built-in (English/US-centric) PII categories. Non-English locale packs add
# further category labels (strings) at runtime, so ``PIIMatch.type`` is a plain
# ``str`` — ``PIIType`` documents the built-ins and is accepted everywhere a type
# is expected.
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
    type: str
    value: str
    start: int
    end: int
    confidence: float
    locale: str | None = None  # set when a locale pack produced the match


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
    def __init__(
        self,
        *,
        enabled_types: set[str] | None = None,
        locales: Sequence[str | LocalePack] | None = None,
        backend: DetectorBackend | None = None,
    ) -> None:
        self.enabled_types = enabled_types
        # Optional ML backend (NER, etc.): its spans merge with the regex spans
        # through the same overlap resolution. None ⇒ deterministic-only.
        self.backend = backend
        self._locale_patterns: list[tuple[str, str, re.Pattern[str], float]] = []
        if locales:
            from .locales import resolve_locales

            self._locale_patterns = resolve_locales(locales)

    @property
    def locales(self) -> list[str]:
        """Locale codes whose packs this detector applies (deduplicated)."""
        return sorted({locale for locale, _type, _pat, _conf in self._locale_patterns})

    def _enabled(self, pii_type: str) -> bool:
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
        for locale, type_label, pattern, confidence in self._locale_patterns:
            if not self._enabled(type_label):
                continue
            for match in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        type=type_label,
                        value=match.group(0).strip(),
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        locale=locale,
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
        if self.backend is not None:
            for span in self.backend.detect(text):
                if not self._enabled(span.label):
                    continue
                matches.append(
                    PIIMatch(
                        type=span.label,
                        value=(span.text or text[span.start : span.end]).strip(),
                        start=span.start,
                        end=span.end,
                        confidence=max(0.0, min(1.0, span.score)),
                    )
                )
        # Drop overlapping lower-confidence matches.
        matches.sort(key=lambda m: (m.start, -m.confidence))
        kept: list[PIIMatch] = []
        for candidate in matches:
            if kept and candidate.start < kept[-1].end and candidate.type == kept[-1].type:
                continue
            kept.append(candidate)
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
    """Replace detected PII spans with typed placeholders.

    Overlapping spans (possible once non-English locale packs add categories that
    can cover the same characters as a built-in one) are resolved to
    non-overlapping spans before replacement, so indices never corrupt — the
    earliest span wins, with the higher-confidence one on a tie.
    """
    if matches is None:
        matches = (detector or PIIDetector()).detect(text)
    eligible = sorted(
        (m for m in matches if m.confidence >= min_confidence),
        key=lambda m: (m.start, -m.confidence),
    )
    kept: list[PIIMatch] = []
    last_end = -1
    for match in eligible:
        if match.start >= last_end:
            kept.append(match)
            last_end = match.end
    result = text
    for match in sorted(kept, key=lambda m: m.start, reverse=True):
        result = result[: match.start] + placeholder.format(type=match.type) + result[match.end :]
    return result
