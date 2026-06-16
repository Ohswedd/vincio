"""Locale packs: non-English PII detectors for the PII engine.

The built-in :class:`~vincio.security.pii.PIIDetector` patterns are
English/US-centric (US SSN, US-style phones, US addresses). Regulated buyers
operate in many jurisdictions, so a PII control that only finds US identifiers
under-reports risk everywhere else.

A :class:`LocalePack` contributes additional national-ID and locale phone
patterns under a locale code. ``PIIDetector(locales=["fr", "de", "in"])`` layers
them on top of the built-ins without changing the English path — locale matches
carry the ``locale`` tag on their :class:`~vincio.security.pii.PIIMatch`.

The packs are deterministic regex (dependency-free); they are not a substitute
for a full NER model, but they reliably catch the structured identifiers that
matter most for compliance.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

__all__ = ["LocalePack", "LOCALE_PACKS", "available_locales", "get_locale_pack", "resolve_locales"]


class LocalePack(BaseModel):
    """A set of locale-specific PII patterns.

    Each pattern is ``(type_label, regex, confidence)``; ``type_label`` becomes
    the :class:`~vincio.security.pii.PIIMatch` ``type`` (e.g. ``"national_id"``,
    ``"tax_id"``), so policies can treat it like any other PII category.
    """

    locale: str
    name: str
    patterns: list[tuple[str, str, float]] = Field(default_factory=list)

    def compiled(self) -> list[tuple[str, str, re.Pattern[str], float]]:
        return [
            (self.locale, type_label, re.compile(pattern), confidence)
            for type_label, pattern, confidence in self.patterns
        ]


# National identifiers and locale phone formats. Patterns are intentionally
# specific (anchored, structured) to keep false positives low.
LOCALE_PACKS: dict[str, LocalePack] = {
    "fr": LocalePack(
        locale="fr",
        name="France",
        patterns=[
            # NIR / INSEE social-security number (13–15 digits, sex+year+month).
            ("national_id", r"\b[12]\s?\d{2}\s?(?:0[1-9]|1[0-2])\s?\d{2}\s?\d{3}\s?\d{3}(?:\s?\d{2})?\b", 0.85),
            ("phone", r"(?<!\d)0[1-9](?:[\s.]?\d{2}){4}(?!\d)", 0.7),
        ],
    ),
    "de": LocalePack(
        locale="de",
        name="Germany",
        patterns=[
            # Steuer-Identifikationsnummer (11 digits).
            ("tax_id", r"(?<!\d)\d{2}\s?\d{3}\s?\d{3}\s?\d{3}(?!\d)", 0.6),
            ("phone", r"(?<!\d)0\d{2,4}[\s/-]?\d{3,8}(?!\d)", 0.5),
        ],
    ),
    "es": LocalePack(
        locale="es",
        name="Spain",
        patterns=[
            # DNI (8 digits + control letter) and NIE (X/Y/Z + 7 digits + letter).
            ("national_id", r"\b\d{8}[A-HJ-NP-TV-Z]\b", 0.9),
            ("national_id", r"\b[XYZ]\d{7}[A-Z]\b", 0.9),
        ],
    ),
    "in": LocalePack(
        locale="in",
        name="India",
        patterns=[
            # Aadhaar (12 digits, grouped) and PAN (5 letters, 4 digits, 1 letter).
            ("national_id", r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", 0.85),
            ("tax_id", r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.9),
        ],
    ),
    "sg": LocalePack(
        locale="sg",
        name="Singapore",
        patterns=[
            # NRIC / FIN (S/T/F/G + 7 digits + checksum letter).
            ("national_id", r"\b[STFG]\d{7}[A-Z]\b", 0.9),
        ],
    ),
    "br": LocalePack(
        locale="br",
        name="Brazil",
        patterns=[
            # CPF (XXX.XXX.XXX-XX).
            ("national_id", r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", 0.95),
        ],
    ),
    "uk": LocalePack(
        locale="uk",
        name="United Kingdom",
        patterns=[
            # National Insurance number (2 letters, 6 digits, 1 letter).
            ("national_id", r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b", 0.85),
        ],
    ),
}


def available_locales() -> list[str]:
    """Locale codes with a registered pack."""
    return sorted(LOCALE_PACKS)


def get_locale_pack(code: str) -> LocalePack:
    """Return the pack for a locale code (e.g. ``"fr"``).

    Accepts BCP-47-ish codes (``"fr-FR"``) by taking the language subtag.
    """
    key = code.lower().split("-")[0]
    if key not in LOCALE_PACKS:
        raise KeyError(f"no locale pack for {code!r}; available: {available_locales()}")
    return LOCALE_PACKS[key]


def resolve_locales(
    locales: list[str | LocalePack],
) -> list[tuple[str, str, re.Pattern[str], float]]:
    """Compile a list of locale codes / packs into detector patterns."""
    compiled: list[tuple[str, str, re.Pattern[str], float]] = []
    for entry in locales:
        pack = entry if isinstance(entry, LocalePack) else get_locale_pack(entry)
        compiled.extend(pack.compiled())
    return compiled
