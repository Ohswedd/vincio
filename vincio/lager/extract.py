"""Claim extraction — documents → Evidence Objects, deterministically.

The default extractor is pure stdlib and CI-gated: abbreviation-aware sentence
spans over the canonical text (byte-exact, tracked with a monotone cursor),
fragments merged forward, list / table / code regions emitted as single typed
objects (never sentence-split), a LAGER-owned entity normalizer (the shared
:func:`~vincio.retrieval.chunking.extract_entities` drops "The FTC" outright,
returns nothing on lowercase prose, and admits pronouns), and an IDF fallback
so every object lands in at least one graph bucket. The :class:`ClaimExtractor`
protocol makes the module replaceable — any extractor whose claims are
byte-exact spans of the canonical text satisfies the contract.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from math import log
from typing import Protocol

from ..core.errors import LagerError
from ..core.types import Document, TrustLevel
from ..retrieval.chunking import extract_entities
from .objects import EXTRACTOR_VERSION, EvidenceObject, canonical_text, document_key

__all__ = ["ClaimExtractor", "DeterministicClaimExtractor", "normalize_entities"]

# Sentence terminators that end a claim only when NOT part of an abbreviation.
_TERMINATOR_RE = re.compile(r"[.!?]")
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "no", "vs", "etc",
    "corp", "inc", "ltd", "co", "dept", "est", "fig", "vol", "approx",
    "e.g", "i.e", "cf", "al", "u.s", "u.k", "a.m", "p.m",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
})
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*```")

# Entity extraction the graph can actually ride on.
_LEAD_ARTICLE_RE = re.compile(
    r"\b(?:The|A|An)\s+((?:[A-Z][\w&.-]*)(?:\s+[A-Z][\w&.-]*)*)"
)
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_PRONOUNS = frozenset({
    "he", "she", "they", "we", "you", "i", "it", "this", "that", "these", "those",
    "his", "her", "their", "our", "its",
})
_LEAD_STOPWORDS = frozenset({"the", "a", "an", "in", "on", "at", "of", "for", "by"})
_REFERRING_OPENER_RE = re.compile(
    r"^(?:he|she|they|it|this|that|these|those|the (?:former|latter|company|team|"
    r"system|service|policy|plan|product|feature|change|result|approach))\b",
    re.IGNORECASE,
)

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_DATE_PATTERNS = (
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.IGNORECASE),
    re.compile(rf"\b({_MONTHS})\s+(\d{{4}})\b", re.IGNORECASE),
    re.compile(r"\bQ([1-4])\s+(\d{4})\b"),
)
_QUARTER_MONTH = {1: 1, 2: 4, 3: 7, 4: 10}
_STOP_TERMS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "for", "with", "by", "from", "as", "it", "its",
    "not", "no", "will", "would", "can", "could", "should", "must", "may",
    "has", "have", "had", "do", "does", "did", "so", "such", "also", "there",
})
_WORD_RE = re.compile(r"[a-z][a-z0-9'-]+")

_MIN_CLAIM_WORDS = 4
_BUCKET_TERMS = 5


def normalize_entities(text: str) -> list[str]:
    """LAGER's entity vocabulary: the shared extractor plus a definite-article
    rescue pass and an acronym pass, with pronouns stopped and leading
    stopwords trimmed (not match-rejected). Lowercased, deduplicated, sorted."""
    found: list[str] = []
    for raw in extract_entities(text):
        tokens = raw.split()
        while tokens and tokens[0].lower() in _LEAD_STOPWORDS:
            tokens.pop(0)
        if not tokens:
            continue
        candidate = " ".join(tokens)
        if candidate.lower() in _PRONOUNS:
            continue
        found.append(candidate)
    for match in _LEAD_ARTICLE_RE.finditer(text):
        found.append(match.group(1))
    for match in _ACRONYM_RE.finditer(text):
        found.append(match.group(0))
    normalized: list[str] = []
    seen: set[str] = set()
    for entity in found:
        key = entity.lower().strip()
        if not key or key in _PRONOUNS or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return sorted(normalized)


def content_terms(text: str) -> list[str]:
    """Lowercased content terms (stopwords removed) — the IDF fallback pool."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP_TERMS]


def parse_observed_at(text: str) -> datetime | None:
    """First recognizable date in the claim (ISO, month-name, month-year, or
    quarter) → an aware UTC datetime; None when the claim carries no date."""
    month_index = {name: i + 1 for i, name in enumerate(_MONTHS.split("|"))}
    iso = _DATE_PATTERNS[0].search(text)
    if iso:
        year, month, day = (int(g) for g in iso.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return datetime(year, month, min(day, 28), tzinfo=UTC)
    full = _DATE_PATTERNS[1].search(text)
    if full:
        month = month_index[full.group(1).lower()]
        return datetime(int(full.group(3)), month, min(int(full.group(2)), 28),
                        tzinfo=UTC)
    month_year = _DATE_PATTERNS[2].search(text)
    if month_year:
        return datetime(int(month_year.group(2)), month_index[month_year.group(1).lower()],
                        1, tzinfo=UTC)
    quarter = _DATE_PATTERNS[3].search(text)
    if quarter:
        return datetime(int(quarter.group(2)), _QUARTER_MONTH[int(quarter.group(1))],
                        1, tzinfo=UTC)
    return None


def is_referring(claim: str) -> bool:
    """True when the claim opens with a referring form (pronoun / definite
    reference) and so needs its antecedent to be self-contained."""
    return bool(_REFERRING_OPENER_RE.match(claim.strip()))


class ClaimExtractor(Protocol):
    """The replaceable extraction contract: canonical document → objects whose
    claims are byte-exact slices of the canonical text."""

    def extract(self, document: Document) -> list[EvidenceObject]: ...


def _line_spans(canon: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(canon):
        if char == "\n":
            spans.append((start, index))
            start = index + 1
    spans.append((start, len(canon)))
    return spans


def _is_abbreviation_boundary(canon: str, dot: int) -> bool:
    """True when the terminator at *dot* ends an abbreviation or an initial —
    not a sentence."""
    if canon[dot] != ".":
        return False
    word_start = dot
    while word_start > 0 and (canon[word_start - 1].isalnum() or canon[word_start - 1] == "."):
        word_start -= 1
    token = canon[word_start:dot].lower().rstrip(".")
    if token in _ABBREVIATIONS:
        return True
    # An initial ("J. Smith") — but never a date/quantity ending a sentence
    # ("…on 2025-11-03." must terminate); numbered list items are handled by
    # the list-region detector, not here.
    return len(token) == 1 and token.isalpha()


def _sentence_spans(canon: str, start: int, end: int) -> list[tuple[int, int]]:
    """Abbreviation-aware sentence spans within [start, end) of the canonical
    text. Byte-exact: each span slices to the sentence, cursor is monotone."""
    spans: list[tuple[int, int]] = []
    cursor = start
    position = start
    while position < end:
        char = canon[position]
        if char in ".!?" and not _is_abbreviation_boundary(canon, position):
            following = position + 1
            if following >= end or canon[following] in " \n\t\"'”)":
                spans.append((cursor, position + 1))
                cursor = position + 1
        elif char == "\n":
            if position > cursor:
                spans.append((cursor, position))
            cursor = position + 1
        position += 1
    if cursor < end:
        spans.append((cursor, end))
    return spans


def _trim(canon: str, span: tuple[int, int]) -> tuple[int, int]:
    """Strip whitespace by moving the span endpoints — never rewriting text."""
    begin, finish = span
    while begin < finish and canon[begin] in " \t\n\"'“”":
        begin += 1
    while finish > begin and canon[finish - 1] in " \t\n\"'“”":
        finish -= 1
    return begin, finish


_SUBSPLIT_RE = re.compile(r";\s+|,\s+(?=(?:but|however|whereas|while)\b)", re.IGNORECASE)


class DeterministicClaimExtractor:
    """The stdlib-only, CI-gated default: same bytes in, same objects out."""

    def extract(self, document: Document) -> list[EvidenceObject]:
        canon = canonical_text(document.text or "")
        if not canon.strip():
            return []
        doc_key = document_key(document.text or "")
        regions = self._regions(canon)
        objects: list[EvidenceObject] = []
        for kind, begin, finish in regions:
            if kind == "prose":
                objects.extend(self._prose_claims(document, canon, doc_key, begin, finish))
            else:
                span = _trim(canon, (begin, finish))
                if span[0] >= span[1]:
                    continue
                objects.append(self._build(document, canon, doc_key, span, kind=kind))
        for obj in objects:
            if not obj.verify(document.text or ""):  # extraction-time self-check
                raise LagerError(
                    f"extracted object {obj.id} does not re-derive from its span",
                    details={"span": list(obj.span), "doc_key": doc_key[:16]},
                )
        return objects

    # -- regions ---------------------------------------------------------------------

    def _regions(self, canon: str) -> list[tuple[str, int, int]]:
        """Segment into prose / list / table / code regions by line shape; a
        structural region becomes ONE object downstream."""
        lines = _line_spans(canon)
        regions: list[tuple[str, int, int]] = []
        current_kind = "prose"
        region_start = 0
        in_code = False

        def close(upto: int) -> None:
            nonlocal region_start
            if upto > region_start:
                regions.append((current_kind, region_start, upto))
            region_start = upto

        for begin, finish in lines:
            line = canon[begin:finish]
            if _CODE_FENCE_RE.match(line):
                if in_code:
                    close(finish)  # include the closing fence
                    in_code = False
                    current_kind = "prose"
                else:
                    close(begin)
                    current_kind = "code"
                    in_code = True
                continue
            if in_code:
                continue
            if _TABLE_LINE_RE.match(line):
                kind = "table"
            elif _LIST_LINE_RE.match(line):
                kind = "list"
            else:
                kind = "prose"
            if kind != current_kind:
                close(begin)
                current_kind = kind
        close(len(canon))
        return [(kind, begin, finish) for kind, begin, finish in regions if finish > begin]

    # -- prose -----------------------------------------------------------------------

    def _prose_claims(
        self, document: Document, canon: str, doc_key: str, begin: int, finish: int
    ) -> list[EvidenceObject]:
        spans = [_trim(canon, s) for s in _sentence_spans(canon, begin, finish)]
        spans = [s for s in spans if s[1] > s[0]]
        # Merge fragments forward (span union) so "Dr." + "Smith joined…" —
        # anything the abbreviation guard missed — never ships as a claim.
        merged: list[tuple[int, int]] = []
        for span in spans:
            words = len(canon[span[0]:span[1]].split())
            if merged and (words < _MIN_CLAIM_WORDS or self._continues(canon, merged[-1])):
                merged[-1] = (merged[-1][0], span[1])
            else:
                merged.append(span)
        objects: list[EvidenceObject] = []
        for span in merged:
            for piece in self._subsplit(canon, span):
                piece = _trim(canon, piece)
                text = canon[piece[0]:piece[1]]
                if len(text.split()) < _MIN_CLAIM_WORDS or text.endswith("?"):
                    continue
                objects.append(self._build(document, canon, doc_key, piece, kind="claim"))
        return objects

    @staticmethod
    def _continues(canon: str, previous: tuple[int, int]) -> bool:
        tail = canon[previous[0]:previous[1]].rstrip()
        return not tail.endswith((".", "!", "?"))

    @staticmethod
    def _subsplit(canon: str, span: tuple[int, int]) -> list[tuple[int, int]]:
        """Connective sub-split with spans kept inside the parent bounds; a side
        shorter than the claim minimum keeps the parent unsplit."""
        text = canon[span[0]:span[1]]
        boundaries = list(_SUBSPLIT_RE.finditer(text))
        if not boundaries:
            return [span]
        pieces: list[tuple[int, int]] = []
        cursor = 0
        for match in boundaries:
            pieces.append((span[0] + cursor, span[0] + match.start()))
            cursor = match.end()
        pieces.append((span[0] + cursor, span[1]))
        texts = [canon[b:f] for b, f in pieces]
        if any(len(t.split()) < _MIN_CLAIM_WORDS for t in texts):
            return [span]
        return pieces

    # -- object assembly ---------------------------------------------------------------

    def _build(
        self,
        document: Document,
        canon: str,
        doc_key: str,
        span: tuple[int, int],
        *,
        kind: str,
    ) -> EvidenceObject:
        claim = canon[span[0]:span[1]]
        entities = normalize_entities(claim)
        words = len(claim.split())
        confidence = 0.5
        if entities:
            confidence += 0.15
        if any(ch.isdigit() for ch in claim):
            confidence += 0.1
        if kind == "claim" and 6 <= words <= 40:
            confidence += 0.1
        if kind != "claim":
            confidence = 0.6
        return EvidenceObject.create(
            claim=claim,
            doc_key=doc_key,
            span=span,
            kind=kind,  # type: ignore[arg-type]
            document_id=document.id,
            source_uri=document.source_uri,
            title=document.title,
            entities=entities,
            confidence=round(min(confidence, 0.95), 4),
            authority=_AUTHORITY_BY_TRUST.get(document.trust_level, 0.5),
            trust_level=document.trust_level,
            observed_at=parse_observed_at(claim),
            metadata={"extractor": f"deterministic/{EXTRACTOR_VERSION}"},
        )


_AUTHORITY_BY_TRUST = {
    TrustLevel.SYSTEM: 0.95,
    TrustLevel.DEVELOPER: 0.9,
    TrustLevel.USER: 0.7,
    TrustLevel.UNTRUSTED_DOCUMENT: 0.5,
    TrustLevel.UNTRUSTED_TOOL: 0.4,
    TrustLevel.UNTRUSTED_EXTERNAL: 0.3,
}


def assign_fallback_terms(objects: list[EvidenceObject]) -> None:
    """Corpus-level IDF pass: every object gets its top distinctive content
    terms (in place, deterministic). Terms are how lowercase prose — which the
    capitalization-based entity extractor cannot see — still lands in graph
    buckets and links across documents; ubiquitous terms are neutralized
    downstream by the graph's document-frequency cut."""
    document_frequency: dict[str, int] = {}
    per_object: list[list[str]] = []
    for obj in objects:
        terms = sorted(set(content_terms(obj.claim)))
        per_object.append(terms)
        for term in terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    total = max(len(objects), 1)
    for obj, terms in zip(objects, per_object, strict=True):
        if not terms:
            continue
        scored = sorted(
            ((log(total / document_frequency[t]), t) for t in terms),
            key=lambda pair: (-pair[0], pair[1]),
        )
        obj.terms = [t for _, t in scored[:_BUCKET_TERMS]]


