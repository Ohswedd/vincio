"""Context anchors — the always-on task frame for chain-call work.

Some documents are not "look it up when asked" evidence; they are the *frame*
of the whole task. A vibe-coder building a CLI editor starts with a bulk of MD
files — a PRD, a brand identity, an architecture note, coding standards — that
are **100% needed for the global context of the task but not in full on every
single call**. Pasting them into every call is token-hungry and re-paid per
step; pure per-query retrieval silently drops a constraint ("brand voice: warm
and concise") the moment a call ("implement the login endpoint") doesn't
lexically match it.

An **anchor** solves this with two tiers:

* the always-on **frame** — a compact, deterministic, content-hash-cached
  :class:`AnchorBrief` (a token-bounded digest that *prefers the normative
  lines* — must / never / always / required — because those are the constraints
  a task must respect) injected into every run as **pinned** evidence, so it is
  never dropped and costs a flat ~few-hundred tokens regardless of how large the
  anchor corpus is;
* on-demand **detail** — the anchor documents are still fully chunked and
  indexed, so a call that needs a specific section retrieves it normally.

Built once, cached by the corpus content hash, re-derivable offline: the same
documents and budget always produce the same brief.
"""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, Field

from ..core.tokens import count_tokens
from ..core.types import Document, EvidenceItem, TrustLevel
from .compression import _verified_truncate

__all__ = ["AnchorBrief", "AnchorSet", "build_anchor_brief"]

_FRAME_HEADER = "Project frame (task anchors — always applies):"

# Lines that state a rule/constraint are the frame's load-bearing content — a
# PRD's requirements, a brand doc's do/don't, a spec's invariants. Preferred so
# they survive the token budget over prose, regardless of which document they
# live in (a tiny rules file must not be starved by a verbose README).
_NORMATIVE_RE = re.compile(
    r"\b(must(?:\s+not)?|shall(?:\s+not)?|should(?:\s+not)?|never|always|required|"
    r"do not|don't|avoid|ensure|prefer|constraint|invariant|mandat\w+|forbidden)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")
_HEADER_RESERVE = 6  # tokens budgeted for each rendered "### title" line


class AnchorBrief(BaseModel):
    """The compact, always-on frame distilled from the anchor documents.

    ``content_hash`` keys the cache and proves the brief re-derives from its
    source corpus; ``text`` is the rendered frame injected as pinned evidence.
    """

    text: str = ""
    tokens: int = 0
    sources: list[str] = Field(default_factory=list)
    document_count: int = 0
    budget_tokens: int = 400
    content_hash: str = ""

    def as_evidence(self) -> EvidenceItem:
        """The pinned evidence item that carries the frame into every run.

        High authority and ``relevance=1.0`` so the compiler's relevance gate
        never touches it; ``pinned`` guarantees inclusion; DEVELOPER trust marks
        it as first-party (not untrusted web/doc content)."""
        return EvidenceItem(
            # Content-derived, stable id: unchanged across the 50-call chain (so
            # the compile cache and receipt_hash stay stable), changed only when
            # the corpus changes (so a diff surfaces the frame update).
            id=f"anchor:{self.content_hash[:16]}",
            source_id="task-frame",
            source_type="document",
            text=self.text,
            pinned=True,
            relevance=1.0,
            authority=0.95,
            provenance=0.95,
            trust_level=TrustLevel.DEVELOPER,
            token_cost=self.tokens,
            metadata={"anchor_brief": True, "sources": self.sources},
        )

    def verify(self, documents: list[Document], *, budget_tokens: int | None = None) -> bool:
        """True iff *documents* re-derive this exact brief (offline honesty)."""
        rebuilt = build_anchor_brief(
            documents, sources=self.sources, brief_tokens=budget_tokens or self.budget_tokens
        )
        return rebuilt.content_hash == self.content_hash and rebuilt.text == self.text


#: Bumped when the brief-building algorithm changes output, so a Vincio upgrade
#: invalidates persisted briefs rather than serving a stale one from cache.
_BRIEF_ALGO_VERSION = "3"


def _text_hash(text: str) -> str:
    """Content address of the *rendered* frame. Hashing the output (not just the
    inputs) means two environments whose tokenizers disagree — and so render
    different briefs — get different ids, and ``verify()`` never false-matches a
    brief it did not actually produce."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_header(document: Document) -> str:
    return (document.title or str(document.metadata.get("source") or "")
            or (document.source_uri or "")).strip()


def _corpus_hash(documents: list[Document], brief_tokens: int) -> str:
    """Cheap change-detector for the in-process AnchorSet cache. Length-framed so
    no two distinct corpora share a preimage, and it covers every field that
    steers rendering (title, the header fallbacks, and body text)."""
    hasher = hashlib.sha256()

    def feed(value: str) -> None:
        raw = value.encode("utf-8")
        hasher.update(len(raw).to_bytes(8, "big"))
        hasher.update(raw)

    feed(_BRIEF_ALGO_VERSION)
    feed(str(brief_tokens))
    hasher.update(len(documents).to_bytes(8, "big"))
    for document in documents:
        feed(document.title or "")
        feed(_doc_header(document))
        feed(document.text or "")
    return hasher.hexdigest()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def build_anchor_brief(
    documents: list[Document],
    *,
    sources: list[str] | None = None,
    brief_tokens: int = 400,
) -> AnchorBrief:
    """Distill *documents* into a token-bounded, deterministic task frame.

    Constraint-first *globally*: every document's normative lines (must / never /
    always / required …) are preferred over prose regardless of which document
    they live in, so a small rules file is never starved by a verbose README.
    Whole sentences only — nothing is ever truncated mid-word — and the assembled
    frame is bounded by dropping whole trailing sentences, never by re-summarizing
    the rendered blocks. The id is the content hash of the rendered text. The
    result never shrinks below the frame-header line, so a ``brief_tokens``
    smaller than that (~10 tokens) yields the bare header rather than nothing.
    """
    real = [d for d in documents if (d.text or "").strip() or (d.title or "").strip()]
    source_names = sources if sources is not None else sorted(
        {str(d.metadata.get("source") or d.title or d.source_uri or "anchor") for d in real}
    )
    if not real:
        return AnchorBrief(
            budget_tokens=brief_tokens, sources=source_names, content_hash=_text_hash(""),
        )

    # Budget for sentence bodies: the whole window minus the frame header and one
    # reserved header line per document (conservative — unused reserve is fine).
    body_budget = max(
        brief_tokens - count_tokens(_FRAME_HEADER) - _HEADER_RESERVE * len(real),
        brief_tokens // 2,
    )
    # Global priority queue: all normative sentences first (in document order),
    # then all prose — each carrying its document index so picks regroup by doc.
    normative: list[tuple[int, str]] = []
    prose: list[tuple[int, str]] = []
    for index, document in enumerate(real):
        seen_local: set[str] = set()
        for sentence in _sentences(document.text or ""):
            key = sentence.lower()
            if key in seen_local:
                continue
            seen_local.add(key)
            (normative if _NORMATIVE_RE.search(sentence) else prose).append((index, sentence))

    picked: dict[int, list[str]] = {index: [] for index in range(len(real))}
    used = 0
    seen: set[str] = set()
    for index, sentence in [*normative, *prose]:
        key = sentence.lower()
        # Skip exact and prefix/superset duplicates (no garbled near-dupes).
        if key in seen or any(key in existing or existing in key for existing in seen):
            continue
        cost = count_tokens(sentence)
        if used + cost > body_budget:
            # Keep scanning: a shorter later sentence may still fit. An oversize
            # *first* sentence must not be admitted either — it would block every
            # later (fitting) constraint and then be trimmed away, emptying the
            # frame.
            continue
        picked[index].append(sentence)
        seen.add(key)
        used += cost
    if used == 0 and (normative or prose):
        # Every sentence alone exceeds the budget (one giant unpunctuated line):
        # carry a verified whole-word cut of the best sentence rather than nothing.
        index, sentence = (normative or prose)[0]
        cut, cut_tokens = _verified_truncate(sentence, max(body_budget, 1))
        if cut:
            picked[index].append(cut)
            used = cut_tokens

    def render(chosen: dict[int, list[str]]) -> str:
        blocks: list[str] = []
        for index, document in enumerate(real):
            if not chosen[index]:
                continue
            title = _doc_header(document)
            header = f"### {title}" if title else "###"
            blocks.append(f"{header}\n{' '.join(chosen[index])}")
        return _FRAME_HEADER + "\n" + "\n".join(blocks) if blocks else _FRAME_HEADER

    rendered = render(picked)
    # Header estimates can nudge the real total over budget; trim whole trailing
    # sentences (block-aware, never mid-word) until it fits.
    order = [*normative, *prose]
    while count_tokens(rendered) > brief_tokens and any(picked[i] for i in picked):
        for index, _sentence in reversed(order):
            if picked[index]:
                picked[index].pop()
                break
        rendered = render(picked)

    return AnchorBrief(
        text=rendered,
        tokens=count_tokens(rendered),
        sources=source_names,
        document_count=len(real),
        budget_tokens=brief_tokens,
        content_hash=_text_hash(rendered),
    )


class AnchorSet:
    """The app's anchor documents and their cached frame.

    Holds anchor docs per source, rebuilds the :class:`AnchorBrief` when the
    corpus changes (content-hash cached, so re-adding identical docs is free),
    and hands the pinned frame evidence to the run loop.
    """

    def __init__(self) -> None:
        self._docs: dict[str, list[Document]] = {}
        self._brief_tokens: dict[str, int] = {}
        self._brief: AnchorBrief | None = None
        self._cache_key: str = ""

    def __bool__(self) -> bool:
        return bool(self._docs)

    @property
    def sources(self) -> list[str]:
        return sorted(self._docs)

    def add(self, source: str, documents: list[Document], *, brief_tokens: int = 400) -> None:
        """Register (or replace) an anchor source and invalidate the cache."""
        self._docs[source] = list(documents)
        self._brief_tokens[source] = brief_tokens
        self._brief = None
        self._cache_key = ""

    def remove(self, source: str) -> bool:
        """Drop an anchor source and invalidate the cache. Returns True if it was
        present. Used by ``erase_source`` so an erased source stops injecting its
        frame — the anchor plane is part of the erasure sweep, not a leak past it."""
        if source not in self._docs:
            return False
        del self._docs[source]
        self._brief_tokens.pop(source, None)
        self._brief = None
        self._cache_key = ""
        return True

    def _all_docs(self) -> list[Document]:
        docs: list[Document] = []
        for source in sorted(self._docs):
            docs.extend(self._docs[source])
        return docs

    def brief(self) -> AnchorBrief | None:
        """The current frame, rebuilt only when the corpus changed."""
        if not self._docs:
            return None
        budget = max(self._brief_tokens.values(), default=400)
        docs = self._all_docs()
        key = _corpus_hash(docs, budget)
        if self._brief is None or self._cache_key != key:
            self._brief = build_anchor_brief(docs, sources=self.sources, brief_tokens=budget)
            self._cache_key = key
        return self._brief

    def brief_evidence(self) -> list[EvidenceItem]:
        """The pinned frame evidence for the run loop (empty when no anchors)."""
        brief = self.brief()
        return [brief.as_evidence()] if brief and brief.text else []
