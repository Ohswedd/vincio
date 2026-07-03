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
from ..memory.summarizers import extractive_summary

__all__ = ["AnchorBrief", "AnchorSet", "build_anchor_brief"]

# Lines that state a rule/constraint are the frame's load-bearing content — a
# PRD's requirements, a brand doc's do/don't, a spec's invariants. Boosted so
# they survive the token budget over prose.
_NORMATIVE_RE = re.compile(
    r"\b(must(?:\s+not)?|shall(?:\s+not)?|should(?:\s+not)?|never|always|required|"
    r"do not|don't|avoid|ensure|prefer|constraint|invariant|mandat\w+|forbidden)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")
_MIN_DOC_TOKENS = 24  # every anchor doc gets at least a header + a line or two
_HEADER_RESERVE = 6  # tokens reserved for a doc's title line


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
_BRIEF_ALGO_VERSION = "1"


def _corpus_hash(documents: list[Document], brief_tokens: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(_BRIEF_ALGO_VERSION.encode())
    hasher.update(b"\x00")
    hasher.update(str(brief_tokens).encode())
    for document in documents:
        hasher.update(b"\x00")
        hasher.update((document.title or "").encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update((document.text or "").encode("utf-8"))
    return hasher.hexdigest()


def _rank_sentences(text: str, budget_tokens: int) -> str:
    """The most frame-relevant sentences of *text* within *budget_tokens*.

    Normative sentences (must / never / always / required …) are preferred over
    prose; ties and the remainder fall back to lead-position extractive summary.
    Deterministic and offline."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return ""
    normative = [s for s in sentences if _NORMATIVE_RE.search(s)]
    picked: list[str] = []
    used = 0
    seen: set[str] = set()
    # 1) as many normative (constraint) lines as fit, in document order
    for sentence in normative:
        key = sentence.lower()
        if key in seen:
            continue
        cost = count_tokens(sentence)
        if used + cost > budget_tokens and picked:
            break
        picked.append(sentence)
        seen.add(key)
        used += cost
    # 2) fill any remaining budget with a lead-biased extractive summary
    if used < budget_tokens:
        remainder = extractive_summary(text, max_tokens=budget_tokens - used)
        for sentence in (s.strip() for s in _SENTENCE_SPLIT.split(remainder) if s.strip()):
            key = sentence.lower()
            if key in seen:
                continue
            cost = count_tokens(sentence)
            if used + cost > budget_tokens and picked:
                break
            picked.append(sentence)
            seen.add(key)
            used += cost
    return " ".join(picked)


def build_anchor_brief(
    documents: list[Document],
    *,
    sources: list[str] | None = None,
    brief_tokens: int = 400,
) -> AnchorBrief:
    """Distill *documents* into a token-bounded, deterministic task frame.

    The budget is shared across documents proportionally to their size (each
    getting at least a header and a line or two), and within a document the
    normative/constraint lines are preferred. The result is capped at
    ``brief_tokens`` and cached by the corpus content hash.
    """
    real = [d for d in documents if (d.text or "").strip() or (d.title or "").strip()]
    source_names = sources if sources is not None else sorted(
        {str(d.metadata.get("source") or d.title or d.source_uri or "anchor") for d in real}
    )
    if not real:
        return AnchorBrief(
            budget_tokens=brief_tokens, sources=source_names,
            content_hash=_corpus_hash([], brief_tokens),
        )
    body_budget = max(brief_tokens - _HEADER_RESERVE, brief_tokens // 2)
    sizes = [max(count_tokens(d.text or ""), 1) for d in real]
    total_size = sum(sizes) or 1
    blocks: list[str] = []
    for document, size in zip(real, sizes, strict=True):
        share = int(body_budget * size / total_size)
        per_doc = max(_MIN_DOC_TOKENS, share)
        title = (document.title or document.metadata.get("source") or "").strip()
        gist = _rank_sentences(document.text or "", per_doc)
        header = f"### {title}" if title else "###"
        blocks.append(f"{header}\n{gist}".strip() if gist else header)
    rendered = "Project frame (task anchors — always applies):\n" + "\n".join(blocks)
    # Hard cap: if proportional shares overshot, trim to budget deterministically.
    if count_tokens(rendered) > brief_tokens:
        rendered = extractive_summary(rendered, max_tokens=brief_tokens, focus=rendered[:200])
        rendered = "Project frame (task anchors — always applies):\n" + rendered
    return AnchorBrief(
        text=rendered,
        tokens=count_tokens(rendered),
        sources=source_names,
        document_count=len(real),
        budget_tokens=brief_tokens,
        content_hash=_corpus_hash(real, brief_tokens),
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
