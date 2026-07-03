"""Token-efficient page reading: a page in, only the passages that matter out.

A fetched page is worth a handful of sentences to the model, not its markup.
The median page runs to tens of thousands of tokens of HTML; the fact the
model needs is usually one paragraph. :func:`extract_page` closes that gap
deterministically, with no model call and no dependency beyond the stdlib
parser:

1. **Structure** — a tolerant :mod:`html.parser` pass collects text blocks
   with their nearest heading, skipping script/style/template subtrees.
2. **Boilerplate** — navigation, header, footer, aside, and form subtrees are
   dropped, as is any block that is mostly link text (menus, tag clouds,
   "related articles" rails).
3. **Relevance** — remaining blocks are scored against the query with a
   self-contained BM25 (plus a small lead-position prior, so query-free
   extraction degrades to the article lead).
4. **Budget** — the best blocks are packed under ``budget_tokens`` (exact
   token counting via :func:`~vincio.core.tokens.count_tokens`) and emitted in
   document order as :class:`PageExcerpt` rows.

The pipeline is pure: the same bytes, query, and budget always produce the
same :class:`PageExtract`, which is what makes a
:class:`~vincio.web.WebEvidence` re-derivable offline from its snapshot.
"""

from __future__ import annotations

import math
import re
from html.parser import HTMLParser

from pydantic import BaseModel, Field

from ..core.tokens import count_tokens

__all__ = ["PageExcerpt", "PageExtract", "extract_page"]

_WORD_RE = re.compile(r"[a-z0-9]+")

# Subtrees that never contain readable content.
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "iframe", "head"})
# Subtrees that are page chrome, not content.
_CHROME_TAGS = frozenset({"nav", "header", "footer", "aside", "form"})
# class/id fragments that mark chrome even on a neutral tag...
_CHROME_HINTS = ("navbar", "sidebar", "cookie", "breadcrumb", "menu-", "-menu")
# ...but only on real container tags, so an unclosed void tag (an <img> or
# <input> with a matching class) can never open a region that never ends.
_CONTAINER_TAGS = frozenset({"div", "section", "span", "ul", "ol", "li", "table"})
# Tags that terminate the current text block.
_BLOCK_TAGS = frozenset(
    {"p", "div", "li", "td", "th", "tr", "ul", "ol", "table", "section", "article",
     "blockquote", "pre", "dd", "dt", "figcaption", "main", "body", "br"}
)
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

_MIN_BLOCK_CHARS = 30
_MAX_LINK_DENSITY = 0.5
_LEAD_BLOCKS = 5  # blocks that receive the lead-position prior
_LEAD_BONUS = 0.35


def _stem(token: str) -> str:
    """Tiny deterministic suffix stripper: 'released', 'releases', and
    'release' all normalize to 'releas'. Ranking-only, so light is right."""
    if len(token) > 3:
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                token = token[: -len(suffix)]
                break
        if token.endswith("e") and len(token) > 3:
            token = token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    return [_stem(token) for token in _WORD_RE.findall(text.lower())]


class PageExcerpt(BaseModel):
    """One budget-selected passage, tagged with its nearest heading."""

    text: str
    section: str = ""
    score: float = 0.0


class PageExtract(BaseModel):
    """The token-budgeted reading of one page.

    ``page_tokens`` is what sending the whole boilerplate-stripped page would
    have cost; ``excerpt_tokens`` is what the selected passages actually cost —
    the ratio is the extraction's token saving, reported as ``reduction``.
    """

    url: str = ""
    title: str = ""
    query: str = ""
    budget_tokens: int = 800
    excerpts: list[PageExcerpt] = Field(default_factory=list)
    excerpt_tokens: int = 0
    page_tokens: int = 0
    content_hash: str = ""

    @property
    def reduction(self) -> float:
        """How many times cheaper the excerpts are than the full page text."""
        return self.page_tokens / self.excerpt_tokens if self.excerpt_tokens else 0.0

    def as_context(self) -> str:
        """Compact, citation-ready rendering for model consumption."""
        lines = [f"[{self.url}] {self.title}".strip()]
        last_section = None
        for excerpt in self.excerpts:
            if excerpt.section and excerpt.section != last_section:
                lines.append(f"## {excerpt.section}")
                last_section = excerpt.section
            lines.append(excerpt.text)
        return "\n".join(lines)


class _Block(BaseModel):
    text: str
    section: str
    chars: int
    link_chars: int
    index: int


class _BlockCollector(HTMLParser):
    """Collect content text blocks with heading context and link density.

    Skipped (script/style/…) and chrome (nav/footer/`class="menu-…"`/…)
    subtrees are excluded by remembering the *tag that opened the region* and
    counting same-named tags until its matching end tag — so a
    ``<div class="menu-wrapper">`` region ends at its own ``</div>`` instead of
    swallowing the rest of the document.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.blocks: list[_Block] = []
        self._skip_tag: str | None = None
        self._skip_depth = 0
        self._chrome_tag: str | None = None
        self._chrome_depth = 0
        self._link_depth = 0
        self._in_title = False
        self._heading: list[str] | None = None
        self._section = ""
        self._parts: list[str] = []
        self._link_chars = 0

    @property
    def _excluded(self) -> bool:
        return self._skip_tag is not None or self._chrome_tag is not None

    def _flush(self) -> None:
        text = " ".join(" ".join(self._parts).split())
        if text:
            self.blocks.append(
                _Block(
                    text=text,
                    section=self._section,
                    chars=len(text),
                    link_chars=min(self._link_chars, len(text)),
                    index=len(self.blocks),
                )
            )
        self._parts = []
        self._link_chars = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":  # lives inside <head>, which is otherwise skipped
            self._in_title = True
            return
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if self._chrome_tag is not None:
            if tag == self._chrome_tag:
                self._chrome_depth += 1
            return
        if tag in _SKIP_TAGS:
            self._skip_tag, self._skip_depth = tag, 1
            return
        marker = " ".join(value or "" for name, value in attrs if name in ("class", "id")).lower()
        if tag in _CHROME_TAGS or (
            tag in _CONTAINER_TAGS and any(hint in marker for hint in _CHROME_HINTS)
        ):
            self._flush()
            self._chrome_tag, self._chrome_depth = tag, 1
            return
        if tag in _HEADING_TAGS:
            self._flush()
            self._heading = []
        elif tag == "a":
            self._link_depth += 1
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._skip_tag = None
            return
        if self._chrome_tag is not None:
            if tag == self._chrome_tag:
                self._chrome_depth -= 1
                if self._chrome_depth <= 0:
                    self._chrome_tag = None
            return
        if tag in _HEADING_TAGS and self._heading is not None:
            heading = " ".join(" ".join(self._heading).split())
            if heading:
                self._section = heading
                if not self.title and tag == "h1":
                    self.title = heading
            self._heading = None
        elif tag == "a":
            self._link_depth = max(0, self._link_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            text = data.strip()
            if text:
                self.title = (self.title + " " + text).strip()
            return
        if self._excluded:
            return
        if self._heading is not None:
            self._heading.append(data)
        else:
            self._parts.append(data)
            if self._link_depth:
                self._link_chars += len(data.strip())

    def close(self) -> None:
        super().close()
        self._flush()


def _content_blocks(html: str) -> tuple[str, list[_Block]]:
    collector = _BlockCollector()
    collector.feed(html)
    collector.close()
    kept = [
        block
        for block in collector.blocks
        if block.chars >= _MIN_BLOCK_CHARS
        and (block.link_chars / block.chars) <= _MAX_LINK_DENSITY
    ]
    return collector.title, kept


def _bm25_scores(blocks: list[_Block], query: str) -> list[float]:
    """Self-contained BM25 (k1=1.5, b=0.75) of each block against the query."""
    terms = _tokenize(query)
    if not terms or not blocks:
        return [0.0] * len(blocks)
    token_lists = [_tokenize(block.text) for block in blocks]
    doc_freq = {term: sum(1 for tokens in token_lists if term in tokens) for term in set(terms)}
    total = len(blocks)
    avg_len = sum(len(tokens) for tokens in token_lists) / total
    k1, b = 1.5, 0.75
    scores: list[float] = []
    for tokens in token_lists:
        length = len(tokens) or 1
        score = 0.0
        for term in terms:
            tf = tokens.count(term)
            if not tf:
                continue
            df = doc_freq[term]
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_len))
        scores.append(score)
    return scores


def extract_page(
    html: str,
    *,
    url: str = "",
    query: str = "",
    budget_tokens: int = 800,
    max_excerpts: int = 12,
) -> PageExtract:
    """Reduce *html* to the passages most relevant to *query*, under a budget.

    With an empty query the lead-position prior alone ranks blocks, so the
    extraction degrades to the article lead rather than failing. The result is
    deterministic in (``html``, ``query``, ``budget_tokens``, ``max_excerpts``).
    """
    title, blocks = _content_blocks(html)
    page_tokens = count_tokens("\n".join(block.text for block in blocks))
    scores = _bm25_scores(blocks, query)
    candidates = list(zip(scores, blocks, strict=True))
    if any(score > 0 for score in scores):
        # The query matched: spend the budget only on matching passages —
        # a smaller answer beats a padded one. A missed query falls through
        # to the lead-ordered fill below, so extraction never returns nothing.
        candidates = [pair for pair in candidates if pair[0] > 0]
    ranked = sorted(
        (
            (score + (_LEAD_BONUS if block.index < _LEAD_BLOCKS else 0.0), block)
            for score, block in candidates
        ),
        key=lambda pair: (-pair[0], pair[1].index),
    )
    selected: list[tuple[float, _Block]] = []
    used_tokens = 0
    for score, block in ranked:
        if len(selected) >= max_excerpts:
            break
        block_tokens = count_tokens(block.text)
        if used_tokens + block_tokens > budget_tokens and selected:
            continue
        if block_tokens > budget_tokens and selected:
            continue
        selected.append((score, block))
        used_tokens += block_tokens
        if used_tokens >= budget_tokens:
            break
    selected.sort(key=lambda pair: pair[1].index)
    excerpts = [
        PageExcerpt(text=block.text, section=block.section, score=round(score, 4))
        for score, block in selected
    ]
    return PageExtract(
        url=url,
        title=title,
        query=query,
        budget_tokens=budget_tokens,
        excerpts=excerpts,
        excerpt_tokens=used_tokens,
        page_tokens=page_tokens,
    )
