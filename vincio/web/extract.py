"""Token-efficient page reading: a page in, only what matters out — at the
depth the task needs.

A fetched page is worth anything from a whole reference article to a single
sentence, depending on what the model is doing. :func:`extract_page` reads at
four depths, deterministically, with no model call and no dependency beyond the
stdlib parser:

* ``mode="excerpt"`` — only the passages relevant to the query, BM25-ranked and
  packed under a token budget (the default; the token-efficient path).
* ``mode="section"`` — the single heading-delimited section that best matches
  the query, whole, so a definition or a procedure arrives with its context.
* ``mode="full"`` — the entire boilerplate-stripped article (still no nav /
  footer / cookie chrome), capped at a budget.
* ``mode="auto"`` — pick per page: a page that already fits the budget is
  returned whole; a strong section match returns that section; otherwise the
  relevant excerpts. This is what a browsing product does without being told.

Every mode shares one structural pass that also **preserves code blocks**
(``<pre>`` verbatim, fenced — critical for reading library docs) and, on
request, **collects links** (href + anchor) so the crawler can walk a site.

The pipeline is pure: the same bytes, query, budget, and mode always produce
the same :class:`PageExtract`, which is what makes a
:class:`~vincio.web.WebEvidence` re-derivable offline from its snapshot.
"""

from __future__ import annotations

import math
import re
from html.parser import HTMLParser
from typing import Literal

from pydantic import BaseModel, Field

from ..core.tokens import count_tokens

__all__ = [
    "PageExcerpt",
    "PageExtract",
    "PageLink",
    "ExtractMode",
    "extract_page",
    "find_in_page",
]

ExtractMode = Literal["excerpt", "section", "full", "auto"]

#: Bumped whenever the extraction algorithm changes in a way that alters output.
#: Recorded on every :class:`~vincio.web.WebEvidence` so a verify failure caused
#: by an extractor upgrade is diagnosable (version mismatch) rather than silent.
EXTRACTOR_VERSION = "2"

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
     "blockquote", "dd", "dt", "figcaption", "main", "body", "br"}
)
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

_MIN_BLOCK_CHARS = 30
_MIN_CODE_CHARS = 8  # code is dense; a short snippet can still be the answer
_MAX_LINK_DENSITY = 0.5
_LEAD_BLOCKS = 5  # blocks that receive the lead-position prior
_LEAD_BONUS = 0.35
# auto-mode thresholds (fixed, for determinism)
_AUTO_SECTION_MIN_SCORE = 2.0  # a section must clearly match to be returned whole
_SECTION_BUDGET_MULTIPLIER = 3  # a whole section may cost a few excerpts' worth


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


class PageLink(BaseModel):
    """One outbound link found on the page: its absolute URL and anchor text."""

    url: str
    anchor: str = ""


class PageExcerpt(BaseModel):
    """One selected passage, tagged with its nearest heading and its kind."""

    text: str
    section: str = ""
    score: float = 0.0
    kind: Literal["text", "code"] = "text"


class PageExtract(BaseModel):
    """The reading of one page at a chosen depth.

    ``page_tokens`` is what sending the whole boilerplate-stripped page would
    have cost; ``excerpt_tokens`` is what the returned passages actually cost —
    the ratio is the token saving, reported as ``reduction``. ``mode`` records
    the depth the reading was taken at (``auto`` resolves to the concrete depth
    it chose, so a replay is unambiguous).
    """

    url: str = ""
    title: str = ""
    query: str = ""
    mode: ExtractMode = "excerpt"
    budget_tokens: int = 800
    excerpts: list[PageExcerpt] = Field(default_factory=list)
    excerpt_tokens: int = 0
    page_tokens: int = 0
    content_hash: str = ""
    links: list[PageLink] = Field(default_factory=list)
    truncated: bool = False
    #: windows around an exact-string ``find`` lookup, when one was requested
    #: (catches short facts the min-block filter drops); auxiliary to ``excerpts``.
    find_matches: list[PageExcerpt] = Field(default_factory=list)
    #: False when the page is a wall / soft-404 / JS-shell rather than content;
    #: ``unavailable_reason`` says which, so the model can route to another source.
    available: bool = True
    unavailable_reason: str = ""

    @property
    def reduction(self) -> float:
        """How many times cheaper the returned passages are than the full page."""
        return self.page_tokens / self.excerpt_tokens if self.excerpt_tokens else 0.0

    def as_context(self) -> str:
        """Compact, citation-ready rendering for model consumption."""
        lines = [f"[{self.url}] {self.title}".strip()]
        if not self.available:
            lines.append(
                f"(content unavailable: {self.unavailable_reason}; "
                "the page is a wall or shell, not readable content — try another source)"
            )
        last_section = None
        for excerpt in self.excerpts:
            if excerpt.section and excerpt.section != last_section:
                lines.append(f"## {excerpt.section}")
                last_section = excerpt.section
            if excerpt.kind == "code":
                lines.append("```\n" + excerpt.text + "\n```")
            else:
                lines.append(excerpt.text)
        return "\n".join(lines)


class _Block(BaseModel):
    text: str
    section: str
    section_index: int  # which heading-delimited section this block belongs to
    chars: int
    link_chars: int
    index: int
    kind: Literal["text", "code"] = "text"


class _ParsedPage(BaseModel):
    title: str
    blocks: list[_Block]
    links: list[PageLink]


class _PageParser(HTMLParser):
    """One structural pass: content blocks (with heading context and link
    density), verbatim code blocks, and outbound links.

    Skipped (script/style/…) and chrome (nav/footer/`class="menu-…"`/…)
    subtrees are excluded by remembering the *tag that opened the region* and
    counting same-named tags until its matching end tag — so a
    ``<div class="menu-wrapper">`` region ends at its own ``</div>`` instead of
    swallowing the rest of the document. Links are collected everywhere except
    script/style, because a site's navigation *is* how a crawler finds pages.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.blocks: list[_Block] = []
        self.links: list[PageLink] = []
        self._skip_tag: str | None = None
        self._skip_depth = 0
        self._chrome_tag: str | None = None
        self._chrome_depth = 0
        self._link_depth = 0
        self._in_title = False
        self._heading: list[str] | None = None
        self._section = ""
        self._section_index = 0
        self._parts: list[str] = []
        self._link_chars = 0
        # verbatim code capture
        self._pre_depth = 0
        self._code_parts: list[str] = []
        # link capture (href + anchor), independent of block capture
        self._link_href: str | None = None
        self._link_anchor: list[str] = []

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
                    section_index=self._section_index,
                    chars=len(text),
                    link_chars=min(self._link_chars, len(text)),
                    index=len(self.blocks),
                    kind="text",
                )
            )
        self._parts = []
        self._link_chars = 0

    def _flush_code(self) -> None:
        code = "".join(self._code_parts).strip("\n")
        if code.strip():
            self.blocks.append(
                _Block(
                    text=code,
                    section=self._section,
                    section_index=self._section_index,
                    chars=len(code),
                    link_chars=0,
                    index=len(self.blocks),
                    kind="code",
                )
            )
        self._code_parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":  # lives inside <head>, which is otherwise skipped
            self._in_title = True
            return
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        # Link capture runs even inside chrome (nav links matter to the crawler),
        # but never inside a skipped subtree (handled above).
        if tag == "a":
            attr_map = {name: (value or "") for name, value in attrs}
            self._link_href = attr_map.get("href")
            self._link_anchor = []
        if self._chrome_tag is not None:
            if tag == self._chrome_tag:
                self._chrome_depth += 1
            if tag == "a":
                return
            return
        if tag == "pre":
            self._flush()
            self._pre_depth += 1
            return
        if self._pre_depth:
            return  # inside a <pre>: everything is verbatim code, no nested blocks
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

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # A self-closing <a .../> never opens an anchor region; ignore for links.
        if tag not in ("a",):
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag == "a" and self._link_href is not None:
            self._record_link()
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._skip_tag = None
            return
        if tag == "pre" and self._pre_depth:
            self._pre_depth -= 1
            if self._pre_depth <= 0:
                self._flush_code()
            return
        if self._pre_depth:
            return
        if self._chrome_tag is not None:
            if tag == self._chrome_tag:
                self._chrome_depth -= 1
                if self._chrome_depth <= 0:
                    self._chrome_tag = None
            return
        if tag in _HEADING_TAGS and self._heading is not None:
            heading = " ".join(" ".join(self._heading).split())
            self._section_index += 1
            if heading:
                self._section = heading
                if not self.title and tag == "h1":
                    self.title = heading
            self._heading = None
        elif tag == "a":
            self._link_depth = max(0, self._link_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._flush()

    def _record_link(self) -> None:
        href = (self._link_href or "").strip()
        anchor = " ".join(" ".join(self._link_anchor).split())
        self._link_href = None
        self._link_anchor = []
        if href and not href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
            self.links.append(PageLink(url=href, anchor=anchor))

    def handle_data(self, data: str) -> None:
        if self._in_title:
            text = data.strip()
            if text:
                self.title = (self.title + " " + text).strip()
            return
        if self._link_href is not None:
            self._link_anchor.append(data)
        if self._pre_depth:
            self._code_parts.append(data)
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
        if self._pre_depth:
            self._flush_code()
        self._flush()


def _parse_page(html: str) -> _ParsedPage:
    parser = _PageParser()
    parser.feed(html)
    parser.close()
    return _ParsedPage(title=parser.title, blocks=parser.blocks, links=parser.links)


def _content_blocks(blocks: list[_Block]) -> list[_Block]:
    kept: list[_Block] = []
    for block in blocks:
        if block.kind == "code":
            if block.chars >= _MIN_CODE_CHARS:
                kept.append(block)
        elif block.chars >= _MIN_BLOCK_CHARS and (block.link_chars / block.chars) <= _MAX_LINK_DENSITY:
            kept.append(block)
    return kept


def _bm25_scores(blocks: list[_Block], query: str) -> list[float]:
    """Self-contained BM25 (k1=1.5, b=0.75) of each block against the query."""
    terms = _tokenize(query)
    if not terms or not blocks:
        return [0.0] * len(blocks)
    token_lists = [_tokenize(block.text) for block in blocks]
    doc_freq = {term: sum(1 for tokens in token_lists if term in tokens) for term in set(terms)}
    total = len(blocks)
    avg_len = sum(len(tokens) for tokens in token_lists) / total or 1.0
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


def _excerpt(block: _Block, score: float) -> PageExcerpt:
    return PageExcerpt(
        text=block.text, section=block.section, score=round(score, 4), kind=block.kind
    )


def _pack_in_order(
    blocks: list[_Block], scores: list[float], budget: int, max_excerpts: int
) -> tuple[list[PageExcerpt], int, bool]:
    """Keep *blocks* in document order until the budget or the count is hit."""
    excerpts: list[PageExcerpt] = []
    used = 0
    truncated = False
    for block, score in zip(blocks, scores, strict=True):
        if len(excerpts) >= max_excerpts:
            truncated = True
            break
        block_tokens = count_tokens(block.text)
        if used + block_tokens > budget and excerpts:
            truncated = True
            break
        excerpts.append(_excerpt(block, score))
        used += block_tokens
    return excerpts, used, truncated


def _select_excerpts(
    blocks: list[_Block], scores: list[float], budget: int, max_excerpts: int
) -> tuple[list[PageExcerpt], int]:
    """The token-efficient path: the highest-scoring blocks, budget-packed,
    returned in document order."""
    candidates = list(zip(scores, blocks, strict=True))
    if any(score > 0 for score in scores):
        candidates = [pair for pair in candidates if pair[0] > 0]
    ranked = sorted(
        (
            (score + (_LEAD_BONUS if block.index < _LEAD_BLOCKS else 0.0), block)
            for score, block in candidates
        ),
        key=lambda pair: (-pair[0], pair[1].index),
    )
    selected: list[tuple[float, _Block]] = []
    used = 0
    for score, block in ranked:
        if len(selected) >= max_excerpts:
            break
        block_tokens = count_tokens(block.text)
        if block_tokens > budget and selected:
            continue
        if used + block_tokens > budget and selected:
            continue
        selected.append((score, block))
        used += block_tokens
        if used >= budget:
            break
    selected.sort(key=lambda pair: pair[1].index)
    return [_excerpt(block, score) for score, block in selected], used


_HEADING_TERM_BONUS = 2.0  # per distinct query term appearing in a section heading

# Content-availability lexicon: markers that a "page" is really a wall or shell.
# Wall/paywall/login/404 markers only fire on a *short* page (a real article that
# merely discusses cookies must not read as a cookie wall).
_UNAVAILABLE_MARKERS = (
    ("requires_javascript", ("enable javascript", "requires javascript",
                             "javascript is disabled", "please enable js")),
    ("cookie_wall", ("we value your privacy", "accept all cookies",
                     "accept cookies to continue", "manage your cookie",
                     "this site uses cookies", "cookie preferences")),
    ("paywall", ("subscribe to continue", "subscribe to read", "subscribers only",
                 "become a member to", "this article is for subscribers",
                 "to continue reading")),
    ("login_required", ("sign in to continue", "log in to continue",
                        "please log in", "create an account to", "you must be logged in")),
    ("not_found", ("page not found", "404 not found", "page you requested",
                   "no longer available", "page doesn't exist", "page does not exist")),
)
_WALL_MAX_BLOCKS = 4  # a wall is a short page; longer pages are treated as content


def _detect_availability(
    html: str, title: str, blocks: list[_Block]
) -> tuple[bool, str]:
    """A deterministic read of whether the fetched page is real content or a
    wall / soft-404 / JS shell the model should route around."""
    # A 200 with substantial markup but almost no extractable content is a
    # client-rendered shell (the content arrives via JavaScript we do not run).
    if len(blocks) < 2 and len(html) > 20_000:
        return False, "requires_javascript"
    if len(blocks) <= _WALL_MAX_BLOCKS:
        haystack = " ".join([title, *(b.text for b in blocks[:_WALL_MAX_BLOCKS])]).lower()
        for reason, markers in _UNAVAILABLE_MARKERS:
            if any(marker in haystack for marker in markers):
                return False, reason
    return True, ""


def _best_section(
    blocks: list[_Block], scores: list[float], query: str = ""
) -> tuple[int, float]:
    """The section index with the highest score, and that score.

    A section's score is the sum of its blocks' BM25 scores plus a bonus for
    each distinct query term that appears in the section's *heading* — a heading
    match ("Make a Request" for `make a request`) is a strong topicality signal
    that raw body density can otherwise miss (a lead paragraph that merely
    repeats the query words should not outrank the section actually about it).
    """
    query_terms = set(_tokenize(query))
    by_section: dict[int, float] = {}
    heading_of: dict[int, str] = {}
    for block, score in zip(blocks, scores, strict=True):
        by_section[block.section_index] = by_section.get(block.section_index, 0.0) + score
        heading_of.setdefault(block.section_index, block.section)
    if not by_section:
        return -1, 0.0
    if query_terms:
        for index, heading in heading_of.items():
            matched = query_terms & set(_tokenize(heading))
            by_section[index] += _HEADING_TERM_BONUS * len(matched)
    # earliest section wins ties (deterministic)
    best_index = min(by_section, key=lambda idx: (-by_section[idx], idx))
    return best_index, by_section[best_index]


def find_in_page(
    html: str, needle: str, *, window_chars: int = 220, max_matches: int = 5
) -> list[PageExcerpt]:
    """Windows of text around each occurrence of *needle* in the page.

    Searches the full parsed text (including blocks below the excerpt-size
    filter), so a short fact — a version number, a date, a config key — that
    normal extraction would drop is still findable. Case-insensitive,
    deterministic, capped at *max_matches*."""
    if not needle.strip():
        return []
    joined = "\n".join(block.text for block in _parse_page(html).blocks)
    lowered = joined.lower()
    target = needle.lower()
    matches: list[PageExcerpt] = []
    start = 0
    while len(matches) < max_matches:
        at = lowered.find(target, start)
        if at < 0:
            break
        left = max(0, at - window_chars)
        right = min(len(joined), at + len(needle) + window_chars)
        window = joined[left:right].strip()
        matches.append(PageExcerpt(text=window, score=1.0, kind="text"))
        start = right
    return matches


def extract_page(
    html: str,
    *,
    url: str = "",
    query: str = "",
    budget_tokens: int = 800,
    max_excerpts: int = 12,
    mode: ExtractMode = "excerpt",
    collect_links: bool = False,
    find: str = "",
) -> PageExtract:
    """Reduce *html* to what the task needs at the chosen *mode*.

    ``mode`` is ``"excerpt"`` (query-relevant passages, default), ``"section"``
    (the best-matching whole section), ``"full"`` (the whole article, budget-
    capped), or ``"auto"`` (choose per page). With an empty query the extraction
    degrades to the article lead. Set ``collect_links=True`` to also return the
    page's outbound links (for crawling). Deterministic in
    (``html``, ``query``, ``budget_tokens``, ``max_excerpts``, ``mode``).
    """
    parsed = _parse_page(html)
    blocks = _content_blocks(parsed.blocks)
    page_tokens = count_tokens("\n".join(block.text for block in blocks))
    scores = _bm25_scores(blocks, query)
    links = list(parsed.links) if collect_links else []

    resolved: ExtractMode = mode
    if mode == "auto":
        _, best_score = _best_section(blocks, scores, query)
        if page_tokens <= budget_tokens:
            resolved = "full"
        elif query and best_score >= _AUTO_SECTION_MIN_SCORE:
            resolved = "section"
        else:
            resolved = "excerpt"

    truncated = False
    if resolved == "full":
        excerpts, used, truncated = _pack_in_order(blocks, scores, budget_tokens, max_excerpts)
    elif resolved == "section":
        best_index, _ = _best_section(blocks, scores, query)
        section_blocks = [b for b in blocks if b.section_index == best_index]
        section_scores = [s for b, s in zip(blocks, scores, strict=True) if b.section_index == best_index]
        section_budget = budget_tokens * _SECTION_BUDGET_MULTIPLIER
        excerpts, used, truncated = _pack_in_order(
            section_blocks, section_scores, section_budget, max_excerpts
        )
        if not excerpts:  # no headings on the page: fall back to excerpts
            excerpts, used = _select_excerpts(blocks, scores, budget_tokens, max_excerpts)
    else:  # excerpt
        excerpts, used = _select_excerpts(blocks, scores, budget_tokens, max_excerpts)

    available, reason = _detect_availability(html, parsed.title, blocks)
    find_matches = find_in_page(html, find) if find else []
    return PageExtract(
        url=url,
        title=parsed.title,
        query=query,
        mode=mode,
        budget_tokens=budget_tokens,
        excerpts=excerpts,
        excerpt_tokens=used,
        page_tokens=page_tokens,
        links=links,
        truncated=truncated,
        find_matches=find_matches,
        available=available,
        unavailable_reason=reason,
    )
