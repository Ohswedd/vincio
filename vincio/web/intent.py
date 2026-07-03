"""Reading the user's intent to browse — deterministically, before any model call.

Two prompt-level signals justify reaching for the web without the model having
to ask for a tool round:

* the user **pastes a URL or a bare domain** ("summarize https://… ", "what does
  numpy.org say about …") — the page is the obvious evidence, so
  :func:`extract_urls` finds it and the browser auto-fetches it;
* the user **directs a search or scopes one to a site** ("search for …",
  "look it up on python.org", "site:docs.rust-lang.org …") —
  :func:`detect_web_intent` surfaces that as a structured
  :class:`WebIntent` the caller can act on.

Everything here is a conservative, offline regex pass over the prompt text: it
never runs on its own, only *offers* signals a governed caller (the runtime
auto-fetch hook, a tool, the research verb) chooses to use. False positives are
cheap (a policy-gated fetch that finds nothing) but kept low by requiring a real
host with a dot and a known-ish TLD shape.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

__all__ = ["WebIntent", "detect_web_intent", "extract_urls", "urls_to_fetch"]

# A full URL with scheme.
_URL_RE = re.compile(r"https?://[^\s<>\)\]\"'}]+", re.IGNORECASE)
# A bare host ("www.example.com", "docs.python.org/3", "numpy.org") not preceded
# by '@' (so it never eats an email local part) — requires a dotted host with a
# 2+ letter final label.
_BARE_HOST_RE = re.compile(
    r"(?<![@\w.])((?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24})"
    r"(/[^\s<>\)\]\"'}]*)?",
    re.IGNORECASE,
)
_SITE_SCOPE_RE = re.compile(
    r"\bsite:([a-z0-9.-]+\.[a-z]{2,24})", re.IGNORECASE
)
_ON_SITE_RE = re.compile(
    r"\b(?:on|from|at|in|via|using|check)\s+((?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+)",
    re.IGNORECASE,
)
_SEARCH_DIRECTIVE_RE = re.compile(
    r"\b(search|google|look\s+up|look\s+it\s+up|find\s+(?:me\s+)?(?:online|on the web)|"
    r"web\s*search|browse|check\s+online|latest|current(?:ly)?|as\s+of\s+(?:today|now))\b",
    re.IGNORECASE,
)
# Verbs that mean "go fetch this page for me" — the gate for auto-fetch, so a URL
# merely *discussed* ("what does GET http://…/ return?") is never fetched.
_FETCH_DIRECTIVE_RE = re.compile(
    r"\b(summar\w+|read|open|fetch|load|browse|scrape|visit|go\s+to|look\s+at|"
    r"check\s+out|according\s+to|based\s+on|from\s+(?:this|the)\s+(?:page|url|link|article|site)|"
    r"what\s+does\s+(?:this|it|the\s+(?:page|link|article))\s+say|tl;?dr)\b",
    re.IGNORECASE,
)
# Regions whose URLs are examples/code, never fetch targets.
_CODE_FENCE_RE = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)
_MAX_RESIDUAL_CHARS = 16  # prompt minus its URLs this short ⇒ the URL is the ask
# Common trailing punctuation to peel off a matched URL/host.
_TRAILING = ".,;:!?)]}>\"'"
# Hosts that are almost never a browse target when bare (avoid false positives on
# casual mentions); an explicit scheme still fetches them.
_COMMON_FALSE_POSITIVES = frozenset({"e.g", "i.e", "etc.al", "vs.the"})


class WebIntent(BaseModel):
    """The browse signals found in a prompt (all optional, all advisory)."""

    urls: list[str] = Field(default_factory=list)
    sites: list[str] = Field(default_factory=list)
    wants_search: bool = False

    @property
    def any(self) -> bool:
        return bool(self.urls or self.sites or self.wants_search)


def _clean(token: str) -> str:
    return token.rstrip(_TRAILING)


def extract_urls(text: str, *, limit: int = 8) -> list[str]:
    """The fetchable URLs a user pasted, in order, deduplicated.

    Full ``http(s)://`` URLs are taken as-is; a bare dotted host (``numpy.org``,
    ``docs.python.org/3``) is promoted to ``https://``. Order is preserved and
    duplicates dropped, capped at *limit*.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []

    def add(url: str) -> None:
        url = _clean(url)
        if url and url not in seen:
            seen.add(url)
            out.append(url)

    # Full URLs first; blank them out so the bare-host pass does not re-match.
    remaining = text
    for match in _URL_RE.finditer(text):
        add(match.group(0))
        remaining = remaining.replace(match.group(0), " ")
    for match in _BARE_HOST_RE.finditer(remaining):
        host = match.group(1).lower().rstrip(".")
        if host in _COMMON_FALSE_POSITIVES:
            continue
        add("https://" + _clean(match.group(0)))
    return out[:limit]


def urls_to_fetch(text: str, *, limit: int = 3) -> list[str]:
    """The URLs the user actually wants fetched — the conservative subset safe
    to auto-fetch without a model round-trip.

    A URL is returned only when the prompt *directs* a fetch ("summarize …",
    "read …", "according to …") or the URL is essentially the whole ask (a
    pasted link with little else). URLs inside code fences are ignored, and a
    URL that is merely mentioned or discussed ("what does GET http://…/ do?") is
    left for the model to fetch deliberately via the tool. This is the trust
    gate for :meth:`~vincio.web.WebBrowser.evidence_for`.
    """
    if not text:
        return []
    cleaned = _CODE_FENCE_RE.sub(" ", text)
    urls = extract_urls(cleaned, limit=limit)
    if not urls:
        return []
    residual = cleaned
    for match in _URL_RE.finditer(cleaned):
        residual = residual.replace(match.group(0), " ")
    for match in _BARE_HOST_RE.finditer(residual):
        residual = residual.replace(match.group(0), " ")
    residual = residual.strip()
    if _FETCH_DIRECTIVE_RE.search(cleaned) or len(residual) <= _MAX_RESIDUAL_CHARS:
        return urls
    return []


def detect_web_intent(text: str) -> WebIntent:
    """Structured browse signals in *text*: pasted URLs, scoped sites, and
    whether the user directed a search."""
    urls = extract_urls(text)
    sites: list[str] = []
    seen: set[str] = set()
    for pattern in (_SITE_SCOPE_RE, _ON_SITE_RE):
        for match in pattern.finditer(text or ""):
            site = match.group(1).lower().rstrip(_TRAILING)
            if "." in site and site not in seen:
                seen.add(site)
                sites.append(site)
    wants_search = bool(_SEARCH_DIRECTIVE_RE.search(text or ""))
    return WebIntent(urls=urls, sites=sites, wants_search=wants_search)
