"""Universal web browsing & search: one governed plane for every model.

Some providers ship a hosted web-search tool, some ship a poor one, and local
models ship none. This subpackage levels that: **any** model Vincio serves —
hosted frontier, OpenAI-compatible gateway, or a llama.cpp GGUF on this
machine — gets the same two tools, ``web_search`` and ``web_read``, executed
by Vincio itself against the open web (DuckDuckGo by default, any
:class:`SearchBackend` by contract) and governed like every other action.

The plane is built around three commitments:

* **Token efficiency** — a page is never forwarded; :func:`extract_page`
  reduces it to the passages relevant to the model's own query under an exact
  token budget (boilerplate-stripped, BM25-ranked, deterministic).
* **Judgement, natively** — when to search, how to write queries, and when to
  stop ship as the built-in :func:`browse_skill`, disclosed to the model
  through the skill library's progressive-disclosure path — the same contract
  for every provider.
* **Proof** — every read lands as a content-hashed :class:`WebEvidence` whose
  excerpts re-derive offline from the snapshot; the :class:`WebSessionReport`
  makes a whole browsing session verifiable from bytes, and every search and
  fetch records on the app's hash-chained audit log.

Enable it in one line — ``app.use_web_search()`` — or drive it directly::

    from vincio.web import WebBrowser, WebPolicy

    browser = WebBrowser(policy=WebPolicy(max_searches=4))
    hits = browser.search_sync("python 3.13 release date")
    page = browser.read_sync(hits[0].url, query="release date")
    assert browser.report().verify(browser.snapshots)
"""

from __future__ import annotations

from .browser import SearchRecord, WebBrowser, WebEvidence, WebSessionReport
from .extract import PageExcerpt, PageExtract, extract_page
from .policy import WebPolicy
from .search import (
    DuckDuckGoBackend,
    SearchBackend,
    SearchResult,
    StaticSearchBackend,
    parse_results_html,
)
from .skill import browse_skill

__all__ = [
    "DuckDuckGoBackend",
    "PageExcerpt",
    "PageExtract",
    "SearchBackend",
    "SearchRecord",
    "SearchResult",
    "StaticSearchBackend",
    "WebBrowser",
    "WebEvidence",
    "WebPolicy",
    "WebSessionReport",
    "browse_skill",
    "extract_page",
    "parse_results_html",
]
