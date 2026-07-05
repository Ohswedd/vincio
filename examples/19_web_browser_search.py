"""Universal web browsing & search — every model gets the open web, governed.

One line — ``app.use_web_search()`` — gives *any* model two tools, ``web_search``
and ``web_read``, executed by Vincio itself. A frontier model calls them natively;
a local model without function calling calls the *same* tools through Vincio's text
protocol — same loop, same governance, same evidence. Pages are never forwarded:
``web_read`` returns only the passages relevant to the model's query under a token
budget, and every read is content-hashed so the session verifies offline from bytes.

This tour runs **fully offline**: the "web" is an injected httpx transport serving
a recorded page, and the models are scripted mocks. For the real thing, drop the
``backend=`` / ``client=`` arguments and Vincio queries DuckDuckGo directly.
"""

from __future__ import annotations

import httpx

from vincio.core.app import ContextApp
from vincio.core.config import VincioConfig
from vincio.core.errors import WebPolicyError
from vincio.core.types import ModelCapabilities
from vincio.providers import MockProvider
from vincio.web import SearchResult, StaticSearchBackend, WebBrowser, WebPolicy

RELEASE_URL = "https://www.python.org/downloads/release/python-3130/"

# -- The offline web: one padded page behind an injected httpx transport. The real
#    signal (the release date) is buried in 150 unrelated sections, so web_read has
#    real work to do reducing the page to the passages that answer the query.
PAGE = (
    "<html><head><title>Python 3.13 Release</title></head><body>"
    "<h1>Python 3.13.0</h1>"
    "<p>Python 3.13.0 is the newest major release of the Python programming language.</p>"
    "<h2>Release date</h2>"
    "<p>Python 3.13.0 was released on October 7, 2024, and ships a new interactive "
    "interpreter plus experimental free-threading support.</p>"
    + "".join(f"<p>Long unrelated section {i} about the history of typography, "
              "padding this page the way real pages are padded.</p>" for i in range(150))
    + "<footer><a href='/legal'>Legal</a></footer></body></html>"
)


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/robots.txt":
        return httpx.Response(404)
    return httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


def _backend() -> StaticSearchBackend:
    return StaticSearchBackend({"latest python version release": [SearchResult(
        rank=1, title="Python Release Python 3.13.0", url=RELEASE_URL,
        snippet="the newest major release of Python")]})


def _config(tag: str) -> VincioConfig:
    import tempfile

    tmp = tempfile.mkdtemp(prefix=f"vincio_web_{tag}_")
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp}/vincio.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = f"{tmp}/audit"
    return config


QUESTION = "What is the latest Python version and when was it released?"
ANSWER = "Python 3.13.0, released October 7, 2024. Source: " + RELEASE_URL

# A native tool-calling model emits structured tool_calls; a tools-blind model emits
# the *same* two calls as fenced text. Vincio runs both through one identical loop.
NATIVE_SCRIPT = [
    {"tool_call": {"name": "web_search", "arguments": {"query": "latest python version release"}}},
    {"tool_call": {"name": "web_read", "arguments": {"url": RELEASE_URL, "query": "release date"}}},
    ANSWER,
]
PROTOCOL_SCRIPT = [
    '```tool_call\n{"name": "web_search", "arguments": {"query": "latest python version release"}}\n```',
    '```tool_call\n{"name": "web_read", "arguments": {"url": "' + RELEASE_URL + '", "query": "release date"}}\n```',
    ANSWER,
]


class LocalGGUFMock(MockProvider):
    """A stand-in for a local model with NO native function calling."""

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(tool_calling=False, structured_output=False)


def main() -> None:
    # 1. Drive the browser directly: search, then a token-budgeted read. web_read
    #    forwards only the relevant excerpts under a budget — never the whole page —
    #    which is what makes browsing affordable at every step of a loop.
    browser = WebBrowser(_backend(), client=_client())
    hits = browser.search_sync("latest python version release")
    page = browser.read_sync(hits[0].url, query="release date", budget_tokens=200)
    print(f"1. Search + read — top hit: {hits[0].title}")
    print(f"   page {page.page_tokens} tokens → excerpts {page.excerpt_tokens} "
          f"({page.reduction:.0f}x cheaper): {page.excerpts[0].text[:64]}…")

    # 2. A native tool-calling model runs the loop through the standard app.run —
    #    Vincio executes the tools and feeds results back until the model answers.
    app = ContextApp(name="web_native", model="mock-1", config=_config("native"),
                     provider=MockProvider(script=list(NATIVE_SCRIPT)))
    app.use_web_search(backend=_backend(), client=_client())
    result = app.run(QUESTION)
    print(f"\n2. Native model — tools {[t.tool_name for t in result.tool_results]} → {result.raw_text}")

    # 3. A local model with NO native tool calling runs the *identical* loop: Vincio
    #    lowers the tools into a text protocol and lifts the calls back out, so one
    #    governed browsing path serves frontier and tools-blind models alike.
    app2 = ContextApp(name="web_local", model="mock-1", config=_config("local"),
                      provider=LocalGGUFMock(script=list(PROTOCOL_SCRIPT)))
    app2.use_web_search(backend=_backend(), client=_client())
    result2 = app2.run(QUESTION)
    print(f"\n3. Tools-blind model — tools {[t.tool_name for t in result2.tool_results]} → {result2.raw_text}")
    assert [t.tool_name for t in result.tool_results] == [t.tool_name for t in result2.tool_results]

    # 4. Governance is pre-egress: SSRF targets and denied domains are refused before
    #    any request leaves the process, and every action lands on the audit trail.
    guarded = WebBrowser(_backend(), client=_client(),
                         policy=WebPolicy(max_searches=1, deny_domains=["tracker.example"]))
    print("\n4. Governance (pre-egress refusal)")
    for url in ("http://169.254.169.254/metadata", "https://ads.tracker.example/p"):
        try:
            guarded.read_sync(url)
        except WebPolicyError as exc:
            print(f"   refused {url} ({exc.code})")
    print(f"   audit trail: {[e.action for e in app.audit.entries if e.action.startswith('web_')]}")

    # 5. Proof: the session re-derives offline from its content-hashed snapshots, and
    #    a single mutated byte is detected — evidence you can attach to a review.
    session = app.web_browser.report()
    evidence = session.reads[0]
    print(f"\n5. Offline verification — {session.searches_used} search, {session.fetches_used} read; "
          f"snapshot {evidence.content_hash[:12]}…")
    print(f"   session verifies from bytes={session.verify(app.web_browser.snapshots)}; "
          f"tamper detected={not evidence.verify(PAGE + '<!-- tampered -->')}")

    # 6. Crawl a bounded subtree into a verifiable collection → retrieval documents
    #    or a Dataset. Bounds (max_crawl_pages) and stop reasons keep it deterministic.
    crawl_pages = {
        "/docs/": "<html><head><title>Docs</title></head><body><h1>Home</h1>"
                  "<p>The documentation index, a real readable block of content here.</p>"
                  "<nav><a href='/docs/guide'>Guide</a></nav></body></html>",
        "/docs/guide": "<html><head><title>Guide</title></head><body><h1>Guide</h1>"
                       "<p>A guide page with enough readable content to be a real block.</p></body></html>",
    }

    def crawl_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        body = crawl_pages.get(request.url.path)
        return (httpx.Response(200, text=body, headers={"content-type": "text/html"})
                if body else httpx.Response(404))

    collection = app.web_crawl(
        "https://docs.example.com/docs/", scope="subtree",
        policy=WebPolicy.preset("scrape", max_crawl_pages=5),
        client=httpx.AsyncClient(transport=httpx.MockTransport(crawl_handler)))
    print(f"\n6. Crawl → collection — {collection.pages_fetched} page(s) "
          f"(stopped: {collection.stopped_reason}) → {len(collection.to_documents())} documents "
          f"or a {len(collection.to_dataset().columns)}-column Dataset")

    print("\nDone — every model browses: governed, token-efficient, and provable.")


if __name__ == "__main__":
    main()
