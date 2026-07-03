"""Universal web browsing & search — every model gets the open web, governed.

One line — ``app.use_web_search()`` — gives *any* model Vincio serves two
tools, ``web_search`` and ``web_read``, executed by Vincio itself against
DuckDuckGo (or any pluggable backend). A frontier model calls them natively;
a local GGUF model without function calling calls the *same* tools through
Vincio's text protocol — same loop, same governance, same evidence. Pages are
never forwarded: ``web_read`` returns only the passages relevant to the
model's own query under an exact token budget, and every read is snapshotted
and content-hashed so the whole session verifies offline from bytes.

This tour runs **fully offline**: the "web" is an injected transport serving
a recorded page, the engine is the deterministic static backend, and the
models are scripted mocks — so you can see exactly what the live path does.
For the real thing, drop the ``backend=`` / ``client=`` arguments and Vincio
queries DuckDuckGo's keyless endpoints directly.

Sections:
  1. Search + token-efficient read, driven directly.
  2. A native tool-calling model runs the loop through ``app.run``.
  3. A model with NO native tool calling runs the *identical* loop.
  4. Governance: pre-egress refusal, budgets, and the audit trail.
  5. Proof: the session's evidence re-derives offline from its snapshots.
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

# -- the recorded web: one page behind an injected transport -------------------

PAGE = (
    "<html><head><title>Python 3.13 Release</title></head><body>"
    "<nav><a href='/'>Home</a><a href='/downloads'>Downloads</a></nav>"
    "<h1>Python 3.13.0</h1>"
    "<p>Python 3.13.0 is the newest major release of the Python programming language.</p>"
    "<h2>Release date</h2>"
    "<p>Python 3.13.0 was released on October 7, 2024, and ships a new interactive "
    "interpreter plus experimental free-threading support.</p>"
    + "".join(
        f"<p>Long unrelated section {i} about the history of typography and paper "
        "sizes, padding this page the way real pages are padded.</p>"
        for i in range(150)
    )
    + "<footer><a href='/legal'>Legal</a></footer></body></html>"
)


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/robots.txt":
        return httpx.Response(404)
    return httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


def _backend() -> StaticSearchBackend:
    return StaticSearchBackend(
        {
            "latest python version release": [
                SearchResult(
                    rank=1,
                    title="Python Release Python 3.13.0",
                    url=RELEASE_URL,
                    snippet="the newest major release of Python",
                )
            ]
        }
    )


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

NATIVE_SCRIPT = [
    {"tool_call": {"name": "web_search", "arguments": {"query": "latest python version release"}}},
    {"tool_call": {"name": "web_read", "arguments": {"url": RELEASE_URL, "query": "release date"}}},
    ANSWER,
]
# The same turns, as the text protocol a tools-blind model emits:
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
    # 1. Drive the browser directly: search, then a token-budgeted read.
    browser = WebBrowser(_backend(), client=_client())
    hits = browser.search_sync("latest python version release")
    page = browser.read_sync(hits[0].url, query="release date", budget_tokens=200)
    print("1. Search + read")
    print(f"   top hit: {hits[0].title} ({hits[0].url})")
    print(f"   page is {page.page_tokens} tokens; excerpts are {page.excerpt_tokens} "
          f"({page.reduction:.0f}x cheaper):")
    for excerpt in page.excerpts:
        print(f"     [{excerpt.section}] {excerpt.text[:72]}…")

    # 2. A native tool-calling model, through the standard run loop.
    app = ContextApp(
        name="web_native", provider=MockProvider(script=list(NATIVE_SCRIPT)),
        model="mock-1", config=_config("native"),
    )
    app.use_web_search(backend=_backend(), client=_client())
    result = app.run(QUESTION)
    print("\n2. Native tool-calling model")
    print(f"   tools called: {[t.tool_name for t in result.tool_results]}")
    print(f"   answer: {result.raw_text}")

    # 3. A local model with NO native tool calling: the identical loop. Vincio
    #    lowers the tools into a text protocol and lifts the calls back out.
    app2 = ContextApp(
        name="web_local", provider=LocalGGUFMock(script=list(PROTOCOL_SCRIPT)),
        model="mock-1", config=_config("local"),
    )
    app2.use_web_search(backend=_backend(), client=_client())
    result2 = app2.run(QUESTION)
    print("\n3. Local model without native tool calling — same loop")
    print(f"   tools called: {[t.tool_name for t in result2.tool_results]}")
    print(f"   answer: {result2.raw_text}")
    assert [t.tool_name for t in result.tool_results] == [t.tool_name for t in result2.tool_results]

    # 4. Governance: refusal happens before any request would leave the process.
    print("\n4. Governance")
    guarded = WebBrowser(
        _backend(), policy=WebPolicy(max_searches=1, deny_domains=["tracker.example"]),
        client=_client(),
    )
    for url in ("http://169.254.169.254/metadata", "https://ads.tracker.example/p"):
        try:
            guarded.read_sync(url)
        except WebPolicyError as exc:
            print(f"   refused pre-egress: {url} ({exc.code})")
    trail = [e.action for e in app.audit.entries if e.action.startswith("web_")]
    print(f"   audit trail: {trail}")

    # 5. Proof: the session re-derives offline from its content-hashed snapshots.
    session = app.web_browser.report()
    evidence = session.reads[0]
    print("\n5. Offline verification")
    print(f"   {session.searches_used} search, {session.fetches_used} read; "
          f"snapshot {evidence.content_hash[:12]}…")
    print(f"   session verifies from bytes: {session.verify(app.web_browser.snapshots)}")
    print(f"   tampered snapshot detected: {not evidence.verify(PAGE + '<!-- tampered -->')}")

    # 6. Adaptive reading depth + code fidelity, driven directly.
    browser = WebBrowser(_backend(), client=_client())
    section = browser.read_sync(RELEASE_URL, query="release date", mode="section")
    print("\n6. Adaptive depth")
    print(f"   section mode returned {len(section.excerpts)} block(s) of the best section; "
          f"code blocks preserved verbatim, fenced for the model")

    # 7. Crawl a site into a verifiable collection → documents or a Dataset.
    crawl_pages = {
        "/docs/": "<html><head><title>Docs</title></head><body><h1>Home</h1>"
                  "<p>The documentation index, a real readable block of content here.</p>"
                  "<nav><a href='/docs/guide'>Guide</a></nav></body></html>",
        "/docs/guide": "<html><head><title>Guide</title></head><body><h1>Guide</h1>"
                       "<p>A guide page with enough readable content to be a real block.</p>"
                       "</body></html>",
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
        client=httpx.AsyncClient(transport=httpx.MockTransport(crawl_handler)),
    )
    print("\n7. Crawl → collection")
    print(f"   crawled {collection.pages_fetched} page(s) (bounded, deterministic, "
          f"stopped: {collection.stopped_reason})")
    print(f"   → {len(collection.to_documents())} retrieval documents, "
          f"or a {len(collection.to_dataset().columns)}-column Dataset")

    print("\nDone — every model browses: governed, token-efficient, adaptive, and provable.")


if __name__ == "__main__":
    main()
