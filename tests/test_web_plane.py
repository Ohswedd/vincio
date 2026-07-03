"""Universal web browsing & search: the DuckDuckGo backend and its parser, the
token-budgeted extractor, the pre-egress WebPolicy rails, the evidence-keeping
WebBrowser session, the ToolProtocolProvider that grants tool use to models
without native function calling, the app verb, and the websearch connector.
All offline: the network is an injected httpx.MockTransport."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from vincio.core.app import ContextApp
from vincio.core.config import VincioConfig
from vincio.core.errors import WebFetchError, WebPolicyError, WebSearchError
from vincio.core.types import (
    Message,
    ModelCapabilities,
    ModelRequest,
    ToolCallRequest,
    ToolSpec,
)
from vincio.providers import (
    MockProvider,
    ToolProtocolProvider,
    lift_tool_calls,
    lower_tool_protocol,
)
from vincio.web import (
    DuckDuckGoBackend,
    SearchResult,
    StaticSearchBackend,
    WebBrowser,
    WebPolicy,
    browse_skill,
    extract_page,
    parse_results_html,
)

RELEASE_URL = "https://www.python.org/downloads/release/python-3130/"

RESULTS_HTML = """
<div class="result results_links_deep web-result">
 <h2 class="result__title">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2Fdownloads%2Frelease%2Fpython-3130%2F&rut=abc">
   Python Release Python 3.13.0</a></h2>
 <a class="result__snippet" href="#">Python 3.13.0 is the <b>newest major release</b>.</a>
</div>
<div class="result result--ad">
 <h2><a class="result__a" href="//duckduckgo.com/y.js?ad_domain=x">Sponsored</a></h2>
</div>
<div class="result">
 <h2 class="result__title">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3.13%2Fwhatsnew%2F&rut=z">
   What's New In Python 3.13</a></h2>
 <a class="result__snippet">Summary of changes.</a>
</div>
"""

LITE_HTML = """
<table>
 <tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2F&rut=q"
   class='result-link'>Welcome to Python.org</a></td></tr>
 <tr><td class='result-snippet'>The official home of the Python language.</td></tr>
</table>
"""

BLOCKED_HTML = "<html><body><div class='anomaly-modal'>detected an anomaly</div></body></html>"

PAGE_HTML = (
    "<html><head><title>Python 3.13 Release</title><style>.x{color:red}</style></head><body>"
    "<nav><a href='/'>Home</a><a href='/dl'>Downloads</a><a href='/doc'>Docs</a></nav>"
    "<h1>Python 3.13.0</h1>"
    "<p>Python 3.13.0 is the newest major release of the Python programming language.</p>"
    "<h2>Release date</h2>"
    "<p>Python 3.13.0 was released on October 7, 2024, and ships a new interactive "
    "interpreter plus experimental free-threading support.</p>"
    "<h2>Unrelated</h2>"
    + "".join(
        f"<p>Filler paragraph {i} about gardening and cooking recipes, long enough "
        "to be a real content block that inflates the page well beyond any budget.</p>"
        for i in range(120)
    )
    + "<footer><a href='/legal'>Legal</a></footer></body></html>"
)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _page_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/robots.txt":
        return httpx.Response(200, text="User-agent: *\nDisallow: /private/")
    return httpx.Response(200, text=PAGE_HTML, headers={"content-type": "text/html"})


def _offline_config(tmp_path) -> VincioConfig:
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/vincio.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    return config


# -- results parsing -----------------------------------------------------------------------------


def test_parse_results_decodes_redirects_and_drops_ads():
    rows = parse_results_html(RESULTS_HTML)
    assert [r.url for r in rows] == [RELEASE_URL, "https://docs.python.org/3.13/whatsnew/"]
    assert rows[0].rank == 1 and rows[0].title == "Python Release Python 3.13.0"
    assert "newest major release" in rows[0].snippet
    assert all("Sponsored" not in r.title for r in rows)


def test_parse_results_lite_layout_and_dedupe():
    rows = parse_results_html(LITE_HTML + LITE_HTML)  # duplicated page: one row
    assert len(rows) == 1
    assert rows[0].url == "https://www.python.org/"
    assert rows[0].snippet == "The official home of the Python language."


def test_parse_results_max_results():
    assert len(parse_results_html(RESULTS_HTML, max_results=1)) == 1


# -- DuckDuckGo backend --------------------------------------------------------------------------


def test_ddg_backend_parses_results():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "python 3.13"
        assert "VincioWeb" in request.headers["user-agent"]
        return httpx.Response(200, text=RESULTS_HTML)

    backend = DuckDuckGoBackend(client=_client(handler))
    rows = asyncio.run(backend.search("python 3.13", max_results=5))
    assert rows[0].url == RELEASE_URL


def test_ddg_backend_falls_back_to_lite_when_blocked():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "html.duckduckgo.com":
            return httpx.Response(200, text=BLOCKED_HTML)
        return httpx.Response(200, text=LITE_HTML)

    backend = DuckDuckGoBackend(client=_client(handler))
    rows = asyncio.run(backend.search("python"))
    assert rows[0].url == "https://www.python.org/"


def test_ddg_backend_blocked_everywhere_is_typed():
    backend = DuckDuckGoBackend(
        client=_client(lambda request: httpx.Response(200, text=BLOCKED_HTML))
    )
    with pytest.raises(WebSearchError) as excinfo:
        asyncio.run(backend.search("python"))
    assert "anomaly" in str(excinfo.value)


def test_ddg_backend_zero_hits_is_a_valid_empty_answer():
    backend = DuckDuckGoBackend(
        client=_client(lambda request: httpx.Response(200, text="<html><body></body></html>"))
    )
    assert asyncio.run(backend.search("qzx")) == []


def test_ddg_backend_http_error_and_empty_query():
    backend = DuckDuckGoBackend(client=_client(lambda request: httpx.Response(500)))
    with pytest.raises(WebSearchError):
        asyncio.run(backend.search("python"))
    with pytest.raises(WebSearchError):
        asyncio.run(DuckDuckGoBackend().search("   "))


def test_ddg_backend_region_and_recency_params():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, text=RESULTS_HTML)

    backend = DuckDuckGoBackend(client=_client(handler), region="us-en")
    asyncio.run(backend.search("python", recency="w"))
    assert seen["kl"] == "us-en" and seen["df"] == "w"


def test_static_backend_scripts_and_records():
    backend = StaticSearchBackend(
        {"q": [SearchResult(rank=9, title="T", url="https://example.org/", snippet="s")]}
    )
    rows = asyncio.run(backend.search("q"))
    assert rows[0].rank == 1  # re-ranked
    assert asyncio.run(backend.search("other")) == []
    assert backend.queries == ["q", "other"]


# -- extraction ----------------------------------------------------------------------------------


def test_extract_page_finds_the_fact_and_drops_chrome():
    extract = extract_page(
        PAGE_HTML, url=RELEASE_URL, query="release date october", budget_tokens=120
    )
    assert extract.title == "Python 3.13 Release"
    assert any("October 7, 2024" in e.text for e in extract.excerpts)
    assert all("Home" not in e.text and "Legal" not in e.text for e in extract.excerpts)
    assert extract.excerpt_tokens <= 120
    assert extract.reduction > 8


def test_extract_page_matching_query_does_not_pad_with_filler():
    extract = extract_page(PAGE_HTML, query="release date october", budget_tokens=400)
    assert all("Filler" not in e.text for e in extract.excerpts)


def test_extract_page_missed_query_degrades_to_lead():
    extract = extract_page(PAGE_HTML, query="zebra xylophone", budget_tokens=80)
    assert extract.excerpts and "newest major release" in extract.excerpts[0].text


def test_extract_page_chrome_region_ends_at_its_own_close_tag():
    # Regression: a hint-classed <div> (Sphinx's menu-wrapper) must stop
    # excluding at its own </div>, not swallow the rest of the document.
    html = (
        "<html><body>"
        "<div class='menu-wrapper'><ul><li><a href='/'>Home</a></li></ul></div>"
        "<p>The actual article content survives after the menu wrapper closes, "
        "long enough to be kept as a real content block.</p>"
        "</body></html>"
    )
    extract = extract_page(html, query="", budget_tokens=100)
    assert len(extract.excerpts) == 1
    assert "actual article content" in extract.excerpts[0].text
    assert "Home" not in extract.excerpts[0].text


def test_extract_page_is_deterministic():
    first = extract_page(PAGE_HTML, query="release date", budget_tokens=100)
    second = extract_page(PAGE_HTML, query="release date", budget_tokens=100)
    assert first.model_dump() == second.model_dump()


def test_extract_as_context_renders_sections():
    extract = extract_page(PAGE_HTML, url=RELEASE_URL, query="release date", budget_tokens=100)
    rendered = extract.as_context()
    assert rendered.startswith(f"[{RELEASE_URL}]")
    assert "## Release date" in rendered


# -- policy --------------------------------------------------------------------------------------


def test_policy_schemes_and_private_hosts():
    policy = WebPolicy()
    for url in (
        "ftp://example.org/x",
        "http://127.0.0.1/x",
        "http://localhost/x",
        "http://10.0.0.8/x",
        "http://[::1]/x",
        "http://intranet/x",
        "http://build01.internal/x",
    ):
        with pytest.raises(WebPolicyError):
            policy.check_url(url)
    policy.check_url("https://example.org/x")
    WebPolicy(allow_private_hosts=True).check_url("http://127.0.0.1/x")


def test_policy_domain_lists():
    policy = WebPolicy(deny_domains=["tracker.example"])
    with pytest.raises(WebPolicyError):
        policy.check_url("https://www.tracker.example/p")
    policy.check_url("https://nottracker.example.org/p")

    allow = WebPolicy(allow_domains=["python.org"])
    allow.check_url("https://docs.python.org/3/")
    with pytest.raises(WebPolicyError):
        allow.check_url("https://example.org/")


# -- browser session -----------------------------------------------------------------------------


def _static_backend() -> StaticSearchBackend:
    return StaticSearchBackend(
        {
            "python release": [
                SearchResult(rank=1, title="Python 3.13.0", url=RELEASE_URL, snippet="release")
            ]
        }
    )


def test_browser_search_budget_and_cache():
    backend = _static_backend()
    browser = WebBrowser(backend, policy=WebPolicy(max_searches=1))
    first = asyncio.run(browser.search("python release"))
    again = asyncio.run(browser.search("Python  Release"))  # normalized + cached
    assert first == again
    assert browser.searches_used == 1 and browser.searches[1].cached
    with pytest.raises(WebPolicyError) as excinfo:
        asyncio.run(browser.search("something else"))
    assert "budget" in str(excinfo.value)


def test_browser_read_produces_verifiable_evidence():
    browser = WebBrowser(_static_backend(), client=_client(_page_handler))
    extract = asyncio.run(browser.read(RELEASE_URL, query="release date october"))
    assert extract.content_hash in browser.snapshots
    report = browser.report()
    assert report.fetches_used == 1
    assert report.verify(browser.snapshots)
    evidence = report.reads[0]
    assert evidence.verify(browser.snapshots[evidence.content_hash])
    assert not evidence.verify(PAGE_HTML + "<!-- tampered -->")


def test_browser_respects_robots_and_fetch_budget():
    browser = WebBrowser(
        _static_backend(), policy=WebPolicy(max_fetches=1), client=_client(_page_handler)
    )
    with pytest.raises(WebPolicyError):
        asyncio.run(browser.read("https://example.org/private/page", query="q"))
    asyncio.run(browser.read("https://example.org/public", query="q"))
    with pytest.raises(WebPolicyError):
        asyncio.run(browser.read("https://example.org/another", query="q"))


def test_browser_read_refuses_binary_and_oversize():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/binary":
            return httpx.Response(
                200, content=b"\x00\x01", headers={"content-type": "application/octet-stream"}
            )
        return httpx.Response(200, text=PAGE_HTML, headers={"content-type": "text/html"})

    browser = WebBrowser(
        _static_backend(), policy=WebPolicy(max_page_bytes=64), client=_client(handler)
    )
    with pytest.raises(WebFetchError):
        asyncio.run(browser.read("https://example.org/binary"))
    with pytest.raises(WebFetchError):  # over the byte ceiling
        asyncio.run(browser.read("https://example.org/page"))


def test_browser_sync_front_door():
    browser = WebBrowser(_static_backend(), client=_client(_page_handler))
    rows = browser.search_sync("python release")
    extract = browser.read_sync(rows[0].url, query="release date")
    assert extract.excerpts


def test_browser_records_on_the_audit_trail():
    class _Audit:
        def __init__(self) -> None:
            self.entries: list[tuple[str, str]] = []

        def record(self, action: str, *, decision: str, details: dict) -> None:
            self.entries.append((action, decision))

    audit = _Audit()
    browser = WebBrowser(_static_backend(), client=_client(_page_handler), audit=audit)
    asyncio.run(browser.search("python release"))
    asyncio.run(browser.read(RELEASE_URL, query="release date"))
    assert ("web_search", "allow") in audit.entries
    assert ("web_read", "allow") in audit.entries


def test_browser_tools_return_compact_json():
    browser = WebBrowser(_static_backend(), client=_client(_page_handler))
    handlers = dict(
        (spec.name, (spec, handler)) for spec, handler in browser.tool_handlers()
    )
    search_spec, search_handler = handlers["web_search"]
    read_spec, read_handler = handlers["web_read"]
    assert search_spec.side_effects == "external" and search_spec.permissions == ["web:search"]
    assert read_spec.permissions == ["web:read"]
    hits = asyncio.run(search_handler(query="python release"))
    assert hits["results"][0]["url"] == RELEASE_URL and "<" not in str(hits)
    page = asyncio.run(read_handler(url=RELEASE_URL, query="release date"))
    assert page["excerpts"] and page["content_hash"]


# -- the universal tool protocol -------------------------------------------------------------


def _tool_request(messages: list[Message]) -> ModelRequest:
    return ModelRequest(
        model="local-gguf",
        messages=messages,
        tools=[
            ToolSpec(
                name="web_search",
                description="Search the web.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
    )


def test_lower_tool_protocol_injects_and_folds():
    request = _tool_request(
        [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Latest python?"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCallRequest(id="tc1", name="web_search", arguments={"query": "q"})],
            ),
            Message(role="tool", content='{"results": []}', tool_call_id="tc1", name="web_search"),
        ]
    )
    lowered = lower_tool_protocol(request)
    assert lowered.tools == []
    system = str(lowered.messages[0].content)
    assert "Tool protocol" in system and "web_search(query: string)" in system
    roles = [m.role for m in lowered.messages]
    assert "tool" not in roles and not any(m.tool_calls for m in lowered.messages)
    assert "[tool_result name=web_search id=tc1]" in str(lowered.messages[-1].content)


def test_lower_tool_protocol_creates_system_message_and_merges_roles():
    request = _tool_request(
        [
            Message(role="user", content="Hi"),
            Message(role="tool", content="r1", tool_call_id="a", name="t"),
            Message(role="tool", content="r2", tool_call_id="b", name="t"),
        ]
    )
    lowered = lower_tool_protocol(request)
    assert lowered.messages[0].role == "system"
    assert [m.role for m in lowered.messages] == ["system", "user"]  # tool turns merged into user


def test_lift_tool_calls_parses_caps_and_keeps_malformed_visible():
    text = (
        'Thinking.\n```tool_call\n{"name": "a", "arguments": {"x": 1}}\n```\n'
        "```tool_call\nnot json\n```\n"
        '```tool_call\n{"name": "b"}\n```'
    )
    clean, calls = lift_tool_calls(text, max_calls=2)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[0].arguments == {"x": 1} and calls[1].arguments == {}
    assert "not json" in clean  # malformed block stays visible
    _, capped = lift_tool_calls(text, max_calls=1)
    assert len(capped) == 1


def test_tool_protocol_provider_engages_only_without_native_support():
    class NonNative(MockProvider):
        def capabilities(self, model: str) -> ModelCapabilities:
            return ModelCapabilities(tool_calling=False)

    native = ToolProtocolProvider(MockProvider(script=["plain"]))
    request = _tool_request([Message(role="user", content="q")])
    response = asyncio.run(native.generate(request))
    assert response.text == "plain"  # passthrough: native mock claims tool_calling
    assert native.inner.requests[0].tools  # tools reached the native provider untouched

    wrapped = ToolProtocolProvider(
        NonNative(script=['```tool_call\n{"name": "web_search", "arguments": {"query": "q"}}\n```'])
    )
    response = asyncio.run(wrapped.generate(request))
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "web_search"
    assert wrapped.inner.requests[0].tools == []
    assert wrapped.capabilities("anything").tool_calling is True


def test_tool_protocol_provider_force_overrides_native_claim():
    forced = ToolProtocolProvider(MockProvider(script=["no calls here"]), force=True)
    request = _tool_request([Message(role="user", content="q")])
    response = asyncio.run(forced.generate(request))
    assert response.tool_calls == [] and response.text == "no calls here"
    assert forced.inner.requests[0].tools == []  # lowered even though mock claims native


# -- the app verb, end to end --------------------------------------------------------------------


def _release_backend() -> StaticSearchBackend:
    return StaticSearchBackend(
        {
            "latest python version release": [
                SearchResult(
                    rank=1,
                    title="Python Release Python 3.13.0",
                    url=RELEASE_URL,
                    snippet="newest major release",
                )
            ]
        }
    )


_ANSWER = "Python 3.13.0, released October 7, 2024. Source: " + RELEASE_URL


def test_use_web_search_native_model_runs_the_loop(tmp_path):
    mock = MockProvider(
        script=[
            {"tool_call": {"name": "web_search", "arguments": {"query": "latest python version release"}}},
            {"tool_call": {"name": "web_read", "arguments": {"url": RELEASE_URL, "query": "release date"}}},
            _ANSWER,
        ]
    )
    app = ContextApp(name="webnative", provider=mock, model="mock-1", config=_offline_config(tmp_path))
    app.use_web_search(backend=_release_backend(), client=_client(_page_handler))
    assert "web_search" in app.enabled_tools and "web_read" in app.enabled_tools
    assert app.skill_library is not None and "web-search" in app.skill_library.index_text()
    result = app.run("What is the latest Python version and when was it released?")
    assert [t.tool_name for t in result.tool_results] == ["web_search", "web_read"]
    assert all(t.status == "ok" for t in result.tool_results)
    assert "October 7, 2024" in result.raw_text
    report = app.web_browser.report()
    assert report.searches_used == 1 and report.fetches_used == 1
    assert report.verify(app.web_browser.snapshots)
    assert any(e.action == "web_search_enabled" for e in app.audit.entries)


def test_use_web_search_protocol_model_runs_the_same_loop(tmp_path):
    class NonNative(MockProvider):
        def capabilities(self, model: str) -> ModelCapabilities:
            return ModelCapabilities(tool_calling=False, structured_output=False)

    mock = NonNative(
        script=[
            '```tool_call\n{"name": "web_search", "arguments": {"query": "latest python version release"}}\n```',
            '```tool_call\n{"name": "web_read", "arguments": {"url": "' + RELEASE_URL + '", "query": "release date"}}\n```',
            _ANSWER,
        ]
    )
    app = ContextApp(
        name="webproto", provider=mock, model="some-local-gguf", config=_offline_config(tmp_path)
    )
    app.use_web_search(backend=_release_backend(), client=_client(_page_handler))
    result = app.run("What is the latest Python version and when was it released?")
    assert [t.tool_name for t in result.tool_results] == ["web_search", "web_read"]
    assert "October 7, 2024" in result.raw_text
    lowered = mock.requests[0]
    assert lowered.tools == [] and "Tool protocol" in str(lowered.messages[0].content)


def test_use_web_search_policy_fields_and_no_wrap(tmp_path):
    app = ContextApp(
        name="webpolicy",
        provider=MockProvider(),
        model="mock-1",
        config=_offline_config(tmp_path),
    )
    app.use_web_search(
        backend=_release_backend(), tool_protocol=False, skill=False, max_searches=2
    )
    assert app.web_browser.policy.max_searches == 2
    assert app.skill_library is None
    assert not isinstance(app._provider_instance, ToolProtocolProvider)


# -- the browsing skill ----------------------------------------------------------------------


def test_browse_skill_progressive_disclosure():
    skill = browse_skill()
    assert skill.name == "web-search"
    assert "When NOT to search" in skill.instructions
    assert skill.match_score("search the web for the latest python release") > 0.0


# -- the websearch connector -----------------------------------------------------------------


def test_websearch_connector_yields_cited_documents():
    from vincio.connectors import connect

    connector = connect(
        "websearch",
        queries=["python release"],
        backend=_static_backend(),
        client=_client(_page_handler),
        max_results=1,
    )
    documents = asyncio.run(connector.load())
    assert len(documents) == 1
    document = documents[0]
    assert document.source_uri == RELEASE_URL
    assert "Python 3.13.0" in document.text
    assert document.metadata["content_hash"] and document.metadata["query"] == "python release"


def test_websearch_connector_snippets_only_mode():
    connector_cls = __import__(
        "vincio.connectors.websearch", fromlist=["WebSearchConnector"]
    ).WebSearchConnector
    connector = connector_cls(
        ["python release"], backend=_static_backend(), fetch_pages=False
    )
    documents = asyncio.run(connector.load())
    assert documents and documents[0].text == "release"
    assert documents[0].metadata["rank"] == 1
