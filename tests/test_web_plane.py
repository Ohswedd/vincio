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
from vincio.core.errors import VincioError, WebFetchError, WebPolicyError, WebSearchError
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


# -- v2: adaptive depth, code blocks, find, availability -------------------------------------

DOCS_HTML = (
    "<html><head><title>Requests: Quickstart</title></head><body>"
    "<nav><a href='/'>Home</a><a href='/install'>Install</a></nav>"
    "<h1>Quickstart</h1><p>Making a request with Requests is very simple.</p>"
    "<h2>Make a Request</h2><p>Begin by importing the Requests module:</p>"
    "<pre><code>import requests\nr = requests.get('https://api.github.com/events')\n"
    "print(r.status_code)</code></pre>"
    "<h2>Passing Parameters</h2><p>You often want to send query-string data in the URL.</p>"
    "<a href='https://requests.readthedocs.io/en/latest/page2'>Next</a>"
    "<footer><a href='/legal'>Legal</a></footer></body></html>"
)


def test_extract_section_mode_returns_the_matching_section_with_code():
    extract = extract_page(
        DOCS_HTML, query="make a request example code", mode="section", budget_tokens=300
    )
    assert any(e.kind == "code" and "requests.get" in e.text for e in extract.excerpts)
    assert any("importing the Requests" in e.text for e in extract.excerpts)
    assert "```" in extract.as_context()  # code fenced for the model


def test_extract_full_mode_and_links():
    extract = extract_page(DOCS_HTML, mode="full", budget_tokens=4000, collect_links=True)
    assert extract.mode == "full"
    assert any("page2" in link.url for link in extract.links)
    assert all("Legal" not in e.text for e in extract.excerpts)  # footer still chrome


def test_extract_auto_mode_small_page_is_returned_whole():
    small = "<html><head><title>T</title></head><body><h1>H</h1><p>" + "word " * 40 + "</p></body></html>"
    extract = extract_page(small, query="word", mode="auto", budget_tokens=4000)
    assert extract.excerpts  # whole small page fits


def test_extract_find_catches_short_facts():
    page = (
        "<html><head><title>Doc</title></head><body><h1>API</h1>"
        "<p>The library version is documented below.</p><p>v4.2.1</p></body></html>"
    )
    from vincio.web import find_in_page

    extract = extract_page(page, find="v4.2.1")
    assert any("4.2.1" in m.text for m in extract.find_matches)
    assert find_in_page(page, "v4.2.1")  # module-level helper too


def test_extract_availability_cookie_wall_and_js_shell():
    wall = extract_page(
        "<html><head><title>Privacy</title></head><body><h1>We value your privacy</h1>"
        "<p>Accept all cookies to continue browsing this site.</p></body></html>"
    )
    assert not wall.available and wall.unavailable_reason == "cookie_wall"
    js = extract_page("<html><body><div id='root'></div><!--" + "x" * 25000 + "--></body></html>")
    assert not js.available and js.unavailable_reason == "requires_javascript"
    good = extract_page(DOCS_HTML, query="request")
    assert good.available


# -- v2: policy hardening --------------------------------------------------------------------


def test_policy_blocks_obfuscated_ip_literals_and_wildcard_dns():
    policy = WebPolicy()
    for bad in (
        "http://0x7f.0.0.1/", "http://127.0.0.01/", "http://127.1/",
        "http://2130706433/", "http://127.0.0.1.nip.io/",
        "http://169.254.169.254/latest/meta-data/",
    ):
        with pytest.raises(WebPolicyError):
            policy.check_url(bad)
    policy.check_url("https://docs.python.org/3/")


def test_policy_canonicalize_strips_only_tracking_params():
    policy = WebPolicy()
    # utm_* and click ids dropped; load-bearing params (ref, id) kept and sorted
    assert policy.canonicalize("https://x.com/a?utm_source=z&ref=abc&id=5&fbclid=q") == (
        "https://x.com/a?id=5&ref=abc"
    )


def test_policy_presets_never_relax_ssrf_or_robots():
    for name in ("default", "research", "scrape", "locked_down"):
        policy = WebPolicy.preset(name)
        assert policy.allow_private_hosts is False
        assert policy.respect_robots is True
    with pytest.raises(WebPolicyError):
        WebPolicy.preset("nonexistent")


# -- v2: browser hardening (redirect SSRF, gzip cap, retries, dedupe) -------------------------


def _hard_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_browser_refuses_redirect_to_private_host():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(302, headers={"location": "http://169.254.169.254/meta"})

    async def nosleep(_: float) -> None:
        return None

    browser = WebBrowser(_static_backend(), client=_hard_client(handler), sleeper=nosleep)
    with pytest.raises(WebPolicyError):
        asyncio.run(browser.read("https://example.org/redirect"))


def test_browser_streams_with_byte_cap_defeats_bomb():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, content=b"x" * 5_000_000, headers={"content-type": "text/html"})

    browser = WebBrowser(
        _static_backend(), policy=WebPolicy(max_page_bytes=1_000_000), client=_hard_client(handler)
    )
    with pytest.raises(WebFetchError):
        asyncio.run(browser.read("https://example.org/bomb"))


def test_browser_retries_transient_then_succeeds():
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(503, headers={"retry-after": "1"})
        return httpx.Response(200, text=PAGE_HTML, headers={"content-type": "text/html"})

    async def nosleep(_: float) -> None:
        return None

    browser = WebBrowser(_static_backend(), client=_hard_client(handler), sleeper=nosleep)
    extract = asyncio.run(browser.read("https://example.org/flaky", query="release"))
    assert extract.title


def test_browser_403_is_typed_bot_blocked():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(403, text="blocked")

    browser = WebBrowser(_static_backend(), client=_hard_client(handler))
    with pytest.raises(WebFetchError) as excinfo:
        asyncio.run(browser.read("https://example.org/blocked"))
    assert excinfo.value.details.get("reason") == "bot_blocked"


def test_browser_canonical_dedup_one_fetch_for_tracking_variants():
    browser = WebBrowser(_static_backend(), client=_client(_page_handler))
    asyncio.run(browser.read("https://example.org/doc?utm_source=a", query="q"))
    asyncio.run(browser.read("https://example.org/doc?utm_source=b", query="q"))
    assert browser.fetches_used == 1  # canonical dedup collapsed the utm variants


def test_read_records_mode_and_verify_survives_reread_at_new_depth():
    browser = WebBrowser(_static_backend(), client=_client(_page_handler))
    first = asyncio.run(browser.read(RELEASE_URL, query="release date", mode="excerpt"))
    second = asyncio.run(browser.read(RELEASE_URL, query="release date", mode="full"))
    assert first.mode == "excerpt" and second.mode == "full"
    assert browser.fetches_used == 1  # re-read from snapshot, no new fetch
    assert browser.report().verify(browser.snapshots)


# -- v2: intent ------------------------------------------------------------------------------


def test_intent_urls_to_fetch_gating():
    from vincio.web import detect_web_intent, urls_to_fetch

    assert urls_to_fetch("summarize https://numpy.org/doc") == ["https://numpy.org/doc"]
    assert urls_to_fetch("https://foo.com/article") == ["https://foo.com/article"]
    assert urls_to_fetch("what does GET http://169.254.169.254 return in AWS?") == []
    assert urls_to_fetch("run `curl https://x.com/api`") == []
    assert detect_web_intent("look it up on docs.python.org site:pypi.org").wants_search
    assert "pypi.org" in detect_web_intent("site:pypi.org requests").sites


# -- v2: search dates + diversity ------------------------------------------------------------


def test_search_parses_dates_and_site():
    html = (
        "<div class='result'><h2 class='result__title'>"
        "<a class='result__a' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2Fnews&rut=x'>"
        "Python News</a></h2>"
        "<a class='result__snippet'>Oct 7, 2024 - Python 3.13 released today.</a></div>"
    )
    rows = parse_results_html(html)
    assert rows[0].site == "python.org"
    assert rows[0].published == "2024-10-07"
    assert rows[0].snippet.startswith("Python 3.13")  # date stripped off the snippet


def test_search_diversify_caps_per_domain():
    from vincio.web import diversify_results

    rows = [
        SearchResult(rank=i, title=f"T{i}", url=f"https://a.com/{i}", snippet="", site="a.com")
        for i in range(4)
    ] + [SearchResult(rank=9, title="B", url="https://b.com/x", snippet="", site="b.com")]
    out = diversify_results(rows, max_per_site=2)
    # b.com is promoted into the top 3; a.com's overflow demoted
    assert out[2].site == "b.com"


# -- v2: crawler -----------------------------------------------------------------------------


def _crawl_site() -> dict[str, str]:
    def page(title: str, links: list[tuple[str, str]]) -> str:
        anchors = "".join(f"<a href='{u}'>{t}</a>" for u, t in links)
        return (
            f"<html><head><title>{title}</title></head><body><h1>{title}</h1>"
            f"<p>Readable content for {title}, long enough to be a real block here.</p>"
            f"<nav>{anchors}</nav></body></html>"
        )

    pages = {
        "/docs/": page("Index", [("/docs/a", "A"), ("/docs/b", "B"), ("/docs/p?page=1", "P1")]),
        "/docs/a": page("A", [("/docs/b", "B"), ("https://other.test/x", "Ext")]),
        "/docs/b": page("B", [("/docs/a", "A")]),
    }
    for i in range(1, 40):
        pages[f"/docs/p?page={i}"] = page(f"P{i}", [(f"/docs/p?page={i + 1}", "Next")])
    return pages


def _crawl_handler(pages):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        key = request.url.path + (("?" + request.url.query.decode()) if request.url.query else "")
        body = pages.get(key) or pages.get(request.url.path)
        return (
            httpx.Response(200, text=body, headers={"content-type": "text/html"})
            if body else httpx.Response(404)
        )

    return handler


def test_crawler_bounded_deterministic_scoped_verifiable():
    from vincio.web import WebCrawler

    async def run():
        async def nosleep(_: float) -> None:
            return None

        pages = _crawl_site()
        browser = WebBrowser(
            policy=WebPolicy.preset("scrape", max_crawl_pages=25),
            client=_hard_client(_crawl_handler(pages)), sleeper=nosleep,
        )
        crawler = WebCrawler(browser=browser)
        col = await crawler.crawl("https://site.test/docs/", scope="subtree", clock=lambda: 0.0)
        return browser, col

    browser, col = asyncio.run(run())
    assert col.pages_fetched <= 25
    assert not any("other.test" in p.url for p in col.pages)  # subtree scope
    assert col.verify(browser.snapshots)
    assert len(col.to_documents()) == col.pages_fetched
    assert len(col.to_dataset().cells[0]) == col.pages_fetched
    # determinism
    browser2, col2 = asyncio.run(run())
    assert [p.url for p in col.pages] == [p.url for p in col2.pages]


def test_crawler_trap_and_depth_defense():
    from vincio.web import WebCrawler

    async def run():
        async def nosleep(_: float) -> None:
            return None

        pages = _crawl_site()
        browser = WebBrowser(
            policy=WebPolicy.preset("scrape", max_crawl_pages=100, max_crawl_depth=2),
            client=_hard_client(_crawl_handler(pages)), sleeper=nosleep,
        )
        return await WebCrawler(browser=browser).crawl(
            "https://site.test/docs/", scope="subtree", clock=lambda: 0.0
        )

    col = asyncio.run(run())
    # depth cap (2) bounds the linear pagination trap well under its 39 pages
    cal = [p for p in col.pages if "page=" in p.url]
    assert len(cal) <= 8


# -- v2: app verbs (web_crawl, research web=, presets) ---------------------------------------


def test_app_web_crawl_verb(tmp_path):
    pages = _crawl_site()
    app = ContextApp(
        name="crawl", provider=MockProvider(), model="mock-1", config=_offline_config(tmp_path)
    )
    collection = app.web_crawl(
        "https://site.test/docs/", scope="subtree",
        policy=WebPolicy.preset("scrape", max_crawl_pages=10),
        client=_hard_client(_crawl_handler(pages)),
    )
    assert collection.pages_fetched >= 2
    assert any(e.action == "web_crawl" for e in app.audit.entries)


def test_use_web_search_preset_and_auto_fetch_directive(tmp_path):
    article = (
        "<html><head><title>Notes</title></head><body><h1>Release</h1>"
        "<p>Vincio 7.6 adds a universal web browsing plane for every model.</p></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text=article, headers={"content-type": "text/html"})

    seen = {}

    def responder(request):
        joined = "\n".join((m.content if isinstance(m.content, str) else "") for m in request.messages)
        seen["web"] = "universal web browsing plane" in joined
        return "ok"

    app = ContextApp(
        name="af", provider=MockProvider(responder=responder), model="mock-1",
        config=_offline_config(tmp_path),
    )
    app.use_web_search(preset="research", client=_hard_client(handler))
    assert app.web_browser.policy.max_searches == 16  # research preset applied
    app.run("Summarize https://example.org/notes for me")
    assert seen["web"] is True  # directive → auto-fetched into context


def test_use_web_search_does_not_auto_fetch_incidental_url(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body><p>x</p></body></html>",
                              headers={"content-type": "text/html"})

    app = ContextApp(
        name="af2", provider=MockProvider(), model="mock-1", config=_offline_config(tmp_path)
    )
    app.use_web_search(client=_hard_client(handler))
    app.run("Is http://169.254.169.254 the metadata endpoint?")
    assert app.web_browser.report().fetches_used == 0


# -- v2: adversarial-review regression fixes -------------------------------------------------


def test_extract_unknown_mode_is_coerced_not_crashed():
    extract = extract_page("<html><body><h1>H</h1><p>" + "word " * 40 + "</p></body></html>", mode="detailed")
    assert extract.mode in ("excerpt", "section", "full")  # coerced, no ValidationError


def test_extract_records_resolved_mode_not_auto():
    small = "<html><head><title>T</title></head><body><h1>H</h1><p>" + "word " * 30 + "</p></body></html>"
    extract = extract_page(small, mode="auto", budget_tokens=4000)
    assert extract.mode == "full"  # auto resolved to a concrete depth


def test_extract_self_closing_chrome_does_not_swallow_document():
    html = (
        "<html><body><div class='cookie-banner'/>"
        "<h1>Real</h1><p>The real article content survives after a self-closing "
        "chrome div, long enough to register as a real content block.</p></body></html>"
    )
    extract = extract_page(html)
    assert any("real article content" in e.text for e in extract.excerpts)


def test_availability_does_not_misflag_link_dense_portal():
    portal = "<html><body>" + "".join(
        f"<p><a href='/x{i}'>Link {i} to a section of the portal</a></p>" for i in range(60)
    ) + "</body></html>"
    assert extract_page(portal).available  # link-dense portal is not a JS shell


def test_search_snippet_date_rejects_impossible_day():
    from vincio.web.search import _parse_snippet_date

    assert _parse_snippet_date("Feb 30, 2024 - impossible")[0] == ""
    assert _parse_snippet_date("Oct 7, 2024 - real")[0] == "2024-10-07"


def test_intent_does_not_read_a_number_as_a_site():
    from vincio.web import detect_web_intent

    assert detect_web_intent("increase it by 3.14 in the config").sites == []
    assert detect_web_intent("check docs.python.org").sites == ["docs.python.org"]


def test_policy_canonicalize_lowercases_host_and_drops_default_port():
    policy = WebPolicy()
    assert policy.canonicalize("https://Example.COM:443/A?b=1") == "https://example.com/A?b=1"
    assert policy.canonicalize("http://Host:80/p") == "http://host/p"


def test_browser_robots_redirect_is_not_followed_to_private_host():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/x"})
        return httpx.Response(
            200, text="<html><body><p>A readable content block long enough to keep.</p></body></html>",
            headers={"content-type": "text/html"},
        )

    async def nosleep(_: float) -> None:
        return None

    browser = WebBrowser(
        _static_backend(), policy=WebPolicy(max_fetches=3),
        client=_hard_client(handler), sleeper=nosleep,
    )
    # robots redirect to a private host is treated as absent (allowed), never followed
    extract = asyncio.run(browser.read("https://ok.example/page"))
    assert extract.excerpts


def test_browser_decodes_declared_non_utf8_charset():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200, content="café résumé".encode("latin-1"),
            headers={"content-type": "text/html; charset=latin-1"},
        )

    browser = WebBrowser(_static_backend(), client=_hard_client(handler))
    extract = asyncio.run(browser.read("https://ok.example/latin", query="cafe"))
    assert "café" in browser.snapshots[extract.content_hash]


def test_browser_failed_fetches_consume_budget():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(404)

    async def nosleep(_: float) -> None:
        return None

    browser = WebBrowser(
        _static_backend(), policy=WebPolicy(max_fetches=2),
        client=_hard_client(handler), sleeper=nosleep,
    )
    for i in range(2):
        with pytest.raises(WebFetchError):
            asyncio.run(browser.read(f"https://ok.example/missing?i={i}"))
    assert browser.fetches_used == 2
    with pytest.raises(WebPolicyError):  # budget now bounds the failed egress
        asyncio.run(browser.read("https://ok.example/missing?i=3"))


def test_search_snippet_void_tag_does_not_break_snippet_region():
    html = (
        "<div class='result'><h2 class='result__title'>"
        "<a class='result__a' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fx.com%2Fa&rut=k'>T</a></h2>"
        "<a class='result__snippet'>Line one<br>line two of the snippet.</a></div>"
    )
    rows = parse_results_html(html)
    assert rows and "line two" in rows[0].snippet


def test_crawl_empty_seeds_raises_typed_error():
    from vincio.web import WebCrawler

    with pytest.raises(VincioError):
        asyncio.run(WebCrawler(policy=WebPolicy.preset("scrape")).crawl([]))


# -- v2: adversarial-review round-2 fixes (gzip bomb, policy, crawl exposure) -----------------


def test_decompress_bounded_defeats_gzip_bomb():
    import gzip
    import io

    from vincio.web.browser import _decompress_bounded

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(b"\x00" * (50 * 1024 * 1024))  # 50MB → tiny compressed
    with pytest.raises(WebFetchError):
        _decompress_bounded(buf.getvalue(), "gzip", 1_000_000)
    # a legitimate small gzip body decodes
    small = io.BytesIO()
    with gzip.GzipFile(fileobj=small, mode="wb") as gz:
        gz.write(b"<html>ok</html>")
    assert _decompress_bounded(small.getvalue(), "gzip", 1_000_000) == b"<html>ok</html>"
    assert _decompress_bounded(b"plain", "identity", 100) == b"plain"
    with pytest.raises(WebFetchError):  # exotic encoding refused, not trusted
        _decompress_bounded(b"x", "br", 100)


def test_web_policy_forbids_unknown_fields():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        WebPolicy(allow_domain=["x.com"])  # typo for allow_domains → loud, not silent


def test_web_policy_preset_override_merges():
    policy = WebPolicy.preset("scrape", max_crawl_pages=7)
    assert policy.max_crawl_pages == 7 and policy.max_fetches == 60  # base + override


def test_app_web_crawl_accepts_dict_policy_and_exposes_browser(tmp_path):
    pages = _crawl_site()
    app = ContextApp(
        name="cd", provider=MockProvider(), model="mock-1", config=_offline_config(tmp_path)
    )
    collection = app.web_crawl(
        "https://site.test/docs/", scope="subtree",
        policy={"max_crawl_pages": 6}, client=_hard_client(_crawl_handler(pages)),
    )
    assert app.web_browser is not None
    assert collection.verify(app.web_browser.snapshots)  # snapshots reachable


def test_crawl_reports_fetch_budget_stop_reason():
    from vincio.web import WebCrawler

    async def run():
        async def nosleep(_: float) -> None:
            return None

        pages = _crawl_site()
        browser = WebBrowser(
            # fetch budget smaller than the page cap ⇒ budget stops the walk
            policy=WebPolicy.preset("scrape", max_fetches=2, max_crawl_pages=25),
            client=_hard_client(_crawl_handler(pages)), sleeper=nosleep,
        )
        return await WebCrawler(browser=browser).crawl(
            "https://site.test/docs/", scope="subtree", clock=lambda: 0.0
        )

    collection = asyncio.run(run())
    assert collection.stopped_reason == "fetch_budget"  # honest, not "frontier_exhausted"


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
