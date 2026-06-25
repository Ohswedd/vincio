"""Real-behavior coverage for vincio.providers.base.

Every test drives the real provider plumbing: the deterministic MockProvider,
real httpx.MockTransport (no network), real error classes, real registry-shaped
guard logic. No unittest.mock / MagicMock / @patch anywhere.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from vincio.core.errors import (
    CapabilityMismatchError,
    ConfigError,
    ModelRetiredError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from vincio.core.types import (
    Message,
    ModelCapabilities,
    ModelEvent,
    ModelRequest,
    ModelResponse,
)
from vincio.providers import MockProvider
from vincio.providers.base import (
    FailoverChain,
    HTTPProvider,
    ModelProvider,
    ProviderRegistry,
    RetryingProvider,
    _retry_delay_from_body,
    failover_failure,
    guard_entry,
    is_lifecycle_error,
    measure_latency_ms,
    parse_sse_lines,
    reasoning_budget_from_effort,
    run_sync,
    screen_entries,
)


def _req(model: str = "mock-model", text: str = "hello") -> ModelRequest:
    return ModelRequest(model=model, messages=[Message(role="user", content=text)])


# --------------------------------------------------------------------------- #
# Test scaffolding: real ModelProvider subclasses (NOT mocks)                  #
# --------------------------------------------------------------------------- #


class GenerateOnlyProvider(ModelProvider):
    """Implements only generate(), so the *base* stream() emulation is exercised."""

    name = "genonly"

    def __init__(self, response: ModelResponse) -> None:
        self._response = response

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return self._response


class FlakyProvider(ModelProvider):
    """Raises a queued error N times, then succeeds — real retry exercise."""

    name = "flaky"

    def __init__(self, errors: list[ProviderError], success: ModelResponse) -> None:
        self._errors = list(errors)
        self._success = success
        self.calls = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return self._success


class AlwaysFailProvider(ModelProvider):
    name = "alwaysfail"

    def __init__(self, error: ProviderError) -> None:
        self._error = error
        self.calls = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        raise self._error


class MidStreamFailProvider(ModelProvider):
    """Yields one event, then raises — to prove mid-stream errors propagate."""

    name = "midstream"

    def __init__(self, error: ProviderError) -> None:
        self._error = error

    async def generate(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
        raise AssertionError("not used")

    async def stream(self, request: ModelRequest):
        yield ModelEvent(type="text_delta", text="partial")
        raise self._error


class FakeRegistry:
    """Registry-shaped object matching what guard_entry/screen_entries call."""

    def __init__(
        self,
        lifecycles: dict[str, str] | None = None,
        caps: dict[str, ModelCapabilities] | None = None,
    ) -> None:
        self._lifecycles = lifecycles or {}
        self._caps = caps or {}

    def lifecycle(self, model: str):
        return self._lifecycles.get(model)

    def guard_capabilities(self, model: str):
        return self._caps.get(model)


# --------------------------------------------------------------------------- #
# is_lifecycle_error                                                          #
# --------------------------------------------------------------------------- #


def test_is_lifecycle_error_true_for_retired_subclass():
    assert is_lifecycle_error(ModelRetiredError("gone", provider="p")) is True


def test_is_lifecycle_error_false_for_non_provider_error():
    # Line 82: a plain Exception is not a lifecycle error.
    assert is_lifecycle_error(ValueError("boom")) is False


def test_is_lifecycle_error_matches_marker_substring():
    exc = ProviderError("the model_not_found upstream", provider="p")
    assert is_lifecycle_error(exc) is True


def test_is_lifecycle_error_false_for_transient_provider_error():
    exc = ProviderUnavailableError("503 backend overloaded", provider="p")
    assert is_lifecycle_error(exc) is False


# --------------------------------------------------------------------------- #
# guard_entry / screen_entries                                               #
# --------------------------------------------------------------------------- #


def test_guard_entry_retired_is_lifecycle():
    reg = FakeRegistry(lifecycles={"old": "retired"})
    reason, is_lc = guard_entry("old", __needs(), reg)
    assert is_lc is True
    assert reason == "'old' is retired"


def test_guard_entry_capability_mismatch_not_lifecycle():
    # request needs vision but model has none -> incapable, not lifecycle.
    reg = FakeRegistry(caps={"blind": ModelCapabilities(vision=False)})
    req = ModelRequest(
        model="blind",
        messages=[Message(role="user", content="x")],
        tools=[],
    )
    from vincio.providers.capabilities import RequestNeeds

    reason, is_lc = guard_entry("blind", RequestNeeds(vision=True), reg)
    assert is_lc is False
    assert reason is not None
    assert "vision" in reason
    del req


def test_guard_entry_passes_known_capable_model():
    reg = FakeRegistry(caps={"good": ModelCapabilities(vision=True, tool_calling=True)})
    reason, is_lc = guard_entry("good", __needs(), reg)
    assert reason is None
    assert is_lc is False


def __needs():
    from vincio.providers.capabilities import RequestNeeds

    return RequestNeeds()


def test_screen_entries_no_guard_returns_all_attemptable():
    p = MockProvider()
    out, lifecycle, incapable = screen_entries(
        [(p, None), (p, "override-model")], _req(), guard=False, registry=None
    )
    assert lifecycle == [] and incapable == []
    # model resolves to override when present, else request.model.
    assert [m for _, _, m in out] == ["mock-model", "override-model"]


def test_screen_entries_partitions_retired_and_incapable():
    capable = MockProvider()
    capable.name = "capable"
    retired_p = MockProvider()
    retired_p.name = "retiredp"
    blind_p = MockProvider()
    blind_p.name = "blindp"
    reg = FakeRegistry(
        lifecycles={"retired-m": "retired"},
        caps={
            "live-m": ModelCapabilities(vision=True),
            "blind-m": ModelCapabilities(vision=False),
        },
    )
    req = ModelRequest(
        model="live-m",
        messages=[
            Message(
                role="user",
                content=[{"type": "image"}],  # forces vision need
            )
        ],
    )
    entries = [(capable, "live-m"), (retired_p, "retired-m"), (blind_p, "blind-m")]
    attemptable, lifecycle, incapable = screen_entries(
        entries, req, guard=True, registry=reg
    )
    assert [m for _, _, m in attemptable] == ["live-m"]
    assert any("retiredp/retired-m" in s for s in lifecycle)
    assert any("blindp/blind-m" in s for s in incapable)


# --------------------------------------------------------------------------- #
# failover_failure                                                            #
# --------------------------------------------------------------------------- #


def test_failover_failure_retired_only_rotates_now():
    err = failover_failure("failover", 0, [], ["a/m: retired"], [])
    assert isinstance(err, ModelRetiredError)
    assert "rotate now" in err.message


def test_failover_failure_incapable_only():
    err = failover_failure("failover", 0, [], [], ["a/m: needs vision"])
    assert isinstance(err, CapabilityMismatchError)
    assert "no capable failover candidate" in err.message


def test_failover_failure_mixed_detail_branches():
    # Lines 124-129: attempt_errors present plus lifecycle plus incapable detail.
    err = failover_failure(
        "failover",
        attempted=2,
        attempt_errors=["p1: 503", "p2: 500"],
        lifecycle=["p3/m: retired"],
        incapable=["p4/m: no tools"],
    )
    assert isinstance(err, ProviderUnavailableError)
    assert err.retryable is False
    assert "all providers failed" in err.message
    assert "rotate now (retired): p3/m: retired" in err.message
    assert "skipped (incapable): p4/m: no tools" in err.message


# --------------------------------------------------------------------------- #
# reasoning_budget_from_effort                                               #
# --------------------------------------------------------------------------- #


def test_reasoning_budget_explicit_wins_and_floors_at_zero():
    assert reasoning_budget_from_effort("high", explicit=5000) == 5000
    assert reasoning_budget_from_effort("high", explicit=-3) == 0


def test_reasoning_budget_none_effort_returns_default():
    assert reasoning_budget_from_effort(None, default=777) == 777


def test_reasoning_budget_known_and_unknown_effort():
    assert reasoning_budget_from_effort("low") == 4096
    assert reasoning_budget_from_effort("high") == 16384
    # Line 178: an unrecognized effort falls back to default.
    assert reasoning_budget_from_effort("ludicrous", default=999) == 999


# --------------------------------------------------------------------------- #
# run_sync                                                                    #
# --------------------------------------------------------------------------- #


def test_run_sync_outside_loop():
    async def coro():
        return 21 + 21

    assert run_sync(coro()) == 42


def test_run_sync_inside_running_loop_uses_thread():
    async def outer():
        async def inner():
            return "from-thread"

        # We are inside a running loop here; run_sync must spawn a thread.
        return run_sync(inner())

    assert asyncio.run(outer()) == "from-thread"


# --------------------------------------------------------------------------- #
# ModelProvider base methods (default stream / sync wrappers / embed)        #
# --------------------------------------------------------------------------- #


def test_base_stream_emulates_from_generate():
    from vincio.core.types import TokenUsage, ToolCallRequest

    resp = ModelResponse(
        text="hi there",
        tool_calls=[ToolCallRequest(name="f", arguments={"a": 1})],
        usage=TokenUsage(input_tokens=3, output_tokens=2),
    )
    provider = GenerateOnlyProvider(resp)

    async def collect():
        return [e async for e in provider.stream(_req())]

    events = asyncio.run(collect())
    types = [e.type for e in events]
    assert types == ["text_delta", "tool_call_delta", "usage", "done"]
    assert events[0].text == "hi there"
    assert events[1].tool_call.name == "f"
    assert events[-1].response is resp


def test_base_stream_skips_text_when_empty():
    resp = ModelResponse(text="")
    provider = GenerateOnlyProvider(resp)

    async def collect():
        return [e.type async for e in provider.stream(_req())]

    types = asyncio.run(collect())
    assert "text_delta" not in types
    assert types == ["usage", "done"]


def test_generate_sync_round_trips_text():
    provider = MockProvider(default_text="sync-result")
    resp = provider.generate_sync(_req())
    assert resp.text == "sync-result"
    assert resp.provider == "mock"


def test_stream_sync_collects_events():
    provider = MockProvider(default_text="abcdefghijklmnopqrstuvwxyz")
    events = list(provider.stream_sync(_req()))
    text = "".join(e.text for e in events if e.type == "text_delta")
    assert text == "abcdefghijklmnopqrstuvwxyz"
    assert events[-1].type == "done"


def test_default_embed_raises_unsupported():
    class NoEmbed(ModelProvider):
        name = "noembed"

        async def generate(self, request):  # pragma: no cover
            raise AssertionError

    provider = NoEmbed()
    with pytest.raises(ProviderError, match="does not support embeddings"):
        asyncio.run(provider.embed(["x"]))


def test_default_list_models_is_empty_and_aclose_is_noop():
    provider = GenerateOnlyProvider(ModelResponse(text="x"))
    assert asyncio.run(provider.list_models()) == []
    assert asyncio.run(provider.aclose()) is None


def test_base_capabilities_falls_back_to_empty_for_unknown_model():
    provider = GenerateOnlyProvider(ModelResponse(text="x"))
    caps = provider.capabilities("definitely-not-a-real-model-xyz")
    assert caps == ModelCapabilities()


# --------------------------------------------------------------------------- #
# HTTPProvider: client lifecycle, key check, header prep                      #
# --------------------------------------------------------------------------- #


class _StaticAuth:
    def headers(self, *, method, url, body, base_headers):
        out = dict(base_headers)
        out["X-Signed"] = f"{method}:{len(body)}"
        return out


class ConcreteHTTP(HTTPProvider):
    """Minimal concrete HTTPProvider so the shared plumbing can be exercised."""

    name = "httptest"
    default_base_url = "https://api.test"

    async def generate(self, request):  # pragma: no cover - not the unit under test
        raise AssertionError("generate not exercised in these tests")


def _http_provider(handler, *, api_key="k", auth=None, base_url="https://api.test"):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ConcreteHTTP(api_key=api_key, base_url=base_url, client=client, auth=auth)


def test_client_property_lazily_creates_owned_client_and_aclose_closes_it():
    # No client injected -> the property builds a real owned httpx.AsyncClient,
    # honoring the configured timeout/limits, and aclose() closes it.
    p = ConcreteHTTP(api_key="k", base_url="https://x", timeout_s=7.0)
    client = p.client
    assert isinstance(client, httpx.AsyncClient)
    assert client is p.client  # cached on subsequent access
    assert client.is_closed is False
    asyncio.run(p.aclose())
    assert client.is_closed is True


def test_client_property_recreates_when_closed():
    p = ConcreteHTTP(api_key="k", base_url="https://x")
    first = p.client
    asyncio.run(first.aclose())
    assert first.is_closed is True
    # Accessing .client again must build a fresh, open client.
    second = p.client
    assert second is not first
    assert second.is_closed is False
    asyncio.run(p.aclose())


def test_injected_client_is_not_owned_and_aclose_leaves_it_open():
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    injected = httpx.AsyncClient(transport=transport)
    p = ConcreteHTTP(api_key="k", base_url="https://x", client=injected)
    # aclose must NOT close a client the provider does not own.
    asyncio.run(p.aclose())
    assert injected.is_closed is False
    asyncio.run(injected.aclose())


def test_check_key_raises_when_required_and_missing():
    p = ConcreteHTTP(api_key=None, base_url="https://x")

    async def go():
        return await p._post_json("/v1", {"a": 1})

    with pytest.raises(ProviderAuthError, match="missing API key"):
        asyncio.run(go())


def test_prepare_static_path_uses_json_kwarg():
    p = ConcreteHTTP(api_key="k", base_url="https://x")
    headers, kwargs = p._prepare("POST", "https://x/v1", {"a": 1})
    assert headers["Authorization"] == "Bearer k"
    assert kwargs == {"json": {"a": 1}}


def test_prepare_auth_strategy_signs_body_and_uses_content():
    p = ConcreteHTTP(api_key="k", base_url="https://x", auth=_StaticAuth())
    headers, kwargs = p._prepare("POST", "https://x/v1", {"a": 1})
    assert "content" in kwargs
    # body is serialized once and signed over those exact bytes.
    assert headers["X-Signed"] == f"POST:{len(kwargs['content'])}"
    assert kwargs["content"] == b'{"a": 1}'


def test_prepare_auth_strategy_empty_body_for_none_payload():
    p = ConcreteHTTP(api_key="k", base_url="https://x", auth=_StaticAuth())
    headers, kwargs = p._prepare("GET", "https://x/models", None)
    assert kwargs == {}
    assert headers["X-Signed"] == "GET:0"


def test_post_json_success_and_auth_header_sent():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True})

    p = _http_provider(handler)
    out = asyncio.run(p._post_json("/v1/chat", {"q": 1}))
    assert out == {"ok": True}
    assert seen["auth"] == "Bearer k"
    assert seen["url"] == "https://api.test/v1/chat"
    asyncio.run(p.aclose())


def test_get_json_and_get_text():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/text"):
            return httpx.Response(200, text="plain-body")
        return httpx.Response(200, json={"models": [1, 2]})

    p = _http_provider(handler)
    assert asyncio.run(p._get_json("/v1/models")) == {"models": [1, 2]}
    assert asyncio.run(p._get_text("/v1/text")) == "plain-body"
    asyncio.run(p.aclose())


# --------------------------------------------------------------------------- #
# HTTPProvider._raise_for_status: every status branch                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("status", "exc"),
    [
        (401, ProviderAuthError),
        (403, ProviderAuthError),
        (408, ProviderTimeoutError),
        (504, ProviderTimeoutError),
        (500, ProviderUnavailableError),
        (529, ProviderUnavailableError),
        (400, ProviderResponseError),
        (422, ProviderResponseError),
    ],
)
def test_raise_for_status_maps_codes(status, exc):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    p = _http_provider(handler)
    with pytest.raises(exc) as ei:
        asyncio.run(p._post_json("/v1", {"a": 1}))
    assert ei.value.details["status"] == status
    asyncio.run(p.aclose())


def test_raise_for_status_below_400_is_noop():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204, json={})

    p = _http_provider(handler)
    # 204/2xx should not raise.
    assert asyncio.run(p._post_json("/v1", {"a": 1})) == {}
    asyncio.run(p.aclose())


def test_raise_for_status_non_json_body_uses_raw_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="<html>broken</html>")

    p = _http_provider(handler)
    with pytest.raises(ProviderResponseError, match="broken"):
        asyncio.run(p._post_json("/v1", {"a": 1}))
    asyncio.run(p.aclose())


def test_rate_limit_uses_retry_after_header():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "12.5"}, json={})

    p = _http_provider(handler)
    with pytest.raises(ProviderRateLimitError) as ei:
        asyncio.run(p._post_json("/v1", {"a": 1}))
    assert ei.value.retry_after_s == 12.5
    asyncio.run(p.aclose())


def test_rate_limit_falls_back_to_body_retry_delay():
    body = {"error": {"details": [{"retryDelay": "29s"}]}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json=body)

    p = _http_provider(handler)
    with pytest.raises(ProviderRateLimitError) as ei:
        asyncio.run(p._post_json("/v1", {"a": 1}))
    assert ei.value.retry_after_s == 29.0
    asyncio.run(p.aclose())


# --------------------------------------------------------------------------- #
# HTTPProvider network-error mapping (timeout / transport)                     #
# --------------------------------------------------------------------------- #


def test_post_json_timeout_maps_to_provider_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    p = _http_provider(handler)
    with pytest.raises(ProviderTimeoutError, match="slow"):
        asyncio.run(p._post_json("/v1", {"a": 1}))
    asyncio.run(p.aclose())


def test_post_json_transport_error_maps_to_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    p = _http_provider(handler)
    with pytest.raises(ProviderUnavailableError, match="refused"):
        asyncio.run(p._post_json("/v1", {"a": 1}))
    asyncio.run(p.aclose())


def test_get_json_timeout_and_transport_errors():
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("t")

    def transport_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("c")

    p1 = _http_provider(timeout_handler)
    with pytest.raises(ProviderTimeoutError):
        asyncio.run(p1._get_json("/m"))
    p2 = _http_provider(transport_handler)
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(p2._get_json("/m"))
    asyncio.run(p1.aclose())
    asyncio.run(p2.aclose())


def test_get_text_timeout_and_transport_errors():
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("t")

    def transport_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("c")

    p1 = _http_provider(timeout_handler)
    with pytest.raises(ProviderTimeoutError):
        asyncio.run(p1._get_text("/m"))
    p2 = _http_provider(transport_handler)
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(p2._get_text("/m"))


# --------------------------------------------------------------------------- #
# HTTPProvider._post_stream                                                   #
# --------------------------------------------------------------------------- #


def test_post_stream_yields_lines():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="data: a\ndata: b\n")

    p = _http_provider(handler)

    async def collect():
        return [line async for line in p._post_stream("/v1/stream", {"q": 1})]

    lines = asyncio.run(collect())
    assert "data: a" in lines
    assert "data: b" in lines
    asyncio.run(p.aclose())


def test_post_stream_error_status_raises_after_aread():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "down"}})

    p = _http_provider(handler)

    async def collect():
        return [line async for line in p._post_stream("/v1/stream", {"q": 1})]

    with pytest.raises(ProviderUnavailableError, match="down"):
        asyncio.run(collect())
    asyncio.run(p.aclose())


def test_post_stream_timeout_maps_to_provider_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("stream-slow")

    p = _http_provider(handler)

    async def collect():
        return [line async for line in p._post_stream("/v1/stream", {"q": 1})]

    with pytest.raises(ProviderTimeoutError, match="stream-slow"):
        asyncio.run(collect())


def test_post_stream_transport_error_maps_to_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("stream-refused")

    p = _http_provider(handler)

    async def collect():
        return [line async for line in p._post_stream("/v1/stream", {"q": 1})]

    with pytest.raises(ProviderUnavailableError, match="stream-refused"):
        asyncio.run(collect())


# --------------------------------------------------------------------------- #
# _retry_delay_from_body                                                      #
# --------------------------------------------------------------------------- #


def test_retry_delay_from_retry_info_detail():
    body = {"error": {"details": [{"retryDelay": "7.5s"}]}}
    assert _retry_delay_from_body(body) == 7.5


def test_retry_delay_from_message_text():
    body = {"error": {"message": "Quota exceeded, retry in 13.2s please"}}
    assert _retry_delay_from_body(body) == 13.2


def test_retry_delay_none_when_not_a_dict():
    assert _retry_delay_from_body("nope") is None


def test_retry_delay_none_when_no_signal():
    assert _retry_delay_from_body({"error": {"message": "no number here"}}) is None


def test_retry_delay_ignores_non_dict_detail_and_bad_format():
    body = {"error": {"details": ["bogus", {"retryDelay": "notanumbers"}]}}
    assert _retry_delay_from_body(body) is None


def test_retry_delay_skips_detail_without_string_delay_then_finds_next():
    # First detail has a non-string retryDelay (loops on, branch 460->456),
    # the second has a valid one which is returned.
    body = {
        "error": {
            "details": [
                {"retryDelay": 5},  # not a str -> skipped
                {"retryDelay": "11s"},
            ]
        }
    }
    assert _retry_delay_from_body(body) == 11.0


# --------------------------------------------------------------------------- #
# parse_sse_lines                                                            #
# --------------------------------------------------------------------------- #


def test_parse_sse_lines_extracts_data_payload():
    assert parse_sse_lines("data: {\"x\":1}") == '{"x":1}'


def test_parse_sse_lines_comment_blank_and_non_data_return_none():
    assert parse_sse_lines("") is None
    assert parse_sse_lines("   ") is None
    assert parse_sse_lines(": keep-alive comment") is None
    assert parse_sse_lines("event: ping") is None


# --------------------------------------------------------------------------- #
# RetryingProvider                                                           #
# --------------------------------------------------------------------------- #


def test_retrying_provider_retries_then_succeeds():
    inner = FlakyProvider(
        [ProviderRateLimitError("429", retry_after_s=0.0)],
        ModelResponse(text="ok"),
    )
    wrapped = RetryingProvider(inner, max_retries=2, base_delay_s=0.0, jitter=0.0)
    assert wrapped.name == "flaky"
    resp = asyncio.run(wrapped.generate(_req()))
    assert resp.text == "ok"
    assert inner.calls == 2
    assert wrapped.retry_count == 1


def test_retrying_provider_non_retryable_propagates_immediately():
    inner = AlwaysFailProvider(ProviderAuthError("bad key", retryable=False))
    wrapped = RetryingProvider(inner, max_retries=3, base_delay_s=0.0)
    with pytest.raises(ProviderAuthError, match="bad key"):
        asyncio.run(wrapped.generate(_req()))
    assert inner.calls == 1  # not retried


def test_retrying_provider_exhausts_retries_and_raises_last():
    inner = AlwaysFailProvider(ProviderUnavailableError("503", retryable=True))
    wrapped = RetryingProvider(inner, max_retries=2, base_delay_s=0.0, jitter=0.0)
    with pytest.raises(ProviderUnavailableError, match="503"):
        asyncio.run(wrapped.generate(_req()))
    assert inner.calls == 3  # initial + 2 retries
    assert wrapped.retry_count == 2


def test_retrying_provider_delay_honors_retry_after_and_caps():
    inner = MockProvider()
    wrapped = RetryingProvider(inner, max_delay_s=5.0, base_delay_s=1.0, jitter=0.0)
    # retry_after below cap is honored exactly.
    d = wrapped._delay(0, ProviderRateLimitError("x", retry_after_s=3.0))
    assert d == 3.0
    # retry_after above cap is clamped to max_delay_s.
    d2 = wrapped._delay(0, ProviderRateLimitError("x", retry_after_s=99.0))
    assert d2 == 5.0
    # no retry_after -> exponential backoff (jitter 0 => exact).
    d3 = wrapped._delay(2, ProviderUnavailableError("x", retryable=True))
    assert d3 == min(5.0, 1.0 * (2**2))


def test_retrying_provider_stream_retries_before_first_event():
    class FlakyStream(ModelProvider):
        name = "flakystream"

        def __init__(self):
            self.attempts = 0

        async def generate(self, request):  # pragma: no cover
            raise AssertionError

        async def stream(self, request):
            self.attempts += 1
            if self.attempts == 1:
                raise ProviderUnavailableError("warmup", retryable=True)
            yield ModelEvent(type="text_delta", text="now")
            yield ModelEvent(type="done")

    inner = FlakyStream()
    wrapped = RetryingProvider(inner, max_retries=2, base_delay_s=0.0, jitter=0.0)

    async def collect():
        return [e async for e in wrapped.stream(_req())]

    events = asyncio.run(collect())
    assert [e.type for e in events] == ["text_delta", "done"]
    assert inner.attempts == 2
    assert wrapped.retry_count == 1


def test_retrying_provider_stream_midstream_error_propagates():
    inner = MidStreamFailProvider(ProviderUnavailableError("mid", retryable=True))
    wrapped = RetryingProvider(inner, max_retries=3, base_delay_s=0.0)

    async def collect():
        out = []
        async for e in wrapped.stream(_req()):
            out.append(e)
        return out

    with pytest.raises(ProviderUnavailableError, match="mid"):
        asyncio.run(collect())


def test_retrying_provider_delegates_caps_embed_list_aclose():
    inner = MockProvider(embedding_dim=8)
    wrapped = RetryingProvider(inner)
    caps = wrapped.capabilities("any")
    assert caps.tool_calling is True
    vecs = asyncio.run(wrapped.embed(["hello world"]))
    assert len(vecs) == 1 and len(vecs[0]) == 8
    assert asyncio.run(wrapped.list_models()) == []
    assert asyncio.run(wrapped.aclose()) is None


# --------------------------------------------------------------------------- #
# FailoverChain                                                              #
# --------------------------------------------------------------------------- #


def test_failover_chain_requires_entries():
    with pytest.raises(ConfigError, match="at least one provider"):
        FailoverChain([])


def test_failover_chain_falls_through_to_second_provider():
    first = AlwaysFailProvider(ProviderUnavailableError("p1 down", retryable=True))
    first.name = "p1"
    second = MockProvider(default_text="second-wins")
    second.name = "p2"
    chain = FailoverChain([(first, None), (second, None)], guard_capabilities=False)
    resp = asyncio.run(chain.generate(_req()))
    assert resp.text == "second-wins"
    assert first.calls == 1


def test_failover_chain_applies_model_override():
    captured = {}

    class Capture(ModelProvider):
        name = "capture"

        async def generate(self, request):
            captured["model"] = request.model
            return ModelResponse(text="ok")

    chain = FailoverChain([(Capture(), "override-x")], guard_capabilities=False)
    asyncio.run(chain.generate(_req(model="orig")))
    assert captured["model"] == "override-x"


def test_failover_chain_all_fail_raises_unavailable():
    p1 = AlwaysFailProvider(ProviderUnavailableError("a down", retryable=True))
    p1.name = "a"
    p2 = AlwaysFailProvider(ProviderUnavailableError("b down", retryable=True))
    p2.name = "b"
    chain = FailoverChain([(p1, None), (p2, None)], guard_capabilities=False)
    with pytest.raises(ProviderUnavailableError, match="all providers failed"):
        asyncio.run(chain.generate(_req()))


def test_failover_chain_classifies_lifecycle_error_during_attempt():
    # The single entry is attempted (guard off) and raises a lifecycle error;
    # since attempted>0 the terminal error is ProviderUnavailableError but the
    # retired detail is surfaced in the "rotate now (retired)" section.
    p1 = AlwaysFailProvider(ModelRetiredError("model_not_found", provider="a"))
    p1.name = "a"
    chain = FailoverChain([(p1, None)], guard_capabilities=False)
    with pytest.raises(ProviderUnavailableError) as ei:
        asyncio.run(chain.generate(_req()))
    assert "rotate now (retired): a/mock-model: model_not_found" in ei.value.message


def test_failover_chain_stream_falls_through():
    first = AlwaysFailProvider(ProviderUnavailableError("p1 down", retryable=True))
    first.name = "p1"
    second = MockProvider(default_text="streamed")
    second.name = "p2"
    chain = FailoverChain([(first, None), (second, None)], guard_capabilities=False)

    async def collect():
        return [e async for e in chain.stream(_req())]

    events = asyncio.run(collect())
    text = "".join(e.text for e in events if e.type == "text_delta")
    assert text == "streamed"


def test_failover_chain_stream_midstream_error_does_not_failover():
    first = MidStreamFailProvider(ProviderUnavailableError("mid", retryable=True))
    first.name = "p1"
    second = MockProvider(default_text="never")
    second.name = "p2"
    chain = FailoverChain([(first, None), (second, None)], guard_capabilities=False)

    async def collect():
        out = []
        async for e in chain.stream(_req()):
            out.append(e)
        return out

    with pytest.raises(ProviderUnavailableError, match="mid"):
        asyncio.run(collect())


def test_failover_chain_stream_classifies_lifecycle():
    p1 = AlwaysFailProvider(ProviderError("is retired", provider="a"))
    p1.name = "a"
    chain = FailoverChain([(p1, None)], guard_capabilities=False)

    async def collect():
        return [e async for e in chain.stream(_req())]

    with pytest.raises(ProviderUnavailableError) as ei:
        asyncio.run(collect())
    assert "rotate now (retired): a/mock-model: is retired" in ei.value.message


def test_failover_chain_capabilities_uses_first_entry():
    first = MockProvider()
    second = MockProvider()
    chain = FailoverChain([(first, None), (second, None)], guard_capabilities=False)
    caps = chain.capabilities("m")
    assert caps.tool_calling is True


def test_failover_chain_list_models_merges_and_aclose():
    from vincio.core.types import ModelProfile

    class ListsProvider(MockProvider):
        def __init__(self, profiles):
            super().__init__()
            self._profiles = profiles
            self.closed = False

        async def list_models(self):
            return self._profiles

        async def aclose(self):
            self.closed = True

    pa = ListsProvider([ModelProfile(name="m1", model="m1", provider="x")])
    pb = ListsProvider(
        [
            ModelProfile(name="m1-dup", model="m1", provider="x"),
            ModelProfile(name="m2", model="m2", provider="x"),
        ]
    )
    chain = FailoverChain([(pa, None), (pb, None)], guard_capabilities=False)
    merged = asyncio.run(chain.list_models())
    assert sorted(p.model for p in merged) == ["m1", "m2"]  # first-seen dedup
    asyncio.run(chain.aclose())
    assert pa.closed and pb.closed


def test_failover_chain_guard_skips_retired_entry_and_uses_capable():
    retired = MockProvider(default_text="retired")
    retired.name = "ret"
    live = MockProvider(default_text="live-answer")
    live.name = "live"
    reg = FakeRegistry(
        lifecycles={"retired-m": "retired"},
        caps={"live-m": ModelCapabilities(tool_calling=True)},
    )
    chain = FailoverChain(
        [(retired, "retired-m"), (live, "live-m")],
        guard_capabilities=True,
        registry=reg,
    )
    resp = asyncio.run(chain.generate(_req()))
    assert resp.text == "live-answer"


def test_failover_chain_guard_with_default_registry_unknown_model_passes():
    # No registry passed + guard on -> _reg() lazily loads the real default
    # registry. An unknown model is never blocked, so the call succeeds.
    p = MockProvider(default_text="unknown-ok")
    p.name = "p"
    chain = FailoverChain([(p, "totally-unknown-model-xyz")], guard_capabilities=True)
    resp = asyncio.run(chain.generate(_req()))
    assert resp.text == "unknown-ok"
    # The lazily-resolved registry is cached for reuse.
    assert chain._reg() is chain._registry


def test_failover_chain_guard_all_retired_raises_rotate_now():
    p = MockProvider()
    p.name = "p"
    reg = FakeRegistry(lifecycles={"dead": "retired"})
    chain = FailoverChain([(p, "dead")], guard_capabilities=True, registry=reg)
    with pytest.raises(ModelRetiredError, match="rotate now"):
        asyncio.run(chain.generate(_req()))


# --------------------------------------------------------------------------- #
# ProviderRegistry                                                           #
# --------------------------------------------------------------------------- #


def test_provider_registry_create_unknown_raises_with_known_list():
    reg = ProviderRegistry()
    reg.register("mock", lambda **kw: MockProvider(**kw))
    with pytest.raises(ConfigError, match="unknown provider 'nope'"):
        reg.create("nope")


def test_provider_registry_create_and_names():
    reg = ProviderRegistry()
    reg.register("mock", lambda **kw: MockProvider(**kw))
    reg.register("alpha", lambda **kw: MockProvider(**kw))
    assert reg.names == ["alpha", "mock"]  # sorted
    p = reg.create("mock", default_text="hi")
    assert isinstance(p, MockProvider)
    assert p.default_text == "hi"


def test_provider_registry_get_or_create_caches_by_kwargs():
    reg = ProviderRegistry()
    reg.register("mock", lambda **kw: MockProvider(**kw))
    a = reg.get_or_create("mock", default_text="x")
    b = reg.get_or_create("mock", default_text="x")
    c = reg.get_or_create("mock", default_text="y")
    assert a is b  # same kwargs -> cached instance
    assert a is not c  # different kwargs -> new instance


# --------------------------------------------------------------------------- #
# measure_latency_ms                                                          #
# --------------------------------------------------------------------------- #


def test_measure_latency_ms_is_non_negative_int():
    import time

    started = time.monotonic()
    ms = measure_latency_ms(started)
    assert isinstance(ms, int)
    assert ms >= 0
