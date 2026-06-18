"""2.0 enterprise endpoints behind a pluggable AuthStrategy: AWS Bedrock
(SigV4), Google Vertex (service-account bearer + regional), Azure OpenAI
(deployment routing + api-version), all over the same HTTPProvider plumbing."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from vincio.core.types import Message, ModelRequest
from vincio.providers import build_provider
from vincio.providers.enterprise import (
    AzureKeyAuth,
    AzureOpenAIProvider,
    BearerTokenAuth,
    BedrockProvider,
    SigV4Auth,
    VertexProvider,
)

_FIXED = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)


# -- fake httpx client -----------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)
        self.headers: dict[str, str] = {}

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Records the last POST and returns a canned response."""

    def __init__(self, response_payload: dict) -> None:
        self.is_closed = False
        self._response = response_payload
        self.last_url: str | None = None
        self.last_headers: dict[str, str] = {}
        self.last_content: bytes | None = None
        self.last_json: dict | None = None

    async def post(self, url, *, headers=None, content=None, json=None):  # noqa: A002
        self.last_url = url
        self.last_headers = headers or {}
        self.last_content = content
        self.last_json = json
        return _FakeResponse(self._response)

    async def aclose(self):
        self.is_closed = True


# -- auth strategies -------------------------------------------------------


def test_sigv4_is_deterministic_and_binds_body():
    auth = SigV4Auth("AKIA", "secret", region="us-east-1", clock=lambda: _FIXED)
    h1 = auth.headers(method="POST", url="https://x.amazonaws.com/model/m/converse", body=b'{"a":1}', base_headers={})
    h2 = auth.headers(method="POST", url="https://x.amazonaws.com/model/m/converse", body=b'{"a":1}', base_headers={})
    assert h1["Authorization"] == h2["Authorization"]  # deterministic with fixed clock
    assert h1["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIA/20260617/us-east-1/bedrock/aws4_request")
    assert h1["x-amz-date"] == "20260617T120000Z"
    # A different body yields a different signature (binding).
    h3 = auth.headers(method="POST", url="https://x.amazonaws.com/model/m/converse", body=b'{"a":2}', base_headers={})
    assert h3["Authorization"] != h1["Authorization"]


def test_sigv4_session_token_is_signed_and_sent():
    auth = SigV4Auth("AKIA", "secret", region="eu-west-1", session_token="TOK", clock=lambda: _FIXED)
    h = auth.headers(method="POST", url="https://x.amazonaws.com/y", body=b"{}", base_headers={})
    assert h["x-amz-security-token"] == "TOK"
    assert "x-amz-security-token" in h["Authorization"]  # appears in SignedHeaders


def test_azure_key_auth():
    h = AzureKeyAuth("secret-key").headers(method="POST", url="u", body=b"{}", base_headers={})
    assert h["api-key"] == "secret-key"
    assert "Authorization" not in h


def test_bearer_token_auth_string_and_callable():
    assert BearerTokenAuth("tok").headers(method="GET", url="u", body=b"", base_headers={})[
        "Authorization"
    ] == "Bearer tok"
    calls = {"n": 0}

    def mint() -> str:
        calls["n"] += 1
        return f"tok{calls['n']}"

    auth = BearerTokenAuth(mint)
    assert auth.headers(method="GET", url="u", body=b"", base_headers={})["Authorization"] == "Bearer tok1"
    assert auth.headers(method="GET", url="u", body=b"", base_headers={})["Authorization"] == "Bearer tok2"


# -- Bedrock ---------------------------------------------------------------


async def test_bedrock_generate_signs_and_parses():
    converse_response = {
        "output": {"message": {"content": [{"text": "Bordeaux is in France."}]}},
        "usage": {"inputTokens": 12, "outputTokens": 6},
        "stopReason": "end_turn",
    }
    client = _FakeClient(converse_response)
    provider = BedrockProvider(
        region="us-east-1", access_key="AKIA", secret_key="secret",
        clock=lambda: _FIXED, client=client,
    )
    req = ModelRequest(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=[Message(role="system", content="Be terse."), Message(role="user", content="Where is Bordeaux?")],
        max_output_tokens=64,
        temperature=0.2,
    )
    resp = await provider.generate(req)
    assert resp.text == "Bordeaux is in France."
    assert resp.usage.input_tokens == 12
    assert resp.finish_reason == "stop"
    # SigV4-signed, sent to the converse path, with the model id URL-encoded.
    assert client.last_url.endswith(
        "/model/anthropic.claude-3-5-sonnet-20240620-v1%3A0/converse"
    )
    assert client.last_headers["Authorization"].startswith("AWS4-HMAC-SHA256")
    # The body was signed over the exact bytes sent (content=, not json=).
    assert client.last_content is not None
    body = json.loads(client.last_content)
    assert body["system"] == [{"text": "Be terse."}]
    assert body["messages"][0]["role"] == "user"
    assert body["inferenceConfig"]["maxTokens"] == 64


async def test_bedrock_stream_yields_full_response():
    client = _FakeClient(
        {"output": {"message": {"content": [{"text": "hi"}]}}, "usage": {}, "stopReason": "end_turn"}
    )
    provider = BedrockProvider(region="us-east-1", access_key="A", secret_key="s", client=client)
    req = ModelRequest(model="m", messages=[Message(role="user", content="hello")])
    events = [e async for e in provider.stream(req)]
    assert any(e.type == "text_delta" and e.text == "hi" for e in events)
    assert events[-1].type == "done"


def test_bedrock_requires_no_api_key():
    assert BedrockProvider.requires_api_key is False


# -- Vertex ----------------------------------------------------------------


async def test_vertex_uses_regional_path_and_bearer():
    gemini_response = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
    }
    client = _FakeClient(gemini_response)
    provider = VertexProvider(
        project="my-proj", region="us-central1", access_token="ya29.token", client=client
    )
    req = ModelRequest(model="gemini-2.5-flash", messages=[Message(role="user", content="hi")])
    resp = await provider.generate(req)
    assert resp.text == "ok"
    assert client.last_url.endswith(
        "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
    )
    assert client.last_headers["Authorization"] == "Bearer ya29.token"


# -- Azure -----------------------------------------------------------------


async def test_azure_deployment_routing_and_api_key():
    chat_response = {
        "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        "model": "gpt-4o",
    }
    client = _FakeClient(chat_response)
    provider = AzureOpenAIProvider(
        api_key="azkey",
        endpoint="https://my-resource.openai.azure.com",
        api_version="2024-10-21",
        deployment="gpt4o-prod",
        client=client,
    )
    req = ModelRequest(model="gpt-4o", messages=[Message(role="user", content="q")])
    resp = await provider.generate(req)
    assert resp.text == "answer"
    assert client.last_url.endswith(
        "/openai/deployments/gpt4o-prod/chat/completions?api-version=2024-10-21"
    )
    assert client.last_headers["api-key"] == "azkey"


def test_azure_defaults_deployment_to_model():
    provider = AzureOpenAIProvider(api_key="k", endpoint="https://r.openai.azure.com")
    req = ModelRequest(model="gpt-4o-mini", messages=[])
    assert "/openai/deployments/gpt-4o-mini/chat/completions" in provider._chat_path(req)


# -- registry --------------------------------------------------------------


def test_enterprise_providers_registered():
    from vincio.providers import _registry

    assert {"bedrock", "vertex", "azure"} <= set(_registry.names)


def test_build_provider_constructs_bedrock():
    provider = build_provider("bedrock", region="us-west-2", access_key="A", secret_key="s", with_retries=False)
    assert isinstance(provider, BedrockProvider)
    assert provider.region == "us-west-2"
