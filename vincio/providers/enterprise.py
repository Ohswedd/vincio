"""Enterprise deployment endpoints: AWS Bedrock, Google Vertex, Azure OpenAI.

These are the surfaces the governance buyer actually runs on. They route
through the *same* :class:`~vincio.providers.base.HTTPProvider`, model registry,
capability guards, swap gate, residency, and audit chain as every other
provider — not a separate proxy. The enabler is the pluggable
:class:`~vincio.providers.base.AuthStrategy`: a per-request signing hook so a
provider is no longer limited to a static api-key header.

* **AWS Bedrock** — SigV4-signed ``converse`` (pure-stdlib signing, no boto3).
* **Google Vertex** — service-account OAuth bearer + regional endpoints,
  reusing the Gemini request/response shape.
* **Azure OpenAI** — deployment-name routing + ``api-version`` + ``api-key``
  (or Azure AD bearer), reusing the OpenAI chat shape.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

from ..core.errors import ProviderResponseError
from ..core.types import (
    FinishReason,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from .base import HTTPProvider
from .google import GoogleProvider
from .openai import OpenAIProvider

__all__ = [
    "AzureKeyAuth",
    "BearerTokenAuth",
    "SigV4Auth",
    "BedrockProvider",
    "VertexProvider",
    "AzureOpenAIProvider",
]


# ---------------------------------------------------------------------------
# Auth strategies
# ---------------------------------------------------------------------------


class AzureKeyAuth:
    """Static Azure ``api-key`` header (or, with ``header='Authorization'`` and a
    bearer token, Azure AD)."""

    def __init__(self, api_key: str, *, header: str = "api-key", prefix: str = "") -> None:
        self.api_key = api_key
        self.header = header
        self.prefix = prefix

    def headers(
        self, *, method: str, url: str, body: bytes, base_headers: dict[str, str]
    ) -> dict[str, str]:
        return {self.header: f"{self.prefix}{self.api_key}", "Content-Type": "application/json"}


class BearerTokenAuth:
    """OAuth2 bearer token, from a string or a zero-arg callable (so a
    short-lived service-account token can be refreshed per request)."""

    def __init__(self, token: str | Callable[[], str]) -> None:
        self._token = token

    def _resolve(self) -> str:
        return self._token() if callable(self._token) else self._token

    def headers(
        self, *, method: str, url: str, body: bytes, base_headers: dict[str, str]
    ) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._resolve()}", "Content-Type": "application/json"}


class SigV4Auth:
    """AWS Signature Version 4 signer (pure stdlib — no boto3 dependency).

    Signs each request over its exact method / canonical URI / query / headers /
    body hash, so the signature binds the bytes actually sent. Supports static
    keys plus an optional session token (STS / assumed roles).
    """

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        *,
        region: str,
        service: str = "bedrock",
        session_token: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.service = service
        self.session_token = session_token
        self._clock = clock or (lambda: datetime.now(UTC))

    @staticmethod
    def _sign(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    def _signing_key(self, datestamp: str) -> bytes:
        k_date = self._sign(("AWS4" + self.secret_key).encode("utf-8"), datestamp)
        k_region = self._sign(k_date, self.region)
        k_service = self._sign(k_region, self.service)
        return self._sign(k_service, "aws4_request")

    def headers(
        self, *, method: str, url: str, body: bytes, base_headers: dict[str, str]
    ) -> dict[str, str]:
        now = self._clock().astimezone(UTC)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        datestamp = now.strftime("%Y%m%d")
        parts = urlsplit(url)
        host = parts.netloc
        canonical_uri = quote(parts.path or "/", safe="/-_.~")
        canonical_qs = parts.query  # callers pass already-encoded, sorted queries
        payload_hash = hashlib.sha256(body).hexdigest()

        signed_parts = {
            "content-type": "application/json",
            "host": host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        }
        if self.session_token:
            signed_parts["x-amz-security-token"] = self.session_token
        signed_headers = ";".join(sorted(signed_parts))
        canonical_headers = "".join(f"{k}:{signed_parts[k]}\n" for k in sorted(signed_parts))
        canonical_request = "\n".join(
            [method, canonical_uri, canonical_qs, canonical_headers, signed_headers, payload_hash]
        )
        credential_scope = f"{datestamp}/{self.region}/{self.service}/aws4_request"
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                amz_date,
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )
        signature = hmac.new(
            self._signing_key(datestamp), string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        authorization = (
            f"AWS4-HMAC-SHA256 Credential={self.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        headers = {
            "Authorization": authorization,
            "x-amz-date": amz_date,
            "x-amz-content-sha256": payload_hash,
            "Content-Type": "application/json",
            "host": host,
        }
        if self.session_token:
            headers["x-amz-security-token"] = self.session_token
        return headers


# ---------------------------------------------------------------------------
# AWS Bedrock (Converse API)
# ---------------------------------------------------------------------------

_BEDROCK_STOP: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "content_filtered": "content_filter",
}


class BedrockProvider(HTTPProvider):
    """AWS Bedrock via the unified Converse API, SigV4-signed.

    Credentials come from explicit keys or the standard environment
    (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN``).
    """

    name = "bedrock"
    requires_api_key = False

    def __init__(
        self,
        *,
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        clock: Callable[[], datetime] | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        import os

        self.region = region
        access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        session_token = session_token or os.environ.get("AWS_SESSION_TOKEN")
        auth = None
        if access_key and secret_key:
            auth = SigV4Auth(
                access_key,
                secret_key,
                region=region,
                service="bedrock",
                session_token=session_token,
                clock=clock,
            )
        super().__init__(
            base_url=base_url or f"https://bedrock-runtime.{region}.amazonaws.com",
            auth=auth,
            **kwargs,
        )

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        system: list[dict[str, str]] = []
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role in ("system", "developer"):
                if message.text:
                    system.append({"text": message.text})
                continue
            role = "assistant" if message.role == "assistant" else "user"
            messages.append({"role": role, "content": [{"text": message.text}]})
        inference: dict[str, Any] = {}
        if request.max_output_tokens is not None:
            inference["maxTokens"] = request.max_output_tokens
        if request.temperature is not None:
            inference["temperature"] = request.temperature
        if request.top_p is not None:
            inference["topP"] = request.top_p
        if request.stop:
            inference["stopSequences"] = request.stop
        payload: dict[str, Any] = {"messages": messages}
        if system:
            payload["system"] = system
        if inference:
            payload["inferenceConfig"] = inference
        return payload

    def _parse_response(
        self, data: dict[str, Any], request: ModelRequest, latency_ms: int
    ) -> ModelResponse:
        content = (data.get("output") or {}).get("message", {}).get("content", [])
        if not content:
            raise ProviderResponseError(
                "no content in response", provider=self.name, retryable=True
            )
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        usage_raw = data.get("usage") or {}
        usage = TokenUsage(
            input_tokens=int(usage_raw.get("inputTokens", 0)),
            output_tokens=int(usage_raw.get("outputTokens", 0)),
        )
        return ModelResponse(
            model=request.model,
            text=text,
            finish_reason=_BEDROCK_STOP.get(data.get("stopReason", ""), "stop"),
            usage=usage,
            latency_ms=latency_ms,
            provider=self.name,
            raw=data,
        )

    def _model_path(self, model: str, *, stream: bool = False) -> str:
        action = "converse-stream" if stream else "converse"
        return f"/model/{quote(model, safe='')}/{action}"

    async def generate(self, request: ModelRequest) -> ModelResponse:
        import time

        start = time.monotonic()
        data = await self._post_json(self._model_path(request.model), self._payload(request))
        return self._parse_response(data, request, int((time.monotonic() - start) * 1000))

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        # Bedrock's converse-stream is an AWS binary event stream; rather than
        # ship a partial binary parser, we serve the full response as a single
        # delta + done so streaming callers work uniformly offline and online.
        response = await self.generate(request)
        if response.text:
            yield ModelEvent(type="text_delta", text=response.text)
        yield ModelEvent(type="usage", usage=response.usage)
        yield ModelEvent(type="done", response=response)


# ---------------------------------------------------------------------------
# Google Vertex AI (Gemini generateContent on the Vertex surface)
# ---------------------------------------------------------------------------


class VertexProvider(GoogleProvider):
    """Google Vertex AI — the Gemini request/response shape on a regional,
    service-account-authenticated endpoint. Reuses :class:`GoogleProvider`'s
    payload/parse; only the URL and auth differ."""

    name = "vertex"
    requires_api_key = False

    def __init__(
        self,
        *,
        project: str,
        region: str = "us-central1",
        access_token: str | Callable[[], str] | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.project = project
        self.region = region
        auth = BearerTokenAuth(access_token) if access_token is not None else None
        super().__init__(
            base_url=base_url or f"https://{region}-aiplatform.googleapis.com",
            auth=auth,
            **kwargs,
        )

    def _content_path(self, request: ModelRequest, action: str) -> str:
        suffix = "?alt=sse" if action == "streamGenerateContent" else ""
        return (
            f"/v1/projects/{self.project}/locations/{self.region}"
            f"/publishers/google/models/{request.model}:{action}{suffix}"
        )


# ---------------------------------------------------------------------------
# Azure OpenAI (deployment-name routing + api-version)
# ---------------------------------------------------------------------------


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI — the OpenAI chat shape with Azure's deployment routing,
    ``api-version`` query, and ``api-key`` (or Azure AD bearer) auth."""

    name = "azure"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str | None = None,
        base_url: str | None = None,
        api_version: str = "2024-10-21",
        deployment: str | None = None,
        ad_token: str | Callable[[], str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.api_version = api_version
        # A fixed deployment overrides per-request model→deployment routing.
        self.deployment = deployment
        resolved_base = (base_url or endpoint or "").rstrip("/")
        if ad_token is not None:
            auth: Any = BearerTokenAuth(ad_token)
        elif api_key:
            auth = AzureKeyAuth(api_key)
        else:
            auth = None
        super().__init__(api_key=api_key, base_url=resolved_base, auth=auth, **kwargs)

    requires_api_key = False  # auth may be Azure AD rather than an api-key

    def _chat_path(self, request: ModelRequest) -> str:
        deployment = self.deployment or request.model
        return (
            f"/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )
