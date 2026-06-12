"""Server authentication: API keys and JWT (HS256) with
tenant-scoped tokens. Stdlib-only JWT verification (HMAC) — no extra deps."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from pydantic import BaseModel

from ..core.errors import AuthenticationError

__all__ = ["AuthContext", "Authenticator", "issue_jwt", "verify_jwt"]


class AuthContext(BaseModel):
    subject: str
    tenant_id: str | None = None
    scopes: list[str] = []
    method: str = "api_key"  # api_key | jwt | anonymous


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def issue_jwt(
    secret: str,
    *,
    subject: str,
    tenant_id: str | None = None,
    scopes: list[str] | None = None,
    expires_in_s: int = 3600,
) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_in_s,
    }
    if tenant_id:
        payload["tenant_id"] = tenant_id
    if scopes:
        payload["scopes"] = scopes
    signing_input = (
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    )
    signature = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def verify_jwt(token: str, secret: str) -> dict[str, Any]:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError as exc:
        raise AuthenticationError("malformed JWT") from exc
    header = json.loads(_b64url_decode(header_b64))
    if header.get("alg") != "HS256":
        raise AuthenticationError(f"unsupported JWT alg {header.get('alg')!r}")
    expected = hmac.new(
        secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(expected, _b64url_decode(signature_b64)):
        raise AuthenticationError("invalid JWT signature")
    payload = json.loads(_b64url_decode(payload_b64))
    if payload.get("exp") is not None and time.time() > payload["exp"]:
        raise AuthenticationError("JWT expired")
    return payload


class Authenticator:
    def __init__(self, *, api_keys: list[str] | None = None, jwt_secret: str | None = None) -> None:
        self.api_keys = set(api_keys or [])
        self.jwt_secret = jwt_secret

    @property
    def enabled(self) -> bool:
        return bool(self.api_keys or self.jwt_secret)

    def authenticate(self, authorization: str | None, api_key_header: str | None) -> AuthContext:
        if not self.enabled:
            return AuthContext(subject="anonymous", method="anonymous")
        if api_key_header and api_key_header in self.api_keys:
            return AuthContext(subject=f"key:{api_key_header[:6]}…", method="api_key")
        if authorization:
            scheme, _, credential = authorization.partition(" ")
            if scheme.lower() == "bearer" and credential:
                if credential in self.api_keys:
                    return AuthContext(subject=f"key:{credential[:6]}…", method="api_key")
                if self.jwt_secret:
                    payload = verify_jwt(credential, self.jwt_secret)
                    return AuthContext(
                        subject=str(payload.get("sub", "unknown")),
                        tenant_id=payload.get("tenant_id"),
                        scopes=list(payload.get("scopes", [])),
                        method="jwt",
                    )
        raise AuthenticationError("missing or invalid credentials")
