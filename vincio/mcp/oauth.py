"""OAuth 2.1 helpers for MCP client (authorization) and server (resource).

MCP standardized on OAuth 2.1: clients discover the authorization server via
Protected-Resource Metadata (PRM), use PKCE, and may register dynamically;
servers validate bearer tokens as a resource server. Vincio provides the
integration seams — PKCE pair generation, bearer headers, and a pluggable token
validator — so the actual token issuance can use any OAuth library or gateway.
Everything runs in your process; Vincio is not an authorization server.
"""

from __future__ import annotations

import base64
import hashlib
import os
from collections.abc import Callable
from typing import Any

from .protocol import MCPError

__all__ = [
    "UNAUTHORIZED",
    "pkce_pair",
    "bearer_headers",
    "static_token_validator",
    "protected_resource_metadata_url",
]

# JSON-RPC-side code used for an unauthorized request (maps to HTTP 401).
UNAUTHORIZED = -32001


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def pkce_pair() -> tuple[str, str]:
    """Return a (code_verifier, code_challenge) PKCE pair (S256)."""
    verifier = _b64url(os.urandom(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def bearer_headers(token: str) -> dict[str, str]:
    """Authorization header for a bearer access token."""
    return {"Authorization": f"Bearer {token}"}


def static_token_validator(tokens: set[str] | list[str]) -> Callable[[str | None], dict[str, Any]]:
    """A resource-server token validator that accepts a fixed allow-list.

    Suitable for tests and simple deployments; swap in JWT/introspection for
    production. The returned callable accepts the raw ``Authorization`` header
    value (or bare token) and raises :class:`MCPError` (code 401) on failure.
    """
    allow = set(tokens)

    def validate(auth: str | None) -> dict[str, Any]:
        if not auth:
            raise MCPError("missing bearer token", code=UNAUTHORIZED, data={"status": 401})
        token = auth.split(" ", 1)[1] if auth.lower().startswith("bearer ") else auth
        if token not in allow:
            raise MCPError("invalid bearer token", code=UNAUTHORIZED, data={"status": 401})
        return {"token": token[:6] + "…"}

    return validate


def protected_resource_metadata_url(resource_url: str) -> str:
    """The PRM discovery URL (RFC 9728) for a resource server."""
    base = resource_url.rstrip("/")
    # PRM lives at the well-known path on the resource origin.
    from urllib.parse import urlsplit

    parts = urlsplit(base)
    return f"{parts.scheme}://{parts.netloc}/.well-known/oauth-protected-resource"
