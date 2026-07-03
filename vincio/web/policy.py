"""The governance rails a browsing session runs inside.

Web access is an *external* side effect, so it gets the same treatment as any
other governed action: a declarative policy checked deterministically **before
any request leaves the process**, with refusal as a typed
:class:`~vincio.core.errors.WebPolicyError` rather than a silent skip. The
policy is deliberately mechanical — domains, schemes, host classes, byte and
call ceilings — the *judgement* layer (when a model should reach for the web at
all) lives in :mod:`vincio.web.skill`.

Private, loopback, and link-local hosts are refused by default: a
model-directed fetcher must not become a server-side request forger against
the network it runs in. Detection is offline (literal IPs and un-dotted or
``.local``/``.internal`` names); it does not resolve DNS, so a public name
pointing at a private address is out of scope and belongs to egress-layer
controls.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from ..core.errors import WebPolicyError
from .search import DEFAULT_USER_AGENT

__all__ = ["WebPolicy"]

_PRIVATE_NAME_SUFFIXES = (".local", ".internal", ".lan", ".home", ".corp")


def _host_is_private(host: str) -> bool:
    """Literal-IP / naming-convention private-host check (no DNS resolution)."""
    host = host.strip("[]").lower()
    if not host:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host == "localhost" or host.endswith(_PRIVATE_NAME_SUFFIXES):
            return True
        # A bare intranet name ("wiki", "build01") has no public TLD dot.
        return "." not in host
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _domain_matches(host: str, domain: str) -> bool:
    """Label-suffix match: ``example.com`` covers ``www.example.com`` only."""
    host, domain = host.lower().rstrip("."), domain.lower().strip().rstrip(".")
    return host == domain or host.endswith("." + domain)


class WebPolicy(BaseModel):
    """Declarative limits for one browsing session.

    An empty ``allow_domains`` admits every domain not explicitly denied;
    a non-empty one is a strict allowlist. Budgets (``max_searches`` /
    ``max_fetches``) are per-session counters enforced by the
    :class:`~vincio.web.WebBrowser`; the byte ceiling and the excerpt token
    budget bound what any single page may cost.
    """

    allow_domains: list[str] = Field(default_factory=list)
    deny_domains: list[str] = Field(default_factory=list)
    max_searches: int = 8
    max_fetches: int = 12
    max_results: int = 5
    max_page_bytes: int = 2_000_000
    excerpt_budget_tokens: int = 800
    respect_robots: bool = True
    allow_private_hosts: bool = False
    timeout_s: float = 15.0
    user_agent: str = DEFAULT_USER_AGENT

    def check_url(self, url: str) -> None:
        """Refuse *url* (typed, pre-egress) unless every rail admits it."""
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise WebPolicyError(
                f"scheme {parts.scheme!r} is not allowed", details={"url": url}
            )
        host = parts.hostname or ""
        if not self.allow_private_hosts and _host_is_private(host):
            raise WebPolicyError(
                f"private or loopback host {host!r} is refused",
                details={"url": url, "host": host},
            )
        for domain in self.deny_domains:
            if _domain_matches(host, domain):
                raise WebPolicyError(
                    f"domain {host!r} is denied by policy",
                    details={"url": url, "domain": domain},
                )
        if self.allow_domains and not any(
            _domain_matches(host, domain) for domain in self.allow_domains
        ):
            raise WebPolicyError(
                f"domain {host!r} is not on the allowlist",
                details={"url": url, "allow_domains": self.allow_domains},
            )
