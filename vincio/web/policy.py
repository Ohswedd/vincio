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

Four presets cover the common shapes — :meth:`WebPolicy.preset`
(``"default"`` / ``"research"`` / ``"scrape"`` / ``"locked_down"``) — and every
field is overridable, so a session is automatic when you want it and pinned
when you need it.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import WebPolicyError
from .search import DEFAULT_USER_AGENT

__all__ = ["WebPolicy"]

_PRIVATE_NAME_SUFFIXES = (".local", ".internal", ".lan", ".home", ".corp")
# Wildcard-DNS services that resolve any embedded IP (incl. private ones) — a
# classic SSRF bypass ("10.0.0.1.nip.io"). Refused wholesale.
_WILDCARD_DNS_SUFFIXES = (".nip.io", ".xip.io", ".sslip.io", ".traefik.me", ".localtest.me")

# Query parameters that are unambiguously tracking cruft, dropped when
# canonicalizing a URL so the same page reached two ways dedupes to one
# fetch/snapshot. Deliberately conservative: only params that never change which
# *resource* a URL resolves to. Ad-network click ids and `utm_*` qualify; `ref`,
# `featured_on`, `spm`, `page`, etc. are load-bearing on some sites and are left
# alone (a caller can strip more via a wider set if they know their targets).
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_PARAMS = frozenset(
    {
        "fbclid", "gclid", "dclid", "gclsrc", "msclkid", "yclid",
        "mc_cid", "mc_eid", "igshid", "_hsenc", "_hsmi", "vero_id",
    }
)


def _is_tracking_param(key: str) -> bool:
    key = key.lower()
    return key in _TRACKING_PARAMS or key.startswith(_TRACKING_PREFIXES)


def _ip_is_private(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _as_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse *host* as an IP literal, **including** the obfuscated IPv4 forms
    (``0x7f.0.0.1``, ``127.1``, ``2130706433``, ``0177.0.0.1``) that
    ``ipaddress`` rejects but ``getaddrinfo`` would resolve — the classic
    private-host bypasses. Returns ``None`` for a genuine hostname."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    # inet_aton accepts the obfuscated/short IPv4 spellings socket resolution
    # honors; normalize them to the canonical address before the private check.
    try:
        packed = socket.inet_aton(host)
    except OSError:
        return None
    return ipaddress.IPv4Address(packed)


def _host_is_private(host: str) -> bool:
    """Private-host check (offline, no DNS): literal IPs in every spelling,
    wildcard-DNS IP embedders, loopback/intranet names, and bare hostnames."""
    host = host.strip("[]").lower().rstrip(".")
    if not host:
        return True
    address = _as_ip(host)
    if address is not None:
        return _ip_is_private(address)
    if host == "localhost" or host.endswith(_PRIVATE_NAME_SUFFIXES):
        return True
    if host.endswith(_WILDCARD_DNS_SUFFIXES):
        return True  # e.g. 10.0.0.1.nip.io resolves to the embedded private IP
    # A bare intranet name ("wiki", "build01") has no public TLD dot.
    return "." not in host


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
    budget bound what any single page may cost; the crawl limits bound a
    :class:`~vincio.web.WebCrawler` walk.

    ``extra="forbid"``: an unknown field is a hard error, not silently dropped —
    so a typo'd rail (``allow_domain=`` for ``allow_domains=``) fails loudly
    rather than leaving the session unexpectedly unrestricted.
    """

    model_config = ConfigDict(extra="forbid")

    allow_domains: list[str] = Field(default_factory=list)
    deny_domains: list[str] = Field(default_factory=list)
    max_searches: int = 8
    max_fetches: int = 12
    max_results: int = 5
    max_page_bytes: int = 2_000_000
    excerpt_budget_tokens: int = 800
    full_page_budget_tokens: int = 6000
    default_mode: str = "auto"
    respect_robots: bool = True
    allow_private_hosts: bool = False
    timeout_s: float = 15.0
    user_agent: str = DEFAULT_USER_AGENT
    # transient-failure resilience
    max_retries: int = 2
    retry_backoff_s: float = 0.5
    max_redirects: int = 5
    # prompt-driven auto-fetch of URLs the user pasted
    auto_fetch: bool = True
    max_auto_fetch: int = 3
    strip_tracking_params: bool = True
    # per-host politeness pacing (seconds between fetches to the same host)
    per_host_delay_s: float = 0.0
    # crawl limits (a WebCrawler walk)
    max_crawl_pages: int = 25
    max_crawl_depth: int = 2
    max_crawl_pages_per_host: int = 40
    max_crawl_bytes: int = 40_000_000
    max_crawl_seconds: float = 120.0
    max_links_per_page: int = 200
    crawl_delay_s: float = 0.0

    @classmethod
    def preset(cls, name: str = "default", **overrides: Any) -> WebPolicy:
        """A named starting policy, with any field overridable by keyword.

        * ``"default"`` — balanced: modest budgets, auto-fetch on, robots on.
        * ``"research"`` — generous budgets and reading depth for deep research.
        * ``"scrape"`` — tuned for a bounded crawl into a collection/dataset:
          more pages, a politeness delay, tracking params stripped.
        * ``"locked_down"`` — no auto-fetch, robots enforced, tight budgets;
          the base for an allowlisted, audited deployment.
        """
        presets: dict[str, dict[str, Any]] = {
            "default": {},
            "research": {
                "max_searches": 16,
                "max_fetches": 24,
                "max_results": 8,
                "excerpt_budget_tokens": 1200,
                "full_page_budget_tokens": 8000,
            },
            "scrape": {
                "max_fetches": 60,
                "max_crawl_pages": 60,
                "max_crawl_depth": 3,
                "crawl_delay_s": 0.5,
                "default_mode": "full",
                "auto_fetch": False,
            },
            "locked_down": {
                "auto_fetch": False,
                "max_searches": 3,
                "max_fetches": 5,
                "max_results": 3,
                "respect_robots": True,
            },
        }
        if name not in presets:
            raise WebPolicyError(
                f"unknown web policy preset {name!r}; "
                f"known: {sorted(presets)}",
                details={"preset": name},
            )
        return cls(**{**presets[name], **overrides})

    def check_url(self, url: str) -> None:
        """Refuse *url* (typed, pre-egress) unless every rail admits it.

        This is the single choke point every fetch — direct, auto-fetched,
        redirected, or crawled — passes through, so a redirect to a private
        host or an off-allowlist domain is refused the same way a direct one is.
        """
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

    def allows_url(self, url: str) -> bool:
        """Non-raising :meth:`check_url`: True iff the URL would be admitted."""
        try:
            self.check_url(url)
        except WebPolicyError:
            return False
        return True

    def canonicalize(self, url: str) -> str:
        """A stable form of *url* for dedup: lower-case the host, drop a default
        port and the fragment, sort query keys, and (when
        ``strip_tracking_params``) remove tracking cruft — so the same page
        reached two ways is fetched and snapshotted once."""
        parts = urlsplit(url)
        query = parse_qsl(parts.query, keep_blank_values=True)
        if self.strip_tracking_params:
            query = [(k, v) for k, v in query if not _is_tracking_param(k)]
        query.sort()
        host = (parts.hostname or "").lower()
        default_port = {"http": 80, "https": 443}.get(parts.scheme)
        netloc = host if (parts.port is None or parts.port == default_port) else f"{host}:{parts.port}"
        path = parts.path or "/"
        return urlunsplit((parts.scheme, netloc, path, urlencode(query), ""))
