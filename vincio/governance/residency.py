"""Data-residency-aware routing.

When a tenant or run requires in-jurisdiction processing, Vincio must be able
to *refuse egress* to a provider region the policy does not allow. This is a
deterministic policy decision — never a model judgment — so it surfaces as a
blocking :class:`~vincio.security.PolicyViolation` on the same audit path as
every other policy, before any request leaves the process.

The strongest residency posture is to point at a **region-pinned endpoint**
(Azure OpenAI regional resource, AWS Bedrock regional endpoint, Vertex AI
regional host, or a sovereign/EU OpenAI-compatible gateway) and let Vincio
*refuse to send* if that endpoint is outside the allowed regions. To support
that, the region is resolved from, in order: an explicit ``provider_regions``
map, then **inferred from the configured endpoint URL** (so the egress decision
reflects the real endpoint, not just a hand-maintained table), then built-in
defaults. Matching is jurisdiction-aware: ``allowed_regions=["eu"]`` admits
``eu-west-1`` and ``europe-west4``.

Vincio can only refuse to *send* a request; it cannot guarantee where a global
provider ultimately runs it. The control here is client-side egress refusal,
which — combined with a region-pinned endpoint — is what an in-jurisdiction
policy needs.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..security.policy import PolicyViolation

__all__ = ["ResidencyPolicy", "residency_violation", "infer_region_from_url"]


# Best-effort default region mapping. Most hosted providers are global, so their
# region is whatever the operator declares or the endpoint reveals. On-prem /
# self-hosted is the one we can assert by construction.
_DEFAULT_PROVIDER_REGIONS: dict[str, str] = {
    "local": "on_prem",
    "mock": "on_prem",
}

# AWS-style region token, e.g. us-east-1, eu-west-1, ap-southeast-2, ca-central-1.
_AWS_REGION_RE = re.compile(r"\b([a-z]{2}(?:-gov)?-[a-z]+-\d)\b")
# GCP-style region token in a hostname, e.g. europe-west4, us-central1, asia-east1.
_GCP_REGION_RE = re.compile(
    r"\b((?:us|europe|asia|australia|northamerica|southamerica|me|africa)-[a-z]+\d)\b"
)
# Jurisdiction hints that appear as a subdomain/path segment (sovereign gateways).
_JURISDICTION_TOKENS = ("eu", "us", "uk", "ca", "au", "apac", "in", "jp", "de", "fr")

# Map a resolved region to a coarse jurisdiction code for allow-matching.
_JURISDICTION_PREFIXES: dict[str, str] = {
    "europe": "eu", "eu": "eu", "de": "eu", "fr": "eu",
    "us": "us", "northamerica": "us",
    "uk": "uk", "gb": "uk",
    "ca": "ca",
    "asia": "apac", "australia": "apac", "ap": "apac", "jp": "apac", "in": "apac",
}


def infer_region_from_url(url: str | None) -> str | None:
    """Infer a provider region from a region-bearing endpoint URL.

    Recognizes AWS (``bedrock-runtime.us-east-1.amazonaws.com``), GCP/Vertex
    (``europe-west4-aiplatform.googleapis.com``), and sovereign-gateway
    jurisdiction subdomains/paths (``https://eu.api.example.com``). Returns the
    region/jurisdiction token, or ``None`` when nothing region-specific is found.
    """
    if not url:
        return None
    lowered = url.lower()
    aws = _AWS_REGION_RE.search(lowered)
    if aws:
        return aws.group(1)
    gcp = _GCP_REGION_RE.search(lowered)
    if gcp:
        return gcp.group(1)
    # Jurisdiction token as a dotted/slashed/hyphenated segment in the host/path.
    for token in _JURISDICTION_TOKENS:
        if re.search(rf"(?:^|[./-]){token}(?:[./-]|$)", lowered.split("://", 1)[-1]):
            return token
    return None


def _jurisdiction(region: str) -> str:
    """Coarse jurisdiction code for a region (e.g. ``eu-west-1`` -> ``eu``)."""
    head = region.lower().split("-")[0]
    return _JURISDICTION_PREFIXES.get(head, head)


def _region_allowed(region: str, allowed: set[str]) -> bool:
    allowed_lower = {a.lower() for a in allowed}
    region_lower = region.lower()
    if region_lower in allowed_lower:
        return True
    if _jurisdiction(region_lower) in allowed_lower:
        return True
    # An allowed jurisdiction prefix admits its specific regions ("eu" -> "eu-west-1").
    return any(region_lower.startswith(f"{a}-") for a in allowed_lower)


class ResidencyPolicy(BaseModel):
    """Pin allowed provider regions and refuse egress to others.

    ``allowed_regions`` empty means the policy is not configured and nothing is
    enforced (backward-compatible default). When non-empty, a run whose resolved
    provider/model/endpoint region is not allowed is blocked. ``deny_on_unknown``
    controls whether an *unknown* region (neither declared nor inferable) is
    treated as a violation — the safe default for in-jurisdiction policies.
    """

    allowed_regions: list[str] = Field(default_factory=list)
    provider_regions: dict[str, str] = Field(default_factory=dict)
    deny_on_unknown: bool = True

    @property
    def enforced(self) -> bool:
        return bool(self.allowed_regions)

    def region_for(
        self, provider: str, model: str | None = None, *, base_url: str | None = None
    ) -> str | None:
        """Resolve the region for a provider/model/endpoint.

        Lookup order: exact ``provider:model`` key, then ``model`` key, then
        ``provider`` key in the operator-declared map; then the region inferred
        from ``base_url``; then built-in defaults. Returns ``None`` when the
        region cannot be determined.
        """
        regions = {**_DEFAULT_PROVIDER_REGIONS, **self.provider_regions}
        if model is not None:
            key = f"{provider}:{model}"
            if key in regions:
                return regions[key]
            if model in regions:
                return regions[model]
        if provider in regions:
            return regions[provider]
        inferred = infer_region_from_url(base_url)
        if inferred is not None:
            return inferred
        return None

    def check(
        self, *, provider: str, model: str | None = None, base_url: str | None = None
    ) -> PolicyViolation | None:
        """Return a blocking :class:`PolicyViolation` if egress is disallowed."""
        if not self.enforced:
            return None
        region = self.region_for(provider, model, base_url=base_url)
        allowed = set(self.allowed_regions)
        if region is None:
            if self.deny_on_unknown:
                return PolicyViolation(
                    policy="data_residency",
                    severity="block",
                    message=(
                        f"residency policy requires region in {sorted(allowed)} but the region of "
                        f"provider {provider!r} (model {model!r}) is unknown"
                    ),
                    details={"provider": provider, "model": model, "region": None,
                             "allowed_regions": sorted(allowed)},
                )
            return None
        if not _region_allowed(region, allowed):
            return PolicyViolation(
                policy="data_residency",
                severity="block",
                message=(
                    f"residency policy forbids egress to region {region!r}; "
                    f"allowed regions are {sorted(allowed)}"
                ),
                details={"provider": provider, "model": model, "region": region,
                         "allowed_regions": sorted(allowed)},
            )
        return None


def residency_violation(
    *,
    provider: str,
    model: str | None,
    allowed_regions: list[str],
    provider_regions: dict[str, str] | None = None,
    deny_on_unknown: bool = True,
    base_url: str | None = None,
) -> PolicyViolation | None:
    """Functional shorthand for a one-off residency check."""
    policy = ResidencyPolicy(
        allowed_regions=allowed_regions,
        provider_regions=provider_regions or {},
        deny_on_unknown=deny_on_unknown,
    )
    return policy.check(provider=provider, model=model, base_url=base_url)
