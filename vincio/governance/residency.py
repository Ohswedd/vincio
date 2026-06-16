"""Data-residency-aware routing.

When a tenant or run requires in-jurisdiction processing, Vincio must be able
to *refuse egress* to a provider region the policy does not allow. This is a
deterministic policy decision — never a model judgment — so it surfaces as a
blocking :class:`~vincio.security.PolicyViolation` on the same audit path as
every other policy, before any request leaves the process.

Vincio can only refuse to *send* a request; it cannot guarantee where a global
provider ultimately runs it. That boundary is documented — the control here is
egress refusal, which is exactly what an in-jurisdiction policy needs from the
client side.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..security.policy import PolicyViolation

__all__ = ["ResidencyPolicy", "residency_violation"]


# Best-effort default region mapping. Most hosted providers are global, so their
# region is whatever the operator declares via ``provider_regions``. On-prem /
# self-hosted is the one we can assert by construction.
_DEFAULT_PROVIDER_REGIONS: dict[str, str] = {
    "local": "on_prem",
    "mock": "on_prem",
}


class ResidencyPolicy(BaseModel):
    """Pin allowed provider regions and refuse egress to others.

    ``allowed_regions`` empty means the policy is not configured and nothing is
    enforced (backward-compatible default). When non-empty, a run whose resolved
    provider/model region is not in the set is blocked. ``deny_on_unknown``
    controls whether an *unknown* region (not declared in ``provider_regions``)
    is treated as a violation — the safe default for in-jurisdiction policies.
    """

    allowed_regions: list[str] = Field(default_factory=list)
    provider_regions: dict[str, str] = Field(default_factory=dict)
    deny_on_unknown: bool = True

    @property
    def enforced(self) -> bool:
        return bool(self.allowed_regions)

    def region_for(self, provider: str, model: str | None = None) -> str | None:
        """Resolve the region for a provider/model.

        Lookup order: exact ``provider:model`` key, then ``model`` key, then
        ``provider`` key in the operator-declared map, then the built-in
        defaults. Returns ``None`` when the region cannot be determined.
        """
        regions = {**_DEFAULT_PROVIDER_REGIONS, **self.provider_regions}
        if model is not None:
            key = f"{provider}:{model}"
            if key in regions:
                return regions[key]
            if model in regions:
                return regions[model]
        return regions.get(provider)

    def check(self, *, provider: str, model: str | None = None) -> PolicyViolation | None:
        """Return a blocking :class:`PolicyViolation` if egress is disallowed."""
        if not self.enforced:
            return None
        region = self.region_for(provider, model)
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
        if region not in allowed:
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
) -> PolicyViolation | None:
    """Functional shorthand for a one-off residency check."""
    policy = ResidencyPolicy(
        allowed_regions=allowed_regions,
        provider_regions=provider_regions or {},
        deny_on_unknown=deny_on_unknown,
    )
    return policy.check(provider=provider, model=model)
