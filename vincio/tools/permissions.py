"""Tool permission model.

Every tool call is checked for: identity, tenant, required scopes, side
effects, approval requirement, and data sensitivity — deterministically,
before execution. Write actions additionally require idempotency keys and
honor an approval callback (human-in-the-loop hook).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from ..core.types import ToolSpec
from ..core.utils import stable_hash
from ..security.access import AccessController, Principal
from ..security.capability import (
    SIDE_EFFECTING,
    CapabilityBroker,
    CapabilityToken,
    TrustLabel,
)
from ..security.secrets import SecretScanner

__all__ = ["ToolPermissionDecision", "ApprovalRequest", "ToolPermissionChecker"]


class ToolPermissionDecision(BaseModel):
    allowed: bool
    reason: str = ""
    requires_approval: bool = False
    checks: list[dict[str, Any]] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    tool: str
    arguments: dict[str, Any]
    principal_user: str | None
    principal_tenant: str | None
    idempotency_key: str
    side_effects: str


# Returns True to approve, False to deny.
ApprovalCallback = Callable[[ApprovalRequest], Awaitable[bool]]


class ToolPermissionChecker:
    def __init__(
        self,
        access: AccessController | None = None,
        *,
        allow_external: bool = True,
        approval_callback: ApprovalCallback | None = None,
        secret_scanner: SecretScanner | None = None,
        broker: CapabilityBroker | None = None,
        require_capability: set[str] | None = None,
    ) -> None:
        self.access = access or AccessController()
        self.allow_external = allow_external
        self.approval_callback = approval_callback
        self.secrets = secret_scanner or SecretScanner()
        # Capability-scoped tools (opt-in): with a broker configured, a
        # side-effecting tool whose arguments are untrusted-tainted (or whose
        # taint is unknown) must present a valid CapabilityToken minted from the
        # user's request, else the call is routed to the approval gate. Leaving
        # ``broker`` None preserves the prior RBAC/ABAC-only behavior exactly.
        self.broker = broker
        self.require_capability = (
            set(require_capability) if require_capability is not None else set(SIDE_EFFECTING)
        )

    def idempotency_key(self, spec: ToolSpec, arguments: dict[str, Any], principal: Principal) -> str:
        return stable_hash(
            {
                "tool": spec.name,
                "arguments": arguments,
                "tenant": principal.tenant_id,
                "user": principal.user_id,
            },
            length=24,
        )

    def check(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        principal: Principal,
        *,
        resource_tenant_id: str | None = None,
        taint: TrustLabel | None = None,
        capability: CapabilityToken | None = None,
    ) -> ToolPermissionDecision:
        checks: list[dict[str, Any]] = []

        # 1. Scope check (RBAC).
        scope_decision = self.access.check_scopes(principal, spec.permissions)
        checks.append({"check": "scopes", "allowed": scope_decision.allowed, "detail": scope_decision.reason})
        if not scope_decision.allowed:
            return ToolPermissionDecision(allowed=False, reason=scope_decision.reason, checks=checks)

        # 2. ABAC rule evaluation for the action/resource pair.
        abac = self.access.evaluate(
            principal,
            action=f"tool:{spec.side_effects}",
            resource=f"tool:{spec.name}",
            context={"side_effects": spec.side_effects},
        )
        checks.append({"check": "abac", "allowed": abac.allowed, "detail": abac.reason})
        if not abac.allowed and abac.rule != "default":
            return ToolPermissionDecision(allowed=False, reason=abac.reason, checks=checks)

        # 3. Tenant boundary.
        if resource_tenant_id is not None:
            try:
                self.access.check_tenant(principal, resource_tenant_id)
                checks.append({"check": "tenant", "allowed": True})
            except Exception as exc:  # noqa: BLE001 - a tenant-boundary error is surfaced as a denied decision
                checks.append({"check": "tenant", "allowed": False, "detail": str(exc)})
                return ToolPermissionDecision(allowed=False, reason=str(exc), checks=checks)

        # 4. External side effects policy.
        if spec.side_effects == "external" and not self.allow_external:
            reason = "external tools are disabled by policy"
            checks.append({"check": "external", "allowed": False})
            return ToolPermissionDecision(allowed=False, reason=reason, checks=checks)

        # 5. Data sensitivity: never pass credentials as tool arguments.
        findings = self.secrets.scan(arguments)
        if findings:
            reason = f"credentials detected in tool arguments: {[f.path for f in findings]}"
            checks.append({"check": "sensitivity", "allowed": False, "detail": reason})
            return ToolPermissionDecision(allowed=False, reason=reason, checks=checks)
        checks.append({"check": "sensitivity", "allowed": True})

        # 6. Capability authority (opt-in): a side-effecting tool carrying an
        # untrusted taint must present a capability minted from the user's
        # request, on whose authority the side effect runs. Absent/invalid →
        # route to the approval gate rather than escalate silently.
        requires_approval = spec.approval_required
        if self.broker is not None and spec.side_effects in self.require_capability:
            tainted = taint is None or taint.is_tainted
            if tainted:
                verdict = self.broker.verify(
                    capability,
                    capability=spec.name,
                    principal_user=principal.user_id,
                    principal_tenant=principal.tenant_id,
                    arguments=arguments,
                )
                checks.append(
                    {"check": "capability", "allowed": verdict.valid, "detail": verdict.reason}
                )
                if not verdict.valid:
                    requires_approval = True
            else:
                checks.append({"check": "capability", "allowed": True, "detail": "trusted authority"})

        # 7. Approval requirement (write guardrails).
        return ToolPermissionDecision(
            allowed=True,
            reason="ok",
            requires_approval=requires_approval,
            checks=checks,
        )

    async def request_approval(self, request: ApprovalRequest) -> bool:
        """Resolve an approval requirement via callback; deny when absent."""
        if self.approval_callback is None:
            return False
        return await self.approval_callback(request)
