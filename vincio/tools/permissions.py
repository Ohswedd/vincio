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
    ) -> None:
        self.access = access or AccessController()
        self.allow_external = allow_external
        self.approval_callback = approval_callback
        self.secrets = secret_scanner or SecretScanner()

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
            except Exception as exc:  # TenantIsolationError
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

        # 6. Approval requirement (write guardrails).
        return ToolPermissionDecision(
            allowed=True,
            reason="ok",
            requires_approval=spec.approval_required,
            checks=checks,
        )

    async def request_approval(self, request: ApprovalRequest) -> bool:
        """Resolve an approval requirement via callback; deny when absent."""
        if self.approval_callback is None:
            return False
        return await self.approval_callback(request)
