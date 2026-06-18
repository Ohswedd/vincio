"""Access control: RBAC, ABAC, tenant isolation, document
permissions, tool permission scopes.

Decisions are deterministic and explainable: every check returns an
:class:`AccessDecision` with the rule that produced it, suitable for audit.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import AccessDeniedError, TenantIsolationError

__all__ = ["Principal", "Role", "AccessRule", "AccessDecision", "AccessController", "AllowListGate"]


class Principal(BaseModel):
    """The acting identity for a run/request."""

    user_id: str | None = None
    tenant_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)  # e.g. "billing:read", "crm:*"
    attributes: dict[str, Any] = Field(default_factory=dict)  # ABAC attributes


class Role(BaseModel):
    name: str
    scopes: list[str] = Field(default_factory=list)


class AccessRule(BaseModel):
    """ABAC rule: effect applies when action/resource patterns and attribute
    conditions all match."""

    id: str
    effect: str = "allow"  # allow | deny
    actions: list[str] = Field(default_factory=lambda: ["*"])
    resources: list[str] = Field(default_factory=lambda: ["*"])
    condition: dict[str, Any] = Field(default_factory=dict)  # attribute equals-match
    priority: int = 100  # lower evaluates first; deny rules should be low


class AccessDecision(BaseModel):
    allowed: bool
    rule: str = ""
    reason: str = ""

    def raise_if_denied(self) -> None:
        if not self.allowed:
            raise AccessDeniedError(self.reason or "access denied", details={"rule": self.rule})


class AccessController:
    def __init__(
        self,
        *,
        roles: list[Role] | None = None,
        rules: list[AccessRule] | None = None,
        tenant_isolation: bool = True,
        default_allow: bool = False,
        require_explicit_tenant: bool = False,
    ) -> None:
        self.roles = {role.name: role for role in (roles or [])}
        self.rules = sorted(rules or [], key=lambda r: r.priority)
        self.tenant_isolation = tenant_isolation
        self.default_allow = default_allow
        # When True, an untagged (``tenant_id is None``) resource is NOT treated
        # as globally readable: tenant access requires an explicit, matching
        # scope on both sides. Closes the cross-tenant fail-open (a correctness
        # and exfiltration risk). Defaults False to preserve the pre-1.7 behavior
        # for one minor; flip it on to fail closed.
        self.require_explicit_tenant = require_explicit_tenant

    # -- RBAC scopes -----------------------------------------------------------

    def effective_scopes(self, principal: Principal) -> set[str]:
        scopes = set(principal.scopes)
        for role_name in principal.roles:
            role = self.roles.get(role_name)
            if role:
                scopes.update(role.scopes)
        return scopes

    def has_scope(self, principal: Principal, required: str) -> bool:
        """Scope match with wildcard support: 'billing:*' grants 'billing:read'."""
        scopes = self.effective_scopes(principal)
        if required in scopes:
            return True
        return any(fnmatch(required, pattern) for pattern in scopes)

    def check_scopes(self, principal: Principal, required: list[str]) -> AccessDecision:
        missing = [scope for scope in required if not self.has_scope(principal, scope)]
        if missing:
            return AccessDecision(
                allowed=False,
                rule="rbac",
                reason=f"missing scopes: {missing}",
            )
        return AccessDecision(allowed=True, rule="rbac", reason="scopes granted")

    # -- ABAC rules --------------------------------------------------------------

    def _condition_matches(self, condition: dict[str, Any], principal: Principal, context: dict[str, Any]) -> bool:
        merged = {**principal.attributes, "user_id": principal.user_id, "tenant_id": principal.tenant_id, **context}
        return all(merged.get(key) == value for key, value in condition.items())

    def evaluate(
        self,
        principal: Principal,
        *,
        action: str,
        resource: str,
        context: dict[str, Any] | None = None,
    ) -> AccessDecision:
        context = context or {}
        for rule in self.rules:
            if not any(fnmatch(action, pattern) for pattern in rule.actions):
                continue
            if not any(fnmatch(resource, pattern) for pattern in rule.resources):
                continue
            if rule.condition and not self._condition_matches(rule.condition, principal, context):
                continue
            allowed = rule.effect == "allow"
            return AccessDecision(
                allowed=allowed,
                rule=rule.id,
                reason=f"rule {rule.id} ({rule.effect}) matched action={action} resource={resource}",
            )
        return AccessDecision(
            allowed=self.default_allow,
            rule="default",
            reason="no rule matched; default " + ("allow" if self.default_allow else "deny"),
        )

    # -- tenant isolation -----------------------------------------------------------

    def check_tenant(self, principal: Principal, resource_tenant_id: str | None) -> None:
        """Enforce tenant boundaries before retrieval (rule 4).

        In strict mode (``require_explicit_tenant``) a resource with no tenant
        tag is not globally readable: both sides must carry an explicit, matching
        tenant. In legacy mode an untagged resource passes (the pre-1.7
        fail-open, kept for one minor)."""
        if not self.tenant_isolation:
            return
        if self.require_explicit_tenant:
            if (
                principal.tenant_id is None
                or resource_tenant_id is None
                or principal.tenant_id != resource_tenant_id
            ):
                raise TenantIsolationError(
                    "tenant boundary violation (explicit scope required)",
                    details={
                        "principal_tenant": principal.tenant_id,
                        "resource_tenant": resource_tenant_id,
                    },
                )
            return
        if resource_tenant_id is None:
            return
        if principal.tenant_id is None or principal.tenant_id != resource_tenant_id:
            raise TenantIsolationError(
                "tenant boundary violation",
                details={
                    "principal_tenant": principal.tenant_id,
                    "resource_tenant": resource_tenant_id,
                },
            )

    def filter_by_tenant(self, principal: Principal, items: list[Any]) -> list[Any]:
        """Drop items belonging to other tenants (items expose .tenant_id).

        In strict mode an untagged item is dropped (it is not globally visible);
        in legacy mode an untagged item is kept (the pre-1.7 fail-open)."""
        if not self.tenant_isolation:
            return list(items)
        kept = []
        for item in items:
            tenant = getattr(item, "tenant_id", None)
            if self.require_explicit_tenant:
                if tenant is not None and tenant == principal.tenant_id:
                    kept.append(item)
            elif tenant is None or tenant == principal.tenant_id:
                kept.append(item)
        return kept

    # -- document permissions -----------------------------------------------------

    def can_read_document(self, principal: Principal, permissions: list[str]) -> bool:
        """Document-level permission check: empty list means unrestricted."""
        if not permissions:
            return True
        return any(self.has_scope(principal, perm) for perm in permissions)


class AllowListGate:
    """A reachability allow-list for the agent fabric (2.2).

    Governs which agents / servers an org will resolve and reach. It is a thin,
    **fail-closed** view over :class:`AccessController`: ``deny`` patterns are
    evaluated first (lowest priority), then ``allow`` patterns; anything matching
    neither falls through to ``default_allow`` (False by default). Patterns are
    fnmatch globs over the resource name or URL (e.g. ``"*.trusted.example"``,
    ``"researcher"``). :meth:`check` returns an explainable
    :class:`AccessDecision` the caller records on the audit chain.
    """

    def __init__(
        self,
        *,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        default_allow: bool = False,
        action: str = "agent_resolve",
    ) -> None:
        self.allow = list(allow or [])
        self.deny = list(deny or [])
        self.action = action
        rules: list[AccessRule] = []
        for i, pattern in enumerate(self.deny):
            rules.append(
                AccessRule(id=f"deny:{pattern}", effect="deny", resources=[pattern], priority=10 + i)
            )
        for i, pattern in enumerate(self.allow):
            rules.append(
                AccessRule(id=f"allow:{pattern}", effect="allow", resources=[pattern], priority=100 + i)
            )
        # Reachability is independent of tenant isolation; that stays on the data
        # plane. The gate only decides whether a name/URL is contactable at all.
        self.controller = AccessController(
            rules=rules, default_allow=default_allow, tenant_isolation=False
        )

    def check(
        self,
        resource: str,
        *,
        principal: Principal | None = None,
        action: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> AccessDecision:
        return self.controller.evaluate(
            principal or Principal(),
            action=action or self.action,
            resource=resource,
            context=context,
        )

    def allows(self, resource: str, **kwargs: Any) -> bool:
        return self.check(resource, **kwargs).allowed
