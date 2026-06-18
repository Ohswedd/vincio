"""Agent directory & capability discovery (2.2).

An :class:`AgentDirectory` indexes :class:`AgentRecord`\\ s — normalized from A2A
Agent Cards, AGNTCY/ACP manifests, or MCP server records — and answers
capability queries (``find``). Every :meth:`resolve` is **governed** by an
:class:`~vincio.security.access.AllowListGate` and **recorded** as an access
decision on the audit chain, so a delegation fabric stays as accountable as a
single in-process tool call.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..a2a.protocol import AgentCard, AgentSkill
from ..core.errors import AccessDeniedError
from ..security.access import AccessDecision, AllowListGate, Principal
from ..stability import experimental

__all__ = ["AgentRecord", "AgentResolution", "AgentDirectory"]

Protocol = Literal["a2a", "acp", "mcp"]


class AgentRecord(BaseModel):
    """A normalized, protocol-neutral directory entry for one agent/server."""

    name: str
    protocol: Protocol = "a2a"
    url: str = ""
    description: str = ""
    version: str = ""
    skills: list[AgentSkill] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)  # capability tags for discovery
    card: dict[str, Any] | None = None  # the raw Agent Card / manifest (wire form)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_agent_card(cls, card: AgentCard, *, url: str = "", protocol: Protocol = "a2a") -> AgentRecord:
        tags = sorted({t for s in card.skills for t in s.tags})
        return cls(
            name=card.name,
            protocol=protocol,
            url=url or card.url,
            description=card.description,
            version=card.version,
            skills=list(card.skills),
            capabilities=tags or [s.id for s in card.skills],
            card=card.to_wire(),
            metadata={"protocol_version": card.protocol_version},
        )

    def _haystack(self) -> str:
        parts = [self.name, self.description, " ".join(self.capabilities)]
        for skill in self.skills:
            parts.extend([skill.id, skill.name, skill.description, " ".join(skill.tags)])
        return " ".join(parts).lower()

    def matches(
        self, *, capability: str | None = None, tag: str | None = None, query: str | None = None
    ) -> bool:
        if capability is not None:
            cap = capability.lower()
            caps = {c.lower() for c in self.capabilities} | {s.id.lower() for s in self.skills}
            if cap not in caps and cap not in self._haystack():
                return False
        if tag is not None:
            tags = {t.lower() for s in self.skills for t in s.tags}
            if tag.lower() not in tags:
                return False
        if query is not None:
            haystack = self._haystack()
            if not any(tok in haystack for tok in query.lower().split()):
                return False
        return True


class AgentResolution(BaseModel):
    """The (non-raising) outcome of a governed resolution."""

    allowed: bool
    decision: AccessDecision
    record: AgentRecord | None = None


@experimental(since="2.2")
class AgentDirectory:
    """A governed, discoverable directory of agents across A2A / ACP / MCP."""

    def __init__(
        self,
        *,
        allow_list: AllowListGate | None = None,
        audit: Any | None = None,
        principal: Principal | None = None,
    ) -> None:
        self.allow_list = allow_list
        self.audit = audit
        self.principal = principal or Principal()
        self._records: dict[str, AgentRecord] = {}

    # -- registration ---------------------------------------------------------

    def register(self, record: AgentRecord | AgentCard, *, url: str = "", protocol: Protocol = "a2a") -> AgentRecord:
        """Register an :class:`AgentRecord` or an A2A :class:`AgentCard`."""
        if isinstance(record, AgentCard):
            record = AgentRecord.from_agent_card(record, url=url, protocol=protocol)
        self._records[record.name] = record
        if self.audit is not None:
            self.audit.record(
                "agent_register",
                resource=record.name,
                decision="registered",
                details={"protocol": record.protocol, "url": record.url, "capabilities": record.capabilities},
            )
        return record

    def all(self) -> list[AgentRecord]:
        return list(self._records.values())

    @property
    def names(self) -> list[str]:
        return sorted(self._records)

    # -- discovery ------------------------------------------------------------

    def find(
        self,
        *,
        capability: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        protocol: Protocol | None = None,
    ) -> list[AgentRecord]:
        """Discover agents by capability/tag/free-text and optional protocol."""
        out = [
            r
            for r in self._records.values()
            if (protocol is None or r.protocol == protocol)
            and r.matches(capability=capability, tag=tag, query=query)
        ]
        return sorted(out, key=lambda r: r.name)

    # -- governed resolution --------------------------------------------------

    def try_resolve(self, name: str, *, principal: Principal | None = None) -> AgentResolution:
        """Resolve ``name`` under the allow-list, recording the decision; never raises."""
        record = self._records.get(name)
        if self.allow_list is None:
            decision = AccessDecision(allowed=True, rule="no_gate", reason="no allow-list configured")
        else:
            decision = self.allow_list.check(name, principal=principal or self.principal)
        if record is None and decision.allowed:
            decision = AccessDecision(
                allowed=False, rule="not_found", reason=f"agent {name!r} not in directory"
            )
        if self.audit is not None:
            self.audit.record(
                "agent_resolve",
                resource=name,
                decision="allow" if decision.allowed else "deny",
                details={
                    "rule": decision.rule,
                    "reason": decision.reason,
                    "protocol": record.protocol if record else None,
                    "url": record.url if record else None,
                },
            )
        return AgentResolution(
            allowed=decision.allowed, decision=decision, record=record if decision.allowed else None
        )

    def resolve(self, name: str, *, principal: Principal | None = None) -> AgentRecord:
        """Resolve ``name`` under the allow-list; raise if denied or unknown.

        The decision is recorded on the audit chain either way.
        """
        resolution = self.try_resolve(name, principal=principal)
        if not resolution.allowed or resolution.record is None:
            raise AccessDeniedError(
                resolution.decision.reason or f"agent {name!r} is not reachable",
                details={"rule": resolution.decision.rule, "agent": name},
            )
        return resolution.record
