"""MCP Apps & elicitation: server-rendered UI and governed mid-call input.

The MCP spec's newer surface adds two interactions that this module lands in the
*same* governed, audited, budgeted runtime Vincio already runs MCP tools and
resources through — never as a hosted service:

* **MCP Apps (server-rendered UI).** A server exposes an interactive UI as a
  ``ui://`` resource (raw HTML or an AG-UI event snapshot). :class:`MCPAppBridge`
  reads those resources from a connected :class:`~vincio.mcp.client.MCPClient`
  and lowers each into an :class:`~vincio.server.agui.AGUIEvent` on the *existing*
  generative-UI channel, so server UI rides one streamed run — inheriting its
  provenance (``origin: mcp:<server>``, an untrusted-external trust level), its
  budget (each render is token-metered and a render over the cap is refused), and
  its audit (every render and refusal lands on the hash-chained audit log).

* **Elicitation (typed mid-call input).** A server may ask the user for a
  structured value mid-call (``elicitation/create``). :class:`ElicitationGate`
  governs that request with the *same* approval and rail machinery that gates a
  write tool today: an :class:`ElicitationPolicy` can require an approval before
  the value is collected, the collected value is screened through the app's input
  :class:`~vincio.security.rails.RailEngine`, and an accepted value is wrapped as
  an untrusted :class:`~vincio.security.TaintedValue` so it is *contained like any
  other untrusted input* — it can never silently authorize a side effect.

Everything here is deterministic and offline; it never depends on a model
judgment and adds no required dependency.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ..core.types import TrustLevel
from ..security.capability import TaintedValue

__all__ = [
    "ElicitationAction",
    "ElicitationRequest",
    "ElicitationResponse",
    "ElicitationPolicy",
    "ElicitationDecision",
    "ElicitationGate",
    "MCPUIRender",
    "MCPAppBridge",
    "is_ui_resource",
]


# ---------------------------------------------------------------------------
# Elicitation — a typed, governed mid-call request for user input
# ---------------------------------------------------------------------------


class ElicitationAction(StrEnum):
    """The MCP elicitation outcome the client returns to the server.

    ``accept`` carries a structured ``content``; ``decline`` is an explicit "no"
    (the user, an approval gate, or a rail refused); ``cancel`` aborts the flow.
    """

    ACCEPT = "accept"
    DECLINE = "decline"
    CANCEL = "cancel"


class ElicitationRequest(BaseModel):
    """A server's typed mid-call request for user input (``elicitation/create``).

    ``schema`` is the JSON Schema (flat, primitive properties per the MCP
    elicitation contract) the server wants the value to satisfy; ``server`` is the
    connected server's name, carried so the gate can audit and taint by source.
    """

    message: str = ""
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
    server: str = ""

    model_config = {"populate_by_name": True}

    @classmethod
    def from_params(cls, params: dict[str, Any], *, server: str = "") -> ElicitationRequest:
        """Build a request from a raw ``elicitation/create`` params object."""
        return cls(
            message=str(params.get("message", "")),
            schema=params.get("requestedSchema") or params.get("schema") or {},
            server=server,
        )


class ElicitationResponse(BaseModel):
    """The wire response to an ``elicitation/create`` request.

    ``content`` is populated only on :attr:`ElicitationAction.ACCEPT`.
    """

    action: ElicitationAction = ElicitationAction.DECLINE
    content: dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        """Render the canonical MCP elicitation response object."""
        wire: dict[str, Any] = {"action": self.action.value}
        if self.action is ElicitationAction.ACCEPT:
            wire["content"] = self.content
        return wire


class ElicitationPolicy(BaseModel):
    """How an :class:`ElicitationGate` governs a mid-call input request.

    The defaults always *contain* an accepted value — it is screened through the
    input rails and tainted untrusted — without blocking the common case (the
    user directly answering). Set :attr:`require_approval` to gate the request
    behind an explicit approval first, exactly like a write tool.
    """

    require_approval: bool = False
    screen_rails: bool = True
    forbid_quarantined: bool = True


@dataclass(slots=True)
class ElicitationDecision:
    """The governed outcome of one elicitation.

    Carries the wire :attr:`response`, the :class:`TaintedValue` an accepted value
    was wrapped in (untrusted, so downstream code that consumes it inherits the
    taint), the input-rail :attr:`rail_check`, whether an approval was granted, and
    a human-readable :attr:`reason`. :meth:`to_wire` returns what the client sends
    back to the server.
    """

    response: ElicitationResponse
    tainted: TaintedValue[dict[str, Any]] | None = None
    rail_check: Any = None
    approved: bool = False
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.response.action is ElicitationAction.ACCEPT

    def to_wire(self) -> dict[str, Any]:
        return self.response.to_wire()


# A collector turns (message, schema) into the user's structured value, or a
# falsy value / None to decline. Sync or async.
Collector = Callable[[str, dict[str, Any]], Any]
# An approver decides whether the request may proceed at all (the approval gate).
Approver = Callable[[ElicitationRequest], bool | Awaitable[bool]]


class ElicitationGate:
    """Govern a server's mid-call input request the way a write tool is governed.

    The flow, in order:

    1. **Approval.** If :attr:`ElicitationPolicy.require_approval`, an ``approver``
       must grant the request (else the gate declines) — the same gate a write
       tool passes before it runs.
    2. **Collect.** The ``collector`` obtains the user's structured value; a
       ``None``/falsy return is an explicit decline.
    3. **Rail screen.** The collected value's text is run through the input
       :class:`~vincio.security.rails.RailEngine`; a blocking rail (PII, secrets,
       injection, …) declines the value, and an injection-flagged value is
       declined when :attr:`ElicitationPolicy.forbid_quarantined`.
    4. **Taint.** An accepted value is wrapped as an untrusted
       :class:`TaintedValue` (``mcp:<server>:elicitation`` source), so it is
       contained like any other untrusted input and cannot launder its way into
       an unauthorized side effect.

    Every decision is recorded on ``audit`` (action ``mcp_elicit``).
    """

    def __init__(
        self,
        collector: Collector | None = None,
        *,
        policy: ElicitationPolicy | None = None,
        rail_engine: Any | None = None,
        approver: Approver | None = None,
        audit: Any | None = None,
    ) -> None:
        self.collector = collector
        self.policy = policy or ElicitationPolicy()
        self.rail_engine = rail_engine
        self.approver = approver
        self.audit = audit

    async def decide(self, request: ElicitationRequest) -> ElicitationDecision:
        """Govern one :class:`ElicitationRequest`; return an :class:`ElicitationDecision`."""
        # 1. Approval gate (like a write tool's approval).
        approved = True
        if self.policy.require_approval:
            approved = await self._approve(request)
            if not approved:
                return self._declined(request, "approval not granted", approved=False)

        # 2. Collect the user's value.
        if self.collector is None:
            return self._declined(request, "no collector configured", approved=approved)
        raw = self.collector(request.message, request.schema_)
        if inspect.isawaitable(raw):
            raw = await raw
        if raw is None or raw is False:
            return self._declined(request, "user declined", approved=approved)
        content = self._as_content(raw)

        # 3. Screen the collected value through the input rails.
        rail_check = None
        quarantined = False
        if self.policy.screen_rails and self.rail_engine is not None and content:
            rail_check = self.rail_engine.check(_flatten(content), direction="input")
            quarantined = any("injection" in (v.details or {}) for v in rail_check.violations)
            if not rail_check.allowed:
                return self._declined(
                    request, "input rail blocked the elicited value", approved=approved,
                    rail_check=rail_check,
                )
            if quarantined and self.policy.forbid_quarantined:
                return self._declined(
                    request, "elicited value flagged as injection (quarantined)",
                    approved=approved, rail_check=rail_check,
                )

        # 4. Accept — wrap untrusted so the value is contained downstream.
        tainted = TaintedValue.untrusted(
            content, source=f"mcp:{request.server}:elicitation", quarantined=quarantined
        )
        self._audit(request, "accept", "elicited value accepted", fields=sorted(content))
        return ElicitationDecision(
            response=ElicitationResponse(action=ElicitationAction.ACCEPT, content=content),
            tainted=tainted, rail_check=rail_check, approved=approved, reason="ok",
        )

    # -- internals -------------------------------------------------------------

    async def _approve(self, request: ElicitationRequest) -> bool:
        if self.approver is None:
            return False
        outcome = self.approver(request)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return bool(outcome)

    @staticmethod
    def _as_content(raw: Any) -> dict[str, Any]:
        if raw is True:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        return {"value": raw}

    def _declined(
        self,
        request: ElicitationRequest,
        reason: str,
        *,
        approved: bool,
        rail_check: Any = None,
    ) -> ElicitationDecision:
        self._audit(request, "decline", reason)
        return ElicitationDecision(
            response=ElicitationResponse(action=ElicitationAction.DECLINE),
            rail_check=rail_check, approved=approved, reason=reason,
        )

    def _audit(self, request: ElicitationRequest, decision: str, reason: str, *, fields: Any = None) -> None:
        if self.audit is None:
            return
        details: dict[str, Any] = {"server": request.server, "reason": reason, "transport": "mcp"}
        if fields is not None:
            details["fields"] = fields
        self.audit.record("mcp_elicit", resource=request.server or "mcp", decision=decision, details=details)


def _flatten(content: dict[str, Any]) -> str:
    """Join a structured value's text for rail screening."""
    return "\n".join(str(v) for v in content.values() if v is not None)


# ---------------------------------------------------------------------------
# MCP Apps — surface a server's UI resource through the AG-UI channel
# ---------------------------------------------------------------------------

# MIME types an MCP-UI / MCP Apps host renders inline.
UI_MIME_TYPES: frozenset[str] = frozenset(
    {"text/html", "text/uri-list", "application/vnd.ag-ui+json"}
)


def is_ui_resource(uri: str, mime_type: str = "") -> bool:
    """Whether a resource is server-rendered UI (a ``ui://`` URI or a UI MIME)."""
    return uri.startswith("ui://") or mime_type in UI_MIME_TYPES


class MCPUIRender(BaseModel):
    """A governed render of a server's UI resource, ready for the AG-UI channel.

    The UI bytes are *untrusted external* content (a third-party server rendered
    them), token-metered against the run, and provenance-tagged. A render whose
    token cost exceeds the bridge's cap is :attr:`refused` (its content dropped)
    so a server UI can never blow the run's budget.
    """

    uri: str
    name: str = ""
    mime_type: str = "text/html"
    content: str = ""
    server: str = ""
    token_cost: int = 0
    trust_level: TrustLevel = TrustLevel.UNTRUSTED_EXTERNAL
    refused: bool = False
    reason: str = ""

    def to_agui(self, *, thread_id: str | None = None, run_id: str | None = None) -> Any:
        """Lower this render into an :class:`~vincio.server.agui.AGUIEvent`."""
        from ..server.agui import mcp_ui_event

        return mcp_ui_event(self, thread_id=thread_id, run_id=run_id)


class MCPAppBridge:
    """Surface a connected MCP server's UI resources through the AG-UI channel.

    Reads the server's ``ui://`` resources, governs each (audit + token budget +
    untrusted provenance), and lowers them into AG-UI events spliced onto the
    *existing* generative-UI stream — so server-rendered UI inherits the run's
    provenance, budget, and audit rather than opening a new, ungoverned path.
    """

    def __init__(
        self,
        client: Any,
        *,
        audit: Any | None = None,
        max_render_tokens: int = 4096,
    ) -> None:
        self.client = client
        self.audit = audit
        self.max_render_tokens = max_render_tokens

    async def renders(self) -> list[MCPUIRender]:
        """Discover and govern every UI resource the server advertises."""
        from ..core.tokens import count_tokens

        out: list[MCPUIRender] = []
        for resource in await self.client.list_ui_resources():
            content = await self.client.read_resource(resource.uri)
            cost = count_tokens(content)
            refused = cost > self.max_render_tokens
            render = MCPUIRender(
                uri=resource.uri,
                name=resource.name,
                mime_type=resource.mime_type,
                content="" if refused else content,
                server=getattr(self.client, "name", ""),
                token_cost=cost,
                refused=refused,
                reason=f"render exceeds max_render_tokens ({self.max_render_tokens})" if refused else "",
            )
            self._audit(render)
            out.append(render)
        return out

    async def to_agui_events(
        self, *, thread_id: str | None = None, run_id: str | None = None
    ) -> list[Any]:
        """The AG-UI events for every non-refused UI render."""
        return [
            render.to_agui(thread_id=thread_id, run_id=run_id)
            for render in await self.renders()
            if not render.refused
        ]

    async def stream(
        self, base: Any, *, thread_id: str | None = None, run_id: str | None = None
    ) -> Any:
        """Splice the server's UI events onto an AG-UI run stream before it finishes."""
        from ..server.agui import AGUIEventType

        ui_events = await self.to_agui_events(thread_id=thread_id, run_id=run_id)
        emitted = False
        async for event in base:
            if not emitted and getattr(event, "type", None) == AGUIEventType.RUN_FINISHED:
                for ui in ui_events:
                    yield ui
                emitted = True
            yield event
        if not emitted:
            for ui in ui_events:
                yield ui

    def _audit(self, render: MCPUIRender) -> None:
        if self.audit is None:
            return
        self.audit.record(
            "mcp_ui_render",
            resource=render.uri,
            decision="refused" if render.refused else "render",
            details={
                "server": render.server,
                "mime_type": render.mime_type,
                "token_cost": render.token_cost,
                "transport": "mcp",
            },
        )
