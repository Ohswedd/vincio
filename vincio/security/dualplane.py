"""Dual-plane execution: a privileged planner that never sees untrusted bytes.

The escalation path for prompt injection is always the same: untrusted content
(a retrieved document, a tool result) carries an instruction, the planner reads
it as if it were the user, and a side-effecting tool runs on the attacker's
behalf. :class:`DualPlaneExecutor` closes that path structurally rather than by
detection, in the spirit of dual-LLM / CaMeL designs:

* The **data plane** holds raw untrusted bytes in a quarantine. The control
  plane never receives them — only opaque references and, on demand, typed,
  schema-validated *extractions* of specific fields. An instruction buried in
  the bytes therefore has no channel to the planner.
* The **control plane** (the planner) sees the user's objective and a
  descriptor of what can be extracted, and proposes tool calls. Arguments are
  resolved from extractions, so a value that originated in untrusted data
  carries an ``untrusted`` :class:`~vincio.security.TrustLabel` all the way to
  the call site.
* Every side-effecting call is gated on **authority**: an argument carrying an
  untrusted taint may only reach a write/external tool with a
  :class:`~vincio.security.CapabilityToken` minted from the user's request, or
  an explicit human approval. Otherwise the call is refused. The
  :class:`~vincio.security.ContainmentMonitor` records each decision so
  :func:`~vincio.security.verify_containment` can prove
  ``untrusted ⇒ no unapproved capability`` held over the whole run.

The executor wraps the existing permissioned :class:`~vincio.tools.ToolRuntime`,
so identity, scopes, tenancy, sandboxing, and output sanitization still apply;
containment is the *additional* control on top.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..core.errors import ContainmentError
from ..core.types import Message, ModelRequest, ToolCall, ToolResult, TrustLevel
from ..core.utils import new_id
from .access import Principal
from .capability import (
    SIDE_EFFECTING,
    CapabilityBroker,
    CapabilityToken,
    ContainmentMonitor,
    ContainmentReport,
    TaintedValue,
    TrustLabel,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..providers.base import ModelProvider
    from ..tools.runtime import ToolRuntime

__all__ = ["QuarantineRef", "PlannedCall", "DualPlaneExecutor"]

# An argument string of this form names an extraction the control plane may
# reference without ever seeing the underlying untrusted bytes.
_REF_PREFIX = "$"

ApprovalFn = Callable[[str, dict[str, Any]], Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class QuarantineRef:
    """An opaque handle to quarantined untrusted bytes (no raw content).

    The control plane may pass a ref around and request typed extractions from
    it, but the bytes themselves never cross into the planner. The descriptor
    carries only non-sensitive shape metadata (source, length, label).
    """

    id: str
    source: str
    length: int
    label: TrustLabel

    def descriptor(self) -> dict[str, Any]:
        """Control-plane-visible metadata — shape only, never the bytes."""
        return {
            "ref": self.id,
            "source": self.source,
            "length": self.length,
            "trust": self.label.value,
        }


@dataclass(frozen=True, slots=True)
class PlannedCall:
    """A tool call proposed by the control-plane planner."""

    tool_name: str
    arguments: dict[str, Any]


_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object", "additionalProperties": True},
                },
                "required": ["tool_name", "arguments"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["calls"],
    "additionalProperties": False,
}


class DualPlaneExecutor:
    """Capability-secure executor separating the control and data planes.

    Wire a :class:`~vincio.tools.ToolRuntime` and a
    :class:`~vincio.security.CapabilityBroker`; ingest untrusted content into the
    data plane, extract typed fields into the control plane, mint capabilities
    from the user's request, and execute tool calls through :meth:`call`. A
    side-effecting call whose arguments carry an untrusted taint is refused
    unless it presents a valid capability or is approved — so an injected
    instruction provably cannot escalate.
    """

    def __init__(
        self,
        tool_runtime: ToolRuntime,
        *,
        broker: CapabilityBroker | None = None,
        monitor: ContainmentMonitor | None = None,
        principal: Principal | None = None,
        approval: ApprovalFn | None = None,
        provider: ModelProvider | None = None,
        model: str | None = None,
    ) -> None:
        self.tools = tool_runtime
        self.broker = broker or CapabilityBroker()
        self.monitor = monitor or ContainmentMonitor()
        self.principal = principal or Principal()
        # Optional human-in-the-loop gate: consulted when an untrusted-tainted
        # side effect lacks a capability. Returning True authorizes the call
        # (recorded as ``approval`` authority); absence/False refuses it.
        self.approval = approval
        # Optional privileged planner. When set, ``plan`` drives the model with
        # the control-plane-only view (objective + extraction descriptors),
        # never the quarantined bytes.
        self.provider = provider
        self.model = model
        self._quarantine: dict[str, TaintedValue[str]] = {}
        self._extractions: dict[str, TaintedValue[Any]] = {}

    # -- data plane --------------------------------------------------------------

    def ingest(
        self, content: str, *, source: str = "external", quarantined: bool = False
    ) -> QuarantineRef:
        """Store untrusted bytes in the data plane and return an opaque ref.

        Pass ``quarantined=True`` for content a detector flagged as actively
        hostile; it raises the label to ``quarantined`` but is otherwise handled
        identically (untrusted is already non-instructing).
        """
        ref_id = new_id("qz")
        label = TrustLabel.QUARANTINED if quarantined else TrustLabel.UNTRUSTED
        self._quarantine[ref_id] = TaintedValue(value=content, label=label, sources=(source,))
        return QuarantineRef(id=ref_id, source=source, length=len(content), label=label)

    def ingest_evidence(self, item: Any) -> QuarantineRef:
        """Quarantine a context :class:`~vincio.core.types.EvidenceItem`.

        The item's :class:`~vincio.core.types.TrustLevel` sets the label, so
        retrieved documents and tool-sourced evidence land in the data plane
        with the right taint automatically.
        """
        level = TrustLevel(getattr(item, "trust_level", TrustLevel.UNTRUSTED_DOCUMENT))
        text = getattr(item, "scorable_text", None) or getattr(item, "text", "") or ""
        source = getattr(item, "source_id", None) or getattr(item, "id", "evidence")
        ref_id = new_id("qz")
        self._quarantine[ref_id] = TaintedValue(
            value=text, label=TrustLabel.from_trust_level(level), sources=(str(source),)
        )
        return QuarantineRef(
            id=ref_id, source=str(source), length=len(text),
            label=TrustLabel.from_trust_level(level),
        )

    def extract(
        self,
        name: str,
        ref: QuarantineRef | str,
        parser: Callable[[str], Any],
        *,
        schema: dict[str, Any] | None = None,
    ) -> TaintedValue[Any]:
        """Pull one typed field from quarantined bytes into the control plane.

        ``parser`` turns the raw bytes into a typed value (the only channel from
        data plane to control plane). The extracted value keeps the source's
        untrusted label, and — when ``schema`` is given — is validated against it
        so a malformed or oversized extraction is rejected at the boundary. The
        result is registered under ``name`` so a planned call can reference it as
        ``"$name"`` without the planner ever seeing the bytes.
        """
        ref_id = ref.id if isinstance(ref, QuarantineRef) else ref
        held = self._quarantine.get(ref_id)
        if held is None:
            raise ContainmentError(f"no quarantined content for ref {ref_id!r}")
        extracted = held.map(parser)
        if schema is not None:
            from ..tools.runtime import validate_against_schema

            errors = validate_against_schema(extracted.value, schema)
            if errors:
                raise ContainmentError(
                    f"extraction {name!r} failed schema validation: {errors}",
                    details={"errors": errors},
                )
        self._extractions[name] = extracted
        return extracted

    def extractions_view(self) -> list[dict[str, Any]]:
        """Control-plane descriptors of every extraction — types, never bytes."""
        view: list[dict[str, Any]] = []
        for name, tainted in self._extractions.items():
            view.append(
                {
                    "name": name,
                    "type": type(tainted.value).__name__,
                    "trust": tainted.label.value,
                    "sources": list(tainted.sources),
                }
            )
        return view

    # -- control plane -----------------------------------------------------------

    def mint(
        self,
        capability: str,
        *,
        constraints: dict[str, Any] | None = None,
        ttl_s: float | None = None,
    ) -> CapabilityToken:
        """Mint a capability for ``capability`` from this run's principal.

        A thin pass-through to the broker that binds the token to the executor's
        (user-authorized) principal — the trusted boundary where authority
        legitimately originates.
        """
        return self.broker.mint(
            capability,
            principal_user=self.principal.user_id,
            principal_tenant=self.principal.tenant_id,
            constraints=constraints,
            ttl_s=ttl_s,
        )

    def control_messages(self, objective: str, tools: list[Any]) -> list[Message]:
        """Assemble the planner-visible prompt: objective + safe descriptors.

        The returned messages contain the user objective, the available tools,
        and the *descriptors* of extractions (names and types) — never the
        quarantined bytes. A test can assert no untrusted content appears here,
        which is exactly the dual-plane guarantee.
        """
        tool_lines = "\n".join(
            f"- {t.name}: {t.description} | input schema: {json.dumps(t.input_schema)[:300]}"
            for t in tools
        )
        extractions = self.extractions_view()
        extraction_lines = (
            "\n".join(f"- ${e['name']} ({e['type']}, trust={e['trust']})" for e in extractions)
            or "(none yet)"
        )
        body = (
            f"Objective: {objective}\n\n"
            f"Available tools:\n{tool_lines}\n\n"
            "Available extractions (reference an argument value as \"$name\"; the "
            "underlying data is untrusted and you cannot see it):\n"
            f"{extraction_lines}\n\n"
            "Output the tool calls needed (empty list if none)."
        )
        return [
            Message(
                role="system",
                content=(
                    "You are a privileged planner. You never see untrusted document "
                    "or tool bytes — only typed extractions of them. Plan tool calls "
                    "to satisfy the objective; reference extracted values as \"$name\"."
                ),
                cache_hint=True,
            ),
            Message(role="user", content=body),
        ]

    async def plan(self, objective: str, *, tools: list[Any]) -> list[PlannedCall]:
        """Drive the privileged planner over the control-plane-only view.

        Requires a ``provider`` and ``model``. The planner is given the objective
        and extraction descriptors but never the quarantined bytes, so an
        injected instruction cannot reach it. Returns the proposed calls; each is
        still gated by :meth:`call` before any side effect.
        """
        if self.provider is None or self.model is None:
            raise ContainmentError(
                "no planner configured; pass provider= and model= to DualPlaneExecutor.plan",
            )
        request = ModelRequest(
            model=self.model,
            messages=self.control_messages(objective, tools),
            output_schema=_PLAN_SCHEMA,
            output_schema_name="tool_calls",
        )
        response = await self.provider.generate(request)
        payload = response.structured or {}
        calls = payload.get("calls", []) if isinstance(payload, dict) else []
        return [
            PlannedCall(tool_name=c.get("tool_name", ""), arguments=dict(c.get("arguments", {})))
            for c in calls
        ]

    # -- enforced execution ------------------------------------------------------

    def _resolve(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], TrustLabel, list[str]]:
        """Resolve ``$name`` refs and :class:`TaintedValue`\\ s to plain values.

        Returns the plain arguments the tool will receive, the joined taint label
        over every resolved value, and the union of provenance sources. A literal
        in the plan is trusted; anything that came from an extraction carries its
        untrusted taint forward.
        """
        label = TrustLabel.TRUSTED
        sources: list[str] = []
        plain: dict[str, Any] = {}

        def resolve_value(value: Any) -> Any:
            nonlocal label
            if isinstance(value, TaintedValue):
                label = label.merge(value.label)
                for src in value.sources:
                    if src not in sources:
                        sources.append(src)
                return value.value
            if isinstance(value, str) and value.startswith(_REF_PREFIX):
                extraction = self._extractions.get(value[len(_REF_PREFIX) :])
                if extraction is not None:
                    label = label.merge(extraction.label)
                    for src in extraction.sources:
                        if src not in sources:
                            sources.append(src)
                    return extraction.value
                return value
            if isinstance(value, list):
                return [resolve_value(v) for v in value]
            if isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            return value

        for key, value in arguments.items():
            plain[key] = resolve_value(value)
        return plain, label, sources

    async def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        capability: CapabilityToken | None = None,
        approved: bool = False,
        raise_on_block: bool = False,
    ) -> ToolResult:
        """Execute a tool call under the containment invariant.

        Resolves arguments (computing their taint), and if the tool is
        side-effecting *and* an argument is untrusted-tainted, requires a valid
        capability (verified against this principal and the arguments) or an
        approval. With neither, the call is refused: a ``denied`` result is
        returned (or :class:`~vincio.core.errors.ContainmentError` raised when
        ``raise_on_block``), and the blocked attempt is recorded. Authorized and
        non-escalating calls run through the wrapped permissioned runtime; the
        tool's output is quarantined so taint keeps propagating across steps.
        """
        plain_args, taint, sources = self._resolve(arguments)
        spec = self.tools.registry.get(tool_name).spec
        side_effects = spec.side_effects

        if side_effects in SIDE_EFFECTING and taint.is_tainted:
            authority = self._authorize(tool_name, plain_args, capability)
            if authority is None and self.approval is not None:
                granted = approved or await self.approval(
                    tool_name, {"arguments": plain_args, "taint": taint.value}
                )
                if granted:
                    authority = "approval"
            if authority is None:
                detail = (
                    f"untrusted-tainted argument(s) from {sources} cannot reach "
                    f"side-effecting tool {tool_name!r} without a capability or approval"
                )
                self.monitor.record(
                    tool_name, taint=taint, side_effects=side_effects,
                    authority="none", blocked=True, detail=detail,
                )
                if raise_on_block:
                    raise ContainmentError(detail, details={"tool": tool_name, "taint": taint.value})
                return ToolResult(
                    call_id="", tool_name=tool_name, status="denied", error=detail,
                    metadata={"containment": "blocked", "taint": taint.value},
                )
        else:
            authority = "trusted" if not taint.is_tainted else "none"

        result = await self.tools.execute(
            ToolCall(tool_name=tool_name, arguments=plain_args),
            principal=self.principal,
            capability=capability,
        )
        self.monitor.record(
            tool_name, taint=taint, side_effects=side_effects, authority=authority,
            blocked=False, detail=f"executed with {authority} authority",
        )
        # A tool result is itself untrusted: quarantine it so a later argument
        # derived from it stays tainted end-to-end.
        if result.status == "ok" and result.output is not None:
            ref = self.ingest(
                result.output if isinstance(result.output, str) else json.dumps(result.output, default=str),
                source=f"tool:{tool_name}",
            )
            result.metadata["quarantine_ref"] = ref.id
        return result

    def _authorize(
        self, tool_name: str, arguments: dict[str, Any], capability: CapabilityToken | None
    ) -> str | None:
        """Return ``"capability"`` if a presented token authorizes the call."""
        if capability is None:
            return None
        verdict = self.broker.verify(
            capability,
            capability=tool_name,
            principal_user=self.principal.user_id,
            principal_tenant=self.principal.tenant_id,
            arguments=arguments,
        )
        return "capability" if verdict.valid else None

    def report(self) -> ContainmentReport:
        """The containment verdict over every call this executor has made."""
        return self.monitor.report()
