"""A conversational, session-aware layer over :class:`~vincio.ContextApp`.

``ContextApp.run`` executes one stateless context-engineering pipeline. A chat
product needs more: it has to thread turns into a session, carry conversational
state forward, gate write tools behind an approval, and remember what the user
said for next time. Hand-wiring that loop around ``run`` is the same boilerplate
in every app. :class:`Assistant` is that loop, written once.

It is a thin, transparent wrapper — every turn is still a full ``ContextApp``
run, so retrieval, grounding, validation, rails, budgets, tracing, and the audit
chain all apply unchanged. The Assistant adds exactly four things:

- **Session threading** — every turn runs under one stable ``session_id`` (and
  optional ``user_id`` / ``tenant_id``), so traces, cost, and memory recall are
  scoped to the conversation.
- **Multi-turn state via memory write-back** — each turn is written back to
  session-scoped memory, so the next turn's pipeline recalls it as scored,
  budgeted context. State flows through the context compiler, not a side channel.
- **Tool approvals** — an approval surface for write tools. Approval-required
  tools are denied by default (a chat reply can never silently run a write tool);
  the caller approves a tool, supplies an interactive callback, or pre-allows it.
- **A recorded transcript** — the running thread, available to the caller and as
  an :class:`~vincio.evals.simulator.Simulator` target for multi-turn evals.

Example::

    assistant = app.assistant(user_id="u-1")
    turn = assistant.send("I was charged twice this month")
    print(turn.text, turn.citations)
    # a write tool surfaces as a pending approval instead of running:
    if turn.approvals:
        assistant.approve("refund_create")
        turn = assistant.send("yes, please refund the duplicate")
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .core.types import RunResult
from .providers.base import run_sync
from .tools.permissions import ApprovalRequest

if TYPE_CHECKING:
    from .core.app import ContextApp

__all__ = ["Assistant", "AssistantTurn", "ApprovalRecord"]

# A caller-supplied interactive approval decision: True approves, False denies.
ApprovalResolver = Callable[[ApprovalRequest], "bool | Awaitable[bool]"]


class ApprovalRecord(BaseModel):
    """A tool-approval decision made during a turn."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # ``approved`` (ran), ``denied`` (caller said no), or ``pending`` (needs the
    # caller's decision — the tool did not run this turn).
    status: str = "pending"
    reason: str = ""


class AssistantTurn(BaseModel):
    """The outcome of one conversational turn."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_message: str
    text: str = ""
    output: Any = None
    citations: list[str] = Field(default_factory=list)
    approvals: list[ApprovalRecord] = Field(default_factory=list)
    memory_writes: list[str] = Field(default_factory=list)
    trace_id: str = ""
    cost_usd: float = 0.0
    result: RunResult | None = None

    @property
    def needs_approval(self) -> bool:
        """True when a tool this turn is waiting on the caller's approval."""
        return any(a.status == "pending" for a in self.approvals)


class Assistant:
    """A multi-turn conversational session over a :class:`ContextApp`.

    Construct via :meth:`ContextApp.assistant`. Drive it with :meth:`send` /
    :meth:`asend`; inspect the thread with :meth:`history`. Conversations are
    sequential, so one Assistant owns its app's conversational state for its
    lifetime.
    """

    def __init__(
        self,
        app: ContextApp,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
        memory_writeback: bool = True,
        auto_approve: list[str] | None = None,
        on_approval: ApprovalResolver | None = None,
        feature: str | None = "assistant",
    ) -> None:
        from .core.utils import new_id

        self.app = app
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.session_id = session_id or new_id("sess")
        self.memory_writeback = memory_writeback
        self.feature = feature
        self._auto_approve: set[str] = set(auto_approve or [])
        self._on_approval = on_approval
        self._transcript: list[dict[str, str]] = []
        self._turn_approvals: list[ApprovalRecord] = []

        # Multi-turn state is carried by session-scoped memory write-back, so the
        # memory engine must exist when write-back is on.
        if self.memory_writeback and self.app.memory is None:
            self.app.add_memory()

        # Install the approval surface. Chain to any callback already configured
        # so an app-level policy still has the final say when the Assistant has
        # no opinion.
        self._prior_callback = self.app.tool_runtime.permissions.approval_callback
        self.app.tool_runtime.permissions.approval_callback = self._resolve_approval

    # -- approvals -----------------------------------------------------------

    async def _resolve_approval(self, request: ApprovalRequest) -> bool:
        if request.tool in self._auto_approve:
            self._turn_approvals.append(
                ApprovalRecord(tool=request.tool, arguments=request.arguments,
                               status="approved", reason="pre-approved")
            )
            return True
        if self._on_approval is not None:
            decision = self._on_approval(request)
            if hasattr(decision, "__await__"):
                decision = await decision  # type: ignore[misc]
            granted = bool(decision)
            self._turn_approvals.append(
                ApprovalRecord(tool=request.tool, arguments=request.arguments,
                               status="approved" if granted else "denied",
                               reason="caller callback")
            )
            return granted
        if self._prior_callback is not None:
            granted = await self._prior_callback(request)
            self._turn_approvals.append(
                ApprovalRecord(tool=request.tool, arguments=request.arguments,
                               status="approved" if granted else "denied",
                               reason="app policy")
            )
            return granted
        # No standing decision: surface it for the caller and do not run the tool.
        self._turn_approvals.append(
            ApprovalRecord(tool=request.tool, arguments=request.arguments,
                           status="pending", reason="awaiting approval")
        )
        return False

    def approve(self, tool: str) -> Assistant:
        """Pre-approve a tool by name for subsequent turns."""
        self._auto_approve.add(tool)
        return self

    def revoke(self, tool: str) -> Assistant:
        """Remove a tool's standing approval."""
        self._auto_approve.discard(tool)
        return self

    @property
    def pending_approvals(self) -> list[ApprovalRecord]:
        """Tools from the most recent turn still awaiting a decision."""
        return [a for a in self._turn_approvals if a.status == "pending"]

    # -- turns ---------------------------------------------------------------

    async def asend(self, text: str, *, files: list[str] | None = None) -> AssistantTurn:
        """Run one conversational turn and return its :class:`AssistantTurn`."""
        self._turn_approvals = []
        result = await self.app.arun(
            text,
            files=files,
            session_id=self.session_id,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            feature=self.feature,
        )
        reply = self._reply_text(result)
        self._transcript.append({"role": "user", "content": text})
        self._transcript.append({"role": "assistant", "content": reply})

        memory_writes: list[str] = []
        if self.memory_writeback and self.app.memory is not None and reply:
            memory_writes = self._write_back(text, reply, result)

        turn = AssistantTurn(
            user_message=text,
            text=reply,
            output=result.output,
            citations=list(result.citations),
            approvals=list(self._turn_approvals),
            memory_writes=memory_writes,
            trace_id=result.trace_id,
            cost_usd=result.cost_usd,
            result=result,
        )
        self.app.events.emit(
            "assistant.turn",
            {
                "session_id": self.session_id,
                "turn": len(self._transcript) // 2,
                "needs_approval": turn.needs_approval,
                "memory_writes": len(memory_writes),
            },
            trace_id=result.trace_id,
        )
        return turn

    def send(self, text: str, *, files: list[str] | None = None) -> AssistantTurn:
        """Synchronous :meth:`asend`."""
        return run_sync(self.asend(text, files=files))

    # -- transcript ----------------------------------------------------------

    def history(self) -> list[dict[str, str]]:
        """The recorded thread as a list of ``{role, content}`` messages."""
        return list(self._transcript)

    def messages(self) -> list[dict[str, str]]:
        """Alias for :meth:`history` (matches the simulator's agent contract)."""
        return self.history()

    def reset(self) -> Assistant:
        """Start a fresh conversation: clear the transcript and rotate the session."""
        from .core.utils import new_id

        self._transcript = []
        self._turn_approvals = []
        self.session_id = new_id("sess")
        return self

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _reply_text(result: RunResult) -> str:
        """The natural-language reply for the transcript."""
        if result.raw_text:
            return result.raw_text
        output = result.output
        if isinstance(output, str):
            return output
        return "" if output is None else str(output)

    def _write_back(self, user_text: str, reply: str, result: RunResult) -> list[str]:
        """Persist the turn to session-scoped memory so the next turn recalls it.

        A turn is written under SESSION scope (which the engine tiers as
        ``episodic``) and typed ``summary`` — a recap of the exchange, not a
        semantic fact — so the guarded write policy never treats two different
        turns as contradictory facts to reconcile.
        """
        assert self.app.memory is not None
        content = f"User said: {user_text.strip()}\nAssistant answered: {reply.strip()}"
        written: list[str] = []
        try:
            item = self.app.memory.remember(
                content,
                session_id=self.session_id,
                user_id=self.user_id,
                tenant_id=self.tenant_id,
                type="summary",
                confidence=0.9,
                source_trace_id=result.trace_id,
                metadata={"kind": "conversation_turn"},
            )
            written.append(item.id)
        except Exception:  # noqa: BLE001 - memory write must never break a reply
            pass
        return written
