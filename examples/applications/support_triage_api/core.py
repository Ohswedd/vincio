"""Support ticket triage — the Vincio core (no web framework here).

This module is the *engine* of the triage service. It has **no** dependency on
FastAPI (or any web framework), so it imports and runs in CI with nothing but
Vincio installed, fully offline on the deterministic mock provider. The HTTP
shell in ``main.py`` is a thin adapter over the plain functions exposed here.

What the engine does, per ticket:

  * classifies the ticket into a typed :class:`Triage` (category / priority /
    summary) using an ``output_schema`` contract — the model's reply is parsed
    and validated into that Pydantic type, or the run fails loudly;
  * remembers the ticket under the reporting user's *semantic* memory scope, and
    recalls that user's prior tickets so triage is personalized over time;
  * registers an **approval-gated write tool** (``escalate_ticket``). A write is
    never fired implicitly — when a ticket warrants escalation the engine returns
    a *pending approval* describing the exact action a human must sign off on.

Run it offline (mock provider, no keys, no network)::

    from core import triage
    print(triage("I was double charged", "u1"))

Point it at a real model by setting one env var (and the matching key)::

    VINCIO_PROVIDER=openai VINCIO_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from vincio import ContextApp
from vincio.memory.stores import InMemoryMemoryStore
from vincio.providers import MockProvider, build_provider


# ---------------------------------------------------------------------------
# Provider helper — offline mock by default; a real provider when configured.
# ---------------------------------------------------------------------------
def _provider() -> tuple[Any, str]:
    """Offline mock by default; a real provider when VINCIO_PROVIDER is set."""
    name = os.environ.get("VINCIO_PROVIDER", "mock")
    if name == "mock":
        return MockProvider(), "mock-1"
    return build_provider(name), os.environ.get("VINCIO_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# The output contract: every triage is this exact shape, validated.
# ---------------------------------------------------------------------------
class Triage(BaseModel):
    """The structured verdict for one support ticket."""

    category: str = Field(description="billing | bug | account | feature | other")
    priority: str = Field(description="low | medium | high | critical")
    summary: str = Field(description="one-sentence summary of the issue")


# Priorities that warrant a human-approved escalation.
_ESCALATE_PRIORITIES = {"high", "critical"}


def _escalate_ticket(ticket_id: str, reason: str, priority: str) -> dict[str, Any]:
    """Escalate a ticket to a human on-call queue (an irreversible write).

    Registered with ``approval_required=True`` so the runtime never fires it
    without an explicit human sign-off — the triage path only ever *proposes* it.
    """
    return {"ticket_id": ticket_id, "escalated": True, "reason": reason, "priority": priority}


# ---------------------------------------------------------------------------
# App construction.
# ---------------------------------------------------------------------------
def build_app() -> ContextApp:
    """Build a fully-configured triage app: typed output, user-scoped semantic
    memory, and an approval-gated escalation write tool."""
    provider, model = _provider()
    app = ContextApp(
        name="support_triage",
        provider=provider,
        model=model,
        output_schema=Triage,
    )
    app.configure(
        role="support_ticket_triage_engine",
        objective="Classify each incoming support ticket and summarize it for routing.",
        rules=[
            "category is exactly one of: billing, bug, account, feature, other.",
            "priority is exactly one of: low, medium, high, critical.",
            "Never invent account, order, or payment details that were not provided.",
        ],
    )
    # Per-user semantic memory so repeat reporters get personalized triage.
    # An in-process store keeps the example deterministic and self-contained: a
    # fresh process starts with empty history (no `.vincio/` written to the cwd).
    # In production, pass a durable store URL (e.g. store="sqlite:///triage.db").
    app.add_memory(scope="user", strategy="semantic", store=InMemoryMemoryStore())
    # An irreversible write: gated behind a human approval, never auto-fired.
    app.add_tool(
        _escalate_ticket,
        name="escalate_ticket",
        permissions=["tickets:write"],
        side_effects="write",
        approval_required=True,
    )
    return app


# A single long-lived app instance backs the service, so user memory persists
# across requests. Built lazily on first use to keep import cheap.
_APP: ContextApp | None = None


def _app() -> ContextApp:
    global _APP
    if _APP is None:
        _APP = build_app()
    return _APP


# ---------------------------------------------------------------------------
# The one public entry point the HTTP layer calls.
# ---------------------------------------------------------------------------
def triage(ticket: str, user_id: str) -> dict[str, Any]:
    """Triage one ticket for one user.

    Returns a plain JSON-able dict: the validated :class:`Triage` fields, any
    pending (human-approval-required) escalations, a count of the user's prior
    related tickets, and the run's trace_id / cost for observability.

    Raises ``ValueError`` on empty input so the HTTP layer can map it to a 422.
    """
    if not ticket or not ticket.strip():
        raise ValueError("ticket text must not be empty")
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty")

    app = _app()

    # Recall this user's prior tickets *before* this one is recorded, so the
    # count reflects history rather than the message we're about to store.
    prior = app.recall(ticket, user_id=user_id, top_k=5)

    result = app.run(ticket)
    verdict: Triage = result.output  # validated Triage instance (or run errors)

    # Record the ticket under the user's semantic memory scope.
    app.remember(
        f"Support ticket [{verdict.category}/{verdict.priority}]: {verdict.summary}",
        user_id=user_id,
        entities=[verdict.category],
    )

    # An escalation is a write — propose it for human approval, never auto-run it.
    pending_approvals: list[dict[str, Any]] = []
    spec = app.tool_registry.get("escalate_ticket").spec
    if verdict.priority.lower() in _ESCALATE_PRIORITIES and spec.approval_required:
        pending_approvals.append(
            {
                "tool": "escalate_ticket",
                "reason": f"{verdict.priority} priority {verdict.category} ticket",
                "arguments": {
                    "ticket_id": result.trace_id,
                    "reason": verdict.summary,
                    "priority": verdict.priority,
                },
                "requires_human_approval": True,
            }
        )

    return {
        "category": verdict.category,
        "priority": verdict.priority,
        "summary": verdict.summary,
        "pending_approvals": pending_approvals,
        "prior_tickets": len(prior),
        "trace_id": result.trace_id,
        "cost_usd": round(result.cost_usd, 6),
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import json

    print(json.dumps(triage("I was double charged", "u1"), indent=2))
