"""The Assistant: a multi-turn chat product in a few lines.

``app.assistant(...)`` is a conversational, session-aware layer over the app.
Every turn is still a full ``ContextApp`` run (retrieval, grounding, validation,
rails, budget, trace, audit), threaded under one session with memory write-back
and an approval surface for write tools — so a chat product is a short loop, not
a hand-wired one.
"""

from _shared import example_provider

from vincio import ContextApp


def refund_create(invoice: str) -> str:
    """Issue a refund for an invoice (a write tool — gated behind approval)."""
    return f"refund issued for {invoice}"


def responder(request):
    """Deterministic offline behaviour; a real model decides these for itself."""
    convo = "\n".join(m.text for m in request.messages).lower()
    if "refund issued for" in convo:  # the tool ran in this turn's loop
        return "Done — the refund for INV-123 posts within 5 business days."
    if "requires approval" in convo:  # the write tool was gated
        return "That refund needs your approval before I can run it."
    if "refund" in convo and "inv-123" in convo:
        return {"tool_call": {"name": "refund_create", "arguments": {"invoice": "INV-123"}}}
    if "refund window" in convo or "my plan" in convo:
        return "You're on the Pro plan, which includes a 30-day refund window."
    return "How can I help with your account today?"


provider, model = example_provider(responder)
app = ContextApp(name="support_chat", provider=provider, model=model)
app.add_tool(refund_create, permissions=["billing:write"], approval_required=True,
             side_effects="write")


if __name__ == "__main__":
    chat = app.assistant(user_id="cust-42")

    turn1 = chat.send("What's my refund window? My plan is Pro.")
    print("assistant:", turn1.text)
    print("  (written back to session memory:", len(turn1.memory_writes), "item)")

    # A write tool surfaces as a pending approval — a chat reply never silently
    # runs it.
    turn2 = chat.send("Please refund invoice INV-123.")
    print("assistant:", turn2.text)
    print("  needs approval:", [a.tool for a in chat.pending_approvals])

    # The human approves; re-sending runs the tool through the permissioned,
    # audited runtime.
    chat.approve("refund_create")
    turn3 = chat.send("Approved — refund invoice INV-123 now.")
    print("assistant:", turn3.text)
    ran = [tr.tool_name for tr in turn3.result.tool_results if tr.status == "ok"]
    print("  tools run:", ran)

    print("\ntranscript:", len(chat.history()), "messages in session", chat.session_id)
