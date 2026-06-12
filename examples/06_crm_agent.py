"""CRM support agent with memory, permissions, and approval-gated writes."""

import asyncio

from _shared import example_provider
from pydantic import BaseModel

from vincio import ContextApp


class RefundDecision(BaseModel):
    decision: str
    reason: str
    evidence_ids: list[str]
    requires_human: bool


provider, model = example_provider(
    script=[
        {"tool_call": {"name": "billing_lookup", "arguments": {"invoice_id": "INV-123"}}},
        '{"decision": "eligible", "reason": "Paid invoice within the 30-day window. [E1]", '
        '"evidence_ids": ["E1"], "requires_human": true}',
    ]
)

app = ContextApp(name="support_refunds", output_schema=RefundDecision, provider=provider, model=model)
app.add_memory(scope="user", strategy="semantic")
app.policies.custom["scopes"] = ["billing:read"]  # the runtime principal's scopes


def billing_lookup(invoice_id: str) -> dict:
    """Look up a billing record."""
    return {"invoice_id": invoice_id, "amount": 49.0, "status": "paid", "age_days": 12}


def refund_create(invoice_id: str, amount: float) -> dict:
    """Issue a refund (write action)."""
    return {"refunded": amount, "invoice_id": invoice_id}


app.add_tool(billing_lookup, permissions=["billing:read"])
app.add_tool(refund_create, permissions=["billing:write"], side_effects="write", approval_required=True)


async def main():
    # The agent can read billing data but cannot issue refunds: the write
    # tool requires the billing:write scope AND human approval.
    agent = app.agent(max_steps=6, planner="react")
    state = await agent.arun(
        "Customer asks for refund on invoice INV-123. Decide eligibility but do not issue a refund."
    )
    print("decision:", state.final_answer)
    print("tools:", [(r.tool_name, r.status) for r in state.tool_results])

    # Remember the interaction outcome for next time.
    app.memory.write_fact(
        "Customer with invoice INV-123 was found refund-eligible", scope="user", owner_id="cust_42"
    )
    print("memory:", [r.item.content for r in app.memory.search("refund eligibility", user_id="cust_42")])


if __name__ == "__main__":
    asyncio.run(main())
