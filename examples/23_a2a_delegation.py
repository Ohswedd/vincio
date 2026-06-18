"""A2A (Agent-to-Agent) — expose a crew and delegate across agents.

Expose a Vincio crew as an A2A agent (Agent Card + task lifecycle), reach it
over the protocol, and plug a remote A2A agent into a local crew as a *bounded,
traced* delegate — the guarantee a raw A2A SDK does not give you.

Runs fully offline using the in-process transport (the same code works over
HTTP behind the FastAPI server). No API keys needed.
"""

from __future__ import annotations

import asyncio

from _shared import example_provider

from vincio import ContextApp
from vincio.a2a import RemoteA2AAgent, connect_a2a_in_process


async def main() -> None:
    provider, model = example_provider()
    app = ContextApp(name="a2a_demo", provider=provider, model=model)

    # A crew with named roles, exposed over A2A in one call.
    crew = app.crew(
        members=[
            {"name": "researcher", "goal": "gather the numbers", "keywords": ["find", "data"]},
            {"name": "writer", "goal": "draft the recommendation"},
        ]
    )
    server = app.serve_a2a(crew, name="research_crew", description="Researches and writes briefs.")

    # The Agent Card advertises the crew's capabilities at /.well-known/agent.json.
    card = server.agent_card()
    print("agent card:", card["name"], "| skills:", [s["name"] for s in card["skills"]])

    # Reach the agent over A2A; the task runs the bounded crew end to end.
    client = connect_a2a_in_process(server)
    task = await client.send("Explain the Q3 refund trend")
    print("task:", task.status.state, "| output:", str(task.status.message.text)[:50])

    # A remote A2A agent as a local crew delegate — bounded + traced around the call.
    delegate = RemoteA2AAgent(
        connect_a2a_in_process(app.serve_a2a(name="pricing_agent")), name="pricing"
    )
    state = await delegate.run("What is the standard refund window?")
    print("delegate termination:", state.termination_reason)
    print("delegate answer:", str(state.final_answer)[:50])


if __name__ == "__main__":
    asyncio.run(main())
