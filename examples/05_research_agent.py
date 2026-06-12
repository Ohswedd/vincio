"""Research agent with tools: bounded ReAct loop."""

from _shared import example_provider

from vincio import ContextApp

provider, model = example_provider(
    script=[
        {"tool_call": {"name": "web_search", "arguments": {"query": "vincio pricing page"}}},
        {"tool_call": {"name": "database_lookup", "arguments": {"table": "invoices", "key": "latest"}}},
        "Report: the pricing page lists $20/month but the latest invoice charged $25 — "
        "a $5 discrepancy, likely the legacy plan rate.",
    ]
)

app = ContextApp(name="research_agent", provider=provider, model=model)


def web_search(query: str) -> dict:
    """Search the web for current information."""
    return {"results": ["Pricing page: Pro plan $20/month"]}


def database_lookup(table: str, key: str) -> dict:
    """Look up internal records."""
    return {"record": {"invoice": "INV-77", "amount": 25.0}}


agent = app.agent(tools=[web_search, database_lookup], planner="react", max_steps=8)

if __name__ == "__main__":
    state = agent.run("Find the latest pricing discrepancy and draft a report")
    print("answer:", state.final_answer)
    print("tools used:", [r.tool_name for r in state.tool_results])
    print("metrics:", state.metrics())
