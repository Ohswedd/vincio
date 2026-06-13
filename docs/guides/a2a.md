# Agent-to-Agent (A2A)

> **Experimental (1.1).** The A2A surface (`vincio.a2a`, `app.serve_a2a`) is new
> and may change within the 1.x line.

[A2A](https://a2a-protocol.org) is the cross-vendor agent interoperability
protocol (Google → Linux Foundation): an **Agent Card** at
`/.well-known/agent.json` plus a JSON-RPC **task lifecycle**
(`submitted → working → input-required → completed/failed`). Vincio both serves
A2A and consumes it — and the edge over a raw A2A SDK is that a Vincio crew or
graph stays **bounded, terminating, and traced** across the delegation.

A2A uses only the core `httpx` dependency. Transports are HTTP and an in-process
transport for offline tests.

## Expose a crew, graph, or app over A2A

```python
from vincio import ContextApp

app = ContextApp(name="research")
crew = app.crew(members=[
    {"name": "researcher", "goal": "gather the numbers", "keywords": ["find"]},
    {"name": "writer", "goal": "draft the recommendation"},
])

server = app.serve_a2a(crew, name="research_crew", url="https://agent.example")
server.agent_card()          # the /.well-known/agent.json document
```

`serve_a2a(target)` accepts a `Crew`, a compiled `StateGraph`, or `None` (the
app itself):

- **Crew** — each member becomes an advertised skill; the task runs the bounded
  crew (per-member budgets, termination guarantees) and returns its output as an
  artifact.
- **Graph** — a human-in-the-loop `interrupt()` surfaces as the
  `input-required` task state; a follow-up `message/send` carrying the same
  `taskId` resumes the checkpointed thread with the caller's answer.
- Pass `token_validator=` for OAuth2/API-key resource-server validation; the
  audit log records an `a2a_serve` entry per inbound task.

Serve it over HTTP behind the [FastAPI server](../reference/api.md), or
in-process for tests.

## Reach a remote agent

```python
from vincio.a2a import connect_a2a

client = connect_a2a("https://other-agent.example/rpc",
                     headers={"Authorization": "Bearer …"})
card = await client.agent_card()
task = await client.send("Summarize the Q3 refund trend")
print(task.status.state, task.status.message.text)
```

## A remote agent as a local crew delegate

A `RemoteA2AAgent` implements the `AgentExecutor` contract, so another vendor's
agent can be a member of *your* crew — while your crew keeps its budget,
termination, and tracing guarantees around the call:

```python
from vincio.a2a import RemoteA2AAgent, connect_a2a
from vincio.agents import AgentRole

pricing = RemoteA2AAgent(connect_a2a("https://pricing.example/rpc"), name="pricing")
crew.add(AgentRole(name="pricing", goal="price the deal"), pricing)
```

## Testing offline

```python
from vincio.a2a import connect_a2a_in_process

client = connect_a2a_in_process(app.serve_a2a(crew, name="research_crew"))
task = await client.send("Explain the refund trend")
assert task.status.state == "completed"
```

See [`examples/23_a2a_delegation.py`](../../examples/23_a2a_delegation.py).
