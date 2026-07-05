# Agent-to-Agent (A2A)

[A2A](https://a2a-protocol.org) is the cross-vendor agent interoperability
protocol (Google → Linux Foundation): an **Agent Card** at
`/.well-known/agent.json` plus a JSON-RPC **task lifecycle**
(`submitted → working → input-required → completed/failed`). Vincio both serves
A2A and consumes it, and the edge over a raw A2A SDK is that a Vincio crew or
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

- **Crew**: each member becomes an advertised skill; the task runs the bounded
  crew (per-member budgets, termination guarantees) and returns its output as an
  artifact.
- **Graph**: a human-in-the-loop `interrupt()` surfaces as the
  `input-required` task state; a follow-up `message/send` carrying the same
  `taskId` resumes the checkpointed thread with the caller's answer.
- Pass `token_validator=` for OAuth2/API-key resource-server validation; the
  audit log records an `a2a_serve` entry per inbound task.

Serve it over HTTP behind the [FastAPI server](../reference/api.md), or
in-process for tests.

### How it works: the JSON-RPC task lifecycle

A2A is a small state machine over JSON-RPC. `message/send` opens an `A2ATask` in
`submitted`; it advances to `working` while the target runs, to `input-required`
when the target needs a human answer, and terminates in `completed` or `failed`.
Vincio's edge is that *your* side of the machine stays bounded and traced:

- A served **crew** runs each member under its own budget and the crew's
  termination guarantee, so an inbound task cannot spin forever; member output
  returns as `A2AArtifact`s on `task.artifacts`.
- A served **graph** checkpoints at every `interrupt()`. The pause surfaces as
  `input-required`; the caller resumes by sending a second `message/send` that
  carries the **same `taskId`** — a fresh id opens a new task and loses the
  checkpointed thread.
- `token_validator=` runs *before* any work, so an unauthenticated caller never
  reaches the crew, and each inbound task writes one `a2a_serve` audit entry.

The returned `A2ATask` carries `.status.state` (the terminal state),
`.status.message` (the reply), and `.artifacts` (typed outputs).

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
agent can be a member of *your* crew, while your crew keeps its budget,
termination, and tracing guarantees around the call:

```python
from vincio.a2a import RemoteA2AAgent, connect_a2a
from vincio.agents import AgentRole

pricing = RemoteA2AAgent(connect_a2a("https://pricing.example/rpc"), name="pricing")
crew.add(AgentRole(name="pricing", goal="price the deal"), pricing)
```

The remote call runs *inside* your crew's budget and round bounds: a slow or
looping counterparty is cut off by the same termination guarantee that bounds a
local member, so a foreign agent can never uncap your run.

## Best practice & gotchas

- **Resume with the original `taskId`.** An `input-required` task is only
  resumable by a follow-up `message/send` carrying the same id; treat the id as
  the handle to the checkpointed thread.
- **A `RemoteA2AAgent` is a black box under your bounds.** You inherit its
  answer, not its guarantees — keep it a *delegate* inside a crew (which caps its
  budget and rounds) rather than the whole task, so an unbounded peer stays
  bounded by you.
- **Authenticate at the edge, not in the handler.** Pass `token_validator=` to
  `serve_a2a`; validation happens before the crew runs, so a rejected caller
  costs nothing.
- **Agent Cards are advertisements, not guarantees.** A card lists advertised
  skills; the actual budget, termination, and audit come from *your* served
  crew/graph, not the peer's card.

## Testing offline

```python
from vincio.a2a import connect_a2a_in_process

client = connect_a2a_in_process(app.serve_a2a(crew, name="research_crew"))
task = await client.send("Explain the refund trend")
assert task.status.state == "completed"
```

See [`examples/10_interop_and_protocols.py`](../../examples/10_interop_and_protocols.py).

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: Model Context Protocol (MCP)](mcp.md)
- [Example: 10_interop_and_protocols.py](../../examples/10_interop_and_protocols.py)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#serving)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
