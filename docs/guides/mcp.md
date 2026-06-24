# Model Context Protocol (MCP)

Vincio speaks [MCP](https://modelcontextprotocol.io) both ways: it **consumes**
MCP servers (their tools, resources, and prompts) and **serves** a `ContextApp`
as an MCP server. The edge over a thin adapter: an MCP tool runs through the
*same* permissioned, sandboxed, audited, budgeted runtime as a native tool, and
an MCP resource becomes cited evidence with provenance.

MCP uses only the core `httpx` dependency — no SDK. Transports are **stdio**,
**Streamable HTTP**, and an **in-process** transport for offline tests.

## Consume an MCP server

```python
from vincio import ContextApp

app = ContextApp(name="assistant")

# stdio — launch the server as a subprocess
app.add_mcp_server("weather", command=["python", "weather_server.py"])

# or Streamable HTTP
app.add_mcp_server("kb", url="https://mcp.example.com/rpc",
                   headers={"Authorization": "Bearer …"})

result = app.run("What's the weather in Paris and what does the policy say?")
```

`add_mcp_server` connects, negotiates capabilities, and registers the server's
surface into the app:

- **Tools** register through the existing `ToolRegistry`, namespaced
  `"<name>.<tool>"`, so they inherit RBAC/ABAC scopes, the permission
  lifecycle, idempotency keys, reliability scoring, the subprocess sandbox, and
  the audit log — unchanged. Pass `permissions=["mcp:weather"]` to additionally
  gate them behind a scope.
- **Resources** become `EvidenceItem`s with `metadata["origin"] = "mcp:<name>"`,
  so the compiler chunks, scores, budgets, and cites them like any document.
  (Pass `resources=False` to skip.)
- **Server-initiated sampling** routes to the app's model provider;
  **elicitation** (a mid-call request for user input) routes to a governed
  `ElicitationGate` — see [Elicitation](#elicitation-governed-mid-call-input).

The live client is kept on `app.mcp_clients["weather"]` for direct calls:

```python
client = app.mcp_clients["weather"]
print(await client.call_tool("get_weather", {"city": "Paris"}))
for tool in await client.list_tools():
    print(tool.name, tool.input_schema)
```

### Discover from a registry (marketplace bridge)

`add_mcp_from_registry` composes discovery, governance, and connection in one
call: an `MCPRegistryClient` (the official MCP Registry or an offline catalog)
finds the server, a governed `AgentDirectory` under an `AllowListGate` decides
reachability — recorded as an audited access decision on the app's audit chain —
and the server's tools land in the permissioned runtime exactly as above.

```python
from vincio.registry import MCPRegistryClient

registry = MCPRegistryClient()                       # or catalog=[...] offline
app.add_mcp_from_registry("weather", registry=registry, allow=["weather"])
# unlisted servers raise AccessDeniedError; the decision is on app.audit
```

Pass `directory=` to reuse an existing governed directory, or `allow` / `deny`
fnmatch globs to build one (fail-closed; defaults to allowing exactly the named
server). For offline or in-process use, pass `server=` (an in-process
`MCPServer`) or `transport=`; otherwise the resolved server's URL or stdio
command is used.

### Lower-level client

```python
from vincio.mcp import connect_stdio, connect_http, connect_in_process

client = connect_stdio(["python", "server.py"])
await client.initialize()
tools = await client.list_tools()
text = await client.call_tool("search", {"q": "refunds"})
docs = await client.read_resource("file:///policy.md")
await client.aclose()
```

OAuth 2.1 is supported via the standard seams: `pkce_pair()` for the client
authorization-code flow and a bearer token in `headers` for Streamable HTTP.

## Serve your app as an MCP server

One `ContextApp` is both a consumer and a provider of context:

```python
server = app.serve_mcp()           # an MCPServer
```

- Registered tools become MCP tools (JSON Schema derived from the same type
  hints), evidence/sources become MCP resources, the prompt spec becomes an MCP
  prompt.
- The deterministic policy engine and the hash-chained audit log are enforced
  on every inbound call (each `tools/call` writes an `mcp_serve` audit entry).
- OAuth 2.1 resource-server validation: pass
  `token_validator=static_token_validator({"…"})` (or your own JWT/introspection
  callback).

Run it over stdio with the CLI or directly:

```bash
vincio mcp serve app.py                 # reads JSON-RPC on stdin
```

```python
from vincio.mcp import serve_stdio
await serve_stdio(server)
```

## MCP Apps: server-rendered UI through the AG-UI channel

[MCP Apps](https://modelcontextprotocol.io) let a server expose interactive UI as
a resource (a `ui://` URI with an HTML or AG-UI body). Vincio surfaces that UI
through its *existing* [generative-UI / AG-UI](agent-fabric.md) channel rather
than opening a new, ungoverned path — so server UI rides one streamed run and
inherits its provenance, budget, and audit.

```python
app.add_mcp_server("dashboards", server=dashboard_server)
bridge = app.mcp_app("dashboards")              # an MCPAppBridge

# Lower each ui:// resource into an AG-UI CUSTOM "mcp.ui" event.
for event in await bridge.to_agui_events():
    ...   # event.name == "mcp.ui"; event.value = {uri, mimeType, content, server, trustLevel}

# Or splice the UI events onto an existing AG-UI run stream (before RUN_FINISHED):
async for event in bridge.stream(run_stream_to_agui(app.astream("..."))):
    ...
```

Each render is **governed**:

- **Provenance.** The UI bytes are *untrusted external* content — a third-party
  server rendered them — so the event carries `trustLevel="untrusted_external"`
  and the originating server.
- **Budget.** The render is token-metered; one whose cost exceeds
  `max_render_tokens` (default 4096) is **refused** (its content dropped, no event
  emitted) so a server UI can never blow the run's budget.
- **Audit.** Every render and refusal lands on the hash-chained audit log as an
  `mcp_ui_render` entry.

A tool result may also **embed** a UI resource (the MCP Apps pattern of returning
UI alongside text). Surface both with `call_tool_ui`, and serve UI from your own
app either as a static resource (`app.serve_mcp(ui_resources=[MCPUIResource…])`) or
by returning an `MCPUIResource` from a tool:

```python
text, ui_resources = await client.call_tool_ui("open_dashboard", {})
```

## Elicitation: governed mid-call input

A server may ask the user for a structured value mid-call (`elicitation/create`).
Vincio gates that request with the **same approval and rail machinery a write tool
passes**, so an elicited value is contained like any other untrusted input:

```python
app.add_rail(name="no_secrets", kind="safety", detectors=["secrets"],
             direction="input", action="block")

app.add_mcp_server(
    "forms",
    server=forms_server,
    elicitation=collect_value,                 # (message, schema) -> dict | None
    elicitation_approval=lambda req: ...,      # optional: gate the request first
)
```

The `ElicitationGate` runs, in order:

1. **Approval.** If the policy requires it, an `elicitation_approval` callable must
   grant the request before any value is collected — the gate a write tool passes.
2. **Collect.** The `elicitation=` collector obtains the user's value (a falsy
   return is an explicit decline).
3. **Rail screen.** The value is run through the app's *input* rails; a secret,
   PII, or injection value is **declined** (an injection-flagged value is declined
   under `forbid_quarantined`, the default).
4. **Taint.** An accepted value is wrapped `TaintedValue.untrusted(...)` with a
   `mcp:<server>:elicitation` source, so downstream code that consumes it inherits
   the taint and it can never silently authorize a side effect.

Every decision is audited (`mcp_elicit`). Build a gate directly for full control:

```python
from vincio.mcp import ElicitationGate, ElicitationPolicy, ElicitationRequest

gate = ElicitationGate(collect_value, policy=ElicitationPolicy(require_approval=True),
                       approver=approve_fn, rail_engine=app.rail_engine, audit=app.audit)
decision = await gate.decide(ElicitationRequest(message="email?", server="forms"))
decision.accepted, decision.tainted, decision.to_wire()
```

## The evolving MCP spec

Vincio tracks the spec's revisions while staying interoperable:

- **Version negotiation.** The client requests the latest revision it implements;
  a server echoes a supported revision and the client records it
  (`client.negotiated_version`). `negotiate_version(requested)` honours a peer
  pinned to an older stable revision in `SUPPORTED_PROTOCOL_VERSIONS`, falling back
  to the latest for an unknown one — so a capability never silently breaks across a
  spec-revision boundary.
- **Stateless-core transport.** `StreamableHTTPTransport(url, stateless=True)`
  never tracks or sends an `Mcp-Session-Id`, so each request is self-contained and
  can be served by any stateless worker behind a load balancer.

## CLI

```bash
vincio mcp tools --command "python weather_server.py"   # inspect a server
vincio mcp tools --url https://mcp.example/rpc --resources --json
vincio mcp add app.py --name weather --command "python weather_server.py"
vincio mcp serve app.py                                  # expose an app
```

## Testing offline

Use the in-process transport — no network, fully deterministic:

```python
from vincio.mcp import MCPServer, connect_in_process

server = MCPServer(name="calc", list_tools=lambda: [...], call_tool=my_handler)
client = connect_in_process(server)
assert await client.call_tool("add", {"a": 2, "b": 3}) == "5"
```

See [`examples/10_interop_and_protocols.py`](../../examples/10_interop_and_protocols.py),
[`examples/10_interop_and_protocols.py`](../../examples/10_interop_and_protocols.py),
and the [threat model](../security/threat-model.md) for the MCP trust boundary.
