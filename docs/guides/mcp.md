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
  **elicitation** routes to the `elicitation=` callback (e.g. a human gate).

The live client is kept on `app.mcp_clients["weather"]` for direct calls:

```python
client = app.mcp_clients["weather"]
print(await client.call_tool("get_weather", {"city": "Paris"}))
for tool in await client.list_tools():
    print(tool.name, tool.input_schema)
```

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

See [`examples/22_mcp_tools_and_resources.py`](../../examples/22_mcp_tools_and_resources.py)
and the [threat model](../security/threat-model.md) for the MCP trust boundary.
