"""MCP (Model Context Protocol) — consume a server *and* expose your app.

Vincio speaks MCP both ways. Here we (1) connect to an MCP server and register
its tools + resources into an app — they run through the *same* permissioned,
sandboxed, audited, budgeted runtime as native tools, and resources become
cited evidence — and (2) expose the app itself as an MCP server.

Runs fully offline using the in-process transport (the same code path works
over stdio or Streamable HTTP). No API keys needed.
"""

from __future__ import annotations

import asyncio

from _shared import example_provider, json_responder

from vincio import ContextApp
from vincio.mcp import MCPServer, build_app_server, connect_in_process


def weather_server() -> MCPServer:
    """A tiny in-process MCP server — stands in for any stdio/HTTP server."""

    def list_tools():
        return [
            {
                "name": "get_weather",
                "description": "Current weather for a city.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

    async def call_tool(name, args):
        if name == "get_weather":
            return {"text": f"{args['city']}: 22°C, sunny"}
        return {"_mcp_error": True, "text": f"unknown tool {name}"}

    def list_resources():
        return [{"uri": "weather://policy", "name": "policy", "mimeType": "text/plain"}]

    async def read_resource(uri):
        return {"uri": uri, "mimeType": "text/plain", "text": "Forecasts refresh every 3 hours."}

    return MCPServer(
        name="weather",
        list_tools=list_tools,
        call_tool=call_tool,
        list_resources=list_resources,
        read_resource=read_resource,
    )


async def main() -> None:
    provider, model = example_provider(json_responder({"answer": "It is sunny."}))
    app = ContextApp(name="mcp_demo", provider=provider, model=model)

    # 1) Consume an MCP server: its tools register through the existing runtime,
    #    its resources become evidence with `origin: mcp:<server>` provenance.
    app.add_mcp_server("weather", server=weather_server(), resources=True)
    print("registered MCP tools:", [t for t in app.enabled_tools if t.startswith("weather.")])
    print("MCP resource as evidence:", app.pending_evidence[0].metadata["origin"])

    # The tool is callable through the connected client (and via the agent loop).
    client = app.mcp_clients["weather"]
    print("get_weather(Paris):", await client.call_tool("get_weather", {"city": "Paris"}))

    # 2) Expose THIS app as an MCP server — one ContextApp, both consumer & provider.
    def summarize(text: str) -> str:
        return text[:40] + ("…" if len(text) > 40 else "")

    app.add_tool(summarize, description="Summarize text to 40 chars.")
    server = build_app_server(app)  # == app.serve_mcp()
    consumer = connect_in_process(server)
    served = await consumer.list_tools()
    print("app exposed over MCP:", [t.name for t in served])
    print("call summarize:", await consumer.call_tool("summarize", {"text": "x" * 60}))


if __name__ == "__main__":
    asyncio.run(main())
