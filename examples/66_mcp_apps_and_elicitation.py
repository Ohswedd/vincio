"""MCP Apps & the evolving MCP spec — server UI and governed mid-call input.

Vincio already speaks MCP in-process — tools through the permissioned runtime,
resources as cited evidence — and streams a run as AG-UI generative-UI events.
This example lands the spec's newer surface in the *same* governed, audited,
budgeted runtime, never as a hosted service:

  1. MCP Apps (server-rendered UI): a server's ``ui://`` resource surfaced
     through the existing AG-UI channel as an ``mcp.ui`` event — inheriting the
     run's provenance (untrusted-external), budget (an oversized render is
     refused), and audit.
  2. A UI resource embedded on a tool result, surfaced the same way.
  3. Elicitation: a server's mid-call request for input, screened through the
     app's input rails and tainted untrusted — so an accepted value is contained
     like any other untrusted input, and a secret value is refused.
  4. Elicitation behind an approval gate, exactly like a write tool.
  5. Evolving-spec parity: protocol-version negotiation and the stateless-core
     transport mode.

Everything here is additive and runs fully offline (no provider, store, or
network). The established MCP client/server paths are unchanged.
"""

from __future__ import annotations

from vincio import ContextApp
from vincio.core.types import ToolCall
from vincio.mcp import (
    SUPPORTED_PROTOCOL_VERSIONS,
    ElicitationGate,
    ElicitationPolicy,
    ElicitationRequest,
    MCPServer,
    MCPUIResource,
    connect_in_process,
    negotiate_version,
)
from vincio.providers import MockProvider


def main() -> None:
    print("MCP Apps & the evolving MCP spec — server UI and governed input\n")

    # 1. MCP Apps: a server's UI resource surfaced through the AG-UI channel.
    print("1. MCP Apps — a server's ui:// resource rides the AG-UI generative-UI channel")
    provider = ContextApp(name="dashboard_server", provider=MockProvider(), model="mock-1")
    ui = MCPUIResource.from_html("ui://sales", "<h1>Live sales</h1>", name="sales dashboard")
    server = provider.serve_mcp(ui_resources=[ui])

    consumer = ContextApp(name="host", provider=MockProvider(), model="mock-1")
    consumer.add_mcp_server("dashboard_server", server=server)
    bridge = consumer.mcp_app("dashboard_server")
    events = bridge_to_events(bridge)
    ev = events[0]
    print(
        f"   AG-UI event: type={ev.type} name={ev.name} uri={ev.value['uri']!r}\n"
        f"   provenance: trustLevel={ev.value['trustLevel']!r} server={ev.value['server']!r}  "
        f"(audited: {audited(consumer, 'mcp_ui_render')})"
    )

    # 2. Budget: an oversized render is token-metered and refused, never streamed.
    print("\n2. Budget — an oversized server render is refused, not streamed")
    big_provider = ContextApp(name="big_server", provider=MockProvider(), model="mock-1")
    big_ui = MCPUIResource.from_html("ui://huge", "<div>" + "row " * 3000 + "</div>", name="huge")
    big_consumer = ContextApp(name="host2", provider=MockProvider(), model="mock-1")
    big_consumer.add_mcp_server("big_server", server=big_provider.serve_mcp(ui_resources=[big_ui]))
    big_bridge = big_consumer.mcp_app("big_server", max_render_tokens=64)
    render = run_async(big_bridge.renders())[0]
    print(
        f"   render token_cost={render.token_cost} (cap 64) -> refused={render.refused}  "
        f"events emitted={len(bridge_to_events(big_bridge))}"
    )

    # 3. A UI resource embedded on a tool result.
    print("\n3. Embedded UI — a tool returns server-rendered UI alongside its text")
    app_server = ContextApp(name="reports", provider=MockProvider(), model="mock-1")

    def open_report() -> MCPUIResource:
        """Open the quarterly report UI."""
        return MCPUIResource.from_html("ui://q3", "<h1>Q3 report</h1>", name="q3", description="Q3 report")

    app_server.add_tool(open_report)
    client = connect_in_process(app_server.serve_mcp())
    text, ui_resources = run_async(client.call_tool_ui("open_report", {}))
    print(f"   tool text={text!r}  embedded UI={[r.uri for r in ui_resources]}")

    # 4. Elicitation: governed by the input rails + taint (contained like any
    #    untrusted input), and a secret value is refused.
    print("\n4. Elicitation — a server's mid-call input is rail-screened and tainted untrusted")
    forms = ContextApp(name="forms_host", provider=MockProvider(), model="mock-1")
    forms.add_rail(
        name="no_secrets", kind="safety", detectors=["secrets"], direction="input", action="block"
    )
    accept_gate = ElicitationGate(
        lambda msg, schema: {"email": "user@example.com"},
        rail_engine=forms.rail_engine,
        audit=forms.audit,
    )
    accepted = run_async(accept_gate.decide(ElicitationRequest(message="your email?", server="forms")))
    print(
        f"   benign value -> {accepted.response.action.value}; "
        f"tainted={accepted.tainted.label.value} source={accepted.tainted.sources[0]!r}"
    )
    refuse_gate = ElicitationGate(
        lambda msg, schema: {"token": "sk-ABCD1234567890abcdef1234567890abcdef"},
        rail_engine=forms.rail_engine,
        audit=forms.audit,
    )
    refused = run_async(refuse_gate.decide(ElicitationRequest(message="api key?", server="forms")))
    print(f"   secret value -> {refused.response.action.value} ({refused.reason})")

    # 5. Elicitation behind an approval gate, exactly like a write tool.
    print("\n5. Approval — an elicitation can be gated behind an approval, like a write tool")
    approval_gate = ElicitationGate(
        lambda msg, schema: {"confirm": True},
        policy=ElicitationPolicy(require_approval=True),
        approver=lambda req: False,  # deny
        rail_engine=forms.rail_engine,
    )
    denied = run_async(approval_gate.decide(ElicitationRequest(message="confirm charge?", server="forms")))
    print(f"   approval denied -> {denied.response.action.value} (approved={denied.approved})")

    # End-to-end: a server tool elicits; the consumer governs and contains it.
    pay = MCPServer(name="pay")
    pay._list_tools = lambda: [{"name": "charge", "description": "", "inputSchema": {"type": "object"}}]

    async def _charge(name, args):
        return {"text": str(await pay.elicit("card token?", schema={"type": "object"}))}

    pay._call_tool = _charge
    pay_host = ContextApp(name="pay_host", provider=MockProvider(), model="mock-1")
    pay_host.add_rail(
        name="no_secrets", kind="safety", detectors=["secrets"], direction="input", action="block"
    )
    pay_host.add_mcp_server(
        "pay", server=pay,
        elicitation=lambda msg, schema: {"token": "sk-ABCD1234567890abcdef1234567890abcdef"},
    )
    charge = run_async(pay_host.tool_runtime.execute(ToolCall(tool_name="pay.charge", arguments={})))
    print(f"   end-to-end server-tool elicit -> {charge.output} (secret contained by the host's rail)")

    # 6. Evolving-spec parity: version negotiation + stateless transport.
    print("\n6. Spec parity — protocol-version negotiation and the stateless-core transport")
    print(
        f"   supported revisions: {list(SUPPORTED_PROTOCOL_VERSIONS)}\n"
        f"   negotiate('2024-11-05') -> {negotiate_version('2024-11-05')!r}  "
        f"negotiate(unknown) -> {negotiate_version('3000-01-01')!r}"
    )

    print("\nDone — MCP Apps and elicitation, in the same governed, audited, budgeted runtime.")


def bridge_to_events(bridge):
    return run_async(bridge.to_agui_events())


def audited(app, action: str) -> bool:
    return any(e.action == action for e in app.audit.entries)


def run_async(coro):
    import asyncio

    return asyncio.run(coro)


if __name__ == "__main__":
    main()
