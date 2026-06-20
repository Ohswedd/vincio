"""Ecosystem & integration breadth: first-party connectors, the entry-point
plugin contract, a signed community pack/skill registry, deeper framework
interop (Haystack + DSPy), and the MCP-server marketplace bridge.

Runs fully offline on the deterministic mock provider — the REST connectors use
an injected httpx mock transport, the warehouse connectors an injected client.
"""

import asyncio

import httpx
from _shared import example_provider

from vincio import (
    BundleRecord,
    CommunityRegistry,
    ContextApp,
    installed_plugins,
    load_plugins,
)
from vincio.connectors import connect
from vincio.interop import from_dspy_module, from_haystack_retriever
from vincio.mcp import build_app_server
from vincio.packs import load_pack
from vincio.plugins import _EP
from vincio.registry import MCPRegistryClient, MCPServerRecord
from vincio.security.access import AllowListGate
from vincio.security.audit import AuditLog, HMACSigner

PROVIDER, MODEL = example_provider()  # MockProvider offline; real provider with env vars


def _mock(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def connectors() -> None:
    """1. First-party connectors feed the document engine with full provenance.
    Eight new sources land behind the same ``connect`` / ``register_connector``
    contract; each runs offline here against an injected client."""

    def jira_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "total": 1,
                "issues": [
                    {
                        "key": "ENG-1",
                        "fields": {
                            "summary": "SSO login fails",
                            "description": {
                                "type": "doc",
                                "content": [
                                    {"type": "paragraph", "content": [{"type": "text", "text": "Refunds within 30 days."}]}
                                ],
                            },
                            "status": {"name": "Open"},
                            "issuetype": {"name": "Bug"},
                            "project": {"key": "ENG"},
                        },
                    }
                ],
            },
        )

    issues = asyncio.run(
        connect("jira", base_url="https://acme.atlassian.net", email="a@b.c", token="t",
                client=_mock(jira_handler)).load()
    )
    doc = issues[0]
    print("1. connectors: jira ->", doc.title)
    print("   provenance:", doc.source_uri, "| connector =", doc.metadata["connector"])
    print("   also available:", "linear, gdrive, sharepoint, salesforce, zendesk, bigquery, snowflake")


def plugins() -> None:
    """2. Third-party providers, metrics, chunkers, rerankers, judges, connectors,
    and packs register on install via entry points, gated by a versioned contract.
    Here we inject a fake distribution's entry points to show the mechanism."""

    def acme_connector_factory(**opts):
        from vincio.core.types import Document

        class _Acme:
            name = "acme"

            async def load(self):
                return [Document(text="from the acme service")]

        return _Acme()

    eps = [
        _EP("acme", "vincio.connectors", "acme-demo", "0.1.0", lambda: acme_connector_factory),
        _EP("api_version", "vincio.plugins", "acme-demo", "0.1.0", lambda: "1.0"),
        # A plugin built for a future, incompatible plugin-API major is gated out.
        _EP("future", "vincio.connectors", "future-demo", "9.0.0", lambda: acme_connector_factory),
        _EP("api_version", "vincio.plugins", "future-demo", "9.0.0", lambda: "2.0"),
    ]
    statuses = {p.name: p.status for p in load_plugins(entry_points=eps)}
    print("2. plugins: acme ->", statuses["acme"], "| future ->", statuses["future"])
    print("   (run `vincio plugins list` to see installed plugins for real)")
    # Installed against the real environment there may be none; just show the call.
    print("   installed in this env:", len(installed_plugins()))
    from vincio.connectors import CONNECTORS

    CONNECTORS.pop("acme", None)  # keep the demo registry clean


def community_registry() -> None:
    """3. A governed, signed index of opt-in packs and SKILL.md bundles. Every
    resolution passes the same allow-list gate the agent fabric uses and is
    recorded as an audited access decision; the bundle's signature is verified."""
    audit = AuditLog(directory=None)
    signer = HMACSigner("publisher-key")  # Ed25519 for third-party verification
    registry = CommunityRegistry(
        allow_list=AllowListGate(allow=["support-pro"]), audit=audit, signer=signer
    )

    # A publisher signs and registers a domain pack bundle.
    pack = load_pack("support").model_copy(update={"name": "support-pro"})
    registry.publish_pack(pack, version="1.2.0", publisher="acme")
    registry.register(BundleRecord(name="evil-pack", kind="pack", payload={"name": "e", "description": ""}))

    loaded = registry.load_pack("support-pro")  # governed + audited + signature-verified
    print("3. registry: loaded", loaded.name, "under the allow-list gate")
    denied = registry.try_resolve("evil-pack")
    print("   unlisted bundle denied:", not denied.allowed, "->", denied.decision.reason)
    decisions = audit.query(action="bundle_resolve")
    print("   audited decisions:", [d.decision for d in decisions])


def interop() -> None:
    """4. Haystack and DSPy assets drop in duck-typed (no heavy imports). A
    Haystack retriever becomes a Vincio source; a compiled DSPy program becomes
    a tool."""

    class HaystackDoc:
        def __init__(self, content, meta, score):
            self.content, self.meta, self.score = content, meta, score

    class HaystackRetriever:
        def run(self, query):
            return {"documents": [HaystackDoc("refunds in 30 days", {"source": "kb"}, 0.92)]}

    hits = asyncio.run(from_haystack_retriever(HaystackRetriever()).search("refund window"))
    print("4. interop: haystack retriever ->", hits[0].chunk.text, f"(score {hits[0].score})")

    class DSPyField:
        def __init__(self, desc):
            self.json_schema_extra = {"desc": desc}

    class DSPySig:
        instructions = "Answer the question from context."
        input_fields = {"question": DSPyField("the user question")}
        output_fields = {"answer": DSPyField("the grounded answer")}

    class DSPyPrediction:
        def __init__(self, data):
            self._data = data

        def toDict(self):
            return self._data

    class CompiledDSPyProgram:
        signature = DSPySig()

        def __call__(self, **kwargs):
            return DSPyPrediction({"answer": f"Refund window for {kwargs['question']}: 30 days"})

    adapter = from_dspy_module(CompiledDSPyProgram())
    print("   dspy module as tool ->", adapter["handler"](question="Pro plan"))


def mcp_marketplace() -> None:
    """5. Discover an MCP server from a registry and land its tools in the
    permissioned runtime — in one governed call. Discovery (registry) +
    governance (allow-list + audit) + connection (the sandboxed runtime)."""
    # A provider app exposes a tool and is served as an MCP server (in-process here).
    provider_app = ContextApp(name="weather_provider", provider=PROVIDER, model=MODEL)

    @provider_app.tool_registry.register(name="get_weather")
    def get_weather(city: str) -> dict:
        """Look up the weather for a city."""
        return {"city": city, "temp_f": 72}

    provider_app.enabled_tools.append("get_weather")
    server = build_app_server(provider_app)

    consumer = ContextApp(name="consumer", provider=PROVIDER, model=MODEL)
    registry = MCPRegistryClient(
        catalog=[
            MCPServerRecord(name="weather", url="https://weather.example/mcp", description="weather"),
            MCPServerRecord(name="evil-server", url="https://evil.example/mcp"),
        ]
    )
    # One call: discover -> govern (allow-list + audit) -> connect.
    consumer.add_mcp_from_registry("weather", registry=registry, server=server, allow=["weather"])
    landed = [t for t in consumer.enabled_tools if t.startswith("weather.")]
    print("5. marketplace bridge: tools landed in the permissioned runtime ->", landed)
    decisions = consumer.audit.query(action="agent_resolve")
    print("   audited reachability decision:", [(d.resource, d.decision) for d in decisions])


if __name__ == "__main__":
    connectors()
    plugins()
    community_registry()
    interop()
    mcp_marketplace()
