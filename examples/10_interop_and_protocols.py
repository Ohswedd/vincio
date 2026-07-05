"""Interop & protocols — one runtime around the whole ecosystem.

Vincio is a hub: it speaks the open agent protocols (MCP both ways, A2A), loads
portable Anthropic-style Agent Skills, adapts assets from the major Python AI
frameworks (LangChain / LlamaIndex / Haystack / DSPy), ships full-stack vertical
packs, and reaches any provider or vector store behind one API. The win: bring
what you already have — tools, retrievers, servers, domains — without rewriting
it, and get Vincio's permissioned, audited, budgeted runtime around all of it.
Runs fully offline on the deterministic mock provider.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _shared import example_provider, json_responder

from vincio import ContextApp, available_packs, load_pack
from vincio.a2a import RemoteA2AAgent, connect_a2a_in_process
from vincio.interop import (
    add_langchain_tool,
    from_dspy_module,
    from_haystack_retriever,
    from_llamaindex_reader,
)
from vincio.mcp import MCPServer, build_app_server, connect_in_process
from vincio.providers import openai_compatible
from vincio.retrieval import build_embedder
from vincio.storage import VECTOR_BACKENDS, build_vector_index


def weather_server() -> MCPServer:
    """A tiny in-process MCP server standing in for any stdio/HTTP server — the
    four callbacks are exactly what a real server provides."""
    async def call_tool(name, args):
        if name == "get_weather":
            return {"text": f"{args['city']}: 22C, sunny"}
        return {"_mcp_error": True, "text": f"unknown tool {name}"}

    async def read_resource(uri):  # must be a coroutine — the client awaits it
        return {"uri": uri, "mimeType": "text/plain", "text": "Forecasts refresh every 3h."}

    return MCPServer(
        name="weather",
        list_tools=lambda: [{"name": "get_weather", "description": "Current weather for a city.",
                             "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}},
                                             "required": ["city"]}}],
        call_tool=call_tool,
        list_resources=lambda: [{"uri": "weather://policy", "name": "policy", "mimeType": "text/plain"}],
        read_resource=read_resource,
    )


async def section_mcp(app: ContextApp) -> None:
    # CLIENT: an MCP server's tools register through the SAME permissioned,
    # sandboxed, audited runtime as native tools; its resources become evidence
    # carrying `origin: mcp:<server>` provenance, so they can be cited.
    app.add_mcp_server("weather", server=weather_server(), resources=True)
    client = app.mcp_clients["weather"]

    # SERVER: expose THIS app over MCP — one ContextApp is both consumer and
    # provider, so its native tools become MCP tools for downstream consumers.
    app.add_tool(lambda text: text[:40], name="summarize", description="Summarize text to 40 chars.")
    consumer = connect_in_process(build_app_server(app))  # == app.serve_mcp()
    served = await consumer.list_tools()
    print("1. MCP: consumed tools", [t for t in app.enabled_tools if t.startswith("weather.")],
          "| provenance", app.pending_evidence[0].metadata["origin"],
          "| exposed", [t.name for t in served])
    print("   get_weather(Paris):", await client.call_tool("get_weather", {"city": "Paris"}))


async def section_a2a(app: ContextApp) -> None:
    # Expose a multi-role crew as an A2A agent in one call. The Agent Card is the
    # protocol's capability advertisement (served at /.well-known/agent.json).
    crew = app.crew(members=[{"name": "researcher", "goal": "gather the numbers", "keywords": ["find"]},
                             {"name": "writer", "goal": "draft the recommendation"}])
    server = app.serve_a2a(crew, name="research_crew", description="Researches and writes briefs.")
    task = await connect_a2a_in_process(server).send("Explain the Q3 refund trend")

    # Plug a REMOTE A2A agent into a local flow as a bounded, traced delegate — it
    # runs under a budget and termination guard, not as an open-ended call.
    delegate = RemoteA2AAgent(connect_a2a_in_process(app.serve_a2a(name="pricing_agent")), name="pricing")
    state = await delegate.run("What is the standard refund window?")
    print("2. A2A: card skills", [s["name"] for s in server.agent_card()["skills"]],
          f"| remote task {task.status.state} | delegate {state.termination_reason}")


SKILL_MD = """---
name: pdf-invoice
description: Extract totals and line items from PDF invoices. Use for invoice/PDF tasks.
keywords: [pdf, invoice, extract, total]
license: Apache-2.0
---

# Extracting PDF invoices

1. Locate the invoice header (vendor, date, invoice number).
2. Read the line-item table; sum the amounts.
3. Reconcile the sum against the stated total.
"""


def write_skill(root: Path) -> Path:
    skill_dir = root / "pdf-invoice"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (skill_dir / "scripts" / "checksum.py").write_text("print('rows: 3, total: 120.00')\n")
    return skill_dir


def section_skills(app: ContextApp, skill_dir: Path) -> None:
    # register_scripts=True exposes bundled scripts as sandboxed tools. Progressive
    # disclosure keeps a skill LIBRARY cheap: a relevant task discloses the full
    # body; an off-topic one costs only the one-line index entry, so unused skills
    # never bloat the context.
    app.add_skill(str(skill_dir), register_scripts=True)
    relevant = app.skill_library.evidence_for("extract the total from this pdf invoice")
    off_topic = app.skill_library.evidence_for("what is the capital of France")
    print("3. skills:", [s.name for s in app.skill_library.skills],
          "| script tool present", "pdf-invoice.checksum" in app.enabled_tools)
    print("   discloses relevant", [e.metadata["kind"] for e in relevant],
          "vs off-topic", [e.metadata["kind"] for e in off_topic])


class WeatherTool:
    """A LangChain-style tool — a real BaseTool works identically (duck-typed)."""
    name = "lc_get_weather"
    description = "Look up the current weather for a city."
    args = {"city": {"type": "string"}}

    def invoke(self, payload: dict) -> dict:
        return {"city": payload["city"], "temp_c": 22, "sky": "sunny"}


class TinyReader:
    """A LlamaIndex-style reader exposing .load_data()."""
    def load_data(self):
        class _Node:
            def __init__(self, text, metadata):
                self._text, self.metadata = text, metadata

            def get_content(self):
                return self._text
        return [_Node("Refunds are processed within 5 business days.", {"file_path": "kb.md"})]


class HaystackRetriever:
    """A Haystack-style retriever exposing .run(query) -> {'documents': [...]}."""
    def run(self, query):
        class _Doc:
            def __init__(self, content, meta, score):
                self.content, self.meta, self.score = content, meta, score
        return {"documents": [_Doc("refunds in 30 days", {"source": "kb"}, 0.92)]}


class CompiledDSPyProgram:
    """A compiled DSPy program: a signature + a callable returning a Prediction."""
    class _Field:
        def __init__(self, desc):
            self.json_schema_extra = {"desc": desc}

    class _Sig:
        instructions = "Answer the question from context."
        input_fields = {"question": None}
        output_fields = {"answer": None}

    def __init__(self):
        self._Sig.input_fields = {"question": self._Field("the user question")}
        self._Sig.output_fields = {"answer": self._Field("the grounded answer")}
        self.signature = self._Sig()

    class _Prediction:
        def __init__(self, data):
            self._data = data

        def toDict(self):
            return self._data

    def __call__(self, **kwargs):
        return self._Prediction({"answer": f"Refund window for {kwargs['question']}: 30 days"})


async def section_interop(app: ContextApp) -> None:
    # Each framework's native asset lands in Vincio's runtime WITHOUT a rewrite —
    # the adapters are duck-typed against the framework's real interface, so a tool,
    # a reader, a retriever and a compiled program all become first-class here.
    add_langchain_tool(app, WeatherTool())                        # tool -> registered
    docs = from_llamaindex_reader(TinyReader())                   # reader -> Documents
    app.add_source("kb", documents=docs, retrieval="hybrid")
    hits = await from_haystack_retriever(HaystackRetriever()).search("refund window")  # retriever -> source
    dspy_tool = from_dspy_module(CompiledDSPyProgram())           # program -> tool
    print("4. interop: langchain tool landed", "lc_get_weather" in app.enabled_tools,
          f"| llamaindex {len(docs)} doc | haystack {hits[0].chunk.text!r} | "
          f"dspy {dspy_tool['handler'](question='Pro plan')}")


def section_packs() -> None:
    # A vertical pack bundles prompt + schema + policies + deterministic rails +
    # domain metrics + scoped memory + residency posture + a golden eval set — a
    # regulated domain configured in one use_pack() call.
    print("5. packs:", available_packs())
    for name in ["healthcare", "kyc", "customer_support", "code_review"]:
        pack = load_pack(name)
        print(f"   {name:16s} schema={pack.output_schema_name:16s} golden={len(pack.eval_cases)}")

    assessment = {"risk_rating": "high", "sanctions_hit": True, "pep": False, "sar_recommended": True,
                  "rationale": "Screening returned a confirmed OFAC match against the beneficial owner."}
    kyc = ContextApp(name="kyc_desk", provider=example_provider(json_responder(assessment))[0],
                     model="mock-1").use_pack("kyc")
    # Standalone demo: relax grounding (production attaches case files, keeps it on).
    kyc.set_policy("answer_only_from_sources", False).set_policy("require_citations", False)
    out = kyc.run("Screen this customer against sanctions and adverse media.").output
    out = out.model_dump() if hasattr(out, "model_dump") else out
    print(f"   applied kyc: risk={out['risk_rating']} SAR={out['sar_recommended']} metrics={kyc.evaluators}")


def section_breadth() -> None:
    # One interface, broad reach: any OpenAI-compatible gateway via a named preset
    # (same call shape for groq/together/fireworks/...), and eleven vector stores
    # behind build_vector_index() with 'memory' as the offline default.
    embedder = build_embedder("local")
    print("6. breadth: groq endpoint", openai_compatible("groq", api_key="demo").base_url,
          f"| local embedder dim {embedder.dim} | {len(VECTOR_BACKENDS)} vector backends")
    print("   built index:", build_vector_index("memory", embedder).name)


async def main() -> None:
    app = ContextApp(name="interop_demo",
                     provider=example_provider(json_responder({"answer": "It is sunny."}))[0], model="mock-1")
    await section_mcp(app)
    await section_a2a(app)
    with tempfile.TemporaryDirectory() as tmp:
        section_skills(app, write_skill(Path(tmp)))
    await section_interop(app)
    section_packs()
    section_breadth()
    print("\nMCP <-> A2A <-> Skills <-> frameworks <-> packs <-> providers — one runtime.")


if __name__ == "__main__":
    asyncio.run(main())
