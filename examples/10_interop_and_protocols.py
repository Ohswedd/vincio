"""Interop & protocols — one interface to the whole ecosystem.

Vincio is a hub: it speaks the open agent protocols (MCP both ways, A2A),
loads portable Anthropic-style Agent Skills, adapts assets from the major
Python AI frameworks (LangChain / LlamaIndex / Haystack / DSPy), pulls from
first-party data connectors, ships full-stack vertical packs, and reaches any
provider or vector store behind a single API. The win: bring what you already
have — tools, retrievers, servers, domains — without rewriting it, and get
Vincio's permissioned, audited, budgeted runtime around all of it for free.

Runs fully offline on the deterministic mock provider. No API keys, no network.
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


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------- #
# 1) MCP — consume a server AND expose your app over the same protocol.
# --------------------------------------------------------------------------- #
def weather_server() -> MCPServer:
    """A tiny in-process MCP server standing in for any stdio/HTTP server.

    The four callbacks are exactly what a real server provides: tool discovery,
    tool invocation, resource discovery, and resource reads.
    """

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
            return {"text": f"{args['city']}: 22C, sunny"}
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


async def section_mcp(app: ContextApp) -> None:
    banner("1. MCP — client and server")

    # CLIENT side: an MCP server's tools register through the SAME permissioned,
    # sandboxed, audited runtime as native tools; its resources become evidence
    # carrying `origin: mcp:<server>` provenance (so they can be cited).
    app.add_mcp_server("weather", server=weather_server(), resources=True)
    print("registered MCP tools  :", [t for t in app.enabled_tools if t.startswith("weather.")])
    print("MCP resource provenance:", app.pending_evidence[0].metadata["origin"])

    # The connected client is callable directly (and from inside the agent loop).
    client = app.mcp_clients["weather"]
    print("get_weather(Paris)    :", await client.call_tool("get_weather", {"city": "Paris"}))

    # SERVER side: expose THIS app over MCP. One ContextApp is both consumer and
    # provider — its native tools become MCP tools for downstream consumers.
    def summarize(text: str) -> str:
        """Summarize text to 40 chars."""
        return text[:40] + ("..." if len(text) > 40 else "")

    app.add_tool(summarize, description="Summarize text to 40 chars.")
    server = build_app_server(app)  # equivalently: app.serve_mcp()
    consumer = connect_in_process(server)
    served = await consumer.list_tools()
    print("app exposed over MCP  :", [t.name for t in served])
    print("call summarize        :", await consumer.call_tool("summarize", {"text": "x" * 60}))


# --------------------------------------------------------------------------- #
# 2) A2A — delegate across agents with bounds + tracing the raw SDK won't give.
# --------------------------------------------------------------------------- #
async def section_a2a(app: ContextApp) -> None:
    banner("2. A2A — agent-to-agent delegation")

    # Expose a multi-role crew as an A2A agent in one call. The Agent Card is the
    # protocol's capability advertisement (served at /.well-known/agent.json).
    crew = app.crew(
        members=[
            {"name": "researcher", "goal": "gather the numbers", "keywords": ["find", "data"]},
            {"name": "writer", "goal": "draft the recommendation"},
        ]
    )
    server = app.serve_a2a(crew, name="research_crew", description="Researches and writes briefs.")
    card = server.agent_card()
    print("agent card            :", card["name"], "| skills:", [s["name"] for s in card["skills"]])

    # Reach it over the protocol; the task runs the bounded crew end to end.
    client = connect_a2a_in_process(server)
    task = await client.send("Explain the Q3 refund trend")
    print("remote task           :", task.status.state, "|", str(task.status.message.text)[:42])

    # Plug a REMOTE A2A agent into a local crew as a bounded, traced delegate —
    # it runs under a budget and termination guard, not as an open-ended call.
    delegate = RemoteA2AAgent(
        connect_a2a_in_process(app.serve_a2a(name="pricing_agent")), name="pricing"
    )
    state = await delegate.run("What is the standard refund window?")
    print("delegate termination  :", state.termination_reason)
    print("delegate answer       :", str(state.final_answer)[:42])


# --------------------------------------------------------------------------- #
# 3) Agent Skills — portable SKILL.md with progressive disclosure.
# --------------------------------------------------------------------------- #
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
    """Lay out an Anthropic-style skill folder: SKILL.md + a bundled script."""
    skill_dir = root / "pdf-invoice"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (skill_dir / "scripts" / "checksum.py").write_text("print('rows: 3, total: 120.00')\n")
    return skill_dir


def section_skills(app: ContextApp, skill_dir: Path) -> None:
    banner("3. Agent Skills — SKILL.md progressive disclosure")

    # register_scripts=True exposes bundled scripts as sandboxed tools.
    app.add_skill(str(skill_dir), register_scripts=True)
    print("loaded skills         :", [s.name for s in app.skill_library.skills])
    print("bundled script tool   :", "pdf-invoice.checksum" in app.enabled_tools)

    # Progressive disclosure: a relevant task discloses the full body; an
    # off-topic one costs only the one-line index entry. This is what keeps a
    # library of skills cheap — unused skills do not bloat the context.
    relevant = app.skill_library.evidence_for("extract the total from this pdf invoice")
    off_topic = app.skill_library.evidence_for("what is the capital of France")
    print("relevant discloses    :", [e.metadata["kind"] for e in relevant])
    print("off-topic discloses   :", [e.metadata["kind"] for e in off_topic])


# --------------------------------------------------------------------------- #
# 4) Framework interop — LangChain / LlamaIndex / Haystack / DSPy adapters.
# --------------------------------------------------------------------------- #
class WeatherTool:
    """A LangChain-style tool. A real ``langchain.tools.BaseTool`` works identically
    because the adapter is duck-typed against name/description/invoke."""

    name = "lc_get_weather"
    description = "Look up the current weather for a city."
    args = {"city": {"type": "string"}}

    def invoke(self, payload: dict) -> dict:
        return {"city": payload["city"], "temp_c": 22, "sky": "sunny"}


class TinyReader:
    """A LlamaIndex-style reader exposing ``.load_data()`` (as real readers do)."""

    def load_data(self):
        class _Node:
            def __init__(self, text, metadata):
                self._text, self.metadata = text, metadata

            def get_content(self):
                return self._text

        return [_Node("Refunds are processed within 5 business days.", {"file_path": "kb.md"})]


class HaystackRetriever:
    """A Haystack-style retriever exposing ``.run(query)`` -> {"documents": [...]}"""

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
    banner("4. Framework interop — import assets, no rewrite")

    # LangChain tool -> registered + enabled in Vincio's runtime.
    add_langchain_tool(app, WeatherTool())
    print("langchain tool landed :", "lc_get_weather" in app.enabled_tools)

    # LlamaIndex reader -> Vincio Documents -> a retrievable source.
    docs = from_llamaindex_reader(TinyReader())
    app.add_source("kb", documents=docs, retrieval="hybrid")
    print("llamaindex docs       :", len(docs), "doc(s) into source 'kb'")

    # Haystack retriever -> a Vincio retrieval source you can .search() async.
    hits = await from_haystack_retriever(HaystackRetriever()).search("refund window")
    print("haystack retriever    :", hits[0].chunk.text, f"(score {hits[0].score})")

    # Compiled DSPy program -> a Vincio tool (handler + schema from the signature).
    dspy_tool = from_dspy_module(CompiledDSPyProgram())
    print("dspy module as tool   :", dspy_tool["handler"](question="Pro plan"))


# --------------------------------------------------------------------------- #
# 5) Vertical packs — a regulated domain configured in one line.
# --------------------------------------------------------------------------- #
def section_packs() -> None:
    banner("5. Vertical packs — full-stack domain starting points")

    # A vertical pack bundles prompt + schema + policies + deterministic rails +
    # domain metrics + scoped memory + data-residency posture + a golden eval set.
    print("available packs       :", available_packs())
    for name in ["healthcare", "kyc", "customer_support", "code_review"]:
        pack = load_pack(name)
        print(
            f"  - {name:16s} schema={pack.output_schema_name:16s} "
            f"rails={[r['name'] for r in pack.rails]} golden={len(pack.eval_cases)}"
        )

    # Apply one with use_pack(...) and run it. The KYC assessment below is what
    # the deterministic provider returns offline.
    assessment = {
        "risk_rating": "high",
        "sanctions_hit": True,
        "pep": False,
        "sar_recommended": True,
        "rationale": "Screening returned a confirmed OFAC match against the beneficial owner.",
    }
    provider, model = example_provider(json_responder(assessment))
    kyc = ContextApp(name="kyc_desk", provider=provider, model=model).use_pack("kyc")
    # Standalone demo: relax source-grounding (in production attach case files and
    # keep grounding + citations on).
    kyc.set_policy("answer_only_from_sources", False).set_policy("require_citations", False)
    print("kyc metrics wired     :", kyc.evaluators)

    result = kyc.run("Screen this customer against sanctions and adverse media.")
    out = result.output
    out = out.model_dump() if hasattr(out, "model_dump") else out
    print("kyc risk_rating       :", out["risk_rating"], "| SAR advised:", out["sar_recommended"])
    print("golden eval ids       :", [c.id for c in load_pack("kyc").eval_cases])


# --------------------------------------------------------------------------- #
# 6) Provider + vector-store breadth behind one interface.
# --------------------------------------------------------------------------- #
def section_breadth() -> None:
    banner("6. Provider + vector-store breadth — one interface")

    # Any OpenAI-compatible gateway via a named preset — no network here, just the
    # resolved endpoint. The same call shape works for groq/together/fireworks/etc.
    print("groq endpoint         :", openai_compatible("groq", api_key="demo").base_url)

    # A local embedder (deterministic, offline) reports its vector dimension.
    embedder = build_embedder("local")
    print("local embedder dim    :", embedder.dim)

    # Vincio reaches eleven vector stores behind build_vector_index(); 'memory'
    # is the offline default and needs no external service.
    print("vector backends       :", VECTOR_BACKENDS)
    index = build_vector_index("memory", embedder)
    print("built index           :", index.name)


async def main() -> None:
    # One app threads MCP, A2A, skills, and framework interop through the same
    # permissioned/audited runtime. The mock provider keeps everything offline.
    provider, model = example_provider(json_responder({"answer": "It is sunny."}))
    app = ContextApp(name="interop_demo", provider=provider, model=model)

    await section_mcp(app)
    await section_a2a(app)

    with tempfile.TemporaryDirectory() as tmp:
        section_skills(app, write_skill(Path(tmp)))

    await section_interop(app)
    section_packs()
    section_breadth()

    print("\nDone. MCP <-> A2A <-> Skills <-> frameworks <-> packs <-> providers, one runtime.")


if __name__ == "__main__":
    asyncio.run(main())
