"""Multimodal, embeddings & retrieval breadth (1.5).

Keeping retrieval best-in-field as the embedding and ingestion frontier moves —
every new embedder, store, and parser sits behind an interface that already
exists, so breadth costs no new concepts:

  1. Matryoshka (MRL) embeddings — shrink the output dimension with `dimensions=`,
     trading storage/latency for a little recall, on any embedder.
  2. Query vs. document input-type hints, plumbed through the vector index.
  3. Contextual (Voyage context-3) and multimodal (Cohere v4 / Voyage) embedders.
  4. New vector stores — Weaviate, Milvus, Elasticsearch/OpenSearch, Vespa —
     behind the one `build_vector_index` factory.
  5. Layout-aware PDF extraction (reading order, tables, figures).
  6. Voice / realtime (optional module): a session whose tool calls run through
     the permissioned runtime.

Runs fully offline: hosted embedders/stores are exercised with an injected mock
transport / fake client, so no network or API keys are needed.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from vincio.core.types import Chunk, ImageRef
from vincio.documents import LayoutBlock, LayoutFigure, PageLayout, assemble_layout
from vincio.realtime import (
    InProcessRealtimeBackend,
    RealtimeEvent,
    RealtimeSession,
    RealtimeToolCall,
)
from vincio.retrieval import (
    LocalHashEmbedder,
    MatryoshkaEmbedder,
    MultimodalInput,
    VectorIndex,
    VoyageContextualEmbedder,
    VoyageMultimodalEmbedder,
    build_embedder,
)
from vincio.storage import build_vector_index


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _chunks() -> list[Chunk]:
    facts = [
        "Pro plan refunds are available within 30 days of purchase.",
        "The subscription renews automatically unless cancelled 60 days ahead.",
        "All data is encrypted at rest with AES-256 and in transit with TLS 1.3.",
    ]
    return [Chunk(document_id="kb", text=text, index=i) for i, text in enumerate(facts)]


def _mock(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def mrl_demo() -> None:
    _section("1. Matryoshka embeddings — recall vs. dimension")
    chunks = _chunks()
    for dimensions in (256, 128, 64, 32):
        index = VectorIndex(MatryoshkaEmbedder(LocalHashEmbedder(dim=256), dimensions))
        await index.add(chunks)
        hits = await index.search("How is data encrypted at rest and in transit?", top_k=1)
        bytes_per_vector = dimensions * 4
        print(f"  dim={dimensions:>3}  {bytes_per_vector:>4} B/vec  top hit: {hits[0].chunk.text[:38]!r}")
    # `build_embedder(..., dimensions=)` wires MRL for hosted embedders too.
    shrunk = build_embedder("local", dimensions=64)
    print(f"  build_embedder('local', dimensions=64) -> dim {shrunk.dim}")


async def input_type_demo() -> None:
    _section("2. Query vs. document input-type hints")
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body.get("task"))  # Jina maps input_type -> task
        n = len(body["input"])
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(i), 1.0]} for i in range(n)]})

    embedder = build_embedder("jina", api_key="demo", client=_mock(handler))
    index = VectorIndex(embedder)
    await index.add(_chunks())  # documents
    await index.search("refunds", top_k=1)  # query
    print(f"  encodings requested: {sorted(t for t in seen if t)}")


async def contextual_and_multimodal_demo() -> None:
    _section("3. Contextual & multimodal embedders (offline mock transport)")

    def contextual_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        groups = body["inputs"]
        data = [
            {"index": gi, "data": [{"index": ci, "embedding": [float(gi), float(ci)]} for ci in range(len(g))]}
            for gi, g in enumerate(groups)
        ]
        return httpx.Response(200, json={"data": data})

    contextual = VoyageContextualEmbedder(api_key="demo", client=_mock(contextual_handler))
    vectors = await contextual.embed(["chunk A", "chunk B"], input_type="document")
    print(f"  voyage-context-3: {len(vectors)} context-aware chunk vectors")

    def multimodal_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        kinds = [part["type"] for part in body["inputs"][0]["content"]]
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5, 0.5]}], "_kinds": kinds})

    multimodal = VoyageMultimodalEmbedder(api_key="demo", client=_mock(multimodal_handler))
    item = MultimodalInput(text="a revenue bar chart", image=ImageRef(url="https://example.com/chart.png"))
    [vec] = await multimodal.embed_multimodal([item])
    print(f"  voyage-multimodal-3: text+image -> one {len(vec)}-d vector in a shared space")


async def vector_store_demo() -> None:
    _section("4. New vector stores — one factory, helpful missing-dep errors")
    # Every new backend is reachable through build_vector_index(...). Here we
    # inject a tiny fake Elasticsearch client so the round trip runs offline.
    class _FakeES:
        def __init__(self) -> None:
            self.docs: dict[str, dict] = {}

            engine = self

            class _Indices:
                def exists(self, index):
                    return bool(engine.docs) or getattr(engine, "_created", False)

                def create(self, index, mappings=None, body=None):
                    engine._created = True

            self.indices = _Indices()

        def index(self, index, id, document, refresh=False):
            self.docs[id] = document

        def count(self, index):
            return {"count": len(self.docs)}

        def delete(self, index, id, refresh=False):
            self.docs.pop(id, None)

        def search(self, index, knn=None, size=10, body=None):
            q = knn["query_vector"]

            def cos(v):
                dot = sum(a * b for a, b in zip(v, q, strict=False))
                return dot / ((sum(a * a for a in v) ** 0.5 or 1) * (sum(b * b for b in q) ** 0.5 or 1))

            ranked = sorted(self.docs.values(), key=lambda d: cos(d["vector"]), reverse=True)
            return {"hits": {"hits": [{"_source": {"json": d["json"]}, "_score": cos(d["vector"])} for d in ranked[:knn["k"]]]}}

    index = build_vector_index("elasticsearch", LocalHashEmbedder(), client=_FakeES())
    await index.add(_chunks())
    hits = await index.search("data encryption", top_k=1)
    print(f"  elasticsearch round trip: {len(index)} chunks; top hit {hits[0].chunk.text[:34]!r}")
    print("  also available: weaviate, milvus, opensearch, vespa (each its own extra)")


def layout_demo() -> None:
    _section("5. Layout-aware extraction — reading order, tables, figures")
    # A two-column page: layout recovers reading order (left column, then right).
    page = PageLayout(
        page_number=1,
        width=200,
        height=300,
        blocks=[
            LayoutBlock(text="Left column first paragraph.", x0=10, top=10, x1=90, bottom=20),
            LayoutBlock(text="Right column first paragraph.", x0=110, top=12, x1=190, bottom=22),
            LayoutBlock(text="Left column second paragraph.", x0=10, top=30, x1=90, bottom=40),
            LayoutBlock(text="Right column second paragraph.", x0=110, top=32, x1=190, bottom=42),
        ],
        figures=[LayoutFigure(page=1, x0=5, top=50, x1=80, bottom=120, caption="Figure 1")],
    )
    doc = assemble_layout([page], title="report")
    print(f"  reading order: {doc.text.replace(chr(10), ' | ')}")
    print(f"  extractor={doc.metadata['extractor']}  figures={doc.metadata['figure_count']}")


async def realtime_demo() -> None:
    _section("6. Voice / realtime (optional module) — tool calls via the runtime")

    def script(text, config):
        return [
            RealtimeEvent(
                type="tool_call",
                tool_call=RealtimeToolCall(call_id="c1", name="get_weather", arguments={"city": "Paris"}),
            )
        ]

    async def dispatch(name: str, arguments: dict) -> dict:
        return {"city": arguments["city"], "temp_c": 14}

    session = RealtimeSession(InProcessRealtimeBackend(script=script), tool_dispatcher=dispatch)
    async with session:
        await session.send_text("What's the weather in Paris?")
        await session.commit()
        async for event in session.events():
            if event.type == "tool_result":
                print(f"  tool '{event.tool_call.name}' dispatched -> {event.data['result']}")
            if event.type == "response.done":
                break


async def main() -> None:
    await mrl_demo()
    await input_type_demo()
    await contextual_and_multimodal_demo()
    await vector_store_demo()
    layout_demo()
    await realtime_demo()
    print("\nEvery new embedder, store, and parser feeds the same compiler — chunked,")
    print("scored, budgeted, and cited exactly like a local file. Nothing downstream changes.")


if __name__ == "__main__":
    asyncio.run(main())
