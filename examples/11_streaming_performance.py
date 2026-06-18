"""End-to-end streaming and the performance features.

Streams a grounded QA run (real token deltas through the full pipeline),
shows incremental partial-JSON output for a structured task, then
demonstrates the content-addressed compile caches and the zero-copy packet.
"""

import asyncio
import tempfile
import time
from pathlib import Path

from _shared import citing_responder, example_provider, write_sample_docs

from vincio import ContextApp, Objective, UserInput

docs_dir = write_sample_docs(Path(tempfile.mkdtemp()) / "docs")
provider, model = example_provider(
    citing_responder("The refund window for the Pro plan is 30 days with no fee. [{ref}]")
)

app = ContextApp(name="streaming_qa", provider=provider, model=model)
app.add_source("docs", path=str(docs_dir), chunking="adaptive", retrieval="hybrid")
app.set_policy("answer_only_from_sources", True)


async def stream_answer() -> None:
    print("— streaming a grounded answer —")
    async for event in app.astream("What is the refund window for the Pro plan?"):
        if event.type == "stage":
            print(f"  [{event.stage}] {event.data}")
        elif event.type == "text_delta":
            print(event.text, end="", flush=True)
        elif event.type == "done":
            result = event.result
            print(f"\n  status={result.status.value} citations={result.citations}")


async def stream_structured() -> None:
    print("\n— incremental partial-JSON output —")
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "confidence"],
    }
    structured_provider, structured_model = example_provider()
    structured = ContextApp(
        name="streaming_structured",
        provider=structured_provider,
        model=structured_model,
        output_schema=schema,
    )
    async for event in structured.astream("Answer: what is 30 days after purchase?"):
        if event.type == "partial_output":
            print(f"  partial (complete={event.output_complete}): {event.partial_output}")
        elif event.type == "done":
            print(f"  final: {event.result.output}")


async def show_compile_caching() -> None:
    print("\n— content-addressed compile caching —")
    question = "What is the refund window for the Pro plan?"
    started = time.perf_counter()
    await app.arun(question)
    cold_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    await app.arun(question)  # chunk/prompt/context compile stages all hit
    warm_ms = (time.perf_counter() - started) * 1000
    print(f"  first run: {cold_ms:.1f} ms, repeat run: {warm_ms:.1f} ms")
    print(f"  context compile cache hits: {app.context_compiler.cache_hits}")
    print(f"  prompt compile cache hits:  {app.prompt_compiler.cache_hits}")


async def show_zero_copy_packet() -> None:
    print("\n— zero-copy context packet —")
    compiled = await app.context_compiler.compile(
        objective=app.objective or Objective("answer policy questions"),
        user_input=UserInput(text="What is the refund window?"),
        evidence=(await app.retrieval.retrieve("refund window", top_k=4)).evidence,
    )
    packet = compiled.packet
    chunks = sum(1 for _ in packet.iter_json())
    print(f"  packet {packet.id}: ~{packet.approx_size_bytes()} bytes, streamed as {chunks} JSON chunks")


async def main() -> None:
    await stream_answer()
    await stream_structured()
    await show_compile_caching()
    await show_zero_copy_packet()


if __name__ == "__main__":
    asyncio.run(main())
