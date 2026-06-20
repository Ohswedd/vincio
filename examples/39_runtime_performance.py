"""Runtime performance & efficiency: the compile hot path.

Shows the spine's performance features, all offline:

- the warm candidate arena reuses the query-independent preparation when the
  candidate set is unchanged, and the compiled-prompt render program reuses the
  rendered stable prefix across tasks that share a spec;
- streaming-first compilation emits the stable prefix before any candidate is
  scored;
- the per-app resident-memory budget holds the packet footprint under a ceiling
  and surfaces it on every run;
- speculative retrieval prefetch (config knob) warms the query embedding before
  retrieval.
"""

import asyncio
import tempfile
import time
from pathlib import Path

from _shared import citing_responder, example_provider, write_sample_docs

from vincio import ContextApp, EvidenceItem, Objective, PromptSpec, UserInput, VincioConfig

docs_dir = write_sample_docs(Path(tempfile.mkdtemp()) / "docs")
provider, model = example_provider(
    citing_responder("Refunds on the Pro plan are available within 30 days. [{ref}]")
)

# A resident-memory ceiling and speculative prefetch are opt-in performance knobs.
config = VincioConfig()
config.performance.memory_budget_mb = 8
config.performance.speculative_prefetch = True

app = ContextApp(name="perf_demo", provider=provider, model=model, config=config)
app.add_source("docs", path=str(docs_dir), retrieval="hybrid")
app.set_policy("answer_only_from_sources", True)

EVIDENCE = [
    EvidenceItem(id="e1", source_id="D1", text="Pro plan refunds are available within 30 days."),
    EvidenceItem(id="e2", source_id="D2", text="Basic plan refunds are available within 14 days."),
    EvidenceItem(id="e3", source_id="D3", text="The subscription renews automatically each year."),
]


async def cache_and_arena() -> None:
    print("— content-addressed cache + warm candidate arena —")
    question = "What is the refund window for the Pro plan?"
    started = time.perf_counter()
    first = await app.arun(question)
    cold_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    await app.arun(question)  # identical inputs → full compile-cache hit
    warm_ms = (time.perf_counter() - started) * 1000
    print(f"  app run: cold {cold_ms:.1f} ms, warm {warm_ms:.1f} ms")
    print(f"  context-compile cache hits: {app.context_compiler.cache_hits}")
    print(f"  resident footprint:         {first.memory_bytes} bytes (held ≤ 8 MB)")

    # The arena reuses the prepared candidate set when only the query changes.
    for query in ("What is the refund window?", "How long to request a Pro refund?"):
        await app.context_compiler.compile(
            objective=Objective("refunds"), user_input=UserInput(text=query), evidence=EVIDENCE
        )
    print(f"  candidate-arena reuses (new query, same evidence): {app.context_compiler.arena_hits}")


def render_program() -> None:
    print("\n— compiled-prompt render program —")
    spec = PromptSpec(
        name="support", role="support agent", objective="Answer from policy",
        rules=["Use only the provided documents", "Cite evidence IDs"],
    )
    for task in ("Refund window?", "Late-payment interest?", "Renewal notice period?"):
        app.prompt_compiler.compile(spec, user_task=task)
    print(f"  stable-prefix reuses across tasks: {app.prompt_compiler.program_hits}")


async def streaming_first_compile() -> None:
    print("\n— streaming-first compilation (prefix before scoring) —")
    async for event in app.context_compiler.compile_streaming(
        objective=Objective("answer policy questions"),
        user_input=UserInput(text="What is the refund window?"),
        evidence=EVIDENCE,
    ):
        if event.type == "prefix":
            print(f"  prefix ready ({len(event.text)} chars) — emitted before scoring")
        elif event.type == "evidence":
            print(f"  evidence selected: {len(event.evidence)} items")
        elif event.type == "done":
            print(f"  done: packet {event.result.packet.id}, {event.result.token_count} tokens")


async def main() -> None:
    await cache_and_arena()
    render_program()
    await streaming_first_compile()


if __name__ == "__main__":
    asyncio.run(main())
