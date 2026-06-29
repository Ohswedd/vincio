"""Streaming and out-of-core bulk processing for datasets larger than memory.

The data plane already fits a huge table into the window as a profile plus a
representative sample. This program walks the big-data rung that processes a
dataset far larger than memory in bounded passes, fully offline on the
deterministic mock:

  * `RowStream` — a lazy, re-iterable, schema-bearing handle over a row source
    (records, a generator factory, or a CSV / JSON-Lines file read line by line),
    iterated in bounded `chunks` and profiled / fitted / sampled in single passes
    whose footprint is invariant to the row count;
  * `stream_aggregate` — a deterministic, bounded-memory group-by whose working
    set tracks the number of groups, not the number of rows, so a source far
    larger than memory aggregates inside a fixed footprint;
  * `encode_stream` — the compact, lossless encoder applied header-once,
    row-by-row, optionally gzip-compressed, so a dataset larger than memory is
    rendered in one bounded pass;
  * the context compiler's streaming candidate pre-filter — a 10k+ evidence pool
    bounded to a cap by a cheap relevance proxy *before* full scoring, so the
    expensive stages never see more than the cap;
  * `app.map_stream` — an analytical transform run over a stream at scale through
    the existing `BatchRunner` (half-cost provider batch APIs, bounded
    concurrency).

Everything below is opt-in and additive; none of it touches a database or a network.
"""

from __future__ import annotations

import asyncio
import gzip

from _shared import example_provider

from vincio import ContextApp
from vincio.context.compiler import ContextCompiler, ContextCompilerOptions
from vincio.core.tokens import count_tokens
from vincio.core.types import Budget, EvidenceItem, Message, ModelRequest, Objective, UserInput
from vincio.data import ColumnSchema, DataType, RowStream, encode_stream, stream_aggregate

SCHEMA = [
    ColumnSchema(name="id", dtype=DataType.INT),
    ColumnSchema(name="region", dtype=DataType.STR),
    ColumnSchema(name="amount", dtype=DataType.FLOAT, unit="USD"),
]
REGIONS = ["NA", "EU", "APAC", "LATAM"]


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def transactions(n: int):
    """A re-iterable row factory standing in for a source larger than memory."""

    def factory():
        for i in range(n):
            yield [i, REGIONS[i % 4], float(i % 1000)]

    return factory


# ---------------------------------------------------------------------------
# 1. A lazy, re-iterable RowStream processed in bounded chunks.
# ---------------------------------------------------------------------------
def section_rowstream() -> None:
    banner("1. RowStream — bounded chunks, never materialized")

    stream = RowStream.from_rows(transactions(1_000_000), SCHEMA, name="txns")
    first = next(iter(stream.chunks(50_000)))
    print(f"   schema: {[(c.name, c.dtype.value) for c in stream.columns]}")
    print(f"   one chunk holds {first.row_count:,} rows resident (of 1,000,000)")

    # A single bounded pass profiles the whole stream; the footprint tracks
    # columns, not rows.
    profile = stream.profile()
    amount = profile.column("amount")
    print(f"   profiled {profile.row_count:,} rows: amount in [{amount.min}, {amount.max}], mean {amount.mean}")


# ---------------------------------------------------------------------------
# 2. A bounded-memory streaming group-by (out-of-core analytics).
# ---------------------------------------------------------------------------
def section_aggregate() -> None:
    banner("2. stream_aggregate — group-by in a fixed footprint")

    stream = RowStream.from_rows(transactions(1_000_000), SCHEMA, name="txns")
    agg = stream_aggregate(stream, group_by="region", measures={"amount": ["sum", "mean", "min", "max"]})
    print(f"   {agg.summary()}")
    for record in agg.result.records():
        print(
            f"     {record['region']}: count={record['count']:,} "
            f"sum={record['amount_sum']:,.0f} mean={record['amount_mean']:.1f}"
        )
    print("   the working set held one accumulator per group, never the rows")


# ---------------------------------------------------------------------------
# 3. The streaming compact encoder, with optional compression.
# ---------------------------------------------------------------------------
def section_encode() -> None:
    banner("3. encode_stream — compact, lossless, header-once (optionally gzip)")

    stream = RowStream.from_rows(transactions(10_000), SCHEMA, name="txns")
    plain = encode_stream(stream)
    compressed = encode_stream(stream, compress=True)
    print(f"   encoded {count_tokens(plain.decode()):,} tokens in {len(plain):,} bytes")
    print(f"   gzip-compressed to {len(compressed):,} bytes ({len(compressed) / len(plain):.1%} of plain)")
    print(f"   round-trips losslessly: {gzip.decompress(compressed) == plain}")


# ---------------------------------------------------------------------------
# 4. The streaming candidate pre-filter bounds a huge evidence pool.
# ---------------------------------------------------------------------------
async def section_prefilter() -> None:
    banner("4. context compiler — bound a 10k+ candidate pool before scoring")

    evidence = [
        EvidenceItem(
            id=f"e{i}",
            text=(
                f"quarterly revenue grew in region {i}"
                if i % 1000 == 0
                else f"unrelated filler note {i} about the weather"
            ),
            source_id=f"s{i}",
        )
        for i in range(10_000)
    ]
    compiler = ContextCompiler(ContextCompilerOptions(max_candidates=200))
    compiled = await compiler.compile(
        objective=Objective(text="analyze quarterly revenue growth", task_type="data_analysis"),
        user_input=UserInput(text="what was the quarterly revenue growth?"),
        evidence=evidence,
        budget=Budget(max_input_tokens=4000),
    )
    survived = sum(1 for i in range(0, 10_000, 1000) if f"e{i}" in {e.id for e in compiled.ir.evidence})
    print(f"   pre-filtered {compiler.prefilter_drops:,} of 10,000 candidates before full scoring")
    print(f"   final packet holds {len(compiled.ir.evidence)} evidence items")
    print(f"   all {survived}/10 relevant items survived the cut")


# ---------------------------------------------------------------------------
# 5. An analytical transform over a stream, at scale on the BatchRunner.
# ---------------------------------------------------------------------------
async def section_map() -> None:
    banner("5. app.map_stream — a per-chunk transform through the BatchRunner")

    provider, model = example_provider()
    app = ContextApp(name="streaming", provider=provider, model=model)
    stream = app.stream_dataset(transactions(1_000), schema=SCHEMA, name="txns")

    def build(chunk, index: int) -> ModelRequest:
        return ModelRequest(
            model=model,
            messages=[Message(role="user", content="Summarize this batch:\n" + chunk.encode())],
        )

    result = await app.map_stream(stream, build, chunk_rows=250)
    print(f"   dispatched {result.chunk_count} chunks through the batch runner")
    print(f"   succeeded: {len(result.succeeded)}; failed: {len(result.failed)}")
    print(f"   batch-discounted cost: ${result.cost_usd:.6f}")


async def main() -> None:
    section_rowstream()
    section_aggregate()
    section_encode()
    await section_prefilter()
    await section_map()
    print("\nDone — a dataset larger than memory was processed in bounded passes, inside a fixed footprint.")


if __name__ == "__main__":
    asyncio.run(main())
