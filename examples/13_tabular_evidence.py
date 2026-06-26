"""Tabular evidence and the compact data encoder.

Structured data is first-class, schema-bearing, columnar evidence in Vincio —
never flattened to prose or dumped as json.dumps. This program walks the 4.1 data
plane, fully offline on the deterministic mock:

  * a typed `Dataset` with a `DataSchema` over column-major cells;
  * the deterministic `DataEncoder` that renders it header-once, lossless and
    far more token-efficient than json.dumps or a Markdown table;
  * columnar-accurate token accounting (the count of the tokens the model gets);
  * `TableEvidence` / `app.table_evidence`, so a dataset is scored, budgeted, and
    cited by the context compiler exactly like text, image, and table evidence.

Everything below is opt-in and additive; none of it touches a database or a network.
"""

from __future__ import annotations

import asyncio
import json

from _shared import example_provider

from vincio import ContextApp, DataEncoder, TableEvidence
from vincio.core.tokens import count_tokens
from vincio.data import ColumnSchema, Dataset, DataType


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


SALES = [
    {"region": "NA", "revenue": 1200.50, "units": 5, "active": True},
    {"region": "EU", "revenue": 980.00, "units": None, "active": False},
    {"region": "APAC", "revenue": 1500.25, "units": 8, "active": True},
]


# ---------------------------------------------------------------------------
# 1. A typed, columnar dataset.
# ---------------------------------------------------------------------------
def section_dataset() -> None:
    banner("1. A typed, columnar Dataset")

    ds = Dataset.from_records(SALES, name="sales")
    print(f"   columns : {ds.column_names}")
    print(f"   dtypes  : {ds.dtypes}")
    print(f"   rows    : {ds.row_count}  ·  nullable units: {ds.columns[2].nullable}")

    # Or declare the schema explicitly, attaching a unit to a column.
    typed = Dataset.from_rows(
        [[r["region"], r["revenue"]] for r in SALES],
        [ColumnSchema(name="region"),
         ColumnSchema(name="revenue", dtype=DataType.FLOAT, unit="USD")],
        name="sales",
    )
    print(f"   typed schema: {[(c.name, c.dtype.value, c.unit) for c in typed.columns]}")


# ---------------------------------------------------------------------------
# 2. The compact encoding — header-once, lossless, token-cheap.
# ---------------------------------------------------------------------------
def section_encoding() -> None:
    banner("2. The compact, lossless encoding")

    ds = Dataset.from_records(SALES, name="sales")
    encoded = ds.encode()
    print("   encoding:")
    for line in encoded.splitlines():
        print(f"     {line}")

    # Lossless: decode reconstructs the columns, types, and cells exactly.
    assert Dataset.from_encoding(encoded).rows() == ds.rows()
    print("   round-trips losslessly: decode(encode(ds)).rows() == ds.rows()")

    # Far fewer tokens than the json.dumps fallback it replaces.
    json_tokens = count_tokens(json.dumps(SALES, indent=2))
    encoded_tokens = count_tokens(encoded)
    print(
        f"   tokens: json.dumps={json_tokens}  encoded={encoded_tokens}  "
        f"({1 - encoded_tokens / json_tokens:.0%} fewer)"
    )


# ---------------------------------------------------------------------------
# 3. Columnar-accurate token accounting and the DataEncoder surface.
# ---------------------------------------------------------------------------
def section_encoder() -> None:
    banner("3. The DataEncoder: token cost and arbitrary values")

    ds = Dataset.from_records(SALES, name="sales")
    encoder = DataEncoder()
    # The token cost is the exact count of the tokens the model receives.
    assert encoder.token_cost(ds) == count_tokens(encoder.encode(ds))
    print(f"   columnar token cost: {ds.token_cost()} (== count_tokens(encoding))")

    # The encoder also replaces json.dumps for arbitrary JSON-like values.
    blob = {"name": "Acme", "tags": ["a", "b"], "rows": [{"p": 1}, {"p": 2}]}
    print("   encode_value (compact, not json.dumps):")
    for line in encoder.encode_value(blob).splitlines():
        print(f"     {line}")


# ---------------------------------------------------------------------------
# 4. A dataset as first-class context evidence.
# ---------------------------------------------------------------------------
async def section_evidence() -> None:
    banner("4. TableEvidence in the context compiler")

    provider, model = example_provider(default_responder=lambda _r: "APAC, at $1500.25.")
    app = ContextApp(name="data", provider=provider, model=model)

    # app.table_evidence accepts records, rows (+columns), a Dataset, or TableData.
    evidence = app.table_evidence(SALES, name="sales", caption="Quarterly sales by region")
    assert isinstance(evidence, TableEvidence)

    item = evidence.to_evidence_item()
    print(f"   modality   : {item.modality}")
    print(f"   token_cost : {item.token_cost} (columnar-accurate)")
    print(f"   prompt text: {item.scorable_text.splitlines()[0]} ...")

    # Attach it to a run by adding it to the pending evidence the runtime reads;
    # the context compiler scores, budgets, orders, and cites it like any other
    # evidence (it also accepts a Dataset or a TableEvidence directly).
    app.pending_evidence.append(item)
    result = await app.arun("Which region had the most revenue?")
    print(f"   answer     : {result.output}")


async def main() -> None:
    section_dataset()
    section_encoding()
    section_encoder()
    await section_evidence()
    print("\nDone — structured data reached the model schema-once and token-cheap, offline.")


if __name__ == "__main__":
    asyncio.run(main())
