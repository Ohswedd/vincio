"""Tabular evidence & the compact data encoder.

Covers the data plane end-to-end, offline and deterministic: the encoding kernel
(round-trip losslessness, columnar token accounting), the typed ``Dataset`` and
its schema, the ``DataEncoder``, ``TableEvidence`` projection into the context
compiler, the ``app.table_evidence`` entry point, and the parser-path replacement
of the pipe-join / ``json.dumps`` rendering.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest

from vincio.core import tabular
from vincio.core.errors import DataError
from vincio.core.tokens import count_tokens
from vincio.core.types import Budget, EvidenceItem, Objective, UserInput
from vincio.data import (
    ColumnSchema,
    DataEncoder,
    Dataset,
    DataType,
    TableEvidence,
)
from vincio.documents.parsers import TableData, parse_csv_table, structure_data
from vincio.output.parsers import extract_citations

# --------------------------------------------------------------------------- #
# Encoding kernel: round-trip losslessness
# --------------------------------------------------------------------------- #


def test_encode_decode_round_trips_mixed_types():
    cols = ["id", "region", "revenue", "units", "active"]
    rows = [
        [1, "NA", 1200.5, 5, True],
        [2, "EU", 980.0, None, False],
        [3, "APAC", 1500.25, 8, True],
    ]
    enc = tabular.encode_table(cols, rows, types=["int", "str", "float", "int", "bool"], name="sales")
    decoded = tabular.decode_table(enc)
    assert decoded.name == "sales"
    assert decoded.columns == cols
    assert decoded.typed_rows() == rows


def test_round_trip_preserves_nulls_empty_strings_and_specials():
    # null vs empty-string, embedded delimiter, quotes, and a newline.
    cols = ["k", "v"]
    rows = [
        ["a", None],           # null
        ["b", ""],             # empty string (distinct from null)
        ["c", "x, y"],         # embedded delimiter
        ["d", 'say "hi"'],     # embedded quotes
        ["e", "line1\nline2"], # embedded newline
    ]
    enc = tabular.encode_table(cols, rows, types=["str", "str"])
    decoded = tabular.decode_table(enc)
    assert decoded.typed_rows() == rows
    # the null and the empty string render differently
    body = enc.splitlines()
    assert body[1].endswith(",")        # null -> trailing empty field
    assert body[2].endswith(',""')      # empty string -> quoted empty


def test_round_trip_preserves_temporal_and_leading_zero_strings():
    cols = ["d", "ts", "zip"]
    rows = [[date(2021, 1, 2), datetime(2021, 1, 2, 3, 4, 5), "01234"]]
    enc = tabular.encode_table(cols, rows, types=["date", "datetime", "str"])
    assert tabular.decode_table(enc).typed_rows() == rows
    assert "01234" in enc  # the leading zero is preserved exactly


def test_exemplar_preamble_is_decode_safe():
    cols, rows = ["id", "name"], [[1, "a"], [2, "b"]]
    enc = tabular.encode_table(
        cols, rows, types=["int", "str"], options=tabular.EncodeOptions(exemplars=2)
    )
    assert enc.splitlines()[0].startswith("#")  # the description line
    assert tabular.decode_table(enc).typed_rows() == rows


def test_count_notation_is_not_a_citation_marker():
    # The row count lives inside the braces (``#3``) so it is never mistaken for
    # an ``[E1]``-style citation marker by a downstream consumer.
    enc = tabular.encode_table(["a"], [[1], [2], [3]], types=["int"], name="t")
    assert extract_citations(enc) == []
    assert "#3" in enc.splitlines()[0]


def test_encode_options_toggle_name_count_types():
    cols, rows = ["a", "b"], [[1, 2]]
    enc = tabular.encode_table(
        cols, rows, types=["int", "int"], name="t",
        options=tabular.EncodeOptions(include_name=False, include_count=False, include_types=False),
    )
    assert enc.splitlines()[0] == "{a,b}"
    assert tabular.decode_table(enc).columns == cols


def test_infer_dtype_keeps_bool_distinct_from_int():
    assert tabular.infer_dtype([True, False]) == "bool"
    assert tabular.infer_dtype([1, 2, 3]) == "int"
    assert tabular.infer_dtype([1, 2.5]) == "float"
    assert tabular.infer_dtype([None, None]) == "null"
    assert tabular.infer_dtype(["x", 1]) == "str"


def test_round_trip_survives_hostile_names_and_headers():
    # A table name beginning with '#' must not be mistaken for a description line.
    enc = tabular.encode_table(["a"], [[1]], types=["int"], name="#tag")
    assert tabular.decode_table(enc).name == "#tag"
    # Newlines in the name and a column name keep each row on one physical line.
    enc = tabular.encode_table(["c\nx"], [[1]], types=["int"], name="my\ntable")
    decoded = tabular.decode_table(enc)
    assert decoded.name == "my\ntable" and decoded.columns == ["c\nx"]
    assert len(enc.splitlines()) == 2  # header + one row, not split by the newlines


def test_encode_table_rejects_over_wide_rows():
    # A row with more cells than declared columns cannot be encoded losslessly.
    with pytest.raises(DataError):
        tabular.encode_table(["a", "b"], [[1, 2, 3]])


def test_decode_tolerates_trailing_comma_in_header():
    # A hand-authored header with a trailing comma is not a phantom column.
    decoded = tabular.decode_table("t{#1,a:int,b:str,}\n1,x")
    assert decoded.columns == ["a", "b"]


def test_encode_value_replaces_json_dumps_compactly():
    obj = {"name": "Acme", "tags": ["a", "b"], "rows": [{"p": 1}, {"p": 2}]}
    rendered = tabular.encode_value(obj)
    # every leaf is present, and it is smaller than json.dumps(indent=2)
    assert "Acme" in rendered and "p:int" in rendered
    assert count_tokens(rendered) < count_tokens(json.dumps(obj, indent=2))


# --------------------------------------------------------------------------- #
# Token efficiency (the SLO property)
# --------------------------------------------------------------------------- #


def _orders(n: int) -> list[dict[str, object]]:
    return [
        {
            "order_id": f"ORD-{i:05d}",
            "customer": f"Customer_{i}",
            "amount_usd": 100.0 + i,
            "status": ["pending", "shipped", "delivered"][i % 3],
        }
        for i in range(n)
    ]


def test_encoding_is_far_more_compact_than_json_dumps():
    records = _orders(25)
    encoded = Dataset.from_records(records, name="orders").encode()
    json_tokens = count_tokens(json.dumps(records, indent=2))
    encoded_tokens = count_tokens(encoded)
    # at least a 30% token reduction versus the json.dumps fallback
    assert encoded_tokens < json_tokens
    assert 1 - encoded_tokens / json_tokens >= 0.30


def test_columnar_token_cost_matches_encoding():
    ds = Dataset.from_records(_orders(10), name="orders")
    assert ds.token_cost() == count_tokens(ds.encode())


# --------------------------------------------------------------------------- #
# Dataset: construction & access
# --------------------------------------------------------------------------- #


def test_from_records_infers_schema_and_nullability():
    ds = Dataset.from_records(
        [{"a": 1, "b": "x"}, {"a": 2}], name="t"  # second row omits b -> nullable
    )
    assert ds.column_names == ["a", "b"]
    assert ds.dtypes == ["int", "str"]
    assert ds.columns[1].nullable is True
    assert ds.row_count == 2 and ds.width == 2


def test_from_rows_and_columns_round_trip_through_encoding():
    ds = Dataset.from_rows(
        [[1, "NA"], [2, "EU"]],
        [ColumnSchema(name="id", dtype=DataType.INT), ColumnSchema(name="region")],
        name="sales",
    )
    again = Dataset.from_encoding(ds.encode())
    assert again.rows() == ds.rows() and again.column_names == ds.column_names
    by_col = Dataset.from_columns({"id": [1, 2], "region": ["NA", "EU"]})
    assert by_col.column("region") == ["NA", "EU"]


def test_from_rows_rejects_width_mismatch():
    with pytest.raises(DataError):
        Dataset.from_rows([[1, 2, 3]], ["a", "b"])


def test_records_head_and_exemplars():
    ds = Dataset.from_records(_orders(5), name="orders")
    assert ds.records()[0]["order_id"] == "ORD-00000"
    head = ds.head(2)
    assert head.rows() == ds.rows()[:2]      # the first n rows, in order
    assert head.dtypes == ds.dtypes          # schema preserved
    assert ds.data_schema.names == ds.column_names


def test_exemplars_skip_nulls_and_dedupe():
    ds = Dataset.from_columns({"x": [None, "x", "x", "y"]})
    assert ds.exemplars(2) == {"x": ["x", "y"]}   # distinct, non-null
    assert ds.exemplars(1) == {"x": ["x"]}


def test_column_unit_round_trips_through_encoding():
    ds = Dataset.from_rows(
        [[1200.5], [980.0]],
        [ColumnSchema(name="revenue", dtype=DataType.FLOAT, unit="USD")],
        name="sales",
    )
    enc = ds.encode()
    assert "revenue:float USD" in enc
    assert Dataset.from_encoding(enc).units == ["USD"]
    # a unit containing a space is quoted in the header and still round-trips
    spaced = Dataset.from_rows(
        [[5]], [ColumnSchema(name="t", dtype=DataType.INT, unit="US Dollar")], name="s"
    )
    assert Dataset.from_encoding(spaced.encode()).units == ["US Dollar"]


def test_from_table_data_bridge_is_lossless_on_strings():
    table = parse_csv_table("id,city,zip\n1,NA,01234\n2,EU,00420", title="locs")
    ds = Dataset.from_table_data(table)
    # numeric id is typed; the zip's leading zeros are preserved (kept as text)
    assert ds.dtypes[0] == "int"
    assert ds.column("zip") == ["01234", "00420"]


# --------------------------------------------------------------------------- #
# DataEncoder
# --------------------------------------------------------------------------- #


def test_data_encoder_encode_decode_and_token_cost():
    enc = DataEncoder()
    ds = Dataset.from_records(_orders(6), name="orders")
    text = enc.encode(ds)
    assert enc.decode(text).rows() == ds.rows()
    assert enc.token_cost(ds) == count_tokens(text)
    # also accepts records (unnamed -> bare brace header) and a legacy TableData
    assert enc.encode([{"a": 1}, {"a": 2}]).splitlines()[0] == "{#2,a:int}"
    table = parse_csv_table("a,b\n1,2", title="t")
    assert "a:int" in enc.encode(table)


def test_data_encoder_value_and_options():
    compact = DataEncoder(include_types=False).encode([{"a": 1, "b": 2}])
    assert ":int" not in compact
    assert "Acme" in DataEncoder().encode_value({"name": "Acme"})


# --------------------------------------------------------------------------- #
# TableEvidence -> EvidenceItem
# --------------------------------------------------------------------------- #


def test_table_evidence_projects_to_evidence_item():
    ds = Dataset.from_records(_orders(4), name="orders")
    te = ds.to_evidence(source_id="orders", caption="Recent orders")
    item = te.to_evidence_item()
    assert item.modality == "table"
    enc = ds.encode()
    assert item.scorable_text == enc
    assert item.text == enc
    assert item.token_cost == count_tokens(enc)
    assert item.estimated_token_cost() == count_tokens(enc)
    assert item.table["encoding"] == enc
    assert item.table["columns"] == ds.column_names
    assert item.table["caption"] == "Recent orders"
    assert item.metadata["row_count"] == 4


def test_table_evidence_from_records_and_citation():
    te = TableEvidence.from_records(_orders(3), name="orders", source_id="orders", citation="D1")
    item = te.to_evidence_item()
    assert item.id == "D1"
    assert item.modality == "table"


# --------------------------------------------------------------------------- #
# EvidenceItem token accounting: encoding-aware, with legacy fallback
# --------------------------------------------------------------------------- #


def test_evidence_item_prefers_encoding_for_cost_and_text():
    ds = Dataset.from_records(_orders(8), name="orders")
    enc = ds.encode()
    item = EvidenceItem(source_id="D", modality="table", table={"encoding": enc, "columns": ds.column_names})
    assert item.estimated_token_cost() == count_tokens(enc)
    assert item.scorable_text == enc


def test_evidence_item_legacy_per_cell_fallback_unchanged():
    # A raw table dict without an encoding keeps the documented per-cell heuristic.
    item = EvidenceItem(
        source_id="D", modality="table", table={"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
    )
    assert item.estimated_token_cost() == 18  # 3 * (4 cells + 2 cols)


# --------------------------------------------------------------------------- #
# Parser path: compact, lossless rendering replaces pipe-join / json.dumps
# --------------------------------------------------------------------------- #


def test_table_data_to_text_is_compact_and_lossless():
    table = parse_csv_table("plan,price\nPro,99\nBasic,19", title="pricing")
    text = table.to_text()
    assert "|" not in text                 # no pipe-join
    assert text.splitlines()[0].startswith("pricing{#2")
    # the cell values survive
    assert "Pro" in text and "99" in text
    # round-trips back to the same cells
    assert tabular.decode_table(text).columns == ["plan", "price"]


def test_table_data_to_text_keeps_footnotes():
    table = TableData(columns=["a"], rows=[["1"]], footnotes=["see appendix"])
    assert "# note: see appendix" in table.to_text()


def test_empty_table_renders_empty():
    # An empty table stays empty, so an empty document/chunk is not given a
    # content-less '{#0}' body.
    assert TableData().to_text() == ""
    assert TableData(title="X").to_text() == ""


def test_structure_data_uses_compact_encoding_not_json():
    text, _sections, _tables = structure_data({"meta": {"k": 1, "deep": {"x": True}}})
    assert "{" not in text or "  \"k\"" not in text  # not pretty-printed json
    assert "k: 1" in text and "x: true" in text


# --------------------------------------------------------------------------- #
# Compiler & app integration
# --------------------------------------------------------------------------- #


async def test_compiler_accepts_dataset_and_table_evidence():
    from vincio.context.compiler import ContextCompiler

    ds = Dataset.from_records(
        [{"region": "NA", "revenue": 1200.5}, {"region": "EU", "revenue": 980.0}], name="sales"
    )
    compiler = ContextCompiler()
    for ev in (ds, ds.to_evidence(source_id="sales")):
        compiled = await compiler.compile(
            objective=Objective(text="revenue by region"),
            user_input=UserInput(text="revenue region"),
            evidence=[ev],
            budget=Budget(max_input_tokens=4000),
        )
        kept = [e for e in compiled.ir.evidence if e.modality == "table"]
        assert kept and kept[0].token_cost > 0
        assert "sales{#2" in kept[0].scorable_text


def test_app_table_evidence_accepts_records_rows_and_table_data(offline_config):
    from vincio.core.app import ContextApp
    from vincio.providers import MockProvider

    app = ContextApp(name="svc", provider=MockProvider(default_text="ok"), config=offline_config)

    from_records = app.table_evidence(_orders(3), name="orders")
    assert isinstance(from_records, TableEvidence)
    assert from_records.to_evidence_item().modality == "table"

    from_rows = app.table_evidence([[1, "NA"], [2, "EU"]], columns=["id", "region"], name="r")
    assert from_rows.dataset.column_names == ["id", "region"]

    table = parse_csv_table("a,b\n1,2", title="t")
    assert isinstance(app.table_evidence(table), TableEvidence)

    with pytest.raises(DataError):
        app.table_evidence([[1, 2]])  # rows without columns/schema
    with pytest.raises(DataError):
        app.table_evidence(42)  # unrecognized input type
