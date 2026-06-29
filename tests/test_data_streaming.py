"""Streaming and out-of-core bulk processing (the data plane's big-data rung).

Covers the lazy, schema-bearing :class:`RowStream` and its chunked file readers,
the bounded-memory streaming group-by, the streaming compact encoder (and its
gzip compression), the BatchRunner-backed map at scale, the context compiler's
streaming candidate pre-filter, and the thin ``app.*`` surface.
"""

from __future__ import annotations

import gzip

import pytest

from vincio import ContextApp, VincioConfig
from vincio.context.compiler import ContextCompiler, ContextCompilerOptions
from vincio.core import tabular
from vincio.core.errors import DataError, StreamError
from vincio.core.types import Budget, EvidenceItem, Message, ModelRequest, Objective, UserInput
from vincio.data import (
    ColumnSchema,
    Dataset,
    DataType,
    RowStream,
    StreamAggregation,
    encode_stream,
    stream_aggregate,
    stream_map,
)
from vincio.providers import MockProvider

SCHEMA = [
    ColumnSchema(name="id", dtype=DataType.INT),
    ColumnSchema(name="region", dtype=DataType.STR),
    ColumnSchema(name="amount", dtype=DataType.FLOAT),
]
REGIONS = ["NA", "EU", "APAC", "LATAM"]


def gen_factory(n: int):
    def factory():
        for i in range(n):
            yield [i, REGIONS[i % 4], float(i % 100)]

    return factory


# --------------------------------------------------------------------------- #
# RowStream construction & access
# --------------------------------------------------------------------------- #


def test_from_records_infers_schema_and_is_reiterable():
    rs = RowStream.from_records(
        [{"r": "NA", "v": 10}, {"r": "EU", "v": 5}, {"r": "NA", "v": 20}], name="sales"
    )
    assert rs.column_names == ["r", "v"]
    assert [c.dtype for c in rs.columns] == [DataType.STR, DataType.INT]
    # re-iterable: two independent passes
    assert [row[0] for row in rs.rows()] == ["NA", "EU", "NA"]
    assert [row[1] for row in rs.rows()] == [10, 5, 20]


def test_from_rows_with_factory_is_reiterable():
    rs = RowStream.from_rows(gen_factory(5), SCHEMA, name="txns")
    assert sum(1 for _ in rs.rows()) == 5
    assert sum(1 for _ in rs.rows()) == 5  # second pass works


def test_from_dataset_roundtrips():
    ds = Dataset.from_records([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], name="d")
    rs = RowStream.from_dataset(ds)
    assert rs.name == "d"
    assert [list(r) for r in rs.rows()] == [[1, "x"], [2, "y"]]


def test_empty_schema_refused():
    with pytest.raises(StreamError):
        RowStream([], [])


def test_one_shot_iterator_allows_one_pass_then_refuses():
    def one_shot():
        yield ["NA", 1]
        yield ["EU", 2]

    rs = RowStream.from_rows(one_shot(), ["r", "v"])
    assert [list(r) for r in rs.rows()] == [["NA", 1], ["EU", 2]]
    with pytest.raises(StreamError):
        list(rs.rows())


def test_chunks_are_bounded_and_cover_all_rows():
    rs = RowStream.from_rows(gen_factory(10), SCHEMA)
    chunks = list(rs.chunks(3))
    assert [c.row_count for c in chunks] == [3, 3, 3, 1]
    assert all(c.column_names == ["id", "region", "amount"] for c in chunks)
    total = sum(c.row_count for c in chunks)
    assert total == 10


def test_chunk_size_must_be_positive():
    rs = RowStream.from_rows(gen_factory(3), SCHEMA)
    with pytest.raises(StreamError):
        list(rs.chunks(0))


def test_materialize_loads_all_rows():
    rs = RowStream.from_rows(gen_factory(4), SCHEMA, name="t")
    ds = rs.materialize()
    assert isinstance(ds, Dataset)
    assert ds.row_count == 4
    assert ds.name == "t"


# --------------------------------------------------------------------------- #
# CSV / JSON-Lines readers
# --------------------------------------------------------------------------- #


def test_from_csv_infers_types_losslessly():
    lines = ["id,region,amount", "1,NA,1200.5", "2,EU,980.0", "007,APAC,1500.25"]
    rs = RowStream.from_csv(lines, name="sales")
    assert [(c.name, c.dtype) for c in rs.columns] == [
        ("id", DataType.STR),  # "007" does not round-trip as int -> stays text (lossless)
        ("region", DataType.STR),
        ("amount", DataType.FLOAT),
    ]
    rows = [list(r) for r in rs.rows()]
    assert rows[0] == ["1", "NA", 1200.5]


def test_from_csv_numeric_inference_when_clean():
    lines = ["id,amount", "1,10.0", "2,20.5"]
    rs = RowStream.from_csv(lines)
    assert [c.dtype for c in rs.columns] == [DataType.INT, DataType.FLOAT]
    assert [list(r) for r in rs.rows()] == [[1, 10.0], [2, 20.5]]


def test_from_csv_quoted_fields_and_nulls():
    lines = ['a,b', '"x,y",1', ',2']  # embedded delimiter quoted; empty field is null
    rs = RowStream.from_csv(lines)
    rows = [list(r) for r in rs.rows()]
    assert rows == [["x,y", 1], [None, 2]]


def test_from_csv_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("id,region,amount\n1,NA,1200.5\n2,EU,980.0\n")
    rs = RowStream.open(str(path))
    assert rs.name == "data"
    rows = [list(r) for r in rs.rows()]
    assert rows == [[1, "NA", 1200.5], [2, "EU", 980.0]]
    # re-iterable from the file (reopened each pass)
    assert sum(1 for _ in rs.rows()) == 2


def test_from_csv_one_shot_line_iterator_is_materialized():
    # A one-shot iterator of lines is read twice (peek + rows); the reader
    # materializes it so both passes see the data.
    def lines():
        yield "a,b"
        yield "1,x"
        yield "2,y"

    rs = RowStream.from_csv(lines())
    assert [list(r) for r in rs.rows()] == [[1, "x"], [2, "y"]]
    assert [list(r) for r in rs.rows()] == [[1, "x"], [2, "y"]]  # re-iterable


def test_from_jsonl_objects():
    lines = ['{"a": 1, "b": "x"}', '{"a": 2, "b": "y"}']
    rs = RowStream.from_jsonl(lines)
    assert [c.dtype for c in rs.columns] == [DataType.INT, DataType.STR]
    assert [list(r) for r in rs.rows()] == [[1, "x"], [2, "y"]]


def test_from_jsonl_array_requires_schema():
    lines = ["[1, 2]", "[3, 4]"]
    with pytest.raises(StreamError):
        RowStream.from_jsonl(lines)
    rs = RowStream.from_jsonl(lines, schema=["a", "b"])
    assert [list(r) for r in rs.rows()] == [[1, 2], [3, 4]]


def test_open_unknown_format_refused(tmp_path):
    path = tmp_path / "data.bin"
    path.write_text("x")
    with pytest.raises(StreamError):
        RowStream.open(str(path))


# --------------------------------------------------------------------------- #
# Bounded-pass operators delegate to the streaming kernels
# --------------------------------------------------------------------------- #


def test_profile_over_stream():
    rs = RowStream.from_rows(gen_factory(1000), SCHEMA, name="t")
    profile = rs.profile()
    assert profile.row_count == 1000
    assert profile.column_count == 3
    amount = profile.column("amount")
    assert amount.min == 0.0
    assert amount.max == 99.0


def test_fit_over_stream_is_within_budget():
    rs = RowStream.from_rows(gen_factory(100_000), SCHEMA, name="t")
    fit = rs.fit(max_tokens=1500)
    assert fit.within_budget
    assert fit.token_cost <= 1500
    assert fit.original_row_count == 100_000


def test_sample_is_bounded_and_records_metadata():
    rs = RowStream.from_rows(gen_factory(10_000), SCHEMA, name="t")
    sample = rs.sample(50, seed=3)
    assert sample.row_count == 50
    assert sample.metadata["sample"]["method"] == "reservoir"
    # deterministic for a given seed
    assert rs.sample(50, seed=3).rows() == sample.rows()


# --------------------------------------------------------------------------- #
# Streaming aggregation (out-of-core group-by)
# --------------------------------------------------------------------------- #


def test_stream_aggregate_groups_and_measures():
    rows = [{"r": "NA", "v": 10.0}, {"r": "EU", "v": 5.0}, {"r": "NA", "v": 20.0}]
    agg = stream_aggregate(rows, group_by="r", measures={"v": ["sum", "mean", "min", "max"]})
    assert isinstance(agg, StreamAggregation)
    assert agg.rows_processed == 3
    assert agg.group_count == 2
    records = {rec["r"]: rec for rec in agg.result.records()}
    assert records["NA"]["v_sum"] == 30.0
    assert records["NA"]["v_mean"] == 15.0
    assert records["NA"]["v_min"] == 10.0
    assert records["NA"]["v_max"] == 20.0
    assert records["NA"]["count"] == 2
    assert records["EU"]["count"] == 1


def test_stream_aggregate_count_only_when_no_measures():
    agg = stream_aggregate([{"k": "a"}, {"k": "b"}, {"k": "a"}], group_by="k")
    assert agg.result.column_names == ["k", "count"]
    assert {r["k"]: r["count"] for r in agg.result.records()} == {"a": 2, "b": 1}


def test_stream_aggregate_multi_key():
    rows = [
        {"r": "NA", "y": 2024, "v": 1.0},
        {"r": "NA", "y": 2024, "v": 3.0},
        {"r": "NA", "y": 2025, "v": 5.0},
    ]
    agg = stream_aggregate(rows, group_by=["r", "y"], measures={"v": "sum"})
    assert agg.group_count == 2
    by = {(r["r"], r["y"]): r["v_sum"] for r in agg.result.records()}
    assert by[("NA", 2024)] == 4.0
    assert by[("NA", 2025)] == 5.0


def test_stream_aggregate_is_deterministic_in_key_order():
    rows = [{"k": k} for k in ["c", "a", "b", "a", "c"]]
    agg = stream_aggregate(rows, group_by="k")
    assert [r["k"] for r in agg.result.records()] == ["a", "b", "c"]


def test_stream_aggregate_unknown_column_refused():
    with pytest.raises(StreamError):
        stream_aggregate([{"k": 1}], group_by="missing")
    with pytest.raises(StreamError):
        stream_aggregate([{"k": 1}], group_by="k", measures={"nope": "sum"})


def test_stream_aggregate_unknown_aggregation_refused():
    with pytest.raises(StreamError):
        stream_aggregate([{"k": 1, "v": 2}], group_by="k", measures={"v": "median"})


def test_stream_aggregate_group_cap_refused():
    rows = [{"k": i} for i in range(100)]
    with pytest.raises(StreamError):
        stream_aggregate(rows, group_by="k", max_groups=10)


def test_stream_aggregate_non_numeric_measure_is_none():
    rows = [{"r": "NA", "label": "x"}, {"r": "NA", "label": "y"}]
    agg = stream_aggregate(rows, group_by="r", measures={"label": "sum"})
    assert agg.result.records()[0]["label_sum"] is None
    assert agg.result.records()[0]["count"] == 2


def test_aggregation_projects_to_evidence():
    agg = stream_aggregate([{"r": "NA", "v": 1.0}], group_by="r", measures={"v": "sum"})
    item = agg.to_evidence_item()
    assert item.modality == "table"
    assert "summary" in agg.summary() or "groups" in agg.summary()


# --------------------------------------------------------------------------- #
# Streaming compact encoding & compression
# --------------------------------------------------------------------------- #


def test_encode_stream_roundtrips_losslessly():
    rs = RowStream.from_records(
        [{"r": "NA", "v": 10.0}, {"r": "EU", "v": 5.0}, {"r": "NA", "v": 20.0}], name="sales"
    )
    encoded = encode_stream(rs)
    decoded = tabular.decode_table(encoded.decode())
    assert decoded.columns == ["r", "v"]
    assert decoded.typed_rows() == [["NA", 10.0], ["EU", 5.0], ["NA", 20.0]]


def test_encode_stream_preserves_null_vs_empty_string():
    rs = RowStream.from_rows([["", None], ["x", "y"]], ["a", "b"])
    decoded = tabular.decode_table(encode_stream(rs).decode())
    # empty string distinguished from null
    assert decoded.rows[0] == ["", None]


def test_encode_stream_gzip_roundtrips():
    rs = RowStream.from_rows(gen_factory(200), SCHEMA, name="t")
    plain = encode_stream(rs)
    gz = encode_stream(rs, compress=True)
    assert gzip.decompress(gz) == plain
    # a repetitive table compresses well
    assert len(gz) < len(plain)


def test_encode_stream_to_sink(tmp_path):
    rs = RowStream.from_rows(gen_factory(50), SCHEMA, name="t")
    path = tmp_path / "out.tbl"
    with open(path, "wb") as handle:
        ret = encode_stream(rs, sink=handle)
    assert ret == b""
    decoded = tabular.decode_table(path.read_text())
    assert len(decoded.rows) == 50


def test_encode_stream_empty():
    rs = RowStream.from_rows([], ["a", "b"])
    decoded = tabular.decode_table(encode_stream(rs).decode())
    assert decoded.columns == ["a", "b"]
    assert decoded.rows == []


def test_tabular_streaming_helpers_compose_to_encode_table():
    columns = ["a", "b"]
    rows = [[1, "x"], [2, "y"]]
    header = tabular.encode_header(columns, types=["int", "str"], name="t")
    body = [tabular.encode_row(r, 2) for r in rows]
    decoded = tabular.decode_table("\n".join([header, *body]))
    assert decoded.typed_rows() == rows


def test_encode_row_refuses_too_wide():
    with pytest.raises(DataError):
        tabular.encode_row([1, 2, 3], 2)


# --------------------------------------------------------------------------- #
# Analytical pipelines at scale on the BatchRunner
# --------------------------------------------------------------------------- #


async def test_stream_map_runs_chunks_through_batch_runner():
    rs = RowStream.from_rows(gen_factory(10), SCHEMA, name="t")

    def build(chunk: Dataset, index: int) -> ModelRequest:
        return ModelRequest(
            model="mock", messages=[Message(role="user", content="summarize:\n" + chunk.encode())]
        )

    result = await stream_map(rs, build, backend=MockProvider(), chunk_rows=4)
    assert result.chunk_count == 3  # 4 + 4 + 2
    assert len(result.succeeded) == 3
    assert not result.failed
    assert sorted(result.by_chunk()) == [0, 1, 2]


async def test_stream_map_requires_a_backend():
    rs = RowStream.from_rows(gen_factory(2), SCHEMA)
    with pytest.raises(StreamError):
        await stream_map(rs, lambda c, i: ModelRequest(model="m", messages=[]))


async def test_stream_map_empty_stream():
    rs = RowStream.from_rows([], ["a"])
    result = await stream_map(rs, lambda c, i: ModelRequest(model="m", messages=[]), backend=MockProvider())
    assert result.chunk_count == 0


# --------------------------------------------------------------------------- #
# Streaming candidate pre-filter (context compiler)
# --------------------------------------------------------------------------- #


def _big_evidence(n: int = 10_000) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for i in range(n):
        if i % 1000 == 0:
            items.append(EvidenceItem(id=f"e{i}", text=f"quarterly revenue grew in region {i}", source_id=f"s{i}"))
        else:
            items.append(EvidenceItem(id=f"e{i}", text=f"unrelated filler note {i} about the weather", source_id=f"s{i}"))
    return items


async def test_prefilter_bounds_pool_and_keeps_relevant():
    evidence = _big_evidence()
    compiler = ContextCompiler(ContextCompilerOptions(max_candidates=200))
    compiled = await compiler.compile(
        objective=Objective(text="analyze quarterly revenue growth", task_type="data_analysis"),
        user_input=UserInput(text="what was the quarterly revenue growth?"),
        evidence=evidence,
        budget=Budget(max_input_tokens=4000),
    )
    assert compiler.prefilter_drops > 0
    pre = [e for e in compiled.excluded_report if str(e.get("reason", "")).startswith("prefiltered")]
    assert len(pre) == compiler.prefilter_drops
    # every relevant item survives into the final packet
    final_ids = {e.id for e in compiled.ir.evidence}
    assert all(f"e{i}" in final_ids for i in range(0, 10_000, 1000))


async def test_prefilter_is_noop_under_cap():
    evidence = _big_evidence(50)
    compiler = ContextCompiler(ContextCompilerOptions(max_candidates=200))
    compiled = await compiler.compile(
        objective=Objective(text="revenue", task_type="data_analysis"),
        user_input=UserInput(text="revenue?"),
        evidence=evidence,
        budget=Budget(max_input_tokens=4000),
    )
    assert compiler.prefilter_drops == 0
    assert not [e for e in compiled.excluded_report if str(e.get("reason", "")).startswith("prefiltered")]


async def test_prefilter_off_by_default_matches_baseline():
    evidence = _big_evidence(500)
    obj = Objective(text="revenue growth", task_type="data_analysis")
    ui = UserInput(text="revenue growth?")
    budget = Budget(max_input_tokens=4000)
    base = await ContextCompiler(ContextCompilerOptions()).compile(
        objective=obj, user_input=ui, evidence=list(evidence), budget=budget
    )
    capped = await ContextCompiler(ContextCompilerOptions(max_candidates=400)).compile(
        objective=obj, user_input=ui, evidence=list(evidence), budget=budget
    )
    # 500 > cap 400, but the few relevant items dominate selection either way.
    base_ids = {e.id for e in base.ir.evidence}
    capped_ids = {e.id for e in capped.ir.evidence}
    assert base_ids == capped_ids


async def test_prefilter_drops_exact_duplicates():
    evidence = _big_evidence(5_000)
    for j in range(20):
        evidence.append(EvidenceItem(id=f"dup{j}", text="quarterly revenue grew in region 0", source_id="sdup"))
    compiler = ContextCompiler(ContextCompilerOptions(max_candidates=300))
    compiled = await compiler.compile(
        objective=Objective(text="revenue", task_type="data_analysis"),
        user_input=UserInput(text="revenue?"),
        evidence=evidence,
        budget=Budget(max_input_tokens=4000),
    )
    reasons = [e["reason"] for e in compiled.excluded_report if str(e.get("reason", "")).startswith("prefiltered")]
    assert "prefiltered_duplicate" in reasons


# --------------------------------------------------------------------------- #
# App surface
# --------------------------------------------------------------------------- #


@pytest.fixture()
def mock_app(tmp_path):
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    return ContextApp(name="stream_test", provider=MockProvider(), model="mock-1", config=config)


def test_app_stream_dataset_from_records(mock_app):
    stream = mock_app.stream_dataset([{"r": "NA", "v": 1.0}, {"r": "EU", "v": 2.0}], name="s")
    assert isinstance(stream, RowStream)
    assert stream.column_names == ["r", "v"]


def test_app_stream_dataset_from_file(mock_app, tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("a,b\n1,x\n2,y\n")
    stream = mock_app.stream_dataset(str(path))
    assert [list(r) for r in stream.rows()] == [[1, "x"], [2, "y"]]


def test_app_stream_dataset_from_rows_needs_columns(mock_app):
    with pytest.raises(DataError):
        mock_app.stream_dataset([[1, 2], [3, 4]])


def test_app_aggregate_stream(mock_app):
    agg = mock_app.aggregate_stream(
        [{"r": "NA", "v": 1.0}, {"r": "NA", "v": 3.0}, {"r": "EU", "v": 2.0}],
        group_by="r",
        measures={"v": ["sum", "mean"]},
    )
    by = {r["r"]: r for r in agg.result.records()}
    assert by["NA"]["v_sum"] == 4.0
    assert by["NA"]["v_mean"] == 2.0


async def test_app_map_stream_uses_app_provider(mock_app):
    stream = mock_app.stream_dataset([{"r": "NA", "v": 1.0}, {"r": "EU", "v": 2.0}], name="s")

    def build(chunk: Dataset, index: int) -> ModelRequest:
        return ModelRequest(model="mock", messages=[Message(role="user", content=chunk.encode())])

    result = await mock_app.map_stream(stream, build, chunk_rows=1)
    assert result.chunk_count == 2
    assert len(result.succeeded) == 2
