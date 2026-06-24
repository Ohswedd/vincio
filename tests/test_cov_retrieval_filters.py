"""Branch-coverage tests for vincio.retrieval.filters.

Targets the native-compiler operator branches (Pinecone/Weaviate/Milvus/
Elasticsearch/SQL), leaf-evaluation edge cases (nin / comparison on None /
TypeError-incomparable), builder validation, and flat_filter_fields. Every
assertion pins an exact compiled structure or predicate outcome.
"""

from __future__ import annotations

import pytest

from vincio.core.types import Chunk
from vincio.retrieval.filters import (
    FilterSpec,
    and_,
    contains,
    eq,
    exists,
    flat_filter_fields,
    in_,
    ne,
    not_,
    or_,
    range_,
)


def _chunk(**kw) -> Chunk:
    base = dict(document_id="d1", text="hello world")
    base.update(kw)
    return Chunk(**base)


# -- leaf evaluation edge cases (lines 265, 269, 276, 279-281) --------------


def test_matches_nin_op():
    c = _chunk(kind="table")
    spec = FilterSpec(field="kind", op="nin", value=["text", "code"])
    assert spec.matches(c) is True
    assert FilterSpec(field="kind", op="nin", value=["table"]).matches(c) is False
    # nin against a None/empty value list treats absence as "not in []".
    assert FilterSpec(field="kind", op="nin", value=None).matches(c) is True


def test_in_op_with_none_value_never_matches():
    c = _chunk(kind="table")
    assert FilterSpec(field="kind", op="in", value=None).matches(c) is False


def test_comparison_on_missing_field_is_false():
    # actual is None -> the `if actual is None: return False` guard (line 269).
    c = _chunk()
    assert FilterSpec(field="metadata.score", op="gte", value=1).matches(c) is False
    assert FilterSpec(field="metadata.score", op="lt", value=1).matches(c) is False


def test_comparison_typeerror_is_swallowed_to_false():
    # str vs int comparison raises TypeError -> caught -> False (lines 279-281).
    c = _chunk(metadata={"label": "high"})
    assert FilterSpec(field="metadata.label", op="gt", value=5).matches(c) is False
    assert FilterSpec(field="metadata.label", op="lte", value=5).matches(c) is False


def test_lt_and_lte_branches():
    c = _chunk(page=5)
    assert FilterSpec(field="page", op="lt", value=6).matches(c) is True
    assert FilterSpec(field="page", op="lt", value=5).matches(c) is False
    assert FilterSpec(field="page", op="lte", value=5).matches(c) is True
    assert FilterSpec(field="page", op="gt", value=4).matches(c) is True


def test_contains_on_string_field():
    c = _chunk(source_uri="s3://bucket/key")
    assert contains("source_uri", "bucket").matches(c) is True
    assert contains("source_uri", "nope").matches(c) is False
    # contains on a non-container (int) is False, not an error.
    assert FilterSpec(field="page", op="contains", value=1).matches(_chunk(page=1)) is False


# -- builder validation (lines 80, 172, 174) -------------------------------


def test_leaf_requires_op():
    with pytest.raises(ValueError, match="leaf FilterSpec requires an op"):
        FilterSpec(field="x")


def test_range_requires_a_bound():
    with pytest.raises(ValueError, match="range_ requires at least one bound"):
        range_("page")


def test_range_single_bound_collapses_to_leaf():
    spec = range_("page", gt=3)
    assert spec.is_leaf is True
    assert spec.op == "gt"
    assert spec.field == "page"


def test_range_multi_bound_is_and_composite():
    spec = range_("page", gte=2, lte=9)
    assert spec.is_leaf is False
    assert len(spec.must) == 2
    assert {b.op for b in spec.must} == {"gte", "lte"}


def test_range_all_four_open_bounds():
    # Exercises every bound branch (gte/gt/lte/lt) inside one composite.
    spec = range_("score", gte=1, gt=0, lte=10, lt=11)
    assert {b.op for b in spec.must} == {"gte", "gt", "lte", "lt"}
    c = _chunk(metadata={"score": 5})
    assert spec.matches(c) is True
    # Boundary: gt=5 excludes 5, lt=5 excludes 5.
    assert range_("score", gt=5).matches(c) is False
    assert range_("score", lt=5).matches(c) is False


# -- flat_filter_fields (lines 227-228, 233, 236) --------------------------


def test_flat_filter_fields_full():
    c = _chunk(
        tenant_id="t1",
        kind="table",
        page=7,
        index=3,
        source_uri="file://x",
        metadata={"author": "alice", "score": 0.9, "ok": True, "nested": {"drop": 1}},
    )
    flat = flat_filter_fields(c)
    assert flat["author"] == "alice"
    assert flat["score"] == 0.9
    assert flat["ok"] is True
    assert "nested" not in flat  # non-scalar metadata dropped (line 227 false branch)
    assert flat["document_id"] == "d1"
    assert flat["tenant_id"] == "t1"
    assert flat["kind"] == "table"
    assert flat["page"] == 7
    assert flat["index"] == 3
    assert flat["source_uri"] == "file://x"


def test_flat_filter_fields_drops_optional_none():
    c = _chunk(tenant_id=None, page=None, source_uri=None)
    flat = flat_filter_fields(c)
    assert flat["tenant_id"] == ""  # untagged tenant stored as empty string
    assert "page" not in flat  # page None -> omitted (line 233 false branch)
    assert "source_uri" not in flat  # source_uri None -> omitted (line 236)


# -- Pinecone / mongo-style compiler (line 357 + ops) ----------------------


def test_pinecone_all_leaf_ops():
    # Values are iterable (str/list) so the eager $in/$nin mapping does not blow up.
    assert eq("k", "v").to_pinecone() == {"k": {"$eq": "v"}}
    assert ne("k", "v").to_pinecone() == {"k": {"$ne": "v"}}
    assert in_("k", [1, 2]).to_pinecone() == {"k": {"$in": [1, 2]}}
    assert FilterSpec(field="k", op="nin", value=[3]).to_pinecone() == {"k": {"$nin": [3]}}
    assert FilterSpec(field="k", op="gt", value="v").to_pinecone() == {"k": {"$gt": "v"}}
    assert FilterSpec(field="k", op="gte", value="v").to_pinecone() == {"k": {"$gte": "v"}}
    assert FilterSpec(field="k", op="lt", value="v").to_pinecone() == {"k": {"$lt": "v"}}
    assert FilterSpec(field="k", op="lte", value="v").to_pinecone() == {"k": {"$lte": "v"}}
    assert contains("k", "x").to_pinecone() == {"k": {"$in": ["x"]}}
    assert exists("k").to_pinecone() == {"k": {"$exists": True}}


def test_pinecone_eager_in_mapping_rejects_scalar_value():
    # The mongo-style compiler builds the full op mapping eagerly, so even an
    # `eq` leaf evaluates `list(value or [])` for the unused `$in` entry -> a
    # non-iterable scalar value raises. This pins the current (brittle) contract.
    with pytest.raises(TypeError):
        eq("k", 1).to_pinecone()


def test_pinecone_strips_metadata_prefix():
    assert eq("metadata.author", "alice").to_pinecone() == {"author": {"$eq": "alice"}}


def test_pinecone_must_not_uses_nor():
    # A leaf-level $nor is produced by a composite whose ONLY clause is must_not.
    assert not_(eq("b", "2")).to_pinecone() == {"$and": [{"$nor": [{"b": {"$eq": "2"}}]}]}
    # Nested inside an and_, the not_ child stays its own $and/$nor sub-clause.
    spec = and_(eq("a", "1"), not_(eq("b", "2")))
    assert spec.to_pinecone() == {
        "$and": [{"a": {"$eq": "1"}}, {"$and": [{"$nor": [{"b": {"$eq": "2"}}]}]}]
    }


def test_pinecone_should_uses_or():
    assert or_(eq("a", "1"), eq("b", "2")).to_pinecone() == {
        "$or": [{"a": {"$eq": "1"}}, {"b": {"$eq": "2"}}]
    }


# -- Weaviate compiler (lines 390, 393, 408-414 value keys) ----------------


def test_weaviate_value_key_per_type():
    assert eq("k", True).to_weaviate()["valueBoolean"] is True
    assert eq("k", 3).to_weaviate()["valueInt"] == 3
    assert eq("k", 1.5).to_weaviate()["valueNumber"] == 1.5
    assert eq("k", "s").to_weaviate()["valueText"] == "s"


def test_weaviate_exists_is_isnull_false():
    assert exists("k").to_weaviate() == {
        "path": ["k"],
        "operator": "IsNull",
        "valueBoolean": False,
    }


def test_weaviate_in_uses_array_value():
    w = in_("kind", ["a", "b"]).to_weaviate()
    assert w == {
        "path": ["kind"],
        "operator": "ContainsAny",
        "valueTextArray": ["a", "b"],
    }


def test_weaviate_contains_wraps_value_in_list():
    w = contains("perm", "admin").to_weaviate()
    assert w == {"path": ["perm"], "operator": "ContainsAny", "valueText": ["admin"]}


def test_weaviate_unsupported_op_raises():
    with pytest.raises(ValueError, match="unsupported op for weaviate: nin"):
        FilterSpec(field="k", op="nin", value=[1]).to_weaviate()


def test_weaviate_single_operand_collapses():
    # An and_ with one child collapses to that child's dict (len(operands)==1).
    w = and_(eq("a", 1)).to_weaviate()
    assert w == {"path": ["a"], "operator": "Equal", "valueInt": 1}


def test_weaviate_not_operand_and_or_operators():
    w = and_(eq("a", 1), not_(eq("b", 2))).to_weaviate()
    assert w["operator"] == "And"
    assert {"operator": "Not", "operands": [eq("b", 2).to_weaviate()]} in w["operands"]
    assert or_(eq("a", 1), eq("b", 2)).to_weaviate()["operator"] == "Or"


# NOTE: every `op` is a typed Literal, so pydantic rejects unknown ops at
# construction time. The per-compiler `raise ValueError("unsupported op ...")`
# lines are therefore defensive/unreachable through the public API and are not
# tested here.


# -- Milvus compiler (lines 428, 430, 432, 435, 451, 453) ------------------


def test_milvus_in_nin_contains_exists():
    assert in_("kind", ["a", "b"]).to_milvus() == 'kind in ["a", "b"]'
    assert FilterSpec(field="kind", op="nin", value=["a"]).to_milvus() == 'kind not in ["a"]'
    assert contains("permissions", "admin").to_milvus() == 'array_contains(permissions, "admin")'
    assert exists("tenant_id").to_milvus() == "tenant_id != null"


def test_milvus_literal_bool_and_escaping():
    # `flag` is not a chunk field -> addressed via metadata["flag"].
    assert eq("flag", True).to_milvus() == 'metadata["flag"] == true'
    assert eq("flag", False).to_milvus() == 'metadata["flag"] == false'
    # embedded double-quote is escaped.
    assert eq("metadata.title", 'a"b').to_milvus() == 'metadata["title"] == "a\\"b"'


def test_milvus_composite_must_should_must_not():
    expr = and_(eq("page", 1), or_(eq("index", 2), eq("kind", 3)), not_(eq("page", 4))).to_milvus()
    assert "(page == 1)" in expr
    assert "((index == 2) or (kind == 3))" in expr
    assert "not (page == 4)" in expr
    assert " and " in expr


# -- Elasticsearch compiler (lines 465, 467, 469, 472-474, 477) ------------


def test_elasticsearch_all_leaf_ops():
    assert eq("kind", "table").to_elasticsearch() == {"term": {"kind": "table"}}
    assert contains("permissions", "admin").to_elasticsearch() == {
        "term": {"permissions": "admin"}
    }
    assert ne("kind", "table").to_elasticsearch() == {
        "bool": {"must_not": [{"term": {"kind": "table"}}]}
    }
    assert in_("kind", ["a", "b"]).to_elasticsearch() == {"terms": {"kind": ["a", "b"]}}
    assert FilterSpec(field="kind", op="nin", value=["a"]).to_elasticsearch() == {
        "bool": {"must_not": [{"terms": {"kind": ["a"]}}]}
    }
    assert exists("kind").to_elasticsearch() == {"exists": {"field": "kind"}}
    assert FilterSpec(field="page", op="gte", value=2).to_elasticsearch() == {
        "range": {"page": {"gte": 2}}
    }


def test_elasticsearch_metadata_path_prefixed():
    assert eq("metadata.author", "alice").to_elasticsearch() == {
        "term": {"metadata.author": "alice"}
    }
    # an unqualified non-chunk field is also routed under metadata.
    assert eq("author", "alice").to_elasticsearch() == {"term": {"metadata.author": "alice"}}


def test_elasticsearch_bool_must_should_must_not():
    # A single composite with all three lists -> one bool with must/should/must_not.
    spec = FilterSpec(
        must=[eq("kind", "a")],
        should=[eq("page", 2)],
        must_not=[eq("index", 3)],
    )
    q = spec.to_elasticsearch()["bool"]
    assert q["must"] == [{"term": {"kind": "a"}}]
    assert q["should"] == [{"term": {"page": 2}}]
    assert q["minimum_should_match"] == 1
    assert q["must_not"] == [{"term": {"index": 3}}]


# -- SQL compiler (lines 505-523, 528, 530) --------------------------------


def test_sql_exists_is_not_null():
    sql, params = exists("tenant_id").to_sql_where(column="json")
    assert sql == "(json ->> %s) IS NOT NULL"
    assert params == ["tenant_id"]


def test_sql_comparison_ops_stringify_value():
    # 'page' is a top-level chunk field -> read via `column ->> key` (not metadata).
    sql, params = FilterSpec(field="page", op="gte", value=5).to_sql_where()
    assert sql == "(json ->> %s) >= %s"
    assert params == ["page", "5"]  # value coerced to str
    # a non-chunk field routes through metadata.
    msql, _ = FilterSpec(field="score", op="gte", value=5).to_sql_where()
    assert msql == "(json -> 'metadata' ->> %s) >= %s"
    assert FilterSpec(field="page", op="ne", value=1).to_sql_where()[0].endswith("<> %s")


def test_sql_in_and_contains():
    sql, params = in_("kind", ["a", "b", "c"]).to_sql_where()
    assert sql == "(json ->> %s) IN (%s, %s, %s)"
    assert params == ["kind", "a", "b", "c"]
    csql, cparams = contains("permissions", "admin").to_sql_where()
    assert csql == "(json ->> %s) IN (%s)"
    assert cparams == ["permissions", "admin"]


def test_sql_nin():
    sql, params = FilterSpec(field="kind", op="nin", value=["a", "b"]).to_sql_where()
    assert sql == "(json ->> %s) NOT IN (%s, %s)"
    assert params == ["kind", "a", "b"]


def test_sql_composite_must_should_must_not():
    sql, params = and_(eq("a", 1), or_(eq("b", 2), eq("c", 3)), not_(eq("d", 4))).to_sql_where()
    assert " AND " in sql
    assert " OR " in sql
    assert "NOT (" in sql
    # params follow the must -> should -> must_not order, each value stringified.
    assert params == ["a", "1", "b", "2", "c", "3", "d", "4"]


def test_sql_empty_composite_is_true():
    # A FilterSpec with no constraints compiles to the TRUE sentinel.
    assert FilterSpec().to_sql_where() == ("TRUE", [])
