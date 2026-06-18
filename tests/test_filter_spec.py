"""2.0 structured FilterSpec: a serializable, pushdown-capable retrieval filter
that replaces the opaque post-filter callable and closes the over-fetch
under-fill and cross-tenant fetch-to-filter gaps."""

from __future__ import annotations

import pytest

from vincio.core.types import Chunk
from vincio.retrieval.filters import (
    FilterSpec,
    and_,
    as_predicate,
    contains,
    eq,
    exists,
    in_,
    ne,
    not_,
    or_,
    range_,
)
from vincio.retrieval.indexes import BM25Index, VectorIndex, build_filter_spec


def _chunk(**kw) -> Chunk:
    base = dict(document_id="d1", text="hello world")
    base.update(kw)
    return Chunk(**base)


def test_matches_leaf_ops():
    c = _chunk(tenant_id="t1", page=5, kind="table", permissions=["admin"], metadata={"y": 2024})
    assert eq("tenant_id", "t1").matches(c)
    assert not eq("tenant_id", "t2").matches(c)
    assert ne("kind", "text").matches(c)
    assert in_("kind", ["table", "code"]).matches(c)
    assert range_("page", gte=3, lte=10).matches(c)
    assert not range_("page", gt=5).matches(c)
    assert contains("permissions", "admin").matches(c)
    assert exists("tenant_id").matches(c)
    assert not exists("metadata.missing").matches(c)
    assert eq("metadata.y", 2024).matches(c)
    assert eq("y", 2024).matches(c)  # unqualified resolves from metadata


def test_matches_composites():
    c = _chunk(tenant_id="t1", kind="table")
    assert and_(eq("tenant_id", "t1"), eq("kind", "table")).matches(c)
    assert not and_(eq("tenant_id", "t1"), eq("kind", "text")).matches(c)
    assert or_(eq("kind", "text"), eq("kind", "table")).matches(c)
    assert not_(eq("kind", "text")).matches(c)


def test_filterspec_is_serializable_roundtrip():
    spec = and_(eq("tenant_id", "t1"), or_(eq("kind", "table"), in_("page", [1, 2])))
    restored = FilterSpec.model_validate_json(spec.model_dump_json())
    c = _chunk(tenant_id="t1", kind="table", page=1)
    assert restored.matches(c)


def test_leaf_cannot_also_be_composite():
    with pytest.raises(ValueError):
        FilterSpec(field="x", op="eq", value=1, must=[eq("y", 2)])


def test_as_predicate_handles_both_forms():
    c = _chunk(tenant_id="t1")
    spec_pred = as_predicate(eq("tenant_id", "t1"))
    assert spec_pred(c) is True
    call_pred = as_predicate(lambda ch: ch.tenant_id == "t1")
    assert call_pred(c) is True
    assert as_predicate(None) is None


async def test_in_memory_index_pushdown_with_filterspec():
    chunks = [
        _chunk(id="a", tenant_id="t1", text="alpha report"),
        _chunk(id="b", tenant_id="t2", text="alpha report"),
        _chunk(id="c", tenant_id="t1", text="alpha report"),
    ]
    bm = BM25Index()
    await bm.add(chunks)
    hits = await bm.search("alpha report", top_k=10, where=eq("tenant_id", "t1"))
    assert {h.chunk.id for h in hits} == {"a", "c"}

    vec = VectorIndex()
    await vec.add(chunks)
    vhits = await vec.search("alpha report", top_k=10, where=eq("tenant_id", "t1"))
    assert all(h.chunk.tenant_id == "t1" for h in vhits)


def test_build_filter_spec_and_tenant_scope():
    spec = build_filter_spec(tenant_id="t1", kinds=["table"])
    assert spec is not None
    c1 = _chunk(tenant_id="t1", kind="table")
    c2 = _chunk(tenant_id="t2", kind="table")
    assert spec.matches(c1)
    assert not spec.matches(c2)
    assert build_filter_spec() is None


def test_app_tenant_filter_returns_filterspec():
    from vincio import ContextApp

    app = ContextApp(name="t")
    f = app.tenant_filter("tenant-9")
    assert isinstance(f, FilterSpec)
    assert f.matches(_chunk(tenant_id="tenant-9"))
    assert not f.matches(_chunk(tenant_id="other"))


# -- native compilers ------------------------------------------------------


def test_compile_pinecone():
    spec = and_(eq("tenant_id", "t1"), in_("kind", ["a", "b"]))
    assert spec.to_pinecone() == {
        "$and": [{"tenant_id": {"$eq": "t1"}}, {"kind": {"$in": ["a", "b"]}}]
    }


def test_compile_milvus():
    expr = and_(eq("tenant_id", "t1"), eq("metadata.author", "alice")).to_milvus()
    assert 'tenant_id == "t1"' in expr
    assert 'metadata["author"] == "alice"' in expr


def test_compile_elasticsearch():
    q = or_(eq("kind", "table"), eq("kind", "code")).to_elasticsearch()
    assert q["bool"]["should"]
    assert q["bool"]["minimum_should_match"] == 1


def test_compile_weaviate():
    w = eq("tenant_id", "t1").to_weaviate()
    assert w == {"path": ["tenant_id"], "operator": "Equal", "valueText": "t1"}


def test_compile_sql_nests_metadata_vs_toplevel():
    sql, params = and_(eq("tenant_id", "t1"), eq("metadata.author", "alice")).to_sql_where(
        column="json"
    )
    # Top-level field reads json ->> key; metadata reads json -> 'metadata' ->> key.
    assert "(json ->> %s)" in sql
    assert "(json -> 'metadata' ->> %s)" in sql
    assert params == ["tenant_id", "t1", "author", "alice"]


def test_compile_qdrant_lazy_import_guarded():
    # to_qdrant requires qdrant_client; assert it raises cleanly when absent,
    # or compiles to a Filter when present.
    spec = eq("tenant_id", "t1")
    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            spec.to_qdrant()
    else:  # pragma: no cover - only when qdrant installed
        assert spec.to_qdrant() is not None
