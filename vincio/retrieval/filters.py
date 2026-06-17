"""Structured, serializable retrieval filters with native backend pushdown.

A :class:`FilterSpec` is a declarative predicate tree (``eq`` / ``in`` / ``range``
/ ``and`` / ``or`` / ``not`` over chunk fields and metadata). Unlike the legacy
:data:`~vincio.retrieval.indexes.SearchFilter` — an opaque ``Callable[[Chunk],
bool]`` that can only ever post-filter rows the client already fetched — a
``FilterSpec`` is data, so it can be:

* evaluated client-side (:meth:`FilterSpec.matches`) on in-memory indexes, and
* compiled to each vector store's *native* filter (Qdrant ``Filter``, a pgvector
  SQL ``WHERE`` on ``jsonb``, Pinecone metadata filter, Weaviate ``where``,
  Milvus ``expr``, Elasticsearch ``bool``) so selective predicates are applied
  in the engine before ``top_k`` is taken.

Native pushdown fixes two problems with post-filtering: the **over-fetch
under-fill** bug (a selective filter silently starves ``top_k`` because the
client only over-fetched a fixed multiple) and the cross-tenant
**fetch-to-filter exfiltration** risk (other tenants' rows are read off the
server before being dropped client-side). Pushing tenant/ACL scope down means
those rows never leave the store.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..core.types import Chunk

__all__ = [
    "FieldOp",
    "FilterSpec",
    "as_predicate",
    "flat_filter_fields",
    "eq",
    "ne",
    "in_",
    "range_",
    "exists",
    "contains",
    "and_",
    "or_",
    "not_",
]

# Comparison operators on a single field. ``contains`` tests membership in a
# list-valued field (e.g. ``permissions``); ``exists`` tests presence.
FieldOp = Literal["eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "exists", "contains"]

# Top-level chunk fields addressable directly; anything else is read from the
# chunk's ``metadata`` dict (``metadata.`` prefix optional for those).
_CHUNK_FIELDS = frozenset(
    {"document_id", "tenant_id", "kind", "page", "index", "source_uri", "permissions", "entities"}
)


class FilterSpec(BaseModel):
    """A declarative filter: either a leaf condition (``field`` + ``op``) or a
    boolean combination (``must`` = AND, ``should`` = OR, ``must_not`` = NOR).

    Build with the module helpers (:func:`eq`, :func:`in_`, :func:`range_`,
    :func:`and_`, :func:`or_`, :func:`not_`) rather than by hand.
    """

    field: str | None = None
    op: FieldOp | None = None
    value: Any = None
    must: list[FilterSpec] = Field(default_factory=list)
    should: list[FilterSpec] = Field(default_factory=list)
    must_not: list[FilterSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_shape(self) -> FilterSpec:
        is_leaf = self.field is not None
        is_composite = bool(self.must or self.should or self.must_not)
        if is_leaf and is_composite:
            raise ValueError("FilterSpec is either a leaf (field+op) or a composite, not both")
        if is_leaf and self.op is None:
            raise ValueError("leaf FilterSpec requires an op")
        return self

    @property
    def is_leaf(self) -> bool:
        return self.field is not None

    # -- client-side evaluation ------------------------------------------------

    def matches(self, chunk: Chunk) -> bool:
        """Evaluate this filter against a chunk in pure Python."""
        if self.is_leaf:
            return _eval_leaf(self, _resolve_field(chunk, self.field or ""))
        if self.must and not all(child.matches(chunk) for child in self.must):
            return False
        if self.should and not any(child.matches(chunk) for child in self.should):
            return False
        if self.must_not and any(child.matches(chunk) for child in self.must_not):
            return False
        # An empty composite matches everything (no constraints).
        return True

    # -- native compilers ------------------------------------------------------

    def to_qdrant(self) -> Any:
        """Compile to a ``qdrant_client.models.Filter`` (lazy import)."""
        from qdrant_client import models as qmodels

        return _to_qdrant(self, qmodels)

    def to_pinecone(self) -> dict[str, Any]:
        """Compile to a Pinecone metadata filter dict."""
        return _to_mongo_style(self)

    def to_weaviate(self) -> dict[str, Any]:
        """Compile to a Weaviate ``where`` filter dict."""
        return _to_weaviate(self)

    def to_milvus(self) -> str:
        """Compile to a Milvus boolean ``expr`` string."""
        return _to_milvus(self)

    def to_elasticsearch(self) -> dict[str, Any]:
        """Compile to an Elasticsearch ``bool`` query dict."""
        return _to_elasticsearch(self)

    def to_sql_where(self, *, column: str = "json") -> tuple[str, list[Any]]:
        """Compile to a parameterized SQL ``WHERE`` fragment over a ``jsonb``
        ``column`` holding the full chunk (pgvector's ``json`` column). Returns
        ``(sql, params)`` with ``%s`` placeholders; top-level chunk fields read
        ``column ->> key`` and metadata fields read ``column -> 'metadata' ->>
        key``.
        """
        params: list[Any] = []
        sql = _to_sql(self, column, params)
        return sql or "TRUE", params


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def eq(field: str, value: Any) -> FilterSpec:
    return FilterSpec(field=field, op="eq", value=value)


def ne(field: str, value: Any) -> FilterSpec:
    return FilterSpec(field=field, op="ne", value=value)


def in_(field: str, values: list[Any]) -> FilterSpec:
    return FilterSpec(field=field, op="in", value=list(values))


def range_(
    field: str,
    *,
    gte: Any = None,
    lte: Any = None,
    gt: Any = None,
    lt: Any = None,
) -> FilterSpec:
    """A range as an AND of bound conditions (only the supplied bounds apply)."""
    bounds: list[FilterSpec] = []
    if gte is not None:
        bounds.append(FilterSpec(field=field, op="gte", value=gte))
    if gt is not None:
        bounds.append(FilterSpec(field=field, op="gt", value=gt))
    if lte is not None:
        bounds.append(FilterSpec(field=field, op="lte", value=lte))
    if lt is not None:
        bounds.append(FilterSpec(field=field, op="lt", value=lt))
    if not bounds:
        raise ValueError("range_ requires at least one bound")
    if len(bounds) == 1:
        return bounds[0]
    return FilterSpec(must=bounds)


def exists(field: str) -> FilterSpec:
    return FilterSpec(field=field, op="exists")


def contains(field: str, value: Any) -> FilterSpec:
    return FilterSpec(field=field, op="contains", value=value)


def and_(*specs: FilterSpec) -> FilterSpec:
    return FilterSpec(must=list(specs))


def or_(*specs: FilterSpec) -> FilterSpec:
    return FilterSpec(should=list(specs))


def not_(spec: FilterSpec) -> FilterSpec:
    return FilterSpec(must_not=[spec])


def as_predicate(where: Any) -> Any:
    """A ``Callable[[Chunk], bool]`` for either a FilterSpec or a legacy callable.

    Lets every index treat both filter forms uniformly when post-filtering.
    """
    if where is None:
        return None
    if isinstance(where, FilterSpec):
        return where.matches
    return where  # already a callable predicate


def flat_filter_fields(chunk: Chunk) -> dict[str, Any]:
    """Scalar, natively-filterable fields for a chunk.

    Vector stores that persist the chunk as an opaque JSON blob also persist
    these flat fields alongside it, so a compiled :class:`FilterSpec` matches
    *server-side* (Pinecone metadata, Weaviate properties, Milvus dynamic field,
    Elasticsearch keywords) — pushing tenant/document/kind/metadata scope into
    the backend instead of fetching rows and dropping them client-side.

    Metadata is flattened first so a top-level chunk field always wins a name
    clash; ``None`` values are dropped (some stores reject null metadata). An
    untagged tenant is stored as ``""`` (see :func:`build_filter_spec`'s
    shared-or-mine scope, which matches both null and empty)."""
    fields: dict[str, Any] = {}
    for key, value in (chunk.metadata or {}).items():
        if isinstance(value, (str, int, float, bool)):
            fields[key] = value
    fields["document_id"] = chunk.document_id
    fields["tenant_id"] = chunk.tenant_id or ""
    fields["kind"] = chunk.kind
    if chunk.page is not None:
        fields["page"] = chunk.page
    fields["index"] = chunk.index
    if chunk.source_uri is not None:
        fields["source_uri"] = chunk.source_uri
    return fields


# ---------------------------------------------------------------------------
# Field resolution + leaf evaluation
# ---------------------------------------------------------------------------


def _resolve_field(chunk: Chunk, field: str) -> Any:
    if field.startswith("metadata."):
        return chunk.metadata.get(field[len("metadata.") :])
    if field in _CHUNK_FIELDS:
        return getattr(chunk, field, None)
    # Unqualified, non-top-level names resolve from metadata.
    return chunk.metadata.get(field)


def _eval_leaf(spec: FilterSpec, actual: Any) -> bool:
    op, value = spec.op, spec.value
    if op == "exists":
        return actual is not None
    if op == "eq":
        return actual == value
    if op == "ne":
        return actual != value
    if op == "in":
        return actual in (value or [])
    if op == "nin":
        return actual not in (value or [])
    if op == "contains":
        return isinstance(actual, (list, tuple, set, str)) and value in actual
    if actual is None:
        return False
    try:
        if op == "gt":
            return actual > value
        if op == "gte":
            return actual >= value
        if op == "lt":
            return actual < value
        if op == "lte":
            return actual <= value
    except TypeError:
        return False
    return False


# ---------------------------------------------------------------------------
# Compilers
# ---------------------------------------------------------------------------


def _qdrant_field(field: str) -> str:
    # Chunks persist with metadata under "metadata"; top-level fields stay flat.
    if field.startswith("metadata."):
        return field
    if field in _CHUNK_FIELDS:
        return field
    return f"metadata.{field}"


def _to_qdrant(spec: FilterSpec, qmodels: Any) -> Any:
    if spec.is_leaf:
        key = _qdrant_field(spec.field or "")
        op, value = spec.op, spec.value
        if op == "eq" or op == "contains":
            return qmodels.Filter(
                must=[qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))]
            )
        if op == "ne":
            return qmodels.Filter(
                must_not=[qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))]
            )
        if op == "in":
            return qmodels.Filter(
                must=[qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=list(value or [])))]
            )
        if op == "nin":
            return qmodels.Filter(
                must_not=[
                    qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=list(value or [])))
                ]
            )
        if op == "exists":
            # "field is present" == "not null" in Qdrant payload terms.
            return qmodels.Filter(
                must_not=[qmodels.IsNullCondition(is_null=qmodels.PayloadField(key=key))]
            )
        if op in ("gt", "gte", "lt", "lte"):
            rng = {"gt": "gt", "gte": "gte", "lt": "lt", "lte": "lte"}[op]
            return qmodels.Filter(
                must=[qmodels.FieldCondition(key=key, range=qmodels.Range(**{rng: value}))]
            )
        raise ValueError(f"unsupported op for qdrant: {op}")
    return qmodels.Filter(
        must=[_to_qdrant(c, qmodels) for c in spec.must] or None,
        should=[_to_qdrant(c, qmodels) for c in spec.should] or None,
        must_not=[_to_qdrant(c, qmodels) for c in spec.must_not] or None,
    )


def _to_mongo_style(spec: FilterSpec) -> dict[str, Any]:
    """Mongo-style operator dict — used by Pinecone (and close to others)."""
    if spec.is_leaf:
        field = spec.field or ""
        field = field[len("metadata.") :] if field.startswith("metadata.") else field
        op, value = spec.op, spec.value
        mapping = {
            "eq": {"$eq": value},
            "ne": {"$ne": value},
            "in": {"$in": list(value or [])},
            "nin": {"$nin": list(value or [])},
            "gt": {"$gt": value},
            "gte": {"$gte": value},
            "lt": {"$lt": value},
            "lte": {"$lte": value},
            "contains": {"$in": [value]},
            "exists": {"$exists": True},
        }
        if op not in mapping:
            raise ValueError(f"unsupported op for pinecone: {op}")
        return {field: mapping[op]}
    clauses: dict[str, Any] = {}
    if spec.must:
        clauses["$and"] = [_to_mongo_style(c) for c in spec.must]
    if spec.should:
        clauses["$or"] = [_to_mongo_style(c) for c in spec.should]
    if spec.must_not:
        clauses["$and"] = clauses.get("$and", []) + [
            {"$nor": [_to_mongo_style(c) for c in spec.must_not]}
        ]
    return clauses


def _to_weaviate(spec: FilterSpec) -> dict[str, Any]:
    if spec.is_leaf:
        field = spec.field or ""
        path = field[len("metadata.") :] if field.startswith("metadata.") else field
        op, value = spec.op, spec.value
        if op == "exists":
            # Weaviate "is present" is IsNull=False.
            return {"path": [path], "operator": "IsNull", "valueBoolean": False}
        operator = {
            "eq": "Equal",
            "ne": "NotEqual",
            "gt": "GreaterThan",
            "gte": "GreaterThanEqual",
            "lt": "LessThan",
            "lte": "LessThanEqual",
            "contains": "ContainsAny",
            "in": "ContainsAny",
        }.get(op or "")
        if operator is None:
            raise ValueError(f"unsupported op for weaviate: {op}")
        value_key = _weaviate_value_key(value)
        if op in ("in", "contains"):
            return {"path": [path], "operator": operator, value_key: value if op == "in" else [value]}
        return {"path": [path], "operator": operator, value_key: value}
    operands = (
        [_to_weaviate(c) for c in spec.must]
        + [_to_weaviate(c) for c in spec.should]
        + [{"operator": "Not", "operands": [_to_weaviate(c)]} for c in spec.must_not]
    )
    operator = "And" if (spec.must or spec.must_not) and not spec.should else "Or"
    if len(operands) == 1:
        return operands[0]
    return {"operator": operator, "operands": operands}


def _weaviate_value_key(value: Any) -> str:
    if isinstance(value, bool):
        return "valueBoolean"
    if isinstance(value, int):
        return "valueInt"
    if isinstance(value, float):
        return "valueNumber"
    if isinstance(value, list):
        return "valueTextArray"
    return "valueText"


def _to_milvus(spec: FilterSpec) -> str:
    if spec.is_leaf:
        field = spec.field or ""
        ref = field if field in _CHUNK_FIELDS else f'metadata["{field.replace("metadata.", "")}"]'
        op, value = spec.op, spec.value
        lit = _milvus_literal(value)
        sym = {"eq": "==", "ne": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}.get(op or "")
        if sym:
            return f"{ref} {sym} {lit}"
        if op == "in":
            return f"{ref} in {_milvus_literal(list(value or []))}"
        if op == "nin":
            return f"{ref} not in {_milvus_literal(list(value or []))}"
        if op == "contains":
            return f"array_contains({ref}, {lit})"
        if op == "exists":
            return f"{ref} != null"
        raise ValueError(f"unsupported op for milvus: {op}")
    parts: list[str] = []
    if spec.must:
        parts.append(" and ".join(f"({_to_milvus(c)})" for c in spec.must))
    if spec.should:
        parts.append("(" + " or ".join(f"({_to_milvus(c)})" for c in spec.should) + ")")
    if spec.must_not:
        parts.append(" and ".join(f"not ({_to_milvus(c)})" for c in spec.must_not))
    return " and ".join(p for p in parts if p)


def _milvus_literal(value: Any) -> str:
    if isinstance(value, str):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_milvus_literal(v) for v in value) + "]"
    return str(value)


def _to_elasticsearch(spec: FilterSpec) -> dict[str, Any]:
    if spec.is_leaf:
        field = spec.field or ""
        path = field if field in _CHUNK_FIELDS else f"metadata.{field.replace('metadata.', '')}"
        op, value = spec.op, spec.value
        if op in ("eq", "contains"):
            return {"term": {path: value}}
        if op == "ne":
            return {"bool": {"must_not": [{"term": {path: value}}]}}
        if op == "in":
            return {"terms": {path: list(value or [])}}
        if op == "nin":
            return {"bool": {"must_not": [{"terms": {path: list(value or [])}}]}}
        if op == "exists":
            return {"exists": {"field": path}}
        if op in ("gt", "gte", "lt", "lte"):
            return {"range": {path: {op: value}}}
        raise ValueError(f"unsupported op for elasticsearch: {op}")
    bool_query: dict[str, list[Any]] = {}
    if spec.must:
        bool_query["must"] = [_to_elasticsearch(c) for c in spec.must]
    if spec.should:
        bool_query["should"] = [_to_elasticsearch(c) for c in spec.should]
        bool_query["minimum_should_match"] = 1  # type: ignore[assignment]
    if spec.must_not:
        bool_query["must_not"] = [_to_elasticsearch(c) for c in spec.must_not]
    return {"bool": bool_query}


def _sql_ref(field: str, column: str) -> tuple[str, str]:
    """Return (extract_sql, key) for a chunk field over a jsonb ``column``
    holding the full chunk. Top-level fields read ``column ->> key``; metadata
    fields read ``column -> 'metadata' ->> key``."""
    is_meta = field.startswith("metadata.") or field not in _CHUNK_FIELDS
    key = field[len("metadata.") :] if field.startswith("metadata.") else field
    if is_meta:
        return f"({column} -> 'metadata' ->> %s)", key
    return f"({column} ->> %s)", key


def _to_sql(spec: FilterSpec, column: str, params: list[Any]) -> str:
    if spec.is_leaf:
        field = spec.field or ""
        op, value = spec.op, spec.value
        ref, name = _sql_ref(field, column)
        if op == "exists":
            # Value-based presence (consistent with FilterSpec.matches): a key
            # serialized as JSON null is "absent", not present.
            params.append(name)
            return f"{ref} IS NOT NULL"
        sym = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}.get(op or "")
        if sym:
            params.append(name)
            params.append(str(value))
            return f"{ref} {sym} %s"
        if op in ("in", "contains"):
            values = list(value) if op == "in" else [value]
            params.append(name)
            placeholders = ", ".join(["%s"] * len(values))
            params.extend(str(v) for v in values)
            return f"{ref} IN ({placeholders})"
        if op == "nin":
            params.append(name)
            placeholders = ", ".join(["%s"] * len(value))
            params.extend(str(v) for v in value)
            return f"{ref} NOT IN ({placeholders})"
        raise ValueError(f"unsupported op for sql: {op}")
    parts: list[str] = []
    if spec.must:
        parts.append(" AND ".join(f"({_to_sql(c, column, params)})" for c in spec.must))
    if spec.should:
        parts.append("(" + " OR ".join(f"({_to_sql(c, column, params)})" for c in spec.should) + ")")
    if spec.must_not:
        parts.append(
            " AND ".join(f"NOT ({_to_sql(c, column, params)})" for c in spec.must_not)
        )
    return " AND ".join(p for p in parts if p)
