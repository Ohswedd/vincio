"""Exercise public helpers the audit (6.6) found reachable but unexercised.

The 6.6 audit ran the symbol-by-symbol reachability rubric across every
subpackage: define the intended public surface from ``__init__``, then confirm
every claim with a repo-wide reference check. It found a set of public symbols
that resolved (so :mod:`vincio._surface` passed them) and were live capabilities
(so they were not dead in the 6.0 sense), yet were referenced *nowhere* in the
code corpus — no internal caller, no test, no example, no benchmark. The
structurally-unexercisable ones (abstract bases users implement, optional-dep
provider backends, production sockets/webhooks) are declared in the frozen
reachability baseline (:mod:`vincio._reachability`,
``docs/reference/reachability.txt``). The *pure, offline-runnable* ones have no
structural excuse, so the honest audit completion is to exercise them — which is
exactly what this module does, one focused behavioural test per symbol. A
passing test here is the evidence the symbol is a real, working capability and
the reason it is **not** in the baseline.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

# -- caching layers -------------------------------------------------------------


def test_retrieval_cache_key_and_miss() -> None:
    from vincio.caching import RetrievalCache

    cache = RetrievalCache()
    key = cache.key("refund window", top_k=5, tenant_id=None, index_version="v1")
    # The key is stable for identical inputs and folds in the index version.
    assert key == cache.key("refund window", top_k=5, tenant_id=None, index_version="v1")
    assert key != cache.key("refund window", top_k=5, tenant_id=None, index_version="v2")
    assert cache.get(key) is None  # cold cache misses, never raises


def test_context_packet_cache_key_and_miss() -> None:
    from vincio.caching import ContextPacketCache

    cache = ContextPacketCache()
    key = cache.key(
        objective="answer the refund question",
        query="refund?",
        evidence_ids=["e1", "e2"],
        memory_ids=[],
        schema_ref=None,
    )
    assert key == cache.key(
        objective="answer the refund question",
        query="refund?",
        evidence_ids=["e1", "e2"],
        memory_ids=[],
        schema_ref=None,
    )
    assert cache.get(key) is None


def test_eval_result_cache_roundtrips() -> None:
    from vincio.caching import EvalResultCache

    cache = EvalResultCache()
    assert cache.get("case-1", "cfg-abc") is None
    cache.set("case-1", "cfg-abc", {"score": 0.9})
    assert cache.get("case-1", "cfg-abc") == {"score": 0.9}
    # A different config hash is a different entry.
    assert cache.get("case-1", "cfg-xyz") is None


# -- structured output registry -------------------------------------------------


def test_schema_registry_register_and_lookup() -> None:
    from vincio.output import SchemaRegistry

    class Invoice(BaseModel):
        total: float

    registry = SchemaRegistry()
    schema = registry.register(Invoice)
    assert schema.name in registry
    assert registry.get(schema.name) is schema


# -- prompt lint rule catalogue -------------------------------------------------


def test_lint_rules_catalogue() -> None:
    from vincio.prompts import LINT_RULES

    assert isinstance(LINT_RULES, dict)
    # Every rule id is a PROMPTNNN code mapped to a human description.
    assert LINT_RULES["PROMPT001"]
    assert all(code.startswith("PROMPT") for code in LINT_RULES)
    assert all(isinstance(desc, str) and desc for desc in LINT_RULES.values())


# -- document loader extension set ----------------------------------------------


def test_supported_extensions_set() -> None:
    from vincio.documents import SUPPORTED_EXTENSIONS

    assert isinstance(SUPPORTED_EXTENSIONS, set)
    # The dependency-free loaders cover the common text formats.
    assert {".json", ".csv", ".md"} <= SUPPORTED_EXTENSIONS
    assert all(ext.startswith(".") for ext in SUPPORTED_EXTENSIONS)


# -- MCP OAuth helpers ----------------------------------------------------------


def test_bearer_headers() -> None:
    from vincio.mcp import bearer_headers

    assert bearer_headers("tok-123") == {"Authorization": "Bearer tok-123"}


def test_pkce_pair_is_s256() -> None:
    import base64
    import hashlib

    from vincio.mcp import pkce_pair

    verifier, challenge = pkce_pair()
    assert verifier and challenge and verifier != challenge
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    assert challenge == expected


# -- AG-UI SSE rendering --------------------------------------------------------


def test_agui_sse_frame() -> None:
    from vincio.server import agui_sse
    from vincio.server.agui import AGUIEvent, AGUIEventType

    event = AGUIEvent(type=AGUIEventType.RUN_STARTED, thread_id="t1", run_id="r1")
    frame = agui_sse(event)
    assert isinstance(frame, str)
    assert "data:" in frame
    assert frame.endswith("\n\n")


# -- prompt-cache economics & layout --------------------------------------------


def test_cache_hit_economics() -> None:
    from vincio.optimize import cache_hit_economics

    econ = cache_hit_economics(
        stable_tokens=800,
        total_tokens=1000,
        calls_per_day=10_000,
        input_cost_per_mtok=3.0,
        cached_cost_per_mtok=0.3,
    )
    assert econ["cacheability"] == 0.8
    assert econ["savings_per_call_usd"] > 0
    assert econ["savings_per_day_usd"] == pytest.approx(
        econ["savings_per_call_usd"] * 10_000, rel=1e-6
    )


def test_analyze_ast_layout_flags_cache_breaker() -> None:
    from vincio.optimize import analyze_ast_layout

    class _Node:
        def __init__(self, kind: str, text: str, stable: bool) -> None:
            self.kind, self.text, self.stable = kind, text, stable

    class _AST:
        def __init__(self, nodes: list[_Node]) -> None:
            self._nodes = nodes

        def ordered(self) -> list[_Node]:
            return self._nodes

        @property
        def nodes(self) -> list[_Node]:
            return self._nodes

    # A stable block placed after dynamic content breaks the cacheable prefix.
    ast_layout = _AST(
        [
            _Node("user_query", "what is the refund window?", stable=False),
            _Node("system", "You are a careful support agent. " * 8, stable=True),
        ]
    )
    advice = analyze_ast_layout(ast_layout)
    assert any(a.code == "CACHE005" for a in advice)
    # A well-ordered layout (stable first) yields no cache-breaker advice.
    assert analyze_ast_layout(_AST([ast_layout.nodes[1], ast_layout.nodes[0]])) == []


# -- markdown metadata extraction -----------------------------------------------


def test_extract_markdown_metadata() -> None:
    from vincio.output import extract_markdown_metadata

    meta, body = extract_markdown_metadata("---\ntitle: Q3 review\nsafe: true\n---\n# Body\n")
    assert meta == {"title": "Q3 review", "safe": True}
    assert body.lstrip().startswith("# Body")
    # No front matter → empty metadata, body untouched.
    assert extract_markdown_metadata("plain body") == ({}, "plain body")


# -- retrieval-eval search adapter ----------------------------------------------


async def test_as_search_fn_adapts_retriever() -> None:
    from vincio.core.types import EvidenceItem
    from vincio.evals import as_search_fn

    item = EvidenceItem(id="e1", source_id="D1", text="the refund window is 30 days")

    class _Result:
        evidence = [item]

    class _Retriever:
        def retrieve(self, query: str, *, top_k: int) -> _Result:
            assert query == "refund?" and top_k == 3
            return _Result()

    search = as_search_fn(_Retriever())
    hits = await search("refund?", 3)
    assert hits == [item]


# -- learned-sparse encoder adapter ---------------------------------------------


async def test_callable_sparse_encoder_delegates() -> None:
    from vincio.retrieval import CallableSparseEncoder

    seen: dict[str, object] = {}

    async def encode_fn(texts: list[str], is_query: bool) -> list[dict[str, float]]:
        seen["texts"], seen["is_query"] = texts, is_query
        return [{"refund": 1.0}]

    encoder = CallableSparseEncoder(encode_fn)
    out = await encoder.encode(["refund window"], is_query=True)
    assert out == [{"refund": 1.0}]
    assert seen == {"texts": ["refund window"], "is_query": True}


# -- Haystack interop -----------------------------------------------------------


def test_from_haystack_documents() -> None:
    from vincio.interop import from_haystack_documents

    class _HSDoc:
        def __init__(self, content: str, meta: dict[str, object]) -> None:
            self.content, self.meta = content, meta

    docs = from_haystack_documents(
        [_HSDoc("refund policy text", {"title": "Refunds", "source": "kb://refunds"})]
    )
    assert len(docs) == 1
    assert docs[0].text == "refund policy text"
    assert docs[0].title == "Refunds"
    assert docs[0].source_uri == "kb://refunds"


# -- observability fan-out exporter ---------------------------------------------


def test_multi_exporter_fans_out() -> None:
    from vincio.observability import MultiExporter
    from vincio.observability.exporters import InMemoryExporter
    from vincio.observability.spans import Trace

    a, b = InMemoryExporter(), InMemoryExporter()
    trace = Trace(id="trace-1")
    MultiExporter([a, b]).export(trace)
    assert a.get("trace-1") is trace
    assert b.get("trace-1") is trace


# -- governance convenience shorthands ------------------------------------------


def test_map_compliance_returns_report() -> None:
    from vincio.governance import map_compliance
    from vincio.governance.frameworks import ComplianceReport

    report = map_compliance()
    assert isinstance(report, ComplianceReport)
    assert isinstance(report.coverage, list)


def test_residency_violation_shorthand() -> None:
    from vincio.governance import residency_violation

    # An EU-hosted provider under a US-only policy is a violation.
    blocked = residency_violation(
        provider="acme",
        model="acme-1",
        allowed_regions=["us"],
        provider_regions={"acme": "eu"},
    )
    assert blocked is not None
    # The same provider under an EU policy is allowed (no violation).
    allowed = residency_violation(
        provider="acme",
        model="acme-1",
        allowed_regions=["eu"],
        provider_regions={"acme": "eu"},
    )
    assert allowed is None


# -- routing threshold optimizer ------------------------------------------------


def test_routing_optimizer_learns_threshold() -> None:
    from vincio.evals.reports import CaseResult, EvalReport
    from vincio.optimize import RoutingOptimizer, RoutingPolicy

    policy = RoutingPolicy(cheap_model="cheap", default_model="mid", strong_model="strong")

    def report(name: str, quality: float) -> EvalReport:
        return EvalReport(
            name=name,
            cases=[
                CaseResult(
                    case_id=f"c{i}",
                    metrics={"lexical_overlap": quality},
                    details={"difficulty": 0.2},
                )
                for i in range(3)
            ],
        )

    # The cheap tier matches the default tier's quality at this difficulty, so the
    # optimizer should be willing to raise the low threshold to route more cheaply.
    learned = RoutingOptimizer().optimize(
        policy,
        {"cheap": report("cheap", 0.95), "default": report("default", 0.95)},
    )
    assert isinstance(learned, RoutingPolicy)
    assert learned.difficulty_threshold_low >= policy.difficulty_threshold_low
    # Missing reports is a clean no-op (returns the policy unchanged).
    assert RoutingOptimizer().optimize(policy, {}) is policy
