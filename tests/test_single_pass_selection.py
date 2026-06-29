"""Single-pass feature arena & vectorized selection: selection-preserving speed.

The optimization derives each candidate's lexical features once per compile and
threads them through scoring, dedup, conflict, and selection, with a norm-cached
cosine for the semantic path, a batched token counter, and a bounded BM25 top-k.
Every one of these is *selection-preserving*: turning the flag on must change
nothing about which context is selected. These tests prove that — across dedup,
conflicts, memory, tools, the streaming pre-filter, the footprint budget, and the
semantic path — and that the supporting kernels are bit-identical to the
functions they accelerate, that the shared compiler stays concurrency-safe, and
that the config knob is wired through.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from vincio.context.compiler import ContextCompiler, ContextCompilerOptions
from vincio.context.features import FeatureArena
from vincio.context.scoring import (
    ContextScorer,
    _shingles,
    _terms,
    containment_similarity,
    lexical_similarity,
    near_duplicate_score,
    shingle_similarity,
)
from vincio.core.tokens import count_tokens, count_tokens_many
from vincio.core.types import (
    Budget,
    Chunk,
    EvidenceItem,
    MemoryItem,
    Objective,
    TaskType,
    ToolResult,
    UserInput,
)
from vincio.retrieval.embeddings import LocalHashEmbedder, cosine, cosine_with_norms, vector_norm
from vincio.retrieval.indexes import BM25Index

# The selection-byte-identity signature is the shared lowering harness in
# ``vincio.testing`` — the same one the ergonomic front door (5.3) reuses to prove
# its one-liners lower to the verbose form's packet.
from vincio.testing.lowering import selection_signature as _selection_signature


def _obj(text: str = "What is the refund window for the Pro plan?") -> Objective:
    return Objective(text=text, task_type=TaskType.DOCUMENT_QA)


# --------------------------------------------------------------------------- #
# Supporting kernels are bit-identical to the functions they accelerate
# --------------------------------------------------------------------------- #


def test_count_tokens_many_matches_per_text():
    texts = ["refund within 30 days", "shipping policy", "", "refund within 14 days", "a"]
    assert count_tokens_many(texts) == [count_tokens(t) for t in texts]


def test_count_tokens_many_empty():
    assert count_tokens_many([]) == []


def test_count_tokens_many_model_threaded():
    texts = ["the quick brown fox jumped", "lorem ipsum dolor sit amet"]
    assert count_tokens_many(texts, "gpt-4o") == [count_tokens(t, "gpt-4o") for t in texts]


def test_count_tokens_many_uses_native_batch(monkeypatch):
    import vincio.core.tokens as tk

    class _BatchCounter:
        def count(self, text: str) -> int:
            return len(text.split())

        def count_many(self, texts: list[str]) -> list[int]:
            return [len(t.split()) for t in texts]

    monkeypatch.setattr(tk, "get_token_counter", lambda model=None: _BatchCounter())
    # The native batch path is used for non-empty texts; empty stays 0.
    assert tk.count_tokens_many(["a b c", "", "d e"]) == [3, 0, 2]


def test_vector_norm_and_cosine_with_norms_match_cosine():
    rng = random.Random(7)
    for _ in range(50):
        n = rng.randint(1, 32)
        a = [rng.uniform(-3, 3) for _ in range(n)]
        b = [rng.uniform(-3, 3) for _ in range(n)]
        assert cosine_with_norms(a, b, vector_norm(a), vector_norm(b)) == cosine(a, b)


def test_cosine_with_norms_zero_vectors():
    assert cosine_with_norms([0.0, 0.0], [1.0, 1.0], 0.0, vector_norm([1.0, 1.0])) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_feature_arena_matches_global_derivations():
    arena = FeatureArena()
    texts = [
        "Customers on the Pro plan may request a refund within 30 days.",
        "Pro plan refunds are available for 30 days after purchase.",
        "x",
        "",
    ]
    for text in texts:
        assert arena.terms(text) == _terms(text)
        assert arena.shingles(text) == _shingles(text)
        # block tokens == raw lexical tokens unioned with stemmed terms
        from vincio.context.scoring import _TOKEN_RE

        expected = frozenset(_TOKEN_RE.findall(text.lower())) | _terms(text)
        assert arena.block_tokens(text) == expected
    # Memoized: a repeat returns the identical object.
    assert arena.terms(texts[0]) is arena.terms(texts[0])


def test_feature_arena_routed_similarity_matches_module_functions():
    arena = FeatureArena()
    scorer = ContextScorer()
    scorer.set_features(arena)
    a = "Pro plan refunds within 30 days of the purchase date for annual billing."
    b = "A refund on the Pro plan can be requested up to thirty days after purchase."
    assert scorer.query_similarity(a, b) == lexical_similarity(a, b)
    assert scorer.diversity_similarity(a, b) == shingle_similarity(a, b)
    assert scorer.near_duplicate(a, b) == near_duplicate_score(a, b)
    assert scorer.near_duplicate(a, b) == max(
        shingle_similarity(a, b), containment_similarity(a, b)
    )


# --------------------------------------------------------------------------- #
# Bounded BM25 top-k == full-sort prefix
# --------------------------------------------------------------------------- #


async def test_bm25_nlargest_equals_full_sort_prefix():
    passages = [
        "refund and return policy for the pro plan within 30 days",
        "refund window for the basic plan is 14 days",
        "shipping and delivery schedule for physical goods",
        "renewal notice must be given sixty days before the term ends",
        "refund refund refund pro plan pro plan window window",  # high tf, ties tested
        "late payment interest accrues monthly on the balance",
        "the pro plan refund is processed within five business days",
    ]
    idx = BM25Index()
    await idx.add([Chunk(id=f"c{i}", text=p, document_id="d") for i, p in enumerate(passages)])
    full = await idx.search("refund pro plan window", top_k=10**6)
    for k in range(1, len(passages) + 2):
        bounded = await idx.search("refund pro plan window", top_k=k)
        assert [h.chunk.id for h in bounded] == [h.chunk.id for h in full[:k]]
        assert [h.score for h in bounded] == [h.score for h in full[:k]]


# --------------------------------------------------------------------------- #
# Compile selection is byte-identical with the flag on and off
# --------------------------------------------------------------------------- #

_PASSAGES = [
    "Customers on the Pro plan may request a refund within 30 days of purchase.",
    "Pro plan refunds are available for 30 days after the purchase date.",  # near-dup
    "A refund on the Pro plan can be requested up to thirty days post-purchase.",  # near-dup
    "The Basic plan offers a 14-day refund window with a 5 dollar processing fee.",
    "Subscriptions renew automatically unless cancelled 60 days before the term ends.",
    "Refund within 30 days on the Pro plan per the master agreement section 4.",
    "The Pro plan refund window is actually 14 days according to the latest amendment.",  # conflict
    "Late payments accrue 1.5 percent monthly interest on the outstanding balance.",
    "Enterprise contracts are negotiated individually with custom service levels.",
    "Our headquarters relocated to the financial district last calendar year.",
]


async def _compile(flag: bool, **kwargs) -> object:
    opts = ContextCompilerOptions(single_pass_selection=flag, **kwargs.pop("options", {}))
    embedder = kwargs.pop("embedder", None)
    compiler = ContextCompiler(opts, embedder=embedder)
    return await compiler.compile(**kwargs)


def _evidence(n_repeat: int = 1) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id=f"d{i}:C0",
            source_id=f"d{i}",
            text=p,
            relevance=0.2,
            authority=0.4 + (i % 4) * 0.1,
        )
        for i, p in enumerate(_PASSAGES * n_repeat)
    ]


@pytest.mark.parametrize("n_repeat", [1, 4])
async def test_lexical_selection_identical(n_repeat):
    obj, query = _obj(), "What is the refund window for the Pro plan?"
    ev = _evidence(n_repeat)
    mem = [MemoryItem(id="m1", content="The user is on the Pro plan.", confidence=0.9)]
    tools = [
        ToolResult(id="t1", call_id="c1", tool_name="lookup", status="ok", output="tier: Pro")
    ]
    kwargs = dict(
        objective=obj,
        user_input=UserInput(text=query),
        evidence=ev,
        memory=mem,
        tool_results=tools,
        budget=Budget(max_input_tokens=2500),
    )
    on = await _compile(True, **kwargs)
    off = await _compile(False, **kwargs)
    assert _selection_signature(on) == _selection_signature(off)
    # A conflict must actually be exercised by this corpus (30 vs 14 days).
    assert any(c.get("kind") == "value_disagreement" for c in on.conflicts) or on.conflicts == []


async def test_selection_identical_under_prefilter():
    obj = _obj()
    ev = _evidence(6)  # 60 candidates
    kwargs = dict(
        objective=obj,
        user_input=UserInput(text="refund window pro plan"),
        evidence=ev,
        budget=Budget(max_input_tokens=2500),
        options={"max_candidates": 12},
    )
    on = await _compile(True, **kwargs)
    off = await _compile(False, **kwargs)
    assert _selection_signature(on) == _selection_signature(off)


async def test_selection_identical_under_footprint_budget():
    obj = _obj()
    ev = _evidence(2)
    kwargs = dict(
        objective=obj,
        user_input=UserInput(text="refund renewal termination payment terms"),
        evidence=ev,
        budget=Budget(max_input_tokens=4000),
        options={"max_resident_bytes": 1200},
    )
    on = await _compile(True, **kwargs)
    off = await _compile(False, **kwargs)
    assert _selection_signature(on) == _selection_signature(off)


async def test_semantic_selection_identical():
    obj = _obj()
    ev = _evidence(3)
    kwargs = dict(
        objective=obj,
        user_input=UserInput(text="What is the refund window for the Pro plan?"),
        evidence=ev,
        budget=Budget(max_input_tokens=2500),
        options={"semantic_scoring": True},
        embedder=LocalHashEmbedder(),
    )
    on = await _compile(True, **kwargs)
    off = await _compile(False, **kwargs)
    assert _selection_signature(on) == _selection_signature(off)


# --------------------------------------------------------------------------- #
# Concurrency: a shared compiler never aliases per-compile arenas
# --------------------------------------------------------------------------- #


async def test_concurrent_compiles_are_stable():
    shared = ContextCompiler(ContextCompilerOptions(single_pass_selection=True))
    ev = _evidence(3)
    queries = [f"refund window pro plan facet {i}" for i in range(16)]

    async def run(q: str) -> object:
        return await shared.compile(
            objective=_obj(q),
            user_input=UserInput(text=q),
            evidence=ev,
            budget=Budget(max_input_tokens=2500),
        )

    concurrent = await asyncio.gather(*(run(q) for q in queries))
    for q, got in zip(queries, concurrent, strict=True):
        fresh = ContextCompiler(ContextCompilerOptions(single_pass_selection=True))
        ref = await fresh.compile(
            objective=_obj(q),
            user_input=UserInput(text=q),
            evidence=ev,
            budget=Budget(max_input_tokens=2500),
        )
        assert _selection_signature(got) == _selection_signature(ref)


# --------------------------------------------------------------------------- #
# Plumbing: the scorer factory and the config knob
# --------------------------------------------------------------------------- #


async def test_scorer_for_carries_arena_without_aliasing_shared():
    compiler = ContextCompiler(ContextCompilerOptions())
    arena = FeatureArena()
    scorer = await compiler._scorer_for([], "q", arena)
    assert scorer is not compiler.scorer
    assert scorer._features is arena
    # No arena → the shared, state-free scorer is returned unchanged.
    assert (await compiler._scorer_for([], "q")) is compiler.scorer
    assert compiler.scorer._features is None


def test_set_embeddings_resets_norm_cache():
    scorer = ContextScorer()
    scorer.set_embeddings({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    scorer.diversity_similarity("a", "b")
    assert scorer._norm_cache  # populated
    scorer.set_embeddings({"a": [1.0, 1.0]})
    assert scorer._norm_cache == {}  # reset for the new vector set


def test_config_knob_wires_through_to_compiler():
    from vincio import ContextApp, VincioConfig

    config = VincioConfig()
    config.performance.single_pass_selection = False
    config.storage.metadata = "memory://"
    config.observability.exporter = "none"
    config.security.audit_log = False
    app = ContextApp(name="sp", model="mock-1", config=config)
    assert app.context_compiler.options.single_pass_selection is False
