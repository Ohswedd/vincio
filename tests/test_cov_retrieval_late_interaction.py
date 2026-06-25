"""Real-behavior coverage for late-interaction (ColBERT-style) retrieval.

Exercises MaxSim scoring, PLAID-style centroid compression, deletion,
metadata filtering, and the empty/edge input paths through the real API
with the deterministic offline ``LocalHashEmbedder``. No mocks.
"""

from __future__ import annotations

import asyncio

import pytest

from vincio.core.types import Chunk
from vincio.retrieval.embeddings import LocalHashEmbedder
from vincio.retrieval.filters import eq
from vincio.retrieval.late_interaction import (
    LateInteractionIndex,
    _dot,
    _kmeans,
    _normalize,
    _tokenize,
)


def _chunk(cid: str, text: str, **meta) -> Chunk:
    return Chunk(id=cid, document_id="doc-" + cid, text=text, metadata=meta)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #


def test_tokenize_lowercases_and_splits_on_non_alnum():
    assert _tokenize("Hello, WORLD! foo-bar 42") == [
        "hello",
        "world",
        "foo",
        "bar",
        "42",
    ]


def test_normalize_produces_unit_vector():
    out = _normalize([3.0, 4.0])
    assert out == pytest.approx([0.6, 0.8])
    assert sum(v * v for v in out) == pytest.approx(1.0)


def test_normalize_zero_vector_uses_unit_denominator():
    # norm is 0 -> falls back to 1.0, vector stays all-zero (no ZeroDivision).
    assert _normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_dot_uses_shortest_length_no_strict():
    # zip(strict=False) -> the trailing element of the longer vector is ignored.
    assert _dot([1.0, 2.0, 3.0], [4.0, 5.0]) == pytest.approx(14.0)


# --------------------------------------------------------------------------- #
# _kmeans
# --------------------------------------------------------------------------- #


def test_kmeans_returns_copies_when_fewer_vectors_than_k():
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    out = _kmeans(vectors, k=5)
    assert out == vectors
    # copies, not aliases — mutating the result must not touch the input.
    out[0][0] = 99.0
    assert vectors[0][0] == 1.0


def test_kmeans_clusters_two_separated_groups_into_unit_centroids():
    a = _normalize([1.0, 0.0])
    b = _normalize([0.0, 1.0])
    # two tight clusters around two orthogonal directions.
    vectors = [a, a, a, b, b, b]
    centroids = _kmeans(vectors, k=2, iters=4)
    assert len(centroids) == 2
    for c in centroids:
        assert sum(v * v for v in c) == pytest.approx(1.0, abs=1e-9)
    # each original direction is near-perfectly matched by some centroid.
    for direction in (a, b):
        assert max(_dot(direction, c) for c in centroids) == pytest.approx(1.0, abs=1e-6)


def test_kmeans_is_deterministic():
    vectors = [_normalize([float(i % 3), float(i % 5), 1.0]) for i in range(20)]
    assert _kmeans(vectors, k=4) == _kmeans(vectors, k=4)


def test_kmeans_handles_empty_cluster_without_error():
    # All vectors point the same direction; with k=3 only one centroid ever wins
    # assignments, leaving the others empty (the `if not members: continue`
    # branch). The run must still return k unit centroids.
    same = _normalize([1.0, 0.0, 0.0])
    vectors = [list(same) for _ in range(8)]
    centroids = _kmeans(vectors, k=3, iters=4)
    assert len(centroids) == 3
    # the populated centroid recovers the common direction.
    assert max(_dot(same, c) for c in centroids) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# add / delete / len
# --------------------------------------------------------------------------- #


def test_len_tracks_added_chunks():
    idx = LateInteractionIndex()
    assert len(idx) == 0
    _run(idx.add([_chunk("a", "alpha beta"), _chunk("b", "gamma delta")]))
    assert len(idx) == 2


def test_add_empty_list_does_not_mark_codes_stale():
    idx = LateInteractionIndex()
    idx._codes_stale = False
    _run(idx.add([]))
    # the `if chunks:` guard skips the staleness flag for an empty batch.
    assert idx._codes_stale is False


def test_add_dedupes_document_tokens():
    idx = LateInteractionIndex()
    _run(idx.add([_chunk("a", "cat cat cat cat dog")]))
    # MaxSim is over distinct token vectors only: {cat, dog} -> 2 vectors.
    assert len(idx._doc_vectors["a"]) == 2


def test_add_truncates_to_max_doc_tokens():
    idx = LateInteractionIndex(max_doc_tokens=3)
    _run(idx.add([_chunk("a", "one two three four five six")]))
    assert len(idx._doc_vectors["a"]) == 3


def test_delete_returns_count_and_removes_state():
    idx = LateInteractionIndex()
    _run(idx.add([_chunk("a", "alpha"), _chunk("b", "beta")]))
    removed = _run(idx.delete(["a", "missing"]))
    assert removed == 1
    assert "a" not in idx.chunks
    assert "a" not in idx._doc_vectors
    assert "b" in idx.chunks


def test_delete_only_missing_ids_returns_zero_and_keeps_codes_fresh():
    idx = LateInteractionIndex()
    _run(idx.add([_chunk("a", "alpha")]))
    idx._codes_stale = False
    removed = _run(idx.delete(["nope", "also-nope"]))
    assert removed == 0
    # nothing removed -> the `if removed:` guard leaves codes fresh.
    assert idx._codes_stale is False


# --------------------------------------------------------------------------- #
# search — uncompressed MaxSim
# --------------------------------------------------------------------------- #


def test_search_empty_index_returns_empty():
    idx = LateInteractionIndex()
    assert _run(idx.search("anything")) == []


def test_search_query_with_no_tokens_returns_empty():
    idx = LateInteractionIndex()
    _run(idx.add([_chunk("a", "alpha beta")]))
    # punctuation-only query tokenizes to nothing.
    assert _run(idx.search("!!! ??? ...")) == []


def test_search_ranks_exact_match_first():
    idx = LateInteractionIndex()
    _run(
        idx.add(
            [
                _chunk("match", "machine learning models"),
                _chunk("other", "gardening soil compost"),
            ]
        )
    )
    hits = _run(idx.search("machine learning"))
    assert [h.chunk.id for h in hits] == ["match", "other"]
    assert hits[0].score > hits[1].score
    assert hits[0].source == "late_interaction"


def test_search_self_query_scores_near_one():
    idx = LateInteractionIndex()
    _run(idx.add([_chunk("a", "quantum entanglement physics")]))
    hits = _run(idx.search("quantum entanglement physics"))
    # every query token matches itself exactly -> mean MaxSim == 1.0.
    assert hits[0].score == pytest.approx(1.0, abs=1e-9)


def test_search_top_k_limits_results():
    idx = LateInteractionIndex()
    _run(idx.add([_chunk(str(i), f"topic{i} shared") for i in range(5)]))
    hits = _run(idx.search("shared", top_k=2))
    assert len(hits) == 2


def test_search_max_query_tokens_truncation_changes_score():
    # With only the first query token kept, the second doc's term cannot match.
    short = LateInteractionIndex(max_query_tokens=1)
    full = LateInteractionIndex(max_query_tokens=8)
    docs = [_chunk("a", "alpha"), _chunk("b", "omega")]
    _run(short.add(docs))
    _run(full.add(docs))
    short_hits = {h.chunk.id: h.score for h in _run(short.search("alpha omega"))}
    full_hits = {h.chunk.id: h.score for h in _run(full.search("alpha omega"))}
    # truncated query never sees "omega", so doc b scores lower than full query.
    assert full_hits["b"] > short_hits["b"]


def test_search_where_filter_excludes_nonmatching_chunks():
    idx = LateInteractionIndex()
    _run(
        idx.add(
            [
                _chunk("keep", "shared topic", lang="en"),
                _chunk("drop", "shared topic", lang="fr"),
            ]
        )
    )
    hits = _run(idx.search("shared topic", where=eq("metadata.lang", "en")))
    assert [h.chunk.id for h in hits] == ["keep"]


def test_search_callable_predicate_post_filters():
    idx = LateInteractionIndex()
    _run(idx.add([_chunk("a", "shared"), _chunk("bb", "shared")]))
    hits = _run(idx.search("shared", where=lambda c: len(c.id) == 1))
    assert [h.chunk.id for h in hits] == ["a"]


def test_maxsim_empty_doc_vectors_scores_zero():
    idx = LateInteractionIndex()
    qv = _run(idx._embed_tokens(["hello"]))
    assert idx._maxsim(qv, []) == 0.0


# --------------------------------------------------------------------------- #
# search — PLAID compression path
# --------------------------------------------------------------------------- #


def _compressed_index() -> LateInteractionIndex:
    # n_centroids=2 with > 2 docs triggers the compressed two-stage path.
    return LateInteractionIndex(
        embedder=LocalHashEmbedder(dim=64),
        compressed=True,
        n_centroids=2,
        n_probe=2,
        rerank_factor=4,
    )


def test_compressed_search_builds_codes_and_finds_match():
    idx = _compressed_index()
    _run(
        idx.add(
            [
                _chunk("ml", "machine learning neural networks"),
                _chunk("cook", "recipe baking flour sugar"),
                _chunk("garden", "soil compost plants water"),
                _chunk("space", "rocket orbit satellite launch"),
            ]
        )
    )
    assert idx._codes_stale is True
    hits = _run(idx.search("machine learning"))
    # codes were built lazily on first compressed search.
    assert idx._codes_stale is False
    assert idx._centroids  # centroids populated
    assert hits[0].chunk.id == "ml"
    # second compressed search: codes are already fresh, so the rebuild branch
    # is skipped and search goes straight to candidate generation.
    again = _run(idx.search("baking recipe"))
    assert idx._codes_stale is False
    assert again[0].chunk.id == "cook"


def test_compressed_codes_rebuilt_after_mutation():
    idx = _compressed_index()
    _run(idx.add([_chunk(str(i), f"word{i} common term") for i in range(4)]))
    _run(idx.search("common"))
    assert idx._codes_stale is False
    # adding invalidates the centroid codes.
    _run(idx.add([_chunk("new", "fresh common term")]))
    assert idx._codes_stale is True
    hits = _run(idx.search("fresh"))
    assert idx._codes_stale is False
    assert hits[0].chunk.id == "new"


def test_compressed_below_centroid_threshold_uses_exhaustive_path():
    # len(chunks) == n_centroids -> NOT > n_centroids -> exhaustive branch,
    # so codes are never built.
    idx = LateInteractionIndex(compressed=True, n_centroids=2)
    _run(idx.add([_chunk("a", "alpha unique"), _chunk("b", "beta unique")]))
    hits = _run(idx.search("alpha"))
    assert idx._codes_stale is True  # never built
    assert hits[0].chunk.id == "a"


def test_build_codes_populates_inverted_lists():
    idx = _compressed_index()
    _run(
        idx.add(
            [
                _chunk("a", "alpha one two"),
                _chunk("b", "beta three four"),
                _chunk("c", "gamma five six"),
            ]
        )
    )
    idx._build_codes()
    assert idx._codes_stale is False
    assert len(idx._centroids) == 2
    # every doc has at least one centroid code, and inverted lists round-trip.
    for cid in ("a", "b", "c"):
        codes = idx._doc_codes[cid]
        assert codes
        for code in codes:
            assert cid in idx._centroid_docs[code]


def test_candidates_returns_subset_ranked_by_approx_maxsim():
    idx = _compressed_index()
    _run(
        idx.add(
            [
                _chunk("ml", "machine learning models training"),
                _chunk("cook", "baking flour sugar eggs"),
                _chunk("garden", "soil compost plants water"),
                _chunk("space", "rocket orbit satellite"),
            ]
        )
    )
    idx._build_codes()
    qv = _run(idx._embed_tokens(_tokenize("machine learning models")))
    candidates = idx._candidates(qv)
    # candidates come only from probed centroids' inverted lists.
    assert "ml" in candidates
    assert set(candidates).issubset(set(idx.chunks))


def test_compressed_search_respects_where_filter():
    idx = _compressed_index()
    _run(
        idx.add(
            [
                _chunk("a", "common shared word", lang="en"),
                _chunk("b", "common shared word", lang="fr"),
                _chunk("c", "common shared word", lang="en"),
                _chunk("d", "common shared word", lang="fr"),
            ]
        )
    )
    hits = _run(idx.search("common shared", where=eq("metadata.lang", "en")))
    assert {h.chunk.id for h in hits} == {"a", "c"}


# --------------------------------------------------------------------------- #
# constructor clamping
# --------------------------------------------------------------------------- #


def test_constructor_clamps_n_probe_and_rerank_factor_to_minimum_one():
    idx = LateInteractionIndex(n_probe=0, rerank_factor=-3)
    assert idx.n_probe == 1
    assert idx.rerank_factor == 1


def test_default_embedder_is_local_hash_embedder():
    idx = LateInteractionIndex()
    assert isinstance(idx.embedder, LocalHashEmbedder)
