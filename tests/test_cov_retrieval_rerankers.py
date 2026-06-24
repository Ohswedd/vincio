"""Real-behavior coverage tests for vincio.retrieval.rerankers.

Every reranker is driven through its real ``rerank`` API on concrete
``SearchHit``/``Chunk`` objects; the LLM path uses the deterministic
``MockProvider`` and the hosted HTTP path uses ``httpx.MockTransport``.
Assertions pin exact ordering, computed scores, truncation, and error
messages — no mocking of the modules under test.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from vincio.core.errors import ConfigError, ProviderAuthError, ProviderError
from vincio.core.types import Chunk
from vincio.core.utils import utcnow
from vincio.providers import MockProvider
from vincio.retrieval.indexes import SearchHit
from vincio.retrieval.rerankers import (
    AuthorityReranker,
    CohereReranker,
    CrossEncoderReranker,
    HeuristicReranker,
    HTTPReranker,
    JinaReranker,
    LLMReranker,
    LocalCrossEncoderReranker,
    RecencyReranker,
    VoyageReranker,
    build_reranker,
    register_reranker,
)


def _chunk(text: str, **kw) -> Chunk:
    return Chunk(document_id=kw.pop("document_id", "doc"), text=text, **kw)


def _hit(text: str, score: float = 1.0, source: str = "idx", **chunk_kw) -> SearchHit:
    return SearchHit(chunk=_chunk(text, **chunk_kw), score=score, source=source)


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- #
# HeuristicReranker
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_heuristic_empty_hits_returns_empty_list():
    assert await HeuristicReranker().rerank("q", [], top_k=5) == []


@pytest.mark.asyncio
async def test_heuristic_exact_phrase_match_outranks_higher_retrieval_score():
    # The phrase-bearing chunk has a LOWER retrieval score but should win on the
    # lexical bonus (+0.4 for an exact phrase substring match).
    hits = [
        _hit("nothing relevant here at all", score=1.0),
        _hit("the quick brown fox jumps", score=0.4),
    ]
    out = await HeuristicReranker().rerank("quick brown fox", hits, top_k=2)
    assert [h.chunk.text for h in out][0] == "the quick brown fox jumps"


@pytest.mark.asyncio
async def test_heuristic_table_structure_boost_for_price_query():
    # A "table" chunk matching the price/cost regex earns the full structure
    # weight (0.15); an otherwise identical text chunk does not.
    table = _hit("12 USD per seat", score=0.5, kind="table")
    text = _hit("12 USD per seat", score=0.5, kind="text")
    out = await HeuristicReranker().rerank("how much does it cost", [text, table], top_k=2)
    assert out[0].chunk.kind == "table"
    assert out[0].score == pytest.approx(out[1].score + 0.15, abs=1e-9)


@pytest.mark.asyncio
async def test_heuristic_blended_score_is_computed_exactly():
    # Single hit, no phrase match, no structure: blended == 0.5*1.0 + 0.35*lex.
    from vincio.context.scoring import lexical_similarity

    rr = HeuristicReranker()
    out = await rr.rerank("zzz", [_hit("alpha beta", score=2.0)], top_k=1)
    lex = lexical_similarity("alpha beta", "zzz")
    assert out[0].score == pytest.approx(0.5 * 1.0 + 0.35 * lex + 0.15 * 0.0, abs=1e-9)


@pytest.mark.asyncio
async def test_heuristic_section_path_structure_branch():
    # A chunk with a section_path that lexically overlaps the query gets the 0.6
    # structure prior; one with an irrelevant path does not.
    matched = _hit("body text", score=0.5, section_path=["billing", "pricing"])
    unmatched = _hit("body text", score=0.5, section_path=["unrelated", "topic"])
    out = await HeuristicReranker().rerank("billing pricing", [unmatched, matched], top_k=2)
    assert out[0].chunk.section_path == ["billing", "pricing"]
    assert out[0].score == pytest.approx(out[1].score + 0.15 * 0.6, abs=1e-9)


@pytest.mark.asyncio
async def test_heuristic_top_k_truncates():
    hits = [_hit(f"chunk {i}", score=float(i)) for i in range(5)]
    out = await HeuristicReranker().rerank("chunk", hits, top_k=2)
    assert len(out) == 2


# --------------------------------------------------------------------------- #
# RecencyReranker
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recency_half_life_decays_to_half():
    now = utcnow()
    fresh = _hit("fresh", score=1.0, created_at=now)
    old = _hit("old", score=1.0, created_at=now - timedelta(days=90))
    out = await RecencyReranker(half_life_days=90.0).rerank("q", [old, fresh], top_k=2)
    assert [h.chunk.text for h in out] == ["fresh", "old"]
    # One half-life back -> ~0.5 of the original score.
    old_out = next(h for h in out if h.chunk.text == "old")
    assert old_out.score == pytest.approx(0.5, abs=0.02)


@pytest.mark.asyncio
async def test_recency_naive_datetime_is_treated_as_utc():
    # A tz-naive created_at must be coerced to UTC (the inner-branch path) and
    # not raise on the subtraction with a tz-aware "now".
    naive = datetime.now(UTC).replace(tzinfo=None)
    hit = _hit("naive", score=1.0, created_at=naive)
    out = await RecencyReranker(half_life_days=30.0).rerank("q", [hit], top_k=1)
    assert out[0].score == pytest.approx(1.0, abs=0.02)


@pytest.mark.asyncio
async def test_recency_missing_created_at_keeps_full_score():
    hit = _hit("no date", score=0.7)  # created_at defaults to None
    out = await RecencyReranker().rerank("q", [hit], top_k=1)
    assert out[0].score == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_recency_future_date_clamped_to_no_decay():
    future = utcnow() + timedelta(days=10)
    hit = _hit("future", score=1.0, created_at=future)
    out = await RecencyReranker().rerank("q", [hit], top_k=1)
    assert out[0].score == pytest.approx(1.0)  # age clamped to >= 0


# --------------------------------------------------------------------------- #
# AuthorityReranker
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_authority_empty_hits():
    assert await AuthorityReranker().rerank("q", [], top_k=3) == []


@pytest.mark.asyncio
async def test_authority_boosts_high_authority_metadata():
    official = _hit("official", score=0.5, metadata={"authority": 0.9})
    chat = _hit("chat log", score=0.5, metadata={"authority": 0.1})
    out = await AuthorityReranker(weight=0.4).rerank("q", [chat, official], top_k=2)
    assert [h.chunk.text for h in out] == ["official", "chat log"]
    # blended = 0.6*(0.5/0.5) + 0.4*0.9 = 0.96
    assert out[0].score == pytest.approx(0.6 * 1.0 + 0.4 * 0.9, abs=1e-9)


@pytest.mark.asyncio
async def test_authority_defaults_to_half_when_metadata_absent():
    hit = _hit("plain", score=0.5)  # no authority key -> 0.5
    out = await AuthorityReranker(weight=0.4).rerank("q", [hit], top_k=1)
    assert out[0].score == pytest.approx(0.6 * 1.0 + 0.4 * 0.5, abs=1e-9)


# --------------------------------------------------------------------------- #
# LLMReranker
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_llm_reranker_empty_hits():
    rr = LLMReranker(MockProvider(), model="m")
    assert await rr.rerank("q", [], top_k=2) == []


@pytest.mark.asyncio
async def test_llm_reranker_uses_structured_scores_and_orders():
    def responder(request):
        return {"scores": [{"id": 0, "relevance": 0.2}, {"id": 1, "relevance": 0.95}]}

    rr = LLMReranker(MockProvider(responder=responder), model="m")
    out = await rr.rerank("q", [_hit("a", 1.0), _hit("b", 1.0)], top_k=2)
    assert [h.chunk.text for h in out] == ["b", "a"]
    assert out[0].score == pytest.approx(0.95)
    assert out[1].score == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_llm_reranker_missing_id_falls_back_to_decayed_retrieval():
    # Only passage 0 is scored; passage 1 falls back to score * 0.01.
    def responder(request):
        return {"scores": [{"id": 0, "relevance": 0.3}]}

    rr = LLMReranker(MockProvider(responder=responder), model="m")
    out = await rr.rerank("q", [_hit("a", 1.0), _hit("b", 5.0)], top_k=2)
    by_text = {h.chunk.text: h.score for h in out}
    assert by_text["a"] == pytest.approx(0.3)
    assert by_text["b"] == pytest.approx(5.0 * 0.01)
    assert out[0].chunk.text == "a"  # 0.3 > 0.05


@pytest.mark.asyncio
async def test_llm_reranker_parses_json_text_when_structured_absent():
    # Provider returns raw JSON text with no structured field; the reranker
    # must json.loads it. (MockProvider populates structured only when the
    # request carries a schema, but a plain string responder bypasses that —
    # here we hand back a ModelResponse with text only.)
    from vincio.core.types import ModelResponse

    def responder(request):
        return ModelResponse(text='{"scores": [{"id": 1, "relevance": 0.8}]}', structured=None)

    rr = LLMReranker(MockProvider(responder=responder), model="m")
    out = await rr.rerank("q", [_hit("a", 1.0), _hit("b", 1.0)], top_k=1)
    assert out[0].chunk.text == "b"
    assert out[0].score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_llm_reranker_invalid_json_text_yields_decay_fallback():
    from vincio.core.types import ModelResponse

    def responder(request):
        return ModelResponse(text="not json at all", structured=None)

    rr = LLMReranker(MockProvider(responder=responder), model="m")
    out = await rr.rerank("q", [_hit("a", 2.0)], top_k=1)
    # No scores parsed -> fallback to score * 0.01.
    assert out[0].score == pytest.approx(2.0 * 0.01)


@pytest.mark.asyncio
async def test_llm_reranker_truncates_long_passages_in_prompt():
    captured = {}

    def responder(request):
        captured["request"] = request
        return {"scores": [{"id": 0, "relevance": 0.5}]}

    rr = LLMReranker(MockProvider(responder=responder), model="m", max_passage_chars=10)
    long_text = "x" * 500
    await rr.rerank("q", [_hit(long_text, 1.0)], top_k=1)
    user_msg = captured["request"].messages[-1].content
    assert "[0] " + "x" * 10 in user_msg
    assert "x" * 11 not in user_msg


# --------------------------------------------------------------------------- #
# CrossEncoderReranker
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cross_encoder_empty_hits():
    async def never_called(query, passages):  # pragma: no cover - asserts not reached
        raise AssertionError("score_fn must not run on empty input")

    assert await CrossEncoderReranker(never_called).rerank("q", [], top_k=2) == []


@pytest.mark.asyncio
async def test_cross_encoder_applies_external_scores_and_sorts():
    async def score_fn(query, passages):
        assert passages == ["a", "b", "c"]
        return [0.1, 0.9, 0.5]

    out = await CrossEncoderReranker(score_fn).rerank(
        "q", [_hit("a"), _hit("b"), _hit("c")], top_k=2
    )
    assert [h.chunk.text for h in out] == ["b", "c"]
    assert out[0].score == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# LocalCrossEncoderReranker
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_local_cross_encoder_empty_hits_short_circuits():
    rr = LocalCrossEncoderReranker(score_fn=lambda q, p: None)  # never invoked
    assert await rr.rerank("q", [], top_k=2) == []


@pytest.mark.asyncio
async def test_local_cross_encoder_injected_score_fn():
    async def score_fn(query, passages):
        return [0.2, 0.8]

    rr = LocalCrossEncoderReranker(score_fn=score_fn)
    out = await rr.rerank("q", [_hit("a"), _hit("b")], top_k=2)
    assert [h.chunk.text for h in out] == ["b", "a"]


@pytest.mark.asyncio
async def test_local_cross_encoder_injected_model_predict_path():
    # A faithful CrossEncoder fake: ``.predict(pairs)`` returns one float per
    # (query, passage) pair. Exercises the run_in_executor / model.predict path.
    class FakeCrossEncoder:
        def predict(self, pairs):
            assert pairs == [("q", "a"), ("q", "b")]
            return [0.3, 0.7]

    rr = LocalCrossEncoderReranker(model=FakeCrossEncoder())
    out = await rr.rerank("q", [_hit("a"), _hit("b")], top_k=2)
    assert [h.chunk.text for h in out] == ["b", "a"]
    assert out[0].score == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_local_cross_encoder_missing_dep_raises_configerror():
    # No score_fn, no model, no fallback, and sentence-transformers is absent in
    # the offline env -> a helpful ConfigError on the install command.
    rr = LocalCrossEncoderReranker()
    with pytest.raises(ConfigError, match=r"vincio\[cross-encoder\]"):
        await rr.rerank("q", [_hit("a")], top_k=1)


@pytest.mark.asyncio
async def test_local_cross_encoder_fallback_degrades_to_heuristic():
    # With fallback=True and the dependency missing, it must reuse the
    # HeuristicReranker rather than raise — phrase match should win.
    rr = LocalCrossEncoderReranker(fallback=True)
    # Equal retrieval scores so the heuristic's lexical/phrase signal decides:
    # the exact-phrase chunk must win, proving the heuristic actually ran.
    hits = [_hit("totally unrelated content", score=1.0), _hit("the magic word", score=1.0)]
    out = await rr.rerank("magic word", hits, top_k=2)
    assert out[0].chunk.text == "the magic word"


# --------------------------------------------------------------------------- #
# HTTPReranker family
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_http_reranker_empty_hits_no_request():
    rr = CohereReranker(api_key="k")  # no client; must not try to call out
    assert await rr.rerank("q", [], top_k=3) == []


@pytest.mark.asyncio
async def test_http_reranker_missing_api_key_raises_auth_error():
    def handler(request):  # pragma: no cover - never reached, auth fails first
        raise AssertionError("request should not be sent without an API key")

    rr = CohereReranker(client=_mock_client(handler))  # api_key=None
    with pytest.raises(ProviderAuthError, match="missing API key for reranker 'cohere'"):
        await rr.rerank("q", [_hit("a")], top_k=1)


@pytest.mark.asyncio
async def test_http_reranker_http_error_status_raises_provider_error():
    def handler(request):
        return httpx.Response(429, text="rate limited")

    rr = CohereReranker(api_key="k", client=_mock_client(handler))
    with pytest.raises(ProviderError, match="reranker 'cohere' error 429"):
        await rr.rerank("q", [_hit("a")], top_k=1)


@pytest.mark.asyncio
async def test_http_reranker_unscored_docs_fall_to_tail():
    # Endpoint scores only index 1; index 0 must appear after, score zeroed.
    def handler(request):
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.7}]})

    rr = CohereReranker(api_key="k", client=_mock_client(handler))
    out = await rr.rerank("q", [_hit("a", 5.0), _hit("b", 5.0)], top_k=5)
    assert [h.chunk.text for h in out] == ["b", "a"]
    assert out[0].score == pytest.approx(0.7)
    assert out[1].score == pytest.approx(0.0)  # tail doc zeroed
    assert all(h.source == "cohere" for h in out)


@pytest.mark.asyncio
async def test_http_reranker_parse_skips_items_without_index():
    # A malformed result item (no "index") is ignored by _parse; the valid one
    # ranks, the unscored doc tails.
    def handler(request):
        return httpx.Response(
            200,
            json={"results": [{"relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}]},
        )

    rr = CohereReranker(api_key="k", client=_mock_client(handler))
    out = await rr.rerank("q", [_hit("a"), _hit("b")], top_k=5)
    assert [h.chunk.text for h in out] == ["a", "b"]
    assert out[0].score == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_http_reranker_creates_and_closes_own_client_on_error():
    # Without an injected client, a created AsyncClient must be closed even when
    # the auth header check fails before any request goes out. ProviderAuthError
    # propagates and no client leak occurs (no event-loop warnings).
    rr = JinaReranker()  # no api_key, no client
    with pytest.raises(ProviderAuthError, match="reranker 'jina'"):
        await rr.rerank("q", [_hit("a")], top_k=1)


@pytest.mark.asyncio
async def test_voyage_reranker_uses_data_key_and_top_k_param():
    seen = {}

    def handler(request):
        import json

        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": 1, "relevance_score": 0.8}]})

    rr = VoyageReranker(api_key="k", client=_mock_client(handler))
    out = await rr.rerank("q", [_hit("a"), _hit("b")], top_k=1)
    assert [h.chunk.text for h in out] == ["b"]
    assert "top_k" in seen["payload"] and "top_n" not in seen["payload"]
    assert seen["payload"]["model"] == "rerank-2"


@pytest.mark.asyncio
async def test_http_reranker_payload_and_base_class_defaults():
    rr = HTTPReranker(api_key="k", base_url="https://x/rerank/", model="m")
    assert rr.base_url == "https://x/rerank"  # trailing slash stripped
    payload = rr._payload("q", ["a", "b"], 3)
    assert payload == {"model": "m", "query": "q", "documents": ["a", "b"], "top_n": 3}


# --------------------------------------------------------------------------- #
# build_reranker / register_reranker
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", [None, "none"])
def test_build_reranker_none_returns_none(kind):
    assert build_reranker(kind) is None


def test_build_reranker_dispatches_each_builtin():
    assert isinstance(build_reranker("heuristic"), HeuristicReranker)
    assert isinstance(build_reranker("recency", half_life_days=10.0), RecencyReranker)
    assert isinstance(build_reranker("authority", weight=0.2), AuthorityReranker)
    assert isinstance(build_reranker("llm", provider=MockProvider(), model="m"), LLMReranker)
    assert isinstance(build_reranker("cross-encoder", fallback=True), LocalCrossEncoderReranker)
    assert isinstance(build_reranker("local"), LocalCrossEncoderReranker)
    assert isinstance(build_reranker("cross_encoder"), LocalCrossEncoderReranker)
    assert isinstance(build_reranker("cohere", api_key="k"), CohereReranker)
    assert isinstance(build_reranker("jina", api_key="k"), JinaReranker)


def test_build_reranker_recency_passes_kwargs():
    rr = build_reranker("recency", half_life_days=7.0)
    assert isinstance(rr, RecencyReranker)
    assert rr.half_life_days == 7.0


def test_build_reranker_unknown_raises_value_error():
    with pytest.raises(ValueError, match="unknown reranker 'bogus'"):
        build_reranker("bogus")


def test_register_reranker_makes_kind_buildable():
    sentinel = HeuristicReranker()

    def factory(**kwargs):
        return sentinel

    returned = register_reranker("cov_custom_rr", factory)
    try:
        assert returned is factory  # register returns the factory (decorator-friendly)
        assert build_reranker("cov_custom_rr") is sentinel
    finally:
        from vincio.retrieval import rerankers as _mod

        _mod._PLUGIN_RERANKERS.pop("cov_custom_rr", None)
