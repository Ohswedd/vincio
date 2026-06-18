"""Rerankers: heuristic, recency, authority, LLM, and a
cross-encoder hook."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import httpx

from ..context.scoring import lexical_similarity
from ..core.errors import ProviderAuthError, ProviderError
from ..core.types import Message, ModelRequest
from ..core.utils import utcnow
from ..providers.base import ModelProvider
from .indexes import SearchHit

__all__ = [
    "Reranker",
    "HeuristicReranker",
    "RecencyReranker",
    "AuthorityReranker",
    "LLMReranker",
    "CrossEncoderReranker",
    "LocalCrossEncoderReranker",
    "HTTPReranker",
    "CohereReranker",
    "JinaReranker",
    "VoyageReranker",
    "build_reranker",
]


class Reranker(Protocol):
    async def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        ...  # pragma: no cover


class HeuristicReranker:
    """Blends retrieval score, exact-phrase/lexical match, term coverage,
    and structural priors (titles/tables score for lookup-style queries)."""

    def __init__(self, *, weight_retrieval: float = 0.5, weight_lexical: float = 0.35, weight_structure: float = 0.15) -> None:
        self.weight_retrieval = weight_retrieval
        self.weight_lexical = weight_lexical
        self.weight_structure = weight_structure

    async def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        max_score = max(h.score for h in hits) or 1.0
        phrase = query.lower().strip()
        rescored: list[SearchHit] = []
        for hit in hits:
            text_lower = hit.chunk.text.lower()
            lexical = lexical_similarity(hit.chunk.text, query)
            if phrase and phrase in text_lower:
                lexical = min(1.0, lexical + 0.4)
            structure = 0.0
            if hit.chunk.kind == "table" and re.search(r"(?i)\b(price|rate|fee|amount|how much|cost|table)\b", query):
                structure = 1.0
            elif hit.chunk.section_path and any(
                lexical_similarity(" ".join(hit.chunk.section_path), query) > 0.3 for _ in (0,)
            ):
                structure = 0.6
            blended = (
                self.weight_retrieval * (hit.score / max_score)
                + self.weight_lexical * lexical
                + self.weight_structure * structure
            )
            rescored.append(SearchHit(chunk=hit.chunk, score=blended, source=hit.source))
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]


class RecencyReranker:
    """Decays scores by chunk age (half-life in days)."""

    def __init__(self, *, half_life_days: float = 90.0) -> None:
        self.half_life_days = half_life_days

    async def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        now = utcnow()
        rescored = []
        for hit in hits:
            created = hit.chunk.created_at
            decay = 1.0
            if created is not None:
                if created.tzinfo is None:
                    from datetime import UTC

                    created = created.replace(tzinfo=UTC)
                age_days = max(0.0, (now - created).total_seconds() / 86_400)
                decay = 0.5 ** (age_days / self.half_life_days)
            rescored.append(SearchHit(chunk=hit.chunk, score=hit.score * decay, source=hit.source))
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]


class AuthorityReranker:
    """Boosts chunks whose metadata carries an authority score (set at
    ingestion: official docs > wikis > chat logs)."""

    def __init__(self, *, weight: float = 0.4) -> None:
        self.weight = weight

    async def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        max_score = max(h.score for h in hits) or 1.0
        rescored = []
        for hit in hits:
            authority = float(hit.chunk.metadata.get("authority", 0.5))
            blended = (1 - self.weight) * (hit.score / max_score) + self.weight * authority
            rescored.append(SearchHit(chunk=hit.chunk, score=blended, source=hit.source))
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]


_RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "relevance": {"type": "number"},
                },
                "required": ["id", "relevance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


class LLMReranker:
    """Scores passages with a (cheap) model in one batched call."""

    def __init__(self, provider: ModelProvider, *, model: str, max_passage_chars: int = 600) -> None:
        self.provider = provider
        self.model = model
        self.max_passage_chars = max_passage_chars

    async def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        passages = "\n\n".join(
            f"[{index}] {hit.chunk.text[: self.max_passage_chars]}" for index, hit in enumerate(hits)
        )
        request = ModelRequest(
            model=self.model,
            messages=[
                Message(
                    role="system",
                    content=(
                        "Score each passage's relevance to the query from 0.0 to 1.0. "
                        "Relevant = contains information that answers the query."
                    ),
                ),
                Message(role="user", content=f"Query: {query}\n\nPassages:\n{passages}"),
            ],
            output_schema=_RERANK_SCHEMA,
            output_schema_name="rerank_scores",
            temperature=0.0,
        )
        response = await self.provider.generate(request)
        payload = response.structured or {}
        if not payload and response.text:
            try:
                payload = json.loads(response.text)
            except json.JSONDecodeError:
                payload = {}
        score_map = {
            int(entry["id"]): float(entry["relevance"])
            for entry in payload.get("scores", [])
            if isinstance(entry, dict) and "id" in entry and "relevance" in entry
        }
        rescored = [
            SearchHit(chunk=hit.chunk, score=score_map.get(index, hit.score * 0.01), source=hit.source)
            for index, hit in enumerate(hits)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]


CrossEncoderFn = Callable[[str, list[str]], Awaitable[list[float]]]


class CrossEncoderReranker:
    """Adapter for an external cross-encoder: pass an async callable that
    scores (query, passages) → relevance list (e.g. a sentence-transformers
    model served behind a function)."""

    def __init__(self, score_fn: CrossEncoderFn) -> None:
        self.score_fn = score_fn

    async def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        scores = await self.score_fn(query, [hit.chunk.text for hit in hits])
        rescored = [
            SearchHit(chunk=hit.chunk, score=score, source=hit.source)
            for hit, score in zip(hits, scores, strict=False)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]


class LocalCrossEncoderReranker:
    """Local cross-encoder reranker via ``sentence-transformers``.

    Batteries-included on-device reranking with real cross-encoder quality and
    no server: lazily loads a ``CrossEncoder`` model and scores each
    (query, passage) pair. Injectable via ``score_fn`` for a fully custom scorer
    or ``model`` for a ``CrossEncoder``-shaped object (offline tests drive the
    real ``model.predict`` path against a faithful fake); with ``fallback=True``
    it degrades to :class:`HeuristicReranker` when the dependency is missing.
    Install with ``pip install "vincio[cross-encoder]"``.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        score_fn: CrossEncoderFn | None = None,
        model: Any = None,
        fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self._score_fn = score_fn
        self._fallback = fallback
        self._model: Any = model
        self._fallback_reranker: HeuristicReranker | None = None

    def _ensure(self) -> None:
        if (
            self._score_fn is not None
            or self._model is not None
            or self._fallback_reranker is not None
        ):
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

            self._model = CrossEncoder(self.model_name)
        except ImportError as exc:
            if self._fallback:
                self._fallback_reranker = HeuristicReranker()
                return
            from ..core.errors import ConfigError

            raise ConfigError(
                'the local cross-encoder reranker requires: pip install '
                '"vincio[cross-encoder]" (or construct with fallback=True / inject score_fn)'
            ) from exc

    async def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        self._ensure()
        if self._fallback_reranker is not None:
            return await self._fallback_reranker.rerank(query, hits, top_k=top_k)
        passages = [hit.chunk.text for hit in hits]
        if self._score_fn is not None:
            scores = await self._score_fn(query, passages)
        else:
            import asyncio

            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                None, lambda: [float(s) for s in self._model.predict([(query, p) for p in passages])]
            )
        rescored = [
            SearchHit(chunk=hit.chunk, score=score, source=hit.source)
            for hit, score in zip(hits, scores, strict=False)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]


class HTTPReranker:
    """Hosted cross-encoder reranker over a JSON rerank endpoint (httpx only).

    Subclasses set the endpoint path, the default model, and how to read the
    per-document relevance scores back out. An ``httpx.AsyncClient`` can be
    injected for offline testing (the established pattern across Vincio's HTTP
    adapters); otherwise one is created and closed per call.
    """

    name = "http"
    default_base_url = ""
    default_model = ""
    results_key = "results"  # response key holding the scored items
    score_key = "relevance_score"
    top_param = "top_n"  # request field naming the truncation count

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model or self.default_model
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self._client = client
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderAuthError(f"missing API key for reranker {self.name!r}", provider=self.name)
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, query: str, documents: list[str], top_k: int) -> dict[str, Any]:
        return {
            "model": self.model,
            "query": query,
            "documents": documents,
            self.top_param: top_k,
        }

    def _parse(self, data: dict[str, Any]) -> list[tuple[int, float]]:
        out: list[tuple[int, float]] = []
        for item in data.get(self.results_key) or []:
            if isinstance(item, dict) and "index" in item:
                out.append((int(item["index"]), float(item.get(self.score_key, 0.0))))
        return out

    async def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        documents = [hit.chunk.text for hit in hits]
        payload = self._payload(query, documents, top_k)
        client = self._client or httpx.AsyncClient(timeout=self.timeout_s)
        try:
            response = await client.post(self.base_url, json=payload, headers=self._headers())
            if response.status_code >= 400:
                raise ProviderError(
                    f"reranker {self.name!r} error {response.status_code}: {response.text[:500]}",
                    provider=self.name,
                )
            scored = self._parse(response.json())
        finally:
            if self._client is None:
                await client.aclose()
        # Sort defensively (don't assume the endpoint pre-sorted); documents it
        # dropped fall to the back, preserving original order.
        scored.sort(key=lambda pair: pair[1], reverse=True)
        ranked = [SearchHit(chunk=hits[i].chunk, score=s, source=self.name) for i, s in scored]
        seen = {i for i, _ in scored}
        tail = [
            SearchHit(chunk=hit.chunk, score=hit.score * 0.0, source=self.name)
            for index, hit in enumerate(hits)
            if index not in seen
        ]
        return (ranked + tail)[:top_k]


class CohereReranker(HTTPReranker):
    """Cohere Rerank (``/v2/rerank``). Needs a Cohere API key; httpx only."""

    name = "cohere"
    default_base_url = "https://api.cohere.com/v2/rerank"
    default_model = "rerank-v3.5"


class JinaReranker(HTTPReranker):
    """Jina AI Reranker (``/v1/rerank``). Needs a Jina API key; httpx only."""

    name = "jina"
    default_base_url = "https://api.jina.ai/v1/rerank"
    default_model = "jina-reranker-v2-base-multilingual"


class VoyageReranker(HTTPReranker):
    """Voyage AI Reranker (``/v1/rerank``). Needs a Voyage API key; httpx only."""

    name = "voyage"
    default_base_url = "https://api.voyageai.com/v1/rerank"
    default_model = "rerank-2"
    results_key = "data"
    top_param = "top_k"


_HTTP_RERANKERS: dict[str, type[HTTPReranker]] = {
    "cohere": CohereReranker,
    "jina": JinaReranker,
    "voyage": VoyageReranker,
}


def build_reranker(kind: str | None, **kwargs) -> Reranker | None:
    if kind in (None, "none"):
        return None
    if kind == "heuristic":
        return HeuristicReranker()
    if kind == "recency":
        return RecencyReranker(**kwargs)
    if kind == "authority":
        return AuthorityReranker(**kwargs)
    if kind == "llm":
        return LLMReranker(**kwargs)
    if kind in ("local", "cross-encoder", "cross_encoder"):
        return LocalCrossEncoderReranker(**kwargs)
    if kind in _HTTP_RERANKERS:
        return _HTTP_RERANKERS[kind](**kwargs)
    raise ValueError(f"unknown reranker {kind!r}")
