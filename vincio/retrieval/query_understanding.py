"""Query understanding strategies: HyDE, multi-query expansion,
decomposition, and step-back prompting.

Each strategy turns one user query into extra search queries that feed the
weighted-RRF fusion in :class:`~vincio.retrieval.engine.RetrievalEngine`
(``retrieve(..., strategies=[...])``). With a provider the expansions are
model-written; without one, deterministic heuristics keep every strategy
usable offline. The strategies used and the queries they produced are
recorded on the query plan and the retrieval result, so they show up in
traces.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from ..core.types import Message, ModelRequest
from ..providers.base import ModelProvider

__all__ = ["QueryExpansion", "QueryUnderstanding", "QUERY_STRATEGIES"]

QUERY_STRATEGIES = ("hyde", "multi_query", "decompose", "step_back")

_QUESTION_PREFIX_RE = re.compile(
    r"(?i)^(?:what|which|who|whom|whose|when|where|why|how)(?:\s+(?:is|are|was|were|do|does|did|can|could|will|would|should|much|many|long|far|often))?\s+"
)
_PROPER_NOUN_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
_NUMBER_RE = re.compile(r"\b\d[\d.,%]*\b")
_SPLIT_RE = re.compile(r"(?i)\b(?:and|as well as|plus|also|;|,\s+and)\b")
_STOPWORDS = frozenset(
    "a an the of for to in on at by with from is are was were be been do does did "
    "can could will would should what which who when where why how".split()
)


class QueryExpansion(BaseModel):
    strategy: str
    queries: list[str] = Field(default_factory=list)
    hypothetical: str | None = None  # HyDE pseudo-document


_EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}
_HYDE_SCHEMA = {
    "type": "object",
    "properties": {"passage": {"type": "string"}},
    "required": ["passage"],
    "additionalProperties": False,
}

_PROMPTS = {
    "multi_query": (
        "Rewrite the search query 3 different ways: different wording, same "
        "intent. Return only the rewrites."
    ),
    "decompose": (
        "Decompose the question into 2-4 self-contained subquestions that "
        "together cover every fact needed to answer it."
    ),
    "step_back": (
        "Write 1-2 broader 'step-back' questions about the general concepts "
        "or policies behind this specific question."
    ),
}


class QueryUnderstanding:
    """LLM-backed query strategies with deterministic offline fallbacks."""

    def __init__(self, provider: ModelProvider | None = None, model: str | None = None) -> None:
        self.provider = provider
        self.model = model

    # -- heuristic fallbacks ------------------------------------------------------

    @staticmethod
    def _keywords(query: str) -> str:
        words = [w for w in re.findall(r"[A-Za-z0-9-]+", query) if w.lower() not in _STOPWORDS]
        return " ".join(words)

    def _heuristic_hyde(self, query: str) -> str:
        # Turn the question into the opening of its own answer so the
        # pseudo-document matches answer-shaped passages.
        statement = _QUESTION_PREFIX_RE.sub("", query.strip()).strip(" ?.")
        if not statement:
            return query
        if statement.lower().startswith(("the ", "a ", "an ")):
            return f"{statement[0].upper()}{statement[1:]} is"
        return f"The {statement} is"

    def _heuristic_multi_query(self, query: str) -> list[str]:
        keywords = self._keywords(query)
        variants = [keywords, f"{keywords} policy details", query.strip(" ?") + " explained"]
        return [v for v in dict.fromkeys(v.strip() for v in variants) if v and v.lower() != query.lower()][:3]

    def _heuristic_decompose(self, query: str) -> list[str]:
        parts = [p.strip(" ?.") for p in _SPLIT_RE.split(query) if p.strip(" ?.")]
        return [p for p in parts if len(p.split()) >= 2][:4] if len(parts) > 1 else []

    def _heuristic_step_back(self, query: str) -> list[str]:
        # Strip proper nouns (except a sentence-initial question word) and
        # numbers so the question generalizes past its specific entities.
        without_nouns = _PROPER_NOUN_RE.sub(
            lambda m: m.group(0) if m.start() == 0 else "", query
        )
        generic = _NUMBER_RE.sub("", without_nouns).strip(" ?.")
        generic = re.sub(r"\s{2,}", " ", generic)
        if len(generic.split()) >= 3 and generic.lower() != query.lower().strip(" ?."):
            return [generic]
        keywords = self._keywords(query)
        return [f"general rules about {keywords}"] if keywords else []

    # -- LLM strategies ------------------------------------------------------------

    async def _llm_queries(self, strategy: str, query: str, objective: str) -> list[str]:
        request = ModelRequest(
            model=self.model or "",
            messages=[
                Message(role="system", content=_PROMPTS[strategy]),
                Message(role="user", content=f"Objective: {objective}\nQuery: {query}"),
            ],
            output_schema=_EXPANSION_SCHEMA,
            output_schema_name=f"{strategy}_queries",
            temperature=0.0,
        )
        response = await self.provider.generate(request)  # type: ignore[union-attr]
        payload = response.structured or json.loads(response.text)
        return [str(q).strip() for q in payload.get("queries", []) if str(q).strip()][:4]

    async def _llm_hyde(self, query: str, objective: str) -> str:
        request = ModelRequest(
            model=self.model or "",
            messages=[
                Message(
                    role="system",
                    content=(
                        "Write a short factual passage (2-3 sentences) that would "
                        "plausibly answer the query, as it might appear in a document. "
                        "It is a search probe, not an answer shown to anyone."
                    ),
                ),
                Message(role="user", content=f"Objective: {objective}\nQuery: {query}"),
            ],
            output_schema=_HYDE_SCHEMA,
            output_schema_name="hyde_passage",
            temperature=0.0,
        )
        response = await self.provider.generate(request)  # type: ignore[union-attr]
        payload = response.structured or json.loads(response.text)
        return str(payload.get("passage", "")).strip()

    # -- public API -----------------------------------------------------------------

    async def expand(
        self, query: str, strategies: list[str], *, objective: str = ""
    ) -> list[QueryExpansion]:
        expansions: list[QueryExpansion] = []
        for strategy in strategies:
            if strategy not in QUERY_STRATEGIES:
                raise ValueError(
                    f"unknown query strategy {strategy!r}; known: {sorted(QUERY_STRATEGIES)}"
                )
            use_llm = self.provider is not None and self.model is not None
            if strategy == "hyde":
                hypothetical = ""
                if use_llm:
                    try:
                        hypothetical = await self._llm_hyde(query, objective)
                    except Exception:  # noqa: BLE001 - strategy falls back to heuristics
                        hypothetical = ""
                hypothetical = hypothetical or self._heuristic_hyde(query)
                expansions.append(
                    QueryExpansion(strategy=strategy, queries=[hypothetical], hypothetical=hypothetical)
                )
                continue
            queries: list[str] = []
            if use_llm:
                try:
                    queries = await self._llm_queries(strategy, query, objective)
                except Exception:  # noqa: BLE001 - strategy falls back to heuristics
                    queries = []
            if not queries:
                fallback = {
                    "multi_query": self._heuristic_multi_query,
                    "decompose": self._heuristic_decompose,
                    "step_back": self._heuristic_step_back,
                }[strategy]
                queries = fallback(query)
            expansions.append(QueryExpansion(strategy=strategy, queries=queries))
        return expansions
