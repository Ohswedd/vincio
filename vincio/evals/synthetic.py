"""Synthetic eval data: bootstrap golden datasets from your own corpora.

``SyntheticGenerator`` turns documents/chunks/raw text into eval cases with
difficulty and coverage controls, and full provenance (every case records the
source it was derived from, and carries the source sentence as a rubric fact
so grounding metrics work out of the box). Offline it uses deterministic
templates; give it a provider and it writes natural questions with an LLM,
falling back to the templates on any failure.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from ..context.compression import split_sentences
from ..context.scoring import lexical_similarity
from ..core.types import Chunk, Document, Message, ModelRequest
from ..providers.base import run_sync
from .datasets import Dataset, EvalCase

__all__ = ["SyntheticGenerator"]

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")

_QA_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                },
                "required": ["question", "answer", "difficulty"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["questions"],
    "additionalProperties": False,
}


class _Source:
    """Normalized view over Document / Chunk / str inputs."""

    def __init__(self, source_id: str, text: str, origin: str) -> None:
        self.source_id = source_id
        self.text = text
        self.origin = origin
        self.sentences = [
            sentence for sentence in split_sentences(text) if len(sentence.split()) >= 5
        ]


class SyntheticGenerator:
    """Generate eval datasets from corpora with difficulty/coverage controls.

    - **Coverage**: sources are sampled round-robin so every document
      contributes before any document contributes twice; near-duplicate
      sentences are skipped.
    - **Difficulty**: ``easy`` asks for a stated fact, ``medium`` masks a
      concrete value (cloze), ``hard`` requires combining facts from two
      different sources (multi-hop).
    - **Provenance**: each case's metadata records the source ids and the
      generator; the source sentences ride in ``rubric['facts']``.
    """

    def __init__(
        self,
        *,
        provider: Any = None,
        model: str | None = None,
        seed: int = 42,
    ) -> None:
        self.provider = provider
        self.model = model
        self.seed = seed

    # -- public API --------------------------------------------------------------

    def generate(
        self,
        corpus: list[Document | Chunk | str],
        *,
        n: int = 20,
        name: str = "synthetic",
        difficulty_mix: dict[str, float] | None = None,
    ) -> Dataset:
        return run_sync(
            self.agenerate(corpus, n=n, name=name, difficulty_mix=difficulty_mix)
        )

    async def agenerate(
        self,
        corpus: list[Document | Chunk | str],
        *,
        n: int = 20,
        name: str = "synthetic",
        difficulty_mix: dict[str, float] | None = None,
    ) -> Dataset:
        sources = self._normalize(corpus)
        if not sources:
            return Dataset(name=name)
        mix = difficulty_mix or {"easy": 0.4, "medium": 0.4, "hard": 0.2}
        total_weight = sum(mix.values()) or 1.0
        counts = {
            difficulty: max(0, round(n * weight / total_weight))
            for difficulty, weight in mix.items()
        }
        while sum(counts.values()) < n:
            counts["medium"] = counts.get("medium", 0) + 1
        cases: list[EvalCase] = []
        generator = "offline"
        if self.provider is not None and self.model:
            cases = await self._generate_llm(sources, counts)
            if cases:
                generator = "llm"
        if not cases:
            cases = self._generate_offline(sources, counts)
        for index, case in enumerate(cases):
            case.id = f"{name}_{index:04d}"
        return Dataset(
            name=name,
            cases=cases[:n],
            metadata={
                "generator": generator,
                "sources": len(sources),
                "difficulty_mix": mix,
            },
        )

    # -- input normalization -------------------------------------------------------

    def _normalize(self, corpus: list[Document | Chunk | str]) -> list[_Source]:
        sources: list[_Source] = []
        for index, item in enumerate(corpus):
            if isinstance(item, Document):
                sources.append(_Source(item.id, item.text, item.title or item.source_uri or item.id))
            elif isinstance(item, Chunk):
                sources.append(_Source(item.id, item.text, item.citation_ref))
            else:
                sources.append(_Source(f"text_{index}", str(item), f"text_{index}"))
        return [source for source in sources if source.sentences]

    # -- offline deterministic generation -------------------------------------------

    def _pick_sentences(
        self, sources: list[_Source], count: int, rng: random.Random, *, used: list[str]
    ) -> list[tuple[_Source, str]]:
        """Round-robin over sources for coverage; skip near-duplicates."""
        picked: list[tuple[_Source, str]] = []
        cursors = {source.source_id: 0 for source in sources}
        order = list(sources)
        rng.shuffle(order)
        while len(picked) < count:
            progressed = False
            for source in order:
                if len(picked) >= count:
                    break
                cursor = cursors[source.source_id]
                while cursor < len(source.sentences):
                    sentence = source.sentences[cursor]
                    cursor += 1
                    if all(lexical_similarity(sentence, seen) < 0.6 for seen in used):
                        picked.append((source, sentence))
                        used.append(sentence)
                        progressed = True
                        break
                cursors[source.source_id] = cursor
            if not progressed:
                break
        return picked

    def _generate_offline(
        self, sources: list[_Source], counts: dict[str, int]
    ) -> list[EvalCase]:
        rng = random.Random(self.seed)
        used: list[str] = []
        cases: list[EvalCase] = []

        for source, sentence in self._pick_sentences(sources, counts.get("easy", 0), rng, used=used):
            topic = " ".join(sentence.split()[:6]).rstrip(".,;:")
            cases.append(
                self._case(
                    question=f"According to the source, what is stated about \"{topic}\"?",
                    answer=sentence,
                    difficulty="easy",
                    facts=[sentence],
                    sources=[source],
                )
            )

        for source, sentence in self._pick_sentences(sources, counts.get("medium", 0), rng, used=used):
            target = self._mask_target(sentence)
            if target is None:
                continue
            masked = sentence.replace(target, "_____", 1)
            cases.append(
                self._case(
                    question=f"Fill in the blank from the source: \"{masked}\"",
                    answer=target,
                    difficulty="medium",
                    facts=[sentence],
                    sources=[source],
                )
            )

        hard_picks = self._pick_sentences(sources, counts.get("hard", 0) * 2, rng, used=used)
        for first, second in zip(hard_picks[0::2], hard_picks[1::2], strict=False):
            source_a, sentence_a = first
            source_b, sentence_b = second
            topic_a = " ".join(sentence_a.split()[:5]).rstrip(".,;:")
            topic_b = " ".join(sentence_b.split()[:5]).rstrip(".,;:")
            cases.append(
                self._case(
                    question=(
                        f"Combining the sources: what do they state about \"{topic_a}\" "
                        f"and \"{topic_b}\"?"
                    ),
                    answer=f"{sentence_a} {sentence_b}",
                    difficulty="hard",
                    facts=[sentence_a, sentence_b],
                    sources=[source_a, source_b],
                    tags=["multi_hop"],
                )
            )
        return cases

    @staticmethod
    def _mask_target(sentence: str) -> str | None:
        number = _NUMBER_RE.search(sentence)
        if number:
            return number.group(0)
        words = _WORD_RE.findall(sentence)
        # Mask the longest content word in the second half of the sentence.
        tail = words[len(words) // 2 :]
        return max(tail, key=len) if tail else None

    def _case(
        self,
        *,
        question: str,
        answer: str,
        difficulty: str,
        facts: list[str],
        sources: list[_Source],
        tags: list[str] | None = None,
    ) -> EvalCase:
        return EvalCase(
            id="pending",
            input=question,
            expected=answer,
            difficulty=difficulty,
            tags=["synthetic", *(tags or [])],
            rubric={"facts": facts},
            context={"reference": [source.text for source in sources]},
            metadata={
                "generator": "vincio.synthetic",
                "source_ids": [source.source_id for source in sources],
                "origins": [source.origin for source in sources],
            },
        )

    # -- LLM-backed generation -------------------------------------------------------

    def _llm_request(self, source: _Source, per_source: int) -> ModelRequest:
        return ModelRequest(
            model=self.model or "",
            messages=[
                Message(
                    role="system",
                    content=(
                        "Write evaluation questions answerable strictly from the given "
                        "source. easy = stated fact, medium = specific value, hard = "
                        "requires combining facts. Answers must be verbatim-supported."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Source:\n{source.text[:6000]}\n\nWrite {per_source} question(s).",
                ),
            ],
            output_schema=_QA_SCHEMA,
            output_schema_name="synthetic_questions",
            temperature=0.3,
        )

    async def _generate_llm(
        self, sources: list[_Source], counts: dict[str, int]
    ) -> list[EvalCase]:
        """LLM-written questions, concurrent per source, honoring the
        difficulty mix: questions whose difficulty bucket is exhausted are
        skipped instead of consuming the budget."""
        from ..core.concurrency import map_bounded

        total = sum(counts.values())
        per_source = max(1, total // len(sources) + 1)

        async def ask(source: _Source) -> dict[str, Any]:
            response = await self.provider.generate(self._llm_request(source, per_source))
            return response.structured or json.loads(response.text)

        try:
            payloads = await map_bounded(ask, sources, limit=8)
        except Exception:  # noqa: BLE001 - fall back to offline templates
            return []
        cases: list[EvalCase] = []
        wanted = dict(counts)
        for source, payload in zip(sources, payloads, strict=True):
            for item in payload.get("questions", []):
                if len(cases) >= total or sum(wanted.values()) <= 0:
                    break
                difficulty = item.get("difficulty", "medium")
                if wanted.get(difficulty, 0) <= 0:
                    continue  # bucket exhausted — don't let it eat the budget
                wanted[difficulty] -= 1
                facts = [
                    sentence
                    for sentence in source.sentences
                    if lexical_similarity(sentence, str(item.get("answer", ""))) >= 0.3
                ][:3]
                cases.append(
                    self._case(
                        question=str(item.get("question", "")),
                        answer=str(item.get("answer", "")),
                        difficulty=difficulty,
                        facts=facts or source.sentences[:1],
                        sources=[source],
                        tags=["llm"],
                    )
                )
        return cases
