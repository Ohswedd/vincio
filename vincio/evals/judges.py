"""Judges: deterministic, model-based, embedding-based, hybrid.

Model judges are calibrated by repeated scoring (mitigation):
``samples > 1`` averages multiple judgments at temperature > 0.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

from ..core.types import Message, ModelRequest
from ..providers.base import ModelProvider
from ..retrieval.embeddings import Embedder, cosine
from .datasets import EvalCase
from .metrics import Metric, MetricResult, RunOutput

__all__ = ["Judge", "DeterministicJudge", "ModelJudge", "EmbeddingJudge", "HybridJudge"]


class Judge(ABC):
    name: str = "judge"

    @abstractmethod
    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult: ...


class DeterministicJudge(Judge):
    def __init__(self, metric: Metric, *, name: str | None = None) -> None:
        self.metric = metric
        self.name = name or getattr(metric, "__name__", "deterministic")

    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult:
        return self.metric(case, output)


_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
        "failures": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "reasoning", "failures"],
    "additionalProperties": False,
}

DEFAULT_RUBRIC = (
    "Score how well the output accomplishes the task: 1.0 = fully correct, "
    "complete, and grounded; 0.0 = wrong or unusable. Penalize unsupported "
    "claims, missing requirements, and format violations."
)


class ModelJudge(Judge):
    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str,
        rubric: str = DEFAULT_RUBRIC,
        name: str = "model_judge",
        samples: int = 1,
        temperature: float = 0.0,
        include_evidence: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model
        self.rubric = rubric
        self.name = name
        self.samples = max(1, samples)
        self.temperature = temperature
        self.include_evidence = include_evidence

    def _request(self, case: EvalCase, output: RunOutput) -> ModelRequest:
        parts = [f"Task input:\n{case.input_text}"]
        if case.expected is not None:
            parts.append(f"Reference answer:\n{json.dumps(case.expected, default=str)[:4000]}")
        if case.rubric:
            parts.append(f"Case rubric:\n{json.dumps(case.rubric, default=str)[:2000]}")
        if self.include_evidence and output.evidence:
            evidence = "\n".join(f"[{e.citation_ref}] {e.text}" for e in output.evidence[:16] if e.text)
            parts.append(f"Evidence available to the system:\n{evidence[:6000]}")
        parts.append(f"System output to judge:\n{output.output_text[:8000]}")
        return ModelRequest(
            model=self.model,
            messages=[
                Message(role="system", content=f"You are a strict evaluator. {self.rubric}"),
                Message(role="user", content="\n\n".join(parts)),
            ],
            output_schema=_JUDGE_SCHEMA,
            output_schema_name="judgment",
            temperature=self.temperature if self.samples == 1 else max(self.temperature, 0.4),
        )

    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult:
        request = self._request(case, output)
        responses = await asyncio.gather(
            *(self.provider.generate(request) for _ in range(self.samples))
        )
        scores: list[float] = []
        failures: list[str] = []
        reasoning = ""
        for response in responses:
            payload = response.structured
            if payload is None:
                try:
                    payload = json.loads(response.text)
                except json.JSONDecodeError:
                    continue
            scores.append(max(0.0, min(1.0, float(payload.get("score", 0.0)))))
            failures.extend(payload.get("failures", []))
            reasoning = payload.get("reasoning", reasoning)
        if not scores:
            return MetricResult(name=self.name, value=0.0, details={"error": "judge returned no parseable score"})
        value = sum(scores) / len(scores)
        return MetricResult(
            name=self.name,
            value=round(value, 4),
            details={
                "samples": len(scores),
                "spread": round(max(scores) - min(scores), 4) if len(scores) > 1 else 0.0,
                "reasoning": reasoning[:500],
                "failures": failures[:10],
            },
        )


class EmbeddingJudge(Judge):
    def __init__(self, embedder: Embedder, *, name: str = "embedding_similarity") -> None:
        self.embedder = embedder
        self.name = name

    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult:
        expected = case.expected if isinstance(case.expected, str) else json.dumps(case.expected, default=str)
        vectors = await self.embedder.embed([expected or "", output.output_text])
        value = max(0.0, cosine(vectors[0], vectors[1]))
        return MetricResult(name=self.name, value=round(value, 4))


class HybridJudge(Judge):
    """Weighted combination of judges (e.g. 0.5 deterministic + 0.5 model)."""

    def __init__(self, judges: list[tuple[Judge, float]], *, name: str = "hybrid") -> None:
        if not judges:
            raise ValueError("HybridJudge requires at least one judge")
        self.judges = judges
        self.name = name

    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult:
        results = await asyncio.gather(*(judge.score(case, output) for judge, _ in self.judges))
        total_weight = sum(weight for _, weight in self.judges) or 1.0
        value = sum(
            result.value * weight for result, (_, weight) in zip(results, self.judges, strict=False)
        ) / total_weight
        details: dict[str, Any] = {
            result.name: result.value for result in results
        }
        return MetricResult(name=self.name, value=round(value, 4), details=details)
