"""Judges: deterministic, model-based, embedding-based, hybrid, G-Eval.

Model judges are calibrated by repeated scoring (mitigation):
``samples > 1`` averages multiple judgments at temperature > 0.
``GEvalJudge`` adds rubric-driven evaluation with auto-generated evaluation
steps and an optional calibration fit against human-labelled scores.
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

__all__ = [
    "Judge",
    "DeterministicJudge",
    "ModelJudge",
    "EmbeddingJudge",
    "HybridJudge",
    "GEvalJudge",
]


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


_GEVAL_STEPS_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 8},
    },
    "required": ["steps"],
    "additionalProperties": False,
}

_GEVAL_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


class GEvalJudge(Judge):
    """Rubric-based G-Eval judge.

    Given plain-language ``criteria``, the judge derives explicit evaluation
    steps once (chain-of-thought, cached for the judge's lifetime), then
    scores each output on a 1–5 form-filling scale. ``samples > 1``
    approximates G-Eval's probability-weighted scoring by averaging repeated
    judgments at temperature > 0. ``calibrate()`` fits a linear correction
    against human-labelled scores and reports the fit quality.
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str,
        criteria: str,
        steps: list[str] | None = None,
        name: str = "g_eval",
        samples: int = 1,
        temperature: float = 0.0,
        threshold: float = 0.5,
        include_evidence: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model
        self.criteria = criteria
        self.steps = list(steps) if steps else None
        self.name = name
        self.samples = max(1, samples)
        self.temperature = temperature
        self.threshold = threshold
        self.include_evidence = include_evidence
        self._calibration: tuple[float, float] | None = None  # (scale, offset)

    async def _ensure_steps(self) -> list[str]:
        if self.steps:
            return self.steps
        request = ModelRequest(
            model=self.model,
            messages=[
                Message(
                    role="system",
                    content=(
                        "You design rigorous LLM evaluation procedures. Given evaluation "
                        "criteria, produce 3-6 concrete, ordered evaluation steps an "
                        "evaluator should follow. Steps must be checkable against the "
                        "output alone."
                    ),
                ),
                Message(role="user", content=f"Evaluation criteria:\n{self.criteria}"),
            ],
            output_schema=_GEVAL_STEPS_SCHEMA,
            output_schema_name="evaluation_steps",
            temperature=0.0,
        )
        response = await self.provider.generate(request)
        payload = response.structured
        if payload is None:
            try:
                payload = json.loads(response.text)
            except json.JSONDecodeError:
                payload = {}
        steps = [str(s) for s in payload.get("steps", []) if str(s).strip()]
        self.steps = steps or [
            "Read the task input and the system output.",
            f"Check the output against the criteria: {self.criteria}",
            "Assign a 1-5 score where 5 fully satisfies the criteria.",
        ]
        return self.steps

    def _request(self, case: EvalCase, output: RunOutput, steps: list[str]) -> ModelRequest:
        numbered = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))
        parts = [f"Task input:\n{case.input_text}"]
        if case.expected is not None:
            parts.append(f"Reference answer:\n{json.dumps(case.expected, default=str)[:4000]}")
        if self.include_evidence and output.evidence:
            evidence = "\n".join(f"[{e.citation_ref}] {e.text}" for e in output.evidence[:16] if e.text)
            parts.append(f"Context available to the system:\n{evidence[:6000]}")
        parts.append(f"System output to judge:\n{output.output_text[:8000]}")
        parts.append("Follow the evaluation steps, then give a 1-5 score (5 = best).")
        return ModelRequest(
            model=self.model,
            messages=[
                Message(
                    role="system",
                    content=(
                        f"You are a strict evaluator.\nCriteria: {self.criteria}\n"
                        f"Evaluation steps:\n{numbered}"
                    ),
                ),
                Message(role="user", content="\n\n".join(parts)),
            ],
            output_schema=_GEVAL_SCORE_SCHEMA,
            output_schema_name="g_eval_score",
            temperature=self.temperature if self.samples == 1 else max(self.temperature, 0.4),
        )

    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult:
        steps = await self._ensure_steps()
        request = self._request(case, output, steps)
        responses = await asyncio.gather(
            *(self.provider.generate(request) for _ in range(self.samples))
        )
        scores: list[float] = []
        reasoning = ""
        for response in responses:
            payload = response.structured
            if payload is None:
                try:
                    payload = json.loads(response.text)
                except json.JSONDecodeError:
                    continue
            raw = payload.get("score")
            if raw is None:
                continue
            scores.append((max(1.0, min(5.0, float(raw))) - 1.0) / 4.0)
            reasoning = payload.get("reasoning", reasoning)
        if not scores:
            return MetricResult(
                name=self.name, value=0.0,
                details={"error": "judge returned no parseable score"},
            )
        value = sum(scores) / len(scores)
        if self._calibration is not None:
            scale, offset = self._calibration
            value = max(0.0, min(1.0, scale * value + offset))
        return MetricResult(
            name=self.name,
            value=round(value, 4),
            passed=value >= self.threshold,
            details={
                "samples": len(scores),
                "spread": round(max(scores) - min(scores), 4) if len(scores) > 1 else 0.0,
                "steps": steps,
                "reasoning": reasoning[:500],
                "calibrated": self._calibration is not None,
            },
        )

    def calibrate(self, pairs: list[tuple[float, float]]) -> dict[str, float]:
        """Fit a linear correction from (judge_score, human_score) pairs.

        Stores ``human ≈ scale * judge + offset`` and applies it to future
        scores. Returns the fit: scale, offset, and Pearson correlation.
        """
        if len(pairs) < 2:
            raise ValueError("calibration requires at least 2 (judge, human) pairs")
        xs = [float(j) for j, _ in pairs]
        ys = [float(h) for _, h in pairs]
        n = len(pairs)
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        var_x = sum((x - mean_x) ** 2 for x in xs)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        scale = cov / var_x if var_x else 1.0
        offset = mean_y - scale * mean_x
        var_y = sum((y - mean_y) ** 2 for y in ys)
        pearson = cov / ((var_x * var_y) ** 0.5) if var_x and var_y else 0.0
        self._calibration = (scale, offset)
        return {"scale": round(scale, 4), "offset": round(offset, 4), "pearson_r": round(pearson, 4), "n": n}


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
