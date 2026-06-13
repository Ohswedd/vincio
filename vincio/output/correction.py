"""Self-correcting output loops (0.7).

Bounded validate → critique → repair cycles with hard cost ceilings. The
critique is deterministic — it is built from the :class:`ValidationReport`,
not from model judgment — and the repair request reuses the structure-only
contract: the model may re-serialize, rename, and re-type, but the prompt
forbids adding, removing, or changing factual content. Semantic, citation,
and policy validators re-run on every cycle, so an "improved" output that
changed the facts still fails.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from ..core.types import Message, ModelRequest
from ..providers.base import ModelProvider
from .schemas import OutputSchema
from .validators import OutputValidator, ValidationReport

__all__ = ["CorrectionResult", "SelfCorrector", "build_critique"]


def build_critique(report: ValidationReport) -> str:
    """Deterministic critique of a failed validation, for the repair request."""
    lines = ["The previous output failed validation:"]
    for step in report.steps:
        if not step.passed:
            lines.append(f"- {step.name}: {step.detail or 'failed'}")
    for error in report.errors:
        line = f"- {error}"
        if line not in lines:
            lines.append(line)
    lines.append(
        "Fix ONLY the structure, field names, types, and formatting. "
        "Do not add, remove, or change any factual content, claims, numbers, "
        "or citations."
    )
    return "\n".join(lines)


class CorrectionResult(BaseModel):
    """Outcome of a self-correction loop."""

    model_config = {"arbitrary_types_allowed": True}

    valid: bool = False
    output: Any = None
    raw_text: str = ""
    report: ValidationReport | None = None
    cycles: int = 0
    cost_usd: float = 0.0
    stopped_reason: str = ""  # valid | max_cycles | cost_ceiling | no_provider
    critiques: list[str] = Field(default_factory=list)


_CORRECTION_SYSTEM_PROMPT = (
    "You repair structured output that failed validation. Re-serialize the "
    "output so it passes, following the critique. You may fix JSON syntax, "
    "field names, types, and formatting ONLY. Never add, remove, or change "
    "factual content, claims, numbers, or citations. Output JSON only."
)


class SelfCorrector:
    """Bounded validate → critique → repair cycles with a cost ceiling.

    Each cycle validates the current output through the full pipeline
    (schema, semantic validators, citations, policy), derives a
    deterministic critique from the failures, and asks the model for a
    structure-only fix. The loop stops at the first valid output, at
    ``max_cycles``, or when ``max_cost_usd`` would be exceeded.
    """

    def __init__(
        self,
        validator: OutputValidator,
        *,
        provider: ModelProvider | None = None,
        model: str | None = None,
        max_cycles: int = 2,
        max_cost_usd: float = 0.05,
        temperature: float = 0.0,
    ) -> None:
        self.validator = validator
        self.provider = provider
        self.model = model
        self.max_cycles = max_cycles
        self.max_cost_usd = max_cost_usd
        self.temperature = temperature

    async def correct(
        self,
        raw_text: str,
        *,
        structured: Any | None = None,
        evidence_ids: set[str] | None = None,
        context: dict[str, Any] | None = None,
        initial_report: ValidationReport | None = None,
    ) -> CorrectionResult:
        result = CorrectionResult(raw_text=raw_text)
        report = initial_report or await self.validator.validate(
            raw_text, structured=structured, evidence_ids=evidence_ids, context=context
        )
        result.report = report
        if report.valid:
            result.valid = True
            result.output = report.output
            result.stopped_reason = "valid"
            return result
        if self.provider is None or self.model is None:
            result.stopped_reason = "no_provider"
            return result

        schema = self.validator.schema
        current_text = raw_text
        for cycle in range(1, self.max_cycles + 1):
            critique = build_critique(report)
            result.critiques.append(critique)
            response = await self.provider.generate(
                self._repair_request(current_text, critique, schema)
            )
            if result.cost_usd + response.cost_usd > self.max_cost_usd and response.cost_usd > 0:
                result.cost_usd += response.cost_usd
                result.cycles = cycle
                result.stopped_reason = "cost_ceiling"
                return result
            result.cost_usd += response.cost_usd
            result.cycles = cycle
            current_text = response.text
            report = await self.validator.validate(
                current_text,
                structured=response.structured,
                evidence_ids=evidence_ids,
                context=context,
            )
            result.report = report
            result.raw_text = current_text
            if report.valid:
                result.valid = True
                result.output = report.output
                result.stopped_reason = "valid"
                return result
            if result.cost_usd >= self.max_cost_usd:
                result.stopped_reason = "cost_ceiling"
                return result
        result.stopped_reason = "max_cycles"
        return result

    def _repair_request(
        self, raw_text: str, critique: str, schema: OutputSchema | None
    ) -> ModelRequest:
        schema_text = (
            f"Schema:\n{json.dumps(schema.json_schema)}\n\n" if schema is not None else ""
        )
        return ModelRequest(
            model=self.model or "",
            messages=[
                Message(role="system", content=_CORRECTION_SYSTEM_PROMPT),
                Message(
                    role="user",
                    content=(
                        f"{critique}\n\n{schema_text}"
                        f"Output to fix:\n{raw_text[:12_000]}"
                    ),
                ),
            ],
            output_schema=schema.json_schema if schema is not None else None,
            output_schema_name=schema.name if schema is not None else None,
            temperature=self.temperature,
        )
