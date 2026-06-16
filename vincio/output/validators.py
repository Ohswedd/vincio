"""Output validation pipeline.

parse → schema_validate → semantic_validate → citation_validate →
policy_validate → repair_if_allowed → final_validate

Every step records its outcome in a :class:`ValidationReport`; repairs only
run within the boundaries of the :class:`RepairPolicy`.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import OutputParseError, OutputSchemaError
from ..security.policy import PolicyEngine
from .parsers import extract_citations, extract_json
from .repair import Repairer
from .schemas import OutputContract, OutputSchema

__all__ = ["ValidationStep", "ValidationReport", "SemanticValidator", "OutputValidator"]

# Semantic validators: (data, context) -> error string or None.
SemanticValidator = Callable[[Any, dict[str, Any]], str | None] | Callable[
    [Any, dict[str, Any]], Awaitable[str | None]
]


class ValidationStep(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    repaired: bool = False


class ValidationReport(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    valid: bool = False
    output: Any = None
    raw_text: str = ""
    steps: list[ValidationStep] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    repair_actions: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def step(self, name: str, passed: bool, detail: str = "", repaired: bool = False) -> None:
        self.steps.append(ValidationStep(name=name, passed=passed, detail=detail, repaired=repaired))
        if not passed and detail:
            self.errors.append(f"{name}: {detail}")


class OutputValidator:
    def __init__(
        self,
        contract: OutputContract,
        *,
        schema: OutputSchema | None = None,
        semantic_validators: dict[str, SemanticValidator] | None = None,
        policy_engine: PolicyEngine | None = None,
        repairer: Repairer | None = None,
    ) -> None:
        self.contract = contract
        self.schema = schema or contract.output_schema()
        self.semantic_validators = semantic_validators or {}
        self.policy_engine = policy_engine
        self.repairer = repairer or Repairer(contract.repair_policy)

    async def validate(
        self,
        raw_text: str,
        *,
        structured: Any | None = None,
        evidence_ids: set[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ValidationReport:
        context = context or {}
        report = ValidationReport(raw_text=raw_text)
        data: Any = structured

        # 1. parse
        if self.schema is not None or self.contract.format == "json":
            if data is None:
                try:
                    data = extract_json(raw_text)
                    report.step("parse", True)
                except OutputParseError as exc:
                    if self.contract.repair_policy.allow_json_repair:
                        try:
                            outcome = self.repairer.repair_parse(raw_text)
                            data = outcome.data
                            report.repair_actions.extend(outcome.actions)
                            report.step("parse", True, "repaired", repaired=True)
                        except Exception as repair_exc:  # noqa: BLE001
                            report.step("parse", False, str(repair_exc))
                            return report
                    else:
                        report.step("parse", False, str(exc))
                        return report
            else:
                report.step("parse", True, "provider-native structured output")
        else:
            data = raw_text
            report.step("parse", True, "text output")

        # 2. schema validation (+ allowed structural repair)
        if self.schema is not None:
            try:
                validated = self.schema.validate(data)
                report.step("schema", True)
                data = validated
            except OutputSchemaError as exc:
                outcome = self.repairer.repair_structure(data, self.schema)
                if outcome.repaired:
                    report.repair_actions.extend(outcome.actions)
                try:
                    validated = self.schema.validate(outcome.data)
                    report.step("schema", True, "repaired", repaired=outcome.repaired)
                    data = validated
                except OutputSchemaError as final_exc:
                    if (
                        self.contract.repair_policy.allow_llm_repair
                        and self.repairer.provider is not None
                    ):
                        try:
                            llm_outcome = await self.repairer.repair_with_model(raw_text, self.schema)
                            data = self.schema.validate(llm_outcome.data)
                            report.repair_actions.extend(llm_outcome.actions)
                            report.step("schema", True, "model repaired", repaired=True)
                        except Exception as llm_exc:  # noqa: BLE001
                            report.step("schema", False, f"{final_exc.message}; llm repair failed: {llm_exc}")
                            return report
                    else:
                        report.step("schema", False, str(final_exc.errors or exc.message))
                        return report

        # 3. semantic validators
        for spec in self.contract.validators:
            validator = self.semantic_validators.get(spec.name)
            if validator is None:
                report.step(f"semantic:{spec.name}", False, "validator not registered")
                if spec.blocking:
                    return report
                continue
            result = validator(data, {**context, **spec.params})
            if inspect.isawaitable(result):
                result = await result
            if result:
                report.step(f"semantic:{spec.name}", False, str(result))
                if spec.blocking:
                    return report
            else:
                report.step(f"semantic:{spec.name}", True)

        # 4. citation validation
        citation_text = raw_text
        if not isinstance(data, str):
            import json as json_module

            try:
                citation_text = raw_text + "\n" + json_module.dumps(
                    data.model_dump(mode="json") if hasattr(data, "model_dump") else data
                )
            except (TypeError, ValueError):
                pass
        citations = extract_citations(citation_text, valid_ids=None)
        if evidence_ids is not None:
            valid_citations = [c for c in citations if c in evidence_ids]
            invalid_citations = [c for c in citations if c not in evidence_ids]
            report.citations = valid_citations
            if self.contract.require_citations:
                if not valid_citations:
                    report.step(
                        "citations",
                        False,
                        "no valid citations found"
                        + (f"; invalid refs: {invalid_citations}" if invalid_citations else ""),
                    )
                    return report
                report.step(
                    "citations",
                    True,
                    f"{len(valid_citations)} valid"
                    + (f", {len(invalid_citations)} invalid" if invalid_citations else ""),
                )
            else:
                report.step("citations", True, f"{len(valid_citations)} found")
        else:
            report.citations = citations
            if self.contract.require_citations and not citations:
                report.step("citations", False, "citations required but none found")
                return report
            report.step("citations", True, f"{len(citations)} found")

        # 5. policy validation (includes programmable output rails)
        if self.policy_engine is not None:
            policy_result = self.policy_engine.check_output(
                raw_text, citations_present=bool(report.citations)
            )
            if not policy_result.allowed:
                report.step(
                    "policy", False, "; ".join(v.message for v in policy_result.blocking)
                )
                return report
            detail = ""
            if policy_result.violations:
                detail = "; ".join(v.message for v in policy_result.violations)
            if policy_result.transformed_text is not None and isinstance(data, str):
                # A redact rail fired on a text output: ship the redacted text.
                data = policy_result.transformed_text
                report.repair_actions.append("output redacted by rail")
            report.step("policy", True, detail)

        # 6-7. final
        report.valid = True
        report.output = data
        return report
