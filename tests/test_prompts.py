"""Prompt engine unit tests (prompt AST)."""

import pytest

from vincio.core.errors import PromptBudgetError, PromptError
from vincio.core.types import Example
from vincio.prompts import (
    CompilerOptions,
    PromptCompiler,
    PromptSpec,
    PromptVariable,
    diff_rendered,
    diff_specs,
    generate_variants,
    lint_spec,
)


@pytest.fixture()
def spec():
    return PromptSpec(
        name="claims",
        role="insurance_claim_decision_engine",
        objective="Determine whether a claim for plan ${plan} is reimbursable",
        rules=["Use only provided documents", "Cite evidence IDs for every claim"],
        definitions={"deductible": "amount paid before coverage applies"},
        examples=[
            Example(input="claim A", output="approved", quality=0.9),
            Example(input="claim B", output="denied", quality=0.5),
        ],
        output_schema={"type": "object", "properties": {"decision": {"type": "string"}}},
        output_format="json",
        citation_policy="Cite evidence IDs in square brackets.",
        insufficient_evidence_behavior="If evidence is missing, say so explicitly.",
        variables=[PromptVariable(name="plan", type="str")],
    )


class TestPromptSpec:
    def test_variable_substitution(self, spec):
        resolved = spec.substitute({"plan": "Gold"})
        assert "Gold" in resolved.objective

    def test_missing_variable_raises(self, spec):
        with pytest.raises(PromptError, match="plan"):
            spec.substitute({})

    def test_undeclared_variable_raises(self):
        bad = PromptSpec(objective="use ${undeclared}")
        with pytest.raises(PromptError, match="undeclared"):
            bad.substitute({})

    def test_variable_type_check(self, spec):
        with pytest.raises(PromptError, match="expected str"):
            spec.substitute({"plan": 42})

    def test_spec_hash_stable(self, spec):
        assert spec.spec_hash == spec.model_copy(deep=True).spec_hash


class TestPromptCompiler:
    def test_compile_markdown(self, spec):
        compiled = PromptCompiler().compile(
            spec,
            user_task="Is claim INV-9 reimbursable?",
            variables={"plan": "Gold"},
            evidence_items=[{"id": "E1", "text": "Policy covers water damage."}],
        )
        assert compiled.messages[0].role == "system"
        assert compiled.messages[0].cache_hint is True
        assert "[E1]" in compiled.user_text
        assert "Gold" in compiled.system_text
        assert compiled.cacheability > 0.5
        assert compiled.prompt_spec_hash and compiled.rendered_hash

    def test_compile_xml(self, spec):
        compiled = PromptCompiler(CompilerOptions(format="xml")).compile(
            spec, user_task="t", variables={"plan": "Gold"}
        )
        assert "<rules>" in compiled.system_text

    def test_compile_json_and_minimal(self, spec):
        for fmt in ("json", "minimal"):
            compiled = PromptCompiler(CompilerOptions(format=fmt)).compile(
                spec, user_task="t", variables={"plan": "Gold"}
            )
            assert compiled.system_text

    def test_dedupes_rules(self, spec):
        duplicated = spec.model_copy(
            update={"rules": ["Use only provided documents", "Use only provided documents"]}
        )
        compiled = PromptCompiler().compile(duplicated, user_task="t", variables={"plan": "x"})
        assert compiled.system_text.count("Use only provided documents") == 1

    def test_example_selection_by_quality(self, spec):
        compiled = PromptCompiler(CompilerOptions(max_examples=1)).compile(
            spec, user_task="t", variables={"plan": "x"}
        )
        assert compiled.excluded_examples == 1
        assert "claim A" in compiled.system_text
        assert "claim B" not in compiled.system_text

    def test_token_budget_enforced(self, spec):
        with pytest.raises(PromptBudgetError):
            PromptCompiler(CompilerOptions(max_prompt_tokens=10)).compile(
                spec, user_task="t", variables={"plan": "x"}
            )

    def test_schema_omitted_when_provider_enforces(self, spec):
        with_schema = PromptCompiler().compile(spec, user_task="t", variables={"plan": "x"})
        without = PromptCompiler().compile(
            spec, user_task="t", variables={"plan": "x"}, provider_enforces_schema=True
        )
        assert "JSON schema" in with_schema.system_text
        assert "JSON schema" not in without.system_text


class TestLint:
    def test_vague_role(self):
        findings = lint_spec(PromptSpec(role="helpful assistant", objective="x"))
        assert any(f.code == "PROMPT001" for f in findings)

    def test_duplicate_and_conflict(self):
        findings = lint_spec(
            PromptSpec(
                role="specific_engine",
                rules=[
                    "Always respond in English",
                    "Always respond in English",
                    "Never respond in English",
                ],
            )
        )
        codes = {f.code for f in findings}
        assert "PROMPT002" in codes
        assert "PROMPT003" in codes

    def test_grounded_task_warnings(self):
        findings = lint_spec(
            PromptSpec(role="engine", rules=["Use only provided documents"])
        )
        codes = {f.code for f in findings}
        assert "PROMPT004" in codes and "PROMPT007" in codes

    def test_schema_in_prose(self):
        findings = lint_spec(
            PromptSpec(
                role="engine",
                rules=["Respond with valid JSON only"],
                output_schema={"type": "object"},
            )
        )
        assert any(f.code == "PROMPT005" for f in findings)

    def test_excessive_examples(self):
        findings = lint_spec(
            PromptSpec(
                role="engine",
                examples=[Example(input=str(i), output="o") for i in range(9)],
            )
        )
        assert any(f.code == "PROMPT008" for f in findings)


class TestVariantsAndDiff:
    def test_generate_variants(self, spec):
        variants = generate_variants(spec, max_variants=6)
        assert len(variants) == 6
        assert all(v.dimensions for v in variants)

    def test_diff_specs(self, spec):
        other = spec.model_copy(update={"objective": "changed"})
        diff = diff_specs(spec, other)
        assert "objective" in diff["changed_fields"]
        assert diff["hash_a"] != diff["hash_b"]

    def test_diff_rendered(self):
        assert "+world" in diff_rendered("hello", "world")
