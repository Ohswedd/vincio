"""Output engine unit tests (schema validation)."""

import pytest
from pydantic import BaseModel

from vincio.output import (
    OutputContract,
    OutputSchema,
    OutputValidator,
    ValidatorSpec,
    extract_citations,
    extract_json,
    lenient_json_loads,
    parse_partial_json,
)


class Risk(BaseModel):
    clause: str
    risk_level: str
    evidence_ids: list[str]


class Report(BaseModel):
    summary: str
    risks: list[Risk]
    score: float


@pytest.fixture()
def contract():
    return OutputContract.from_schema(OutputSchema.from_pydantic(Report), require_citations=True)


class TestParsers:
    def test_extract_json_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_extract_json_fenced(self):
        assert extract_json('text\n```json\n{"a": 1}\n```\nmore') == {"a": 1}

    def test_extract_json_embedded(self):
        assert extract_json('The answer is {"a": [1, 2]} ok?') == {"a": [1, 2]}

    def test_lenient_trailing_comma_single_quotes(self):
        assert lenient_json_loads("{'a': True, 'b': None,}") == {"a": True, "b": None}

    def test_partial_json(self):
        value, complete = parse_partial_json('{"a": "x", "items": [{"b": 1')
        assert value == {"a": "x", "items": [{"b": 1}]}
        assert complete is False

    def test_extract_citations(self):
        citations = extract_citations("Fact one [E1]. Fact two [D1:C7]. Junk [not a ref!]")
        assert citations == ["E1", "D1:C7"]


class TestValidation:
    @pytest.mark.asyncio
    async def test_repair_and_typed_output(self, contract):
        raw = (
            "Result [D1:C2]:\n```json\n"
            "{'summary': 'risk found', 'risks': [{'clause': 'renewal', 'risk_level': 'high', "
            "'evidence_ids': ['D1:C2']}], 'score': '0.9',}\n```"
        )
        report = await OutputValidator(contract).validate(raw, evidence_ids={"D1:C2"})
        assert report.valid
        assert isinstance(report.output, Report)
        assert report.output.score == 0.9
        assert report.citations == ["D1:C2"]

    @pytest.mark.asyncio
    async def test_missing_citations_fail(self, contract):
        report = await OutputValidator(contract).validate(
            '{"summary": "x", "risks": [], "score": 1.0}', evidence_ids={"E9"}
        )
        assert not report.valid
        assert any(s.name == "citations" and not s.passed for s in report.steps)

    @pytest.mark.asyncio
    async def test_schema_failure_not_silently_repaired(self, contract):
        report = await OutputValidator(contract).validate(
            '{"summary": "x"}', evidence_ids={"E1"}
        )  # missing required fields entirely — repair must NOT invent them
        assert not report.valid

    @pytest.mark.asyncio
    async def test_semantic_validator_blocks(self):
        contract = OutputContract.from_schema(
            OutputSchema.from_pydantic(Report),
            validators=[ValidatorSpec(name="max_score", params={"limit": 0.5})],
        )

        def max_score(data, ctx):
            return "score too high" if data.score > ctx["limit"] else None

        validator = OutputValidator(contract, semantic_validators={"max_score": max_score})
        report = await validator.validate('{"summary": "x", "risks": [], "score": 0.9}')
        assert not report.valid
        assert "score too high" in report.errors[0]

    @pytest.mark.asyncio
    async def test_text_contract_passthrough(self):
        report = await OutputValidator(OutputContract(format="text")).validate("plain answer")
        assert report.valid
        assert report.output == "plain answer"
