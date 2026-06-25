"""Real-behavior coverage tests for vincio.output.repair.

Exercises the structural repairer through its public API with the deterministic
MockProvider (no mocks/patches). Targets the uncovered branches: lenient JSON
parse-repair, type coercion across every JSON type, optional-field filling,
the repair loop's progress/exhaustion logic, the never-invent invariant, and
the policy-forbidden / missing-provider error paths.
"""

from __future__ import annotations

import pytest

from vincio.core.errors import OutputParseError, OutputRepairForbiddenError
from vincio.core.types import ModelResponse
from vincio.output.repair import Repairer, RepairOutcome
from vincio.output.schemas import OutputSchema, RepairPolicy
from vincio.providers.mock import MockProvider


def _schema(json_schema: dict, name: str = "t") -> OutputSchema:
    return OutputSchema.from_json_schema(json_schema, name=name)


# --------------------------------------------------------------------------- #
# repair_parse: lenient JSON + policy gate
# --------------------------------------------------------------------------- #


def test_repair_parse_fixes_python_literals_and_trailing_comma():
    out = Repairer().repair_parse("{'ok': True, 'n': None,}")
    assert isinstance(out, RepairOutcome)
    assert out.data == {"ok": True, "n": None}
    assert out.repaired is True
    assert out.actions == ["lenient JSON parse"]


def test_repair_parse_forbidden_by_policy():
    repairer = Repairer(RepairPolicy(allow_json_repair=False))
    with pytest.raises(OutputRepairForbiddenError, match="JSON repair disabled"):
        repairer.repair_parse('{"a": 1}')


def test_repair_parse_unrepairable_raises_parse_error():
    # lenient_json_loads gives up on this -> propagates OutputParseError.
    with pytest.raises(OutputParseError, match="could not parse JSON"):
        Repairer().repair_parse("this is not json at all <<<")


# --------------------------------------------------------------------------- #
# repair_structure: already-valid short-circuit
# --------------------------------------------------------------------------- #


def test_repair_structure_already_valid_is_noop():
    schema = _schema({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]})
    out = Repairer().repair_structure({"n": 7}, schema)
    assert out.data == {"n": 7}
    assert out.repaired is False
    assert out.actions == []


# --------------------------------------------------------------------------- #
# type coercion: each JSON type, via nested-object schema
# --------------------------------------------------------------------------- #


def test_coerce_string_to_number():
    schema = _schema(
        {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}
    )
    out = Repairer().repair_structure({"x": "3.5"}, schema)
    assert out.data == {"x": 3.5}
    assert out.repaired is True
    assert any("3.5" in a and "number" in a for a in out.actions)


def test_coerce_string_to_integer():
    schema = _schema(
        {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    )
    out = Repairer().repair_structure({"x": "42"}, schema)
    assert out.data == {"x": 42}
    assert isinstance(out.data["x"], int)


def test_coerce_float_to_integer_only_when_whole():
    schema = _schema(
        {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    )
    # 4.0 is integral -> coerced.
    out = Repairer().repair_structure({"x": 4.0}, schema)
    assert out.data == {"x": 4}
    # 4.5 is not integral -> no coercion, stays invalid, no progress, returned as-is.
    out2 = Repairer().repair_structure({"x": 4.5}, schema)
    assert out2.data == {"x": 4.5}
    assert out2.repaired is False


def test_coerce_non_numeric_string_to_integer_left_unchanged():
    # float('xyz') raises -> the integer-coercion path swallows it and returns
    # the value untouched (never invents a number).
    schema = _schema(
        {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    )
    out = Repairer().repair_structure({"x": "xyz"}, schema)
    assert out.data == {"x": "xyz"}
    assert out.repaired is False


def test_coerce_number_to_string():
    schema = _schema(
        {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
    )
    out = Repairer().repair_structure({"label": 12}, schema)
    assert out.data == {"label": "12"}


def test_coerce_string_to_boolean_truthy_and_falsy():
    schema = _schema(
        {"type": "object", "properties": {"flag": {"type": "boolean"}}, "required": ["flag"]}
    )
    assert Repairer().repair_structure({"flag": "yes"}, schema).data == {"flag": True}
    assert Repairer().repair_structure({"flag": "FALSE"}, schema).data == {"flag": False}


def test_coerce_string_to_boolean_unrecognized_is_not_invented():
    schema = _schema(
        {"type": "object", "properties": {"flag": {"type": "boolean"}}, "required": ["flag"]}
    )
    out = Repairer().repair_structure({"flag": "maybe"}, schema)
    # Ambiguous value must NOT be invented into True/False.
    assert out.data == {"flag": "maybe"}
    assert out.repaired is False
    assert out.actions == []


def test_coerce_non_numeric_string_to_number_left_unchanged():
    schema = _schema(
        {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}
    )
    out = Repairer().repair_structure({"x": "abc"}, schema)
    assert out.data == {"x": "abc"}
    assert out.repaired is False


def test_coerce_scalar_wrapped_in_array_with_item_coercion():
    schema = _schema(
        {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["tags"],
        }
    )
    out = Repairer().repair_structure({"tags": 9}, schema)
    # scalar 9 -> wrapped, then the int item is coerced to a string "9".
    assert out.data == {"tags": ["9"]}
    assert any("wrapped scalar in array" in a for a in out.actions)


def test_coerce_items_inside_existing_array():
    schema = _schema(
        {
            "type": "object",
            "properties": {"nums": {"type": "array", "items": {"type": "integer"}}},
            "required": ["nums"],
        }
    )
    out = Repairer().repair_structure({"nums": ["1", "2", 3]}, schema)
    assert out.data == {"nums": [1, 2, 3]}
    # the per-index action prefix is emitted for coerced entries.
    assert any(a.startswith("nums: [0]:") for a in out.actions)


def test_coerce_union_type_with_null_picks_non_null():
    # schema_type given as a list: ["null","number"] -> coerce against number.
    schema = _schema(
        {
            "type": "object",
            "properties": {"x": {"type": ["null", "number"]}},
            "required": ["x"],
        }
    )
    out = Repairer().repair_structure({"x": "2.0"}, schema)
    assert out.data == {"x": 2.0}


# --------------------------------------------------------------------------- #
# fill optional
# --------------------------------------------------------------------------- #


def test_fill_optional_with_default():
    # 'a' arrives as a string so the data is invalid -> the repair loop runs,
    # coerces 'a', and fills the missing optional 'b' from its default.
    schema = _schema(
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string", "default": "hi"}},
            "required": ["a"],
        }
    )
    out = Repairer().repair_structure({"a": "1"}, schema)
    assert out.data == {"a": 1, "b": "hi"}
    assert any("filled optional 'b' with default" in a for a in out.actions)


def test_fill_optional_with_null_via_type_list():
    schema = _schema(
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "note": {"type": ["string", "null"]}},
            "required": ["a"],
        }
    )
    out = Repairer().repair_structure({"a": "1"}, schema)
    assert out.data == {"a": 1, "note": None}
    assert any("filled optional 'note' with null" in a for a in out.actions)


def test_fill_optional_with_null_via_anyof():
    schema = _schema(
        {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "extra": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["a"],
        }
    )
    out = Repairer().repair_structure({"a": "1"}, schema)
    assert out.data["extra"] is None
    assert any("filled optional 'extra' with null" in a for a in out.actions)


def test_fill_optional_skips_anyof_without_null_branch():
    # An optional prop whose anyOf has no null option and no default cannot be
    # safely filled -> it is left absent (the loop continues past it).
    schema = _schema(
        {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "u": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            },
            "required": ["a"],
        }
    )
    out = Repairer().repair_structure({"a": "1"}, schema)
    assert out.data == {"a": 1}
    assert "u" not in out.data


def test_fill_optional_does_not_fill_required_missing_field():
    # A missing *required* field is a real validation failure, never invented.
    schema = _schema(
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string", "default": "x"}},
            "required": ["a", "b"],
        }
    )
    out = Repairer().repair_structure({"a": 1}, schema)
    # 'b' is required+missing; it must not be auto-filled despite the default.
    assert "b" not in out.data
    assert out.repaired is False


def test_fill_optional_skipped_when_policy_disabled():
    schema = _schema(
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string", "default": "x"}},
            "required": ["a"],
        }
    )
    # 'a' invalid forces the loop in; coercion still runs but fill is disabled.
    out = Repairer(RepairPolicy(allow_fill_optional=False)).repair_structure({"a": "1"}, schema)
    assert out.data == {"a": 1}
    assert "b" not in out.data


def test_coercion_skipped_when_policy_disabled():
    schema = _schema(
        {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    )
    out = Repairer(RepairPolicy(allow_type_coercion=False)).repair_structure({"x": "5"}, schema)
    assert out.data == {"x": "5"}
    assert out.repaired is False


# --------------------------------------------------------------------------- #
# repair loop: combination, no-progress break, max-attempts
# --------------------------------------------------------------------------- #


def test_repair_combines_coercion_and_fill_to_validity():
    schema = _schema(
        {
            "type": "object",
            "properties": {
                "n": {"type": "integer"},
                "opt": {"type": "string", "default": "d"},
            },
            "required": ["n"],
        }
    )
    out = Repairer().repair_structure({"n": "3"}, schema)
    assert out.data == {"n": 3, "opt": "d"}
    assert schema.is_valid(out.data)


def test_repair_no_progress_breaks_without_inventing():
    # A flat string can never satisfy an object schema and nothing coerces it,
    # so the loop must break (not spin) and return the input untouched.
    schema = _schema({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]})
    out = Repairer().repair_structure("plain string", schema)
    assert out.data == "plain string"
    assert out.repaired is False
    assert out.actions == []


def test_max_repair_attempts_exhausted_returns_best_effort():
    # Build a case that needs >1 round but cap attempts at 1 so the loop exits
    # while still invalid. Round 1 coerces the int item; the still-missing
    # required 'b' (no default) cannot be filled, so it stays invalid.
    schema = _schema(
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        }
    )
    out = Repairer(RepairPolicy(max_repair_attempts=1)).repair_structure({"a": "1"}, schema)
    assert out.data == {"a": 1}  # coercion happened
    assert not schema.is_valid(out.data)  # still invalid: 'b' never invented
    assert out.repaired is True


def test_zero_attempts_returns_input_unrepaired():
    schema = _schema({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]})
    out = Repairer(RepairPolicy(max_repair_attempts=0)).repair_structure({"x": "5"}, schema)
    assert out.data == {"x": "5"}
    assert out.repaired is False


# --------------------------------------------------------------------------- #
# repair_with_model: policy gate, missing provider, success paths
# --------------------------------------------------------------------------- #


async def test_repair_with_model_forbidden_by_default_policy():
    repairer = Repairer(provider=MockProvider(), model="mock-1")
    schema = _schema({"type": "object", "properties": {"x": {"type": "integer"}}})
    with pytest.raises(OutputRepairForbiddenError, match="LLM repair disabled"):
        await repairer.repair_with_model('{"x": 1}', schema)


async def test_repair_with_model_requires_provider_and_model():
    repairer = Repairer(RepairPolicy(allow_llm_repair=True))
    schema = _schema({"type": "object", "properties": {"x": {"type": "integer"}}})
    with pytest.raises(OutputRepairForbiddenError, match="requires a provider and model"):
        await repairer.repair_with_model('{"x": 1}', schema)


async def test_repair_with_model_requires_model_even_with_provider():
    repairer = Repairer(RepairPolicy(allow_llm_repair=True), provider=MockProvider())
    schema = _schema({"type": "object", "properties": {"x": {"type": "integer"}}})
    with pytest.raises(OutputRepairForbiddenError, match="requires a provider and model"):
        await repairer.repair_with_model('{"x": 1}', schema)


async def test_repair_with_model_uses_structured_response():
    captured: list = []

    def responder(request):
        captured.append(request)
        return ModelResponse(text='{"x": 5}', structured={"x": 5})

    provider = MockProvider(responder=responder)
    repairer = Repairer(RepairPolicy(allow_llm_repair=True), provider=provider, model="mock-1")
    schema = _schema(
        {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    )
    out = await repairer.repair_with_model('{"x": "5"}', schema)
    assert out.data == {"x": 5}
    assert out.repaired is True
    assert out.actions == ["model reserialization"]
    # The request carried the schema and the structure-only instruction.
    req = captured[0]
    assert req.output_schema_name == "t"
    assert req.temperature == 0.0
    assert any("Do not add, remove, or change" in m.content for m in req.messages)


async def test_repair_with_model_falls_back_to_lenient_parse_when_unstructured():
    # structured=None forces the lenient_json_loads branch on response.text.
    def responder(request):
        return ModelResponse(text="{'x': 7,}", structured=None)

    provider = MockProvider(responder=responder)
    repairer = Repairer(RepairPolicy(allow_llm_repair=True), provider=provider, model="mock-1")
    schema = _schema(
        {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    )
    out = await repairer.repair_with_model("garbage", schema)
    assert out.data == {"x": 7}
    assert out.actions == ["model reserialization"]


async def test_repair_with_model_truncates_oversized_raw_text():
    captured: list = []

    def responder(request):
        captured.append(request)
        return ModelResponse(text="{}", structured={})

    provider = MockProvider(responder=responder)
    repairer = Repairer(RepairPolicy(allow_llm_repair=True), provider=provider, model="mock-1")
    schema = _schema({"type": "object", "properties": {}})
    huge = "A" * 20_000
    await repairer.repair_with_model(huge, schema)
    user_msg = next(m for m in captured[0].messages if m.role == "user")
    # The 20k input is clipped to the 12k window in the prompt.
    assert user_msg.content.count("A") == 12_000
