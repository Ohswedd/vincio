"""Coverage-focused, behavior-real tests for vincio.output.streaming.

Targets the prefix-checker (`validate_partial` / `_check_partial`) across
every schema-shape branch (objects, arrays, scalars, nullable unions,
anyOf/oneOf combiners, $ref resolution) and the `StreamingValidator`
incremental lifecycle (interval gating, unparseable deltas, mid-stream
abort, finalize repair / parse-failure / post-repair-invalid).

All tests are deterministic and offline. No mocks, no network.
"""

from __future__ import annotations

from vincio.output import (
    OutputSchema,
    StreamingValidator,
    validate_partial,
)
from vincio.output.schemas import RepairPolicy

# ---------------------------------------------------------------------------
# validate_partial / _check_partial — scalar branches
# ---------------------------------------------------------------------------


class TestScalarChecks:
    def test_string_arrived_as_int_is_definite_error(self):
        errors = validate_partial({"name": 7}, {"properties": {"name": {"type": "string"}}})
        assert errors == ["$.name: expected string, got int"]

    def test_integer_arrived_as_string_is_error(self):
        errors = validate_partial({"n": "5"}, {"properties": {"n": {"type": "integer"}}})
        assert errors == ["$.n: expected integer, got str"]

    def test_bool_is_not_an_integer(self):
        # bool is a subclass of int in Python; the checker must reject it.
        errors = validate_partial({"n": True}, {"properties": {"n": {"type": "integer"}}})
        assert errors == ["$.n: expected integer, got bool"]

    def test_number_accepts_int_and_float_but_not_bool(self):
        props = {"properties": {"x": {"type": "number"}}}
        assert validate_partial({"x": 3}, props) == []
        assert validate_partial({"x": 3.5}, props) == []
        assert validate_partial({"x": False}, props) == ["$.x: expected number, got bool"]

    def test_number_arrived_as_string_is_error(self):
        errors = validate_partial({"x": "nope"}, {"properties": {"x": {"type": "number"}}})
        assert errors == ["$.x: expected number, got str"]

    def test_boolean_arrived_as_int_is_error(self):
        errors = validate_partial({"flag": 1}, {"properties": {"flag": {"type": "boolean"}}})
        assert errors == ["$.flag: expected boolean, got int"]

    def test_none_scalar_is_tolerated_as_not_yet_arrived(self):
        # A null value for a typed scalar is "not arrived yet" — no error.
        assert validate_partial({"name": None}, {"properties": {"name": {"type": "string"}}}) == []


# ---------------------------------------------------------------------------
# nullable unions ("type": [...])
# ---------------------------------------------------------------------------


class TestNullableUnion:
    def test_null_allowed_when_null_in_type_list(self):
        schema = {"properties": {"v": {"type": ["string", "null"]}}}
        assert validate_partial({"v": None}, schema) == []

    def test_union_uses_first_non_null_type_for_checking(self):
        schema = {"properties": {"v": {"type": ["integer", "null"]}}}
        assert validate_partial({"v": 3}, schema) == []
        assert validate_partial({"v": "x"}, schema) == ["$.v: expected integer, got str"]

    def test_value_present_but_null_not_permitted_falls_through(self):
        # type list with no "null" — a real value is still type-checked.
        schema = {"properties": {"v": {"type": ["string"]}}}
        assert validate_partial({"v": 9}, schema) == ["$.v: expected string, got int"]

    def test_union_of_only_null_yields_no_type_check(self):
        # non_null empty -> schema_type None -> no scalar branch fires.
        schema = {"properties": {"v": {"type": ["null"]}}}
        assert validate_partial({"v": 123}, schema) == []

    def test_null_value_with_null_absent_from_list_falls_through(self):
        # data is None but "null" not in the list: no early return at the null
        # guard; schema_type becomes the first non-null type, and None is then
        # tolerated as "not arrived yet" by the later `if data is None` guard.
        schema = {"properties": {"v": {"type": ["integer", "string"]}}}
        assert validate_partial({"v": None}, schema) == []


# ---------------------------------------------------------------------------
# object branch
# ---------------------------------------------------------------------------


class TestObjectChecks:
    def test_expected_object_got_list_is_error(self):
        schema = {"properties": {"o": {"type": "object", "properties": {}}}}
        errors = validate_partial({"o": [1, 2]}, schema)
        assert errors == ["$.o: expected object, got list"]

    def test_expected_object_got_none_is_tolerated(self):
        schema = {"properties": {"o": {"type": "object", "properties": {}}}}
        assert validate_partial({"o": None}, schema) == []

    def test_unknown_field_only_flagged_when_closed_and_has_properties(self):
        closed = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
        }
        assert validate_partial({"a": "x", "b": 1}, closed) == ["$: unknown field 'b'"]
        # Open object tolerates extras.
        open_obj = {"type": "object", "properties": {"a": {"type": "string"}}}
        assert validate_partial({"a": "x", "b": 1}, open_obj) == []

    def test_closed_object_with_no_properties_does_not_flag(self):
        # `closed and properties` requires properties to be non-empty.
        closed_empty = {"type": "object", "properties": {}, "additionalProperties": False}
        assert validate_partial({"anything": 1}, closed_empty) == []

    def test_properties_presence_implies_object_even_without_type(self):
        # "properties" key alone routes through the object branch.
        schema = {"properties": {"a": {"type": "string"}}}
        assert validate_partial("not a dict", schema) == ["$: expected object, got str"]


# ---------------------------------------------------------------------------
# array branch
# ---------------------------------------------------------------------------


class TestArrayChecks:
    def test_expected_array_got_dict_is_error(self):
        schema = {"type": "array", "items": {"type": "integer"}}
        assert validate_partial({"not": "list"}, schema) == ["$: expected array, got dict"]

    def test_expected_array_got_none_is_tolerated(self):
        schema = {"type": "array", "items": {"type": "integer"}}
        assert validate_partial(None, schema) == []

    def test_array_items_are_checked_with_index_in_path(self):
        schema = {"type": "array", "items": {"type": "integer"}}
        errors = validate_partial([1, "two", 3], schema)
        assert errors == ["$[1]: expected integer, got str"]

    def test_array_without_items_schema_tolerates_anything(self):
        schema = {"type": "array"}
        assert validate_partial([1, "x", {"k": 1}], schema) == []

    def test_non_dict_items_schema_is_ignored(self):
        # A malformed `items` that is not a dict (here a bool, JSON-Schema's
        # "true" wildcard) reaches _check_partial and is skipped, not crashed.
        schema = {"type": "array", "items": True}
        assert validate_partial([1, "x"], schema) == []


# ---------------------------------------------------------------------------
# anyOf / oneOf combiners
# ---------------------------------------------------------------------------


class TestCombiners:
    def test_anyof_passes_when_one_branch_matches(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        assert validate_partial(42, schema) == []
        assert validate_partial("hi", schema) == []

    def test_anyof_fails_when_no_branch_matches(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        assert validate_partial({"obj": 1}, schema) == ["$: matches no anyOf branch"]

    def test_oneof_combiner_reported_by_name(self):
        schema = {"oneOf": [{"type": "string"}, {"type": "boolean"}]}
        assert validate_partial(3, schema) == ["$: matches no oneOf branch"]
        assert validate_partial(True, schema) == []

    def test_empty_combiner_list_is_ignored(self):
        # An empty anyOf list is falsy -> combiner skipped, falls through.
        schema = {"anyOf": [], "type": "integer"}
        assert validate_partial("x", schema) == ["$: expected integer, got str"]


# ---------------------------------------------------------------------------
# $ref resolution
# ---------------------------------------------------------------------------


class TestRefResolution:
    def test_ref_resolves_into_defs_and_checks_type(self):
        root = {
            "$defs": {"Name": {"type": "string"}},
            "properties": {"who": {"$ref": "#/$defs/Name"}},
        }
        assert validate_partial({"who": "Ada"}, root) == []
        assert validate_partial({"who": 1}, root) == ["$.who: expected string, got int"]

    def test_unresolvable_ref_yields_empty_schema_no_error(self):
        # Missing target resolves to {} -> no type, nothing flagged.
        root = {"properties": {"who": {"$ref": "#/$defs/Missing"}}}
        assert validate_partial({"who": 12345}, root) == []

    def test_non_local_ref_is_left_unresolved(self):
        # A ref not starting with "#/" is returned as-is (no type to check).
        schema = {"$ref": "http://example/Thing", "type": "string"}
        # _resolve_ref returns the original schema unchanged, so type check applies.
        assert validate_partial(5, schema) == ["$: expected string, got int"]


# ---------------------------------------------------------------------------
# StreamingValidator lifecycle
# ---------------------------------------------------------------------------


def _invoice_schema() -> OutputSchema:
    return OutputSchema.from_json_schema(
        {
            "type": "object",
            "properties": {"vendor": {"type": "string"}, "total": {"type": "number"}},
            "required": ["vendor", "total"],
            "additionalProperties": False,
        },
        name="invoice",
    )


class TestStreamingValidatorFeed:
    def test_feed_below_interval_returns_none_and_accumulates(self):
        v = StreamingValidator(_invoice_schema(), min_interval_chars=100)
        assert v.feed('{"vendor": ') is None
        assert v.feed('"Acme"') is None
        # Text accumulated even though no parse ran.
        assert v.text == '{"vendor": "Acme"'

    def test_empty_delta_does_not_advance_length(self):
        v = StreamingValidator(_invoice_schema(), min_interval_chars=1)
        # Empty delta: no append, length unchanged, still below interval -> None.
        assert v.feed("") is None
        assert v.text == ""

    def test_feed_unparseable_text_returns_none(self):
        v = StreamingValidator(_invoice_schema(), min_interval_chars=1)
        # Whitespace-only is unparseable -> parse_partial_json returns (None, False).
        event = v.feed("    ")
        assert event is None

    def test_interval_gates_then_fires_on_crossing(self):
        v = StreamingValidator(_invoice_schema(), min_interval_chars=10)
        assert v.feed("'''''''''") is None  # 9 chars < 10, no parse attempted
        event = v.feed("'")  # now 10 chars -> parse runs; text is unparseable JSON
        assert event is None  # "''''''''''" cannot be balanced into JSON
        # text is preserved regardless
        assert v.text == "''''''''''"

    def test_valid_prefix_event_mid_stream(self):
        v = StreamingValidator(_invoice_schema(), min_interval_chars=1)
        event = v.feed('{"vendor": "Ac')
        assert event is not None
        assert event.valid_prefix is True
        assert event.errors == []
        assert event.complete is False
        assert event.data == {"vendor": "Ac"}
        assert event.chars_seen == len('{"vendor": "Ac')
        assert v.last_event is event

    def test_invalid_prefix_detected_and_recorded(self):
        v = StreamingValidator(_invoice_schema(), min_interval_chars=1)
        event = v.feed('{"vendor": "Acme", "total": "oops"')
        assert event is not None
        assert event.valid_prefix is False
        assert event.errors == ["$.total: expected number, got str"]

    def test_no_schema_means_no_errors_ever(self):
        v = StreamingValidator(schema=None, min_interval_chars=1)
        event = v.feed('{"total": "definitely-not-a-number"}')
        assert event is not None
        assert event.valid_prefix is True
        assert event.errors == []
        assert event.complete is True


class TestStreamingValidatorFinalize:
    def test_finalize_on_empty_stream_cannot_parse(self):
        v = StreamingValidator(_invoice_schema())
        event = v.finalize()
        assert event.data is None
        assert event.valid_prefix is False
        assert event.errors == ["could not parse structured output"]
        assert event.complete is False
        assert event.repaired is False

    def test_finalize_clean_complete_output_no_repair(self):
        v = StreamingValidator(_invoice_schema(), min_interval_chars=1000)
        v.feed('{"vendor": "Acme", "total": 12.5}')
        event = v.finalize()
        assert event.repaired is False
        assert event.repair_actions == []
        assert event.valid_prefix is True
        assert event.errors == []
        assert event.data == {"vendor": "Acme", "total": 12.5}
        assert event.complete is True

    def test_finalize_repairs_coercible_type(self):
        v = StreamingValidator(_invoice_schema())
        v.feed('{"vendor": "Acme", "total": "12.5"}')
        event = v.finalize()
        assert event.repaired is True
        assert event.repair_actions  # at least one action recorded
        assert event.data["total"] == 12.5
        assert event.valid_prefix is True

    def test_finalize_unrepairable_reports_post_repair_mismatch(self):
        # Disable type coercion so "twelve" cannot be repaired into a number.
        policy = RepairPolicy(allow_type_coercion=False, allow_fill_optional=False)
        v = StreamingValidator(_invoice_schema(), repair_policy=policy)
        v.feed('{"vendor": "Acme", "total": "twelve"}')
        event = v.finalize()
        assert event.repaired is False
        assert event.valid_prefix is False
        assert event.errors == ["output does not match schema after streaming repair"]
        # Even unrepaired, a parsed object counts as complete.
        assert event.complete is True

    def test_finalize_truncated_stream_marked_complete_when_parseable(self):
        # Truncated JSON: partial parses (complete=False from parser) but the
        # presence of a parsed object makes the finalize event complete=True.
        v = StreamingValidator(schema=None)
        v.feed('{"vendor": "Acme", "total": 1')
        event = v.finalize()
        assert event.data == {"vendor": "Acme", "total": 1}
        assert event.complete is True
        assert event.errors == []
