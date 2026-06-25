"""Real-behavior coverage for the deterministic verification kernels.

Every test drives the real kernel API over hand-chosen inputs and asserts a
specific verdict (status + recomputed value) or a raised error with its message.
No mocks: the kernels are pure and offline, so the inputs themselves are the
fixtures.
"""

from __future__ import annotations

import pytest

from vincio.core.types import EvidenceItem
from vincio.verify.certificates import VerificationContext
from vincio.verify.kernels import (
    ArithmeticVerifier,
    CitationVerifier,
    Constraint,
    ConstraintVerifier,
    SchemaVerifier,
    TemporalVerifier,
    UnitVerifier,
    default_verifiers,
    safe_eval_arithmetic,
)

CTX = VerificationContext()


# --------------------------------------------------------------------------- #
# safe_eval_arithmetic — the recursive-descent parser's error/edge branches    #
# --------------------------------------------------------------------------- #


def test_safe_eval_precedence_and_modulo():
    # term binds tighter than expr; % is modulo, not percent.
    assert safe_eval_arithmetic("2 + 3 * 4") == 14.0
    assert safe_eval_arithmetic("17 % 5") == 2.0
    assert safe_eval_arithmetic("(2 + 3) * 4") == 20.0


def test_safe_eval_unary_plus_and_minus():
    # The leading "+" branch (line 105-106) and unary minus.
    assert safe_eval_arithmetic("+7") == 7.0
    assert safe_eval_arithmetic("-3 * +2") == -6.0


def test_safe_eval_empty_expression_raises():
    with pytest.raises(ValueError, match="empty expression"):
        safe_eval_arithmetic("   ")


def test_safe_eval_unbalanced_parentheses_raises():
    with pytest.raises(ValueError, match="unbalanced parentheses"):
        safe_eval_arithmetic("(2 + 3")


def test_safe_eval_expected_number_raises():
    # An operator where an atom is required hits the "expected a number" branch.
    with pytest.raises(ValueError, match="expected a number, got '\\*'"):
        safe_eval_arithmetic("2 * * 3")


def test_safe_eval_division_by_zero_raises():
    with pytest.raises(ValueError, match="division by zero"):
        safe_eval_arithmetic("4 / 0")


def test_safe_eval_modulo_by_zero_raises():
    with pytest.raises(ValueError, match="modulo by zero"):
        safe_eval_arithmetic("4 % 0")


def test_safe_eval_trailing_tokens_raises():
    # A bare ")" leaves the closing paren unconsumed -> trailing tokens.
    with pytest.raises(ValueError, match="trailing tokens"):
        safe_eval_arithmetic("2 + 3 )")


def test_safe_eval_unexpected_character_raises():
    with pytest.raises(ValueError, match="unexpected character"):
        safe_eval_arithmetic("2 + a")


# --------------------------------------------------------------------------- #
# ArithmeticVerifier                                                            #
# --------------------------------------------------------------------------- #


def test_arithmetic_equality_verified():
    [check] = ArithmeticVerifier().check("So 2 + 3 = 5 total.", CTX)
    assert check.status == "verified"
    assert check.name == "equality"
    assert check.evidence["computed"] == 5.0
    assert check.evidence["claimed"] == 5.0


def test_arithmetic_equality_refuted_on_recompute_disagreement():
    [check] = ArithmeticVerifier().check("Clearly 2 + 2 = 5.", CTX)
    assert check.status == "refuted"
    assert check.evidence["computed"] == 4.0
    assert check.evidence["claimed"] == 5.0
    assert "claimed 5" in check.detail


def test_arithmetic_percent_of_verified_and_refuted():
    verified, refuted = ArithmeticVerifier().check(
        "10% of 200 is 20, but 50% of 8 equals 5.", CTX
    )
    assert verified.name == "percent_of"
    assert verified.status == "verified"
    assert verified.evidence["computed"] == 20.0
    assert refuted.status == "refuted"
    assert refuted.evidence["computed"] == 4.0
    assert refuted.evidence["claimed"] == 5.0


def test_arithmetic_assignment_without_operator_skipped():
    # "5 = 5" matches the equality regex but its lhs carries no operator, so it is
    # an assignment, not a computed claim -> skipped, leaving inapplicable (197).
    [check] = ArithmeticVerifier().check("The value 5 = 5 holds.", CTX)
    assert check.status == "inapplicable"
    assert check.detail == "no arithmetic equality found"


def test_arithmetic_division_equality_verified():
    # Exercises the successful "/" branch of the term parser (line 121).
    [check] = ArithmeticVerifier().check("Note 20 / 4 = 5.", CTX)
    assert check.status == "verified"
    assert check.evidence["computed"] == 5.0


def test_arithmetic_iso_date_left_side_is_not_subtraction():
    # An ISO date matches the equality regex but is deferred to temporal (198-199).
    [check] = ArithmeticVerifier().check("The release 2024-01-05 = 2024", CTX)
    assert check.status == "inapplicable"


def test_arithmetic_malformed_expression_is_skipped():
    # The lhs matches the equality regex but safe_eval raises -> continue (203-204);
    # with no other claim the kernel is inapplicable.
    [check] = ArithmeticVerifier().check("Compute 5 / 0 = 0 now.", CTX)
    assert check.status == "inapplicable"


def test_arithmetic_accepts_non_string_answer():
    # answer is stringified; the embedded equality is then checked.
    [check] = ArithmeticVerifier().check(["1 + 1 = 2"], CTX)
    assert check.status == "verified"


# --------------------------------------------------------------------------- #
# UnitVerifier                                                                  #
# --------------------------------------------------------------------------- #


def test_units_conversion_verified():
    [check] = UnitVerifier().check("Note that 5 km = 5000 m.", CTX)
    assert check.status == "verified"
    assert check.name == "conversion"
    assert check.evidence["computed"] == 5000.0


def test_units_conversion_refuted_on_wrong_factor():
    [check] = UnitVerifier().check("Wrongly, 5 km = 500 m.", CTX)
    assert check.status == "refuted"
    assert check.evidence["computed"] == 5000.0
    assert check.evidence["claimed"] == 500.0


def test_units_dimensional_mismatch_refuted():
    [check] = UnitVerifier().check("Absurdly, 5 km = 5000 kg.", CTX)
    assert check.status == "refuted"
    assert check.name == "dimension"
    assert check.evidence["dim1"] == "length"
    assert check.evidence["dim2"] == "mass"


def test_units_unknown_unit_skipped_inapplicable():
    # "parsecs" is not in the unit table -> the match is skipped (line 268).
    [check] = UnitVerifier().check("Roughly 5 km = 1 parsecs.", CTX)
    assert check.status == "inapplicable"
    assert check.detail == "no unit conversion found"


# --------------------------------------------------------------------------- #
# TemporalVerifier                                                              #
# --------------------------------------------------------------------------- #


def test_temporal_duration_verified():
    [check] = TemporalVerifier().check(
        "from 2024-01-01 to 2024-01-11 is 10 days", CTX
    )
    assert check.status == "verified"
    assert check.evidence["computed"] == 10


def test_temporal_duration_refuted_off_by_one():
    [check] = TemporalVerifier().check(
        "from 2024-01-01 to 2024-01-11 is 11 days", CTX
    )
    assert check.status == "refuted"
    assert check.evidence["computed"] == 10
    assert check.evidence["claimed"] == 11


def test_temporal_ordering_before_verified_and_after_refuted():
    verified = TemporalVerifier().check("2024-01-01 is before 2024-02-01", CTX)[0]
    refuted = TemporalVerifier().check("2024-03-01 is after 2024-12-01", CTX)[0]
    assert verified.status == "verified"
    assert verified.evidence["relation"] == "before"
    assert refuted.status == "refuted"
    assert refuted.evidence["relation"] == "after"


def test_temporal_invalid_duration_date_skipped():
    # 2024-13-40 matches the regex but is not a real date -> continue (line 341).
    [check] = TemporalVerifier().check(
        "from 2024-13-40 to 2024-01-11 is 10 days", CTX
    )
    assert check.status == "inapplicable"
    assert check.detail == "no temporal claim found"


def test_temporal_invalid_ordering_date_skipped():
    # An impossible date in an ordering claim is skipped (line 358).
    [check] = TemporalVerifier().check("2024-02-30 is before 2024-03-01", CTX)
    assert check.status == "inapplicable"


# --------------------------------------------------------------------------- #
# Constraint / ConstraintVerifier                                              #
# --------------------------------------------------------------------------- #


def test_constraint_compare_unknown_operator_raises():
    with pytest.raises(ValueError, match="unknown operator '~~'"):
        Constraint.compare("x", "~~", 3)


def test_constraint_compare_sets_smt_only_for_relational_ops():
    rel = Constraint.compare("x", "<=", 10)
    membership = Constraint.compare("x", "in", [1, 2, 3])
    # The relational op records a declarative SMT tuple; "in" does not (420->422).
    assert rel._smt == ("x", "<=", 10)
    assert membership._smt is None


def test_constraint_compare_missing_variable_is_unsatisfied():
    # The lambda guards on `var in a`, so an absent variable is not satisfied.
    assert Constraint.compare("x", ">", 0).satisfied_by({"y": 5}) is False
    assert Constraint.compare("x", ">", 0).satisfied_by({"x": 5}) is True


def test_constraint_all_different_distinct_vs_collision():
    c = Constraint.all_different(["a", "b", "c"])
    assert c.satisfied_by({"a": 1, "b": 2, "c": 3}) is True
    assert c.satisfied_by({"a": 1, "b": 1, "c": 3}) is False


def test_constraint_verifier_verified_and_refuted():
    constraints = [
        Constraint.compare("x", "<=", 10),
        Constraint.compare("x", ">", 100),
    ]
    checks = ConstraintVerifier(constraints).check({"x": 5}, CTX)
    statuses = {c.detail for c in checks}
    assert statuses == {"satisfied", "violated"}
    assert checks[0].status == "verified"
    assert checks[1].status == "refuted"


def test_constraint_verifier_inapplicable_with_no_constraints():
    [check] = ConstraintVerifier([]).check({"x": 1}, CTX)
    assert check.status == "inapplicable"
    assert check.detail == "no constraints supplied"


def test_constraint_verifier_inapplicable_when_answer_not_assignment():
    # Constraints exist, the answer is not a mapping, and facts is not a dict
    # either -> the kernel cannot form an assignment (lines 461-463). Pydantic
    # validates facts as a dict, so we set the non-dict fallback directly to
    # drive the kernel's own isinstance guard.
    ctx = VerificationContext(constraints=[Constraint.compare("x", ">", 0)])
    object.__setattr__(ctx, "facts", ["not", "a", "dict"])
    [check] = ConstraintVerifier().check("not a dict", ctx)
    assert check.status == "inapplicable"
    assert check.detail == "answer is not an assignment"


def test_constraint_verifier_reads_constraints_from_context():
    ctx = VerificationContext(constraints=[Constraint.compare("x", "==", 7)])
    [check] = ConstraintVerifier().check({"x": 7}, ctx)
    assert check.status == "verified"


def test_constraint_verifier_falls_back_to_context_facts():
    # answer is not a dict, so the assignment is taken from context.facts.
    ctx = VerificationContext(facts={"x": 3})
    [check] = ConstraintVerifier([Constraint.compare("x", "<", 5)]).check(None, ctx)
    assert check.status == "verified"


def test_constraint_verifier_faulty_predicate_refutes_not_crashes():
    # A predicate that raises is recorded as refuted with the error (468-471).
    def boom(_a: dict) -> bool:
        raise RuntimeError("kaboom")

    bad = Constraint.predicate("explodes", boom)
    [check] = ConstraintVerifier([bad]).check({"x": 1}, CTX)
    assert check.status == "refuted"
    assert "predicate error: kaboom" in check.detail


# --------------------------------------------------------------------------- #
# SchemaVerifier                                                                #
# --------------------------------------------------------------------------- #

_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
}


def test_schema_verified_on_conforming_value():
    [check] = SchemaVerifier(_SCHEMA).check({"n": 3}, CTX)
    assert check.status == "verified"
    assert check.detail == "conforms to schema"


def test_schema_refuted_lists_errors():
    [check] = SchemaVerifier(_SCHEMA).check({"n": "not-an-int"}, CTX)
    assert check.status == "refuted"
    assert check.evidence["errors"]


def test_schema_inapplicable_without_schema():
    [check] = SchemaVerifier().check({"n": 3}, CTX)
    assert check.status == "inapplicable"
    assert check.detail == "no schema supplied"


def test_schema_reads_schema_from_context():
    ctx = VerificationContext(schema=_SCHEMA)
    [check] = SchemaVerifier().check({"n": 1}, ctx)
    assert check.status == "verified"


def test_schema_dumps_pydantic_model_answer():
    # A model answer is dumped via model_dump before validation (line 508).
    item = EvidenceItem(source_id="s", text="t")
    open_schema = {"type": "object"}
    [check] = SchemaVerifier(open_schema).check(item, CTX)
    assert check.status == "verified"


# --------------------------------------------------------------------------- #
# CitationVerifier                                                              #
# --------------------------------------------------------------------------- #

_EVIDENCE = [
    EvidenceItem(source_id="src", text="The tower is 324 meters tall and opened in 1889.")
]


def test_citation_entailed_claim_verified():
    [check] = CitationVerifier(_EVIDENCE).check("The tower is 324 meters tall.", CTX)
    assert check.status == "verified"
    assert check.name == "entailment"


def test_citation_numeric_contradiction_refuted():
    # Strict support requires every number to appear; 500 is not in the evidence.
    [check] = CitationVerifier(_EVIDENCE).check("The tower is 500 meters tall.", CTX)
    assert check.status == "refuted"
    assert "not entailed" in check.detail


def test_citation_inapplicable_without_evidence():
    [check] = CitationVerifier([]).check("The tower is 324 meters tall.", CTX)
    assert check.status == "inapplicable"
    assert check.detail == "no evidence supplied"


def test_citation_inapplicable_without_verifiable_claim():
    # No factual/number-bearing claim -> no verifiable claims found (line 556).
    [check] = CitationVerifier(_EVIDENCE).check("Hello there!", CTX)
    assert check.status == "inapplicable"
    assert check.detail == "no verifiable claim found"


def test_citation_reads_evidence_from_context():
    ctx = VerificationContext(evidence=_EVIDENCE)
    [check] = CitationVerifier().check("The tower is 324 meters tall.", ctx)
    assert check.status == "verified"


def test_citation_extracts_text_from_dict_answer():
    # A dict answer's "text" key is used as the claim source (lines 528-529).
    [check] = CitationVerifier(_EVIDENCE).check(
        {"text": "The tower is 324 meters tall."}, CTX
    )
    assert check.status == "verified"


def test_citation_stringifies_non_text_answer():
    # A list answer has no text key and no model_dump, so _claim_text falls back
    # to str() (line 530); the bracketed surrogate is not strictly supported.
    [check] = CitationVerifier(_EVIDENCE).check(["The tower is 324 meters tall."], CTX)
    assert check.name == "entailment"
    assert check.status == "refuted"


def test_citation_extracts_text_from_pydantic_answer():
    # A pydantic answer with a "text" field feeds _claim_text (lines 525-527).
    answer = EvidenceItem(source_id="ans", text="The tower is 324 meters tall.")
    [check] = CitationVerifier(_EVIDENCE).check(answer, CTX)
    assert check.status == "verified"


# --------------------------------------------------------------------------- #
# default_verifiers                                                             #
# --------------------------------------------------------------------------- #


def test_default_verifiers_are_the_six_kernels():
    kinds = [v.kind for v in default_verifiers()]
    assert kinds == ["arithmetic", "units", "temporal", "constraints", "schema", "citation"]


def test_default_verifiers_all_inapplicable_on_empty_text():
    # Every default kernel safely returns inapplicable over a claimless answer.
    statuses = {
        c.status
        for v in default_verifiers()
        for c in v.check("just some prose with no claims", CTX)
    }
    assert statuses == {"inapplicable"}
