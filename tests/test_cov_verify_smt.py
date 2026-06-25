"""Real-behavior tests for the optional SMT / CAS verification backends.

The Z3 and SymPy backends sit behind ``pip install "vincio[verify]"``. This suite
runs offline (no solver installed), so it nails the *not-installed* contract: the
availability probes report False, ``_require`` raises a precise ``LoaderError``,
and both verifiers refuse to run rather than silently passing. The pure-Python
``_comparisons`` extraction (which needs no solver) is exercised directly. The
solver-dependent code paths each ``pytest.importorskip`` their backend, so they
run only when the solver is present and skip cleanly when it is absent.
"""

from __future__ import annotations

import pytest

from vincio.core.errors import LoaderError
from vincio.verify.certificates import VerificationContext
from vincio.verify.kernels import Constraint
from vincio.verify.smt import (
    CasArithmeticVerifier,
    SmtConstraintVerifier,
    cas_available,
    smt_available,
)

_HAS_Z3 = smt_available()
_HAS_SYMPY = cas_available()


# --------------------------------------------------------------------------- #
# Availability probes                                                          #
# --------------------------------------------------------------------------- #


def test_smt_available_returns_bool_and_is_consistent_with_import():
    """smt_available() reflects whether z3 actually imports, as a real bool."""
    result = smt_available()
    assert result is True or result is False
    try:
        import z3  # noqa: F401
    except ImportError:
        assert result is False
    else:
        assert result is True


def test_cas_available_returns_bool_and_is_consistent_with_import():
    """cas_available() reflects whether sympy actually imports, as a real bool."""
    result = cas_available()
    assert result is True or result is False
    try:
        import sympy  # noqa: F401
    except ImportError:
        assert result is False
    else:
        assert result is True


@pytest.mark.skipif(_HAS_Z3, reason="z3 is installed; this is the not-installed branch")
def test_smt_available_false_when_z3_absent():
    """With z3 absent (the offline default) the probe is exactly False."""
    assert smt_available() is False


@pytest.mark.skipif(_HAS_SYMPY, reason="sympy is installed; this is the not-installed branch")
def test_cas_available_false_when_sympy_absent():
    """With sympy absent (the offline default) the probe is exactly False."""
    assert cas_available() is False


# --------------------------------------------------------------------------- #
# SmtConstraintVerifier — not-installed refusal                                #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(_HAS_Z3, reason="z3 installed: solver path would run instead of refusing")
def test_smt_check_raises_loadererror_with_install_hint_when_z3_absent():
    """check() refuses (raises) rather than silently passing when z3 is missing."""
    verifier = SmtConstraintVerifier([Constraint.compare("x", "<=", 10)])
    ctx = VerificationContext(constraints=[Constraint.compare("x", "<=", 10)])
    with pytest.raises(LoaderError, match=r'z3-solver is required.*pip install "vincio\[verify\]"'):
        verifier.check({"x": 5}, ctx)


@pytest.mark.skipif(_HAS_Z3, reason="z3 installed")
def test_smt_check_refuses_even_with_no_constraints_when_z3_absent():
    """The z3 import guard fires before the empty-comparison early return."""
    verifier = SmtConstraintVerifier()
    ctx = VerificationContext()
    with pytest.raises(LoaderError, match="z3-solver is required"):
        verifier.check({}, ctx)


def test_smt_verifier_kind_is_smt():
    """The verifier advertises its kind for certificate grouping."""
    assert SmtConstraintVerifier().kind == "smt"


def test_smt_verifier_stores_constructor_constraints_as_a_list_copy():
    """Constraints are copied into an independent list (None -> empty)."""
    assert SmtConstraintVerifier()._constraints == []
    source = [Constraint.compare("x", ">", 0)]
    verifier = SmtConstraintVerifier(source)
    assert verifier._constraints == source
    source.append(Constraint.compare("y", "<", 9))
    assert len(verifier._constraints) == 1  # copy, not alias


# --------------------------------------------------------------------------- #
# _comparisons — pure-Python extraction (needs no solver)                      #
# --------------------------------------------------------------------------- #


def test_comparisons_extracts_only_smt_tagged_constraints():
    """compare() constraints carry an _smt spec; predicate()/all_different() do not."""
    cmp_le = Constraint.compare("x", "<=", 10)
    cmp_gt = Constraint.compare("y", ">", 3)
    non_smt_pred = Constraint.predicate("x even", lambda a: a.get("x", 0) % 2 == 0)
    non_smt_alldiff = Constraint.all_different(["x", "y"])
    ctx = VerificationContext(constraints=[cmp_le, non_smt_pred, cmp_gt, non_smt_alldiff])

    comparisons = SmtConstraintVerifier._comparisons(ctx)

    assert comparisons == [("x", "<=", 10), ("y", ">", 3)]


def test_comparisons_empty_when_no_declarative_constraints():
    """A context with only predicate/all_different constraints yields no comparisons."""
    ctx = VerificationContext(
        constraints=[
            Constraint.predicate("p", lambda a: True),
            Constraint.all_different(["a", "b"]),
        ]
    )
    assert SmtConstraintVerifier._comparisons(ctx) == []


def test_comparisons_ignores_objects_without_smt_attribute():
    """Foreign constraint-like objects lacking ``_smt`` are skipped, not crashed on."""

    class Foreign:
        description = "not a real constraint"

    ctx = VerificationContext(constraints=[Foreign(), Constraint.compare("z", "==", 7)])
    assert SmtConstraintVerifier._comparisons(ctx) == [("z", "==", 7)]


def test_comparisons_preserves_each_operator_and_bound():
    """Every comparison operator round-trips through the _smt spec unchanged."""
    ops = ["==", "!=", "<", "<=", ">", ">="]
    constraints = [Constraint.compare(f"v{i}", op, i) for i, op in enumerate(ops)]
    ctx = VerificationContext(constraints=constraints)
    extracted = SmtConstraintVerifier._comparisons(ctx)
    assert extracted == [(f"v{i}", op, i) for i, op in enumerate(ops)]


# --------------------------------------------------------------------------- #
# CasArithmeticVerifier — not-installed refusal                               #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(_HAS_SYMPY, reason="sympy installed: CAS path would run instead of refusing")
def test_cas_check_raises_loadererror_with_install_hint_when_sympy_absent():
    """check() refuses (raises) rather than silently passing when sympy is missing."""
    verifier = CasArithmeticVerifier()
    ctx = VerificationContext()
    with pytest.raises(LoaderError, match=r'sympy is required.*pip install "vincio\[verify\]"'):
        verifier.check("1/3 = 0.333", ctx)


@pytest.mark.skipif(_HAS_SYMPY, reason="sympy installed")
def test_cas_check_refuses_before_inspecting_answer_text_when_sympy_absent():
    """The sympy import guard fires before any equality parsing (even on empty text)."""
    verifier = CasArithmeticVerifier()
    ctx = VerificationContext()
    with pytest.raises(LoaderError, match="sympy is required"):
        verifier.check("no equality here at all", ctx)


def test_cas_verifier_kind_is_cas():
    """The verifier advertises its kind for certificate grouping."""
    assert CasArithmeticVerifier().kind == "cas"


# --------------------------------------------------------------------------- #
# Solver-dependent paths — only runnable with vincio[verify] installed         #
# --------------------------------------------------------------------------- #


def test_smt_inapplicable_when_no_comparison_constraints_present():
    """With z3 present, a no-comparison context yields a single inapplicable check."""
    pytest.importorskip("z3")
    verifier = SmtConstraintVerifier()
    ctx = VerificationContext()
    checks = verifier.check({}, ctx)
    assert len(checks) == 1
    assert checks[0].status == "inapplicable"
    assert checks[0].kind == "smt"


def test_smt_refutes_unsatisfiable_system():
    """x > 5 and x < 1 is unsatisfiable: the solver refutes the system."""
    pytest.importorskip("z3")
    verifier = SmtConstraintVerifier()
    ctx = VerificationContext(
        constraints=[Constraint.compare("x", ">", 5), Constraint.compare("x", "<", 1)]
    )
    checks = verifier.check({"x": 3}, ctx)
    assert checks[0].status == "refuted"
    assert "unsatisfiable" in checks[0].detail


def test_smt_verifies_assignment_that_models_constraints():
    """A satisfiable system with a conforming assignment verifies."""
    pytest.importorskip("z3")
    verifier = SmtConstraintVerifier()
    ctx = VerificationContext(constraints=[Constraint.compare("x", "<=", 10)])
    checks = verifier.check({"x": 4}, ctx)
    assert checks[0].status == "verified"


def test_smt_refutes_assignment_that_is_not_a_model():
    """A satisfiable system but a violating assignment is refuted, not verified."""
    pytest.importorskip("z3")
    verifier = SmtConstraintVerifier()
    ctx = VerificationContext(constraints=[Constraint.compare("x", "<=", 10)])
    checks = verifier.check({"x": 99}, ctx)
    assert checks[0].status == "refuted"
    assert "not a model" in checks[0].detail


def test_cas_inapplicable_when_no_arithmetic_equality_present():
    """With sympy present but no equality in the text, the check is inapplicable."""
    pytest.importorskip("sympy")
    verifier = CasArithmeticVerifier()
    checks = verifier.check("the answer is purely prose", VerificationContext())
    assert len(checks) == 1
    assert checks[0].status == "inapplicable"


def test_cas_verifies_exact_rational_equality():
    """2 + 2 = 4 holds exactly under rational arithmetic."""
    pytest.importorskip("sympy")
    verifier = CasArithmeticVerifier()
    checks = verifier.check("2 + 2 = 4", VerificationContext())
    assert any(c.status == "verified" for c in checks)


def test_cas_refutes_inexactly_rounded_equality():
    """1/3 = 0.333 is refuted because exact rational arithmetic disagrees."""
    pytest.importorskip("sympy")
    verifier = CasArithmeticVerifier()
    checks = verifier.check("1/3 = 0.333", VerificationContext())
    assert any(c.status == "refuted" for c in checks)
