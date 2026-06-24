"""Optional SMT / CAS verification backends.

The deterministic kernels in :mod:`vincio.verify.kernels` are the default and need
no extra. For the cases that warrant a solver — proving a constraint system is
*consistent* (not merely that one assignment happens to satisfy it), or checking
an arithmetic equality with **exact** rational arithmetic instead of a float
tolerance — these backends sit behind ``pip install "vincio[verify]"`` (Z3 and
SymPy). They are strictly opt-in: nothing in the offline path imports them.
"""

from __future__ import annotations

import re
from typing import Any

from ..core.errors import LoaderError
from .certificates import Check, VerificationContext
from .kernels import _EQUALITY_RE, Constraint

__all__ = [
    "smt_available",
    "cas_available",
    "SmtConstraintVerifier",
    "CasArithmeticVerifier",
]


def smt_available() -> bool:
    """True when the Z3 SMT solver is importable (``vincio[verify]``)."""
    try:
        import z3  # noqa: F401
    except ImportError:
        return False
    return True


def cas_available() -> bool:
    """True when SymPy is importable (``vincio[verify]``)."""
    try:
        import sympy  # noqa: F401
    except ImportError:
        return False
    return True


def _require(module: str, available: bool) -> None:
    if not available:
        raise LoaderError(
            f"{module} is required for this verifier; install it with "
            'pip install "vincio[verify]"'
        )


class SmtConstraintVerifier:
    """Proves a constraint system is **consistent** with the answer via Z3.

    Where :class:`~vincio.verify.kernels.ConstraintVerifier` checks one assignment,
    this asks the solver whether the constraints (as ``var op bound`` comparisons)
    are jointly satisfiable and whether the answer's assignment is a model — a
    refutation when the system is unsatisfiable or the assignment is not a model.
    Requires ``vincio[verify]``.
    """

    kind = "smt"

    def __init__(self, constraints: list[Constraint] | None = None) -> None:
        self._constraints = list(constraints or [])

    def check(self, answer: Any, context: VerificationContext) -> list[Check]:
        _require("z3-solver", smt_available())
        import z3

        comparisons = self._comparisons(context)
        if not comparisons:
            return [Check(name="smt", kind=self.kind, status="inapplicable",
                          detail="no declarative comparison constraints")]
        assignment = answer if isinstance(answer, dict) else context.facts
        solver = z3.Solver()
        variables: dict[str, Any] = {}
        for var, op, bound in comparisons:
            sym = variables.setdefault(var, z3.Real(var))
            solver.add(self._z3_relation(sym, op, bound))
        consistent = solver.check() == z3.sat
        if not consistent:
            return [Check(name="smt", kind=self.kind, status="refuted",
                          detail="constraint system is unsatisfiable")]
        # The assignment must itself be a model.
        model = z3.Solver()
        for var, op, bound in comparisons:
            model.add(self._z3_relation(z3.Real(var), op, bound))
        if isinstance(assignment, dict):
            for var, value in assignment.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    model.add(z3.Real(var) == value)
        ok = model.check() == z3.sat
        return [Check(name="smt", kind=self.kind, status="verified" if ok else "refuted",
                      detail="assignment models the constraints" if ok
                      else "assignment is not a model of the constraints")]

    @staticmethod
    def _comparisons(context: VerificationContext) -> list[tuple[str, str, Any]]:
        out: list[tuple[str, str, Any]] = []
        for c in context.constraints:
            spec = getattr(c, "_smt", None)
            if spec is not None:
                out.append(spec)
        return out

    @staticmethod
    def _z3_relation(sym: Any, op: str, bound: Any) -> Any:
        return {
            "==": sym == bound, "!=": sym != bound,
            "<": sym < bound, "<=": sym <= bound,
            ">": sym > bound, ">=": sym >= bound,
        }[op]


class CasArithmeticVerifier:
    """Re-checks ``a op b = c`` equalities with **exact** rational arithmetic via SymPy.

    Catches the rounding the float-tolerant native kernel would accept (a
    long-division equality stated to too few digits), proving the equality exactly.
    Requires ``vincio[verify]``.
    """

    kind = "cas"

    def check(self, answer: Any, context: VerificationContext) -> list[Check]:
        _require("sympy", cas_available())
        import sympy

        text = answer if isinstance(answer, str) else str(answer)
        checks: list[Check] = []
        for match in _EQUALITY_RE.finditer(text):
            lhs = match.group("lhs").strip()
            if not re.search(r"[+\-*/%]", lhs):
                continue
            try:
                computed = sympy.Rational(sympy.sympify(lhs, rational=True))
                claimed = sympy.Rational(match.group("rhs"))
            except (sympy.SympifyError, ValueError, TypeError):
                continue
            ok = bool(sympy.simplify(computed - claimed) == 0)
            checks.append(Check(
                name="cas_equality", kind=self.kind,
                status="verified" if ok else "refuted",
                detail=f"{lhs} = {computed} (exact), claimed {claimed}",
            ))
        if not checks:
            return [Check(name="cas", kind=self.kind, status="inapplicable",
                          detail="no arithmetic equality found")]
        return checks
