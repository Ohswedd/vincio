"""Deterministic verification kernels.

Each kernel is a :class:`~vincio.verify.certificates.ReasoningVerifier` that
recomputes a checkable class of claim and emits a sound :class:`Check`:

* :class:`ArithmeticVerifier` — recomputes arithmetic equalities (``a op b = c``,
  ``n% of m = k``) with a safe expression evaluator (never ``eval``).
* :class:`UnitVerifier` — checks unit conversions and refuses a dimensional
  mismatch (``5 km = 5000 m`` holds; ``5 km = 5000 kg`` is refuted).
* :class:`TemporalVerifier` — checks date ordering and duration claims against a
  real calendar.
* :class:`ConstraintVerifier` — checks an assignment satisfies a set of typed
  constraints (the constraint-satisfaction / SAT-style kernel).
* :class:`SchemaVerifier` — checks structural conformance to a JSON schema.
* :class:`CitationVerifier` — checks every verifiable claim is entailed by cited
  evidence (reusing the deterministic strict-support kernel).

All kernels are pure, offline, and dependency-free. An optional SMT/CAS backend
(:mod:`vincio.verify.smt`) sits behind ``vincio[verify]`` for the cases that need
a solver; the kernels here are the default and need no extra.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from ..core.types import EvidenceItem
from ..evals.metrics import _supported_strict, _verifiable_claims
from ..tools.runtime import validate_against_schema
from .certificates import Check, VerificationContext

__all__ = [
    "ArithmeticVerifier",
    "UnitVerifier",
    "TemporalVerifier",
    "ConstraintVerifier",
    "SchemaVerifier",
    "CitationVerifier",
    "Constraint",
    "default_verifiers",
    "safe_eval_arithmetic",
]

_TOLERANCE = 1e-6


# --------------------------------------------------------------------------- #
# Safe arithmetic evaluator (recursive descent over numbers and + - * / % ( )) #
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"\s*(?:(\d+(?:\.\d+)?)|([()+\-*/%]))")


def _tokenize_expr(text: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        match = _TOKEN_RE.match(text, pos)
        if not match or match.start() != pos:
            raise ValueError(f"unexpected character at {pos!r}")
        tokens.append(match.group(1) or match.group(2))
        pos = match.end()
    return tokens


def safe_eval_arithmetic(expression: str) -> float:
    """Evaluate a numeric arithmetic expression deterministically and safely.

    Supports ``+ - * / %`` and parentheses over decimal numbers via a recursive
    descent parser — never the Python ``eval``. ``%`` is the modulo operator.
    Raises :class:`ValueError` on a malformed or non-numeric expression.
    """
    tokens = _tokenize_expr(expression)
    if not tokens:
        raise ValueError("empty expression")
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def consume() -> str:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def parse_atom() -> float:
        tok = peek()
        if tok == "(":
            consume()
            value = parse_expr()
            if peek() != ")":
                raise ValueError("unbalanced parentheses")
            consume()
            return value
        if tok == "-":
            consume()
            return -parse_atom()
        if tok == "+":
            consume()
            return parse_atom()
        if tok is None or not re.fullmatch(r"\d+(?:\.\d+)?", tok):
            raise ValueError(f"expected a number, got {tok!r}")
        return float(consume())

    def parse_term() -> float:
        value = parse_atom()
        while peek() in ("*", "/", "%"):
            op = consume()
            rhs = parse_atom()
            if op == "*":
                value *= rhs
            elif op == "/":
                if rhs == 0:
                    raise ValueError("division by zero")
                value /= rhs
            else:
                if rhs == 0:
                    raise ValueError("modulo by zero")
                value %= rhs
        return value

    def parse_expr() -> float:
        value = parse_term()
        while peek() in ("+", "-"):
            op = consume()
            rhs = parse_term()
            value = value + rhs if op == "+" else value - rhs
        return value

    result = parse_expr()
    if pos != len(tokens):
        raise ValueError("trailing tokens in expression")
    return result


# --------------------------------------------------------------------------- #
# Arithmetic                                                                    #
# --------------------------------------------------------------------------- #

# ``a op b = c``, with optional surrounding text. The left side is any run of
# numbers, operators, parentheses and whitespace; the right side is one number.
_EQUALITY_RE = re.compile(
    r"(?P<lhs>[-+]?(?:\d[\d\s.+\-*/%()]*\d|\d))\s*=\s*(?P<rhs>[-+]?\d+(?:\.\d+)?)"
)
_PERCENT_OF_RE = re.compile(
    r"(?P<pct>\d+(?:\.\d+)?)\s*%\s*of\s*(?P<base>\d+(?:\.\d+)?)\s*(?:is|=|equals)\s*"
    r"(?P<result>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# An ISO date (``2024-01-05``) is digits joined by ``-``, which the arithmetic
# parser would otherwise read as subtraction; those claims belong to the temporal
# kernel, so the arithmetic kernel skips an equality whose left side contains one.
_DATE_IN_EXPR_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _TOLERANCE + 1e-9 * max(abs(a), abs(b))


class ArithmeticVerifier:
    """Recomputes arithmetic equalities stated in an answer.

    Extracts ``a op b = c`` and ``n% of m is k`` claims and recomputes the left
    side with :func:`safe_eval_arithmetic`; a mismatch is a refutation. With no
    arithmetic claim the check is inapplicable.
    """

    kind = "arithmetic"

    def check(self, answer: Any, context: VerificationContext) -> list[Check]:
        text = answer if isinstance(answer, str) else str(answer)
        checks: list[Check] = []
        for match in _PERCENT_OF_RE.finditer(text):
            pct = float(match.group("pct"))
            base = float(match.group("base"))
            claimed = float(match.group("result"))
            computed = pct / 100.0 * base
            ok = _close(computed, claimed)
            checks.append(
                Check(
                    name="percent_of",
                    kind=self.kind,
                    status="verified" if ok else "refuted",
                    detail=f"{pct}% of {base} = {computed:g}, claimed {claimed:g}",
                    evidence={"computed": computed, "claimed": claimed},
                )
            )
        for match in _EQUALITY_RE.finditer(text):
            lhs = match.group("lhs").strip()
            if not re.search(r"[+\-*/%]", lhs):
                continue  # "x = 5" with no operation is an assignment, not a claim
            if _DATE_IN_EXPR_RE.search(lhs):
                continue  # an ISO date is a temporal claim, not subtraction
            rhs = float(match.group("rhs"))
            try:
                computed = safe_eval_arithmetic(lhs)
            except ValueError:
                continue
            ok = _close(computed, rhs)
            checks.append(
                Check(
                    name="equality",
                    kind=self.kind,
                    status="verified" if ok else "refuted",
                    detail=f"{lhs} = {computed:g}, claimed {rhs:g}",
                    evidence={"expression": lhs, "computed": computed, "claimed": rhs},
                )
            )
        if not checks:
            return [Check(name="arithmetic", kind=self.kind, status="inapplicable",
                          detail="no arithmetic equality found")]
        return checks


# --------------------------------------------------------------------------- #
# Units                                                                         #
# --------------------------------------------------------------------------- #

# factor to the dimension's base unit.
_UNITS: dict[str, tuple[str, float]] = {
    # length (base: metre)
    "mm": ("length", 0.001), "cm": ("length", 0.01), "m": ("length", 1.0),
    "km": ("length", 1000.0), "in": ("length", 0.0254), "ft": ("length", 0.3048),
    "mi": ("length", 1609.344),
    # mass (base: gram)
    "mg": ("mass", 0.001), "g": ("mass", 1.0), "kg": ("mass", 1000.0),
    "t": ("mass", 1_000_000.0), "lb": ("mass", 453.59237),
    # time (base: second)
    "ms": ("time", 0.001), "s": ("time", 1.0), "sec": ("time", 1.0), "min": ("time", 60.0),
    "h": ("time", 3600.0), "hr": ("time", 3600.0), "day": ("time", 86400.0),
    "days": ("time", 86400.0),
    # data (base: byte)
    "b": ("data", 1.0), "byte": ("data", 1.0), "kb": ("data", 1000.0),
    "mb": ("data", 1_000_000.0), "gb": ("data", 1_000_000_000.0),
    "tb": ("data", 1_000_000_000_000.0),
}

_CONVERSION_RE = re.compile(
    r"(?P<x>\d+(?:\.\d+)?)\s*(?P<u1>[a-zA-Z]+)\s*(?:=|is|equals)\s*"
    r"(?P<y>\d+(?:\.\d+)?)\s*(?P<u2>[a-zA-Z]+)",
)


class UnitVerifier:
    """Checks unit conversions and refuses a dimensional mismatch.

    Extracts ``X unit1 = Y unit2`` and recomputes the conversion against a small
    unit table (length / mass / time / data). A converted value that disagrees is
    refuted; a conversion across **incompatible dimensions** (``5 km = 5000 kg``)
    is refuted as a dimensional error. Unknown units are skipped (inapplicable).
    """

    kind = "units"

    def check(self, answer: Any, context: VerificationContext) -> list[Check]:
        text = answer if isinstance(answer, str) else str(answer)
        checks: list[Check] = []
        for match in _CONVERSION_RE.finditer(text):
            u1 = match.group("u1").lower()
            u2 = match.group("u2").lower()
            if u1 not in _UNITS or u2 not in _UNITS:
                continue
            x = float(match.group("x"))
            y = float(match.group("y"))
            dim1, f1 = _UNITS[u1]
            dim2, f2 = _UNITS[u2]
            if dim1 != dim2:
                checks.append(
                    Check(
                        name="dimension",
                        kind=self.kind,
                        status="refuted",
                        detail=f"{u1} is {dim1}, {u2} is {dim2}: dimensional mismatch",
                        evidence={"u1": u1, "u2": u2, "dim1": dim1, "dim2": dim2},
                    )
                )
                continue
            computed = x * f1 / f2
            ok = _close(computed, y)
            checks.append(
                Check(
                    name="conversion",
                    kind=self.kind,
                    status="verified" if ok else "refuted",
                    detail=f"{x} {u1} = {computed:g} {u2}, claimed {y:g} {u2}",
                    evidence={"computed": computed, "claimed": y},
                )
            )
        if not checks:
            return [Check(name="units", kind=self.kind, status="inapplicable",
                          detail="no unit conversion found")]
        return checks


# --------------------------------------------------------------------------- #
# Temporal                                                                      #
# --------------------------------------------------------------------------- #

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DURATION_RE = re.compile(
    r"from\s+(?P<a>\d{4}-\d{2}-\d{2})\s+to\s+(?P<b>\d{4}-\d{2}-\d{2})\s+is\s+"
    r"(?P<n>\d+)\s+days?",
    re.IGNORECASE,
)
_ORDER_RE = re.compile(
    r"(?P<a>\d{4}-\d{2}-\d{2})\s+is\s+(?P<rel>before|after)\s+(?P<b>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def _parse_date(text: str) -> date | None:
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


class TemporalVerifier:
    """Checks date ordering and duration claims against a real calendar.

    Recomputes ``from A to B is N days`` durations and ``A is before/after B``
    orderings using actual date arithmetic, so an off-by-one or a reversed
    ordering is refuted. With no temporal claim the check is inapplicable.
    """

    kind = "temporal"

    def check(self, answer: Any, context: VerificationContext) -> list[Check]:
        text = answer if isinstance(answer, str) else str(answer)
        checks: list[Check] = []
        for match in _DURATION_RE.finditer(text):
            a = _parse_date(match.group("a"))
            b = _parse_date(match.group("b"))
            if a is None or b is None:
                continue
            claimed = int(match.group("n"))
            computed = (b - a).days
            ok = computed == claimed
            checks.append(
                Check(
                    name="duration",
                    kind=self.kind,
                    status="verified" if ok else "refuted",
                    detail=f"{a} to {b} is {computed} days, claimed {claimed}",
                    evidence={"computed": computed, "claimed": claimed},
                )
            )
        for match in _ORDER_RE.finditer(text):
            a = _parse_date(match.group("a"))
            b = _parse_date(match.group("b"))
            if a is None or b is None:
                continue
            rel = match.group("rel").lower()
            ok = (a < b) if rel == "before" else (a > b)
            checks.append(
                Check(
                    name="ordering",
                    kind=self.kind,
                    status="verified" if ok else "refuted",
                    detail=f"{a} {rel} {b} is {ok}",
                    evidence={"a": str(a), "b": str(b), "relation": rel},
                )
            )
        if not checks:
            return [Check(name="temporal", kind=self.kind, status="inapplicable",
                          detail="no temporal claim found")]
        return checks


# --------------------------------------------------------------------------- #
# Constraints (constraint-satisfaction / SAT-style)                            #
# --------------------------------------------------------------------------- #

_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "in": lambda a, b: a in b,
    "not in": lambda a, b: a not in b,
}


class Constraint:
    """One declarative constraint over a variable assignment.

    Built via the constructors :meth:`compare` (a relation between a variable and
    a bound, e.g. ``x <= 10``), :meth:`all_different` (a set of variables must
    take distinct values), and :meth:`predicate` (an arbitrary boolean function of
    the assignment). :class:`ConstraintVerifier` evaluates each over a candidate
    assignment; a violated constraint is a refutation.
    """

    def __init__(self, description: str, fn: Any) -> None:
        self.description = description
        self._fn = fn
        # Declarative ``(var, op, bound)`` form for the optional SMT backend; set
        # only by :meth:`compare` (the cases a solver can reason over symbolically).
        self._smt: tuple[str, str, Any] | None = None

    def satisfied_by(self, assignment: dict[str, Any]) -> bool:
        """True when the assignment satisfies this constraint."""
        return bool(self._fn(assignment))

    @classmethod
    def compare(cls, var: str, op: str, bound: Any) -> Constraint:
        """A relation ``var op bound`` (op in ==, !=, <, <=, >, >=, in, not in)."""
        if op not in _OPS:
            raise ValueError(f"unknown operator {op!r}; choose from {sorted(_OPS)}")
        checker = _OPS[op]
        constraint = cls(f"{var} {op} {bound!r}", lambda a: var in a and checker(a[var], bound))
        if op in {"==", "!=", "<", "<=", ">", ">="}:
            constraint._smt = (var, op, bound)
        return constraint

    @classmethod
    def all_different(cls, variables: list[str]) -> Constraint:
        """All listed variables must take distinct values."""
        def fn(a: dict[str, Any]) -> bool:
            values = [a[v] for v in variables if v in a]
            return len(values) == len(set(values))
        return cls(f"all_different({', '.join(variables)})", fn)

    @classmethod
    def predicate(cls, description: str, fn: Any) -> Constraint:
        """An arbitrary boolean function ``assignment -> bool``."""
        return cls(description, fn)


class ConstraintVerifier:
    """Checks a candidate assignment satisfies a set of typed constraints.

    The assignment is the ``answer`` (a mapping of variable to value); the
    constraints come from :attr:`VerificationContext.constraints`. Every satisfied
    constraint is a verified check and a single violated one is a refutation, so
    an assignment that breaks any constraint cannot carry a verified certificate.
    With no constraints the check is inapplicable.
    """

    kind = "constraints"

    def __init__(self, constraints: list[Constraint] | None = None) -> None:
        self._constraints = list(constraints or [])

    def check(self, answer: Any, context: VerificationContext) -> list[Check]:
        constraints = self._constraints or [
            c for c in context.constraints if isinstance(c, Constraint)
        ]
        if not constraints:
            return [Check(name="constraints", kind=self.kind, status="inapplicable",
                          detail="no constraints supplied")]
        assignment = answer if isinstance(answer, dict) else context.facts
        if not isinstance(assignment, dict):
            return [Check(name="constraints", kind=self.kind, status="inapplicable",
                          detail="answer is not an assignment")]
        checks: list[Check] = []
        for constraint in constraints:
            try:
                ok = constraint.satisfied_by(assignment)
            except Exception as exc:  # noqa: BLE001 - a faulty predicate refutes, not crashes
                checks.append(Check(name=constraint.description, kind=self.kind,
                                    status="refuted", detail=f"predicate error: {exc}"))
                continue
            checks.append(
                Check(
                    name=constraint.description,
                    kind=self.kind,
                    status="verified" if ok else "refuted",
                    detail="satisfied" if ok else "violated",
                )
            )
        return checks


# --------------------------------------------------------------------------- #
# Schema                                                                        #
# --------------------------------------------------------------------------- #


class SchemaVerifier:
    """Checks an answer structurally conforms to a JSON schema.

    The schema comes from :attr:`VerificationContext.schema` (or is passed at
    construction); conformance is decided by the same deterministic validator the
    tool runtime uses. A structural violation is refuted; no schema is inapplicable.
    """

    kind = "schema"

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        self._schema = schema

    def check(self, answer: Any, context: VerificationContext) -> list[Check]:
        schema = self._schema or context.schema_
        if not schema:
            return [Check(name="schema", kind=self.kind, status="inapplicable",
                          detail="no schema supplied")]
        value = answer
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        errors = validate_against_schema(value, schema)
        if errors:
            return [Check(name="schema", kind=self.kind, status="refuted",
                          detail="; ".join(errors[:5]), evidence={"errors": errors})]
        return [Check(name="schema", kind=self.kind, status="verified",
                      detail="conforms to schema")]


# --------------------------------------------------------------------------- #
# Citation entailment                                                          #
# --------------------------------------------------------------------------- #


def _claim_text(answer: Any) -> str:
    if isinstance(answer, str):
        return answer
    if hasattr(answer, "model_dump"):
        dumped = answer.model_dump(mode="json")
        return dumped.get("text", str(dumped)) if isinstance(dumped, dict) else str(dumped)
    if isinstance(answer, dict):
        return str(answer.get("text", answer))
    return str(answer)


class CitationVerifier:
    """Checks every verifiable claim in an answer is entailed by cited evidence.

    Reuses the deterministic strict-support kernel (lexical support **plus** every
    number in the claim must appear in the supporting evidence, so a numeric
    contradiction is caught). A verifiable claim not supported by the supplied
    evidence is refuted; with no evidence or no verifiable claim the check is
    inapplicable. Citation markers are honoured when present but not required —
    support is decided against the evidence set.
    """

    kind = "citation"

    def __init__(self, evidence: list[EvidenceItem] | None = None) -> None:
        self._evidence = evidence

    def check(self, answer: Any, context: VerificationContext) -> list[Check]:
        evidence = self._evidence if self._evidence is not None else context.evidence
        if not evidence:
            return [Check(name="citation", kind=self.kind, status="inapplicable",
                          detail="no evidence supplied")]
        claims = _verifiable_claims(_claim_text(answer))
        if not claims:
            return [Check(name="citation", kind=self.kind, status="inapplicable",
                          detail="no verifiable claim found")]
        checks: list[Check] = []
        for claim in claims:
            supported = _supported_strict(claim, evidence)
            checks.append(
                Check(
                    name="entailment",
                    kind=self.kind,
                    status="verified" if supported else "refuted",
                    detail=("entailed by evidence" if supported
                            else "not entailed by any cited evidence"),
                    evidence={"claim": claim[:200]},
                )
            )
        return checks


def default_verifiers() -> list[Any]:
    """The default offline kernel set behind ``app.verify_reasoning``.

    Arithmetic, units, temporal, schema, constraints, and citation — every kernel
    is deterministic and dependency-free, returning ``inapplicable`` when its kind
    of claim is absent, so the set is safe to run over any answer.
    """
    return [
        ArithmeticVerifier(),
        UnitVerifier(),
        TemporalVerifier(),
        ConstraintVerifier(),
        SchemaVerifier(),
        CitationVerifier(),
    ]
