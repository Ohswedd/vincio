"""Verified tool use & synthesized programs.

Two kinds of proof-carrying code in the tool plane:

* :class:`ToolContract` — pre- and post-conditions a tool declares as a contract
  on its behaviour (not merely its schema). The runtime checks :meth:`check_pre`
  against the actual arguments before the call and :meth:`check_post` against the
  actual result after it; a breach raises
  :class:`~vincio.core.errors.ToolContractError`, so a tool that returns an
  out-of-contract value is caught at the boundary.
* :class:`SynthesizedProgram` — a small, **verified** data transform built from a
  whitelisted, deterministic op set (no ``eval``, no I/O). :func:`synthesize`
  runs it on representative examples, checks its declared properties (schema
  conformance, row-count relations, field invariants), and binds the verdict into
  the same :class:`~vincio.verify.certificates.Certificate` an answer carries —
  the program's properties are proven before it is allowed to run.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..tools.runtime import validate_against_schema
from .certificates import Certificate, Check, build_certificate
from .kernels import safe_eval_arithmetic

__all__ = [
    "ToolClause",
    "ToolContract",
    "ProgramOp",
    "ProgramProperty",
    "ProgramSpec",
    "SynthesizedProgram",
    "synthesize",
]


# --------------------------------------------------------------------------- #
# Tool contracts                                                               #
# --------------------------------------------------------------------------- #


class ToolClause(BaseModel):
    """One named pre- or post-condition over a tool call.

    ``predicate`` returns a truthy value when the clause **holds**; a falsy value
    is a breach, and a returned string is used as the breach message. A
    pre-condition predicate takes the arguments mapping; a post-condition
    predicate takes ``(arguments, result)``.
    """

    model_config = {"arbitrary_types_allowed": True}

    description: str
    predicate: Callable[..., Any]

    def evaluate(self, *args: Any) -> str | None:
        """Return ``None`` when the clause holds, else a breach message."""
        try:
            outcome = self.predicate(*args)
        except Exception as exc:  # noqa: BLE001 - a faulty predicate is a breach, not a crash
            return f"{self.description}: predicate error: {exc}"
        if outcome:
            return None
        return self.description if not isinstance(outcome, str) else outcome


class ToolContract(BaseModel):
    """Pre- and post-conditions checked against a tool's actual call and result.

    ``requires`` clauses are checked against the arguments before execution;
    ``ensures`` clauses against ``(arguments, result)`` after it. Build clauses
    with :meth:`requires_that` / :meth:`ensures_that`, or pass :class:`ToolClause`
    lists directly. The runtime raises on the first breach, so the contract is an
    enforced boundary, not documentation.
    """

    model_config = {"arbitrary_types_allowed": True}

    requires: list[ToolClause] = Field(default_factory=list)
    ensures: list[ToolClause] = Field(default_factory=list)

    def requires_that(self, description: str, predicate: Callable[[dict[str, Any]], Any]) -> ToolContract:
        """Add a pre-condition over the arguments mapping."""
        self.requires.append(ToolClause(description=description, predicate=predicate))
        return self

    def ensures_that(
        self, description: str, predicate: Callable[[dict[str, Any], Any], Any]
    ) -> ToolContract:
        """Add a post-condition over ``(arguments, result)``."""
        self.ensures.append(ToolClause(description=description, predicate=predicate))
        return self

    def check_pre(self, arguments: dict[str, Any]) -> list[str]:
        """Return the pre-condition breaches for ``arguments`` (empty when clean)."""
        return [m for c in self.requires if (m := c.evaluate(arguments)) is not None]

    def check_post(self, arguments: dict[str, Any], result: Any) -> list[str]:
        """Return the post-condition breaches for ``(arguments, result)``."""
        return [m for c in self.ensures if (m := c.evaluate(arguments, result)) is not None]


# --------------------------------------------------------------------------- #
# Synthesized programs                                                         #
# --------------------------------------------------------------------------- #

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
OpKind = Literal["select", "rename", "derive", "filter"]
_FILTER_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


class ProgramOp(BaseModel):
    """One whitelisted transform step over a list of record dicts.

    Kinds:

    * ``select`` — keep only ``fields``.
    * ``rename`` — rename keys per ``mapping`` (``{old: new}``).
    * ``derive`` — set ``field`` to a safe arithmetic ``expr`` over existing
      numeric fields (e.g. ``price * quantity``); evaluated without ``eval``.
    * ``filter`` — keep rows where ``field`` compares to ``value`` under ``op``.

    Every op is pure and deterministic; there is no attribute access, call, or
    I/O — only the declared transform.
    """

    op: OpKind
    fields: list[str] = Field(default_factory=list)
    mapping: dict[str, str] = Field(default_factory=dict)
    field: str = ""
    expr: str = ""
    op_symbol: str = "=="
    value: Any = None

    def apply(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply this op to ``rows`` and return the transformed rows."""
        if self.op == "select":
            return [{k: r[k] for k in self.fields if k in r} for r in rows]
        if self.op == "rename":
            return [{self.mapping.get(k, k): v for k, v in r.items()} for r in rows]
        if self.op == "derive":
            out = []
            for r in rows:
                row = dict(r)
                row[self.field] = self._eval(self.expr, r)
                out.append(row)
            return out
        if self.op == "filter":
            checker = _FILTER_OPS.get(self.op_symbol)
            if checker is None:
                raise ValueError(f"unknown filter operator {self.op_symbol!r}")
            return [r for r in rows if self.field in r and checker(r[self.field], self.value)]
        raise ValueError(f"unknown op {self.op!r}")

    @staticmethod
    def _eval(expr: str, record: dict[str, Any]) -> float:
        def replace(match: re.Match[str]) -> str:
            name = match.group(0)
            if name not in record:
                raise ValueError(f"derive references unknown field {name!r}")
            value = record[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"derive field {name!r} is not numeric")
            return repr(float(value))

        substituted = _IDENT_RE.sub(replace, expr)
        return safe_eval_arithmetic(substituted)


class ProgramProperty(BaseModel):
    """A declarative property a synthesized program must satisfy.

    Kinds:

    * ``schema`` — every output row conforms to ``schema``.
    * ``row_count`` — output length relates to input length by ``relation`` (one of
      ``preserved``, ``le``, ``ge``).
    * ``field_nonnegative`` — ``field`` is ``>= 0`` in every output row.
    * ``field_range`` — ``minimum <= field <= maximum`` in every output row.
    """

    kind: Literal["schema", "row_count", "field_nonnegative", "field_range"]
    field: str = ""
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    relation: Literal["preserved", "le", "ge"] = "preserved"
    minimum: float | None = None
    maximum: float | None = None

    model_config = {"populate_by_name": True}

    def evaluate(
        self, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        """Return ``(holds, detail)`` for this property over a run's in/out rows."""
        if self.kind == "schema":
            if not self.schema_:
                return False, "schema property has no schema"
            for i, row in enumerate(outputs):
                errors = validate_against_schema(row, self.schema_)
                if errors:
                    return False, f"row {i} violates schema: {errors[0]}"
            return True, f"all {len(outputs)} rows conform to schema"
        if self.kind == "row_count":
            ok = {
                "preserved": len(outputs) == len(inputs),
                "le": len(outputs) <= len(inputs),
                "ge": len(outputs) >= len(inputs),
            }[self.relation]
            return ok, f"out={len(outputs)} {self.relation} in={len(inputs)}"
        if self.kind == "field_nonnegative":
            for i, row in enumerate(outputs):
                v = row.get(self.field)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
                    return False, f"row {i} field {self.field!r} = {v} < 0"
            return True, f"field {self.field!r} non-negative in all rows"
        # field_range
        for i, row in enumerate(outputs):
            v = row.get(self.field)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if self.minimum is not None and v < self.minimum:
                return False, f"row {i} field {self.field!r} = {v} < {self.minimum}"
            if self.maximum is not None and v > self.maximum:
                return False, f"row {i} field {self.field!r} = {v} > {self.maximum}"
        return True, f"field {self.field!r} within range in all rows"


class ProgramSpec(BaseModel):
    """The declaration of a verified transform: its ops and the properties it must hold."""

    name: str
    ops: list[ProgramOp] = Field(default_factory=list)
    properties: list[ProgramProperty] = Field(default_factory=list)

    def transform(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply the op pipeline to ``rows`` (no property checking)."""
        out = [dict(r) for r in rows]
        for op in self.ops:
            out = op.apply(out)
        return out


class SynthesizedProgram(BaseModel):
    """A verified transform paired with the certificate proving its properties.

    :attr:`holds` is true only when the certificate verified. :meth:`run` re-applies
    the verified ops to new records and, by default, re-checks the declared
    properties on the output — so a property that the synthesis examples proved is
    re-asserted at every use, raising
    :class:`~vincio.core.errors.ProgramSynthesisError` if it ever fails to hold.
    """

    model_config = {"arbitrary_types_allowed": True}

    spec: ProgramSpec
    certificate: Certificate

    @property
    def holds(self) -> bool:
        """True when the synthesis certificate positively verified."""
        return self.certificate.holds

    def run(self, rows: list[dict[str, Any]], *, recheck: bool = True) -> list[dict[str, Any]]:
        """Apply the verified transform, re-checking properties unless disabled."""
        from ..core.errors import ProgramSynthesisError

        outputs = self.spec.transform(rows)
        if recheck:
            for prop in self.spec.properties:
                ok, detail = prop.evaluate(rows, outputs)
                if not ok:
                    raise ProgramSynthesisError(
                        f"program {self.spec.name!r} violated property {prop.kind!r} at run time: "
                        f"{detail}"
                    )
        return outputs


def synthesize(
    spec: ProgramSpec,
    examples: list[dict[str, Any]],
    *,
    require: bool = True,
) -> SynthesizedProgram:
    """Verify ``spec``'s properties on ``examples`` and emit a proof-carrying program.

    Runs the op pipeline on the representative ``examples``, checks every declared
    :class:`ProgramProperty` on the output, and binds the verdicts into a
    content-bound :class:`Certificate`. With ``require`` (the default), a refuted
    property raises :class:`~vincio.core.errors.ProgramSynthesisError` — the
    program is not returned unless its properties are proven; pass ``require=False``
    to inspect a refuted certificate instead.
    """
    from ..core.errors import ProgramSynthesisError

    try:
        outputs = spec.transform(examples)
        transform_check = Check(
            name="transform", kind="program", status="verified",
            detail=f"pipeline of {len(spec.ops)} op(s) ran on {len(examples)} example(s)",
        )
    except Exception as exc:  # noqa: BLE001 - a malformed op refutes the program
        cert = build_certificate(
            spec.name,
            [Check(name="transform", kind="program", status="refuted", detail=str(exc))],
            kinds=["program"],
        )
        if require:
            raise ProgramSynthesisError(
                f"program {spec.name!r} failed to run on examples: {exc}"
            ) from exc
        return SynthesizedProgram(spec=spec, certificate=cert)

    checks: list[Check] = [transform_check]
    for prop in spec.properties:
        ok, detail = prop.evaluate(examples, outputs)
        checks.append(Check(
            name=f"property:{prop.kind}" + (f":{prop.field}" if prop.field else ""),
            kind="program",
            status="verified" if ok else "refuted",
            detail=detail,
        ))
    cert = build_certificate(spec.name, checks, kinds=["program"])
    if require and cert.refuted:
        raise ProgramSynthesisError(
            f"program {spec.name!r} refuted: "
            + "; ".join(c.detail for c in cert.refutations)
        )
    return SynthesizedProgram(spec=spec, certificate=cert)
