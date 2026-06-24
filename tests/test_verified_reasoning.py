"""Verified reasoning & neuro-symbolic certificates.

Covers proof-carrying answers (the deterministic kernels, certificate soundness
and content-binding, the refuse-or-repair loop), runtime verification & shielding
(behavior specs, the step-by-step monitor, the action-blocking shield wired into
the tool runtime), and verified tool use (pre/post-condition contracts and
proof-carrying synthesized programs).
"""

from __future__ import annotations

import pytest

from vincio import (
    ArithmeticVerifier,
    BehaviorEvent,
    BehaviorSpec,
    CitationVerifier,
    CompositeVerifier,
    ConstraintVerifier,
    ContextApp,
    EventPattern,
    ProgramOp,
    ProgramProperty,
    ProgramSpec,
    RuntimeMonitor,
    SchemaVerifier,
    Shield,
    TemporalVerifier,
    ToolContract,
    UnitVerifier,
    VincioConfig,
    synthesize,
)
from vincio.core.errors import (
    BehaviorViolationError,  # noqa: F401 - exported surface check
    CertificateRefutedError,
    ProgramSynthesisError,
    ReasoningVerificationError,
    ToolContractError,
)
from vincio.core.types import EvidenceItem, ToolCall
from vincio.providers import MockProvider
from vincio.verify import Constraint, VerificationContext, build_certificate, derive_status
from vincio.verify.certificates import Check
from vincio.verify.kernels import safe_eval_arithmetic


def _app() -> ContextApp:
    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    return ContextApp(name="solver", provider=MockProvider(default_text="ok"), config=cfg)


# --------------------------------------------------------------------------- #
# Certificate core                                                            #
# --------------------------------------------------------------------------- #


def test_derive_status_precedence():
    assert derive_status([Check(name="a", kind="k", status="verified")]) == "verified"
    assert derive_status([
        Check(name="a", kind="k", status="verified"),
        Check(name="b", kind="k", status="refuted"),
    ]) == "refuted"  # a single refutation sinks it
    assert derive_status([Check(name="a", kind="k", status="inapplicable")]) == "inapplicable"
    assert derive_status([]) == "inapplicable"


def test_certificate_is_content_bound_and_tamper_evident():
    cert = build_certificate("2 + 2 = 5", [
        Check(name="equality", kind="arithmetic", status="refuted", detail="4 != 5"),
    ])
    assert cert.verify()
    # Flip the verdict and re-seal the status: the re-derivation catches it.
    cert.checks[0].status = "verified"
    cert.status = "verified"
    assert not cert.verify()


def test_certificate_hash_catches_edited_check():
    cert = build_certificate("x", [Check(name="a", kind="k", status="verified")])
    cert.checks[0].detail = "tampered"
    assert not cert.verify()


# --------------------------------------------------------------------------- #
# Arithmetic / units / temporal kernels                                       #
# --------------------------------------------------------------------------- #


def test_arithmetic_kernel_verifies_and_refutes():
    cv = CompositeVerifier([ArithmeticVerifier()])
    assert cv.certify("We get 12 * 3 = 36 and 10% of 200 is 20.").status == "verified"
    assert cv.certify("So 2 + 2 = 5.").status == "refuted"
    assert cv.certify("no math here").status == "inapplicable"


def test_arithmetic_ignores_plain_assignment():
    # "x = 5" with no operator is an assignment, not a checkable equality.
    cv = CompositeVerifier([ArithmeticVerifier()])
    assert cv.certify("let x = 5").status == "inapplicable"


def test_unit_kernel_dimensional_mismatch_is_refuted():
    cv = CompositeVerifier([UnitVerifier()])
    assert cv.certify("5 km = 5000 m").status == "verified"
    assert cv.certify("2 h = 120 min").status == "verified"
    assert cv.certify("5 km = 4000 m").status == "refuted"
    assert cv.certify("5 km = 5000 kg").status == "refuted"  # dimensional error


def test_arithmetic_does_not_misread_iso_dates():
    # An ISO date "2024-01-05" must not be read as subtraction (2024 - 1 - 5).
    cv = CompositeVerifier([ArithmeticVerifier()])
    assert cv.certify("The meeting on 2024-01-05 = 12 attendees.").status == "inapplicable"


def test_temporal_kernel_uses_real_calendar():
    cv = CompositeVerifier([TemporalVerifier()])
    assert cv.certify("from 2024-01-01 to 2024-01-08 is 7 days").status == "verified"
    assert cv.certify("from 2024-01-01 to 2024-01-08 is 5 days").status == "refuted"
    assert cv.certify("2024-01-01 is before 2024-02-01").status == "verified"
    assert cv.certify("2024-03-01 is before 2024-02-01").status == "refuted"


# --------------------------------------------------------------------------- #
# Constraint / schema / citation kernels                                      #
# --------------------------------------------------------------------------- #


def test_constraint_kernel_checks_assignment():
    cv = CompositeVerifier([ConstraintVerifier()])
    ctx = VerificationContext(constraints=[
        Constraint.compare("x", "<=", 10),
        Constraint.compare("x", ">", 0),
        Constraint.all_different(["a", "b"]),
    ])
    assert cv.certify({"x": 5, "a": 1, "b": 2}, ctx).status == "verified"
    assert cv.certify({"x": 50, "a": 1, "b": 2}, ctx).status == "refuted"
    assert cv.certify({"x": 5, "a": 1, "b": 1}, ctx).status == "refuted"  # not all-different


def test_schema_kernel_checks_structure():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 0}},
        "required": ["n"],
    }
    cv = CompositeVerifier([SchemaVerifier(schema)])
    assert cv.certify({"n": 3}).status == "verified"
    assert cv.certify({"n": -1}).status == "refuted"
    assert cv.certify({}).status == "refuted"


def test_citation_kernel_requires_entailment():
    evidence = [EvidenceItem(source_id="D1", text="The refund window is 30 days.")]
    cv = CompositeVerifier([CitationVerifier(evidence)])
    assert cv.certify("The refund window is 30 days.").status == "verified"
    # A numeric contradiction is caught by strict number checking.
    assert cv.certify("The refund window is 90 days.").status == "refuted"


# --------------------------------------------------------------------------- #
# Safe arithmetic evaluator                                                   #
# --------------------------------------------------------------------------- #


def test_safe_eval_arithmetic_correctness_and_safety():
    assert safe_eval_arithmetic("2 + 3 * 4") == 14
    assert safe_eval_arithmetic("(2 + 3) * 4") == 20
    assert safe_eval_arithmetic("10 % 3") == 1
    assert safe_eval_arithmetic("-5 + 2") == -3
    with pytest.raises(ValueError):
        safe_eval_arithmetic("__import__('os')")  # no identifiers, no eval
    with pytest.raises(ValueError):
        safe_eval_arithmetic("1 / 0")


# --------------------------------------------------------------------------- #
# app.verify_reasoning                                                        #
# --------------------------------------------------------------------------- #


def test_verify_reasoning_refuses_and_audits():
    app = _app()
    va = app.verify_reasoning("The total is 2 + 2 = 5.")
    assert not va.holds and va.refused and va.stopped_reason == "refused"
    assert va.certificate.refuted
    assert any(e.action == "reasoning_verification" for e in app.audit.entries)


def test_verify_reasoning_self_corrects():
    app = _app()
    fixed = app.verify_reasoning("2 + 2 = 5", regenerate=lambda ans, crit: "2 + 2 = 4")
    assert fixed.holds and fixed.attempts == 2 and not fixed.refused


def test_verify_reasoning_self_correction_is_bounded():
    app = _app()
    # A regenerator that emits a distinct but still-wrong answer each cycle stops at
    # max_cycles, still refused — bounding the loop.
    seq = iter(["2 + 2 = 6", "2 + 2 = 7", "2 + 2 = 8"])
    va = app.verify_reasoning("2 + 2 = 5", regenerate=lambda a, c: next(seq), max_cycles=2)
    assert va.refused and va.attempts == 3  # initial + 2 cycles


def test_verify_reasoning_stops_on_non_progress():
    app = _app()
    # A regenerator that returns the same wrong answer stops early, not at max_cycles.
    va = app.verify_reasoning("2 + 2 = 5", regenerate=lambda a, c: "2 + 2 = 6", max_cycles=5)
    assert va.refused and va.attempts == 2


def test_verify_reasoning_raise_on_refute():
    app = _app()
    with pytest.raises(CertificateRefutedError):
        app.verify_reasoning("2 + 2 = 5", raise_on_refute=True)


def test_certificate_refuted_error_is_in_family():
    assert issubclass(CertificateRefutedError, ReasoningVerificationError)


# --------------------------------------------------------------------------- #
# Runtime verification & shielding                                            #
# --------------------------------------------------------------------------- #


def test_behavior_spec_forbid():
    spec = BehaviorSpec(name="no-unapproved-write", forbid=[
        EventPattern(kind="tool_call", where={"side_effects": "write", "approved": False}),
    ])
    mon = RuntimeMonitor(spec)
    v = mon.observe(BehaviorEvent(kind="tool_call", name="delete",
                                  attributes={"side_effects": "write", "approved": False}))
    assert not v.ok
    ok = mon.observe(BehaviorEvent(kind="tool_call", name="delete",
                                   attributes={"side_effects": "write", "approved": True}))
    assert ok.ok


def test_behavior_spec_precedence():
    spec = BehaviorSpec(name="cite-first").precede(
        EventPattern(kind="retrieval"), EventPattern(kind="claim"),
        description="claim before retrieval",
    )
    assert not RuntimeMonitor(spec).check_trajectory([BehaviorEvent(kind="claim")]).ok
    assert RuntimeMonitor(spec).check_trajectory(
        [BehaviorEvent(kind="retrieval"), BehaviorEvent(kind="claim")]
    ).ok


def test_behavior_spec_invariant():
    spec = BehaviorSpec(name="residency")
    spec.invariant("eu_only", lambda e: e.attributes.get("region") in ("eu", "on_prem"))
    mon = RuntimeMonitor(spec)
    assert not mon.observe(BehaviorEvent(kind="action", attributes={"region": "us"})).ok
    assert mon.observe(BehaviorEvent(kind="action", attributes={"region": "eu"})).ok


def test_shield_block_repair_monitor():
    spec = BehaviorSpec(name="no-write", forbid=[EventPattern(kind="tool_call",
                        where={"side_effects": "write", "approved": False})])
    bad = BehaviorEvent(kind="tool_call", name="del",
                        attributes={"side_effects": "write", "approved": False})

    blocked = Shield(spec, mode="block").guard(bad)
    assert not blocked.allowed and blocked.violations

    def repair(event, violations):
        return BehaviorEvent(kind="tool_call", name="del",
                             attributes={"side_effects": "write", "approved": True})

    repaired = Shield(spec, mode="repair", repair=repair).guard(bad)
    assert repaired.allowed and repaired.repaired is not None

    observed = Shield(spec, mode="monitor").guard(bad)
    assert observed.allowed and observed.violations  # recorded but not stopped


def test_shield_rollback_does_not_poison_precedence():
    # A blocked event must not count as a prior event for a later precedence check.
    spec = BehaviorSpec(name="cite-first").precede(
        EventPattern(kind="retrieval"), EventPattern(kind="claim"))
    forbid_spec = BehaviorSpec(name="no-x", forbid=[EventPattern(kind="x")])
    shield = Shield([forbid_spec, spec], mode="block")
    shield.guard(BehaviorEvent(kind="x"))            # blocked, rolled back
    # A claim still has no prior retrieval, so it is independently caught.
    decision = shield.guard(BehaviorEvent(kind="claim"))
    assert not decision.allowed


# --------------------------------------------------------------------------- #
# Shield + contracts wired into the tool runtime                              #
# --------------------------------------------------------------------------- #


async def test_shield_blocks_unapproved_write_tool():
    app = _app()

    def delete_account(account_id: str) -> dict:
        return {"deleted": account_id}

    app.add_tool(delete_account, side_effects="write")
    app.shield(BehaviorSpec(name="nw", forbid=[EventPattern(kind="tool_call",
               where={"side_effects": "write", "approved": False})]), use=True)

    blocked = await app.tool_runtime.execute(
        ToolCall(tool_name="delete_account", arguments={"account_id": "x"}))
    assert blocked.status == "denied" and "shield" in (blocked.error or "")

    allowed = await app.tool_runtime.execute(
        ToolCall(tool_name="delete_account", arguments={"account_id": "x"}), approved=True)
    assert allowed.status == "ok" and allowed.output == {"deleted": "x"}


async def test_tool_contract_pre_and_post_conditions():
    app = _app()

    def charge(amount: float) -> dict:
        return {"amount": amount}

    contract = (
        ToolContract()
        .requires_that("amount > 0", lambda a: a["amount"] > 0)
        .ensures_that("returns id", lambda a, r: "id" in r)
    )
    app.add_tool(charge, side_effects="write", contract=contract)

    with pytest.raises(ToolContractError):
        await app.tool_runtime.execute(
            ToolCall(tool_name="charge", arguments={"amount": -5}), approved=True)
    with pytest.raises(ToolContractError):
        await app.tool_runtime.execute(
            ToolCall(tool_name="charge", arguments={"amount": 5}), approved=True)


async def test_tool_contract_passes_when_satisfied():
    app = _app()

    def charge(amount: float) -> dict:
        return {"amount": amount, "id": "ch_1"}

    contract = ToolContract().ensures_that("returns id", lambda a, r: "id" in r)
    app.add_tool(charge, side_effects="read", contract=contract)
    result = await app.tool_runtime.execute(ToolCall(tool_name="charge", arguments={"amount": 5}))
    assert result.status == "ok" and result.output["id"] == "ch_1"


# --------------------------------------------------------------------------- #
# Synthesized programs                                                        #
# --------------------------------------------------------------------------- #


def test_synthesize_verified_program_runs_and_rechecks():
    spec = ProgramSpec(
        name="line-total",
        ops=[ProgramOp(op="derive", field="total", expr="price * qty")],
        properties=[
            ProgramProperty(kind="row_count", relation="preserved"),
            ProgramProperty(kind="field_nonnegative", field="total"),
        ],
    )
    program = synthesize(spec, [{"price": 3.0, "qty": 2}, {"price": 5.0, "qty": 4}])
    assert program.holds and program.certificate.verify()
    out = program.run([{"price": 2.0, "qty": 10}])
    assert out[0]["total"] == 20.0


def test_synthesize_refuted_property_raises():
    spec = ProgramSpec(
        name="neg",
        ops=[ProgramOp(op="derive", field="d", expr="price - 100")],
        properties=[ProgramProperty(kind="field_nonnegative", field="d")],
    )
    with pytest.raises(ProgramSynthesisError):
        synthesize(spec, [{"price": 1.0}])
    # require=False returns a refuted certificate instead of raising.
    program = synthesize(spec, [{"price": 1.0}], require=False)
    assert not program.holds and program.certificate.refuted


def test_synthesized_program_rechecks_at_run_time():
    spec = ProgramSpec(
        name="nonneg",
        ops=[ProgramOp(op="select", fields=["v"])],
        properties=[ProgramProperty(kind="field_nonnegative", field="v")],
    )
    program = synthesize(spec, [{"v": 1}])
    assert program.holds
    with pytest.raises(ProgramSynthesisError):
        program.run([{"v": -1}])  # property no longer holds on new data


def test_program_filter_and_schema_property():
    spec = ProgramSpec(
        name="positives",
        ops=[ProgramOp(op="filter", field="v", op_symbol=">", value=0)],
        properties=[
            ProgramProperty(kind="row_count", relation="le"),
            ProgramProperty(kind="schema", schema={
                "type": "object", "properties": {"v": {"type": "integer", "minimum": 1}}}),
        ],
    )
    program = synthesize(spec, [{"v": 1}, {"v": -1}, {"v": 3}])
    assert program.holds
    assert program.run([{"v": 5}, {"v": -2}]) == [{"v": 5}]


def test_app_synthesize_program_audits():
    app = _app()
    spec = ProgramSpec(name="t", ops=[ProgramOp(op="derive", field="x", expr="a + b")],
                       properties=[ProgramProperty(kind="row_count", relation="preserved")])
    program = app.synthesize_program(spec, [{"a": 1, "b": 2}])
    assert program.holds
    assert any(e.action == "program_synthesis" for e in app.audit.entries)


# --------------------------------------------------------------------------- #
# Optional SMT / CAS backends                                                 #
# --------------------------------------------------------------------------- #


def test_smt_backend_optional():
    from vincio.verify.smt import SmtConstraintVerifier, smt_available

    if not smt_available():
        pytest.skip("z3 not installed (vincio[verify])")
    cv = CompositeVerifier([SmtConstraintVerifier()])
    ctx = VerificationContext(constraints=[
        Constraint.compare("x", ">", 0), Constraint.compare("x", "<", 10)])
    assert cv.certify({"x": 5}, ctx).status == "verified"
    # An unsatisfiable system is refuted.
    bad = VerificationContext(constraints=[
        Constraint.compare("x", ">", 10), Constraint.compare("x", "<", 1)])
    assert cv.certify({"x": 5}, bad).status == "refuted"


def test_cas_backend_optional():
    from vincio.verify.smt import CasArithmeticVerifier, cas_available

    if not cas_available():
        pytest.skip("sympy not installed (vincio[verify]))")
    cv = CompositeVerifier([CasArithmeticVerifier()])
    assert cv.certify("1 / 3 * 3 = 1").status == "verified"
