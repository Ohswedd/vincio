"""Continuous assurance cases & production certification.

Deterministic, offline coverage of the assurance argument tree (claims discharged
by the platform's existing verdicts, bound by hash), continuous re-checking and the
assurance-regression gate, the signed incident + safety-case learning loop, and the
portable certification report — including every tamper, staleness, and missing-piece
path the soundness and regression SLOs rest on.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from vincio import (
    CertificationReport,
    Claim,
    ContextApp,
    Evidence,
    Incident,
    VincioConfig,
    assurance_regression_gate,
    certify,
)
from vincio.assurance import EVIDENCE_KINDS
from vincio.core.errors import AssuranceError
from vincio.core.utils import utcnow
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner

# -- fixtures -----------------------------------------------------------------


def _app(name: str = "svc") -> ContextApp:
    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), config=cfg)


def _gov_evidence(app: ContextApp, **kw) -> Evidence:
    return Evidence.from_governance(app.verify_governance(record=False), **kw)


# -- evidence binding ---------------------------------------------------------


def test_evidence_from_gate_supports_when_passed():
    class _Verdict:
        passed = True
        reason = "no regression"

    ev = Evidence.from_gate(_Verdict())
    assert ev.kind == "eval_gate"
    assert ev.supports is True
    assert ev.detail == "no regression"
    assert ev.verify() and ev.holds()


def test_evidence_from_gate_bool_and_failure():
    assert Evidence.from_gate(True).supports is True
    failed = Evidence.from_gate(False)
    assert failed.supports is False
    assert not failed.holds()  # a failing gate does not discharge a claim


def test_evidence_from_governance_held(app=None):
    app = _app()
    ev = _gov_evidence(app)
    assert ev.kind == "governance_proof"
    assert ev.supports is True
    assert ev.source_hash  # bound to the report's content digest
    assert ev.holds()


def test_evidence_from_certificate_verified_vs_refuted():
    app = _app()
    verified = app.verify_reasoning("2 + 2 = 4", record=False)
    refuted = app.verify_reasoning("2 + 2 = 5", record=False)
    assert Evidence.from_certificate(verified.certificate).supports is True
    assert Evidence.from_certificate(refuted.certificate).supports is False


def test_evidence_from_audit_chain():
    app = _app()
    app.verify_governance()  # writes an entry to the chain
    ev = Evidence.from_audit(app.audit)
    assert ev.kind == "audit_segment"
    assert ev.supports is True
    assert ev.source_hash == app.audit.merkle_root()


def test_evidence_from_sbom():
    from vincio.governance.aibom import generate_aibom

    app = _app()
    ev = Evidence.from_sbom(generate_aibom(app))
    assert ev.kind == "sbom"
    assert ev.supports is True


def test_evidence_kinds_taxonomy():
    assert set(EVIDENCE_KINDS) >= {
        "eval_gate",
        "governance_proof",
        "reasoning_certificate",
        "audit_segment",
        "identity_chain",
        "sbom",
        "external",
    }


def test_evidence_tamper_caught_from_bytes():
    ev = Evidence.from_gate(True)
    assert ev.verify()
    ev.supports = False  # flip the verdict without re-sealing
    assert not ev.verify()  # the content hash no longer recomputes
    assert not ev.holds()


def test_evidence_staleness():
    app = _app()
    fresh = _gov_evidence(app, horizon_days=30)
    stale = _gov_evidence(app, horizon_days=30, recorded_at=utcnow() - timedelta(days=40))
    assert fresh.is_fresh() and fresh.holds()
    assert not stale.is_fresh()
    assert not stale.holds()  # intact and supportive, but expired


# -- the argument tree --------------------------------------------------------


def test_leaf_claim_needs_supporting_evidence():
    bare = Claim(id="x", statement="unsupported")
    assert not bare.evaluate().holds
    supported = Claim(id="x", statement="ok", evidence=[Evidence.from_gate(True)])
    assert supported.evaluate().holds


def test_required_evidence_missing_is_pinpointed():
    claim = Claim(id="x", statement="needs a gate", required_evidence=["eval_gate"])
    status = claim.evaluate()
    assert not status.holds
    assert status.missing == ["eval_gate"]
    # supplying the demanded kind discharges it
    claim.evidence.append(Evidence.from_gate(True))
    assert claim.evaluate().holds


def test_parent_strategy_all_and_any():
    good = Claim(id="g", statement="g", evidence=[Evidence.from_gate(True)])
    bad = Claim(id="b", statement="b", evidence=[Evidence.from_gate(False)])
    all_claim = Claim(id="p", statement="all", strategy="all", subclaims=[good, bad])
    any_claim = Claim(id="p", statement="any", strategy="any", subclaims=[good, bad])
    assert not all_claim.evaluate().holds
    assert any_claim.evaluate().holds


def test_claim_find():
    leaf = Claim(id="leaf", statement="l")
    root = Claim(id="root", statement="r", subclaims=[leaf])
    assert root.find("leaf") is leaf
    assert root.find("nope") is None


# -- the case -----------------------------------------------------------------


def test_assurance_case_builds_signs_and_holds():
    app = _app()
    case = app.assurance_case(
        "The assistant is fit for production",
        context="EU deployment",
        subclaims=[
            Claim(id="governance", statement="Controls hold", evidence=[_gov_evidence(app)]),
            Claim(id="quality", statement="Meets the bar", evidence=[Evidence.from_gate(True)]),
        ],
    )
    assert case.verify()
    assert case.signature  # signed with the app's per-app key
    report = case.check()
    assert report.holds
    assert report.verify()
    assert report.failing_claims == []


def test_case_check_pinpoints_falsified_claim():
    app = _app()
    case = app.assurance_case(
        "Fit",
        subclaims=[Claim(id="quality", statement="q", evidence=[Evidence.from_gate(True)])],
    )
    assert case.check().holds
    case.goal.subclaims[0].evidence[0].supports = False  # an evidence verdict flips
    report = case.check()
    assert not report.holds
    assert "quality:eval_gate" in report.falsified
    assert "quality" in report.failing_claims and "goal" in report.failing_claims


def test_case_tamper_breaks_integrity_and_check_raises():
    app = _app()
    case = app.assurance_case("Fit", subclaims=[Claim(id="q", statement="q")])
    case.goal.statement = "Something else entirely"  # tamper the sealed tree
    assert not case.verify()
    with pytest.raises(AssuranceError):
        case.check()


def test_case_report_hash_catches_edited_verdict():
    app = _app()
    case = app.assurance_case(
        "Fit", subclaims=[Claim(id="q", statement="q", evidence=[Evidence.from_gate(True)])]
    )
    report = case.check()
    assert report.verify()
    report.holds = False  # forge the verdict
    assert not report.verify()


def test_case_signature_verifies_and_rejects_wrong_key():
    app = _app()
    case = app.assurance_case(
        "Fit", subclaims=[Claim(id="q", statement="q", evidence=[Evidence.from_gate(True)])]
    )
    signer = app._resolve_contract_signer(None, True)
    assert case.verify_signature(signer)
    other = HMACSigner("different-secret", key_id="other")
    assert not case.verify_signature(other)


def test_unsigned_case_when_sign_off():
    app = _app()
    case = app.assurance_case(
        "Fit",
        sign=False,
        subclaims=[Claim(id="q", statement="q", evidence=[Evidence.from_gate(True)])],
    )
    assert not case.signature
    assert case.verify()


# -- continuous assurance & regression gate -----------------------------------


def test_assurance_regression_gate_blocks_a_falsified_claim():
    app = _app()
    case = app.assurance_case(
        "Fit",
        subclaims=[Claim(id="quality", statement="q", evidence=[Evidence.from_gate(True)])],
    )
    before = case.check()
    passed, reason = assurance_regression_gate(before, before)
    assert passed and "no assurance regression" in reason

    case.goal.subclaims[0].evidence[0].supports = False
    after = case.check()
    passed, reason = assurance_regression_gate(before, after)
    assert not passed
    assert "quality" in reason


def test_assurance_regression_gate_fails_on_non_holding_after():
    app = _app()
    # Nothing held before (the case was already failing), so no claim regressed —
    # but the after-case still does not hold, which the gate must block.
    case = app.assurance_case(
        "a", subclaims=[Claim(id="q", statement="q", evidence=[Evidence.from_gate(False)])]
    )
    before = case.check()
    after = case.check()
    passed, reason = assurance_regression_gate(before, after)
    assert not passed and "does not hold" in reason


# -- incidents & safety-case learning -----------------------------------------


def test_incident_is_content_bound_and_signable():
    inc = Incident(id="inc-1", description="bad answer", falsified_claim="quality").seal()
    assert inc.verify()
    signer = HMACSigner("secret", key_id="ops")
    inc.sign(signer)
    assert inc.verify_signature(signer)
    inc.falsified_claim = "something-else"  # re-point after sealing
    assert not inc.verify()


def test_case_learns_from_incident_and_revalidates():
    app = _app()
    case = app.assurance_case(
        "Fit",
        subclaims=[Claim(id="quality", statement="q", evidence=[Evidence.from_gate(True)])],
    )
    assert case.check().holds
    inc = Incident(
        id="inc-1",
        description="quality regressed",
        falsified_claim="quality",
        required_evidence=["eval_gate"],
    ).seal()
    case.learn_from(inc)
    # learning demands fresh evidence before the case re-validates
    report = case.check()
    assert not report.holds
    remediation = case.goal.find("quality").subclaims[0]
    assert remediation.required_evidence == ["eval_gate"]
    # signature is dropped because the tree changed
    assert not case.signature
    # discharging the remediation re-validates the case
    case.discharge(remediation.id, Evidence.from_gate(True, label="fix verified"))
    assert case.check().holds


def test_learn_from_unknown_claim_raises():
    app = _app()
    case = app.assurance_case("Fit", subclaims=[Claim(id="q", statement="q")])
    inc = Incident(id="x", falsified_claim="does-not-exist").seal()
    with pytest.raises(AssuranceError):
        case.learn_from(inc)


def test_discharge_unknown_claim_raises():
    app = _app()
    case = app.assurance_case("Fit", subclaims=[Claim(id="q", statement="q")])
    with pytest.raises(AssuranceError):
        case.discharge("nope", Evidence.from_gate(True))


# -- certification report -----------------------------------------------------


def test_certify_emits_portable_verifiable_report():
    app = _app()
    case = app.assurance_case(
        "Fit",
        subclaims=[Claim(id="g", statement="controls", evidence=[_gov_evidence(app)])],
    )
    report = app.certify(case)
    assert isinstance(report, CertificationReport)
    assert report.certified
    assert report.verify()
    assert report.provenance.get("vincio_version") == "3.49.0"
    assert "sbom" in report.provenance
    # round-trips through JSON for a downstream auditor
    assert report.to_json()


def test_certify_non_holding_case_records_residual_risks():
    app = _app()
    case = app.assurance_case(
        "Fit", subclaims=[Claim(id="q", statement="q", evidence=[Evidence.from_gate(False)])]
    )
    report = app.certify(case)
    assert not report.certified
    assert any("q" in r for r in report.residual_risks)
    assert report.verify()  # an honest "not certified" still verifies


def test_certification_verify_is_freshness_aware():
    app = _app()
    # Evidence valid for 30 days; certify today.
    case = app.assurance_case(
        "Fit",
        subclaims=[
            Claim(id="g", statement="controls", evidence=[_gov_evidence(app, horizon_days=30)])
        ],
    )
    report = app.certify(case)
    assert report.verify()  # holds as of issue time
    # 40 days later the proof has expired — the certification no longer holds.
    assert not report.verify(as_of=report.assurance.as_of + timedelta(days=40))


def test_certification_verify_catches_forged_certified_flag():
    app = _app()
    case = app.assurance_case(
        "Fit", subclaims=[Claim(id="q", statement="q", evidence=[Evidence.from_gate(True)])]
    )
    report = app.certify(case)
    report.certified = False
    assert not report.verify()


def test_certification_verify_catches_underlying_artifact_tamper():
    app = _app()
    case = app.assurance_case(
        "Fit", subclaims=[Claim(id="q", statement="q", evidence=[Evidence.from_gate(True)])]
    )
    report = app.certify(case)
    assert report.verify()
    report.case.goal.subclaims[0].evidence[0].supports = False  # tamper underlying evidence
    assert not report.verify()


def test_certify_free_function_with_signer():
    app = _app()
    case = app.assurance_case(
        "Fit",
        sign=False,
        subclaims=[Claim(id="q", statement="q", evidence=[Evidence.from_gate(True)])],
    )
    signer = HMACSigner("secret", key_id="ci")
    report = certify(case, signer=signer)
    assert report.verify()
    assert report.verify_signature(signer)


# -- audit integration --------------------------------------------------------


def test_assurance_lands_on_audit_chain():
    app = _app()
    case = app.assurance_case(
        "Fit", subclaims=[Claim(id="q", statement="q", evidence=[Evidence.from_gate(True)])]
    )
    app.certify(case)
    actions = {e.action for e in app.audit.entries}
    assert "assurance_case" in actions
    assert "assurance_certification" in actions
    assert app.audit.verify_chain()
