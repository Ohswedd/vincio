"""Formal verification of governance invariants.

The platform enforces residency, erasure, the budget cap, and injection
containment at runtime; this proves — by exhaustive bounded model checking,
ahead of any run — that those controls satisfy their specifications across the
whole typed input space, and that a violation yields a minimal counterexample.
"""

from __future__ import annotations

import dataclasses

import pytest

from vincio import (
    ContextApp,
    Counterexample,
    GovernanceVerifier,
    Invariant,
    InvariantResult,
    VerificationReport,
    VincioConfig,
)
from vincio.core.errors import GovernanceVerificationError
from vincio.governance.verification import (
    budget_invariant,
    containment_invariant,
    default_invariants,
    erasure_invariant,
    residency_invariant,
    within_budget,
)
from vincio.providers.mock import MockProvider
from vincio.security.capability import AUTHORIZED, TrustLabel, requires_authority


@pytest.fixture()
def config(tmp_path):
    cfg = VincioConfig()
    cfg.storage.metadata = f"sqlite:///{tmp_path}/vincio.db"
    cfg.observability.exporter = "memory"
    cfg.security.audit_dir = str(tmp_path / "audit")
    return cfg


def make_app(config, **kwargs):
    return ContextApp(name="verify_test", provider=MockProvider(), config=config, **kwargs)


# --------------------------------------------------------------------------- #
# The four platform invariants hold across their whole state space
# --------------------------------------------------------------------------- #


def test_all_default_invariants_hold():
    report = GovernanceVerifier().verify(record=False)
    assert report.held
    assert {r.category for r in report.results} == {
        "containment",
        "residency",
        "budget",
        "erasure",
    }
    # "held" means every point in the bounded domain was checked — a proof, not a sample.
    for result in report.results:
        assert result.held
        assert result.states_checked == result.domain_size
        assert result.counterexample is None
    assert report.states_checked == sum(r.domain_size for r in report.results)


@pytest.mark.parametrize(
    "factory",
    [containment_invariant, residency_invariant, budget_invariant, erasure_invariant],
)
def test_each_invariant_holds_individually(factory):
    report = GovernanceVerifier([factory()]).verify(record=False)
    assert report.held
    assert report.results[0].states_checked == report.results[0].domain_size


def test_default_invariants_are_the_four():
    cats = [inv.category for inv in default_invariants()]
    assert cats == ["containment", "residency", "budget", "erasure"]


# --------------------------------------------------------------------------- #
# Deterministic, reproducible, content-bound artifact
# --------------------------------------------------------------------------- #


def test_report_digest_is_deterministic():
    a = GovernanceVerifier().verify(record=False)
    b = GovernanceVerifier().verify(record=False)
    assert a.content_sha256 == b.content_sha256
    assert a.verify() and b.verify()


def test_report_content_binding_detects_tampering():
    report = GovernanceVerifier().verify(record=False)
    assert report.verify()
    # Flip a recorded verdict without recomputing the digest -> binding breaks.
    report.results[0].held = not report.results[0].held
    assert not report.verify()


def test_states_checked_counts_the_whole_domain_when_holding():
    inv = budget_invariant()
    report = GovernanceVerifier([inv]).verify(record=False)
    assert report.results[0].states_checked == inv.domain_size


# --------------------------------------------------------------------------- #
# Counterexample, not just a verdict
# --------------------------------------------------------------------------- #


def test_fail_open_residency_yields_counterexample():
    # deny_on_unknown=False admits an unresolvable region -> out-of-jurisdiction egress.
    report = GovernanceVerifier([residency_invariant(deny_on_unknown=False)]).verify(record=False)
    assert not report.held
    cx = report.counterexamples[0]
    assert isinstance(cx, Counterexample)
    assert cx.category == "residency"
    # Minimization keeps the adversarial unknown region and the tightest allowed set.
    assert cx.assignment["region"] is None
    assert "admitted" in cx.explanation


def test_weak_budget_cap_yields_counterexample():
    # A cap that checks only what is already spent (ignoring the projection) lets a
    # run push the total over the limit.
    weak = budget_invariant(admits=lambda spent, projected, limit: spent < limit)
    report = GovernanceVerifier([weak]).verify(record=False)
    assert not report.held
    cx = report.counterexamples[0]
    assert cx.category == "budget"
    # Minimal witness: nothing spent yet, one large projected run, the smaller limit.
    assert cx.assignment["spent"] == 0.0
    assert cx.assignment["projected"] + cx.assignment["spent"] >= cx.assignment["limit"]


def test_bypassed_containment_gate_yields_counterexample():
    # Model a gate that forgot to block: every call executes. The verifier finds the
    # injected-instruction escalation the real gate makes impossible.
    base = containment_invariant()

    def no_gate(state):
        taint = TrustLabel(state["taint"])
        return not (
            state["side_effects"] in {"write", "external"}
            and taint.is_tainted
            and state["authority"] not in AUTHORIZED
        )

    bypassed = dataclasses.replace(base, id="containment_bypassed", predicate=no_gate)
    report = GovernanceVerifier([bypassed]).verify(record=False)
    assert not report.held
    cx = report.counterexamples[0]
    assert cx.assignment["authority"] in ("none", "trusted")
    assert cx.assignment["side_effects"] in ("write", "external")
    assert cx.assignment["taint"] in ("untrusted", "quarantined")


def test_counterexample_is_minimized():
    # A predicate that fails only when `extra` is at a non-default value should have
    # every other variable relaxed back to its default in the reported witness.
    from vincio.governance.verification import StateVariable

    inv = Invariant(
        id="toy",
        statement="toy",
        category="toy",
        variables=(
            StateVariable("a", (0, 1, 2)),
            StateVariable("b", ("x", "y")),
            StateVariable("trigger", (False, True)),
        ),
        predicate=lambda s: not s["trigger"],
    )
    report = GovernanceVerifier([inv]).verify(record=False)
    cx = report.counterexamples[0]
    assert cx.assignment == {"a": 0, "b": "x", "trigger": True}


def test_raising_predicate_is_a_violation_not_a_crash():
    from vincio.governance.verification import StateVariable

    def boom(_state):
        raise RuntimeError("predicate blew up")

    inv = Invariant(
        id="raiser",
        statement="raises",
        category="toy",
        variables=(StateVariable("x", (0, 1)),),
        predicate=boom,
    )
    report = GovernanceVerifier([inv]).verify(record=False)
    assert not report.held
    assert report.counterexamples


# --------------------------------------------------------------------------- #
# The shared budget primitive
# --------------------------------------------------------------------------- #


def test_within_budget_is_a_projected_hard_cap():
    assert within_budget(0.0, 0.5, 1.0)  # room to spare
    assert not within_budget(0.6, 0.5, 1.0)  # projection pushes over
    assert not within_budget(1.0, 0.0, 1.0)  # already at the cap
    assert within_budget(5.0, 5.0, None)  # no cap


# --------------------------------------------------------------------------- #
# The containment gate is one shared predicate (refactor regression)
# --------------------------------------------------------------------------- #


def test_requires_authority_matches_the_gate_alphabet():
    # Side-effecting + tainted needs an authority; everything else is exempt.
    assert requires_authority(TrustLabel.UNTRUSTED, "write")
    assert requires_authority(TrustLabel.QUARANTINED, "external")
    assert not requires_authority(TrustLabel.TRUSTED, "write")
    assert not requires_authority(TrustLabel.UNTRUSTED, "read")
    assert not requires_authority(TrustLabel.UNTRUSTED, "pure")


# --------------------------------------------------------------------------- #
# Auditable & offline via the app surface
# --------------------------------------------------------------------------- #


def test_app_verify_governance_holds_and_audits(config):
    app = make_app(config)
    report = app.verify_governance()
    assert isinstance(report, VerificationReport)
    assert report.held
    assert report.audit_entry_id is not None
    actions = [e.action for e in app.audit.entries]
    assert "governance_verification" in actions
    entry = next(e for e in app.audit.entries if e.action == "governance_verification")
    assert entry.decision == "allow"
    assert entry.details["held"] is True
    assert entry.details["content_sha256"] == report.content_sha256
    assert app.audit.verify_chain()


def test_app_verify_governance_is_deterministic(config):
    app = make_app(config)
    a = app.verify_governance(record=False)
    b = app.verify_governance(record=False)
    assert a.content_sha256 == b.content_sha256


def test_app_verify_governance_reflects_fail_open_posture(config):
    # An app that turned off fail-closed residency is caught by the verifier.
    config.governance.allowed_regions = ["eu"]
    config.governance.deny_on_unknown_region = False
    app = make_app(config)
    report = app.verify_governance()
    assert not report.held
    residency = next(r for r in report.results if r.category == "residency")
    assert residency.counterexample is not None
    entry = next(e for e in app.audit.entries if e.action == "governance_verification")
    assert entry.decision == "deny"
    assert app.audit.verify_chain()


def test_raise_on_violation_raises_with_counterexamples(config):
    app = make_app(config)
    bad = [residency_invariant(deny_on_unknown=False)]
    with pytest.raises(GovernanceVerificationError) as excinfo:
        app.verify_governance(bad, raise_on_violation=True)
    assert excinfo.value.counterexamples
    assert excinfo.value.code == "GOVERNANCE_INVARIANT_VIOLATED"


def test_holding_report_does_not_raise(config):
    app = make_app(config)
    # Default invariants hold, so raise_on_violation is a no-op.
    report = app.verify_governance(raise_on_violation=True)
    assert report.held


def test_custom_invariant_list_is_honored(config):
    app = make_app(config)
    report = app.verify_governance([erasure_invariant()])
    assert report.held
    assert len(report.results) == 1
    assert report.results[0].category == "erasure"


def test_counterexample_render_is_readable():
    report = GovernanceVerifier([residency_invariant(deny_on_unknown=False)]).verify(record=False)
    rendered = report.counterexamples[0].render()
    assert rendered.startswith("[residency_in_jurisdiction_egress]")
    assert "state:" in rendered


def test_invariant_result_shape():
    report = GovernanceVerifier([erasure_invariant()]).verify(record=False)
    result = report.results[0]
    assert isinstance(result, InvariantResult)
    assert result.digest  # content-hashed per invariant
    assert result.domain_size > 0
