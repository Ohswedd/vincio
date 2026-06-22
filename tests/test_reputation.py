"""Tests for cross-fleet reputation & reliability-weighted federated aggregation.

All deterministic and offline. A reputation ledger earns a per-member reliability
score from how each contribution fared against the no-regression gate, the secure
aggregator weights a member's pull on the consensus geometry by that score, and the
discount is bounded (a weight never leaves ``[floor, 1]``) and reversible (adoption
still clears the very same gate). No network and no real model are involved.
"""

from __future__ import annotations

import pytest

import vincio
from vincio import (
    ContextApp,
    ContributionBuilder,
    FederatedPolicy,
    MemberReputation,
    PrivacyConfig,
    ReputationConfig,
    ReputationError,
    ReputationLedger,
    ReputationReport,
    SecureAggregator,
    VincioConfig,
)
from vincio.core.errors import OptimizationError
from vincio.evals.datasets import Dataset, EvalCase
from vincio.optimize.distill import TrainingExample, TrainingSet
from vincio.optimize.federated import FederatedError, _frobenius, _top_eigenvectors, _zeros
from vincio.optimize.reputation import REPUTATION_ACTION, ReputationWeights
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import LocalHashEmbedder

DIM = 64
FLEET = ["org-a", "org-b"]
QA_A = [
    ("what is the refund policy", "Refunds are processed within 30 days."),
    ("how do I reset my password", "Use the reset link on the login page."),
]
QA_B = [
    ("what are the shipping options", "We ship worldwide via DHL in 5-7 days."),
    ("how do I contact support", "Email support@example.com any time."),
]
QA_ALL = QA_A + QA_B


def _ts(qa: list[tuple[str, str]], name: str = "federated-adapter") -> TrainingSet:
    return TrainingSet(
        name=name,
        examples=[
            TrainingExample(
                messages=[{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            )
            for q, a in qa
        ],
    )


def _golden(qa: list[tuple[str, str]]) -> Dataset:
    return Dataset(
        name="golden",
        cases=[EvalCase(id=f"c{i}", input=q, expected=a) for i, (q, a) in enumerate(qa)],
    )


def _config() -> VincioConfig:
    config = VincioConfig()
    config.observability.exporter = "memory"
    return config


def _embedder() -> LocalHashEmbedder:
    return LocalHashEmbedder(dim=DIM)


def _app(name: str, default_text: str = "I am not sure about that.") -> ContextApp:
    app = ContextApp(name=name, provider=MockProvider(default_text=default_text), config=_config())
    app.embedder = _embedder()
    return app


async def _contribution(member_id, qa, *, privacy=None, participants=FLEET, weight=1.0):
    builder = ContributionBuilder(embedder=_embedder(), privacy=privacy)
    return await builder.build(
        _ts(qa), "gguf-local", member_id=member_id, participants=participants, reputation_weight=weight
    )


# ---------------------------------------------------------------------------
# Reputation as a Beta-Bernoulli posterior over gate outcomes
# ---------------------------------------------------------------------------


def test_fresh_member_gets_the_prior_benefit_of_the_doubt():
    """An unseen member starts at the prior mean — neither trusted blindly nor frozen."""
    config = ReputationConfig()
    ledger = ReputationLedger(config)
    expected = config.prior_success / (config.prior_success + config.prior_failure)
    assert ledger.reputation("newcomer") == pytest.approx(expected)
    # The weight sits inside the band and below a proven member's.
    assert config.weight_floor <= ledger.weight("newcomer") <= config.weight_ceiling


def test_successes_raise_and_failures_lower_reputation():
    """A pass raises reputation; a regression lowers it; the order is strict."""
    ledger = ReputationLedger()
    base = ledger.reputation("m")
    ledger.record_outcome("m", passed=True, round_id="r1")
    raised = ledger.reputation("m")
    ledger.record_outcome("m", passed=False, round_id="r2")
    ledger.record_outcome("m", passed=False, round_id="r3")
    lowered = ledger.reputation("m")
    assert raised > base > lowered


def test_persistent_regressor_is_discounted_toward_the_floor():
    """A member that keeps failing the gate decays toward — never below — the floor."""
    config = ReputationConfig(weight_floor=0.1)
    ledger = ReputationLedger(config)
    for i in range(40):
        ledger.record_outcome("bad", passed=False, round_id=f"r{i}")
    for i in range(40):
        ledger.record_outcome("good", passed=True, round_id=f"r{i}")
    assert ledger.weight("bad") < ledger.weight("good")
    assert ledger.weight("bad") >= config.weight_floor  # discounted, never zeroed
    assert ledger.weight("good") <= config.weight_ceiling


def test_decay_lets_a_reformed_member_recover():
    """With decay < 1 a member's recent passes outweigh its stale failures."""
    ledger = ReputationLedger(ReputationConfig(decay=0.5))
    for i in range(5):
        ledger.record_outcome("m", passed=False, round_id=f"r{i}")
    sunk = ledger.reputation("m")
    for i in range(5):
        ledger.record_outcome("m", passed=True, round_id=f"s{i}")
    assert ledger.reputation("m") > sunk


def test_weight_mapping_is_monotonic_and_bounded():
    """The reputation→weight map is monotonic and stays within the configured band."""
    config = ReputationConfig(weight_floor=0.2, weight_ceiling=0.9)
    assert config.weight_of(0.0) == pytest.approx(0.2)
    assert config.weight_of(1.0) == pytest.approx(0.9)
    assert config.weight_of(0.5) == pytest.approx(0.55)
    # Out-of-range reputations clamp into the band.
    assert config.weight_of(-1.0) == pytest.approx(0.2)
    assert config.weight_of(2.0) == pytest.approx(0.9)


def test_incoherent_config_is_refused():
    """An incoherent reputation configuration raises at construction."""
    with pytest.raises(ReputationError, match="weight_floor"):
        ReputationLedger(ReputationConfig(weight_floor=0.9, weight_ceiling=0.1))
    with pytest.raises(ReputationError, match="prior"):
        ReputationLedger(ReputationConfig(prior_success=0.0))
    with pytest.raises(ReputationError, match="decay"):
        ReputationLedger(ReputationConfig(decay=1.5))


def test_snapshot_and_report_expose_the_track_record():
    """A member's standing is an auditable number — successes, failures, weight."""
    ledger = ReputationLedger()
    ledger.record_outcome("m", passed=True, round_id="r1")
    ledger.record_outcome("m", passed=False, round_id="r2")
    snap = ledger.snapshot("m")
    assert isinstance(snap, MemberReputation)
    assert snap.successes == 1.0 and snap.failures == 1.0 and snap.rounds == 2
    assert snap.last_round == "r2"
    report = ledger.report()
    assert [r.member_id for r in report.rows] == ["m"]
    assert report.rows[0].weight == ledger.weight("m")


# ---------------------------------------------------------------------------
# Reputation lives on the audit chain — accrued from gate outcomes, replayable
# ---------------------------------------------------------------------------


def test_outcomes_are_recorded_on_the_audit_chain():
    """Each update lands on the hash-chained log: a pass allows, a regression denies."""
    app = _app("org-a")
    ledger = app.use_reputation_ledger()
    ledger.record_outcome("org-b", passed=True, round_id="r1")
    ledger.record_outcome("org-b", passed=False, round_id="r2")
    entries = app.audit.query(action=REPUTATION_ACTION)
    assert len(entries) == 2
    assert entries[0].decision == "allow" and entries[1].decision == "deny"
    assert entries[1].details["reputation"] == ledger.reputation("org-b")
    assert app.audit.verify_chain()


def test_ledger_replays_from_the_audit_chain():
    """Reputation is reconstructable from the chain alone — no raw traffic needed."""
    app = _app("org-a")
    ledger = app.use_reputation_ledger()
    for _ in range(4):
        ledger.record_outcome("org-b", passed=False, round_id="r")
    for _ in range(4):
        ledger.record_outcome("org-a", passed=True, round_id="r")
    replayed = ReputationLedger.from_audit(app.audit)
    for member in ("org-a", "org-b"):
        assert replayed.weight(member) == pytest.approx(ledger.weight(member))
        assert replayed.snapshot(member).rounds == ledger.snapshot(member).rounds


# ---------------------------------------------------------------------------
# Reliability-weighted aggregation — discount without singling out
# ---------------------------------------------------------------------------


async def test_weighting_shrinks_a_members_pull_before_masking():
    """A reputation weight scales the signal down while masks still cancel exactly."""
    on = PrivacyConfig(secure_aggregation=True)
    discounted = await _contribution("org-b", QA_B, privacy=on, weight=0.25)
    assert discounted.reputation_weight == 0.25
    # The weight scales the underlying signal down (seen on the unmasked path).
    off = PrivacyConfig(secure_aggregation=False)
    full_off = await _contribution("org-b", QA_B, privacy=off, participants=None, weight=1.0)
    plain = await _contribution("org-b", QA_B, privacy=off, participants=None, weight=0.25)
    assert _frobenius(plain.scatter) < _frobenius(full_off.scatter)
    # The masked sum still cancels: a full org-a + discounted org-b unmasks cleanly
    # against the matching unmasked pair (the masks are independent of the weight).
    a_on = await _contribution("org-a", QA_A, privacy=on, weight=1.0)
    a_off = await _contribution("org-a", QA_A, privacy=off, participants=None, weight=1.0)
    masked_sum = _zeros(DIM, DIM)
    _add(masked_sum, a_on.scatter)
    _add(masked_sum, discounted.scatter)
    plain_sum = _zeros(DIM, DIM)
    _add(plain_sum, a_off.scatter)
    _add(plain_sum, plain.scatter)
    residual = _frobenius(
        [[masked_sum[i][j] - plain_sum[i][j] for j in range(DIM)] for i in range(DIM)]
    )
    assert residual == pytest.approx(0.0, abs=1e-9)


def _add(target, other):
    for i, row in enumerate(other):
        for j, v in enumerate(row):
            target[i][j] += v


async def test_aggregator_discounts_the_regressor_toward_the_reliable_member():
    """A reputation-weighted merge leans the consensus toward the reliable member."""
    off = PrivacyConfig(secure_aggregation=False)
    good = await _contribution("good", QA_A, privacy=off, participants=None)
    bad = await _contribution("bad", QA_B, privacy=off, participants=None)
    # The reliable member's own leading direction (its solo consensus).
    good_basis, _ = _top_eigenvectors(good.scatter, 1)
    good_dir = good_basis[0]

    ledger = ReputationLedger(ReputationConfig(weight_floor=0.05))
    for i in range(30):
        ledger.record_outcome("bad", passed=False, round_id=f"r{i}")
    for i in range(30):
        ledger.record_outcome("good", passed=True, round_id=f"r{i}")

    unweighted = SecureAggregator(privacy=off, rank=1).aggregate([good, bad])
    weighted = SecureAggregator(privacy=off, rank=1, reputation=ledger).aggregate([good, bad])

    def align(subspace):
        return abs(sum(a * b for a, b in zip(subspace.basis[0], good_dir, strict=True)))

    # Discounting the regressor pulls the consensus closer to the reliable member.
    assert align(weighted) > align(unweighted)
    assert weighted.provenance["reputation_weighted"] is True
    assert weighted.provenance["reputation_weights"]["bad"] < weighted.provenance[
        "reputation_weights"
    ]["good"]


async def test_aggregator_refuses_to_reweight_a_masked_contribution():
    """A masked contribution must carry its weight; the aggregator never re-weights it."""
    on = PrivacyConfig(secure_aggregation=True)
    a = await _contribution("org-a", QA_A, privacy=on, weight=1.0)
    b = await _contribution("org-b", QA_B, privacy=on, weight=1.0)  # built unweighted
    ledger = ReputationLedger()
    for _ in range(6):
        ledger.record_outcome("org-b", passed=False, round_id="r")  # ledger wants < 1.0
    with pytest.raises(FederatedError, match="cannot re-weight masked"):
        SecureAggregator(privacy=on, reputation=ledger).aggregate([a, b])


async def test_explicit_weights_override_the_ledger():
    """An explicit weight map takes precedence over a bound ledger."""
    off = PrivacyConfig(secure_aggregation=False)
    a = await _contribution("org-a", QA_A, privacy=off, participants=None)
    b = await _contribution("org-b", QA_B, privacy=off, participants=None)
    merged = SecureAggregator(privacy=off, rank=4).aggregate(
        [a, b], weights={"org-a": 1.0, "org-b": 0.1}
    )
    assert merged.provenance["reputation_weights"] == {"org-a": 1.0, "org-b": 0.1}


async def test_unweighted_round_is_byte_identical_to_before():
    """Without a ledger the merge is unchanged — reputation is strictly opt-in."""
    privacy = PrivacyConfig()
    a = await _contribution("org-a", QA_A, privacy=privacy)
    b = await _contribution("org-b", QA_B, privacy=privacy)
    plain = SecureAggregator(privacy=privacy, rank=8).aggregate([a, b])
    assert plain.provenance["reputation_weighted"] is False
    assert plain.provenance["reputation_weights"] == {"org-a": 1.0, "org-b": 1.0}


# ---------------------------------------------------------------------------
# The gated round — bounded and reversible
# ---------------------------------------------------------------------------


def _policy(**kw) -> FederatedPolicy:
    base = dict(min_examples=4, min_samples=4, require_significance=False)
    base.update(kw)
    return FederatedPolicy(**base)


async def test_round_weights_contributions_and_records_the_verdict():
    """A bound ledger weights the round and records its gate verdict back."""
    app = _app("org-a")
    ledger = app.use_reputation_ledger()
    # org-b arrives with a poor track record; org-a is unproven.
    for _ in range(8):
        ledger.record_outcome("org-b", passed=False, round_id="seed")
    ctl = app.federated_improvement(_policy(), dataset=_golden(QA_ALL))
    a = await ctl.build_contribution(member_id="org-a", participants=FLEET, training_set=_ts(QA_A))
    b = await ctl.build_contribution(member_id="org-b", participants=FLEET, training_set=_ts(QA_B))
    # The discounted member's contribution was built with its (sub-1.0) weight.
    assert b.reputation_weight == ledger.weight("org-b")
    assert b.reputation_weight < a.reputation_weight
    result = await ctl.aadopt(contributions=[a, b], training_set=_ts(QA_ALL))
    assert result.adopted is True
    assert result.reputation_weights is not None
    # The round's pass credited every contributor on the audit chain.
    updates = app.audit.query(action=REPUTATION_ACTION)
    assert any(e.details["round_id"] == "round" and e.decision == "allow" for e in updates)
    assert ledger.snapshot("org-a").rounds >= 1 and ledger.snapshot("org-b").rounds >= 1


async def test_reputation_never_bypasses_the_no_regression_gate():
    """Even reliability-weighted, a regressing adapter is refused and reversible."""

    def echo(req):
        return "GOOD answer " + req.messages[-1].text.split()[-1]

    reg_a = [(f"q item {w}", f"GOOD answer {w}") for w in ("alpha", "beta")]
    reg_b = [(f"q item {w}", f"GOOD answer {w}") for w in ("gamma", "delta")]
    reg_all = reg_a + reg_b
    app = ContextApp(name="org-a", provider=MockProvider(responder=echo), config=_config())
    app.embedder = _embedder()
    ledger = app.use_reputation_ledger()
    # Give both members pristine reputations — a high weight must STILL not let a
    # regressing adapter through: the gate is the hard backstop.
    for _ in range(5):
        ledger.record_outcome("org-a", passed=True, round_id="seed")
        ledger.record_outcome("org-b", passed=True, round_id="seed")

    a = await _contribution("org-a", reg_a, weight=ledger.weight("org-a"))
    b = await _contribution("org-b", reg_b, weight=ledger.weight("org-b"))
    bad_local = TrainingSet(
        name="federated-adapter",
        examples=[
            TrainingExample(
                messages=[{"role": "user", "content": q}, {"role": "assistant", "content": "wrong"}]
            )
            for q, _ in reg_all
        ],
    )
    ctl = app.federated_improvement(_policy(gate=0.6), dataset=_golden(reg_all))
    result = await ctl.aadopt(contributions=[a, b], training_set=bad_local)
    assert result.adopted is False
    assert ctl.registry.versions("federated-adapter") == []
    assert app.local_adapter is None
    # The regression debited every contributor (a deny on the chain).
    denies = [e for e in app.audit.query(action=REPUTATION_ACTION) if e.decision == "deny"]
    assert {e.resource for e in denies} == {"org-a", "org-b"}


async def test_record_reputation_can_be_disabled():
    """A round with ``record_reputation=False`` leaves the ledger untouched."""
    app = _app("org-a")
    app.use_reputation_ledger()
    ctl = app.federated_improvement(_policy(record_reputation=False), dataset=_golden(QA_ALL))
    a = await ctl.build_contribution(member_id="org-a", participants=FLEET, training_set=_ts(QA_A))
    b = await ctl.build_contribution(member_id="org-b", participants=FLEET, training_set=_ts(QA_B))
    await ctl.aadopt(contributions=[a, b], training_set=_ts(QA_ALL))
    assert app.audit.query(action=REPUTATION_ACTION) == []


# ---------------------------------------------------------------------------
# App facade & stability surface
# ---------------------------------------------------------------------------


def test_caller_details_cannot_clobber_canonical_audit_fields():
    """Caller-supplied details never overwrite the fields replay depends on."""
    app = _app("org-a")
    ledger = app.use_reputation_ledger()
    # Hostile details that try to forge the canonical replay fields.
    ledger.record_outcome(
        "org-b",
        passed=False,
        round_id="r1",
        details={"member_id": "org-evil", "passed": True, "round_id": "forged"},
    )
    entry = app.audit.query(action=REPUTATION_ACTION)[-1]
    assert entry.details["member_id"] == "org-b"
    assert entry.details["passed"] is False
    assert entry.details["round_id"] == "r1"
    # Replay reconstructs the true member's failure, not the forged identity.
    replayed = ReputationLedger.from_audit(app.audit)
    assert replayed.snapshot("org-b").failures == 1.0
    assert replayed.members() == ["org-b"]


def test_app_reputation_report_facade():
    """``app.reputation_report`` rolls up each member's standing; empty without a ledger."""
    app = _app("org-a")
    assert isinstance(app.reputation_report(), ReputationReport)
    assert app.reputation_report().rows == []
    ledger = app.use_reputation_ledger()
    ledger.record_outcome("org-b", passed=True, round_id="r")
    report = app.reputation_report()
    assert [r.member_id for r in report.rows] == ["org-b"]
    assert app.reputation_report("org-b").rows[0].weight == ledger.weight("org-b")


def test_record_round_credits_every_contributor():
    """A round-level record credits/debits each named member once."""
    ledger = ReputationLedger()
    snaps = ledger.record_round(
        ["org-a", "org-b"],
        passed=True,
        round_id="r",
        weights=ReputationWeights(weights={"org-a": 1.0, "org-b": 0.5}),
    )
    assert {s.member_id for s in snaps} == {"org-a", "org-b"}
    assert all(s.successes == 1.0 for s in snaps)


def test_public_surface_exports_reputation_symbols():
    for name in (
        "ReputationLedger",
        "ReputationConfig",
        "MemberReputation",
        "ReputationReport",
        "ReputationError",
    ):
        assert name in vincio.__all__
        assert hasattr(vincio, name)


def test_reputation_error_is_optimization_error():
    assert issubclass(ReputationError, OptimizationError)
    assert ReputationError("x").code
