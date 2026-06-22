"""Differential-privacy memory & training: a composing per-subject budget.

Covers the RDP/moments accountant math, cross-round composition, the budget gate
(refuse / down-weight), per-subject isolation, the audit trail and report, and the
two wired integrations — memory consolidation and federated contributions.
"""

from __future__ import annotations

import math

import pytest

from vincio import (
    ContextApp,
    FederatedPolicy,
    PrivacyBudget,
    PrivacyBudgetError,
    PrivacyMechanism,
    VincioConfig,
)
from vincio.core.error_catalog import ERROR_CATALOG
from vincio.core.types import MemoryScope, MemoryType
from vincio.governance.privacy import (
    PrivacyAccountant,
    PrivacyDecision,
    PrivacyReport,
    gaussian_rdp,
    rdp_to_epsilon,
)
from vincio.optimize.distill import TrainingExample, TrainingSet
from vincio.optimize.federated import PrivacyConfig
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import LocalHashEmbedder

DELTA = 1e-5


def _config() -> VincioConfig:
    config = VincioConfig()
    config.observability.exporter = "memory"
    # An ephemeral in-memory metadata store keeps each app isolated and
    # deterministic (persisted spends never leak across apps or pollute cwd).
    config.storage.metadata = "memory://"
    return config


def _training_set(n: int = 4) -> TrainingSet:
    return TrainingSet(
        name="fed",
        examples=[
            TrainingExample(
                messages=[
                    {"role": "user", "content": f"q {i}"},
                    {"role": "assistant", "content": f"a {i}"},
                ]
            )
            for i in range(n)
        ],
    )


# --------------------------------------------------------------------------- #
# The accountant's math
# --------------------------------------------------------------------------- #


def test_full_batch_gaussian_rdp_is_exact():
    orders = (2, 4, 8, 16)
    z = 3.0
    rdp = gaussian_rdp(z, sample_rate=1.0, orders=orders)
    for r, alpha in zip(rdp, orders, strict=True):
        assert r == pytest.approx(alpha / (2 * z * z))


def test_subsampling_amplifies_privacy():
    full = gaussian_rdp(4.0, sample_rate=1.0)
    sub = gaussian_rdp(4.0, sample_rate=0.1)
    # Every order's RDP is no larger under sub-sampling, and strictly smaller for
    # the orders that matter — amplification by sub-sampling.
    assert all(s <= f + 1e-12 for s, f in zip(sub, full, strict=True))
    assert rdp_to_epsilon(sub, delta=DELTA) < rdp_to_epsilon(full, delta=DELTA)


def test_zero_noise_is_not_private():
    rdp = gaussian_rdp(0.0)
    assert all(r == math.inf for r in rdp)
    assert rdp_to_epsilon(rdp, delta=DELTA) == math.inf


def test_zero_curve_spends_nothing():
    assert rdp_to_epsilon([0.0, 0.0, 0.0], delta=DELTA) == 0.0
    assert rdp_to_epsilon([], delta=DELTA) == 0.0


def test_epsilon_requires_valid_delta():
    with pytest.raises(ValueError):
        rdp_to_epsilon([1.0], delta=0.0)
    with pytest.raises(ValueError):
        rdp_to_epsilon([1.0], delta=1.0)


def test_steps_scale_rdp_linearly():
    one = gaussian_rdp(2.0, steps=1)
    five = gaussian_rdp(2.0, steps=5)
    assert all(b == pytest.approx(5 * a) for a, b in zip(one, five, strict=True))


def test_composition_is_tighter_than_naive_sum():
    mech = PrivacyMechanism(noise_multiplier=4.0)
    single = mech.epsilon(delta=DELTA)
    acc = PrivacyAccountant(delta=DELTA)
    for _ in range(4):
        acc.record("s", mech)
    composed = acc.spent("s")
    # Composition grows the budget but stays well under the naive 4×ε sum.
    assert single < composed < 4 * single


def test_mechanism_scaled_reduces_cost_quadratically():
    mech = PrivacyMechanism(noise_multiplier=4.0)
    half = mech.scaled(0.5)
    # Scaling sensitivity by w raises the noise multiplier to z/w.
    assert half.noise_multiplier == pytest.approx(8.0)
    # ...and drops RDP by w² (= 0.25).
    base = gaussian_rdp(4.0)
    scaled = gaussian_rdp(8.0)
    assert all(s == pytest.approx(0.25 * b) for s, b in zip(scaled, base, strict=True))
    assert mech.scaled(0.0).noise_multiplier == math.inf


# --------------------------------------------------------------------------- #
# The budget gate
# --------------------------------------------------------------------------- #


def test_unbounded_subject_always_allows():
    acc = PrivacyAccountant(delta=DELTA)
    decision = acc.check("nobody", PrivacyMechanism(noise_multiplier=1.0))
    assert isinstance(decision, PrivacyDecision)
    assert decision.action == "allow"
    assert decision.limit_epsilon == math.inf


def test_budget_refuses_when_exceeded():
    acc = PrivacyAccountant(default_budget=PrivacyBudget(epsilon=2.0, delta=DELTA), delta=DELTA)
    mech = PrivacyMechanism(noise_multiplier=4.0)
    allowed = 0
    refused = False
    for _ in range(8):
        decision = acc.check("alice", mech)
        if decision.allowed:
            acc.record("alice", mech)
            allowed += 1
        else:
            refused = True
            break
    assert refused
    assert allowed >= 1
    assert acc.spent("alice") <= 2.0 + 1e-9


def test_charge_raises_on_refusal_and_tallies():
    acc = PrivacyAccountant(default_budget=PrivacyBudget(epsilon=1.0, delta=DELTA), delta=DELTA)
    mech = PrivacyMechanism(noise_multiplier=4.0)  # ~1.23 ε on its own > 1.0
    with pytest.raises(PrivacyBudgetError) as exc:
        acc.charge("alice", mech, operation="round")
    assert exc.value.code == "PRIVACY_BUDGET_EXCEEDED"
    assert exc.value.details["subject_id"] == "alice"
    assert acc.report("alice").rows[0].refusals == 1


def test_downweight_fits_within_budget():
    acc = PrivacyAccountant(
        default_budget=PrivacyBudget(epsilon=1.5, delta=DELTA, on_breach="downweight"),
        delta=DELTA,
    )
    mech = PrivacyMechanism(noise_multiplier=4.0)
    weights = []
    for _ in range(8):
        try:
            spend = acc.charge("carol", mech)
        except PrivacyBudgetError:
            break
        weights.append(spend.downweight)
    assert any(w < 1.0 for w in weights)  # a release was clipped harder
    assert acc.spent("carol") <= 1.5 + 1e-6  # never over the ceiling


def test_per_subject_budgets_are_isolated():
    acc = PrivacyAccountant(default_budget=PrivacyBudget(epsilon=2.0, delta=DELTA), delta=DELTA)
    acc.set_budget(PrivacyBudget(subject_id="vip", epsilon=10.0, delta=DELTA))
    mech = PrivacyMechanism(noise_multiplier=4.0)
    acc.record("alice", mech)
    assert acc.spent("bob") == 0.0  # untouched subject spent nothing
    assert acc.budget_for("vip").epsilon == 10.0  # specific budget wins over default
    assert acc.budget_for("alice").epsilon == 2.0  # default applies otherwise


def test_spend_is_monotonic():
    acc = PrivacyAccountant(delta=DELTA)
    mech = PrivacyMechanism(noise_multiplier=4.0)
    seen = [0.0]
    for _ in range(5):
        acc.record("s", mech)
        seen.append(acc.spent("s"))
    assert all(b >= a - 1e-12 for a, b in zip(seen, seen[1:], strict=False))


def test_reset_clears_spends():
    acc = PrivacyAccountant(delta=DELTA)
    acc.record("a", PrivacyMechanism(noise_multiplier=4.0))
    acc.record("b", PrivacyMechanism(noise_multiplier=4.0))
    acc.reset("a")
    assert acc.spent("a") == 0.0
    assert acc.spent("b") > 0.0
    acc.reset()
    assert acc.spent("b") == 0.0


# --------------------------------------------------------------------------- #
# Audit trail & report
# --------------------------------------------------------------------------- #


def test_report_rolls_up_spent_and_remaining():
    acc = PrivacyAccountant(default_budget=PrivacyBudget(epsilon=2.0, delta=DELTA), delta=DELTA)
    mech = PrivacyMechanism(noise_multiplier=4.0)
    acc.record("alice", mech)
    report = acc.report()
    assert isinstance(report, PrivacyReport)
    row = report.rows[0]
    assert row.subject_id == "alice"
    assert row.spent_epsilon > 0.0
    assert row.remaining_epsilon == pytest.approx(2.0 - row.spent_epsilon)
    assert row.operations == 1
    assert report.total_spent_epsilon == pytest.approx(row.spent_epsilon)


def test_spends_recorded_on_audit_chain():
    app = ContextApp(name="dp", provider=MockProvider(default_text="ok"), config=_config())
    app.use_privacy_accountant(default_budget=PrivacyBudget(epsilon=2.0, delta=DELTA))
    app.privacy_accountant.record("alice", PrivacyMechanism(noise_multiplier=4.0), operation="x")
    actions = {e.action for e in app.audit.entries}
    assert "privacy_spend" in actions
    assert app.audit.verify_chain()


# --------------------------------------------------------------------------- #
# Memory consolidation integration
# --------------------------------------------------------------------------- #


_FACTS = [
    "the user prefers metric units and a dark theme",
    "the user's home airport is SFO and they fly on Tuesdays",
    "the user is allergic to penicillin",
    "the user manages a team of six engineers in Berlin",
]


def _seed(app: ContextApp, session_id: str, tag: str) -> None:
    # Distinct content so the write policy keeps each as its own memory.
    for i, fact in enumerate(_FACTS):
        app.memory.write_fact(
            f"{fact} ({tag}.{i})",
            scope=MemoryScope.SESSION,
            owner_id=session_id,
            type=MemoryType.FACT,
            confidence=0.9,
        )


async def test_memory_consolidation_is_gated_and_refused_over_budget():
    app = ContextApp(name="dp", provider=MockProvider(default_text="ok"), config=_config())
    app.use_privacy_accountant(
        default_budget=PrivacyBudget(epsilon=2.0, delta=DELTA),
        default_mechanism=PrivacyMechanism(noise_multiplier=4.0),
    )
    app.add_memory()
    reports = []
    for k in range(4):
        _seed(app, "sess-alice", f"r{k}")
        reports.append(await app.memory.consolidate("sess-alice", user_id="alice"))
    # First consolidation fits the budget and promotes a summary.
    assert reports[0].promoted >= 1
    assert reports[0].privacy_refused is False
    assert reports[0].privacy_epsilon is not None
    # A later one exceeds the budget and is refused — nothing promoted.
    assert any(r.privacy_refused and r.promoted == 0 for r in reports)
    # The refusal is on the audit chain and in the per-subject report.
    assert "privacy_refused" in {e.action for e in app.audit.entries}
    assert app.privacy_report("alice").rows[0].refusals >= 1


async def test_memory_consolidation_is_refuse_only_even_under_a_downweight_budget():
    # A deterministic consolidation cannot be made noisier, so a down-weight budget
    # must NOT let it under-charge: it pays full cost or is refused outright.
    app = ContextApp(name="dp", provider=MockProvider(default_text="ok"), config=_config())
    app.use_privacy_accountant(
        default_budget=PrivacyBudget(epsilon=2.0, delta=DELTA, on_breach="downweight"),
        default_mechanism=PrivacyMechanism(noise_multiplier=4.0),
    )
    app.add_memory()
    reports = []
    for k in range(4):
        _seed(app, "sess-alice", f"r{k}")
        reports.append(await app.memory.consolidate("sess-alice", user_id="alice"))
    # No spend is ever down-weighted (every recorded spend is at full cost)...
    assert all(s.downweight == 1.0 for s in app.privacy_accountant.spends("alice"))
    # ...and an over-budget consolidation is refused, not silently under-charged.
    assert any(r.privacy_refused for r in reports)
    assert app.privacy_accountant.spent("alice") <= 2.0 + 1e-9


async def test_memory_consolidation_unaccounted_without_an_accountant():
    app = ContextApp(name="plain", provider=MockProvider(default_text="ok"), config=_config())
    app.add_memory()  # no accountant attached
    _seed(app, "sess-bob", "x")
    report = await app.memory.consolidate("sess-bob", user_id="bob")
    assert report.privacy_refused is False
    assert report.privacy_epsilon is None  # unaccounted path is unchanged


# --------------------------------------------------------------------------- #
# Federated contribution integration
# --------------------------------------------------------------------------- #


async def test_federated_contribution_composes_the_subject_budget():
    app = ContextApp(name="org-a", provider=MockProvider(default_text="x"), config=_config())
    app.embedder = LocalHashEmbedder(dim=64)
    app.use_privacy_accountant(default_budget=PrivacyBudget(epsilon=1.5, delta=DELTA))
    policy = FederatedPolicy(
        privacy=PrivacyConfig(min_contributors=2, clip_norm=1.0, dp_epsilon=0.8, dp_delta=DELTA),
        consent_subject="alice",
    )
    controller = app.federated_improvement(policy)
    allowed = 0
    refused = False
    for _ in range(6):
        try:
            await controller.build_contribution(
                member_id="org-a", participants=["org-a", "org-b"], training_set=_training_set()
            )
            allowed += 1
        except PrivacyBudgetError:
            refused = True
            break
    assert allowed >= 1
    assert refused
    assert app.privacy_accountant.spent("alice") <= 1.5 + 1e-9


async def test_federated_downweight_releases_a_more_private_contribution():
    app = ContextApp(name="org-b", provider=MockProvider(default_text="x"), config=_config())
    app.embedder = LocalHashEmbedder(dim=64)
    app.use_privacy_accountant(
        default_budget=PrivacyBudget(epsilon=1.5, delta=DELTA, on_breach="downweight")
    )
    base_epsilon = 0.8
    policy = FederatedPolicy(
        privacy=PrivacyConfig(
            min_contributors=2, clip_norm=1.0, dp_epsilon=base_epsilon, dp_delta=DELTA
        ),
        consent_subject="alice",
    )
    controller = app.federated_improvement(policy)
    saw_downweight = False
    for _ in range(8):
        try:
            contribution = await controller.build_contribution(
                member_id="org-b", participants=["org-a", "org-b"], training_set=_training_set()
            )
        except PrivacyBudgetError:
            break
        spend = app.privacy_accountant.spends("alice")[-1]
        if spend.downweight < 1.0:
            saw_downweight = True
            # The leak guard: a down-weighted release must be GENUINELY more private,
            # not merely clipped harder. The Gaussian mechanism's ε is scaled down by
            # the same factor (more noise relative to sensitivity), so the released
            # contribution records a smaller dp_epsilon — and the discounted cost the
            # accountant charged matches the geometry actually released.
            assert contribution.dp_epsilon == pytest.approx(base_epsilon * spend.downweight)
            assert contribution.dp_epsilon < base_epsilon
    assert saw_downweight
    # The budget is never exceeded, and the released ε never silently stays full.
    assert app.privacy_accountant.spent("alice") <= 1.5 + 1e-6


async def test_federated_without_dp_epsilon_is_unaccounted():
    app = ContextApp(name="org-c", provider=MockProvider(default_text="x"), config=_config())
    app.embedder = LocalHashEmbedder(dim=64)
    app.use_privacy_accountant(default_budget=PrivacyBudget(epsilon=0.001, delta=DELTA))
    # No Gaussian mechanism configured (dp_epsilon=None): nothing to account, so a
    # tiny budget does not refuse a clipping-only contribution.
    policy = FederatedPolicy(
        privacy=PrivacyConfig(min_contributors=2, clip_norm=1.0),
        consent_subject="alice",
    )
    controller = app.federated_improvement(policy)
    contribution = await controller.build_contribution(
        member_id="org-c", participants=["org-a", "org-c"], training_set=_training_set()
    )
    assert contribution.n_examples == 4
    assert app.privacy_accountant.spent("alice") == 0.0


# --------------------------------------------------------------------------- #
# App surface
# --------------------------------------------------------------------------- #


def test_app_set_privacy_budget_creates_accountant():
    app = ContextApp(name="dp", provider=MockProvider(default_text="ok"), config=_config())
    assert app.privacy_accountant is None
    app.set_privacy_budget(subject_id="alice", epsilon=1.0)
    assert app.privacy_accountant is not None
    assert app.privacy_accountant.budget_for("alice").epsilon == 1.0


def test_app_privacy_report_empty_without_accountant():
    app = ContextApp(name="dp", provider=MockProvider(default_text="ok"), config=_config())
    report = app.privacy_report()
    assert isinstance(report, PrivacyReport)
    assert report.rows == []


def test_use_privacy_accountant_wires_existing_memory():
    app = ContextApp(name="dp", provider=MockProvider(default_text="ok"), config=_config())
    app.add_memory()  # memory created first
    app.use_privacy_accountant(default_budget=PrivacyBudget(epsilon=2.0))
    assert app.memory.privacy_accountant is app.privacy_accountant


def test_privacy_budget_error_has_catalog_entry():
    assert "PRIVACY_BUDGET_EXCEEDED" in ERROR_CATALOG
    assert PrivacyBudgetError("x").code == "PRIVACY_BUDGET_EXCEEDED"
    assert PrivacyBudgetError("x").remediation


def test_spends_persist_and_reload_from_store():
    import tempfile

    from vincio.storage.sqlite import SQLiteMetadataStore

    with tempfile.TemporaryDirectory() as d:
        store = SQLiteMetadataStore(f"{d}/m.db")
        acc = PrivacyAccountant(store=store)
        acc.record("alice", PrivacyMechanism(noise_multiplier=4.0))
        acc.record("alice", PrivacyMechanism(noise_multiplier=4.0))
        spent = acc.spent("alice")
        # A fresh accountant over the same store recovers the composed budget.
        reloaded = PrivacyAccountant(store=store)
        assert reloaded.spent("alice") == pytest.approx(spent)
        assert len(reloaded.spends("alice")) == 2
