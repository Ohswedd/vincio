"""Tests for federated / cross-org self-improvement.

All deterministic and offline: each member fits a local subspace on its own
grounded data, contributes a numeric, raw-text-free, masked scatter, a secure
aggregation merges the fleet's contributions into a shared subspace, the adopting
member re-fits its own adapter against that geometry, and the result clears the
same no-regression gate a local promotion does — versioned and reversible. No
network and no real model are involved; the in-process adapter shapes generation
directly through the deterministic mock provider.
"""

from __future__ import annotations

import pytest

import vincio
from vincio import (
    ContextApp,
    Contribution,
    ContributionBuilder,
    FederatedPolicy,
    PrivacyConfig,
    SecureAggregator,
    VincioConfig,
)
from vincio.core.errors import OptimizationError
from vincio.evals.datasets import Dataset, EvalCase
from vincio.governance.consent import ConsentLedger, LawfulBasis, Purpose
from vincio.optimize.distill import TrainingExample, TrainingSet
from vincio.optimize.federated import (
    FederatedError,
    _add_into,
    _frobenius,
    _zeros,
    refit_with_subspace,
)
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import LocalHashEmbedder

QA_A = [
    ("what is the refund policy", "Refunds are processed within 30 days."),
    ("how do I reset my password", "Use the reset link on the login page."),
]
QA_B = [
    ("what are the shipping options", "We ship worldwide via DHL in 5-7 days."),
    ("how do I contact support", "Email support@example.com any time."),
]
QA_ALL = QA_A + QA_B
FLEET = ["org-a", "org-b"]
DIM = 64


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


async def _contribution(
    member_id: str,
    qa: list[tuple[str, str]],
    *,
    privacy: PrivacyConfig | None = None,
    participants: list[str] | None = FLEET,
    base_model: str = "gguf-local",
) -> Contribution:
    builder = ContributionBuilder(embedder=_embedder(), privacy=privacy)
    return await builder.build(
        _ts(qa), base_model, member_id=member_id, participants=participants
    )


# ---------------------------------------------------------------------------
# Privacy: contributions carry no raw traffic
# ---------------------------------------------------------------------------


async def test_contribution_carries_no_raw_text():
    """A serialized contribution contains no prompt or response text — only numbers."""
    contribution = await _contribution("org-a", QA_A)
    blob = contribution.model_dump_json()
    for prompt, target in QA_A:
        assert prompt not in blob
        assert target not in blob
    # It is the numeric scatter, the count, and an attestation — nothing else.
    assert contribution.n_examples == len(QA_A)
    assert len(contribution.scatter) == DIM
    assert contribution.embed_dim == DIM


async def test_contribution_clipping_bounds_sensitivity():
    """Clipping caps a contribution's Frobenius norm at the configured bound."""
    privacy = PrivacyConfig(clip_norm=0.5, secure_aggregation=False)
    contribution = await _contribution("org-a", QA_A, privacy=privacy, participants=None)
    assert contribution.clipped is True
    assert _frobenius(contribution.scatter) <= 0.5 + 1e-9


def test_privacy_noise_sigma():
    """The Gaussian-mechanism scale is zero without ε and positive with it."""
    assert PrivacyConfig(dp_epsilon=None).noise_sigma() == 0.0
    sigma = PrivacyConfig(dp_epsilon=1.0, clip_norm=1.0, dp_delta=1e-5).noise_sigma()
    assert sigma > 0.0
    # Tighter epsilon injects strictly more noise.
    assert PrivacyConfig(dp_epsilon=0.5).noise_sigma() > PrivacyConfig(dp_epsilon=2.0).noise_sigma()


async def test_dp_noise_applied_and_deterministic():
    """DP noise perturbs the scatter and is reproducible under a fixed seed."""
    privacy = PrivacyConfig(dp_epsilon=1.0, secure_aggregation=False, seed=7)
    noised = await _contribution("org-a", QA_A, privacy=privacy, participants=None)
    exact = await _contribution(
        "org-a", QA_A, privacy=PrivacyConfig(secure_aggregation=False), participants=None
    )
    delta = _frobenius(
        [[noised.scatter[i][j] - exact.scatter[i][j] for j in range(DIM)] for i in range(DIM)]
    )
    assert delta > 1e-9  # noise actually changed the payload
    again = await _contribution("org-a", QA_A, privacy=privacy, participants=None)
    assert noised.scatter == again.scatter  # seeded → reproducible


# ---------------------------------------------------------------------------
# Secure aggregation
# ---------------------------------------------------------------------------


async def test_secure_aggregation_masks_individual_but_cancels_in_sum():
    """A masked update is unrecoverable alone, yet the masked sum equals the truth."""
    on = PrivacyConfig(secure_aggregation=True)
    off = PrivacyConfig(secure_aggregation=False)
    mA = await _contribution("org-a", QA_A, privacy=on)
    mB = await _contribution("org-b", QA_B, privacy=on)
    uA = await _contribution("org-a", QA_A, privacy=off)
    uB = await _contribution("org-b", QA_B, privacy=off)

    # The masked individual update differs from the true one — not recoverable.
    individual_delta = _frobenius(
        [[mA.scatter[i][j] - uA.scatter[i][j] for j in range(DIM)] for i in range(DIM)]
    )
    assert individual_delta > 1e-6
    assert mA.masked is True

    # The masks cancel exactly in the aggregate: masked sum == unmasked sum.
    masked_sum = _zeros(DIM, DIM)
    _add_into(masked_sum, mA.scatter)
    _add_into(masked_sum, mB.scatter)
    unmasked_sum = _zeros(DIM, DIM)
    _add_into(unmasked_sum, uA.scatter)
    _add_into(unmasked_sum, uB.scatter)
    residual = _frobenius(
        [[masked_sum[i][j] - unmasked_sum[i][j] for j in range(DIM)] for i in range(DIM)]
    )
    assert residual == pytest.approx(0.0, abs=1e-9)


async def test_aggregator_refuses_below_min_contributors():
    """Round-level k-anonymity: a round below ``min_contributors`` is refused."""
    privacy = PrivacyConfig(min_contributors=2)
    aggregator = SecureAggregator(privacy=privacy)
    only = await _contribution("org-a", QA_A, privacy=privacy)
    with pytest.raises(FederatedError, match="at least 2"):
        aggregator.aggregate([only])


async def test_aggregator_refuses_mixed_base_model_and_dim():
    """A round merges exactly one base model and one embedding dimension."""
    privacy = PrivacyConfig()
    a = await _contribution("org-a", QA_A, privacy=privacy, base_model="model-x")
    b = await _contribution("org-b", QA_B, privacy=privacy, base_model="model-y")
    with pytest.raises(FederatedError, match="multiple base models"):
        SecureAggregator(privacy=privacy).aggregate([a, b])

    big = ContributionBuilder(embedder=LocalHashEmbedder(dim=32), privacy=privacy)
    c = await big.build(_ts(QA_B), "model-x", member_id="org-c", participants=FLEET)
    with pytest.raises(FederatedError, match="multiple embedding dimensions"):
        SecureAggregator(privacy=privacy).aggregate([a, c])


async def test_aggregator_enforces_residency_allow_list():
    """A contribution from a non-allowed residency region is refused at the merge."""
    privacy = PrivacyConfig()
    builder = ContributionBuilder(embedder=_embedder(), privacy=privacy)
    a = await builder.build(
        _ts(QA_A), "m", member_id="org-a", participants=FLEET, residency="eu"
    )
    b = await builder.build(
        _ts(QA_B), "m", member_id="org-b", participants=FLEET, residency="us"
    )
    with pytest.raises(FederatedError, match="non-allowed residency"):
        SecureAggregator(privacy=privacy, allowed_regions=["eu"]).aggregate([a, b])
    # With both regions allowed, the merge proceeds.
    merged = SecureAggregator(privacy=privacy, allowed_regions=["eu", "us"]).aggregate([a, b])
    assert merged.contributor_count == 2


async def test_aggregate_is_deterministic_and_covers_the_fleet():
    """The same fleet yields the same subspace, of rank at least each member's."""
    privacy = PrivacyConfig()
    a = await _contribution("org-a", QA_A, privacy=privacy)
    b = await _contribution("org-b", QA_B, privacy=privacy)
    s1 = SecureAggregator(privacy=privacy, rank=8).aggregate([a, b])
    a2 = await _contribution("org-a", QA_A, privacy=privacy)
    b2 = await _contribution("org-b", QA_B, privacy=privacy)
    s2 = SecureAggregator(privacy=privacy, rank=8).aggregate([a2, b2])
    assert s1.digest == s2.digest
    # The fleet subspace spans at least as many directions as the strongest member.
    assert s1.rank >= max(a.local_rank, b.local_rank)
    assert s1.privacy.contributor_count == 2
    assert s1.privacy.secure_aggregation is True


# ---------------------------------------------------------------------------
# Adoption: refit against the shared subspace
# ---------------------------------------------------------------------------


async def test_refit_uses_fleet_basis_and_own_targets():
    """The refit adapter takes the fleet geometry but only the member's own text."""
    privacy = PrivacyConfig()
    a = await _contribution("org-a", QA_A, privacy=privacy)
    b = await _contribution("org-b", QA_B, privacy=privacy)
    subspace = SecureAggregator(privacy=privacy, rank=8).aggregate([a, b])
    adapter = await refit_with_subspace(subspace, _ts(QA_A), embedder=_embedder())
    # Basis is the fleet's; targets are the member's own local answers.
    assert adapter.basis == subspace.basis
    assert set(adapter.targets) == {a for _, a in QA_A}
    assert adapter.provenance["federated_subspace_digest"] == subspace.digest
    assert adapter.metadata["federated"] is True


async def test_refit_rejects_dim_mismatch():
    """Adopting a subspace with the wrong embedder family is refused."""
    privacy = PrivacyConfig()
    a = await _contribution("org-a", QA_A, privacy=privacy)
    b = await _contribution("org-b", QA_B, privacy=privacy)
    subspace = SecureAggregator(privacy=privacy).aggregate([a, b])
    with pytest.raises(FederatedError, match="!= subspace dim"):
        await refit_with_subspace(subspace, _ts(QA_A), embedder=LocalHashEmbedder(dim=32))


# ---------------------------------------------------------------------------
# End-to-end gated round
# ---------------------------------------------------------------------------


def _policy(**kw) -> FederatedPolicy:
    base = dict(min_examples=4, min_samples=4, require_significance=False)
    base.update(kw)
    return FederatedPolicy(**base)


async def test_round_adopts_when_at_least_as_good():
    """A federated round adopts the adapter when it is at-least-as-good as base."""
    app = _app("org-a")
    a = await _contribution("org-a", QA_A)
    b = await _contribution("org-b", QA_B)
    ctl = app.federated_improvement(_policy(), dataset=_golden(QA_ALL))
    result = await ctl.aadopt(contributions=[a, b], training_set=_ts(QA_ALL))
    assert result.adopted is True
    assert result.verdict is not None and result.verdict.delta >= 0.0
    assert result.contributor_count == 2
    assert result.subspace_rank >= 1
    assert result.privacy is not None and result.privacy.secure_aggregation is True
    # The live app now answers the grounded way the fleet taught it.
    assert app.run(QA_A[0][0]).raw_text == QA_A[0][1]
    # The adapter is versioned in the registry.
    assert [v.version for v in ctl.registry.versions("federated-adapter")] == [1]


async def test_round_streams_expected_phases():
    """The streaming round emits observe → aggregate → refit → gate → adopt."""
    app = _app("org-a")
    a = await _contribution("org-a", QA_A)
    b = await _contribution("org-b", QA_B)
    ctl = app.federated_improvement(_policy(), dataset=_golden(QA_ALL))
    phases = [ev.phase async for ev in ctl.astream(contributions=[a, b], training_set=_ts(QA_ALL))]
    assert phases == ["observe", "aggregate", "refit", "gate", "adopt"]
    # Every decision is stamped on the audit chain.
    actions = [e.details["action"] for e in app.audit.query(action="federated_improvement")]
    assert "adopted" in actions


async def test_regressing_adapter_is_refused_and_reversible():
    """A regressing federated adapter is refused; the registry head is untouched."""

    def echo(req):
        return "GOOD answer " + req.messages[-1].text.split()[-1]

    reg_qa_a = [(f"q item {w}", f"GOOD answer {w}") for w in ("alpha", "beta")]
    reg_qa_b = [(f"q item {w}", f"GOOD answer {w}") for w in ("gamma", "delta")]
    reg_all = reg_qa_a + reg_qa_b
    app = ContextApp(name="org-a", provider=MockProvider(responder=echo), config=_config())
    app.embedder = _embedder()

    # Each member contributes geometry honestly, but the adopter's local data is
    # mislabeled, so the refit adapter regresses the base and must be refused.
    a = await _contribution("org-a", reg_qa_a)
    b = await _contribution("org-b", reg_qa_b)
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
    # The refusal is audited as a deny.
    denies = [e for e in app.audit.query(action="federated_improvement") if e.details["action"] == "refused"]
    assert denies


async def test_no_dataset_means_no_adoption():
    """Without a held-out dataset the round refits but refuses to adopt."""
    app = _app("org-a")
    a = await _contribution("org-a", QA_A)
    b = await _contribution("org-b", QA_B)
    ctl = app.federated_improvement(_policy())
    result = await ctl.aadopt(contributions=[a, b], training_set=_ts(QA_ALL))
    assert result.adopted is False
    assert "cannot gate" in result.reason


async def test_dry_run_gates_without_adopting():
    """A dry-run round gates but does not register or apply the adapter."""
    app = _app("org-a")
    a = await _contribution("org-a", QA_A)
    b = await _contribution("org-b", QA_B)
    ctl = app.federated_improvement(_policy(dry_run=True), dataset=_golden(QA_ALL))
    result = await ctl.aadopt(contributions=[a, b], training_set=_ts(QA_ALL))
    assert result.adopted is False
    assert "dry run" in result.reason
    assert ctl.registry.versions("federated-adapter") == []
    assert app.local_adapter is None


# ---------------------------------------------------------------------------
# Governance: consent gating
# ---------------------------------------------------------------------------


async def test_contribution_requires_consent_when_policy_demands_it():
    """A contribution is denied without an active TRAINING consent, allowed with one."""
    app = _app("org-a")
    # A store-less ledger so the test does not read or write shared on-disk consent.
    app.use_consent_ledger(ConsentLedger())
    policy = _policy(require_consent=True, consent_subject="tenant-a")
    ctl = app.federated_improvement(policy)
    with pytest.raises(FederatedError, match="consent denied"):
        await ctl.build_contribution(
            member_id="org-a", participants=FLEET, training_set=_ts(QA_A)
        )
    # Grant TRAINING consent and the contribution proceeds, recording the basis.
    app.consent_ledger.grant(
        "tenant-a", [Purpose.TRAINING], lawful_basis=LawfulBasis.LEGITIMATE_INTERESTS
    )
    contribution = await ctl.build_contribution(
        member_id="org-a", participants=FLEET, training_set=_ts(QA_A)
    )
    assert contribution.consent_basis == LawfulBasis.LEGITIMATE_INTERESTS.value


async def test_consent_required_without_ledger_raises():
    """Requiring consent without a ledger attached is a configuration error."""
    app = _app("org-a")
    ctl = app.federated_improvement(_policy(require_consent=True, consent_subject="t"))
    with pytest.raises(FederatedError, match="no consent ledger"):
        await ctl.build_contribution(member_id="org-a", training_set=_ts(QA_A))


# ---------------------------------------------------------------------------
# App facade integration
# ---------------------------------------------------------------------------


def test_app_contribute_and_adopt_facades():
    """``app.contribute_federated`` and ``app.adopt_federated`` drive a full round."""
    app_a = _app("org-a")
    app_b = _app("org-b")
    contribution_a = app_a.contribute_federated(
        member_id="org-a", participants=FLEET, training_set=_ts(QA_A)
    )
    contribution_b = app_b.contribute_federated(
        member_id="org-b", participants=FLEET, training_set=_ts(QA_B)
    )
    assert contribution_a.masked is True and contribution_b.masked is True
    result = app_a.adopt_federated(
        _golden(QA_ALL),
        [contribution_a, contribution_b],
        training_set=_ts(QA_ALL),
        policy=_policy(),
    )
    assert result.adopted is True
    assert app_a.run(QA_A[0][0]).raw_text == QA_A[0][1]
    # Unloading restores the base model exactly — the reversibility knob.
    app_a.use_local_adapter(None)
    assert app_a.run(QA_A[0][0]).raw_text == "I am not sure about that."


def test_residency_tag_is_stamped_from_app_posture():
    """A member's residency tag is taken from its governance posture."""
    config = _config()
    config.governance.allowed_regions = ["eu-west"]
    app = ContextApp(name="org-a", provider=MockProvider(), config=config)
    app.embedder = _embedder()
    contribution = app.contribute_federated(
        member_id="org-a", participants=FLEET, training_set=_ts(QA_A)
    )
    assert contribution.residency == "eu-west"


# ---------------------------------------------------------------------------
# Stability surface
# ---------------------------------------------------------------------------


def test_public_surface_exports_federated_symbols():
    for name in (
        "PrivacyConfig",
        "Contribution",
        "ContributionBuilder",
        "FederatedSubspace",
        "SecureAggregator",
        "FederatedPolicy",
        "FederatedRoundResult",
        "FederatedImprovement",
    ):
        assert name in vincio.__all__
        assert hasattr(vincio, name)


def test_federated_error_is_optimization_error():
    assert issubclass(FederatedError, OptimizationError)
    assert FederatedError("x").code
