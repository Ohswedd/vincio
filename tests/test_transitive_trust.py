"""Cross-org transitive trust & Sybil-resistant attestation weighting.

Pooling every counted issuer's attested evidence with equal pull lets a clutch of
unknown peers out-evidence a few an importer has lived through, and an adversary can
spin up Sybil issuers that all vouch the same way. The trust kernel weighs each
issuer's evidence by the importer's *own* trust in that issuer — a bounded,
transitive web-of-trust rooted in its local ledger — so corroboration from a trusted
peer counts for more than volume from an unknown one, an unknown issuer still counts
(floored, never zeroed), and a mutually-vouching Sybil cluster cannot outvote a few
trusted ones. The weighting is opt-in: with no ``trust`` / ``trust_config`` the
combination pools with equal pull exactly as before.
"""

from __future__ import annotations

import pytest

from vincio import (
    ContextApp,
    IssuerTrust,
    ReputationLedger,
    TrustConfig,
    TrustModel,
    attest_reputation,
    build_trust_model,
    combine_attestations,
    settle_contract,
)
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner


def _signer(name: str) -> HMACSigner:
    return HMACSigner(f"{name}-key", key_id=name)


def _contract(seller: str, *, buyer: str = "acme", price: float = 0.10) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope="work", price_usd=price)
    ).seal()


def _records(seller: str, *, buyer: str = "acme", settled: int = 0, breached: int = 0):
    out = [settle_contract(_contract(seller, buyer=buyer), cost_usd=0.05) for _ in range(settled)]
    out += [
        settle_contract(_contract(seller, buyer=buyer, price=0.04), cost_usd=0.09)
        for _ in range(breached)
    ]
    return out


def _attestation(issuer: str, subject: str, *, settled: int = 0, breached: int = 0):
    """A signed attestation by ``issuer`` over ``subject``'s earned standing."""
    recs = _records(subject, buyer=issuer, settled=settled, breached=breached)
    return attest_reputation(recs, subject, issuer=issuer).sign(_signer(issuer))


def _base_ledger(*known: str, rounds: int = 10) -> ReputationLedger:
    """A local ledger with first-hand passing history for each ``known`` member."""
    ledger = ReputationLedger()
    for member in known:
        for _ in range(rounds):
            ledger.record_outcome(member, passed=True, round_id="r")
    return ledger


# -- TrustConfig --------------------------------------------------------------


def test_trust_config_defaults_are_coherent() -> None:
    cfg = TrustConfig().validate_coherent()
    assert cfg.max_depth == 1
    assert 0.0 < cfg.hop_decay <= 1.0
    assert 0.0 <= cfg.trust_floor <= cfg.trust_ceiling <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_depth": -1},
        {"hop_decay": 0.0},
        {"hop_decay": 1.5},
        {"trust_floor": 0.5, "trust_ceiling": 0.4},
        {"trust_floor": -0.1},
        {"trust_ceiling": 1.2},
    ],
)
def test_trust_config_rejects_incoherent(kwargs: dict) -> None:
    with pytest.raises(SettlementError):
        TrustConfig(**kwargs).validate_coherent()


def test_trust_of_maps_reputation_into_band_monotonically() -> None:
    cfg = TrustConfig(trust_floor=0.2, trust_ceiling=1.0)
    assert cfg.trust_of(0.0) == 0.2
    assert cfg.trust_of(1.0) == 1.0
    assert cfg.trust_of(0.5) == pytest.approx(0.6)
    assert cfg.trust_of(0.9) > cfg.trust_of(0.1)


def test_clamp_trust_holds_the_band() -> None:
    cfg = TrustConfig(trust_floor=0.1, trust_ceiling=0.9)
    assert cfg.clamp_trust(-1.0) == 0.1
    assert cfg.clamp_trust(2.0) == 0.9
    assert cfg.clamp_trust(0.5) == 0.5


# -- direct (hop 0) trust -----------------------------------------------------


def test_known_issuer_is_trusted_first_hand() -> None:
    base = _base_ledger("acme")
    a = _attestation("acme", "vendor", settled=4)
    model = build_trust_model([a], base=base, config=TrustConfig())
    assessment = model.assessment("acme")
    assert assessment is not None
    assert assessment.direct is True
    assert assessment.depth == 0
    assert assessment.trust == pytest.approx(base.weight("acme"))
    assert "acme" in model.direct_issuers()


def test_unknown_issuer_falls_back_to_the_floor_never_zeroed() -> None:
    base = _base_ledger("acme")
    stranger = _attestation("stranger", "vendor", settled=4)
    model = build_trust_model([stranger], base=base, config=TrustConfig(trust_floor=0.1))
    # The stranger is unassessed (no direct evidence, no trusted voucher) → floor.
    assert model.assessment("stranger") is None
    assert model.trust_in("stranger") == 0.1
    assert model.trust_in("stranger") > 0.0


def test_no_base_leaves_everyone_at_the_floor() -> None:
    a = _attestation("acme", "vendor", settled=4)
    model = build_trust_model([a], base=None, config=TrustConfig())
    assert model.trust_in("acme") == TrustConfig().trust_floor
    assert model.direct_issuers() == []


# -- issuer-weighted pooling --------------------------------------------------


def test_trusted_issuer_outweighs_an_unknown_one_with_equal_evidence() -> None:
    base = _base_ledger("acme")
    trusted = _attestation("acme", "vendor", settled=4)
    unknown = _attestation("stranger", "vendor", settled=4)
    prior = combine_attestations([trusted, unknown], base=base, trust_config=TrustConfig())
    standing = prior.standing("vendor")
    assert standing is not None
    # Both issuers counted, but the trusted issuer's evidence pulls harder.
    assert standing.issuers == ["acme", "stranger"]
    assert standing.issuer_trust["acme"] > standing.issuer_trust["stranger"]
    assert standing.issuer_trust["stranger"] == TrustConfig().trust_floor
    # The verdict pinpoints the applied multiplier, never silent.
    verdict = prior.verdict_for("acme", "vendor")
    assert verdict is not None and verdict.counted and verdict.trust == standing.issuer_trust["acme"]


def test_trust_scales_mass_not_the_attested_ratio() -> None:
    # A heavily-discounted issuer's evidence shrinks in mass but keeps its ratio:
    # all-positive evidence stays all-positive, just with less pull.
    base = _base_ledger("acme")
    unknown = _attestation("stranger", "vendor", settled=10)
    prior = combine_attestations([unknown], base=base, trust_config=TrustConfig(trust_floor=0.1))
    standing = prior.standing("vendor")
    assert standing is not None
    assert standing.failures == 0.0
    assert standing.successes == pytest.approx(1.0)  # 10 successes × 0.1 floor


def test_trust_only_lowers_pull_never_amplifies() -> None:
    # The ceiling is 1.0, so a fully-trusted issuer contributes its full mass and no
    # issuer ever contributes *more* than it attested.
    base = _base_ledger("acme", rounds=200)  # acme's weight saturates toward 1.0
    a = _attestation("acme", "vendor", settled=6)
    weighted = combine_attestations([a], base=base, trust_config=TrustConfig())
    plain = combine_attestations([a])
    assert weighted.standing("vendor").successes <= plain.standing("vendor").successes + 1e-9


# -- Sybil resistance ---------------------------------------------------------


def test_sybil_cluster_cannot_outvote_a_trusted_issuer() -> None:
    base = _base_ledger("acme")
    # Five unknown Sybils all vouch the subject is great; one trusted issuer says it
    # regressed. Pull follows earned trust, so the trusted negative evidence wins.
    sybils = [_attestation(f"sybil{i}", "vendor", settled=4) for i in range(5)]
    trusted_bad = _attestation("acme", "vendor", breached=4)
    weighted = combine_attestations(
        [*sybils, trusted_bad], base=base, trust_config=TrustConfig()
    )
    plain = combine_attestations([*sybils, trusted_bad])
    # Without trust the Sybils dominate (high reputation); with trust the few trusted
    # adverse outcomes pull the standing down well below the unweighted pooling.
    assert weighted.standing("vendor").reputation < plain.standing("vendor").reputation
    assert all(weighted.standing("vendor").issuer_trust[f"sybil{i}"] == 0.1 for i in range(5))


def test_mutually_vouching_sybils_stay_at_the_floor() -> None:
    base = _base_ledger("acme")
    # A ring of unknown issuers attesting *each other* as counterparties — none is
    # reachable from the trusted root, so the ring manufactures no trust.
    ring = [
        _attestation("x", "y", settled=8),
        _attestation("y", "z", settled=8),
        _attestation("z", "x", settled=8),
        _attestation("x", "vendor", settled=8),
    ]
    model = build_trust_model(ring, base=base, config=TrustConfig(max_depth=3))
    assert model.trust_in("x") == 0.1
    assert model.trust_in("y") == 0.1
    assert model.trust_in("z") == 0.1
    assert model.transitive_issuers() == []


# -- bounded transitivity -----------------------------------------------------


def test_trusted_issuer_lends_transitive_trust_one_hop() -> None:
    base = _base_ledger("acme")
    # acme (known) vouches for broker as a counterparty; broker then attests vendor.
    acme_on_broker = _attestation("acme", "broker", settled=8)
    broker_on_vendor = _attestation("broker", "vendor", settled=4)
    model = build_trust_model(
        [acme_on_broker, broker_on_vendor], base=base, config=TrustConfig()
    )
    broker = model.assessment("broker")
    assert broker is not None
    assert broker.transitive is True
    assert broker.depth == 1
    assert broker.vouched_by == ["acme"]
    assert broker.trust > 0.1  # lifted above the floor by the trusted voucher
    assert broker.trust < model.trust_in("acme")  # but attenuated by the hop decay


def test_hop_decay_attenuates_with_distance() -> None:
    base = _base_ledger("acme")
    chain = [
        _attestation("acme", "b1", settled=8),
        _attestation("b1", "b2", settled=8),
        _attestation("b2", "vendor", settled=8),
    ]
    near = build_trust_model(chain, base=base, config=TrustConfig(max_depth=2, hop_decay=0.5))
    # b1 is one hop out, b2 two hops — the further link carries less trust.
    assert near.assessment("b1").depth == 1
    assert near.assessment("b2").depth == 2
    assert near.trust_in("b2") < near.trust_in("b1")


def test_depth_bound_stops_transitivity() -> None:
    base = _base_ledger("acme")
    acme_on_broker = _attestation("acme", "broker", settled=8)
    broker_on_vendor = _attestation("broker", "vendor", settled=4)
    # max_depth=0 → direct trust only; broker is never reached.
    model = build_trust_model(
        [acme_on_broker, broker_on_vendor], base=base, config=TrustConfig(max_depth=0)
    )
    assert model.trust_in("broker") == 0.1
    assert model.transitive_issuers() == []


def test_a_chain_beyond_the_depth_bound_does_not_manufacture_standing() -> None:
    base = _base_ledger("acme")
    chain = [
        _attestation("acme", "b1", settled=8),
        _attestation("b1", "b2", settled=8),
        _attestation("b2", "vendor", settled=8),
    ]
    # With max_depth=1 only b1 is reached; b2 (two hops) stays at the floor.
    model = build_trust_model(chain, base=base, config=TrustConfig(max_depth=1))
    assert model.trust_in("b1") > 0.1
    assert model.trust_in("b2") == 0.1


def test_self_vouching_does_not_bootstrap_trust() -> None:
    base = _base_ledger("acme")
    # An issuer attesting itself as a subject cannot lend itself transitive trust.
    self_vouch = _attestation("stranger", "stranger", settled=8)
    on_vendor = _attestation("stranger", "vendor", settled=8)
    model = build_trust_model([self_vouch, on_vendor], base=base, config=TrustConfig())
    assert model.trust_in("stranger") == 0.1


# -- transitivity through combine_attestations --------------------------------


def test_combine_uses_vouching_attestations_for_transitive_trust() -> None:
    base = _base_ledger("acme")
    acme_on_broker = _attestation("acme", "broker", settled=8)
    broker_on_vendor = _attestation("broker", "vendor", settled=4)
    stranger_on_vendor = _attestation("stranger", "vendor", settled=4)
    prior = combine_attestations(
        [acme_on_broker, broker_on_vendor, stranger_on_vendor],
        subject="vendor",
        base=base,
        trust_config=TrustConfig(),
    )
    # Only vendor is pooled; the acme→broker attestation feeds trust, not the standing.
    assert prior.subjects() == ["vendor"]
    standing = prior.standing("vendor")
    assert standing.issuers == ["broker", "stranger"]
    # broker (transitively trusted) out-pulls the unknown stranger.
    assert standing.issuer_trust["broker"] > standing.issuer_trust["stranger"]
    assert standing.issuer_trust["stranger"] == 0.1


# -- explicit trust sources ---------------------------------------------------


def test_explicit_trust_callable_weights_issuers() -> None:
    trust = {"acme": 1.0, "stranger": 0.2}
    a = _attestation("acme", "vendor", settled=5)
    s = _attestation("stranger", "vendor", settled=5)
    prior = combine_attestations([a, s], trust=lambda issuer: trust.get(issuer, 0.1))
    standing = prior.standing("vendor")
    assert standing.issuer_trust["acme"] == 1.0
    assert standing.issuer_trust["stranger"] == 0.2
    assert prior.trust_in("acme") == 1.0


def test_explicit_trust_model_is_interchangeable_with_a_ledger() -> None:
    base = _base_ledger("acme")
    a = _attestation("acme", "vendor", settled=4)
    s = _attestation("stranger", "vendor", settled=4)
    model = build_trust_model([a, s], base=base, config=TrustConfig())
    assert isinstance(model, TrustModel)
    # weight() aliases trust_in(), so the model drops in wherever a ledger would.
    assert model.weight("acme") == model.trust_in("acme")
    prior = combine_attestations([a, s], trust=model)
    assert prior.standing("vendor").issuer_trust["acme"] == model.trust_in("acme")


def test_a_bad_trust_callable_falls_back_to_full_pull() -> None:
    def explode(_issuer: str) -> float:
        raise RuntimeError("boom")

    a = _attestation("acme", "vendor", settled=4)
    prior = combine_attestations([a], trust=explode)
    # A misbehaving trust source must not break weighting — full pull, like no trust.
    assert prior.standing("vendor").issuer_trust["acme"] == 1.0


# -- backward compatibility ---------------------------------------------------


def test_no_trust_pools_with_equal_pull_unchanged() -> None:
    a = _attestation("acme", "vendor", settled=4)
    s = _attestation("stranger", "vendor", settled=4)
    prior = combine_attestations([a, s])
    standing = prior.standing("vendor")
    assert standing.successes == 8.0
    assert standing.issuer_trust == {}  # no weighting recorded
    assert prior.trust is None
    assert prior.trust_in("acme") == 1.0  # full pull when unweighted


def test_base_alone_does_not_enable_trust_weighting() -> None:
    # Passing a base ledger (for the local-wins rule) must NOT silently weight by
    # trust — weighting is opt-in via trust / trust_config only.
    base = _base_ledger("acme")
    a = _attestation("acme", "vendor", settled=4)
    s = _attestation("stranger", "vendor", settled=4)
    prior = combine_attestations([a, s], base=base)
    assert prior.standing("vendor").successes == 8.0
    assert prior.standing("vendor").issuer_trust == {}


# -- determinism --------------------------------------------------------------


def test_two_importers_reading_the_same_attestations_agree() -> None:
    base = _base_ledger("acme")
    atts = [
        _attestation("acme", "vendor", settled=4),
        _attestation("stranger", "vendor", settled=4),
    ]
    forward = combine_attestations(atts, base=base, trust_config=TrustConfig())
    reverse = combine_attestations(
        list(reversed(atts)), base=base, trust_config=TrustConfig()
    )
    assert forward.standing("vendor").successes == reverse.standing("vendor").successes
    assert forward.weight("vendor") == reverse.weight("vendor")


# -- app surface --------------------------------------------------------------


def test_app_import_reputation_with_trust_config_roots_in_local_ledger() -> None:
    app = ContextApp(name="importer", provider=MockProvider(default_text="ok"))
    app.use_reputation_ledger()
    # The importer knows acme first-hand; stranger is unknown.
    for _ in range(10):
        app.reputation_ledger.record_outcome("acme", passed=True, round_id="r")
    trusted = _attestation("acme", "vendor", settled=4)
    unknown = _attestation("stranger", "vendor", settled=4)
    prior = app.import_reputation([trusted, unknown], trust_config=TrustConfig())
    standing = prior.standing("vendor")
    assert standing.issuer_trust["acme"] > standing.issuer_trust["stranger"]
    assert app.imported_reputation is prior


def test_app_gather_reputation_weights_by_trust() -> None:
    def peer(name: str, *, settled: int) -> ContextApp:
        org = ContextApp(name=name, provider=MockProvider(default_text="ok"))
        org.use_settlement_book()
        for _ in range(settled):
            org.settle(_contract("vendor", buyer=name), cost_usd=0.05)
        return org

    importer = ContextApp(name="importer", provider=MockProvider(default_text="ok"))
    importer.use_reputation_ledger()
    for _ in range(10):
        importer.reputation_ledger.record_outcome("trusted-peer", passed=True, round_id="r")

    gathered = importer.gather_reputation(
        "vendor",
        peers={
            "trusted-peer": peer("trusted-peer", settled=4).serve_attestations(),
            "unknown-peer": peer("unknown-peer", settled=4).serve_attestations(),
        },
        trust_config=TrustConfig(),
        weight=False,
    )
    standing = gathered.standing("vendor")
    assert standing.issuer_trust["trusted-peer"] > standing.issuer_trust["unknown-peer"]
    assert standing.issuer_trust["unknown-peer"] == 0.1


# -- reporting / introspection ------------------------------------------------


def test_issuer_trust_is_traceable() -> None:
    assessment = IssuerTrust(
        issuer="broker", trust=0.4, depth=1, vouched_by=["acme"], reputation=0.9
    )
    assert assessment.transitive is True
    direct = IssuerTrust(issuer="acme", trust=0.9, depth=0, direct=True)
    assert direct.transitive is False
