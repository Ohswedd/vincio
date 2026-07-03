"""The 7.5 keyword-rename runways: ``verify_with=``→``verifier=`` and ``at=``→``as_of=``.

Every public settlement entry point (free functions, ``SettlementBook`` methods,
and the ``ContextApp`` settlement verbs) takes the canonical ``verifier=``
ChainSigner keyword, and every validity-instant keyword on the identity surface
is ``as_of=``. The old keywords stay accepted through the 7.x line: passing one
warns :class:`VincioDeprecationWarning` and behaves identically, passing both
raises the module's ``VincioError`` subclass, and the library itself never emits
the warning (every internal forward uses the canonical name).
"""

from __future__ import annotations

import inspect
import warnings
from datetime import UTC, datetime, timedelta

import pytest

from vincio import (
    AgentCredential,
    AgentIdentity,
    ContextApp,
    DelegationChain,
    Grant,
    IdentityDocument,
    KeyRecord,
    Keyring,
    arbitrate,
    attest_custody,
    attest_liabilities,
    attest_reputation,
    combine_attestations,
    gather_reputation,
    guard_collateral,
    net_settlements,
    post_collateral_pool,
    settle_contract,
)
from vincio.core.errors import IdentityError, SettlementError
from vincio.core.utils import utcnow
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement import SettlementBook
from vincio.stability import VincioDeprecationWarning

ACME = HMACSigner("acme-key", key_id="acme")
VENDOR = HMACSigner("vendor-key", key_id="vendor")
ATTESTOR = HMACSigner("attestor-key", key_id="attestor")

DEPRECATED_MATCH = "deprecated since Vincio 7.5.*removed in 8.0"


def _app(name: str = "clearer") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def _org(name: str, *, settled: int = 2, subject: str = "vendor") -> ContextApp:
    app = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    app.use_settlement_book()
    for _ in range(settled):
        app.settle(_contract(buyer=name, seller=subject), cost_usd=0.05)
    return app


def _contract(*, buyer: str = "acme", seller: str = "vendor", price: float = 0.10) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope="work", price_usd=price)
    ).seal()


def _settled(buyer: str = "acme", seller: str = "vendor", price: float = 0.10):
    return settle_contract(_contract(buyer=buyer, seller=seller, price=price), cost_usd=0.05)


def _agreed(contract: Contract, *, cost: float = 0.08):
    return [
        settle_contract(contract, cost_usd=cost).sign(ACME, party="acme"),
        settle_contract(contract, cost_usd=cost).sign(VENDOR, party="vendor"),
    ]


def _attestation(issuer: str = "acme", signer: HMACSigner = ACME):
    records = [_settled(buyer=issuer) for _ in range(2)]
    return attest_reputation(records, "vendor", issuer=issuer).sign(signer)


# ---------------------------------------------------------------------------
# verify_with= → verifier= : old kwarg warns and behaves identically
# ---------------------------------------------------------------------------


def test_net_settlements_old_kwarg_warns_and_matches():
    records = [_settled()]
    canonical = net_settlements(records, owner="clearer", verifier=ACME)
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = net_settlements(records, owner="clearer", verify_with=ACME)
    assert legacy.gross_edges == canonical.gross_edges == 1
    assert legacy.source_hashes == canonical.source_hashes
    assert legacy.total_cleared_usd == canonical.total_cleared_usd


def test_arbitrate_old_kwarg_warns_and_matches():
    contract = _contract()
    canonical = arbitrate(_agreed(contract), arbiter="arb", verifier=ACME)
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = arbitrate(_agreed(contract), arbiter="arb", verify_with=ACME)
    assert legacy.status == canonical.status == "upheld"
    assert legacy.upheld_balance_usd == canonical.upheld_balance_usd


def test_combine_attestations_old_kwarg_warns_and_matches():
    att = _attestation()
    canonical = combine_attestations([att], verifier=ACME)
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = combine_attestations([att], verify_with=ACME)
    assert legacy.standing("vendor") is not None
    assert legacy.standing("vendor").reputation == canonical.standing("vendor").reputation


def test_guard_collateral_old_kwarg_warns_and_matches():
    pool = post_collateral_pool([_contract(price=100.0)], fraction=0.1)
    pool.sign(VENDOR, party="vendor")
    canonical = guard_collateral([pool], verifier=VENDOR)
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = guard_collateral([pool], verify_with=VENDOR)
    assert legacy.pledged_usd == canonical.pledged_usd
    assert legacy.status == canonical.status


async def test_gather_reputation_old_kwarg_warns_and_matches():
    peers = {"acme": _org("acme").serve_attestations()}
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = await gather_reputation("vendor", peers=peers, verify_with=ACME)
    canonical = await gather_reputation("vendor", peers=peers, verifier=ACME)
    assert legacy.peers_reachable == canonical.peers_reachable


def test_book_check_root_consistency_old_kwarg_warns_and_matches():
    book = SettlementBook("auditor")
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    a = attest_liabilities("vendor", {"acme": 60.0}, attestor="attestor", as_of=as_of)
    a.sign(ATTESTOR, party="attestor")
    b = attest_liabilities("vendor", {"globex": 40.0}, attestor="attestor", as_of=as_of)
    b.sign(ATTESTOR, party="attestor")
    canonical = book.check_root_consistency([("acme", a), ("globex", b)], verifier=ATTESTOR)
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = book.check_root_consistency(
            [("acme", a), ("globex", b)], verify_with=ATTESTOR
        )
    assert legacy.consistent == canonical.consistent is False
    assert legacy.equivocating_posters == canonical.equivocating_posters


def test_app_clear_settlements_old_kwarg_warns_and_matches():
    app = _app()
    records = [_settled()]
    canonical = app.clear_settlements(records=records, verifier=ACME)
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = app.clear_settlements(records=records, verify_with=ACME)
    assert legacy.source_hashes == canonical.source_hashes
    assert legacy.total_cleared_usd == canonical.total_cleared_usd


def test_app_prove_solvency_old_kwarg_warns_and_canonical_not_ignored():
    app = _app()
    reserves = attest_custody("vendor", 80.0)  # unsigned — the verifier checks the attestor
    owed = attest_liabilities("vendor", 60.0, attestor="attestor")
    owed.sign(ATTESTOR)
    canonical = app.prove_solvency(reserves, owed, verifier=ATTESTOR)
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = app.prove_solvency(reserves, owed, verify_with=ATTESTOR)
    assert legacy.status == canonical.status
    # The canonical keyword must actually reach the underlying verifier path: a
    # verifier that did not sign the attestation refuses the signature.
    with pytest.raises(SettlementError, match="invalid attestor signature"):
        app.prove_solvency(reserves, owed, verifier=HMACSigner("other-key", key_id="other"))


def test_app_import_reputation_old_kwarg_warns_and_matches():
    app = _app()
    att = _attestation()
    canonical = app.import_reputation([att], verifier=ACME)
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = app.import_reputation([att], verify_with=ACME)
    assert legacy.standing("vendor").reputation == canonical.standing("vendor").reputation


# ---------------------------------------------------------------------------
# verify_with= → verifier= : passing both raises SettlementError
# ---------------------------------------------------------------------------


def test_net_settlements_both_kwargs_raise():
    with pytest.raises(SettlementError, match="both verifier="):
        net_settlements([_settled()], verifier=ACME, verify_with=ACME)


def test_book_both_kwargs_raise():
    book = SettlementBook("auditor")
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    a = attest_liabilities("vendor", {"acme": 60.0}, attestor="attestor", as_of=as_of)
    with pytest.raises(SettlementError, match="both verifier="):
        book.check_root_consistency([("acme", a)], verifier=ATTESTOR, verify_with=ATTESTOR)


def test_app_both_kwargs_raise():
    app = _app()
    with pytest.raises(SettlementError, match="both verifier="):
        app.clear_settlements(records=[_settled()], verifier=ACME, verify_with=ACME)


# ---------------------------------------------------------------------------
# verify_with= → verifier= : the canonical name is silent, library never warns
# ---------------------------------------------------------------------------


def test_settlement_canonical_paths_never_warn():
    app = _app()
    records = [_settled()]
    att = _attestation()
    pool = post_collateral_pool([_contract(price=100.0)], fraction=0.1)
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    owed = app.attest_liabilities("vendor", {"acme": 60.0}, as_of=as_of)
    reserves = app.attest_custody("vendor", {"omnibus": 80.0}, as_of=as_of)
    with warnings.catch_warnings():
        warnings.simplefilter("error", VincioDeprecationWarning)
        net_settlements(records, owner="clearer", verifier=ACME)
        arbitrate(_agreed(_contract()), arbiter="arb", verifier=ACME)
        combine_attestations([att], verifier=ACME)
        guard_collateral([pool])
        app.clear_settlements(records=records)
        app.arbitrate(_agreed(_contract()))
        app.check_completeness(owed, {"acme": 60.0})
        app.prove_solvency(reserves, owed)
        app.check_root_consistency([owed])
        app.check_history_consistency([owed])
        app.resolve_insolvency(reserves, owed)
        app.guard_collateral([pool])
        app.import_reputation([att], verifier=ACME)


def test_app_attest_and_gather_reputation_never_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", VincioDeprecationWarning)
        acme = _org("acme")
        acme.attest_reputation("vendor")  # internal source.attest(verifier=None)
        buyer = ContextApp(name="buyer", provider=MockProvider(default_text="ok"))
        buyer.use_reputation_ledger()
        buyer.gather_reputation("vendor", peers={"acme": acme.serve_attestations()})


# ---------------------------------------------------------------------------
# at= → as_of= : every identity method — old kwarg warns and behaves identically
# ---------------------------------------------------------------------------


def _identity(name: str = "agent", seed: bytes = b"\x01" * 32) -> AgentIdentity:
    return AgentIdentity.generate(name, seed=seed)


def _chain() -> DelegationChain:
    principal = AgentIdentity.generate("principal", seed=b"\x11" * 32)
    agent = AgentIdentity.generate("agent", seed=b"\x12" * 32)
    d1 = principal.delegate(agent, capabilities=["retrieve"], budget_usd=40.0)
    return DelegationChain(links=[d1])


def _credential():
    org = AgentIdentity.generate("org", seed=b"\x21" * 32)
    agent = AgentIdentity.generate("agent", seed=b"\x22" * 32)
    return org.issue_credential(agent, {"admitted_capabilities": "retrieve"})


def test_grant_permits_old_kwarg_warns_and_matches():
    cutoff = utcnow()
    g = Grant(capabilities=["x"], not_after=cutoff)
    late = cutoff + timedelta(seconds=1)
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        assert g.permits("x", at=cutoff) is g.permits("x", as_of=cutoff) is True
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        assert g.permits("x", at=late) is g.permits("x", as_of=late) is False


def test_key_record_active_at_old_kwarg_warns_and_positional_stays():
    record = _identity().document.active_key
    now = utcnow()
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        assert record.active_at(at=now) is record.active_at(now) is True
    with warnings.catch_warnings():
        warnings.simplefilter("error", VincioDeprecationWarning)
        assert record.active_at(now) is True  # positional call stays silent
    with pytest.raises(IdentityError, match="requires an as_of"):
        record.active_at()
    with pytest.raises(IdentityError, match="both as_of="):
        record.active_at(now, at=now)


def test_verify_signature_old_kwarg_warns_and_matches():
    identity = _identity(seed=b"\x02" * 32)
    sig = identity.sign("doc-msg")
    canonical = identity.document.verify_signature("doc-msg", sig, as_of=utcnow())
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = identity.document.verify_signature("doc-msg", sig, at=utcnow())
    assert legacy.valid is canonical.valid is True
    assert legacy.kid == canonical.kid


def test_delegation_verify_old_kwarg_warns_and_matches():
    chain = _chain()
    link = chain.links[0]
    now = utcnow()
    canonical = link.verify(as_of=now)
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = link.verify(at=now)
    assert legacy.valid is canonical.valid is True


def test_delegation_chain_verify_old_kwarg_warns_and_matches():
    chain = _chain()
    now = utcnow()
    canonical = chain.verify(as_of=now)
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = chain.verify(at=now)
    assert legacy.valid is canonical.valid is True


def test_delegation_chain_permits_old_kwarg_warns_and_matches():
    chain = _chain()
    now = utcnow()
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        assert chain.permits("retrieve", at=now) is chain.permits("retrieve", as_of=now) is True


def test_delegation_chain_require_permits_old_kwarg_warns_and_matches():
    chain = _chain()
    now = utcnow()
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        assert chain.require_permits("retrieve", at=now) is chain


def test_credential_verify_old_kwarg_warns_and_matches():
    cred = _credential()
    now = utcnow()
    canonical = cred.verify(as_of=now)
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        legacy = cred.verify(at=now)
    assert legacy.valid is canonical.valid is True


def test_credential_require_valid_old_kwarg_warns_and_matches():
    cred = _credential()
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        assert cred.require_valid(at=utcnow()) is cred


def test_credential_admits_old_kwarg_warns_and_matches():
    cred = _credential()
    now = utcnow()
    with pytest.warns(VincioDeprecationWarning, match=DEPRECATED_MATCH):
        assert cred.admits("retrieve", at=now) is cred.admits("retrieve", as_of=now) is True


def test_identity_both_kwargs_raise():
    cred = _credential()
    now = utcnow()
    with pytest.raises(IdentityError, match="both as_of="):
        cred.verify(as_of=now, at=now)


def test_identity_canonical_paths_never_warn():
    identity = _identity(seed=b"\x03" * 32)
    chain = _chain()
    cred = _credential()
    sig = identity.sign("msg")
    now = utcnow()
    with warnings.catch_warnings():
        warnings.simplefilter("error", VincioDeprecationWarning)
        Grant(capabilities=["x"]).permits("x", as_of=now)
        identity.document.active_key.active_at(now)
        identity.document.verify_signature("msg", sig, as_of=now)
        chain.links[0].verify(as_of=now)
        chain.verify(as_of=now)
        chain.permits("retrieve", as_of=now)
        chain.require_permits("retrieve", as_of=now)
        cred.verify(as_of=now)
        cred.require_valid(as_of=now)
        cred.admits("retrieve", as_of=now)


def test_keyring_rotate_and_revoke_event_timestamps_stay_at():
    # rotate/revoke stamp an *event*, not a validity instant — excluded from the
    # rename; at= stays canonical there and must not warn.
    keyring = Keyring.create(name="agent", seed=b"\x04" * 32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", VincioDeprecationWarning)
        keyring.rotate(at=utcnow())
        keyring.revoke(at=utcnow())


# ---------------------------------------------------------------------------
# The docstrings document the runway
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj",
    [
        net_settlements,
        arbitrate,
        combine_attestations,
        guard_collateral,
        gather_reputation,
        SettlementBook.attest,
        ContextApp.clear_settlements,
        ContextApp.import_reputation,
    ],
    ids=lambda o: getattr(o, "__qualname__", str(o)),
)
def test_settlement_docstrings_name_the_deprecated_alias(obj):
    doc = " ".join((inspect.getdoc(obj) or "").split())
    assert "``verify_with`` is a deprecated alias" in doc
    assert "``verifier``" in doc


@pytest.mark.parametrize(
    "obj",
    [
        Grant.permits,
        KeyRecord.active_at,
        IdentityDocument.verify_signature,
        DelegationChain.verify,
        DelegationChain.permits,
        DelegationChain.require_permits,
        AgentCredential.verify,
        AgentCredential.require_valid,
        AgentCredential.admits,
    ],
    ids=lambda o: getattr(o, "__qualname__", str(o)),
)
def test_identity_docstrings_name_the_deprecated_alias(obj):
    doc = " ".join((inspect.getdoc(obj) or "").split())
    assert "deprecated alias for ``as_of``" in doc or "deprecated keyword alias" in doc
