"""Cross-org reputation gossip & attestation exchange.

A bounded, pull-based exchange of the existing signed attestations and revocations
over the A2A fabric — the discovery analogue for reputation. An importer queries a
bounded set of governed peers for what they hold about a subject, verifies every
fetched artifact from the bytes, deduplicates by content hash, and folds the result
into the *same* combination — so gossip changes only where the evidence comes from,
never how it is weighed: a denied peer is skipped, a forged artifact is refused, a
revocation a peer gossips excludes the withdrawn claim, and every peer visited and
artifact fetched lands on the audit chain.
"""

from __future__ import annotations

from vincio import (
    AttestationExchange,
    ContextApp,
    GatheredReputation,
    ReputationBundle,
    attestation_a2a_server,
    gather_reputation,
)
from vincio.a2a import AgentCard, connect_a2a_in_process
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.settlement.exchange import EXCHANGE_FETCH_ACTION, EXCHANGE_PEER_ACTION


def _contract(buyer: str, seller: str = "vendor", price: float = 0.10) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope="work", price_usd=price)
    ).seal()


def _org(name: str, *, settled: int = 3, breached: int = 0, subject: str = "vendor") -> ContextApp:
    """An org with a settlement book holding ``settled`` / ``breached`` records on a subject."""
    app = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    app.use_settlement_book()
    for _ in range(settled):
        app.settle(_contract(name, subject), cost_usd=0.05)
    for _ in range(breached):
        app.settle(_contract(name, subject, price=0.04), cost_usd=0.09)
    return app


def _buyer(name: str = "buyer") -> ContextApp:
    app = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    app.use_reputation_ledger()
    return app


# -- the wire bundle ----------------------------------------------------------


def test_bundle_round_trips_through_the_wire() -> None:
    acme = _org("acme")
    att = acme.attest_reputation("vendor")
    rev = acme.revoke_attestation(att, reason="regressed")
    bundle = ReputationBundle(subject="vendor", attestations=[att], revocations=[rev])
    restored = ReputationBundle.from_wire(bundle.to_wire())
    assert restored.subject == "vendor"
    assert restored.attestations[0].content_hash == att.content_hash
    assert restored.revocations[0].content_hash == rev.content_hash
    assert not restored.is_empty


def test_empty_bundle_is_empty() -> None:
    assert ReputationBundle(subject="vendor").is_empty


# -- the peer (server) side ---------------------------------------------------


async def test_peer_serves_its_own_current_signed_attestation() -> None:
    acme = _org("acme", settled=3)
    server = acme.serve_attestations()
    exchange = AttestationExchange(connect_a2a_in_process(server), peer_id="acme")
    bundle = await exchange.fetch("vendor")
    assert len(bundle.attestations) == 1
    att = bundle.attestations[0]
    assert att.issuer == "acme"
    assert att.settled == 3
    # Signed as the issuer, verifiable from the bytes alone.
    assert att.verify(acme.contract_signer).valid


async def test_peer_with_no_history_returns_an_attestation_free_bundle() -> None:
    acme = _org("acme", settled=3, subject="vendor")
    server = acme.serve_attestations()
    exchange = AttestationExchange(connect_a2a_in_process(server), peer_id="acme")
    bundle = await exchange.fetch("stranger")  # acme has no history with 'stranger'
    assert bundle.attestations == []
    assert bundle.is_empty


async def test_peer_serves_held_revocations_about_the_subject() -> None:
    acme = _org("acme")
    att = acme.attest_reputation("vendor")
    acme.revoke_attestation(att, reason="regressed")
    server = acme.serve_attestations()  # uses the app's retained revocations
    exchange = AttestationExchange(connect_a2a_in_process(server), peer_id="acme")
    bundle = await exchange.fetch("vendor")
    assert len(bundle.revocations) == 1
    assert bundle.revocations[0].issuer == "acme"


async def test_peer_serves_an_explicit_signed_snapshot() -> None:
    acme = _org("acme", settled=5)
    snapshot = acme.attest_reputation("vendor")  # a fixed signed claim
    server = acme.serve_attestations(attestations=[snapshot])
    exchange = AttestationExchange(connect_a2a_in_process(server), peer_id="acme")
    bundle = await exchange.fetch("vendor")
    assert len(bundle.attestations) == 1
    assert bundle.attestations[0].content_hash == snapshot.content_hash
    # A different subject is filtered out of the snapshot.
    assert (await exchange.fetch("other")).attestations == []


# -- gathering across peers ---------------------------------------------------


async def test_gather_pools_evidence_across_peers() -> None:
    acme = _org("acme", settled=3)
    globex = _org("globex", settled=2)
    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor", peers={"acme": acme.serve_attestations(), "globex": globex.serve_attestations()}
    )
    assert isinstance(result, GatheredReputation)
    assert result.attestations_gathered == 2
    assert result.peers_reachable == 2
    standing = result.standing("vendor")
    assert standing is not None
    assert standing.issuers == ["acme", "globex"]
    assert standing.successes == 5
    # The assembled prior is attached and weights the negotiation path.
    assert buyer.imported_reputation is result.reputation


async def test_gather_discounts_a_regressor_without_zeroing_it() -> None:
    reliable_issuer = _org("acme", settled=6, subject="reliable")
    flaky_issuer = _org("globex", settled=0, breached=6, subject="flaky")
    buyer = _buyer()
    reliable = await buyer.agather_reputation(
        "reliable", peers={"acme": reliable_issuer.serve_attestations()}, weight=False
    )
    flaky = await buyer.agather_reputation(
        "flaky", peers={"globex": flaky_issuer.serve_attestations()}, weight=False
    )
    # A reliable counterparty outweighs a regressing one, but the regressor stays
    # above the floor — discounted, never zeroed.
    assert reliable.weight("reliable") > flaky.weight("flaky") >= 0.1


async def test_gather_deduplicates_by_content_hash() -> None:
    acme = _org("acme", settled=3)
    # The same peer reachable under two ids returns the same signed attestation.
    server = acme.serve_attestations()
    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor", peers={"acme": server, "acme-mirror": server}, weight=False
    )
    assert result.attestations_gathered == 1  # deduped by content hash
    assert result.duplicates == 1


async def test_gather_is_order_independent() -> None:
    acme = _org("acme", settled=3)
    globex = _org("globex", settled=2)
    buyer = _buyer()
    a = await buyer.agather_reputation(
        "vendor", peers={"acme": acme.serve_attestations(), "globex": globex.serve_attestations()},
        weight=False,
    )
    b = await buyer.agather_reputation(
        "vendor", peers={"globex": globex.serve_attestations(), "acme": acme.serve_attestations()},
        weight=False,
    )
    assert a.weight("vendor") == b.weight("vendor")


# -- governance ---------------------------------------------------------------


async def test_directory_allow_list_skips_a_denied_peer() -> None:
    acme = _org("acme", settled=3)
    evil = _org("evil", settled=9)  # claims a glowing record
    buyer = _buyer()
    directory = buyer.agent_directory(allow=["acme"])
    directory.register(AgentCard(name="acme", description="peer"))
    directory.register(AgentCard(name="evil", description="peer"))
    result = await buyer.agather_reputation(
        "vendor",
        peers={"acme": acme.serve_attestations(), "evil": evil.serve_attestations()},
        directory=directory,
    )
    assert result.peers_reachable == 1
    assert result.standing("vendor").issuers == ["acme"]
    evil_visit = result.visit_for("evil")
    assert evil_visit is not None and not evil_visit.allowed
    assert "evil" not in result.peers_contributing


async def test_bounded_fan_out_caps_the_peers_visited() -> None:
    acme = _org("acme", settled=3)
    globex = _org("globex", settled=2)
    zeta = _org("zeta", settled=4)
    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor",
        peers={
            "acme": acme.serve_attestations(),
            "globex": globex.serve_attestations(),
            "zeta": zeta.serve_attestations(),
        },
        max_peers=2,
        weight=False,
    )
    assert result.peers_reachable == 2  # only two were queried
    assert result.attestations_gathered == 2


# -- integrity: nothing trusted that does not verify --------------------------


async def test_a_forged_artifact_from_a_peer_is_refused() -> None:
    acme = _org("acme", settled=3)
    forged = acme.attest_reputation("vendor")
    forged.signatures[0].signature = "deadbeef"  # forge the issuer signature
    server = acme.serve_attestations(attestations=[forged])
    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor", peers={"acme": server}, verifier=acme.contract_signer, weight=False
    )
    assert result.attestations_gathered == 0
    assert result.standing("vendor") is None


async def test_a_tampered_artifact_from_a_peer_is_refused() -> None:
    acme = _org("acme", settled=3)
    tampered = acme.attest_reputation("vendor")
    tampered.settled = 99  # the stored hash no longer recomputes
    server = acme.serve_attestations(attestations=[tampered])
    buyer = _buyer()
    result = await buyer.agather_reputation("vendor", peers={"acme": server}, weight=False)
    assert result.attestations_gathered == 0


async def test_an_unreachable_peer_is_skipped_not_fatal() -> None:
    acme = _org("acme", settled=3)

    class _DeadExchange:
        peer_id = "dead"

        async def fetch(self, subject: str) -> ReputationBundle:
            raise SettlementError("peer unreachable")

    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor", peers={"acme": acme.serve_attestations(), "dead": _DeadExchange()}, weight=False
    )
    assert result.attestations_gathered == 1  # acme still counted
    dead = result.visit_for("dead")
    assert dead is not None and not dead.reachable


# -- revocation gossip --------------------------------------------------------


async def test_a_gossiped_revocation_excludes_the_withdrawn_claim() -> None:
    acme = _org("acme", settled=4)
    globex = _org("globex", settled=2)
    att = acme.attest_reputation("vendor")
    acme.revoke_attestation(att, reason="vendor regressed")  # retained, then gossiped
    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor",
        peers={"acme": acme.serve_attestations(), "globex": globex.serve_attestations()},
        weight=False,
    )
    assert result.revocations_gathered == 1
    assert len(result.reputation.revoked) == 1
    assert result.reputation.revoked[0].issuer == "acme"
    # acme's withdrawn claim is excluded; globex's evidence still stands.
    assert result.standing("vendor").issuers == ["globex"]


# -- freshness ----------------------------------------------------------------


async def test_held_artifacts_merge_with_gossiped_ones() -> None:
    acme = _org("acme", settled=3)
    globex = _org("globex", settled=2)
    # The importer already holds globex's attestation out of band.
    held = globex.attest_reputation("vendor")
    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor",
        peers={"acme": acme.serve_attestations()},
        held_attestations=[held],
        weight=False,
    )
    assert result.attestations_gathered == 2
    assert result.standing("vendor").issuers == ["acme", "globex"]


# -- auditability -------------------------------------------------------------


async def test_every_peer_and_artifact_is_audited() -> None:
    acme = _org("acme", settled=3)
    globex = _org("globex", settled=2)
    buyer = _buyer()
    await buyer.agather_reputation(
        "vendor", peers={"acme": acme.serve_attestations(), "globex": globex.serve_attestations()}
    )
    peer_records = buyer.audit.query(action=EXCHANGE_PEER_ACTION)
    fetch_records = buyer.audit.query(action=EXCHANGE_FETCH_ACTION)
    assert len(peer_records) == 2  # both peers visited
    assert len(fetch_records) == 2  # both attestations fetched
    assert buyer.audit.verify_chain()


async def test_record_audit_off_records_nothing() -> None:
    acme = _org("acme", settled=3)
    buyer = _buyer()
    await buyer.agather_reputation(
        "vendor", peers={"acme": acme.serve_attestations()}, record_audit=False, weight=False
    )
    assert buyer.audit.query(action=EXCHANGE_PEER_ACTION) == []
    assert buyer.audit.query(action=EXCHANGE_FETCH_ACTION) == []


# -- module-level orchestrator & error handling -------------------------------


async def test_module_gather_reputation_matches_app_method() -> None:
    acme = _org("acme", settled=3)
    result = await gather_reputation("vendor", peers={"acme": acme.serve_attestations()})
    assert result.attestations_gathered == 1
    assert result.standing("vendor").issuers == ["acme"]


async def test_a_bad_connection_is_skipped_and_pinpointed() -> None:
    acme = _org("acme", settled=3)
    # A misconfigured peer (not a client/server/exchange) is skipped, not fatal.
    result = await gather_reputation(
        "vendor", peers={"acme": acme.serve_attestations(), "bogus": object()}
    )
    assert result.attestations_gathered == 1
    bogus = result.visit_for("bogus")
    assert bogus is not None and not bogus.reachable
    assert "not an AttestationExchange" in (bogus.reason or "")


async def test_attestation_a2a_server_builds_from_a_book_directly() -> None:
    acme = _org("acme", settled=3)
    server = attestation_a2a_server(acme.settlement_book, name="acme-peer")
    exchange = AttestationExchange(connect_a2a_in_process(server), peer_id="acme")
    bundle = await exchange.fetch("vendor")
    assert bundle.attestations[0].issuer == "acme"


def test_serve_attestations_advertises_the_exchange_skill() -> None:
    acme = _org("acme", settled=3)
    server = acme.serve_attestations()
    skills = [s.id for s in server.card.skills]
    assert "attestation-exchange" in skills


def test_sync_gather_wrapper() -> None:
    acme = _org("acme", settled=3)
    buyer = _buyer()
    result = buyer.gather_reputation("vendor", peers={"acme": acme.serve_attestations()})
    assert result.attestations_gathered == 1
