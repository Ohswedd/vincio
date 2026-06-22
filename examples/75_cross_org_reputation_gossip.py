"""Cross-org reputation gossip & attestation exchange — discovering standing.

Attestations are portable, time-aware, and revocable — but an importer still has to
be *handed* the right bundle out of band: it has no way to **discover** who has
attested a counterparty, or to learn that an issuer has since revoked one, without a
hosted registry. This example adds the next rung — a bounded, **pull-based**
exchange of the existing signed artifacts over the A2A fabric — so an importer
assembles a *current* prior from what its peers hold, never from a central bulletin
board.

Five steps, all offline and deterministic:

  1. Two orgs each keep a settlement book on a shared vendor and expose it as a
     queryable attestation peer (``app.serve_attestations`` over A2A).
  2. A buyer with no local history *pulls* their signed attestations
     (``app.gather_reputation``), verifies each from the bytes, and folds them into
     one bounded, evidence-weighted prior — gossip changes only where the evidence
     comes from.
  3. Governed & bounded: a peer the directory's allow-list denies is skipped and
     pinpointed; ``max_peers`` caps the fan-out — discovery never trusts an unlisted
     source or fans out without bound.
  4. Pull, never trust: a forged artifact a peer serves is refused — nothing is
     counted that does not verify from the bytes, exactly as a handed bundle is.
  5. A revocation a peer gossips excludes the withdrawn claim, pinpointed — so the
     assembled prior reflects *current* standing, not a frozen snapshot.

Everything here is opt-in and additive; this is a library capability inside your
process, never a hosted reputation registry or a push-based gossip bus.
"""

from __future__ import annotations

import asyncio

from vincio import ContextApp
from vincio.a2a import AgentCard
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider


def a_contract(buyer: str, seller: str = "vendor", price: float = 0.10) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope="transcribe", price_usd=price)
    ).seal()


def attestor(name: str, *, settled: int = 3) -> ContextApp:
    """An org with a settlement book recording how a vendor delivered for it."""
    org = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    org.use_settlement_book()
    for _ in range(settled):
        org.settle(a_contract(name), cost_usd=0.06)
    return org


async def main() -> None:
    # 1. Two orgs each hold their own signed record of the vendor and expose it as a
    #    queryable peer. They never push — they only answer a pull.
    acme = attestor("acme", settled=3)
    globex = attestor("globex", settled=2)
    acme_peer = acme.serve_attestations()
    globex_peer = globex.serve_attestations()
    print(
        f"1. acme and globex expose their standing over A2A "
        f"({[s.id for s in acme_peer.card.skills][0]!r} skill)."
    )

    # 2. A buyer with no local history pulls their signed attestations and folds them
    #    into one bounded, evidence-weighted prior — current, never handed.
    buyer = ContextApp(name="buyer", provider=MockProvider(default_text="ok"))
    buyer.use_reputation_ledger()
    gathered = await buyer.agather_reputation(
        "vendor", peers={"acme": acme_peer, "globex": globex_peer}
    )
    standing = gathered.standing("vendor")
    print(
        f"2. Gathered {gathered.attestations_gathered} attestation(s) from "
        f"{gathered.peers_reachable} peer(s): vendor standing rests on "
        f"{standing.issuers}, weight={gathered.weight('vendor'):.3f}."
    )

    # 3. Governed & bounded: an unlisted peer is skipped and pinpointed.
    evil = attestor("evil", settled=9)  # claims a glowing record
    directory = buyer.agent_directory(allow=["acme", "globex"])
    for nm in ("acme", "globex", "evil"):
        directory.register(AgentCard(name=nm, description="attestation peer"))
    governed = await buyer.agather_reputation(
        "vendor",
        peers={"acme": acme_peer, "globex": globex_peer, "evil": evil.serve_attestations()},
        directory=directory,
        weight=False,
    )
    skipped = governed.visit_for("evil")
    print(
        f"3. Governed: evil denied (allowed={skipped.allowed}, {skipped.reason!r}); "
        f"{governed.peers_reachable} allowed peer(s) contributed — discovery never "
        f"trusts an unlisted source."
    )

    # 4. Pull, never trust: a forged artifact a peer serves is refused.
    forged_att = acme.attest_reputation("vendor")
    forged_att.signatures[0].signature = "deadbeef"  # forge the issuer signature
    forged_peer = acme.serve_attestations(attestations=[forged_att])
    guarded = await buyer.agather_reputation(
        "vendor", peers={"acme": forged_peer}, verify_with=acme.contract_signer, weight=False
    )
    print(
        f"4. A forged artifact is refused ({guarded.attestations_gathered} counted): "
        f"nothing is trusted that does not verify from the bytes alone."
    )

    # 5. A revocation a peer gossips excludes the withdrawn claim, pinpointed.
    live_att = acme.attest_reputation("vendor")
    acme.revoke_attestation(live_att, reason="vendor regressed this quarter")
    current = await buyer.agather_reputation(
        "vendor", peers={"acme": acme.serve_attestations(), "globex": globex_peer}, weight=False
    )
    print(
        f"5. acme gossiped a revocation; the buyer excludes the withdrawn claim "
        f"({len(current.reputation.revoked)} revoked, pinpointed) — globex's evidence "
        f"still stands for {current.standing('vendor').issuers}. "
        f"Every peer and artifact was audited "
        f"({len(buyer.audit.query(action='reputation_fetch'))} fetches on the chain)."
    )


if __name__ == "__main__":
    asyncio.run(main())
