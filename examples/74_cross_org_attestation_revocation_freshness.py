"""Cross-org attestation revocation & freshness — standing that stays current.

A ``ReputationAttestation`` is a *point-in-time* claim, but standing changes: a
counterparty reliable a year ago may have regressed, and an issuer may need to
**withdraw** an attestation it can no longer stand behind. Today's portable prior
would trust a signed attestation forever. This example adds the next rung — making
portable reputation **time-aware and revocable** — so an imported prior reflects
*current* standing, not a frozen snapshot, without becoming a hosted revocation
service.

Five steps, all offline and deterministic:

  1. Two orgs attest a vendor's standing; one attestation carries an issuer-declared
     validity window (``horizon_days``), the other does not.
  2. Against an as-of clock, a stale attestation (past its window) is excluded and
     pinpointed — it no longer anchors the pooled prior.
  3. Within the window, an older attestation **decays** by a half-life, contributing
     less evidence the older it is — it eases out of the prior rather than dominating.
  4. An issuer signs a content-bound ``AttestationRevocation`` to withdraw a claim it
     can no longer stand behind; the importer excludes it — pinpointed, never silently
     honored — while another issuer's evidence still stands.
  5. A forged revocation, or one naming another org's attestation, cannot cancel a
     claim: revocation reads only the existing signed artifacts, verified from bytes.

Everything here is opt-in and additive; this is a library capability inside your
process, never a hosted revocation or reputation service.
"""

from __future__ import annotations

from datetime import timedelta

from vincio import ContextApp, attest_reputation, combine_attestations, revoke_attestation
from vincio.core.utils import utcnow
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement import AttestationConfig

ACME = HMACSigner("acme-key", key_id="acme")
GLOBEX = HMACSigner("globex-key", key_id="globex")


def a_contract(seller: str = "vendor", price: float = 0.10) -> Contract:
    return Contract(
        buyer="acme", seller=seller, terms=ContractTerms(scope="transcribe", price_usd=price)
    ).seal()


def records(*, seller: str = "vendor", n: int = 4):
    """Settle ``n`` jobs the vendor delivered under price (each a fulfilled record)."""
    from vincio import settle_contract

    return [settle_contract(a_contract(seller), cost_usd=0.06) for _ in range(n)]


def main() -> None:
    now = utcnow()

    # 1. Two orgs attest the vendor. acme declares a 30-day validity window; globex
    #    does not (it asserts the standing holds until it is revoked).
    acme_old = attest_reputation(records(n=4), "vendor", issuer="acme", horizon_days=30)
    acme_old.issued_at = now - timedelta(days=90)  # issued a quarter ago
    acme_old.seal().sign(ACME)
    globex_att = attest_reputation(records(n=2), "vendor", issuer="globex").sign(GLOBEX)
    print(
        f"1. Attestations: acme (90 days old, 30-day window) and globex (no expiry); "
        f"acme expires_at set={acme_old.expires_at is not None}."
    )

    # 2. Against an as-of clock, acme's stale attestation is excluded and pinpointed.
    fresh_prior = combine_attestations([acme_old, globex_att], as_of=now)
    standing = fresh_prior.standing("vendor")
    print(
        f"2. As of now, {len(fresh_prior.stale)} stale attestation excluded "
        f"({fresh_prior.stale[0].reason}); pooled standing rests on "
        f"{standing.issuers} alone."
    )

    # 3. Within its window, an older attestation decays by a half-life: at one
    #    half-life (30 days) an 8-success attestation contributes ~4.
    decay_cfg = AttestationConfig(half_life_days=30)
    aged = attest_reputation(records(n=8), "vendor", issuer="acme")
    aged.issued_at = now - timedelta(days=30)
    aged.seal().sign(ACME)
    decayed = combine_attestations([aged], config=decay_cfg, as_of=now)
    print(
        f"3. Half-life decay: an 8-success attestation aged one half-life contributes "
        f"{decayed.standing('vendor').successes:g} successes — it eases out, never anchors."
    )

    # 4. An issuer withdraws a claim it can no longer stand behind, via app surfaces.
    acme = ContextApp(name="acme", provider=MockProvider(default_text="ok"))
    acme.use_settlement_book()
    for _ in range(4):
        acme.settle(a_contract("vendor"), cost_usd=0.06)
    live_att = acme.attest_reputation("vendor")
    revocation = acme.revoke_attestation(live_att, reason="vendor regressed this quarter")

    buyer = ContextApp(name="buyer", provider=MockProvider(default_text="ok"))
    buyer.use_reputation_ledger()
    prior = buyer.import_reputation([live_att, globex_att], revocations=[revocation])
    print(
        f"4. acme revoked its attestation ({revocation.attestation_hash[:12]}…); the "
        f"buyer excludes it ({len(prior.revoked)} revoked, pinpointed) — globex's "
        f"evidence still stands for {prior.standing('vendor').issuers}."
    )

    # 5. A forged revocation cannot cancel a claim — verified from the bytes alone.
    forged = revoke_attestation(live_att).sign(ACME)
    forged.signatures[0].signature = "deadbeef"  # forge the signature
    guarded = combine_attestations([live_att], revocations=[forged], verify_with=ACME)
    print(
        f"5. A forged revocation is ignored (revocation valid={revocation.verify(acme.contract_signer).valid}, "
        f"forged honored={bool(guarded.revoked)}): the withdrawn claim survives, so no "
        f"org can cancel another's attestation."
    )


if __name__ == "__main__":
    main()
