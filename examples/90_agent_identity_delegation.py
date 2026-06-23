"""Agent identity, delegation & cryptographic accountability.

The platform signs contracts, settlements, attestations, and audit entries — but
*who* a signing key belonged to was an out-of-band assumption (a `key_id` string a
verifier had to trust). This example shows the identity substrate that makes the
answer to *who authorized this action, down what chain, within what bounds*
mechanical and verifiable from the bytes.

Six steps, all offline and deterministic (pure-Python RFC 8032 Ed25519; the native
`cryptography` backend is used automatically behind `vincio[crypto]`):

  1. Mint a portable, **self-certifying** identity — its DID is *derived from* its
     public key, so anyone resolves the verifying key from the id alone, and its
     signed `IdentityDocument` verifies from the bytes.
  2. **Rotate** the signing key along a signed chain: a signature made under the old
     key stays valid, the new key signs new history, and a rotated-away key cannot
     forge it.
  3. **Delegate** bounded authority (a subset of capabilities, a budget cap, an
     expiry) from a principal to an agent, and let the agent **sub-delegate** to a
     sub-agent — each link only *attenuating* the last.
  4. Verify the whole `DelegationChain` offline and ask whether it **permits** a
     concrete action; an over-reaching sub-delegation is **refused from the bytes**.
  5. Issue a verifiable **`AgentCredential`** an importer checks offline and folds
     into the capability-gated admission path.
  6. **Bind** the identity as the app's signer, so every audit entry, contract, and
     settlement records its **DID** — accountability as a cryptographic fact.

This is a library capability inside your process — never a hosted identity provider,
a certificate authority, or a key-escrow service.
"""

from __future__ import annotations

from vincio import AgentIdentity, ContextApp, DelegationChain, Grant
from vincio.providers import MockProvider


def main() -> None:
    app = ContextApp(name="acme-platform", provider=MockProvider(default_text="ok"))

    # 1. A portable, self-certifying identity. The DID *is* the public key.
    principal = app.identity(
        "billing-principal", capabilities=["retrieve", "summarize", "settle"], use=True
    )
    print(f"1. Identity: {principal.did}")
    print(
        f"   document verifies offline: {principal.document.verify().valid}; "
        f"capabilities advertised: {principal.capabilities}"
    )

    # 2. Rotate the key along a signed chain. Old signatures survive; the rotated-away
    #    key cannot forge new history.
    legacy_sig = principal.sign("quarterly-close-2026Q1")
    principal.rotate()
    print(
        f"2. Rotated key (chain length {len(principal.document.keys)}): "
        f"document still verifies={principal.document.verify().valid}; "
        f"pre-rotation signature still valid={principal.verify('quarterly-close-2026Q1', legacy_sig)}"
    )

    # 3. Delegate bounded authority to an agent, which sub-delegates to a sub-agent.
    agent = AgentIdentity.generate("billing-agent")
    sub_agent = AgentIdentity.generate("invoice-worker")
    to_agent = principal.delegate(
        agent, capabilities=["retrieve", "summarize"], budget_usd=500.0, max_delegations=2
    )
    to_sub = to_agent.delegate(
        agent, sub_agent, capabilities=["retrieve"], budget_usd=100.0
    )
    chain = DelegationChain(links=[to_agent, to_sub])
    print(
        f"3. Delegation chain: {chain.principal[:24]}… → agent → sub-agent "
        f"({len(chain.links)} links); leaf grant = {sorted(chain.effective_grant.capability_set)} "
        f"≤ ${chain.effective_grant.budget_usd:.0f}"
    )

    # 4. The chain verifies offline and answers concrete authorization questions.
    verdict = chain.verify(root_issuer=principal.did)
    print(
        f"4. Chain verifies offline: {verdict.valid}; "
        f"permits retrieve@$60: {chain.permits('retrieve', budget_usd=60.0)}; "
        f"permits summarize (attenuated away): {chain.permits('summarize')}; "
        f"permits retrieve@$200 (over leaf budget): {chain.permits('retrieve', budget_usd=200.0)}"
    )
    # An over-reaching sub-delegation that amplifies its parent is refused from the bytes.
    forged = to_agent.delegate(
        agent, sub_agent, grant=Grant(capabilities=["retrieve", "settle"], budget_usd=100.0)
    )
    forged_chain = DelegationChain(links=[to_agent, forged])
    refused = forged_chain.verify(root_issuer=principal.did)
    print(f"   over-reaching sub-delegation refused: {not refused.valid} — {refused.reason}")

    # 5. A verifiable credential an importer checks offline and folds into admission.
    credential = app.issue_credential(
        agent, {"admitted_capability": "retrieve", "operated_by": "org-acme"}
    )
    print(
        f"5. Credential for {credential.subject[:24]}…: verifies offline="
        f"{credential.verify().valid}; admits 'retrieve'={credential.admits('retrieve')}; "
        f"admits 'settle'={credential.admits('settle')}"
    )

    # 6. Because the identity is the app's signer, the audit chain binds to its DID.
    entry = app.audit.record("invoice_issued", resource="INV-1042", decision="allow")
    print(
        f"6. Audit entry signed by the identity: key_id == DID "
        f"({entry.key_id == principal.did}); chain intact={app.audit.verify_chain()}"
    )

    assert verdict.valid and not refused.valid and credential.verify().valid
    assert entry.key_id == principal.did and app.audit.verify_chain()


if __name__ == "__main__":
    main()
