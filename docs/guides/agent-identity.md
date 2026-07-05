# Agent identity, delegation & cryptographic accountability

Vincio signs contracts, settlements, attestations, audit entries, and engagement
narratives. But *who* a signing key belongs to was an out-of-band assumption, a
`key_id` string a verifier had to trust. For production multi-org and multi-agent
deployments that is the weak link: accountability is only as good as the answer to
*who authorized this action, down what chain, within what bounds*.

This guide covers the identity substrate (`vincio.security.identity`) that makes that
answer **mechanical and verifiable from the bytes**. Everything runs **fully offline
and dependency-free**, Ed25519 is implemented in pure Python (RFC 8032), and the
native, constant-time `cryptography` backend is used automatically when you install
`pip install "vincio[crypto]"` (byte-for-byte the same signatures).

> This is a library capability inside your process, never a hosted identity provider,
> a certificate authority, or a key-escrow service.

## Portable, self-certifying identity

`app.identity(...)` mints an `AgentIdentity` built on an Ed25519 key whose **DID is
derived from the public key**:

```python
agent = app.identity("billing-agent", capabilities=["retrieve", "summarize"])
agent.did                       # "did:vincio:ed25519:<hex public key>"
agent.document.verify().valid   # True, content-bound and signed, checks from the bytes
```

Because the DID *is* the key, a verifier resolves the verifying key from the
identifier alone, `public_key_from_did(agent.did)`, with no registry or network
call. The `IdentityDocument` carries the identity's advertised capabilities, an
optional operating `controller` org, and its full key history; `verify()` recomputes
the content hash, re-derives the DID from the genesis key, checks the rotation chain,
and checks the document signature.

An `AgentIdentity` satisfies the [`ChainSigner`](../reference/api.md) protocol
(`key_id` is the DID, plus `sign` / `verify`), so it drops into every place the
platform already takes a signer.

## Key rotation & revocation

A `Keyring` rotates keys along a **signed rotation chain**, each new key authorized
by the one before it, so a signature is validated against the key that was current
*at signing time*:

```python
legacy = agent.sign("2026-Q1-close")
agent.rotate()                                  # mint a new key the current one signs in
agent.verify("2026-Q1-close", legacy)           # True, old signatures stay valid
agent.document.verify().rotation_chain_ok       # True, the chain is signed end to end
```

A rotated-away (retired) or revoked key cannot sign **new** history: the document is
re-sealed under the current active key, and `IdentityDocument.verify_signature(msg,
sig, at=...)` reports both which key produced a signature and whether that key was
active at a given instant. Revoking the active key first rotates to a fresh key (so a
signer remains) and then marks the old key revoked from that moment, modelling a
compromise where the attacker's key can no longer forge history, but everything it
legitimately signed before remains verifiable.

## Delegation chains & attenuated authority

A signed `Delegation` mints a bounded `Grant`, a subset of capabilities, a budget
cap, an expiry, an audience, and a re-delegation depth, from a principal to an
agent. An agent can **sub-delegate** to a sub-agent, and the links compose into a
`DelegationChain`:

```python
from vincio import DelegationChain

to_agent = principal.delegate(agent, capabilities=["retrieve", "summarize"],
                              budget_usd=500.0, max_delegations=2)
to_sub   = to_agent.delegate(agent, sub_agent, capabilities=["retrieve"], budget_usd=100.0)

chain = DelegationChain(links=[to_agent, to_sub])
chain.verify(root_issuer=principal.did).valid       # True, verifies offline end to end
chain.permits("retrieve", budget_usd=60.0)          # True
chain.permits("summarize")                          # False, attenuated away at the leaf
```

The core invariant is **attenuation**: each link must only narrow its parent's grant.
Capabilities only shrink, the budget only tightens, the expiry only shortens, the
audience only narrows, and each hop consumes one re-delegation. An over-reaching
sub-delegation (one that adds a capability, raises the cap, or extends the expiry) is
**refused from the bytes**, `chain.verify(...)` reports `attenuation_ok == False`,
and a tampered grant fails its signature check. Because every issuer is a DID, the
chain verifies with no external key registry; when a link was signed with a *rotated*
key it carries a compact `KeyAuthorization` proving that key descends from the
issuer's genesis key, so it stays offline-verifiable.

`chain.require_permits(capability, budget_usd=..., audience=...)` raises an
[`IdentityError`](../reference/errors.md#identity_verification_failed) when the chain
does not authorize an action, drop it in front of a tool call, a contract signature,
or a saga handoff so the action carries provenance of authority.

## Verifiable credentials

An org issues a signed `AgentCredential`, a verifiable claim about an agent, that an
importer verifies offline and folds into the existing admission / registry path:

```python
org  = app.identity("org-acme", use=True)
cred = app.issue_credential(agent, {"admitted_capability": "retrieve",
                                     "operated_by": "org-acme"})
cred.verify().valid       # True, from the bytes alone
cred.admits("retrieve")   # True, folds into the capability-gated admission path
```

A tampered claim or a forged issuer is caught the way every other content-bound
artifact is.

## Accountable audit, contracts & settlements

Because an `AgentIdentity` is a drop-in `ChainSigner`, binding it as the app's signer
makes every signed artifact record the identity's **DID**:

```python
app.use_identity(agent)                       # bind as the content / contract / audit signer
entry = app.audit.record("invoice_issued", resource="INV-1042", decision="allow")
entry.key_id == agent.did                     # True
```

Subsequent audit entries, negotiated contracts, settlement records, and signed
manifests all carry the DID as their `key_id`, so a verifier resolves the signer from
the identifier and checks the signature from the bytes, accountability becomes a
cryptographic fact, not a logged string. (The app adopts the identity as the
audit-chain signer only when the log is still empty, so the chain stays verifiable
under one key; bind the identity at startup.)

## Best practice & gotchas

- **Bind the identity at startup, before the log has entries.**
  `app.use_identity(agent)` adopts the identity as the audit-chain signer *only
  while the chain is still empty*, so it stays verifiable under one key. Do it
  first thing; you cannot swap the chain signer mid-run without breaking
  single-key verification.
- **Signatures are time-indexed, not latest-key-wins.** After `rotate()`, old
  signatures stay valid because the rotation chain proves the new key descends
  from the one that signed them; `IdentityDocument.verify_signature(msg, sig,
  at=...)` reports which key produced a signature *and* whether that key was
  active at a given instant.
- **Revoking the active key rotates first.** Revocation mints a fresh key (so a
  signer always remains), then marks the old key revoked from that moment —
  history it legitimately signed before stays verifiable; new forgeries with the
  compromised key do not.
- **Attenuation only ever narrows.** A sub-delegation that *adds* a capability,
  *raises* the budget cap, or *extends* the expiry is refused from the bytes
  (`attenuation_ok == False`). Put `chain.require_permits(...)` in front of the
  privileged call so an over-reach fails closed with an `IdentityError` rather
  than proceeding unauthorized.
- **No registry to keep in sync.** A verifier resolves the key with
  `public_key_from_did`, and a link signed with a *rotated* key carries a compact
  `KeyAuthorization`, so the whole chain verifies offline with no network call.
- **Install the native backend for production.** The bundled pure-Python Ed25519
  is byte-for-byte identical, but install `pip install "vincio[crypto]"` for the
  constant-time `cryptography` implementation under load.

## See also

- The runnable example: [`examples/09_security_governance.py`](../../examples/09_security_governance.py)
- [SLOs](../reference/slo.md#agent-identity-delegation--cryptographic-accountability), the identity-integrity and delegation-attenuation budgets
- [Audit integrity](../../SECURITY.md), how the identity binds to the hash-chained audit log

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Example: 09_security_governance.py](../../examples/09_security_governance.py)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#governance)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
