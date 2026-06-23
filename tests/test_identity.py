"""Agent identity, delegation & cryptographic accountability (3.46).

Covers the self-certifying DID, the pure-Python Ed25519 kernel (against the
RFC 8032 vectors), the signed rotation chain (a rotated/revoked key cannot forge
new history while old signatures stay valid), delegation attenuation (each link
only narrows authority — an over-reaching sub-delegation is refused from the
bytes), verifiable credentials, and the app-facade binding that makes every audit
entry carry the signer's DID.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from vincio import (
    AgentCredential,
    AgentIdentity,
    Delegation,
    DelegationChain,
    Grant,
    IdentityDocument,
    Keyring,
    did_from_public_key,
    is_vincio_did,
    public_key_from_did,
)
from vincio.core.errors import IdentityError
from vincio.core.utils import utcnow
from vincio.security import _ed25519 as ed

# ---------------------------------------------------------------------------
# Ed25519 kernel — RFC 8032 conformance
# ---------------------------------------------------------------------------

RFC8032_VECTORS = [
    (
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
    (
        "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "af82",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
    ),
]


@pytest.mark.parametrize("seed_h, pub_h, msg_h, sig_h", RFC8032_VECTORS)
def test_ed25519_rfc8032_vectors(seed_h, pub_h, msg_h, sig_h):
    seed = bytes.fromhex(seed_h)
    msg = bytes.fromhex(msg_h)
    assert ed.public_key_from_seed(seed).hex() == pub_h
    sig = ed.sign(seed, msg)
    assert sig.hex() == sig_h  # deterministic, byte-for-byte RFC 8032
    assert ed.verify(bytes.fromhex(pub_h), msg, sig)


def test_ed25519_rejects_tamper():
    seed = bytes.fromhex(RFC8032_VECTORS[1][0])
    pub = ed.public_key_from_seed(seed)
    sig = ed.sign(seed, b"hello")
    assert ed.verify(pub, b"hello", sig)
    assert not ed.verify(pub, b"hell0", sig)
    bad = bytearray(sig)
    bad[0] ^= 1
    assert not ed.verify(pub, b"hello", bytes(bad))
    # A non-canonical / wrong-length signature is refused, not raised.
    assert not ed.verify(pub, b"hello", b"\x00" * 64)
    assert not ed.verify(pub, b"hello", b"short")


def test_ed25519_seed_validation():
    with pytest.raises(ValueError):
        ed.public_key_from_seed(b"too-short")
    with pytest.raises(ValueError):
        ed.sign(b"too-short", b"x")


# ---------------------------------------------------------------------------
# DID — self-certifying & offline-resolvable
# ---------------------------------------------------------------------------


def test_did_round_trips_with_public_key():
    identity = AgentIdentity.generate("agent", seed=b"\x01" * 32)
    pub = bytes.fromhex(identity.did[len("did:vincio:ed25519:") :])
    assert did_from_public_key(pub) == identity.did
    assert public_key_from_did(identity.did) == pub
    assert is_vincio_did(identity.did)


def test_did_rejects_malformed():
    assert not is_vincio_did("did:example:123")
    with pytest.raises(IdentityError):
        public_key_from_did("did:example:123")
    with pytest.raises(IdentityError):
        public_key_from_did("did:vincio:ed25519:zz")
    with pytest.raises(IdentityError):
        did_from_public_key(b"\x00" * 16)


def test_identity_seed_is_deterministic():
    a = AgentIdentity.generate("a", seed=b"\x07" * 32)
    b = AgentIdentity.generate("b", seed=b"\x07" * 32)
    assert a.did == b.did  # same seed → same DID


# ---------------------------------------------------------------------------
# Identity document & ChainSigner protocol
# ---------------------------------------------------------------------------


def test_document_verifies_offline():
    identity = AgentIdentity.generate("agent", capabilities=["retrieve"], seed=b"\x02" * 32)
    doc = identity.document
    assert doc.verify().valid
    assert doc.subject == identity.did
    assert "retrieve" in doc.capabilities


def test_document_tamper_detected():
    identity = AgentIdentity.generate("agent", seed=b"\x02" * 32)
    doc = IdentityDocument.from_wire(identity.document.to_wire())
    assert doc.verify().valid
    doc.capabilities.append("write")  # tamper after sealing
    result = doc.verify()
    assert not result.valid
    assert not result.hash_ok


def test_identity_is_chainsigner():
    from vincio.security.audit import ChainSigner

    identity = AgentIdentity.generate("agent", seed=b"\x03" * 32)
    assert isinstance(identity, ChainSigner)
    assert identity.key_id == identity.did
    sig = identity.sign("payload")
    assert identity.verify("payload", sig)
    assert not identity.verify("payload!", sig)


# ---------------------------------------------------------------------------
# Rotation & revocation — identity integrity
# ---------------------------------------------------------------------------


def test_rotation_keeps_old_signatures_valid():
    identity = AgentIdentity.generate("agent", seed=b"\x04" * 32)
    before = identity.sign("legacy-message")
    genesis_kid = identity.document.active_key.kid
    identity.rotate()
    assert identity.document.active_key.kid != genesis_kid
    assert identity.document.verify().valid
    # The signature made under the genesis key still verifies after rotation.
    assert identity.verify("legacy-message", before)
    # And the new active key signs new history.
    after = identity.sign("new-message")
    assert identity.verify("new-message", after)


def test_rotation_chain_is_signed():
    identity = AgentIdentity.generate("agent", seed=b"\x05" * 32)
    identity.rotate()
    identity.rotate()
    assert len(identity.document.keys) == 3
    assert identity.document.verify().rotation_chain_ok
    # Forge a key record into the chain with no valid rotation signature.
    doc = IdentityDocument.from_wire(identity.document.to_wire())
    doc.keys[-1].rotation_sig = "00" * 64
    assert not doc.verify().rotation_chain_ok


def test_revoked_key_cannot_forge_new_history():
    identity = AgentIdentity.generate("agent", seed=b"\x06" * 32)
    compromised_kid = identity.document.active_key.kid
    identity.revoke(compromised_kid)
    # The document still verifies (the revocation is authentic, signed by a fresh key).
    assert identity.document.verify().valid
    revoked = identity.document.resolve(compromised_kid)
    assert revoked.status == "revoked"
    assert revoked.revoked_at is not None
    # A signature the revoked key produces *after* revocation is rejected as of now.
    forged = ed.sign(b"\x06" * 32, b"after-compromise").hex()
    check = identity.document.verify_signature("after-compromise", forged, at=utcnow())
    assert check.valid  # the bytes match the (now-revoked) key
    assert not check.active_at_check  # but the key was not active at signing time


def test_verify_signature_time_pinned():
    identity = AgentIdentity.generate("agent", seed=b"\x08" * 32)
    sig = identity.sign("doc-msg")
    kid = identity.document.active_key.kid
    check = identity.document.verify_signature("doc-msg", sig, at=utcnow())
    assert check.valid and check.active_at_check and check.kid == kid


# ---------------------------------------------------------------------------
# Grant attenuation
# ---------------------------------------------------------------------------


def test_grant_attenuation_rules():
    parent = Grant(capabilities=["a", "b", "c"], budget_usd=100.0, max_delegations=2)
    assert Grant(capabilities=["a"], budget_usd=50.0, max_delegations=1).attenuates(parent)
    # capability not in parent
    assert not Grant(capabilities=["a", "z"], budget_usd=10.0, max_delegations=1).attenuates(parent)
    # larger budget
    assert not Grant(capabilities=["a"], budget_usd=200.0, max_delegations=1).attenuates(parent)
    # unbounded budget under a bounded parent
    assert not Grant(capabilities=["a"], budget_usd=None, max_delegations=1).attenuates(parent)
    # depth not decremented
    assert not Grant(capabilities=["a"], budget_usd=10.0, max_delegations=2).attenuates(parent)


def test_grant_wildcard_cannot_be_introduced():
    parent = Grant(capabilities=["read"])
    assert not Grant(capabilities=["*"]).attenuates(parent)
    wild = Grant(capabilities=["*"], max_delegations=1)
    assert Grant(capabilities=["read"], max_delegations=0).attenuates(wild)


def test_grant_expiry_and_audience_attenuation():
    now = utcnow()
    parent = Grant(capabilities=["x"], not_after=now + timedelta(days=2), audience="org-a")
    assert Grant(
        capabilities=["x"], not_after=now + timedelta(days=1), audience="org-a"
    ).attenuates(parent)
    # later expiry amplifies
    assert not Grant(
        capabilities=["x"], not_after=now + timedelta(days=3), audience="org-a"
    ).attenuates(parent)
    # broadening the audience amplifies
    assert not Grant(capabilities=["x"], not_after=now, audience="org-b").attenuates(parent)
    assert not Grant(capabilities=["x"], not_after=now, audience="").attenuates(parent)


# ---------------------------------------------------------------------------
# Delegation chains — delegation attenuation SLO
# ---------------------------------------------------------------------------


def _three_identities():
    return (
        AgentIdentity.generate("principal", seed=b"\x11" * 32),
        AgentIdentity.generate("agent", seed=b"\x12" * 32),
        AgentIdentity.generate("subagent", seed=b"\x13" * 32),
    )


def test_delegation_chain_verifies_and_permits():
    principal, agent, sub = _three_identities()
    d1 = principal.delegate(
        agent, capabilities=["retrieve", "summarize"], budget_usd=100.0, max_delegations=2
    )
    d2 = d1.delegate(agent, sub, capabilities=["retrieve"], budget_usd=40.0)
    chain = DelegationChain(links=[d1, d2])
    v = chain.verify(root_issuer=principal.did)
    assert v.valid, v.reason
    assert chain.principal == principal.did
    assert chain.subject == sub.did
    assert chain.permits("retrieve", budget_usd=30.0)
    assert not chain.permits("summarize")  # attenuated away at the leaf
    assert not chain.permits("retrieve", budget_usd=50.0)  # over the leaf budget


def test_over_reaching_sub_delegation_refused():
    principal, agent, sub = _three_identities()
    d1 = principal.delegate(agent, capabilities=["retrieve"], budget_usd=40.0, max_delegations=2)
    # Forge a child that amplifies (adds a capability the parent never had).
    forged = d1.delegate(
        agent, sub, grant=Grant(capabilities=["retrieve", "write"], budget_usd=40.0)
    )
    chain = DelegationChain(links=[d1, forged])
    result = chain.verify(root_issuer=principal.did)
    assert not result.valid
    assert not result.attenuation_ok
    assert "amplif" in result.reason
    with pytest.raises(IdentityError):
        chain.require_permits("write")


def test_delegation_signature_tamper_detected():
    principal, agent, _ = _three_identities()
    d1 = principal.delegate(agent, capabilities=["retrieve"], budget_usd=10.0)
    assert d1.verify().valid
    tampered = Delegation.from_wire(d1.to_wire())
    tampered.grant.budget_usd = 999.0  # raise the cap after signing
    assert not tampered.verify().valid


def test_delegation_only_subject_can_sub_delegate():
    principal, agent, sub = _three_identities()
    d1 = principal.delegate(agent, capabilities=["retrieve"], budget_usd=10.0, max_delegations=1)
    # `sub` is not the delegate of d1, so it cannot sub-delegate from it.
    with pytest.raises(IdentityError):
        d1.delegate(sub, agent, capabilities=["retrieve"])


def test_delegation_chain_linkage_required():
    principal, agent, sub = _three_identities()
    d1 = principal.delegate(agent, capabilities=["retrieve"], budget_usd=40.0, max_delegations=2)
    other = AgentIdentity.generate("other", seed=b"\x14" * 32)
    # A link whose issuer is not the previous subject breaks the chain.
    stray = other.delegate(sub, capabilities=["retrieve"], budget_usd=10.0)
    chain = DelegationChain(links=[d1, stray])
    assert not chain.verify(root_issuer=principal.did).linkage_ok


def test_delegation_expiry_enforced():
    principal, agent, _ = _three_identities()
    past = utcnow() - timedelta(days=1)
    d1 = principal.delegate(agent, capabilities=["retrieve"], not_after=past)
    chain = DelegationChain(links=[d1])
    assert not chain.verify().not_expired
    assert not chain.permits("retrieve")


def test_delegation_chain_root_issuer_enforced():
    principal, agent, sub = _three_identities()
    d1 = principal.delegate(agent, capabilities=["retrieve"], budget_usd=40.0, max_delegations=2)
    d2 = d1.delegate(agent, sub, capabilities=["retrieve"], budget_usd=10.0)
    chain = DelegationChain(links=[d1, d2])
    assert chain.verify(root_issuer=principal.did).valid
    assert not chain.verify(root_issuer=sub.did).linkage_ok


def test_delegation_signed_with_rotated_key_verifies():
    principal, agent, _ = _three_identities()
    principal.rotate()  # now signing with a non-genesis key
    d1 = principal.delegate(agent, capabilities=["retrieve"], budget_usd=10.0)
    assert d1.authority is not None  # carries the rotation proof
    assert d1.verify().valid  # still offline-verifiable from the bytes
    # Forging the authority path (claiming a key not descended from genesis) fails.
    forged = Delegation.from_wire(d1.to_wire())
    forged.authority.path[0].public_key = "00" * 32
    assert not forged.verify().valid


# ---------------------------------------------------------------------------
# Verifiable credentials
# ---------------------------------------------------------------------------


def test_credential_verify_and_admits():
    org = AgentIdentity.generate("org-acme", seed=b"\x21" * 32)
    agent = AgentIdentity.generate("agent", seed=b"\x22" * 32)
    cred = org.issue_credential(
        agent, {"admitted_capability": "retrieve", "operated_by": "org-acme"}
    )
    assert cred.verify().valid
    assert cred.admits("retrieve")
    assert not cred.admits("write")
    assert "retrieve" in cred.admitted_capabilities


def test_credential_multiple_capabilities():
    org = AgentIdentity.generate("org", seed=b"\x23" * 32)
    agent = AgentIdentity.generate("agent", seed=b"\x24" * 32)
    cred = org.issue_credential(agent, {"admitted_capabilities": "retrieve, summarize ,rank"})
    assert set(cred.admitted_capabilities) == {"retrieve", "summarize", "rank"}
    assert cred.admits("rank")


def test_credential_tamper_and_expiry():
    org = AgentIdentity.generate("org", seed=b"\x25" * 32)
    agent = AgentIdentity.generate("agent", seed=b"\x26" * 32)
    cred = org.issue_credential(agent, {"admitted_capability": "retrieve"})
    tampered = AgentCredential.from_wire(cred.to_wire())
    tampered.claims["admitted_capability"] = "write"
    assert not tampered.verify().valid
    expired = org.issue_credential(
        agent, {"admitted_capability": "retrieve"}, not_after=utcnow() - timedelta(days=1)
    )
    assert not expired.verify().valid
    assert not expired.admits("retrieve")
    with pytest.raises(IdentityError):
        expired.require_valid()


def test_credential_must_be_signed_by_issuer():
    org = AgentIdentity.generate("org", seed=b"\x27" * 32)
    other = AgentIdentity.generate("other", seed=b"\x28" * 32)
    cred = AgentCredential(issuer=org.did, subject=other.did, claims={"x": "y"})
    with pytest.raises(IdentityError):
        cred.sign(other)  # not the issuer


# ---------------------------------------------------------------------------
# Keyring direct API
# ---------------------------------------------------------------------------


def test_keyring_create_and_rotate():
    keyring = Keyring.create(name="svc", capabilities=["a"], seed=b"\x31" * 32)
    assert keyring.document.verify().valid
    assert keyring.authorization() is None  # genesis key needs no proof
    keyring.rotate()
    assert keyring.authorization() is not None  # rotated key carries a proof
    assert keyring.document.verify().valid


def test_keyring_create_validates_seed():
    with pytest.raises(IdentityError):
        Keyring.create(seed=b"short")


# ---------------------------------------------------------------------------
# App-facade binding — accountable audit
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    from vincio import ContextApp, VincioConfig
    from vincio.providers import MockProvider

    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    return ContextApp(name="acme", provider=MockProvider(default_text="ok"), config=cfg)


def test_app_identity_minted_and_audited(app):
    identity = app.identity("billing-agent", capabilities=["retrieve"])
    assert is_vincio_did(identity.did)
    actions = [e.action for e in app.audit.entries]
    assert "identity" in actions


def test_app_use_identity_binds_did_to_audit(app):
    identity = app.identity("acme-signer", use=True)
    # The bound identity signs subsequent audit entries with its DID as key_id.
    entry = app.audit.record("custom", resource="r", decision="allow")
    assert entry.key_id == identity.did
    verdict = app.audit.verify_chain()
    assert getattr(verdict, "valid", verdict) is True


def test_app_issue_credential_requires_identity(app):
    agent = app.identity("agent")
    with pytest.raises(IdentityError):
        app.issue_credential(agent, {"admitted_capability": "retrieve"})
    app.identity("org", use=True)
    cred = app.issue_credential(agent, {"admitted_capability": "retrieve"})
    assert cred.verify().valid
    assert any(e.action == "credential" for e in app.audit.entries)


def test_app_bound_identity_signs_contracts(app):
    """An app bound to an identity signs negotiated contracts with its DID."""
    identity = app.identity("acme", use=True)
    signer = app._resolve_contract_signer(None, True)
    assert signer is identity
    assert signer.key_id == identity.did
