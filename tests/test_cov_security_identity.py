"""Targeted coverage for ``vincio.security.identity``.

Hits the error/edge paths the broad suite leaves uncovered: malformed-DID
recovery, every branch of ``Grant.permits`` and the attenuation helpers,
``KeyRecord.active_at`` boundaries, ``IdentityDocument.verify`` failure reasons,
rotation-chain refusal, the ``_verify_authority`` rotation-path checks, empty /
broken / over-reaching delegation chains, ``require_permits`` / ``require_valid``
raising, and the keyring no-active-key guards. Everything is offline and
deterministic via fixed 32-byte seeds — no network, no mocks.
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
    Keyring,
    did_from_public_key,
    is_vincio_did,
    public_key_from_did,
)
from vincio.core.errors import IdentityError
from vincio.core.utils import utcnow
from vincio.security import _ed25519 as ed
from vincio.security.identity import (
    DID_PREFIX,
    KeyAuthorization,
    KeyRecord,
    key_fingerprint,
)


def _seed(b: int) -> bytes:
    return bytes([b]) * 32


def _identity(b: int, *, name: str = "", caps: list[str] | None = None) -> AgentIdentity:
    return AgentIdentity.generate(name, capabilities=caps, seed=_seed(b))


# ---------------------------------------------------------------------------
# DID helpers — malformed recovery paths (lines 130-131, 133)
# ---------------------------------------------------------------------------


def test_public_key_from_did_rejects_non_hex_material():
    bad = DID_PREFIX + "z" * 64  # right length, not hex
    with pytest.raises(IdentityError, match="malformed DID key material"):
        public_key_from_did(bad)


def test_public_key_from_did_rejects_non_vincio():
    with pytest.raises(IdentityError, match="not a vincio DID"):
        public_key_from_did("did:example:123")


def test_is_vincio_did_rejects_wrong_length_and_nonstring():
    assert is_vincio_did(DID_PREFIX + "ab") is False
    assert is_vincio_did(12345) is False  # type: ignore[arg-type]


def test_did_round_trip_recovers_exact_key():
    pub = ed.public_key_from_seed(_seed(7))
    assert public_key_from_did(did_from_public_key(pub)) == pub


def test_did_from_public_key_rejects_wrong_size():
    with pytest.raises(IdentityError, match="32 bytes"):
        did_from_public_key(b"\x00" * 31)


# ---------------------------------------------------------------------------
# Grant.permits — the budget / expiry / audience branches (lines 245, 247)
# ---------------------------------------------------------------------------


def test_grant_permits_budget_over_cap_refused():
    g = Grant(capabilities=["pay"], budget_usd=100.0)
    assert g.permits("pay", budget_usd=100.0) is True
    assert g.permits("pay", budget_usd=100.01) is False


def test_grant_permits_expiry_boundary():
    cutoff = utcnow()
    g = Grant(capabilities=["x"], not_after=cutoff)
    assert g.permits("x", as_of=cutoff) is True
    assert g.permits("x", as_of=cutoff + timedelta(seconds=1)) is False


def test_grant_permits_audience_mismatch_refused():
    g = Grant(capabilities=["x"], audience="org:acme")
    assert g.permits("x", audience="org:acme") is True
    assert g.permits("x", audience="org:evil") is False
    # An unfixed-audience grant ignores the requested audience.
    assert Grant(capabilities=["x"]).permits("x", audience="anything") is True


def test_grant_permits_capability_not_in_set():
    assert Grant(capabilities=["read"]).permits("write") is False


# ---------------------------------------------------------------------------
# Attenuation helper branches (lines 273, 281)
# ---------------------------------------------------------------------------


def test_unbounded_child_under_bounded_parent_is_amplification():
    parent = Grant(capabilities=["*"], budget_usd=100.0)
    child = Grant(capabilities=["pay"], budget_usd=None)
    assert child.attenuates(parent) is False  # budget widened


def test_expiryless_child_under_expiring_parent_refused():
    cutoff = utcnow()
    parent = Grant(capabilities=["*"], not_after=cutoff)
    child = Grant(capabilities=["x"], not_after=None)
    assert child.attenuates(parent) is False  # expiry widened


def test_depleted_parent_grants_no_further_delegation():
    parent = Grant(capabilities=["*"], max_delegations=0)
    child = Grant(capabilities=["x"], max_delegations=0)
    assert child.attenuates(parent) is False  # parent depleted


def test_child_without_depth_under_bounded_parent_refused():
    parent = Grant(capabilities=["*"], max_delegations=2)
    child = Grant(capabilities=["x"], max_delegations=None)
    assert child.attenuates(parent) is False


def test_depth_attenuates_consumes_one_hop():
    parent = Grant(capabilities=["*"], max_delegations=2)
    assert Grant(capabilities=["x"], max_delegations=1).attenuates(parent) is True
    assert Grant(capabilities=["x"], max_delegations=2).attenuates(parent) is False


def test_child_cannot_introduce_wildcard():
    parent = Grant(capabilities=["read"])
    child = Grant(capabilities=["*"])
    assert child.attenuates(parent) is False  # wildcard introduced


def test_child_capabilities_must_be_subset():
    parent = Grant(capabilities=["read"])
    assert Grant(capabilities=["read", "write"]).attenuates(parent) is False
    assert Grant(capabilities=["read"]).attenuates(parent) is True


def test_audience_broadening_refused():
    parent = Grant(capabilities=["*"], audience="org:acme")
    assert Grant(capabilities=["x"], audience="org:other").attenuates(parent) is False
    assert Grant(capabilities=["x"], audience="org:acme").attenuates(parent) is True


def test_larger_child_budget_refused_smaller_allowed():
    parent = Grant(capabilities=["*"], budget_usd=100.0)
    assert Grant(capabilities=["x"], budget_usd=150.0).attenuates(parent) is False
    assert Grant(capabilities=["x"], budget_usd=50.0).attenuates(parent) is True


def test_later_child_expiry_refused():
    cutoff = utcnow()
    parent = Grant(capabilities=["*"], not_after=cutoff)
    assert Grant(capabilities=["x"], not_after=cutoff + timedelta(hours=1)).attenuates(parent) is (
        False
    )
    assert Grant(capabilities=["x"], not_after=cutoff - timedelta(hours=1)).attenuates(parent) is (
        True
    )


# ---------------------------------------------------------------------------
# KeyRecord.active_at boundaries (lines 333, 335)
# ---------------------------------------------------------------------------


def test_key_record_active_at_boundaries():
    created = utcnow()
    rec = KeyRecord(
        kid="k1",
        public_key=ed.public_key_from_seed(_seed(1)).hex(),
        created_at=created,
        not_after=created + timedelta(hours=1),
    )
    assert rec.active_at(created - timedelta(seconds=1)) is False  # before creation
    assert rec.active_at(created) is True
    assert rec.active_at(created + timedelta(hours=2)) is False  # past not_after


def test_key_record_revoked_inactive_at_or_after_revocation():
    created = utcnow()
    revoked = created + timedelta(hours=1)
    rec = KeyRecord(
        kid="k1",
        public_key=ed.public_key_from_seed(_seed(1)).hex(),
        created_at=created,
        status="revoked",
        revoked_at=revoked,
    )
    assert rec.active_at(created) is True  # signed before revocation
    assert rec.active_at(revoked) is False  # at the instant of revocation


# ---------------------------------------------------------------------------
# IdentityDocument.verify — failure reasons (lines 465, 476, 495, 498-503)
# ---------------------------------------------------------------------------


def test_unsealed_document_reports_not_sealed():
    doc = _identity(2).document.model_copy(deep=True)
    doc.content_hash = ""
    result = doc.verify()
    assert result.valid is False
    assert result.reason == "document is not sealed"


def test_tampered_facts_break_content_hash():
    doc = _identity(2).document.model_copy(deep=True)
    doc.capabilities = ["sneaky"]  # facts changed, hash now stale
    result = doc.verify()
    assert result.valid is False
    assert result.hash_ok is False
    assert result.reason == "content hash does not match the document facts"


def test_subject_did_must_match_genesis_key():
    ident = _identity(2)
    doc = ident.document.model_copy(deep=True)
    other = ed.public_key_from_seed(_seed(99))
    doc.subject = did_from_public_key(other)  # DID no longer matches genesis
    doc.seal()  # re-hash so hash_ok passes and we reach the DID check
    result = doc.verify()
    assert result.valid is False
    assert result.did_matches_genesis is False
    assert result.reason == "subject DID does not match the genesis key"


def test_rotation_chain_refuses_missing_predecessor():
    ident = _identity(2)
    ident.rotate(seed=_seed(3))
    doc = ident.document.model_copy(deep=True)
    # Break the link: child references a predecessor kid that is not present.
    doc.keys[1].prev_kid = "k_missing"
    doc.seal()
    result = doc.verify()
    assert result.rotation_chain_ok is False
    assert result.reason == "rotation chain does not verify"


def test_rotation_chain_refuses_forged_rotation_signature():
    ident = _identity(2)
    ident.rotate(seed=_seed(3))
    doc = ident.document.model_copy(deep=True)
    doc.keys[1].rotation_sig = "00" * 64  # predecessor never signed this
    doc.seal()
    result = doc.verify()
    assert result.rotation_chain_ok is False


def test_rotation_chain_refuses_backdated_after_revocation():
    ident = _identity(2)
    ident.rotate(seed=_seed(3))  # genesis -> k2
    ident.revoke(ident.document.keys[0].kid)  # revoke genesis
    doc = ident.document.model_copy(deep=True)
    genesis = doc.keys[0]
    # Forge a rotation dated strictly after the genesis was revoked.
    doc.keys[1].created_at = genesis.revoked_at + timedelta(seconds=1)
    doc.keys[1].rotation_sig = ""  # also recompute below
    # Re-sign the rotation statement as if genesis were still current is impossible
    # (its seed is dropped), so we only assert the date guard refuses the chain.
    doc.seal()
    result = doc.verify()
    assert result.rotation_chain_ok is False


def test_document_signature_failure_reason():
    ident = _identity(2)
    doc = ident.document.model_copy(deep=True)
    # Corrupt the signature but keep facts/hash intact.
    doc.signature = "00" * 64
    result = doc.verify()
    assert result.valid is False
    assert result.signature_ok is False
    assert result.reason == "document signature does not verify"


# ---------------------------------------------------------------------------
# IdentityDocument key lookup fallbacks (lines 420->419, 422, 433)
# ---------------------------------------------------------------------------


def test_genesis_key_none_for_empty_keys():
    from vincio.security.identity import IdentityDocument

    doc = IdentityDocument(subject=_identity(2).did, keys=[])
    assert doc.genesis_key is None
    assert doc.active_key is None
    assert doc.resolve("nope") is None


def test_document_audit_details_and_wire_round_trip():
    ident = _identity(2, name="bot", caps=["read"])
    details = ident.document.audit_details()
    assert details["subject"] == ident.did
    assert details["name"] == "bot"
    assert details["keys"] == 1
    assert details["active_kid"] == ident.document.active_key.kid
    restored = type(ident.document).from_wire(ident.document.to_wire())
    assert restored.verify().valid is True
    assert restored.subject == ident.did


def test_document_verify_skips_signature_when_absent():
    ident = _identity(2)
    doc = ident.document.model_copy(deep=True)
    doc.signature = ""  # no signature -> sig check is vacuously ok
    result = doc.verify()
    assert result.signature_ok is True
    assert result.valid is True


def test_key_authorization_facts_lists_records():
    ident = _identity(2)
    ident.rotate(seed=_seed(3))
    auth = ident.keyring.authorization()
    facts = auth.facts()
    assert len(facts) == 2
    assert facts[0]["prev_kid"] == ""
    assert facts[1]["prev_kid"] == facts[0]["kid"]


def test_document_rotation_chain_refuses_document_level_backdate():
    from vincio.security.identity import IdentityDocument

    # Hand-build a document whose child rotation is validly signed but dated AFTER
    # the genesis was revoked: the document-level chain check must refuse it.
    genesis_seed, child_seed = _seed(82), _seed(83)
    g_pub = ed.public_key_from_seed(genesis_seed)
    c_pub = ed.public_key_from_seed(child_seed)
    did = did_from_public_key(g_pub)
    t0 = utcnow()
    genesis = KeyRecord(
        kid=key_fingerprint(g_pub),
        public_key=g_pub.hex(),
        created_at=t0,
        status="revoked",
        revoked_at=t0 + timedelta(minutes=1),
    )
    child = KeyRecord(
        kid=key_fingerprint(c_pub),
        public_key=c_pub.hex(),
        created_at=t0 + timedelta(minutes=5),  # after the genesis revocation
        status="active",
        prev_kid=genesis.kid,
    )
    child.rotation_sig = ed.sign(
        genesis_seed, child.rotation_message(did).encode("utf-8")
    ).hex()
    doc = IdentityDocument(subject=did, keys=[genesis, child])
    assert doc._verify_rotation_chain() is False
    result = doc.seal().verify()
    assert result.rotation_chain_ok is False


def test_resolve_returns_named_key():
    ident = _identity(2)
    kid = ident.document.active_key.kid
    assert ident.document.resolve(kid).kid == kid
    assert ident.document.resolve("absent") is None


# ---------------------------------------------------------------------------
# verify_signature — kid path & no-match (lines 526, 538)
# ---------------------------------------------------------------------------


def test_verify_signature_by_named_kid():
    ident = _identity(2)
    msg = "audit-entry-42"
    sig = ident.sign(msg)
    kid = ident.document.active_key.kid
    check = ident.document.verify_signature(msg, sig, kid=kid)
    assert check.valid is True
    assert check.kid == kid


def test_verify_signature_unknown_kid_resolves_to_none():
    ident = _identity(2)
    sig = ident.sign("m")
    # kid given but absent -> the single candidate is None and is skipped.
    check = ident.document.verify_signature("m", sig, kid="ghost")
    assert check.valid is False
    assert check.reason == "no key in the document produced this signature"


def test_verify_signature_no_key_matches():
    ident = _identity(2)
    check = ident.document.verify_signature("msg", "00" * 64)
    assert check.valid is False
    assert check.reason == "no key in the document produced this signature"


def test_verify_signature_inactive_at_reports_reason():
    ident = _identity(2)
    msg = "old-message"
    sig = ident.sign(msg)
    before_birth = ident.document.active_key.created_at - timedelta(days=1)
    check = ident.document.verify_signature(msg, sig, as_of=before_birth)
    assert check.valid is True  # the signature is genuine
    assert check.active_at_check is False
    assert "was not active at" in check.reason


# ---------------------------------------------------------------------------
# _hexbytes ValueError swallow (lines 564-565)
# ---------------------------------------------------------------------------


def test_signature_with_non_hex_does_not_crash():
    ident = _identity(2)
    # A non-hex signature string flows through _hexbytes -> b"" and fails cleanly.
    check = ident.document.verify_signature("msg", "nothex!!")
    assert check.valid is False


# ---------------------------------------------------------------------------
# Keyring guards (lines 640, 664, 696, 731, 746)
# ---------------------------------------------------------------------------


def test_keyring_active_kid_raises_without_active_key():
    kr = Keyring.create(seed=_seed(5))
    kr.document.keys[0].status = "retired"
    with pytest.raises(IdentityError, match="no active key"):
        _ = kr.active_kid


def test_keyring_active_public_raises_without_active_key():
    kr = Keyring.create(seed=_seed(5))
    kr.document.keys[0].status = "retired"
    with pytest.raises(IdentityError, match="no active key"):
        kr.active_public()


def test_keyring_revoke_unknown_kid_raises():
    kr = Keyring.create(seed=_seed(5))
    with pytest.raises(IdentityError, match="unknown key 'ghost'"):
        kr.revoke("ghost")


def test_keyring_revoke_non_active_key_directly():
    kr = Keyring.create(seed=_seed(5))
    genesis_kid = kr.active_kid
    kr.rotate(seed=_seed(6))  # genesis now retired, k2 active
    revoked = kr.revoke(genesis_kid)  # revoke the (non-active) old key, no extra rotation
    assert revoked.kid == genesis_kid
    assert revoked.status == "revoked"
    assert kr.active_kid != genesis_kid  # active key unchanged by revoking an old key


def test_keyring_create_validates_seed_length():
    with pytest.raises(IdentityError, match="32 bytes"):
        Keyring.create(seed=b"\x00" * 16)


def test_keyring_signer_wraps_into_identity():
    kr = Keyring.create(name="payer", seed=_seed(5))
    ident = kr.signer()
    assert isinstance(ident, AgentIdentity)
    assert ident.did == kr.did
    assert ident.name == "payer"


def test_keyring_authorization_none_without_rotation():
    kr = Keyring.create(seed=_seed(5))
    assert kr.authorization() is None  # active key is genesis


def test_keyring_authorization_path_after_rotation():
    kr = Keyring.create(seed=_seed(5))
    kr.rotate(seed=_seed(6))
    auth = kr.authorization()
    assert auth is not None
    assert len(auth.path) == 2
    assert auth.path[0].prev_kid == ""  # genesis first
    assert auth.path[1].prev_kid == auth.path[0].kid


# ---------------------------------------------------------------------------
# Keyring rotation chain & revocation behaviour
# ---------------------------------------------------------------------------


def test_revoke_active_rotates_then_marks_revoked():
    kr = Keyring.create(seed=_seed(5))
    genesis_kid = kr.active_kid
    revoked = kr.revoke()  # revoke active -> rotates away first
    assert revoked.kid == genesis_kid
    assert revoked.status == "revoked"
    assert revoked.revoked_at is not None
    assert kr.active_kid != genesis_kid  # a fresh signer remains
    # The revoked key's seed is dropped so it can never sign again.
    assert genesis_kid not in kr._seeds


def test_rotation_retires_prior_key_and_signs_chain():
    kr = Keyring.create(seed=_seed(5))
    genesis = kr.document.keys[0]
    new = kr.rotate(seed=_seed(6))
    assert genesis.status == "retired"
    assert new.prev_kid == genesis.kid
    # The genesis key really signed the new key's rotation statement.
    message = new.rotation_message(kr.did).encode("utf-8")
    assert ed.verify(genesis.public_bytes(), message, bytes.fromhex(new.rotation_sig)) is True
    assert kr.document.verify().valid is True


# ---------------------------------------------------------------------------
# AgentIdentity reads & generate (lines 794, 806)
# ---------------------------------------------------------------------------


def test_identity_generate_exposes_capabilities():
    ident = _identity(8, name="bot", caps=["read", "write"])
    assert ident.capabilities == ["read", "write"]
    assert ident.name == "bot"
    assert ident.key_id == ident.did


def test_identity_rotate_keeps_old_signature_valid():
    ident = _identity(8)
    msg = "pre-rotation"
    sig = ident.sign(msg)
    assert ident.verify(msg, sig) is True
    ident.rotate(seed=_seed(9))
    assert ident.verify(msg, sig) is True  # old signature still resolves


def test_identity_revoke_via_facade():
    ident = _identity(8)
    genesis_kid = ident.document.active_key.kid
    revoked = ident.revoke()
    assert revoked.kid == genesis_kid
    assert revoked.status == "revoked"
    assert ident.document.active_key.kid != genesis_kid


def test_keyring_rotate_without_active_key_raises():
    kr = Keyring.create(seed=_seed(5))
    kr.document.keys[0].status = "retired"
    with pytest.raises(IdentityError, match="cannot rotate an identity with no active key"):
        kr.rotate(seed=_seed(6))


# ---------------------------------------------------------------------------
# Delegation signing guards (lines 1009, 1076, 1081)
# ---------------------------------------------------------------------------


def test_delegation_must_be_signed_by_issuer():
    issuer = _identity(10)
    impostor = _identity(11)
    deleg = Delegation(issuer=issuer.did, subject=impostor.did, grant=Grant(capabilities=["x"]))
    with pytest.raises(IdentityError, match="signed by its issuer"):
        deleg.sign(impostor)


def test_delegation_verify_unsigned_fails_on_signature():
    issuer = _identity(10)
    deleg = Delegation(
        issuer=issuer.did, subject=_identity(11).did, grant=Grant(capabilities=["x"])
    ).seal()
    result = deleg.verify()
    assert result.valid is False
    assert result.signature_ok is False
    # An unsigned delegation is not an authority-binding failure.
    assert result.authority_ok is True
    assert result.reason == "signature does not verify against a key bound to the issuer DID"


def test_delegation_sig_binding_refused_without_signer_key():
    issuer = _identity(10)
    deleg = issuer.delegate(_identity(11), capabilities=["x"])
    deleg.signer_key = ""  # signature present but no signer key -> binding refused
    result = deleg.verify()
    assert result.signature_ok is False


def test_delegation_sig_binding_refused_on_corrupt_signature():
    issuer = _identity(10)
    deleg = issuer.delegate(_identity(11), capabilities=["x"])
    # Keep facts/hash and signer_key intact, but corrupt the signature bytes so
    # ed.verify fails at the binding step (not the empty-string short-circuit).
    deleg.signature = "ab" * 64
    result = deleg.verify()
    assert result.hash_ok is True  # facts untouched
    assert result.signature_ok is False
    assert result.reason == "signature does not verify against a key bound to the issuer DID"


def test_delegation_audit_details_and_wire_round_trip():
    issuer = _identity(10)
    cutoff = utcnow() + timedelta(days=1)
    deleg = issuer.delegate(
        _identity(11), capabilities=["pay", "read"], budget_usd=25.0, not_after=cutoff
    )
    details = deleg.audit_details()
    assert details["issuer"] == issuer.did
    assert details["capabilities"] == ["pay", "read"]
    assert details["budget_usd"] == 25.0
    assert details["not_after"] == cutoff.isoformat()
    restored = Delegation.from_wire(deleg.to_wire())
    assert restored.verify().valid is True
    assert restored.content_hash == deleg.content_hash


def test_delegation_hash_tamper_detected():
    issuer = _identity(10)
    deleg = issuer.delegate(_identity(11), capabilities=["x"])
    deleg.subject = _identity(12).did  # changes facts, content_hash now stale
    result = deleg.verify()
    assert result.hash_ok is False
    assert result.reason == "content hash does not match the delegation facts"


# ---------------------------------------------------------------------------
# Delegation signed with a rotated key — _verify_authority paths (1172-1197)
# ---------------------------------------------------------------------------


def test_only_subject_can_sub_delegate():
    issuer = _identity(40)
    agent = _identity(41)
    other = _identity(42)
    root = issuer.delegate(agent, capabilities=["read"], max_delegations=2)
    with pytest.raises(IdentityError, match="only the delegate"):
        root.delegate(other, _identity(43))  # other is not the subject


def test_sub_delegate_inherits_then_narrows():
    issuer = _identity(40)
    agent = _identity(41)
    sub = _identity(42)
    root = issuer.delegate(
        agent,
        capabilities=["read", "write"],
        budget_usd=100.0,
        audience="org:acme",
        max_delegations=3,
    )
    # No overrides -> the child inherits caps/budget/audience and consumes one hop.
    child = root.delegate(agent, sub)
    assert child.grant.capability_set == {"read", "write"}
    assert child.grant.budget_usd == 100.0
    assert child.grant.audience == "org:acme"
    assert child.grant.max_delegations == 2  # one hop consumed
    assert child.grant.attenuates(root.grant) is True


def test_sub_delegate_expires_in_sets_child_expiry():
    issuer = _identity(40)
    agent = _identity(41)
    sub = _identity(42)
    issued = utcnow()
    root = issuer.delegate(agent, capabilities=["read"], max_delegations=2, issued_at=issued)
    child = root.delegate(agent, sub, expires_in=timedelta(hours=1), issued_at=issued)
    assert child.grant.not_after == issued + timedelta(hours=1)


def test_sub_delegate_explicit_depth_override():
    issuer = _identity(40)
    agent = _identity(41)
    sub = _identity(42)
    root = issuer.delegate(agent, capabilities=["read"], max_delegations=5)
    child = root.delegate(agent, sub, max_delegations=1)  # explicit override wins
    assert child.grant.max_delegations == 1


def test_sub_delegate_unbounded_parent_keeps_child_unbounded_depth():
    issuer = _identity(40)
    agent = _identity(41)
    sub = _identity(42)
    root = issuer.delegate(agent, capabilities=["read"])  # max_delegations None
    child = root.delegate(agent, sub)
    assert child.grant.max_delegations is None


def test_delegate_expires_in_sets_root_expiry():
    issuer = _identity(40)
    agent = _identity(41)
    issued = utcnow()
    root = issuer.delegate(agent, capabilities=["read"], expires_in=timedelta(hours=2),
                           issued_at=issued)
    assert root.grant.not_after == issued + timedelta(hours=2)


def test_delegation_signed_with_rotated_key_verifies():
    issuer = _identity(10)
    issuer.rotate(seed=_seed(20))  # now signing with a non-genesis key
    deleg = issuer.delegate(_identity(11), capabilities=["x"], budget_usd=5.0)
    assert deleg.authority is not None  # carries the rotation path
    assert deleg.verify().valid is True


def test_delegation_authority_refused_when_path_missing():
    from vincio.security.identity import _verify_authority

    issuer = _identity(10)
    # No authority at all -> refused.
    assert _verify_authority(None, issuer.did, ed.public_key_from_seed(_seed(20)), utcnow()) is False
    # Empty path -> refused.
    assert (
        _verify_authority(
            KeyAuthorization(path=[]), issuer.did, ed.public_key_from_seed(_seed(20)), utcnow()
        )
        is False
    )


def test_delegation_authority_refused_when_genesis_not_first():
    from vincio.security.identity import _verify_authority

    issuer = _identity(10)
    issuer.rotate(seed=_seed(20))
    auth = issuer.keyring.authorization()
    # Mangle the first record so it is no longer the genesis (has a prev_kid).
    auth.path[0].prev_kid = "x"
    assert (
        _verify_authority(auth, issuer.did, ed.public_key_from_seed(_seed(20)), utcnow()) is False
    )


def test_delegation_authority_refused_when_genesis_mismatches_did():
    from vincio.security.identity import _verify_authority

    issuer = _identity(10)
    issuer.rotate(seed=_seed(20))
    auth = issuer.keyring.authorization()
    other = _identity(33)
    # Genesis key bytes don't match the (other) DID.
    assert (
        _verify_authority(auth, other.did, ed.public_key_from_seed(_seed(20)), utcnow()) is False
    )


def test_delegation_authority_refused_on_forged_rotation_sig():
    from vincio.security.identity import _verify_authority

    issuer = _identity(10)
    issuer.rotate(seed=_seed(20))
    auth = issuer.keyring.authorization()
    auth.path[1].rotation_sig = "00" * 64  # predecessor never authorized this key
    signer = issuer.keyring.active_public()
    assert _verify_authority(auth, issuer.did, signer, utcnow()) is False


def test_delegation_authority_refused_on_broken_kid_chain():
    from vincio.security.identity import _verify_authority

    issuer = _identity(10)
    issuer.rotate(seed=_seed(20))
    auth = issuer.keyring.authorization()
    auth.path[1].prev_kid = "wrong"  # child no longer points at the genesis kid
    signer = issuer.keyring.active_public()
    assert _verify_authority(auth, issuer.did, signer, utcnow()) is False


def test_delegation_authority_refused_when_predecessor_revoked_before_rotation():
    from vincio.security.identity import _verify_authority

    # Hand-build a genuine 2-link rotation path, then back-date the genesis
    # revocation BEFORE the child's creation: the child was minted after the
    # predecessor was already revoked, so the path must be refused (line 1193).
    genesis_seed, child_seed = _seed(80), _seed(81)
    g_pub = ed.public_key_from_seed(genesis_seed)
    c_pub = ed.public_key_from_seed(child_seed)
    did = did_from_public_key(g_pub)
    t0 = utcnow()
    genesis = KeyRecord(
        kid=key_fingerprint(g_pub),
        public_key=g_pub.hex(),
        created_at=t0,
        status="revoked",
        revoked_at=t0 + timedelta(minutes=1),
    )
    child = KeyRecord(
        kid=key_fingerprint(c_pub),
        public_key=c_pub.hex(),
        created_at=t0 + timedelta(minutes=5),  # AFTER genesis was revoked
        prev_kid=genesis.kid,
    )
    child.rotation_sig = ed.sign(
        genesis_seed, child.rotation_message(did).encode("utf-8")
    ).hex()
    auth = KeyAuthorization(path=[genesis, child])
    # The rotation signature itself is valid, but the date guard refuses it.
    assert _verify_authority(auth, did, c_pub, t0 + timedelta(minutes=6)) is False


def test_delegation_authority_refused_on_signer_key_mismatch():
    from vincio.security.identity import _verify_authority

    issuer = _identity(10)
    issuer.rotate(seed=_seed(20))
    auth = issuer.keyring.authorization()
    wrong_key = ed.public_key_from_seed(_seed(77))
    # Path is intact, but the claimed signer key is not the path's last key.
    assert _verify_authority(auth, issuer.did, wrong_key, utcnow()) is False


# ---------------------------------------------------------------------------
# DelegationChain — empty, broken linkage, attenuation, permits (1244-1347)
# ---------------------------------------------------------------------------


def _two_link_chain() -> tuple[AgentIdentity, AgentIdentity, AgentIdentity, DelegationChain]:
    principal = _identity(40)
    agent = _identity(41)
    sub = _identity(42)
    root = principal.delegate(
        agent, capabilities=["read", "write"], budget_usd=100.0, max_delegations=2
    )
    leaf = root.delegate(agent, sub, capabilities=["read"], budget_usd=50.0)
    chain = DelegationChain(links=[root, leaf])
    return principal, agent, sub, chain


def test_empty_chain_is_invalid():
    result = DelegationChain(links=[]).verify()
    assert result.valid is False
    assert result.reason == "empty delegation chain"
    assert DelegationChain(links=[]).permits("x") is False


def test_chain_extend_appends_link():
    principal, agent, sub, chain = _two_link_chain()
    assert len(chain.links) == 2
    extended = DelegationChain(links=[chain.links[0]]).extend(chain.links[1])
    assert len(extended.links) == 2
    assert extended.links[1] is chain.links[1]


def test_valid_chain_permits_and_verifies():
    principal, agent, sub, chain = _two_link_chain()
    result = chain.verify()
    assert result.valid is True
    assert result.principal == principal.did
    assert result.subject == sub.did
    assert chain.permits("read", budget_usd=50.0) is True
    assert chain.permits("write") is False  # leaf attenuated it away
    assert chain.permits("read", budget_usd=51.0) is False  # over leaf budget


def test_chain_root_issuer_mismatch_breaks_linkage():
    principal, agent, sub, chain = _two_link_chain()
    result = chain.verify(root_issuer=_identity(99).did)
    assert result.linkage_ok is False
    assert result.reason == "the chain linkage is broken (issuer/subject or parent hash mismatch)"


def test_chain_root_with_parent_hash_breaks_linkage():
    principal = _identity(40)
    agent = _identity(41)
    root = principal.delegate(agent, capabilities=["read"])
    root.parent_hash = "deadbeef"  # a root must sub-delegate from nothing
    root.sign(principal)  # re-seal/sign so the hash matches facts again
    result = DelegationChain(links=[root]).verify()
    assert result.linkage_ok is False


def test_chain_over_reaching_sub_delegation_refused():
    principal = _identity(40)
    agent = _identity(41)
    sub = _identity(42)
    root = principal.delegate(agent, capabilities=["read"], budget_usd=10.0, max_delegations=2)
    # Forge a child that AMPLIFIES (adds a capability + raises the budget).
    forged = Delegation(
        issuer=agent.did,
        subject=sub.did,
        grant=Grant(capabilities=["read", "write"], budget_usd=999.0, max_delegations=1),
        parent_id=root.id,
        parent_hash=root.content_hash,
    ).sign(agent)
    result = DelegationChain(links=[root, forged]).verify()
    assert result.attenuation_ok is False
    assert result.reason == "a sub-delegation amplifies its parent's authority"


def test_chain_expired_link_reported():
    principal = _identity(40)
    agent = _identity(41)
    past = utcnow() - timedelta(days=2)
    root = principal.delegate(agent, capabilities=["read"], not_after=past, issued_at=past)
    result = DelegationChain(links=[root]).verify()
    assert result.not_expired is False
    assert result.reason == "a delegation in the chain has expired"


def test_chain_link_issuer_must_match_parent_subject():
    principal = _identity(40)
    agent = _identity(41)
    sub = _identity(42)
    root = principal.delegate(agent, capabilities=["read"], max_delegations=2)
    # A second link issued by someone who is NOT the root's subject.
    rogue = _identity(44)
    leaf = Delegation(
        issuer=rogue.did,
        subject=sub.did,
        grant=Grant(capabilities=["read"]),
        parent_id=root.id,
        parent_hash=root.content_hash,
    ).sign(rogue)
    result = DelegationChain(links=[root, leaf]).verify()
    assert result.linkage_ok is False


def test_chain_broken_parent_hash_linkage():
    principal, agent, sub, chain = _two_link_chain()
    chain.links[1].parent_hash = "00" * 32  # no longer matches parent content hash
    result = chain.verify()
    assert result.linkage_ok is False


def test_require_permits_raises_with_reason():
    principal, agent, sub, chain = _two_link_chain()
    with pytest.raises(IdentityError, match="does not authorize the action"):
        chain.require_permits("write")  # attenuated away at the leaf


def test_require_permits_returns_self_when_allowed():
    principal, agent, sub, chain = _two_link_chain()
    assert chain.require_permits("read", budget_usd=10.0) is chain


def test_chain_audit_details_and_wire_round_trip():
    principal, agent, sub, chain = _two_link_chain()
    details = chain.audit_details()
    assert details["links"] == 2
    assert details["principal"] == principal.did
    assert details["capabilities"] == ["read"]
    restored = DelegationChain.from_wire(chain.to_wire())
    assert restored.verify().valid is True
    assert restored.subject == sub.did


# ---------------------------------------------------------------------------
# AgentCredential verify / require_valid (lines 902, 1446, 1451, 1454, 1487)
# ---------------------------------------------------------------------------


def test_issue_credential_with_expires_in_sets_not_after():
    org = _identity(50)
    agent = _identity(51)
    issued = utcnow()
    cred = org.issue_credential(
        agent, {"role": "worker"}, expires_in=timedelta(days=1), issued_at=issued
    )
    assert cred.not_after == issued + timedelta(days=1)
    assert cred.verify().valid is True


def test_issue_credential_explicit_not_after_wins():
    org = _identity(50)
    cutoff = utcnow() + timedelta(days=30)
    cred = org.issue_credential(
        _identity(51), {"role": "x"}, not_after=cutoff, expires_in=timedelta(days=1)
    )
    # Explicit not_after takes precedence over expires_in.
    assert cred.not_after == cutoff


def test_credential_single_admitted_capability():
    org = _identity(50)
    cred = org.issue_credential(_identity(51), {"admitted_capability": "pay"})
    assert cred.admitted_capabilities == ["pay"]
    assert cred.admits("pay") is True


def test_credential_unsigned_fails_signature():
    org = _identity(50)
    cred = AgentCredential(
        issuer=org.did, subject=_identity(51).did, claims={"a": "b"}
    ).seal()
    result = cred.verify()
    assert result.signature_ok is False
    assert result.reason == "signature does not verify against a key bound to the issuer DID"


def test_credential_signed_with_rotated_key_verifies():
    org = _identity(50)
    org.rotate(seed=_seed(60))
    cred = org.issue_credential(_identity(51), {"admitted_capability": "pay"})
    assert cred.authority is not None
    assert cred.verify().valid is True
    assert cred.admits("pay") is True


def test_credential_tamper_breaks_hash():
    org = _identity(50)
    cred = org.issue_credential(_identity(51), {"role": "worker"})
    cred.claims = {"role": "admin"}  # facts changed, hash stale
    result = cred.verify()
    assert result.hash_ok is False
    assert result.reason == "content hash does not match the credential facts"


def test_credential_expired_reported():
    org = _identity(50)
    past = utcnow() - timedelta(days=2)
    cred = org.issue_credential(
        _identity(51), {"role": "x"}, not_after=past, issued_at=past - timedelta(days=1)
    )
    result = cred.verify()
    assert result.not_expired is False
    assert result.reason == "credential has expired"


def test_credential_require_valid_raises_on_tamper():
    org = _identity(50)
    cred = org.issue_credential(_identity(51), {"role": "worker"})
    cred.signature = "00" * 64
    with pytest.raises(IdentityError, match="failed verification"):
        cred.require_valid()


def test_credential_require_valid_returns_self_when_ok():
    org = _identity(50)
    cred = org.issue_credential(_identity(51), {"role": "worker"})
    assert cred.require_valid() is cred


def test_credential_admits_false_on_invalid_credential():
    org = _identity(50)
    cred = org.issue_credential(_identity(51), {"admitted_capability": "pay"})
    cred.signature = "00" * 64  # tampered -> verify fails inside admits
    assert cred.admits("pay") is False


def test_credential_admits_multiple_capabilities():
    org = _identity(50)
    cred = org.issue_credential(
        _identity(51), {"admitted_capabilities": "read, write , pay"}
    )
    assert cred.admitted_capabilities == ["read", "write", "pay"]
    assert cred.admits("write") is True
    assert cred.admits("delete") is False


def test_credential_must_be_signed_by_issuer():
    org = _identity(50)
    impostor = _identity(52)
    cred = AgentCredential(issuer=org.did, subject=_identity(51).did, claims={"a": "b"})
    with pytest.raises(IdentityError, match="signed by its issuer"):
        cred.sign(impostor)


def test_credential_audit_details_and_wire():
    org = _identity(50)
    cred = org.issue_credential(_identity(51), {"role": "worker"})
    details = cred.audit_details()
    assert details["issuer"] == org.did
    assert details["claims"] == {"role": "worker"}
    restored = AgentCredential.from_wire(cred.to_wire())
    assert restored.verify().valid is True


# ---------------------------------------------------------------------------
# key_fingerprint stability
# ---------------------------------------------------------------------------


def test_key_fingerprint_is_stable_and_prefixed():
    pub = ed.public_key_from_seed(_seed(1))
    fp = key_fingerprint(pub)
    assert fp.startswith("k")
    assert len(fp) == 17  # "k" + 16 hex
    assert key_fingerprint(pub) == fp  # deterministic
    assert key_fingerprint(ed.public_key_from_seed(_seed(2))) != fp
