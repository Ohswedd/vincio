"""Community pack & skill registry — governed (allow-list), audited, and signed."""

from __future__ import annotations

import pytest

from vincio.core.errors import AccessDeniedError, VincioError
from vincio.packs import Pack, load_pack
from vincio.registry import BundleRecord, CommunityRegistry
from vincio.security.access import AllowListGate
from vincio.security.audit import AuditLog, HMACSigner

SKILL_MD = (
    "---\n"
    "name: pdf-invoice\n"
    "description: Extract invoice fields from a PDF.\n"
    "keywords: [pdf, invoice]\n"
    "---\n"
    "# Steps\n1. OCR the PDF.\n2. Extract totals.\n"
)


def _registry(*, signer=None, allow=None, audit=None):
    return CommunityRegistry(
        allow_list=AllowListGate(allow=allow if allow is not None else ["*"]),
        audit=audit,
        signer=signer,
    )


def test_publish_and_load_pack_under_gate():
    audit = AuditLog(directory=None)
    reg = _registry(allow=["support-pro"], audit=audit)
    pack = load_pack("support").model_copy(update={"name": "support-pro"})
    record = reg.publish_pack(pack, version="1.2.0", publisher="acme")
    assert record.digest and record.kind == "pack"

    loaded = reg.load_pack("support-pro")
    assert isinstance(loaded, Pack)
    assert loaded.name == "support-pro"
    decisions = audit.query(action="bundle_resolve")
    assert any(d.decision == "allow" and d.resource == "support-pro" for d in decisions)


def test_publish_and_load_skill():
    reg = _registry(allow=["pdf-invoice"])
    reg.publish_skill(SKILL_MD, name="pdf-invoice", version="0.1.0", description="PDF invoice")
    skill = reg.load_skill("pdf-invoice")
    assert skill.name == "pdf-invoice"
    assert "OCR" in skill.instructions
    assert "pdf" in skill.keywords


def test_resolution_denied_by_allow_list_is_audited():
    audit = AuditLog(directory=None)
    reg = _registry(allow=["support-*"], audit=audit)
    reg.register(BundleRecord(name="evil-pack", kind="pack", payload={"name": "evil", "description": "x"}))
    with pytest.raises(AccessDeniedError):
        reg.load_pack("evil-pack")
    decisions = audit.query(action="bundle_resolve")
    assert any(d.decision == "deny" and d.resource == "evil-pack" for d in decisions)


def test_signature_required_when_signer_configured():
    signer = HMACSigner("k")
    reg = CommunityRegistry(allow_list=AllowListGate(allow=["*"]), signer=signer)
    # Register an unsigned record -> resolution fails integrity (signature required).
    reg.register(BundleRecord(name="unsigned", kind="pack", payload={"name": "u", "description": "d"}))
    res = reg.try_resolve("unsigned")
    assert not res.allowed
    assert res.decision.rule == "integrity"


def test_tamper_after_signing_is_detected():
    signer = HMACSigner("k")
    publisher = CommunityRegistry(allow_list=AllowListGate(allow=["*"]), signer=signer)
    pack = load_pack("legal").model_copy(update={"name": "legal-pro"})
    record = publisher.publish_pack(pack, version="1.0.0")

    # An attacker edits the payload but keeps the old signature/digest.
    tampered = record.model_copy(update={"payload": {**record.payload, "role": "HIJACKED"}})
    verifier = CommunityRegistry(
        allow_list=AllowListGate(allow=["*"]), signer=signer, index=[tampered]
    )
    res = verifier.try_resolve("legal-pro")
    assert not res.allowed
    assert "digest" in res.decision.reason or "signature" in res.decision.reason


def test_valid_signature_verifies_and_loads():
    signer = HMACSigner("k")
    reg = CommunityRegistry(allow_list=AllowListGate(allow=["*"]), signer=signer)
    pack = load_pack("finance").model_copy(update={"name": "finance-pro"})
    reg.publish_pack(pack, version="2.0.0", publisher="acme")
    res = reg.try_resolve("finance-pro")
    assert res.allowed and res.verified
    loaded = reg.load_pack("finance-pro")
    assert loaded.name == "finance-pro"


def test_ed25519_third_party_verification():
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric import ed25519

    from vincio.security.audit import Ed25519Signer

    private = ed25519.Ed25519PrivateKey.generate()
    public = private.public_key()
    pub_signer = Ed25519Signer(public_key=public)
    priv_signer = Ed25519Signer(private_key=private)

    publisher = CommunityRegistry(allow_list=AllowListGate(allow=["*"]), signer=priv_signer)
    pack = load_pack("support").model_copy(update={"name": "support-signed"})
    record = publisher.publish_pack(pack, version="1.0.0")

    # A consumer with only the public key can verify and load.
    consumer = CommunityRegistry(allow_list=AllowListGate(allow=["*"]), signer=pub_signer, index=[record])
    assert consumer.load_pack("support-signed").name == "support-signed"


def test_find_by_kind_and_tag():
    reg = _registry()
    reg.register(BundleRecord(name="p1", kind="pack", tags=["support"], payload={"name": "p1", "description": ""}))
    reg.register(BundleRecord(name="s1", kind="skill", tags=["pdf"], payload_text="---\nname: s1\ndescription: d\n---\nbody"))
    assert [r.name for r in reg.find(kind="pack")] == ["p1"]
    assert [r.name for r in reg.find(tag="pdf")] == ["s1"]


def test_signed_index_root_tamper_evident():
    signer = HMACSigner("k")
    reg = CommunityRegistry(allow_list=AllowListGate(allow=["*"]), signer=signer)
    reg.publish_pack(load_pack("support").model_copy(update={"name": "a"}), version="1.0.0")
    sig = reg.sign_index()
    assert reg.verify_index(sig)
    # Adding a bundle changes the root, invalidating the old index signature.
    reg.publish_pack(load_pack("legal").model_copy(update={"name": "b"}), version="1.0.0")
    assert not reg.verify_index(sig)


def test_load_pack_rejects_skill_bundle():
    reg = _registry()
    reg.publish_skill(SKILL_MD, name="pdf-invoice")
    with pytest.raises(VincioError):
        reg.load_pack("pdf-invoice")
