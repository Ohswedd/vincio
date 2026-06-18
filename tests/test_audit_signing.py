"""Signed audit chain + Merkle checkpoints: tamper-evidence against a
privileged attacker who can recompute the public hash chain."""

from __future__ import annotations

import json

from vincio.security.audit import (
    AuditLog,
    HMACSigner,
    MerkleCheckpoint,
    merkle_proof,
    merkle_root,
    verify_audit_file,
    verify_merkle_proof,
)


def test_unsigned_log_is_backward_compatible(tmp_path):
    log = AuditLog(directory=tmp_path)
    log.record("run", run_id="r1")
    log.record("output", run_id="r1")
    assert log.verify_chain() is True
    result = log.verify_file()
    assert result.intact is True
    assert result.entries == 2
    assert result.signed_entries == 0
    assert result.signatures_ok is None  # no verifier supplied


def test_signed_entries_carry_signature_and_verify(tmp_path):
    signer = HMACSigner("super-secret-key", key_id="k1")
    log = AuditLog(directory=tmp_path, signer=signer)
    log.record("run", run_id="r1")
    log.record("tool_call", run_id="r1", resource="search")
    for entry in log.entries:
        assert entry.signature
        assert entry.key_id == "k1"
    assert log.verify_chain() is True

    result = verify_audit_file(log.path, verifier=signer)
    assert result.intact is True
    assert result.signed_entries == 2
    assert result.signatures_ok is True


def test_privileged_attacker_recomputing_hashes_still_fails(tmp_path):
    signer = HMACSigner("the-real-key")
    log = AuditLog(directory=tmp_path, signer=signer)
    log.record("run", run_id="r1", details={"prompt": "original"})
    log.record("output", run_id="r1", details={"answer": "original"})

    # Attacker rewrites a row's content and recomputes the public hash chain
    # (they know the algorithm) — but cannot forge the HMAC without the key.
    lines = [json.loads(line) for line in log.path.read_text().splitlines() if line.strip()]
    from vincio.security.audit import AuditEntry

    tampered = AuditEntry.model_validate(lines[0])
    tampered.details = {"prompt": "TAMPERED"}
    tampered.entry_hash = tampered.compute_hash()  # recompute hash like an attacker
    lines[0] = json.loads(tampered.model_dump_json())
    # Recompute the chain forward so prev_hash links stay consistent.
    lines[1]["prev_hash"] = lines[0]["entry_hash"]
    second = AuditEntry.model_validate(lines[1])
    lines[1]["entry_hash"] = second.compute_hash()
    log.path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    # Hash chain alone now looks intact (attacker recomputed it)...
    no_verifier = verify_audit_file(log.path)
    assert no_verifier.intact is True
    # ...but signature verification catches the forgery.
    with_verifier = verify_audit_file(log.path, verifier=signer)
    assert with_verifier.intact is False
    assert with_verifier.reason == "signature mismatch"


def test_wrong_key_does_not_verify(tmp_path):
    log = AuditLog(directory=tmp_path, signer=HMACSigner("key-a"))
    log.record("run", run_id="r1")
    bad = verify_audit_file(log.path, verifier=HMACSigner("key-b"))
    assert bad.intact is False
    assert bad.signatures_ok is False


def test_merkle_root_and_inclusion_proof():
    hashes = [f"{i:032x}" for i in range(7)]
    root = merkle_root(hashes)
    assert root
    for i in range(len(hashes)):
        proof = merkle_proof(hashes, i)
        assert verify_merkle_proof(hashes[i], proof, root) is True
    # A wrong leaf does not verify under the root.
    assert verify_merkle_proof("deadbeef", merkle_proof(hashes, 0), root) is False


def test_merkle_root_empty():
    assert merkle_root([]) == ""


def test_checkpoint_signs_root_and_persists(tmp_path):
    signer = HMACSigner("ckpt-key", key_id="ck")
    log = AuditLog(directory=tmp_path, signer=signer)
    log.record("run", run_id="r1")
    log.record("output", run_id="r1")
    checkpoint = log.checkpoint()
    assert isinstance(checkpoint, MerkleCheckpoint)
    assert checkpoint.count == 2
    assert checkpoint.root == log.merkle_root()
    assert checkpoint.signature
    assert signer.verify(checkpoint.root, checkpoint.signature) is True
    # Persisted to the sidecar.
    assert log.merkle_path.is_file()
    persisted = [
        MerkleCheckpoint.model_validate_json(line)
        for line in log.merkle_path.read_text().splitlines()
        if line.strip()
    ]
    assert persisted[-1].root == checkpoint.root


def test_app_wires_signer_from_config(tmp_path):
    from vincio import ContextApp
    from vincio.core.config import SecurityConfig, VincioConfig

    cfg = VincioConfig(
        security=SecurityConfig(
            audit_dir=str(tmp_path / "audit"),
            audit_signing_key="config-key",
            audit_signing_key_id="cfg",
        )
    )
    app = ContextApp(name="signed", config=cfg)
    assert app.audit.signer is not None
    entry = app.audit.record("run", run_id="r1")
    assert entry.signature
    assert entry.key_id == "cfg"
