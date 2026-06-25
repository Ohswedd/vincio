"""Real-behavior coverage for vincio.security.audit.

Targets the previously-uncovered branches: empty-key rejection, the
checkpoint empty/unsigned branches, in-memory chain-break and signature
failures in verify_chain, the missing-file / unparseable paths of
verify_audit_file, the merkle_proof bounds error, and the whole
apply_retention age-based pruning function.

Everything is deterministic and offline; no mocks, no network, no key.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from vincio.core.utils import to_jsonable, utcnow
from vincio.security.audit import (
    AuditEntry,
    AuditLog,
    ChainVerification,
    HMACSigner,
    MerkleCheckpoint,
    RetentionPolicy,
    apply_retention,
    merkle_proof,
    merkle_root,
    verify_audit_file,
    verify_merkle_proof,
)

# ---------------------------------------------------------------------------
# HMACSigner edge cases
# ---------------------------------------------------------------------------


def test_hmac_signer_rejects_empty_string_key():
    with pytest.raises(ValueError, match="non-empty key"):
        HMACSigner("")


def test_hmac_signer_rejects_empty_bytes_key():
    with pytest.raises(ValueError, match="non-empty key"):
        HMACSigner(b"")


def test_hmac_signer_accepts_bytes_key_and_round_trips():
    signer = HMACSigner(b"\x00\x01raw-bytes", key_id="kb")
    sig = signer.sign("hello")
    assert signer.verify("hello", sig) is True
    assert signer.verify("hello-tampered", sig) is False


def test_hmac_verify_empty_signature_is_false():
    signer = HMACSigner("k")
    # Empty / None signature must not be treated as a valid match.
    assert signer.verify("msg", "") is False
    assert signer.verify("msg", None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# merkle_proof bounds
# ---------------------------------------------------------------------------


def test_merkle_proof_index_out_of_range_raises():
    hashes = [f"{i:032x}" for i in range(3)]
    with pytest.raises(IndexError, match="out of range"):
        merkle_proof(hashes, 3)
    with pytest.raises(IndexError, match="out of range"):
        merkle_proof(hashes, -1)


def test_merkle_proof_single_leaf_is_empty():
    # A one-element tree: the leaf is the root, proof has no siblings.
    assert merkle_proof(["a" * 32], 0) == []


def test_merkle_root_odd_count_duplicates_last_node():
    # Three leaves forces the odd-level duplication branch; root must be
    # stable and reproducible, and every leaf must prove inclusion.
    hashes = [f"{i:032x}" for i in range(3)]
    root = merkle_root(hashes)
    assert root == merkle_root(hashes)  # deterministic
    assert root != hashes[0]
    for i in range(3):
        assert verify_merkle_proof(hashes[i], merkle_proof(hashes, i), root) is True


def test_merkle_proof_records_sibling_sides():
    hashes = [f"{i:032x}" for i in range(4)]
    # Leaf 0's first sibling sits to its right ("R"); leaf 1's to its left.
    assert merkle_proof(hashes, 0)[0][1] == "R"
    assert merkle_proof(hashes, 1)[0][1] == "L"
    root = merkle_root(hashes)
    # Tampered proof must fail under the genuine root.
    bad = merkle_proof(hashes, 0)
    bad[0] = ("0" * 32, bad[0][1])
    assert verify_merkle_proof(hashes[0], bad, root) is False


# ---------------------------------------------------------------------------
# checkpoint branches: empty chain, and unsigned log
# ---------------------------------------------------------------------------


def test_checkpoint_on_empty_log_has_empty_root_and_no_ids():
    log = AuditLog(directory=None)
    ckpt = log.checkpoint()
    assert isinstance(ckpt, MerkleCheckpoint)
    assert ckpt.count == 0
    assert ckpt.root == ""
    assert ckpt.first_entry_id is None
    assert ckpt.last_entry_id is None
    # Empty root must NOT be signed even on a signed log (root falsy branch).
    assert ckpt.signature == ""


def test_checkpoint_empty_root_unsigned_even_with_signer():
    signer = HMACSigner("k")
    log = AuditLog(directory=None, signer=signer)
    ckpt = log.checkpoint()  # zero entries -> empty root -> `root` is falsy
    assert ckpt.root == ""
    assert ckpt.signature == ""
    assert ckpt.key_id == ""


def test_checkpoint_unsigned_log_records_ids_but_no_signature(tmp_path):
    log = AuditLog(directory=tmp_path)  # no signer
    first = log.record("run", run_id="r1")
    last = log.record("output", run_id="r1")
    ckpt = log.checkpoint()
    assert ckpt.count == 2
    assert ckpt.first_entry_id == first.id
    assert ckpt.last_entry_id == last.id
    assert ckpt.root == log.merkle_root()
    assert ckpt.signature == ""  # unsigned branch
    assert ckpt.key_id == ""


# ---------------------------------------------------------------------------
# verify_chain in-memory failure branches
# ---------------------------------------------------------------------------


def test_verify_chain_detects_tampered_entry_hash_in_memory():
    log = AuditLog(directory=None)
    log.record("run", run_id="r1")
    log.record("output", run_id="r1")
    assert log.verify_chain() is True
    # Mutate a field without recomputing -> entry_hash != compute_hash().
    log.entries[0].action = "memory_write"
    assert log.verify_chain() is False


def test_verify_chain_detects_broken_prev_hash_link_in_memory():
    log = AuditLog(directory=None)
    log.record("run", run_id="r1")
    log.record("output", run_id="r1")
    # Break the link between entry 0 and entry 1.
    log.entries[1].prev_hash = "0" * 32
    assert log.verify_chain() is False


def test_verify_chain_fails_when_signature_does_not_verify():
    signer = HMACSigner("right-key")
    log = AuditLog(directory=None, signer=signer)
    log.record("run", run_id="r1")
    assert log.verify_chain() is True
    # Hash chain still links, but a different verifier rejects the signature
    # (line 324: check.verify(...) is False -> return False).
    assert log.verify_chain(verifier=HMACSigner("wrong-key")) is False


def test_checkpoint_signed_root_persists_and_verifies(tmp_path):
    signer = HMACSigner("ck-key", key_id="ck")
    log = AuditLog(directory=tmp_path, signer=signer)
    log.record("run", run_id="r1")
    ckpt = log.checkpoint()
    # Non-empty root on a signed log -> signature + key_id set (lines 302-303).
    assert ckpt.root
    assert ckpt.key_id == "ck"
    assert signer.verify(ckpt.root, ckpt.signature) is True
    # Persisted to the sidecar file.
    persisted = [
        MerkleCheckpoint.model_validate_json(line)
        for line in log.merkle_path.read_text().splitlines()
        if line.strip()
    ]
    assert persisted[-1].root == ckpt.root
    assert persisted[-1].signature == ckpt.signature


def test_verify_file_persisted_signed_chain_validates(tmp_path):
    signer = HMACSigner("file-key")
    log = AuditLog(directory=tmp_path, signer=signer)
    log.record("run", run_id="r1")
    log.record("output", run_id="r1")
    # verify_file with the log's own signer re-reads disk and checks sigs.
    result = log.verify_file()
    assert result.intact is True
    assert result.entries == 2
    assert result.signed_entries == 2
    assert result.signatures_ok is True


def test_verify_chain_signed_entry_skipped_without_verifier():
    # Signed entries but neither verifier nor signer passed -> chain still ok
    # because the signature branch only runs when `check` is not None.
    signer = HMACSigner("k")
    log = AuditLog(directory=None, signer=signer)
    log.record("run", run_id="r1")
    # Drop the signer; verify_chain with no verifier ignores signatures.
    log.signer = None
    assert log.verify_chain() is True


# ---------------------------------------------------------------------------
# verify_file / verify_audit_file edge paths
# ---------------------------------------------------------------------------


def test_verify_file_in_memory_log_reports_intact_zero():
    log = AuditLog(directory=None)
    log.record("run", run_id="r1")  # in memory only, no path
    result = log.verify_file()
    assert result.intact is True
    assert result.entries == 0


def test_verify_audit_file_missing_file_is_intact_zero(tmp_path):
    result = verify_audit_file(tmp_path / "nope.jsonl")
    assert isinstance(result, ChainVerification)
    assert result.intact is True
    assert result.entries == 0


def test_verify_audit_file_unparseable_line_localized(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(directory=tmp_path)
    log.record("run", run_id="r1")
    log.record("output", run_id="r1")
    # Corrupt line 2 into non-JSON so model_validate_json raises.
    lines = path.read_text().splitlines()
    lines[1] = "{not valid json"
    path.write_text("\n".join(lines) + "\n")
    result = verify_audit_file(path)
    assert result.intact is False
    assert result.broken_at == 2
    assert result.reason is not None
    assert result.reason.startswith("unparseable:")


def test_verify_audit_file_skips_blank_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(directory=tmp_path)
    log.record("run", run_id="r1")
    log.record("output", run_id="r1")
    # Inject blank lines that must be ignored without breaking the chain.
    raw = path.read_text().splitlines()
    path.write_text(raw[0] + "\n\n   \n" + raw[1] + "\n")
    result = verify_audit_file(path)
    assert result.intact is True
    assert result.entries == 2


def test_verify_audit_file_prev_hash_mismatch_localized(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(directory=tmp_path)
    log.record("run", run_id="r1")
    log.record("output", run_id="r1")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[1]["prev_hash"] = "f" * 32  # break the link to entry 0
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result = verify_audit_file(path)
    assert result.intact is False
    assert result.broken_at == 2
    assert result.reason == "prev_hash mismatch"


def test_verify_audit_file_signature_mismatch_localized(tmp_path):
    path = tmp_path / "audit.jsonl"
    signer = HMACSigner("genuine")
    log = AuditLog(directory=tmp_path, signer=signer)
    log.record("run", run_id="r1")
    log.record("output", run_id="r1")
    # Verifying with the wrong key: hashes match but signature does not
    # (lines 415-417: signed entry, verifier rejects -> localized failure).
    result = verify_audit_file(path, verifier=HMACSigner("impostor"))
    assert result.intact is False
    assert result.reason == "signature mismatch"
    assert result.broken_at == 1
    assert result.signatures_ok is False
    assert result.signed_entries == 1


def test_verify_audit_file_signed_count_without_verifier(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(directory=tmp_path, signer=HMACSigner("k"))
    log.record("run", run_id="r1")
    # No verifier: signatures counted but not checked -> signatures_ok None.
    result = verify_audit_file(path)
    assert result.intact is True
    assert result.signed_entries == 1
    assert result.signatures_ok is None


def test_verify_audit_file_entry_hash_mismatch_localized(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(directory=tmp_path)
    log.record("run", run_id="r1")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[0]["details"] = {"prompt": "edited-without-rehash"}
    path.write_text(json.dumps(rows[0]) + "\n")
    result = verify_audit_file(path)
    assert result.intact is False
    assert result.broken_at == 1
    assert result.reason == "entry_hash mismatch"


# ---------------------------------------------------------------------------
# query filters
# ---------------------------------------------------------------------------


def test_query_filters_by_each_field_and_limit():
    log = AuditLog(directory=None)
    log.record("run", user_id="u1", tenant_id="t1", run_id="r1")
    log.record("output", user_id="u1", tenant_id="t1", run_id="r1")
    log.record("run", user_id="u2", tenant_id="t2", run_id="r2")

    assert {e.action for e in log.query(action="run")} == {"run"}
    assert [e.run_id for e in log.query(action="run")] == ["r1", "r2"]
    assert all(e.user_id == "u1" for e in log.query(user_id="u1"))
    assert all(e.tenant_id == "t2" for e in log.query(tenant_id="t2"))
    assert len(log.query(run_id="r1")) == 2
    assert log.query(user_id="nobody") == []


def test_query_limit_returns_most_recent_tail():
    log = AuditLog(directory=None)
    for i in range(5):
        log.record("run", run_id=f"r{i}")
    last_two = log.query(limit=2)
    assert [e.run_id for e in last_two] == ["r3", "r4"]


# ---------------------------------------------------------------------------
# apply_retention — entirely uncovered before
# ---------------------------------------------------------------------------


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(to_jsonable(r)) for r in records) + "\n")


def test_apply_retention_missing_file_removes_zero(tmp_path):
    assert apply_retention(tmp_path / "absent.jsonl", max_age_days=1) == 0


def test_apply_retention_drops_only_old_records(tmp_path):
    path = tmp_path / "data.jsonl"
    now = utcnow()
    old = (now - timedelta(days=10)).isoformat()
    fresh = (now - timedelta(days=1)).isoformat()
    _write_jsonl(
        path,
        [
            {"timestamp": old, "id": "old"},
            {"timestamp": fresh, "id": "fresh"},
        ],
    )
    removed = apply_retention(path, max_age_days=5)
    assert removed == 1
    surviving = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [r["id"] for r in surviving] == ["fresh"]
    # Re-running with nothing old removes zero and leaves the file untouched.
    assert apply_retention(path, max_age_days=5) == 0


def test_apply_retention_drops_all_records_writes_empty(tmp_path):
    path = tmp_path / "data.jsonl"
    old = (utcnow() - timedelta(days=30)).isoformat()
    _write_jsonl(path, [{"timestamp": old, "id": "a"}, {"timestamp": old, "id": "b"}])
    removed = apply_retention(path, max_age_days=1)
    assert removed == 2
    # All removed -> file written with no trailing newline-only content.
    assert path.read_text() == ""


def test_apply_retention_uses_fallback_timestamp_fields(tmp_path):
    path = tmp_path / "data.jsonl"
    old = (utcnow() - timedelta(days=10)).isoformat()
    fresh = (utcnow() - timedelta(days=1)).isoformat()
    # No "timestamp" field; falls back to start_time then created_at.
    _write_jsonl(
        path,
        [
            {"start_time": old, "id": "by_start"},
            {"created_at": fresh, "id": "by_created"},
        ],
    )
    removed = apply_retention(path, max_age_days=5)
    assert removed == 1
    surviving = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [r["id"] for r in surviving] == ["by_created"]


def test_apply_retention_handles_z_suffix_and_naive_timestamps(tmp_path):
    path = tmp_path / "data.jsonl"
    # 'Z' UTC suffix and a naive (tz-less) timestamp must both parse.
    old_z = (utcnow() - timedelta(days=10)).isoformat().replace("+00:00", "Z")
    naive_fresh = (utcnow() - timedelta(days=1)).replace(tzinfo=None).isoformat()
    _write_jsonl(path, [{"timestamp": old_z, "id": "z"}, {"timestamp": naive_fresh, "id": "naive"}])
    removed = apply_retention(path, max_age_days=5)
    assert removed == 1
    surviving = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [r["id"] for r in surviving] == ["naive"]


def test_apply_retention_keeps_unparseable_and_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    old = (utcnow() - timedelta(days=10)).isoformat()
    # Build a file with: a bad-JSON line, a record with no usable timestamp,
    # a blank line, and one genuinely old record.
    path.write_text(
        "{not json\n"
        + json.dumps({"id": "no_ts", "value": 1}) + "\n"
        + "\n"
        + json.dumps({"timestamp": old, "id": "old"}) + "\n"
    )
    removed = apply_retention(path, max_age_days=5)
    assert removed == 1
    kept = path.read_text().splitlines()
    # Bad-JSON and the no-timestamp record are preserved; blank line dropped.
    assert "{not json" in kept
    assert any('"no_ts"' in line for line in kept)
    assert not any('"old"' in line for line in kept)


def test_apply_retention_custom_timestamp_field(tmp_path):
    path = tmp_path / "data.jsonl"
    old = (utcnow() - timedelta(days=10)).isoformat()
    fresh = (utcnow() - timedelta(days=1)).isoformat()
    _write_jsonl(path, [{"when": old, "id": "old"}, {"when": fresh, "id": "fresh"}])
    removed = apply_retention(path, max_age_days=5, timestamp_field="when")
    assert removed == 1
    surviving = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [r["id"] for r in surviving] == ["fresh"]


# ---------------------------------------------------------------------------
# RetentionPolicy / AuditEntry plumbing
# ---------------------------------------------------------------------------


def test_retention_policy_defaults_keep_forever():
    policy = RetentionPolicy()
    assert policy.traces is None
    assert policy.audit is None
    custom = RetentionPolicy(audit=30, traces=7)
    assert custom.audit == 30
    assert custom.traces == 7


def test_audit_entry_compute_hash_changes_with_details():
    e1 = AuditEntry(action="run", details={"prompt": "a"})
    e2 = e1.model_copy(update={"details": {"prompt": "b"}})
    assert e1.compute_hash() != e2.compute_hash()
    # Signature/key_id are deliberately excluded from the hash (1.x compat).
    e3 = e1.model_copy(update={"signature": "deadbeef", "key_id": "k"})
    assert e3.compute_hash() == e1.compute_hash()


def test_record_persists_jsonl_line_per_entry(tmp_path):
    log = AuditLog(directory=tmp_path)
    log.record("run", run_id="r1")
    log.record("output", run_id="r1")
    lines = [line for line in log.path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    reloaded = AuditEntry.model_validate_json(lines[1])
    assert reloaded.prev_hash == log.entries[0].entry_hash
