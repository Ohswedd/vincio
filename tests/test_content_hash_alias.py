"""Canonical ``content_hash`` read surface (7.5) for the four ``content_sha256`` fields.

Two fields renamed with a validation alias, a dual-key wire emit (the legacy
``content_sha256`` key stays in every dump until 8.0), and a warn-and-forward
``content_sha256`` getter/setter for the runway (``VerificationReport``,
``SourceErased``); two kept their stored field because persisted signatures
cover the literal serialized key, gaining a read-only ``content_hash`` alias
instead (``ErasureProof``, ``ProvenanceManifest``). Every byte that feeds a
persisted or signed artifact must stay identical — the frozen byte anchors also
live in tests/test_canonical_json_goldens.py; the goldens here re-pin the forms
this rename could plausibly have disturbed.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from vincio import ContextApp
from vincio.core.events import SourceErased
from vincio.governance import (
    HmacSigner,
    build_erasure_proof,
    mark_synthetic_content,
    verify_erasure_proof,
    verify_manifest,
)
from vincio.governance.lineage import ErasureProof
from vincio.governance.verification import (
    GovernanceVerifier,
    InvariantResult,
    VerificationReport,
)
from vincio.stability import VincioDeprecationWarning

# ---------------------------------------------------------------------------
# VerificationReport — SAFE FULL RENAME (not signed; digest excludes the key)
# ---------------------------------------------------------------------------


class TestVerificationReportRename:
    def test_digest_payload_and_digest_byte_frozen(self):
        # The digested bytes never contained the field name, so the rename
        # cannot move them — pinned to the pre-rename golden.
        report = VerificationReport(
            held=True,
            results=[
                InvariantResult(
                    id="i", statement="s", category="c", held=True, states_checked=1, domain_size=1
                )
            ],
        )
        assert report.digest_payload() == (
            '{"held":true,"results":[{"category":"c","counterexample":null,'
            '"domain_size":1,"held":true,"id":"i"}]}'
        )
        assert report.digest() == (
            "c62df220e31d0f2e867ec094e7619abcc70e8892425329af89f07937ece91ebc"
        )
        assert report.digest() == hashlib.sha256(report.digest_payload().encode()).hexdigest()

    def test_old_serialized_payload_validates_via_alias(self):
        old = VerificationReport.model_validate({"held": True, "content_sha256": "abc"})
        assert old.content_hash == "abc"
        with pytest.warns(VincioDeprecationWarning, match="content_hash"):
            assert old.content_sha256 == "abc"  # deprecated alias, warns

    def test_old_and_new_constructor_kwargs_both_bind(self):
        assert VerificationReport(held=True, content_sha256="old").content_hash == "old"
        assert VerificationReport(held=True, content_hash="new").content_hash == "new"

    def test_dump_dual_emits_legacy_key_until_8_0(self):
        # The rename runway on the WIRE: a consumer keyed on a persisted
        # report's content_sha256 keeps reading until 8.0, and the dual-key
        # dump round-trips through the alias.
        dump = VerificationReport(held=True, content_hash="h").model_dump()
        assert dump["content_hash"] == "h"
        assert dump["content_sha256"] == "h"
        assert VerificationReport.model_validate(dump).content_hash == "h"

    def test_deprecated_alias_getter_and_setter_warn_and_forward(self):
        report = VerificationReport(held=True, content_hash="h")
        with pytest.warns(VincioDeprecationWarning, match="content_hash"):
            assert report.content_sha256 == "h"
        with pytest.warns(VincioDeprecationWarning, match="content_hash"):
            report.content_sha256 = "x"
        assert report.content_hash == "x"

    def test_live_verifier_digest_recomputes_post_rename(self):
        live = GovernanceVerifier().verify(record=False)
        assert live.content_hash
        assert live.verify()
        with pytest.warns(VincioDeprecationWarning):
            assert live.content_sha256 == live.content_hash


# ---------------------------------------------------------------------------
# ErasureProof — PROPERTY ALIAS (persisted signatures cover the wire key)
# ---------------------------------------------------------------------------


class TestErasureProofAlias:
    def _proof(self) -> ErasureProof:
        proof = ErasureProof(
            source="s",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            claim_generator="vincio/test",
            removed={"chunks": 1},
            removed_ids={"chunks": ["c1"]},
        )
        proof.content_sha256 = proof.digest()
        return proof

    def test_digest_and_canonical_read_byte_frozen(self):
        proof = self._proof()
        assert proof.content_sha256 == (
            "88c29fba7f8011400cb5b5aaef76f7ca8242b7c2be134834ef1c5762301efeda"
        )
        assert proof.content_hash == proof.content_sha256  # new canonical read

    def test_signing_payload_keeps_frozen_wire_key(self):
        proof = self._proof()
        assert '"content_sha256":"88c29fba' in proof.signing_payload()
        assert '"content_hash"' not in proof.signing_payload()
        # The exact signature previously-persisted proofs depend on.
        assert HmacSigner("erasure-key", key_id="kid").sign(proof.signing_payload()) == (
            "4b7b252b2614713d778de5e518de0c13d3298a43f15e5c3b2e008ac130a0ca85"
        )

    def test_wire_dump_unchanged(self):
        dump = self._proof().to_dict()
        assert "content_sha256" in dump
        assert "content_hash" not in dump

    def test_build_and_verify_roundtrip_unchanged(self):
        signer = HmacSigner("erasure-key", key_id="kid")
        proof = build_erasure_proof("s", {"chunks": ["c1"]}, signer=signer)
        assert proof.content_hash == proof.content_sha256
        assert verify_erasure_proof(proof, signer=signer)


# ---------------------------------------------------------------------------
# ProvenanceManifest — PROPERTY ALIAS (signed manifests live inside user media)
# ---------------------------------------------------------------------------


class TestProvenanceManifestAlias:
    def test_alias_reads_field_and_bytes_unchanged(self):
        manifest = mark_synthetic_content("Generated answer.", model_id="m", signer=HmacSigner("k"))
        assert manifest.content_hash == manifest.content_sha256
        assert '"content_sha256":' in manifest.signing_payload()
        assert '"content_hash"' not in manifest.signing_payload()
        assert manifest.to_dict()["content_binding"]["hash"] == manifest.content_hash
        assert verify_manifest(manifest, "Generated answer.", signer=HmacSigner("k"))


# ---------------------------------------------------------------------------
# SourceErased — SAFE FULL RENAME + dual-key emit through the runway
# ---------------------------------------------------------------------------


class TestSourceErasedRename:
    def test_old_dict_payload_validates_via_alias(self):
        old = SourceErased.model_validate({"source": "s", "content_sha256": "h"})
        assert old.content_hash == "h"
        with pytest.warns(VincioDeprecationWarning, match="content_hash"):
            assert old.content_sha256 == "h"

    def test_dual_key_payload_validates(self):
        # The 7.5 emit carries both keys; the alias binds content_hash and the
        # leftover old key is an allowed extra.
        dual = SourceErased.model_validate(
            {"source": "s", "content_hash": "h", "content_sha256": "h"}
        )
        assert dual.content_hash == "h"

    def test_model_round_trip_keeps_the_legacy_wire_key(self):
        # A payload normalized through the typed model (validate -> dump) must
        # not drop content_sha256 while the runway is open — event consumers
        # keyed on the old name keep reading until 8.0.
        round_tripped = SourceErased.model_validate(
            {"source": "s", "content_sha256": "h"}
        ).model_dump()
        assert round_tripped["content_hash"] == "h"
        assert round_tripped["content_sha256"] == "h"

    def test_deprecated_alias_getter_and_setter_warn(self):
        evt = SourceErased(source="s", content_hash="h")
        with pytest.warns(VincioDeprecationWarning, match="content_hash"):
            assert evt.content_sha256 == "h"
        with pytest.warns(VincioDeprecationWarning, match="content_hash"):
            evt.content_sha256 = "h2"
        assert evt.content_hash == "h2"

    def test_erase_source_emits_both_keys(self, sample_docs_dir, offline_config, tmp_cwd):
        app = ContextApp(name="erase-dual", config=offline_config)
        app.add_source("docs", path=str(sample_docs_dir), retrieval="bm25")
        seen: dict = {}
        app.events.subscribe("governance.source_erased", lambda e: seen.update(e.payload))
        result = app.erase_source("docs")
        assert result.proof is not None
        assert seen["content_hash"] == result.proof.content_hash
        # deprecated wire key dual-emitted (removal in 8.0)
        assert seen["content_sha256"] == seen["content_hash"]
