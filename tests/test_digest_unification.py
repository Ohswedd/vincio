"""Canonical ``digest()`` accessor unification (7.5).

Three content-hash accessors align on ``digest()`` with a deprecation runway
(``DocumentArtifact.sha256``, ``Recording.compute_digest``,
``PromptNode.content_hash``); the deliberate KEEPs stay stable and undecorated
(``BundleRecord.compute_digest`` — the serialized ``digest`` field owns the
name; ``digest_payload()`` — returns the pre-image, not the hash;
``CandidateArena.fingerprint`` — keys an external signature, not self). Every
aligned accessor is byte-identical to its old name, and the library itself
never emits a deprecation warning on its internal paths.
"""

from __future__ import annotations

import hashlib
import warnings

import pytest

from vincio.generation import DocumentBuilder
from vincio.generation.model import DocumentModel
from vincio.generation.render import DocumentArtifact, render
from vincio.observability.record_replay import RecordedEdge, Recording
from vincio.prompts import PromptCompiler, PromptSpec
from vincio.prompts.ast import PromptAST, PromptNode
from vincio.security.audit import AuditLog
from vincio.stability import StabilityLevel, VincioDeprecationWarning, stability_of


def _artifact() -> DocumentArtifact:
    return render(DocumentModel(title="x"), "markdown")


def _recording() -> Recording:
    return Recording(
        run_id="r",
        input="what is 6x7",
        output_text="forty two",
        edges=[RecordedEdge.of("model_call", 0, "m", {"text": "forty two"})],
    )


# ---------------------------------------------------------------------------
# Byte identity: digest() returns exactly what the old name returned
# ---------------------------------------------------------------------------


class TestByteIdentity:
    def test_document_artifact_digest(self):
        art = _artifact()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VincioDeprecationWarning)
            old = art.sha256()
        assert art.digest() == old == hashlib.sha256(art.content).hexdigest()

    def test_recording_digest_and_fidelity_chain(self):
        rec = _recording()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VincioDeprecationWarning)
            old = rec.compute_digest()
        assert rec.digest() == old
        # The fidelity chain is name-agnostic: sealing with digest() verifies.
        rec.fidelity_digest = rec.digest()
        assert rec.verify()

    def test_prompt_node_digest(self):
        node = PromptNode(kind="rule", text="t")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VincioDeprecationWarning)
            old = node.content_hash
        assert node.digest() == old

    def test_prompt_ast_spec_hash_golden(self):
        # spec_hash hashes node digests (values only), so the accessor rename
        # cannot move it — pinned to the pre-rename golden.
        ast = PromptAST(nodes=[PromptNode(kind="rule", text="a"), PromptNode(kind="rule", text="b")])
        assert ast.spec_hash == "6e0c428dcf0268d0"


# ---------------------------------------------------------------------------
# Deprecation runway: old names warn, escalate, and introspect
# ---------------------------------------------------------------------------


class TestDeprecationRunway:
    def test_sha256_warns(self):
        with pytest.warns(VincioDeprecationWarning, match=r"Use digest\(\)"):
            _artifact().sha256()

    def test_compute_digest_warns(self):
        with pytest.warns(VincioDeprecationWarning, match=r"Use digest\(\)"):
            _recording().compute_digest()

    def test_content_hash_property_warns(self):
        node = PromptNode(kind="rule", text="t")
        with pytest.warns(VincioDeprecationWarning, match=r"Use digest\(\)"):
            _ = node.content_hash

    def test_warning_escalates_to_error(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", VincioDeprecationWarning)
            with pytest.raises(VincioDeprecationWarning):
                _artifact().sha256()

    def test_stability_records(self):
        for accessor in (DocumentArtifact.sha256, Recording.compute_digest):
            record = stability_of(accessor)
            assert record["level"] is StabilityLevel.DEPRECATED
            assert record["since"] == "7.5" and record["removed_in"] == "8.0"
            assert record["alternative"] == "digest()"
        # The property's record lives on its getter.
        record = stability_of(PromptNode.content_hash.fget)
        assert record["level"] is StabilityLevel.DEPRECATED
        assert record["alternative"] == "digest()"


# ---------------------------------------------------------------------------
# The library itself never warns on its internal paths
# ---------------------------------------------------------------------------


class TestLibraryIsWarningClean:
    def test_builder_audit_path(self):
        audit = AuditLog(directory=None)
        with warnings.catch_warnings():
            warnings.simplefilter("error", VincioDeprecationWarning)
            artifact = DocumentBuilder(audit_log=audit).build("# T\n\nBody.", format="markdown")
        assert artifact.digest()
        entry = next(e for e in audit.entries if e.action == "document_generate")
        # frozen audit-detail key, canonical value
        assert entry.details["content_sha256"] == artifact.digest()

    def test_recording_finalize_and_report(self):
        rec = _recording()
        with warnings.catch_warnings():
            warnings.simplefilter("error", VincioDeprecationWarning)
            rec.fidelity_digest = rec.digest()
            report = rec.fidelity_report()
        assert report["ok"] is True and report["digest_ok"] is True

    def test_prompt_compile_and_spec_hash(self):
        spec = PromptSpec(
            name="p",
            objective="Answer briefly",
            rules=["Cite evidence", "Cite evidence"],  # exercises _dedupe
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", VincioDeprecationWarning)
            compiled = PromptCompiler().compile(spec, user_task="t")
        assert compiled.prompt_spec_hash


# ---------------------------------------------------------------------------
# The deliberate KEEPs stay stable (not deprecated, names unchanged)
# ---------------------------------------------------------------------------


class TestKeepVerdicts:
    def test_bundle_record_keeps_compute_digest(self):
        # The serialized ``digest`` field owns the name and signing_message
        # binds it — the accessor keeps its distinct verb.
        from vincio.registry.community import BundleRecord

        assert "digest" in BundleRecord.model_fields
        assert stability_of(BundleRecord.compute_digest)["level"] is StabilityLevel.STABLE

    def test_digest_payload_pair_keeps_preimage_semantics(self):
        # digest_payload() returns the bytes the digest covers, not the hash.
        from vincio.governance.lineage import ErasureProof
        from vincio.governance.verification import VerificationReport

        report = VerificationReport(held=True)
        assert report.digest() == hashlib.sha256(report.digest_payload().encode()).hexdigest()
        proof = ErasureProof(source="s")
        assert proof.digest() == hashlib.sha256(proof.digest_payload().encode()).hexdigest()
        for accessor in (VerificationReport.digest_payload, ErasureProof.digest_payload):
            assert stability_of(accessor)["level"] is StabilityLevel.STABLE

    def test_candidate_arena_keeps_fingerprint(self):
        # fingerprint() keys a caller-supplied signature — it is not a content
        # hash of the arena itself.
        from vincio.context.arena import CandidateArena

        assert stability_of(CandidateArena.fingerprint)["level"] is StabilityLevel.STABLE
