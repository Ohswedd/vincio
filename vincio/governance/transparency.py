"""EU AI Act transparency artifacts — generated, deadline-agnostic.

The EU AI Act's GenAI transparency duties (machine-readable synthetic-content
marking, AI-interaction disclosure, and a training/grounding-data summary) take
effect 2 Aug 2026. Vincio supplies the *artifacts and hooks*, configurable and
date-agnostic — it does not hard-code a deadline or become a compliance service.

* :func:`mark_synthetic_content` emits a C2PA-style **provenance manifest** that
  binds to the output by SHA-256 — text *or* raw media bytes (1.9), with the
  IPTC ``trainedAlgorithmicMedia`` / ``compositeWithTrainedAlgorithmicMedia``
  digital source type — suitable for attaching as content credentials.
* :func:`embed_provenance` writes the manifest into a generated asset's file
  metadata (PNG text chunk, dependency-free) and runs an optional invisible-
  watermark hook; :func:`write_sidecar_manifest` attaches it as a ``*.c2pa.json``
  sidecar for formats that can't carry embedded credentials.
* :func:`ai_disclosure` returns a plain-language **interaction disclosure**.
* :func:`data_summary` summarizes the **grounding data** a run used (or any
  evidence/sources) for the training/grounding-data-summary duty.

None of these embed cryptographic signatures (that needs a signing authority);
they are the manifest and the hook, which you sign and attach in your pipeline.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
import zlib
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..core.media import media_sha256
from ..core.utils import utcnow
from ..security.secrets import SecretString

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.types import EvidenceItem, RunResult

__all__ = [
    "ProvenanceManifest",
    "ContentSigner",
    "HmacSigner",
    "mark_synthetic_content",
    "verify_manifest",
    "embed_provenance",
    "extract_embedded_manifest",
    "verify_embedded_manifest",
    "write_sidecar_manifest",
    "ai_disclosure",
    "data_summary",
]

# IPTC digital-source-type terms (C2PA standard values).
_TRAINED_ALGORITHMIC_MEDIA = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
# Edited/composited content with an AI-generated component (edit_image, redline).
_COMPOSITE_TRAINED_MEDIA = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
)

_DISCLOSURES = {
    "en": "You are interacting with an AI system. Responses are AI-generated and may be inaccurate; verify important information.",
    "es": "Está interactuando con un sistema de IA. Las respuestas son generadas por IA y pueden ser inexactas; verifique la información importante.",
    "fr": "Vous interagissez avec un système d'IA. Les réponses sont générées par IA et peuvent être inexactes; vérifiez les informations importantes.",
    "de": "Sie interagieren mit einem KI-System. Die Antworten werden von KI generiert und können ungenau sein; überprüfen Sie wichtige Informationen.",
}


class ProvenanceManifest(BaseModel):
    """A C2PA-style content-provenance manifest for AI-generated output."""

    claim_generator: str  # e.g. "vincio/1.6.0"
    is_synthetic: bool = True
    digital_source_type: str = _TRAINED_ALGORITHMIC_MEDIA
    model_id: str | None = None
    provider: str | None = None
    # IANA media type of the bound asset (``text/plain`` for a text answer,
    # ``image/png`` / ``audio/mpeg`` for generated media); part of the binding.
    media_type: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    content_sha256: str | None = None
    assertions: list[dict[str, Any]] = Field(default_factory=list)
    # Optional cryptographic signature over the manifest's binding payload
    # (``{alg, key_id, value}``); attached when a signer is supplied.
    signature: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def signing_payload(self) -> str:
        """Deterministic bytes the signature covers (binds the credential)."""
        return json.dumps(
            {
                "claim_generator": self.claim_generator,
                "is_synthetic": self.is_synthetic,
                "digital_source_type": self.digital_source_type,
                "model_id": self.model_id,
                "provider": self.provider,
                "media_type": self.media_type,
                "created_at": self.created_at.isoformat(),
                "content_sha256": self.content_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render in a C2PA-manifest-shaped dict (attach as content credentials)."""
        manifest: dict[str, Any] = {
            "claim_generator": self.claim_generator,
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "digitalSourceType": self.digital_source_type,
                                "softwareAgent": self.claim_generator,
                            }
                        ]
                    },
                },
                {
                    "label": "vincio.ai_generation",
                    "data": {
                        "is_synthetic": self.is_synthetic,
                        "model_id": self.model_id,
                        "provider": self.provider,
                        "media_type": self.media_type,
                        "created_at": self.created_at.isoformat(),
                    },
                },
                *self.assertions,
            ],
            "content_binding": {"alg": "SHA-256", "hash": self.content_sha256},
        }
        if self.signature is not None:
            manifest["signature"] = self.signature
        return manifest

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


@runtime_checkable
class ContentSigner(Protocol):
    """Signs a manifest's binding payload. ``key_id`` labels the key used."""

    key_id: str

    def sign(self, payload: str) -> str: ...

    def verify(self, payload: str, signature: str) -> bool: ...


class HmacSigner:
    """HMAC-SHA256 signer over a shared secret (symmetric).

    A pragmatic, dependency-free signer for environments without a full PKI:
    the same secret signs and verifies. For third-party-verifiable provenance,
    supply your own asymmetric :class:`ContentSigner` instead.
    """

    def __init__(self, secret: str | SecretString, *, key_id: str = "hmac-default") -> None:
        self._secret = secret if isinstance(secret, SecretString) else SecretString(secret)
        self.key_id = key_id

    def sign(self, payload: str) -> str:
        return hmac.new(
            self._secret.reveal().encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def verify(self, payload: str, signature: str) -> bool:
        return hmac.compare_digest(self.sign(payload), signature)


def mark_synthetic_content(
    content: str | bytes,
    *,
    model_id: str | None = None,
    provider: str | None = None,
    media_type: str | None = None,
    edited: bool = False,
    digital_source_type: str | None = None,
    extra_assertions: list[dict[str, Any]] | None = None,
    signer: ContentSigner | None = None,
) -> ProvenanceManifest:
    """Build a provenance manifest marking ``content`` as AI-generated.

    ``content`` may be a text answer *or* raw media bytes (a generated image or
    audio clip, 1.9): the manifest is bound by SHA-256 over whichever was given,
    so a downstream consumer can confirm the credential matches the bytes it
    received. ``media_type`` records the asset's IANA type (defaults to
    ``text/plain`` for ``str``); ``edited=True`` marks an edit/composite with the
    ``compositeWithTrainedAlgorithmicMedia`` source type, and ``digital_source_type``
    overrides it outright. Pass a ``signer`` to attach a cryptographic signature
    over the binding payload — verify it later with :func:`verify_manifest`.
    """
    import vincio

    resolved_type = media_type or ("text/plain" if isinstance(content, str) else None)
    source_type = digital_source_type or (
        _COMPOSITE_TRAINED_MEDIA if edited else _TRAINED_ALGORITHMIC_MEDIA
    )
    manifest = ProvenanceManifest(
        claim_generator=f"vincio/{vincio.__version__}",
        digital_source_type=source_type,
        model_id=model_id,
        provider=provider,
        media_type=resolved_type,
        content_sha256=media_sha256(content),
        assertions=list(extra_assertions or []),
    )
    if signer is not None:
        manifest.signature = {
            "alg": "HMAC-SHA256" if isinstance(signer, HmacSigner) else "custom",
            "key_id": getattr(signer, "key_id", "default"),
            "value": signer.sign(manifest.signing_payload()),
        }
    return manifest


def verify_manifest(
    manifest: ProvenanceManifest,
    content: str | bytes,
    *,
    signer: ContentSigner | None = None,
) -> bool:
    """Verify a manifest against the content (text or media bytes) it describes.

    Always checks the SHA-256 content binding. If the manifest carries a
    signature, a ``signer`` with the matching key must be supplied to verify it
    (returns ``False`` when a signature is present but no verifier is given, so
    an unverifiable credential is never reported as valid).
    """
    if manifest.content_sha256 != media_sha256(content):
        return False
    if manifest.signature is not None:
        if signer is None:
            return False
        return signer.verify(manifest.signing_payload(), manifest.signature.get("value", ""))
    return True


def _png_text_chunk(keyword: str, text: str) -> bytes:
    """A PNG ``iTXt`` chunk carrying UTF-8 ``text`` under ``keyword``."""
    data = (
        keyword.encode("latin-1")
        + b"\x00"  # null separator
        + b"\x00\x00"  # compression flag + method (uncompressed)
        + b"\x00"  # language tag (empty) + null
        + b"\x00"  # translated keyword (empty) + null
        + text.encode("utf-8")
    )
    chunk_type = b"iTXt"
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


_PNG_MANIFEST_KEYWORD = b"c2pa.manifest"


def _embed_png_manifest(data: bytes, manifest: ProvenanceManifest) -> bytes:
    """Insert the manifest JSON as an iTXt chunk before a PNG's IEND.

    Returns ``data`` unchanged when it is not a PNG (no false claim of
    embedding); callers fall back to a sidecar for other formats. The embedded
    credential binds the *pre-insert* bytes by SHA-256 and is **self-verifying**:
    inserting the chunk is reversible, so :func:`verify_embedded_manifest` removes
    the chunk to reconstruct the original bytes and confirm the digest. The
    signature is dropped from the embedded copy (its payload is bound to the
    in-memory manifest the sidecar carries).
    """
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return data
    iend = data.rfind(b"IEND")
    if iend < 4:
        return data
    insert_at = iend - 4  # before the IEND chunk's length field
    embedded = manifest.model_copy(update={"content_sha256": media_sha256(data), "signature": None})
    chunk = _png_text_chunk(_PNG_MANIFEST_KEYWORD.decode(), embedded.to_json(indent=0))
    return data[:insert_at] + chunk + data[insert_at:]


def _itxt_text(chunk_data: bytes) -> str | None:
    """Decode the UTF-8 text payload of an iTXt chunk (any keyword)."""
    sep = chunk_data.find(b"\x00")
    if sep < 0:
        return None
    rest = chunk_data[sep + 1 :]
    if len(rest) < 2:
        return None
    rest = rest[2:]  # skip compression flag + method
    lang_end = rest.find(b"\x00")
    if lang_end < 0:
        return None
    rest = rest[lang_end + 1 :]
    trans_end = rest.find(b"\x00")
    if trans_end < 0:
        return None
    return rest[trans_end + 1 :].decode("utf-8", "ignore")


def _find_png_manifest_chunk(data: bytes) -> tuple[int, int, str] | None:
    """Locate the ``c2pa.manifest`` iTXt chunk: ``(chunk_start, chunk_end, json)``."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    pos = 8
    n = len(data)
    while pos + 8 <= n:
        length = int.from_bytes(data[pos : pos + 4], "big")
        ctype = data[pos + 4 : pos + 8]
        chunk_end = pos + 12 + length  # length(4) + type(4) + data(length) + crc(4)
        if chunk_end > n:
            break
        if ctype == b"iTXt":
            chunk_data = data[pos + 8 : pos + 8 + length]
            if chunk_data.startswith(_PNG_MANIFEST_KEYWORD + b"\x00"):
                text = _itxt_text(chunk_data)
                if text is not None:
                    return (pos, chunk_end, text)
        pos = chunk_end
    return None


def extract_embedded_manifest(data: bytes) -> ProvenanceManifest | None:
    """Extract a C2PA manifest embedded in PNG bytes by :func:`embed_provenance`,
    or ``None`` when no embedded credential is present."""
    found = _find_png_manifest_chunk(data)
    if found is None:
        return None
    try:
        payload = json.loads(found[2])
    except (ValueError, TypeError):
        return None
    gen: dict[str, Any] = {}
    for assertion in payload.get("assertions", []) or []:
        if isinstance(assertion, dict) and assertion.get("label") == "vincio.ai_generation":
            gen = assertion.get("data", {}) or {}
            break
    binding = payload.get("content_binding", {}) or {}
    return ProvenanceManifest(
        claim_generator=payload.get("claim_generator", ""),
        is_synthetic=bool(gen.get("is_synthetic", True)),
        model_id=gen.get("model_id"),
        provider=gen.get("provider"),
        media_type=gen.get("media_type"),
        content_sha256=binding.get("hash"),
    )


def verify_embedded_manifest(data: bytes) -> bool:
    """Verify a PNG's embedded C2PA credential against the file it travels in.

    Removes the ``c2pa.manifest`` chunk to reconstruct the original bytes and
    confirms the embedded digest matches — so an extracted credential is
    independently verifiable, not merely present. Returns ``False`` when there is
    no embedded manifest or the binding does not hold.
    """
    found = _find_png_manifest_chunk(data)
    if found is None:
        return False
    manifest = extract_embedded_manifest(data)
    if manifest is None or manifest.content_sha256 is None:
        return False
    reconstructed = data[: found[0]] + data[found[1] :]
    return manifest.content_sha256 == media_sha256(reconstructed)


def embed_provenance(
    data: bytes,
    manifest: ProvenanceManifest,
    *,
    watermark_hook: Callable[[bytes], bytes] | None = None,
) -> bytes:
    """Embed provenance into generated media bytes, returning the new bytes.

    Applies the optional ``watermark_hook`` first (an invisible-watermark
    function you supply — Vincio ships the hook point, not a watermarking model),
    then writes the C2PA manifest into the file's metadata where the container
    supports it dependency-free (PNG text chunk today). For formats that cannot
    carry embedded credentials the bytes are returned watermarked-or-unchanged
    and the manifest should travel as a :func:`write_sidecar_manifest` sidecar.
    Always re-bind the manifest (:func:`mark_synthetic_content`) to the returned
    bytes if a watermark altered them. A ``watermark_hook`` must return non-empty
    bytes and must not corrupt the container; a hook that strips a PNG signature
    raises rather than silently shipping an unembedded-but-reported-stamped asset.
    """
    if watermark_hook is not None:
        was_png = data.startswith(b"\x89PNG\r\n\x1a\n")
        result = watermark_hook(data)
        if not result:
            from ..core.errors import MediaGenerationError

            raise MediaGenerationError("watermark_hook returned empty bytes")
        if was_png and not result.startswith(b"\x89PNG\r\n\x1a\n"):
            from ..core.errors import MediaGenerationError

            raise MediaGenerationError(
                "watermark_hook corrupted the PNG signature; manifest could not be embedded"
            )
        data = result
    return _embed_png_manifest(data, manifest)


def write_sidecar_manifest(asset_path: str | Path, manifest: ProvenanceManifest) -> Path:
    """Write ``manifest`` next to ``asset_path`` as a ``<name>.c2pa.json`` sidecar.

    The dependency-free way to attach content credentials to any asset format —
    the manifest is bound to the asset by SHA-256, so the pair is verifiable
    even when the bytes themselves can't carry embedded metadata.
    """
    asset = Path(asset_path)
    sidecar = asset.with_name(asset.name + ".c2pa.json")
    sidecar.write_text(manifest.to_json(), encoding="utf-8")
    return sidecar


def ai_disclosure(*, language: str = "en", system_name: str | None = None) -> str:
    """Return a plain-language AI-interaction disclosure string.

    Falls back to English for unknown locales. ``system_name`` prefixes the
    notice when supplied (e.g. for branding).
    """
    base = _DISCLOSURES.get(language.lower().split("-")[0], _DISCLOSURES["en"])
    if system_name:
        return f"{system_name}: {base}"
    return base


def data_summary(
    source: RunResult | list[EvidenceItem],
    *,
    title: str = "Grounding data summary",
) -> dict[str, Any]:
    """Summarize the grounding/training data behind a run (or evidence list).

    Accepts a :class:`~vincio.core.types.RunResult` (uses its evidence and
    citations) or a bare list of :class:`~vincio.core.types.EvidenceItem`.
    Produces aggregate counts — by source type, trust level, and grounding
    coverage — suitable for a training-/grounding-data-summary export.
    """
    evidence: list[Any]
    citations: list[str] = []
    if hasattr(source, "evidence"):
        evidence = list(source.evidence)  # type: ignore[union-attr]
        citations = list(getattr(source, "citations", []) or [])
    else:
        evidence = list(source)  # type: ignore[arg-type]

    by_source_type = Counter(getattr(e, "source_type", "unknown") for e in evidence)
    by_trust = Counter(getattr(getattr(e, "trust_level", None), "value", "unknown") for e in evidence)
    unique_sources = sorted({getattr(e, "source_id", "") for e in evidence if getattr(e, "source_id", "")})
    cited = {c.split(":")[0] for c in citations}

    return {
        "title": title,
        "generated_at": utcnow().isoformat(),
        "evidence_items": len(evidence),
        "unique_sources": len(unique_sources),
        "source_ids": unique_sources,
        "by_source_type": dict(by_source_type),
        "by_trust_level": dict(by_trust),
        "citations": len(citations),
        "cited_sources": sorted(cited),
        "grounding_coverage": round(len(cited) / len(unique_sources), 4) if unique_sources else 0.0,
    }
