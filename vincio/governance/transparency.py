"""EU AI Act transparency artifacts — generated, deadline-agnostic.

The EU AI Act's GenAI transparency duties (machine-readable synthetic-content
marking, AI-interaction disclosure, and a training/grounding-data summary) take
effect 2 Aug 2026. Vincio supplies the *artifacts and hooks*, configurable and
date-agnostic — it does not hard-code a deadline or become a compliance service.

* :func:`mark_synthetic_content` emits a C2PA-style **provenance manifest** that
  binds to the output by SHA-256 (the IPTC ``trainedAlgorithmicMedia`` digital
  source type), suitable for attaching as content credentials / metadata.
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
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

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
    "ai_disclosure",
    "data_summary",
]

# IPTC digital-source-type term for AI-generated content (C2PA standard value).
_TRAINED_ALGORITHMIC_MEDIA = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"

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
    content: str,
    *,
    model_id: str | None = None,
    provider: str | None = None,
    extra_assertions: list[dict[str, Any]] | None = None,
    signer: ContentSigner | None = None,
) -> ProvenanceManifest:
    """Build a provenance manifest marking ``content`` as AI-generated.

    The manifest is bound to the exact output by SHA-256, so a downstream
    consumer can confirm the credential matches the content it received. Pass a
    ``signer`` (e.g. :class:`HmacSigner`, or your own :class:`ContentSigner`) to
    attach a cryptographic signature over the binding payload — verify it later
    with :func:`verify_manifest`.
    """
    import vincio

    manifest = ProvenanceManifest(
        claim_generator=f"vincio/{vincio.__version__}",
        model_id=model_id,
        provider=provider,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
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
    content: str,
    *,
    signer: ContentSigner | None = None,
) -> bool:
    """Verify a manifest against the content it claims to describe.

    Always checks the SHA-256 content binding. If the manifest carries a
    signature, a ``signer`` with the matching key must be supplied to verify it
    (returns ``False`` when a signature is present but no verifier is given, so
    an unverifiable credential is never reported as valid).
    """
    if manifest.content_sha256 != hashlib.sha256(content.encode("utf-8")).hexdigest():
        return False
    if manifest.signature is not None:
        if signer is None:
            return False
        return signer.verify(manifest.signing_payload(), manifest.signature.get("value", ""))
    return True


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
