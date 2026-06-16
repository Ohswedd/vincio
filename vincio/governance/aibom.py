"""AI-BOM: an AI bill of materials for blast-radius assessment.

The shipped release pipeline already emits a **CycloneDX SBOM** (Python
dependencies) and **SLSA provenance**. An AI-BOM adds the layer those miss: the
*AI* components a deployment depends on — the base model and version, the
embedding and rerank models, fine-tune datasets, and prompt/registry versions —
each with an optional **SHA-256 hash** so that when a model or dataset is found
compromised, you can answer "are we affected?" mechanically.

The output is CycloneDX-1.6-shaped (component types ``machine-learning-model``
and ``data``) so it slots next to the dependency SBOM. It is generated from the
live configuration, so it cannot drift from what the app actually loads.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from ..core.utils import utcnow
from ..observability.costs import default_price_table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.app import ContextApp
    from ..core.config import VincioConfig

__all__ = [
    "AIComponent",
    "AIBOM",
    "generate_aibom",
    "sha256_file",
    "sha256_text",
]

# CycloneDX component types used for AI artifacts.
_MODEL_TYPE = "machine-learning-model"
_DATA_TYPE = "data"


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's contents (for local model/dataset weights)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a string (for prompts / registry-pinned content)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AIComponent(BaseModel):
    """One AI artifact in the bill of materials."""

    type: str = _MODEL_TYPE  # machine-learning-model | data
    name: str
    role: str = "model"  # model | embedding-model | rerank-model | dataset | prompt
    provider: str | None = None
    version: str | None = None
    sha256: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    @property
    def bom_ref(self) -> str:
        ident = self.name
        if self.version:
            ident = f"{ident}@{self.version}"
        return f"{self.role}:{ident}"

    def to_cyclonedx(self) -> dict[str, Any]:
        comp: dict[str, Any] = {
            "type": self.type,
            "bom-ref": self.bom_ref,
            "name": self.name,
        }
        if self.provider:
            comp["publisher"] = self.provider
        if self.version:
            comp["version"] = self.version
        if self.sha256:
            comp["hashes"] = [{"alg": "SHA-256", "content": self.sha256}]
        props = {"vincio:role": self.role, **{f"vincio:{k}": v for k, v in self.properties.items()}}
        comp["properties"] = [{"name": k, "value": str(v)} for k, v in props.items()]
        return comp

    def verify(self, *, path: str | Path | None = None, text: str | None = None) -> bool:
        """Recompute the hash from a local artifact and compare to ``sha256``.

        Returns ``True`` when no expected hash is recorded (nothing to verify),
        and ``False`` on mismatch — so a compromised/swapped artifact is caught.
        """
        if self.sha256 is None:
            return True
        if path is not None:
            return sha256_file(path) == self.sha256
        if text is not None:
            return sha256_text(text) == self.sha256
        return False


class AIBOM(BaseModel):
    """An AI bill of materials, serializable as CycloneDX 1.6 JSON."""

    bom_format: str = "CycloneDX"
    spec_version: str = "1.6"
    generated_at: datetime = Field(default_factory=utcnow)
    vincio_version: str = ""
    application: str = "vincio_app"
    components: list[AIComponent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def by_role(self, role: str) -> list[AIComponent]:
        return [c for c in self.components if c.role == role]

    def verify_all(self, artifacts: dict[str, str | Path] | None = None) -> dict[str, bool]:
        """Verify every component that has a recorded hash.

        ``artifacts`` maps a component ``bom_ref`` to a local path whose hash is
        recomputed. Components with a hash but no provided artifact verify as
        ``False`` (cannot confirm), so a silent gap is never reported as intact.
        """
        artifacts = artifacts or {}
        out: dict[str, bool] = {}
        for comp in self.components:
            if comp.sha256 is None:
                continue
            path = artifacts.get(comp.bom_ref)
            out[comp.bom_ref] = comp.verify(path=path) if path is not None else False
        return out

    def to_cyclonedx(self) -> dict[str, Any]:
        return {
            "bomFormat": self.bom_format,
            "specVersion": self.spec_version,
            "version": 1,
            "metadata": {
                "timestamp": self.generated_at.isoformat(),
                "component": {"type": "application", "name": self.application},
                "tools": [{"vendor": "vincio", "name": "vincio governance aibom",
                           "version": self.vincio_version}],
                **self.metadata,
            },
            "components": [c.to_cyclonedx() for c in self.components],
        }

    def to_json(self, *, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_cyclonedx(), indent=indent, default=str)


def generate_aibom(
    target: ContextApp | VincioConfig,
    *,
    datasets: list[AIComponent] | None = None,
    prompts: list[AIComponent] | None = None,
    extra: list[AIComponent] | None = None,
) -> AIBOM:
    """Build an :class:`AIBOM` from the live configuration.

    The base model, embedder, and reranker are read from the config. Pass
    ``datasets`` (e.g. fine-tune sets) and ``prompts`` (registry-pinned versions
    with their hashes) to record the full AI dependency surface.
    """
    import vincio

    cfg: VincioConfig = cast("VincioConfig", getattr(target, "config", None) or target)
    price_table = default_price_table()
    tracker = getattr(target, "cost_tracker", None)
    if tracker is not None and getattr(tracker, "price_table", None) is not None:
        price_table = tracker.price_table

    model_id = getattr(target, "model", None) or cfg.provider.model
    components: list[AIComponent] = [
        AIComponent(
            type=_MODEL_TYPE,
            name=model_id,
            role="model",
            provider=cfg.provider.default,
            version=cfg.metadata.get("model_version"),
            properties={
                "pricing_input_per_mtok": price_table.lookup(model_id).input_per_mtok,
                "pricing_output_per_mtok": price_table.lookup(model_id).output_per_mtok,
            },
        )
    ]
    if cfg.provider.fallback_models:
        for fb in cfg.provider.fallback_models:
            components.append(AIComponent(type=_MODEL_TYPE, name=fb, role="model",
                                          provider=cfg.provider.default,
                                          properties={"fallback": True}))
    if cfg.retrieval.embedder:
        components.append(AIComponent(
            type=_MODEL_TYPE, name=cfg.retrieval.embedder, role="embedding-model",
            properties={"dimensions": cfg.retrieval.embedding_dimensions or "native"}))
    if cfg.retrieval.reranker:
        components.append(AIComponent(
            type=_MODEL_TYPE, name=cfg.retrieval.reranker, role="rerank-model"))

    components.extend(datasets or [])
    components.extend(prompts or [])
    components.extend(extra or [])

    return AIBOM(
        vincio_version=vincio.__version__,
        application=cfg.project,
        components=components,
        metadata={"provider": cfg.provider.default},
    )
