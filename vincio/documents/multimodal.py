"""Multimodal evidence: convert images into visual evidence items.

A vision-capable provider observes each image and produces region-tagged
observations that become :class:`EvidenceItem` records with provenance —
so visual facts can be cited, scored, and budgeted like any other evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import DocumentError
from ..core.types import ContentPart, EvidenceItem, ImageRef, Message, ModelRequest, TrustLevel
from ..providers.base import ModelProvider

__all__ = ["ImageObservation", "ImageAnalyzer", "image_evidence_items"]


class ImageObservation(BaseModel):
    region: str = "full"  # e.g. "top-right", "table area"
    observation: str
    confidence: float = 0.8
    metadata: dict[str, Any] = Field(default_factory=dict)


OBSERVE_PROMPT = """Analyze this image and report concrete observations.
Return a JSON array; each element: {"region": "<location in image>",
"observation": "<one factual statement>", "confidence": <0..1>}.
Report text content, UI state, chart values, and notable elements.
Only state what is visually verifiable. Return JSON only."""

OBSERVATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "observation": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["region", "observation", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["observations"],
    "additionalProperties": False,
}


class ImageAnalyzer:
    def __init__(self, provider: ModelProvider, *, model: str) -> None:
        capabilities = provider.capabilities(model)
        if not capabilities.vision:
            raise DocumentError(f"model {model!r} does not support vision input")
        self.provider = provider
        self.model = model

    async def observe(self, image: str | Path | ImageRef) -> list[ImageObservation]:
        image_ref = image if isinstance(image, ImageRef) else ImageRef(path=str(image))
        request = ModelRequest(
            model=self.model,
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentPart(type="text", text=OBSERVE_PROMPT),
                        ContentPart(type="image", image=image_ref),
                    ],
                )
            ],
            output_schema=OBSERVATION_SCHEMA,
            output_schema_name="image_observations",
            temperature=0.0,
        )
        response = await self.provider.generate(request)
        payload = response.structured
        if payload is None:
            try:
                payload = json.loads(response.text)
            except json.JSONDecodeError as exc:
                raise DocumentError(f"image analysis returned non-JSON output: {response.text[:200]}") from exc
        raw_observations = payload.get("observations", payload if isinstance(payload, list) else [])
        observations: list[ImageObservation] = []
        for entry in raw_observations:
            try:
                observations.append(ImageObservation.model_validate(entry))
            except ValueError:
                continue
        return observations


def image_evidence_items(
    image_id: str,
    observations: list[ImageObservation],
    *,
    source_uri: str | None = None,
) -> list[EvidenceItem]:
    """Convert observations to citable evidence (image_evidence)."""
    items: list[EvidenceItem] = []
    for index, observation in enumerate(observations, start=1):
        items.append(
            EvidenceItem(
                id=f"{image_id}:R{index}",
                source_id=image_id,
                source_type="image",
                text=f"[{observation.region}] {observation.observation}",
                media_ref=source_uri,
                trust_level=TrustLevel.UNTRUSTED_DOCUMENT,
                provenance=0.9,
                authority=observation.confidence,
                metadata={"region": observation.region, "confidence": observation.confidence},
            )
        )
    return items
