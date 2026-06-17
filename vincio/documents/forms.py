"""Forms / KYC structured extraction.

A :class:`DocumentAI` protocol unifies Textract / Azure Document Intelligence /
Google Document AI behind one shape, with an offline
:class:`HeuristicFormExtractor` so the dominant invoice/receipt/ID use-case works
without a cloud call. Extracted :class:`FormField`\\ s carry confidence and (when
available) a bounding box, and convert to citable
:class:`~vincio.core.types.EvidenceItem`\\ s — a filled form is grounded evidence
like any other source.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from ..core.types import Document, EvidenceItem, TrustLevel

__all__ = [
    "FormField",
    "DocumentAI",
    "HeuristicFormExtractor",
    "TextractDocumentAI",
    "AzureDocumentAI",
    "GoogleDocumentAI",
    "form_fields_to_evidence",
]


class FormField(BaseModel):
    """One extracted key/value field with provenance."""

    name: str
    value: str
    confidence: float = 0.5
    page: int | None = None
    # (x0, top, x1, bottom) in the source document's coordinate system.
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict = Field(default_factory=dict)


class DocumentAI(Protocol):
    async def extract_fields(self, source: str | Path) -> list[FormField]:  # pragma: no cover
        ...


# Common label → canonical-name hints for the heuristic extractor (KYC + invoice).
_LABEL_ALIASES = {
    "name": "name",
    "full name": "name",
    "date of birth": "date_of_birth",
    "dob": "date_of_birth",
    "id number": "id_number",
    "identification number": "id_number",
    "passport number": "passport_number",
    "nationality": "nationality",
    "address": "address",
    "invoice number": "invoice_number",
    "invoice no": "invoice_number",
    "invoice date": "invoice_date",
    "due date": "due_date",
    "total": "total",
    "total due": "total",
    "amount due": "total",
    "subtotal": "subtotal",
    "tax": "tax",
    "vat": "tax",
    "account number": "account_number",
}

_KV_RE = re.compile(r"^\s*([A-Za-z][\w ./&-]{1,40}?)\s*[:#]\s*(.+?)\s*$")


class HeuristicFormExtractor:
    """Offline key/value extractor — ``Label: value`` lines + known aliases.

    Deterministic and dependency-free: confidence reflects whether the label is
    a recognized KYC/invoice field. The honest fallback when no Document-AI
    backend is configured.
    """

    def __init__(self, *, min_confidence: float = 0.0) -> None:
        self.min_confidence = min_confidence

    def extract(self, source: str | Document) -> list[FormField]:
        text = source.text if isinstance(source, Document) else str(source)
        fields: list[FormField] = []
        seen: set[str] = set()
        for line in text.splitlines():
            match = _KV_RE.match(line)
            if not match:
                continue
            raw_label = match.group(1).strip()
            value = match.group(2).strip()
            if not value or len(value) > 200:
                continue
            canonical = _LABEL_ALIASES.get(raw_label.lower())
            name = canonical or re.sub(r"\s+", "_", raw_label.lower())
            if name in seen:
                continue
            seen.add(name)
            confidence = 0.85 if canonical else 0.55
            if confidence < self.min_confidence:
                continue
            fields.append(
                FormField(name=name, value=value, confidence=confidence,
                          metadata={"label": raw_label})
            )
        return fields

    async def extract_fields(self, source: str | Path) -> list[FormField]:
        path = Path(source)
        if path.is_file():
            from .loaders import load_document

            return self.extract(load_document(path))
        return self.extract(str(source))


class _CloudDocumentAI:
    """Shared base for the cloud Document-AI adapters (optional deps)."""

    extra = ""
    name = "document-ai"

    def __init__(self, **client_kwargs: object) -> None:
        self.client_kwargs = client_kwargs

    async def extract_fields(self, source: str | Path) -> list[FormField]:  # pragma: no cover
        raise NotImplementedError(
            f"{type(self).__name__} requires the {self.extra!r} backend SDK; "
            "install it and implement extract_fields, or use HeuristicFormExtractor offline"
        )


class TextractDocumentAI(_CloudDocumentAI):
    """AWS Textract key/value extraction (``vincio[s3]`` + boto3)."""

    extra = "s3"
    name = "textract"


class AzureDocumentAI(_CloudDocumentAI):
    """Azure AI Document Intelligence key/value extraction."""

    extra = "azure"
    name = "azure-document-intelligence"


class GoogleDocumentAI(_CloudDocumentAI):
    """Google Document AI key/value extraction."""

    extra = "gcp"
    name = "google-document-ai"


def form_fields_to_evidence(
    fields: list[FormField],
    *,
    source_id: str,
    source_uri: str | None = None,
) -> list[EvidenceItem]:
    """Convert extracted form fields to citable evidence (bbox + confidence)."""
    items: list[EvidenceItem] = []
    for index, field in enumerate(fields, start=1):
        items.append(
            EvidenceItem(
                id=f"{source_id}:F{index}",
                source_id=source_id,
                source_type="document",
                text=f"{field.name}: {field.value}",
                media_ref=source_uri,
                page=field.page,
                trust_level=TrustLevel.UNTRUSTED_DOCUMENT,
                authority=field.confidence,
                provenance=0.9,
                metadata={
                    "field": field.name,
                    "confidence": field.confidence,
                    "bbox": list(field.bbox) if field.bbox else None,
                },
            )
        )
    return items
