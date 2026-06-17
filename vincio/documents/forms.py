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
from typing import Any, Protocol

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


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict *or* an SDK object — so the response parsers work
    against both a live SDK result and an offline synthetic dict (tested)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _bbox_from_box(box: Any) -> tuple[float, float, float, float] | None:
    """Textract ``BoundingBox`` (Left/Top/Width/Height) → (x0, top, x1, bottom)."""
    if not box:
        return None
    left = float(_attr(box, "Left", 0.0) or 0.0)
    top = float(_attr(box, "Top", 0.0) or 0.0)
    width = float(_attr(box, "Width", 0.0) or 0.0)
    height = float(_attr(box, "Height", 0.0) or 0.0)
    return (left, top, left + width, top + height)


def _bbox_from_polygon(polygon: Any) -> tuple[float, float, float, float] | None:
    """A flat or point-list polygon → an axis-aligned (x0, top, x1, bottom) box."""
    if not polygon:
        return None
    xs: list[float] = []
    ys: list[float] = []
    if isinstance(polygon[0], (int, float)):  # flat [x0, y0, x1, y1, ...]
        xs = [float(v) for v in polygon[0::2]]
        ys = [float(v) for v in polygon[1::2]]
    else:  # list of {x, y} points
        xs = [float(_attr(p, "x", 0.0) or 0.0) for p in polygon]
        ys = [float(_attr(p, "y", 0.0) or 0.0) for p in polygon]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


class TextractDocumentAI:
    """AWS Textract FORMS key/value extraction.

    Dependency-injected: build the client yourself and pass it in, so no SDK is a
    hard dependency::

        import boto3
        ai = TextractDocumentAI(client=boto3.client("textract"))  # vincio[s3]

    The blocking ``analyze_document`` call runs in a worker thread; the response
    parsing (:meth:`parse`) is a pure function tested offline.
    """

    name = "textract"

    def __init__(self, client: Any, *, feature_types: tuple[str, ...] = ("FORMS",)) -> None:
        self.client = client
        self.feature_types = list(feature_types)

    async def extract_fields(self, source: str | Path) -> list[FormField]:
        import asyncio

        data = Path(source).read_bytes()
        response = await asyncio.to_thread(
            self.client.analyze_document,
            Document={"Bytes": data},
            FeatureTypes=self.feature_types,
        )
        return self.parse(response)

    @staticmethod
    def parse(response: Any) -> list[FormField]:
        blocks = _attr(response, "Blocks", []) or []
        by_id = {_attr(b, "Id"): b for b in blocks}

        def child_text(block: Any) -> str:
            words: list[str] = []
            for rel in _attr(block, "Relationships", []) or []:
                if _attr(rel, "Type") != "CHILD":
                    continue
                for cid in _attr(rel, "Ids", []) or []:
                    child = by_id.get(cid)
                    if _attr(child, "BlockType") == "WORD":
                        words.append(str(_attr(child, "Text", "")))
                    elif _attr(child, "BlockType") == "SELECTION_ELEMENT":
                        words.append("☑" if _attr(child, "SelectionStatus") == "SELECTED" else "☐")
            return " ".join(w for w in words if w).strip()

        fields: list[FormField] = []
        for block in blocks:
            if _attr(block, "BlockType") != "KEY_VALUE_SET":
                continue
            if "KEY" not in (_attr(block, "EntityTypes", []) or []):
                continue
            key_text = child_text(block)
            if not key_text:
                continue
            value_text = ""
            for rel in _attr(block, "Relationships", []) or []:
                if _attr(rel, "Type") == "VALUE":
                    for vid in _attr(rel, "Ids", []) or []:
                        value_text = child_text(by_id.get(vid))
                        break
            confidence = float(_attr(block, "Confidence", 0.0) or 0.0)
            geometry = _attr(block, "Geometry")
            fields.append(
                FormField(
                    name=re.sub(r"\s+", "_", key_text.rstrip(":").lower()),
                    value=value_text,
                    confidence=round(confidence / 100.0, 4) if confidence > 1 else confidence,
                    page=_attr(block, "Page"),
                    bbox=_bbox_from_box(_attr(geometry, "BoundingBox")),
                    metadata={"label": key_text},
                )
            )
        return fields


class AzureDocumentAI:
    """Azure AI Document Intelligence key/value extraction.

    Dependency-injected: pass a built ``DocumentIntelligenceClient`` and the
    model id (default ``prebuilt-document``)::

        ai = AzureDocumentAI(client=di_client, model_id="prebuilt-document")

    Parsing (:meth:`parse`) over the analyze result is a pure, offline-tested
    function.
    """

    name = "azure-document-intelligence"

    def __init__(self, client: Any, *, model_id: str = "prebuilt-document") -> None:
        self.client = client
        self.model_id = model_id

    async def extract_fields(self, source: str | Path) -> list[FormField]:
        import asyncio

        data = Path(source).read_bytes()

        def run() -> Any:
            poller = self.client.begin_analyze_document(self.model_id, body=data)
            return poller.result()

        return self.parse(await asyncio.to_thread(run))

    @staticmethod
    def parse(result: Any) -> list[FormField]:
        fields: list[FormField] = []
        for pair in _attr(result, "key_value_pairs", []) or []:
            key = _attr(pair, "key")
            value = _attr(pair, "value")
            key_text = str(_attr(key, "content", "") or "").strip()
            if not key_text:
                continue
            regions = _attr(key, "bounding_regions", []) or []
            region = regions[0] if regions else None
            fields.append(
                FormField(
                    name=re.sub(r"\s+", "_", key_text.rstrip(":").lower()),
                    value=str(_attr(value, "content", "") or "").strip(),
                    confidence=float(_attr(pair, "confidence", 0.5) or 0.5),
                    page=_attr(region, "page_number"),
                    bbox=_bbox_from_polygon(_attr(region, "polygon")),
                    metadata={"label": key_text},
                )
            )
        return fields


class GoogleDocumentAI:
    """Google Document AI form-field extraction.

    Dependency-injected: pass a built ``DocumentProcessorServiceClient`` and the
    fully-qualified processor name::

        ai = GoogleDocumentAI(client=docai_client, processor="projects/…/processors/…")

    Parsing (:meth:`parse`) resolves each field's text from the document layout
    and is a pure, offline-tested function.
    """

    name = "google-document-ai"

    def __init__(self, client: Any, *, processor: str, mime_type: str = "application/pdf") -> None:
        self.client = client
        self.processor = processor
        self.mime_type = mime_type

    async def extract_fields(self, source: str | Path) -> list[FormField]:
        import asyncio

        data = Path(source).read_bytes()

        def run() -> Any:
            request = {
                "name": self.processor,
                "raw_document": {"content": data, "mime_type": self.mime_type},
            }
            return self.client.process_document(request=request).document

        return self.parse(await asyncio.to_thread(run))

    @staticmethod
    def parse(document: Any) -> list[FormField]:
        full_text = str(_attr(document, "text", "") or "")

        def layout_text(layout: Any) -> str:
            # Newer responses expose resolved text directly.
            direct = _attr(layout, "content")
            if direct:
                return str(direct).strip()
            anchor = _attr(layout, "text_anchor")
            segments = _attr(anchor, "text_segments", []) or []
            parts: list[str] = []
            for seg in segments:
                start = int(_attr(seg, "start_index", 0) or 0)
                end = int(_attr(seg, "end_index", 0) or 0)
                parts.append(full_text[start:end])
            return "".join(parts).strip()

        fields: list[FormField] = []
        for page in _attr(document, "pages", []) or []:
            page_number = _attr(page, "page_number")
            for field in _attr(page, "form_fields", []) or []:
                name_layout = _attr(field, "field_name")
                value_layout = _attr(field, "field_value")
                key_text = layout_text(name_layout)
                if not key_text:
                    continue
                bbox = _bbox_from_polygon(
                    _attr(_attr(name_layout, "bounding_poly"), "normalized_vertices")
                    or _attr(_attr(name_layout, "bounding_poly"), "vertices")
                )
                fields.append(
                    FormField(
                        name=re.sub(r"\s+", "_", key_text.rstrip(":").lower()),
                        value=layout_text(value_layout),
                        confidence=float(_attr(field, "field_name_confidence", 0.5) or 0.5),
                        page=page_number,
                        bbox=bbox,
                        metadata={"label": key_text},
                    )
                )
        return fields


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
