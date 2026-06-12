"""Vincio document engine: loaders, parsers, OCR, multimodal."""

from .loaders import (
    SUPPORTED_EXTENSIONS,
    load_directory,
    load_document,
    load_docx,
    load_pdf,
    load_xlsx,
)
from .multimodal import ImageAnalyzer, ImageObservation, image_evidence_items
from .ocr import OCREngine, TesseractOCR, VisionModelOCR
from .parsers import (
    CodeSymbol,
    Section,
    TableData,
    extract_code_symbols,
    extract_markdown_sections,
    extract_markdown_tables,
    infer_table_schema,
    parse_csv_table,
    strip_html,
    table_quality_checks,
)

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "load_directory",
    "load_document",
    "load_docx",
    "load_pdf",
    "load_xlsx",
    "ImageAnalyzer",
    "ImageObservation",
    "image_evidence_items",
    "OCREngine",
    "TesseractOCR",
    "VisionModelOCR",
    "CodeSymbol",
    "Section",
    "TableData",
    "extract_code_symbols",
    "extract_markdown_sections",
    "extract_markdown_tables",
    "infer_table_schema",
    "parse_csv_table",
    "strip_html",
    "table_quality_checks",
]
