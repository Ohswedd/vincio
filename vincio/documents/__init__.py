"""Vincio document engine: loaders, parsers, OCR, layout, multimodal."""

from .layout import (
    LayoutBlock,
    LayoutFigure,
    LayoutWord,
    PageLayout,
    assemble_layout,
    extract_pdf_layout,
    group_words_into_lines,
    order_blocks,
)
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
    "LayoutWord",
    "LayoutBlock",
    "LayoutFigure",
    "PageLayout",
    "group_words_into_lines",
    "order_blocks",
    "assemble_layout",
    "extract_pdf_layout",
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
