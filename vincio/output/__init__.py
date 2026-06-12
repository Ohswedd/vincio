"""Vincio output engine."""

from .parsers import (
    extract_citations,
    extract_json,
    extract_markdown_metadata,
    lenient_json_loads,
    parse_partial_json,
)
from .repair import Repairer, RepairOutcome
from .schemas import OutputContract, OutputSchema, RepairPolicy, SchemaRegistry, ValidatorSpec
from .validators import OutputValidator, ValidationReport, ValidationStep

__all__ = [
    "extract_citations",
    "extract_json",
    "extract_markdown_metadata",
    "lenient_json_loads",
    "parse_partial_json",
    "RepairOutcome",
    "Repairer",
    "OutputContract",
    "OutputSchema",
    "RepairPolicy",
    "SchemaRegistry",
    "ValidatorSpec",
    "OutputValidator",
    "ValidationReport",
    "ValidationStep",
]
