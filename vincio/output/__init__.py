"""Vincio output engine."""

from __future__ import annotations

from .constrained import (
    DecodingMode,
    choice_schema,
    negotiate_decoding,
    regex_schema,
    to_strict_json_schema,
)
from .correction import CorrectionResult, SelfCorrector, build_critique
from .parsers import (
    extract_citations,
    extract_json,
    extract_markdown_metadata,
    lenient_json_loads,
    parse_partial_json,
)
from .repair import Repairer, RepairOutcome
from .routing import SchemaRoute, SchemaRouter
from .schemas import OutputContract, OutputSchema, RepairPolicy, SchemaRegistry, ValidatorSpec
from .streaming import StreamingValidationEvent, StreamingValidator, validate_partial
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
    "DecodingMode",
    "negotiate_decoding",
    "to_strict_json_schema",
    "choice_schema",
    "regex_schema",
    "StreamingValidator",
    "StreamingValidationEvent",
    "validate_partial",
    "SelfCorrector",
    "CorrectionResult",
    "build_critique",
    "SchemaRoute",
    "SchemaRouter",
]
