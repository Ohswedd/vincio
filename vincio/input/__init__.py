"""Vincio input engine: normalization, classification, routing."""

from .classifiers import (
    AmbiguityReport,
    TaskClassification,
    classify_file,
    classify_task,
    detect_ambiguity,
)
from .normalizers import detect_language, normalize_text
from .routers import InputRouter, RoutedInput

__all__ = [
    "AmbiguityReport",
    "TaskClassification",
    "classify_file",
    "classify_task",
    "detect_ambiguity",
    "detect_language",
    "normalize_text",
    "InputRouter",
    "RoutedInput",
]
