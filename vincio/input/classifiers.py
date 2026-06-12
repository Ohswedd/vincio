"""Task/intent classification, file classification, ambiguity detection
(items 3–5, 9; task router taxonomy)."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, Field

from ..core.types import FileRef, TaskType, UserInput

__all__ = [
    "TaskClassification",
    "classify_task",
    "classify_file",
    "detect_ambiguity",
    "AmbiguityReport",
]


class TaskClassification(BaseModel):
    task_type: TaskType
    confidence: float
    signals: list[str] = Field(default_factory=list)


# Keyword/structure signals per task type, evaluated in priority order.
_TASK_SIGNALS: list[tuple[TaskType, re.Pattern[str], float, str]] = [
    (TaskType.DOCUMENT_COMPARISON, re.compile(r"(?i)\b(compare|difference between|diff|versus|vs\.?)\b.*\b(document|contract|version|file|proposal)s?\b"), 0.8, "compare+document"),
    (TaskType.COMPLIANCE_REVIEW, re.compile(r"(?i)\b(complian(?:ce|t)|gdpr|hipaa|sox|regulat(?:ion|ory)|audit(?:ing)? requirements?|policy violation)\b"), 0.8, "compliance terms"),
    (TaskType.EXTRACTION, re.compile(r"(?i)\b(extract|pull out|parse out|list all|find all|identify (?:all|every))\b.*\b(field|value|name|date|amount|entit|clause|number)s?\b"), 0.75, "extract+fields"),
    (TaskType.SUMMARIZATION, re.compile(r"(?i)\b(summari[sz]e|summary|tl;?dr|key points|brief overview|condense)\b"), 0.8, "summarize"),
    (TaskType.CLASSIFICATION, re.compile(r"(?i)\b(classify|categori[sz]e|label|which (?:category|class|type|bucket)|triage)\b"), 0.8, "classify"),
    (TaskType.DATA_ANALYSIS, re.compile(r"(?i)\b(analy[sz]e|trend|average|median|correlat|aggregate|chart|plot|statistics|spreadsheet|csv)\b"), 0.65, "analysis terms"),
    (TaskType.CODING, re.compile(r"(?i)\b(code|function|bug|refactor|implement|python|javascript|typescript|sql query|stack ?trace|exception|compile error)\b"), 0.7, "code terms"),
    (TaskType.PLANNING, re.compile(r"(?i)\b(plan|roadmap|schedule|milestones?|steps to|strategy for|break down)\b"), 0.6, "planning terms"),
    (TaskType.TOOL_ACTION, re.compile(r"(?i)\b(create|update|delete|send|issue|book|order|cancel|refund|schedule a|look ?up)\b.*\b(ticket|invoice|record|email|meeting|order|refund|account|crm|database)\b"), 0.65, "action+object"),
    (TaskType.CREATIVE_GENERATION, re.compile(r"(?i)\b(write|draft|compose|generate)\b.*\b(story|poem|blog|article|tagline|slogan|tweet|post|essay|song)\b"), 0.7, "creative writing"),
    (TaskType.DOCUMENT_QA, re.compile(r"(?i)\b(according to|in the (?:document|contract|report|manual|pdf)|based on the|what does the (?:document|contract|policy) say)\b"), 0.75, "doc reference"),
]

_QUESTION_RE = re.compile(r"(?i)^(who|what|when|where|why|how|which|does|do|did|is|are|can|could|will)\b|\?\s*$")


def classify_task(user_input: UserInput | str, *, has_sources: bool = False) -> TaskClassification:
    text = user_input if isinstance(user_input, str) else (user_input.text or "")
    files = [] if isinstance(user_input, str) else user_input.files
    signals: list[str] = []

    for task_type, pattern, confidence, signal in _TASK_SIGNALS:
        if pattern.search(text):
            signals.append(signal)
            return TaskClassification(task_type=task_type, confidence=confidence, signals=signals)

    if files or has_sources:
        if _QUESTION_RE.search(text.strip()):
            signals.append("question+documents")
            return TaskClassification(task_type=TaskType.DOCUMENT_QA, confidence=0.7, signals=signals)
        signals.append("documents attached")
        return TaskClassification(task_type=TaskType.DOCUMENT_QA, confidence=0.5, signals=signals)
    if _QUESTION_RE.search(text.strip()):
        signals.append("question form")
        return TaskClassification(task_type=TaskType.GENERAL, confidence=0.5, signals=signals)
    return TaskClassification(task_type=TaskType.GENERAL, confidence=0.4, signals=signals)


# Optional async LLM classifier hook: receives text, returns a TaskType value.
LLMTaskClassifier = Callable[[str], Awaitable[str]]


_FILE_KINDS: dict[str, str] = {
    ".pdf": "document",
    ".docx": "document",
    ".doc": "document",
    ".pptx": "presentation",
    ".txt": "text",
    ".md": "text",
    ".rst": "text",
    ".html": "web",
    ".htm": "web",
    ".csv": "tabular",
    ".tsv": "tabular",
    ".xlsx": "tabular",
    ".xls": "tabular",
    ".json": "data",
    ".jsonl": "data",
    ".yaml": "data",
    ".yml": "data",
    ".xml": "data",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".svg": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".flac": "audio",
    ".eml": "email",
    ".msg": "email",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".tsx": "code",
    ".jsx": "code",
    ".go": "code",
    ".rs": "code",
    ".java": "code",
    ".rb": "code",
    ".c": "code",
    ".h": "code",
    ".cpp": "code",
    ".cs": "code",
    ".sql": "code",
    ".sh": "code",
}


def classify_file(file: FileRef | str) -> str:
    """Coarse media classification by extension (item 5)."""
    path = file if isinstance(file, str) else file.path
    return _FILE_KINDS.get(Path(path).suffix.lower(), "unknown")


class AmbiguityReport(BaseModel):
    ambiguous: bool
    reasons: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)


_DEICTIC_RE = re.compile(r"(?i)\b(this|that|these|those|it|them)\b\s+(?:one|thing|stuff|item)?")
_UNRESOLVED_REF_RE = re.compile(r"(?i)\b(the (?:above|previous|earlier|attached|same))\b")


def detect_ambiguity(user_input: UserInput | str) -> AmbiguityReport:
    """Heuristic ambiguity detection (item 9)."""
    text = user_input if isinstance(user_input, str) else (user_input.text or "")
    files = [] if isinstance(user_input, str) else user_input.files
    reasons: list[str] = []
    questions: list[str] = []
    stripped = text.strip()

    if not stripped:
        reasons.append("empty input")
        questions.append("What would you like me to do?")
        return AmbiguityReport(ambiguous=True, reasons=reasons, clarifying_questions=questions)

    word_count = len(stripped.split())
    if word_count <= 2 and not files:
        reasons.append("very short input with no attachments")
        questions.append(f"Can you give more detail about what you need regarding '{stripped}'?")
    if _UNRESOLVED_REF_RE.search(stripped) and not files:
        reasons.append("reference to prior/attached content that is not present")
        questions.append("Which document or content are you referring to?")
    if _DEICTIC_RE.match(stripped) and word_count < 6 and not files:
        reasons.append("unresolved pronoun reference")
        questions.append("What does the reference point to?")
    return AmbiguityReport(
        ambiguous=bool(reasons), reasons=reasons, clarifying_questions=questions
    )
