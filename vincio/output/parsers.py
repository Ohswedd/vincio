"""Output parsing (parse step).

Robust extraction of structured data from model text: fenced JSON blocks,
inline JSON objects/arrays, lenient JSON repair-parsing, citation
extraction, and a streaming-safe partial JSON reader.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..core.errors import OutputParseError

__all__ = [
    "extract_json",
    "lenient_json_loads",
    "extract_citations",
    "parse_partial_json",
    "extract_markdown_metadata",
]

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
_CITATION_RE = re.compile(r"\[((?:[A-Za-z]+[\w.-]*:)?[A-Za-z0-9][\w.:-]*)\]")


def _find_json_span(text: str) -> str | None:
    """Locate the first balanced top-level JSON object/array in text."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        return text[start : index + 1]
            start = text.find(opener, start + 1)
    return None


def lenient_json_loads(text: str) -> Any:
    """Parse JSON tolerating common model mistakes: trailing commas,
    single-quoted strings, unquoted keys, Python literals."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    repaired = text.strip()
    # Python literals → JSON.
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    # Trailing commas.
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    # Unquoted object keys.
    repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    # Single-quoted strings (after key quoting to avoid double-processing).
    single_quoted = re.sub(
        r"'((?:[^'\\]|\\.)*)'",
        lambda m: json.dumps(m.group(1).replace("\\'", "'")),
        repaired,
    )
    try:
        return json.loads(single_quoted)
    except json.JSONDecodeError as exc:
        raise OutputParseError(f"could not parse JSON: {exc}", details={"text": text[:500]}) from exc


def extract_json(text: str) -> Any:
    """Extract and parse the JSON payload from model output text."""
    if not text or not text.strip():
        raise OutputParseError("empty output")
    stripped = text.strip()
    # 1. Whole-output JSON.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # 2. Fenced block.
    fence = _FENCE_RE.search(text)
    if fence:
        return lenient_json_loads(fence.group(1).strip())
    # 3. First balanced JSON span.
    span = _find_json_span(text)
    if span is not None:
        return lenient_json_loads(span)
    # 4. Lenient parse of the whole thing.
    return lenient_json_loads(stripped)


def parse_partial_json(text: str) -> tuple[Any | None, bool]:
    """Best-effort parse of streaming/truncated JSON.

    Returns (value, complete). Balances unclosed brackets/strings so partial
    structured output can be displayed during streaming.
    """
    stripped = text.strip()
    if not stripped:
        return None, False
    try:
        return json.loads(stripped), True
    except json.JSONDecodeError:
        pass
    # Balance the string: close open quotes and brackets.
    stack: list[str] = []
    in_string = False
    escape = False
    for char in stripped:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()
    balanced = stripped
    if in_string:
        balanced += '"'
    # Drop a trailing comma/colon fragment before closing.
    balanced = re.sub(r"[,:]\s*$", "", balanced)
    balanced += "".join(reversed(stack))
    try:
        return json.loads(balanced), False
    except json.JSONDecodeError:
        return None, False


def extract_citations(text: str, *, valid_ids: set[str] | None = None) -> list[str]:
    """Extract citation refs like [E1], [D1:C7], [doc:p4] from output text."""
    citations: list[str] = []
    for match in _CITATION_RE.finditer(text or ""):
        ref = match.group(1)
        if valid_ids is not None and ref not in valid_ids:
            continue
        if ref not in citations:
            citations.append(ref)
    return citations


_MD_META_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def extract_markdown_metadata(text: str) -> tuple[dict[str, Any], str]:
    """Split a hybrid markdown report into (front-matter metadata, body)
    (mode 5)."""
    match = _MD_META_RE.match(text or "")
    if not match:
        return {}, text
    import yaml

    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(metadata, dict):
        return {}, text
    return metadata, text[match.end() :]
