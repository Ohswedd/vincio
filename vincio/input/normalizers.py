"""Input normalization: text cleanup and language detection.
Pure-python, deterministic, offline."""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize_text", "detect_language"]

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​‌‍﻿]")
_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")

# Smart quotes and typographic characters → ASCII equivalents.
_TRANSLATE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
        "…": "...",
    }
)


def normalize_text(text: str) -> str:
    """Unicode NFC, strip control/zero-width chars, normalize quotes/spaces."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text)
    text = text.translate(_TRANSLATE)
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


# Stopword profiles for fast offline language identification. Scores are
# computed as stopword hits per token; the best-scoring language above a
# floor wins, else "en" for Latin scripts and the script name otherwise.
_LANG_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the and of to in is you that it for on are with as was at be this have from or had by not but what all were when there can".split()),
    "es": frozenset("el la de que y en un ser se no haber por con su para como estar tener le lo todo pero más hacer este ya o cuando".split()),
    "fr": frozenset("le de un être et à il avoir ne je son que se qui ce dans en du elle au pour pas vous par sur faire plus".split()),
    "de": frozenset("der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er".split()),
    "it": frozenset("di e il la che è per un in non sono io ho lui ma si con come ci questo qui hanno del alla più o anche".split()),
    "pt": frozenset("o de a e que do da em um para é com não uma os no se na por mais as dos como mas foi ao ele".split()),
    "nl": frozenset("de het een en van ik te dat die in je niet zijn is was op aan met als voor had er maar om hem dan".split()),
}


def detect_language(text: str) -> str:
    """Best-effort ISO 639-1 language code; falls back to script detection."""
    if not text:
        return "en"
    # Script-based shortcuts for non-Latin scripts.
    counts: dict[str, int] = {}
    for char in text[:2000]:
        if char.isalpha():
            try:
                script = unicodedata.name(char).split()[0]
            except ValueError:
                continue
            counts[script] = counts.get(script, 0) + 1
    if counts:
        dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
        script_map = {
            "CJK": "zh",
            "HIRAGANA": "ja",
            "KATAKANA": "ja",
            "HANGUL": "ko",
            "CYRILLIC": "ru",
            "ARABIC": "ar",
            "HEBREW": "he",
            "DEVANAGARI": "hi",
            "THAI": "th",
            "GREEK": "el",
        }
        if dominant in script_map and counts[dominant] >= max(1, sum(counts.values()) // 3):
            return script_map[dominant]
    tokens = re.findall(r"[a-zà-ÿäöüß']+", text.lower())[:400]
    if not tokens:
        return "en"
    best_lang, best_score = "en", 0.0
    for lang, stopwords in _LANG_STOPWORDS.items():
        score = sum(1 for token in tokens if token in stopwords) / len(tokens)
        if score > best_score:
            best_lang, best_score = lang, score
    return best_lang if best_score >= 0.05 else "en"
