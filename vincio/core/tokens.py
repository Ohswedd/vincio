"""Token counting.

Vincio needs deterministic, fast token estimates for budgeting and scoring.
The default counter is a calibrated heuristic that works offline with zero
dependencies. When ``tiktoken`` is installed (``pip install "vincio[tokenizers]"``)
exact BPE counts are used for known model families.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

__all__ = ["TokenCounter", "HeuristicTokenCounter", "TiktokenCounter", "get_token_counter", "count_tokens"]


class TokenCounter(Protocol):
    def count(self, text: str) -> int:  # pragma: no cover - protocol
        ...


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class HeuristicTokenCounter:
    """Deterministic token estimate without external dependencies.

    Calibrated against cl100k/o200k behavior: words shorter than 7 chars are
    usually one token; longer words split roughly every 4 characters;
    punctuation is one token each. Accurate to within ~±10% on English prose,
    which is sufficient for budgeting (budgets keep a safety margin).
    """

    def count(self, text: str) -> int:
        if not text:
            return 0
        tokens = 0
        for piece in _WORD_RE.findall(text):
            if len(piece) <= 6:
                tokens += 1
            else:
                tokens += (len(piece) + 3) // 4
        return max(1, tokens)


class TiktokenCounter:
    """Exact BPE counting via tiktoken (optional dependency)."""

    def __init__(self, model: str = "gpt-4o") -> None:
        import tiktoken

        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("o200k_base")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))


@lru_cache(maxsize=32)
def get_token_counter(model: str | None = None) -> TokenCounter:
    """Return the best available counter for *model*.

    Prefers tiktoken when installed; falls back to the heuristic counter.
    Results are cached per model name.
    """
    try:
        return TiktokenCounter(model or "gpt-4o")
    except Exception:
        return HeuristicTokenCounter()


def count_tokens(text: str, model: str | None = None) -> int:
    return get_token_counter(model).count(text)
