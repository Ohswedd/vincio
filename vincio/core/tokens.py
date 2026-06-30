"""Token counting.

Vincio needs deterministic, fast token estimates for budgeting and scoring.
The default counter is a calibrated heuristic that works offline with zero
dependencies. When ``tiktoken`` is installed (``pip install "vincio[tokenizers]"``)
exact BPE counts are used for known model families.

Provider-native exact counters (Anthropic ``count_tokens``, Gemini
``countTokens``) sit behind the :class:`TokenCounter` Protocol and are selected
by resolved model id via :func:`register_token_counter` — the registry
foundation lets a provider plug in its exact counter without changing the
offline default. ``count_tokens`` is memoized so repeated compiler passes and
incremental recompiles never re-tokenize the same text.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from typing import Protocol

from .diagnostics import note_suppressed

__all__ = [
    "TokenCounter",
    "HeuristicTokenCounter",
    "TiktokenCounter",
    "register_token_counter",
    "get_token_counter",
    "count_tokens",
    "count_tokens_many",
]


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


# Registered provider-native counters, matched by model id. Each entry is a
# (matcher, factory): matcher is a str prefix or a predicate over the model id;
# factory builds the counter lazily. More-specific (longer-prefix) matches win.
_Matcher = Callable[[str], bool]
_REGISTERED: list[tuple[_Matcher, int, Callable[[str], TokenCounter]]] = []


def register_token_counter(
    matcher: str | _Matcher, factory: Callable[[str], TokenCounter]
) -> None:
    """Register a provider-native :class:`TokenCounter` for matching model ids.

    *matcher* is a model-id prefix (e.g. ``"claude-"``) or a predicate; *factory*
    receives the resolved model id and returns the counter. Selection prefers the
    longest matching prefix, then registration order. Offline default behavior is
    unchanged until a counter is registered.
    """
    if isinstance(matcher, str):
        prefix = matcher
        specificity = len(prefix)

        def _match(model: str, _p: str = prefix) -> bool:
            return model.startswith(_p)

        _REGISTERED.append((_match, specificity, factory))
    else:
        _REGISTERED.append((matcher, 0, factory))
    # Invalidate both the counter cache and the per-text memo, so a counter
    # registered after some text was already counted takes effect.
    get_token_counter.cache_clear()
    _count_cached.cache_clear()


def _select_registered(model: str) -> Callable[[str], TokenCounter] | None:
    best: tuple[int, Callable[[str], TokenCounter]] | None = None
    for matcher, specificity, factory in _REGISTERED:
        try:
            if matcher(model) and (best is None or specificity > best[0]):
                best = (specificity, factory)
        except Exception:  # noqa: BLE001 - a bad matcher must not break counting
            continue
    return best[1] if best else None


@lru_cache(maxsize=64)
def get_token_counter(model: str | None = None) -> TokenCounter:
    """Return the best available counter for *model*.

    Resolution order: a registered provider-native counter (selected by model
    id), then tiktoken when installed, then the offline heuristic. Cached per
    model name.
    """
    if model is not None:
        factory = _select_registered(model)
        if factory is not None:
            try:
                return factory(model)
            except Exception:
                note_suppressed("tokens.native_counter_build")
    try:
        return TiktokenCounter(model or "gpt-4o")
    except Exception:
        note_suppressed("tokens.tiktoken_build")
        return HeuristicTokenCounter()


@lru_cache(maxsize=16_384)
def _count_cached(text: str, model: str | None) -> int:
    return get_token_counter(model).count(text)


def count_tokens(text: str, model: str | None = None) -> int:
    """Token count for *text*, memoized by ``(text, model)``.

    The O(n²) dedupe/conflict loops and incremental recompiles re-count the same
    strings repeatedly; the bounded cache makes those passes free after the first.
    """
    if not text:
        return 0
    return _count_cached(text, model)


def count_tokens_many(texts: list[str], model: str | None = None) -> list[int]:
    """Token counts for a batch of *texts*, in order.

    The compiler's normalization pass counts every candidate's tokens at once.
    Resolving the counter once for the whole batch (it is itself cached) and
    counting each text through the same per-text memo makes the result
    element-for-element identical to ``[count_tokens(t, model) for t in texts]``
    while a large pool is counted in a single pass — and a batch-capable counter
    (one exposing ``count_many``) amortizes the whole batch in one native call.
    """
    if not texts:
        return []
    counter = get_token_counter(model)
    batch = getattr(counter, "count_many", None)
    if batch is not None:
        return [max(0, int(n)) if text else 0 for text, n in zip(texts, batch(texts), strict=True)]
    return [_count_cached(text, model) if text else 0 for text in texts]
