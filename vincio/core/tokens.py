"""Token counting.

Vincio needs deterministic, fast token estimates for budgeting and scoring.
The default counter is a calibrated heuristic that works offline with zero
dependencies. When ``tiktoken`` is installed (``pip install "vincio[tokenizers]"``)
exact BPE counts are used for known OpenAI model families.

:func:`register_token_counter` is the extension point a provider uses to plug in
its exact, offline counter, selected by resolved model id: when a provider that
can count *exactly and offline* is built, it registers its counter here (the
OpenAI provider registers ``tiktoken`` for the ``gpt-*`` / ``o*`` families; an
in-process GGUF model registers its own tokenizer for its served model id), so
counting becomes model-id-driven rather than tied to a single global default. A
hosted provider whose only exact count is a network round-trip (Anthropic
``count_tokens``, Gemini ``countTokens``) ships **no** hot-path counter — that
round-trip is unsuitable for the per-candidate scoring loop — but a deployment
that wants exact remote counts can register one through the same hook. With no
exact counter registered for a model the offline heuristic (or ``tiktoken``) is
used, unchanged. ``count_tokens`` is memoized so repeated compiler passes and
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
# (matcher, specificity, factory, key): matcher is a str prefix or a predicate
# over the model id; factory builds the counter lazily; key (optional) makes a
# registration idempotent so a provider built more than once registers once.
# More-specific (longer-prefix) matches win.
_Matcher = Callable[[str], bool]
_REGISTERED: list[tuple[_Matcher, int, Callable[[str], TokenCounter], str | None]] = []


def register_token_counter(
    matcher: str | _Matcher,
    factory: Callable[[str], TokenCounter],
    *,
    key: str | None = None,
) -> None:
    """Register a provider-native :class:`TokenCounter` for matching model ids.

    *matcher* is a model-id prefix (e.g. ``"gpt-"``) or a predicate; *factory*
    receives the resolved model id and returns the counter. Selection prefers the
    longest matching prefix, then registration order. Offline default behavior is
    unchanged until a counter is registered.

    Pass a stable *key* to make the registration idempotent: re-registering the
    same key replaces the prior entry rather than appending a duplicate, so a
    provider that registers its counter every time it is built (see
    :func:`vincio.providers.register_provider_token_counters`) stays a single
    entry.
    """
    if key is not None:
        _REGISTERED[:] = [entry for entry in _REGISTERED if entry[3] != key]
    if isinstance(matcher, str):
        prefix = matcher
        specificity = len(prefix)

        def _match(model: str, _p: str = prefix) -> bool:
            return model.startswith(_p)

        _REGISTERED.append((_match, specificity, factory, key))
    else:
        _REGISTERED.append((matcher, 0, factory, key))
    # Invalidate both the counter cache and the per-text memo, so a counter
    # registered after some text was already counted takes effect.
    get_token_counter.cache_clear()
    _count_cached.cache_clear()


def _registered_keys() -> set[str]:
    """The set of keys already registered, so an idempotent re-registration can be
    skipped without clearing the shared token memo (see
    :func:`vincio.providers.register_provider_token_counters`)."""
    return {entry[3] for entry in _REGISTERED if entry[3] is not None}


def _select_registered(model: str) -> Callable[[str], TokenCounter] | None:
    best: tuple[int, Callable[[str], TokenCounter]] | None = None
    for matcher, specificity, factory, _key in _REGISTERED:
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
    counter = get_token_counter(model)
    try:
        return counter.count(text)
    except Exception:
        # A registered provider-native counter may fail at *count* time, not just
        # at build time — an in-process GGUF tokenizer loads the model lazily on
        # first count and raises if it cannot. Token counting is a hot, total path
        # (budgeting, scoring); a failure must never break the run, so fall back to
        # the offline heuristic, observably. Memoized like any other result.
        note_suppressed("tokens.counter_count_failed")
        return HeuristicTokenCounter().count(text)


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
        try:
            counts = batch(texts)
        except Exception:
            # Same best-effort contract as the per-text path: a native batch count
            # that fails falls back to the guarded per-text path, never breaking.
            note_suppressed("tokens.counter_count_many_failed")
        else:
            return [max(0, int(n)) if text else 0 for text, n in zip(texts, counts, strict=True)]
    return [_count_cached(text, model) if text else 0 for text in texts]
