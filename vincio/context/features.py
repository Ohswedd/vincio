"""Per-compile feature arena: derive each candidate's text features once.

The dedup, conflict, and selection passes each re-derive a candidate's stemmed
content terms, word shingles, and similarity-blocking tokens from its text. On a
small pool the global ``lru_cache`` on those derivations absorbs the repeats, but
on the 10k+ pools the streaming pre-filter exercises that bounded cache thrashes —
the same text is re-tokenized pass after pass as entries evict one another, and
``_block_tokens`` is computed twice per kept candidate in the dedup index alone.

The arena derives each feature **exactly once per compile**, keyed by content, in
an unbounded per-compile dict that is discarded when the compile finishes, so the
O(n²) passes pay the derivation only once however large the pool grows — without
the bounded global cache evicting it mid-compile. The values are byte-identical to
the global derivations, so the context that is selected never changes.

Like the warm :class:`~vincio.context.arena.CandidateArena`, a feature arena is
created fresh per compile and never shared, so concurrent compiles on the same
compiler never alias one another's memo.
"""

from __future__ import annotations

from .scoring import _TOKEN_RE, _shingles, _terms

__all__ = ["FeatureArena"]


class FeatureArena:
    """Unbounded per-compile memo of a candidate's derived text features."""

    def __init__(self) -> None:
        self._terms_by_text: dict[str, frozenset[str]] = {}
        self._shingles_by_text: dict[tuple[str, int], frozenset[str]] = {}
        self._block_by_text: dict[str, frozenset[str]] = {}

    def terms(self, text: str) -> frozenset[str]:
        """Stemmed content terms of *text* (identical to ``scoring._terms``)."""
        cached = self._terms_by_text.get(text)
        if cached is None:
            cached = _terms(text)
            self._terms_by_text[text] = cached
        return cached

    def shingles(self, text: str, size: int = 3) -> frozenset[str]:
        """Word shingles of *text* (identical to ``scoring._shingles``)."""
        key = (text, size)
        cached = self._shingles_by_text.get(key)
        if cached is None:
            cached = _shingles(text, size)
            self._shingles_by_text[key] = cached
        return cached

    def block_tokens(self, text: str) -> frozenset[str]:
        """Similarity-blocking tokens of *text*: the raw lexical tokens unioned
        with the stemmed content terms. Identical membership to the compiler's
        ``_block_tokens``; reuses the memoized terms so the union is derived once.
        """
        cached = self._block_by_text.get(text)
        if cached is None:
            cached = frozenset(_TOKEN_RE.findall(text.lower())) | self.terms(text)
            self._block_by_text[text] = cached
        return cached
