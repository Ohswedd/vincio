"""Tokenizer fertility telemetry — the non-English "token tax".

The same sentence costs more tokens in some languages than others (tokenizer
"fertility" = tokens per word). For non-English, high-fertility languages that
tax can be 2–3×, and it hides inside an aggregate cost number. This tracker
surfaces fertility **per language** (and optionally per tenant), so the token
tax is visible and *routable* — you can attribute it, budget for it, or pick a
tokenizer-friendlier model for a language.

It reuses the same token counter the budgeter uses, so the numbers match what
billing sees. Fully deterministic and offline.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from ..core.tokens import count_tokens

__all__ = ["LanguageFertility", "FertilityTracker"]

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class LanguageFertility(BaseModel):
    """Aggregated tokenizer statistics for one language."""

    language: str
    samples: int = 0
    chars: int = 0
    words: int = 0
    tokens: int = 0

    @property
    def tokens_per_word(self) -> float:
        return round(self.tokens / self.words, 4) if self.words else 0.0

    @property
    def tokens_per_char(self) -> float:
        return round(self.tokens / self.chars, 4) if self.chars else 0.0


class FertilityTracker:
    """Track tokens-per-word per language to surface the non-English token tax."""

    def __init__(self, *, model: str | None = None, baseline_language: str = "en") -> None:
        self.model = model
        self.baseline_language = baseline_language
        self._by_language: dict[str, LanguageFertility] = {}
        self._by_tenant_language: dict[tuple[str, str], LanguageFertility] = {}

    @staticmethod
    def _normalize(language: str) -> str:
        return language.lower().split("-")[0] if language else "unknown"

    def record(self, text: str, *, language: str, tenant: str | None = None) -> int:
        """Record a text sample for a language; returns its token count."""
        language = self._normalize(language)
        tokens = count_tokens(text, self.model)
        words = len(_WORD_RE.findall(text))
        chars = len(text)

        agg = self._by_language.setdefault(language, LanguageFertility(language=language))
        agg.samples += 1
        agg.chars += chars
        agg.words += words
        agg.tokens += tokens

        if tenant is not None:
            key = (tenant, language)
            tagg = self._by_tenant_language.setdefault(
                key, LanguageFertility(language=language))
            tagg.samples += 1
            tagg.chars += chars
            tagg.words += words
            tagg.tokens += tokens
        return tokens

    def token_tax(self, language: str) -> float:
        """Fertility of ``language`` relative to the baseline language.

        ``1.0`` means parity with the baseline; ``2.0`` means twice as many
        tokens per character. Returns ``1.0`` when the baseline is unmeasured.
        """
        language = self._normalize(language)
        baseline = self._by_language.get(self.baseline_language)
        target = self._by_language.get(language)
        if not baseline or not target or baseline.tokens_per_char == 0:
            return 1.0
        return round(target.tokens_per_char / baseline.tokens_per_char, 4)

    def report(self) -> dict[str, object]:
        """Per-language fertility plus the token-tax multiplier vs. baseline."""
        languages = {
            lang: {
                **stats.model_dump(),
                "tokens_per_word": stats.tokens_per_word,
                "tokens_per_char": stats.tokens_per_char,
                "token_tax": self.token_tax(lang),
            }
            for lang, stats in sorted(self._by_language.items())
        }
        return {
            "baseline_language": self.baseline_language,
            "model": self.model,
            "languages": languages,
            "by_tenant": {
                f"{tenant}:{lang}": {
                    "tokens_per_word": stats.tokens_per_word,
                    "tokens_per_char": stats.tokens_per_char,
                    "tokens": stats.tokens,
                    "samples": stats.samples,
                }
                for (tenant, lang), stats in sorted(self._by_tenant_language.items())
            },
        }
