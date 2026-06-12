"""Prompt cache layout tuning.

Analyzes compiled prompts for cacheability and produces actionable layout
advice: which content to move into the stable prefix, which volatile
content breaks the cache, and the expected cached-token gain.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from ..core.tokens import count_tokens
from ..prompts.ast import PromptAST
from ..prompts.compiler import CompiledPrompt

__all__ = ["CacheAdvice", "CacheTuningReport", "analyze_prompt_cacheability", "analyze_ast_layout"]

_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|\b(?:today|now|current time)\b", re.IGNORECASE
)
_RANDOM_ID_RE = re.compile(r"\b(?:run|req|trace|ctx|session)_[a-f0-9]{8,}\b")


class CacheAdvice(BaseModel):
    code: str
    message: str
    estimated_token_gain: int = 0


class CacheTuningReport(BaseModel):
    cacheability: float
    stable_prefix_tokens: int
    total_tokens: int
    advice: list[CacheAdvice] = Field(default_factory=list)

    @property
    def potential_cacheability(self) -> float:
        gain = sum(a.estimated_token_gain for a in self.advice)
        if self.total_tokens == 0:
            return 0.0
        return min(1.0, (self.stable_prefix_tokens + gain) / self.total_tokens)


def analyze_prompt_cacheability(compiled: CompiledPrompt) -> CacheTuningReport:
    """Audit a compiled prompt against cache-hygiene rules."""
    advice: list[CacheAdvice] = []
    system_text = compiled.system_text

    # Rule: avoid timestamps/random IDs in the prefix — they kill the cache.
    if _TIMESTAMP_RE.search(system_text):
        advice.append(
            CacheAdvice(
                code="CACHE001",
                message="timestamp-like content in the stable prefix invalidates the cache every call; move it to the user message",
                estimated_token_gain=compiled.stable_prefix_tokens,
            )
        )
    if _RANDOM_ID_RE.search(system_text):
        advice.append(
            CacheAdvice(
                code="CACHE002",
                message="random/run identifiers in the stable prefix invalidate the cache; move them to the suffix",
                estimated_token_gain=compiled.stable_prefix_tokens,
            )
        )

    # Rule: large user/context block with small prefix → candidates for prefix
    # promotion (examples and schemas are stable per app version).
    suffix_tokens = compiled.token_count - compiled.stable_prefix_tokens
    if compiled.cacheability < 0.5 and suffix_tokens > 400:
        advice.append(
            CacheAdvice(
                code="CACHE003",
                message=(
                    "less than half the prompt is cacheable; move static content "
                    "(examples, schema, rules) into the system prefix"
                ),
                estimated_token_gain=int(suffix_tokens * 0.3),
            )
        )
    if not any(m.cache_hint for m in compiled.messages):
        advice.append(
            CacheAdvice(
                code="CACHE004",
                message="no cache_hint set on the stable prefix; providers with explicit cache control won't cache it",
                estimated_token_gain=compiled.stable_prefix_tokens,
            )
        )
    return CacheTuningReport(
        cacheability=compiled.cacheability,
        stable_prefix_tokens=compiled.stable_prefix_tokens,
        total_tokens=compiled.token_count,
        advice=advice,
    )


def analyze_ast_layout(ast: PromptAST) -> list[CacheAdvice]:
    """Detect stable nodes ordered after volatile ones (cache breakers)."""
    advice: list[CacheAdvice] = []
    seen_volatile = False
    for node in ast.ordered():
        if not node.stable:
            seen_volatile = True
        elif seen_volatile:
            tokens = count_tokens(node.text)
            advice.append(
                CacheAdvice(
                    code="CACHE005",
                    message=f"stable node {node.kind!r} appears after dynamic content; reorder to extend the cacheable prefix",
                    estimated_token_gain=tokens,
                )
            )
    return advice


def cache_hit_economics(
    *,
    stable_tokens: int,
    total_tokens: int,
    calls_per_day: int,
    input_cost_per_mtok: float,
    cached_cost_per_mtok: float,
) -> dict[str, Any]:
    """Estimate daily savings from prompt caching for capacity planning."""
    savings_per_call = stable_tokens * (input_cost_per_mtok - cached_cost_per_mtok) / 1_000_000
    return {
        "cacheability": round(stable_tokens / total_tokens, 4) if total_tokens else 0.0,
        "savings_per_call_usd": round(max(0.0, savings_per_call), 8),
        "savings_per_day_usd": round(max(0.0, savings_per_call) * calls_per_day, 6),
    }
