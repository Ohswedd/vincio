"""Provider-aware prompt-cache strategy.

The compiler already lays content out stable-prefix-first and marks the stable
system block with ``cache_hint`` (its cache-aware layout). This module turns
that hint into a *provider-aware* caching decision at request time:

- For providers with explicit breakpoints (**Anthropic** ``cache_control``):
  attach the chosen **TTL** (5-minute or 1-hour) to the stable prefix when it is
  long enough to be worth caching. Because Anthropic caches in
  tools → system → messages order, one breakpoint on the system block caches the
  tools and system together.
- For providers with **automatic** prefix caching (**OpenAI**, **Gemini**): the
  stable→volatile ordering the compiler already produces is what maximizes the
  auto-cache hit rate; there is nothing to mark, so the strategy is a no-op and
  hit-rate telemetry comes straight from the provider's reported
  ``cached_input_tokens``.

The pass is purely additive — it only *adds* a TTL to breakpoints the compiler
already chose; it never removes one — so enabling it cannot change which prompts
are cacheable, only how they are cached and measured.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from ..core.tokens import count_tokens
from ..core.types import Message, ModelCapabilities

__all__ = ["PromptCacheStrategy", "cache_hit_rate"]


def cache_hit_rate(input_tokens: int, cached_input_tokens: int) -> float:
    """Fraction of input tokens served from the provider prompt cache."""
    if input_tokens <= 0:
        return 0.0
    return min(1.0, cached_input_tokens / input_tokens)


class PromptCacheStrategy(BaseModel):
    """Decide and place provider cache breakpoints for a request's messages."""

    enabled: bool = True
    ttl: Literal["5m", "1h"] = "5m"
    # Anthropic does not cache prefixes below ~1024 tokens; below this the
    # cache-write is pure overhead, so we leave the TTL off (the breakpoint the
    # compiler set is harmless and ignored by the provider).
    min_prefix_tokens: int = 1024

    def apply(
        self,
        messages: list[Message],
        *,
        capabilities: ModelCapabilities,
        model: str | None = None,
    ) -> tuple[list[Message], dict[str, Any]]:
        """Return ``(messages, info)`` with the cache TTL applied where useful.

        ``info`` carries telemetry: whether a breakpoint was set, the
        prefix-token count, and the TTL — surfaced on the model-call span.
        """
        info: dict[str, Any] = {
            "applied": False,
            "supported": bool(capabilities.prompt_caching),
            "ttl": self.ttl,
            "prefix_tokens": 0,
            "breakpoints": 0,
        }
        if not self.enabled or not capabilities.prompt_caching:
            return messages, info

        # The stable prefix is the leading run of system/developer or
        # already-hinted messages, ending at the first volatile message.
        prefix_tokens = 0
        last_breakpoint = -1
        for i, message in enumerate(messages):
            if message.role in ("system", "developer") or message.cache_hint:
                prefix_tokens += count_tokens(message.text, model)
                if message.cache_hint or message.role in ("system", "developer"):
                    last_breakpoint = i
            else:
                break
        info["prefix_tokens"] = prefix_tokens
        if last_breakpoint < 0 or prefix_tokens < self.min_prefix_tokens:
            return messages, info

        out: list[Message] = []
        breakpoints = 0
        for i, message in enumerate(messages):
            if i == last_breakpoint or message.cache_hint:
                out.append(message.model_copy(update={"cache_hint": True, "cache_ttl": self.ttl}))
                breakpoints += 1
            else:
                out.append(message)
        info["applied"] = True
        info["breakpoints"] = breakpoints
        return out, info
