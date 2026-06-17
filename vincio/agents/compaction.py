"""In-loop context compaction for the agent executor (1.10).

Long agent runs accumulate tool observations and reasoning turns until the
context overflows. The :class:`ContextCompactor` replaces the executor's old
fixed ``[-8]`` / ``[:24]`` slicing with a *token-budgeted* policy: the most
recent turns are kept verbatim, and everything older is folded into a rolling
**extractive summary** (deterministic and offline via
:func:`~vincio.memory.summarizers.extractive_summary`). Under budget it keeps
everything, so short runs are unchanged; over budget it prunes the oldest
observations and preserves their gist, so the loop never blows the window.
"""

from __future__ import annotations

from ..core.tokens import count_tokens
from ..core.types import Message
from ..memory.summarizers import extractive_summary

__all__ = ["ContextCompactor"]


class ContextCompactor:
    """Token-budgeted compaction of context blocks and ReAct message logs."""

    def __init__(self, *, max_tokens: int = 6000, keep_recent: int = 6, summary_tokens: int = 200) -> None:
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.summary_tokens = summary_tokens

    def compact_blocks(self, blocks: list[str], *, budget: int | None = None) -> tuple[str | None, list[str]]:
        """Keep the most recent blocks within ``budget`` tokens; summarize the rest.

        Returns ``(rolling_summary_or_None, kept_blocks)`` in original order. When
        the blocks already fit, the summary is ``None`` and all blocks are kept.
        """
        if not blocks:
            return None, []
        budget = budget if budget is not None else self.max_tokens
        kept_reversed: list[str] = []
        tokens = 0
        cut = 0  # number of oldest blocks folded into the summary
        for i in range(len(blocks) - 1, -1, -1):
            block = blocks[i]
            cost = count_tokens(block)
            keep_for_recency = len(blocks) - i <= self.keep_recent
            if kept_reversed and tokens + cost > budget and not keep_for_recency:
                cut = i + 1
                break
            kept_reversed.append(block)
            tokens += cost
        kept = list(reversed(kept_reversed))
        if cut == 0:
            return None, kept
        older = "\n".join(blocks[:cut])
        summary = extractive_summary(older, max_tokens=self.summary_tokens)
        return (summary or None), kept

    def compact_messages(
        self, messages: list[Message], *, budget: int | None = None
    ) -> list[Message]:
        """Compact a ReAct message log: keep the system message, the first user
        turn, and the most recent turns; fold the middle into one summary turn."""
        budget = budget if budget is not None else self.max_tokens
        total = sum(count_tokens(m.text) for m in messages)
        if total <= budget or len(messages) <= self.keep_recent + 2:
            return messages
        head: list[Message] = []
        rest = list(messages)
        # Preserve the leading system + first user message as the anchor.
        while rest and rest[0].role in ("system", "user") and len(head) < 2:
            head.append(rest.pop(0))
        keep = self.keep_recent if self.keep_recent else 0
        # Never start the recent window on a lone tool result — that would orphan
        # it from the assistant tool-call turn that providers require to precede it.
        while keep < len(rest) and rest[len(rest) - keep].role == "tool":
            keep += 1
        recent = rest[-keep:] if keep else []
        middle = rest[: len(rest) - len(recent)]
        if not middle:
            return messages
        folded = extractive_summary(
            "\n".join(f"{m.role}: {m.text}" for m in middle), max_tokens=self.summary_tokens
        )
        summary_msg = Message(
            role="user",
            content=f"[Earlier steps compacted]\n{folded}" if folded else "[Earlier steps compacted]",
        )
        return [*head, summary_msg, *recent]
