"""Multi-turn user simulator for conversational evaluation.

A :class:`Simulator` drives a multi-turn session against an agent from a
``Persona`` + goal: it produces the user's turns, feeds them to the agent, and
records the thread. Like :class:`~vincio.evals.synthetic.SyntheticGenerator`, it
is **LLM-backed with a deterministic template fallback** — with no provider it
runs fully offline and seed-deterministic (same seed → identical conversation),
which is what makes simulated sessions usable as CI golden cases.

The resulting :class:`SimulatedConversation` converts to an :class:`EvalCase`
whose ``context['messages']`` is the whole thread, so the conversational metrics
(``conversation_outcome``, ``intent_resolution``, ``knowledge_retention``,
``conversation_relevance``) score the outcome over the entire session.
"""

from __future__ import annotations

import random
import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from ..providers.base import run_sync
from .datasets import EvalCase

__all__ = ["Persona", "SimulatedConversation", "Simulator"]

# An agent under test: given the conversation so far (a list of role/content
# dicts), return the assistant's next reply. May be sync or async.
AgentCallable = Callable[[list[dict[str, str]]], Awaitable[str] | str]

_STOPWORDS = {
    "the", "a", "an", "to", "of", "and", "or", "for", "with", "my", "i", "is",
    "are", "do", "does", "can", "you", "me", "how", "what", "please", "need",
    "help", "want", "about", "on", "in", "it", "this", "that",
}


class Persona(BaseModel):
    """A simulated user: who they are and what they're trying to accomplish."""

    name: str = "user"
    goal: str = ""
    traits: list[str] = Field(default_factory=list)
    # Facts the user knows and may state during the conversation; used to test
    # knowledge_retention (the agent must not re-ask for a stated fact).
    facts: dict[str, str] = Field(default_factory=dict)
    max_turns: int = 4


class SimulatedConversation(BaseModel):
    """The recorded thread of a simulated multi-turn session."""

    persona: str
    goal: str
    turns: list[dict[str, str]] = Field(default_factory=list)
    goal_achieved: bool = False
    rounds: int = 0
    generator: str = "offline"  # offline | llm

    def messages(self) -> list[dict[str, str]]:
        return list(self.turns)

    def to_eval_case(
        self, *, id: str = "sim", expected: Any = None, tags: list[str] | None = None
    ) -> EvalCase:
        """Turn the thread into an EvalCase the conversational metrics can score."""
        keywords = _keywords(self.goal)
        assistant_turns = [t["content"] for t in self.turns if t["role"] == "assistant"]
        return EvalCase(
            id=id,
            input=self.turns[0]["content"] if self.turns else self.goal,
            context={"messages": self.turns, "goal": self.goal},
            expected=expected if expected is not None else (assistant_turns[-1] if assistant_turns else None),
            rubric={"goal": self.goal, "goal_keywords": keywords},
            tags=tags or ["simulated"],
            metadata={"persona": self.persona, "goal_achieved": self.goal_achieved, "rounds": self.rounds},
        )


def _keywords(text: str, *, limit: int = 6) -> list[str]:
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()) if w not in _STOPWORDS]
    seen: list[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
    return seen[:limit]


class Simulator:
    """Drive multi-turn sessions from a persona + goal (offline-deterministic)."""

    def __init__(
        self,
        *,
        provider: Any = None,
        model: str | None = None,
        seed: int = 42,
        max_turns: int = 4,
        goal_threshold: float = 0.18,
    ) -> None:
        self.provider = provider
        self.model = model
        self.seed = seed
        self.max_turns = max_turns
        self.goal_threshold = goal_threshold

    # -- public API ----------------------------------------------------------

    async def asimulate(
        self, agent: AgentCallable, persona: Persona, *, max_turns: int | None = None
    ) -> SimulatedConversation:
        turns: list[dict[str, str]] = []
        rng = random.Random(self.seed)
        limit = max_turns or persona.max_turns or self.max_turns
        keywords = _keywords(persona.goal)
        generator = "offline"
        achieved = False
        for turn_index in range(limit):
            user_text = await self._user_turn(persona, turns, turn_index, rng)
            if self.provider is not None and self.model and turn_index == 0:
                generator = "llm"
            turns.append({"role": "user", "content": user_text})
            reply = agent(list(turns))
            if hasattr(reply, "__await__"):
                reply = await reply  # type: ignore[assignment]
            reply_text = str(reply or "")
            turns.append({"role": "assistant", "content": reply_text})
            if self._goal_satisfied(persona.goal, keywords, reply_text):
                achieved = True
                break
        return SimulatedConversation(
            persona=persona.name,
            goal=persona.goal,
            turns=turns,
            goal_achieved=achieved,
            rounds=len([t for t in turns if t["role"] == "user"]),
            generator=generator,
        )

    def simulate(
        self, agent: AgentCallable, persona: Persona, *, max_turns: int | None = None
    ) -> SimulatedConversation:
        """Synchronous wrapper; accepts a sync or async ``agent``."""
        async def _async_agent(messages: list[dict[str, str]]) -> str:
            result = agent(messages)
            if hasattr(result, "__await__"):
                return str(await result)  # type: ignore[arg-type]
            return str(result)

        return run_sync(self.asimulate(_async_agent, persona, max_turns=max_turns))

    # -- user-turn generation ------------------------------------------------

    async def _user_turn(
        self, persona: Persona, turns: list[dict[str, str]], turn_index: int, rng: random.Random
    ) -> str:
        if self.provider is not None and self.model:
            text = await self._llm_user_turn(persona, turns)
            if text:
                return text
        return self._template_user_turn(persona, turns, turn_index, rng)

    async def _llm_user_turn(self, persona: Persona, turns: list[dict[str, str]]) -> str:
        from ..core.types import Message, ModelRequest

        traits = ", ".join(persona.traits) or "concise"
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns) or "(start of conversation)"
        system = (
            f"You are role-playing a user named {persona.name} ({traits}). "
            f"Your goal: {persona.goal}. Write only your next short message to the "
            "assistant — no quotes, no narration. If the goal is met, acknowledge briefly."
        )
        request = ModelRequest(
            model=self.model or "",
            messages=[
                Message(role="system", content=system),
                Message(role="user", content=f"Conversation so far:\n{transcript}\n\nYour next message:"),
            ],
            temperature=0.7,
        )
        try:
            response = await self.provider.generate(request)
        except Exception:  # noqa: BLE001 - fall back to templates on any provider error
            return ""
        return (response.text or "").strip()

    def _template_user_turn(
        self, persona: Persona, turns: list[dict[str, str]], turn_index: int, rng: random.Random
    ) -> str:
        if turn_index == 0:
            opening = persona.goal or "I need some help."
            return opening if opening.endswith(("?", ".")) else f"{opening}."
        keywords = _keywords(persona.goal)
        # Deterministically state a known fact on the second turn (tests retention).
        if turn_index == 1 and persona.facts:
            key = sorted(persona.facts)[0]
            return f"Just so you know, {key} is {persona.facts[key]}."
        follow_ups = [
            "Can you go into more detail?",
            "What should I do next?",
            "Are there any caveats I should know?",
            "Can you give a concrete example?",
        ]
        if keywords:
            focus = keywords[rng.randrange(len(keywords))]
            follow_ups.insert(0, f"Can you say more about {focus}?")
        return follow_ups[rng.randrange(len(follow_ups))]

    def _goal_satisfied(self, goal: str, keywords: list[str], reply: str) -> bool:
        if not goal:
            return False
        from ..context.scoring import lexical_similarity

        reply_lower = reply.lower()
        if keywords:
            hits = sum(1 for k in keywords if k in reply_lower)
            if hits / len(keywords) >= 0.6:
                return True
        return lexical_similarity(goal, reply) >= self.goal_threshold
