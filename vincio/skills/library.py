"""A library of loaded skills with progressive-disclosure evidence selection."""

from __future__ import annotations

from ..core.tokens import count_tokens
from ..core.types import EvidenceItem, TrustLevel
from .skill import Skill

__all__ = ["SkillLibrary"]


class SkillLibrary:
    """Holds skills and turns them into scored, budgeted evidence per task.

    Progressive disclosure has two levels:

    * **Level 1 — always disclosed.** A compact index (one line per skill) so
      the model knows which skills exist. Cheap; always in budget.
    * **Level 2 — disclosed on relevance.** A skill's full instructions are
      emitted as an evidence item only when the task matches it above
      ``threshold``; the context compiler then scores, budgets, and cites it
      like any other evidence — so an unused skill costs only its index line.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    @property
    def skills(self) -> list[Skill]:
        return list(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def index_text(self) -> str:
        lines = [s.summary_line() for s in self._skills.values()]
        return "Available skills (load the relevant one's steps):\n" + "\n".join(lines)

    def relevant(
        self, query: str, *, threshold: float = 0.05, limit: int = 3
    ) -> list[tuple[Skill, float]]:
        scored = [(s, s.match_score(query)) for s in self._skills.values()]
        hits = [(s, score) for s, score in scored if score >= threshold]
        hits.sort(key=lambda pair: pair[1], reverse=True)
        return hits[:limit]

    def evidence_for(
        self, query: str, *, threshold: float = 0.05, limit: int = 3
    ) -> list[EvidenceItem]:
        """Index item (always) + relevant skill bodies (on match)."""
        if not self._skills:
            return []
        index_text = self.index_text()
        items = [
            EvidenceItem(
                id="skill-index",
                source_id="skills",
                source_type="document",
                text=index_text,
                trust_level=TrustLevel.DEVELOPER,
                relevance=0.9,  # keep the index in budget
                authority=0.9,
                provenance=1.0,
                token_cost=count_tokens(index_text),
                metadata={"origin": "skill:index", "kind": "skill_index"},
            )
        ]
        for skill, score in self.relevant(query, threshold=threshold, limit=limit):
            body = f"# Skill: {skill.name}\n\n{skill.description}\n\n{skill.instructions}".strip()
            items.append(
                EvidenceItem(
                    id=f"skill:{skill.name}",
                    source_id=skill.name,
                    source_type="document",
                    text=body,
                    trust_level=TrustLevel.DEVELOPER,
                    # Bias relevance up by the match so a clearly-applicable
                    # skill survives budgeting; the compiler still scores it.
                    relevance=min(1.0, 0.5 + score),
                    authority=0.8,
                    provenance=1.0,
                    token_cost=count_tokens(body),
                    metadata={"origin": f"skill:{skill.name}", "kind": "skill"},
                )
            )
        return items
