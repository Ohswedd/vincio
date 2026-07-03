"""The browsing skill: when to search, what to search, when to stop.

Giving a model a search tool is the easy half; the hard half is judgement —
knowing when the web helps, writing queries that find the fact, reading only
what the question needs, and stopping. That judgement ships here as a
first-class :class:`~vincio.skills.Skill` (the Agent Skills ``SKILL.md``
shape), not as prompt text bolted onto a template:

* its **summary line** joins the always-disclosed skill index, so every run
  knows web access exists and what it is for;
* its **full instructions** surface through the skill library's progressive
  disclosure only when the task looks web-relevant, scored and budgeted by the
  context compiler like any other evidence — zero standing token cost on tasks
  that will never search.

Because the same :class:`~vincio.skills.Skill` object serves every provider,
the when/what/how contract is identical for a hosted frontier model and a
local GGUF one — the first step of teaching skills to the model *through the
context plane* rather than through provider-specific system prompts.
"""

from __future__ import annotations

from ..skills.skill import Skill

__all__ = ["browse_skill"]

_INSTRUCTIONS = """\
You can call `web_search(query)` to search the open web and `web_read(url, query)`
to read the passages of a page relevant to a query. Use them with judgement:

## When to search
- The fact can change over time: prices, versions, releases, schedules, scores,
  office-holders, statistics, anything dated after your training data.
- The user asks about current events, or explicitly asks you to look something up.
- You need an exact quote, figure, or citation you cannot reproduce with
  confidence, or the question names an entity you barely know.

## When NOT to search
- Stable, well-known knowledge; definitions; mathematics or pure reasoning.
- Anything already answered by the provided context, documents, or memory.
- Private or organization-internal questions the open web cannot answer.

## How to search
- Write 2-5 significant keywords, not a full sentence: prefer
  `python 3.13 release date` over `when was python 3.13 released?`.
- Make one precise search before considering a broader one; add or swap one
  term to refine, never repeat a failed query verbatim.
- Read only the most promising 1-2 results with `web_read`, passing the fact
  you need as the query — it returns only the relevant passages, not the page.

## How to finish
- Stop as soon as the fact is grounded; do not keep searching for confirmation
  you already have. If two sources disagree, read one more and say so.
- Cite the URL of every page whose content you used in your answer.
- If the web does not answer the question, say what you searched and what you
  found instead of guessing.
"""


def browse_skill() -> Skill:
    """The built-in web-browsing skill, ready for ``app.skill_library``."""
    return Skill(
        name="web-search",
        description=(
            "Search the open web (web_search) and read the relevant passages of a "
            "page (web_read) for facts that are recent, volatile, niche, or need "
            "an exact citation."
        ),
        instructions=_INSTRUCTIONS,
        keywords=[
            "search", "web", "internet", "browse", "online", "lookup",
            "current", "latest", "today", "recent", "news", "price",
            "version", "release", "update", "weather", "score", "stock",
        ],
        metadata={"builtin": True, "plane": "web"},
    )
