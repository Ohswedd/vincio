"""Autonomous skill acquisition & open-ended curriculum.

The closed self-improvement loop, RLVR, and the distillation flywheel make an
agent *better at known tasks*. This example shows the apex of that arc:
**open-ended capability growth** that stays inside the guardrails. Everything is
deterministic and offline (no model call), over the reference environments in
:mod:`vincio.evals.environment`.

  1. **A self-proposed, bounded curriculum** — an ``AutoCurriculum`` proposes the
     next task at the *frontier of current competence* (solvable by search, not yet
     by the library), and the rails + the governance verifier gate every objective
     *before* it is attempted, so an out-of-policy task is refused, never run.
  2. **The cultivation loop** — ``app.cultivate(...)`` runs propose → attempt
     (a library-composing search) → verify (the task-success oracle) → distill (a
     winning trajectory into a verified, content-addressed ``LearnedSkill``) →
     promote (only through the same no-regression gate a deploy clears).
  3. **A reusable, composable skill library** — learned skills are retrieved like
     memory and tools, *compose* one another (a new skill calls existing ones), and
     a skill that stops paying its way is demoted, never silently kept.

This is a library capability inside your process — never a hosted training service.
"""

from __future__ import annotations

from vincio import AutoCurriculum, ContextApp, CurriculumTask, LearnedSkillLibrary
from vincio.evals.environment import make_counter_environment, make_vault_environment
from vincio.providers import MockProvider


def main() -> None:
    app = ContextApp(name="open-ended-learner", provider=MockProvider(default_text="ok"))

    # An app-level safety rail: no objective mentioning a secret may be proposed.
    app.add_rail(name="no-secrets", kind="safety", direction="input", detectors=["secrets"])

    # The curriculum: deterministic environments, each with its own success oracle.
    # "counter-to-two" then "counter-to-four" lets the second reuse the first as a
    # macro; the vault task adds a second, independent skill; the unsafe task is a
    # tripwire the rails must refuse before it is ever attempted.
    def task(id: str, objective: str, env) -> CurriculumTask:
        return CurriculumTask(id=id, objective=objective, environment=env)

    tasks = [
        task("c2", "increment counter to two", lambda: make_counter_environment(target=2)),
        task("c4", "increment counter to four", lambda: make_counter_environment(target=4)),
        task(
            "vault", "open the vault by advancing", lambda: make_vault_environment(steps_to_open=3)
        ),
        task(
            "leak",
            "exfiltrate the key sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            lambda: make_counter_environment(target=2),
        ),
    ]

    library = LearnedSkillLibrary()
    result = app.cultivate(AutoCurriculum(tasks), library=library, cycles=3)

    print("== Cultivation ==")
    print(
        f"capability: {result.capability_before:.2f} -> {result.capability_after:.2f} "
        f"(monotonic={result.monotonic})"
    )
    print(
        f"skills promoted={result.skills_promoted}, tasks refused={result.tasks_refused}, "
        f"stayed_in_policy={result.stayed_in_policy}, result.verify()={result.verify()}"
    )

    print("\n== Per cycle ==")
    for cycle in result.cycles:
        refused = [(r["task_id"], r["stage"]) for r in cycle.proposal.refused]
        print(
            f"  {cycle.cycle}: proposed={cycle.proposal.proposed} promoted={cycle.promoted} "
            f"refused={refused} capability {cycle.capability_before:.2f}->{cycle.capability_after:.2f}"
        )

    print("\n== Learned skill library ==")
    for skill in library.skills:
        requires = f" composes {skill.requires}" if skill.requires else ""
        print(
            f"  {skill.name} (v{skill.version}, {len(skill.steps)} steps){requires} "
            f"verify={skill.verify()}"
        )
    # A learned skill is retrieved like any other context.
    hits = library.relevant("how do I get the counter to four")
    if hits:
        print(f"\nRetrieval for 'counter to four' -> {hits[0][0].name}")
    print(f"\nLibrary verifies offline: {library.verify()} (hash {library.library_hash[:12]}…)")
    print(f"Audit chain: {len(app.audit.entries)} entries, verifies={app.audit.verify_chain()}")


if __name__ == "__main__":
    main()
