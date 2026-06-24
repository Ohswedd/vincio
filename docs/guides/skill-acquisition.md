# Autonomous skill acquisition & open-ended curriculum

Vincio's closed self-improvement loop (trace → dataset → eval → optimize → promote),
RLVR, and the distillation flywheel all make an agent *better at known tasks*. This
is the apex of that arc: **open-ended capability growth**. An agent proposes its own
tasks at the edge of its competence, distills successful trajectories into a
reusable, versioned skill library, and bootstraps — Voyager / ADAS-shaped — under the
*same* no-regression gate a promotion already clears, so growth is **safe and
reversible** rather than unbounded drift.

Everything here is opt-in, additive, deterministic, and offline. It composes the
existing eval harness, optimizer gates, rails, and governance verifier — it does not
introduce a new runtime, a hosted trainer, or a network dependency.

> **Two libraries, one word.** `vincio.cultivate.LearnedSkillLibrary` holds skills the
> agent **learned itself** (distilled from trajectories, content-addressed). It is
> distinct from `vincio.skills.SkillLibrary`, which holds human-authored `SKILL.md`
> procedural knowledge. A learned skill projects to a `Skill`
> (`LearnedSkill.to_skill()`) so both are retrieved and cited through the same
> progressive-disclosure path.

## The cultivation loop

`app.cultivate(curriculum)` runs **propose → attempt → verify → distill → promote**
across cycles, so capability compounds across runs:

```python
from vincio import AutoCurriculum, ContextApp, CurriculumTask, LearnedSkillLibrary
from vincio.evals.environment import make_counter_environment, make_vault_environment
from vincio.providers import MockProvider

app = ContextApp(name="learner", provider=MockProvider(default_text="ok"))

tasks = [
    CurriculumTask(id="c2", objective="increment counter to two",
                   environment=lambda: make_counter_environment(target=2)),
    CurriculumTask(id="c4", objective="increment counter to four",
                   environment=lambda: make_counter_environment(target=4)),
    CurriculumTask(id="vault", objective="open the vault by advancing",
                   environment=lambda: make_vault_environment(steps_to_open=3)),
]

library = LearnedSkillLibrary()
result = app.cultivate(AutoCurriculum(tasks), library=library, cycles=3)

assert result.capability_after >= result.capability_before   # monotone
assert result.stayed_in_policy                               # nothing out-of-policy ran
assert result.verify()                                       # re-derived from the bytes
```

Each cycle:

1. **Propose** — the `AutoCurriculum` ranks the tasks at the *frontier of current
   competence*: solvable by a bounded search, but not yet by retrieving an existing
   skill. A task already mastered, or one beyond the search bound, falls out.
2. **Attempt** — a deterministic, library-composing `SkillSearch` looks for the
   shortest procedure that satisfies the task. Its moves are the task's primitive
   actions **plus one macro per existing skill**, so a solution can *call skills the
   agent already has*.
3. **Verify** — the winning procedure must pass the task-success **oracle**
   (`env.verify()`). No skill is distilled from an unverified attempt.
4. **Distill** — the winning trajectory becomes a `LearnedSkill`: an objective, a
   precondition, the ordered `steps` (primitive actions and sub-skill calls), the
   verifier it satisfies, and provenance (the cycle, the reward, the trajectory hash).
5. **Promote** — the skill is added **only if it clears the no-regression gate**:
   capability on a held-out frontier set must not fall (the same `no_regression_gate`
   a prompt or policy promotion uses). A skill that stops paying its way is demoted.

## A reusable, composable skill library

A `LearnedSkill` is **verified, content-addressed, versioned, and composable**:

```python
skill = library.get("skill-increment-counter-to-four")
skill.verify()            # recompute the content hash from the bytes — tamper-evident
skill.requires            # ['skill-increment-counter-to-two'] — it composes another
library.compose(skill.name)   # the flattened primitive actions
library.relevant("get the counter to four")   # retrieved like memory and tools
```

* **Content-addressed dedup.** A skill's identity is the hash of its procedure
  (objective, precondition, steps, verifier). Adding a byte-for-byte duplicate is a
  no-op; adding a *changed* procedure under an existing name bumps its `version`.
* **Composition.** A `SkillStep` is either a primitive `EnvAction` **or** an
  invocation of another skill, expanded recursively by `compose()` (a cycle or a
  missing sub-skill is refused, never executed).
* **Demotion.** `_prune` removes any active skill whose removal does not lower
  capability — dead weight is demoted, never silently kept. A skill another skill
  depends on is load-bearing and is kept.
* **Offline-verifiable.** `library.verify()` recomputes every active skill's hash,
  and `library.library_hash` binds the active set; `to_dict()` / `from_dict()`
  round-trips it through a content-addressed store.

## Autonomy that stays inside the guardrails

`AutoCurriculum` gates **every proposed objective before it is ever attempted**:

* **Rails.** The objective's instruction is screened by the programmable
  `RailEngine` (`app.cultivate` wires the app's own rails by default). A blocked
  objective is *pinpointed and refused* — it never reaches the attempt step.
* **Governance verifier.** The `GovernanceVerifier` must prove the app's controls
  (containment, residency, budget, erasure) still hold for the round; if they do
  not, the whole round is refused. A failing verifier *fails closed*.

```python
app.add_rail(name="no-secrets", kind="safety", direction="input", detectors=["secrets"])
result = app.cultivate(AutoCurriculum(tasks_including_an_unsafe_one))
# the unsafe objective appears in cycle.proposal.refused with stage="rails",
# never in cycle.attempted — result.stayed_in_policy is True.
```

The `CurriculumProposal` is content-bound: `proposal.verify()` recomputes its hash
and checks that **no refused objective was slipped into the proposed set**, so the
stay-in-policy property is reconstructable from the bytes — the autonomous-growth
analogue of the shield's prevention-by-construction.

## What it is held to

A `skill_acquisition` VincioBench family gates two SLOs, offline against the
deterministic reference environments:

| SLO | What it proves |
|---|---|
| **Capability monotonicity** | A full propose → attempt → verify → distill → promote run ends *at least as capable* as it began; promotion reuses the gated no-regression check, dead weight is demoted, and a tampered capability number is caught. |
| **Stay-in-policy safety** | An objective a rail blocks — or any objective when the governance invariants do not hold — is refused and never attempted; the proposal's hash catches a refused objective relabelled as proposed. |

It is **a library capability inside your process** — never a hosted training service,
a managed curriculum, or a network dependency. See `examples/92_skill_acquisition.py`
for a runnable end-to-end walk-through.
