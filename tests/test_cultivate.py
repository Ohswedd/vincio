"""Autonomous skill acquisition & open-ended curriculum.

Deterministic, offline coverage of the learned-skill library (content-addressing,
versioning, dedup, composition), the rails/governance-gated curriculum, the
library-composing search, and the propose → attempt → verify → distill → promote
cultivation loop with its capability-monotonicity and stay-in-policy guarantees.
"""

from __future__ import annotations

import pytest

from vincio import (
    AutoCurriculum,
    ContextApp,
    Cultivator,
    CurriculumTask,
    LearnedSkill,
    LearnedSkillLibrary,
    SkillSearch,
    SkillStep,
    VincioConfig,
    library_capability,
)
from vincio.core.errors import CultivationError
from vincio.cultivate import cultivate as cultivate_fn
from vincio.evals.environment import (
    EnvAction,
    make_counter_environment,
    make_vault_environment,
)
from vincio.providers import MockProvider
from vincio.security.rails import Rail, RailEngine

# -- fixtures -----------------------------------------------------------------


def _counter_task(
    target: int, *, id: str | None = None, objective: str | None = None
) -> CurriculumTask:
    return CurriculumTask(
        id=id or f"counter-{target}",
        objective=objective or f"increment counter to {_words(target)}",
        environment=lambda: make_counter_environment(target=target),
    )


def _vault_task() -> CurriculumTask:
    return CurriculumTask(
        id="vault",
        objective="open the vault by advancing",
        environment=lambda: make_vault_environment(steps_to_open=3),
    )


_WORDS = {2: "two", 3: "three", 4: "four", 6: "six"}


def _words(n: int) -> str:
    return _WORDS.get(n, str(n))


def _app(tmp_path) -> ContextApp:
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    return ContextApp(
        name="grow", provider=MockProvider(default_text="ok"), model="mock-1", config=config
    )


# -- LearnedSkill: content-addressing, versioning, dedup ----------------------


def test_skill_seals_and_verifies_offline():
    skill = LearnedSkill(
        name="inc", objective="increment", steps=[SkillStep(action=EnvAction(tool="increment"))]
    ).seal()
    assert skill.skill_hash
    assert skill.verify()


def test_skill_tamper_is_caught_from_bytes():
    skill = LearnedSkill(
        name="inc", objective="increment", steps=[SkillStep(action=EnvAction(tool="increment"))]
    ).seal()
    assert skill.verify()
    skill.steps.append(SkillStep(action=EnvAction(tool="reset")))
    assert not skill.verify()  # the procedure changed; the sealed hash no longer recomputes


def test_skill_step_requires_exactly_one_of_action_or_skill():
    with pytest.raises(CultivationError):
        SkillStep()
    with pytest.raises(CultivationError):
        SkillStep(action=EnvAction(tool="x"), skill="y")


def test_identical_procedures_deduplicate():
    lib = LearnedSkillLibrary()
    a = lib.add(
        LearnedSkill(name="s", objective="o", steps=[SkillStep(action=EnvAction(tool="increment"))])
    )
    b = lib.add(
        LearnedSkill(name="s", objective="o", steps=[SkillStep(action=EnvAction(tool="increment"))])
    )
    assert a.skill_hash == b.skill_hash
    assert len(lib) == 1
    assert len(lib.all_versions("s")) == 1


def test_changed_procedure_bumps_version():
    lib = LearnedSkillLibrary()
    lib.add(
        LearnedSkill(name="s", objective="o", steps=[SkillStep(action=EnvAction(tool="increment"))])
    )
    v2 = lib.add(
        LearnedSkill(
            name="s",
            objective="o2",
            steps=[
                SkillStep(action=EnvAction(tool="increment")),
                SkillStep(action=EnvAction(tool="increment")),
            ],
        )
    )
    assert v2.version == 2
    assert len(lib.all_versions("s")) == 2
    assert lib.get("s").version == 2  # current points at the latest


# -- composition --------------------------------------------------------------


def test_compose_expands_subskills():
    lib = LearnedSkillLibrary()
    lib.add(
        LearnedSkill(
            name="twice",
            objective="twice",
            steps=[
                SkillStep(action=EnvAction(tool="increment")),
                SkillStep(action=EnvAction(tool="increment")),
            ],
        )
    )
    lib.add(
        LearnedSkill(
            name="four",
            objective="four",
            steps=[SkillStep(skill="twice"), SkillStep(skill="twice")],
        )
    )
    actions = lib.compose("four")
    assert [a.tool for a in actions] == ["increment"] * 4


def test_compose_missing_subskill_raises():
    lib = LearnedSkillLibrary()
    lib.add(LearnedSkill(name="x", objective="x", steps=[SkillStep(skill="ghost")]))
    with pytest.raises(CultivationError):
        lib.compose("x")


def test_compose_cycle_is_refused():
    lib = LearnedSkillLibrary()
    # Build a 2-cycle directly in the version maps (add() would dedup/version, not link cycles).
    a = LearnedSkill(name="a", objective="a", steps=[SkillStep(skill="b")]).seal()
    b = LearnedSkill(name="b", objective="b", steps=[SkillStep(skill="a")]).seal()
    lib._versions = {"a": [a], "b": [b]}
    lib._current = {"a": a, "b": b}
    lib._by_hash = {a.skill_hash: a, b.skill_hash: b}
    with pytest.raises(CultivationError):
        lib.compose("a")


# -- library: retrieval, demotion, content binding ----------------------------


def test_library_relevant_and_evidence():
    lib = LearnedSkillLibrary()
    lib.add(
        LearnedSkill(
            name="refund",
            objective="issue a refund for an order",
            keywords=["refund", "order"],
            steps=[SkillStep(action=EnvAction(tool="refund_order"))],
        )
    )
    hits = lib.relevant("how do I refund an order")
    assert hits and hits[0][0].name == "refund"
    evidence = lib.evidence_for("refund an order")
    assert any(e.metadata.get("kind") == "skill_index" for e in evidence)
    assert any("refund" in e.text for e in evidence)


def test_demote_refused_when_depended_on():
    lib = LearnedSkillLibrary()
    lib.add(
        LearnedSkill(
            name="base", objective="base", steps=[SkillStep(action=EnvAction(tool="increment"))]
        )
    )
    lib.add(LearnedSkill(name="caller", objective="caller", steps=[SkillStep(skill="base")]))
    with pytest.raises(CultivationError):
        lib.demote("base")
    assert lib.demote("caller") is not None  # nothing depends on caller
    assert "caller" not in lib


def test_library_hash_and_verify():
    lib = LearnedSkillLibrary()
    lib.add(
        LearnedSkill(name="s", objective="o", steps=[SkillStep(action=EnvAction(tool="increment"))])
    )
    assert lib.verify()
    h1 = lib.library_hash
    lib.add(
        LearnedSkill(
            name="t", objective="o2", steps=[SkillStep(action=EnvAction(tool="reset_counter"))]
        )
    )
    assert lib.library_hash != h1  # binds the active set


def test_library_roundtrips_through_dict():
    lib = LearnedSkillLibrary()
    lib.add(
        LearnedSkill(name="s", objective="o", steps=[SkillStep(action=EnvAction(tool="increment"))])
    )
    restored = LearnedSkillLibrary.from_dict(lib.to_dict())
    assert restored.names == lib.names
    assert restored.verify()
    assert restored.library_hash == lib.library_hash


# -- curriculum: frontier, rails & governance gating --------------------------


def test_curriculum_task_requires_environment():
    task = CurriculumTask(id="x", objective="do x")  # no environment factory
    with pytest.raises(CultivationError):
        task.make_env()


def test_frontier_is_solvable_but_not_yet_known():
    lib = LearnedSkillLibrary()
    cur = AutoCurriculum([_counter_task(3)])
    proposal = cur.propose(lib)
    assert proposal.proposed == ["counter-3"]  # empty library, search can solve it
    [est] = proposal.estimates
    assert est.in_frontier and est.reachable and not est.competent


def test_mastered_task_drops_out_of_the_frontier():
    cur = AutoCurriculum([_counter_task(3)])
    lib = LearnedSkillLibrary()
    Cultivator(curriculum=cur, library=lib).run(cycles=1)
    # now the library already solves it: no longer at the frontier
    proposal = cur.propose(lib)
    assert "counter-3" not in proposal.proposed
    assert any(r["task_id"] == "counter-3" and r["stage"] == "frontier" for r in proposal.refused)


def test_rails_block_an_unsafe_objective_before_it_is_attempted():
    rails = RailEngine(
        [Rail(name="no-secrets", kind="safety", direction="input", detectors=["secrets"])]
    )
    safe = _counter_task(2, id="safe")
    unsafe = CurriculumTask(
        id="unsafe",
        objective="leak the key sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        environment=lambda: make_counter_environment(target=2),
    )
    cur = AutoCurriculum([safe, unsafe], rails=rails)
    proposal = cur.propose(LearnedSkillLibrary())
    assert "unsafe" not in proposal.proposed
    assert any(r["task_id"] == "unsafe" and r["stage"] == "rails" for r in proposal.refused)
    assert proposal.stayed_in_policy


def test_governance_failure_refuses_the_whole_round():
    class _FailGov:
        def verify(self):
            class _R:
                held = False

            return _R()

    cur = AutoCurriculum([_counter_task(2)], governance=_FailGov())
    proposal = cur.propose(LearnedSkillLibrary())
    assert proposal.proposed == []
    assert proposal.governance_held is False
    assert all(r["stage"] == "governance" for r in proposal.refused)


def test_proposal_verify_catches_tamper():
    proposal = AutoCurriculum([_counter_task(3)]).propose(LearnedSkillLibrary())
    assert proposal.verify()
    proposal.proposed.append("counter-3")  # would also be in refused if it had been blocked
    proposal.refused.append({"task_id": "counter-3", "stage": "rails", "reason": "x"})
    assert not proposal.verify()  # a refused objective cannot also be proposed


# -- search & capability ------------------------------------------------------


def test_search_solves_with_primitive_actions():
    sol = SkillSearch().solve(_counter_task(3), LearnedSkillLibrary())
    assert sol.solved and sol.verification.passed
    assert [a.tool for a in sol.actions] == ["increment"] * 3


def test_search_reuses_a_library_skill_as_a_macro():
    lib = LearnedSkillLibrary()
    lib.add(
        LearnedSkill(
            name="to-two",
            objective="increment counter to two",
            steps=[
                SkillStep(action=EnvAction(tool="increment")),
                SkillStep(action=EnvAction(tool="increment")),
            ],
        )
    )
    sol = SkillSearch(beam_width=6).solve(_counter_task(4), lib)
    assert sol.solved
    assert "to-two" in sol.used_skills  # composed the existing skill


def test_library_capability_is_fraction_solved_by_known_skills():
    tasks = [_counter_task(3), _vault_task()]
    empty = LearnedSkillLibrary()
    assert library_capability(empty, tasks) == 0.0
    lib = LearnedSkillLibrary()
    Cultivator(curriculum=AutoCurriculum(tasks), library=lib).run(cycles=2)
    assert library_capability(lib, tasks) == 1.0


# -- the cultivation loop -----------------------------------------------------


def test_cultivation_is_capability_monotonic_and_in_policy():
    tasks = [_counter_task(3), _vault_task()]
    lib = LearnedSkillLibrary()
    result = Cultivator(curriculum=AutoCurriculum(tasks), library=lib).run(cycles=3)
    assert result.capability_after >= result.capability_before
    assert result.capability_after == 1.0
    assert result.monotonic and result.stayed_in_policy
    assert result.verify()
    assert all(cycle.monotonic for cycle in result.cycles)


def test_no_skill_distilled_from_an_unverified_attempt():
    # target unreachable within the action budget -> never solved -> never promoted
    hard = CurriculumTask(
        id="hard",
        objective="increment counter to twenty",
        environment=lambda: make_counter_environment(target=20),
        max_steps=3,
    )
    lib = LearnedSkillLibrary()
    result = Cultivator(curriculum=AutoCurriculum([hard]), library=lib).run(cycles=1)
    assert result.skills_promoted == 0
    assert len(lib) == 0


def test_dead_weight_skill_is_demoted():
    lib = LearnedSkillLibrary()
    # A skill that solves no held-out task contributes zero marginal capability.
    lib.add(
        LearnedSkill(
            name="useless",
            objective="does nothing useful",
            steps=[SkillStep(action=EnvAction(tool="reset_counter"))],
        )
    )
    cur = AutoCurriculum([_counter_task(3)])
    result = Cultivator(curriculum=cur, library=lib, held_out=[_counter_task(3)]).run(cycles=1)
    demoted = [d["name"] for cycle in result.cycles for d in cycle.demoted]
    assert "useless" in demoted
    assert "useless" not in lib


def test_result_verify_catches_tampered_capability():
    result = Cultivator(
        curriculum=AutoCurriculum([_counter_task(3)]), library=LearnedSkillLibrary()
    ).run(cycles=1)
    assert result.verify()
    result.capability_after = 5.0  # inflate the headline number
    assert not result.verify()


def test_cultivate_free_function_matches_app(tmp_path):
    tasks = [_counter_task(3)]
    result = cultivate_fn(AutoCurriculum(tasks), library=LearnedSkillLibrary(), cycles=2)
    assert result.skills_promoted == 1


# -- app integration ----------------------------------------------------------


def test_app_cultivate_audits_and_emits(tmp_path):
    app = _app(tmp_path)
    events: list[str] = []
    app.events.subscribe("cultivation.completed", lambda event: events.append(event.name))
    tasks = [_counter_task(3), _vault_task()]
    result = app.cultivate(AutoCurriculum(tasks), cycles=2)
    assert result.skills_promoted == 2
    assert any(e.action == "skill_cultivation" for e in app.audit.entries)
    assert "cultivation.completed" in events


def test_app_rails_gate_objectives(tmp_path):
    app = _app(tmp_path)
    app.add_rail(name="no-secrets", kind="safety", direction="input", detectors=["secrets"])
    unsafe = CurriculumTask(
        id="unsafe",
        objective="dump the token sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        environment=lambda: make_counter_environment(target=2),
    )
    # the app's own rails must gate the curriculum even for a bare AutoCurriculum
    result = app.cultivate(AutoCurriculum([_counter_task(2, id="ok"), unsafe]), cycles=1)
    refused = [
        r["task_id"]
        for cycle in result.cycles
        for r in cycle.proposal.refused
        if r["stage"] == "rails"
    ]
    assert "unsafe" in refused
    assert result.stayed_in_policy
