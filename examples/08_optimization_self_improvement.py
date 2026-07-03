"""Optimization & the self-improving loop.

One coherent tour of how a Vincio app gets *better over time* without leaving the
process or breaking its safety discipline. The spine is the closed loop —
trace -> dataset -> eval -> optimize -> promote — and every rung above it (reflective
GEPA-style search, the distillation flywheel, on-policy reinforcement from verifiable
rewards, the declarative self-improvement policy, canary-gated deploy with rollback,
on-device LoRA adaptation, and federated cross-org improvement under a differential-
privacy budget) reuses that same gated, audited path. Every promotion must clear a
no-regression gate before it goes live, so improvement is monotonic by construction.

Runs fully offline on the deterministic mock provider — no API keys, no network.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _shared import example_provider, write_sample_docs

from vincio import (
    AdapterRegistry,
    AutoCurriculum,
    ContextApp,
    ContributionBuilder,
    CurriculumTask,
    FederatedPolicy,
    LearnedSkillLibrary,
    LocalAdaptationPolicy,
    PrivacyBudget,
    PrivacyConfig,
    PrivacyMechanism,
    SecureAggregator,
    VincioConfig,
)
from vincio.core.types import MemoryScope, MemoryType
from vincio.evals.datasets import Dataset, EvalCase
from vincio.evals.environment import (
    EnvAction,
    EnvironmentSimulator,
    build_counter_environment,
    build_retail_environment,
    build_vault_environment,
    scripted_policy,
)
from vincio.observability.sessions import record_feedback
from vincio.optimize import (
    BootstrapFinetune,
    CandidateOutcome,
    FitnessWeights,
    ImprovementLoop,
    LearningTask,
    OracleReward,
    RewardModel,
    RewardSample,
    SelfImprovementPolicy,
    export_training_set,
)
from vincio.optimize.distill import TrainingExample, TrainingSet
from vincio.retrieval.embeddings import LocalHashEmbedder


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


# A self-contained config: in-memory metadata + in-process trace exporter, so the
# whole demo is reproducible on re-run and never touches disk or network.
def _config() -> VincioConfig:
    config = VincioConfig()
    config.storage.metadata = "memory://"
    config.observability.exporter = "memory"
    config.security.audit_log = False
    return config


# Sample docs (refund_policy.md + terms.md) the QA app grounds its answers on.
DOCS_DIR = write_sample_docs(Path(tempfile.mkdtemp()) / "docs")


def _format_sensitive_responder(request):
    """Offline stand-in for a real model. It answers correctly everywhere but only
    *cites* its evidence when the prompt is XML-rendered (contains a closing tag) —
    a genuine, learnable signal the optimizer can discover and exploit."""
    text = "\n".join(m.text for m in request.messages)
    answer = "The refund window for the Pro plan is 30 days."
    if "</" not in text:  # plain prompt: correct but uncited
        return answer
    import re

    match = re.search(r"\[([\w.:-]+:C\d+)\]", text)
    ref = match.group(1) if match else "E1"
    return f"{answer} [{ref}]"


def _qa_app() -> ContextApp:
    provider, model = example_provider(_format_sensitive_responder)
    config = _config()
    config.memory.write_back = ["facts"]  # grounded run claims become candidate memories
    app = ContextApp(name="optimize_demo", provider=provider, model=model, config=config)
    app.add_source("docs", path=str(DOCS_DIR))
    app.add_memory()
    return app


# ---------------------------------------------------------------------------
# 1. The closed loop: trace -> dataset -> eval -> optimize -> promote
# ---------------------------------------------------------------------------
async def section_closed_loop() -> ImprovementLoop:
    banner("1. The closed loop (trace -> dataset -> eval -> optimize -> promote)")
    app = _qa_app()

    # Real production traffic: run a few paraphrases, then mark the answers as
    # approved. Captured traces with positive feedback become the eval dataset.
    questions = [
        "What is the refund window for the Pro plan?",
        "How long do Pro customers have to request a refund?",
        "Within how many days can a Pro plan be refunded?",
        "Pro plan refund period?",
    ]
    for q in questions:
        await app.arun(q, user_id="u1")
    for trace in app.tracer.exporter.traces:
        record_feedback(trace, score=1.0)  # users approved these answers

    # Grounded output claims are auto-written back to memory as candidates — the
    # loop feeds the knowledge base while it feeds the optimizer.
    facts = [
        item
        for item in app.memory.store.all_items(statuses=("candidate",))
        if item.metadata.get("origin") == "run_fact"
    ]
    print(f"  auto-memory: {len(facts)} grounded fact(s) written back as candidates")

    # The loop optimizes for citation quality. The baseline answers correctly but
    # does not cite; promotion requires a variant that cites WITHOUT regressing
    # groundedness (the explicit gate) or safety/schema (built-in rules).
    loop = ImprovementLoop(
        app,
        metrics=["lexical_overlap", "groundedness", "citation_accuracy", "cost", "latency"],
        weights=FitnessWeights(accuracy_metric="citation_accuracy"),
        gates={"groundedness": ">= 0.5"},
        experiment="refund_qa",
    )
    result = await loop.arun(min_feedback_score=0.5, max_variants=6, subset_size=4)
    print(f"  dataset from traces: {result.dataset_name} ({result.dataset_size} cases, "
          f"fp {result.dataset_fingerprint})")
    print(f"  promoted: {result.promoted} — {result.reason}")
    if result.promoted_ref:
        # The winner is pushed to the prompt registry and tagged production.
        version = loop.registry.get(loop.prompt_name, tag="production")
        print(f"  registry: {result.promoted_ref} tags={version.tags} "
              f"eval_runs={len(version.eval_runs)}")
    return loop


# ---------------------------------------------------------------------------
# 2. Reflective optimization (GEPA-style) — read failures, propose the fix
# ---------------------------------------------------------------------------
def section_reflective() -> None:
    banner("2. Reflective optimization (GEPA-style)")

    # The model answers correctly only when the prompt plans first, so a reflection
    # that adds a plan-then-answer step should beat the baseline — a fix the
    # optimizer discovers by reading WHY the baseline lost, not by blind search.
    def responder(req):
        text = "\n".join(m.text for m in req.messages).lower()
        if "plan the steps" in text or "briefly plan" in text:
            return "The Pro plan refund window is 30 days."
        return "I am not sure."

    provider, _ = example_provider(responder)
    app = ContextApp(name="reflect", provider=provider, model="teacher", config=_config())
    dataset = Dataset(
        name="refunds",
        cases=[
            EvalCase(id=f"c{i}", input="What is the Pro plan refund window?",
                     expected="The Pro plan refund window is 30 days.")
            for i in range(6)
        ],
    )
    # budget hard-bounds the number of rollouts; the optimizer evolves a Pareto
    # frontier of edits under it (deterministic under the default seed).
    result = app.reflective_optimize(
        dataset, metrics=["lexical_overlap", "cost", "latency"],
        weights=FitnessWeights(latency=0.0), budget=8, minibatch_size=3,
    )
    for reflection in result.reflections:
        if reflection.edits:
            print(f"  reflection diagnosis: {reflection.diagnosis}")
    print(f"  rollouts spent: {result.evaluations} (hard-bounded by budget)")
    print(f"  promoted: {result.promoted} — {result.reason}")


# ---------------------------------------------------------------------------
# 3. The distillation flywheel — grounded traces -> gated cheaper student
# ---------------------------------------------------------------------------
async def section_distillation() -> None:
    banner("3. Distillation flywheel (grounded export -> gated student)")
    from types import SimpleNamespace

    from vincio.core.types import EvidenceItem

    evidence = [
        EvidenceItem(id="D1:C0", source_id="D1",
                     text="Customers on the Pro plan may request refunds within 30 days.",
                     provenance=0.9)
    ]

    def trace(tid, inp, out, ev):
        return SimpleNamespace(
            id=tid, run_id=tid, session_id=None, status="ok", feedback=[],
            attributes={"input": inp, "output": out, "evidence": [e.model_dump() for e in ev]},
        )

    traces = [
        trace("t1", "Refund window?", "The Pro plan refund window is 30 days.", evidence),
        trace("t2", "Mascot?", "The mascot is a purple axolotl with 12 legs.", evidence),  # ungrounded
    ]
    # Only grounded traces become training data — the ungrounded claim is dropped.
    training_set = export_training_set(traces, require_grounding=True, min_support=0.4)
    print(f"  exported {len(training_set)} grounded example(s); "
          f"dropped {training_set.metadata['dropped_ungrounded']} ungrounded")

    # Teacher -> student gate: the student is promoted only if it HOLDS quality
    # (>= min_quality_ratio of the teacher) while being cheaper.
    async def evaluate_model(model, ds):
        from vincio.evals.reports import CaseResult, EvalReport

        q, cost = (0.95, 0.01) if model == "teacher" else (0.93, 0.002)
        return EvalReport(cases=[
            CaseResult(case_id=f"c{i}", metrics={"lexical_overlap": q, "cost": cost})
            for i in range(len(ds))
        ])

    dataset = Dataset(name="held", cases=[EvalCase(id=f"c{i}", input="q") for i in range(6)])
    loop = BootstrapFinetune(evaluate_model, min_quality_ratio=0.9)
    result = await loop.distill(training_set, dataset, teacher="teacher", student="student")
    print(f"  promoted student: {result.promoted} — holds {result.quality_ratio:.0%} quality "
          f"at {result.cost_savings:.0%} lower cost")
    if result.cascade:
        print(f"  cascade rungs: {[r.model for r in result.cascade.rungs]}")


# ---------------------------------------------------------------------------
# 4. RLVR — on-policy reinforcement from verifiable rewards (app.learn)
# ---------------------------------------------------------------------------
def _run_env(actions: list[dict]):
    """Drive the deterministic retail environment through a fixed action list."""
    env = build_retail_environment("cancel_refund")
    policy = scripted_policy([EnvAction(**a) for a in actions])
    return EnvironmentSimulator().run(env, policy)


# The correct trajectory cancels before refunding; the violation refunds a
# still-processing order, which the task-success oracle rejects.
_CORRECT = [
    {"kind": "tool", "tool": "cancel_order", "arguments": {"order_id": "O1002"}},
    {"kind": "tool", "tool": "refund_order", "arguments": {"order_id": "O1002"}},
]
_VIOLATION = [{"kind": "tool", "tool": "refund_order", "arguments": {"order_id": "O1002"}}]


def section_rlvr() -> None:
    banner("4. RLVR — on-policy reinforcement from verifiable rewards (app.learn)")
    good, bad = _run_env(_CORRECT), _run_env(_VIOLATION)

    # The reward is not a learned guess: it is the task-success ORACLE the platform
    # already computes — the database end state turns into a checkable reward.
    task = LearningTask(
        id="refund",
        prompt="Cancel order O1002 and refund it.",
        candidates=[
            CandidateOutcome(
                action="cancel_then_refund",
                sample=RewardSample(task_id="refund", verification=good.verification),
                text="cancel order O1002, then issue the refund",
            ),
            CandidateOutcome(
                action="refund_only",
                sample=RewardSample(task_id="refund", verification=bad.verification),
                text="issue the refund",
            ),
        ],
    )
    app = ContextApp(name="rlvr", config=_config())
    # A GRPO-style update: group-relative advantage, a KL-to-reference clamp so the
    # policy stays near the reference, and a monotonic no-regression gate so the
    # served policy never regresses the baseline reward.
    result = app.learn(
        [task], reward=RewardModel([OracleReward()]),
        kl_max=0.5, iterations=6, learning_rate=0.8,
    )
    print(f"  promoted={result.promoted} — {result.reason}")
    print(f"  expected reward: {result.baseline_reward} -> {result.policy_reward} "
          f"(Δ={result.reward_delta:+.4f})")
    print(f"  KL to reference: {result.kl_to_reference} (bound {result.kl_bound}, "
          f"within={result.kl_within_bound})")
    print(f"  monotonic (never regresses baseline): {result.reward_monotonic}")
    print(f"  verdict (same shape a prompt deploy produces): passed={result.verdict.passed}")


# ---------------------------------------------------------------------------
# 5. The declarative self-improvement policy (one governed contract)
# ---------------------------------------------------------------------------
async def section_self_improvement_policy(loop: ImprovementLoop) -> None:
    banner("5. Declarative SelfImprovementPolicy (one governed contract)")
    app = loop.app

    # One policy composes everything above: autonomous experiment proposal, online
    # drift response, meta-optimization, and canary-gated promotion — all under a
    # single eval budget and the same gates. dry_run streams the plan without
    # mutating the live prompt, so we can show the cycle deterministically.
    dataset = Dataset(
        name="golden",
        cases=[
            EvalCase(id=f"c{i}", input=q,
                     expected="Refunds within 30 days for the Pro plan.")
            for i, q in enumerate([
                "What is the refund window for the Pro plan?",
                "How long do Pro customers have to request a refund?",
                "Within how many days can a Pro plan be refunded?",
                "Pro plan refund period?",
            ])
        ],
    )
    policy = SelfImprovementPolicy(
        metrics=["lexical_overlap", "groundedness", "citation_accuracy", "cost", "latency"],
        gates={"groundedness": ">= 0.5"},
        eval_budget=24.0,
        dry_run=True,  # plan only — do not mutate the live prompt in the demo
    )
    controller = app.self_improvement(policy, dataset=dataset)
    phases = []
    async for event in controller.astream():
        phases.append(event.phase)
    # The cycle emits observe -> proposal -> meta -> ... -> promote/rollback.
    print(f"  policy metrics watched: {policy.metrics[:3]}…")
    print(f"  cycle phases streamed: {' -> '.join(phases)}")
    print(f"  every decision is audited; dry_run={policy.dry_run} (live prompt untouched)")


# ---------------------------------------------------------------------------
# 6. Gated deploy with canary + rollback
# ---------------------------------------------------------------------------
def section_gated_deploy(loop: ImprovementLoop) -> None:
    banner("6. Gated deploy with canary + rollback")
    app = loop.app
    from vincio.optimize import CanarySpec
    from vincio.prompts.compiler import CompilerOptions
    from vincio.prompts.optimizers import PromptVariant

    # Section 1 promoted an XML-rendered prompt — the live prompt now cites its
    # evidence (the mock model only cites when the prompt is XML-rendered).
    dataset = Dataset(
        name="canary",
        cases=[
            EvalCase(id=f"c{i}", input="What is the refund window for the Pro plan?",
                     expected="The refund window for the Pro plan is 30 days. [docs:refund_policy.md:C0]")
            for i in range(4)
        ],
    )
    # Qualify candidates on citation_accuracy with a strict no-regression threshold.
    spec = CanarySpec(metric="citation_accuracy", regression_threshold=0.05,
                      require_significance=False)

    # A genuinely regressing candidate: the same prompt but rendered as markdown,
    # which drops the XML tags the model needs to cite. Citation accuracy collapses,
    # so the offline canary refuses it and rolls the live prompt back to known-good.
    regressing = PromptVariant(
        name="markdown-no-cite",
        spec=app.prompt_spec,
        compiler_options=CompilerOptions(format="markdown"),
    )
    refused = app.deploy(regressing, dataset=dataset, canary=spec, rollback_on_fail=True)
    verdict = refused.verdict
    print(f"  regressing candidate: deployed={refused.deployed} — {refused.reason}")
    if verdict is not None:
        print(f"  canary verdict: baseline={verdict.baseline:.2f} "
              f"candidate={verdict.candidate:.2f} (Δ={verdict.delta:+.2f})")
    print(f"  rolled back to last known-good: {refused.rolled_back_to}")

    # A safe candidate: re-deploying the current live prompt is a no-op regression
    # (a tie passes — deploying a no-regression change is safe), so it clears.
    safe = app.deploy(app.prompt_spec, dataset=dataset, canary=spec)
    print(f"  no-regression candidate: deployed={safe.deployed} — {safe.reason}")


# ---------------------------------------------------------------------------
# 7. On-device LoRA local adaptation (in-process, gated, reversible)
# ---------------------------------------------------------------------------
# A slice of grounded edge traffic the in-process model has served.
_QA = [
    ("what is the refund policy", "Refunds are processed within 30 days."),
    ("how do I reset my password", "Use the reset link on the login page."),
    ("what are the shipping options", "We ship worldwide via DHL in 5-7 days."),
    ("how do I contact support", "Email support@example.com any time."),
]


def _training_set(qa) -> TrainingSet:
    return TrainingSet(
        name="local-adapter",
        examples=[
            TrainingExample(messages=[
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ])
            for q, a in qa
        ],
    )


def _golden(qa) -> Dataset:
    return Dataset(name="golden",
                   cases=[EvalCase(id=f"c{i}", input=q, expected=a) for i, (q, a) in enumerate(qa)])


def section_local_adaptation() -> None:
    banner("7. On-device LoRA local adaptation (in-process, gated, reversible)")
    from vincio.providers.mock import MockProvider

    # A base model that does not know the grounded answers. The continual loop fits
    # a parameter-efficient low-rank adapter on-device from the grounded training
    # set — pure Python, no network — and promotes it ONLY when the locally-adapted
    # model is at-least-as-good as the base on a held-out set (the same
    # no-regression discipline a hosted fine-tune clears).
    app = ContextApp(name="edge", provider=MockProvider(default_text="I am not sure about that."),
                     config=_config())
    registry = AdapterRegistry()
    policy = LocalAdaptationPolicy(min_examples=4, min_samples=4, require_significance=False)
    result = app.adapt_locally(_golden(_QA), training_set=_training_set(_QA),
                               policy=policy, registry=registry)
    print(f"  promoted={result.promoted}  base={result.verdict.baseline:.2f} -> "
          f"adapted={result.verdict.candidate:.2f}  (Δ={result.verdict.delta:+.2f})")
    print(f"  live run now answers the grounded way: {app.run(_QA[0][0]).raw_text!r}")

    # Reversible: unloading the adapter restores the base model exactly.
    app.use_local_adapter(None)
    print(f"  after unload (base restored): {app.run(_QA[0][0]).raw_text!r}")


# ---------------------------------------------------------------------------
# 8. Federated cross-org improvement + a differential-privacy accountant
# ---------------------------------------------------------------------------
async def section_federated_dp() -> None:
    banner("8. Federated cross-org improvement + differential-privacy accountant")
    from vincio.providers.mock import MockProvider

    DELTA = 1e-5
    DIM = 64
    emb = LocalHashEmbedder(dim=DIM)
    fleet = ["org-a", "org-b"]
    qa_a = [("what is the refund policy", "Refunds are processed within 30 days."),
            ("how do I reset my password", "Use the reset link on the login page.")]
    qa_b = [("what are the shipping options", "We ship worldwide via DHL in 5-7 days."),
            ("how do I contact support", "Email support@example.com any time.")]
    qa_all = qa_a + qa_b

    # Each member contributes a numeric, raw-text-free subspace summary — the
    # clipped, masked scatter carries geometry, never a prompt or response.
    privacy = PrivacyConfig(min_contributors=2, secure_aggregation=True, clip_norm=1.0)
    builder = ContributionBuilder(embedder=emb, privacy=privacy)
    contribution_a = await builder.build(_training_set(qa_a), "gguf-local",
                                          member_id="org-a", participants=fleet)
    contribution_b = await builder.build(_training_set(qa_b), "gguf-local",
                                          member_id="org-b", participants=fleet)
    blob = contribution_a.model_dump_json()
    leaked = [q for q, a in qa_a if q in blob or a in blob]
    print(f"  org-a contribution: {contribution_a.n_examples} examples, masked={contribution_a.masked}; "
          f"raw traffic leaked on the wire: {leaked} (none)")

    # Secure aggregation: the per-member masks cancel exactly in the sum, so the
    # aggregator recovers the fleet subspace without seeing any one update. It
    # refuses a round below the k-anonymity contributor floor.
    subspace = SecureAggregator(privacy=privacy, rank=8).aggregate([contribution_a, contribution_b])
    print(f"  merged rank-{subspace.rank} subspace from {subspace.contributor_count} contributors")
    try:
        SecureAggregator(privacy=privacy).aggregate([contribution_a])
    except Exception as exc:  # noqa: BLE001 - demonstrating the k-anonymity refusal
        print(f"  single-member round refused: {type(exc).__name__}")

    # The adopting member re-fits its own adapter against the shared geometry,
    # gated for no regression on a held-out set spanning the whole fleet's tasks.
    app = ContextApp(name="org-a", provider=MockProvider(default_text="I am not sure about that."),
                     config=_config())
    app.embedder = emb
    fed_policy = FederatedPolicy(min_examples=4, min_samples=4, require_significance=False)
    fed = app.adopt_federated(_golden(qa_all), [contribution_a, contribution_b],
                              training_set=_training_set(qa_all), policy=fed_policy)
    print(f"  adopted={fed.adopted}  base={fed.verdict.baseline:.2f} -> "
          f"adapted={fed.verdict.candidate:.2f}  (Δ={fed.verdict.delta:+.2f})")

    # The differential-privacy accountant: a subject's data is touched again and
    # again (memory consolidation, every federated round). A Rényi/moments
    # accountant tracks the cumulative (ε, δ) and REFUSES a release that would
    # exceed the per-subject budget — privacy composes far tighter than naive sum.
    print("\n  -- per-subject differential-privacy budget --")
    dp_app = ContextApp(name="dp", provider=MockProvider(default_text="ok"), config=_config())
    dp_app.use_privacy_accountant(
        default_budget=PrivacyBudget(epsilon=2.0, delta=DELTA),
        default_mechanism=PrivacyMechanism(noise_multiplier=4.0),
    )
    dp_app.add_memory()
    facts = ["the user prefers metric units and a dark theme",
             "the user's home airport is SFO and they fly on Tuesdays",
             "the user is allergic to penicillin",
             "the user manages a team of six engineers in Berlin"]
    for round_no in range(2):
        for i, fact in enumerate(facts):
            dp_app.memory.write_fact(f"{fact} (note {round_no}.{i})", scope=MemoryScope.SESSION,
                                     owner_id="sess-alice", type=MemoryType.FACT, confidence=0.9)
        report = await dp_app.memory.consolidate("sess-alice", user_id="alice")
        charged = f"{report.privacy_epsilon:.3f}" if report.privacy_epsilon is not None else "—"
        print(f"  consolidation {round_no}: promoted={report.promoted}  cumulative ε={charged}")
    spent = dp_app.privacy_accountant.spent("alice")
    single = PrivacyMechanism(noise_multiplier=4.0).epsilon(delta=DELTA)
    print(f"  composed ε after 2 rounds: {spent:.3f}  (naive sum would be {2 * single:.3f})")

    # Once the budget is spent, the next consolidation is REFUSED — the episodes
    # simply stay in their short-lived episodic form.
    for i, fact in enumerate(facts):
        dp_app.memory.write_fact(f"{fact} (note 2.{i})", scope=MemoryScope.SESSION,
                                 owner_id="sess-alice", type=MemoryType.FACT, confidence=0.9)
    refused = await dp_app.memory.consolidate("sess-alice", user_id="alice")
    print(f"  consolidation 2: promoted={refused.promoted}  privacy_refused={refused.privacy_refused}")
    print(f"  audit chain verified: {dp_app.audit.verify_chain()}")


# ---------------------------------------------------------------------------
# 9. Autonomous skill acquisition — the open-ended cultivate loop
# ---------------------------------------------------------------------------
def section_skill_acquisition() -> None:
    banner("9. Autonomous skill acquisition (the cultivate loop)")
    from vincio.providers.mock import MockProvider

    app = ContextApp(name="open-ended-learner",
                     provider=MockProvider(default_text="ok"), config=_config())

    # A self-proposed, bounded curriculum over deterministic reference environments,
    # each carrying its own success oracle. "counter-to-two" then "counter-to-four"
    # lets the second REUSE the first as a macro (a skill that composes a skill);
    # the vault task adds a second, independent skill. Each objective is gated by
    # rails + the governance verifier BEFORE it is attempted, never blindly run.
    tasks = [
        CurriculumTask(id="c2", objective="increment counter to two",
                       environment=lambda: build_counter_environment(target=2)),
        CurriculumTask(id="c4", objective="increment counter to four",
                       environment=lambda: build_counter_environment(target=4)),
        CurriculumTask(id="vault", objective="open the vault by advancing",
                       environment=lambda: build_vault_environment(steps_to_open=3)),
    ]

    # cultivate = propose -> attempt (library-composing search) -> verify (oracle) ->
    # distill (a winning trajectory into a content-addressed LearnedSkill) -> promote
    # (only through the SAME no-regression gate a prompt deploy clears).
    library = LearnedSkillLibrary()
    result = app.cultivate(AutoCurriculum(tasks), library=library, cycles=3)

    # Capability only ever moves up (monotonic by construction), and every objective
    # stayed inside policy — verify() re-checks the whole run end-to-end offline.
    print(f"  capability: {result.capability_before:.2f} -> {result.capability_after:.2f} "
          f"(monotonic={result.monotonic})")
    print(f"  skills promoted={result.skills_promoted}  stayed_in_policy={result.stayed_in_policy}  "
          f"verify()={result.verify()}")
    # The learned-skill library is content-addressed: it re-derives and verifies with
    # no model and no network — a portable, offline-checkable capability artifact.
    print(f"  learned skills: {[s.name for s in library.skills]}")
    print(f"  library verifies offline: {library.verify()} (hash {library.library_hash[:12]}…)")


async def main() -> None:
    loop = await section_closed_loop()
    section_reflective()
    await section_distillation()
    section_rlvr()
    await section_self_improvement_policy(loop)
    section_gated_deploy(loop)
    section_local_adaptation()
    await section_federated_dp()
    section_skill_acquisition()
    print("\nOne gated, audited path — improvement is monotonic by construction.")


if __name__ == "__main__":
    asyncio.run(main())
