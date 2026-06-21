"""Test-time compute & reasoning orchestration.

Reasoning-model thinking budgets and parallel test-time search are the cheapest
quality lever left, and the platform already owns the pieces to orchestrate them:
cost-aware action selection, critics and validators that act as verifiers, and a
provider-neutral reasoning-effort knob. This makes test-time compute a
first-class, budgeted, cache-aware dimension of the run rather than a per-call
knob.

Four steps, all offline and deterministic (no model required):

  1. ReasoningController: thinking effort scales with the task's difficulty and
     classification, and a hard reasoning-token ceiling means a hard task can
     never silently exhaust the run.
  2. Reasoning-trace-aware caching: when a re-ask shares a thinking prefix that
     was already paid for, the controller recognizes the warm prefix and steps
     the effort down — the reasoning analogue of prompt-prefix caching.
  3. Verifier-guided best-of-N: draw candidates, score each with the platform's
     existing judge ensemble, keep the best, and early-exit the instant the
     verifier clears the bar (so an easy ask costs one draw, a hard one spends
     the budget).
  4. Self-consistency and beam search: a majority vote that early-exits once the
     lead is unbeatable, and a beam over partial tool-use trajectories scored by
     a critic.

Everything here is opt-in and additive; nothing below is required to run Vincio.
"""

from __future__ import annotations

import asyncio

from vincio import (
    ContextApp,
    ReasoningController,
    ReasoningPolicy,
    TaskType,
    TestTimeSearch,
)
from vincio.evals.datasets import EvalCase
from vincio.evals.ensemble import JudgeEnsemble
from vincio.evals.judges import DeterministicJudge
from vincio.evals.metrics import MetricResult, RunOutput
from vincio.optimize import CallableVerifier, JudgeVerifier, SearchBudget


async def reasoning_controller() -> None:
    print("1. ReasoningController — effort scales with difficulty, under a hard ceiling")
    ctl = ReasoningController(ReasoningPolicy(max_reasoning_tokens=8192))

    easy = ctl.decide(task=TaskType.CLASSIFICATION, text="Is this email spam? yes/no")
    hard = ctl.decide(
        task=TaskType.DOCUMENT_COMPARISON,
        text="Compare and reason about why these two merger agreements differ " * 12,
        remaining_output_tokens=4096,
    )
    print(f"   easy task → effort={easy.effort:8s} budget={easy.thinking_budget_tokens:5d} "
          f"(difficulty {easy.difficulty})")
    print(f"   hard task → effort={hard.effort:8s} budget={hard.thinking_budget_tokens:5d} "
          f"(difficulty {hard.difficulty})")
    print(f"   hard task held under the budget share: capped={hard.budget_capped}")

    capped = ReasoningController(ReasoningPolicy(max_reasoning_tokens=1500)).decide(difficulty=0.95)
    print(f"   hard ceiling clamps a hard task: {capped.thinking_budget_tokens} tokens "
          f"(ceiling_capped={capped.ceiling_capped})")


async def reasoning_trace_cache() -> None:
    print("\n2. Reasoning-trace-aware caching — a warm thinking prefix steps effort down")
    ctl = ReasoningController()  # carries its own ReasoningTraceCache

    cold = ctl.decide(difficulty=0.5, prefix_hash="prefixA", model="m-reasoner")
    print(f"   cold prefix → effort={cold.effort} (warm={cold.warm_prefix})")

    # The run paid for this thinking prefix once; record it (the runtime does this
    # automatically when a controller is installed via app.use_reasoning_controller).
    ctl.record_trace(prefix_hash="prefixA", model="m-reasoner", effort=cold.effort,
                     reasoning_tokens=64)
    warm = ctl.decide(difficulty=0.5, prefix_hash="prefixA", model="m-reasoner")
    print(f"   warm prefix → effort={warm.effort} (warm={warm.warm_prefix})")
    print(f"   trace cache: {ctl.trace_cache.stats()}")


# A reference answer the judge panel rewards proximity to.
TARGET = "the refund window for the pro plan is 30 days"
CANDIDATES = [
    "Refunds take roughly a month I think",
    "Pro plan refunds within some window",
    "The refund window for the Pro plan is 30 days.",  # the strong one
    "Refund policy unclear",
]


def _overlap(case: EvalCase, out: RunOutput) -> MetricResult:
    want = set(TARGET.split())
    got = set(out.raw_text.lower().replace(".", "").split())
    return MetricResult(name="overlap", value=len(want & got) / len(want))


async def verifier_guided_best_of_n() -> None:
    print("\n3. Verifier-guided best-of-N — scored by an existing judge ensemble")
    # The verifier is the platform's judge ensemble; disagreement would lower its
    # confidence, here a single deterministic judge keeps it reproducible.
    ensemble = JudgeEnsemble([DeterministicJudge(_overlap, name="overlap")])
    verifier = JudgeVerifier(ensemble, case=EvalCase(id="q", input="What is the refund window?"))

    search = TestTimeSearch(
        lambda i: CANDIDATES[i],
        verifier=verifier,
        budget=SearchBudget(max_candidates=4, confidence_target=0.95),
    )
    result = await search.best_of_n()
    print(f"   drew {result.n_generated}/4 candidates; best score {result.best.score:.2f}")
    print(f"   winner: {result.best.answer_text!r}")
    print(f"   early_exit={result.early_exit} — {result.stop_reason}")


async def self_consistency_and_beam() -> None:
    print("\n4. Self-consistency vote + beam over tool-use trajectories")
    # Self-consistency: 4× "30 days", 1× "14 days" → majority, locked early.
    votes = ["30 days", "30 days", "14 days", "30 days", "30 days"]
    sc = TestTimeSearch(lambda i: votes[i], budget=SearchBudget(max_candidates=5))
    result = await sc.self_consistency()
    print(f"   majority answer {result.best.answer_text!r} "
          f"(vote share {result.confidence:.0%}, drew {result.n_generated}/5)")
    print(f"   early_exit={result.early_exit} — {result.stop_reason}")

    # Beam search over partial tool-use trajectories: a critic rewards the plan
    # that cancels before it refunds.
    ideal = ["cancel_order", "refund_order"]
    tools = ["cancel_order", "refund_order", "email_customer"]

    def expand(state: list[str]) -> list[list[str]]:
        if len(state) >= len(ideal):
            return []
        return [state + [tool] for tool in tools]

    def critic(candidate) -> float:
        seq = candidate.output
        matched = sum(1 for i, tool in enumerate(seq) if i < len(ideal) and tool == ideal[i])
        return matched / len(ideal)

    beam = TestTimeSearch(
        lambda i: None, verifier=CallableVerifier(critic), budget=SearchBudget(max_candidates=64)
    )
    result = await beam.beam_search(
        root=[], expand=expand, beam_width=2, max_depth=2,
        state_text=lambda s: " → ".join(s),
    )
    print(f"   best trajectory: {' → '.join(result.best.output)} (score {result.best.score:.2f}, "
          f"scored {result.n_scored} nodes)")


async def runtime_wiring() -> None:
    print("\n5. Runtime wiring — install the controller once, every run sets its own effort")
    app = ContextApp(name="ttc_demo")
    app.use_reasoning_controller(ReasoningPolicy(max_reasoning_tokens=4096))
    print("   app.use_reasoning_controller() installed; runs now fill reasoning_effort")
    print("   from the task classification + live budget unless RunConfig pins one.")
    print(f"   controller: {type(app.reasoning_controller).__name__} "
          f"with a {type(app.reasoning_controller.trace_cache).__name__}")


async def main() -> None:
    await reasoning_controller()
    await reasoning_trace_cache()
    await verifier_guided_best_of_n()
    await self_consistency_and_beam()
    await runtime_wiring()


if __name__ == "__main__":
    asyncio.run(main())
