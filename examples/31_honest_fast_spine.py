"""The honest, fast spine (1.7).

Vincio's promises made literally true, and the model-knowledge layer turned into
data:

  1. Enforced Budget — max_cost_usd / max_input_tokens / max_output_tokens /
     max_steps become hard caps on app.run(); an opt-out preserves the old
     soft-cap behavior for one minor.
  2. ModelRegistry — a data-driven catalog (capabilities, pricing, lifecycle)
     keyed by exact model id; capability guards and the price table read from it,
     and an unknown model warns instead of silently billing $0.
  3. Semantic context scoring — opt-in embedding-cosine relevance, MMR selection,
     and salient-unit value-contradiction detection (vs the old bag-of-words).
  4. RunHandle — cooperative cancellation identical across streaming and
     non-streaming; a cancelled run is still fully recorded.
  5. Significance-gated promotion + the trace-replay executor — promotions carry
     a p-value / confidence interval / effect size, and ReplayRunner re-runs
     captured traces and diffs output, trajectory, and cost.

Runs fully offline with the deterministic mock provider.
"""

from __future__ import annotations

import asyncio

from _shared import example_provider

from vincio import ContextApp, default_model_registry
from vincio.core.types import EvidenceItem, RunConfig, UserInput
from vincio.observability.costs import ModelPrice


def model_registry_demo() -> None:
    print("== ModelRegistry — capabilities, pricing & lifecycle as data ==")
    reg = default_model_registry()
    for model in ("gpt-5.2", "gpt-4o", "claude-haiku-4-5", "gemini-2.0-flash"):
        caps = reg.capabilities(model)
        print(f"  {model:18s} reasoning={caps.reasoning!s:5} "
              f"ctx={caps.max_context_tokens:>9,} lifecycle={reg.lifecycle(model)}")
    # Dated snapshots resolve by prefix; a successor is suggested for older models.
    print(f"  gpt-4o-2024-11-20 -> {reg.resolve('gpt-4o-2024-11-20').model}")
    print(f"  successor(gemini-2.0-flash) -> {reg.successor('gemini-2.0-flash')}")


def enforced_budget_demo() -> None:
    print("\n== Enforced Budget — the advertised cap is now a hard cap ==")
    provider, model = example_provider()
    app = ContextApp("budget_demo", provider=provider, model=model)
    # Price the model so a tiny cost cap actually bites.
    app.cost_tracker.price_table.set(model, ModelPrice(input_per_mtok=1e6))
    app.budget = app.budget.model_copy(update={"max_cost_usd": 1e-9})

    capped = app.run("summarize the refund policy")
    print(f"  hard cap:   status={capped.status.value}  error={capped.error}")
    soft = app.run("summarize the refund policy", config=RunConfig(enforce_budget_caps=False))
    print(f"  opt-out:    status={soft.status.value}  (legacy soft-cap preserved)")


async def cancellation_demo() -> None:
    print("\n== RunHandle — cooperative cancellation, still fully recorded ==")
    from vincio.core.types import ModelResponse
    from vincio.providers.base import ModelProvider

    class SlowProvider(ModelProvider):
        name = "slow"

        async def generate(self, request: object) -> ModelResponse:
            await asyncio.sleep(5)
            return ModelResponse(text="late")

    app = ContextApp("cancel_demo", provider=SlowProvider())
    handle = app.submit("a long-running question")
    await asyncio.sleep(0.05)
    handle.cancel()
    try:
        await handle.result()
    except asyncio.CancelledError:
        pass
    audited = [e for e in app.audit.entries if e.action == "run"]
    print(f"  cancelled run audited: {bool(audited)} (decision={audited[-1].decision if audited else '-'})")


async def semantic_and_conflict_demo() -> None:
    print("\n== Value-level contradiction (salient-unit disagreement) ==")
    from vincio.context.compiler import ContextCompiler, ContextCompilerOptions
    from vincio.core.types import Objective, TaskType

    compiler = ContextCompiler(ContextCompilerOptions())
    packet = await compiler.compile(
        objective=Objective("refund window", task_type=TaskType.DOCUMENT_QA),
        user_input=UserInput(text="refund window"),
        evidence=[
            EvidenceItem(id="a", source_id="kb", relevance=0.0,
                         text="Customers can request a refund within 30 days of the purchase date for any item."),
            EvidenceItem(id="b", source_id="kb", relevance=0.0,
                         text="Customers can request a refund within 14 days of the delivery date for any item."),
        ],
    )
    for conflict in packet.conflicts:
        print(f"  conflict [{conflict.get('kind')}]: {conflict['a']} vs {conflict['b']} "
              f"differ on {conflict.get('differing')}")


async def replay_demo() -> None:
    print("\n== Significance-gated promotion + trace replay ==")
    from vincio.evals.experiments import ab_test
    from vincio.evals.replay import ReplayRunner, _CaptureExporter
    from vincio.evals.reports import CaseResult, EvalReport

    baseline = EvalReport(cases=[CaseResult(case_id=f"c{i}", metrics={"m": 0.6}) for i in range(8)])
    candidate = EvalReport(cases=[CaseResult(case_id=f"c{i}", metrics={"m": 0.9}) for i in range(8)])
    verdict = ab_test(baseline, candidate, "m")
    print(f"  ab_test: Δ={verdict['delta']}  p={verdict['p_value']}  "
          f"effect={verdict['effect_size']}  CI=[{verdict['ci_low']}, {verdict['ci_high']}]")

    provider, model = example_provider()
    app = ContextApp("replay_demo", provider=provider, model=model)
    cap = _CaptureExporter(app.tracer.exporter)
    app.tracer.exporter = cap
    run = await app.arun("what is the refund window?")
    trace = cap.captured[run.trace_id]
    app.tracer.exporter = cap._inner
    result = await ReplayRunner(app).replay([trace])
    print(f"  replay: {result.summary()}")


def main() -> None:
    model_registry_demo()
    enforced_budget_demo()
    asyncio.run(cancellation_demo())
    asyncio.run(semantic_and_conflict_demo())
    asyncio.run(replay_demo())
    print("\nThe spine's promises are now literally true — on one trace, one audit chain.")


if __name__ == "__main__":
    main()
