"""Provider/model rotation & swap regression — the migration safety net (1.8).

A model swap is the most common and the riskiest change in production. Vincio
makes it a gated, statistically-backed discipline rather than a hope:

  1. Capability-aware routing — the router (and failover) refuse to substitute a
     model that cannot serve the request, and pick the cheapest *capable* one.
  2. SwapGate — replay golden traces + an eval/cost/latency/behavioral diff with
     significance, and PASS/FAIL the migration. A model reaches the live path
     only if it clears the gate.
  3. Model-swap regression — hold prompt/data/config fixed, swap only the model,
     and report per-metric significance, the cost/latency trade, and the
     worst-regressed slices, with flake quarantine.
  4. Shadow & canary — qualify a candidate on live traffic without touching the
     user, with automatic rollback on regression.
  5. Lifecycle watcher — propose a migration off a deprecated/retired model.
  6. Google/Vertex batch parity — half-cost batch on every provider.

Runs fully offline with the deterministic mock provider.
"""

from __future__ import annotations

import asyncio
from datetime import date

from _shared import example_provider

from vincio import ContextApp
from vincio.core.types import ContentPart, ImageRef, Message, ModelRequest
from vincio.evals.datasets import Dataset, EvalCase
from vincio.optimize.routing import Router
from vincio.providers import MockProvider
from vincio.providers.batch import BatchRequest, BatchRunner, InProcessBatchBackend
from vincio.providers.lifecycle import LifecycleWatcher


def _swap_app() -> ContextApp:
    """An app whose mock degrades on the cheap candidate, so a swap regresses."""

    def responder(request):
        if request.model == "gpt-5.2-nano":
            return "I'm not sure."
        return "The capital of France is Paris."

    provider, model = example_provider()
    if model == "mock-1":  # offline: drive the model-sensitive responder
        provider = MockProvider(responder=responder)
        model = "gpt-5.2"
    return ContextApp("swap_demo", provider=provider, model=model)


def _dataset() -> Dataset:
    return Dataset(name="geo", cases=[
        EvalCase(id=f"c{i}", input="What is the capital of France?",
                 expected="The capital of France is Paris.")
        for i in range(6)
    ])


def routing_demo() -> None:
    print("== Capability-aware routing — cheapest *capable* model per request ==")
    router = Router.from_models(
        MockProvider(), ["gpt-5.2", "gpt-5.2-mini", "gpt-5.2-nano"], strategy="cheapest"
    )
    text = ModelRequest(model="x", messages=[Message(role="user", content="classify this ticket")])
    plain = router.pick(text)
    print(f"  text request   -> {plain.model} (est ${plain.est_cost_usd:.6f})")
    budgeted = router.pick(text, budget_usd=0.0)
    print(f"  budget $0       -> {budgeted.model} (downgraded={budgeted.downgraded})")
    vision = ModelRequest(model="x", messages=[Message(role="user", content=[
        ContentPart(type="image", image=ImageRef(url="data:image/png;base64,AAAA"))])])
    vrouter = Router.from_models(MockProvider(), ["mistral-small-latest", "gpt-5.2"])
    pick = vrouter.pick(vision)
    print(f"  vision request  -> {pick.model} (skipped {list(pick.skipped)})")


def regression_demo() -> None:
    print("\n== Model-swap regression — swap only the model, prove it's safe ==")
    app, dataset = _swap_app(), _dataset()
    report = app.swap_regression(dataset, candidate_model="gpt-5.2-nano", baseline_model="gpt-5.2",
                                 repeats=2)
    for metric, test in sorted(report.metric_tests.items()):
        flag = "REGRESSION" if metric in report.regressions else "ok"
        print(f"  {metric:22s} {test['mean_a']:.3f} -> {test['mean_b']:.3f}  "
              f"p={test['p_value']:.4f}  [{flag}]")
    print(f"  cost x{report.cost['ratio']}  ·  regressed={report.regressed}")


def gate_demo() -> None:
    print("\n== SwapGate — gated PASS/FAIL on the swap ==")
    app, dataset = _swap_app(), _dataset()
    bad = app.gate_swap("gpt-5.2-nano", baseline_model="gpt-5.2", dataset=dataset)
    good = app.gate_swap("gpt-5.2-mini", baseline_model="gpt-5.2", dataset=dataset)
    print(f"  gpt-5.2 -> gpt-5.2-nano : {'PASS' if bad.passed else 'FAIL'} — {bad.reason}")
    print(f"  gpt-5.2 -> gpt-5.2-mini : {'PASS' if good.passed else 'FAIL'} — {good.reason}")


async def shadow_canary_demo() -> None:
    print("\n== Shadow & canary — qualify on live traffic, auto-rollback ==")
    shadow_app = ContextApp("shadow_demo", provider=MockProvider(default_text="primary answer"),
                            model="gpt-5.2")
    shadow = shadow_app.shadow("gpt-5.2-mini", block=True,
                               candidate_provider=MockProvider(default_text="candidate answer"))
    req = ModelRequest(model="gpt-5.2", messages=[Message(role="user", content="hello")])
    await shadow.generate(req)
    print(f"  shadow: user gets primary; candidate similarity = {shadow.diff()['mean_output_similarity']}")

    from vincio.core.types import ModelResponse, TokenUsage
    from vincio.providers.shadow import CanaryRouter

    good = MockProvider(responder=lambda r: ModelResponse(
        model=r.model, text="ok", finish_reason="stop", usage=TokenUsage(input_tokens=5, output_tokens=2)))
    bad = MockProvider(responder=lambda r: ModelResponse(
        model=r.model, text="", finish_reason="content_filter", usage=TokenUsage(input_tokens=5)))
    canary = CanaryRouter(good, bad, percent=50.0, min_samples=4, regression_threshold=0.2)
    for _ in range(40):
        await canary.generate(req)
        if canary.rolled_back:
            break
    print(f"  canary: rolled_back={canary.rolled_back} ({canary.state().rollback_reason})")


def lifecycle_demo() -> None:
    print("\n== Lifecycle watcher — propose a migration off a sunsetting model ==")
    watcher = LifecycleWatcher()
    for alert in watcher.scan(["gemini-2.0-flash", "gpt-4o"], as_of=date(2026, 6, 17)):
        print(f"  [{alert.severity}] {alert.message}")
    proposal = watcher.propose_migration("gemini-2.0-flash")
    print(f"  proposal: {proposal.from_model} -> {proposal.to_model} ({proposal.kind})")


async def batch_parity_demo() -> None:
    print("\n== Google/Vertex batch parity — half-cost batch everywhere ==")
    runner = BatchRunner(InProcessBatchBackend(MockProvider(default_text="ok")), discount=0.5)
    reqs = [BatchRequest(custom_id=f"g{i}", request=ModelRequest(
        model="gemini-2.5-flash", messages=[Message(role="user", content="hi")])) for i in range(4)]
    result = await runner.run(reqs)
    print(f"  batched {len(result.succeeded)} requests at half cost (${result.cost_usd:.8f})")


def main() -> None:
    routing_demo()
    regression_demo()
    gate_demo()
    asyncio.run(shadow_canary_demo())
    lifecycle_demo()
    asyncio.run(batch_parity_demo())
    print("\nThe swap is gated, not guessed — on one trace and one audit chain.")


if __name__ == "__main__":
    main()
