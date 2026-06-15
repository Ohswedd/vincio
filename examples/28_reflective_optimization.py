"""Reflective optimization & the data flywheel (1.4).

Sharpen the optimizer to the 2025–26 state of the art, and add the one lever
the field is missing — turning production traces into cheaper inference — while
every promotion stays gated, grounded, and audited:

  1. Reflective optimizer (GEPA-style): read the eval failures, reflect on why
     the prompt lost, propose targeted edits, evolve a Pareto frontier.
  2. MIPROv2-style joint instruction + few-shot proposal.
  3. Distillation flywheel: curate grounded traces into fine-tuning JSONL, then
     gate a cheaper student on holding quality before promoting it to a cascade.
  4. Learned prompt compression (LLMLingua-style), faithfulness-gated.
  5. Optimizer-judge calibration: tune the judge's own steps for κ agreement.

Runs fully offline on the deterministic mock provider.
"""

from __future__ import annotations

from types import SimpleNamespace

from vincio import ContextApp, VincioConfig
from vincio.context.llmlingua import LLMLinguaCompressor, compression_faithfulness
from vincio.core.tokens import count_tokens
from vincio.core.types import EvidenceItem
from vincio.evals import Dataset, EvalCase
from vincio.evals.datasets import EvalCase as Case
from vincio.evals.judges import GEvalJudge
from vincio.evals.metrics import RunOutput
from vincio.optimize import (
    BootstrapFinetune,
    CompressionTuner,
    FitnessWeights,
    export_training_set,
)
from vincio.providers import MockProvider


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _app(responder) -> ContextApp:
    cfg = VincioConfig()
    cfg.storage.metadata = "memory://"
    cfg.observability.exporter = "memory"
    cfg.security.audit_log = False
    return ContextApp(name="reflect", provider=MockProvider(responder=responder), model="teacher", config=cfg)


REFUND_DATASET = Dataset(
    name="refunds",
    cases=[
        EvalCase(id=f"c{i}", input="What is the Pro plan refund window?",
                 expected="The Pro plan refund window is 30 days.")
        for i in range(6)
    ],
)
EVIDENCE = [
    EvidenceItem(id="D1:C0", source_id="D1",
                 text="Customers on the Pro plan may request refunds within 30 days of purchase.",
                 provenance=0.9)
]


def reflective_demo() -> None:
    _section("1. Reflective optimizer (GEPA-style)")

    # The model answers correctly only when the prompt plans first — so a
    # reflection that adds a plan-then-answer step beats the baseline.
    def responder(req):
        text = "\n".join(m.text for m in req.messages)
        if "plan the steps" in text.lower() or "briefly plan" in text.lower():
            return "The Pro plan refund window is 30 days."
        return "I am not sure."

    app = _app(responder)
    result = app.reflective_optimize(
        REFUND_DATASET, metrics=["semantic_similarity", "cost", "latency"],
        weights=FitnessWeights(latency=0.0), budget=8, minibatch_size=3,
    )
    for reflection in result.reflections:
        if reflection.edits:
            print(f"  reflection: {reflection.diagnosis}")
    print(f"  rollouts spent: {result.evaluations} (hard-bounded by budget)")
    print(f"  promoted: {result.promoted} — {result.reason}")


def mipro_demo() -> None:
    _section("2. MIPROv2-style joint instruction + example proposal")

    def responder(req):
        text = "\n".join(m.text for m in req.messages)
        return ("The Pro plan refund window is 30 days." if "evidence" in text.lower()
                else "I am not sure.")

    app = _app(responder)
    result = app.reflective_optimize(
        REFUND_DATASET, strategy="mipro", metrics=["semantic_similarity", "cost", "latency"],
        weights=FitnessWeights(latency=0.0), budget=10, minibatch_size=3,
    )
    print(f"  strategy: {result.strategy} · promoted: {result.promoted}")
    print(f"  reason: {result.reason}")


def distill_demo() -> None:
    _section("3. Distillation flywheel (grounded export → gated student)")

    def trace(tid, inp, out, ev):
        return SimpleNamespace(
            id=tid, run_id=tid, session_id=None, status="ok", feedback=[],
            attributes={"input": inp, "output": out, "evidence": [e.model_dump() for e in ev]},
        )

    traces = [
        trace("t1", "Refund window?", "The Pro plan refund window is 30 days.", EVIDENCE),
        trace("t2", "Mascot?", "The mascot is a purple axolotl with 12 legs.", EVIDENCE),  # ungrounded
    ]
    training_set = export_training_set(traces, require_grounding=True, min_support=0.4)
    print(f"  exported {len(training_set)} grounded example(s); "
          f"dropped {training_set.metadata['dropped_ungrounded']} ungrounded")
    print(f"  openai JSONL: {training_set.to_jsonl(format='openai')[:70]}...")

    # Teacher → student gate: the student must hold quality and be cheaper.
    async def evaluate_model(model, ds):
        from vincio.evals.reports import CaseResult, EvalReport

        q, cost = (0.95, 0.01) if model == "teacher" else (0.93, 0.002)
        return EvalReport(cases=[CaseResult(case_id=f"c{i}", metrics={"semantic_similarity": q, "cost": cost})
                                 for i in range(len(ds))])

    loop = BootstrapFinetune(evaluate_model, min_quality_ratio=0.9)
    import asyncio

    result = asyncio.run(loop.distill(training_set, REFUND_DATASET, teacher="teacher", student="student"))
    print(f"  promoted student: {result.promoted} — holds {result.quality_ratio:.0%} quality "
          f"at {result.cost_savings:.0%} lower cost")
    if result.cascade:
        print(f"  cascade: {[r.model for r in result.cascade.rungs]}")


def compression_demo() -> None:
    _section("4. Learned prompt compression (faithfulness-gated)")
    text = (
        "The Pro plan offers a refund window of 30 days from the date of purchase. "
        "Customers who are not satisfied may contact support to request a full refund. "
        "The Enterprise plan provides a 90 day evaluation period and dedicated onboarding."
    )
    budget = count_tokens(text) // 2
    result = LLMLinguaCompressor()(text, "Pro plan refund window", budget)
    print(f"  {result.original_tokens} → {result.compressed_tokens} tokens "
          f"({result.ratio:.0%} of original), method={result.method}")
    print(f"  faithfulness (salient units preserved): "
          f"{compression_faithfulness(text, result.text):.0%}; '30' kept: {'30' in result.text}")

    # Adoption is gated: install only if quality + faithfulness hold under eval.
    async def evaluate(compressor, ds):
        from vincio.evals.reports import CaseResult, EvalReport

        learned = compressor is not None
        return EvalReport(cases=[CaseResult(case_id=f"c{i}", metrics={
            "semantic_similarity": 0.99 if learned else 1.0,
            "faithfulness": 0.95 if learned else 1.0,
            "input_tokens": 60.0 if learned else 100.0,
        }) for i in range(len(ds))])

    import asyncio

    tuner = CompressionTuner(evaluate)
    decision, _ = asyncio.run(tuner.tune(LLMLinguaCompressor(), REFUND_DATASET))
    print(f"  adopted: {decision.adopted} — {decision.reason}")


def judge_calibration_demo() -> None:
    _section("5. Optimizer-judge calibration (closing the loop on the loop)")

    def responder(req):
        text = "\n".join(m.text for m in req.messages)
        grounded_steps = "does not support" in text.lower()
        is_good = "GOOD_OUTPUT" in text
        if grounded_steps:
            return {"score": 5 if is_good else 1, "reasoning": "r"}
        return {"score": 3, "reasoning": "r"}

    judge = GEvalJudge(MockProvider(responder=responder), model="mock-1", criteria="Is the answer faithful?",
                       steps=["Read the output.", "Give a 1-5 score."])
    samples = [
        (Case(id=f"c{i}", input="q"),
         RunOutput(raw_text="GOOD_OUTPUT" if i % 2 == 0 else "BAD_OUTPUT"),
         1.0 if i % 2 == 0 else 0.0)
        for i in range(6)
    ]
    app = _app(lambda r: "")
    result = app.calibrate_judge(judge, samples)
    print(f"  adopted procedure: {result.adopted}")
    print(f"  Cohen's κ: {result.kappa_before:.2f} → {result.kappa_after:.2f}; "
          f"gating weight {result.gating_weight_before:.0f} → {result.gating_weight_after:.0f}")


def main() -> None:
    reflective_demo()
    mipro_demo()
    distill_demo()
    compression_demo()
    judge_calibration_demo()


if __name__ == "__main__":
    main()
