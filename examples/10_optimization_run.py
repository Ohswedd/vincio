"""Optimization run: search prompt variants by fitness,
promote only through safety gates."""

import asyncio
import tempfile
from pathlib import Path

from _shared import citing_responder, example_provider, write_sample_docs

from vincio import ContextApp, Dataset
from vincio.evals import EvalCase, EvalRunner
from vincio.optimize import FitnessWeights, PromptOptimizer

docs_dir = write_sample_docs(Path(tempfile.mkdtemp()) / "docs")
provider, model = example_provider(
    citing_responder("The refund window for the Pro plan is 30 days. [{ref}]")
)

app = ContextApp(name="optimize_demo", provider=provider, model=model)
app.add_source("docs", path=str(docs_dir))
app.set_policy("answer_only_from_sources", True)

dataset = Dataset(
    name="refunds",
    cases=[
        EvalCase(id=f"c{i}", input=question, expected="Refunds within 30 days for the Pro plan.")
        for i, question in enumerate(
            [
                "What is the refund window for the Pro plan?",
                "How long do Pro customers have to request a refund?",
                "Within how many days can a Pro plan be refunded?",
                "Pro plan refund period?",
            ]
        )
    ],
)

METRICS = ["lexical_overlap", "groundedness", "schema_validity", "cost", "latency"]


async def evaluate_variant(variant, ds):
    """Swap the app's prompt spec/options for the candidate and eval it."""
    original_spec, original_options = app.prompt_spec, app.prompt_compiler.options
    app.prompt_spec = variant.spec
    app.prompt_compiler.options = variant.compiler_options
    try:
        return await EvalRunner(app, metrics=METRICS, concurrency=4).arun(ds)
    finally:
        app.prompt_spec, app.prompt_compiler.options = original_spec, original_options


async def main():
    optimizer = PromptOptimizer(
        evaluate_variant,
        weights=FitnessWeights(groundedness=2.0),
        gates={"groundedness": ">= 0.8"},
    )
    result = await optimizer.optimize(app.prompt_spec, dataset, max_variants=6, subset_size=4)
    print(f"baseline fitness: {result.baseline_fitness:.4f}")
    for entry in result.history:
        print(f"  [{entry['phase']:>8}] {entry['name']}: {entry['fitness']:.4f}")
    print(f"\npromoted: {result.promoted} — {result.reason}")
    if result.best:
        print("winning dimensions:", result.best.params)


if __name__ == "__main__":
    asyncio.run(main())
