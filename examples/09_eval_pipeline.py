"""Eval pipeline: dataset → runner → gates → report → baseline diff."""

import tempfile
from pathlib import Path

from _shared import citing_responder, example_provider, write_sample_docs

from vincio import ContextApp, Dataset
from vincio.evals import EvalCase, EvalRunner

docs_dir = write_sample_docs(Path(tempfile.mkdtemp()) / "docs")
provider, model = example_provider(
    citing_responder("The refund window for the Pro plan is 30 days. [{ref}]")
)

app = ContextApp(name="eval_demo", provider=provider, model=model)
app.add_source("docs", path=str(docs_dir))
app.set_policy("answer_only_from_sources", True)

dataset = Dataset(
    name="refunds",
    cases=[
        EvalCase(id="c1", input="What is the refund window for the Pro plan?",
                 expected="Refunds within 30 days for the Pro plan.", tags=["refund"]),
        EvalCase(id="c2", input="How long is the Pro refund window?",
                 expected="30 days", tags=["refund"], difficulty="easy"),
    ],
)

if __name__ == "__main__":
    runner = EvalRunner(
        app,
        metrics=["groundedness", "citation_accuracy", "semantic_similarity", "cost", "latency"],
        gates={"groundedness": ">= 0.9", "p95_latency": "<= 10000"},
        concurrency=4,
    )
    baseline = runner.run(dataset, name="baseline")
    current = runner.run(dataset, baseline=baseline, name="current")
    current.print_summary()
    print("\ngates:", {k: v["passed"] for k, v in current.gates.items()})
    print("regressions vs baseline:", current.metadata["baseline_diff"]["regressed_cases"])
    out = Path(tempfile.mkdtemp()) / "report.json"
    current.save(out)
    print("saved:", out)
