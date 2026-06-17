"""The closed-loop ecosystem (0.8).

One continuous, reproducible cycle that no single-purpose library can ship:
production traces become datasets, datasets drive gated optimization, the
winner is promoted into the prompt registry and applied live — while runs
write grounded facts back to memory, eval-scored relevance tunes retrieval,
the optimizer keeps a cost/quality Pareto frontier instead of one score, and
budget allocation is learned from eval outcomes.
"""

import asyncio
import re
import tempfile
from pathlib import Path

from _shared import example_provider, write_sample_docs

from vincio import ContextApp, TaskType, VincioConfig
from vincio.observability.sessions import record_feedback
from vincio.optimize import (
    BudgetLearner,
    ImprovementLoop,
    ObjectiveSpec,
    ParetoFrontier,
    ParetoPoint,
    RetrievalFeedback,
    objective_vector,
    records_from_dataset,
)

docs_dir = write_sample_docs(Path(tempfile.mkdtemp()) / "docs")


def format_sensitive_responder(request):
    """Offline stand-in for a real model: it answers correctly everywhere but
    only cites its sources reliably for XML-rendered prompts — a genuine
    signal for the optimizer to find."""
    text = "\n".join(m.text for m in request.messages)
    answer = "The refund window for the Pro plan is 30 days."
    if "</" not in text:
        return answer
    match = re.search(r"\[([\w.:-]+:C\d+)\]", text)
    ref = match.group(1) if match else "E1"
    return f"{answer} [{ref}]"


provider, model = example_provider(format_sensitive_responder)

config = VincioConfig()
config.storage.metadata = "memory://"  # keep the demo self-contained
config.observability.exporter = "memory"  # capture traces in-process
config.memory.write_back = ["facts"]  # 1. auto-memory from runs

app = ContextApp(name="closed_loop_demo", provider=provider, model=model, config=config)
app.add_source("docs", path=str(docs_dir))
app.add_memory()


async def main() -> None:
    # --- 1. Auto-memory: grounded output claims become candidate memories ---

    await app.arun("What is the refund window for the Pro plan?", user_id="u1")
    facts = [
        item
        for item in app.memory.store.all_items(statuses=("candidate",))
        if item.metadata.get("origin") == "run_fact"
    ]
    print(f"auto-memory: {len(facts)} grounded fact(s) written as candidates")
    for item in facts:
        print(f"  - {item.content!r} (support {item.metadata['support']})")

    # --- 2. Simulate production traffic with user feedback ----------------------

    for question in (
        "How long do Pro customers have to request a refund?",
        "Within how many days can a Pro plan be refunded?",
        "Pro plan refund period?",
    ):
        await app.arun(question, user_id="u1")
    for trace in app.tracer.exporter.traces:
        record_feedback(trace, score=1.0)  # users approved these answers

    # --- 3. The loop: trace → dataset → eval → optimize → promote ---------------

    # Optimize for citation quality: the baseline answers correctly but does
    # not cite; promotion requires a variant that cites without regressing
    # groundedness (the gate) or safety/schema (built-in rules).
    from vincio.optimize import FitnessWeights

    loop = ImprovementLoop(
        app,
        metrics=["lexical_overlap", "groundedness", "citation_accuracy", "cost", "latency"],
        weights=FitnessWeights(accuracy_metric="citation_accuracy"),
        gates={"groundedness": ">= 0.5"},
        experiment="refund_qa",
    )
    result = await loop.arun(min_feedback_score=0.5, max_variants=6, subset_size=4)
    print(f"\nloop dataset: {result.dataset_name} ({result.dataset_size} cases, "
          f"fingerprint {result.dataset_fingerprint})")
    for step in result.steps:
        print(f"  [{step['stage']}] " + ", ".join(f"{k}={v}" for k, v in step.items() if k != "stage"))
    print(f"promoted: {result.promoted} — {result.reason}")
    if result.promoted_ref:
        version = loop.registry.get(loop.prompt_name, tag="production")
        print(f"registry: {result.promoted_ref} tags={version.tags} "
              f"eval_runs={len(version.eval_runs)}")

    # --- 4. Pareto: the frontier behind the decision ----------------------------

    optimization = result.optimization
    if optimization is not None and optimization.baseline is not None:
        specs = [
            ObjectiveSpec(name="citations", metric="citation_accuracy"),
            ObjectiveSpec(name="groundedness", metric="groundedness"),
            ObjectiveSpec(name="latency_s", metric="latency", direction="min", scale=0.001),
        ]
        points = [
            ParetoPoint(name=c.name, objectives=objective_vector(c.full_report, specs))
            for c in [optimization.baseline, *optimization.candidates]
            if c.full_report is not None
        ]
        frontier = ParetoFrontier.build(points, specs=specs)
        print(f"\npareto frontier: {len(frontier.front)}/{len(points)} non-dominated")
        knee = frontier.knee()
        if knee:
            print(f"  knee point: {knee.name} {knee.objectives}")

    # --- 5. Retrieval feedback: relevance labels tune fusion weights ------------

    labelled = loop.curate(loop.capture())
    for case in labelled.cases:  # label the chunks that should come back
        case.rubric["relevant_ids"] = [
            e.id for e in (await app.retrieval.retrieve(case.input_text, top_k=1)).evidence
        ]
    feedback = RetrievalFeedback(app.retrieval, records_from_dataset(labelled), top_k=4)
    tuned = await feedback.tune_index_weights()
    print(f"\nretrieval feedback: applied={tuned.applied} "
          f"({tuned.baseline_score} → {tuned.tuned_score}); {tuned.reason}")

    # --- 6. Learned budgeting: allocation tuned from eval outcomes --------------

    async def evaluate_allocation(fractions, ds):
        from vincio.context.budgeting import BudgetAllocator
        from vincio.evals import EvalRunner

        app.context_compiler.allocator = BudgetAllocator(
            learned={TaskType.DOCUMENT_QA.value: fractions}
        )
        try:
            runner = EvalRunner(app, metrics=["lexical_overlap", "cost", "latency"])
            return await runner.arun(ds)
        finally:
            app.context_compiler.allocator = BudgetAllocator()

    learner = BudgetLearner(evaluate_allocation)
    budget_result, learned = await learner.learn(
        labelled, task_type=TaskType.DOCUMENT_QA, candidates=4, subset_size=4
    )
    print(f"\nlearned budgeting: promoted={budget_result.promoted} — {budget_result.reason}")
    if learned is not None:
        app.use_learned_budgets(learned)  # installs the eval-tuned table
        print(f"  installed: {learned.get(TaskType.DOCUMENT_QA)}")


if __name__ == "__main__":
    asyncio.run(main())
