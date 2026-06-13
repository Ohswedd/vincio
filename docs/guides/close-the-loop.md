# Guide: close the loop

0.8 ships the milestone no single-purpose library can: one continuous,
reproducible improvement cycle — **trace → dataset → eval → optimize →
promote** — plus the feedback paths that let every organ tune the others:
runs write grounded facts back to memory, eval-scored relevance tunes
retrieval, the optimizer keeps a cost/quality Pareto frontier instead of one
score, budget allocation is learned from eval outcomes, and richer offline
search strategies drive the evolution loop. Everything flows through the
same packet, ledger, and trace as the rest of the library.

## The improvement loop

`ImprovementLoop` wires together pieces that already exist — the tracer's
exporter, `dataset_from_traces`, the eval runner, the gated evolution loop,
the prompt registry, and the experiment tracker — into one call:

```python
loop = app.improvement_loop(
    metrics=["semantic_similarity", "groundedness", "cost", "latency"],
    gates={"groundedness": ">= 0.8"},      # promotion gates
    experiment="support_qa",
)
result = loop.run(min_feedback_score=0.5)  # curate from approved traces
result.promoted, result.promoted_ref, result.dataset_fingerprint
```

One cycle does five things:

1. **Capture** — loads the traces your production runs already write
   (`loop.capture()`; any exporter with `load_all()` or `.traces`).
2. **Curate** — `dataset_from_traces` turns them into an eval dataset,
   keeping only successful runs whose mean user feedback clears
   `min_feedback_score`. The dataset's case-id fingerprint is recorded so
   the decision is reproducible.
3. **Evaluate** — the current prompt is the baseline, measured by the same
   metric objects used everywhere else.
4. **Optimize** — the gated evolution loop searches prompt variants;
   promotion is blocked on safety/schema regression, cost ceilings, and
   your gates. Candidate evaluations are **memory-write-free**: an eval run
   never pollutes user memory or hands later candidates different recall
   state than earlier ones saw.
5. **Promote** — the winner is pushed to the `PromptRegistry`, tagged
   (`production` by default), linked to the eval report that justified it,
   applied to the live app, written to the hash-chained audit log
   (`loop_promotion`), and announced on the event bus (`loop.promoted`).
   `dry_run=True` reports the decision without acting on it.

Both the baseline and the winner land in the `ExperimentTracker` (same
metadata store as runs and packets), so `tracker.compare()` and `ab_test()`
work across loop cycles. From the CLI:

```bash
vincio loop run --app app.py --min-feedback 0.5 --gate groundedness=">= 0.8"
vincio loop run --app app.py --dataset golden.jsonl --dry-run
```

## Auto-memory from runs

With `memory.write_back: [facts]` in `vincio.yaml`, verifiable claims from a
run's output that the cited evidence actually supports become **candidate**
memories — with measured support, evidence provenance, and the same guarded
admission (privacy, stability, contradiction, confidence) as every other
write:

```yaml
memory:
  write_back: [facts]      # also: input | evidence | tools
  fact_min_support: 0.5    # evidence support a claim needs
  max_facts_per_run: 5
```

Extraction is deterministic (`extract_grounded_facts`): a sentence must look
like a factual claim *and* reach `fact_min_support` lexical support against
the run's cited evidence. Facts land as candidates (`origin: run_fact`,
status penalty in recall until confirmed) and are utility-scored against the
task before they ever enter a packet — high-confidence grounding in, stale
hallucination out.

## Retrieval feedback

Relevance labels that already live on eval cases (`rubric.relevant_ids`)
tune retrieval directly:

```python
from vincio.optimize import RetrievalFeedback, records_from_report, recommend_chunking

records = records_from_report(report, dataset)   # or records_from_dataset(dataset)
feedback = RetrievalFeedback(app.retrieval, records, top_k=8)
result = await feedback.tune()                   # fusion weights + reranker blend
result.applied, result.index_weights_after, result.reranker_weights_after
```

The search is deterministic (fixed grids, no randomness) and **gated**: the
engine's per-index RRF fusion weights and the heuristic reranker's blend
only change when the tuned configuration measurably beats the current one on
recall@k + MRR over the records — otherwise nothing moves.
`recommend_chunking(reports_by_config, baseline=...)` closes the chunking
side: it picks the chunking config whose eval report scored best, and stays
on the baseline unless beaten by `min_improvement`.

## Cost/quality Pareto optimization

`pareto_loop` keeps the full multi-objective frontier instead of collapsing
accuracy, groundedness, latency, and cost into one number:

```python
from vincio.optimize import ObjectiveSpec, pareto_loop

result = await pareto_loop(
    candidates, evaluate_fn, dataset, baseline=baseline,
    objectives=[
        ObjectiveSpec(name="accuracy", metric="semantic_similarity"),
        ObjectiveSpec(name="cost", metric="cost", direction="min"),
    ],
    constraints={"cost": 0.01},   # at most a cent per case
    prefer="accuracy",            # or omit: the knee point wins
)
result.frontier.front             # non-dominated points
result.frontier.knee()            # best summed normalized goodness
```

Screening still uses scalar fitness (cheap); the final pick comes from the
frontier of full-dataset reports and goes through the same promotion safety
rules — a frontier point that regresses safety or fails a gate never wins.

## Learned context budgeting

Per-task budget allocation tuned from eval outcomes instead of fixed tables:

```python
from vincio.optimize import BudgetLearner

learner = BudgetLearner(evaluate_allocation)          # (fractions, dataset) -> EvalReport
result, learned = await learner.learn(dataset, task_type=TaskType.DOCUMENT_QA)
if learned:                                           # promoted through the gated loop
    learned.save(".vincio/budgets.json")
    app.use_learned_budgets(learned)                  # or the saved path
```

The learner perturbs the baseline allocation in bounded steps (move a slice
of budget from a donor block to a receiver, renormalize) and adopts a table
only when it beats the baseline through the same safety gates as every
other optimizer. Tasks without a learned table keep the fixed defaults.

## Context-aware offline search

The evolution loop's candidate proposals can now condition on what already
scored well:

```python
result = await ContextOptimizer(evaluate_config).optimize(
    dataset, budget=12, strategy="hill_climb",   # or "anneal", "random"
)
```

`hill_climb` mutates the best config one knob at a time; `anneal` walks with
a cooling schedule (early batches explore, late batches exploit). Both are
deterministic under a seed, hard-bounded by the evaluation budget, and feed
the same gated promotion path — a guided search can never bypass the safety
rules. `guided_search(space, evaluate, strategy=...)` exposes the primitive
for custom spaces.

## What lands where (interconnection)

| Event | Where it's recorded |
|---|---|
| Loop promotion | `PromptRegistry` version (tagged, eval-linked) + `loop_promotion` audit entry + `loop.promoted` event |
| Baseline & winner reports | `ExperimentTracker` (same metadata store as runs) |
| Dataset provenance | case metadata (trace/run/session ids) + fingerprint on the `LoopResult` |
| Grounded facts | candidate memories with `origin: run_fact`, support, evidence ids |
| Retrieval tuning | engine weights (applied only when gated improvement holds) |
| Learned budgets | `LearnedAllocations` JSON → `BudgetAllocator(learned=...)` |

The VincioBench `loop` family measures all of it offline — promotion fires
and is deterministic, gates block regressions, the registry is tagged and
eval-linked, grounded facts are written (and ungrounded ones never are),
retrieval tuning is gated, the frontier excludes dominated points, learned
budgets promote, and guided search respects its budget — under 14 CI-gated
budgets.
