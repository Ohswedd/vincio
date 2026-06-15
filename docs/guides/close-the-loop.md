# Guide: close the loop

Vincio ships the milestone no single-purpose library can: one continuous,
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

## Reflective optimization (GEPA-style)

Blind search proposes; reflection *diagnoses*. The `ReflectiveOptimizer` reads
the eval report's failures, reflects on why the prompt lost, and proposes
targeted edits — then verifies each child on the same gated, Pareto-aware
machinery:

```python
result = app.reflective_optimize(
    dataset,
    metrics=["semantic_similarity", "groundedness", "cost", "latency"],
    gates={"groundedness": ">= 0.8"},
    budget=12,            # hard cap on evaluation rollouts
    minibatch_size=8,     # cheap screening before a full rollout
)
result.promoted, result.reason
result.frontier.front     # the evolved Pareto frontier
[r.diagnosis for r in result.reflections]  # why each edit was proposed
```

A child is screened on a minibatch and earns a full-dataset rollout *only* when
it beats its parent, so the sample-efficiency win GEPA reports (beating RL with
far fewer rollouts) holds under a hard budget — deterministic under a seed.
`strategy="mipro"` switches to MIPROv2-style joint instruction+example proposal.
The result is a drop-in `OptimizationResult`, so the improvement loop runs it
unchanged:

```python
loop = app.improvement_loop(optimizer="reflective", gates={"groundedness": ">= 0.8"})
result = loop.run(min_feedback_score=0.5)   # promotes through the same gated path
```

```bash
vincio optimize reflective --app app.py --dataset golden.jsonl --strategy mipro
vincio loop run --app app.py --reflective --gate groundedness=">= 0.8"
```

The default `HeuristicReflector` is deterministic and offline (it maps a sagging
metric to the edit a careful prompt engineer would make); `LLMReflector` adds a
model-backed reflection with the heuristic as a fallback, so behaviour stays
reproducible in tests and air-gapped runs.

## The distillation flywheel

The one lever the rest of the field is missing: turn the runs you already make
into *cheaper inference*. The faithful, flag-free path is to keep the
`RunResult`s — they carry the full output and cited evidence, and the runtime
stamps the input — and export from them:

```python
results = [app.run(q) for q in prompts]
ts = app.export_training_set(runs=results, path="train.jsonl")
ts.grounded_fraction                   # 1.0 — every example is evidence-supported
ts.save("train_anthropic.jsonl", format="anthropic")
```

If you'd rather curate from the traces production already writes (with feedback
filtering), enable capture so the full output and evidence are recorded — this
covers streaming runs too:

```python
app.enable_training_capture()          # record full output + cited evidence on every trace
# ... run production traffic (incl. app.astream) ...
ts = app.export_training_set(min_feedback_score=0.5, path="train.jsonl")
```

Nothing ungrounded is exported — an example whose answer the evidence does not
support is dropped, not trained on. The teacher→student loop then promotes a
cheaper student into the runtime cascade **only** when it holds quality:

```python
result = app.distill(ts, held_out, teacher="gpt-5.2", student="gpt-5.2-mini")
result.promoted, result.quality_ratio, result.cost_savings
# on promotion, result.cascade (student → teacher) is installed via use_cascade
```

The student is gated like every other promotion: it must preserve a quality
ratio of the teacher, cost strictly less, and regress neither safety nor schema
validity. `vincio distill --traces-dir .vincio/traces --output train.jsonl`
exports from the CLI.

## Learned prompt compression

Extractive compression keeps whole sentences; learned compression goes finer.
`LLMLinguaCompressor` scores every token's importance and drops the
low-information ones while protecting the answer (numbers, amounts, entities,
citation markers, query terms):

```python
from vincio.context import LLMLinguaCompressor, compression_faithfulness

c = LLMLinguaCompressor()
result = c(evidence_text, query="refund window", max_tokens=120)
result.method                          # "llmlingua"
compression_faithfulness(evidence_text, result.text)  # salient units preserved
```

It is a drop-in for the compiler's inline compression step, but adoption is
**faithfulness-gated** — installed only when it shrinks the prompt without
losing the cited-fact set or regressing quality under eval:

```python
result = app.gate_compression(golden)   # measures faithfulness + quality + tokens
result.adopted, result.token_savings, result.learned_faithfulness
# or opt in directly, ungated:
app.use_learned_compression()
```

## Continuous quality

The same metric objects that gate releases offline also watch live traffic
and feed the loop back. Online evaluators score a sampled fraction of
finished runs off the hot path (deterministic 1-in-N sampling), writing each
score as a time series; a `DriftMonitor` compares those scores against a
golden baseline and raises a regression the moment it appears:

```python
app.add_online_evaluator("goal_accuracy", sample_rate=0.2)   # scored after the response is final
app.add_online_evaluator("answer_relevance", sample_rate=0.1)
series = app.online_evaluators[0].series()                   # rows from the metadata store

from vincio.evals import DriftMonitor
monitor = DriftMonitor(bus=app.events, store=app.store, score_threshold=0.1)
monitor.set_score_baseline("goal_accuracy", baseline_values)
report = monitor.check_scores("goal_accuracy", [r["metric_value"] for r in series])
report.drifted, report.delta, report.z_score                 # raises drift.detected on the bus
```

Drift fires a `drift.detected` event and persists the baseline; the CLI
`vincio eval drift baseline.json current.json` exits non-zero so a scheduled
check can gate. Because the live scores are trajectory metrics like any
other, they flow straight into the optimizer's fitness — the
`AGENTIC_OBJECTIVES` preset keeps a frontier over `goal_accuracy`,
`tool_call_accuracy`, `step_efficiency`, and `cost`:

```python
from vincio.optimize import AGENTIC_OBJECTIVES, pareto_loop

result = await pareto_loop(candidates, evaluate_fn, dataset,
                           baseline=baseline, objectives=AGENTIC_OBJECTIVES)
```

A metric earns the right to *gate* CI only once it has demonstrably agreed
with people. An `AnnotationQueue` pairs judge scores with human labels and
reports Cohen's κ; the judge's gating weight is `1.0` only when calibrated κ
clears the bar — chance-level agreement carries no veto:

```python
from vincio.evals import AnnotationQueue

q = AnnotationQueue(name="judge_cal")
item = q.add(run_id="r1", judge_score=0.9); q.label(item.id, human_score=1.0)
q.agreement()                       # {"cohens_kappa": ..., "exact_agreement": ..., "n": ...}
q.gating_weight(threshold=0.6)      # 0.0 until κ clears the bar, then 1.0
```

`GEvalJudge.calibrate(pairs)` returns the same `cohens_kappa`, and
`vincio eval annotate labels.jsonl` reports it from the CLI — the LLM judge
only joins the gate after it has earned the trust.

And the judge that gates the optimizer can itself be optimized.
`app.calibrate_judge(judge, samples)` reflectively proposes alternative
evaluation procedures, scores each against the labelled samples, and installs
the one that best agrees with people — only when its κ strictly beats the
incumbent — leaving the judge calibrated for CI gating:

```python
result = app.calibrate_judge(geval, labelled_samples)  # (case, output, human_score)
result.adopted, result.kappa_before, result.kappa_after
result.gating_weight_before, result.gating_weight_after
```

## What lands where (interconnection)

| Event | Where it's recorded |
|---|---|
| Loop promotion | `PromptRegistry` version (tagged, eval-linked) + `loop_promotion` audit entry + `loop.promoted` event |
| Baseline & winner reports | `ExperimentTracker` (same metadata store as runs) |
| Dataset provenance | case metadata (trace/run/session ids) + fingerprint on the `LoopResult` |
| Grounded facts | candidate memories with `origin: run_fact`, support, evidence ids |
| Retrieval tuning | engine weights (applied only when gated improvement holds) |
| Learned budgets | `LearnedAllocations` JSON → `BudgetAllocator(learned=...)` |
| Online scores | metadata store time series (kind `eval_results`) + `eval.online` event |
| Detected drift | persisted baseline (kind `drift_baselines`) + `drift.detected` event |
| Reflective promotion | same as loop promotion (registry + audit + event); `optimize.reflective` event when applied to the app |
| Exported training set | grounded fine-tuning JSONL + `distill.exported` event |
| Promoted student | runtime `ModelCascade` (cheap→strong) + `distill.promoted` event |
| Adopted compressor | `ContextCompiler.compressor` + `compression.adopted` event (only when faithfulness-gated) |

The VincioBench `loop` family measures all of it offline — promotion fires
and is deterministic, gates block regressions, the registry is tagged and
eval-linked, grounded facts are written (and ungrounded ones never are),
retrieval tuning is gated, the frontier excludes dominated points, learned
budgets promote, guided search respects its budget, the reflective optimizer
beats the baseline within its rollout budget, distillation exports only grounded
examples and gates the student on quality, and learned compression preserves the
cited-fact set under a faithfulness gate — under 23 CI-gated budgets.
