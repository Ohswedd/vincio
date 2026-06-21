# Reference: public API index

This page is generated from `vincio.__all__` — the exact set of names
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) applies to —
with each symbol's signature and docstring summary. It is gated for
docstring coverage: no public symbol ships undocumented. For the curated,
grouped narrative see [api.md](api.md).

**209** public symbols.

## Classes

### `AIBOM(**data)`

An AI bill of materials, serializable as CycloneDX 1.6 JSON.

### `AdaptiveSampler(cases, sample, gate, metric=…, budget, seed_samples=…, confidence=…, weights=…)`

Decide a mean-aggregate gate with the fewest samples by allocating the budget to the highest-variance cases and stopping as soon as the verdict is certain.

### `AgentDirectory(allow_list=…, audit=…, principal=…)`

A governed, discoverable directory of agents across A2A / ACP / MCP.

### `AgentRole(**data)`

A named role in a crew: who the agent is and what share it gets.

### `AlertManager(sinks=…)`

Evaluates :class:`AlertRule`\ s over a metric stream and dispatches alerts.

### `AlertRule(**data)`

One alerting rule over a metric stream.

### `AlertSink(*args, **kwargs)`

Base class for protocol classes.

### `AllowListGate(allow=…, deny=…, default_allow=…, action=…)`

A reachability allow-list for the agent fabric.

### `AnnexIVBuilder(classifier=…)`

Render EU AI Act **Annex IV** technical documentation as a cited document.

### `ApprovalRecord(**data)`

A tool-approval decision made during a turn.

### `Assistant(app, user_id=…, tenant_id=…, session_id=…, memory_writeback=…, auto_approve=…, on_approval=…, feature=…)`

A multi-turn conversational session over a :class:`ContextApp`.

### `AssistantTurn(**data)`

The outcome of one conversational turn.

### `BatchRunner(backend, price_table=…, tracer=…, discount=…, poll_interval_s=…, timeout_s=…, clock=…)`

Submit a batch, poll it to completion, reconcile, and cost-track.

### `BenchmarkAdapter(tasks=…, fixture_path=…)`

Base contract for a leaderboard adapter.

### `Blackboard(event_bus=…)`

Versioned, author-attributed shared memory for agent teams.

### `BootstrapFinetune(evaluate_model, quality_metric=…, min_quality_ratio=…, gates=…, trainer=…, swap_gate=…, dedupe_embedder=…)`

Teacher → student distillation with a gated quality hold.

### `Budget(**data)`

Hard resource limits for a run (budgets, termination).

### `BudgetManager(ledger, events=…)`

Enforces :class:`CostBudget`\ s and detects spend anomalies.

### `BundleRecord(**data)`

One governed, content-bound entry in the community index.

### `CalibrationReport(**data)`

The verdict of :meth:`WorldModel.calibrate` — the model's planning weight.

### `CanaryRouter(primary, candidate, percent=…, candidate_model=…, score_fn=…, min_samples=…, window=…, regression_threshold=…, on_rollback=…, prompt_registry=…, prompt_name=…, events=…)`

Ramp a percentage of live traffic onto a candidate, with auto-rollback.

### `CanarySpec(**data)`

How a candidate is qualified before it is deployed live.

### `CapabilityBroker(secret=…, default_ttl_s=…)`

Mints and verifies :class:`CapabilityToken`\ s from the user's authority.

### `CapabilityToken(**data)`

An unforgeable, capability-scoped grant minted from the user's request.

### `CausalAttributor(app, dataset, factors, metric=…, aggregate=…, repeats=…, concurrency=…)`

Attribute a metric delta to the components a release changed, by Shapley counterfactual replay over the dataset.

### `CircuitBreaker(inner, failure_threshold=…, min_calls=…, window=…, latency_threshold_ms=…, cooldown_s=…, half_open_max=…, events=…, clock=…)`

Per-provider circuit breaker with half-open probing.

### `CitationContract(**data)`

Field/claim-level citation requirements for a cited report.

### `CitedReportBuilder(entailment=…, audit_log=…, tenant_id=…)`

Resolve citations, verify per-claim support, render a cited report.

### `CommunityRegistry(allow_list=…, audit=…, signer=…, principal=…, require_signature=…, index=…)`

A governed, signed, audited index of community packs and skills.

### `ComplianceFramework(*args, **kwds)`

A governance framework whose controls Vincio maps onto.

### `ComplianceReport(**data)`

A coverage matrix across the mapped frameworks.

### `ConsentLedger(store=…, audit=…, default_allow=…)`

Records and checks consent, binding data to a purpose + lawful basis.

### `Constraint(text=…, **data)`

!!! abstract "Usage Documentation" [Models](../concepts/models.md)

### `ContainmentMonitor()`

Records capability exercises so containment can be proven after a run.

### `ContainmentReport(**data)`

The verdict of checking the containment invariant over a run.

### `ContentCapturePolicy(**data)`

Gate + redact prompt/completion content at the telemetry export boundary.

### `ContextApp(name=…, objective=…, output_schema=…, config=…, provider=…, model=…, budget=…, policies=…, prompt_spec=…)`

The top-level Vincio application: one object that compiles prompts, memory, retrieval, tools, schemas, and policies into validated, observable, model-ready context and runs the end-to-end pipeline.

### `ContextBudget(**data)`

A per-run context budget: the residency analogue of a dollar budget.

### `ContextCompactor(store=…, memory=…, owner_id=…, scope=…, summary_tokens=…, summarizer=…)`

Hierarchical, provenance-preserving compaction of cold run spans.

### `ContextGovernor(budget=…, decay=…, compactor=…, keep_recent_spans=…, decay_threshold=…, compact_batch=…)`

Per-run controller that holds a context budget across a long horizon.

### `ContinuousImprovementController(app, metrics=…, golden=…, registry=…, prompt_name=…, monitor=…, sustain=…, cooldown_s=…, eval_budget=…, quality_floor=…, reoptimize=…, gates=…, clock=…)`

Drive gated re-optimization / re-eval / rollback from live signals.

### `ControllerDecision(**data)`

The record of one controller evaluation — stamped on the audit chain.

### `CostAwareSelector(models, registry=…, quality_floor=…, events=…)`

Picks the cheapest capable model per action, escalating on low confidence.

### `CostBudget(**data)`

A spend limit on a scope, with an enforcement action on breach.

### `CostLedger(price_table=…, store=…, max_events=…)`

In-process append-only ledger of attributed cost events.

### `Crew(name=…, process=…, blackboard=…, tracer=…, manager_provider=…, manager_model=…, max_rounds=…, concurrency=…, cost_tracker=…, cost_ledger=…)`

A multi-agent team that collaborates over a shared blackboard.

### `Dataset(**data)`

!!! abstract "Usage Documentation" [Models](../concepts/models.md)

### `DeployResult(**data)`

Outcome of a canary-gated prompt/policy deployment.

### `DistributedCheckpointer(store=…, coordinator=…, owner=…, lease_ttl_s=…)`

A :class:`Checkpointer` that lease-guards and CAS-commits each super-step.

### `DocumentArtifact(**data)`

A rendered document: its bytes, format, and media type.

### `DocumentBuilder(audit_log=…, tenant_id=…)`

Render validated results into cited, contract-checked, audited documents.

### `DocumentContract(**data)`

The structural contract a generated document must satisfy.

### `DualPlaneExecutor(tool_runtime, broker=…, monitor=…, principal=…, approval=…, provider=…, model=…)`

Capability-secure executor separating the control and data planes.

### `Environment(*args, **kwargs)`

The stateful-environment contract: ``reset`` / ``step`` / ``observe`` / ``verify``.

### `EnvironmentSimulator(max_steps=…)`

Drive an agent *policy* through an :class:`Environment` to a verified end state.

### `ErasureProof(**data)`

A signed, content-bound manifest of exactly what an erasure removed.

### `ErasureResult(**data)`

Outcome of a right-to-erasure-by-source sweep.

### `EvidenceItem(**data)`

A provenance-aware unit of evidence (text, image, or table).

### `Example(**data)`

!!! abstract "Usage Documentation" [Models](../concepts/models.md)

### `ExperimentProposer(app, targets=…, eval_budget=…, golden_suite=…, gates=…)`

Rank where the system is weakest and schedule the highest-ROI experiment.

### `FRIAGenerator(classifier=…)`

Generate an Article 27 **fundamental-rights impact assessment** (FRIA).

### `FastEmbedEmbedder(model_name=…, dim=…, encode_fn=…, model=…, fallback=…)`

Local ONNX dense embedder via ``fastembed``.

### `FertilityTracker(model=…, baseline_language=…)`

Track tokens-per-word per language to surface the non-English token tax.

### `GGUFProvider(model_path=…, llama=…, n_ctx=…, embedding=…, **kwargs)`

Native in-process GGUF / llama.cpp provider with on-device embedding.

### `GoldenRegressionSuite(path=…, name=…)`

A held-out, *growing* golden regression set with per-case provenance.

### `GuardedBanditRouter(entries, bandit=…, safe_model=…, reward_fn=…, context_fn=…, epsilon=…, alpha=…, context_dim=…, seed=…, regret_budget=…, rollback_margin=…, store=…, app_name=…, events=…)`

A live routing bandit with a safety floor, regret tracking, and auto-rollback.

### `HTNDomain(**data)`

A library of operators and methods the planner decomposes against.

### `HealthAwareFailover(entries, guard_capabilities=…, registry=…)`

Failover chain that tries healthy providers first.

### `ImageGenRequest(**data)`

!!! abstract "Usage Documentation" [Models](../concepts/models.md)

### `ImageProvider()`

Abstract image generation/editing provider.

### `ImprovementLoop(app, registry=…, tracker=…, metrics=…, weights=…, gates=…, max_cost_per_case=…, experiment=…, prompt_name=…, concurrency=…, optimizer=…, strategy=…, reflector=…, golden_suite=…)`

Runs the trace → dataset → eval → optimize → promote cycle on an app.

### `IndexedTraceStore(path=…, percentile_window=…)`

SQLite-backed, indexed trace + cost store with pre-aggregated rollups.

### `Instruction(text=…, **data)`

!!! abstract "Usage Documentation" [Models](../concepts/models.md)

### `JudgeCalibrator(judge, reflector=…, kappa_bins=…, trust_threshold=…, min_kappa_gain=…)`

Tune a :class:`~vincio.evals.judges.GEvalJudge`'s evaluation steps to maximize agreement with human labels, then leave the judge calibrated.

### `JudgeEnsemble(judges, aggregate=…, disagreement_threshold=…, name=…)`

A panel of judges scored together, with disagreement surfaced as uncertainty and the panel as a whole calibrated against human labels.

### `JudgeVerifier(judge, case=…, name=…)`

Score candidates with any :class:`~vincio.evals.judges.Judge` or :class:`~vincio.evals.ensemble.JudgeEnsemble`.

### `KeyPool(providers, rpm=…, tpm=…, breaker=…, labels=…, max_attempts=…, base_backoff_s=…, max_backoff_s=…, seed=…, events=…, clock=…)`

Round-robin pool over multiple keys/regions of one logical provider.

### `LLMLinguaCompressor(scorer=…, min_keep_ratio=…, coarse_overshoot=…)`

Token-importance compressor (callable, drop-in for ``extractive_compress``).

### `LawfulBasis(*args, **kwds)`

GDPR Article 6(1) lawful bases for processing.

### `LearningResult(**data)`

The outcome of a :class:`TrajectoryOptimizer` run.

### `LifecycleWatcher(registry=…, warn_within_days=…, events=…)`

Watch pinned models for sunset and propose migrations off them.

### `LineageRecord(**data)`

The full provenance chain for one source.

### `LoopResult(**data)`

Outcome of one improvement-loop cycle, with full provenance.

### `MPCResult(**data)`

The outcome of driving a :class:`ModelPredictivePlanner` to a verified end.

### `MPCStep(**data)`

The record of one real, committed step of a model-predictive plan.

### `MatryoshkaEmbedder(inner, dimensions)`

Matryoshka (MRL) dimension truncation over any embedder.

### `MemoryEngine(store=…, write_policy=…, decay_lambda=…, min_confidence=…, graph_enabled=…, embedder=…, vector_weight=…, retention_weight=…, ttl_days=…, audit=…, consent_ledger=…)`

Layered, guarded, decaying long-term memory with hybrid recall.

### `MemoryItem(**data)`

A scoped, scored, decaying memory.

### `MemoryScope(*args, **kwds)`

Enum where members are also (and must be) strings

### `MemoryType(*args, **kwds)`

Enum where members are also (and must be) strings

### `MockImageProvider(size=…, default_model=…)`

Deterministic offline image provider.

### `MockSpeechProvider(sample_rate=…)`

Deterministic offline TTS: a real WAV whose length scales with the text.

### `ModelCard(**data)`

Machine-readable documentation for a single model.

### `ModelCascade(**data)`

An ordered cheap→strong model ladder for confidence-based escalation.

### `ModelPredictivePlanner(model, actions=…, goal_value=…, horizon=…, beam_width=…, max_real_steps=…, goal_bar=…, length_penalty=…, reward_weight=…, action_cost=…, cost_weight=…, require_calibrated=…)`

Plan by searching imagined rollouts under a :class:`WorldModel` (MPC).

### `ModelRegistry(profiles=…, version=…)`

A catalog of :class:`ModelProfile` keyed by exact model id.

### `Objective(text=…, **data)`

What the application is trying to accomplish.

### `OpenAIFineTuneBackend(provider)`

Drives the OpenAI fine-tuning API over an :class:`OpenAIProvider`.

### `OutputContract(**data)`

The full output contract.

### `OutputSchema(name, json_schema, model=…)`

A named, provider-agnostic structured-output contract.

### `Pack(**data)`

A domain bundle: prompt config + schema + policies + evaluators + evals.

### `PlanRepairer(max_repairs=…, budget_shock_fraction=…)`

Repairs a running :class:`StepDAG` in place. Deterministic and offline.

### `PluginInfo(**data)`

A discovered plugin entry point and its registration status.

### `PoisoningDetector(threshold=…, min_authority=…, min_provenance=…, classifier=…, injection_detector=…)`

Flag likely-poisoned retrieved evidence from authority/provenance signals.

### `PolicySet(**data)`

Deterministic per-run policies (policies).

### `Predict(sig, provider, model, temperature=…, prompt_spec=…, max_output_tokens=…)`

Execute a signature against a provider with full output validation.

### `PredictedStep(**data)`

The world model's prediction for one ``(observation, action)``.

### `PrometheusExporter(namespace=…)`

Scrape-friendly Prometheus metrics for the served plane.

### `PromptSpec(**data)`

Declarative prompt definition compiled to an AST.

### `ProvenanceManifest(**data)`

A C2PA-style content-provenance manifest for AI-generated output.

### `Purpose(*args, **kwds)`

Why personal data is processed (GDPR Art. 5(1)(b) purpose limitation).

### `Rail(**data)`

One programmable rail.

### `RealtimeSession(backend=…, config=…, tool_dispatcher=…)`

A bidirectional realtime session.

### `ReasoningController(policy=…, trace_cache=…)`

Pick a thinking effort + token budget per step from task + budget signals.

### `ReasoningDecision(**data)`

The record of one reasoning-effort pick — stamped on the trace.

### `ReasoningPolicy(**data)`

The effort policy: difficulty bands, guardrails, and reuse behavior.

### `ReasoningTrace(**data)`

One cached reasoning trace: how much thinking a warm prefix already cost.

### `ReasoningTraceCache(max_entries=…, max_resident_bytes=…)`

Bounded LRU of reasoning traces under a resident-memory budget.

### `ReflectiveOptimizer(evaluate_variant, weights=…, gates=…, max_cost_per_case=…, objectives=…, reflector=…, constraints=…, prefer=…)`

GEPA-style reflective prompt optimizer.

### `RelevanceDecay(**data)`

Exponential intra-run relevance decay (the memory recency model, per run).

### `ResearchAgent(app, budget=…, strategies=…, judge=…, min_support=…, require_citations=…)`

Search → read → reflect → verify → synthesize, cited and budget-bounded.

### `ResearchBudget(**data)`

Explicit breadth/depth/source/token bounds for one research run.

### `ResearchReport(**data)`

The cited, budgeted, eval-scored output of a research run.

### `ResidencyPolicy(**data)`

Pin allowed provider regions and refuse egress to others.

### `RetrievalEvaluator(k_values=…)`

Score a retriever against a :class:`RetrievalGoldenSet` on the IR metrics.

### `RetrievalGoldenSet(**data)`

A fixed query set scored against a fixed corpus.

### `RewardModel(rewards, success_threshold=…, name=…)`

Compose verifiable rewards into one dense, confidence-weighted signal.

### `RewardVerifier(reward, name=…)`

Score candidates with any :class:`~vincio.optimize.rewards.VerifiableReward` or :class:`~vincio.optimize.rewards.RewardModel`.

### `RiskTierClassifier(purpose=…, domains=…, prohibited_practices=…, human_oversight=…, interacts_with_humans=…, generates_content=…)`

Place an app into the EU AI Act risk tiers from its declared profile.

### `Router(entries, strategy=…, registry=…, price_table=…, budget_usd=…, guard_capabilities=…, events=…)`

A registry-backed router: pick the cheapest / fastest / least-busy *capable* model per request, inside your own process and audit boundary.

### `RunConfig(**data)`

Per-run overrides (A2).

### `RunHandle(task)`

Handle to an in-flight run started by :meth:`ContextApp.submit`.

### `RunResult(**data)`

Result of a ContextApp run.

### `RunStreamEvent(**data)`

Event emitted by the streaming run flow (``ContextApp.astream``).

### `ScheduleResult(**data)`

The aggregate result of one scheduling pass.

### `SchemaRouter(default=…)`

Routes a run (or a piece of structured data) to one of several schemas.

### `ScopedMemory(engine, scope, owner_id)`

Mem0-style handle bound to one owner: ``engine.for_user("u1")``.

### `SearchBudget(**data)`

Bounds one search: candidate cap, optional cost cap, optional deadline.

### `SearchResult(**data)`

The outcome of a search: the winner, every candidate, and why it stopped.

### `SelfImprovementController(app, policy=…, dataset=…, golden=…, registry=…, prompt_name=…)`

Drive a :class:`SelfImprovementPolicy` as one streaming controller.

### `SelfImprovementPolicy(**data)`

One declarative, governed contract for continual self-improvement.

### `Send(node, state=…, **data)`

Dynamic fan-out instruction for map-reduce super-steps.

### `ShadowProvider(primary, candidate, candidate_model=…, block=…, price_table=…, recorder=…, events=…, max_observations=…)`

Return the primary's answer; dual-dispatch the candidate for offline diff.

### `ShardedIndex(shards, router=…, max_concurrency=…)`

Routes writes across shards and merges parallel reads (Index protocol).

### `Signature(**data)`

Base class for typed input → output signatures.

### `SpeechProvider()`

Helper class that provides a standard way to create an ABC using inheritance.

### `SpeechRequest(**data)`

!!! abstract "Usage Documentation" [Models](../concepts/models.md)

### `StabilityLevel(*args, **kwds)`

Stability contract for a public symbol.

### `StateGraph(name=…, state_schema=…, reducers=…, defaults=…)`

Build-time graph definition; ``compile()`` produces the runnable form.

### `SubgraphScheduler(workers=…, store=…, coordinator=…, lease_ttl_s=…, budget=…, deadline_s=…, clock=…)`

Runs independent sub-graphs concurrently under a fair-share budget + SLA.

### `SubgraphTask(graph, input=…, id=…, thread_id=…, weight=…)`

One independent sub-graph to schedule.

### `SwapGate(app, metrics=…, quality_metric=…, gates=…, alpha=…, drift_threshold=…, behavior_threshold=…, repeats=…, flake_quarantine=…)`

Gate a model/provider change on replayed golden traces + an eval/cost/ latency/behavioral diff with statistical backing.

### `SwapVerdict(**data)`

PASS / FAIL verdict for promoting a model into the live path.

### `SystemCard(**data)`

Documentation for the whole system: model + retrieval + memory + safety.

### `TaintedValue(value, label=…, sources=…)`

A value carried together with its :class:`TrustLabel` and provenance.

### `TaskType(*args, **kwds)`

Task taxonomy used by the input router.

### `TestTimeSearch(generate, verifier=…, budget=…)`

Verifier-guided test-time search bounded by a :class:`SearchBudget`.

### `TimerService(graph, clock=…)`

Resumes due timers and delivers events for one compiled graph.

### `ToolEnvironment(name, initial_state, tools, task, instructions=…)`

A deterministic, in-process environment whose world is a dict mutated by tools.

### `TrainingSet(**data)`

A curated, grounded fine-tuning corpus.

### `TrajectoryAdvantage(value_fn, include=…, max_players=…)`

Attribute a trajectory's outcome reward to the steps that earned it.

### `TrajectoryOptimizer(reward_model, policy=…, learning_rate=…, kl_max=…, iterations=…, group_normalize=…, min_reward_improvement=…)`

GRPO-style on-policy update over a deterministic policy, safety-gated.

### `Transition(**data)`

One recorded ``(observation, action) → next_observation`` step.

### `TrustLabel(*args, **kwds)`

A typed information-flow label on a value or context candidate.

### `TwoStageIndex(embedder=…, coarse_dims=…, quantization=…, rerank_factor=…)`

Matryoshka + quantized coarse search, full-precision exact rerank.

### `UserInput(**data)`

Structured task input.

### `VerifiableReward()`

Base contract: map a :class:`RewardSample` to a :class:`RewardSignal`.

### `Verifier(*args, **kwargs)`

Scores a candidate answer or trajectory. Reuse an existing critic via the adapters in this module rather than implementing this directly.

### `VerifierScore(**data)`

A verifier's verdict on one candidate: a value, a confidence, a reason.

### `VincioConfig(**data)`

Top-level project configuration.

### `VincioDeprecationWarning(*args, **kwargs)`

Emitted when a deprecated Vincio API is used.

### `VincioError(message, details=…, hint=…, docs_url=…)`

Base class for all Vincio errors.

### `VincioExperimentalWarning(*args, **kwargs)`

Emitted on first use of an :func:`experimental` API.

### `VoiceAgent(app, backend=…, config=…, research=…, memory_os=…, rails=…, owner_id=…, research_tool=…, **backend_kwargs)`

A grounded, remembering, guarded voice session over a :class:`ContextApp`.

### `WorkerPoolBackend(workers=…, store=…, coordinator=…, lease_ttl_s=…)`

In-process reference distributed executor — lock-free, durable, fan-out.

### `Workflow(name, tracer=…, approval_fn=…)`

A deterministic, resumable DAG of steps.

### `WorldModel(transitions=…)`

A deterministic, offline-learned dynamics model of a tool environment.

## Functions

### `InputField(desc=…, default=…, **kwargs)`

Declare a signature input field.

### `OutputField(desc=…, **kwargs)`

Declare a signature output field.

### `attribute_regression(app, dataset, factors, metric=…, aggregate=…, repeats=…)`

Attribute a metric regression to the changed ``factors`` by Shapley counterfactual replay — the convenience entry point behind a failing gate.

### `available_packs()`

Names of all packs that can be loaded (built-in + installed plugins + registered).

### `compose(*steps, name=…, tracer=…)`

Compose steps left to right: ``compose(a, b) == compose(a) | b``.

### `default_model_registry()`

Process-wide registry, seeded from the built-in catalog plus the ``VINCIO_MODEL_REGISTRY`` overlay (if set). Constructed lazily and cached.

### `deprecated(since, removed_in, alternative=…)`

Mark a function or class as deprecated.

### `discover_plugins(groups=…, entry_points=…)`

List installed Vincio plugins without registering them.

### `enable_rich_reprs()`

Attach ``_repr_html_`` / ``_repr_markdown_`` to the core result types.

### `experimental(since, note=…)`

Mark a function or class as experimental (no stability guarantee).

### `generate_redline(original, revised, format=…, title=…)`

Generate a tracked-change redline between two texts.

### `installed_plugins()`

All installed Vincio plugins across every group (alias for discovery).

### `load_benchmark(name, **kwargs)`

Construct a benchmark adapter by name.

### `load_config(path=…, overrides=…)`

Load configuration from a file (or discover it), env vars, and overrides.

### `load_pack(name)`

Load a pack by name (built-in modules import lazily; installed plugin packs register via the ``vincio.packs`` entry-point group on first miss).

### `load_plugins(groups=…, entry_points=…)`

Register every compatible installed plugin into its registry.

### `make_finetune_backend(provider)`

Build the right fine-tune backend for a provider instance.

### `make_retail_environment(task_id=…)`

A τ-bench-style retail world: orders mutated by tools, verified by end state.

### `model_swap_regression(app, dataset, baseline_model=…, candidate_model, metrics=…, quality_metric=…, alpha=…, repeats=…, flake_quarantine=…, flake_threshold=…, slice_prefix=…)`

Swap only the model on a fixed dataset and report a statistically grounded regression analysis (the body of ``vincio eval regress``).

### `provider_trainer(backend, registry=…, inherit_from=…, pricing=…, suffix=…, fmt=…, poll_interval_s=…, max_polls=…)`

Build an *executed* :data:`StudentTrainer` over a fine-tune backend.

### `record_transitions(env, action_sequences, include_failures=…)`

Drive ``env`` through each action sequence, recording every tool step.

### `retrieval_regression(search_fn, golden, config, store=…, metrics=…, gates=…, top_k=…, alpha=…, min_delta=…, k_values=…)`

Evaluate ``config`` on ``golden``, record an artifact, and gate vs. baseline.

### `serve_viewer(store, host=…, port=…)`

Start the served observability plane over ``store`` (opt-in, self-hosted).

### `signature(spec, instructions=…, name=…)`

Build a Signature type from a DSPy-style string spec::

### `sleep_for(state, seconds, clock=…)`

Pause the graph for ``seconds`` of wall-clock time, durably.

### `sleep_until(state, when, clock=…)`

Pause the graph until ``when`` (a datetime or ISO string), durably.

### `stability_of(obj)`

Return the stability record for ``obj``.

### `task_goal_value(checks)`

A goal-value function: the fraction of an environment task's checks an observation's state satisfies (the planner's default verifier).

### `verify_containment(events)`

Check ``untrusted ⇒ no unapproved capability`` over recorded events.

### `verify_erasure_proof(proof, signer=…)`

Verify a proof's content binding and (if present) its signature.

### `wait_for_event(state, name)`

Pause the graph until an event named ``name`` is delivered; return its payload.

## Values

### `API_VERSION`

str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str
