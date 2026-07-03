# Reference: public API index

This page is generated from `vincio.__all__`, the exact set of names
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) applies to,
with each symbol's signature and docstring summary. It is gated for
docstring coverage: no public symbol ships undocumented. For the curated,
grouped narrative see [api.md](api.md).

**552** public symbols.

## Classes

### `A2ANegotiator(client, member_id, role=…)`

A negotiating :class:`Party` whose moves are made by a remote A2A agent.

### `AIBOM(**data)`

An AI bill of materials, serializable as CycloneDX 1.6 JSON.

### `ActionOutcome(**data)`

The result of one full perceive → gate → act → verify → undo cycle.

### `ActionPolicy(**data)`

The pre-gate rail: what is in scope and what needs approval.

### `AdaptationResult(**data)`

The outcome of one gated on-device adaptation cycle.

### `AdaptedProvider(base, adapter, embedder=…)`

Apply a :class:`LocalAdapter` to any base provider at generation time.

### `AdapterGate(metric=…, regression_threshold=…, require_significance=…, min_samples=…, alpha=…)`

No-regression gate for an on-device adapter, the model-swap gate's analog.

### `AdapterRegistry(directory=…)`

A versioned, reversible store of on-device adapters.

### `AdaptiveSampler(cases, sample, gate, metric=…, budget, seed_samples=…, confidence=…, weights=…)`

Decide a mean-aggregate gate with the fewest samples by allocating the budget to the highest-variance cases and stopping as soon as the verdict is certain.

### `AdmissionConfig(**data)`

How a counterparty's standing maps to a bounded exposure posture.

### `AdmissionDecision(**data)`

A bounded, offline-verifiable exposure posture for one counterparty.

### `AdmissionPolicy(config=…)`

A graduated-exposure policy over the standing the fabric already earns.

### `AdmissionVerification(**data)`

The (non-raising) outcome of verifying an admission decision offline.

### `AgentCredential(**data)`

A signed, verifiable claim an org makes about an agent.

### `AgentDirectory(allow_list=…, audit=…, principal=…)`

A governed, discoverable directory of agents across A2A / ACP / MCP.

### `AgentIdentity(keyring, name=…)`

A portable agent identity: a keyring, its document, and an accountable signer.

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

### `AnalysisAgent(app, budget=…, engine=…, propose_followups=…, max_followups=…)`

Plan → query → inspect → refine → synthesize, cited and budget-bounded.

### `AnalysisResult(**data)`

A bounded, multi-step analysis rendered as a **cited analytical narrative**.

### `AnnexIVBuilder(classifier=…)`

Render EU AI Act **Annex IV** technical documentation as a cited document.

### `ApprovalRecord(**data)`

A tool-approval decision made during a turn.

### `ArithmeticVerifier()`

Recomputes arithmetic equalities stated in an answer.

### `Assistant(app, user_id=…, tenant_id=…, session_id=…, memory_writeback=…, auto_approve=…, on_approval=…, feature=…)`

A multi-turn conversational session over a :class:`ContextApp`.

### `AssistantTurn(**data)`

The outcome of one conversational turn.

### `AssuranceCase(**data)`

A signed, content-bound assurance argument the platform keeps continuously valid.

### `AssuranceReport(**data)`

The content-bound outcome of re-checking a case against current evidence.

### `AttestationExchange(client, peer_id=…)`

A peer reached over A2A that an importer pulls signed artifacts from.

### `AttestationRevocation(**data)`

A signed, offline-verifiable withdrawal of a prior attestation by its hash.

### `AutoCurriculum(tasks, rails=…, governance=…, world_model=…, search=…, max_tasks=…)`

Propose the next frontier tasks, gated by rails and the governance verifier.

### `BatchRunner(backend, price_table=…, tracer=…, discount=…, poll_interval_s=…, timeout_s=…, clock=…)`

Submit a batch, poll it to completion, reconcile, and cost-track.

### `BehaviorEvent(**data)`

One observable step in an agent's trajectory.

### `BehaviorSpec(**data)`

A temporal-logic-lite property over an event trajectory, as plain data.

### `BenchmarkAdapter(tasks=…, fixture_path=…)`

Base contract for a leaderboard adapter.

### `BenchmarkDataset(**data)`

A pinned set of :class:`~vincio.evals.benchmarks.BenchmarkTask`s and its provenance tier ceiling.

### `BenchmarkRegistry(with_builtins=…)`

A niche-grouped catalog of :class:`BenchmarkSpec`s.

### `BenchmarkSpec(**data)`

One catalog entry: a benchmark, the adapter that scores it, and its provenance.

### `BenchmarkSuite(registry=…, concurrency=…, seed=…, checkpoint_dir=…)`

Run benchmarks over a model or app, deterministically and resumably.

### `BeneficiaryClaim(**data)`

One beneficiary's bounded claim on the poster's held capital.

### `BindingCandidate(**data)`

One ranked candidate for a capability binding, the decision's evidence.

### `BindingWeights(**data)`

How a candidate's signals combine into one ranking score.

### `Blackboard(event_bus=…)`

Versioned, author-attributed shared memory for agent teams.

### `BootstrapFinetune(evaluate_model, quality_metric=…, min_quality_ratio=…, gates=…, trainer=…, swap_gate=…, dedupe_embedder=…)`

Teacher → student distillation with a gated quality hold.

### `Budget(**data)`

Hard resource limits for a run (budgets, termination).

### `BudgetManager(ledger, events=…)`

Enforces :class:`CostBudget`/:class:`EnergyBudget`\ s and detects spend anomalies.

### `BundleRecord(**data)`

One governed, content-bound entry in the community index.

### `CalibrationExample(**data)`

One labelled near-miss observation used to calibrate the threshold.

### `CalibrationReport(**data)`

The verdict of :meth:`WorldModel.calibrate`, the model's planning weight.

### `CanaryRouter(primary, candidate, percent=…, candidate_model=…, score_fn=…, min_samples=…, window=…, regression_threshold=…, on_rollback=…, prompt_registry=…, prompt_name=…, events=…)`

Ramp a percentage of live traffic onto a candidate, with auto-rollback.

### `CanarySpec(**data)`

How a candidate is qualified before it is deployed live.

### `CapabilityBinder(directory, reputation=…, settlement_book=…, weights=…, principal=…)`

Resolves a capability-declaring saga step to a participant at dispatch time.

### `CapabilityBroker(secret=…, default_ttl_s=…)`

Mints and verifies :class:`CapabilityToken`\ s from the user's authority.

### `CapabilityToken(**data)`

An unforgeable, capability-scoped grant minted from the user's request.

### `CausalAttributor(app, dataset, factors, metric=…, aggregate=…, repeats=…, concurrency=…)`

Attribute a metric delta to the components a release changed, by Shapley counterfactual replay over the dataset.

### `CellCitation(**data)`

A reference to one source cell an answer rests on.

### `CellRef(**data)`

A reference to one source cell a series value came from.

### `Certificate(**data)`

A typed, content-bound, offline-verifiable proof over an answer.

### `CertificationReport(**data)`

The signed, content-bound certificate that an app is fit for production.

### `Chart(**data)`

A rendered chart, **content-bound and data-bound**.

### `ChartSpec(**data)`

A spec-driven chart definition: title, mark, channel encoding, the plotted columns, and the **values** it depicts (a projection of the source result onto the encoded columns). :meth:`to_vega_lite` renders it as a portable, embedded-data Vega-Lite v5 spec a consumer can render with any Vega-Lite runtime.

### `ChartType(*args, **kwds)`

The closed mark vocabulary a chart declares — the deterministic subset of Vega-Lite marks that also rasterizes cleanly through matplotlib.

### `Check(**data)`

One kernel's verdict on an answer.

### `Choreography(saga, participants, coordinator=…, store=…, audit=…, events=…, signer=…, binder=…, clock=…, raise_on_compensation_failure=…)`

Drives a :class:`~vincio.choreography.saga.Saga` across organizations.

### `CircuitBreaker(inner, failure_threshold=…, min_calls=…, window=…, latency_threshold_ms=…, cooldown_s=…, half_open_max=…, events=…, clock=…)`

Per-provider circuit breaker with half-open probing.

### `CitationContract(**data)`

Field/claim-level citation requirements for a cited report.

### `CitationVerifier(evidence=…)`

Checks every verifiable claim in an answer is entailed by cited evidence.

### `CitedReportBuilder(entailment=…, audit_log=…, tenant_id=…)`

Resolve citations, verify per-claim support, render a cited report.

### `CitedSeries(**data)`

A named numeric series bound to the source cells it was read from.

### `Claim(**data)`

A node in the assurance argument: a statement, its decomposition, its evidence.

### `ClaimStatus(**data)`

The re-derived verdict for one claim and its subtree.

### `CollateralLedger(**data)`

A poster's cross-pool rehypothecation view, a bounded re-use guard.

### `CollateralLedgerVerification(**data)`

The (non-raising) outcome of verifying a collateral ledger offline.

### `CollateralPool(**data)`

A counterparty's single posted stake backing many contracts, a margin account.

### `CollateralPoolVerification(**data)`

The (non-raising) outcome of verifying a collateral pool offline.

### `CommunityRegistry(allow_list=…, audit=…, signer=…, principal=…, require_signature=…, index=…)`

A governed, signed, audited index of community packs and skills.

### `CompletenessProof(**data)`

A signed, offline-verifiable completeness check over a liability attestation.

### `CompletenessVerification(**data)`

The (non-raising) outcome of verifying a completeness check offline.

### `ComplianceFramework(*args, **kwds)`

A governance framework whose controls Vincio maps onto.

### `ComplianceReport(**data)`

A coverage matrix across the mapped frameworks.

### `CompositeVerifier(verifiers)`

Runs an ordered set of verifiers and folds their checks into one certificate.

### `ComputerEnvironment(backend, app=…, policy=…, approve=…, auto_undo=…, max_steps=…)`

A grounded, verified, reversible computer-use action plane.

### `ComputerRun(**data)`

The outcome of driving a policy through the action plane to a goal.

### `ComputerTask(**data)`

A computer-use goal: a natural-language instruction plus a declarative end-state verifier and an action budget. The verifier reads the same :class:`~vincio.evals.environment.StateCheck` paths an environment oracle does, so a run's success is verifiable end-state, not turn-by-turn plausibility.

### `ConsentLedger(store=…, audit=…, default_allow=…)`

Records and checks consent, binding data to a purpose + lawful basis.

### `Constraint(text=…, **data)`

!!! abstract "Usage Documentation" [Models](../concepts/models.md)

### `ConstraintVerifier(constraints=…)`

Checks a candidate assignment satisfies a set of typed constraints.

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

### `ContinualAdaptation(app, policy=…, dataset=…, registry=…, embedder=…, base_model=…, trainer=…)`

Drive continual on-device adaptation as a streaming, gated loop.

### `ContinuousImprovementController(app, metrics=…, golden=…, registry=…, prompt_name=…, monitor=…, sustain=…, cooldown_s=…, eval_budget=…, quality_floor=…, reoptimize=…, gates=…, clock=…)`

Drive gated re-optimization / re-eval / rollback from live signals.

### `Contract(**data)`

A signed, audited, offline-verifiable agreement over typed terms.

### `ContractFulfillment(**data)`

Whether delivered work met the contract's terms, the enforcement verdict.

### `ContractTerms(**data)`

The typed, negotiated terms of an agreement.

### `ContractVerification(**data)`

The (non-raising) outcome of verifying a contract offline.

### `Contribution(**data)`

One member's privacy-preserving federated update, numeric, no raw traffic.

### `ContributionBuilder(embedder=…, privacy=…)`

Build a :class:`Contribution` from a member's local data, never its text.

### `ControllerDecision(**data)`

The record of one controller evaluation, stamped on the audit chain.

### `CorrelationClaim(**data)`

A stated correlation between two cited series, optionally asserting causation.

### `CorrelationVerifier(claims=…)`

Recomputes a correlation and refutes correlation-stated-as-causation.

### `CostAwareSelector(models, registry=…, quality_floor=…, events=…)`

Picks the cheapest capable model per action, escalating on low confidence.

### `CostBudget(**data)`

A spend limit on a scope, with an enforcement action on breach.

### `CostLedger(price_table=…, store=…, max_events=…)`

In-process append-only ledger of attributed cost events.

### `Counterexample(**data)`

A concrete, minimal state that violates an invariant.

### `CredentialVerification(**data)`

The (non-raising) outcome of verifying an agent credential offline.

### `CreditorRecovery(**data)`

One creditor's outcome in an :class:`InsolvencyResolution` waterfall.

### `Crew(name=…, process=…, blackboard=…, tracer=…, manager_provider=…, manager_model=…, max_rounds=…, concurrency=…, cost_tracker=…, cost_ledger=…)`

A multi-agent team that collaborates over a shared blackboard.

### `CrossOrgEngagement(app, buyer=…, seller=…, scope=…, coordinator=…)`

A purely-compositional facade threading the whole cross-org fabric in one call-path.

### `CultivationResult(**data)`

The content-bound, offline-verifiable outcome of a cultivation run.

### `Cultivator(app=…, curriculum, library=…, held_out=…, rails=…, governance=…, search=…, min_capability_gain=…, tolerance=…, prune=…, record=…)`

Drive the cultivation loop over a :class:`LearnedSkillLibrary`.

### `CurriculumProposal(**data)`

A content-bound, offline-verifiable curriculum round.

### `CurriculumTask(**data)`

A candidate objective: a deterministic environment plus its success oracle.

### `CustodyAttestation(**data)`

A signed, offline-verifiable proof-of-reserves over a poster's held capital.

### `CustodyAttestationVerification(**data)`

The (non-raising) outcome of verifying a custody attestation offline.

### `CycleReport(**data)`

What one cultivation cycle proposed, learned, promoted, and demoted.

### `DataCatalog(datasets=…)`

A named set of registered :class:`~vincio.data.Dataset`\s a query grounds against and executes over.

### `DataEncoder(delimiter=…, include_name=…, include_count=…, include_types=…, include_units=…, exemplars=…, max_rows=…)`

Render tabular data header-once in a compact, token-oriented form.

### `DataEngagement(app, dataset=…, question=…, analyst=…)`

A purely-compositional facade threading the whole data plane in one call-path.

### `DataEngagementSignature(**data)`

One party's signature over a data-engagement narrative's content hash.

### `DataEngagementVerification(**data)`

The (non-raising) outcome of verifying a data engagement offline.

### `DataNarrative(**data)`

A signed, content-bound, hash-chained narrative of a whole data engagement.

### `DataQualityRails(constraints=…, detect_anomalies=…, anomaly_threshold=…, anomaly_action=…, max_examples=…, pii_detector=…, secret_scanner=…, injection_detector=…)`

Screen tabular data deterministically against a set of column constraints, with optional numeric anomaly detection.

### `DataQualityReport(**data)`

The outcome of screening a dataset. ``allowed`` is false when any blocking rule fired; the violations carry the detail.

### `DataStage(**data)`

One step of a data engagement, bound into the narrative's hash chain.

### `Dataset(**data)`

!!! abstract "Usage Documentation" [Models](../concepts/models.md)

### `DatasetProfile(**data)`

A dataset's deterministic, fixed-size column profile.

### `Delegation(**data)`

A signed grant of bounded authority from one identity to another.

### `DelegationChain(**data)`

An ordered chain of delegations from a principal down to an acting agent.

### `DelegationChainVerification(**data)`

The (non-raising) outcome of verifying a delegation chain offline.

### `DelegationVerification(**data)`

The (non-raising) outcome of verifying one delegation offline.

### `DeployResult(**data)`

Outcome of a canary-gated prompt/policy deployment.

### `Discharge(**data)`

A signed, content-bound release of part of what a poster owes one creditor.

### `DischargeVerification(**data)`

The (non-raising) outcome of verifying a liability discharge offline.

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

### `EdgeEnvironment(**data)`

A report of the runtime the edge core is executing on.

### `EdgeManifest(**data)`

The static WASM-buildability certificate for the edge core.

### `EdgeParityReport(**data)`

The result of verifying the edge build is the same library, not a fork.

### `EdgeProfile(**data)`

A bounded resident-memory and latency profile for a constrained target.

### `EdgeRequest(**data)`

A self-contained context-engineering request for the edge runtime.

### `EdgeResult(**data)`

The outcome of one edge compile.

### `EdgeRuntime(profile=…, rails=…)`

A bounded, in-process context-engineering runtime for the edge.

### `EnergyBudget(**data)`

An energy/carbon limit on a scope, refused on breach.

### `EnergyEstimate(**data)`

A run (or call)'s estimated energy and carbon, with its breakdown.

### `EnergyIntensityTable(**data)`

Resolves a model + region into an energy/carbon estimate.

### `EnergyProfile(**data)`

Per-model energy intensity, in watt-hours per million tokens.

### `EnergyReport(**data)`

Estimated energy + carbon rolled up by dimension.

### `EngagementNarrative(**data)`

A signed, content-bound, hash-chained narrative of a whole cross-org engagement.

### `EngagementSignature(**data)`

One party's signature over an engagement narrative's content hash.

### `EngagementStage(**data)`

One step of a cross-org engagement, bound into the narrative's hash chain.

### `EngagementVerification(**data)`

The (non-raising) outcome of verifying an engagement narrative offline.

### `Environment(*args, **kwargs)`

The stateful-environment contract: ``reset`` / ``step`` / ``observe`` / ``verify``.

### `EnvironmentSimulator(max_steps=…)`

Drive an agent *policy* through an :class:`Environment` to a verified end state.

### `EquivocationProof(**data)`

A signed, offline-verifiable proof that a poster signed two conflicting liability roots.

### `EquivocationProofVerification(**data)`

The (non-raising) outcome of verifying a liability equivocation proof offline.

### `ErasureProof(**data)`

A signed, content-bound manifest of exactly what an erasure removed.

### `ErasureResult(**data)`

Outcome of a right-to-erasure-by-source sweep.

### `Escrow(**data)`

Posted collateral bound to a contract, held, released, or forfeited.

### `EscrowConfig(**data)`

How a breach's measured shortfall maps to a bounded forfeiture.

### `EscrowVerification(**data)`

The (non-raising) outcome of verifying an escrow offline.

### `EventCitation(**data)`

A reference to one source *event* cell a windowed answer rests on.

### `EventPattern(**data)`

A predicate that matches a :class:`BehaviorEvent`.

### `Evidence(**data)`

A platform verdict bound by hash to discharge one sub-claim.

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

### `FederatedContribution(**data)`

One organization's aggregated, source-bound contribution to a finding.

### `FederatedDataEngagement(app, query=…, coordinator=…, layer=…)`

A governed, compositional facade for analytics across organizations.

### `FederatedFinding(**data)`

A reconciled cross-org answer for one metric and one dimension group.

### `FederatedImprovement(app, policy=…, dataset=…, registry=…, embedder=…, base_model=…, reputation=…)`

Drive one gated, privacy-preserving federated round for the adopting member.

### `FederatedMember(org, app, table=…, layer=…, region=…, subject=…)`

One organization participating in a federated analytics engagement.

### `FederatedNarrative(**data)`

A signed, content-bound, hash-chained narrative of a federated engagement.

### `FederatedPolicy(**data)`

The opt-in contract for one gated federated-improvement round.

### `FederatedQuery(**data)`

The shape of one governed metric run across organizations.

### `FederatedRoundResult(**data)`

The outcome of one gated federated-improvement round.

### `FederatedSignature(**data)`

One party's signature over a federated narrative's content hash.

### `FederatedStage(**data)`

One step of a federated engagement, bound into the narrative's hash chain.

### `FederatedSubspace(**data)`

The fleet-consensus low-rank subspace distilled from a secure aggregation.

### `FederatedVerification(**data)`

The (non-raising) outcome of verifying a federated engagement offline.

### `FertilityTracker(model=…, baseline_language=…)`

Track tokens-per-word per language to surface the non-English token tax.

### `Figure(**data)`

A chart or table embedded in a cited report, **data-bound** to its source.

### `Flow(provider=…, model=…, name=…, output_schema=…, app=…, config=…)`

An immutable, fluent pipeline that lowers to one governed run packet.

### `ForecastClaim(**data)`

A stated projection from a declared deterministic forecast model.

### `ForecastVerifier(claims=…)`

Re-runs a declared deterministic forecast over the cited series and checks it.

### `FrontierEstimate(**data)`

Where a task sits relative to current competence.

### `GGUFProvider(model_path=…, llama=…, n_ctx=…, embedding=…, lora_path=…, lora_scale=…, **kwargs)`

Native in-process GGUF / llama.cpp provider with on-device embedding.

### `GatheredReputation(subject, visits, attestations, revocations, reputation, duplicates=…)`

A current prior assembled by pulling signed artifacts from a set of peers.

### `GoldenRegressionSuite(path=…, name=…)`

A held-out, *growing* golden regression set with per-case provenance.

### `GovernanceVerifier(invariants=…, audit_log=…, claim_generator=…)`

Proves governance invariants by exhaustive bounded model checking.

### `Grant(**data)`

A bounded grant of authority: the capabilities, budget, expiry, and audience.

### `GuardedBanditRouter(entries, bandit=…, safe_model=…, reward_fn=…, context_fn=…, epsilon=…, alpha=…, context_dim=…, seed=…, regret_budget=…, rollback_margin=…, store=…, app_name=…, events=…)`

A live routing bandit with a safety floor, regret tracking, and auto-rollback.

### `HTNDomain(**data)`

A library of operators and methods the planner decomposes against.

### `HealthAwareFailover(entries, guard_capabilities=…, registry=…)`

Failover chain that tries healthy providers first.

### `HistoryConsistencyProof(**data)`

A signed, offline-verifiable proof a poster's liability history is monotone over time.

### `HistoryConsistencyProofVerification(**data)`

The (non-raising) outcome of verifying a liability history-consistency proof offline.

### `HistoryConsistencyReport(**data)`

The outcome of walking a set of liability snapshots for cross-time monotonicity.

### `IdentityDocument(**data)`

A signed, content-bound description of an agent identity.

### `IdentityVerification(**data)`

The (non-raising) outcome of verifying an identity document offline.

### `ImageGenRequest(**data)`

!!! abstract "Usage Documentation" [Models](../concepts/models.md)

### `ImageProvider()`

Abstract image generation/editing provider.

### `ImprovementLoop(app, registry=…, tracker=…, metrics=…, weights=…, gates=…, max_cost_per_case=…, experiment=…, prompt_name=…, concurrency=…, optimizer=…, strategy=…, reflector=…, golden_suite=…)`

Runs the trace → dataset → eval → optimize → promote cycle on an app.

### `Incident(**data)`

A signed observation that a sub-claim no longer holds in production.

### `InclusionProof(**data)`

An offline-verifiable proof that one creditor's claim is in a liability attestation.

### `InclusionProofVerification(**data)`

The (non-raising) outcome of verifying a liability inclusion proof offline.

### `IndexedTraceStore(path=…, percentile_window=…)`

SQLite-backed, indexed trace + cost store with pre-aggregated rollups.

### `InsolvencyBreach(**data)`

A proven shortfall: the obligations owed exceed the reserves actually held.

### `InsolvencyResolution(**data)`

A signed, offline-verifiable resolution distributing reserves across ranked liabilities.

### `InsolvencyResolutionVerification(**data)`

The (non-raising) outcome of verifying an insolvency resolution offline.

### `Instruction(text=…, **data)`

!!! abstract "Usage Documentation" [Models](../concepts/models.md)

### `IntervalClaim(**data)`

A stated interval over a cited series.

### `IntervalVerifier(claims=…)`

Recomputes a stated confidence or prediction interval from the cited series.

### `Invariant(id, statement, category, variables, predicate, explain=…)`

A formal governance property checked over a bounded, typed state space.

### `InvariantResult(**data)`

The verdict of checking one :class:`Invariant` over its whole state space.

### `IssuePreference(**data)`

A party's preference over one numeric issue.

### `IssuerTrust(**data)`

The importer's resolved trust in one issuer, pinpointed, never silent.

### `JudgeCalibrator(judge, reflector=…, kappa_bins=…, trust_threshold=…, min_kappa_gain=…)`

Tune a :class:`~vincio.evals.judges.GEvalJudge`'s evaluation steps to maximize agreement with human labels, then leave the judge calibrated.

### `JudgeEnsemble(judges, aggregate=…, disagreement_threshold=…, name=…)`

A panel of judges scored together, with disagreement surfaced as uncertainty and the panel as a whole calibrated against human labels.

### `JudgeVerifier(judge, case=…, name=…)`

Score candidates with any :class:`~vincio.evals.judges.Judge` or :class:`~vincio.evals.ensemble.JudgeEnsemble`.

### `KVPrefixPool(kv_bytes_per_token=…, max_entries=…, max_resident_bytes=…)`

Bounded tracker of cross-request shared stable-prefix KV reuse.

### `KeyAuthorization(**data)`

An offline proof that a signing key descends from an identity's genesis key.

### `KeyPool(providers, rpm=…, tpm=…, breaker=…, labels=…, max_attempts=…, base_backoff_s=…, max_backoff_s=…, seed=…, events=…, clock=…)`

Round-robin pool over multiple keys/regions of one logical provider.

### `KeyRecord(**data)`

One public key in an identity's rotation history.

### `Keyring(document, seeds)`

Holds an identity's private keys and maintains its signed rotation chain.

### `LLMLinguaCompressor(scorer=…, min_keep_ratio=…, coarse_overshoot=…)`

Token-importance compressor (callable, drop-in for ``extractive_compress``).

### `LawfulBasis(*args, **kwds)`

GDPR Article 6(1) lawful bases for processing.

### `Leaderboard(**data)`

A ranked comparison of models over a shared benchmark set.

### `LearnedSemanticCache(embedder, policy=…, calibration=…, clock=…)`

Bounded, calibrated, auditable near-miss response cache.

### `LearnedSkill(**data)`

A verified, content-addressed, versioned, composable learned procedure.

### `LearnedSkillLibrary(skills=…)`

A content-addressed library of learned skills with versioning and dedup.

### `LearningResult(**data)`

The outcome of a :class:`TrajectoryOptimizer` run.

### `LiabilityAttestation(**data)`

A signed, offline-verifiable proof-of-liabilities over a poster's total obligations.

### `LiabilityAttestationVerification(**data)`

The (non-raising) outcome of verifying a liability attestation offline.

### `LiabilityLine(**data)`

One obligation owed, backing the poster's attested total liabilities.

### `LifecycleWatcher(registry=…, warn_within_days=…, events=…)`

Watch pinned models for sunset and propose migrations off them.

### `LineageRecord(**data)`

The full provenance chain for one source.

### `LocalAdaptationPolicy(**data)`

The opt-in contract for continual on-device adaptation.

### `LocalAdapter(**data)`

A versioned, content-addressed, portable LoRA-class adapter.

### `LocalLoRATrainer(embedder=…, rank=…, gate=…, scale=…, backend=…)`

Fit a :class:`LocalAdapter` on-device from a grounded training set.

### `LoopResult(**data)`

Outcome of one improvement-loop cycle, with full provenance.

### `MPCResult(**data)`

The outcome of driving a :class:`ModelPredictivePlanner` to a verified end.

### `MPCStep(**data)`

The record of one real, committed step of a model-predictive plan.

### `MatryoshkaEmbedder(inner, dimensions)`

Matryoshka (MRL) dimension truncation over any embedder.

### `MemberReputation(**data)`

One member's reputation snapshot, its track record as an auditable number.

### `MemoryEngine(store=…, write_policy=…, decay_lambda=…, min_confidence=…, graph_enabled=…, embedder=…, vector_weight=…, retention_weight=…, ttl_days=…, audit=…, consent_ledger=…, privacy_accountant=…, privacy_mechanism=…)`

Layered, guarded, decaying long-term memory with hybrid recall.

### `MemoryItem(**data)`

A scoped, scored, decaying memory.

### `MemoryScope(*args, **kwds)`

Enum where members are also (and must be) strings

### `MemoryType(*args, **kwds)`

Enum where members are also (and must be) strings

### `MerkleStep(**data)`

One step of an :class:`InclusionProof`'s authentication path.

### `Meter(contract_id, run_id=…)`

Accumulates the usage of work delivered under one contract.

### `MeterReading(**data)`

The deterministic roll-up of a meter's accrued usage for one contract.

### `MockImageProvider(size=…, default_model=…)`

Deterministic offline image provider.

### `MockScreen(app)`

Deterministic in-process screen over a :class:`ScreenApp`, no browser, no network. Tracks the current screen, typed field values, and durable flags, and re-derives a stable :class:`ScreenState` from them, so a run is reproducible and CI-golden. Supports exact snapshot restore as an undo fallback.

### `MockSpeechProvider(sample_rate=…)`

Deterministic offline TTS: a real WAV whose length scales with the text.

### `MockVideoProvider(default_model=…)`

Deterministic offline video provider.

### `ModelCard(**data)`

Machine-readable documentation for a single model.

### `ModelCascade(**data)`

An ordered cheap→strong model ladder for confidence-based escalation.

### `ModelPredictivePlanner(model, actions=…, goal_value=…, horizon=…, beam_width=…, max_real_steps=…, goal_bar=…, length_penalty=…, reward_weight=…, action_cost=…, cost_weight=…, require_calibrated=…)`

Plan by searching imagined rollouts under a :class:`WorldModel` (MPC).

### `ModelRegistry(profiles=…, version=…)`

A catalog of :class:`ModelProfile` keyed by exact model id.

### `MonitorVerdict(**data)`

The outcome of checking one event (or a whole trajectory).

### `MonotonicityBreach(**data)`

A creditor's obligation that shrank between two snapshots without a backing discharge.

### `Negotiation(buyer, seller, budget=…, signer=…, audit=…, events=…, clock=…)`

Drives a bounded alternating-offers bargain between a buyer and a seller.

### `NegotiationBudget(**data)`

The guaranteed-termination budget for a negotiation.

### `NegotiationPosition(**data)`

A party's private stance: per-issue preferences and a concession curve.

### `NegotiationResult(**data)`

The outcome of a bounded negotiation, a deal, or a partial no-deal.

### `NettingSet(**data)`

A content-bound, offline-verifiable multilateral clearing of a fleet's books.

### `NotebookSession(engagement, auto_display=…)`

An interactive, governed data-analysis session for notebooks and REPLs.

### `Objective(text=…, **data)`

What the application is trying to accomplish.

### `Offer(**data)`

One move in a negotiation: a proposal, an acceptance, or a walk-away.

### `OmissionBreach(**data)`

A creditor's proven claim the attested liabilities omit or under-state.

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

### `PooledContract(**data)`

One contract a :class:`CollateralPool` backs, with its share and disposition.

### `PortableReputation(standings, verdicts, config, base=…, as_of=…, trust=…)`

An imported, evidence-weighted prior combined from several issuers' attestations.

### `Predict(sig, provider, model, temperature=…, prompt_spec=…, max_output_tokens=…)`

Execute a signature against a provider with full output validation.

### `PredictedStep(**data)`

The world model's prediction for one ``(observation, action)``.

### `PrivacyAccountant(default_budget=…, default_mechanism=…, orders=…, delta=…, audit=…, store=…)`

A composing, per-subject differential-privacy budget over the learning loop.

### `PrivacyBudget(**data)`

A per-subject (or default) ``(ε, δ)`` privacy ceiling.

### `PrivacyBudgetError(message, details=…, hint=…, docs_url=…)`

A learning step was refused because it would exceed a subject's DP budget.

### `PrivacyConfig(**data)`

How a contribution is made privacy-preserving before it leaves a member.

### `PrivacyDecision(**data)`

An explainable verdict on whether a proposed release fits the budget.

### `PrivacyMechanism(**data)`

One differentially-private release, as accounted against a budget.

### `PrivacyReport(**data)`

Per-subject DP budget roll-up, the privacy analogue of the cost report.

### `PrivacySpend(**data)`

One accounted privacy release for a subject, a row on the audit chain.

### `ProgramOp(**data)`

One whitelisted transform step over a list of record dicts.

### `ProgramProperty(**data)`

A declarative property a synthesized program must satisfy.

### `ProgramSpec(**data)`

The declaration of a verified transform: its ops and the properties it must hold.

### `PrometheusExporter(namespace=…)`

Scrape-friendly Prometheus metrics for the served plane.

### `PromptSpec(**data)`

Declarative prompt definition compiled to an AST.

### `ProvenanceManifest(**data)`

A C2PA-style content-provenance manifest for AI-generated output.

### `ProvenanceTier(*args, **kwds)`

How real a benchmark number is — ordered ``STATIC < RECORDED < LIVE``.

### `Purpose(*args, **kwds)`

Why personal data is processed (GDPR Art. 5(1)(b) purpose limitation).

### `QueryPlan(**data)`

A schema-grounded, read-only-verified query that has **not yet run**.

### `QueryResult(**data)`

A query's result, schema-bearing and **cell-level cited**.

### `Rail(**data)`

One programmable rail.

### `RealtimeSession(backend=…, config=…, tool_dispatcher=…)`

A bidirectional realtime session.

### `ReasoningController(policy=…, trace_cache=…)`

Pick a thinking effort + token budget per step from task + budget signals.

### `ReasoningDecision(**data)`

The record of one reasoning-effort pick, stamped on the trace.

### `ReasoningPolicy(**data)`

The effort policy: difficulty bands, guardrails, and reuse behavior.

### `ReasoningTrace(**data)`

One cached reasoning trace: how much thinking a warm prefix already cost.

### `ReasoningTraceCache(max_entries=…, max_resident_bytes=…)`

Bounded LRU of reasoning traces under a resident-memory budget.

### `ReasoningVerifier(*args, **kwargs)`

A pluggable, deterministic checker that turns an answer into checks.

### `Reconciliation(**data)`

Whether two parties' settlement records tie out, the dispute verdict.

### `ReflectiveOptimizer(evaluate_variant, weights=…, gates=…, max_cost_per_case=…, objectives=…, reflector=…, constraints=…, prefer=…)`

GEPA-style reflective prompt optimizer.

### `RelevanceDecay(**data)`

Exponential intra-run relevance decay (the memory recency model, per run).

### `RemoteParticipant(client, org_id)`

A choreography :class:`Participant` whose steps run in a remote A2A org.

### `ReputationAttestation(**data)`

A signed, offline-verifiable attestation of a counterparty's earned standing.

### `ReputationBundle(**data)`

The signed artifacts a peer holds about one subject, its reply to a query.

### `ReputationConfig(**data)`

How a member's gate track record maps to an aggregation weight.

### `ReputationError(message, details=…, hint=…, docs_url=…)`

A reputation operation could not proceed.

### `ReputationLedger(config=…, audit=…, events=…, store=…)`

A per-member, gate-earned reputation that weights federated aggregation.

### `ReputationReport(**data)`

Per-member reputation roll-up, alongside the cost and privacy reports.

### `ResearchAgent(app, budget=…, strategies=…, judge=…, min_support=…, require_citations=…)`

Search → read → reflect → verify → synthesize, cited and budget-bounded.

### `ResearchBudget(**data)`

Explicit breadth/depth/source/token bounds for one research run.

### `ResearchReport(**data)`

The cited, budgeted, eval-scored output of a research run.

### `ReserveLine(**data)`

One custodied holding backing the poster's proven reserves.

### `ResidencyPolicy(**data)`

Pin allowed provider regions and refuse egress to others.

### `Resolution(**data)`

A content-bound, offline-verifiable adjudication of a disputed contract.

### `RetrievalEvaluator(k_values=…)`

Score a retriever against a :class:`RetrievalGoldenSet` on the IR metrics.

### `RetrievalGoldenSet(**data)`

A fixed query set scored against a fixed corpus.

### `ReuseBreach(**data)`

A contract pledged across more than one pool, the same collateral, twice.

### `RewardModel(rewards, success_threshold=…, name=…)`

Compose verifiable rewards into one dense, confidence-weighted signal.

### `RewardVerifier(reward, name=…)`

Score candidates with any :class:`~vincio.optimize.rewards.VerifiableReward` or :class:`~vincio.optimize.rewards.RewardModel`.

### `RiskTierClassifier(purpose=…, domains=…, prohibited_practices=…, human_oversight=…, interacts_with_humans=…, generates_content=…)`

Place an app into the EU AI Act risk tiers from its declared profile.

### `RootCommitment(**data)`

A compact, signed digest of one liability attestation's root, for cross-creditor compare.

### `RootCommitmentVerification(**data)`

The (non-raising) outcome of verifying a liability root commitment offline.

### `RootConsistencyReport(**data)`

The outcome of comparing a set of liability roots for cross-creditor non-equivocation.

### `Router(entries, strategy=…, registry=…, price_table=…, budget_usd=…, guard_capabilities=…, events=…)`

A registry-backed router: pick the cheapest / fastest / least-busy *capable* model per request, inside your own process and audit boundary.

### `RowStream(source, schema, name=…, source_id=…)`

A lazy, re-iterable, schema-bearing handle over an out-of-core row source.

### `RunConfig(**data)`

Per-run overrides (A2).

### `RunHandle(task)`

Handle to an in-flight run started by :meth:`ContextApp.submit`.

### `RunResult(**data)`

Result of a ContextApp run.

### `RunStore(dsn=…)`

Persist and query :class:`SuiteRun`s over SQLite (default) or Postgres.

### `RunStreamEvent(**data)`

Event emitted by the streaming run flow (``ContextApp.astream``).

### `RuntimeMonitor(specs)`

Checks a :class:`BehaviorSpec` set against a trajectory, step-by-step.

### `Saga(**data)`

A cross-org compensating workflow: an ordered list of steps.

### `SagaJournal(**data)`

The durable, resumable, offline-verifiable record of one saga run.

### `SagaResult(**data)`

The outcome of a cross-org saga run, completion, a clean unwind, or a pause.

### `SagaStep(**data)`

One step of a :class:`Saga`: a forward action and its compensation.

### `ScheduleResult(**data)`

The aggregate result of one scheduling pass.

### `SchemaRouter(default=…)`

Routes a run (or a piece of structured data) to one of several schemas.

### `SchemaVerifier(schema=…)`

Checks an answer structurally conforms to a JSON schema.

### `ScopedMemory(engine, scope, owner_id)`

Mem0-style handle bound to one owner: ``engine.for_user("u1")``.

### `ScreenApp(**data)`

A deterministic, in-process app a :class:`MockScreen` drives, the offline, WebArena / OSWorld-shaped harness: named screens, form fields, click-driven transitions, and effects that set durable flags.

### `ScreenState(**data)`

A perceived snapshot of the UI, the *observe* half of the loop.

### `SearchBudget(**data)`

Bounds one search: candidate cap, optional cost cap, optional deadline.

### `SearchResult(**data)`

The outcome of a search: the winner, every candidate, and why it stopped.

### `SecureAggregator(privacy=…, rank=…, allowed_regions=…, reputation=…)`

Merge masked contributions into a :class:`FederatedSubspace`, never seeing one.

### `SelfImprovementController(app, policy=…, dataset=…, golden=…, registry=…, prompt_name=…)`

Drive a :class:`SelfImprovementPolicy` as one streaming controller.

### `SelfImprovementPolicy(**data)`

One declarative, governed contract for continual self-improvement.

### `SemanticCacheGate(quality_floor=…, scorer=…)`

Gate a learned semantic cache on replayed cases before it ships.

### `SemanticCachePolicy(**data)`

Opt-in policy for the learned semantic cache.

### `SemanticGateCase(**data)`

One probe for the cache gate: a query and its live (reference) answer.

### `SemanticLayer(**data)`

Measures, dimensions, and derived columns defined once over one table.

### `Send(node, state=…, **data)`

Dynamic fan-out instruction for map-reduce super-steps.

### `SenioritySchedule(**data)`

A signed, offline-verifiable ranking of a poster's obligations into priority tranches.

### `SeniorityTranche(**data)`

One priority rank of a :class:`SenioritySchedule`, the creditors paid at that level.

### `SeniorityVerification(**data)`

The (non-raising) outcome of verifying a seniority schedule offline.

### `SetOffStatement(**data)`

A signed, offline-verifiable statement of the obligations running both ways.

### `SetOffVerification(**data)`

The (non-raising) outcome of verifying a set-off statement offline.

### `SettlementBook(owner, signer=…, audit=…, events=…, store=…, reputation=…, book_id=…)`

An org's durable, hash-chained, offline-verifiable ledger of settlements.

### `SettlementRecord(**data)`

A signed, offline-verifiable reconciliation of delivery against a contract.

### `SettlementReport(**data)`

Per-counterparty settlement roll-up, alongside the cost report.

### `ShadowProvider(primary, candidate, candidate_model=…, block=…, price_table=…, recorder=…, events=…, max_observations=…)`

Return the primary's answer; dual-dispatch the candidate for offline diff.

### `ShardedIndex(shards, router=…, max_concurrency=…)`

Routes writes across shards and merges parallel reads (Index protocol).

### `Shield(specs, mode=…, repair=…)`

Prevents a behavioural violation before the action executes.

### `ShieldDecision(**data)`

A shield's ruling on a proposed event.

### `Signature(**data)`

Base class for typed input → output signatures.

### `SignatureCheck(**data)`

Which key verified a signature and whether it was valid at a given time.

### `SkillProvenance(**data)`

Where a learned skill came from, the audit trail of its acquisition.

### `SkillSearch(beam_width=…, max_depth=…)`

Bounded, deterministic beam search that composes the skill library.

### `SkillStep(**data)`

One step of a learned procedure: a primitive action **or** a sub-skill call.

### `Solution(**data)`

The outcome of searching for (or retrieving) a procedure for a task.

### `SolvencyProof(**data)`

A signed, offline-verifiable proof-of-solvency over a poster's reserves and liabilities.

### `SolvencyProofVerification(**data)`

The (non-raising) outcome of verifying a solvency proof offline.

### `SpeechProvider()`

Helper class that provides a standard way to create an ABC using inheritance.

### `SpeechRequest(**data)`

!!! abstract "Usage Documentation" [Models](../concepts/models.md)

### `StabilityLevel(*args, **kwds)`

Stability contract for a public symbol.

### `StateGraph(name=…, state_schema=…, reducers=…, defaults=…)`

Build-time graph definition; ``compile()`` produces the runnable form.

### `StatisticalClaim(**data)`

Base of the analytical-claim family the statistical kernels certify.

### `StepBinding(**data)`

The resolved run-time binding for one capability step.

### `StepOutcome(**data)`

A participant's result for one dispatched step.

### `StepRecord(**data)`

One immutable, hash-chained entry in a :class:`SagaJournal`.

### `StepRequest(**data)`

The typed envelope dispatched to a participant for one step, the handoff.

### `StreamWindow(**data)`

A windowing policy over an unbounded event stream, carrying the streaming analogues of the data plane's batch primitives.

### `SubgraphScheduler(workers=…, store=…, coordinator=…, lease_ttl_s=…, budget=…, deadline_s=…, clock=…)`

Runs independent sub-graphs concurrently under a fair-share budget + SLA.

### `SubgraphTask(graph, input=…, id=…, thread_id=…, weight=…)`

One independent sub-graph to schedule.

### `SuiteReport(run, title=…, cite_failures=…)`

Render one :class:`SuiteRun` to Markdown / HTML / JSON / CSV / PDF.

### `SuiteRun(**data)`

A whole suite run: one model over a set of benchmarks at one tier.

### `SwapGate(app, metrics=…, quality_metric=…, gates=…, alpha=…, drift_threshold=…, behavior_threshold=…, repeats=…, flake_quarantine=…)`

Gate a model/provider change on replayed golden traces + an eval/cost/ latency/behavioral diff with statistical backing.

### `SwapVerdict(**data)`

PASS / FAIL verdict for promoting a model into the live path.

### `SynthesizedProgram(**data)`

A verified transform paired with the certificate proving its properties.

### `SystemCard(**data)`

Documentation for the whole system: model + retrieval + memory + safety.

### `TableEvidence(**data)`

A :class:`~vincio.data.Dataset` presented as first-class context evidence.

### `TaintedValue(value, label=…, sources=…)`

A value carried together with its :class:`TrustLabel` and provenance.

### `TaskType(*args, **kwds)`

Task taxonomy used by the input router.

### `TemporalVerifier()`

Checks date ordering and duration claims against a real calendar.

### `TestTimeSearch(generate, verifier=…, budget=…)`

Verifier-guided test-time search bounded by a :class:`SearchBudget`.

### `ThresholdCalibrator(target_precision=…, min_floor=…)`

Fit a calibrated acceptance threshold from labelled near-miss examples.

### `TimerService(graph, clock=…)`

Resumes due timers and delivers events for one compiled graph.

### `ToolClause(**data)`

One named pre- or post-condition over a tool call.

### `ToolContract(**data)`

Pre- and post-conditions checked against a tool's actual call and result.

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

### `TrendClaim(**data)`

A stated linear trend over a cited series.

### `TrendVerifier(claims=…)`

Recomputes a stated linear trend and its goodness-of-fit from cited cells.

### `TrustConfig(**data)`

How the importer's trust in an issuer scales the evidence it contributes.

### `TrustLabel(*args, **kwds)`

A typed information-flow label on a value or context candidate.

### `TrustModel(assessments, config)`

The importer's bounded, transitive trust in each issuer, the Sybil-resistant kernel.

### `TwoStageIndex(embedder=…, coarse_dims=…, quantization=…, rerank_factor=…)`

Matryoshka + quantized coarse search, full-precision exact rerank.

### `UIAction(**data)`

A typed action bound to a target by a stable selector, not a coordinate.

### `UIElement(**data)`

A typed, addressable element grounded from the screen + accessibility tree.

### `UnderReservedBreach(**data)`

A proven-reserves shortfall: the pools pledge more than the custodian attests.

### `UnitVerifier()`

Checks unit conversions and refuses a dimensional mismatch.

### `UsageEvent(**data)`

One unit of delivered usage accrued against a contract.

### `UserInput(**data)`

Structured task input.

### `VerifiableReward()`

Base contract: map a :class:`RewardSample` to a :class:`RewardSignal`.

### `VerificationContext(**data)`

The grounding a verifier may consult while certifying an answer.

### `VerificationReport(**data)`

The verdict of a governance-verification pass over all invariants.

### `VerifiedAnswer(**data)`

An answer paired with the certificate a deterministic verifier produced.

### `Verifier(*args, **kwargs)`

Scores a candidate answer or trajectory. Reuse an existing critic via the adapters in this module rather than implementing this directly.

### `VerifierScore(**data)`

A verifier's verdict on one candidate: a value, a confidence, a reason.

### `VideoGenRequest(**data)`

!!! abstract "Usage Documentation" [Models](../concepts/models.md)

### `VideoProvider()`

Abstract video generation/editing provider.

### `VincioConfig(**data)`

Top-level project configuration.

### `VincioDeprecationWarning(*args, **kwargs)`

Emitted when a deprecated Vincio API is used.

### `VincioError(message, details=…, hint=…, docs_url=…)`

Base class for all Vincio errors.

### `VincioExperimentalWarning(*args, **kwargs)`

Emitted on first use of an :func:`experimental` API.

### `Violation(**data)`

A single property breach pinned to the event that caused it.

### `VoiceAgent(app, backend=…, config=…, research=…, memory_os=…, rails=…, owner_id=…, research_tool=…, **backend_kwargs)`

A grounded, remembering, guarded voice session over a :class:`ContextApp`.

### `WaterfallTranche(**data)`

The per-tranche distribution summary of an :class:`InsolvencyResolution`.

### `WindowedQueryResult(**data)`

A governed query's result over one closed window, **event-level cited**.

### `WorkerPoolBackend(workers=…, store=…, coordinator=…, lease_ttl_s=…)`

In-process reference distributed executor, lock-free, durable, fan-out.

### `Workflow(name, tracer=…, approval_fn=…)`

A deterministic, resumable DAG of steps.

### `WorldModel(transitions=…)`

A deterministic, offline-learned dynamics model of a tool environment.

## Functions

### `InputField(desc=…, default=…, **kwargs)`

Declare a signature input field.

### `OutputField(desc=…, **kwargs)`

Declare a signature output field.

### `admit(subject, reputation=…, ledger=…, standing=…, config=…)`

Decide a counterparty's admitted exposure from its standing.

### `analyze_dataset(objective, data, table=…, budget=…, engine=…, injection_detector=…, screen=…, extra_questions=…)`

Run a bounded, multi-step analysis over a dataset and return a cited analytical narrative — the offline, deterministic core of the data-analysis agent.

### `arbitrate(records, contract_id=…, arbiter=…, verifier=…, verify_with=…)`

Adjudicate a disputed contract from the records its parties submit.

### `assurance_regression_gate(before, after)`

Block a build when a previously-discharged claim is no longer discharged.

### `attest_custody(poster, reserves, custodian=…, as_of=…)`

Attest a poster's proven reserves into an (unsigned) :class:`CustodyAttestation`.

### `attest_liabilities(poster, liabilities, attestor=…, as_of=…, prior=…)`

Attest a poster's total obligations into an (unsigned) :class:`LiabilityAttestation`.

### `attest_reputation(records, subject, issuer=…, resolutions=…, config=…, verifier=…, horizon_days=…, note=…, verify_with=…)`

Issue an attestation of ``subject``'s earned standing from signed records.

### `attestation_a2a_server(book, revocations=…, attestations=…, config=…, name=…, url=…, description=…, tracer=…, token_validator=…, audit=…)`

Expose an org's settlement book as a queryable attestation peer over A2A.

### `attribute_regression(app, dataset, factors, metric=…, aggregate=…, repeats=…)`

Attribute a metric regression to the changed ``factors`` by Shapley counterfactual replay, the convenience entry point behind a failing gate.

### `available_packs()`

Names of all packs that can be loaded (built-in + installed plugins + registered).

### `build_finetune_backend(provider)`

Build the right fine-tune backend for a provider instance.

### `build_retail_environment(task_id=…)`

A τ-bench-style retail world: orders mutated by tools, verified by end state.

### `build_seniority_schedule(poster, tranches, as_of=…)`

Rank a poster's obligations into a sealed, unsigned :class:`SenioritySchedule`.

### `build_set_off_statement(poster, creditor, owed_usd, owing_usd, references=…, as_of=…)`

Collapse the mutual obligations between a poster and one creditor into a statement.

### `build_trust_model(attestations, base=…, config=…, attestation_config=…, verifier=…, verify_with=…)`

Build the importer's bounded, transitive trust over a set of issuers.

### `build_web_checkout()`

A deterministic, in-process checkout app and its goal, the offline, WebArena / OSWorld-shaped reference scenario.

### `buyer_position(max_price_usd, ideal_price_usd=…, max_sla_seconds, ideal_sla_seconds=…, min_quality=…, ideal_quality=…, weights=…, concession=…, min_utility=…)`

Build a buyer position: wants low price, fast SLA, high quality.

### `certify(case, signer=…, residual_risks=…, provenance=…, as_of=…)`

Build a :class:`CertificationReport` from a checked assurance case.

### `chat(provider=…, model=…, name=…, tools=…, writes=…, approve=…, web=…, user_id=…, tenant_id=…, session_id=…, memory_writeback=…, on_approval=…, role=…, objective=…, rules=…, app=…, config=…)`

Open a multi-turn, session-aware chat in one expression.

### `check_completeness(liabilities, claims, verifier=…, as_of=…)`

Fold a set of creditor claims against a liability attestation into a completeness check.

### `check_history_consistency(attestations, discharges=…, verifier=…)`

Walk a set of liability snapshots for cross-time monotonicity (no debt silently dropped).

### `check_root_consistency(attestations, verifier=…)`

Compare a set of liability attestations for cross-creditor root non-equivocation.

### `choreography_a2a_server(handlers, org_id=…, name=…, url=…, description=…, tracer=…, token_validator=…, audit=…)`

Expose a local org's choreography handlers over A2A.

### `combine_attestations(attestations, subject=…, config=…, verifier=…, base=…, allow_self=…, revocations=…, as_of=…, trust=…, trust_config=…, verify_with=…)`

Combine several issuers' attestations into one bounded, evidence-weighted prior.

### `compose(*steps, name=…, tracer=…)`

Compose steps left to right: ``compose(a, b) == compose(a) | b``.

### `default_model_registry()`

Process-wide registry, seeded from the built-in catalog plus the ``VINCIO_MODEL_REGISTRY`` overlay (if set). Constructed lazily and cached.

### `default_verifiers()`

The default offline kernel set behind ``app.verify_reasoning``.

### `deprecated(since, removed_in, alternative=…)`

Mark a function or class as deprecated.

### `did_from_public_key(public_key)`

Derive the self-certifying DID for an Ed25519 public key.

### `discharge_liability(poster, creditor, amount_usd, as_of=…, note=…)`

Build an (unsigned) :class:`Discharge` releasing part of what ``poster`` owes ``creditor``.

### `discover_plugins(groups=…, entry_points=…)`

List installed Vincio plugins without registering them.

### `draw_pool(pool, record, config=…)`

Settle one backed contract against a settlement record (draw or release).

### `edge_environment()`

Detect the current runtime and report its edge-relevant capabilities.

### `edge_manifest()`

Certify that the edge core imports no native/optional dependency.

### `enable_rich_reprs()`

Attach ``_repr_html_`` / ``_repr_markdown_`` to the core and data-plane types.

### `evaluation(dataset=…, metrics=…, gates=…, provider=…, model=…, name=…, role=…, objective=…, rules=…, app=…, config=…)`

Build an offline evaluation in one expression.

### `experimental(since, note=…)`

Mark a function or class as experimental (no stability guarantee).

### `extractor(schema, provider=…, model=…, name=…, role=…, objective=…, rules=…, app=…, config=…)`

Build a typed structured-extraction task from a schema in one expression.

### `gather_reputation(subject, peers, directory=…, principal=…, config=…, verifier=…, base=…, allow_self=…, held_attestations=…, held_revocations=…, as_of=…, trust=…, trust_config=…, max_peers=…, audit=…, record_audit=…, verify_with=…)`

Pull signed attestations and revocations from a bounded set of peers.

### `generate_chart(result, type=…, x=…, y=…, color=…, title=…, renderer=…, signer=…, infer_type=…)`

Turn a cited query result into a **content-bound, data-bound** chart.

### `generate_redline(original, revised, format=…, title=…)`

Generate a tracked-change redline between two texts.

### `guard_collateral(pools, poster=…, held=…, custody=…, solvency=…, verifier=…, verify_with=…)`

Fold a counterparty's collateral pools into a bounded, offline-verifiable re-use guard.

### `installed_plugins()`

All installed Vincio plugins across every group (alias for discovery).

### `is_vincio_did(did)`

Whether ``did`` is a well-formed ``did:vincio:ed25519`` identifier.

### `is_wasm_runtime()`

True when running on a WASM target (Emscripten/Pyodide or WASI).

### `key_fingerprint(public_key)`

A short, stable key id (``k<16 hex>``) for a public key, used as ``kid``.

### `library_capability(library, tasks, search=…)`

Fraction of *tasks* the library solves by applying an existing skill.

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

### `make_web_checkout()`

A deterministic, in-process checkout app and its goal, the offline, WebArena / OSWorld-shaped reference scenario.

### `model_swap_regression(app, dataset, baseline_model=…, candidate_model, metrics=…, quality_metric=…, alpha=…, repeats=…, flake_quarantine=…, flake_threshold=…, slice_prefix=…)`

Swap only the model on a fixed dataset and report a statistically grounded regression analysis (the body of ``vincio eval regress``).

### `negotiation_a2a_server(party, name=…, url=…, description=…, tracer=…, token_validator=…, audit=…)`

Expose a local negotiating :class:`Party` over A2A.

### `net_books(books, owner=…, verifier=…, require_intact=…, verify_with=…)`

Net a fleet of :class:`~vincio.settlement.book.SettlementBook`\ s into one set.

### `net_settlements(records, owner=…, fleet=…, verifier=…, verify_with=…)`

Fold a fleet's settled contracts into a minimal cleared set of obligations.

### `notebook_session(app, dataset=…, question=…, analyst=…, auto_display=…, rich=…)`

Open a governed, notebook-native analysis session over *app*.

### `post_collateral_pool(contracts, poster=…, posted=…, decisions=…, fraction=…, config=…)`

Post one stake backing many contracts into an (unsigned) :class:`CollateralPool`.

### `post_escrow(contract, decision=…, fraction=…, amount=…, poster=…, beneficiary=…, config=…)`

Post collateral against a contract into an (unsigned) :class:`Escrow`.

### `prove_equivocation(first, second, verifier=…, first_creditor=…, second_creditor=…)`

Fold two conflicting liability attestations into a non-repudiable :class:`EquivocationProof`.

### `prove_solvency(custody, liabilities, poster=…, completeness=…, as_of=…, verifier=…)`

Fold a reserve proof against a liability proof into a proof-of-solvency.

### `provider_trainer(backend, registry=…, inherit_from=…, pricing=…, suffix=…, fmt=…, poll_interval_s=…, max_polls=…)`

Build an *executed* :data:`StudentTrainer` over a fine-tune backend.

### `public_key_from_did(did)`

Recover the Ed25519 public key embedded in a ``did:vincio:ed25519`` DID.

### `query_dataset(request, data, dialect=…, question=…, ops=…, table=…, max_rows=…, engine=…, injection_detector=…, screen_question=…)`

Plan → verify → execute → cite, in one call.

### `query_metric(request, data, layer, by=…, where=…, order_by=…, descending=…, limit=…, engine=…, max_rows=…, injection_detector=…, screen=…)`

Resolve a governed metric over *data* with *layer* and run it — the one-shot free function behind :meth:`SemanticLayer.query`.

### `rag(sources=…, provider=…, model=…, name=…, grounded=…, evaluators=…, role=…, objective=…, rules=…, output_schema=…, chunking=…, retrieval=…, app=…, config=…)`

Build a grounded-RAG question answerer in one expression.

### `reconcile(a, b, tolerance=…)`

Tie two independently-produced settlement records out against each other.

### `record_transitions(env, action_sequences, include_failures=…)`

Drive ``env`` through each action sequence, recording every tool step.

### `register_benchmark(spec, replace=…)`

Register a benchmark on the default registry — the public extension point.

### `resolve_insolvency(custody, liabilities, schedule=…, poster=…, completeness=…, solvency=…, set_off=…, as_of=…, verifier=…)`

Distribute a poster's proven reserves across its ranked liabilities into a resolution.

### `retrieval_regression(search_fn, golden, config, store=…, metrics=…, gates=…, top_k=…, alpha=…, min_delta=…, k_values=…)`

Evaluate ``config`` on ``golden``, record an artifact, and gate vs. baseline.

### `revoke_attestation(attestation, subject=…, issuer=…, replacement=…, reason=…)`

Issue a revocation withdrawing a prior attestation by its hash.

### `select_offer(results, buyer_position, reputation=…)`

Pick the best deal among competing sellers by reputation-weighted utility.

### `seller_position(min_price_usd, ideal_price_usd, min_sla_seconds=…, ideal_sla_seconds=…, max_quality=…, ideal_quality=…, weights=…, concession=…, min_utility=…)`

Build a seller position: wants high price, a loose SLA, a low quality floor.

### `serve_viewer(store, host=…, port=…)`

Start the served observability plane over ``store`` (opt-in, self-hosted).

### `set_off_from_records(poster, creditor, liabilities, records, as_of=…, verifier=…)`

Derive a set-off statement straight from the existing signed, content-bound artifacts.

### `settle_contract(contract, reading=…, cost_usd=…, latency_ms=…, quality=…, run_id=…, saga_id=…)`

Reconcile delivery against a contract into an (unsigned) settlement record.

### `settle_escrow(escrow, record, config=…)`

Resolve a posted escrow against a settlement record (release or forfeit).

### `settle_saga(result, contracts, run_id=…)`

Settle every contract a cross-org saga ran under, from its durable journal.

### `signature(spec, instructions=…, name=…)`

Build a Signature type from a DSPy-style string spec::

### `sleep_for(state, seconds, clock=…)`

Pause the graph for ``seconds`` of wall-clock time, durably.

### `sleep_until(state, when, clock=…)`

Pause the graph until ``when`` (a datetime or ISO string), durably.

### `stability_of(obj)`

Return the stability record for ``obj``.

### `statistical_verifiers()`

The four statistical kernels — trend, correlation, interval, forecast.

### `stream_aggregate(data, group_by, measures=…, max_groups=…)`

Group a stream by one or more columns and reduce measures over each group in a single bounded-memory pass.

### `synthesize(spec, examples, require=…)`

Verify ``spec``'s properties on ``examples`` and emit a proof-carrying program.

### `task_goal_value(checks)`

A goal-value function: the fraction of an environment task's checks an observation's state satisfies (the planner's default verifier).

### `tool_agent(tools=…, writes=…, approve=…, web=…, provider=…, model=…, name=…, role=…, objective=…, rules=…, app=…, config=…)`

Build an approval-gated tool-using agent in one expression.

### `verify_containment(events)`

Check ``untrusted ⇒ no unapproved capability`` over recorded events.

### `verify_edge_parity(request=…, profile=…)`

Prove the edge runtime is the server compiler under a profile, not a fork.

### `verify_erasure_proof(proof, signer=…)`

Verify a proof's content binding and (if present) its signature.

### `wait_for_event(state, name)`

Pause the graph until an event named ``name`` is delivered; return its payload.

## Values

### `API_VERSION`

str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str
