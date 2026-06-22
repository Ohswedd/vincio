# Reference: error catalog

Every `vincio` error derives from `VincioError` and carries a stable
`.code`, a `.remediation` hint, and a `.docs_url` deep link into this
page. Catch the whole family with one `except VincioError`, branch on
`.code` for programmatic handling, and surface `.remediation` to users.

```python
from vincio import VincioError

try:
    app.run("...")
except VincioError as exc:
    print(exc.code, exc.message)
    print("fix:", exc.remediation)
    print("docs:", exc.docs_url)
```

Error message strings are not part of the stable API; the `.code` values
and this catalog are. This page is generated from
`vincio.core.error_catalog` and gated for completeness — no error ships
without an entry here.

## Codes

### VINCIO_ERROR

**Vincio error.** Catch-all base error. Inspect `.code`, `.message`, and `.details`; every Vincio failure derives from VincioError so one except clause covers the family.

### CONFIG_ERROR

**Invalid configuration.** Run `vincio config validate` to locate the offending field, and `vincio config migrate` if the file predates the current schema.

### PROVIDER_ERROR

**Model provider failure.** Check the provider's status and your network; wrap the provider in a FailoverChain or CircuitBreaker so a single backend cannot stall a run.

### PROVIDER_AUTH

**Provider authentication failed.** The API key is missing, wrong, or lacks scope. Set the standard env var (e.g. OPENAI_API_KEY) or `provider.api_keys` indirection in vincio.yaml, and confirm the key is active for the target model.

### PROVIDER_RATE_LIMIT

**Provider rate limit.** Back off and retry — the error is retryable and carries `retry_after_s`. Add a RateLimiter or KeyPool, or lower `performance.max_concurrency`.

### PROVIDER_TIMEOUT

**Provider timed out.** Raise `provider.timeout_s`, reduce the request size, or rely on the automatic retry; persistent timeouts indicate provider degradation — fail over to a healthy model.

### PROVIDER_UNAVAILABLE

**Provider unavailable.** The backend is temporarily down (retryable). Configure `provider.fallback_models` so a FailoverChain routes around the outage.

### PROVIDER_RESPONSE

**Malformed provider response.** The provider returned an unparseable or contract-violating payload. Verify the model id is correct for the endpoint and that any OpenAI-compatible base URL implements the expected schema.

### CIRCUIT_OPEN

**Circuit breaker open.** The breaker tripped after repeated failures and is failing fast. Let it cool down, or provide a fallback model so the failover chain skips the unhealthy entry immediately.

### BATCH_ERROR

**Batch API failure.** A Batch submission/poll/reconciliation failed. Re-submit the batch or fall back to synchronous `run`; inspect `.details` for the provider job id.

### FINETUNE_ERROR

**Fine-tune job failure.** The distillation fine-tune could not be submitted or reached a failed/cancelled state. Check the training file format and the provider job status before re-running the flywheel.

### CAPABILITY_MISMATCH

**Model capability mismatch.** The routed model structurally cannot serve the request (see `.missing`, e.g. vision/tools/context). Escalate to a capable model rather than retrying; enable `guard_capabilities` on the router/failover chain.

### MODEL_RETIRED

**Model retired.** The pinned model is past its registry retirement date. Run `vincio providers lifecycle` for a migration proposal and repin to the successor model.

### PROMPT_ERROR

**Prompt compilation error.** The prompt spec is malformed. Run `vincio prompt lint` to surface the offending rule and location.

### PROMPT_LINT

**Prompt lint failure.** A blocking lint rule fired (see `.findings`). Fix the flagged sections or relax the rule; `vincio prompt lint` reports each finding with a hint.

### PROMPT_BUDGET

**Prompt over token budget.** The compiled prompt exceeds the token budget. Trim instructions/examples, raise `budget.max_input_tokens`, or enable context compression.

### CONTEXT_ERROR

**Context compilation error.** The context compiler could not assemble a packet. Inspect the source candidates and scoring configuration in `.details`.

### CONTEXT_COMPILE

**Context compile failure.** Candidate collection or packing failed. Check that sources are indexed and the embedder is reachable; review the excluded-context report.

### BUDGET_EXCEEDED

**Token budget exceeded.** Selected context exceeds the budget (`.used` vs `.limit`). Raise the token budget, lower `retrieval.top_k`, or enable compression/packing.

### INPUT_ERROR

**Invalid input.** The run input could not be normalized or classified. Provide non-empty text or a supported file type.

### DOCUMENT_ERROR

**Document processing error.** A document could not be parsed. Confirm the format is supported and the file is not corrupt; install the relevant extra (e.g. `vincio[pdf]`).

### LOADER_ERROR

**Document loader error.** No loader matched, or a loader failed. Register one with `register_loader`, or install the extra its format requires.

### RETRIEVAL_ERROR

**Retrieval failure.** A retrieval backend errored. Verify the index exists and the vector store URL in `storage.vector` is reachable.

### INDEX_ERROR

**Index failure.** Building or querying an index failed. Rebuild with `vincio index build`, and confirm the embedder dimension matches the stored vectors.

### MEMORY_ERROR

**Memory engine error.** A memory operation failed. Check the memory store URL and that the owner/scope arguments are supplied.

### MEMORY_POLICY

**Memory policy violation.** The write policy rejected this memory (`memory.write_policy`). Loosen the policy to `open`, or supply the required owner/consent metadata.

### MEMORY_CONFLICT

**Memory conflict.** A new memory contradicts an existing one. Use `MemoryEngine.correct()` to supersede it history-preservingly instead of overwriting.

### TOOL_ERROR

**Tool execution error.** A tool raised. Inspect `.tool` and the tool's own exception; make the tool defensive or wrap the call site.

### TOOL_NOT_FOUND

**Tool not found.** No tool with that name is registered. Register it with `app.add_tool`, and check for a typo against `app.enabled_tools`.

### TOOL_PERMISSION

**Tool permission denied.** The caller's role lacks permission for this tool. Grant the permission in the registry, or call with an authorized principal.

### TOOL_VALIDATION

**Tool argument validation failed.** The arguments do not match the tool's schema. Correct the call against the derived JSON Schema in `.details`.

### TOOL_TIMEOUT

**Tool timed out.** The tool exceeded its time limit. Raise the tool timeout, or make the tool faster/asynchronous.

### TOOL_APPROVAL_REQUIRED

**Tool approval required.** A write/side-effecting tool is gated behind human approval. Approve the pending call (e.g. `assistant.approve(...)`) or add it to an `auto_approve` allow-list.

### SANDBOX_ERROR

**Sandbox isolation failure.** The isolation backend is unavailable or too weak for the requested level. Install/configure a real backend, or lower the isolation requirement only if you trust the code.

### AGENT_ERROR

**Agent execution error.** The agent loop failed. Inspect the trace span tree (`vincio trace show`) to find the failing step.

### AGENT_STEP

**Agent step failed.** A single plan step errored (see `.step_id`). The executor may repair the plan; if it recurs, narrow the step's tool or inputs.

### AGENT_BUDGET_EXHAUSTED

**Agent budget exhausted.** The agent hit its cost/token budget before finishing. Raise the budget or reduce the task scope.

### AGENT_MAX_STEPS

**Agent step limit reached.** The agent reached `max_steps` without converging. Raise `max_steps`, or decompose the task; inspect the trace for a loop.

### GRAPH_ERROR

**Graph definition or execution error.** The stateful graph is misconfigured or a node failed. Check channel reducers and that every edge target exists.

### CHECKPOINT_CONFLICT

**Checkpoint version conflict.** Another worker advanced the thread first (optimistic-concurrency loss). Re-acquire the lease and resume from the new head — this is non-fatal.

### WORKFLOW_ERROR

**Workflow error.** The deterministic workflow failed. Inspect the step graph and any compensation handlers.

### WORKFLOW_STEP

**Workflow step failed.** A workflow step raised (see `.step`). Add a retry or compensation, or fix the step's logic; resume from the last checkpoint.

### OUTPUT_ERROR

**Structured output error.** The model output failed contract handling. Review the schema and the raw text in `.details`.

### OUTPUT_PARSE

**Output parse failure.** The output is not valid JSON for the schema. Enable provider-native constrained decoding or bounded self-correction (`enable_self_correction`).

### OUTPUT_SCHEMA

**Output schema validation failed.** The parsed output violates the schema (see `.errors`). Tighten the prompt examples or enable structure-only repair.

### OUTPUT_REPAIR_FORBIDDEN

**Output repair forbidden.** Repair was disabled but the output needs it. Allow self-correction, or fix the prompt so the first attempt validates.

### CITATION_INVALID

**Citation validation failed.** A cited claim does not resolve to supporting evidence. Require citations and answer-only-from-sources, or relax the citation contract.

### GENERATION_ERROR

**Document/media generation error.** Rendering or a generation provider failed. Install the relevant extra (`vincio[gen-docx|gen-pdf|gen-pptx]`) and check the provider credentials.

### DOCUMENT_CONTRACT

**Document contract violation.** The rendered document violates its contract and formatting-only repair could not fix it (see `.violations`). Adjust the content or the TableSpec/structure requirements.

### MEDIA_GENERATION

**Media generation failure.** An image, video, or speech provider call failed. Verify the media provider credentials and that the requested model supports the modality.

### EVAL_ERROR

**Evaluation error.** An eval run failed. Check the dataset format and that every referenced metric/judge is registered.

### DATASET_ERROR

**Dataset error.** The dataset could not be loaded or is malformed. Validate the JSONL rows against the expected case schema.

### GATE_FAILED

**Quality gate failed.** A CI gate threshold was not met (see `.failures`). Fix the regression, or adjust the gate expression if the new baseline is intended.

### BENCHMARK_ERROR

**Benchmark adapter error.** A benchmark adapter failed to load or score. Confirm the task-set hash and that the recorded fixtures or live solver are wired correctly.

### OPTIMIZATION_ERROR

**Optimization error.** An optimization run failed. Check the dataset, fitness weights, and that the prompt spec is valid before retrying.

### REWARD_ERROR

**Reward derivation error.** A verifiable reward could not be derived from the sample. Provide the signal the reward needs (an environment verification, adapter gold, or judge inputs) before calling app.learn.

### CACHE_ERROR

**Cache error.** A cache backend failed. Verify the cache URL in `storage.cache`; an in-memory cache (`memory://`) always works as a fallback.

### SECURITY_ERROR

**Security policy error.** A security control failed or blocked the operation. Review the active rails and policy settings under `security`.

### ACCESS_DENIED

**Access denied.** The principal lacks rights for this resource. Grant the role/scope via the AccessController, or call with an authorized identity.

### TENANT_ISOLATION

**Tenant isolation violation.** A cross-tenant access was attempted, or a run is missing its tenant tag. Pass `tenant_id` on the run; keep `security.tenant_isolation` on.

### INJECTION_DETECTED

**Prompt injection detected.** Untrusted content carried instruction-like text. Keep `block_untrusted_instructions` on, quarantine the source, and review the injection finding in `.details`.

### CONTAINMENT_BLOCKED

**Containment blocked an untrusted capability.** An argument derived from untrusted data reached a write/external tool without authority. Mint a CapabilityToken from the user's request via CapabilityBroker (or route the call through the approval gate) before the side effect; the DualPlaneExecutor enforces this automatically.

### PII_POLICY

**PII policy violation.** Detected PII violates the active policy. Enable redaction (`redact_pii_in_context`), or add the locale pack the data requires.

### EGRESS_BLOCKED

**Egress DLP blocked the request.** The outbound request carried secrets or sensitive identifiers. Remove the leaked credential; set `security.egress_dlp: warn` only if the match is a false positive.

### GOVERNANCE_ERROR

**Governance/compliance error.** A governance artifact (card/BOM/lineage) could not be produced. Check that the app has the required sources and metadata configured.

### RESIDENCY_VIOLATION

**Data residency violation.** The resolved provider region is not in `governance.allowed_regions` (see `.region`/`.allowed`). Pin the provider region or route to an in-jurisdiction model.

### ERASURE_ERROR

**Erasure could not complete.** A right-to-erasure-by-source operation did not complete atomically. Retry `app.erase_source(...)`; inspect which stores were swept in `.details`.

### GOVERNANCE_INVARIANT_VIOLATED

**Governance invariant violated.** The formal verifier found a counterexample to a governance invariant (containment/residency/budget/erasure). Inspect `.counterexamples` for the minimal violating state, or call `app.verify_governance()` without `raise_on_violation` to get the full VerificationReport.

### PRIVACY_BUDGET_EXCEEDED

**Differential-privacy budget exceeded.** A consolidation or learning round would push a subject's cumulative (ε, δ) past its PrivacyBudget. Raise the subject's epsilon, set `on_breach='downweight'` to admit a clipped-harder release, or refuse the step; inspect spent/remaining ε in `.details` and `app.privacy_report()`.

### STORAGE_ERROR

**Storage backend error.** A storage backend failed. Verify the URL/credentials for the relevant `storage.*` setting and that the schema is migrated.

### SERVER_ERROR

**Server error.** The HTTP API server hit an internal error. Check the server logs and that every served app file exposes a ContextApp as `app`.

### AUTHENTICATION_ERROR

**Server authentication failed.** The request's API key or JWT was missing or invalid. Send a valid credential matching `server.api_keys`/`server.jwt_secret`.

### SKILL_ERROR

**Agent Skill error.** A SKILL.md bundle could not be parsed or loaded. Validate the front matter and that referenced scripts exist.

### OBSERVABILITY_ERROR

**Observability error.** A tracing, recording, or replay operation failed. Inspect `.details` and confirm the trace/recording exists and is readable.

### REPLAY_DIVERGENCE

**Recording no longer replays.** Live code asked for an edge (a model call, tool output, or retrieval) absent from the recording, or the recording failed to load/verify. Re-record against the current code, or use `Replayer.branch(...)` to re-execute the changed suffix against the recorded prefix.

### ENERGY_BUDGET_INVALID

**Energy budget misconfigured.** Set an energy budget with at least one ceiling: pass `limit_wh` (watt-hours), `limit_co2e_grams` (grams CO₂e), or both to `app.set_energy_budget(...)`.

### EDGE_ERROR

**Edge runtime request invalid or over profile.** Give the `EdgeRequest` a `task` or `objective`; under `strict=True`, raise the `EdgeProfile`'s `max_resident_bytes` / `max_input_tokens` or trim the request's evidence so the packet fits the edge profile.

### NEGOTIATION_ERROR

**Negotiation could not proceed.** Check the `NegotiationPosition` is coherent (the reservation must be no better for the party than its ideal) and the `NegotiationBudget` has positive `max_rounds`. A negotiation that runs out of rounds without a deal does not raise — it returns a partial NegotiationResult with `status='no_agreement'`.

### CONTRACT_VIOLATION

**Contract failed verification or was breached.** The contract's content hash did not recompute, a signature is missing or invalid, or delivered work breached the agreed price/SLA/quality (see `.breaches`). Re-verify with the signer both parties used, or renegotiate; use `contract.to_budget()` to enforce the terms up front.

### CHOREOGRAPHY_ERROR

**Cross-org choreography could not proceed.** Register a participant binding for every org a `Saga` step names, give the saga at least one uniquely-named step, and pass a `saga_id` that exists in the durable store when calling `resume`. A saga whose forward step fails does not raise — it compensates and returns a SagaResult with `status='compensated'`.

### COMPENSATION_FAILED

**Saga could not unwind cleanly.** A compensating step itself failed, leaving a half-completed cross-org transaction partially unwound (see `.failures`). Resume the saga to retry the outstanding compensations once the participant is reachable, or reconcile the residue manually; the journal pinpoints every compensation that did not complete.

### SETTLEMENT_ERROR

**Settlement could not proceed.** Meter non-negative usage, sign a settlement only as its buyer or seller, and supply the contract terms a saga's steps ran under when settling it. A settlement whose delivered work breaches the agreed terms does not raise — it reconciles to a SettlementRecord with `status='breached'` (see `.breaches`); re-verify a record or book with the signer the parties used.
