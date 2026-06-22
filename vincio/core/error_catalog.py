"""The Vincio error catalog: stable codes, remediation hints, and docs links.

Every :class:`~vincio.core.errors.VincioError` carries a stable ``.code``. This
module is the single source of truth that turns that code into an actionable,
human-facing message: a short *title*, a *remediation* hint that says what to do
next, and a deep link into the error reference. Decoupling the message from the
code keeps error strings free to change without breaking programmatic handling,
and makes the surface **internationalizable** — a locale pack registers
translated titles and hints keyed by the same stable codes (English is the
shipped reference locale).

The catalog is gated for completeness: every error class in the hierarchy must
have an entry here, and ``docs/reference/errors.md`` is generated from it, so a
new error cannot ship without a code, a remediation, and a documented anchor.

Protocol errors that repurpose ``.code`` as a numeric wire code (the JSON-RPC
``A2AError`` / ``MCPError``) are deliberately outside this string catalog; see
:data:`PROTOCOL_ERROR_CLASSES`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DOCS_BASE_URL",
    "PROTOCOL_ERROR_CLASSES",
    "ErrorCatalogEntry",
    "ERROR_CATALOG",
    "catalog_entry",
    "title_for",
    "remediation_for",
    "docs_url_for",
    "docs_anchor_for",
    "register_error_locale",
    "set_default_error_locale",
    "default_error_locale",
    "available_error_locales",
    "render_error_reference",
]

# Deep-link base for error documentation. Anchors are the lower-cased code.
DOCS_BASE_URL = "https://github.com/Ohswedd/vincio/blob/main/docs/reference/errors.md"

# Error classes that override ``.code`` with a numeric protocol code instead of a
# stable string code (JSON-RPC). They are intentionally excluded from the string
# catalog; the completeness gate asserts this set is exactly the non-string set.
PROTOCOL_ERROR_CLASSES = frozenset({"A2AError", "MCPError"})


@dataclass(frozen=True, slots=True)
class ErrorCatalogEntry:
    """One catalog row: the actionable metadata behind a stable error code."""

    code: str
    title: str
    remediation: str


def _entry(code: str, title: str, remediation: str) -> ErrorCatalogEntry:
    return ErrorCatalogEntry(code=code, title=title, remediation=remediation)


# The English reference catalog. Keyed by the exact ``VincioError.code`` string.
# Remediations are imperative and specific: they name the knob to turn, not just
# the failure. Keep entries ordered by subsystem to mirror ``core/errors.py``.
ERROR_CATALOG: dict[str, ErrorCatalogEntry] = {
    e.code: e
    for e in (
        _entry(
            "VINCIO_ERROR",
            "Vincio error",
            "Catch-all base error. Inspect `.code`, `.message`, and `.details`; "
            "every Vincio failure derives from VincioError so one except clause "
            "covers the family.",
        ),
        # --- configuration ---
        _entry(
            "CONFIG_ERROR",
            "Invalid configuration",
            "Run `vincio config validate` to locate the offending field, and "
            "`vincio config migrate` if the file predates the current schema.",
        ),
        # --- providers ---
        _entry(
            "PROVIDER_ERROR",
            "Model provider failure",
            "Check the provider's status and your network; wrap the provider in a "
            "FailoverChain or CircuitBreaker so a single backend cannot stall a run.",
        ),
        _entry(
            "PROVIDER_AUTH",
            "Provider authentication failed",
            "The API key is missing, wrong, or lacks scope. Set the standard env "
            "var (e.g. OPENAI_API_KEY) or `provider.api_keys` indirection in "
            "vincio.yaml, and confirm the key is active for the target model.",
        ),
        _entry(
            "PROVIDER_RATE_LIMIT",
            "Provider rate limit",
            "Back off and retry — the error is retryable and carries `retry_after_s`. "
            "Add a RateLimiter or KeyPool, or lower `performance.max_concurrency`.",
        ),
        _entry(
            "PROVIDER_TIMEOUT",
            "Provider timed out",
            "Raise `provider.timeout_s`, reduce the request size, or rely on the "
            "automatic retry; persistent timeouts indicate provider degradation — "
            "fail over to a healthy model.",
        ),
        _entry(
            "PROVIDER_UNAVAILABLE",
            "Provider unavailable",
            "The backend is temporarily down (retryable). Configure "
            "`provider.fallback_models` so a FailoverChain routes around the outage.",
        ),
        _entry(
            "PROVIDER_RESPONSE",
            "Malformed provider response",
            "The provider returned an unparseable or contract-violating payload. "
            "Verify the model id is correct for the endpoint and that any "
            "OpenAI-compatible base URL implements the expected schema.",
        ),
        _entry(
            "CIRCUIT_OPEN",
            "Circuit breaker open",
            "The breaker tripped after repeated failures and is failing fast. Let it "
            "cool down, or provide a fallback model so the failover chain skips the "
            "unhealthy entry immediately.",
        ),
        _entry(
            "BATCH_ERROR",
            "Batch API failure",
            "A Batch submission/poll/reconciliation failed. Re-submit the batch or "
            "fall back to synchronous `run`; inspect `.details` for the provider job id.",
        ),
        _entry(
            "FINETUNE_ERROR",
            "Fine-tune job failure",
            "The distillation fine-tune could not be submitted or reached a "
            "failed/cancelled state. Check the training file format and the "
            "provider job status before re-running the flywheel.",
        ),
        _entry(
            "CAPABILITY_MISMATCH",
            "Model capability mismatch",
            "The routed model structurally cannot serve the request (see `.missing`, "
            "e.g. vision/tools/context). Escalate to a capable model rather than "
            "retrying; enable `guard_capabilities` on the router/failover chain.",
        ),
        _entry(
            "MODEL_RETIRED",
            "Model retired",
            "The pinned model is past its registry retirement date. Run "
            "`vincio providers lifecycle` for a migration proposal and repin to the "
            "successor model.",
        ),
        # --- prompt engine ---
        _entry(
            "PROMPT_ERROR",
            "Prompt compilation error",
            "The prompt spec is malformed. Run `vincio prompt lint` to surface the "
            "offending rule and location.",
        ),
        _entry(
            "PROMPT_LINT",
            "Prompt lint failure",
            "A blocking lint rule fired (see `.findings`). Fix the flagged sections "
            "or relax the rule; `vincio prompt lint` reports each finding with a hint.",
        ),
        _entry(
            "PROMPT_BUDGET",
            "Prompt over token budget",
            "The compiled prompt exceeds the token budget. Trim instructions/examples, "
            "raise `budget.max_input_tokens`, or enable context compression.",
        ),
        # --- context compiler ---
        _entry(
            "CONTEXT_ERROR",
            "Context compilation error",
            "The context compiler could not assemble a packet. Inspect the source "
            "candidates and scoring configuration in `.details`.",
        ),
        _entry(
            "CONTEXT_COMPILE",
            "Context compile failure",
            "Candidate collection or packing failed. Check that sources are indexed "
            "and the embedder is reachable; review the excluded-context report.",
        ),
        _entry(
            "BUDGET_EXCEEDED",
            "Token budget exceeded",
            "Selected context exceeds the budget (`.used` vs `.limit`). Raise the "
            "token budget, lower `retrieval.top_k`, or enable compression/packing.",
        ),
        # --- input ---
        _entry(
            "INPUT_ERROR",
            "Invalid input",
            "The run input could not be normalized or classified. Provide non-empty "
            "text or a supported file type.",
        ),
        # --- documents ---
        _entry(
            "DOCUMENT_ERROR",
            "Document processing error",
            "A document could not be parsed. Confirm the format is supported and the "
            "file is not corrupt; install the relevant extra (e.g. `vincio[pdf]`).",
        ),
        _entry(
            "LOADER_ERROR",
            "Document loader error",
            "No loader matched, or a loader failed. Register one with "
            "`register_loader`, or install the extra its format requires.",
        ),
        # --- retrieval ---
        _entry(
            "RETRIEVAL_ERROR",
            "Retrieval failure",
            "A retrieval backend errored. Verify the index exists and the vector "
            "store URL in `storage.vector` is reachable.",
        ),
        _entry(
            "INDEX_ERROR",
            "Index failure",
            "Building or querying an index failed. Rebuild with `vincio index build`, "
            "and confirm the embedder dimension matches the stored vectors.",
        ),
        # --- memory ---
        _entry(
            "MEMORY_ERROR",
            "Memory engine error",
            "A memory operation failed. Check the memory store URL and that the "
            "owner/scope arguments are supplied.",
        ),
        _entry(
            "MEMORY_POLICY",
            "Memory policy violation",
            "The write policy rejected this memory (`memory.write_policy`). Loosen the "
            "policy to `open`, or supply the required owner/consent metadata.",
        ),
        _entry(
            "MEMORY_CONFLICT",
            "Memory conflict",
            "A new memory contradicts an existing one. Use `MemoryEngine.correct()` "
            "to supersede it history-preservingly instead of overwriting.",
        ),
        # --- tools ---
        _entry(
            "TOOL_ERROR",
            "Tool execution error",
            "A tool raised. Inspect `.tool` and the tool's own exception; make the "
            "tool defensive or wrap the call site.",
        ),
        _entry(
            "TOOL_NOT_FOUND",
            "Tool not found",
            "No tool with that name is registered. Register it with `app.add_tool`, "
            "and check for a typo against `app.enabled_tools`.",
        ),
        _entry(
            "TOOL_PERMISSION",
            "Tool permission denied",
            "The caller's role lacks permission for this tool. Grant the permission "
            "in the registry, or call with an authorized principal.",
        ),
        _entry(
            "TOOL_VALIDATION",
            "Tool argument validation failed",
            "The arguments do not match the tool's schema. Correct the call against "
            "the derived JSON Schema in `.details`.",
        ),
        _entry(
            "TOOL_TIMEOUT",
            "Tool timed out",
            "The tool exceeded its time limit. Raise the tool timeout, or make the "
            "tool faster/asynchronous.",
        ),
        _entry(
            "TOOL_APPROVAL_REQUIRED",
            "Tool approval required",
            "A write/side-effecting tool is gated behind human approval. Approve the "
            "pending call (e.g. `assistant.approve(...)`) or add it to an "
            "`auto_approve` allow-list.",
        ),
        _entry(
            "SANDBOX_ERROR",
            "Sandbox isolation failure",
            "The isolation backend is unavailable or too weak for the requested "
            "level. Install/configure a real backend, or lower the isolation "
            "requirement only if you trust the code.",
        ),
        # --- agents ---
        _entry(
            "AGENT_ERROR",
            "Agent execution error",
            "The agent loop failed. Inspect the trace span tree (`vincio trace show`) "
            "to find the failing step.",
        ),
        _entry(
            "AGENT_STEP",
            "Agent step failed",
            "A single plan step errored (see `.step_id`). The executor may repair the "
            "plan; if it recurs, narrow the step's tool or inputs.",
        ),
        _entry(
            "AGENT_BUDGET_EXHAUSTED",
            "Agent budget exhausted",
            "The agent hit its cost/token budget before finishing. Raise the budget "
            "or reduce the task scope.",
        ),
        _entry(
            "AGENT_MAX_STEPS",
            "Agent step limit reached",
            "The agent reached `max_steps` without converging. Raise `max_steps`, or "
            "decompose the task; inspect the trace for a loop.",
        ),
        _entry(
            "GRAPH_ERROR",
            "Graph definition or execution error",
            "The stateful graph is misconfigured or a node failed. Check channel "
            "reducers and that every edge target exists.",
        ),
        _entry(
            "CHECKPOINT_CONFLICT",
            "Checkpoint version conflict",
            "Another worker advanced the thread first (optimistic-concurrency loss). "
            "Re-acquire the lease and resume from the new head — this is non-fatal.",
        ),
        # --- workflows ---
        _entry(
            "WORKFLOW_ERROR",
            "Workflow error",
            "The deterministic workflow failed. Inspect the step graph and any "
            "compensation handlers.",
        ),
        _entry(
            "WORKFLOW_STEP",
            "Workflow step failed",
            "A workflow step raised (see `.step`). Add a retry or compensation, or "
            "fix the step's logic; resume from the last checkpoint.",
        ),
        # --- output ---
        _entry(
            "OUTPUT_ERROR",
            "Structured output error",
            "The model output failed contract handling. Review the schema and the "
            "raw text in `.details`.",
        ),
        _entry(
            "OUTPUT_PARSE",
            "Output parse failure",
            "The output is not valid JSON for the schema. Enable provider-native "
            "constrained decoding or bounded self-correction (`enable_self_correction`).",
        ),
        _entry(
            "OUTPUT_SCHEMA",
            "Output schema validation failed",
            "The parsed output violates the schema (see `.errors`). Tighten the prompt "
            "examples or enable structure-only repair.",
        ),
        _entry(
            "OUTPUT_REPAIR_FORBIDDEN",
            "Output repair forbidden",
            "Repair was disabled but the output needs it. Allow self-correction, or "
            "fix the prompt so the first attempt validates.",
        ),
        _entry(
            "CITATION_INVALID",
            "Citation validation failed",
            "A cited claim does not resolve to supporting evidence. Require citations "
            "and answer-only-from-sources, or relax the citation contract.",
        ),
        # --- generation ---
        _entry(
            "GENERATION_ERROR",
            "Document/media generation error",
            "Rendering or a generation provider failed. Install the relevant extra "
            "(`vincio[gen-docx|gen-pdf|gen-pptx]`) and check the provider credentials.",
        ),
        _entry(
            "DOCUMENT_CONTRACT",
            "Document contract violation",
            "The rendered document violates its contract and formatting-only repair "
            "could not fix it (see `.violations`). Adjust the content or the "
            "TableSpec/structure requirements.",
        ),
        _entry(
            "MEDIA_GENERATION",
            "Media generation failure",
            "An image, video, or speech provider call failed. Verify the media "
            "provider credentials and that the requested model supports the modality.",
        ),
        # --- evals ---
        _entry(
            "EVAL_ERROR",
            "Evaluation error",
            "An eval run failed. Check the dataset format and that every referenced "
            "metric/judge is registered.",
        ),
        _entry(
            "DATASET_ERROR",
            "Dataset error",
            "The dataset could not be loaded or is malformed. Validate the JSONL rows "
            "against the expected case schema.",
        ),
        _entry(
            "GATE_FAILED",
            "Quality gate failed",
            "A CI gate threshold was not met (see `.failures`). Fix the regression, or "
            "adjust the gate expression if the new baseline is intended.",
        ),
        _entry(
            "BENCHMARK_ERROR",
            "Benchmark adapter error",
            "A benchmark adapter failed to load or score. Confirm the task-set hash "
            "and that the recorded fixtures or live solver are wired correctly.",
        ),
        # --- optimization ---
        _entry(
            "OPTIMIZATION_ERROR",
            "Optimization error",
            "An optimization run failed. Check the dataset, fitness weights, and that "
            "the prompt spec is valid before retrying.",
        ),
        _entry(
            "REWARD_ERROR",
            "Reward derivation error",
            "A verifiable reward could not be derived from the sample. Provide the "
            "signal the reward needs (an environment verification, adapter gold, or "
            "judge inputs) before calling app.learn.",
        ),
        # --- caching ---
        _entry(
            "CACHE_ERROR",
            "Cache error",
            "A cache backend failed. Verify the cache URL in `storage.cache`; an "
            "in-memory cache (`memory://`) always works as a fallback.",
        ),
        # --- security ---
        _entry(
            "SECURITY_ERROR",
            "Security policy error",
            "A security control failed or blocked the operation. Review the active "
            "rails and policy settings under `security`.",
        ),
        _entry(
            "ACCESS_DENIED",
            "Access denied",
            "The principal lacks rights for this resource. Grant the role/scope via "
            "the AccessController, or call with an authorized identity.",
        ),
        _entry(
            "TENANT_ISOLATION",
            "Tenant isolation violation",
            "A cross-tenant access was attempted, or a run is missing its tenant tag. "
            "Pass `tenant_id` on the run; keep `security.tenant_isolation` on.",
        ),
        _entry(
            "INJECTION_DETECTED",
            "Prompt injection detected",
            "Untrusted content carried instruction-like text. Keep "
            "`block_untrusted_instructions` on, quarantine the source, and review "
            "the injection finding in `.details`.",
        ),
        _entry(
            "CONTAINMENT_BLOCKED",
            "Containment blocked an untrusted capability",
            "An argument derived from untrusted data reached a write/external tool "
            "without authority. Mint a CapabilityToken from the user's request via "
            "CapabilityBroker (or route the call through the approval gate) before "
            "the side effect; the DualPlaneExecutor enforces this automatically.",
        ),
        _entry(
            "PII_POLICY",
            "PII policy violation",
            "Detected PII violates the active policy. Enable redaction "
            "(`redact_pii_in_context`), or add the locale pack the data requires.",
        ),
        _entry(
            "EGRESS_BLOCKED",
            "Egress DLP blocked the request",
            "The outbound request carried secrets or sensitive identifiers. Remove "
            "the leaked credential; set `security.egress_dlp: warn` only if the "
            "match is a false positive.",
        ),
        # --- governance ---
        _entry(
            "GOVERNANCE_ERROR",
            "Governance/compliance error",
            "A governance artifact (card/BOM/lineage) could not be produced. Check "
            "that the app has the required sources and metadata configured.",
        ),
        _entry(
            "RESIDENCY_VIOLATION",
            "Data residency violation",
            "The resolved provider region is not in `governance.allowed_regions` "
            "(see `.region`/`.allowed`). Pin the provider region or route to an "
            "in-jurisdiction model.",
        ),
        _entry(
            "ERASURE_ERROR",
            "Erasure could not complete",
            "A right-to-erasure-by-source operation did not complete atomically. "
            "Retry `app.erase_source(...)`; inspect which stores were swept in "
            "`.details`.",
        ),
        _entry(
            "GOVERNANCE_INVARIANT_VIOLATED",
            "Governance invariant violated",
            "The formal verifier found a counterexample to a governance invariant "
            "(containment/residency/budget/erasure). Inspect `.counterexamples` for "
            "the minimal violating state, or call `app.verify_governance()` without "
            "`raise_on_violation` to get the full VerificationReport.",
        ),
        _entry(
            "PRIVACY_BUDGET_EXCEEDED",
            "Differential-privacy budget exceeded",
            "A consolidation or learning round would push a subject's cumulative "
            "(ε, δ) past its PrivacyBudget. Raise the subject's epsilon, set "
            "`on_breach='downweight'` to admit a clipped-harder release, or refuse "
            "the step; inspect spent/remaining ε in `.details` and "
            "`app.privacy_report()`.",
        ),
        # --- storage ---
        _entry(
            "STORAGE_ERROR",
            "Storage backend error",
            "A storage backend failed. Verify the URL/credentials for the relevant "
            "`storage.*` setting and that the schema is migrated.",
        ),
        # --- server ---
        _entry(
            "SERVER_ERROR",
            "Server error",
            "The HTTP API server hit an internal error. Check the server logs and "
            "that every served app file exposes a ContextApp as `app`.",
        ),
        _entry(
            "AUTHENTICATION_ERROR",
            "Server authentication failed",
            "The request's API key or JWT was missing or invalid. Send a valid "
            "credential matching `server.api_keys`/`server.jwt_secret`.",
        ),
        # --- skills ---
        _entry(
            "SKILL_ERROR",
            "Agent Skill error",
            "A SKILL.md bundle could not be parsed or loaded. Validate the front "
            "matter and that referenced scripts exist.",
        ),
        # --- observability ---
        _entry(
            "OBSERVABILITY_ERROR",
            "Observability error",
            "A tracing, recording, or replay operation failed. Inspect `.details` "
            "and confirm the trace/recording exists and is readable.",
        ),
        _entry(
            "REPLAY_DIVERGENCE",
            "Recording no longer replays",
            "Live code asked for an edge (a model call, tool output, or retrieval) "
            "absent from the recording, or the recording failed to load/verify. "
            "Re-record against the current code, or use `Replayer.branch(...)` to "
            "re-execute the changed suffix against the recorded prefix.",
        ),
        _entry(
            "ENERGY_BUDGET_INVALID",
            "Energy budget misconfigured",
            "Set an energy budget with at least one ceiling: pass `limit_wh` "
            "(watt-hours), `limit_co2e_grams` (grams CO₂e), or both to "
            "`app.set_energy_budget(...)`.",
        ),
        _entry(
            "EDGE_ERROR",
            "Edge runtime request invalid or over profile",
            "Give the `EdgeRequest` a `task` or `objective`; under `strict=True`, "
            "raise the `EdgeProfile`'s `max_resident_bytes` / `max_input_tokens` "
            "or trim the request's evidence so the packet fits the edge profile.",
        ),
        # --- agent negotiation & contracting ---
        _entry(
            "NEGOTIATION_ERROR",
            "Negotiation could not proceed",
            "Check the `NegotiationPosition` is coherent (the reservation must be "
            "no better for the party than its ideal) and the `NegotiationBudget` "
            "has positive `max_rounds`. A negotiation that runs out of rounds "
            "without a deal does not raise — it returns a partial NegotiationResult "
            "with `status='no_agreement'`.",
        ),
        _entry(
            "CONTRACT_VIOLATION",
            "Contract failed verification or was breached",
            "The contract's content hash did not recompute, a signature is missing "
            "or invalid, or delivered work breached the agreed price/SLA/quality "
            "(see `.breaches`). Re-verify with the signer both parties used, or "
            "renegotiate; use `contract.to_budget()` to enforce the terms up front.",
        ),
    )
}


# --- internationalization ---------------------------------------------------

# Reference locale. Additional locales register translated (title, remediation)
# pairs keyed by the same stable code; lookups fall back to English.
_DEFAULT_LOCALE = "en"
_LOCALES: dict[str, dict[str, tuple[str, str]]] = {
    _DEFAULT_LOCALE: {e.code: (e.title, e.remediation) for e in ERROR_CATALOG.values()}
}
_active_locale = _DEFAULT_LOCALE


def register_error_locale(locale: str, entries: dict[str, tuple[str, str]]) -> None:
    """Register translated ``code -> (title, remediation)`` pairs for *locale*.

    Partial locales are allowed; any code without a translation falls back to
    the English reference text. The locale code is matched case-insensitively.
    """
    key = locale.lower()
    bucket = _LOCALES.setdefault(key, {})
    for code, pair in entries.items():
        if code not in ERROR_CATALOG:
            raise KeyError(f"unknown error code in locale {locale!r}: {code!r}")
        bucket[code] = (pair[0], pair[1])


def set_default_error_locale(locale: str) -> None:
    """Set the process-wide locale used when none is passed to lookups."""
    global _active_locale
    _active_locale = locale.lower()


def default_error_locale() -> str:
    """Return the active default locale code."""
    return _active_locale


def available_error_locales() -> tuple[str, ...]:
    """Return the registered locale codes (always includes ``"en"``)."""
    return tuple(sorted(_LOCALES))


def _resolve(code: str, locale: str | None) -> tuple[str, str] | None:
    chosen = (locale or _active_locale).lower()
    table = _LOCALES.get(chosen)
    if table is not None and code in table:
        return table[code]
    en = _LOCALES[_DEFAULT_LOCALE]
    return en.get(code)


# --- lookups ----------------------------------------------------------------


def catalog_entry(code: str) -> ErrorCatalogEntry | None:
    """Return the English catalog entry for a stable code, or ``None``."""
    return ERROR_CATALOG.get(code)


def title_for(code: str, *, locale: str | None = None) -> str | None:
    """Return the human-readable title for a code in *locale* (or default)."""
    pair = _resolve(code, locale)
    return pair[0] if pair else None


def remediation_for(code: str, *, locale: str | None = None) -> str | None:
    """Return the remediation hint for a code in *locale* (or default)."""
    pair = _resolve(code, locale)
    return pair[1] if pair else None


def docs_anchor_for(code: str) -> str:
    """Return the in-page anchor for a code in the error reference."""
    return code.lower()


def docs_url_for(code: str) -> str | None:
    """Return the deep link into the error reference for a known code."""
    if code not in ERROR_CATALOG:
        return None
    return f"{DOCS_BASE_URL}#{docs_anchor_for(code)}"


# --- reference generation ---------------------------------------------------


def render_error_reference() -> str:
    """Render ``docs/reference/errors.md`` from the catalog (single source).

    A golden test asserts the committed page equals this output, so the docs
    links every error carries always resolve to a real anchor.
    """
    lines: list[str] = [
        "# Reference: error catalog",
        "",
        "Every `vincio` error derives from `VincioError` and carries a stable",
        "`.code`, a `.remediation` hint, and a `.docs_url` deep link into this",
        "page. Catch the whole family with one `except VincioError`, branch on",
        "`.code` for programmatic handling, and surface `.remediation` to users.",
        "",
        "```python",
        "from vincio import VincioError",
        "",
        "try:",
        "    app.run(\"...\")",
        "except VincioError as exc:",
        "    print(exc.code, exc.message)",
        "    print(\"fix:\", exc.remediation)",
        "    print(\"docs:\", exc.docs_url)",
        "```",
        "",
        "Error message strings are not part of the stable API; the `.code` values",
        "and this catalog are. This page is generated from",
        "`vincio.core.error_catalog` and gated for completeness — no error ships",
        "without an entry here.",
        "",
        "## Codes",
        "",
    ]
    for entry in ERROR_CATALOG.values():
        lines.append(f"### {entry.code}")
        lines.append("")
        lines.append(f"**{entry.title}.** {entry.remediation}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
