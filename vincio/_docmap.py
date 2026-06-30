"""The connected-docs capability map and docs graph (5.4).

The docs are ~80 leaf pages — a concept, a guide, a reference entry, and a
runnable example per subsystem — held together by one hand-ordered index and
little else. This module is the *connective tissue*: a single, reviewable source
of truth that binds every public ``app.*`` verb to the **concept** that explains
it, the **guide** that applies it, the **example** that demonstrates it, and the
**reference anchor** that specifies it, grouped by the six capability facades
(``runs`` / ``knowledge`` / ``governance`` / ``optimization`` / ``serving`` /
``training``).

From that one source it renders, deterministically and dependency-free (the way
:mod:`vincio._apiref` renders ``api-generated.md``):

* the **capability map** (``docs/reference/capability-map.md``),
* a single-sourced **Related** cross-link block injected into every concept and
  guide page so a reader traverses *laterally* instead of returning to the index,
* the staged **learning path** (``docs/learning-path.md``),
* the generated **app-method index** appended to ``docs/reference/api.md`` so
  every public ``app.*`` method is documented there, and
* ``llms.txt``, regenerated from ``vincio.__all__`` and gated for freshness
  exactly the way ``api-generated.md`` and the error catalog already are.

It also exposes the **docs-graph checks** the deepened completeness gate and the
``docs_conformance`` VincioBench family run: every internal link resolves, every
concept reaches a guide + an example + a reference anchor, every public ``app.*``
method appears in ``api.md``, no page is orphaned, and ``llms.txt`` is current.

This is connective tissue, not a new domain: deterministic, dependency-free, and
offline — never a hosted docs site, a search service, or a docs-as-a-service
control plane. ``vincio docs map`` / ``check`` / ``serve`` drive it from the CLI;
the richer link-renderer behind ``serve`` rides the opt-in ``vincio[docs]`` extra
while the map and the coverage check run on the standard library alone.
"""

from __future__ import annotations

import os.path
import posixpath
import re
from dataclasses import dataclass, field

from ._apiref import docstring_summary, public_symbols, symbol_kind, symbol_signature

__all__ = [
    "FACETS",
    "Topic",
    "TOPICS",
    "app_verbs",
    "topic_for_verb",
    "uncovered_verbs",
    "render_capability_map",
    "render_related_block",
    "render_learning_path",
    "render_app_method_index",
    "render_llms_txt",
    "MarkdownLink",
    "iter_markdown_links",
    "link_integrity_report",
    "capability_map_coverage",
    "navigation_reachability",
    "orphan_pages",
    "llms_txt_current",
    "docs_graph_report",
    "concept_pages",
    "guide_pages",
    "sync_docs",
]

# --- paths ------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCS = os.path.join(_ROOT, "docs")
_EXAMPLES = os.path.join(_ROOT, "examples")
_LLMS = os.path.join(_ROOT, "llms.txt")
_CAPABILITY_MAP = "docs/reference/capability-map.md"
_LEARNING_PATH = "docs/learning-path.md"
_API_REF = "docs/reference/api.md"
_INDEX = "docs/README.md"

# Managed-block markers — a generated region the renderer owns inside an
# otherwise hand-written page (the Related block on a concept/guide, the
# app-method index inside api.md). Editing between them is overwritten on sync.
_RELATED_BEGIN = "<!-- BEGIN GENERATED: related (vincio._docmap) -->"
_RELATED_END = "<!-- END GENERATED: related -->"
_APPINDEX_BEGIN = "<!-- BEGIN GENERATED: app-method-index (vincio._docmap) -->"
_APPINDEX_END = "<!-- END GENERATED: app-method-index -->"


# --- the six capability facades --------------------------------------------

# Ordered to match docs/README.md's reading order and the facade properties on
# ContextApp (runs / knowledge / governance / optimization / serving / training).
FACETS: tuple[tuple[str, str, str], ...] = (
    ("runs", "Runs", "Execute the pipeline: configure, run, stream, orchestrate, and produce deliverables."),
    ("knowledge", "Knowledge", "Feed the compiler: sources, retrieval, memory, structured data, and the analytics plane."),
    ("governance", "Governance", "Prove it is safe: compliance, security, privacy, identity, verification, assurance, and the cross-org trust fabric."),
    ("optimization", "Optimization", "Make it better and cheaper: cost, evaluation, self-improvement, routing, caching, and energy."),
    ("serving", "Serving", "Expose it: MCP / A2A servers, realtime, the governed fabric, deploy, and the edge runtime."),
    ("training", "Training", "Teach it: trace capture, dataset export, distillation, on-policy learning, local adaptation, federation, and skill acquisition."),
)
_FACET_KEYS = tuple(k for k, _t, _b in FACETS)


# --- the doc graph: one reviewable source of truth -------------------------


@dataclass(frozen=True)
class Topic:
    """One node of the doc graph: a coherent capability and everything that
    documents it.

    ``concept`` / ``guides`` / ``examples`` are repo-relative paths (POSIX);
    ``verbs`` are public ``ContextApp`` method names (async ``a*`` variants are
    attached automatically). A topic may carry no verbs (a pure concept↔guide
    binding such as the prompt compiler) and need not carry a concept.
    """

    key: str
    facet: str
    title: str
    summary: str
    concept: str | None = None
    guides: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    verbs: tuple[str, ...] = ()


def _c(name: str) -> str:
    return f"docs/concepts/{name}.md"


def _g(name: str) -> str:
    return f"docs/guides/{name}.md"


def _e(name: str) -> str:
    return f"examples/{name}.py"


TOPICS: tuple[Topic, ...] = (
    # ---- runs ----
    Topic(
        "execution", "runs", "Execution & the run pipeline",
        "Configure an app and run it — single-shot, streaming, background, or batch — "
        "through the governed pipeline that produces a validated, traced, costed result.",
        guides=(_g("performance"), _g("cookbook")),
        examples=(_e("01_quickstart"),),
        verbs=("run", "stream", "submit", "batch", "configure", "set_policy", "use_pack", "stats", "aclose"),
    ),
    Topic(
        "prompt", "runs", "Prompt compiler",
        "Prompts are compiled, not concatenated: a typed PromptSpec becomes a "
        "cache-aware, lint-checked, stable-prefix prompt.",
        concept=_c("prompt-compiler"),
        guides=(_g("optimize-context"),),
        examples=(_e("01_quickstart"),),
    ),
    Topic(
        "ergonomic", "runs", "The ergonomic front door",
        "One-line, task-shaped constructors and a fluent Flow that lower to the same "
        "governed packet as the verbose builder path.",
        concept=_c("ergonomic-surface"),
        guides=(_g("cookbook"),),
        examples=(_e("00_one_liners"),),
        verbs=("task",),
    ),
    Topic(
        "agents", "runs", "Agents & orchestration",
        "Bounded DAG agents, crews, durable graphs, planners, deep research, and the "
        "conversational assistant — the run pipeline driven in a loop.",
        concept=_c("agents"),
        guides=(_g("orchestrate-agents"), _g("add-tools")),
        examples=(_e("04_agents_and_tools"), _e("05_orchestration")),
        verbs=("agent", "crew", "graph", "workflow", "research", "assistant", "voice_agent",
               "predictor", "reasoning", "use_reasoning_controller"),
    ),
    Topic(
        "tools", "runs", "Tools & skills",
        "Register functions as permissioned, approval-gated tools, attach Agent Skills, "
        "and surface provider-native hosted tools.",
        guides=(_g("add-tools"), _g("agent-skills")),
        examples=(_e("04_agents_and_tools"),),
        verbs=("add_tool", "add_skill", "use_hosted_tools"),
    ),
    Topic(
        "structured-output", "runs", "Structured output",
        "Typed Pydantic contracts, constrained decoding, multi-schema routing, and "
        "bounded structure-only self-correction.",
        guides=(_g("structured-output"),),
        examples=(_e("06_structured_output"),),
        verbs=("add_output_schema", "enable_self_correction"),
    ),
    Topic(
        "test-time-compute", "runs", "Test-time compute",
        "Best-of-N, self-consistency, and beam search over the run pipeline for a "
        "harder answer at a controlled budget.",
        guides=(_g("reasoning"),),
        examples=(_e("11_advanced_context"),),
        verbs=("test_time_search",),
    ),
    Topic(
        "computer-use", "runs", "Computer-use action plane",
        "A grounded perceive → pre-gate → act → post-verify → undo loop over a "
        "pluggable screen backend.",
        guides=(_g("computer-use"),),
        examples=(_e("04_agents_and_tools"),),
        verbs=("computer_use", "enable_computer_use"),
    ),
    Topic(
        "generation", "runs", "Documents & media generation",
        "Cited DOCX/PDF/PPTX/HTML/Markdown, redlines, image generation, and TTS with "
        "C2PA provenance flowing out of a run.",
        guides=(_g("generate-documents"),),
        examples=(_e("09_security_governance"),),
        verbs=("build_document", "cited_report", "generate_image", "synthesize_speech",
               "generate_video", "edit_video"),
    ),
    # ---- knowledge ----
    Topic(
        "context", "knowledge", "Context packets & long-horizon governance",
        "The central unit: candidate evidence scored, deduped, budgeted, and packed; "
        "and the per-run governor that keeps a long horizon inside its token, residency, "
        "and KV-cache budget.",
        concept=_c("context-packets"),
        guides=(_g("optimize-context"), _g("performance")),
        examples=(_e("11_advanced_context"),),
        verbs=("govern_packet", "use_context_governor", "context_budget_report"),
    ),
    Topic(
        "retrieval", "knowledge", "Retrieval (RAG)",
        "BM25 + dense + learned-sparse + late-interaction fused in one RRF, query "
        "understanding, chunking, GraphRAG, and pushed-down filters.",
        concept=_c("retrieval"),
        guides=(_g("build-rag-app"), _g("connectors")),
        examples=(_e("02_retrieval_rag"),),
        verbs=("add_source", "ingest_files"),
    ),
    Topic(
        "memory", "knowledge", "Memory",
        "Layered, guarded, decaying, conflict-resolving, privacy-scoped memory with "
        "hybrid vector+graph recall and as-of correction.",
        concept=_c("memory"),
        guides=(_g("assistant"), _g("close-the-loop")),
        examples=(_e("03_memory"),),
        verbs=("add_memory", "remember", "recall", "enable_memory_os"),
    ),
    Topic(
        "inputs", "knowledge", "Rich inputs",
        "Audio transcripts and video brought in as first-class evidence the compiler "
        "scores beside text and images.",
        guides=(_g("connectors"), _g("video")),
        examples=(_e("02_retrieval_rag"),),
        verbs=("load_media", "load_video"),
    ),
    Topic(
        "tabular", "knowledge", "Tabular evidence",
        "A typed, columnar Dataset, a lossless header-once encoder, and TableEvidence "
        "the compiler scores token-cheap.",
        concept=_c("tabular-evidence"),
        guides=(_g("analyze-data"),),
        examples=(_e("13_tabular_evidence"),),
        verbs=("table_evidence", "register_dataset", "data_catalog"),
    ),
    Topic(
        "profiling", "knowledge", "Profiling, sampling & quality rails",
        "Bounded-memory profiling, reservoir/stratified sampling, fit-to-window under a "
        "token budget, and deterministic quality screening.",
        concept=_c("dataset-profiling"),
        guides=(_g("analyze-data"),),
        examples=(_e("14_dataset_profiling"),),
        verbs=("profile_dataset", "sample_dataset", "fit_dataset", "screen_data"),
    ),
    Topic(
        "text-to-query", "knowledge", "Governed text-to-query",
        "A question grounded to a read-only-verified, cost-bounded query with cell-level "
        "provenance that verifies offline.",
        concept=_c("governed-text-to-query"),
        guides=(_g("analyze-data"),),
        examples=(_e("15_governed_text_to_query"),),
        verbs=("query_data",),
    ),
    Topic(
        "data-analysis", "knowledge", "Data-analysis agent",
        "A bounded plan → query → inspect → refine → synthesize loop producing a cited "
        "narrative that verifies offline.",
        concept=_c("data-analysis-agent"),
        guides=(_g("analyze-data"),),
        examples=(_e("16_data_analysis_agent"),),
        verbs=("analyze_data",),
    ),
    Topic(
        "charts", "knowledge", "Charts & cited artifacts",
        "A spec-driven chart that is content-bound (C2PA) and data-bound (a back-reference "
        "to the exact source cells).",
        concept=_c("charts-and-cited-artifacts"),
        guides=(_g("analyze-data"), _g("generate-documents")),
        examples=(_e("17_charts_cited_artifacts"),),
        verbs=("generate_chart",),
    ),
    Topic(
        "streaming", "knowledge", "Streaming & out-of-core",
        "A lazy, re-iterable RowStream over a source larger than memory, a bounded-memory "
        "group-by, and a streaming candidate pre-filter.",
        concept=_c("streaming-and-out-of-core"),
        guides=(_g("analyze-data"), _g("performance")),
        examples=(_e("18_streaming_out_of_core"),),
        verbs=("map_stream", "stream_dataset", "aggregate_stream"),
    ),
    Topic(
        "semantic-layer", "knowledge", "Semantic layer & governed metrics",
        "Measures, dimensions, and derived columns defined once so a question maps to a "
        "governed metric computed one way everywhere.",
        concept=_c("semantic-layer-and-governed-metrics"),
        guides=(_g("analyze-data"),),
        examples=(_e("19_semantic_layer_governed_metrics"),),
        verbs=("semantic_layer", "query_metric", "metric_lineage"),
    ),
    Topic(
        "realtime-analytics", "knowledge", "Real-time & streaming analytics",
        "The profiling, query, governed-metric, and quality primitives over an unbounded "
        "event stream, windowed (tumbling / sliding / session) inside a bounded footprint.",
        concept=_c("realtime-streaming-analytics"),
        guides=(_g("analyze-data"),),
        examples=(_e("23_realtime_streaming_analytics"),),
        verbs=("stream_analytics",),
    ),
    Topic(
        "data-engagement", "knowledge", "Data engagement (the analytics capstone)",
        "The whole analytics plane threaded into a hash-chained, signed, data-bound "
        "DataNarrative that verifies offline.",
        concept=_c("data-engagement"),
        guides=(_g("analyze-data"),),
        examples=(_e("20_data_engagement"),),
        verbs=("data_engagement",),
    ),
    # ---- governance ----
    Topic(
        "compliance", "governance", "Cards, compliance & EU AI Act",
        "Model and system cards, a framework coverage matrix, an AI-BOM, lineage, content "
        "marking, and the EU AI Act conformity pack.",
        guides=(_g("governance"),),
        examples=(_e("09_security_governance"),),
        verbs=("model_card", "system_card", "compliance_report", "aibom", "trace_lineage",
               "mark_output", "risk_tier", "annex_iv", "fria"),
    ),
    Topic(
        "residency-erasure", "governance", "Residency, erasure & consent",
        "Residency-aware egress refusal, provable right-to-erasure by source, and a "
        "consent ledger consulted by access and recall.",
        guides=(_g("governance"),),
        examples=(_e("09_security_governance"),),
        verbs=("check_residency", "set_residency", "erase_source", "use_consent_ledger"),
    ),
    Topic(
        "security-rails", "governance", "Rails, tenancy & security",
        "Programmable input/output rails, custom rail predicates, and tenant-scoped "
        "retrieval and memory.",
        guides=(_g("reliability-guardrails"),),
        examples=(_e("09_security_governance"),),
        verbs=("add_rail", "register_rail_predicate", "tenant_filter"),
    ),
    Topic(
        "privacy", "governance", "Differential-privacy accounting",
        "Per-subject (ε, δ) accounting for consolidation and federated rounds, with "
        "bounded per-member influence.",
        guides=(_g("differential-privacy"),),
        examples=(_e("08_optimization_self_improvement"),),
        verbs=("set_privacy_budget", "privacy_report", "use_privacy_accountant"),
    ),
    Topic(
        "identity", "governance", "Agent identity & delegation",
        "DID-based self-certifying identity, attenuating delegation chains, and signed, "
        "verifiable credentials folded into admission.",
        guides=(_g("agent-identity"),),
        examples=(_e("09_security_governance"),),
        verbs=("identity", "issue_credential", "principal_for", "use_identity"),
    ),
    Topic(
        "governance-verification", "governance", "Governance-invariant verification",
        "Bounded model checking that proves containment, residency, the budget cap, and "
        "the erasure binding hold across their whole state space.",
        guides=(_g("governance-verification"),),
        examples=(_e("09_security_governance"),),
        verbs=("verify_governance",),
    ),
    Topic(
        "verified-reasoning", "governance", "Verified reasoning & shielding",
        "Proof-carrying answers checked by deterministic kernels, runtime behaviour "
        "shielding, and proof-carrying synthesized programs.",
        guides=(_g("verified-reasoning"),),
        examples=(_e("09_security_governance"),),
        verbs=("verify_reasoning", "behavior_monitor", "shield", "use_shield", "synthesize_program"),
    ),
    Topic(
        "assurance", "governance", "Continuous assurance & certification",
        "An assurance argument tree bound to the platform's own verdicts, with freshness "
        "horizons, incidents, and a certification report.",
        guides=(_g("assurance"),),
        examples=(_e("09_security_governance"),),
        verbs=("assurance_case", "certify"),
    ),
    Topic(
        "negotiation", "governance", "Negotiation & contracting",
        "A bounded offer/counter-offer bargain that mints a typed, signed, "
        "offline-verifiable Contract enforced like any other budget.",
        guides=(_g("negotiation"),),
        examples=(_e("12_cross_org_economy"),),
        verbs=("negotiate", "enforce_contract", "serve_negotiation"),
    ),
    Topic(
        "choreography", "governance", "Cross-org workflow choreography",
        "Durable, compensating sagas across organizations over A2A and the negotiated "
        "contract, resumable and hash-chained.",
        guides=(_g("choreography"),),
        examples=(_e("12_cross_org_economy"),),
        verbs=("choreograph", "resume_choreography", "serve_choreography"),
    ),
    Topic(
        "settlement", "governance", "Settlement, clearing & the trust fabric",
        "Metered, signed, offline-verifiable settlement of delivered work; multilateral "
        "netting; arbitration; portable reputation; collateral; solvency proofs; and the "
        "cross-org engagement capstone.",
        guides=(_g("settlement"),),
        examples=(_e("12_cross_org_economy"),),
        verbs=("meter", "settle", "settle_saga", "settle_escrow", "settlement_report",
               "use_settlement_book", "clear_settlements", "arbitrate", "attest_reputation",
               "revoke_attestation", "import_reputation", "serve_attestations", "gather_reputation",
               "admit", "post_escrow", "post_collateral_pool", "draw_pool", "guard_collateral",
               "attest_custody", "attest_liabilities", "check_completeness", "inclusion_proof",
               "prove_solvency", "discharge_liability", "check_history_consistency",
               "check_root_consistency", "build_seniority_schedule", "resolve_insolvency",
               "build_set_off_statement", "cross_org_engagement"),
    ),
    # ---- optimization ----
    Topic(
        "cost", "optimization", "Cost & FinOps",
        "A per-run cost report, cost budgets, provider resolution, and prompt caching.",
        guides=(_g("cost-and-reliability"),),
        examples=(_e("07_evaluation_observability"),),
        verbs=("set_cost_budget", "cost_report", "resolve_provider", "enable_prompt_caching"),
    ),
    Topic(
        "observability", "optimization", "Observability",
        "One trace, one cost, and one hash-chained audit entry per run, with OpenTelemetry "
        "export, a local viewer, and a served alerting plane.",
        concept=_c("observability"),
        guides=(_g("cost-and-reliability"), _g("performance")),
        examples=(_e("07_evaluation_observability"),),
    ),
    Topic(
        "energy", "optimization", "Energy & carbon accounting",
        "A deterministic, offline per-run energy/carbon estimate on the cost-report "
        "surface, budgeted and refused like a dollar.",
        guides=(_g("cost-and-reliability"),),
        examples=(_e("11_advanced_context"),),
        verbs=("set_energy_budget", "energy_report", "use_energy_accounting"),
    ),
    Topic(
        "evaluation", "optimization", "Evaluation & continuous quality",
        "Golden datasets, 30+ metrics, calibrated judges and ensembles, experiments with "
        "significance, online evaluation, and metric-backed guardrails.",
        concept=_c("evals"),
        guides=(_g("run-evals"), _g("test-llm-apps"), _g("agentic-eval")),
        examples=(_e("07_evaluation_observability"),),
        verbs=("add_evaluator", "add_validator", "add_optimizer", "add_online_evaluator",
               "add_metric_rail", "aflush_online", "evaluate", "eval_target", "experiment",
               "calibrate_judge"),
    ),
    Topic(
        "self-improvement", "optimization", "Self-improvement",
        "The closed loop — trace → dataset → eval → optimize → promote — with reflective "
        "optimization, learned compression, and learned budgets.",
        guides=(_g("close-the-loop"), _g("optimize-context")),
        examples=(_e("08_optimization_self_improvement"),),
        verbs=("improvement_loop", "reflective_optimize", "self_improvement", "gate_compression",
               "use_learned_compression", "use_learned_budgets", "use_semantic_context_scoring"),
    ),
    Topic(
        "routing", "optimization", "Routing, cascades & rotation",
        "Cost/latency routing over registry pricing, model cascades, guarded bandits, and "
        "swap-regression rotation with canary and shadow.",
        guides=(_g("cost-and-reliability"),),
        examples=(_e("08_optimization_self_improvement"),),
        verbs=("use_cascade", "use_router", "use_bandit_router", "gate_swap", "swap_regression",
               "watch_lifecycle", "shadow", "canary"),
    ),
    Topic(
        "caching", "optimization", "Learned caching & KV reuse",
        "A learned semantic cache with a threshold tuned from traces and near-miss "
        "KV-prefix reuse, each gated against regression.",
        guides=(_g("performance"),),
        examples=(_e("11_advanced_context"),),
        verbs=("use_semantic_cache", "semantic_cache_report", "use_kv_prefix_reuse", "kv_prefix_report"),
    ),
    Topic(
        "reputation", "optimization", "Cross-fleet reputation weighting",
        "A per-member reliability score earned from how each contribution fared against "
        "the no-regression gate, weighting aggregation without singling a member out.",
        guides=(_g("close-the-loop"),),
        examples=(_e("08_optimization_self_improvement"),),
        verbs=("use_reputation_ledger", "reputation_report"),
    ),
    # ---- serving ----
    Topic(
        "serving", "serving", "Serving surfaces & protocols",
        "Serve tools and agents over MCP and A2A, surface a server-driven UI resource, and "
        "promote a candidate through a canary-gated deploy.",
        guides=(_g("mcp"), _g("a2a")),
        examples=(_e("10_interop_and_protocols"),),
        verbs=("serve_mcp", "serve_a2a", "deploy", "mcp_app", "add_mcp_server", "add_mcp_from_registry"),
    ),
    Topic(
        "realtime", "serving", "Voice & realtime",
        "Bidirectional voice/realtime sessions with VAD, interruption, and in-session "
        "tools through the permissioned runtime.",
        guides=(_g("realtime"),),
        examples=(_e("10_interop_and_protocols"),),
        verbs=("realtime_session",),
    ),
    Topic(
        "fabric", "serving", "The governed agent fabric",
        "A governed, discoverable agent directory over the A2A Agent Card with allow-list "
        "gated, audited resolution.",
        guides=(_g("agent-fabric"),),
        examples=(_e("10_interop_and_protocols"),),
        verbs=("agent_directory",),
    ),
    Topic(
        "edge", "serving", "Edge / WASM runtime",
        "The dependency-free context-engineering core packaged for constrained and browser "
        "targets, parity-not-a-fork.",
        guides=(_g("edge"),),
        examples=(_e("11_advanced_context"),),
        verbs=("edge_runtime",),
    ),
    # ---- training ----
    Topic(
        "training", "training", "Trace capture, export & distillation",
        "Record grounded traces, curate faithful fine-tuning sets, run gated distillation, "
        "and learn on-policy from verifiable rewards.",
        guides=(_g("close-the-loop"),),
        examples=(_e("08_optimization_self_improvement"),),
        verbs=("enable_training_capture", "export_training_set", "distill", "learn"),
    ),
    Topic(
        "local-adaptation", "training", "On-device local adaptation",
        "A parameter-efficient adapter fit on-device from grounded traces, applied behind "
        "a no-regression gate and reversible.",
        guides=(_g("close-the-loop"),),
        examples=(_e("08_optimization_self_improvement"),),
        verbs=("adapt_locally", "local_adaptation", "use_local_adapter"),
    ),
    Topic(
        "federated", "training", "Federated self-improvement",
        "A member's raw-text-free, clipped, DP-noised subspace update merged into a shared "
        "consensus subspace under a k-anonymity floor.",
        guides=(_g("differential-privacy"),),
        examples=(_e("08_optimization_self_improvement"),),
        verbs=("contribute_federated", "federated_improvement", "adopt_federated"),
    ),
    Topic(
        "skill-acquisition", "training", "Autonomous skill acquisition",
        "An open-ended curriculum that proposes, attempts, verifies, distills, and promotes "
        "new skills under the no-regression gate.",
        guides=(_g("skill-acquisition"),),
        examples=(_e("08_optimization_self_improvement"),),
        verbs=("cultivate",),
    ),
)


# Guides that document cross-cutting matter rather than one capability. They still
# carry a single-sourced Related block so a reader traverses laterally; they bind
# to their sibling guides plus the navigational hubs.
_META_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "recipes-and-extending",
        "Recipes & extending Vincio",
        (_g("cookbook"), _g("integrations"), _g("plugins"), _g("vertical-packs")),
    ),
    (
        "migrating",
        "Migrating from another library",
        (_g("migrate-from-langchain"), _g("migrate-from-llamaindex"),
         _g("migrate-from-mem0"), _g("migrate-from-ragas")),
    ),
)


# --- verb ⇄ topic resolution -----------------------------------------------


def app_verbs() -> list[str]:
    """Return every public ``ContextApp`` method name, sorted."""
    import inspect

    import vincio

    return sorted(
        name
        for name, obj in inspect.getmembers(vincio.ContextApp, predicate=inspect.isfunction)
        if not name.startswith("_")
    )


def _declared_verbs() -> set[str]:
    declared: set[str] = set()
    for topic in TOPICS:
        declared.update(topic.verbs)
    return declared


def _verb_topic_index() -> dict[str, Topic]:
    """Map every public verb to its topic, attaching async ``a*`` variants to
    their sync sibling's topic automatically."""
    index: dict[str, Topic] = {}
    for topic in TOPICS:
        for verb in topic.verbs:
            index[verb] = topic
    # Attach async variants whose sync sibling is declared.
    for verb in app_verbs():
        if verb in index:
            continue
        if verb.startswith("a") and verb[1:] in index:
            index[verb] = index[verb[1:]]
    return index


def topic_for_verb(verb: str) -> Topic | None:
    """Return the topic that owns *verb*, or ``None`` if unassigned."""
    return _verb_topic_index().get(verb)


def uncovered_verbs() -> list[str]:
    """Return public ``app.*`` verbs the doc graph does not yet place.

    The map gate requires this to be empty so a newly-added verb cannot ship
    without a home in the capability map.
    """
    index = _verb_topic_index()
    return sorted(v for v in app_verbs() if v not in index)


# --- link helpers -----------------------------------------------------------

_LINK_RE = re.compile(r"(?<!\!)\[(?P<text>[^\]]*)\]\((?P<target>[^)]+)\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_ANCHOR_DEFN_RE = re.compile(r'<a\s+(?:id|name)=["\']([^"\']+)["\']', re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_INLINE_CODE_RE = re.compile(r"`[^`]*`")

# Generated index pages whose outbound links mirror third-party docstrings we
# neither author nor control (Pydantic injects `[Models](../concepts/models.md)`
# usage links into BaseModel docstrings). They are gated for freshness elsewhere
# (tests/test_api_reference.py); the authored-doc link graph excludes their
# *outbound* links, but they remain valid inbound link targets.
_LINK_SCAN_EXCLUDE = frozenset({"docs/reference/api-generated.md"})


@dataclass(frozen=True)
class MarkdownLink:
    """An internal markdown link located in a docs page."""

    source: str  # repo-relative POSIX path of the page the link lives in
    text: str
    target: str  # the raw link target, e.g. "../guides/x.md#anchor"


def _rel(from_file: str, to_file: str) -> str:
    """Repo-relative POSIX link from *from_file* to *to_file*."""
    return posixpath.relpath(to_file, start=posixpath.dirname(from_file))


def _slugify(heading: str) -> str:
    """GitHub-flavoured heading slug (lowercase, punctuation stripped, spaces→-)."""
    text = heading.strip().lower()
    text = text.replace("`", "")
    text = re.sub(r"[^\w\s-]", "", text)
    text = text.replace(" ", "-")
    return text.strip("-")


def _read(repo_rel: str) -> str:
    with open(os.path.join(_ROOT, repo_rel), encoding="utf-8") as fh:
        return fh.read()


def _anchors_in(repo_rel: str) -> set[str]:
    """Heading slugs plus explicit ``<a id=...>`` anchors in a markdown file."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    try:
        text = _read(repo_rel)
    except OSError:
        return anchors
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            slug = _slugify(m.group(2))
            if slug in seen:
                seen[slug] += 1
                anchors.add(f"{slug}-{seen[slug]}")
            else:
                seen[slug] = 0
                anchors.add(slug)
        for am in _ANCHOR_DEFN_RE.finditer(line):
            anchors.add(am.group(1))
    return anchors


def _strip_code(text: str) -> str:
    """Blank out fenced code blocks and inline code spans (links there are code,
    not navigation), preserving line count so anchors/headings still resolve."""
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in text.splitlines():
        m = _FENCE_RE.match(line)
        if m:
            token = m.group(1)
            if not in_fence:
                in_fence, fence_marker = True, token[0]
                out.append("")
                continue
            if token[0] == fence_marker:
                in_fence, fence_marker = False, ""
                out.append("")
                continue
        out.append("" if in_fence else _INLINE_CODE_RE.sub("", line))
    return "\n".join(out)


def iter_markdown_links(repo_rel: str) -> list[MarkdownLink]:
    """Return the internal markdown links in a docs page (skips http/mailto and
    links that live inside code blocks)."""
    links: list[MarkdownLink] = []
    text = _strip_code(_read(repo_rel))
    for m in _LINK_RE.finditer(text):
        target = m.group("target").strip()
        if target.startswith(("http://", "https://", "mailto:", "tel:")):
            continue
        if " " in target.split("#", 1)[0]:
            # Real link targets have no whitespace in the path part.
            continue
        links.append(MarkdownLink(source=repo_rel, text=m.group("text"), target=target))
    return links


def _resolve_link(link: MarkdownLink) -> str | None:
    """Return an error string if a link does not resolve, else ``None``."""
    target = link.target
    path_part, _, anchor = target.partition("#")
    if not path_part:
        # Pure in-page anchor: must exist in the source file.
        if anchor and anchor not in _anchors_in(link.source):
            return f"{link.source}: in-page anchor #{anchor} not found"
        return None
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(link.source), path_part))
    abs_path = os.path.join(_ROOT, resolved)
    if not os.path.exists(abs_path):
        return f"{link.source}: link target {target!r} → missing {resolved}"
    if anchor and resolved.endswith(".md"):
        if anchor not in _anchors_in(resolved):
            return f"{link.source}: anchor #{anchor} not found in {resolved}"
    return None


def _all_doc_pages() -> list[str]:
    """Every markdown page under docs/, repo-relative POSIX, sorted."""
    pages: list[str] = []
    for dirpath, _dirs, files in os.walk(_DOCS):
        for name in files:
            if name.endswith(".md"):
                rel = os.path.relpath(os.path.join(dirpath, name), _ROOT)
                pages.append(rel.replace(os.sep, "/"))
    return sorted(pages)


def concept_pages() -> list[str]:
    """Every concept page, repo-relative POSIX, sorted."""
    return sorted(p for p in _all_doc_pages() if p.startswith("docs/concepts/"))


def guide_pages() -> list[str]:
    """Every guide page, repo-relative POSIX, sorted."""
    return sorted(p for p in _all_doc_pages() if p.startswith("docs/guides/"))


# --- renderers --------------------------------------------------------------


def _summary_for(verb: str) -> str:
    import vincio

    summary = docstring_summary(getattr(vincio.ContextApp, verb))
    return summary or "—"


def _link(label: str, repo_rel_target: str, *, from_file: str, anchor: str = "") -> str:
    href = _rel(from_file, repo_rel_target)
    if anchor:
        href = f"{href}#{anchor}"
    return f"[{label}]({href})"


def _example_label(example_repo_rel: str) -> str:
    return posixpath.basename(example_repo_rel)


def render_capability_map() -> str:
    """Render ``docs/reference/capability-map.md`` from the doc graph."""
    here = _CAPABILITY_MAP
    verbs = set(app_verbs())
    index = _verb_topic_index()
    total = len([v for v in verbs if v in index])
    lines: list[str] = [
        "# Reference: capability map",
        "",
        "This page is generated by `vincio._docmap` from the doc graph — the single",
        "source of truth that binds every public `app.*` verb to the **concept** that",
        "explains it, the **guide** that applies it, the **example** that demonstrates",
        "it, and the **reference** that specifies it. It is the map of *what Vincio can",
        "do* and *where each capability is documented*, grouped by the six capability",
        "facades (`app.runs` / `app.knowledge` / `app.governance` / `app.optimization`",
        "/ `app.serving` / `app.training`).",
        "",
        f"It covers **{total}** public `ContextApp` methods. It is gated for coverage:",
        "every public `app.*` verb appears here, every link resolves, and every concept",
        "reaches a guide, an example, and a reference anchor. For the exhaustive",
        "docstring-driven symbol index see [api-generated.md](api-generated.md); for the",
        "curated narrative see [api.md](api.md); for a staged reading order see the",
        f"[learning path]({_rel(here, _LEARNING_PATH)}).",
        "",
    ]
    for facet_key, facet_title, facet_blurb in FACETS:
        facet_topics = [t for t in TOPICS if t.facet == facet_key]
        lines.append(f"## {facet_title}")
        lines.append("")
        lines.append(f"_{facet_blurb}_")
        lines.append("")
        for topic in facet_topics:
            lines.append(f"### {topic.title}")
            lines.append("")
            lines.append(topic.summary)
            lines.append("")
            nav: list[str] = []
            if topic.concept:
                nav.append("Concept: " + _link(
                    _concept_title(topic.concept), topic.concept, from_file=here))
            if topic.guides:
                nav.append("Guides: " + ", ".join(
                    _link(_guide_title(g), g, from_file=here) for g in topic.guides))
            if topic.examples:
                nav.append("Examples: " + ", ".join(
                    _link(_example_label(e), e, from_file=here) for e in topic.examples))
            nav.append("Reference: " + _link(
                facet_title, _API_REF, from_file=here, anchor=facet_key))
            lines.append(" · ".join(nav))
            lines.append("")
            topic_verbs = sorted(v for v in verbs if index.get(v) is topic)
            if topic_verbs:
                lines.append("| Method | What it does |")
                lines.append("|---|---|")
                for verb in topic_verbs:
                    lines.append(f"| `app.{verb}` | {_summary_for(verb)} |")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _concept_title(repo_rel: str) -> str:
    for topic in TOPICS:
        if topic.concept == repo_rel:
            return topic.title
    return _title_from_page(repo_rel)


def _guide_title(repo_rel: str) -> str:
    return _title_from_page(repo_rel)


_TITLE_CACHE: dict[str, str] = {}


def _title_from_page(repo_rel: str) -> str:
    """The first ``# Heading`` of a page, used as a human link label."""
    if repo_rel in _TITLE_CACHE:
        return _TITLE_CACHE[repo_rel]
    title = posixpath.splitext(posixpath.basename(repo_rel))[0].replace("-", " ").title()
    try:
        for line in _read(repo_rel).splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
    except OSError:
        pass
    # Strip a leading "Guide: " / "Concept: " so the Related block can apply a
    # single, consistent label without doubling it.
    for prefix in ("Guide:", "Concept:"):
        if title.lower().startswith(prefix.lower()):
            title = title[len(prefix):].strip()
            break
    _TITLE_CACHE[repo_rel] = title
    return title


def _topics_for_page(repo_rel: str) -> list[Topic]:
    return [t for t in TOPICS if t.concept == repo_rel or repo_rel in t.guides]


def render_related_block(repo_rel: str) -> str:
    """Render the single-sourced **Related** block for a concept or guide page.

    Returns the managed block (markers included). The links let a reader move
    *laterally* — to the concept that explains the page, the guides that apply
    it, the example that demonstrates it, the capability map, and the learning
    path — instead of returning to the index.
    """
    here = repo_rel
    items: list[str] = []
    seen: set[str] = {repo_rel}

    def add(label: str, target: str, anchor: str = "") -> None:
        key = f"{target}#{anchor}"
        if target in seen or key in seen:
            return
        seen.add(target)
        seen.add(key)
        items.append("- " + _link(label, target, from_file=here, anchor=anchor))

    topics = _topics_for_page(repo_rel)
    if topics:
        # Concept of this page (when it is a guide), or guides (when a concept).
        for topic in topics:
            if topic.concept and topic.concept != repo_rel:
                add("Concept: " + _concept_title(topic.concept), topic.concept)
        for topic in topics:
            for guide in topic.guides:
                if guide != repo_rel:
                    add("Guide: " + _guide_title(guide), guide)
        for topic in topics:
            for example in topic.examples:
                add("Example: " + _example_label(example), example)
        # A sibling concept in the same facet, for lateral discovery.
        facet = topics[0].facet
        for sib in TOPICS:
            if sib.facet == facet and sib.concept and sib.concept != repo_rel:
                add("Concept: " + sib.title, sib.concept)
                break
        facet_key = topics[0].facet
        add("Reference: capability map", _CAPABILITY_MAP)
        add("Reference: API", _API_REF, anchor=facet_key)
    else:
        # A cross-cutting guide: bind to its sibling guides in the meta group.
        for _key, group_title, members in _META_GROUPS:
            if repo_rel in members:
                items.append(f"_{group_title}:_")
                for member in members:
                    if member != repo_rel:
                        add(_guide_title(member), member)
                break
        add("Reference: capability map", _CAPABILITY_MAP)

    add("Documentation index", _INDEX)
    add("Learning path", _LEARNING_PATH)

    body = "\n".join(
        [_RELATED_BEGIN, "", "## Related", "", *items, "", _RELATED_END]
    )
    return body


def render_learning_path() -> str:
    """Render ``docs/learning-path.md`` — a staged getting-started → depth spine."""
    here = _LEARNING_PATH

    def g(name: str, label: str) -> str:
        return _link(label, _g(name), from_file=here)

    def c(name: str, label: str) -> str:
        return _link(label, _c(name), from_file=here)

    def ex(name: str, label: str) -> str:
        return _link(label, _e(name), from_file=here)

    def ref(name: str, label: str) -> str:
        return _link(label, f"docs/reference/{name}", from_file=here)

    lines = [
        "# Learning path",
        "",
        "A staged route through Vincio, from your first grounded app to the full",
        "platform. Each stage builds on the one before; follow it top to bottom, or",
        "jump to the stage that matches what you are building. The",
        f"[documentation index]({_rel(here, _INDEX)}) is the exhaustive map and the",
        f"[capability map]({_rel(here, _CAPABILITY_MAP)}) binds every `app.*` verb to",
        "the page that documents it.",
        "",
        "## Stage 1 — Get running",
        "",
        "Install, scaffold, and run your first grounded app offline on the deterministic",
        "mock provider.",
        "",
        f"- {ref('../getting-started.md', 'Getting started')} — install, scaffold, first run, first eval.",
        f"- {ex('00_one_liners', 'Example: the one-line front door')} and {c('ergonomic-surface', 'the ergonomic surface')}.",
        f"- {ex('01_quickstart', 'Example: the five-minute tour')}.",
        f"- {g('cookbook', 'Cookbook')} — short, runnable recipes to copy.",
        "",
        "## Stage 2 — The core model",
        "",
        "Understand what happens between your input and the validated output.",
        "",
        f"- {c('context-packets', 'Context packets & the context compiler')} — the central unit.",
        f"- {c('prompt-compiler', 'The prompt compiler')} — prompts are compiled, not concatenated.",
        f"- {c('retrieval', 'Retrieval')} and {g('build-rag-app', 'build a RAG app')}.",
        f"- {c('memory', 'Memory')} — scoped, decaying, conflict-resolving recall.",
        "",
        "## Stage 3 — Build a real application",
        "",
        "Add tools, typed output, guardrails, and the structured-data plane.",
        "",
        f"- {g('add-tools', 'Add tools')} and {g('structured-output', 'structured output')}.",
        f"- {g('reliability-guardrails', 'Reliability & guardrails')}.",
        f"- {c('tabular-evidence', 'Tabular evidence')} → {g('analyze-data', 'analyze data')} → {c('data-engagement', 'the data engagement')}.",
        f"- {g('generate-documents', 'Generate documents & media')}.",
        "",
        "## Stage 4 — Evaluate and improve",
        "",
        "Turn quality into numbers, gate CI on them, and close the optimization loop.",
        "",
        f"- {c('evals', 'Evaluation')} → {g('run-evals', 'run evals')} → {g('test-llm-apps', 'test with pytest')}.",
        f"- {g('agentic-eval', 'Agentic evaluation & continuous quality')}.",
        f"- {g('close-the-loop', 'Close the loop')} and {g('optimize-context', 'optimize context')}.",
        f"- {g('cost-and-reliability', 'Cost, reliability & scale')} and {g('performance', 'performance')}.",
        "",
        "## Stage 5 — Orchestrate and interoperate",
        "",
        "Compose multi-agent systems and connect them across processes and vendors.",
        "",
        f"- {c('agents', 'Agents & workflows')} → {g('orchestrate-agents', 'orchestrate multi-agent systems')}.",
        f"- {g('mcp', 'MCP')}, {g('a2a', 'A2A')}, {g('agent-skills', 'Agent Skills')}, and {g('agent-fabric', 'the agent fabric')}.",
        f"- {g('reasoning', 'Reasoning control')} and {g('realtime', 'voice & realtime')}.",
        "",
        "## Stage 6 — Govern, secure, and assure",
        "",
        "Produce compliance evidence and formal guarantees from the live system.",
        "",
        f"- {g('governance', 'Enterprise governance')} and {g('governance-verification', 'formal verification')}.",
        f"- {g('verified-reasoning', 'Verified reasoning')}, {g('assurance', 'continuous assurance')}, and {g('agent-identity', 'agent identity')}.",
        f"- {g('differential-privacy', 'Differential-privacy memory & training')}.",
        f"- {ref('../security/threat-model.md', 'The threat model')}.",
        "",
        "## Stage 7 — The cross-organization economy & advanced runtimes",
        "",
        "Transact across organizations and run Vincio beyond the default server path.",
        "",
        f"- {g('negotiation', 'Negotiation & contracting')} → {g('choreography', 'choreography')} → {g('settlement', 'settlement')}.",
        f"- {ex('12_cross_org_economy', 'Example: the cross-org economy')}.",
        f"- {g('edge', 'Edge / WASM runtime')}, {g('computer-use', 'computer-use')}, {g('video', 'video')}, and {g('skill-acquisition', 'skill acquisition')}.",
        "",
        "## Keep going",
        "",
        f"- {ref('capability-map.md', 'Capability map')} — every `app.*` verb and where it is documented.",
        f"- {ref('api.md', 'API reference')} and {ref('cli.md', 'CLI reference')}.",
        f"- {g('migrate-from-langchain', 'Migrating from another library')}.",
        "",
        "Run `vincio docs check` to prove this graph is intact, or `vincio docs map`",
        "to regenerate the capability map.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_app_method_index() -> str:
    """Render the managed app-method index block appended to ``api.md``.

    Guarantees every public ``app.*`` method is documented in the curated API
    reference, grouped under a stable per-facet anchor the capability map links
    to.
    """
    index = _verb_topic_index()
    verbs = app_verbs()
    lines: list[str] = [
        _APPINDEX_BEGIN,
        "",
        "## ContextApp methods by facet",
        "",
        f"Every one of the **{len(verbs)}** public `ContextApp` methods, grouped by the",
        "six capability facades. Generated by `vincio._docmap` from the doc graph and",
        "gated for completeness, so a newly-added method cannot ship undocumented. Each",
        "facet anchor is the reference target the",
        "[capability map](capability-map.md) links every verb to.",
        "",
    ]
    for facet_key, facet_title, _blurb in FACETS:
        facet_verbs = sorted(v for v in verbs if (index.get(v) and index[v].facet == facet_key))
        lines.append(f"### {facet_title}")
        lines.append("")
        lines.append("| Method | What it does |")
        lines.append("|---|---|")
        for verb in facet_verbs:
            lines.append(f"| `app.{verb}` | {_summary_for(verb)} |")
        lines.append("")
    lines.append(_APPINDEX_END)
    return "\n".join(lines)


# --- llms.txt ---------------------------------------------------------------

_LLMS_PREAMBLE = """\
# Vincio

> Vincio is a Python platform for building AI applications you can trust in
> production. It compiles everything that goes *into* a model — prompts, memory,
> retrieved evidence, tools, schemas, and policies — into an optimized,
> validated, observable, provider-neutral **context packet**, then checks,
> measures, and traces everything that comes *out*.

Package: `pip install vincio` · Python 3.11+ · Apache 2.0 · SemVer.
Main entry point: `from vincio import ContextApp`.

This file is generated from `vincio.__all__` by `vincio._docmap` and gated for
freshness (a new public symbol must appear here), the way `api-generated.md` and
the error catalog are. It is a complete, machine-readable digest of the public
surface; for prose see the docs under `docs/` (start at `docs/learning-path.md`
and `docs/reference/capability-map.md`).

## Install

```bash
pip install vincio                  # core (only pydantic, httpx, pyyaml, typing-extensions)
pip install "vincio[openai]"        # + a provider (also: anthropic, google, mistral)
pip install "vincio[chroma]"        # + a vector store (also: pinecone, lancedb, pgvector, ...)
pip install "vincio[server]"        # + the FastAPI server (vincio serve / from vincio.server import create_app)
pip install "vincio[all]"           # every optional integration
```

Every heavy integration (vector stores, OCR, server, OpenTelemetry, charts, ...)
is an opt-in extra; the core is dependency-light and runs offline.

## Quickstart

```python
from vincio import ContextApp

# Uses your configured provider (set a provider+key, e.g. provider="openai" with
# OPENAI_API_KEY in the env). The DEFAULT provider is OpenAI, so configure one.
app = ContextApp(name="docs_qa", provider="openai", model="gpt-4o-mini")
app.add_source("docs", path="./docs", retrieval="hybrid")
app.set_policy("answer_only_from_sources", True)
result = app.run("How do I configure SSO?")
result.output; result.citations; result.trace_id; result.cost_usd

# Run FULLY OFFLINE (no key, no network): pass the bundled deterministic mock.
# It auto-generates schema-valid output, so the whole pipeline runs in CI.
from vincio.providers import MockProvider
app = ContextApp(name="dev", provider=MockProvider(), model="mock-1")

# Typed output: pass a Pydantic class; result.output is a validated instance
# (the mock fills it schema-valid offline; a real model fills it for real).
from pydantic import BaseModel
class Triage(BaseModel):
    label: str; confidence: float
app = ContextApp(name="triage", provider=MockProvider(), model="mock-1", output_schema=Triage)
app.run("export button 500s").output.label
```

## Ergonomic front door (vincio.tasks)

One-line, task-shaped constructors over `ContextApp` for the common jobs, plus
one fluent `Flow`. Each is `@experimental`, re-exported at top level, and lowers
to the exact same governed `ContextApp.run` packet as the verbose builder path
(retrieval, grounding, validation, rails, budgets, tracing, audit chain all
apply unchanged). `.app` on every facade is the escape hatch to all deep methods.

```python
from vincio import rag, extractor, tool_agent, evaluation, chat, Flow

rag("./docs").ask("How do I configure SSO?")             # grounded RAG Q&A
extractor(Triage).extract("export button 500s")          # typed extraction
tool_agent(tools=[lookup], writes=[refund]).run(task)    # approval-gated tools
evaluation(dataset, gates={"groundedness": ">= 0.8"}).run()   # offline eval + CI gate
chat().send("What's my refund window?")                  # a multi-turn Assistant
Flow(provider=p, model=m).retrieve("./docs").ground().run(question)  # one packet, fluent
```

## Mental model

- One object, `ContextApp`, owns the pipeline. Configure it, then `run`/`arun`/
  `stream`/`astream`/`submit`/`batch`. The surface is also grouped into six
  lazily-constructed capability facades — `app.runs`, `app.knowledge`,
  `app.governance`, `app.optimization`, `app.serving`, `app.training` — each a
  narrow view delegating to the same implementation.
- The run pipeline is one path: normalize → classify → policy → memory recall →
  retrieve → compile context (score / dedupe / conflict / compress / budget) →
  compile prompt (cache-aware) → model (+ bounded tool loop) → validate (schema /
  citations / policy, principled repair) → evaluate → trace → memory write.
- Deterministic where it matters: security, permissions, validation, and budgets
  are enforced in code, never gated on model output.
- Offline development uses the bundled `MockProvider` (pass it explicitly, or set
  a provider+key for a real run). It emits schema-valid output so the whole
  pipeline — validation, evals, traces, audit, cost — runs with no network.
- Every run yields one `RunResult` (typed output, citations, trace_id, cost_usd,
  usage, eval_scores, excluded_context), one trace, one cost entry, and one
  hash-chained audit entry.
- Errors all derive from `VincioError` and carry a stable `.code`, a
  `.remediation`, and a `.docs_url`. Catch the family with one `except`.
- Optional heavy features ride extras (`vincio[...]`); the dependency-light,
  offline-first path is always the default.

## Examples (three tiers, all runnable offline)

- `examples/notebooks/*.ipynb` — Google Colab-ready notebooks (one `pip install`,
  offline by default): quickstart, RAG, agents & tools, evaluation, data analysis.
- `examples/00`–`22` — complete, heavily-commented feature tours, one per subsystem.
- `examples/applications/` — real-world small backends: a FastAPI grounded-RAG
  service, a ticket-triage API, a structured-extraction service, and a CLI
  research agent. Each splits an offline-testable `core.py` from a FastAPI
  `main.py` and runs on the mock or a real model with one env var.
"""

_LLMS_GOTCHAS = """\
## Gotchas for generated code

- The DEFAULT provider is OpenAI. A bare `ContextApp(name=...)` needs a provider
  and key; to run with NO key, pass `provider=MockProvider()` explicitly (from
  `vincio.providers`). The mock auto-generates schema-valid output offline.
- `result.output` is a validated Pydantic instance only when `output_schema=` is
  set; otherwise read `result.raw_text`. Check `result.status` / `result.error`.
- Grounding is a policy: `app.set_policy("answer_only_from_sources", True)` plus
  an evaluator like `groundedness`. The `rag(...)` front door wires both.
- Async methods are the `a`-prefixed variants (`arun`, `astream`, `abatch`, …);
  the sync names wrap them and work with or without a running event loop.
- Write/side-effecting tools are denied by default and surfaced for approval;
  pass an `approval_required=` tool plus an approval callback / allow-list.
- The data plane uses app METHODS, not top-level functions: `app.register_dataset`,
  `app.query_data`, `app.analyze_data`, `app.generate_chart`, `app.data_catalog`.
- Heavy backends are extras: install `vincio[openai]`, `vincio[retrieval]`,
  `vincio[server]`, `vincio[charts]`, `vincio[docs]`, etc. as needed.
- The frozen public surface is exactly `vincio.__all__`; import from the top
  level. Subpackage paths are internal and may move.
"""


def render_llms_txt() -> str:
    """Regenerate ``llms.txt`` from ``vincio.__all__`` and the doc graph.

    Composes a curated preamble + mental model, the complete public-symbol index
    (every name in ``vincio.__all__`` with its signature and one-line summary),
    the ``app.*`` capability map grouped by facet, and the error catalog — all
    derived from the live package, so a new public symbol or error code appears
    here automatically and the freshness gate catches any drift.
    """
    from .core.error_catalog import ERROR_CATALOG

    parts: list[str] = [_LLMS_PREAMBLE.rstrip(), ""]

    # -- the app.* capability map, compact ----------------------------------
    index = _verb_topic_index()
    verbs = app_verbs()
    parts.append("## Capability map (app.* by facet)")
    parts.append("")
    parts.append(
        "Every public `app.*` verb, grouped by capability facade. See "
        "docs/reference/capability-map.md for the concept/guide/example each links to."
    )
    parts.append("")
    for facet_key, facet_title, blurb in FACETS:
        facet_verbs = sorted(v for v in verbs if (index.get(v) and index[v].facet == facet_key))
        parts.append(f"### {facet_title}")
        parts.append(f"_{blurb}_")
        parts.append("")
        for verb in facet_verbs:
            parts.append(f"- `app.{verb}` — {_summary_for(verb)}")
        parts.append("")

    # -- the complete public-symbol index, from vincio.__all__ --------------
    by_kind: dict[str, list[tuple[str, object]]] = {"class": [], "function": [], "data": []}
    for name, obj in public_symbols():
        by_kind[symbol_kind(obj)].append((name, obj))
    total = sum(len(v) for v in by_kind.values())
    parts.append(f"## Public API ({total} public symbols)")
    parts.append("")
    parts.append(
        "Every introspectable name in `vincio.__all__` — the exact set Semantic "
        "Versioning applies to (the `__version__` value aside). Import every name "
        "from the top-level `vincio` package."
    )
    parts.append("")
    for kind, heading in (("class", "Classes"), ("function", "Functions"), ("data", "Values")):
        entries = by_kind[kind]
        if not entries:
            continue
        parts.append(f"### {heading}")
        parts.append("")
        for name, obj in entries:
            if kind == "data":
                parts.append(f"- `{name}` — {docstring_summary(obj) or '—'}")
            else:
                parts.append(f"- `{symbol_signature(name, obj)}` — {docstring_summary(obj) or '—'}")
        parts.append("")

    # -- the error catalog --------------------------------------------------
    parts.append(f"## Error catalog ({len(ERROR_CATALOG)} codes)")
    parts.append("")
    parts.append(
        "Every error derives from `VincioError` with a stable `.code`, a "
        "`.remediation`, and a `.docs_url`. Branch on `.code`."
    )
    parts.append("")
    for entry in ERROR_CATALOG.values():
        parts.append(f"- `{entry.code}` — {entry.title}. {entry.remediation}")
    parts.append("")

    parts.append(_LLMS_GOTCHAS.rstrip())
    return "\n".join(parts).rstrip() + "\n"


# --- graph checks -----------------------------------------------------------


@dataclass
class GraphCheck:
    """The result of one docs-graph check: a name, a verdict, and the offenders."""

    name: str
    ok: bool
    problems: list[str] = field(default_factory=list)


def link_integrity_report() -> GraphCheck:
    """Every internal markdown link under docs/ resolves (path + anchor)."""
    problems: list[str] = []
    for page in _all_doc_pages():
        if page in _LINK_SCAN_EXCLUDE:
            continue
        for link in iter_markdown_links(page):
            err = _resolve_link(link)
            if err:
                problems.append(err)
    return GraphCheck("link_integrity", not problems, sorted(problems))


def capability_map_coverage() -> GraphCheck:
    """Every public ``app.*`` verb is placed, appears in the capability map, and
    appears in ``api.md``; and every concept reaches a guide, an example, and a
    reference anchor."""
    problems: list[str] = []
    problems.extend(f"verb not in capability map: app.{v}" for v in uncovered_verbs())

    cap_map = render_capability_map()
    for verb in app_verbs():
        if f"`app.{verb}`" not in cap_map:
            problems.append(f"verb missing from rendered capability map: app.{verb}")

    try:
        api = _read(_API_REF)
    except OSError:
        api = ""
    for verb in app_verbs():
        if f"app.{verb}" not in api:
            problems.append(f"verb not documented in api.md: app.{verb}")

    declared: set[str] = set()
    for topic in TOPICS:
        declared.update(topic.verbs)
    for verb in declared:
        if verb not in set(app_verbs()):
            problems.append(f"doc graph references a non-existent verb: app.{verb}")

    # Every concept page reaches a guide + an example + a reference anchor.
    documented_concepts = {t.concept for t in TOPICS if t.concept}
    for concept in concept_pages():
        topics = [t for t in TOPICS if t.concept == concept]
        if not topics:
            problems.append(f"concept page not in the doc graph: {concept}")
            continue
        if not any(t.guides for t in topics):
            problems.append(f"concept reaches no guide: {concept}")
        if not any(t.examples for t in topics):
            problems.append(f"concept reaches no example: {concept}")
    for concept in documented_concepts:
        if concept and not os.path.exists(os.path.join(_ROOT, concept)):
            problems.append(f"doc graph references a missing concept page: {concept}")

    return GraphCheck("capability_map_coverage", not problems, sorted(problems))


def navigation_reachability() -> GraphCheck:
    """Every concept and guide carries a current single-sourced Related block,
    and the generated pages (capability map, learning path, llms.txt, api index)
    are present and current."""
    problems: list[str] = []
    for page in concept_pages() + guide_pages():
        try:
            text = _read(page)
        except OSError:
            problems.append(f"page unreadable: {page}")
            continue
        if _RELATED_BEGIN not in text or _RELATED_END not in text:
            problems.append(f"page missing its Related block: {page}")
            continue
        current = render_related_block(page)
        block = _extract_block(text, _RELATED_BEGIN, _RELATED_END)
        if block != current:
            problems.append(f"page Related block is stale: {page}")

    # Generated pages exist and are current.
    for repo_rel, renderer in (
        (_CAPABILITY_MAP, render_capability_map),
        (_LEARNING_PATH, render_learning_path),
    ):
        try:
            if _read(repo_rel) != renderer():
                problems.append(f"generated page is stale: {repo_rel}")
        except OSError:
            problems.append(f"generated page missing: {repo_rel}")

    try:
        api = _read(_API_REF)
        if _APPINDEX_BEGIN not in api or _extract_block(
            api, _APPINDEX_BEGIN, _APPINDEX_END
        ) != render_app_method_index():
            problems.append("api.md app-method index is missing or stale")
    except OSError:
        problems.append("api.md missing")

    return GraphCheck("navigation_reachability", not problems, sorted(problems))


def orphan_pages() -> GraphCheck:
    """No docs page is unreachable: every page (except the index hub) is linked
    to by at least one other docs page."""
    pages = set(_all_doc_pages())
    linked: set[str] = set()
    for page in pages:
        for link in iter_markdown_links(page):
            path_part, _, _ = link.target.partition("#")
            if not path_part:
                continue
            resolved = posixpath.normpath(
                posixpath.join(posixpath.dirname(page), path_part)
            )
            linked.add(resolved)
    hubs = {_INDEX}
    orphans = sorted(p for p in pages if p not in linked and p not in hubs)
    return GraphCheck("no_orphans", not orphans, [f"orphan page: {p}" for p in orphans])


def llms_txt_current() -> GraphCheck:
    """``llms.txt`` equals the freshly-rendered output (the freshness gate)."""
    try:
        on_disk = open(_LLMS, encoding="utf-8").read()
    except OSError:
        return GraphCheck("llms_txt_current", False, ["llms.txt missing"])
    ok = on_disk == render_llms_txt()
    return GraphCheck("llms_txt_current", ok, [] if ok else ["llms.txt is stale"])


def docs_graph_report() -> list[GraphCheck]:
    """Run every docs-graph check and return the verdicts."""
    return [
        link_integrity_report(),
        capability_map_coverage(),
        navigation_reachability(),
        orphan_pages(),
        llms_txt_current(),
    ]


# --- sync (the generator) ---------------------------------------------------


def _extract_block(text: str, begin: str, end: str) -> str:
    start = text.find(begin)
    stop = text.find(end)
    if start == -1 or stop == -1:
        return ""
    return text[start : stop + len(end)]


def _inject_block(text: str, block: str, begin: str, end: str) -> str:
    start = text.find(begin)
    stop = text.find(end)
    if start != -1 and stop != -1:
        return text[:start] + block + text[stop + len(end) :]
    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return text + sep + block + "\n"


def sync_docs(*, write: bool = True) -> list[str]:
    """Render every generated artifact and (when *write*) write it.

    Returns the repo-relative paths that changed (or would change). This is the
    one command that keeps the docs graph in sync: the capability map, the
    learning path, the api.md app-method index, every Related block, and
    llms.txt. ``vincio docs map`` and the docs-graph gate both call it.
    """
    changed: list[str] = []

    def apply(repo_rel: str, content: str) -> None:
        path = os.path.join(_ROOT, repo_rel)
        try:
            existing = open(path, encoding="utf-8").read()
        except OSError:
            existing = None
        if existing == content:
            return
        changed.append(repo_rel)
        if write:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)

    # Whole-file generated pages.
    apply(_CAPABILITY_MAP, render_capability_map())
    apply(_LEARNING_PATH, render_learning_path())
    apply(_LLMS, render_llms_txt())

    # Managed block inside api.md.
    api = _read(_API_REF)
    new_api = _inject_block(api, render_app_method_index(), _APPINDEX_BEGIN, _APPINDEX_END)
    apply(_API_REF, new_api)

    # Related blocks on every concept and guide.
    for page in concept_pages() + guide_pages():
        text = _read(page)
        new_text = _inject_block(text, render_related_block(page), _RELATED_BEGIN, _RELATED_END)
        apply(page, new_text)

    return changed


def _main(argv: list[str]) -> int:  # pragma: no cover - dev tool
    check = "--check" in argv
    changed = sync_docs(write=not check)
    if check:
        report = docs_graph_report()
        failed = [c for c in report if not c.ok]
        for c in report:
            print(f"[{'PASS' if c.ok else 'FAIL'}] {c.name}")
            for problem in c.problems[:50]:
                print(f"    - {problem}")
        if changed:
            print(f"\n{len(changed)} generated artifact(s) would change — run "
                  "`python -m vincio._docmap` to regenerate:")
            for path in changed:
                print(f"    - {path}")
        return 1 if (failed or changed) else 0
    print(f"synced {len(changed)} doc artifact(s)" if changed else "docs already in sync")
    for path in changed:
        print(f"    - {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - dev tool
    import sys

    raise SystemExit(_main(sys.argv[1:]))
