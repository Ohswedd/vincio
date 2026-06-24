"""Frontier context engineering.

The hardest, longest, most expensive agent runs are won at the context layer, not
the model. This single program walks the frontier controls Vincio layers on top of
any provider — all fully offline on the deterministic mock:

  * unified reasoning control: one portable effort knob under a hard token ceiling;
  * test-time compute: verifier-guided best-of-N, self-consistency, and beam search
    that early-exit the instant the answer is locked;
  * a long-horizon context governor that keeps the resident footprint flat as the
    horizon grows 10x, with intra-run decay and provenance-preserving compaction;
  * world-model / simulation-based planning (plan against a learned model, not the
    live world);
  * the learned semantic cache with near-miss KV reuse;
  * the causal record-replay debugger; and
  * per-run energy / carbon accounting plus the honest compile hot-path numbers.

Everything below is opt-in and additive; none of it is required to run Vincio.
"""

from __future__ import annotations

import asyncio
import time

from _shared import example_provider

from vincio import (
    ContextApp,
    ContextBudget,
    ContextCompactor,
    ContextGovernor,
    LearnedSemanticCache,
    Objective,
    ReasoningController,
    ReasoningPolicy,
    RelevanceDecay,
    SemanticCachePolicy,
    TaskType,
    TestTimeSearch,
    UserInput,
    VincioConfig,
)
from vincio.agents import ModelPredictivePlanner, WorldModel, record_transitions
from vincio.context.evidence_store import content_hash
from vincio.core.types import EvidenceItem, RunConfig
from vincio.edge import EdgeProfile, EdgeRequest, EdgeRuntime, verify_edge_parity
from vincio.evals.datasets import EvalCase
from vincio.evals.ensemble import JudgeEnsemble
from vincio.evals.environment import (
    EnvAction,
    make_retail_environment,
    make_vault_environment,
)
from vincio.evals.judges import DeterministicJudge
from vincio.evals.metrics import MetricResult, RunOutput
from vincio.observability import Recorder, Replayer
from vincio.optimize import JudgeVerifier, SearchBudget
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import LocalHashEmbedder


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def _memory_config() -> VincioConfig:
    # Keep observability fully in-process so the example needs no exporter/network.
    config = VincioConfig()
    config.observability.exporter = "memory"
    return config


# ---------------------------------------------------------------------------
# 1. Unified reasoning control: one portable knob under a hard ceiling.
# ---------------------------------------------------------------------------
def section_reasoning_control() -> None:
    banner("1. Unified reasoning control (effort + hard token ceiling)")

    # The same `reasoning_effort` maps to OpenAI effort, Anthropic extended
    # thinking, and Gemini thinking budgets. A reasoning-capable mock emulates it
    # offline; thinking tokens are recorded on the span AND billed.
    app = ContextApp(
        name="reasoning",
        provider=MockProvider(default_text="42", reasoning=True),
        model="mock-1",
    )
    for effort in ("low", "high"):
        result = app.run("How many r's in strawberry?", config=RunConfig(reasoning_effort=effort))
        print(f"   effort={effort:<4} reasoning_tokens={result.usage.reasoning_tokens:<4} "
              f"cost=${result.cost_usd:.6f}")

    # The controller chooses effort from the task — but a hard token ceiling means
    # a hard task can never silently exhaust the budget.
    ctl = ReasoningController(ReasoningPolicy(max_reasoning_tokens=8192))
    easy = ctl.decide(task=TaskType.CLASSIFICATION, text="Is this email spam? yes/no")
    hard = ctl.decide(
        task=TaskType.DOCUMENT_COMPARISON,
        text="Compare why these two merger agreements differ " * 12,
        remaining_output_tokens=4096,
    )
    print(f"   easy task → effort={easy.effort:<8} budget={easy.thinking_budget_tokens} tokens")
    print(f"   hard task → effort={hard.effort:<8} budget={hard.thinking_budget_tokens} tokens")
    capped = ReasoningController(ReasoningPolicy(max_reasoning_tokens=1500)).decide(difficulty=0.95)
    print(f"   ceiling clamps a hard task to {capped.thinking_budget_tokens} tokens "
          f"(ceiling_capped={capped.ceiling_capped})")


# ---------------------------------------------------------------------------
# 2. Test-time compute: verifier-guided search with early exit.
# ---------------------------------------------------------------------------
# A reference answer the judge panel rewards proximity to.
_TARGET = "the refund window for the pro plan is 30 days"
_CANDIDATES = [
    "Refunds take roughly a month I think",
    "Pro plan refunds within some window",
    "The refund window for the Pro plan is 30 days.",  # the strong one
    "Refund policy unclear",
]


def _overlap(case: EvalCase, out: RunOutput) -> MetricResult:
    # Word-overlap with the target — a deterministic stand-in for a real judge.
    want = set(_TARGET.split())
    got = set(out.raw_text.lower().replace(".", "").split())
    return MetricResult(name="overlap", value=len(want & got) / len(want))


async def section_test_time_compute() -> None:
    banner("2. Test-time compute (best-of-N, self-consistency, beam — with early exit)")

    # Best-of-N: draw candidates, score each with the platform's judge ensemble,
    # and early-exit the instant the verifier clears the confidence bar.
    verifier = JudgeVerifier(
        JudgeEnsemble([DeterministicJudge(_overlap, name="overlap")]),
        case=EvalCase(id="q", input="What is the refund window?"),
    )
    bon = TestTimeSearch(
        lambda i: _CANDIDATES[i],
        verifier=verifier,
        budget=SearchBudget(max_candidates=4, confidence_target=0.95),
    )
    r = await bon.best_of_n()
    print(f"   best-of-N: drew {r.n_generated}/4  best={r.best.score:.2f}  "
          f"winner={r.best.answer_text!r}")
    print(f"             early_exit={r.early_exit} — {r.stop_reason}")

    # Self-consistency: a majority vote that locks in once the lead is unbeatable.
    votes = ["30 days", "30 days", "14 days", "30 days", "30 days"]
    sc = TestTimeSearch(lambda i: votes[i], budget=SearchBudget(max_candidates=5))
    r = await sc.self_consistency()
    print(f"   self-consistency: {r.best.answer_text!r}  "
          f"(vote share {r.confidence:.0%}, drew {r.n_generated}/5, early_exit={r.early_exit})")


# ---------------------------------------------------------------------------
# 3. Long-horizon context governor: bounded residency, decay, compaction.
# ---------------------------------------------------------------------------
_NEEDLE = "The Pro plan refund window is exactly 30 days from the purchase date."


def _filler(i: int) -> str:
    return f"Filler observation {i}: telemetry, logs, metrics, traces, spans, counters."


def _governed(horizon: int) -> ContextGovernor:
    # A per-run context budget: tokens, resident bytes, intra-run decay, and a
    # floor of recent spans always kept. Folds cold spans into summaries.
    gov = ContextGovernor(
        ContextBudget(max_tokens=400, max_resident_bytes=6000),
        compactor=ContextCompactor(summary_tokens=48),
        decay=RelevanceDecay(half_life_steps=8),
        keep_recent_spans=3,
    )
    gov.admit(_NEEDLE, relevance=0.95, source_ids=["needle"])
    for i in range(horizon):
        gov.admit(_filler(i), relevance=0.5)
    return gov


def section_long_horizon() -> None:
    banner("3. Long-horizon context governor (residency stays flat, needle survives)")

    # Intra-run decay: a span admitted many steps ago loses weight before it can
    # crowd out fresh signal.
    decay = RelevanceDecay(half_life_steps=8)
    print(f"   decay: fresh weight {decay.weight(0):.3f}, "
          f"one half-life {decay.weight(8):.3f}, age-40 {decay.weight(40):.3f}")

    # Provenance-preserving compaction: cold spans fold into a summary, full text
    # is paged to a content-addressed store, and paged back on demand.
    compactor = ContextCompactor(summary_tokens=40)
    cold = [EvidenceItem(text=_NEEDLE, source_id="needle"),
            *[EvidenceItem(text=_filler(i), source_id=f"f{i}") for i in range(5)]]
    spans = ContextGovernor(ContextBudget(), compactor=compactor).admit_evidence(cold)
    summary, record = compactor.compact(spans)
    print(f"   compaction: {len(spans)} cold spans → 1 summary "
          f"({record.tokens_before}→{record.tokens_after} tokens, "
          f"{len(record.covered_hashes)} hashes kept)")
    recovered = compactor.page_in([content_hash(_NEEDLE)])
    print(f"   paged back on demand: {recovered[content_hash(_NEEDLE)][:40]!r}…")

    # The headline: a 10x longer horizon barely moves the resident footprint, while
    # naive accumulation would blow up — and the needle is still recallable.
    small, large = _governed(20), _governed(200)
    print(f"   resident: {small.resident_bytes}B (1x) → {large.resident_bytes}B (10x) "
          f"= {large.resident_bytes / small.resident_bytes:.2f}x")
    rep = large.report()
    print(f"   at 10x: {rep.compaction_count} compactions saved {rep.compacted_tokens_saved} "
          f"tokens, within_budget={rep.within_budget}")
    hits = large.recall("Pro plan refund window days purchase", top_k=3)
    print(f"   recall@10x still finds the needle: {any('30 days' in h for h in hits)}")


# ---------------------------------------------------------------------------
# 4. World-model / simulation-based planning.
# ---------------------------------------------------------------------------
def _act(tool: str, **kwargs: object) -> EnvAction:
    return EnvAction(kind="tool", tool=tool, arguments=kwargs)


def section_world_model() -> None:
    banner("4. World-model planning (plan against a learned model, not the live world)")

    # Learn each tool's effect + precondition from recorded experience — including
    # a refund that fails on a processing order and one that succeeds after a cancel.
    explore = [
        [_act("refund_order", order_id="O1002")],
        [_act("cancel_order", order_id="O1002"), _act("refund_order", order_id="O1002")],
        [_act("cancel_order", order_id="O1002")],
        [_act("refund_order", order_id="O1001")],
        [_act("update_address", order_id="O1002", address="9 New Rd")],
    ]
    transitions = record_transitions(make_retail_environment("cancel_refund"), explore)
    model = WorldModel(transitions)
    base = make_retail_environment("cancel_refund").observe()
    fail = model.predict(base, _act("refund_order", order_id="O1002"))
    after = model.predict(base, _act("cancel_order", order_id="O1002")).observation
    ok = model.predict(after, _act("refund_order", order_id="O1002"))
    print(f"   learned precondition: refund on processing order ok={fail.ok}, "
          f"after cancel ok={ok.ok}")

    # The model earns planning weight only after calibrating against the real env.
    report = model.calibrate(transitions)
    print(f"   calibration: state_acc={report.state_accuracy:.2f}, "
          f"trusted={report.trusted} (weight {report.weight:.2f})")

    # Receding-horizon planner: search imagined rollouts, commit the best first move.
    planner = ModelPredictivePlanner(model, horizon=3, beam_width=16)
    result = planner.plan(make_retail_environment("cancel_refund"))
    print(f"   planned (real steps={result.real_steps}): "
          f"{' → '.join(a.tool for a in result.committed)}  success={result.success}")

    # Planning beats reacting: on a world with a tempting shortcut that dead-ends,
    # the imagined-rollout planner reaches the vault; a one-step reactor is trapped.
    vault_explore = [
        [_act("advance"), _act("advance"), _act("advance"), _act("open_vault")],
        [_act("open_vault")],
        [_act("advance"), _act("open_vault")],
        [_act("advance"), _act("advance"), _act("open_vault")],
        [_act("shortcut")],
        [_act("shortcut"), _act("open_vault")],
        [_act("advance")],
        [_act("advance"), _act("advance")],
        [_act("advance"), _act("advance"), _act("advance")],
        [_act("shortcut"), _act("advance")],
    ]
    vt = record_transitions(make_vault_environment(), vault_explore)
    vmodel = WorldModel(vt).fit(vt)
    vmodel.calibrate(vt)
    reactive = ModelPredictivePlanner(vmodel, horizon=1, beam_width=64, max_real_steps=6).plan(
        make_vault_environment())
    planned = ModelPredictivePlanner(vmodel, horizon=5, beam_width=64, max_real_steps=6).plan(
        make_vault_environment())
    print(f"   reactive (1-step): success={reactive.success} (took the shortcut, trapped)")
    print(f"   imagined-rollout : success={planned.success} "
          f"({' → '.join(a.tool for a in planned.committed)})")


# ---------------------------------------------------------------------------
# 5. Learned semantic cache + near-miss KV reuse.
# ---------------------------------------------------------------------------
_SCOPE = "support:refunds"
_REFUND_A = "what is the refund policy for orders"
_REFUND_B = "what is the refund policy for returns"
_UNRELATED = "how do I reset my account password"


async def section_semantic_cache() -> None:
    banner("5. Learned semantic cache + near-miss KV reuse")

    # The cache learns its own acceptance threshold from labelled trace pairs so a
    # near-miss is served only when it clears a precision target.
    cache = LearnedSemanticCache(
        LocalHashEmbedder(),
        policy=SemanticCachePolicy(target_precision=0.95, min_floor=0.5, ttl_s=None),
    )
    cal = await cache.calibrate_from_pairs(
        [(_REFUND_A, _REFUND_B, True), (_REFUND_A, _UNRELATED, False)]
    )
    print(f"   calibrated threshold={cal.threshold:.3f}  "
          f"precision={cal.achieved_precision:.2f}  calibrated={cal.calibrated}")

    # A semantically-equivalent request is served from cache for free.
    await cache.store(_REFUND_A, {"text": "Refunds within 30 days."},
                      policy_scope=_SCOPE, response_tokens=8)
    hit = await cache.lookup(_REFUND_B, policy_scope=_SCOPE)
    print(f"   near-miss {_REFUND_B!r} → served {hit.value['text']!r} "
          f"(similarity {hit.similarity:.3f} ≥ {hit.threshold:.3f})")
    # An unrelated request is refused below the bar.
    miss = await cache.lookup(_UNRELATED, policy_scope=_SCOPE)
    print(f"   below-bar {_UNRELATED!r} → served={miss is not None} "
          f"(rejected={cache.stats().near_misses_rejected})")

    # Through the run path: the near-miss is served free and a distinct question
    # still hits the provider but reuses the shared prompt head's KV footprint.
    app = ContextApp(name="support", provider=MockProvider(default_text="Refunds within 30 days."),
                     config=_memory_config())
    app.use_semantic_cache(SemanticCachePolicy(enabled=True, threshold=0.6, ttl_s=None))
    app.use_kv_prefix_reuse()
    app.run(_REFUND_A)                      # cold: real call computes the head's KV
    served = app.run(_REFUND_B)             # near-miss: served from cache, billed $0
    app.run("what are the international shipping options for large parcels")
    kv = app.kv_prefix_report()
    print(f"   run-path near-miss billed ${served.cost_usd:.4f}; "
          f"kv pool: families={kv.families} reuses={kv.reuses} bytes_reused={kv.kv_bytes_reused}")


# ---------------------------------------------------------------------------
# 6. Causal record-replay debugger.
# ---------------------------------------------------------------------------
def _replay_app(final_answer: str) -> ContextApp:
    # A model that calls a `lookup` tool once, then answers from it.
    script = [{"tool_call": {"name": "lookup", "arguments": {"q": "refund-policy"}}}, final_answer]
    app = ContextApp(config=_memory_config(), provider=MockProvider(script=list(script)))

    @app.tool_registry.register(name="lookup")
    def lookup(q: str) -> str:
        return "Refunds are accepted within 30 days of purchase."

    app.enabled_tools.append("lookup")
    return app


async def section_record_replay() -> None:
    banner("6. Causal record-replay debugger (replay byte-for-byte, detect drift)")

    # Record every non-deterministic edge of a run into a portable, verifiable
    # Recording.
    result, recording = await Recorder(_replay_app("Refunds within 30 days. [policy]")).record(
        "What is the refund policy?")
    print(f"   recorded {len(recording.edges)} edges "
          f"(model={len(recording.model_calls)}, tool={len(recording.tool_calls)})  "
          f"verified={recording.verify()}")

    # Faithful replay: the replay app's live provider would answer "WRONG", but the
    # recording — not the provider — serves every edge, so it reproduces exactly.
    faithful = await Replayer(_replay_app("WRONG (live)")).replay(recording)
    print(f"   faithful replay: faithful={faithful.faithful} "
          f"identical={faithful.output_identical} "
          f"served_from_recording={faithful.served_from_recording}")

    # Divergence: change the objective and the recorded edge no longer matches —
    # the debugger reports exactly where the run drifted.
    diverged_app = _replay_app("WRONG (live)")
    diverged_app.configure(objective="Summarize the cancellation policy instead")
    diverged = await Replayer(diverged_app).replay(recording)
    detail = diverged.divergences[0] if diverged.divergences else None
    print(f"   diverged replay: faithful={diverged.faithful} "
          f"divergences={len(diverged.divergences)}"
          + (f" — {detail.kind}" if detail else ""))


# ---------------------------------------------------------------------------
# 7. Energy / carbon accounting (the sustainability analogue of cost).
# ---------------------------------------------------------------------------
async def section_energy_carbon() -> None:
    banner("7. Per-run energy / carbon accounting")

    def app_for(region: str) -> ContextApp:
        app = ContextApp(name="energy", provider=MockProvider(), config=_memory_config())
        app.use_energy_accounting(region=region)
        return app

    # Every run accrues energy (Wh) and carbon (gCO2e) on the same surface as cost.
    eu = app_for("eu")
    eu_run = await eu.arun("Summarize our quarterly sustainability disclosure.")
    print(f"   EU run: ${eu_run.cost_usd:.6f}  {eu_run.energy_wh:.4f} Wh  "
          f"{eu_run.co2e_grams:.4f} gCO2e")

    # Same compute, cleaner grid → less carbon.
    fr_run = await app_for("fr").arun("Summarize our quarterly sustainability disclosure.")
    print(f"   FR run: {fr_run.energy_wh:.4f} Wh  {fr_run.co2e_grams:.4f} gCO2e "
          f"(same compute, cleaner grid)")

    # Budgeted like a dollar: a carbon cap refuses the over-budget runs, and every
    # estimate + refusal lands on the hash-chained audit log.
    budgeted = app_for("us")
    one = await budgeted.arun("probe the per-run carbon")
    budgeted.set_energy_budget(scope="global", limit_co2e_grams=one.co2e_grams * 1.5, period="total")
    statuses = [(await budgeted.arun(f"section {i}")).status.value for i in range(4)]
    print(f"   under a {one.co2e_grams * 1.5:.4f} gCO2e cap, statuses: {statuses}")
    refusals = [e for e in budgeted.audit.entries if e.action == "energy_budget"]
    print(f"   {len(refusals)} refusals on the chain; verifies={budgeted.audit.verify_chain()}")


# ---------------------------------------------------------------------------
# 8. The compile hot path — honest performance numbers.
# ---------------------------------------------------------------------------
_EVIDENCE = [
    EvidenceItem(id="e1", source_id="D1", text="Pro plan refunds are available within 30 days."),
    EvidenceItem(id="e2", source_id="D2", text="Basic plan refunds are available within 14 days."),
    EvidenceItem(id="e3", source_id="D3", text="The subscription renews automatically each year."),
]


async def section_compile_hot_path() -> None:
    banner("8. Compile hot path (cache + warm arena — honest numbers)")

    # A resident-memory ceiling and speculative prefetch are opt-in perf knobs.
    provider, model = example_provider(default_responder=lambda r: "Refunds within 30 days. [D1]")
    config = _memory_config()
    config.performance.memory_budget_mb = 8
    config.performance.speculative_prefetch = True
    app = ContextApp(name="perf", provider=provider, model=model, config=config)

    question = "What is the refund window for the Pro plan?"
    t0 = time.perf_counter()
    first = await app.arun(question)
    cold_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    await app.arun(question)  # identical inputs → full compile-cache hit
    warm_ms = (time.perf_counter() - t0) * 1000
    # These are real wall-clock numbers from this machine — reported, not promised.
    print(f"   app run: cold {cold_ms:.1f} ms, warm {warm_ms:.1f} ms")
    print(f"   compile-cache hits: {app.context_compiler.cache_hits}  "
          f"resident footprint: {first.memory_bytes} bytes (≤ 8 MB)")

    # The warm candidate arena reuses prepared candidates when only the query changes.
    for query in ("What is the refund window?", "How long to request a Pro refund?"):
        await app.context_compiler.compile(
            objective=Objective("refunds"), user_input=UserInput(text=query), evidence=_EVIDENCE)
    print(f"   candidate-arena reuses (new query, same evidence): "
          f"{app.context_compiler.arena_hits}")


# ---------------------------------------------------------------------------
# 9. Edge / WASM in-process runtime — the same context engineering at the edge.
# ---------------------------------------------------------------------------
def _edge_corpus(n: int) -> list[EvidenceItem]:
    return [EvidenceItem(source_id=f"clause{j}", relevance=0.85, authority=0.7,
                         text=f"Clause {j}: the refund window is {30 + j} days.")
            for j in range(n)]


def section_edge_runtime() -> None:
    banner("9. Edge / WASM in-process runtime (same compile, no provider, byte-identical)")

    # `EdgeRuntime` runs the dependency-free core (compile → score → rail → pack)
    # behind a thin in-process boundary, so the same pipeline runs in a browser
    # (Pyodide/WASM) with no provider, no network, and no caller-owned event loop.
    runtime = EdgeRuntime(EdgeProfile.browser())
    result = runtime.run(EdgeRequest(
        task="What is the refund window?", task_type=TaskType.DOCUMENT_QA,
        instructions=["Answer only from the evidence.", "Cite the source id."],
        evidence=_edge_corpus(4)))
    print(f"   run offline: profile={result.profile}  kept={len(result.packet.evidence_items)}  "
          f"tokens={result.token_count}  resident={result.resident_bytes}B  within={result.within_profile}")

    # The `EdgeProfile` is a hard resident-footprint cap: as the corpus grows 10×,
    # eviction holds the bound — same discipline as the server's memory budget.
    profile = EdgeProfile(name="capped", max_resident_bytes=4096, max_input_tokens=4096)
    capped = EdgeRuntime(profile)
    small = capped.run(EdgeRequest(task="refund window", evidence=_edge_corpus(4)))
    big = capped.run(EdgeRequest(task="refund window", evidence=_edge_corpus(40)))
    print(f"   bounded: 4 docs → kept {len(small.packet.evidence_items)} ({small.resident_bytes}B), "
          f"40 docs → kept {len(big.packet.evidence_items)} ({big.resident_bytes}B ≤ cap "
          f"{profile.max_resident_bytes}B — eviction held)")

    # Parity, not a fork: the edge compile is byte-identical to a server compile
    # over the same inputs — the spec hashes match, so there is one codepath.
    parity = verify_edge_parity()
    print(f"   parity: held={parity.held}  edge_hash={parity.edge_spec_hash[:12]}  "
          f"server_hash={parity.server_spec_hash[:12]} (identical → one codepath)")


async def main() -> None:
    section_reasoning_control()
    await section_test_time_compute()
    section_long_horizon()
    section_world_model()
    await section_semantic_cache()
    await section_record_replay()
    await section_energy_carbon()
    await section_compile_hot_path()
    section_edge_runtime()
    print("\nThe whole frontier-context stack, exercised offline on the mock provider.")


if __name__ == "__main__":
    asyncio.run(main())
