"""Frontier context engineering.

The hardest, longest agent runs are won at the context layer, not the model.
This tour focuses on four frontier controls Vincio layers on any provider, shown
deeply and fully offline:

  1. Unified reasoning control: one portable effort knob under a hard token ceiling.
  2. Test-time compute: verifier-guided best-of-N / self-consistency that early-
     exit the instant the answer is locked.
  3. A long-horizon context governor: resident footprint stays flat as the horizon
     grows 10x, via intra-run decay + provenance-preserving compaction.
  4. The learned semantic cache with near-miss KV reuse.
Everything is opt-in and additive. (Closing note points to the rest.)
"""

from __future__ import annotations

import asyncio

from vincio import (
    ContextApp,
    ContextBudget,
    ContextCompactor,
    ContextGovernor,
    LearnedSemanticCache,
    ReasoningController,
    ReasoningPolicy,
    RelevanceDecay,
    SemanticCachePolicy,
    TaskType,
    TestTimeSearch,
    VincioConfig,
)
from vincio.context.evidence_store import content_hash
from vincio.core.types import EvidenceItem, RunConfig
from vincio.evals.datasets import EvalCase
from vincio.evals.ensemble import JudgeEnsemble
from vincio.evals.judges import DeterministicJudge
from vincio.evals.metrics import MetricResult, RunOutput
from vincio.optimize import JudgeVerifier, SearchBudget
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import LocalHashEmbedder


def _memory_config() -> VincioConfig:
    config = VincioConfig()
    config.observability.exporter = "memory"  # fully in-process, no exporter/network
    return config


def section_reasoning_control() -> None:
    # One `reasoning_effort` maps to OpenAI effort, Anthropic extended thinking and
    # Gemini thinking budgets. Thinking tokens are recorded on the span AND billed.
    app = ContextApp(name="reasoning", provider=MockProvider(default_text="42", reasoning=True),
                     model="mock-1")
    tokens = {e: app.run("How many r's in strawberry?", config=RunConfig(reasoning_effort=e)
                         ).usage.reasoning_tokens for e in ("low", "high")}

    # The controller chooses effort FROM the task, but a hard token ceiling means a
    # hard task can never silently exhaust the budget (ceiling_capped says so).
    ctl = ReasoningController(ReasoningPolicy(max_reasoning_tokens=8192))
    easy = ctl.decide(task=TaskType.CLASSIFICATION, text="Is this email spam? yes/no")
    hard = ctl.decide(task=TaskType.DOCUMENT_COMPARISON,
                      text="Compare why these two merger agreements differ " * 12,
                      remaining_output_tokens=4096)
    capped = ReasoningController(ReasoningPolicy(max_reasoning_tokens=1500)).decide(difficulty=0.95)
    print(f"1. reasoning control: effort->tokens {tokens} | task-chosen easy={easy.effort} "
          f"hard={hard.effort} | ceiling clamps hard task to {capped.thinking_budget_tokens} "
          f"(capped={capped.ceiling_capped})")


_TARGET = "the refund window for the pro plan is 30 days"
_CANDIDATES = ["Refunds take roughly a month I think", "Pro plan refunds within some window",
               "The refund window for the Pro plan is 30 days.", "Refund policy unclear"]


def _overlap(case: EvalCase, out: RunOutput) -> MetricResult:
    want = set(_TARGET.split())
    got = set(out.raw_text.lower().replace(".", "").split())
    return MetricResult(name="overlap", value=len(want & got) / len(want))


async def section_test_time_compute() -> None:
    # Best-of-N: draw candidates, score each with the platform's judge ensemble,
    # and EARLY-EXIT the instant the verifier clears the confidence bar — you spend
    # generations only until the answer is good enough, not a fixed N.
    verifier = JudgeVerifier(JudgeEnsemble([DeterministicJudge(_overlap, name="overlap")]),
                             case=EvalCase(id="q", input="What is the refund window?"))
    bon = await TestTimeSearch(lambda i: _CANDIDATES[i], verifier=verifier,
                               budget=SearchBudget(max_candidates=4, confidence_target=0.95)).best_of_n()

    # Self-consistency: a majority vote that locks in once the lead is unbeatable.
    votes = ["30 days", "30 days", "14 days", "30 days", "30 days"]
    sc = await TestTimeSearch(lambda i: votes[i], budget=SearchBudget(max_candidates=5)).self_consistency()
    print(f"2. test-time compute: best-of-N drew {bon.n_generated}/4 best={bon.best.score:.2f} "
          f"({bon.best.answer_text!r}) early_exit={bon.early_exit} | "
          f"self-consistency {sc.best.answer_text!r} @ {sc.confidence:.0%} (drew {sc.n_generated}/5)")


_NEEDLE = "The Pro plan refund window is exactly 30 days from the purchase date."


def _filler(i: int) -> str:
    return f"Filler observation {i}: telemetry, logs, metrics, traces, spans, counters."


def _governed(horizon: int) -> ContextGovernor:
    # A per-run context budget: tokens, resident bytes, intra-run decay, and a
    # floor of recent spans always kept. Cold spans fold into summaries.
    gov = ContextGovernor(ContextBudget(max_tokens=400, max_resident_bytes=6000),
                          compactor=ContextCompactor(summary_tokens=48),
                          decay=RelevanceDecay(half_life_steps=8), keep_recent_spans=3)
    gov.admit(_NEEDLE, relevance=0.95, source_ids=["needle"])
    for i in range(horizon):
        gov.admit(_filler(i), relevance=0.5)
    return gov


def section_long_horizon() -> None:
    # Intra-run decay: a span admitted many steps ago loses weight before it can
    # crowd out fresh signal.
    decay = RelevanceDecay(half_life_steps=8)

    # Provenance-preserving compaction: cold spans fold into a summary while their
    # full text is paged to a content-addressed store and paged back on demand — so
    # compaction never loses a citation, it just defers the bytes.
    compactor = ContextCompactor(summary_tokens=40)
    cold = [EvidenceItem(text=_NEEDLE, source_id="needle"),
            *[EvidenceItem(text=_filler(i), source_id=f"f{i}") for i in range(5)]]
    spans = ContextGovernor(ContextBudget(), compactor=compactor).admit_evidence(cold)
    _, record = compactor.compact(spans)
    recovered = compactor.page_in([content_hash(_NEEDLE)])[content_hash(_NEEDLE)]

    # The headline: a 10x longer horizon barely moves the resident footprint (naive
    # accumulation would blow up), and the needle is still recallable.
    small, large = _governed(20), _governed(200)
    rep = large.report()
    hits = large.recall("Pro plan refund window days purchase", top_k=3)
    print(f"3. long-horizon: decay fresh {decay.weight(0):.2f} -> age-40 {decay.weight(40):.2f} | "
          f"compaction {record.tokens_before}->{record.tokens_after} tokens, paged back {recovered[:24]!r}")
    print(f"   residency 1x={small.resident_bytes}B -> 10x={large.resident_bytes}B "
          f"({large.resident_bytes / small.resident_bytes:.2f}x), within_budget={rep.within_budget}, "
          f"needle still found={any('30 days' in h for h in hits)}")


_REFUND_A = "what is the refund policy for orders"
_REFUND_B = "what is the refund policy for returns"
_UNRELATED = "how do I reset my account password"


async def section_semantic_cache() -> None:
    # The cache LEARNS its acceptance threshold from labelled trace pairs, so a
    # near-miss is served only when it clears a precision target — no hand-tuned
    # similarity cutoff that silently serves wrong answers.
    cache = LearnedSemanticCache(LocalHashEmbedder(),
                                 policy=SemanticCachePolicy(target_precision=0.95, min_floor=0.5, ttl_s=None))
    cal = await cache.calibrate_from_pairs([(_REFUND_A, _REFUND_B, True), (_REFUND_A, _UNRELATED, False)])
    await cache.store(_REFUND_A, {"text": "Refunds within 30 days."}, policy_scope="refunds", response_tokens=8)
    hit = await cache.lookup(_REFUND_B, policy_scope="refunds")  # semantically equivalent -> served free
    miss = await cache.lookup(_UNRELATED, policy_scope="refunds")  # below the bar -> refused
    print(f"4. semantic cache: calibrated threshold={cal.threshold:.3f} | near-miss served "
          f"{hit.value['text']!r} (sim {hit.similarity:.2f} >= {hit.threshold:.2f}), unrelated served={miss is not None}")

    # Through the run path: a near-miss is served free, and a distinct question still
    # hits the provider but reuses the shared prompt head's KV footprint.
    app = ContextApp(name="support", provider=MockProvider(default_text="Refunds within 30 days."),
                     config=_memory_config())
    app.use_semantic_cache(SemanticCachePolicy(enabled=True, threshold=0.6, ttl_s=None))
    app.use_kv_prefix_reuse()
    app.run(_REFUND_A)                       # cold: real call computes the head's KV
    served = app.run(_REFUND_B)              # near-miss: served from cache, billed $0
    app.run("what are the international shipping options for large parcels")
    kv = app.kv_prefix_report()
    print(f"   run-path near-miss billed ${served.cost_usd:.4f} | kv reuses={kv.reuses} "
          f"bytes_reused={kv.kv_bytes_reused}")


async def main() -> None:
    section_reasoning_control()
    await section_test_time_compute()
    section_long_horizon()
    await section_semantic_cache()
    # The same frontier-context stack also includes:
    #   * world-model / simulation planning — plan against a learned model, not the
    #     live world (agents.WorldModel + ModelPredictivePlanner)
    #   * the causal record-replay debugger — replay a run byte-for-byte, detect drift
    #     (observability.Recorder / Replayer)
    #   * per-run energy / carbon accounting with a carbon budget — app.use_energy_accounting
    #   * the compile hot path — cache + warm candidate arena + streaming compile
    #   * the edge / WASM in-process runtime — same compile, no provider, byte-identical
    #     to the server (vincio.edge.EdgeRuntime, verify_edge_parity)
    print("\nFrontier context engineering: portable, verifier-guided, bounded, cached.")


if __name__ == "__main__":
    asyncio.run(main())
