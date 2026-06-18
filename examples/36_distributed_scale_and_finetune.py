"""Scale out & train for real.

Horizontal scale and a real, still-self-hosted operational plane, all offline
on the deterministic mock:

  1. Distributed durable execution: a graph thread is lease-guarded and every
     super-step is CAS-committed, so two workers can't double-execute; a worker
     pool fans a batch out; a graph fans out in-step with Send map-reduce.
  2. Executed distillation: an executed StudentTrainer submits/polls a (mocked)
     fine-tune job, registers the trained model, and the swap gate decides
     promotion — a cheaper model only if it provably doesn't regress.
  3. Served observability & alerting: an indexed trace/cost store powers a
     served dashboard; an alert rule engine pages on an SRE burn rate.
  4. Quantized two-stage retrieval: coarse search on quantized/truncated
     vectors, exact rerank on full precision — recall of the exact index.
  5. Batteries-included local neural models: a GGUF in-process provider and a
     fastembed dense embedder, behind the same provider/embedder interfaces.

Everything here is opt-in and additive; the single-process asyncio path stays
the default and nothing below is required to run Vincio.
"""

from __future__ import annotations

import asyncio
import operator
from types import SimpleNamespace

from vincio.agents import (
    END,
    DistributedCheckpointer,
    InMemoryGraphCoordinator,
    Send,
    StateGraph,
    WorkerPoolBackend,
)
from vincio.core.errors import CheckpointConflictError
from vincio.core.types import Chunk
from vincio.observability import (
    AlertManager,
    AlertRule,
    IndexedTraceStore,
    MemoryAlertSink,
    Span,
    Trace,
    ViewerApp,
)
from vincio.providers import GGUFProvider
from vincio.retrieval import FastEmbedEmbedder, LocalHashEmbedder, TwoStageIndex, VectorIndex
from vincio.storage.base import InMemoryMetadataStore


async def distributed_execution() -> None:
    print("== 1. distributed durable execution ==")
    store = InMemoryMetadataStore()
    coordinator = InMemoryGraphCoordinator()

    # A holds the thread lease; B is refused — no double-execution.
    a = DistributedCheckpointer(store, coordinator=coordinator, owner="A", lease_ttl_s=100)
    a.on_thread_start("t1")
    b = DistributedCheckpointer(store, coordinator=coordinator, owner="B", lease_ttl_s=100)
    try:
        b.on_thread_start("t1")
    except CheckpointConflictError as exc:
        print(f"   lease refused second worker: {exc.code}")
    a.on_thread_end("t1")

    # Worker pool fans a batch out across lease-coordinated workers.
    graph = StateGraph("inc")
    graph.add_node("inc", lambda s: {"n": s["n"] + 1})
    graph.add_edge("inc", END)
    results = await WorkerPoolBackend(workers=4).run_batch(graph, [{"n": i} for i in range(8)])
    print(f"   worker-pool fan-out: {[r.state['n'] for r in results]}")

    # Send map-reduce: a dispatcher fans out to workers, a reducer collects.
    # The channel default makes the reducer fold the first write into an
    # empty list, so a non-defensive ``operator.add`` needs no upstream seed node.
    mr = StateGraph("mapreduce", reducers={"out": operator.add}, defaults={"out": list})
    mr.add_node("dispatch", lambda s: [Send("double", {"x": v}) for v in s["items"]])
    mr.add_node("double", lambda s: {"out": [s["x"] * 2]})
    mr.add_node("reduce", lambda s: {"total": sum(s["out"])})
    mr.set_entry("dispatch")
    mr.add_edge("dispatch", "reduce")
    mr.add_edge("reduce", END)
    out = await mr.compile().ainvoke({"items": [1, 2, 3, 4]})
    print(f"   Send map-reduce total: {out.state['total']}")


async def executed_distillation() -> None:
    print("\n== 2. executed distillation, swap-gated ==")
    from vincio.evals.datasets import Dataset, EvalCase
    from vincio.evals.reports import CaseResult, EvalReport
    from vincio.optimize import BootstrapFinetune, TrainingExample, TrainingSet

    quality = {"teacher": (0.95, 0.01), "student-ft": (0.93, 0.002)}

    def report(q: float, cost: float) -> EvalReport:
        m = {"lexical_overlap": q, "groundedness": q, "schema_validity": 1.0,
             "safety": 1.0, "cost": cost, "latency": 50.0}
        return EvalReport(cases=[CaseResult(case_id=f"c{i}", metrics=dict(m)) for i in range(8)])

    async def evaluate(model: str, ds: object) -> EvalReport:
        return report(*quality[model])

    async def trainer(ts: TrainingSet, base: str) -> str:
        # In production: submit + poll a real fine-tune job and register the model.
        return "student-ft"

    class Gate:
        async def evaluate(self, *, candidate_model, baseline_model, dataset):
            return SimpleNamespace(passed=True, reason="no significant regression")

    ts = TrainingSet(
        examples=[TrainingExample(messages=[{"role": "user", "content": "q"},
                                            {"role": "assistant", "content": "a"}], grounded=True)]
    )
    ds = Dataset(name="held_out", cases=[EvalCase(id=f"c{i}", input="q", expected="a") for i in range(8)])
    loop = BootstrapFinetune(evaluate, trainer=trainer, min_quality_ratio=0.9, swap_gate=Gate())
    result = await loop.distill(ts, ds, teacher="teacher", student="student")
    print(f"   trained student: {result.trained_student}")
    print(f"   promoted: {result.promoted} (swap gate passed: {result.swap_passed})")
    print(f"   cost savings vs teacher: {result.cost_savings:.0%}")


def served_observability() -> None:
    print("\n== 3. served observability & alerting ==")
    store = IndexedTraceStore(":memory:")
    for i in range(10):
        status = "error" if i % 5 == 0 else "ok"
        trace = Trace(id=f"t{i}", app_name="demo", tenant_id="acme", status=status)
        trace.spans.append(Span(name="run", type="run", trace_id=f"t{i}",
                                attributes={"cost_usd": 0.002, "model": "gpt-5.2-mini"}))
        store.export(trace)
    pct = store.percentiles("cost")
    print(f"   indexed store: {store.stats()['traces']} traces, cost p95=${pct.p95:.4f}")
    by_tenant = store.cost_by_dimension("tenant")
    print(f"   cost by tenant: {[(s.key, round(s.cost_usd, 4)) for s in by_tenant]}")

    status, _ctype, body = ViewerApp(store).handle("/api/stats")
    print(f"   served /api/stats: HTTP {status} ({len(body)} bytes)")

    sink = MemoryAlertSink()
    alerts = AlertManager(sinks=[sink])
    alerts.add_rule(AlertRule(name="error_burn", metric="error_rate", kind="burn_rate",
                              threshold=14.4, slo_target=0.99))
    fired = alerts.observe("error_rate", store.stats()["error_rate"])
    print(f"   burn-rate alerts fired: {[a.rule for a in fired]}")


async def quantized_retrieval() -> None:
    print("\n== 4. quantized two-stage retrieval ==")
    docs = [
        Chunk(id="c0", document_id="d", text="The refund window is 30 days for the Pro plan."),
        Chunk(id="c1", document_id="d", text="Password reset links expire after one hour."),
        Chunk(id="c2", document_id="d", text="Enterprise plans include a dedicated manager."),
        Chunk(id="c3", document_id="d", text="Refunds are issued within thirty days of purchase."),
    ]
    embedder = LocalHashEmbedder(dim=128)
    exact = VectorIndex(embedder=embedder)
    two_stage = TwoStageIndex(embedder=embedder, quantization="scalar", coarse_dims=64, rerank_factor=4)
    await exact.add(docs)
    await two_stage.add(docs)
    for query in ["refund window", "password reset"]:
        e = (await exact.search(query, top_k=1))[0].chunk.id
        t = (await two_stage.search(query, top_k=1))[0].chunk.id
        print(f"   '{query}': exact={e} two-stage={t} {'(match)' if e == t else '(MISS)'}")


async def local_models() -> None:
    print("\n== 5. batteries-included local neural models ==")

    class FakeLlama:  # stands in for an in-process GGUF model
        def create_chat_completion(self, messages, **kw):
            return {"choices": [{"message": {"content": "answer from a local GGUF model"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 6}}

        def embed(self, text):
            return [float(len(text))] * 8

    from vincio.core.types import Message, ModelRequest

    provider = GGUFProvider(llama=FakeLlama())  # in production: model_path="model.gguf"
    resp = await provider.generate(
        ModelRequest(model="local-gguf", messages=[Message(role="user", content="hi")])
    )
    print(f"   GGUF generate: {resp.text!r} ({resp.usage.input_tokens} in tokens)")
    print(f"   GGUF on-device embedding dim: {len(await provider.embed(['hello']))} x {len((await provider.embed(['hello']))[0])}")

    # fastembed dense embedder; falls back to the offline hash embedder here.
    embedder = FastEmbedEmbedder(dim=32, fallback=True)
    vectors = await embedder.embed(["air-gapped semantic search"])
    print(f"   fastembed embedder (offline fallback) dim: {len(vectors[0])}")


async def main() -> None:
    await distributed_execution()
    await executed_distillation()
    served_observability()
    await quantized_retrieval()
    await local_models()
    print("\nAll capabilities ran offline. The single-process path stays the default.")


if __name__ == "__main__":
    asyncio.run(main())
