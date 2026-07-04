"""LAGER — reasoning-driven retrieval: evidence objects, a knowledge graph,
and a lazy loop instead of top-k chunks.

Classic RAG retrieves top-k chunks by query similarity, then generates. That
structurally fails on multi-hop questions (the bridge fact shares no words with
the query), wastes tokens (a 400-token chunk carries one useful sentence), and
has no honest "this corpus cannot answer". LAGER inverts it: the corpus becomes
atomic, byte-exact **Evidence Objects** in a typed knowledge graph, and
retrieval is **lazy** — needs-driven, incremental, self-terminating.

This tour runs fully offline with a scripted mock model.

Sections:
  1. Ingest — documents become verifiable Evidence Objects + a typed graph.
  2. Lazy retrieval — one round for an easy query, graph hops for a hard one.
  3. Honest abstention — an unanswerable query names what is missing.
  4. Grounded answering + offline verification through the app.
"""

from __future__ import annotations

import tempfile

from vincio.core.app import ContextApp
from vincio.core.config import VincioConfig
from vincio.core.types import Document
from vincio.lager import LagerEngine
from vincio.providers import MockProvider

CORPUS = [
    Document(title="incident report", text=(
        "The checkout service suffered a full outage on 2025-11-03. "
        "Customers could not complete purchases for three hours. "
        "The incident review assigned the root cause to the payments gateway."
    )),
    Document(title="gateway runbook", text=(
        "The payments gateway rejected all connections because its TLS "
        "certificate had expired. Certificate management is owned by the "
        "platform team."
    )),
    Document(title="marketing notes", text=(
        "The marketing team launched a new checkout banner in October. "
        "Checkout conversion rates improved after the redesign. "
        "The outage dashboard shows overall availability trends for checkout."
    )),
]


def _config() -> VincioConfig:
    tmp = tempfile.mkdtemp(prefix="vincio_lager_")
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp}/vincio.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = f"{tmp}/audit"
    return config


def main() -> None:
    # 1. Ingest: claims lifted byte-exactly, connected in a typed graph.
    engine = LagerEngine()
    count = engine.ingest(CORPUS)
    edges = sum(len(v) for v in engine.graph.edges.values())
    print("1. Ingest")
    print(f"   {len(CORPUS)} documents → {count} evidence objects, {edges} typed edges")
    sample = engine.objects[0]
    print(f"   e.g. {sample.id}: \"{sample.claim[:60]}…\" span={sample.span}")
    print(f"   byte-exact re-derivation: {sample.verify(CORPUS[0].text)}")

    # 2. Lazy retrieval: rounds scale with the question, not a fixed k.
    print("\n2. Lazy retrieval")
    easy = engine.retrieve("who owns certificate management")
    print(f"   easy query   → {easy.rounds} round, {len(easy.objects)} objects, "
          f"{easy.token_cost} tokens, exit {easy.exit_reason}")
    hard = engine.retrieve("why did the checkout outage happen")
    print(f"   why-question → {hard.rounds} rounds, {len(hard.objects)} objects, "
          f"{hard.token_cost} tokens, exit {hard.exit_reason}")
    bridge = next((o.claim for o in hard.objects if "root cause" in o.claim), None)
    print(f"   bridge found via graph (zero word overlap with the query): {bridge!r}")
    print("   the decision trace (every round, explainable):")
    for step in hard.gain_trace:
        print(f"     round {step['round']}: +{len(step['added'])} objects, "
              f"gain {step['gain']}")

    # 3. Honest abstention: unanswerable queries name what is missing.
    impossible = engine.retrieve("what is the chief executive compensation package")
    print("\n3. Honest abstention")
    print(f"   sufficient={impossible.sufficient}, exit {impossible.exit_reason}")
    print(f"   uncovered needs: {impossible.uncovered_needs}")

    # 4. Through the app: the lazy pack replaces top-k on every run and rides
    #    the same screened, compiled, cited pipeline.
    def responder(request):
        joined = "\n".join(m.content if isinstance(m.content, str) else ""
                           for m in request.messages)
        return ("The outage was caused by the payments gateway's expired TLS "
                "certificate." if "root cause" in joined else "insufficient evidence")

    app = ContextApp(name="lager-demo", provider=MockProvider(responder=responder),
                     model="mock-1", config=_config())
    app.add_source("kb", documents=CORPUS)
    app.use_lager()
    result = app.run("why did the checkout outage happen")
    print("\n4. Through the app (app.use_lager)")
    print(f"   model answered from the lazy pack: {result.raw_text!r}")
    pack = app.retrieve_evidence("why did the checkout outage happen")
    print(f"   offline verification of the pack: {app.lager_engine.verify(pack)}")

    print("\nDone — retrieval as a consequence of reasoning: minimal, traceable, "
          "graph-guided, and honest about what it cannot answer.")


if __name__ == "__main__":
    main()
