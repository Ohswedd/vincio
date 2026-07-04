# LAGER — reasoning-driven retrieval

Classic RAG runs retrieval *before* reasoning: embed the query, take the top-k
chunks, stuff them into the prompt, hope the answer is in there. That shape has
three structural failures no amount of tuning fixes:

- **the unit is wrong** — a 400-token chunk carries one useful sentence and 380
  tokens of padding;
- **the ranking is wrong for multi-hop** — the bridge fact ("the root cause was
  the payments gateway") shares zero words with the question ("why did checkout
  go down?"), so query-similarity ranking structurally cannot surface it, while
  distractors that merely repeat the question's words rank high;
- **the stopping rule is wrong** — k is fixed regardless of whether the query
  needs one fact or a causal chain, and there is no honest "this corpus cannot
  answer".

**LAGER** (Lazy Graph Evidence Retrieval) inverts the paradigm — *reasoning
drives retrieval* (RDR):

## Evidence Objects, not chunks

Ingestion transforms the corpus into **Evidence Objects**: atomic claims lifted
**byte-exactly** from the canonical source text (`claim ==
canon(text)[span]`), each carrying provenance (source, span, section),
normalized entities, typed relations, a deterministic confidence, temporal
validity, and a content-derived id — `eo:` + a hash of the document *content*,
span, and claim, so re-ingesting the same bytes in another process yields
identical ids, and any edit changes them. Lists, tables, and code blocks become
single typed objects, never sentence-shredded. Every object re-derives from its
source offline: `object.verify(document_text)` is a byte comparison, not a
model's opinion.

## A typed knowledge graph

Objects connect through typed, weighted edges — `follows` (narrative order),
`depends_on` (a pronoun-opening claim points at its antecedent, so packs stay
self-contained), `supports` (same-bucket affinity), and `contradicts` via a
**gated** detector: the raw heuristic fires on ordinary temporal/scope
variation ("raised prices in March" vs "… in June"), so LAGER suppresses
differing time scopes, qualifier-only divergence, and multi-slot changes,
gating a CI precision floor on labeled hard negatives. Every edge records the
basis that created it.

## The lazy loop

Retrieval is incremental and self-terminating. The planner turns the query into
explicit **information needs** (lookup / relation / temporal / aggregate /
causal), freezing the coverage denominators up front. Round 0 is one hybrid
seed (lexical + entity + optional dense, RRF-fused) — an easy query costs
exactly one round. Later rounds expand the graph outward from what is already
acquired (`depends_on` > `contradicts` > `supports` > `follows` > buckets >
document coherence), and the picker prefers candidates that carry the *kind* of
evidence an uncovered need requires — a causal marker for a why-need beats a
lexical echo of the question.

Coverage is **structural, not lexical**: a relation need requires an entity
path; a temporal need a dated claim; an aggregate need a quantity; a causal
need a causal-marker claim that is graph-linked to the topic — because the
answering claim often shares no words with the question. The loop has exactly
five exits, each recorded on the trace: `E0` empty frontier, `E1` sufficient,
`E2` diminishing gain, `E3` token budget, `E4` max rounds. An unanswerable
query returns `sufficient=False` **with the uncovered needs named**, so
generation abstains instead of guessing.

## Measured, not asserted

The offline `lager` bench family gates every claim against the real in-repo
baseline (never a strawman): the bridge found where same-budget hybrid top-k
honestly misses it, ≥3× fewer evidence tokens at equal correctness (~23× on the
fixture at the default 400-token-chunk configuration), one-round easy queries,
no fixed k, cross-process byte-identical determinism, contradiction precision ≥
0.8 on hard negatives, tamper-failing verification, and a full per-round gain
trace. Measured live (`benchmarks/lager_uplift_live.py`, same model, three
arms): LAGER answered 100% of the multi-hop questions at ~9× fewer input
tokens per call than the classic pipeline's 75%.

## Using it

```python
from vincio.lager import LagerEngine

engine = LagerEngine()
engine.ingest(documents)
pack = engine.retrieve("why did the checkout outage happen?")
for step in pack.gain_trace:       # every retrieval decision, explainable
    print(step)
answer = await engine.answer("why did the checkout outage happen?",
                             provider=provider, model="claude-sonnet-5")
```

Or attached to an app — the lazy loop replaces top-k for every run, riding the
same untrusted-content screen, compile, citation, and receipt pipeline:

```python
app.add_source("kb", documents=docs)
app.use_lager()
report = app.run("why did the checkout outage happen?")
pack = app.retrieve_evidence("who owns certificate management?")  # direct
```

Every module is replaceable — the extractor (any `ClaimExtractor` whose claims
are byte-exact spans), the embedder, the planner, and the controller options —
which is what makes LAGER a research platform for retrieval architectures
beyond chunk-centric RAG, not a single fixed pipeline.

See also: [Retrieval (RAG)](retrieval.md),
[Context anchors](context-anchors.md), [Context packets](context-packets.md).

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: build a RAG app](../guides/build-rag-app.md)
- [Example: 21_lager_reasoning_retrieval.py](../../examples/21_lager_reasoning_retrieval.py)
- [Concept: Context packets & long-horizon governance](context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
