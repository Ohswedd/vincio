# Cookbook: task-shaped recipes

The cookbook is a curated set of runnable, tested recipes, each a small,
end-to-end app for a concrete task. Every recipe runs **fully offline** on the
deterministic mock provider and is part of the
[example gate](test-llm-apps.md), so the code behind each recipe is proven to
run, not just described. Point any of them at a real model with
`VINCIO_PROVIDER` / `VINCIO_MODEL`.

| Recipe | Task | Builds on |
|---|---|---|
| [Contract redlining](../../examples/09_security_governance.py) | Review a clause and emit a tracked-change redline | `legal` pack + `generate_redline` |
| [Incident triage](../../examples/04_agents_and_tools.py) | Turn an alert + runbooks into a typed, grounded triage decision | typed output + grounding |
| [Data-room Q&A](../../examples/02_retrieval_rag.py) | Diligence Q&A over a virtual data room with citations | hybrid RAG + citation policy |
| [Multimodal RAG over slides & PDFs](../../examples/02_retrieval_rag.py) | Answer from text, a table, and a slide image in one packet | multimodal context packet |

The capability examples that the recipes lean on:

| Example | Shows |
|---|---|
| [Vertical packs](../../examples/10_interop_and_protocols.py) | a regulated domain configured in one `use_pack`, see [vertical packs](vertical-packs.md) |
| [The Assistant](../../examples/01_quickstart.py) | a multi-turn chat product with tool approvals, see [the Assistant](assistant.md) |
| [The voice agent](../../examples/01_quickstart.py) | a grounded, guarded spoken assistant, see [voice & realtime](realtime.md) |

## Contract redlining

Review a contract clause-by-clause with the `legal` pack, then emit a
tracked-change redline (markdown by default; DOCX with `vincio[gen-docx]`). The
review is grounded in the contract text; the redline is a deterministic diff of
original vs. revised, so the change set is auditable.

## Incident triage

Attach your runbooks as a source, classify an alert into a typed `Triage`
(severity / component / mitigation / runbook ref), and answer only from the
runbooks, so the on-call action is cited, not improvised.

## Data-room Q&A

Load the documents of a virtual data room and answer diligence questions with
`answer_only_from_sources` + `require_citations`, so every claim traces back to a
specific file. The same shape powers M&A diligence, vendor review, and audit prep.

## Multimodal RAG over slides & PDFs

Text, a table, and a slide image are first-class candidates in **one scored,
budgeted [context packet](../concepts/context-packets.md)**. The context compiler
scores and orders them together with modality-aware token cost, so an answer can
draw on a chart slide and a metrics table alongside prose, and cite each.

## More

These recipes compose the same primitives the rest of the docs cover in depth:
[build a RAG app](build-rag-app.md), [structured output](structured-output.md),
[reliability & guardrails](reliability-guardrails.md), and
[generate documents & media](generate-documents.md).

## Running a recipe

Each recipe is a numbered file under `examples/`. Run it as-is and it uses the
deterministic mock — no key, no network:

```bash
python examples/09_security_governance.py          # contract redlining, offline

VINCIO_PROVIDER=openai VINCIO_MODEL=gpt-5.2 \
  python examples/09_security_governance.py         # same code, a real model
```

The example file is the source of truth — the prose above summarizes what it
does, but the code is what the [example gate](test-llm-apps.md) proves runs.
Read the file to copy the exact wiring.

## Gotchas

- **The mock proves the pipeline, not the answer.** Recipes run offline because
  the mock emits schema-valid output — that guarantees the plumbing (retrieval,
  validation, rails, rendering) works, but it is not a quality signal. Point at a
  real model before judging output.
- **Some recipes want an extra.** The redline recipe renders Markdown with no
  dependency but needs `pip install "vincio[gen-docx]"` for DOCX; a recipe that
  loads PDFs needs `vincio[pdf]`.
- **A real provider changes cost and latency.** `VINCIO_PROVIDER` / `VINCIO_MODEL`
  swap the model without touching the recipe, but now every run bills and blocks
  on the network — keep the mock for iteration.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: The ergonomic front door](../concepts/ergonomic-surface.md)
- [Guide: Performance & streaming](performance.md)
- [Example: 01_quickstart.py](../../examples/01_quickstart.py)
- [Example: 00_one_liners.py](../../examples/00_one_liners.py)
- [Concept: Prompt compiler](../concepts/prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
