# Cookbook: task-shaped recipes

The cookbook is a curated set of runnable, tested recipes — each a small,
end-to-end app for a concrete task. Every recipe runs **fully offline** on the
deterministic mock provider and is part of the
[example gate](test-llm-apps.md), so the code behind each recipe is proven to
run, not just described. Point any of them at a real model with
`VINCIO_PROVIDER` / `VINCIO_MODEL`.

| Recipe | Task | Builds on |
|---|---|---|
| [Contract redlining](../../examples/45_recipe_contract_redlining.py) | Review a clause and emit a tracked-change redline | `legal` pack + `generate_redline` |
| [Incident triage](../../examples/46_recipe_incident_triage.py) | Turn an alert + runbooks into a typed, grounded triage decision | typed output + grounding |
| [Data-room Q&A](../../examples/47_recipe_data_room_qa.py) | Diligence Q&A over a virtual data room with citations | hybrid RAG + citation policy |
| [Multimodal RAG over slides & PDFs](../../examples/48_recipe_multimodal_rag.py) | Answer from text, a table, and a slide image in one packet | multimodal context packet |

The capability examples that the recipes lean on:

| Example | Shows |
|---|---|
| [Vertical packs](../../examples/42_vertical_packs.py) | a regulated domain configured in one `use_pack` — see [vertical packs](vertical-packs.md) |
| [The Assistant](../../examples/43_assistant.py) | a multi-turn chat product with tool approvals — see [the Assistant](assistant.md) |
| [The voice agent](../../examples/44_voice_agent.py) | a grounded, guarded spoken assistant — see [voice & realtime](realtime.md) |

## Contract redlining

Review a contract clause-by-clause with the `legal` pack, then emit a
tracked-change redline (markdown by default; DOCX with `vincio[gen-docx]`). The
review is grounded in the contract text; the redline is a deterministic diff of
original vs. revised, so the change set is auditable.

## Incident triage

Attach your runbooks as a source, classify an alert into a typed `Triage`
(severity / component / mitigation / runbook ref), and answer only from the
runbooks — so the on-call action is cited, not improvised.

## Data-room Q&A

Load the documents of a virtual data room and answer diligence questions with
`answer_only_from_sources` + `require_citations`, so every claim traces back to a
specific file. The same shape powers M&A diligence, vendor review, and audit prep.

## Multimodal RAG over slides & PDFs

Text, a table, and a slide image are first-class candidates in **one scored,
budgeted [context packet](../concepts/context-packets.md)**. The context compiler
scores and orders them together with modality-aware token cost, so an answer can
draw on a chart slide and a metrics table alongside prose — and cite each.

## More

These recipes compose the same primitives the rest of the docs cover in depth:
[build a RAG app](build-rag-app.md), [structured output](structured-output.md),
[reliability & guardrails](reliability-guardrails.md), and
[generate documents & media](generate-documents.md).
