# Learning path

A staged route through Vincio, from your first grounded app to the full
platform. Each stage builds on the one before; follow it top to bottom, or
jump to the stage that matches what you are building. The
[documentation index](README.md) is the exhaustive map and the
[capability map](reference/capability-map.md) binds every `app.*` verb to
the page that documents it.

## Stage 1 — Get running

Install, scaffold, and run your first grounded app offline on the deterministic
mock provider.

- [Getting started](getting-started.md) — install, scaffold, first run, first eval.
- [Example: the one-line front door](../examples/00_one_liners.py) and [the ergonomic surface](concepts/ergonomic-surface.md).
- [Example: the five-minute tour](../examples/01_quickstart.py).
- [Cookbook](guides/cookbook.md) — short, runnable recipes to copy.

## Stage 2 — The core model

Understand what happens between your input and the validated output.

- [Context packets & the context compiler](concepts/context-packets.md) — the central unit.
- [The prompt compiler](concepts/prompt-compiler.md) — prompts are compiled, not concatenated.
- [Retrieval](concepts/retrieval.md) and [build a RAG app](guides/build-rag-app.md).
- [Memory](concepts/memory.md) — scoped, decaying, conflict-resolving recall.

## Stage 3 — Build a real application

Add tools, typed output, guardrails, and the structured-data plane.

- [Add tools](guides/add-tools.md) and [structured output](guides/structured-output.md).
- [Reliability & guardrails](guides/reliability-guardrails.md).
- [Tabular evidence](concepts/tabular-evidence.md) → [analyze data](guides/analyze-data.md) → [the data engagement](concepts/data-engagement.md).
- [Generate documents & media](guides/generate-documents.md).

## Stage 4 — Evaluate and improve

Turn quality into numbers, gate CI on them, and close the optimization loop.

- [Evaluation](concepts/evals.md) → [run evals](guides/run-evals.md) → [test with pytest](guides/test-llm-apps.md).
- [The open evaluation plane](concepts/open-evaluation-plane.md) → [run the three-track benchmark platform](guides/run-benchmark-suite.md) ([example](../examples/16_open_evaluation_plane.py)).
- [Agentic evaluation & continuous quality](guides/agentic-eval.md).
- [Close the loop](guides/close-the-loop.md) and [optimize context](guides/optimize-context.md).
- [Cost, reliability & scale](guides/cost-and-reliability.md) and [performance](guides/performance.md).

## Stage 5 — Orchestrate and interoperate

Compose multi-agent systems and connect them across processes and vendors.

- [Agents & workflows](concepts/agents.md) → [orchestrate multi-agent systems](guides/orchestrate-agents.md).
- [MCP](guides/mcp.md), [A2A](guides/a2a.md), [Agent Skills](guides/agent-skills.md), and [the agent fabric](guides/agent-fabric.md).
- [Reasoning control](guides/reasoning.md) and [voice & realtime](guides/realtime.md).

## Stage 6 — Govern, secure, and assure

Produce compliance evidence and formal guarantees from the live system.

- [Enterprise governance](guides/governance.md) and [formal verification](guides/governance-verification.md).
- [Verified reasoning](guides/verified-reasoning.md), [continuous assurance](guides/assurance.md), and [agent identity](guides/agent-identity.md).
- [Differential-privacy memory & training](guides/differential-privacy.md).
- [The threat model](security/threat-model.md).

## Stage 7 — The cross-organization economy & advanced runtimes

Transact across organizations and run Vincio beyond the default server path.

- [Negotiation & contracting](guides/negotiation.md) → [choreography](guides/choreography.md) → [settlement](guides/settlement.md).
- [Example: the cross-org economy](../examples/12_cross_org_economy.py).
- [Edge / WASM runtime](guides/edge.md), [computer-use](guides/computer-use.md), [video](guides/video.md), and [skill acquisition](guides/skill-acquisition.md).

## Keep going

- [Capability map](reference/capability-map.md) — every `app.*` verb and where it is documented.
- [API reference](reference/api.md) and [CLI reference](reference/cli.md).
- [Migrating from another library](guides/migrate-from-langchain.md).

Run `vincio docs check` to prove this graph is intact, or `vincio docs map`
to regenerate the capability map.
