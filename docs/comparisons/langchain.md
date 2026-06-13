# Vincio vs LangChain / LangGraph

LangChain offers a broad ecosystem of chains, agents, and integrations,
with LangGraph for stateful workflows and LangSmith for observability.

**Where Vincio differs**

- **The central unit is the context packet, not the chain.** Vincio compiles
  prompts, evidence, memory, tools, and policies into a scored, budgeted,
  provenance-aware packet — with an excluded-context report explaining every
  omission.
- **Compiler passes instead of string templates.** Prompts are typed ASTs
  with lint rules, hashes, diffing, and cache-aware layout.
- **Evals and optimization are built in**, not externalized to a separate
  SaaS: golden datasets, grounding metrics, judges, CI gates, and gated
  prompt/context/routing optimization ship in the core library.
- **Observability is native and provider-neutral** — every run produces a
  full trace (JSONL or OpenTelemetry) with cost tracking, no platform
  account required.
- **Deterministic security**: PII/secret detection, injection defense,
  RBAC/ABAC, tenant isolation, and audit logs are enforced in code, not by
  the model.
- **Durable graphs without a platform (0.6).** `app.graph()` gives
  LangGraph-style stateful graphs — conditional edges, checkpoints on your
  own storage, resume, edit-and-resume, time-travel forks, and
  `interrupt()` human gates — with bounded steps and the same trace/eval
  loop as every other run. And when you want LangGraph itself,
  `LangGraphBackend` exports the same graph definition to it: orchestrate
  without lock-in.

**Where LangChain is a fit:** the widest catalog of third-party
integrations and a large community. You don't have to choose — `vincio.interop`
(0.9) brings LangChain tools, retrievers, loaders, and embeddings into Vincio
(duck-typed, no import needed) and hands Vincio's back: `add_langchain_tool(app,
tool)`, `from_langchain_retriever(r)`, `from_langchain_loader(loader)`,
`from_langchain_embeddings(e)`, and `to_langchain_*` (with `vincio[langchain]`).
Migrate incrementally — see
[Coming from LangChain to Vincio](../guides/migrate-from-langchain.md).
