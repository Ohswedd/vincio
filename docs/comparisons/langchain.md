# Vincio vs LangChain / LangGraph

LangChain is the broadest LLM framework in Python: a large catalog of
third-party integrations, chains and the LCEL expression language for composing
them, LangGraph for stateful workflows, and LangSmith for hosted observability.

Everything below is **built-in, in-library** capability — not what a separate
SaaS or hosted product adds on top.

**Where Vincio differs**

- **The central unit is the context packet, not the chain.** Vincio compiles
  prompts, evidence, memory, tools, and policies into a scored, budgeted,
  provenance-aware packet, with an excluded-context report and a diffable
  `CompileReceipt` explaining every omission.
- **Compiler passes instead of string templates.** Prompts are typed ASTs
  with lint rules, hashes, diffing, and cache-aware layout.
- **Composition is a typed pipeline, not string piping.** `Flow` threads
  `retrieve → ground → evaluate → run` as an immutable, inspectable
  pipeline — the Vincio analog to LCEL — but every stage is a scored,
  budgeted compiler pass that lowers to the exact same governed run, not a
  string transform.
- **Evals and optimization are built in**, not externalized to a separate
  SaaS: golden datasets, grounding metrics, judges, CI gates, and gated
  prompt/context/routing optimization ship in the core library.
- **Observability is native and provider-neutral**: every run produces a
  full trace (JSONL or OpenTelemetry) with cost tracking, no platform
  account required.
- **Deterministic security**: PII/secret detection, injection defense,
  RBAC/ABAC, tenant isolation, and audit logs are enforced in code, not by
  the model.
- **Durable graphs without a platform.** `app.graph()` gives
  LangGraph-style stateful graphs, conditional edges, checkpoints on your
  own storage, resume, edit-and-resume, time-travel forks, and
  `interrupt()` human gates, with bounded steps and the same trace/eval
  loop as every other run. And when you want LangGraph itself,
  `LangGraphBackend` exports the same graph definition to it: orchestrate
  without lock-in.

**Where LangChain is a fit:** the widest catalog of third-party
integrations and the largest community — genuine strengths a younger,
narrower library doesn't match. You don't have to choose: `vincio.interop`
brings LangChain tools, retrievers, loaders, and embeddings into Vincio
(duck-typed, no import needed) and hands Vincio's back: `add_langchain_tool(app,
tool)`, `from_langchain_retriever(r)`, `from_langchain_loader(loader)`,
`from_langchain_embeddings(e)`, and `to_langchain_*` (with `vincio[langchain]`).
Migrate incrementally, see
[Coming from LangChain to Vincio](../guides/migrate-from-langchain.md).
