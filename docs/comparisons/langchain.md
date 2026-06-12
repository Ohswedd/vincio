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

**Where LangChain is a fit:** the widest catalog of third-party
integrations and a large community. Vincio's tool registry and provider
abstraction make it straightforward to wrap any LangChain component as a
Vincio tool when you need one.
