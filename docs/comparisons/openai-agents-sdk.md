# Vincio vs OpenAI Agents SDK

The OpenAI Agents SDK provides agents with instructions, function tools,
handoffs, and guardrails, executed by a hosted-model-centric runner.

**Where Vincio differs**

- **Provider-neutral.** Vincio agents, crews, and graphs run against any
  provider (OpenAI, Anthropic, Google, Mistral, local, mock) behind one
  interface — and offline in CI with deterministic mocks.
- **Durable state is first-class.** Stateful graphs checkpoint every step on
  your own storage (memory/SQLite/Postgres), with resume, edit-and-resume,
  and time-travel forks — no session service required.
- **Budgets and termination are enforced in code.** Steps, tool calls,
  tokens, cost, and latency are hard limits checked by the executor, not
  conventions.
- **Guardrails are the security engine.** Injection defense, PII redaction,
  RBAC/ABAC, tool approval gates, and the audit log are deterministic
  subsystems shared with the rest of the library.
- **No lock-in either way.** `OpenAIAgentsBackend` exports Vincio agents and
  crews to SDK `Agent` objects (a crew becomes a manager with handoffs), so
  you can adopt the SDK runtime where it helps and keep Vincio's compiler,
  evals, and traces.

**Where the Agents SDK is a fit:** tight integration with OpenAI-hosted
tools (web search, computer use) and the OpenAI platform's tracing UI.
