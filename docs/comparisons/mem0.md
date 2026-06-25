# Vincio vs Mem0

Mem0 is a popular memory layer for AI applications: user/agent/session
scopes, extraction, vector + graph storage, and a managed platform.

**Where Vincio differs**

- **Memories are governed, not just stored.** Every write passes a
  deterministic policy: credential/PII screening, stability scoring (volatile
  statements rejected), contradiction detection with supersede/flag
  resolution, confidence assignment, and provenance (`source_trace_id`).
- **Recall is utility-scored against the task** before a memory ever enters
  a packet: hybrid lexical + vector + graph relevance is weighed against
  recency, decayed confidence, scope match, stability, status, token cost,
  and privacy risk, the same scored-budgeted treatment evidence gets.
- **Forgetting is a feature**: exponential confidence decay with usage and
  confirmation boosts, per-scope TTLs, and importance-weighted retention,
  plus user-driven `edit` / `forget` / `export` / `erase` that flow through a
  hash-chained audit log (GDPR-style access and erasure).
- **Consolidation keeps provenance.** Episodic→semantic promotion records
  `consolidated_from` on every promoted memory and archives the episodes
  with a backref, nothing is silently merged away.
- **Measured, not assumed**: the memory eval harness (recall precision,
  contradiction rate, staleness, personalization lift) runs in VincioBench
  and gates releases in CI.
- **One system, no hosted dependency**: memory shares the runtime's stores,
  traces, security engine, and audit log, and runs fully offline.

**Where Mem0 is a fit:** a hosted, drop-in memory API across many separate
apps with managed infrastructure. Vincio's `remember` / `recall` /
`for_user(...)` ergonomics mirror that surface, so migration is mostly
mechanical.
