# Vincio vs CrewAI

CrewAI popularized role-based multi-agent teams: agents with roles, goals,
and backstories collaborate on tasks sequentially or under a manager.

**Where Vincio differs**

- **Bounded by construction.** Every crew member runs a bounded
  `AgentExecutor` under a scaled share of the crew budget, the crew checks
  its budget before each delegation, and hierarchical review is capped at
  `max_rounds`. Termination is a guarantee, not a hope.
- **A real blackboard, not message passing.** Members coordinate through
  versioned, author-attributed shared memory that snapshots to JSON, crew
  runs can be persisted, diffed, and replayed.
- **Deterministic offline behavior.** Manager delegation is LLM-planned with
  a schema-validated plan and a deterministic keyword-routing fallback, so
  crews run in CI with the mock provider and the same code path.
- **One trace, eval-ready.** The crew emits a `crew` span, each member a
  `crew_agent` span, and `CrewResult.metrics()` aggregates per-member agent
  metrics, the same objects the eval runner gates releases with.
- **Context is compiled, not concatenated.** Members read evidence through
  the context compiler, so scoring, budgeting, provenance, and injection
  defense apply to every agent automatically.

**Where CrewAI is a fit:** a large gallery of community crew templates and
quick role-play-style prototypes. Vincio's `OpenAIAgentsBackend` pattern
shows the shape of interop: crews are plain data (roles + executors), easy to
export to other runtimes.
