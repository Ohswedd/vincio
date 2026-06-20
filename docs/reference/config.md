# Reference: configuration

Configuration is layered: defaults < `vincio.yaml` < `VINCIO_*` environment
variables < explicit constructor arguments.

Generate a JSON Schema for editor completion with `vincio config schema --output
vincio.schema.json` and reference it from a `# yaml-language-server: $schema=…`
comment at the top of the file (`vincio init` writes both for you). Check a
config with `vincio config validate` and print the effective merged result with
`vincio config show`.

## vincio.yaml

```yaml
schema_version: 1            # config schema version; `vincio config migrate` upgrades older files
project: contract_ai

provider:
  default: openai            # openai | anthropic | google | mistral | local | mock
                             # + OpenAI-compatible presets: groq | together | fireworks |
                             #   openrouter | deepseek | perplexity | xai | nvidia (keys via <NAME>_API_KEY)
  model: gpt-5.2
  fallback_models: []        # used by FailoverChain
  base_urls: {}              # e.g. local: "http://localhost:8000/v1"
  timeout_s: 120
  max_retries: 2

storage:
  metadata: sqlite:///.vincio/vincio.db   # memory:// | sqlite:///… | postgres://…
  vector: memory://                       # memory:// | qdrant://… | postgres://…
  documents_dir: .vincio/documents
  analytics: null                         # duckdb:///.vincio/analytics.duckdb

observability:
  exporter: jsonl            # jsonl | memory | console | otel | none
  traces_dir: .vincio/traces
  sample_rate: 1.0

security:
  tenant_isolation: true
  pii_detection: true
  injection_detection: true
  audit_log: true
  audit_dir: .vincio/audit
  retention_days: null
  egress_dlp: warn           # always-on egress DLP of the assembled request: off | warn | block
  audit_signing_key: ""      # HMAC key → tamper-evident signed audit chain (empty = unsigned, 1.x behavior)
  audit_signing_key_id: hmac # key id recorded on each signed entry

governance:                  # enterprise governance & compliance; all opt-in
  allowed_regions: []        # non-empty pins data-residency: runs may only egress to these regions
  provider_regions: {}       # e.g. {openai: us, anthropic: us} — region per provider/model
  deny_on_unknown_region: true
  content_marking: false     # attach a synthetic-content manifest + AI disclosure to every run
  locales: []                # non-English PII packs: fr | de | es | in | sg | br | uk
  card_format: vincio        # vincio | open_model_card | ai_card

retrieval:
  top_k: 8
  candidate_multiplier: 4
  chunk_size_tokens: 400
  chunk_overlap_tokens: 50
  chunking: recursive        # fixed | recursive | semantic | heading_aware | table_aware | code_aware | sentence_window | hierarchical | parent_document | contextual | adaptive
  reranker: heuristic        # heuristic | recency | authority | llm | cohere | jina | voyage | null
  embedder: local            # local | jina | voyage | cohere | voyage-context | voyage-multimodal | cohere-multimodal | openai | google | mistral | <provider>
  embedding_dimensions: null # int | None — Matryoshka output-dimension truncation; null keeps the native dimension
  query_strategies: []       # hyde | multi_query | decompose | step_back

memory:
  enabled: true
  decay_lambda: 0.01         # per day
  min_confidence: 0.25
  max_items_per_run: 8
  write_policy: guarded      # guarded | open | off
  hybrid_recall: true        # fuse lexical + vector + graph signals per query
  vector_weight: 0.5         # vector share of fused relevance
  retention_weight: 0.5      # importance-weighted retention strength
  ttl_days:                  # default TTL per scope; unlisted scopes never expire
    session: 30
  write_back: [input]        # input | evidence | tools | facts
  fact_min_support: 0.5      # evidence support an output claim needs to become
                             # a candidate memory (write_back includes "facts")
  max_facts_per_run: 5

cache:
  response_cache: false
  tool_cache: true
  embedding_cache: true
  semantic_cache: false
  semantic_threshold: 0.97
  ttl_s: 3600
  max_entries: 10000
  prompt_compile_cache: true   # content-addressed; unchanged inputs never recompile
  chunk_cache: true            # content-addressed; unchanged docs never re-chunk
  context_compile_cache: true  # content-addressed; unchanged compile inputs hit
  provider_cache: true         # provider-aware prompt caching: attach an Anthropic
                               # cache_control breakpoint with a TTL to the compiler's
                               # stable prefix, and record cache-hit-rate telemetry;
                               # auto-cache providers rely on stable→volatile ordering
  provider_cache_ttl: 5m       # breakpoint TTL: "5m" or "1h"
  provider_cache_min_prefix_tokens: 1024  # min stable-prefix length before a breakpoint
                                          # TTL is applied

performance:
  max_concurrency: 8           # bound for retrieval/memory/ingest fan-out
  tool_parallelism: 4          # concurrent tool calls within one model round
  embed_batch_size: 64         # max texts per provider embedding call
  embed_window_ms: 5.0         # BatchingEmbedder micro-batch window
  coalesce_requests: true      # identical in-flight model calls share one request
  max_connections: 100         # provider HTTP connection pool
  max_keepalive_connections: 20
  slim_packets: false          # packets reference evidence text by content hash
  partial_parse_min_chars: 24  # min new chars between partial-JSON parses when streaming

server:
  host: 127.0.0.1
  port: 8042
  api_keys: []
  jwt_secret: null
  cors_origins: []

budget:
  max_input_tokens: 100000
  max_output_tokens: 4096
  max_latency_ms: 120000
  max_cost_usd: 1.0
  max_steps: 16
  max_tool_calls: 32

policies:
  privacy: tenant_isolated   # open | tenant_isolated | user_isolated
  safety: standard           # minimal | standard | strict
  answer_only_from_sources: false
  require_citations: false
  allow_memory_writes: true
  allow_external_tools: true
  redact_pii_in_context: false
  block_untrusted_instructions: true
```

## Schema versioning & migrations

A config carries a `schema_version`. When the schema evolves, Vincio migrates
older files mechanically instead of letting them silently drift:

- **On load**, `load_config` applies all pending migrations *in memory*, so a
  stale file always validates against the current schema (the file on disk is
  untouched).
- **`vincio config migrate [path]`** persists the upgrade, reporting each
  applied step and the concrete changes it made. Use `--check` in CI to fail if
  a file is behind (it exits non-zero without writing), `--dry-run` to preview,
  or `--output` to write elsewhere. A leading `# yaml-language-server` schema
  hint is preserved.
- **`vincio doctor`** flags a project whose `vincio.yaml` is behind the current
  schema and points at `config migrate`.

Files written before versioning have no `schema_version` and load as version 0,
then migrate forward. Migrations are idempotent — re-running is a no-op.

```bash
vincio config migrate                 # upgrade ./vincio.yaml in place
vincio config migrate --check         # CI gate: non-zero if a migration is pending
vincio config migrate old.yaml --dry-run
```

## Environment variables

`VINCIO_<SECTION>__<FIELD>` overrides any config field:

```bash
export VINCIO_PROVIDER__MODEL=gpt-5.2-mini
export VINCIO_RETRIEVAL__TOP_K=12
export VINCIO_SECURITY__TENANT_ISOLATION=false
```

API keys resolve from standard env vars (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`) or
`provider.api_keys` indirection.
