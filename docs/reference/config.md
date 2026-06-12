# Reference: configuration

Configuration is layered: defaults < `vincio.yaml` < `VINCIO_*` environment
variables < explicit constructor arguments.

## vincio.yaml

```yaml
project: contract_ai

provider:
  default: openai            # openai | anthropic | google | mistral | local | mock
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

retrieval:
  top_k: 8
  candidate_multiplier: 4
  chunk_size_tokens: 400
  chunk_overlap_tokens: 50
  chunking: recursive        # fixed | recursive | semantic | heading_aware | table_aware | code_aware | adaptive
  reranker: heuristic        # heuristic | recency | authority | llm | null
  embedder: local            # local | openai | google | mistral

memory:
  enabled: true
  decay_lambda: 0.01         # per day
  min_confidence: 0.25
  max_items_per_run: 8
  write_policy: guarded      # guarded | open | off

cache:
  response_cache: false
  tool_cache: true
  embedding_cache: true
  semantic_cache: false
  semantic_threshold: 0.97
  ttl_s: 3600
  max_entries: 10000

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
