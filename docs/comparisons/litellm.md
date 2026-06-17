# Vincio vs LiteLLM / gateways (Bifrost, Portkey)

LiteLLM, Bifrost, and Portkey are LLM gateways: a unified API over 100+
providers, with failover, retries, circuit breaking, key/region load
balancing, spend tracking and budgets, and caching — operated as a separate
proxy/service that sits in front of your app.

**Where Vincio differs**

- **The same reliability and FinOps controls, in-process.** Batch execution,
  circuit breakers, health-aware failover, key pooling, runtime cascades,
  cost attribution, enforced budgets, and provider-aware caching all run as a
  Python library inside your process — no extra network hop, no second
  service to deploy, scale, or page on.
- **One trace, not two systems.** A gateway is a proxy hop with its own logs;
  Vincio's reliability and cost events land on the *same* trace as the rest of
  the run. Model spans carry `cache_hit_rate` and `cached_input_tokens`;
  `circuit.opened`/`circuit.closed`, `cost.anomaly`, and
  `cost.budget_exceeded` ride the same event bus the agent does.
- **Resilience composes from real objects.** `CircuitBreaker(inner)` is itself
  a `ModelProvider` (CLOSED/OPEN/HALF_OPEN, failure-rate and latency
  thresholds); `HealthAwareFailover` tries healthy breakers first and skips
  open ones in microseconds via a non-retryable `CircuitOpenError`; `KeyPool`
  does round-robin across keys/regions with dual RPM+TPM token buckets and
  full-jitter backoff that honors `retry_after`. The recommended pattern is
  plain composition: `CircuitBreaker(RetryingProvider(provider))`.
- **Budgets that enforce, not just report.** `app.set_cost_budget(...)` sets a
  limit scoped to `tenant`/`feature`/`user`/`global` over a `run`/`hour`/`day`/
  `month`/`total` period, and on breach will `cap`, `degrade` (to a cheaper
  `degrade_model`), or `queue_to_batch`. `anomaly_factor` emits a
  `cost.anomaly` event; the decision is written to the hash-chained audit log
  as a `cost_budget` action and a `PolicyViolation` — governed by the same
  deterministic policy engine as everything else in the run.
- **Cost attribution down to the feature.** `arun`/`astream` take
  `tenant_id`/`user_id`/`session_id` and a `feature` dimension; `CostLedger`
  records every call and `app.cost_report(by="tenant"|"feature"|"user"|
  "model"|"provider"|"run")` (or `vincio cost report --by ...`) rolls it up —
  no separate analytics store.
- **Batch and cascades are first-class.** `app.batch(inputs, discount=0.5)`
  (and `vincio batch app.py`) runs at provider batch pricing through
  `InProcessBatchBackend` offline or `OpenAIBatchBackend`/
  `AnthropicBatchBackend` live; `app.use_cascade(models=[...])` escalates from
  cheap to capable only when `response_confidence` falls below
  `min_confidence`, with `max_escalations` as a hard cap.
- **Provider-aware prompt caching.** `app.enable_prompt_caching(ttl="5m"|"1h",
  min_prefix_tokens=1024)` applies caching only where `ModelCapabilities`
  support it, and it's default-on via the `vincio.yaml` cache section
  (`provider_cache`, `provider_cache_ttl`, `provider_cache_min_prefix_tokens`).
- **Capability-aware routing and a gated swap (1.8).** Where LiteLLM Router
  load-balances by health and cost, Vincio's `app.use_router([...])` routes by
  **capability *and* cost** — it refuses a model that can't serve the request
  (vision/tools/schema/reasoning/context), so failover never silently returns a
  wrong answer. And a model swap is *gated*, not just compared:
  `app.gate_swap(candidate, dataset=, traces=)` replays golden traces and runs an
  eval + cost + latency + behavioral diff with statistical significance
  (PASS/FAIL), `app.swap_regression(...)` / `vincio eval regress` quantifies "is
  the cheaper model safe?", and `app.shadow(...)` / `app.canary(...)` qualify a
  candidate on live traffic with automatic rollback.

| Capability | Gateway (LiteLLM / Bifrost / Portkey) | Vincio |
| --- | --- | --- |
| Deployment | Separate proxy/service, extra hop | In-process library, no hop |
| Failover / circuit breaking | In the proxy | `HealthAwareFailover`, `CircuitBreaker` |
| Routing | By health/cost in the proxy | By **capability + cost** (`use_router`), refuses mismatches |
| Key/region load balancing | In the proxy | `KeyPool` (RPM+TPM buckets) |
| Spend tracking | Gateway dashboard | `CostLedger` / `cost_report` |
| Budgets | Report + block | Enforced: `cap`/`degrade`/`queue_to_batch` + audit |
| Batch pricing | Varies | `app.batch(discount=0.5)` (OpenAI/Anthropic/Google), `vincio batch` |
| Model swap | Manual config edit | **Gated**: `SwapGate` (replay + significance), shadow/canary auto-rollback |
| Caching | Proxy-level | Provider-aware `PromptCacheStrategy` |
| Governance / audit | Gateway logs | Same policy engine + hash-chained audit, one trace |

**Where gateways are a fit:** their provider breadth and their
language-agnostic proxy deployment model are real strengths a Python library
doesn't replicate — if your fleet is polyglot or you want one spend/routing
control plane shared across many unrelated apps, a gateway is the right tool.
Vincio doesn't try to be that proxy: it reaches any OpenAI-compatible endpoint
plus named gateway presets, so it can even sit *behind* one of these gateways
and let Vincio own the in-process policy, budgets, and trace while the gateway
owns the network edge.
