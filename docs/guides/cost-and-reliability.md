# Cost, reliability & scale

Vincio runs the FinOps and resilience layer in-process: batch at half cost,
compose retries / failover / circuit-breaking, pool keys across regions,
cascade cheap→strong by confidence, attribute and budget every dollar, cache
provider prefixes, and index incrementally or sharded — all on the same trace,
audit log, and policy engine as the rest of the run. Everything here is
`@experimental` since 1.3 unless noted, and the examples run offline.

## Batch at ~50% cost

`app.batch([...])` runs many inputs through the in-process batch backend at the
batch discount (`discount=0.5` by default). Reach for it whenever the work is
asynchronous and bulk: eval suites, bulk extraction over a corpus, synthetic
data generation:

```python
from vincio import ContextApp

app = ContextApp(name="extract")

inputs = [f"Extract the invoice total from: {doc}" for doc in docs]
results = app.batch(inputs)               # list[RunResult], ~50% cost

for r in results:
    print(r.output)
```

The same app runs sync (`app.run`) or batch (`app.batch`) — switch by calling
the other method, no rewrite. Use `app.abatch(...)` for the async form.

For explicit reconcile-by-id and partial-failure handling, drop to
`BatchRunner` with `BatchRequest`/`custom_id`:

```python
from vincio.core.types import ModelRequest
from vincio.providers import BatchRequest, BatchRunner

runner = BatchRunner(provider, discount=0.5)   # provider → in-process backend

requests = [
    BatchRequest(custom_id=doc.id, request=ModelRequest(messages=[...]))
    for doc in docs
]
result = await runner.run(requests)             # BatchRunResult

print(result.cost_usd, len(result.succeeded), len(result.failed))
by_id = result.by_id()                          # custom_id -> BatchResult
for doc in docs:
    res = by_id[doc.id]
    if res.ok:
        save(doc.id, res.response)
    else:
        retry_later(doc.id, res.error)          # partial failures don't abort the job
```

A `BatchResult` carries `custom_id`, `response`, `error`, and `.ok` — one
failed request never sinks the rest. Switch the backend to ship to a real
provider's batch API without touching the request code:

```python
from vincio.providers import OpenAIBatchBackend, InProcessBatchBackend

backend = OpenAIBatchBackend(openai_provider, completion_window="24h")  # 50% off
# backend = InProcessBatchBackend(provider, concurrency=8)             # offline default
runner = BatchRunner(backend)
```

From the shell: `vincio batch app.py --input X --input Y [--input-file lines.txt]
[--discount 0.5] [--output results.json]` (exit 1 if any request failed).

## The reliability pattern

Three failure modes, three tools, composed inner→outer:

- **Transient** (a timeout, a 503) — retry it: `RetryingProvider`.
- **Persistent** (one key/region is down) — fail over: `HealthAwareFailover`.
- **Systemic** (a provider is melting down) — stop hammering it: `CircuitBreaker`.

Wrap retries on the inside and the breaker on the outside, so the breaker sees
the outcome *after* retries and trips on sustained failure rather than a single
hiccup:

```python
from vincio.providers import CircuitBreaker, RetryingProvider, HealthAwareFailover

protected = CircuitBreaker(RetryingProvider(provider))   # retry transient, break systemic
```

Then put a failover chain over two breaker-wrapped providers. An open breaker
raises a non-retryable `CircuitOpenError`, so the chain skips a sick provider in
microseconds instead of waiting on it:

```python
primary   = CircuitBreaker(RetryingProvider(provider_a))
secondary = CircuitBreaker(RetryingProvider(provider_b))

failover = HealthAwareFailover([
    (primary,   "primary"),
    (secondary, "secondary"),
])

response = await failover.generate(request)
```

`CircuitBreaker` is itself a `ModelProvider`, so it drops in anywhere. Inspect
it with `.state` (`CLOSED`/`OPEN`/`HALF_OPEN`), `.healthy`, `.failure_rate()`,
and `.snapshot()`. State transitions emit `circuit.opened` / `circuit.closed` /
`circuit.half_open` on the event bus.

## Key pooling across keys & regions

`KeyPool` spreads load round-robin over a list of providers (different keys or
regions), each guarded by a health-aware breaker, with dual RPM + TPM token
buckets and full-jitter backoff that honors any `retry_after`:

```python
from vincio.providers import KeyPool

pool = KeyPool(
    [provider_us, provider_eu, provider_ap],
    rpm=600,                       # requests/min across the pool
    tpm=400_000,                   # tokens/min across the pool
    labels=["us", "eu", "ap"],
    base_backoff_s=0.5,
    max_backoff_s=30.0,
)

response = await pool.generate(request)   # picks a healthy key, rate-limits, backs off
```

It's a `ModelProvider` too, so wrap it or feed it to `HealthAwareFailover` like
any other. For standalone limiting, `RateLimiter(rpm=..., tpm=...)` exposes
`.acquire(tokens=0)`, `.available(tokens)`, and `.wait_time(tokens)`.

## Runtime model cascades

`app.use_cascade([...])` starts each run on the cheap model and only escalates
to a stronger one when confidence is low — paying for the big model only on the
hard inputs:

```python
app.use_cascade(["gpt-5.2-mini", "gpt-5.2"], min_confidence=0.5)

result = app.run("Classify this ticket: ...")   # cheap first, escalates if unsure
```

The default confidence comes from `response_confidence` (1.0 on a clean stop,
0.0 on a length/content-filter/error stop, 0.2 when a schema was expected but
didn't parse). Supply your own callable to score escalation any way you like:

```python
def my_confidence(response) -> float:
    return 0.9 if "yes" in response.text.lower() else 0.3

app.use_cascade(["cheap", "strong"], confidence=my_confidence, max_escalations=1)
```

Behind the scenes this builds a `ModelCascade` of `CascadeRung`s; the offline
`RoutingOptimizer` keeps tuning the per-rung thresholds from your eval data, so
the cascade gets cheaper without getting worse.

## Cost attribution & budgets

Pass `feature=` and `tenant_id=` to a run and every model call is tagged on the
cost ledger (alongside `user_id` / `session_id`):

```python
result = await app.arun(
    "Summarize this thread",
    tenant_id="acme",
    feature="inbox_summary",
)
```

Roll the ledger up along any dimension:

```python
report = app.cost_report(by="tenant")      # or "feature" / "user" / "model" / "provider" / "run"
report.print_summary()
print(report.total_usd)
for row in report.rows:
    print(row.key, row.cost_usd, row.calls, row.cached_input_tokens)
```

Set enforced budgets with an automatic action on breach — cap (deny), degrade
to a cheaper model, or queue the work to batch — and an anomaly factor that
fires `cost.anomaly` when spend spikes:

```python
app.set_cost_budget(
    limit_usd=50.0,
    scope="tenant",
    id="acme",
    period="day",
    on_breach="degrade",
    degrade_model="gpt-5.2-mini",
    anomaly_factor=3.0,            # >3x the expected burn rate → cost.anomaly event
)
```

On breach the `BudgetManager` returns a `BudgetDecision`
(`allow`/`cap`/`degrade`/`queue_to_batch`); a deny or degrade is a
`PolicyViolation` on the audit path under the `cost_budget` action, and
`cost.budget_exceeded` fires on the bus. From the shell:
`vincio cost report --by tenant|feature|user|model|provider|run [--db .vincio/vincio.db] [--json]`.

Attribution and budgets cover `ContextApp` runs — `run` / `arun` / `astream` /
`batch`, including their tool loops and self-correction. Lower-level
`app.agent()` / `app.crew()` handles still aggregate spend on `app.cost_tracker`
but do not carry per-run attribution dimensions.

## Provider-aware prompt caching

`app.enable_prompt_caching(ttl="5m")` caches the stable prefix of your prompt
(system instructions, fixed context, examples) so repeat runs only pay for the
changing tail:

```python
app.enable_prompt_caching(ttl="1h", min_prefix_tokens=1024)
```

The strategy is provider-aware: on Anthropic it emits `cache_control` breakpoints
on the prefix; on OpenAI and Gemini it orders messages so their automatic prefix
caching kicks in. Prefixes shorter than `min_prefix_tokens` aren't worth caching
and are left alone. It's default-on via config:

```yaml
# vincio.yaml
cache:
  provider_cache: true
  provider_cache_ttl: "5m"
  provider_cache_min_prefix_tokens: 1024
```

Cache hits are telemetry, not a feeling: the model span gains `cache_hit_rate`
and `cached_input_tokens`, and those `cached_input_tokens` flow into the cost
ledger so a cached call costs visibly less in `cost_report`.

## Incremental & sharded indexing

`LiveIndex.upsert` does content-hash change detection: re-embedding only the
chunks whose text actually changed and leaving the rest untouched:

```python
stats = await index.upsert(chunks)          # UpsertStats
print(stats.added, stats.updated, stats.unchanged, stats.reembedded)
```

For a live feed, `upsert_stream` consumes an async iterable in bounded batches:

```python
stats = await index.upsert_stream(change_feed(), batch_size=64, ttl_seconds=3600)
```

To scale past one backend, `ShardedIndex` fans a single `Index` across several.
A document's chunks co-locate by default (hashed `document_id`); search runs
every shard in parallel and merges the global `top_k`:

```python
from vincio.retrieval import ShardedIndex

index = ShardedIndex([shard_a, shard_b, shard_c], max_concurrency=8)
await index.add(chunks)                      # routed per document
hits = await index.search(query, top_k=10)   # fan-out + merge
```

Pass a custom `router=lambda chunk: ...` to shard by tenant, region, or any key
you choose.

## Edge over gateways

LLM gateways (LiteLLM, Bifrost, Portkey) give you failover, circuit breaking,
cascades, cost attribution, budgets, and batch — behind a separate proxy hop.
Vincio gives you the same, **in-process**: no extra network hop, governed by your
own policy engine, and on **one trace** with the rest of the run. Offline-first,
no vendor SDKs (core depends on `httpx` only).
