# Cost, reliability & scale

Vincio runs the FinOps and resilience layer in-process: batch at half cost,
compose retries / failover / circuit-breaking, pool keys across regions,
cascade cheap→strong by confidence, attribute and budget every dollar, cache
provider prefixes, and index incrementally or sharded, all on the same trace,
audit log, and policy engine as the rest of the run. The examples run offline.

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

The same app runs sync (`app.run`) or batch (`app.batch`), switch by calling
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

A `BatchResult` carries `custom_id`, `response`, `error`, and `.ok`, one
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

- **Transient** (a timeout, a 503), retry it: `RetryingProvider`.
- **Persistent** (one key/region is down), fail over: `HealthAwareFailover`.
- **Systemic** (a provider is melting down), stop hammering it: `CircuitBreaker`.

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
to a stronger one when confidence is low, paying for the big model only on the
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

Set enforced budgets with an automatic action on breach, cap (deny), degrade
to a cheaper model, or queue the work to batch, and an anomaly factor that
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

Attribution and budgets cover the whole runtime, `run` / `arun` / `astream` /
`batch` (including tool loops and self-correction) **and** the `app.agent()` /
`app.crew()` handles. Pass `tenant_id` / `user_id` / `feature` to a handle's
`run`/`arun` and every agent or crew (manager + member) model call is attributed
on the same ledger:

```python
app.agent(tools=[...]).run("research the refund policy", tenant_id="acme", feature="research")
app.crew(members=[...]).run("draft the report", tenant_id="acme", feature="report")
```

A response-cache hit costs nothing; it is billed `$0` and recorded as a free
event, so `cost_report` reflects real spend, not what an uncached run would have
cost.

## Energy & carbon accounting

The cost report makes a run's dollar spend an auditable number; the same surface
also reports a run's **energy** (watt-hours) and estimated **carbon** (grams
CO₂e), the disclosure sustainability-reporting regimes increasingly require. It
is **opt-in** and additive: until you enable it, `result.energy_wh` and
`result.co2e_grams` stay `0.0`.

```python
app.use_energy_accounting(region="eu")     # estimate carbon against the EU grid

result = app.run("summarize the quarterly report")
print(result.energy_wh, result.co2e_grams)  # this run's footprint

app.energy_report(by="model").print_summary()
```

The estimate is **mechanical and offline**, no external service. Each run accrues
energy from its own token accounting against a per-model intensity (watt-hours per
million tokens, seeded from the `ModelRegistry` by tier, decode dominates prefill,
a stronger tier draws more), scaled by a datacenter power-overhead factor (`pue`);
carbon is that energy at a per-region grid factor (g CO₂e/kWh) from a built-in
table. `region=` pins the deployment region (an operator knows where their
inference runs); otherwise the residency policy's resolved region is used, then a
world-average fallback. Override the model intensity, the grid factors, or the PUE
for a measured deployment:

```python
table = app.cost_tracker.energy_table
table.set("my-model", EnergyProfile(wh_per_input_mtok=40, wh_per_output_mtok=400))
app.use_energy_accounting(region="on_prem", carbon_intensity={"on_prem": 12.0}, pue=1.05)
```

`energy_report(by=...)` rolls up from the *same* attributed events as `cost_report`
(by tenant / feature / user / model / provider / run), so a run's energy is
attributed exactly where its dollars are.

**Budget it like a dollar.** An energy or carbon envelope refuses a run that would
exceed it, the way a hard cost cap refuses spend:

```python
app.set_energy_budget(scope="tenant", id="acme", limit_co2e_grams=500.0, period="day")
app.set_energy_budget(limit_wh=1000.0, period="hour")   # an energy ceiling, globally
```

When a scope's accrued energy or carbon over the period reaches the ceiling, the
run is **denied**, an `energy_budget` decision on the audit chain and an
`energy.budget_exceeded` event on the bus, exactly parallel to `cost_budget`. Both
the per-run estimate and every refusal land on the hash-chained, verifiable audit
log, so the sustainability figure an auditor sees is a number, not a claim.

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

## Provider/model rotation & swap regression

A model swap is the most common and the riskiest change in production. Vincio
makes it a gated, statistically-backed discipline rather than a hope.

### Capability-aware routing

Before any substitution, Vincio intersects what the *request* needs (vision,
tool calling, structured output, reasoning, a wide enough context window) with
what a candidate model *can do*, read from the `ModelRegistry`. A registry-backed
router picks the cheapest / fastest / least-busy **capable** model per request,
and can downgrade to honor a per-request budget:

```python
app.use_router(["gpt-5.2-nano", "gpt-5.2-mini", "gpt-5.2"], strategy="cheapest")
result = app.run("classify this ticket")   # routed to the cheapest capable model
```

`FailoverChain` and `HealthAwareFailover` guard capabilities by default: a model
that cannot serve the request is **skipped** (not silently returning a wrong
answer), a retired model raises `ModelRetiredError` ("rotate now"), and a
terminal lifecycle/config error (a removed/unknown model) is classified
distinctly from a transient outage. Unknown models are never blocked; pass
`guard_capabilities=False` to restore the previous attempt-everything behavior.

When a [residency policy](governance.md) is configured, every model a run can
reach, the router's candidates, a cascade's rungs, a budget-degrade target, and
a shadow/canary candidate, is residency-checked at the run boundary, so a
rotation can never egress to a disallowed region.

### The swap gate

`app.gate_swap(...)` replays golden traces and runs an eval + cost + latency +
behavioral diff with statistical significance, returning a PASS/FAIL verdict. A
model is promoted into the live path only if it clears the gate:

```python
verdict = app.gate_swap("gpt-5.2-mini", baseline_model="gpt-5.2",
                        dataset=golden, traces=captured_traces)
if verdict.passed:
    app.model = "gpt-5.2-mini"
```

`vincio providers regress --app app.py --candidate-model gpt-5.2-mini --dataset golden.jsonl`
runs the same gate from the CLI (exit code 1 on FAIL).

### Model-swap regression (is the cheaper model safe?)

`app.swap_regression(...)` holds prompt, data, and config fixed, swaps **only**
the model, and reports per-metric significance, per-case deltas, the cost/latency
trade, and the worst-regressed slices. `repeats=N` runs each case N times for a
per-case mean/stdev, and flake quarantine excludes noisy cases from the gate so
non-mock provider variance never flips it on a single run:

```python
report = app.swap_regression(golden, candidate_model="gpt-5.2-nano",
                             baseline_model="gpt-5.2", repeats=3)
report.regressed          # True if a quality metric regressed significantly
report.cost["ratio"]      # candidate / baseline cost
report.worst_slices       # the slices that regressed most
```

CLI: `vincio eval regress golden.jsonl --app app.py --baseline-model gpt-5.2
--candidate-model gpt-5.2-nano --repeats 3`.

### Shadow & canary with auto-rollback

Qualify a candidate on live traffic without touching the user. A `ShadowProvider`
returns the primary's response while asynchronously dual-dispatching the
candidate for offline diff; a `CanaryRouter` ramps a percentage of traffic onto
the candidate, scores both arms online, and **auto-rolls-back** to the last
known-good model (and prompt-registry head) on regression:

```python
shadow = app.shadow("gpt-5.2-mini")          # users still get the primary
canary = app.canary("gpt-5.2-mini", percent=5.0, regression_threshold=0.05)
```

Both implement `ModelProvider`, so they nest inside `CircuitBreaker` / `KeyPool`.

### Lifecycle watcher

`app.watch_lifecycle()` reads the registry's deprecation/retirement dates and
emits early sunset warnings, then proposes a migration, to a model's declared
successor or a cheaper, at-least-as-capable Pareto-dominating model, that can
rewrite a `ModelCascade`, `RoutingPolicy`, or `config.model` in place:

```python
result = app.watch_lifecycle()
for proposal in result["proposals"]:
    print(proposal.from_model, "→", proposal.to_model, proposal.kind)
```

CLI: `vincio providers lifecycle --app app.py` and `vincio providers list`.

### Live discovery & Google/Vertex batch parity

`vincio providers discover <provider>` reconciles a provider's live model list
into the registry (offline-safe, the shipped catalog stands when no endpoint is
available). A `GoogleBatchBackend` completes half-cost batch parity with
OpenAI/Anthropic.

### Honest pricing: the catalog coverage gate

The registry's built-in catalog ships as reviewable data
(`vincio/providers/model_catalog.json`) and prices the **real current lineup of
every provider** — the OpenAI o-series / `gpt-5` / `gpt-4.1` families and
`text-embedding-3-*`, the Anthropic 3.x tier beside 4.x / Fable, Mistral
medium / codestral / pixtral / `mistral-embed`, the `openai_compat` presets, and
Google reconciled to live reality — so a current model never resolves to nothing
and silently bills $0. Each profile carries a `priced_as_of`, and freshness is
held against an `as_of`-deterministic horizon measured from the catalog's
**release date**, not the wall clock, so a frozen release reports the same
verdict forever.

```python
from vincio import default_model_registry

report = default_model_registry().coverage_report()
assert report.ok          # complete, honest ($0-free), fresh, routing-stable
```

`vincio registry coverage` runs the same drift detector from the shell (exit
non-zero on a gap), and `vincio registry sync <provider>` is a **review-only**
helper that diffs a provider's live `list_models()` into a candidate overlay for
you to price and merge — it never mutates the shipped catalog. An arbitrary model
id the catalog does not cover still warns once via `ModelUnknownWarning` rather
than billing $0.

## Edge over gateways

LLM gateways (LiteLLM, Bifrost, Portkey) give you failover, circuit breaking,
cascades, cost attribution, budgets, and batch, behind a separate proxy hop, and
they route by health, not by **capability**. Vincio gives you the same in-process
(no extra network hop, governed by your own policy engine, on **one trace**), and
adds what a proxy structurally cannot: it refuses a capability-mismatched
substitution, **gates** every swap on replayed golden traces with statistical
significance, and qualifies a candidate on live shadow/canary traffic with
automatic rollback. Offline-first, no vendor SDKs (core depends on `httpx` only).
