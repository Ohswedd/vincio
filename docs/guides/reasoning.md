# Universal reasoning, native thinking & the Responses API

> One adaptive reasoning architecture for every model, plus a portable native
> thinking knob and honest cost accounting.

## Universal reasoning for every model

`reasoning_effort` only affects models that expose a native thinking control. To
give the full task-understand → decompose → gather evidence → solve → verify →
correct → synthesize flow to **every** model, install the universal engine:

```python
app.use_reasoning_engine(web=True)

easy = app.run("Rewrite this title")       # direct, exactly one model pass
hard = app.run("Compare the latest releases, verify the claims, and recommend one")
receipt = hard.metadata["universal_reasoning"]
print(receipt["depth"], receipt["strategy"], receipt["passes"], receipt["web_verified"])
```

Routing is hybrid. Exact syntax and confidently detected English use the
token-free deterministic classifier. Non-English and uncertain-language input
uses one compact structured call to the configured model, so every language the
model understands receives native task, depth, web and tool classification
without Vincio maintaining a locale allow-list. The classification call is
traced, DLP-screened and included in returned token/cost totals. Its output can
only select among registered capabilities; deterministic policy remains the
authority.

```python
from vincio import UniversalReasoningPolicy

app.use_reasoning_engine(
    policy=UniversalReasoningPolicy(
        semantic_routing="auto",  # auto | off | always
        semantic_routing_confidence=0.55,
    )
)
result = app.run("ウェブを使わずに、現在のCEOを教えてください。")
receipt = result.metadata["universal_reasoning"]
print(receipt["detected_language"], receipt["semantic_routing_succeeded"])
```

Low-confidence or malformed semantic routing falls back to standard depth and
does not gain web or tool access. A one-step budget skips the routing probe so
the actual answer always retains a model-call slot. Deep tasks use bounded
independent candidates and correction only when verification or material
disagreement justifies it.

Transient upstreams are handled in two layers. Provider adapters mark an
HTTP 200 whose payload carries no completion as retryable, so the standard
retry wrapper absorbs one-off faults within milliseconds; if *every* pass in a
reasoning run still dies before producing an answer, the engine spends its
reserved correction slot on one salvage attempt spaced further out
(`salvage_transient_failures=True`, `salvage_backoff_ms=1500`), recorded as a
`salvage` pass and as `receipt["salvaged"]`. A persistently rate-limited
upstream remains beyond client-side repair: the run then fails honestly rather
than fabricating an answer.

A deep, genuinely multi-step request additionally triggers the internal plan
mode (`plan_mode="auto"`, or `"off"`/`"always"`): one bounded planning call
returns typed, dependency-ordered `PlannedStep`s — goal, kind, and the
deterministic check each step's output should survive — that structure every
candidate pass. The plan can shape prompts and evidence queries only; it can
never open the web the deterministic policy declined, and an invalid or
low-confidence plan falls back to the heuristic decomposition. Look for
`plan_mode_used`, `plan_steps` and `plan_tokens` in the receipt. Verification
also refuses fabricated grounding: a URL or "according to …" domain found in
neither the attached evidence nor the request refutes the answer, the flagged
sources land in `receipt["fabricated_sources"]`, and each refutation's reason
is recorded in `receipt["refutation_notes"]`.

Browser need is a separate decision with four observable outcomes:
`not_needed`, `search`, `disabled`, or `user_declined`. Explicit searches,
requested URLs, unstable facts and high-stakes questions select governed web
evidence. Local uses such as “current paragraph”, “version control”, and “score
this essay” remain offline. A requested URL is read directly; search queries
strip request mechanics, prefer source-domain diversity, and are never repeated
inside candidate calls. `web_verified` proves the retrieved bytes still match
their snapshots; `answer_verification` reports whether the chosen answer passed
the reasoning/evidence checks.

For supported numerical and logical shapes, a task-bound kernel recomputes the
expected fact from the request itself and places it in `plan.verified_facts`.
This closes a gap ordinary self-critique cannot: an equality can be internally
valid yet answer the wrong interpretation. If all model attempts remain refuted,
Vincio emits only a complete deterministic fallback when those facts prove one;
otherwise it refuses the answer.
Use `UniversalReasoningPolicy(web="required")` when unverifiable live claims
must fail closed. With the default `auto`, missing or declined web access permits
an explicit uncertainty response but refutes and withholds an unsupported
current assertion. `web="off"` disables egress; it does not make unstable facts
safe to assert.
For unavailable multilingual live evidence, the candidate must prefix calibrated
uncertainty with `[UNVERIFIED]`; this marker is language-independent and
deterministically verified.

Call `app.reason(...)` instead when you want the full `UniversalReasoningResult`
for one request without changing later runs. Its plan and pass records contain
operational decisions, validation status, tokens and cost—never model scratch
work or chain-of-thought. See the [universal reasoning concept](../concepts/universal-reasoning.md).

Native reasoning models expose a "think harder" control under different names — OpenAI
calls it `reasoning_effort`, Anthropic a thinking `budget_tokens`, Gemini a
`thinkingBudget`. Wire your app to one of them and you have coupled it to that
provider. Vincio gives you **one** provider-neutral knob and lowers it to
whatever the target model speaks:

```python
from vincio import ContextApp
from vincio.core.types import RunConfig

app = ContextApp(name="assistant")

result = app.run("How many r's are in strawberry?",
                 config=RunConfig(reasoning_effort="high"))   # minimal | low | medium | high
print(result.usage.reasoning_tokens)   # thinking tokens, recorded and billed
```

`reasoning_effort` is a four-level ordinal ladder (`minimal → low → medium →
high`). Providers that don't expose reasoning ignore it entirely — the same
code runs on a non-reasoning model with no branch on your side.

## How the effort knob maps

The effort level is translated once, in the provider layer, and providers that
lack a native effort dial receive a **thinking-token budget** derived from it
(`reasoning_budget_from_effort`: `minimal → 1024`, `low → 4096`, `medium →
8192`, `high → 16384`):

| Provider | Mapping |
|---|---|
| OpenAI reasoning models (o-series, GPT-5) | native `reasoning_effort` (Chat Completions / Responses) |
| Anthropic (Opus/Sonnet, extended thinking) | a thinking `budget_tokens` derived from the effort; sampling is left at default while thinking is on |
| Gemini 2.5 / 3 | a `thinkingConfig.thinkingBudget` derived from the effort |

For explicit control, set `thinking_budget_tokens=` on the `RunConfig` instead
of an effort level — it wins over `reasoning_effort` and pins the exact budget.
Whether a model supports reasoning at all is provider-declared
(`provider.capabilities(model).reasoning`), so you can gate on it before
spending:

```python
if provider.capabilities(model).reasoning:      # the provider + model you built
    config = RunConfig(reasoning_effort="high")
```

## Let the platform pick the effort: the ReasoningController

A fixed effort level overpays on the easy steps and starves the hard ones. The
thinking budget is the cheapest quality lever left, so it should be a *policy*
over signals the platform already computes, not a constant. `app.reasoning()`
builds a [`ReasoningController`](../reference/api.md) that turns the task
classification, a deterministic difficulty estimate (the same estimator the
capability-aware router uses), and the live remaining budget into a
`ReasoningDecision` — effort, thinking budget, and a human-readable `reason`:

```python
from vincio.core.types import RunConfig

ctl = app.reasoning()                       # default ReasoningPolicy
d = ctl.decide(text="Prove the pigeonhole principle for n+1 items.",
               remaining_output_tokens=4096)
print(d.effort, d.thinking_budget_tokens, d.reason)
#   high  2048  "difficulty 0.71 → high; capped at 50% of remaining output budget (2048 tokens)"

app.run("...", config=RunConfig(reasoning_effort=d.effort))
```

Install it once and every run that does **not** pin its own `reasoning_effort`
has the effort chosen for it (only on reasoning-capable models):

```python
app.use_reasoning_controller()              # or pass a ReasoningPolicy / controller
report = app.run("summarize this invoice")  # easy → minimal effort, cheap
report = app.run("reconcile these three ledgers and explain the discrepancy")  # hard → escalated
```

### How the decision is made

`ReasoningPolicy` sets difficulty **bands** and two hard **guardrails**:

- **Difficulty banding.** Difficulty in `[0, 1]` maps to a base effort:
  `< low_effort_below` (0.3) is easy (`minimal`), `> high_effort_above` (0.65)
  is hard (`high`), in between is `medium`. `min_effort` / `max_effort` clamp
  the range.
- **Low-confidence escalation.** If a prior attempt's `confidence` is below
  `quality_floor` (0.6), the controller steps effort up one rung — a retry
  thinks harder, deterministically.
- **Warm-prefix step-down.** When a `ReasoningTraceCache` shows the call's
  stable thinking prefix was already reasoned through, effort steps *down* one
  rung: the expensive part was already paid for. Record each paid prefix with
  `ctl.record_trace(prefix_hash=..., model=..., reasoning_tokens=...)`.
- **Hard ceiling + budget share.** The thinking budget is clamped to
  `max_reasoning_tokens` (16384, held by an SLO) *and* to `budget_fraction`
  (0.5) of the remaining output budget, whichever is smaller — reasoning can
  never silently crowd out the answer.

Every `ReasoningDecision` (`difficulty`, `escalated`, `warm_prefix`,
`ceiling_capped`, `budget_capped`, `reason`) is stamped on the trace, so the
choice is always explained, never silent. The decision is deterministic given
its inputs, so a seeded run reproduces exactly.

## Cost accounting

Thinking tokens are recorded on the `model_call` span (`reasoning_tokens`) and
**billed** at the output rate, including Gemini thinking tokens
(`thoughtsTokenCount`), which are folded into billable output, not costed at $0.
This is the honest tradeoff: higher effort buys quality with output-priced
tokens you may never see in the reply. Budget for reasoning like output, watch
`result.usage.reasoning_tokens`, and prefer the controller so easy steps stop
paying the hard-step price.

## When to spend, when not

- **Use higher effort when** the step is genuinely hard: multi-hop reasoning,
  proofs, planning, ambiguous extraction, code with tricky invariants — the
  places where a wrong answer costs more than the extra tokens.
- **Avoid it when** the task is retrieval-bound, formatting, classification, or
  simple summarization: thinking tokens there are pure cost with no quality
  lift. Let the controller band these down to `minimal`.
- **Prefer the controller over a global `high`.** A blanket high effort is the
  most common way to double a bill for no measurable win; adaptive effort spends
  where difficulty is, and the trace shows you where that was.

## Gotchas

- **The native effort knob is ignored on non-reasoning models.** Use
  `app.use_reasoning_engine()` when those models need actual provider-independent
  decomposition, verification and correction rather than an ignored knob.
- **`thinking_budget_tokens` overrides `reasoning_effort`.** Set one or the
  other; if you set both, the explicit budget wins.
- **The controller only fills an *unset* effort.** A run that pins
  `reasoning_effort` on its `RunConfig` bypasses the controller entirely — that
  is the escape hatch, not a bug.
- **Warm-prefix reuse needs the cache fed.** Step-down only fires if you called
  `record_trace(...)` for that prefix/model earlier; a fresh cache never steps
  down.

## OpenAI Responses API

An optional adapter targets OpenAI's stateful Responses API behind the same
`ModelProvider` interface, `previous_response_id` preserves reasoning across
tool calls without resending context. Chat Completions stays the portable
default.

```python
from vincio import ContextApp
from vincio.providers import build_provider

provider = build_provider("openai_responses", api_key="…")
app = ContextApp(name="x", provider=provider, model="gpt-5.2")
```

See [`examples/11_advanced_context.py`](../../examples/11_advanced_context.py).

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Universal reasoning](../concepts/universal-reasoning.md)
- [Concept: The Big Brain](../concepts/big-brain.md)
- [Example: 22_universal_reasoning.py](../../examples/22_universal_reasoning.py)
- [Example: 11_advanced_context.py](../../examples/11_advanced_context.py)
- [Concept: Prompt compiler](../concepts/prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
