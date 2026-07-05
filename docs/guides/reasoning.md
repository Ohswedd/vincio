# Reasoning control & the Responses API

> One portable knob for thinking/reasoning across providers, with honest cost
> accounting.

Reasoning models expose a "think harder" control under different names — OpenAI
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

- **Effort is ignored on non-reasoning models** — silently, by design. Gate on
  `capabilities(model).reasoning` if you need to *know* it took effect.
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

- [Example: 11_advanced_context.py](../../examples/11_advanced_context.py)
- [Concept: Prompt compiler](../concepts/prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
