# Reasoning control & the Responses API

> One portable knob for thinking/reasoning across providers, with honest cost
> accounting.

Reasoning models expose a "think harder" control under different names. Vincio
gives you **one** provider-neutral knob:

```python
from vincio.core.types import RunConfig

result = app.run("How many r's are in strawberry?",
                 config=RunConfig(reasoning_effort="high"))   # low | medium | high | minimal
print(result.usage.reasoning_tokens)   # thinking tokens, recorded and billed
```

`reasoning_effort` maps per provider, and providers that don't expose reasoning
ignore it:

| Provider | Mapping |
|---|---|
| OpenAI reasoning models (o-series, GPT-5) | `reasoning_effort` (Chat Completions / Responses) |
| Anthropic (Opus/Sonnet, extended thinking) | a thinking `budget_tokens` derived from the effort; sampling is left at default while thinking is on |
| Gemini 2.5 / 3 | a `thinkingConfig.thinkingBudget` derived from the effort |

For explicit control, set `thinking_budget_tokens=` instead of an effort level.
Whether a model supports reasoning is provider-declared
(`provider.capabilities(model).reasoning`).

## Cost accounting

Thinking tokens are recorded on the `model_call` span (`reasoning_tokens`) and
**billed** at the output rate — including Gemini thinking tokens
(`thoughtsTokenCount`), which are counted as billable output, not costed at $0.

## OpenAI Responses API

An optional adapter targets OpenAI's stateful Responses API behind the same
`ModelProvider` interface — `previous_response_id` preserves reasoning across
tool calls without resending context. Chat Completions stays the portable
default.

```python
from vincio.providers import build_provider

provider = build_provider("openai_responses", api_key="…")
app = ContextApp(name="x", provider=provider, model="gpt-5.2")
```

See [`examples/11_advanced_context.py`](../../examples/11_advanced_context.py).
