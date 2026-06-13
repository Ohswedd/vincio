"""Unified reasoning control across providers, with honest cost accounting (1.1).

One portable knob — ``reasoning_effort`` — maps to OpenAI reasoning effort,
Anthropic extended thinking, and Gemini thinking budgets; providers that don't
expose reasoning ignore it. Thinking tokens are recorded on the model span and
**billed** (fixing a latent gap where Gemini thinking tokens were counted but
not costed). An optional OpenAI Responses API adapter preserves reasoning across
tool calls.

Runs fully offline with a reasoning-capable mock provider.
"""

from __future__ import annotations

from vincio import ContextApp
from vincio.core.types import RunConfig
from vincio.providers import MockProvider, build_provider


def main() -> None:
    # A reasoning-capable provider (offline mock emulates thinking tokens).
    app = ContextApp(
        name="reasoning_demo",
        provider=MockProvider(default_text="42", reasoning=True),
        model="mock-1",
    )

    for effort in ("low", "high"):
        result = app.run("How many r's are in strawberry?", config=RunConfig(reasoning_effort=effort))
        print(
            f"effort={effort:<4} reasoning_tokens={result.usage.reasoning_tokens:<4} "
            f"cost=${result.cost_usd:.6f}"
        )

    # Which models expose reasoning? (capabilities are provider-declared)
    print("\nreasoning capability by model:")
    for name, model in (("openai", "gpt-5.2"), ("anthropic", "claude-opus-4-8"), ("google", "gemini-2.5-pro")):
        provider = build_provider(name, with_retries=False, api_key="demo")
        print(f"  {model:<20} reasoning={provider.capabilities(model).reasoning}")

    # The Responses API adapter (server-state, reasoning preserved across tools)
    # is available behind the same interface; Chat Completions stays the default.
    print("\nResponses API provider:", build_provider("openai_responses", with_retries=False, api_key="demo").name)


if __name__ == "__main__":
    main()
