"""Reasoning & Responses surface: effort/thinking controls, cost accounting."""

from __future__ import annotations

import httpx
import pytest

from vincio import ContextApp
from vincio.core.types import ModelRequest, RunConfig, TokenUsage
from vincio.observability.costs import PriceTable
from vincio.providers import MockProvider, build_provider
from vincio.providers.anthropic import AnthropicProvider
from vincio.providers.google import GoogleProvider
from vincio.providers.openai import OpenAIProvider
from vincio.providers.openai_responses import OpenAIResponsesProvider


def _req(model: str, **kw) -> ModelRequest:
    from vincio.core.types import Message

    return ModelRequest(model=model, messages=[Message(role="user", content="hi")], **kw)


# -- capability detection -----------------------------------------------------


def test_reasoning_capability_detection():
    assert OpenAIProvider(api_key="x").capabilities("gpt-5.2").reasoning is True
    assert OpenAIProvider(api_key="x").capabilities("gpt-4o").reasoning is False
    assert AnthropicProvider(api_key="x").capabilities("claude-opus-4-8").reasoning is True
    assert AnthropicProvider(api_key="x").capabilities("claude-haiku-4-5").reasoning is False
    assert GoogleProvider(api_key="x").capabilities("gemini-2.5-pro").reasoning is True
    assert GoogleProvider(api_key="x").capabilities("gemini-2.0-flash").reasoning is False


# -- payload wiring -----------------------------------------------------------


def test_openai_payload_includes_reasoning_effort_only_when_supported():
    provider = OpenAIProvider(api_key="x")
    payload = provider._payload(_req("gpt-5.2", reasoning_effort="high"))
    assert payload["reasoning_effort"] == "high"
    # Non-reasoning model: omitted.
    assert "reasoning_effort" not in provider._payload(_req("gpt-4o", reasoning_effort="high"))
    # "minimal" downgrades to "low" on non-gpt-5 reasoning models.
    payload_o = provider._payload(_req("o3-mini", reasoning_effort="minimal"))
    assert payload_o["reasoning_effort"] == "low"


def test_anthropic_thinking_payload_drops_sampling_and_grows_max_tokens():
    provider = AnthropicProvider(api_key="x")
    payload = provider._payload(
        _req("claude-opus-4-8", reasoning_effort="high", temperature=0.7, max_output_tokens=2000)
    )
    assert payload["thinking"]["type"] == "enabled"
    assert payload["thinking"]["budget_tokens"] == 16384
    assert payload["max_tokens"] > 16384  # must exceed the thinking budget
    assert "temperature" not in payload  # only default sampling allowed with thinking


def test_google_thinking_config_payload():
    provider = GoogleProvider(api_key="x")
    payload = provider._payload(_req("gemini-2.5-pro", thinking_budget_tokens=5000))
    assert payload["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 5000


# -- cost accounting (the bug fix) --------------------------------------------


def test_google_folds_thinking_tokens_into_billable_output():
    """Regression: Gemini thinking tokens were recorded but excluded from
    billable output, so they were billed at $0."""
    provider = GoogleProvider(api_key="x")
    usage = provider._parse_usage(
        {"promptTokenCount": 100, "candidatesTokenCount": 40, "thoughtsTokenCount": 60}
    )
    assert usage.reasoning_tokens == 60
    assert usage.output_tokens == 100  # 40 visible + 60 thinking, both billable


def test_reasoning_tokens_are_billed_at_output_rate():
    table = PriceTable()
    table.set("gemini-2.5-pro", __import__("vincio.observability.costs", fromlist=["ModelPrice"]).ModelPrice(
        input_per_mtok=1.0, output_per_mtok=10.0
    ))
    no_think = TokenUsage(input_tokens=1000, output_tokens=1000)
    with_think = TokenUsage(input_tokens=1000, output_tokens=2000, reasoning_tokens=1000)
    cost_no = table.cost("gemini-2.5-pro", no_think)
    cost_yes = table.cost("gemini-2.5-pro", with_think)
    # Extra 1000 reasoning tokens cost an extra 1000 * $10/Mtok.
    assert round(cost_yes - cost_no, 8) == round(1000 * 10.0 / 1_000_000, 8)


# -- mock provider emulation --------------------------------------------------


def test_mock_provider_emulates_reasoning_tokens():
    provider = MockProvider(default_text="answer", reasoning=True)
    assert provider.capabilities("mock-1").reasoning is True
    resp = provider.generate_sync(_req("mock-1", reasoning_effort="high"))
    assert resp.usage.reasoning_tokens == 128


def test_mock_without_reasoning_ignores_effort():
    provider = MockProvider(default_text="answer")  # reasoning=False
    resp = provider.generate_sync(_req("mock-1", reasoning_effort="high"))
    assert resp.usage.reasoning_tokens == 0


# -- run-level wiring ---------------------------------------------------------


def test_run_records_reasoning_on_trace():
    from vincio.core.config import VincioConfig

    config = VincioConfig()
    config.observability.exporter = "memory"
    app = ContextApp(
        name="t",
        provider=MockProvider(default_text="ok", reasoning=True),
        model="mock-1",
        config=config,
    )
    result = app.run("explain", config=RunConfig(reasoning_effort="medium"))
    trace = app.tracer.exporter.get(result.trace_id)
    render = next(s for s in trace.spans if s.type == "prompt_render")
    assert render.attributes["reasoning"] == "medium"
    model_span = next(s for s in trace.spans if s.type == "model_call")
    assert model_span.attributes["reasoning_tokens"] == 64  # medium → 64


# -- OpenAI Responses adapter -------------------------------------------------


def test_responses_provider_payload_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        assert request.url.path.endswith("/responses")
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "gpt-5.2",
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "the answer"}],
                    },
                ],
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 30,
                    "output_tokens_details": {"reasoning_tokens": 12},
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIResponsesProvider(api_key="x", client=client)
    resp = provider.generate_sync(_req("gpt-5.2", reasoning_effort="high"))
    assert resp.text == "the answer"
    assert resp.usage.reasoning_tokens == 12
    assert seen["payload"]["reasoning"] == {"effort": "high"}
    assert "input" in seen["payload"]  # Responses shape, not chat "messages"


def test_responses_provider_registered():
    provider = build_provider("openai_responses", with_retries=False, api_key="x")
    assert provider.name == "openai_responses"


@pytest.mark.asyncio
async def test_responses_provider_tool_call_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_2",
                "model": "gpt-5.2",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": '{"q": "x"}',
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    provider = OpenAIResponsesProvider(
        api_key="x", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    resp = await provider.generate(_req("gpt-5.2"))
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls[0].name == "lookup"
    assert resp.tool_calls[0].arguments == {"q": "x"}
