"""DS4 local inference — a self-hosted DeepSeek V4 box as a first-class provider.

`DS4 <https://github.com/antirez/ds4>`_ is antirez's self-contained inference
engine for DeepSeek V4 (Flash / PRO). A running ``ds4-server`` serves an
OpenAI-compatible API on ``127.0.0.1:8000``, and Vincio's :class:`Ds4Provider`
makes it a first-class provider — flowing through the same registry, cost table,
reasoning controller, residency, and audit chain as every hosted model.

Point Vincio at your own box with either of:

    export VINCIO_PROVIDER=ds4                 # then run any example/app
    app = ContextApp(name="qa", provider="ds4", model="deepseek-v4-flash")

This tour runs **fully offline** — with no DS4 binary — by replaying a *recorded*
DS4 exchange through the real provider via an injected transport, so you can see
exactly what the live path does. Set ``VINCIO_PROVIDER=ds4`` to run it against a
real ds4-server instead.

Sections:
  1. Chat through the DS4 provider — the thinking trace stays out of the answer.
  2. Streaming — deltas plus disk-KV usage.
  3. Thinking modes ↔ the reasoning controller (on with a budget; off when plain).
  4. Residency — a localhost endpoint is on-prem, fail-closed.
  5. The catalog — self-hosted and honestly $0, coverage green.
"""

from __future__ import annotations

import json
import os

from vincio.core.types import Message, ModelRequest
from vincio.governance.residency import residency_violation
from vincio.providers import Ds4Provider, build_provider
from vincio.providers.cache_strategy import cache_hit_rate
from vincio.providers.registry import ModelRegistry

MODEL = "deepseek-v4-flash"


# -- a recorded DS4 exchange, replayed offline via an injected transport -------
# This is what a real ds4-server returns; injecting it as the provider's client
# exercises the genuine parse/stream/usage paths with no binary and no network.


class _Recorded:
    def __init__(self, payload: dict) -> None:
        self._payload, self.status_code, self.headers = payload, 200, {}
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class _RecordedStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines, self.status_code = lines, 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b""


class RecordedDs4Transport:
    """A minimal stand-in for httpx that replays a canned chat + SSE response."""

    def __init__(self, *, chat: dict, stream: list[str]) -> None:
        self.is_closed = False
        self._chat, self._stream = chat, stream

    async def post(self, url, *, headers=None, content=None, json=None):  # noqa: A002
        return _Recorded(self._chat)

    def stream(self, method, url, *, headers=None, content=None, json=None):  # noqa: A002
        return _RecordedStream(self._stream)

    async def aclose(self) -> None:
        self.is_closed = True


CHAT_FIXTURE = {
    "id": "ds4-1",
    "model": MODEL,
    "choices": [
        {
            "message": {
                "content": "Bordeaux is in south-western France, on the Garonne.",
                # DeepSeek returns its thinking separately; it must not leak.
                "reasoning_content": "The user asks where Bordeaux is located...",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 1200,
        "completion_tokens": 12,
        "prompt_cache_hit_tokens": 1024,  # DS4's disk-KV cache reuse
        "completion_tokens_details": {"reasoning_tokens": 48},
    },
}
STREAM_FIXTURE = [
    "data: " + json.dumps({"model": MODEL, "choices": [{"delta": {"content": "Bordeaux is "}}]}),
    "data: " + json.dumps({"model": MODEL, "choices": [{"delta": {"content": "in France."}}]}),
    "data: "
    + json.dumps(
        {
            "model": MODEL,
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 3, "prompt_cache_hit_tokens": 1024},
        }
    ),
    "data: [DONE]",
]


def make_provider() -> Ds4Provider:
    """The real provider against a live box, or a recorded transport offline."""
    if os.environ.get("VINCIO_PROVIDER") == "ds4":
        provider = build_provider("ds4", with_retries=False)
        assert isinstance(provider, Ds4Provider)
        return provider
    return Ds4Provider(client=RecordedDs4Transport(chat=CHAT_FIXTURE, stream=STREAM_FIXTURE))


async def main() -> None:
    provider = make_provider()
    print("Provider:", provider.name, "| endpoint:", provider.base_url, "| API key required:", provider.requires_api_key)

    # 1. Chat. The answer never carries DeepSeek's separate thinking trace, the
    #    provider is 'ds4', and a self-hosted box bills nothing.
    reply = await provider.generate(
        ModelRequest(
            model=MODEL,
            messages=[Message(role="system", content="Be terse."), Message(role="user", content="Where is Bordeaux?")],
            reasoning_effort="medium",
        )
    )
    print("\n1. Chat")
    print("   answer:", reply.text)
    print("   provider:", reply.provider, "| cost_usd:", reply.cost_usd, "| reasoning_tokens:", reply.usage.reasoning_tokens)
    print("   disk-KV cache hit rate:", round(cache_hit_rate(reply.usage.input_tokens, reply.usage.cached_input_tokens), 3))

    # 2. Streaming — the same answer, incrementally, with usage at the end.
    events = [
        e
        async for e in provider.stream(
            ModelRequest(model=MODEL, messages=[Message(role="user", content="Where is Bordeaux?")])
        )
    ]
    streamed = "".join(e.text for e in events if e.type == "text_delta")
    done = [e for e in events if e.type == "done"][-1]
    print("\n2. Streaming")
    print("   streamed:", streamed, "| cached_input_tokens:", done.response.usage.cached_input_tokens)

    # 3. Thinking modes are driven by the reasoning controller. Effort / budget
    #    turn DS4's thinking on; a plain request turns it off explicitly.
    thinking = provider._payload(
        ModelRequest(model=MODEL, messages=[Message(role="user", content="prove it")],
                     reasoning_effort="high", thinking_budget_tokens=4096)
    )
    plain = provider._payload(ModelRequest(model=MODEL, messages=[Message(role="user", content="hi")]))
    print("\n3. Thinking control")
    print("   with effort=high, budget=4096 →", thinking["chat_template_kwargs"])
    print("   plain request →", plain["chat_template_kwargs"])

    # 4. Residency: a localhost box is on-prem by construction. An in-jurisdiction
    #    policy that includes on_prem admits it; one that does not refuses egress.
    admitted = residency_violation(provider="ds4", model=MODEL, allowed_regions=["on_prem"]) is None
    refused = residency_violation(provider="ds4", model=MODEL, allowed_regions=["eu"])
    print("\n4. Residency (fail-closed)")
    print("   allowed_regions=['on_prem'] → admitted:", admitted)
    print("   allowed_regions=['eu']      → refused:", refused is not None, f"({refused.severity})" if refused else "")

    # 5. The catalog: self-hosted, honestly $0, and the coverage gate stays green.
    registry = ModelRegistry()
    profile = registry.resolve(MODEL)
    report = registry.coverage_report()
    print("\n5. Catalog")
    print(f"   {profile.model}: provider={profile.provider} self_hosted={profile.self_hosted} "
          f"in/out=${profile.input_cost_per_mtok}/${profile.output_cost_per_mtok} tier={profile.tier}")
    print("   registry coverage ok:", report.ok, "| presets priced/self-hosted:", report.presets_priced)

    print("\nDone — a DS4 box is a first-class provider: thinking, KV cache, residency, honest $0.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
