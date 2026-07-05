"""DS4 local inference — a self-hosted DeepSeek V4 box as a first-class provider.

`DS4 <https://github.com/antirez/ds4>`_ serves an OpenAI-compatible API on
``127.0.0.1:8000``; Vincio's :class:`Ds4Provider` makes it a first-class provider,
flowing through the same registry, cost table, reasoning controller, residency, and
audit chain as every hosted model. Point Vincio at your box with either of::

    export VINCIO_PROVIDER=ds4
    app = ContextApp(name="qa", provider="ds4", model="deepseek-v4-flash")

This tour runs **fully offline** with no DS4 binary: it replays a *recorded* DS4
exchange through the real provider via an injected transport, so you see exactly
what the live path does. Set ``VINCIO_PROVIDER=ds4`` to run against a real box.
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


# -- The offline mechanism: a recorded DS4 exchange replayed via an injected -------
# transport. Injecting these canned responses as the provider's httpx client
# exercises the *genuine* parse/stream/usage paths with no binary and no network —
# the same technique to unit-test any provider against a pinned server response.


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
    """A minimal httpx stand-in that replays a canned chat + SSE response."""

    def __init__(self, *, chat: dict, stream: list[str]) -> None:
        self.is_closed = False
        self._chat, self._stream = chat, stream

    async def post(self, url, *, headers=None, content=None, json=None):  # noqa: A002
        return _Recorded(self._chat)

    def stream(self, method, url, *, headers=None, content=None, json=None):  # noqa: A002
        return _RecordedStream(self._stream)

    async def aclose(self) -> None:
        self.is_closed = True


# What a real ds4-server returns. Note the thinking trace is a *separate* field
# (reasoning_content) and prompt_cache_hit_tokens reports DS4's disk-KV reuse.
CHAT_FIXTURE = {
    "id": "ds4-1", "model": MODEL,
    "choices": [{"message": {
        "content": "Bordeaux is in south-western France, on the Garonne.",
        "reasoning_content": "The user asks where Bordeaux is located...",  # must not leak
    }, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1200, "completion_tokens": 12,
              "prompt_cache_hit_tokens": 1024,
              "completion_tokens_details": {"reasoning_tokens": 48}},
}
STREAM_FIXTURE = [
    "data: " + json.dumps({"model": MODEL, "choices": [{"delta": {"content": "Bordeaux is "}}]}),
    "data: " + json.dumps({"model": MODEL, "choices": [{"delta": {"content": "in France."}}]}),
    "data: " + json.dumps({"model": MODEL, "choices": [{"delta": {}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 1200, "completion_tokens": 3,
                                     "prompt_cache_hit_tokens": 1024}}),
    "data: [DONE]",
]


def make_provider() -> Ds4Provider:
    """The real provider against a live box, or the recorded transport offline."""
    if os.environ.get("VINCIO_PROVIDER") == "ds4":
        provider = build_provider("ds4", with_retries=False)
        assert isinstance(provider, Ds4Provider)
        return provider
    return Ds4Provider(client=RecordedDs4Transport(chat=CHAT_FIXTURE, stream=STREAM_FIXTURE))


async def main() -> None:
    provider = make_provider()
    print(f"Provider {provider.name} @ {provider.base_url} · api key required: {provider.requires_api_key}")

    # 1. Chat. The provider parses out DeepSeek's separate thinking trace so it never
    #    leaks into the answer, reports reasoning_tokens, and — self-hosted — bills $0.
    reply = await provider.generate(ModelRequest(
        model=MODEL, reasoning_effort="medium",
        messages=[Message(role="system", content="Be terse."),
                  Message(role="user", content="Where is Bordeaux?")]))
    print(f"\n1. Chat → {reply.text!r}")
    print(f"   cost_usd={reply.cost_usd} reasoning_tokens={reply.usage.reasoning_tokens} "
          f"disk-KV hit rate={cache_hit_rate(reply.usage.input_tokens, reply.usage.cached_input_tokens):.2f}")

    # 2. Streaming: the same answer as incremental deltas, with usage on the final
    #    'done' event — the disk-KV cache stats survive the streaming path too.
    events = [e async for e in provider.stream(
        ModelRequest(model=MODEL, messages=[Message(role="user", content="Where is Bordeaux?")]))]
    streamed = "".join(e.text for e in events if e.type == "text_delta")
    done = [e for e in events if e.type == "done"][-1]
    print(f"\n2. Streaming → {streamed!r} (cached_input_tokens={done.response.usage.cached_input_tokens})")

    # 3. Thinking is driven by the shared reasoning controller, not DS4-specific
    #    flags: effort/budget turn thinking on; a plain request turns it off. Same
    #    knobs as every hosted reasoning model, so app code stays provider-agnostic.
    thinking = provider._payload(ModelRequest(
        model=MODEL, reasoning_effort="high", thinking_budget_tokens=4096,
        messages=[Message(role="user", content="prove it")]))
    plain = provider._payload(ModelRequest(model=MODEL, messages=[Message(role="user", content="hi")]))
    print(f"\n3. Thinking control — effort=high,budget=4096 → {thinking['chat_template_kwargs']}; "
          f"plain → {plain['chat_template_kwargs']}")

    # 4. Residency is fail-closed: a localhost box is on_prem by construction, so a
    #    policy that includes on_prem admits it and one that doesn't refuses egress.
    admitted = residency_violation(provider="ds4", model=MODEL, allowed_regions=["on_prem"]) is None
    refused = residency_violation(provider="ds4", model=MODEL, allowed_regions=["eu"])
    print(f"\n4. Residency — allowed=['on_prem'] admits={admitted}; "
          f"allowed=['eu'] refuses={refused is not None} ({refused.severity if refused else '-'})")

    # 5. The catalog carries the box honestly: self_hosted=True, $0 in/out, and the
    #    coverage gate stays green (a self-hosted $0 is legitimate, not a silent miss).
    registry = ModelRegistry()
    profile = registry.resolve(MODEL)
    print(f"\n5. Catalog — {profile.model}: self_hosted={profile.self_hosted} "
          f"${profile.input_cost_per_mtok}/${profile.output_cost_per_mtok} "
          f"coverage ok={registry.coverage_report().ok}")

    print("\nDone — a DS4 box is a first-class provider: thinking, KV cache, residency, honest $0.")


if __name__ == "__main__":
    import asyncio
    import sys

    from vincio.core.errors import ProviderUnavailableError

    try:
        asyncio.run(main())
    except ProviderUnavailableError as exc:
        # Only reachable in live mode (VINCIO_PROVIDER=ds4): no box answered.
        print(f"\nNo DS4 server reachable: {exc}", file=sys.stderr)
        print("Start one with `ds4-server`, or unset VINCIO_PROVIDER to run offline "
              "against the recorded transport.", file=sys.stderr)
        sys.exit(1)
