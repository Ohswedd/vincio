"""Native-grade tool use for models without native function calling.

Vincio's tool loop (:meth:`RunEngine._model_tool_loop`) is driven entirely by
``ModelResponse.tool_calls`` — which means a model whose provider cannot render
tools natively (an in-process GGUF model, a bare completion endpoint, a gateway
that dropped the ``tools`` field) could never call *any* tool. This module
closes that gap by composition, the same way :class:`~vincio.providers.base.RetryingProvider`
adds retries: wrap any provider in :class:`ToolProtocolProvider` and every
model behind it becomes tool-capable.

When a request carries tools and the wrapped model does not claim
``tool_calling`` (or ``force=True``), the wrapper *lowers* the request — the
tool schemas become a compact protocol block in the system message, prior
tool-call turns are folded back into plain text — sends it, then *lifts* the
reply: fenced ``tool_call`` JSON blocks are parsed into ordinary
:class:`~vincio.core.types.ToolCallRequest` rows and ``finish_reason`` becomes
``tool_calls``. The runtime, registry, permissions, budgets, and audit chain
are untouched — they see exactly what a native provider would have produced.

This mirrors how structured output already degrades gracefully
(``DecodingMode.PROMPT``): capability emulated in the prompt when the wire
protocol lacks it, invisible above the provider seam.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

from ..core.types import (
    Message,
    ModelCapabilities,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolSpec,
)
from .base import ModelProvider

__all__ = ["ToolProtocolProvider", "lower_tool_protocol", "lift_tool_calls"]

_FENCE_RE = re.compile(r"```tool_call\s*\n(.*?)```", re.DOTALL)

_PROTOCOL_HEADER = """\
# Tool protocol
You can call the tools listed below. To call one, emit a fenced block:

```tool_call
{{"name": "<tool_name>", "arguments": {{...}}}}
```

Rules: emit at most {max_calls} tool_call block(s) per reply, and no other text
when calling tools. Each result arrives in the next message as a
[tool_result name=... id=...] block. When you have what you need, reply with
your final answer and no tool_call block.

Available tools:
{tool_cards}"""


def _tool_card(spec: ToolSpec) -> str:
    schema = spec.input_schema or {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    arguments = ", ".join(
        f"{name}: {definition.get('type', 'any')}"
        + ("" if name in required else "?")
        + (f" — {definition['description']}" if definition.get("description") else "")
        for name, definition in properties.items()
    )
    return f"- {spec.name}({arguments}): {spec.description}"


def lower_tool_protocol(request: ModelRequest, *, max_calls: int = 4) -> ModelRequest:
    """Rewrite *request* so a tools-blind model can serve it.

    The tool schemas become a protocol block appended to the system message
    (created if absent); assistant tool-call turns are re-rendered as the
    fenced blocks the model would have emitted; ``tool`` results become
    ``[tool_result ...]`` user turns; adjacent same-role messages merge so the
    transcript stays alternation-safe. ``tools`` is emptied.
    """
    protocol = _PROTOCOL_HEADER.format(
        max_calls=max_calls,
        tool_cards="\n".join(_tool_card(spec) for spec in request.tools),
    )
    folded: list[Message] = []
    for message in request.messages:
        if message.role == "tool":
            body = message.content if isinstance(message.content, str) else ""
            folded.append(
                Message(
                    role="user",
                    content=(
                        f"[tool_result name={message.name or ''} "
                        f"id={message.tool_call_id or ''}]\n{body}"
                    ),
                )
            )
        elif message.role == "assistant" and message.tool_calls:
            blocks = "\n".join(
                "```tool_call\n"
                + json.dumps({"name": call.name, "arguments": call.arguments}, sort_keys=True)
                + "\n```"
                for call in message.tool_calls
            )
            text = message.content if isinstance(message.content, str) else ""
            folded.append(
                Message(role="assistant", content=(text + "\n" + blocks).strip())
            )
        else:
            folded.append(message.model_copy())
    if folded and folded[0].role == "system":
        first = folded[0]
        body = first.content if isinstance(first.content, str) else ""
        folded[0] = first.model_copy(update={"content": body + "\n\n" + protocol})
    else:
        folded.insert(0, Message(role="system", content=protocol))
    merged: list[Message] = []
    for message in folded:
        if (
            merged
            and message.role == merged[-1].role
            and isinstance(message.content, str)
            and isinstance(merged[-1].content, str)
            and not message.tool_calls
            and not merged[-1].tool_calls
        ):
            merged[-1] = merged[-1].model_copy(
                update={"content": merged[-1].content + "\n\n" + message.content}
            )
        else:
            merged.append(message)
    return request.model_copy(update={"messages": merged, "tools": []})


def lift_tool_calls(text: str, *, max_calls: int = 4) -> tuple[str, list[ToolCallRequest]]:
    """Parse fenced ``tool_call`` blocks out of *text*.

    Returns the text with parsed blocks removed, and the calls in order
    (capped at *max_calls*). A block whose JSON does not parse — or parses to
    something other than ``{"name": str, ...}`` — is left in the text, visibly,
    rather than silently dropped: the model (or a human reading the trace) can
    see and repair it.
    """
    calls: list[ToolCallRequest] = []
    spans: list[tuple[int, int]] = []
    for match in _FENCE_RE.finditer(text):
        if len(calls) >= max_calls:
            break
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            continue
        arguments = payload.get("arguments")
        calls.append(
            ToolCallRequest(
                name=payload["name"],
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
        spans.append(match.span())
    for start, end in reversed(spans):
        text = text[:start] + text[end:]
    return text.strip(), calls


class ToolProtocolProvider(ModelProvider):
    """Composition wrapper: every model behind it becomes tool-capable.

    Engages only when the request carries tools **and** the wrapped model does
    not claim native ``tool_calling`` (unknown models default to not claiming
    it); a natively capable model passes through byte-untouched. ``force=True``
    applies the protocol unconditionally — useful when a gateway advertises
    tool calling it does not honor ("some that do, do it poorly").
    """

    name = "tool_protocol"

    def __init__(
        self, inner: ModelProvider, *, force: bool = False, max_calls_per_turn: int = 4
    ) -> None:
        self.inner = inner
        self.force = force
        self.max_calls_per_turn = max_calls_per_turn

    def _engages(self, request: ModelRequest) -> bool:
        if not request.tools:
            return False
        if self.force:
            return True
        return not self.inner.capabilities(request.model).tool_calling

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if not self._engages(request):
            return await self.inner.generate(request)
        lowered = lower_tool_protocol(request, max_calls=self.max_calls_per_turn)
        response = await self.inner.generate(lowered)
        text, calls = lift_tool_calls(response.text, max_calls=self.max_calls_per_turn)
        if not calls:
            return response
        return response.model_copy(
            update={"text": text, "tool_calls": calls, "finish_reason": "tool_calls"}
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        if not self._engages(request):
            async for event in self.inner.stream(request):
                yield event
            return
        # Protocol turns cannot stream text deltas faithfully (the fenced
        # blocks must be lifted whole), so emulate from generate().
        response = await self.generate(request)
        if response.text:
            yield ModelEvent(type="text_delta", text=response.text)
        for tool_call in response.tool_calls:
            yield ModelEvent(type="tool_call_delta", tool_call=tool_call)
        yield ModelEvent(type="usage", usage=response.usage)
        yield ModelEvent(type="done", response=response)

    def capabilities(self, model: str) -> ModelCapabilities:
        """The wrapped model's matrix, with ``tool_calling`` claimed: that is
        the wrapper's whole point, and it keeps the capability guard admitting
        tool-carrying requests routed at wrapped models."""
        return self.inner.capabilities(model).model_copy(update={"tool_calling": True})

    def exact_token_counter(self, model: str):  # noqa: ANN201 - mirrors base signature
        return self.inner.exact_token_counter(model)

    def token_id_prefixes(self) -> tuple[str, ...]:
        return self.inner.token_id_prefixes()

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return await self.inner.embed(texts, model)

    async def list_models(self):  # noqa: ANN201 - mirrors base signature
        return await self.inner.list_models()

    async def aclose(self) -> None:
        await self.inner.aclose()
