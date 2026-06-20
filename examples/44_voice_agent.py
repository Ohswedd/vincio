"""End-to-end voice agent: a spoken assistant with the full stack behind it.

``app.voice_agent(...)`` wires a realtime session to the deep-research agent (an
in-session ``research`` tool that answers from your sources with citations), the
self-editing memory OS, and the app's deterministic input/output rails — so a
spoken turn inherits the same grounding, budget, and audit guarantees as the text
path. Tool calls route through the permissioned, sandboxed, audited runtime.

Runs offline on the dependency-free in-process backend; point it at OpenAI
Realtime or Gemini Live with ``backend="openai"`` / ``"gemini"`` and
``pip install "vincio[realtime]"``.
"""

import asyncio

from _shared import example_provider

from vincio import ContextApp
from vincio.core.types import Document
from vincio.realtime import RealtimeEvent, RealtimeToolCall

provider, model = example_provider(lambda r: "The refund window is 30 days for Pro customers. [E1]")
app = ContextApp(name="voice_support", provider=provider, model=model)
app.add_source("kb", documents=[
    Document(id="policy", title="Refund policy", text="The refund window is 30 days for Pro customers."),
])
# Spoken replies are screened on the way out, just like the text path.
app.add_rail(name="pii_out", kind="safety", direction="output", detectors=["pii"], action="redact")


def script(text, config):
    """Offline backend script: a spoken turn that looks the answer up, then
    speaks a reply (with a PII slip the output rail will redact)."""
    return [
        RealtimeEvent(type="tool_call",
                      tool_call=RealtimeToolCall(call_id="c1", name="research",
                                                 arguments={"question": text})),
        RealtimeEvent(type="response.text",
                      text="Your account 123-45-6789 is on Pro, so the refund window is 30 days."),
        RealtimeEvent(type="response.done"),
    ]


async def main() -> None:
    agent = app.voice_agent(backend="inprocess", script=script)
    async with agent:
        await agent.send_text("How long do I have to request a refund?")
        await agent.commit()
        async for event in agent.events():
            if event.type == "tool_result":
                result = event.data["result"]
                print("research →", result["answer"], "| citations:", result["citations"])
            elif event.type == "response.text":
                print("spoken   →", event.text)
                if event.data.get("redacted"):
                    print("           (rails redacted:", event.data["redacted"], ")")
            elif event.type == "turn.end":
                break
    print("\nin-session tools (permissioned runtime):", app.enabled_tools)


if __name__ == "__main__":
    asyncio.run(main())
