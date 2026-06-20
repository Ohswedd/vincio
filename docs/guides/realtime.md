# Voice & realtime (optional module)

> **Optional module.** A separate, opt-in module for stateful
> bidirectional voice/realtime sessions. It is explicitly scoped as a *stateful
> bidirectional* module — **not** core context engineering — and lives behind
> the `vincio[realtime]` extra. The dependency-free in-process backend is the
> default and the offline path; the hosted backends speak WebSocket.

`vincio.realtime` gives you a provider-neutral realtime session over a
pluggable backend — **OpenAI Realtime**, **Gemini Live**, or a deterministic
**in-process** backend for tests and offline development. The session owns the
protocol-agnostic concerns; the backend owns the wire.

The thing a raw realtime SDK does not give you: **in-session tool calls run
through the same permissioned, sandboxed, audited tool runtime** as every other
Vincio tool. Voice does not get a privileged side channel.

```bash
pip install "vincio[realtime]"   # adds websockets for the hosted backends
```

## A session in one screen

```python
from vincio.realtime import RealtimeSession, RealtimeConfig

session = RealtimeSession(config=RealtimeConfig(model="gpt-realtime", voice="alloy"))
async with session:
    await session.send_text("What's the weather in Paris?")
    await session.commit()                 # end the user's turn
    async for event in session.events():
        if event.type == "response.text":
            print(event.text, end="")
        elif event.type == "turn.end":
            break
```

`RealtimeSession` is an async context manager. You **send** (`send_text`,
`send_audio`, `commit`, `interrupt`) and you **receive** a normalized event
stream (`events()`). Every backend emits the same `RealtimeEvent` types:
`session.started`, `input.transcript`, `vad.speech_start` / `vad.speech_stop`,
`turn.start` / `turn.end`, `response.text` / `response.audio` / `response.done`,
`tool_call`, `tool_result`, `interrupted`, and `error`.

## Voice-activity detection and barge-in

Server-side VAD turns speech into turns, and **interruption** (barge-in)
cancels an in-flight response:

```python
session = RealtimeSession(config=RealtimeConfig(vad=VADConfig(threshold=0.5)))
async with session:
    await session.send_audio(speech_pcm)   # vad.speech_start
    await session.send_audio(silence_pcm)  # vad.speech_stop → turn ends
    async for event in session.events():
        if user_started_talking_again:
            await session.interrupt()       # stop paying for a response nobody is listening to
```

## In-session tools through the permissioned runtime

The headline integration: wire a session from a `ContextApp` and its tool calls
flow through the app's permissioned, sandboxed, audited tool runtime — exactly
like a native tool call.

```python
from vincio import ContextApp

app = ContextApp(name="concierge")
app.add_tool(get_weather, permission="read_only")

session = app.realtime_session(backend="openai")   # or "gemini" / "inprocess"
async with session:
    await session.send_text("Weather in Paris?")
    await session.commit()
    async for event in session.events():
        if event.type == "tool_result":
            print(event.data["result"])    # get_weather ran through the audited runtime
        elif event.type == "response.done":
            break
```

Standalone (without an app), pass any `tool_dispatcher`:

```python
async def dispatch(name: str, arguments: dict) -> dict:
    return await my_tools[name](**arguments)

session = RealtimeSession(tool_dispatcher=dispatch)
```

## Backends

| Backend | `connect_realtime(...)` | Transport | Extra |
|---|---|---|---|
| In-process | `"inprocess"` | none (deterministic, scriptable) | none |
| OpenAI Realtime | `"openai"` | WebSocket | `vincio[realtime]` |
| Gemini Live | `"gemini"` | WebSocket | `vincio[realtime]` |

`connect_realtime("inprocess", script=...)` drives the model's response from a
pure function, so realtime flows — turns, VAD, interruption, tool round-trips —
are reproducible and fully testable offline with no network.

## End-to-end voice agent

A raw session gives you the wire. A spoken assistant usually wants the rest of
the stack behind it: the ability to *look things up* (grounded, cited,
budget-bounded), to *remember* across the conversation, and to be *guarded* on
both the spoken-in and spoken-out boundary exactly like the text path.
`app.voice_agent(...)` assembles that from parts that already exist, returning a
`VoiceAgent`:

```python
app.add_source("kb", path="./help-center")
agent = app.voice_agent()                       # research + memory OS + rails, all on
async with agent:
    await agent.send_text("What is the refund window?")
    await agent.commit()
    async for event in agent.events():          # the event stream is rail-screened
        if event.type == "tool_result":
            print(event.data["result"])         # the cited research answer
        elif event.type == "response.text":
            print(event.text)                   # redacted/blocked per the app's output rails
        elif event.type == "turn.end":
            break
```

It wires three things onto the session:

- **Deep research** — registers [`app.research`](../concepts/agents.md) as an
  in-session `research` tool, so a spoken question runs the cited search → read →
  verify → synthesize loop and answers from sources, not the model's memory.
- **Memory OS** — enables the self-editing memory tools
  ([`enable_memory_os`](../concepts/memory.md)), so the agent can recall and
  update its own memory mid-conversation on the audited, permissioned path.
- **Rails** — runs the app's deterministic input/output
  [rails](reliability-guardrails.md) over every spoken transcript and reply,
  redacting or blocking before audio is produced, recorded on the audit chain.

Pass `research=False`, `memory_os=False`, or `rails=False` to opt out of any one;
`backend="openai"` / `"gemini"` switch to the hosted wire. Tool calls (including
`research` and the memory ops) route through the app's permissioned, sandboxed,
budgeted, audited runtime — a voice turn cannot do anything a text turn could
not. See [`examples/44_voice_agent.py`](../../examples/44_voice_agent.py) for a
runnable, offline end-to-end version.

## Scope

This module is deliberately small and opt-in. The base session is **not** wired
into the context compiler, evals, or the closed loop — a realtime audio session
is a different shape of computation from a compiled context packet. The
`VoiceAgent` is the bridge for the cases that *do* want grounding, memory, and
rails. What both share is the tool runtime: realtime tool calls are permissioned,
sandboxed, and audited like everything else. See the
[security threat model](../security/threat-model.md) for how tool calls
(including realtime ones) are governed.
