"""Causal record-replay debugger.

A trace tells you *what* a run did; it does not let you *re-run* it. This example
shows the next rung: recording a whole agent run edge-by-edge, then replaying it
byte-for-byte — every model response, tool output, and the negotiated
capabilities served from the recording — so a past run becomes something you can
step, inspect, and branch instead of a one-shot you can only read about.

Four steps, all offline and deterministic (driven by the mock provider):

  1. Record: run a tool-using task and capture every non-deterministic edge into
     a portable, content-addressed Recording.
  2. Replay (faithful): replay against an app whose *live* provider would answer
     differently — the recording, not the provider, serves the answer, so the
     run reproduces byte-for-byte.
  3. Replay (divergence): change the prompt and replay — the recorded edge no
     longer matches, and the debugger reports exactly where the run drifted.
  4. Branch-and-edit: fork the recording, change the tool's output, and
     re-execute only the affected suffix — the unchanged prefix is still served
     from the recording, so a fix is validated against the exact failing run.

Everything here is opt-in and additive; nothing below is required to run Vincio.
"""

from __future__ import annotations

import asyncio

from vincio import ContextApp, VincioConfig
from vincio.context.evidence_store import InMemoryEvidenceStore
from vincio.observability import BranchEdit, Recorder, Recording, Replayer
from vincio.providers.mock import MockProvider


def _config() -> VincioConfig:
    config = VincioConfig()
    config.observability.exporter = "memory"
    return config


def build_app(final_answer: str) -> ContextApp:
    """An app whose model calls a `lookup` tool once, then answers from it."""
    script = [
        {"tool_call": {"name": "lookup", "arguments": {"q": "refund-policy"}}},
        final_answer,
    ]
    app = ContextApp(config=_config(), provider=MockProvider(script=list(script)))

    @app.tool_registry.register(name="lookup")
    def lookup(q: str) -> str:
        return "Refunds are accepted within 30 days of purchase."

    app.enabled_tools.append("lookup")
    return app


async def record_a_run() -> Recording:
    print("1. Record — capture every non-deterministic edge of a run")
    app = build_app("Refunds are accepted within 30 days. [policy]")
    result, recording = await Recorder(app).record("What is the refund policy?")
    print(f"   output: {result.raw_text!r}")
    print(
        f"   edges:  {len(recording.edges)} "
        f"(model={len(recording.model_calls)}, tool={len(recording.tool_calls)})"
    )
    print(f"   digest: {recording.fidelity_digest}  verified={recording.verify()}")

    # A recording is portable: write it to a content-addressed store and read it
    # back; the address both locates and verifies the bytes.
    store = InMemoryEvidenceStore()
    address = recording.put(store)
    Recording.from_store(store, address)
    print(f"   stored at content address {address} (round-trips and re-verifies)")
    return recording


async def replay_faithfully(recording: Recording) -> None:
    print("\n2. Replay (faithful) — the recording, not the provider, drives the run")
    # The replay app's live provider would answer "WRONG (live)" — faithful replay
    # still reproduces the recorded answer because every edge is served back.
    replay_app = build_app("WRONG (live) — should never appear")
    result = await Replayer(replay_app).replay(recording)
    print(f"   faithful={result.faithful}  identical={result.output_identical}")
    print(f"   served from recording: {result.served_from_recording} edges")
    print(f"   replayed output: {result.replayed_output!r}")


async def replay_with_divergence(recording: Recording) -> None:
    print("\n3. Replay (divergence) — changed code is detected, not silently re-run")
    diverged = build_app("WRONG (live)")
    diverged.configure(objective="Summarize the cancellation policy instead")
    result = await Replayer(diverged).replay(recording)
    print(f"   faithful={result.faithful}  divergences={len(result.divergences)}")
    if result.divergences:
        first = result.divergences[0]
        print(f"   first divergence: {first.kind} — {first.detail}")


async def branch_and_edit(recording: Recording) -> None:
    print("\n4. Branch-and-edit — re-execute only the affected suffix")
    tool_key = recording.tool_calls[0].key
    branch = await Replayer(build_app("unused")).branch(
        recording,
        edits=[
            BranchEdit(
                kind="tool_call",
                key=tool_key,
                value={
                    "call_id": "edit",
                    "tool_name": "lookup",
                    "status": "ok",
                    "output": "Refunds are accepted within 7 days of purchase.",
                },
            )
        ],
        fallback=MockProvider(default_text="Refunds are accepted within 7 days. [policy]"),
    )
    print(f"   edits applied: {branch.edits_applied}")
    print(
        f"   served from recording (unchanged prefix): {branch.served_from_recording}; "
        f"re-executed (affected suffix): {branch.reexecuted}"
    )
    print(f"   branched output: {branch.output!r}")


async def main() -> None:
    recording = await record_a_run()
    await replay_faithfully(recording)
    await replay_with_divergence(recording)
    await branch_and_edit(recording)
    print("\nA past run, replayed byte-for-byte — then forked, edited, and re-validated.")


if __name__ == "__main__":
    asyncio.run(main())
