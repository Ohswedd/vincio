"""Context anchors — keep a PRD / spec / brand frame across a whole coding task.

A vibe-coder building a CLI editor starts with a bulk of MD files — a PRD, a
brand guide, coding standards — that bind *every* step but should not be
re-pasted into *every* model call. Two bad options:

* stuff every MD file into every call → token-hungry, re-paid per step;
* pure per-query RAG → a step like "add a settings panel" doesn't lexically
  match "brand voice: warm and concise", so the constraint silently drops.

Mark the source ``anchor=True`` and Vincio distills it **once** into a compact,
constraint-first, content-hash-cached brief that is injected as **pinned**
evidence into every call — the frame is always present at a flat few-hundred-
token cost, while on-demand detail still flows through normal retrieval. This
tour runs fully offline with a scripted mock model.

Sections:
  1. The brief — a large corpus distilled to a tiny constraint-first frame.
  2. The frame is present on every call, even a query that never mentions it.
  3. The frame is guaranteed even under a tiny token window (never dropped).
  4. On-demand detail still retrieves alongside the frame.
"""

from __future__ import annotations

import tempfile

from vincio.context.anchors import build_anchor_brief
from vincio.core.app import ContextApp
from vincio.core.config import VincioConfig
from vincio.core.tokens import count_tokens
from vincio.core.types import Document
from vincio.providers import MockProvider

# A realistic bulk of standards: a few binding rules buried in narrative.
FILLER = " ".join(
    f"Background {i}: rationale, personas, and considerations that inform the "
    "product but are not rules a coding step must honor on every call."
    for i in range(60)
)
PRD = Document(title="PRD", text=(
    "Build a CLI code editor for vibe coders. It must support a plugin system. "
    "Users should never lose unsaved work. The editor must start under 200ms. " + FILLER))
BRAND = Document(title="Brand identity", text=(
    "Our voice is warm, concise, and encouraging. Always address the user directly. "
    "Never use jargon or corporate speak. Error messages must offer a next step. " + FILLER))
STANDARDS = Document(title="Coding standards", text=(
    "The core is event-sourced. All state changes go through a command bus. "
    "Rendering must be decoupled from the model layer. " + FILLER))
CORPUS = [PRD, BRAND, STANDARDS]


def _config() -> VincioConfig:
    tmp = tempfile.mkdtemp(prefix="vincio_anchors_")
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp}/vincio.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = f"{tmp}/audit"
    return config


def main() -> None:
    # 1. The brief: a big corpus → a tiny, constraint-first frame.
    corpus_tokens = sum(count_tokens(d.text) for d in CORPUS)
    brief = build_anchor_brief(CORPUS, brief_tokens=200)
    print("1. The brief")
    print(f"   corpus {corpus_tokens} tokens → brief {brief.tokens} tokens "
          f"({corpus_tokens / max(brief.tokens, 1):.0f}x smaller), constraint-first:")
    for line in brief.text.splitlines()[:6]:
        print(f"     {line}")

    # A model that reports which frame constraints reached its context.
    def responder(request):
        joined = "\n".join((m.content if isinstance(m.content, str) else "") for m in request.messages)
        seen = [c for c in ("Never use jargon", "never lose unsaved", "command bus")
                if c in joined]
        return "frame:" + ("+".join(seen) if seen else "NONE")

    app = ContextApp(name="coder", provider=MockProvider(responder=responder),
                     model="mock-1", config=_config())
    app.add_source("spec", documents=CORPUS, anchor=True, brief_tokens=200)

    # 2. The frame is present on every call — even a query that never mentions it.
    print("\n2. Frame present on every call (queries never mention the constraints)")
    for query in ("add a settings panel", "parse command-line flags", "write buffer tests"):
        result = app.run(query)
        print(f"   '{query}' → {result.raw_text}")

    # 3. Guaranteed even under a tiny window (the frame is never dropped).
    from vincio.core.types import Budget

    tight_config = _config()
    tight_config.budget = Budget(max_input_tokens=1200, max_output_tokens=200)
    tight = ContextApp(name="tight", provider=MockProvider(responder=responder),
                       model="mock-1", config=tight_config)
    tight.add_source("spec", documents=CORPUS, anchor=True, brief_tokens=200)
    result = tight.run("implement the plugin loader")
    used = result.usage.input_tokens
    print("\n3. Guaranteed under a tiny 1200-token window")
    print(f"   input tokens {used} <= 1200: {used <= 1200}; "
          f"frame still present: {'NONE' not in result.raw_text}")

    # 4. On-demand detail retrieves alongside the frame.
    app.add_source("api", documents=[Document(
        title="API", text="The login endpoint validates a bearer token and returns a session cookie.")])
    detail_result = app.run("how does the login endpoint work")
    has_detail = any("bearer token" in (e.text or "") for e in detail_result.evidence)
    print("\n4. On-demand detail + frame together")
    print(f"   frame present: {'NONE' not in detail_result.raw_text}; "
          f"specific detail retrieved: {has_detail}")

    print("\nDone — the task frame is always present, guaranteed, and ~"
          f"{corpus_tokens // max(brief.tokens, 1)}x cheaper than re-pasting the corpus.")


if __name__ == "__main__":
    main()
