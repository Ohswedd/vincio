"""Context anchors — the always-on task frame for chain-call work: a compact,
constraint-first, content-hash-cached brief injected as pinned evidence that is
guaranteed into every packet (never dropped by gate, dedup, conflict, budget, or
footprint), never starves retrieved detail, and never breaks the hard token
budget. Plus the dynamic-retrieval knobs (grow-only adaptive k, embedder=auto).
All offline and deterministic."""

from __future__ import annotations

import asyncio

import pytest

from vincio.context.anchors import AnchorSet, build_anchor_brief
from vincio.context.compiler import ContextCompiler, ContextCompilerOptions
from vincio.core.app import ContextApp
from vincio.core.config import VincioConfig
from vincio.core.types import (
    Budget,
    Document,
    EvidenceItem,
    Instruction,
    Objective,
    TaskType,
    UserInput,
)
from vincio.providers import MockProvider

PRD = Document(
    title="PRD",
    text=(
        "Build a CLI code editor for vibe coders. It must support a plugin system. "
        "Users should never lose unsaved work. The editor must start under 200ms. "
        "Later we may add AI completion and assorted quality-of-life features."
    ),
)
BRAND = Document(
    title="Brand identity",
    text=(
        "Our voice is warm, concise, and encouraging. Always address the user "
        "directly. Never use jargon or corporate speak. Error messages must offer "
        "a next step."
    ),
)
ARCH = Document(
    title="Architecture",
    text=(
        "The core is event-sourced. All state changes go through a command bus. "
        "Rendering must be decoupled from the model layer."
    ),
)


def _offline_config(tmp_path) -> VincioConfig:
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/vincio.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    return config


# -- the brief builder -----------------------------------------------------------------------


def test_brief_is_token_bounded_and_constraint_first():
    brief = build_anchor_brief([PRD, BRAND, ARCH], brief_tokens=160)
    assert brief.tokens <= 200  # bounded (with rendering overhead)
    # the load-bearing normative constraints survive the budget
    for constraint in ("must support a plugin", "never lose unsaved", "Never use jargon",
                       "command bus"):
        assert constraint in brief.text, f"lost constraint: {constraint}"


def test_brief_is_deterministic_and_verifies():
    a = build_anchor_brief([PRD, BRAND], brief_tokens=140)
    b = build_anchor_brief([PRD, BRAND], brief_tokens=140)
    assert a.text == b.text and a.content_hash == b.content_hash
    assert a.verify([PRD, BRAND], budget_tokens=140)
    assert not a.verify([PRD], budget_tokens=140)  # a changed corpus does not verify


def test_brief_evidence_is_pinned_with_stable_content_id():
    brief = build_anchor_brief([PRD], brief_tokens=120)
    ev = brief.as_evidence()
    assert ev.pinned and ev.relevance == 1.0
    assert ev.id == f"anchor:{brief.content_hash[:16]}"  # stable, content-derived
    assert ev.metadata.get("anchor_brief") is True


def test_anchor_set_caches_and_rebuilds_on_change():
    s = AnchorSet()
    s.add("prd", [PRD], brief_tokens=120)
    first = s.brief()
    assert s.brief() is first  # cached
    s.add("brand", [BRAND], brief_tokens=120)  # corpus changed
    assert s.brief() is not first
    assert "Never use jargon" in s.brief().text


def test_anchor_set_rebuilds_when_brief_tokens_change():
    s = AnchorSet()
    s.add("prd", [PRD], brief_tokens=60)
    small = s.brief().text
    s.add("prd", [PRD], brief_tokens=300)  # same corpus, larger budget
    assert s.brief().text != small


# -- the compiler guarantee ------------------------------------------------------------------


def _compile(evidence, budget, task=TaskType.CODING, query="implement the login endpoint", **opt):
    compiler = ContextCompiler(ContextCompilerOptions(**opt))
    return asyncio.run(
        compiler.compile(
            objective=Objective("build the app", task_type=task),
            user_input=UserInput(text=query),
            instructions=[Instruction("Answer from the sources")],
            evidence=evidence,
            budget=budget,
        )
    )


def _ids(result):
    return [e.get("id") for e in result.packet.evidence_items]


FRAME = build_anchor_brief([PRD, BRAND, ARCH], brief_tokens=140).as_evidence()
DETAIL = EvidenceItem(
    id="d1", source_id="arch.md",
    text="The login endpoint validates a bearer token and returns a session cookie.",
    relevance=0.9,
)


def test_frame_present_and_first_on_normal_budget():
    result = _compile([FRAME, DETAIL], Budget(max_input_tokens=8000))
    assert FRAME.id in _ids(result)
    assert _ids(result)[0] == FRAME.id  # stable position: frame before detail


def test_tier2_detail_survives_alongside_the_frame():
    # the query matches the detail; the frame must NOT suppress it
    result = _compile([FRAME, DETAIL], Budget(max_input_tokens=8000), query="login endpoint bearer token")
    assert FRAME.id in _ids(result) and "d1" in _ids(result)


def test_budget_invariant_holds_on_tiny_window_no_crash():
    result = _compile(
        [FRAME, DETAIL], Budget(max_input_tokens=1000, max_output_tokens=200),
    )
    assert result.token_count <= 1000  # never exceeds the window
    assert FRAME.id in _ids(result)  # frame still guaranteed


def test_frame_survives_classification_with_zero_evidence_allocation():
    # CLASSIFICATION allocates ~0 to evidence; the frame is reserved off the top
    result = _compile([FRAME, DETAIL], Budget(max_input_tokens=2000), task=TaskType.CLASSIFICATION)
    assert result.token_count <= 2000
    assert FRAME.id in _ids(result)


def test_frame_survives_footprint_ceiling():
    result = _compile(
        [FRAME, DETAIL], Budget(max_input_tokens=8000),
        max_resident_bytes=200,  # tight footprint forces eviction
    )
    assert FRAME.id in _ids(result)  # pinned is eviction-exempt


def test_frame_survives_a_conflicting_higher_authority_chunk():
    contradiction = EvidenceItem(
        id="c1", source_id="old.md", authority=0.99,
        text="The editor must start under 900ms, not 200ms.", relevance=0.9,
    )
    result = _compile([FRAME, contradiction], Budget(max_input_tokens=8000),
                      query="editor startup time")
    assert FRAME.id in _ids(result)  # never dropped by conflict resolution


def test_frame_overflow_ladder_never_drops_the_frame():
    huge = build_anchor_brief([PRD, BRAND, ARCH], brief_tokens=4000).as_evidence()
    # cap = 50% of a 400-token window = 200 → frame is compressed/truncated to fit
    result = _compile([huge], Budget(max_input_tokens=400, max_output_tokens=50))
    assert result.token_count <= 400
    assert huge.id in _ids(result)  # fitted, not dropped


# -- cache correctness -----------------------------------------------------------------------


def test_pinned_flag_keys_the_compile_signature():
    # same text, different pinned state → different compiled packet (no stale cache hit)
    item_a = EvidenceItem(id="x", source_id="s", text="a shared body about the topic", relevance=0.9)
    item_b = item_a.model_copy(update={"pinned": True})
    budget = Budget(max_input_tokens=8000)
    a = _compile([item_a], budget)
    b = _compile([item_b], budget)
    assert a.packet.evidence_items != b.packet.evidence_items or a.token_count != b.token_count \
        or _ids(a) == _ids(b)  # at minimum the signature differs so no wrong cache reuse
    # direct signature check: the arena/compile signature must distinguish the
    # pinned state, so a source flipped to anchor=True with unchanged text can
    # never serve a stale required=False candidate from cache.
    compiler = ContextCompiler(ContextCompilerOptions())
    sig_a = compiler._candidate_signature(
        evidence=[item_a], memory=[], tool_results=[], privacy="public", tenant_id=None
    )
    sig_b = compiler._candidate_signature(
        evidence=[item_b], memory=[], tool_results=[], privacy="public", tenant_id=None
    )
    assert sig_a != sig_b


# -- end-to-end: frame retention across chained calls ----------------------------------------


def test_frame_retained_across_chained_calls_even_when_query_mismatches(tmp_path):
    def responder(request):
        joined = "\n".join((m.content if isinstance(m.content, str) else "") for m in request.messages)
        return "SAW_FRAME" if "Never use jargon" in joined else "NO_FRAME"

    app = ContextApp(
        name="coder", provider=MockProvider(responder=responder), model="mock-1",
        config=_offline_config(tmp_path),
    )
    app.add_source("spec", documents=[PRD, BRAND, ARCH], anchor=True, brief_tokens=160)
    assert app.task_brief() and "Never use jargon" in app.task_brief()
    # queries that do NOT lexically mention the brand constraint
    for query in ("parse command-line flags", "add unit tests for the buffer", "refactor the parser"):
        assert app.run(query).raw_text == "SAW_FRAME", f"frame lost on: {query}"


def test_web_off_style_run_without_anchors_is_unaffected(tmp_path):
    app = ContextApp(name="plain", provider=MockProvider(), model="mock-1", config=_offline_config(tmp_path))
    assert app.task_brief() is None
    assert not app.anchors
    result = app.run("hello")  # no anchors → no pinned evidence, ordinary compile
    assert not any(e.metadata.get("anchor_brief") for e in result.evidence)


# -- dynamic retrieval -----------------------------------------------------------------------


def test_embedder_auto_resolves_without_crashing():
    from vincio.retrieval.embeddings import FastEmbedEmbedder, LocalHashEmbedder, build_embedder

    embedder = build_embedder("auto")
    assert isinstance(embedder, LocalHashEmbedder | FastEmbedEmbedder)


def test_adaptive_top_k_is_grow_only_and_off_by_default():
    from vincio.retrieval.engine import RetrievalEngine
    from vincio.retrieval.indexes import BM25Index

    idx = BM25Index()
    off = RetrievalEngine([idx])
    assert off._effective_top_k(5, off._heuristic_plan("anything precise")) == 5  # off = no change
    on = RetrievalEngine([idx], adaptive_top_k=True, adaptive_top_k_ceiling=20)
    broad = on._effective_top_k(5, on._heuristic_plan("what is a and how does b relate to c and d"))
    precise = on._effective_top_k(5, on._heuristic_plan("login"))
    assert broad >= precise >= 5  # never below the floor; broad grows
    assert broad <= 20  # capped


def test_adaptive_top_k_default_off_in_config():
    assert VincioConfig().retrieval.adaptive_top_k is False
    assert VincioConfig().retrieval.embedder == "local"  # default not env-dependent


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
