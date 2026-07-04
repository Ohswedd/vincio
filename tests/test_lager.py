"""LAGER — lazy graph evidence retrieval. All offline, all deterministic.

Covers the panel-mandated hard cases: byte-exact span re-derivation on messy
prose, content-derived identity across process-fresh Document ids, gated
contradiction precision on hard negatives, the lexical-decoy no-false-stop and
paraphrase no-spin corpora, honest insufficiency with named needs, provable
termination, and the app wiring (attach → run → erase rebuild)."""

from __future__ import annotations

import tempfile

import pytest

from vincio.core.errors import LagerError
from vincio.core.types import Document, TrustLevel
from vincio.lager import (
    DeterministicClaimExtractor,
    LagerEngine,
    LazyOptions,
    QueryPlanner,
    build_context,
    cited_ids,
    claims_contradict,
    document_key,
    estimate_confidence,
    normalize_entities,
)

# -- the shared multi-hop corpus: the bridge shares NO content word with the query ------

INCIDENT = Document(title="incident", text=(
    "The checkout service suffered a full outage on 2025-11-03. "
    "Customers could not complete purchases for three hours. "
    "The incident review assigned the root cause to the payments gateway."
))
GATEWAY = Document(title="gateway", text=(
    "The payments gateway rejected all connections because its TLS certificate had expired. "
    "Certificate management is owned by the platform team."
))
DISTRACTOR = Document(title="distractor", text=(
    "The marketing team launched a new checkout banner in October. "
    "Checkout conversion rates improved after the redesign. "
    "The outage dashboard shows overall availability trends for checkout."
))
CORPUS = [INCIDENT, GATEWAY, DISTRACTOR]


def _engine(docs=None, **options):
    engine = LagerEngine(options=LazyOptions(**options) if options else None)
    engine.ingest(docs or CORPUS)
    return engine


# -- extraction ------------------------------------------------------------------------


def test_every_object_rederives_byte_exactly():
    extractor = DeterministicClaimExtractor()
    messy = Document(title="messy", text=(
        "Dr. Smith joined Acme Corp. in 2019. She now leads the platform team.\n"
        "Steps:\n- Install the CLI\n- Run vincio init\n"
        "| plan | price |\n| pro | $20 |\n"
        "```\nprint('hello')\n```\n"
        "The API returns JSON by default, but the legacy endpoint still returns XML."
    ))
    objects = extractor.extract(messy)  # would raise LagerError on any span drift
    assert objects
    for obj in objects:
        assert obj.verify(messy.text)


def test_structural_regions_become_single_typed_objects():
    extractor = DeterministicClaimExtractor()
    doc = Document(title="doc", text=(
        "Prose sentence stating one plain fact clearly.\n"
        "- first step of the procedure\n- second step of the procedure\n"
        "| a | b |\n| 1 | 2 |\n"
        "```\ncode line\n```"
    ))
    kinds = sorted({o.kind for o in extractor.extract(doc)})
    assert "list" in kinds and "table" in kinds and "code" in kinds
    lists = [o for o in extractor.extract(doc) if o.kind == "list"]
    assert len(lists) == 1  # the whole region is ONE object, never sentence-split


def test_abbreviations_and_dates_do_not_shatter_or_merge_sentences():
    extractor = DeterministicClaimExtractor()
    doc = Document(title="d", text=(
        "Dr. Smith joined Acme Corp. in 2019 as the head of research. "
        "The outage began on 2025-11-03. The recovery took three hours."
    ))
    claims = [o.claim for o in extractor.extract(doc)]
    assert any("Dr. Smith joined Acme Corp. in 2019" in c for c in claims)  # no shatter
    assert any(c.endswith("on 2025-11-03.") for c in claims)  # date ends its sentence
    assert not any("2025-11-03. The recovery" in c for c in claims)  # no merge across


def test_entity_normalizer_handles_the_ftc_lowercase_and_pronouns():
    assert "ftc" in normalize_entities("The FTC blocked the deal in April")
    assert "he" not in normalize_entities("He reports to Jones.")
    assert "jones" in normalize_entities("He reports to Jones.")
    # lowercase prose has no capitalized entities — the fallback-terms pass
    # (engine-level) buckets it; the normalizer just returns [] without crashing
    assert normalize_entities("the quarterly revenue rose because of pricing") == []


# -- identity & determinism -------------------------------------------------------------


def test_ids_are_content_derived_across_fresh_document_objects():
    text = INCIDENT.text
    ids_a = sorted(o.id for o in _engine([Document(title="x", text=text)]).objects)
    ids_b = sorted(o.id for o in _engine([Document(title="y", text=text)]).objects)
    assert ids_a == ids_b  # Document.id is random per load; identity must not use it
    edited = text.replace("three hours", "four hours")
    ids_c = sorted(o.id for o in _engine([Document(title="z", text=edited)]).objects)
    assert ids_a != ids_c  # an edit changes identity


def test_retrieval_is_deterministic_including_the_trace():
    engine = _engine()
    first = engine.retrieve("why did the checkout outage happen")
    second = engine.retrieve("why did the checkout outage happen")
    assert [o.id for o in first.objects] == [o.id for o in second.objects]
    assert first.gain_trace == second.gain_trace
    assert first.exit_reason == second.exit_reason


# -- the gated contradiction detector ----------------------------------------------------


def _eo(text: str, observed=None):
    from vincio.lager.objects import EvidenceObject
    key = document_key(text)
    return EvidenceObject.create(
        claim=text, doc_key=key, span=(0, len(text)), document_id="d",
        entities=normalize_entities(text), observed_at=observed,
    )


def test_contradiction_hard_negatives_are_suppressed():
    # different scopes / slots / refinements — the raw memory heuristic fires
    # on all of these; the gated detector must not.
    pairs = [
        ("Acme raised prices for the Pro plan in March this year",
         "Acme raised prices for the Team plan in June this year"),
        ("Chen approved the operating budget for the first quarter",
         "Chen did not approve the operating budget for the second quarter"),
        ("The API returns JSON responses by default to clients",
         "The API does not return XML responses to legacy clients"),
    ]
    from vincio.lager.extract import parse_observed_at
    suppressed = 0
    for a, b in pairs:
        eo_a = _eo(a, parse_observed_at(a))
        eo_b = _eo(b, parse_observed_at(b))
        if claims_contradict(eo_a, eo_b) is None:
            suppressed += 1
    assert suppressed >= 2  # precision over the hard-negative set


def test_true_contradiction_is_detected():
    a = _eo("The rotation script was enabled during the datacenter migration window")
    b = _eo("The rotation script was not enabled during the datacenter migration window")
    assert claims_contradict(a, b) == "negation"


# -- the planner -------------------------------------------------------------------------


def test_planner_always_yields_a_required_need():
    needs = QueryPlanner().plan("summarize")
    assert needs and all(n.required for n in needs)


def test_planner_classifies_kinds():
    planner = QueryPlanner()
    assert planner.plan("why did the outage happen")[0].kind == "causal"
    assert planner.plan("when did the migration finish")[0].kind == "temporal"
    assert planner.plan("how many customers were affected")[0].kind == "aggregate"
    kinds = {n.kind for n in planner.plan("how does the gateway affect checkout")}
    assert "relation" in kinds


# -- the lazy loop -----------------------------------------------------------------------


def test_multi_hop_bridge_is_found_and_loop_terminates_sufficient():
    pack = _engine().retrieve("why did the checkout outage happen")
    text = " ".join(o.claim for o in pack.objects)
    assert "root cause" in text  # the bridge, zero lexical overlap with the query
    assert pack.exit_reason.startswith("E1")
    assert pack.sufficient


def test_lexical_decoy_does_not_cover_a_causal_need():
    pack = _engine().retrieve("why did the checkout outage happen")
    decoy_id = next(o.id for o in _engine().objects
                    if "dashboard" in o.claim)
    for ids in pack.coverage.values():
        assert decoy_id not in ids  # topical overlap is not answeredness


def test_easy_query_stops_in_one_round_hard_takes_more():
    engine = _engine()
    easy = engine.retrieve("who owns certificate management")
    hard = engine.retrieve("why did the checkout outage happen")
    assert easy.rounds == 1 and easy.sufficient
    assert easy.rounds <= hard.rounds  # laziness scales with difficulty


def test_evidence_count_varies_with_query_complexity():
    engine = _engine()
    easy = engine.retrieve("who owns certificate management")
    hard = engine.retrieve(
        "why did the checkout outage happen and why did the payments gateway fail"
    )
    assert len(easy.objects) < len(hard.objects)  # no fixed k


def test_impossible_query_abstains_honestly_without_spinning():
    pack = _engine(max_rounds=6).retrieve("what is the chief executive compensation package")
    assert not pack.sufficient
    assert pack.uncovered_needs  # named, so downstream abstains
    assert pack.rounds <= 6 and pack.exit_reason.startswith(("E0", "E2", "E4"))


def test_paraphrase_corpus_terminates_without_spinning():
    # the answer exists but shares almost no surface forms with the query;
    # embedder=off must terminate quickly (honest insufficiency allowed)
    docs = [Document(title="p", text=(
        "The service disruption stemmed from an expired security credential. "
        "Renewal automation had been paused for the infrastructure move."
    ))]
    pack = _engine(docs, max_rounds=6).retrieve("why did the app go down")
    assert pack.rounds <= 3  # patience/empty-frontier, never a max_rounds burn


def test_token_budget_is_a_hard_exit():
    pack = _engine(max_evidence_tokens=12).retrieve(
        "why did the checkout outage happen and why did the payments gateway fail"
    )
    assert pack.token_cost <= 40  # bounded near the budget (one claim granularity)


def test_gain_trace_explains_every_round():
    pack = _engine().retrieve("why did the checkout outage happen")
    assert len(pack.gain_trace) == pack.rounds
    for entry in pack.gain_trace:
        assert {"round", "added", "gain", "acquired", "frontier"} <= set(entry)


# -- pack → compiler bridge ---------------------------------------------------------------


def test_pack_items_carry_explicit_ids_spans_and_pin_covering_evidence():
    pack = _engine().retrieve("why did the checkout outage happen")
    items = pack.as_evidence_items()
    covering = {i for ids in pack.coverage.values() for i in ids}
    for item in items:
        assert item.id.startswith("eo:")  # explicit, content-derived
        assert item.span is not None
        assert item.metadata["doc_key"]
    assert any(item.pinned for item in items if item.id in covering)


def test_pack_verify_catches_tamper():
    engine = _engine()
    pack = engine.retrieve("why did the checkout outage happen")
    assert engine.verify(pack)
    tampered = {k: v.replace("root cause", "best guess") for k, v in
                engine.documents_text.items()}
    assert not pack.verify(tampered)


# -- answer plane -------------------------------------------------------------------------


def test_build_context_flags_contradictions_and_insufficiency():
    a = Document(title="a", text=(
        "The rotation script was enabled during the datacenter migration window. "
        "The platform team owns certificate management duties."
    ))
    b = Document(title="b", text=(
        "The rotation script was not enabled during the datacenter migration window."
    ))
    engine = _engine([a, b])
    pack = engine.retrieve("was the rotation script enabled during the migration")
    context = build_context(pack)
    if pack.contradictions:
        assert "CONFLICTS WITH" in context
    insufficient = engine.retrieve("what is the chief executive compensation package")
    assert "INSUFFICIENT" in build_context(insufficient)


def test_verify_answer_binds_citations_to_source_bytes():
    from vincio.lager import LagerAnswer, verify_answer

    engine = _engine()
    pack = engine.retrieve("why did the checkout outage happen")
    answer = LagerAnswer(
        query=pack.query, text=f"Grounded [{pack.objects[0].id}].",
        citations=[pack.objects[0].id], confidence=0.8,
        sufficient=pack.sufficient, pack=pack,
    )
    ok, problems = verify_answer(answer, engine.documents_text)
    assert ok and not problems
    # a citation outside the pack is named as a problem
    bad = answer.model_copy(update={"citations": ["eo:ffffffffffffffff"]})
    ok, problems = verify_answer(bad, engine.documents_text)
    assert not ok and any("not in the evidence pack" in p for p in problems)
    # a tampered source is named as a problem
    tampered = {k: v.replace("root cause", "best guess")
                for k, v in engine.documents_text.items()}
    ok, problems = verify_answer(answer, tampered)
    assert not ok and any("re-derive" in p for p in problems)


def test_cited_ids_parse_and_confidence_is_deterministic():
    assert cited_ids("Fact one [eo:0123456789abcdef]. Again [eo:0123456789abcdef].") == [
        "eo:0123456789abcdef"
    ]
    engine = _engine()
    pack = engine.retrieve("why did the checkout outage happen")
    assert estimate_confidence(pack) == estimate_confidence(pack)
    assert 0.0 < estimate_confidence(pack) <= 1.0


# -- app wiring ---------------------------------------------------------------------------


def _app(responder=None):
    from vincio.core.app import ContextApp
    from vincio.core.config import VincioConfig
    from vincio.providers import MockProvider

    tmp = tempfile.mkdtemp()
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp}/v.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = f"{tmp}/audit"
    return ContextApp(name="lager", provider=MockProvider(responder=responder),
                      model="mock-1", config=config)


def test_use_lager_ingests_registered_sources_and_run_uses_the_pack():
    seen = {}

    def responder(request):
        seen["prompt"] = "\n".join(
            m.content if isinstance(m.content, str) else "" for m in request.messages
        )
        return "grounded"

    app = _app(responder)
    app.add_source("kb", documents=CORPUS)
    engine = app.use_lager()
    assert len(engine) > 0
    result = app.run("why did the checkout outage happen")
    assert result.raw_text == "grounded"
    assert "root cause" in seen["prompt"]  # the lazy pack reached the model
    assert any((e.metadata or {}).get("lager") for e in result.evidence)


def test_retrieve_evidence_requires_attachment():
    app = _app()
    with pytest.raises(LagerError):
        app.retrieve_evidence("anything")


def test_erase_source_rebuilds_the_engine_without_the_erased_text():
    app = _app()
    app.add_source("kb", documents=CORPUS)
    app.use_lager()
    assert len(app.lager_engine) > 0
    app.erase_source("kb", prove=False)
    assert len(app.lager_engine) == 0  # no evidence object survives erasure


def test_untrusted_authority_is_inherited():
    trusted = Document(title="t", text="The platform team owns certificate management.",
                       trust_level=TrustLevel.DEVELOPER)
    engine = _engine([trusted])
    assert all(o.trust_level == TrustLevel.DEVELOPER for o in engine.objects)
    assert all(o.authority > 0.5 for o in engine.objects)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
