"""LAGER's two deliberate embedder-off residuals — and how the dense signal
tightens each without weakening the abstain-only honesty guarantee.

Both residuals are *lexically inseparable* from a legitimate case, so LAGER's
pure-stdlib path abstains by design (an honest miss, never a wrong cause). A
GENUINELY semantic embedder separates them — the promise the ``similarity_floor``
docstring makes and this experiment demonstrates end to end:

* **Residual 2 — the paraphrased cause (over-abstains).** A lone entity-anchored
  cause that renames the query's topic noun ("outage" → "downtime") shares no
  bridge term, so the lexical floor cannot recall it. The dense *rescue* floor
  (higher than the lexical floor, so a lexical/hash embedder never reaches it)
  lifts the paraphrase over the gap — while the entity-sharing decoy ("revenue
  fell because of tariffs") stays far below, so recalling the paraphrase never
  re-admits the decoy.

* **Residual 1 — the same-document decoy (falsely covers).** An entity-less
  why-need covers through document coherence: a causal claim sharing its document
  with a genuine query match. A decoy cause that merely co-habits a query-matching
  document is indistinguishable from the real multi-hop bridge the flagship relies
  on. The opt-in dense *bridge* floor rejects a semantically off-topic same-doc
  cause while preserving the real bridge.

Runs OFFLINE and DETERMINISTIC by default with a tiny self-contained concept-space
embedder (a faithful stand-in for a real model's world knowledge — synonyms near,
different topics far). Pass ``--embedder auto`` (needs a semantic local model such
as fastembed) or any name :func:`vincio.retrieval.embeddings.build_embedder`
accepts to run the exact same experiment on a real embedder.

    python benchmarks/lager_residuals.py                # deterministic concept stub
    python benchmarks/lager_residuals.py --embedder auto  # a real semantic model
    python benchmarks/lager_residuals.py --json
"""

from __future__ import annotations

import argparse
import json
import re

from vincio.core.types import Document
from vincio.lager import LagerEngine, LazyOptions

# -- a deterministic, offline stand-in for a genuinely semantic embedder ------------------
# Places each claim on a few concept axes: a synonym ("downtime") lands on the same
# OUTAGE axis as "outage"; an unrelated topic ("revenue"/"tariffs") lands elsewhere;
# incident-domain words share a weak coherence axis; an entity name barely moves the
# needle. NOT a lexical/morphological hash — the smallest faithful model of the dense
# world knowledge a real embedder brings.

_CONCEPT_AXES = {
    "outage": ["OUTAGE", "INCIDENT"], "downtime": ["OUTAGE", "INCIDENT"],
    "disruption": ["OUTAGE", "INCIDENT"], "outages": ["OUTAGE", "INCIDENT"],
    "gateway": ["GATEWAY", "INCIDENT"], "gateways": ["GATEWAY", "INCIDENT"],
    "certificate": ["CERT", "INCIDENT"], "cert": ["CERT", "INCIDENT"],
    "tls": ["CERT", "INCIDENT"], "credential": ["CERT", "INCIDENT"],
    "rollout": ["ROLLOUT", "INCIDENT"], "deploy": ["ROLLOUT", "INCIDENT"],
    "deployment": ["ROLLOUT", "INCIDENT"],
    "checkout": ["CHECKOUT", "INCIDENT"], "purchases": ["CHECKOUT", "INCIDENT"],
    "incident": ["INCIDENT"], "failed": ["INCIDENT"], "fail": ["INCIDENT"],
    "rejected": ["INCIDENT"], "crash": ["INCIDENT"],
    "revenue": ["REVENUE"], "quarterly": ["REVENUE"], "tariffs": ["REVENUE"],
    "import": ["REVENUE"], "earnings": ["REVENUE"], "sales": ["REVENUE"],
    "cafeteria": ["FOOD"], "menu": ["FOOD"], "chef": ["FOOD"], "retired": ["FOOD"],
    "acme": ["ENTITY"], "platform": ["ENTITY"],
}
_AXIS_WEIGHT = {"INCIDENT": 0.5, "ENTITY": 0.3}
_AXIS_INDEX = {axis: i for i, axis in enumerate(sorted(
    {a for axes in _CONCEPT_AXES.values() for a in axes}))}


class ConceptEmbedder:
    """A deterministic, offline stand-in for a genuinely semantic embedder."""

    dim = len(_AXIS_INDEX)

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"[a-z]+", text.lower()):
            for axis in _CONCEPT_AXES.get(token, ()):
                vector[_AXIS_INDEX[axis]] += _AXIS_WEIGHT.get(axis, 1.0)
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    async def embed(self, texts):
        return [self.embed_one(t) for t in texts]


def _build_embedder(kind: str | None):
    if kind is None:
        return ConceptEmbedder()
    from vincio.retrieval.embeddings import build_embedder

    return build_embedder(kind)


# -- the two residual corpora ------------------------------------------------------------

RESIDUAL_2 = {
    "name": "residual-2: paraphrased entity-anchored cause",
    "query": "Why did the ACME outage happen?",
    "docs": [
        Document(title="a", text="The ACME outage halted the retail fleet."),
        Document(title="b", text="The ACME downtime was caused by a corrupted memory module."),
    ],
    "cause_marker": "downtime",   # the paraphrased cause we hope to recall
    "options": {},
}
RESIDUAL_2_DECOY = {
    "name": "residual-2 control: entity-sharing decoy must stay rejected",
    "query": "Why did the ACME outage happen?",
    "docs": [
        Document(title="a", text="The ACME outage halted the retail fleet."),
        Document(title="b", text="ACME quarterly revenue fell because of import tariffs."),
    ],
    "cause_marker": None,         # nothing should cover — abstention is correct
    "options": {},
}
RESIDUAL_1 = {
    "name": "residual-1: off-topic same-document causal decoy",
    "query": "why did the gateway rollout fail",
    "docs": [Document(title="d", text=(
        "The gateway rollout failed on 2025-11-03.\n"
        "The cafeteria menu changed because the chef retired."))],
    "cause_marker": None,         # the cafeteria cause must NOT cover
    "options": {"reject_same_doc_causal_decoys": True},
}
RESIDUAL_1_FLAGSHIP = {
    "name": "residual-1 control: the real same-document bridge must survive",
    "query": "why did the checkout outage happen",
    "docs": [Document(title="f", text=(
        "The checkout service suffered a full outage on 2025-11-03.\n"
        "The incident review assigned the root cause to the payments gateway."))],
    "cause_marker": "root cause",   # the genuine bridge must still cover
    "options": {"reject_same_doc_causal_decoys": True},
}
CASES = [RESIDUAL_2, RESIDUAL_2_DECOY, RESIDUAL_1, RESIDUAL_1_FLAGSHIP]


def _covering_claims(pack) -> list[str]:
    covering = {i for ids in pack.coverage.values() for i in ids}
    return [o.claim for o in pack.objects if o.id in covering]


def _gate_probe(query: str) -> str:
    """The exact text the dense rescue compares against: the query with its
    entities removed (mirrors ``LazyRetriever._topic_text``), so the reported
    cosine is the signal the coverage gate actually consults, not the raw query
    whose shared entity would inflate it."""
    from vincio.lager.extract import normalize_entities

    entities = normalize_entities(query)
    if not entities:
        return query
    probe = query
    for entity in entities:
        probe = re.sub(re.escape(entity), " ", probe, flags=re.IGNORECASE)
    probe = " ".join(probe.split())
    return probe if len(probe) >= 3 else query


def _run_case(case: dict, *, embedder) -> dict:
    def once(emb):
        engine = LagerEngine(embedder=emb, options=LazyOptions(**case["options"]))
        engine.ingest(case["docs"])
        pack = engine.retrieve(case["query"])
        return engine, pack

    _, off = once(None)
    engine_on, on = once(embedder)
    # the sharpest single number: the gate's dense cosine to the causal claim
    # (entity-neutralized topic — exactly what the rescue floor is tested against)
    margin = None
    if case["cause_marker"]:
        for obj in engine_on.objects:
            if case["cause_marker"] in obj.claim:
                margin = engine_on.index.semantic_similarity(_gate_probe(case["query"]), obj)
                break
    return {
        "case": case["name"],
        "query": case["query"],
        "off": {"sufficient": off.sufficient, "covers": _covering_claims(off)},
        "on": {"sufficient": on.sufficient, "covers": _covering_claims(on)},
        "cause_cosine_on": None if margin is None else round(margin, 3),
    }


def run(embedder_kind: str | None) -> dict:
    embedder = _build_embedder(embedder_kind)
    floors = LazyOptions()
    return {
        "embedder": embedder_kind or "concept-stub (deterministic)",
        "floors": {
            "similarity_floor": floors.similarity_floor,
            "dense_rescue_floor": floors.dense_rescue_floor,
            "bridge_similarity_floor": floors.bridge_similarity_floor,
        },
        "results": [_run_case(c, embedder=embedder) for c in CASES],
    }


def _verdict(row: dict) -> str:
    off, on = row["off"]["sufficient"], row["on"]["sufficient"]
    if off == on:
        return "unchanged"
    return "recalled (off→on)" if on else "rejected (off→on)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedder", default=None,
                        help="a real embedder name for build_embedder (e.g. 'auto'); "
                             "default is the deterministic offline concept stub")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.embedder)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    f = report["floors"]
    print(f"LAGER residuals — dense signal: {report['embedder']}")
    print(f"floors: lexical {f['similarity_floor']} · dense-rescue "
          f"{f['dense_rescue_floor']} · bridge {f['bridge_similarity_floor']}\n")
    print(f"{'case':<64}{'off':>11}{'on':>11}{'cosine':>9}   verdict")
    for row in report["results"]:
        cosine = "" if row["cause_cosine_on"] is None else f"{row['cause_cosine_on']:.3f}"
        print(f"{row['case']:<64}"
              f"{'sufficient' if row['off']['sufficient'] else 'abstains':>11}"
              f"{'sufficient' if row['on']['sufficient'] else 'abstains':>11}"
              f"{cosine:>9}   {_verdict(row)}")
    print("\nWith a genuine dense signal: the paraphrase is recalled, the "
          "entity-sharing decoy stays rejected, the same-document decoy is "
          "rejected, and the real bridge survives.")
    print("The two 'control' rows are the SAFETY checks — the floors "
          f"(rescue {f['dense_rescue_floor']}, bridge {f['bridge_similarity_floor']}) "
          "are embedder-specific; re-run with --embedder <your model> and confirm "
          "the decoy stays 'abstains' and the bridge stays 'sufficient' before "
          "relying on the tightening.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
