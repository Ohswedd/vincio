"""Golden-value pins for the two canonical-JSON conventions.

Convention A (``vincio.core.utils.json_dumps`` / ``stable_hash``) and
Convention B (``vincio.core.utils.compact_json`` / ``compact_hash`` /
``sha256_text``) produce different bytes by design, and the hashes below live
inside persisted / signed artifacts — task-set pins, determinism digests, run
ids, community-bundle digests, erasure proofs, and C2PA-bound chart bytes. Each
call site is therefore pinned to its convention forever. These goldens were
computed from the production functions **before** the 7.5.0 consolidation onto
the shared ``core.utils`` helpers and must never change: a diff here means an
artifact-breaking byte change, not a refactor.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace

from vincio.core.utils import compact_hash, compact_json, sha256_text, stable_hash


def test_compact_hash_pins_the_computer_use_observation_recipe():
    # The exact byte recipe the computer-use observation digest has always
    # used: sort_keys, compact separators, default=str, ensure_ascii=True
    # (non-ASCII pins the \uXXXX escaping).
    payload = {"url": "https://ex.com", "title": "Café — résumé", "text": "a\nb", "n": 3}
    assert compact_hash(payload) == "b18c49173d8c989b"


def test_task_set_hash_golden():
    from vincio.evals.benchmarks import BenchmarkTask, compute_task_set_hash

    tasks = [
        BenchmarkTask(id="t2", gold={"a": 1}),
        BenchmarkTask(id="t1", gold="café"),
    ]
    assert compute_task_set_hash(tasks) == "dd13a19a21eaf1ee"


def _benchmark_run():
    from vincio.evals.suite.results import BenchmarkRun, ItemResult
    from vincio.evals.suite.tiers import ProvenanceTier

    return BenchmarkRun(
        benchmark_id="bm1",
        niche="rag",
        tier=ProvenanceTier.STATIC,
        task_set_hash="abc123",
        items=[
            ItemResult(task_id="b", success=True, score=0.51),
            ItemResult(task_id="a", success=False, score=0.12345678),
        ],
    )


def test_benchmark_run_determinism_digest_golden():
    assert _benchmark_run().determinism_digest == "3d696601a734cc67"


def test_suite_run_determinism_digest_stays_spaced():
    # The suite-level pin uses the historical spaced-separator form; persisted
    # determinism pins depend on those exact bytes, so it must NEVER move to
    # the compact helper (the compact form provably differs).
    from vincio.evals.suite.results import SuiteRun
    from vincio.evals.suite.tiers import ProvenanceTier

    run = SuiteRun(run_id="r1", tier=ProvenanceTier.STATIC, runs=[_benchmark_run()])
    assert run.determinism_digest == "4a92923d8cec6b81"
    assert run.determinism_digest != "7bafff96b1f971e8"  # the compact-form digest


def test_suite_run_id_golden():
    from vincio.evals.suite.engine import BenchmarkSuite
    from vincio.evals.suite.tiers import ProvenanceTier

    run_id = BenchmarkSuite._run_id(
        SimpleNamespace(seed=42),
        [SimpleNamespace(id="b2"), SimpleNamespace(id="b1")],
        ProvenanceTier.STATIC,
        "mock",
        None,
    )
    assert run_id == "run_c53d47c1f9f18a74"


def test_bundle_record_digest_golden():
    from vincio.registry.community import BundleRecord

    record = BundleRecord(name="p1", kind="pack", payload={"b": [1, 2], "a": "café"})
    assert record.compute_digest() == (
        "9dcc79a68b275451b55cf72e719853e1675d2337d7e0c1b620461c10b56c6f25"
    )


def test_index_root_primitive_golden():
    assert sha256_text("p1:abc\np2:def") == (
        "3747132ce45d9368efd4bb2bf0a3a4d00966a64956314e529b0aadb0daa3b581"
    )


def test_chart_spec_to_json_golden():
    # These are the rendered artifact bytes a C2PA credential binds
    # (content_sha256) and chart_hash covers — byte-stable forever.
    from vincio.data.charts import ChartChannel, ChartEncoding, ChartSpec, ChartType

    spec = ChartSpec(
        title="Rev — café",
        mark=ChartType.BAR,
        encoding=ChartEncoding(
            x=ChartChannel(field="region", type="nominal"),
            y=ChartChannel(field="revenue", type="quantitative"),
        ),
        columns=["region", "revenue"],
        values=[{"region": "EU", "revenue": 1.5}, {"region": "US", "revenue": 2.0}],
    )
    rendered = spec.to_json()
    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16] == "89bf4cdc027e879d"
    assert rendered == compact_json(spec.to_vega_lite())


def test_erasure_proof_content_digest_golden():
    # digest_payload stays on the historical spaced form — the resulting
    # content_sha256 lives inside signed, persisted proofs.
    from vincio.governance.lineage import build_erasure_proof

    proof = build_erasure_proof("s1", {"vec": ["id2", "id1"], "doc": ["d1"]})
    assert proof.content_sha256 == (
        "56ee1a34254ba218cf10f0fb3effe0886e7ef9652a0f3d0d272bc5ff8306770d"
    )


def test_erasure_proof_signing_payload_golden():
    from vincio.governance.lineage import ErasureProof

    proof = ErasureProof(
        source="s1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        claim_generator="vincio/test",
        removed={"vec": 2, "doc": 1},
        removed_ids={"vec": ["id2", "id1"], "doc": ["d1"]},
        content_sha256=(
            "56ee1a34254ba218cf10f0fb3effe0886e7ef9652a0f3d0d272bc5ff8306770d"
        ),
    )
    payload = proof.signing_payload()
    assert payload == (
        '{"audit_merkle_root":null,"claim_generator":"vincio/test",'
        '"content_sha256":"56ee1a34254ba218cf10f0fb3effe0886e7ef9652a0f3d0d272bc5ff8306770d",'
        '"created_at":"2026-01-01T00:00:00+00:00","removed":{"doc":1,"vec":2},"source":"s1"}'
    )
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16] == "0fb0182a75451ee4"


def test_sha256_text_golden():
    assert sha256_text("café — weights") == (
        "6830629898b2acd7b5bc4909586921e3b83f483931ad7b572f68973cd599e1db"
    )


def test_uplift_results_digest_golden():
    # Stays on its local spaced recipe — the digest becomes a persisted run id.
    from vincio.evals.suite.uplift import UpliftResult, _results_digest

    assert _results_digest([UpliftResult(benchmark_id="b1", direct=0.5, vincio=0.75)]) == (
        "e9b098d3c0cb6093"
    )


def test_conventions_diverge_by_design():
    # Documents WHY both conventions exist: same value, different bytes — on
    # non-ASCII (escaping) AND on pure ASCII (separators / to_jsonable pass).
    for payload in ({"t": "café"}, {"t": "cafe"}):
        assert stable_hash(payload) != compact_hash(payload)
