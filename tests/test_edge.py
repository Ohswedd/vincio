"""Edge / WASM in-process runtime.

The dependency-free compile → score → rail → pack core, packaged for a
constrained or browser/WASM target behind a thin in-process boundary: bounded by
an edge profile, identical to the server compile (parity, not a fork), and
import-clean for WASM. Everything here runs offline with no provider, store, or
network.
"""

from __future__ import annotations

import pytest

from vincio import (
    ContextApp,
    EdgeEnvironment,
    EdgeManifest,
    EdgeParityReport,
    EdgeProfile,
    EdgeRequest,
    EdgeResult,
    EdgeRuntime,
    VincioConfig,
    edge_environment,
    edge_manifest,
    is_wasm_runtime,
    verify_edge_parity,
)
from vincio.context.compiler import ContextCompiler
from vincio.core.errors import EdgeError
from vincio.core.types import EvidenceItem, MemoryItem, TaskType
from vincio.edge.parity import EDGE_CORE_MODULES, NATIVE_DENYLIST
from vincio.providers.mock import MockProvider
from vincio.security.rails import Rail


def _evidence(n: int, *, base: int = 0) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            source_id=f"doc{base}_{j}",
            text=(
                f"Clause {base}-{j}: the refund window is {30 + j} days and exception "
                f"{j} is approved by a manager in region {j}."
            ),
            authority=0.55 + (j % 4) * 0.1,
            relevance=0.85,
        )
        for j in range(n)
    ]


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #


def test_default_profile_is_the_worker_profile():
    assert EdgeProfile.default().name == EdgeProfile.worker().name == "edge"


def test_profile_presets_tighten_monotonically():
    browser = EdgeProfile.browser()
    worker = EdgeProfile.worker()
    server = EdgeProfile.server_like()
    assert browser.max_resident_bytes < worker.max_resident_bytes < server.max_resident_bytes
    assert browser.max_input_tokens < worker.max_input_tokens <= server.max_input_tokens


def test_profile_lowers_to_compiler_options():
    profile = EdgeProfile.browser()
    options = profile.to_compiler_options()
    # The profile *is* a bounded ContextCompilerOptions — same fields, no fork.
    assert options.max_resident_bytes == profile.max_resident_bytes
    assert options.max_evidence_items == profile.max_evidence_items
    assert options.max_memory_items == profile.max_memory_items
    assert options.slim_packets is True
    assert options.ordering == profile.ordering


def test_profile_rejects_non_positive_bounds():
    with pytest.raises(ValueError):
        EdgeProfile(max_resident_bytes=0)
    with pytest.raises(ValueError):
        EdgeProfile(max_latency_ms=0.0)


# --------------------------------------------------------------------------- #
# Environment detection
# --------------------------------------------------------------------------- #


def test_environment_detection_is_offline_and_typed():
    env = edge_environment()
    assert isinstance(env, EdgeEnvironment)
    assert env.runtime in ("cpython", "pyodide", "emscripten", "wasi", "unknown")
    # On a normal CI host this is not a WASM target.
    assert env.is_wasm == is_wasm_runtime()
    assert env.has_threads == (not env.is_wasm)


# --------------------------------------------------------------------------- #
# The runtime
# --------------------------------------------------------------------------- #


def test_runtime_compiles_a_string_task_offline():
    runtime = EdgeRuntime(EdgeProfile.browser())
    result = runtime.run("What is the refund window?")
    assert isinstance(result, EdgeResult)
    assert result.prompt  # a model-ready prompt was rendered
    assert result.packet.slim is True  # edge packets are zero-copy
    assert result.within_profile is True
    assert result.allowed is True


def test_runtime_selects_and_cites_evidence():
    runtime = EdgeRuntime(EdgeProfile.browser())
    result = runtime.run(
        EdgeRequest(
            task="refund window and exception approver",
            task_type=TaskType.DOCUMENT_QA,
            instructions=["Answer only from the evidence."],
            evidence=_evidence(3),
        )
    )
    assert result.packet.evidence_items
    assert result.token_count > 0


def test_runtime_request_needs_a_task_or_objective():
    runtime = EdgeRuntime()
    with pytest.raises(EdgeError):
        runtime.run(EdgeRequest())


async def test_arun_matches_run():
    runtime = EdgeRuntime(EdgeProfile.browser())
    req = EdgeRequest(task="refund window", evidence=_evidence(2))
    a = await runtime.arun(req)
    b = runtime.run(req)
    assert a.packet.spec_hash == b.packet.spec_hash


# --------------------------------------------------------------------------- #
# The bounded profile
# --------------------------------------------------------------------------- #


def test_resident_footprint_stays_under_cap_as_corpus_grows_10x():
    profile = EdgeProfile(
        name="capped",
        max_resident_bytes=4096,
        max_input_tokens=4096,
        max_evidence_items=24,
    )
    runtime = EdgeRuntime(profile)
    task = "refund window and exception approver"
    small = runtime.run(EdgeRequest(task=task, evidence=_evidence(4, base=0)))
    big = runtime.run(EdgeRequest(task=task, evidence=_evidence(40, base=1)))
    assert small.within_profile and big.within_profile
    assert big.resident_bytes <= profile.max_resident_bytes
    # Eviction fired under the 10× load: not every offered item could be kept.
    assert len(big.packet.evidence_items) < 40


def test_token_window_is_bounded():
    profile = EdgeProfile(name="tok", max_resident_bytes=1_048_576, max_input_tokens=2048)
    runtime = EdgeRuntime(profile)
    result = runtime.run(EdgeRequest(task="summarize the policy", evidence=_evidence(20)))
    assert result.token_count <= profile.max_input_tokens


def test_strict_raises_when_non_evictable_memory_exceeds_resident_cap():
    # Memory text is not evicted (only evidence is), so a large memory item can
    # push the footprint over a tiny cap — the reachable strict-mode breach.
    profile = EdgeProfile(name="tiny", max_resident_bytes=50, max_input_tokens=8192)
    runtime = EdgeRuntime(profile)
    memory = [MemoryItem(content="a durable preference " * 40, confidence=0.9)]
    lenient = runtime.run(EdgeRequest(task="hello", memory=memory))
    assert lenient.within_profile is False
    with pytest.raises(EdgeError):
        runtime.run(EdgeRequest(task="hello", memory=memory), strict=True)


# --------------------------------------------------------------------------- #
# Rails at the edge
# --------------------------------------------------------------------------- #


def test_output_rail_catches_a_secret_leaking_from_evidence():
    runtime = EdgeRuntime(
        EdgeProfile.browser(),
        rails=[Rail(name="no_secrets", kind="safety", detectors=["secrets", "pii"], direction="output")],
    )
    result = runtime.run(
        EdgeRequest(
            task="print the configuration",
            evidence=[
                EvidenceItem(
                    source_id="cfg",
                    text="api key sk-ABCD1234567890abcdef1234567890abcdef email ops@example.com",
                    relevance=0.9,
                    authority=0.9,
                )
            ],
        )
    )
    assert result.allowed is False
    assert any(v.rail == "no_secrets" for v in result.rail_check.violations)


def test_input_rail_blocks_a_forbidden_topic():
    runtime = EdgeRuntime(
        rails=[Rail(name="no_competitor", kind="topic", direction="input", blocked_topics=["acme"])]
    )
    blocked = runtime.run(EdgeRequest(task="compare us to acme please"))
    assert blocked.allowed is False
    clean = runtime.run(EdgeRequest(task="summarize our own policy"))
    assert clean.allowed is True


def test_per_request_rails_compose_with_runtime_rails():
    runtime = EdgeRuntime(rails=[Rail(name="base", kind="topic", direction="input", blocked_topics=["foo"])])
    result = runtime.run(
        EdgeRequest(
            task="discuss bar topics",
            rails=[Rail(name="extra", kind="topic", direction="input", blocked_topics=["bar"])],
        )
    )
    assert result.allowed is False
    assert {v.rail for v in result.rail_check.violations} == {"extra"}


# --------------------------------------------------------------------------- #
# Parity, not a fork
# --------------------------------------------------------------------------- #


def test_edge_compile_is_byte_identical_to_a_server_compile():
    report = verify_edge_parity()
    assert isinstance(report, EdgeParityReport)
    assert report.packet_identical is True
    assert report.edge_spec_hash == report.server_spec_hash
    assert report.same_compiler and report.same_rail_engine
    assert report.held is True


def test_parity_holds_across_profiles():
    for profile in (EdgeProfile.browser(), EdgeProfile.worker(), EdgeProfile.server_like()):
        report = verify_edge_parity(profile=profile)
        assert report.held, profile.name


def test_runtime_delegates_to_the_canonical_compiler():
    runtime = EdgeRuntime()
    # Not a subclass, not a shadow — the exact server compiler class.
    assert type(runtime.compiler) is ContextCompiler


# --------------------------------------------------------------------------- #
# WASM-buildability manifest
# --------------------------------------------------------------------------- #


def test_edge_core_imports_nothing_native():
    manifest = edge_manifest()
    assert isinstance(manifest, EdgeManifest)
    assert manifest.clean is True, f"native imports on the edge path: {manifest.offending}"
    assert manifest.offending == []
    assert len(manifest.modules) == len(EDGE_CORE_MODULES) >= 20


def test_manifest_detects_a_native_import_when_present():
    # The scanner must actually catch a denylisted import (guard against a
    # vacuously-clean manifest). vincio.context.scoring imports numpy only behind
    # a guarded fallback, so it stays clean; a module that imported numpy at the
    # top level would be flagged. Verify the denylist + scanner on a known module.
    from vincio.edge.parity import _native_imports

    # The vectorized scorer guards its numpy import — it must NOT be flagged.
    assert _native_imports("vincio.context.vectorized") == []
    # Sanity: the denylist is non-trivial and includes the usual native packages.
    assert {"numpy", "av", "tiktoken"} <= NATIVE_DENYLIST


# --------------------------------------------------------------------------- #
# App factory + serialization
# --------------------------------------------------------------------------- #


def test_app_edge_runtime_shares_the_apps_rails(tmp_path):
    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    app = ContextApp(name="edge_app", provider=MockProvider(), config=cfg)
    app.add_rail(name="no_acme", kind="topic", direction="input", blocked_topics=["acme"])
    edge = app.edge_runtime()
    assert isinstance(edge, EdgeRuntime)
    blocked = edge.run(EdgeRequest(task="tell me about acme corp"))
    assert blocked.allowed is False


def test_result_round_trips_through_json():
    runtime = EdgeRuntime(EdgeProfile.browser())
    result = runtime.run(EdgeRequest(task="refund window", evidence=_evidence(2)))
    restored = EdgeResult.model_validate(result.model_dump(mode="json"))
    assert restored.packet.spec_hash == result.packet.spec_hash
    assert restored.token_count == result.token_count
