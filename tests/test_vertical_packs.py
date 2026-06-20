"""Vertical packs: full-stack domain bundles that preconfigure retrieval,
memory, rails, metrics, residency, and a golden eval set, and run offline."""

from __future__ import annotations

import json

import pytest

from vincio import ContextApp, available_packs, load_pack
from vincio.evals.metrics import METRICS
from vincio.providers import MockProvider

VERTICALS = ["healthcare", "ediscovery", "kyc", "customer_support", "code_review"]


def test_verticals_are_listed():
    assert set(VERTICALS) <= set(available_packs())


@pytest.mark.parametrize("name", VERTICALS)
def test_vertical_pack_is_well_formed(name):
    pack = load_pack(name)
    assert pack.name == name
    assert pack.is_vertical, "a vertical pack must preconfigure retrieval/memory/residency"
    assert pack.role and pack.objective
    assert pack.output_schema and pack.output_schema["type"] == "object"
    # Every declared evaluator must be a real metric.
    for metric in pack.evaluators:
        assert metric in METRICS, f"{name}: unknown metric {metric!r}"
    # Golden eval set ships with the pack.
    dataset = pack.dataset()
    assert len(dataset) >= 3
    # Retrieval knobs are real config fields (besides the add_source mode hint).
    from vincio.core.config import RetrievalConfig

    for key in pack.retrieval:
        assert key == "mode" or hasattr(RetrievalConfig(), key), f"{name}: bad retrieval knob {key}"


@pytest.mark.parametrize("name", VERTICALS)
def test_vertical_pack_applies_full_stack(name):
    pack = load_pack(name)
    app = ContextApp(name="t", provider=MockProvider(), model="mock-1").use_pack(name)
    # schema
    assert app.output_contract.schema_name == pack.output_schema_name
    # metrics
    for metric in pack.evaluators:
        assert metric in app.evaluators
    # memory
    if pack.memory is not None:
        assert app.memory is not None
    # rails
    rail_names = {r.name for r in app.rail_engine.rails}
    for rail in pack.rails:
        assert rail["name"] in rail_names
    # residency
    if pack.residency:
        assert app.residency.enforced
        assert "on_prem" in app.residency.allowed_regions  # offline path stays in jurisdiction
    # retrieval knobs landed on the config
    for key, value in pack.retrieval.items():
        if key == "mode":
            continue
        assert getattr(app.config.retrieval, key) == value


@pytest.mark.parametrize("name", VERTICALS)
def test_vertical_pack_runs_offline(name):
    """A residency-pinned vertical pack must still run on the offline mock."""
    app = ContextApp(name="t", provider=MockProvider(), model="mock-1").use_pack(name)
    app.set_policy("answer_only_from_sources", False).set_policy("require_citations", False)
    result = app.run("a representative question for this domain")
    assert result.error is None


def test_residency_pack_still_refuses_out_of_region_endpoint():
    """deny_on_unknown is off for offline safety, but an *identifiable*
    out-of-jurisdiction endpoint is still refused."""
    app = ContextApp(name="t", provider=MockProvider(), model="mock-1").use_pack("healthcare")
    assert app.residency.enforced
    # A us-only posture refuses an EU-region endpoint.
    violation = app.residency.check(
        provider="vertex", model="m", base_url="https://europe-west4-aiplatform.googleapis.com"
    )
    assert violation is not None


def test_healthcare_redacts_pii_on_output():
    payload = {
        "answer": "Patient SSN 123-45-6789 has a penicillin allergy.",
        "phi_detected": True,
        "needs_clinician": False,
    }
    provider = MockProvider(responder=lambda request: json.dumps(payload))
    app = ContextApp(name="clinic", provider=provider, model="mock-1").use_pack("healthcare")
    app.set_policy("answer_only_from_sources", False).set_policy("require_citations", False)
    result = app.run("Does the patient have allergies?")
    assert result.error is None
    # The phi_redact output rail masks the identifier in the structured deliverable.
    output = result.output
    answer = output.answer if hasattr(output, "answer") else output["answer"]
    assert "123-45-6789" not in answer
    assert "REDACTED" in answer


def test_run_with_kyc_pack_enforces_schema():
    payload = {
        "risk_rating": "high",
        "sanctions_hit": True,
        "pep": False,
        "sar_recommended": True,
        "rationale": "Confirmed OFAC match.",
    }
    provider = MockProvider(responder=lambda request: json.dumps(payload))
    app = ContextApp(name="kyc", provider=provider, model="mock-1").use_pack("kyc")
    app.set_policy("answer_only_from_sources", False).set_policy("require_citations", False)
    result = app.run("Screen this customer")
    assert result.error is None
    output = result.output
    output = output.model_dump() if hasattr(output, "model_dump") else output
    assert output["risk_rating"] == "high"
    assert output["sanctions_hit"] is True


def test_vertical_pack_apply_is_idempotent():
    app = ContextApp(name="t", provider=MockProvider(), model="mock-1")
    app.use_pack("kyc")
    app.use_pack("kyc")
    names = [r.name for r in app.rail_engine.rails]
    assert names.count("pii_redact") == 1
