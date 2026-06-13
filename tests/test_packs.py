"""0.9 domain packs: loading, application, and an end-to-end run."""

from __future__ import annotations

import json

import pytest

from vincio import ContextApp
from vincio.core.errors import ConfigError
from vincio.packs import Pack, available_packs, load_pack, register_pack
from vincio.providers import MockProvider

BUILTINS = ["support", "engineering", "finance", "legal"]


def test_available_packs_lists_builtins():
    assert set(BUILTINS) <= set(available_packs())


@pytest.mark.parametrize("name", BUILTINS)
def test_builtin_pack_is_well_formed(name):
    pack = load_pack(name)
    assert pack.name == name
    assert pack.output_schema and pack.output_schema["type"] == "object"
    assert pack.role and pack.objective
    dataset = pack.dataset()
    assert len(dataset) >= 1
    spec = pack.prompt_spec()
    assert spec.output_format == "json"


def test_load_unknown_pack_raises():
    with pytest.raises(ConfigError):
        load_pack("astrology")


def test_use_pack_configures_app():
    app = ContextApp(name="helpdesk", provider=MockProvider(), model="mock-1")
    app.use_pack("support")
    assert "support" in app.prompt_spec.role or app.prompt_spec.role == "customer support assistant"
    assert app.prompt_spec.objective.startswith("Classify the ticket")
    assert app.policies.answer_only_from_sources is True
    assert app.output_contract.schema_name == "support_resolution"
    assert "groundedness" in app.evaluators


def test_use_pack_set_schema_false_keeps_contract():
    app = ContextApp(name="x", provider=MockProvider(), model="mock-1")
    before = app.output_contract.schema_name
    app.use_pack("legal", set_schema=False)
    assert app.output_contract.schema_name == before
    assert app.policies.require_citations is True  # policies still applied


def test_register_and_apply_custom_pack():
    custom = Pack(name="my_domain", description="x", role="r", objective="o", rules=["only facts"])
    register_pack(custom)
    assert "my_domain" in available_packs()
    app = ContextApp(name="c", provider=MockProvider(), model="mock-1").use_pack("my_domain")
    assert app.prompt_spec.role == "r"


def test_use_pack_rejects_bad_type():
    app = ContextApp(name="x", provider=MockProvider(), model="mock-1")
    with pytest.raises(ConfigError):
        app.use_pack(123)


def test_use_pack_rails_are_idempotent():
    # Re-applying a pack must not install its rail twice (it would be
    # evaluated twice on every generation).
    app = ContextApp(name="x", provider=MockProvider(), model="mock-1")
    app.use_pack("finance")
    app.use_pack("finance")
    names = [rail.name for rail in app.rail_engine.rails]
    assert names.count("no_pii_leak") == 1


def test_add_evaluator_callable_without_name_is_consistent():
    # A callable lacking __name__ (e.g. functools.partial) must register the
    # metric under the same key it records in app.evaluators.
    from functools import partial

    from vincio.evals.metrics import METRICS

    app = ContextApp(name="x", provider=MockProvider(), model="mock-1")

    def base(output, case):  # a real metric signature
        return 1.0

    app.add_evaluator(partial(base))
    registered = app.evaluators[-1]
    assert registered in METRICS


def test_run_with_support_pack_enforces_schema():
    payload = {
        "category": "billing",
        "priority": "high",
        "response": "We have refunded the duplicate charge.",
        "needs_human": False,
    }
    provider = MockProvider(responder=lambda request: json.dumps(payload))
    app = ContextApp(name="helpdesk", provider=provider, model="mock-1").use_pack("support")
    # The pack requires citations + source-grounding; relax both here to isolate
    # schema enforcement (citation behavior is covered by the output tests).
    app.set_policy("answer_only_from_sources", False).set_policy("require_citations", False)
    result = app.run("I was charged twice this month")
    assert result.error is None
    output = result.output
    output = output.model_dump() if hasattr(output, "model_dump") else output
    assert output["category"] == "billing"
    assert output["needs_human"] is False
