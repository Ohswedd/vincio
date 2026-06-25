"""Real-behavior coverage tests for ``vincio.observability.otel``.

The OpenTelemetry SDK is an *optional* dependency. Offline (no ``opentelemetry``
installed) two things are deterministically exercisable through the real API:

* :func:`vincio.observability.otel._genai_span` — the pure mapping from a Vincio
  :class:`~vincio.observability.spans.Span` to an OTel GenAI span name +
  ``gen_ai.*`` semantic-convention attribute set. Every span-type branch and
  every optional-attribute branch is asserted on exact values here.
* :class:`~vincio.observability.otel.OTelExporter` — constructing it without the
  SDK installed must raise a :class:`~vincio.core.errors.ConfigError` with the
  install hint, never a bare ``ImportError``.

The span-emission / metrics-histogram code paths (``_export`` / ``_record_metrics``)
require the live SDK and are intentionally not exercised here.
"""

from __future__ import annotations

import pytest

from vincio.core.errors import ConfigError
from vincio.observability.otel import OTelExporter, _genai_span
from vincio.observability.spans import Span


def _span(span_type: str, *, name: str = "s", **attributes: object) -> Span:
    return Span(name=name, type=span_type, attributes=dict(attributes))


# --------------------------------------------------------------------------- #
# model_call -> "chat {model}" with gen_ai.* chat attributes
# --------------------------------------------------------------------------- #
def test_model_call_base_attributes_and_name() -> None:
    name, attrs = _genai_span(_span("model_call", model="gpt-4o"))
    assert name == "chat gpt-4o"
    assert attrs == {
        "gen_ai.operation.name": "chat",
        "gen_ai.system": "vincio",
        "gen_ai.request.model": "gpt-4o",
    }


def test_model_call_missing_model_falls_back_to_unknown() -> None:
    name, attrs = _genai_span(_span("model_call"))
    assert name == "chat unknown"
    assert attrs["gen_ai.request.model"] == "unknown"


def test_model_call_empty_string_model_falls_back_to_unknown() -> None:
    # `span.attributes.get("model") or "unknown"` -> empty string is falsy.
    name, attrs = _genai_span(_span("model_call", model=""))
    assert name == "chat unknown"
    assert attrs["gen_ai.request.model"] == "unknown"


def test_model_call_includes_all_usage_signals_with_coercion() -> None:
    name, attrs = _genai_span(
        _span(
            "model_call",
            model="claude",
            input_tokens="120",  # string -> int(...) coercion
            output_tokens=34,
            cost_usd="0.0021",  # string -> float(...) coercion
            finish="stop",
        )
    )
    assert name == "chat claude"
    assert attrs["gen_ai.usage.input_tokens"] == 120
    assert isinstance(attrs["gen_ai.usage.input_tokens"], int)
    assert attrs["gen_ai.usage.output_tokens"] == 34
    assert attrs["gen_ai.usage.cost"] == pytest.approx(0.0021)
    assert isinstance(attrs["gen_ai.usage.cost"], float)
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]


def test_model_call_zero_tokens_are_still_emitted() -> None:
    # 0 is not None, so the `is not None` guard must keep it (a `if tokens:`
    # truthiness bug would silently drop a legitimate zero count).
    _, attrs = _genai_span(_span("model_call", model="m", input_tokens=0, output_tokens=0))
    assert attrs["gen_ai.usage.input_tokens"] == 0
    assert attrs["gen_ai.usage.output_tokens"] == 0


def test_model_call_zero_cost_is_emitted() -> None:
    _, attrs = _genai_span(_span("model_call", model="m", cost_usd=0.0))
    assert attrs["gen_ai.usage.cost"] == 0.0


def test_model_call_omits_absent_optional_attributes() -> None:
    # No tokens / cost / finish provided -> none of those keys appear.
    _, attrs = _genai_span(_span("model_call", model="m"))
    for absent in (
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.cost",
        "gen_ai.response.finish_reasons",
    ):
        assert absent not in attrs


def test_model_call_empty_finish_string_is_dropped() -> None:
    # `if span.attributes.get("finish")` -> empty string is falsy, so no
    # finish_reasons key is emitted.
    _, attrs = _genai_span(_span("model_call", model="m", finish=""))
    assert "gen_ai.response.finish_reasons" not in attrs


# --------------------------------------------------------------------------- #
# tool_call -> "execute_tool {tool}"
# --------------------------------------------------------------------------- #
def test_tool_call_uses_tool_attribute() -> None:
    name, attrs = _genai_span(_span("tool_call", name="span-name", tool="web_search"))
    assert name == "execute_tool web_search"
    assert attrs == {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.system": "vincio",
        "gen_ai.tool.name": "web_search",
    }


def test_tool_call_falls_back_to_span_name_without_tool_attr() -> None:
    name, attrs = _genai_span(_span("tool_call", name="calculator"))
    assert name == "execute_tool calculator"
    assert attrs["gen_ai.tool.name"] == "calculator"


# --------------------------------------------------------------------------- #
# agent-family span types -> "invoke_agent {name}"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("span_type", ["crew", "crew_agent", "graph_node", "compose_node"])
def test_agent_family_types_map_to_invoke_agent(span_type: str) -> None:
    name, attrs = _genai_span(_span(span_type, agent="Researcher"))
    assert name == "invoke_agent Researcher"
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.system"] == "vincio"
    assert attrs["gen_ai.agent.name"] == "Researcher"


def test_agent_name_resolution_prefers_agent_over_role_node_name() -> None:
    name, attrs = _genai_span(
        _span("crew", name="span", agent="A", role="B", node="C"),
    )
    assert name == "invoke_agent A"
    assert attrs["gen_ai.agent.name"] == "A"


def test_agent_name_falls_back_to_role_then_node_then_span_name() -> None:
    # role used when agent absent
    _, role_attrs = _genai_span(_span("crew", name="span", role="Planner", node="N"))
    assert role_attrs["gen_ai.agent.name"] == "Planner"
    # node used when agent + role absent
    _, node_attrs = _genai_span(_span("graph_node", name="span", node="ingest"))
    assert node_attrs["gen_ai.agent.name"] == "ingest"
    # span name used when none of agent/role/node present
    _, name_attrs = _genai_span(_span("compose_node", name="finalize"))
    assert name_attrs["gen_ai.agent.name"] == "finalize"


def test_agent_id_attribute_from_agent_id_key() -> None:
    _, attrs = _genai_span(_span("graph_node", agent="A", agent_id=42))
    assert attrs["gen_ai.agent.id"] == "42"  # coerced to str


def test_agent_id_attribute_from_id_key_fallback() -> None:
    _, attrs = _genai_span(_span("crew_agent", agent="A", id="x-7"))
    assert attrs["gen_ai.agent.id"] == "x-7"


def test_agent_id_omitted_when_no_id_present() -> None:
    _, attrs = _genai_span(_span("compose_node", agent="A"))
    assert "gen_ai.agent.id" not in attrs


# --------------------------------------------------------------------------- #
# fallback: any other span type -> "{type}:{name}" with no gen_ai attributes
# --------------------------------------------------------------------------- #
def test_unmapped_span_type_uses_type_colon_name_and_no_genai_attrs() -> None:
    name, attrs = _genai_span(_span("retrieval", name="vector_lookup"))
    assert name == "retrieval:vector_lookup"
    assert attrs == {}


def test_custom_span_type_fallback() -> None:
    name, attrs = _genai_span(_span("custom", name="thing"))
    assert name == "custom:thing"
    assert attrs == {}


# --------------------------------------------------------------------------- #
# OTelExporter requires the optional SDK
# --------------------------------------------------------------------------- #
def test_exporter_without_sdk_raises_configerror_with_install_hint() -> None:
    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        pass
    else:  # pragma: no cover - only when the optional SDK is installed
        pytest.skip("opentelemetry SDK installed; ConfigError path not reachable")
    with pytest.raises(ConfigError, match=r'vincio\[otel\]'):
        OTelExporter()


def test_exporter_configerror_message_mentions_opentelemetry() -> None:
    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        pass
    else:  # pragma: no cover - only when the optional SDK is installed
        pytest.skip("opentelemetry SDK installed; ConfigError path not reachable")
    with pytest.raises(ConfigError, match="OpenTelemetry export requires"):
        OTelExporter(service_name="custom", content_policy=None)
