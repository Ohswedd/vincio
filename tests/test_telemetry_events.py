"""2.0 async-first storage, typed event catalog, and unified OTel telemetry."""

from __future__ import annotations

import logging

from vincio.core.events import (
    EVENT_CATALOG,
    EVENT_SCHEMA_VERSION,
    EgressDLP,
    Event,
    EventBus,
    RunCompleted,
    payload_model_for,
)
from vincio.observability.otel import _genai_span
from vincio.observability.spans import Span
from vincio.storage.base import (
    InMemoryMetadataStore,
    acount,
    adelete,
    aget,
    aquery,
    asave,
)

# -- typed event catalog ---------------------------------------------------


def test_event_carries_schema_version():
    assert Event(name="x").schema_version == EVENT_SCHEMA_VERSION


def test_catalog_lookup_and_models():
    assert payload_model_for("run.completed") is RunCompleted
    assert payload_model_for("security.egress_dlp") is EgressDLP
    assert payload_model_for("does.not.exist") is None
    assert "run.completed" in EVENT_CATALOG


def test_publish_typed_payload():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe("run.completed", seen.append)
    event = bus.publish(RunCompleted(run_id="r1", status="succeeded"))
    assert event.name == "run.completed"
    assert event.payload["run_id"] == "r1"
    assert seen and seen[0].payload["status"] == "succeeded"


def test_payload_extra_fields_allowed():
    # Forward-compat: extra keys do not reject a documented payload.
    payload = RunCompleted(run_id="r", status="ok", extra_dim="anything")
    assert payload.model_dump()["extra_dim"] == "anything"


def test_emit_validates_leniently_against_catalog(caplog):
    bus = EventBus()
    delivered: list[Event] = []
    bus.subscribe("run.completed", delivered.append)
    # A malformed payload (missing required run_id) logs a warning but still emits.
    with caplog.at_level(logging.WARNING, logger="vincio.events"):
        event = bus.emit("run.completed", {"status": "ok"})
    assert event.payload == {"status": "ok"}  # still delivered
    assert delivered
    assert any("catalog schema" in r.message for r in caplog.records)


def test_emit_untyped_event_unaffected():
    bus = EventBus()
    got: list[Event] = []
    bus.subscribe("custom.thing", got.append)
    bus.emit("custom.thing", {"whatever": 1})
    assert got[0].payload == {"whatever": 1}


# -- async-first storage ---------------------------------------------------


async def test_async_helpers_over_sync_store_threaded():
    store = InMemoryMetadataStore()
    await asave(store, "runs", {"id": "r1", "status": "ok"})
    assert (await aget(store, "runs", "r1"))["status"] == "ok"
    rows = await aquery(store, "runs", where={"status": "ok"})
    assert len(rows) == 1
    assert await acount(store, "runs") == 1
    assert await adelete(store, "runs", "r1") is True
    assert await acount(store, "runs") == 0


async def test_async_helpers_prefer_native_async_methods():
    calls: list[str] = []

    class AsyncNativeStore:
        async def asave(self, kind, record):
            calls.append("asave")

        async def aget(self, kind, record_id):
            calls.append("aget")
            return {"id": record_id}

        async def aquery(self, kind, *, where=None, limit=100, offset=0):
            calls.append("aquery")
            return []

        async def adelete(self, kind, record_id):
            calls.append("adelete")
            return True

        async def acount(self, kind):
            calls.append("acount")
            return 0

    store = AsyncNativeStore()
    await asave(store, "runs", {"id": "r"})
    await aget(store, "runs", "r")
    await aquery(store, "runs")
    await adelete(store, "runs", "r")
    await acount(store, "runs")
    assert calls == ["asave", "aget", "aquery", "adelete", "acount"]


# -- unified telemetry: OTel agentic conventions ---------------------------


def _span(span_type: str, name: str = "s", **attrs) -> Span:
    return Span(name=name, type=span_type, attributes=attrs)


def test_genai_model_span_includes_cost_and_tokens():
    name, attrs = _genai_span(
        _span("model_call", model="claude-opus-4-8", input_tokens=100, output_tokens=20, cost_usd=0.0123, finish="stop")
    )
    assert name == "chat claude-opus-4-8"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["gen_ai.usage.cost"] == 0.0123
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]


def test_genai_tool_span():
    name, attrs = _genai_span(_span("tool_call", name="search", tool="search"))
    assert name == "execute_tool search"
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "search"


def test_genai_agent_span_uses_invoke_agent_convention():
    name, attrs = _genai_span(_span("crew_agent", name="researcher", agent="researcher", agent_id="a1"))
    assert name == "invoke_agent researcher"
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.agent.name"] == "researcher"
    assert attrs["gen_ai.agent.id"] == "a1"


def test_genai_graph_node_maps_to_invoke_agent():
    name, attrs = _genai_span(_span("graph_node", name="plan"))
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert name == "invoke_agent plan"
