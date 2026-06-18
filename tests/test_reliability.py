"""Reliability tests: constrained generation, streaming validation,
typed signatures, rails, self-correction, multi-schema routing, and the
provider reliability fixes."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel

from vincio import ContextApp, Rail
from vincio.core.types import ModelResponse, RunStatus
from vincio.output import (
    DecodingMode,
    OutputContract,
    OutputSchema,
    OutputValidator,
    SchemaRouter,
    SelfCorrector,
    StreamingValidator,
    choice_schema,
    negotiate_decoding,
    regex_schema,
    to_strict_json_schema,
    validate_partial,
)
from vincio.prompts import InputField, OutputField, Predict, Signature, signature
from vincio.prompts.optimizers import generate_variants
from vincio.providers import MockProvider
from vincio.providers.base import HTTPProvider, _retry_delay_from_body
from vincio.security.rails import RailEngine


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str = "USD"


class Ticket(BaseModel):
    label: str
    confidence: float


# ---------------------------------------------------------------------------
# constrained generation
# ---------------------------------------------------------------------------


class TestConstrainedGeneration:
    def test_strict_schema_closes_objects_and_requires_all(self):
        strict = to_strict_json_schema(Invoice.model_json_schema())
        assert strict["additionalProperties"] is False
        assert sorted(strict["required"]) == ["currency", "total", "vendor"]
        # The optional field became nullable instead of absent.
        assert strict["properties"]["currency"]["type"] == ["string", "null"]
        assert "default" not in strict["properties"]["currency"]

    def test_strict_schema_recurses_into_nested_defs(self):
        class Line(BaseModel):
            description: str
            amount: float | None = None

        class Order(BaseModel):
            lines: list[Line]

        strict = to_strict_json_schema(Order.model_json_schema())
        line = strict["$defs"]["Line"]
        assert line["additionalProperties"] is False
        assert sorted(line["required"]) == ["amount", "description"]
        # amount was Optional → its anyOf keeps the null branch, unchanged.
        assert any(o.get("type") == "null" for o in line["properties"]["amount"]["anyOf"])

    def test_strict_transform_does_not_mutate_original(self):
        original = Invoice.model_json_schema()
        snapshot = json.dumps(original, sort_keys=True)
        to_strict_json_schema(original)
        assert json.dumps(original, sort_keys=True) == snapshot

    def test_negotiate_decoding(self):
        capable = MockProvider().capabilities("mock-1")
        assert negotiate_decoding(capable, {"type": "object"}) is DecodingMode.NATIVE
        assert negotiate_decoding(capable, None) is DecodingMode.NONE
        incapable = capable.model_copy(update={"structured_output": False})
        assert negotiate_decoding(incapable, {"type": "object"}) is DecodingMode.PROMPT

    def test_choice_and_regex_schemas_validate(self):
        schema = OutputSchema.from_json_schema(choice_schema(["bug", "billing"]))
        assert schema.is_valid({"choice": "bug"})
        assert not schema.is_valid({"choice": "feature"})
        pattern = OutputSchema.from_json_schema(regex_schema(r"^INV-\d{4}$"))
        assert pattern.is_valid({"value": "INV-0042"})
        assert not pattern.is_valid({"value": "42"})

    @pytest.mark.asyncio
    async def test_runtime_sends_strict_schema_natively(self, offline_config, tmp_cwd):
        provider = MockProvider()
        app = ContextApp(
            name="t", provider=provider, model="mock-1",
            output_schema=Invoice, config=offline_config,
        )
        result = await app.arun("Extract the invoice")
        assert result.status == RunStatus.SUCCEEDED
        sent = provider.requests[-1].output_schema
        assert sent["additionalProperties"] is False
        assert sorted(sent["required"]) == ["currency", "total", "vendor"]
        # Validation ran against the original schema and produced a typed model.
        assert isinstance(result.output, Invoice)


# ---------------------------------------------------------------------------
# streaming validation
# ---------------------------------------------------------------------------


class TestStreamingValidation:
    def test_validate_partial_tolerates_missing_required(self):
        schema = Invoice.model_json_schema()
        assert validate_partial({"vendor": "Acme"}, schema) == []

    def test_validate_partial_catches_definite_mismatch(self):
        schema = Invoice.model_json_schema()
        errors = validate_partial({"vendor": "Acme", "total": "twelve"}, schema)
        assert errors and "total" in errors[0]

    def test_validate_partial_unknown_field_on_closed_object(self):
        schema = to_strict_json_schema(Invoice.model_json_schema())
        errors = validate_partial({"surprise": 1}, schema)
        assert errors and "unknown field" in errors[0]

    def test_feed_and_finalize(self):
        validator = StreamingValidator(
            OutputSchema.from_pydantic(Invoice), min_interval_chars=8
        )
        events = [
            e
            for e in (
                validator.feed('{"vendor": "Ac'),
                validator.feed('me", "total": 12'),
                validator.feed('.5, "currency": "EUR"}'),
            )
            if e is not None
        ]
        assert events, "expected at least one mid-stream parse"
        assert all(e.valid_prefix for e in events)
        final = validator.finalize()
        assert final.valid_prefix
        assert final.data == {"vendor": "Acme", "total": 12.5, "currency": "EUR"}

    def test_invalid_prefix_detected_mid_stream(self):
        validator = StreamingValidator(
            OutputSchema.from_pydantic(Invoice), min_interval_chars=1
        )
        event = validator.feed('{"vendor": "Acme", "total": true')
        assert event is not None
        assert not event.valid_prefix

    def test_finalize_repairs_structure(self):
        schema = OutputSchema.from_json_schema(
            {
                "type": "object",
                "properties": {"vendor": {"type": "string"}, "total": {"type": "number"}},
                "required": ["vendor", "total"],
            },
            name="invoice",
        )
        validator = StreamingValidator(schema)
        validator.feed('{"vendor": "Acme", "total": "12.5"}')
        final = validator.finalize()
        assert final.repaired
        assert final.data["total"] == 12.5

    @pytest.mark.asyncio
    async def test_astream_emits_valid_prefix_events(self, offline_config, tmp_cwd):
        app = ContextApp(
            name="t", provider=MockProvider(), model="mock-1",
            output_schema=Invoice, config=offline_config,
        )
        partials = [
            event
            async for event in app.astream("Extract the invoice")
            if event.type == "partial_output"
        ]
        assert partials
        assert all(event.valid_prefix is not None for event in partials)
        assert all(event.valid_prefix for event in partials)


# ---------------------------------------------------------------------------
# typed signatures
# ---------------------------------------------------------------------------


class Triage(Signature):
    """Classify a support ticket."""

    ticket: str = InputField(desc="the raw ticket text")
    label: str = OutputField(desc="bug | billing | feature | other")
    confidence: float = OutputField()


class TestSignatures:
    def test_field_partition(self):
        assert list(Triage.input_fields()) == ["ticket"]
        assert list(Triage.output_fields()) == ["label", "confidence"]

    def test_output_schema(self):
        schema = Triage.output_schema()
        assert schema.name == "Triage"
        assert sorted(schema.json_schema["required"]) == ["confidence", "label"]

    def test_to_prompt_spec_is_optimization_target(self):
        spec = Triage.to_prompt_spec()
        assert spec.objective == "Classify a support ticket."
        assert spec.output_schema is not None
        variants = generate_variants(spec, max_variants=6)
        assert variants  # signatures feed the prompt optimizer directly

    def test_render_inputs_type_checks(self):
        from vincio.core.errors import PromptError

        assert "ticket: the app crashes" in Triage.render_inputs(ticket="the app crashes")
        with pytest.raises(PromptError):
            Triage.render_inputs(ticket=42)
        with pytest.raises(PromptError):
            Triage.render_inputs()
        with pytest.raises(PromptError):
            Triage.render_inputs(ticket="x", extra="y")

    def test_string_signature(self):
        QA = signature("question, context -> answer, confidence: float", name="QA")
        assert list(QA.input_fields()) == ["question", "context"]
        assert list(QA.output_fields()) == ["answer", "confidence"]

    def test_string_signature_rejects_garbage(self):
        from vincio.core.errors import PromptError

        with pytest.raises(PromptError):
            signature("no separator here")
        with pytest.raises(PromptError):
            signature("a -> b: tuple")

    def test_predict_returns_typed_result(self):
        predict = Predict(Triage, provider=MockProvider(), model="mock-1")
        result = predict(ticket="The export button 500s")
        assert isinstance(result.label, str)
        assert isinstance(result.confidence, float)
        assert result.report.valid

    def test_app_predictor(self, offline_config, tmp_cwd):
        app = ContextApp(name="t", provider=MockProvider(), model="mock-1", config=offline_config)
        result = app.predictor(Triage)(ticket="Refund my invoice")
        assert result.output.label


# ---------------------------------------------------------------------------
# rails
# ---------------------------------------------------------------------------


class TestRails:
    def test_topic_rail_blocks(self):
        engine = RailEngine()
        engine.add(Rail(name="no_legal", kind="topic", blocked_topics=["legal advice"]))
        check = engine.check("Can you give me legal advice on this?", direction="input")
        assert not check.allowed
        assert check.violations[0].rail == "no_legal"

    def test_allowed_topics_rail(self):
        engine = RailEngine()
        engine.add(
            Rail(name="on_topic", kind="topic", allowed_topics=["refund", "invoice"], direction="input")
        )
        assert engine.check("What about my refund?", direction="input").allowed
        assert not engine.check("Tell me a story about dragons", direction="input").allowed

    def test_format_rail(self):
        engine = RailEngine()
        engine.add(Rail(name="short", kind="format", max_chars=10, direction="output"))
        assert engine.check("ok", direction="output").allowed
        assert not engine.check("this is far too long", direction="output").allowed

    def test_safety_rail_reuses_detectors(self):
        engine = RailEngine()
        engine.add(Rail(name="no_pii", kind="safety", detectors=["pii"]))
        check = engine.check("Reach me at jane.doe@example.com", direction="output")
        assert not check.allowed
        assert "pii" in check.violations[0].details

    def test_redact_rail_transforms_text(self):
        engine = RailEngine()
        engine.add(Rail(name="mask_pii", kind="safety", action="redact", detectors=["pii"]))
        check = engine.check("Reach me at jane.doe@example.com", direction="output")
        assert check.allowed  # redact, not block
        assert check.transformed_text is not None
        assert "jane.doe@example.com" not in check.transformed_text

    def test_custom_rail_predicate(self):
        engine = RailEngine()
        engine.register("too_many_words", lambda text, params: (
            f"more than {params['limit']} words" if len(text.split()) > params["limit"] else None
        ))
        engine.add(Rail(name="brevity", kind="custom", predicate="too_many_words", params={"limit": 3}))
        assert engine.check("one two three", direction="output").allowed
        check = engine.check("one two three four", direction="output")
        assert not check.allowed
        assert "more than 3 words" in check.violations[0].message

    def test_unregistered_predicate_rejected(self):
        from vincio.core.errors import SecurityError

        with pytest.raises(SecurityError):
            RailEngine().add(Rail(name="x", kind="custom", predicate="missing"))

    @pytest.mark.asyncio
    async def test_input_rail_denies_run_and_audits(self, offline_config, tmp_cwd):
        app = ContextApp(name="t", provider=MockProvider(), model="mock-1", config=offline_config)
        app.add_rail(name="no_crypto", kind="topic", direction="input", blocked_topics=["crypto"])
        result = await app.arun("Give me crypto investment tips")
        assert result.status == RunStatus.DENIED
        assert "blocked topic" in (result.error or "")
        denied = [e for e in app.audit.entries if e.action == "run" and e.decision == "deny"]
        assert denied and any("rail:no_crypto" in str(e.details) for e in denied)

    @pytest.mark.asyncio
    async def test_output_rail_blocks_via_validation(self, offline_config, tmp_cwd):
        app = ContextApp(
            name="t",
            provider=MockProvider(default_text="our competitor AcmeCorp is great"),
            model="mock-1",
            config=offline_config,
        )
        app.add_rail(
            name="no_competitors", kind="topic", direction="output", blocked_topics=["acmecorp"]
        )
        result = await app.arun("Summarize")
        assert result.status == RunStatus.FAILED
        assert any(
            step["name"] == "policy" and not step["passed"]
            for step in result.validation["steps"]
        )


# ---------------------------------------------------------------------------
# self-correction
# ---------------------------------------------------------------------------


def _invoice_validator() -> OutputValidator:
    schema = OutputSchema.from_pydantic(Invoice)
    return OutputValidator(OutputContract.from_schema(schema), schema=schema)


class TestSelfCorrection:
    @pytest.mark.asyncio
    async def test_corrects_invalid_output(self):
        provider = MockProvider(
            script=[ModelResponse(text='{"vendor": "Acme", "total": 12.5, "currency": "USD"}')]
        )
        corrector = SelfCorrector(
            _invoice_validator(), provider=provider, model="mock-1", max_cycles=2
        )
        result = await corrector.correct('{"vendor": "Acme"}')
        assert result.valid
        assert result.cycles == 1
        assert result.stopped_reason == "valid"
        assert result.critiques and "schema" in result.critiques[0]

    @pytest.mark.asyncio
    async def test_valid_output_short_circuits(self):
        provider = MockProvider()
        corrector = SelfCorrector(_invoice_validator(), provider=provider, model="mock-1")
        result = await corrector.correct('{"vendor": "Acme", "total": 1.0, "currency": "USD"}')
        assert result.valid
        assert result.cycles == 0
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_max_cycles_bounds_the_loop(self):
        provider = MockProvider(responder=lambda request: ModelResponse(text="still not json"))
        corrector = SelfCorrector(
            _invoice_validator(), provider=provider, model="mock-1", max_cycles=2
        )
        result = await corrector.correct("garbage")
        assert not result.valid
        assert result.cycles == 2
        assert result.stopped_reason == "max_cycles"
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_cost_ceiling_stops_the_loop(self):
        def costly(request):
            return ModelResponse(text="nope", cost_usd=1.0)

        provider = MockProvider(responder=costly)
        corrector = SelfCorrector(
            _invoice_validator(), provider=provider, model="mock-1",
            max_cycles=5, max_cost_usd=0.5,
        )
        result = await corrector.correct("garbage")
        assert not result.valid
        assert result.stopped_reason == "cost_ceiling"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_app_self_correction_recovers_run(self, offline_config, tmp_cwd):
        provider = MockProvider(
            script=[
                ModelResponse(text='{"vendor": "Acme"}'),  # fails validation
                ModelResponse(text='{"vendor": "Acme", "total": 9.0, "currency": "USD"}'),
            ]
        )
        app = ContextApp(
            name="t", provider=provider, model="mock-1",
            output_schema=Invoice, config=offline_config,
        )
        app.enable_self_correction(max_cycles=1)
        result = await app.arun("Extract")
        assert result.status == RunStatus.SUCCEEDED
        assert isinstance(result.output, Invoice)
        assert result.output.total == 9.0
        # Repair landed in the audit log (decision=repair).
        assert any(
            e.action == "output_validation" and e.decision == "repair"
            for e in app.audit.entries
        )


# ---------------------------------------------------------------------------
# multi-schema routing
# ---------------------------------------------------------------------------


class TestSchemaRouting:
    def test_keyword_routing(self):
        router = SchemaRouter()
        router.add(Invoice, keywords=["invoice", "vendor"])
        router.add(Ticket, keywords=["bug", "crash"])
        assert router.route("Extract this invoice").name == "Invoice"
        assert router.route("The app crashed again").name == "Ticket"
        assert router.route("Hello there") is None

    def test_task_type_routing(self):
        router = SchemaRouter()
        router.add(Invoice, task_types=["extraction"])
        assert router.route("anything", task_type="extraction").name == "Invoice"
        assert router.route("anything", task_type="general") is None

    def test_content_classification_and_validate_any(self):
        router = SchemaRouter()
        router.add(Invoice)
        router.add(Ticket)
        name, validated = router.validate_any({"label": "bug", "confidence": 0.9})
        assert name == "Ticket"
        assert isinstance(validated, Ticket)
        from vincio.core.errors import OutputSchemaError

        with pytest.raises(OutputSchemaError):
            router.validate_any({"neither": True})

    @pytest.mark.asyncio
    async def test_app_routes_schema_by_keywords(self, offline_config, tmp_cwd):
        app = ContextApp(name="t", provider=MockProvider(), model="mock-1", config=offline_config)
        app.add_output_schema(Invoice, keywords=["invoice"])
        app.add_output_schema(Ticket, keywords=["bug"])
        result = await app.arun("There is a bug in the export")
        assert result.status == RunStatus.SUCCEEDED
        assert isinstance(result.output, Ticket)
        result = await app.arun("Process this invoice please")
        assert isinstance(result.output, Invoice)


# ---------------------------------------------------------------------------
# provider reliability fixes (shipped with 0.7)
# ---------------------------------------------------------------------------


class _DummyHTTPProvider(HTTPProvider):
    name = "dummy"
    requires_api_key = False

    async def generate(self, request):  # pragma: no cover - not exercised
        raise NotImplementedError

    async def stream(self, request):  # pragma: no cover - not exercised
        raise NotImplementedError

    async def embed(self, texts, model=None):  # pragma: no cover - not exercised
        raise NotImplementedError

    def capabilities(self, model):  # pragma: no cover - not exercised
        raise NotImplementedError


class TestProviderReliabilityFixes:
    def test_retry_delay_parsed_from_google_error_body(self):
        body = {
            "error": {
                "message": "Resource exhausted. Please retry in 29.1s.",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "29s",
                    }
                ],
            }
        }
        assert _retry_delay_from_body(body) == 29.0
        no_details = {"error": {"message": "quota; retry in 12.5s"}}
        assert _retry_delay_from_body(no_details) == 12.5
        assert _retry_delay_from_body({"error": {"message": "nope"}}) is None
        assert _retry_delay_from_body(None) is None

    def test_client_recreated_across_event_loops(self):
        provider = _DummyHTTPProvider(api_key=None)

        async def get_client():
            return provider.client

        first = asyncio.run(get_client())
        second = asyncio.run(get_client())
        assert first is not second  # a client bound to a dead loop is replaced

        async def same_loop():
            return provider.client, provider.client

        a, b = asyncio.run(same_loop())
        assert a is b  # within one loop the client is reused
