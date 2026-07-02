"""DS4 local-inference provider: a running ds4-server (self-hosted DeepSeek V4)
as a first-class Vincio provider — OpenAI-compatible transport, thinking modes
driven by the reasoning controller, disk-KV accounting, honest $0 self-hosted
pricing, and fail-closed on-prem residency. All offline via an injected client."""

from __future__ import annotations

import json

import pytest

from vincio.core.errors import ConfigError
from vincio.core.types import Message, ModelRequest
from vincio.governance.residency import (
    ResidencyPolicy,
    infer_region_from_url,
    residency_violation,
)
from vincio.providers import Ds4Provider, build_provider, openai_compatible
from vincio.providers.cache_strategy import cache_hit_rate
from vincio.providers.openai_compat import PRESETS
from vincio.providers.registry import ModelRegistry

DS4_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-flash-q4", "deepseek-v4-pro-q4")


# -- fake transport --------------------------------------------------------


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)
        self.headers: dict[str, str] = {}

    def json(self) -> dict:
        return self._payload


class _StreamCtx:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status_code = 200

    async def __aenter__(self) -> _StreamCtx:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b""


class _FakeClient:
    """Records the last request and returns a canned chat or SSE response."""

    def __init__(self, *, response: dict | None = None, lines: list[str] | None = None) -> None:
        self.is_closed = False
        self._response = response
        self._lines = lines or []
        self.last_url: str | None = None
        self.last_headers: dict[str, str] = {}
        self.last_json: dict | None = None

    async def post(self, url, *, headers=None, content=None, json=None):  # noqa: A002
        self.last_url = url
        self.last_headers = headers or {}
        self.last_json = json
        return _Resp(self._response or {})

    def stream(self, method, url, *, headers=None, content=None, json=None):  # noqa: A002
        self.last_url = url
        self.last_headers = headers or {}
        self.last_json = json
        return _StreamCtx(self._lines)

    async def aclose(self) -> None:
        self.is_closed = True


def _chat_response(**over) -> dict:
    base = {
        "id": "resp-1",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "message": {
                    "content": "Bordeaux is in France.",
                    # DeepSeek returns the thinking trace separately; it must NOT
                    # leak into the answer text.
                    "reasoning_content": "The user asks where Bordeaux is...",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 6,
            "prompt_cache_hit_tokens": 32,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    }
    base.update(over)
    return base


# -- registration ----------------------------------------------------------


def test_build_provider_resolves_first_class_ds4():
    provider = build_provider("ds4", with_retries=False)
    assert isinstance(provider, Ds4Provider)
    assert provider.name == "ds4"
    assert provider.base_url == "http://127.0.0.1:8000/v1"
    assert provider.requires_api_key is False


def test_ds4_registered_and_preset_present():
    from vincio.providers import _registry

    assert "ds4" in _registry.names
    assert "ds4" in PRESETS
    # The first-class provider owns the name; the preset factory did not shadow it.
    assert isinstance(build_provider("ds4", with_retries=False), Ds4Provider)


def test_openai_compatible_ds4_preset_is_keyless_passthrough():
    provider = openai_compatible("ds4")
    # Generic passthrough (not the first-class subclass), pointed at the DS4 box,
    # and — crucially — keyless, so it never demands an API key at call time.
    assert provider.name == "ds4"
    assert provider.base_url == "http://127.0.0.1:8000/v1"
    assert provider.requires_api_key is False


def test_other_presets_still_require_a_key():
    # The requires_api_key wiring is preset-specific, not a blanket change.
    assert openai_compatible("groq", api_key="k").requires_api_key is True


# -- headers / auth --------------------------------------------------------


def test_headers_are_keyless_by_default():
    headers = Ds4Provider()._headers()
    assert headers == {"Content-Type": "application/json"}


def test_headers_attach_bearer_when_a_key_is_set():
    headers = Ds4Provider(api_key="proxy-token")._headers()
    assert headers["Authorization"] == "Bearer proxy-token"


# -- payload: thinking modes ----------------------------------------------


def test_plain_request_turns_thinking_off_explicitly():
    payload = Ds4Provider()._payload(
        ModelRequest(model="deepseek-v4-flash", messages=[Message(role="user", content="hi")])
    )
    assert payload["chat_template_kwargs"] == {"thinking": False}
    assert "reasoning_effort" not in payload


def test_reasoning_request_drives_thinking_on_with_budget():
    payload = Ds4Provider()._payload(
        ModelRequest(
            model="deepseek-v4-flash",
            messages=[Message(role="user", content="prove it")],
            reasoning_effort="high",
            thinking_budget_tokens=4096,
        )
    )
    assert payload["chat_template_kwargs"]["thinking"] is True
    assert payload["chat_template_kwargs"]["thinking_budget"] == 4096
    # The effort also rides the OpenAI surface for a DS4 build that honors it.
    assert payload["reasoning_effort"] == "high"


def test_effort_only_request_derives_budget_from_effort():
    payload = Ds4Provider()._payload(
        ModelRequest(
            model="deepseek-v4-pro",
            messages=[Message(role="user", content="q")],
            reasoning_effort="low",
        )
    )
    # No explicit budget → derived from the effort ladder (a positive number).
    assert payload["chat_template_kwargs"]["thinking"] is True
    assert payload["chat_template_kwargs"]["thinking_budget"] > 0


def test_strict_json_schema_flag_dropped_for_self_hosted():
    payload = Ds4Provider()._payload(
        ModelRequest(
            model="deepseek-v4-flash",
            messages=[Message(role="user", content="q")],
            output_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
    )
    assert payload["response_format"]["type"] == "json_schema"
    assert "strict" not in payload["response_format"]["json_schema"]


# -- generate / stream round-trip -----------------------------------------


async def test_generate_round_trips_and_excludes_thinking_from_text():
    client = _FakeClient(response=_chat_response())
    provider = Ds4Provider(client=client)
    request = ModelRequest(
        model="deepseek-v4-flash",
        messages=[Message(role="system", content="Be terse."), Message(role="user", content="Where is Bordeaux?")],
        reasoning_effort="medium",
    )
    response = await provider.generate(request)

    assert response.text == "Bordeaux is in France."  # reasoning_content excluded
    assert response.provider == "ds4"
    assert response.usage.reasoning_tokens == 20
    assert response.cost_usd == 0.0  # self-hosted bills nothing
    assert client.last_url.endswith("/chat/completions")
    assert client.last_json["chat_template_kwargs"]["thinking"] is True


async def test_generate_folds_disk_kv_hits_into_cached_input_tokens():
    client = _FakeClient(response=_chat_response())
    provider = Ds4Provider(client=client)
    response = await provider.generate(
        ModelRequest(model="deepseek-v4-flash", messages=[Message(role="user", content="q")])
    )
    # DS4's prompt_cache_hit_tokens surface as cached_input_tokens so the cache
    # telemetry and cost/energy report see the KV reuse.
    assert response.usage.cached_input_tokens == 32
    assert cache_hit_rate(response.usage.input_tokens, response.usage.cached_input_tokens) == pytest.approx(
        32 / 40
    )


async def test_stream_round_trip_yields_deltas_usage_and_done():
    lines = [
        "data: " + json.dumps({"model": "deepseek-v4-flash", "choices": [{"delta": {"content": "Bord"}}]}),
        "data: " + json.dumps({"model": "deepseek-v4-flash", "choices": [{"delta": {"content": "eaux."}}]}),
        "data: "
        + json.dumps(
            {
                "model": "deepseek-v4-flash",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 2, "prompt_cache_hit_tokens": 32},
            }
        ),
        "data: [DONE]",
    ]
    provider = Ds4Provider(client=_FakeClient(lines=lines))
    events = [e async for e in provider.stream(
        ModelRequest(model="deepseek-v4-flash", messages=[Message(role="user", content="hi")])
    )]
    text = "".join(e.text for e in events if e.type == "text_delta")
    done = [e for e in events if e.type == "done"][-1]
    assert text == "Bordeaux."
    assert done.response.usage.cached_input_tokens == 32
    assert done.response.finish_reason == "stop"


# -- embeddings ------------------------------------------------------------


async def test_embed_refuses_clearly():
    with pytest.raises(ConfigError, match="not embeddings"):
        await Ds4Provider().embed(["some text"])


# -- capabilities ----------------------------------------------------------


def test_capabilities_read_from_registry_first():
    caps = Ds4Provider().capabilities("deepseek-v4-flash")
    assert caps.reasoning is True
    assert caps.tool_calling is True
    assert caps.prompt_caching is True
    assert caps.max_context_tokens == 163_840


def test_capabilities_fallback_for_unregistered_quant():
    caps = Ds4Provider().capabilities("deepseek-v4-flash-custom-q3")
    assert caps.reasoning is True
    assert caps.prompt_caching is True
    assert caps.vision is False


def test_no_tiktoken_family_claimed():
    # DeepSeek is not OpenAI BPE, so no exact tiktoken counter is claimed
    # (the offline heuristic counter is used instead).
    assert Ds4Provider().token_id_prefixes() == ()
    assert Ds4Provider().exact_token_counter("deepseek-v4-flash") is None


# -- residency: fail-closed on-prem ---------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/v1",
        "http://localhost:8000/v1",
        "http://[::1]:8000/v1",
        "http://0.0.0.0:8000/v1",
    ],
)
def test_loopback_endpoints_resolve_to_on_prem(url):
    assert infer_region_from_url(url) == "on_prem"


def test_ds4_default_region_is_on_prem():
    assert ResidencyPolicy().region_for("ds4", "deepseek-v4-flash") == "on_prem"


def test_ds4_admitted_when_on_prem_allowed():
    assert residency_violation(
        provider="ds4", model="deepseek-v4-flash", allowed_regions=["on_prem"]
    ) is None


def test_ds4_refused_fail_closed_when_on_prem_not_allowed():
    violation = residency_violation(
        provider="ds4", model="deepseek-v4-flash", allowed_regions=["eu"]
    )
    assert violation is not None
    assert violation.policy == "data_residency"
    assert violation.severity == "block"


def test_localhost_passthrough_egress_refused_by_url_inference():
    # Even the generic passthrough (provider name openai_compat) is pinned on-prem
    # by its localhost endpoint, so an eu-only policy refuses it.
    violation = residency_violation(
        provider="openai_compat",
        model="deepseek-v4-flash",
        allowed_regions=["eu"],
        base_url="http://127.0.0.1:8000/v1",
    )
    assert violation is not None and violation.severity == "block"


# -- catalog & coverage gate ----------------------------------------------


@pytest.mark.parametrize("model", DS4_MODELS)
def test_ds4_models_are_self_hosted_and_free(model):
    profile = ModelRegistry().resolve(model)
    assert profile is not None
    assert profile.provider == "ds4"
    assert profile.self_hosted is True
    assert profile.input_cost_per_mtok == 0.0
    assert profile.output_cost_per_mtok == 0.0
    assert profile.capabilities.reasoning is True


def test_coverage_gate_stays_green_with_self_hosted_ds4():
    report = ModelRegistry().coverage_report()
    assert report.ok, report.gaps + report.unpriced + report.stale
    # The self-hosted $0 preset headline is *covered*, not flagged unpriced.
    assert report.presets_priced is True
    assert not any("ds4" in g for g in report.gaps)


def test_self_hosted_zero_is_not_a_silent_zero_drift():
    report = ModelRegistry().coverage_report()
    # A self-hosted model at $0 must never appear as a silent-$0 drift.
    assert report.no_silent_zero is True
    assert not any(m.startswith("deepseek-v4") for m in report.unpriced)


def test_silent_zero_gate_still_bites_a_paid_model():
    # Regression guard: making self-hosted $0 legitimate must not blind the gate to
    # a genuinely paid model that drops to $0.
    from vincio.core.types import ModelCapabilities, ModelProfile

    registry = ModelRegistry()
    registry.register(
        ModelProfile(
            name="ghost", provider="openai", model="ghost-chat",
            capabilities=ModelCapabilities(tool_calling=True),
        )
    )
    report = registry.coverage_report()
    assert report.no_silent_zero is False
    assert "ghost-chat" in report.unpriced


# -- energy accounting -----------------------------------------------------


def test_ds4_models_carry_a_non_zero_energy_estimate():
    from vincio.core.types import TokenUsage
    from vincio.observability.energy import default_energy_table

    table = default_energy_table()
    estimate = table.estimate(
        "deepseek-v4-pro", TokenUsage(input_tokens=1000, output_tokens=500), region="on_prem"
    )
    # A self-hosted model still accrues energy (tier-derived) and on-prem carbon.
    assert estimate.energy_wh > 0
    assert estimate.region == "on_prem"
    assert estimate.co2e_grams > 0
