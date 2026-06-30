"""The provider-native token-counter registry is wired at provider init (6.3).

``register_token_counter`` was a real, tested extension point but nothing in the
platform registered a counter through it. 6.3 wires it: a provider that can count
a model's tokens exactly and offline registers its counter when it is built, so
counting becomes model-id-driven. These tests pin that wiring and prove the
offline default is unchanged for a provider that supplies no exact counter.
"""

from __future__ import annotations

import pytest

import vincio.core.tokens as tokens
from vincio.core.config import ProviderConfig
from vincio.providers import build_provider, register_provider_token_counters
from vincio.providers.base import ModelProvider
from vincio.providers.local import GGUFProvider


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Save and restore the process-global counter registry around each test."""
    saved = list(tokens._REGISTERED)
    tokens._REGISTERED.clear()
    tokens.get_token_counter.cache_clear()
    tokens._count_cached.cache_clear()
    yield
    tokens._REGISTERED[:] = saved
    tokens.get_token_counter.cache_clear()
    tokens._count_cached.cache_clear()


class _FixedCounter:
    def __init__(self, n: int) -> None:
        self._n = n

    def count(self, text: str) -> int:
        return self._n if text else 0


class _StubProvider(ModelProvider):
    name = "stub"

    async def generate(self, request):  # pragma: no cover - not exercised
        raise NotImplementedError

    def token_id_prefixes(self) -> tuple[str, ...]:
        return ("stub-",)

    def exact_token_counter(self, model: str):
        return _FixedCounter(7)


def test_registration_routes_counts_by_model_id():
    register_provider_token_counters(_StubProvider(), models=("stub-exact",))
    # A prefix-matched model and an exact-model match both resolve to the counter.
    assert tokens.count_tokens("anything at all", "stub-foo") == 7
    assert tokens.count_tokens("anything at all", "stub-exact") == 7


def test_unmatched_model_falls_back_to_offline_default():
    text = "hello world example text"
    # Baseline: the count for a model no matcher claims, with an empty registry.
    baseline = tokens.count_tokens(text, "other-model")
    tokens.get_token_counter.cache_clear()
    tokens._count_cached.cache_clear()
    register_provider_token_counters(_StubProvider())
    # Registering the stub's matchers does not change an unmatched model's count.
    assert tokens.count_tokens(text, "other-model") == baseline


def test_registration_is_idempotent_by_key():
    provider = _StubProvider()
    register_provider_token_counters(provider, models=("stub-exact",))
    first = len(tokens._REGISTERED)
    register_provider_token_counters(provider, models=("stub-exact",))
    register_provider_token_counters(provider, models=("stub-exact",))
    assert len(tokens._REGISTERED) == first  # one entry per (class, matcher)


def test_more_specific_registration_wins():
    register_provider_token_counters(_StubProvider())  # prefix "stub-" -> 7

    def specific(model: str) -> _FixedCounter:
        return _FixedCounter(99)

    tokens.register_token_counter("stub-special", specific, key="test:specific")
    # Longest matching prefix wins, so the more specific counter is selected.
    assert tokens.count_tokens("x", "stub-special-1") == 99
    assert tokens.count_tokens("x", "stub-other") == 7


def test_openai_provider_registers_tiktoken_when_available():
    pytest.importorskip("tiktoken")
    build_provider("openai", ProviderConfig(model="gpt-4o", max_retries=0), api_key="x")
    counter = tokens.get_token_counter("gpt-4o")
    assert type(counter).__name__ == "TiktokenCounter"
    # The registry is genuinely exercised: an entry exists for the gpt- family.
    assert any(matcher("gpt-4o") for matcher, _spec, _f, _key in tokens._REGISTERED)


def test_provider_without_exact_counter_registers_nothing():
    # The mock and local providers expose no offline-exact counter.
    build_provider("mock", ProviderConfig(model="mock", max_retries=0))
    build_provider("local", ProviderConfig(model="llama3", max_retries=0))
    assert tokens._REGISTERED == []


def test_gguf_in_process_tokenizer_counts_exactly():
    class _FakeLlama:
        def tokenize(self, data: bytes, add_bos: bool = False, special: bool = False):
            return list(range(len(data.split())))  # one token per whitespace word

    provider = GGUFProvider(llama=_FakeLlama())
    register_provider_token_counters(provider, models=("my-gguf",))
    assert tokens.count_tokens("one two three four five", "my-gguf") == 5


def test_gguf_without_model_registers_nothing():
    register_provider_token_counters(GGUFProvider(), models=("x",))
    assert tokens._REGISTERED == []
