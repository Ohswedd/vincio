"""Provider abstraction.

Providers translate the provider-neutral :class:`ModelRequest` into vendor
APIs and back. All providers are async-first; ``generate_sync`` is provided
for synchronous callers. Retry/backoff and failover are implemented once,
provider-neutrally, in :class:`RetryingProvider` and :class:`FailoverChain`.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from typing import Any, Protocol, runtime_checkable

import httpx

from ..core.errors import (
    CapabilityMismatchError,
    ConfigError,
    ModelRetiredError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from ..core.tokens import TokenCounter, _registered_keys, register_token_counter
from ..core.types import (
    ModelCapabilities,
    ModelEvent,
    ModelProfile,
    ModelRequest,
    ModelResponse,
)
from .capabilities import RequestNeeds, capability_check, requirements_for

__all__ = [
    "ModelProvider",
    "HTTPProvider",
    "AuthStrategy",
    "RetryingProvider",
    "FailoverChain",
    "ProviderRegistry",
    "parse_sse_lines",
    "run_sync",
    "reasoning_budget_from_effort",
    "is_lifecycle_error",
    "guard_entry",
    "screen_entries",
    "failover_failure",
]


# Substring markers a provider uses for a terminal model-lifecycle/config error
# (a retired/removed/unknown model), as opposed to a transient availability one.
_LIFECYCLE_ERROR_MARKERS = (
    "model_not_found",
    "model not found",
    "does not exist",
    "decommission",
    "is retired",
    "no longer available",
    "no longer supported",
    "has been deprecated",
    "invalid model",
    "unknown model",
)


def is_lifecycle_error(exc: Exception) -> bool:
    """Whether *exc* is a terminal model-lifecycle/config error (retired / removed
    / unknown model) rather than a transient availability error — so a failover
    chain can surface "rotate now" instead of burying it in "all providers failed"."""
    if isinstance(exc, ModelRetiredError):
        return True
    if not isinstance(exc, ProviderError):
        return False
    message = (exc.message or "").lower()
    return any(marker in message for marker in _LIFECYCLE_ERROR_MARKERS)


def guard_entry(
    model: str, needs: RequestNeeds, registry: Any
) -> tuple[str | None, bool]:
    """Pre-flight one failover/route candidate against the registry.

    Returns ``(skip_reason, is_lifecycle)``: ``(None, False)`` when the entry may
    be attempted, a reason string when it must be skipped, with ``is_lifecycle``
    True for a retired model (terminal) and False for a capability mismatch.
    """
    if registry.lifecycle(model) == "retired":
        return (f"{model!r} is retired", True)
    verdict = capability_check(needs, registry.guard_capabilities(model), model=model)
    if not verdict.ok:
        return (verdict.reason, False)
    return (None, False)


def failover_failure(
    name: str,
    attempted: int,
    attempt_errors: list[str],
    lifecycle: list[str],
    incapable: list[str],
) -> ProviderError:
    """Build the terminal error for an exhausted failover chain, classifying a
    retired-only failure ("rotate now") and a capability-only one distinctly from
    a plain availability failure."""
    if attempted == 0 and lifecycle and not incapable and not attempt_errors:
        return ModelRetiredError(
            "rotate now — every failover candidate is retired: " + " | ".join(lifecycle),
            provider=name,
        )
    if attempted == 0 and incapable and not attempt_errors:
        return CapabilityMismatchError(
            "no capable failover candidate: " + " | ".join(incapable + lifecycle),
            provider=name,
        )
    detail = list(attempt_errors)
    if lifecycle:
        detail.append("rotate now (retired): " + "; ".join(lifecycle))
    if incapable:
        detail.append("skipped (incapable): " + "; ".join(incapable))
    return ProviderUnavailableError(
        "all providers failed: " + " | ".join(detail), provider=name, retryable=False
    )


def screen_entries(
    entries: list[tuple[ModelProvider, str | None]],
    request: ModelRequest,
    *,
    guard: bool,
    registry: Any,
) -> tuple[list[tuple[ModelProvider, str | None, str]], list[str], list[str]]:
    """Pre-screen failover/router entries against the capability+lifecycle guard.

    Returns ``(attemptable, lifecycle, incapable)`` where ``attemptable`` is the
    list of ``(provider, model_override, model)`` to try in order, and the other
    two are human-readable skip reasons (retired vs capability-mismatched). When
    ``guard`` is False, every entry is attemptable. Shared by ``FailoverChain``
    and ``HealthAwareFailover`` so the guard logic lives in one place.
    """
    if not guard:
        return ([(p, mo, mo or request.model) for p, mo in entries], [], [])
    needs = requirements_for(request)
    attemptable: list[tuple[ModelProvider, str | None, str]] = []
    lifecycle: list[str] = []
    incapable: list[str] = []
    for provider, model_override in entries:
        model = model_override or request.model
        reason, is_lc = guard_entry(model, needs, registry)
        if reason is not None:
            (lifecycle if is_lc else incapable).append(f"{provider.name}/{model}: {reason}")
            continue
        attemptable.append((provider, model_override, model))
    return attemptable, lifecycle, incapable


# Effort → thinking-token budget for providers that take an explicit budget
# (Anthropic, Gemini) rather than an effort level (OpenAI). Conservative
# defaults; callers override with ModelRequest.thinking_budget_tokens.
_EFFORT_BUDGET = {"minimal": 1024, "low": 4096, "medium": 8192, "high": 16384}


def reasoning_budget_from_effort(
    effort: str | None, explicit: int | None = None, *, default: int = 8192
) -> int:
    """Resolve a thinking-token budget from an effort level (or an explicit value)."""
    if explicit is not None:
        return max(0, explicit)
    if effort is None:
        return default
    return _EFFORT_BUDGET.get(effort, default)


def run_sync(coro):
    """Run *coro* to completion from sync code, inside or outside a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # We're inside a running loop (e.g. Jupyter): execute in a fresh thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class ModelProvider(ABC):
    """Abstract model provider."""

    name: str = "base"

    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Execute a single model call."""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Stream a model call. Default: emulate streaming from generate()."""
        response = await self.generate(request)
        if response.text:
            yield ModelEvent(type="text_delta", text=response.text)
        for tool_call in response.tool_calls:
            yield ModelEvent(type="tool_call_delta", tool_call=tool_call)
        yield ModelEvent(type="usage", usage=response.usage)
        yield ModelEvent(type="done", response=response)

    def capabilities(self, model: str) -> ModelCapabilities:
        """Capability matrix for *model*, from the model registry when known."""
        from .registry import default_model_registry

        profile = default_model_registry().resolve(model)
        return profile.capabilities if profile is not None else ModelCapabilities()

    def exact_token_counter(self, model: str) -> TokenCounter | None:
        """An exact, *offline* token counter for *model*, or ``None``.

        The hook the :func:`~vincio.core.tokens.register_token_counter` registry
        is wired through: a provider that can count a model's tokens exactly
        without a network round-trip (the OpenAI provider via ``tiktoken``; an
        in-process GGUF model via its own tokenizer) returns a counter here, and
        it is registered for the provider's served model ids when the provider is
        built (see :func:`register_provider_token_counters`). The default returns
        ``None`` — the offline heuristic (or ``tiktoken``) is used, unchanged. A
        hosted provider whose only exact count is a network call deliberately
        returns ``None``, so the per-candidate scoring loop never blocks on the
        network; a deployment that wants exact remote counts registers one
        through the same public hook.
        """
        return None

    def token_id_prefixes(self) -> tuple[str, ...]:
        """Model-id prefixes whose tokens this provider counts exactly offline.

        Registered as registry matchers (longest matching prefix wins) when the
        provider is built, so a deployment can still register a more specific
        counter that out-ranks the provider default. The base returns an empty
        tuple — no family claimed."""
        return ()

    def generate_sync(self, request: ModelRequest) -> ModelResponse:
        return run_sync(self.generate(request))

    def stream_sync(self, request: ModelRequest) -> Iterator[ModelEvent]:
        """Synchronous streaming: collects events from the async iterator."""

        async def collect() -> list[ModelEvent]:
            return [event async for event in self.stream(request)]

        yield from run_sync(collect())

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed texts. Providers that support embeddings override this."""
        raise ProviderError(
            f"provider {self.name!r} does not support embeddings", provider=self.name
        )

    async def list_models(self) -> list[ModelProfile]:
        """Discover the models the provider currently serves.

        Providers that expose a model-list endpoint (OpenAI ``/v1/models``,
        Anthropic ``/v1/models``, Gemini ``ListModels``, Ollama ``/api/tags``,
        OpenAI-compatible ``/v1/models``) override this and map the response onto
        :class:`~vincio.core.types.ModelProfile`\\ s for reconciliation into the
        :class:`~vincio.providers.registry.ModelRegistry`. The default returns an
        empty list — offline-safe, so the shipped catalog remains authoritative.
        """
        return []

    async def aclose(self) -> None:
        """Release underlying resources (HTTP clients)."""
        return None


def register_provider_token_counters(
    provider: ModelProvider, *, models: Iterable[str] = ()
) -> None:
    """Register *provider*'s exact, offline token counters into the global registry.

    Called when a provider is built (see
    :func:`~vincio.providers.build_provider`): registers a prefix matcher for each
    family in ``provider.token_id_prefixes()`` and an exact-model matcher for each
    id in *models* (the app's resolved model — covering an in-process GGUF model
    whose id no prefix matches), skipping any family or model for which the
    provider exposes no offline-exact :meth:`~ModelProvider.exact_token_counter`.
    Idempotent: each registration is keyed by provider class and matcher, so a
    provider built more than once stays a single registry entry and the offline
    default is unchanged for a provider that supplies no exact counter.
    """

    def _factory(model: str) -> TokenCounter:
        counter = provider.exact_token_counter(model)
        if counter is None:  # pragma: no cover - guarded by the probes below
            raise ProviderError(
                f"no exact token counter for {model!r}", provider=provider.name
            )
        return counter

    def _exact_match(model_id: str) -> Callable[[str], bool]:
        def match(candidate: str) -> bool:
            return candidate == model_id

        return match

    # Skip keys already registered: a provider built once per request would
    # otherwise re-register every build and each registration clears the shared
    # token memo the compiler's hot loops depend on. Idempotent and side-effect-free
    # when nothing is new.
    existing = _registered_keys()
    cls = type(provider).__name__
    prefixes = tuple(provider.token_id_prefixes())
    if prefixes and provider.exact_token_counter(prefixes[0]) is not None:
        for prefix in prefixes:
            key = f"{cls}:prefix:{prefix}"
            if key not in existing:
                register_token_counter(prefix, _factory, key=key)
    for model in models:
        key = f"{cls}:model:{model}"
        if not model or key in existing or provider.exact_token_counter(model) is None:
            continue
        register_token_counter(_exact_match(model), _factory, key=key)


@runtime_checkable
class AuthStrategy(Protocol):
    """Computes the auth headers for a single outbound provider request.

    Lets :class:`HTTPProvider` go beyond a static api-key header: enterprise
    endpoints (AWS Bedrock SigV4, Google
    Vertex service-account OAuth, Azure ``api-key``/AAD) plug their per-request
    signing in here, routed through the same registry, capability guards, swap
    gate, residency, and audit chain as every other provider.

    Given the exact ``method`` / ``url`` / ``body`` bytes about to be sent (so a
    signature binds them) plus the provider's ``base_headers``, return the full
    header set to send.
    """

    def headers(
        self, *, method: str, url: str, body: bytes, base_headers: dict[str, str]
    ) -> dict[str, str]: ...


class HTTPProvider(ModelProvider):
    """Shared httpx plumbing for HTTP API providers."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        auth: AuthStrategy | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.timeout_s = timeout_s
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self._client = client
        self._owns_client = client is None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        # optional per-request auth strategy. When None, the static
        # ``_headers()`` path is used unchanged (every 1.x provider).
        self.auth = auth

    default_base_url: str = ""
    requires_api_key: bool = True

    @property
    def client(self) -> httpx.AsyncClient:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        # Recreate a client we own if it is closed or was bound to a different
        # event loop. Reusing a client across asyncio.run() calls (the natural
        # sync usage of generate_sync/stream_sync) otherwise raises
        # "Event loop is closed" when httpx touches a connection from a dead loop.
        stale_loop = (
            self._owns_client
            and loop is not None
            and self._client_loop is not None
            and self._client_loop is not loop
        )
        if self._client is None or self._client.is_closed or stale_loop:
            self._client = httpx.AsyncClient(
                timeout=self.timeout_s,
                limits=httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_keepalive_connections,
                ),
            )
            self._owns_client = True
            self._client_loop = loop
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    def _check_key(self) -> None:
        if self.requires_api_key and not self.api_key:
            raise ProviderAuthError(
                f"missing API key for provider {self.name!r}", provider=self.name
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _prepare(
        self, method: str, url: str, payload: dict[str, Any] | None
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Resolve (headers, httpx-call-kwargs) for one request.

        When an :class:`AuthStrategy` is set, the body is serialized once and
        signed over those exact bytes (sent via ``content=`` so httpx does not
        re-serialize and break the signature). Otherwise the legacy ``json=``
        path with static ``_headers()`` is used unchanged.
        """
        if self.auth is None:
            return self._headers(), ({} if payload is None else {"json": payload})
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        headers = self.auth.headers(
            method=method, url=url, body=body, base_headers=self._headers()
        )
        return headers, ({} if payload is None else {"content": body})

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            body = {"raw": response.text[:2000]}
        message = json.dumps(body)[:2000]
        kw: dict[str, Any] = {"provider": self.name, "details": {"status": response.status_code}}
        if response.status_code in (401, 403):
            raise ProviderAuthError(f"authentication failed: {message}", **kw)
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            retry_after_s = float(retry_after) if retry_after else _retry_delay_from_body(body)
            raise ProviderRateLimitError(
                f"rate limited: {message}",
                retry_after_s=retry_after_s,
                **kw,
            )
        if response.status_code in (408, 504):
            raise ProviderTimeoutError(f"timeout: {message}", **kw)
        if response.status_code >= 500 or response.status_code == 529:
            raise ProviderUnavailableError(f"provider unavailable: {message}", **kw)
        raise ProviderResponseError(f"provider error {response.status_code}: {message}", **kw)

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._check_key()
        url = f"{self.base_url}{path}"
        headers, call_kwargs = self._prepare("POST", url, payload)
        try:
            response = await self.client.post(url, headers=headers, **call_kwargs)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc), provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc), provider=self.name) from exc
        self._raise_for_status(response)
        return response.json()

    async def _get_json(self, path: str) -> dict[str, Any]:
        self._check_key()
        url = f"{self.base_url}{path}"
        headers, _ = self._prepare("GET", url, None)
        try:
            response = await self.client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc), provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc), provider=self.name) from exc
        self._raise_for_status(response)
        return response.json()

    async def _get_text(self, path: str) -> str:
        self._check_key()
        url = f"{self.base_url}{path}"
        headers, _ = self._prepare("GET", url, None)
        try:
            response = await self.client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc), provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc), provider=self.name) from exc
        self._raise_for_status(response)
        return response.text

    async def _post_stream(self, path: str, payload: dict[str, Any]) -> AsyncIterator[str]:
        self._check_key()
        url = f"{self.base_url}{path}"
        headers, call_kwargs = self._prepare("POST", url, payload)
        try:
            async with self.client.stream(
                "POST", url, headers=headers, **call_kwargs
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._raise_for_status(response)
                async for line in response.aiter_lines():
                    yield line
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc), provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc), provider=self.name) from exc


def _retry_delay_from_body(body: Any) -> float | None:
    """Extract a retry delay from a provider error body when no Retry-After header is set.

    Google's Gemini API returns the cooldown in the JSON body — either as a
    ``RetryInfo.retryDelay`` detail ("29s") or in the message text ("retry in
    29.1s"). Honoring it lets free-tier RPM limits self-heal via the retry loop.
    """
    if not isinstance(body, dict):
        return None
    error = body.get("error") or {}
    for detail in error.get("details", []) or []:
        if not isinstance(detail, dict):
            continue
        delay = detail.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                pass
    message = error.get("message") if isinstance(error, dict) else None
    if isinstance(message, str):
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", message)
        if match:
            return float(match.group(1))
    return None


def parse_sse_lines(line: str) -> str | None:
    """Extract the data payload from an SSE line; returns None for non-data lines."""
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if line.startswith("data:"):
        return line[len("data:") :].strip()
    return None


class RetryingProvider(ModelProvider):
    """Wraps a provider with bounded exponential backoff on retryable errors."""

    def __init__(
        self,
        inner: ModelProvider,
        *,
        max_retries: int = 2,
        base_delay_s: float = 0.5,
        max_delay_s: float = 60.0,
        jitter: float = 0.2,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.jitter = jitter
        self.retry_count = 0

    def _delay(self, attempt: int, error: ProviderError) -> float:
        if isinstance(error, ProviderRateLimitError) and error.retry_after_s:
            return min(self.max_delay_s, error.retry_after_s)
        delay = min(self.max_delay_s, self.base_delay_s * (2**attempt))
        return delay * (1.0 + random.uniform(-self.jitter, self.jitter))

    async def generate(self, request: ModelRequest) -> ModelResponse:
        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self.inner.generate(request)
            except ProviderError as exc:
                if not exc.retryable or attempt == self.max_retries:
                    raise
                last_error = exc
                self.retry_count += 1
                await asyncio.sleep(self._delay(attempt, exc))
        raise last_error or ProviderError("retries exhausted", provider=self.name)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        # Retry only before first event is yielded; mid-stream errors propagate.
        for attempt in range(self.max_retries + 1):
            yielded = False
            try:
                async for event in self.inner.stream(request):
                    yielded = True
                    yield event
                return
            except ProviderError as exc:
                if yielded or not exc.retryable or attempt == self.max_retries:
                    raise
                self.retry_count += 1
                await asyncio.sleep(self._delay(attempt, exc))

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.inner.capabilities(model)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return await self.inner.embed(texts, model)

    async def list_models(self) -> list[ModelProfile]:
        return await self.inner.list_models()

    async def aclose(self) -> None:
        await self.inner.aclose()


def _merge_model_lists(profile_lists: list[list[ModelProfile]]) -> list[ModelProfile]:
    """Union discovered profiles across chain entries, first-seen wins per id."""
    seen: dict[str, ModelProfile] = {}
    for profiles in profile_lists:
        for profile in profiles:
            seen.setdefault(profile.model, profile)
    return list(seen.values())


class FailoverChain(ModelProvider):
    """Provider failover: try providers/models in order.

    Each entry is ``(provider, model_override | None)``. On non-retryable
    auth errors or exhausted retries, the next entry is attempted.

    With ``guard_capabilities`` (default on) every entry is pre-flighted
    against the :class:`~vincio.providers.registry.ModelRegistry` before it is
    attempted: a *retired* model is skipped as a terminal lifecycle error and a
    *capability-mismatched* one (cannot serve the request's vision/tools/schema/
    reasoning/context) is skipped rather than silently returning a wrong answer.
    When every entry is exhausted, a retired-only failure raises
    :class:`~vincio.core.errors.ModelRetiredError` ("rotate now"); a
    capability-only failure raises :class:`CapabilityMismatchError`; otherwise the
    usual :class:`ProviderUnavailableError`. Set ``guard_capabilities=False`` to
    restore the previous attempt-everything behavior. Unknown models are never
    blocked.
    """

    name = "failover"

    def __init__(
        self,
        entries: list[tuple[ModelProvider, str | None]],
        *,
        guard_capabilities: bool = True,
        registry: Any | None = None,
    ) -> None:
        if not entries:
            raise ConfigError("FailoverChain requires at least one provider")
        self.entries = entries
        self.guard_capabilities = guard_capabilities
        self._registry = registry

    def _reg(self) -> Any:
        if self._registry is None:
            from .registry import default_model_registry

            self._registry = default_model_registry()
        return self._registry

    def _failure(
        self, attempted: int, attempt_errors: list[str], lifecycle: list[str], incapable: list[str]
    ) -> ProviderError:
        return failover_failure(self.name, attempted, attempt_errors, lifecycle, incapable)

    def _screen(
        self, request: ModelRequest
    ) -> tuple[list[tuple[ModelProvider, str | None, str]], list[str], list[str]]:
        registry = self._reg() if self.guard_capabilities else None
        return screen_entries(
            self.entries, request, guard=self.guard_capabilities, registry=registry
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        attemptable, lifecycle, incapable = self._screen(request)
        attempt_errors: list[str] = []
        for provider, model_override, model in attemptable:
            attempt_request = request if not model_override else request.model_copy(
                update={"model": model_override}
            )
            try:
                return await provider.generate(attempt_request)
            except ProviderError as exc:
                if is_lifecycle_error(exc):
                    lifecycle.append(f"{provider.name}/{model}: {exc.message}")
                else:
                    attempt_errors.append(f"{provider.name}: {exc.message}")
        raise self._failure(len(attemptable), attempt_errors, lifecycle, incapable)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        attemptable, lifecycle, incapable = self._screen(request)
        attempt_errors: list[str] = []
        for provider, model_override, model in attemptable:
            attempt_request = request if not model_override else request.model_copy(
                update={"model": model_override}
            )
            yielded = False
            try:
                async for event in provider.stream(attempt_request):
                    yielded = True
                    yield event
                return
            except ProviderError as exc:
                if yielded:
                    raise
                if is_lifecycle_error(exc):
                    lifecycle.append(f"{provider.name}/{model}: {exc.message}")
                else:
                    attempt_errors.append(f"{provider.name}: {exc.message}")
        raise self._failure(len(attemptable), attempt_errors, lifecycle, incapable)

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.entries[0][0].capabilities(model)

    async def list_models(self) -> list[ModelProfile]:
        return _merge_model_lists([await p.list_models() for p, _ in self.entries])

    async def aclose(self) -> None:
        for provider, _ in self.entries:
            await provider.aclose()


class ProviderRegistry:
    """Named provider construction + instance cache."""

    def __init__(self) -> None:
        self._factories: dict[str, Any] = {}
        self._instances: dict[str, ModelProvider] = {}

    def register(self, name: str, factory: Any) -> None:
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> ModelProvider:
        if name not in self._factories:
            raise ConfigError(
                f"unknown provider {name!r}; known: {sorted(self._factories)}"
            )
        return self._factories[name](**kwargs)

    def get_or_create(self, name: str, **kwargs: Any) -> ModelProvider:
        key = f"{name}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        if key not in self._instances:
            self._instances[key] = self.create(name, **kwargs)
        return self._instances[key]

    @property
    def names(self) -> list[str]:
        return sorted(self._factories)


def measure_latency_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)
