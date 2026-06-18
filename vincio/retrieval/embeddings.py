"""Embedding interfaces (dense retrieval, extras).

- :class:`LocalHashEmbedder` — deterministic, dependency-free embeddings
  (token-hash bag with char-trigram smoothing). Useful for tests, offline
  mode, and as a fallback; captures lexical similarity, not deep semantics.
- :class:`ProviderEmbedder` — embeddings via any provider that implements
  ``embed`` (OpenAI, Google, Mistral, local servers); large inputs are
  split into bounded batches embedded concurrently.
- :class:`CachedEmbedder` — wraps any embedder with a thread-safe,
  content-addressed cache (in-memory by default, any
  :class:`~vincio.caching.base.CacheBackend` for persistence).
- :class:`BatchingEmbedder` — micro-batches concurrent ``embed`` calls into
  one provider call (request coalescing for embeddings).
- :class:`MatryoshkaEmbedder` — truncates any embedder's output to a smaller
  dimension (MRL), so vectors shrink with minimal quality loss.
- :class:`VoyageContextualEmbedder` — contextual chunk embeddings whose
  per-chunk vector carries the surrounding document context.
- :class:`MultimodalEmbedder` (Cohere v4 / Voyage multimodal) — unified
  text+image embeddings in one vector space.

All embedders accept an optional ``input_type`` hint (``"document"`` for the
corpus, ``"query"`` at search time); embedders that don't use it ignore it.
:func:`embed_texts` dispatches the hint only to embedders that advertise
support, so custom embedders implementing only ``embed(texts)`` keep working.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
from collections import OrderedDict
from typing import Any, Literal, Protocol, cast

import httpx
from pydantic import BaseModel

from ..core.concurrency import gather_bounded
from ..core.errors import ConfigError, ProviderAuthError, ProviderError
from ..core.types import ImageRef
from ..providers.base import ModelProvider, run_sync

__all__ = [
    "Embedder",
    "InputType",
    "LocalHashEmbedder",
    "ProviderEmbedder",
    "CachedEmbedder",
    "BatchingEmbedder",
    "MatryoshkaEmbedder",
    "FastEmbedEmbedder",
    "ColBERTTokenEmbedder",
    "HTTPEmbedder",
    "JinaEmbedder",
    "VoyageEmbedder",
    "CohereEmbedder",
    "VoyageContextualEmbedder",
    "MultimodalInput",
    "MultimodalEmbedder",
    "VoyageMultimodalEmbedder",
    "CohereMultimodalEmbedder",
    "build_embedder",
    "embed_texts",
    "mrl_truncate",
    "cosine",
]

InputType = Literal["document", "query"]


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        ...


async def embed_texts(
    embedder: Embedder, texts: list[str], *, input_type: InputType | None = None
) -> list[list[float]]:
    """Embed *texts*, passing the ``input_type`` hint only to embedders that
    advertise support (``supports_input_type``). Custom embedders implementing
    only ``embed(texts)`` are called the old way, so nothing breaks."""
    if input_type is not None and getattr(embedder, "supports_input_type", False):
        return await embedder.embed(texts, input_type=input_type)  # type: ignore[call-arg]
    return await embedder.embed(texts)


def mrl_truncate(vector: list[float], dimensions: int) -> list[float]:
    """Matryoshka truncation: keep the first ``dimensions`` components and
    L2-renormalize. MRL-trained models pack the most information into the
    leading dimensions, so a truncated vector stays a faithful, comparable
    embedding. A no-op when ``dimensions`` covers the whole vector."""
    if dimensions <= 0 or dimensions >= len(vector):
        return list(vector)
    head = vector[:dimensions]
    norm = math.sqrt(sum(x * x for x in head)) or 1.0
    return [x / norm for x in head]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class LocalHashEmbedder:
    """Deterministic offline embeddings.

    Combines word-level and character-trigram hash features so morphological
    variants ("termination"/"terminate") land near each other.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        tokens = text.lower().split()
        features: list[str] = []
        for token in tokens:
            token = "".join(c for c in token if c.isalnum())
            if not token:
                continue
            features.append(f"w:{token}")
            padded = f"^{token}$"
            features.extend(f"t:{padded[i:i+3]}" for i in range(len(padded) - 2))
        return features

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature in self._features(text):
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 if feature.startswith("w:") else 0.35
            vector[index] += sign * weight
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]


class ProviderEmbedder:
    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str | None = None,
        dim: int = 1536,
        batch_size: int = 64,
        concurrency: int = 4,
    ) -> None:
        self.provider = provider
        self.model = model
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self.concurrency = max(1, concurrency)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) <= self.batch_size:
            vectors = await self.provider.embed(texts, self.model)
        else:
            batches = [
                texts[start : start + self.batch_size]
                for start in range(0, len(texts), self.batch_size)
            ]
            results = await gather_bounded(
                (self.provider.embed(batch, self.model) for batch in batches),
                limit=self.concurrency,
            )
            vectors = [vector for batch_vectors in results for vector in batch_vectors]
        if vectors:
            self.dim = len(vectors[0])
        return vectors


def _content_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class CachedEmbedder:
    """Content-addressed embedding cache, safe under concurrent use.

    Keys are SHA-256 hashes of the text, so identical content embedded from
    any path (chunking, queries, semantic cache) reuses one vector. An
    optional persistent ``backend`` (any :class:`~vincio.caching.base
    .CacheBackend`) survives process restarts; the in-memory LRU fronts it.
    When the inner embedder is input-type-aware the hint is folded into the
    key, so a ``"query"`` vector never aliases a ``"document"`` vector.
    """

    def __init__(self, inner: Embedder, *, max_entries: int = 50_000, backend: Any | None = None) -> None:
        self.inner = inner
        self.max_entries = max_entries
        self.backend = backend  # CacheBackend | None (duck-typed: get/set)
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def supports_input_type(self) -> bool:
        return bool(getattr(self.inner, "supports_input_type", False))

    def _key(self, text: str, input_type: InputType | None) -> str:
        base = _content_key(text)
        if input_type is not None and self.supports_input_type:
            return f"{input_type}:{base}"
        return base

    def _get_cached(self, key: str) -> list[float] | None:
        with self._lock:
            vector = self._cache.get(key)
            if vector is not None:
                self._cache.move_to_end(key)
                return vector
        if self.backend is not None:
            persisted = self.backend.get(f"emb:{key}")
            if persisted is not None:
                self._store_memory(key, persisted)
                return persisted
        return None

    def _store_memory(self, key: str, vector: list[float]) -> None:
        with self._lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)

    async def embed(
        self, texts: list[str], *, input_type: InputType | None = None
    ) -> list[list[float]]:
        keys = [self._key(text, input_type) for text in texts]
        resolved: dict[str, list[float]] = {}
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        for text, key in zip(texts, keys, strict=True):
            if key in resolved:
                continue
            vector = self._get_cached(key)
            if vector is not None:
                resolved[key] = vector
            elif key not in missing_keys:
                missing_keys.append(key)
                missing_texts.append(text)
        if missing_texts:
            vectors = await embed_texts(self.inner, missing_texts, input_type=input_type)
            for key, vector in zip(missing_keys, vectors, strict=True):
                resolved[key] = vector
                self._store_memory(key, vector)
                if self.backend is not None:
                    self.backend.set(f"emb:{key}", vector, tags=["embeddings"])
            self.misses += len(missing_texts)
        self.hits += len(texts) - len(missing_texts)
        return [resolved[key] for key in keys]

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        return run_sync(self.embed(texts))


class BatchingEmbedder:
    """Coalesces concurrent ``embed`` calls into provider micro-batches.

    Callers that issue small ``embed`` requests at the same time (parallel
    retrieval subqueries, agent steps) are merged into one provider call,
    flushed when the batch fills or after ``window_ms`` of quiet — fewer
    round-trips, same results. Duplicate texts within a flush are sent once;
    differing ``input_type`` hints never share a coalesced vector.
    """

    def __init__(self, inner: Embedder, *, max_batch: int = 64, window_ms: float = 5.0) -> None:
        self.inner = inner
        self.max_batch = max(1, max_batch)
        self.window_ms = window_ms
        self.flushes = 0
        self._pending: list[tuple[str, InputType | None, asyncio.Future[list[float]]]] = []
        self._timer: asyncio.Task | None = None

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def supports_input_type(self) -> bool:
        return bool(getattr(self.inner, "supports_input_type", False))

    async def embed(
        self, texts: list[str], *, input_type: InputType | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future[list[float]]] = []
        for text in texts:
            future: asyncio.Future[list[float]] = loop.create_future()
            self._pending.append((text, input_type, future))
            futures.append(future)
        full_batch: list[tuple[str, InputType | None, asyncio.Future[list[float]]]] | None = None
        if len(self._pending) >= self.max_batch:
            full_batch, self._pending = self._pending, []
        elif self._timer is None or self._timer.done():
            self._timer = asyncio.ensure_future(self._delayed_flush())
        if full_batch is not None:
            await self._flush(full_batch)
        return list(await asyncio.gather(*futures))

    async def aclose(self) -> None:
        """Flush anything pending and stop the timer."""
        batch, self._pending = self._pending, []
        if self._timer is not None and not self._timer.done():
            self._timer.cancel()
            try:
                await self._timer
            except asyncio.CancelledError:
                pass
        if batch:
            await self._flush(batch)

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(self.window_ms / 1000.0)
        batch, self._pending = self._pending, []
        if batch:
            await self._flush(batch)

    async def _flush(
        self, batch: list[tuple[str, InputType | None, asyncio.Future[list[float]]]]
    ) -> None:
        self.flushes += 1
        # Group by input-type so a "query" vector never satisfies a "document"
        # waiter; within a group, identical texts are sent once.
        groups: dict[InputType | None, dict[str, list[asyncio.Future[list[float]]]]] = {}
        for text, input_type, future in batch:
            groups.setdefault(input_type, {}).setdefault(text, []).append(future)
        for input_type, unique in groups.items():
            texts = list(unique)
            try:
                vectors = await embed_texts(self.inner, texts, input_type=input_type)
            except BaseException as exc:  # propagate to every waiter
                for waiters in unique.values():
                    for future in waiters:
                        if not future.done():
                            future.set_exception(exc)
                            future.exception()  # consumed below or by the waiter
                if isinstance(exc, asyncio.CancelledError):
                    raise
                continue
            for text, vector in zip(texts, vectors, strict=True):
                for future in unique[text]:
                    if not future.done():
                        future.set_result(vector)


class MatryoshkaEmbedder:
    """Matryoshka (MRL) dimension truncation over any embedder.

    Wraps an inner embedder and truncates every output vector to
    ``dimensions`` (then L2-renormalizes), so storage and search cost shrink
    with the leading-dimension quality MRL-trained models preserve. Works with
    any embedder — provider, hosted, or the offline hash embedder — and
    composes with :class:`CachedEmbedder` / :class:`BatchingEmbedder`. The
    ``input_type`` hint passes through to the inner embedder.
    """

    def __init__(self, inner: Embedder, dimensions: int) -> None:
        if dimensions <= 0:
            raise ConfigError("MatryoshkaEmbedder requires dimensions > 0")
        self.inner = inner
        self.dimensions = dimensions

    @property
    def dim(self) -> int:
        return self.dimensions

    @property
    def supports_input_type(self) -> bool:
        return bool(getattr(self.inner, "supports_input_type", False))

    async def embed(
        self, texts: list[str], *, input_type: InputType | None = None
    ) -> list[list[float]]:
        vectors = await embed_texts(self.inner, texts, input_type=input_type)
        return [mrl_truncate(vector, self.dimensions) for vector in vectors]

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        return run_sync(self.embed(texts))


class HTTPEmbedder:
    """Hosted embedding endpoint over httpx (OpenAI-style ``/embeddings``).

    Jina and Voyage speak the OpenAI embedding dialect, so they subclass this
    directly; Cohere's v2 shape overrides payload/parse. An
    ``httpx.AsyncClient`` can be injected for offline testing.

    Set ``dimensions`` for Matryoshka output truncation (sent to the provider
    when it supports it, then enforced client-side so the returned vector is
    exactly that long). Set ``input_type`` per call (``"document"`` /
    ``"query"``) for providers that distinguish corpus from query encodings.
    """

    name = "http"
    default_base_url = ""
    default_model = ""
    supports_input_type = True
    supports_dimensions = True
    _dimensions_field = "dimensions"  # native MRL field name in the request

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        dim: int = 1024,
        dimensions: int | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model or self.default_model
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.dimensions = dimensions
        self.dim = dimensions or dim
        self._client = client
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderAuthError(f"missing API key for embedder {self.name!r}", provider=self.name)
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, texts: list[str], input_type: InputType | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            payload[self._dimensions_field] = self.dimensions
        return payload

    def _parse(self, data: dict[str, Any]) -> list[list[float]]:
        items = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]

    async def _post(self, payload: dict[str, Any]) -> list[list[float]]:
        client = self._client or httpx.AsyncClient(timeout=self.timeout_s)
        try:
            response = await client.post(self.base_url, json=payload, headers=self._headers())
            if response.status_code >= 400:
                raise ProviderError(
                    f"embedder {self.name!r} error {response.status_code}: {response.text[:500]}",
                    provider=self.name,
                )
            vectors = self._parse(response.json())
        finally:
            if self._client is None:
                await client.aclose()
        if self.dimensions is not None:
            vectors = [mrl_truncate(vector, self.dimensions) for vector in vectors]
        if vectors:
            self.dim = len(vectors[0])
        return vectors

    async def embed(
        self, texts: list[str], *, input_type: InputType | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        return await self._post(self._payload(texts, input_type))


class JinaEmbedder(HTTPEmbedder):
    """Jina AI embeddings (``/v1/embeddings``); OpenAI-compatible shape with a
    ``task`` hint for retrieval (passage vs query) and ``dimensions`` MRL."""

    name = "jina"
    default_base_url = "https://api.jina.ai/v1/embeddings"
    default_model = "jina-embeddings-v3"

    _JINA_TASK = {"document": "retrieval.passage", "query": "retrieval.query"}

    def _payload(self, texts: list[str], input_type: InputType | None) -> dict[str, Any]:
        payload = super()._payload(texts, input_type)
        if input_type is not None:
            payload["task"] = self._JINA_TASK[input_type]
        return payload


class VoyageEmbedder(HTTPEmbedder):
    """Voyage AI embeddings (``/v1/embeddings``); OpenAI-compatible shape.

    Voyage names its MRL field ``output_dimension`` and takes a document/query
    ``input_type``."""

    name = "voyage"
    default_base_url = "https://api.voyageai.com/v1/embeddings"
    default_model = "voyage-3"
    _dimensions_field = "output_dimension"

    def _payload(self, texts: list[str], input_type: InputType | None) -> dict[str, Any]:
        payload = super()._payload(texts, input_type)
        if input_type is not None:
            payload["input_type"] = input_type
        return payload


class CohereEmbedder(HTTPEmbedder):
    """Cohere v2 embeddings (``/v2/embed``).

    Cohere asks for an ``input_type`` (``search_document`` for the corpus,
    ``search_query`` at query time) and returns vectors under
    ``embeddings.float`` rather than the OpenAI ``data[].embedding`` shape.
    Pass ``dimensions`` for Matryoshka output truncation (``embed-v4.0``).
    """

    name = "cohere"
    default_base_url = "https://api.cohere.com/v2/embed"
    default_model = "embed-english-v3.0"
    _dimensions_field = "output_dimension"

    _COHERE_INPUT_TYPE = {"document": "search_document", "query": "search_query"}

    def __init__(self, *, input_type: str = "search_document", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.input_type = input_type

    def _payload(self, texts: list[str], input_type: InputType | None) -> dict[str, Any]:
        resolved = self._COHERE_INPUT_TYPE[input_type] if input_type is not None else self.input_type
        payload: dict[str, Any] = {
            "model": self.model,
            "texts": texts,
            "input_type": resolved,
            "embedding_types": ["float"],
        }
        if self.dimensions is not None:
            payload[self._dimensions_field] = self.dimensions
        return payload

    def _parse(self, data: dict[str, Any]) -> list[list[float]]:
        embeddings = data.get("embeddings") or {}
        if isinstance(embeddings, dict):
            return list(embeddings.get("float") or [])
        return list(embeddings)  # older "embed-floats" responses returned a bare list


class VoyageContextualEmbedder(HTTPEmbedder):
    """Voyage contextual chunk embeddings (``/v1/contextualizedembeddings``).

    Each chunk's vector is computed *with* its sibling chunks as context, so
    the embedding encodes where the chunk sits in the document — complementing
    Vincio's ``contextualize_chunks`` LLM-prefix approach without rewriting the
    text. ``embed(texts)`` treats the batch as one document's chunks;
    :meth:`embed_grouped` embeds several documents at once (each inner list is
    one document). Default model ``voyage-context-3``.
    """

    name = "voyage-context"
    default_base_url = "https://api.voyageai.com/v1/contextualizedembeddings"
    default_model = "voyage-context-3"
    _dimensions_field = "output_dimension"

    def _grouped_payload(
        self, documents: list[list[str]], input_type: InputType | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "inputs": documents}
        if input_type is not None:
            payload["input_type"] = input_type
        if self.dimensions is not None:
            payload[self._dimensions_field] = self.dimensions
        return payload

    def _parse_grouped(self, data: dict[str, Any]) -> list[list[list[float]]]:
        groups = sorted(data.get("data") or [], key=lambda g: g.get("index", 0))
        out: list[list[list[float]]] = []
        for group in groups:
            chunks = sorted(group.get("data") or [], key=lambda c: c.get("index", 0))
            vectors = [c["embedding"] for c in chunks]
            if self.dimensions is not None:
                vectors = [mrl_truncate(v, self.dimensions) for v in vectors]
            out.append(vectors)
        return out

    async def embed_grouped(
        self, documents: list[list[str]], *, input_type: InputType | None = None
    ) -> list[list[list[float]]]:
        """Embed several documents at once; returns one vector list per document."""
        if not documents:
            return []
        client = self._client or httpx.AsyncClient(timeout=self.timeout_s)
        try:
            response = await client.post(
                self.base_url, json=self._grouped_payload(documents, input_type), headers=self._headers()
            )
            if response.status_code >= 400:
                raise ProviderError(
                    f"embedder {self.name!r} error {response.status_code}: {response.text[:500]}",
                    provider=self.name,
                )
            groups = self._parse_grouped(response.json())
        finally:
            if self._client is None:
                await client.aclose()
        flat = [v for group in groups for v in group]
        if flat:
            self.dim = len(flat[0])
        return groups

    async def embed(
        self, texts: list[str], *, input_type: InputType | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        groups = await self.embed_grouped([texts], input_type=input_type)
        return groups[0] if groups else []


class MultimodalInput(BaseModel):
    """One multimodal embedding input: text, an image, or both."""

    text: str | None = None
    image: ImageRef | None = None


class MultimodalEmbedder(HTTPEmbedder):
    """Base for unified text+image embedders (one shared vector space).

    ``embed(texts)`` embeds text-only inputs; :meth:`embed_multimodal` accepts
    :class:`MultimodalInput` items (text, image, or both) so an image and the
    text that describes it land near each other and can be retrieved together.
    Images are sent inline as base64 data URLs (or by ``url`` when set).
    """

    def _encode_image(self, image: ImageRef) -> str:
        if not image.url and not image.path:
            raise ConfigError("MultimodalInput image needs a path or url")
        # Shared encoder: remote URL passthrough or a base64 data URL (with the
        # size guardrail), identical to the chat providers.
        from ..core.media import image_to_data_url

        return image_to_data_url(image)

    def _multimodal_payload(
        self, items: list[MultimodalInput], input_type: InputType | None
    ) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    async def embed_multimodal(
        self, items: list[MultimodalInput], *, input_type: InputType | None = None
    ) -> list[list[float]]:
        if not items:
            return []
        return await self._post(self._multimodal_payload(items, input_type))

    async def embed(
        self, texts: list[str], *, input_type: InputType | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        return await self.embed_multimodal(
            [MultimodalInput(text=text) for text in texts], input_type=input_type
        )


class VoyageMultimodalEmbedder(MultimodalEmbedder):
    """Voyage multimodal embeddings (``/v1/multimodalembeddings``).

    Each input is a content list of text and image parts; the returned vector
    lives in the same space as text-only Voyage vectors. Default model
    ``voyage-multimodal-3``."""

    name = "voyage-multimodal"
    default_base_url = "https://api.voyageai.com/v1/multimodalembeddings"
    default_model = "voyage-multimodal-3"
    _dimensions_field = "output_dimension"

    def _content(self, item: MultimodalInput) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if item.text:
            parts.append({"type": "text", "text": item.text})
        if item.image is not None:
            encoded = self._encode_image(item.image)
            if encoded.startswith("data:"):
                parts.append({"type": "image_base64", "image_base64": encoded})
            else:
                parts.append({"type": "image_url", "image_url": encoded})
        return parts

    def _multimodal_payload(
        self, items: list[MultimodalInput], input_type: InputType | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "inputs": [{"content": self._content(item)} for item in items],
        }
        if input_type is not None:
            payload["input_type"] = input_type
        if self.dimensions is not None:
            payload[self._dimensions_field] = self.dimensions
        return payload


class CohereMultimodalEmbedder(MultimodalEmbedder):
    """Cohere v4 unified text+image embeddings (``/v2/embed``, ``embed-v4.0``).

    Cohere v4 embeds text and images into one space; inputs carry a content
    list and the corpus/query ``input_type``. Returns ``embeddings.float``."""

    name = "cohere-multimodal"
    default_base_url = "https://api.cohere.com/v2/embed"
    default_model = "embed-v4.0"
    _dimensions_field = "output_dimension"

    _COHERE_INPUT_TYPE = {"document": "search_document", "query": "search_query"}

    def __init__(self, *, input_type: str = "search_document", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.input_type = input_type

    def _content(self, item: MultimodalInput) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if item.text:
            parts.append({"type": "text", "text": item.text})
        if item.image is not None:
            parts.append({"type": "image_url", "image_url": {"url": self._encode_image(item.image)}})
        return parts

    def _multimodal_payload(
        self, items: list[MultimodalInput], input_type: InputType | None
    ) -> dict[str, Any]:
        resolved = self._COHERE_INPUT_TYPE[input_type] if input_type is not None else self.input_type
        payload: dict[str, Any] = {
            "model": self.model,
            "input_type": resolved,
            "embedding_types": ["float"],
            "inputs": [{"content": self._content(item)} for item in items],
        }
        if self.dimensions is not None:
            payload[self._dimensions_field] = self.dimensions
        return payload

    def _parse(self, data: dict[str, Any]) -> list[list[float]]:
        embeddings = data.get("embeddings") or {}
        if isinstance(embeddings, dict):
            return list(embeddings.get("float") or [])
        return list(embeddings)


_HTTP_EMBEDDERS: dict[str, type[HTTPEmbedder]] = {
    "jina": JinaEmbedder,
    "voyage": VoyageEmbedder,
    "cohere": CohereEmbedder,
    "voyage-context": VoyageContextualEmbedder,
    "voyage_context": VoyageContextualEmbedder,
    "voyage-multimodal": VoyageMultimodalEmbedder,
    "voyage_multimodal": VoyageMultimodalEmbedder,
    "cohere-multimodal": CohereMultimodalEmbedder,
    "cohere-v4": CohereMultimodalEmbedder,
}


_DISCOVERED_EMBEDDERS: dict[str, Any] | None = None


def _discovered_embedders() -> dict[str, Any]:
    """Third-party embedders advertised under the ``vincio.embedders`` entry-point
    group (discovered once, then cached)."""
    global _DISCOVERED_EMBEDDERS
    if _DISCOVERED_EMBEDDERS is None:
        from ..providers.registry import discover_entry_points

        _DISCOVERED_EMBEDDERS = discover_entry_points("vincio.embedders")
    return _DISCOVERED_EMBEDDERS


class FastEmbedEmbedder:
    """Local ONNX dense embedder via ``fastembed``.

    Batteries-included on-device embeddings with real semantic quality and true
    offline inference — no server. Lazily loads the ONNX model the first time it
    runs; with ``fallback=True`` it degrades to the deterministic
    :class:`LocalHashEmbedder` so the dependency-free path still works. Pass
    ``encode_fn`` for a custom encoder or ``model`` for a ``TextEmbedding``-shaped
    object (offline tests drive the real ``model.embed`` executor path against a
    faithful fake). Install with ``pip install "vincio[fastembed]"``.
    """

    supports_input_type = False
    supports_dimensions = False

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        dim: int = 384,
        encode_fn: Any = None,
        model: Any = None,
        fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self._encode_fn = encode_fn
        self._fallback = fallback
        self._model: Any = model
        self._fallback_embedder: LocalHashEmbedder | None = None

    def _ensure(self) -> None:
        if (
            self._encode_fn is not None
            or self._model is not None
            or self._fallback_embedder is not None
        ):
            return
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        except ImportError as exc:
            if self._fallback:
                self._fallback_embedder = LocalHashEmbedder(dim=self.dim)
                return
            raise ConfigError(
                'the local ONNX embedder requires: pip install "vincio[fastembed]" '
                "(or construct with fallback=True / inject encode_fn)"
            ) from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        if self._encode_fn is not None:
            return [list(map(float, v)) for v in self._encode_fn(texts)]
        if self._fallback_embedder is not None:
            return await self._fallback_embedder.embed(texts)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: [list(map(float, v)) for v in self._model.embed(list(texts))]
        )


class ColBERTTokenEmbedder:
    """Token-level dense embedder for late interaction / ColBERT.

    :class:`~vincio.retrieval.late_interaction.LateInteractionIndex` embeds
    individual tokens and scores by MaxSim; this provides ColBERT-quality token
    vectors behind the same :class:`Embedder` interface. Injectable via
    ``encode_fn``, with a deterministic fallback so the offline path holds.
    """

    supports_input_type = False
    supports_dimensions = False

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        *,
        dim: int = 128,
        encode_fn: Any = None,
        fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self._encode_fn = encode_fn
        self._fallback = fallback
        self._fallback_embedder: LocalHashEmbedder | None = None

    def _ensure(self) -> None:
        if self._encode_fn is not None or self._fallback_embedder is not None:
            return
        if self._fallback:
            self._fallback_embedder = LocalHashEmbedder(dim=self.dim)
            return
        raise ConfigError(
            "the ColBERT token embedder requires a local model; inject encode_fn "
            "or construct with fallback=True"
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        if self._encode_fn is not None:
            return [list(map(float, v)) for v in self._encode_fn(texts)]
        assert self._fallback_embedder is not None
        return await self._fallback_embedder.embed(texts)


def build_embedder(
    kind: str = "local",
    *,
    config: Any | None = None,
    dimensions: int | None = None,
    **kwargs: Any,
) -> Embedder:
    """Construct an embedder by name.

    - ``local`` → :class:`LocalHashEmbedder` (deterministic, dependency-free)
    - ``fastembed`` → :class:`FastEmbedEmbedder` (local ONNX, batteries-included)
    - ``jina`` / ``voyage`` / ``cohere`` → hosted HTTP embedders (httpx only)
    - ``voyage-context`` → :class:`VoyageContextualEmbedder` (contextual chunks)
    - ``voyage-multimodal`` / ``cohere-multimodal`` (``cohere-v4``) →
      unified text+image embedders
    - any provider name (``openai``, ``google``, ``mistral``, ``groq``, …) →
      :class:`ProviderEmbedder` over :func:`vincio.providers.build_provider`

    ``dimensions`` enables Matryoshka (MRL) truncation: hosted embedders that
    support it request the shorter vector natively; everything else is wrapped
    in :class:`MatryoshkaEmbedder` so the output is exactly ``dimensions`` long.
    """
    if kind == "local":
        base: Embedder = LocalHashEmbedder(**kwargs)
    elif kind in ("fastembed", "onnx"):
        base = FastEmbedEmbedder(**kwargs)
    elif kind == "colbert":
        base = ColBERTTokenEmbedder(**kwargs)
    elif kind in _HTTP_EMBEDDERS:
        if dimensions is not None:
            kwargs.setdefault("dimensions", dimensions)
        base = _HTTP_EMBEDDERS[kind](**kwargs)
    elif kind in _discovered_embedders():
        if dimensions is not None:
            kwargs.setdefault("dimensions", dimensions)
        base = _discovered_embedders()[kind](**kwargs)
    else:
        try:
            from ..providers import build_provider

            provider = build_provider(kind, config)
        except Exception as exc:  # noqa: BLE001 - surface a clear configuration error
            raise ConfigError(f"unknown embedder {kind!r}: {exc}") from exc
        model = kwargs.pop("model", None)
        base = ProviderEmbedder(provider, model=model, **kwargs)
    if dimensions is not None and not getattr(base, "supports_dimensions", False):
        # MatryoshkaEmbedder wraps any Embedder; it exposes ``dim`` as a
        # read-only property, which the Protocol's settable attr can't express.
        base = cast("Embedder", MatryoshkaEmbedder(base, dimensions))
    return base
