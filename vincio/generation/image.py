"""Image generation / editing provider abstraction.

A neutral surface — ``generate_image`` / ``edit_image`` / ``variation`` — over
OpenAI ``gpt-image-1``, Gemini/Imagen, and a generic HTTP/Replicate adapter,
with a deterministic :class:`MockImageProvider` for offline tests. Every
generated asset carries a media-aware C2PA manifest bound to its bytes, a usage
cost, and (at the app boundary) is metered against the run budget — so a
generated image is as auditable as a text answer.
"""

from __future__ import annotations

import base64
import hashlib
import struct
import zlib
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import MediaGenerationError
from ..core.media import encode_image_bytes
from ..core.types import ImageRef
from ..governance.transparency import ProvenanceManifest
from .media import attach_media_provenance, image_cost

__all__ = [
    "ImageGenRequest",
    "GeneratedImage",
    "ImageGenResponse",
    "ImageProvider",
    "MockImageProvider",
    "OpenAIImageProvider",
    "GoogleImageProvider",
    "HTTPImageProvider",
]

ImageQuality = Literal["low", "medium", "high", "standard", "hd", "auto"]


class ImageGenRequest(BaseModel):
    prompt: str
    n: int = Field(1, ge=1)
    size: str = "1024x1024"
    quality: ImageQuality = "auto"
    format: Literal["png", "jpeg", "webp"] = "png"
    seed: int | None = None
    background: str | None = None  # "transparent" | "opaque" | "auto"
    reference_images: list[ImageRef] = Field(default_factory=list)
    mask: ImageRef | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def media_type(self) -> str:
        return {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}[self.format]


class GeneratedImage(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    data: bytes
    media_type: str = "image/png"
    revised_prompt: str | None = None
    seed: int | None = None
    manifest: ProvenanceManifest | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def save(self, path: str) -> str:
        from pathlib import Path

        Path(path).write_bytes(self.data)
        return path

    def to_ref(self, path: str) -> ImageRef:
        """Save and return an :class:`ImageRef` pointing at the saved file."""
        self.save(path)
        return ImageRef(path=path, media_type=self.media_type)


class ImageGenResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    images: list[GeneratedImage] = Field(default_factory=list)
    model: str = ""
    provider: str = ""
    cost_usd: float = 0.0
    revised_prompt: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)


class ImageProvider(ABC):
    """Abstract image generation/editing provider."""

    name: str = "image"

    @abstractmethod
    async def generate_image(self, request: ImageGenRequest, *, model: str) -> ImageGenResponse: ...

    async def edit_image(
        self, image: ImageRef, request: ImageGenRequest, *, model: str
    ) -> ImageGenResponse:
        raise MediaGenerationError(f"provider {self.name!r} does not support image editing")

    async def variation(
        self, image: ImageRef, *, model: str, n: int = 1
    ) -> ImageGenResponse:
        raise MediaGenerationError(f"provider {self.name!r} does not support image variations")

    async def aclose(self) -> None:
        return None

    # Shared: stamp each produced image with bound provenance.
    def _stamp(
        self, raw: list[bytes], *, model: str, media_type: str, edited: bool, revised: str | None
    ) -> list[GeneratedImage]:
        out: list[GeneratedImage] = []
        for data in raw:
            stamped, manifest = attach_media_provenance(
                data, media_type=media_type, model=model, provider=self.name, edited=edited
            )
            out.append(
                GeneratedImage(
                    data=stamped, media_type=media_type, revised_prompt=revised, manifest=manifest
                )
            )
        return out


# -- PNG helper (dependency-free, for the mock) -------------------------------


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """A minimal valid RGB PNG filled with one colour (for offline tests)."""
    row = bytes([0]) + bytes(rgb) * width  # filter byte 0 + pixels
    raw = row * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


class MockImageProvider(ImageProvider):
    """Deterministic offline image provider.

    Produces a real, openable PNG whose colour is derived from the prompt (so a
    given prompt is reproducible and two prompts differ), with a cost from the
    overridable media price table. Used by tests and offline development.
    """

    name = "mock-image"

    def __init__(self, *, size: int = 16, default_model: str = "mock-image") -> None:
        self.size = size
        self.default_model = default_model
        self.requests: list[ImageGenRequest] = []

    def _render(self, request: ImageGenRequest, *, salt: str = "") -> list[bytes]:
        out: list[bytes] = []
        for i in range(max(1, request.n)):
            digest = hashlib.sha256(f"{salt}{request.prompt}{request.seed}{i}".encode()).digest()
            rgb = (digest[0], digest[1], digest[2])
            out.append(_solid_png(self.size, self.size, rgb))
        return out

    async def generate_image(
        self, request: ImageGenRequest, *, model: str = "mock-image"
    ) -> ImageGenResponse:
        self.requests.append(request)
        raw = self._render(request)
        images = self._stamp(
            raw, model=model, media_type="image/png", edited=False,
            revised=f"mock revision of: {request.prompt[:80]}",
        )
        return ImageGenResponse(
            images=images,
            model=model,
            provider=self.name,
            cost_usd=image_cost(model, n=request.n, quality=request.quality),
            revised_prompt=images[0].revised_prompt if images else None,
        )

    async def edit_image(
        self, image: ImageRef, request: ImageGenRequest, *, model: str = "mock-image"
    ) -> ImageGenResponse:
        self.requests.append(request)
        raw = self._render(request, salt="edit:")
        images = self._stamp(raw, model=model, media_type="image/png", edited=True, revised=None)
        return ImageGenResponse(
            images=images, model=model, provider=self.name,
            cost_usd=image_cost(model, n=request.n, quality=request.quality),
        )

    async def variation(
        self, image: ImageRef, *, model: str = "mock-image", n: int = 1
    ) -> ImageGenResponse:
        req = ImageGenRequest(prompt="variation", n=n)
        raw = self._render(req, salt="var:")
        images = self._stamp(raw, model=model, media_type="image/png", edited=True, revised=None)
        return ImageGenResponse(
            images=images, model=model, provider=self.name,
            cost_usd=image_cost(model, n=n, quality="auto"),
        )


# -- OpenAI gpt-image-1 -------------------------------------------------------


class OpenAIImageProvider(ImageProvider):
    """OpenAI Images API (``gpt-image-1``) over httpx."""

    name = "openai"

    def __init__(self, api_key: str | None = None, *, base_url: str = "https://api.openai.com/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client: Any = None

    def _http(self) -> Any:
        import httpx

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            from ..core.errors import ProviderAuthError

            raise ProviderAuthError("missing OpenAI API key", provider=self.name)
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _decode(payload: dict[str, Any]) -> tuple[list[bytes], str | None]:
        images: list[bytes] = []
        revised: str | None = None
        for item in payload.get("data") or []:
            revised = item.get("revised_prompt") or revised
            if item.get("b64_json"):
                images.append(base64.b64decode(item["b64_json"]))
        return images, revised

    async def generate_image(
        self, request: ImageGenRequest, *, model: str = "gpt-image-1"
    ) -> ImageGenResponse:
        body = {
            "model": model,
            "prompt": request.prompt,
            "n": request.n,
            "size": request.size,
        }
        if request.quality != "auto":
            body["quality"] = request.quality
        if request.background:
            body["background"] = request.background
        response = await self._http().post(
            f"{self.base_url}/images/generations", headers=self._headers(), json=body
        )
        if response.status_code >= 400:
            raise MediaGenerationError(f"OpenAI image error {response.status_code}: {response.text[:500]}")
        raw, revised = self._decode(response.json())
        images = self._stamp(raw, model=model, media_type=request.media_type, edited=False, revised=revised)
        return ImageGenResponse(
            images=images, model=model, provider=self.name, revised_prompt=revised,
            cost_usd=image_cost(model, n=request.n, quality=request.quality),
        )

    async def edit_image(
        self, image: ImageRef, request: ImageGenRequest, *, model: str = "gpt-image-1"
    ) -> ImageGenResponse:
        media_type, b64 = encode_image_bytes(image)
        files = {"image": ("image", base64.b64decode(b64), media_type)}
        data = {"model": model, "prompt": request.prompt, "n": str(request.n), "size": request.size}
        if request.mask is not None:
            mask_type, mask_b64 = encode_image_bytes(request.mask)
            files["mask"] = ("mask", base64.b64decode(mask_b64), mask_type)
        response = await self._http().post(
            f"{self.base_url}/images/edits", headers=self._headers(), data=data, files=files
        )
        if response.status_code >= 400:
            raise MediaGenerationError(f"OpenAI edit error {response.status_code}: {response.text[:500]}")
        raw, revised = self._decode(response.json())
        images = self._stamp(raw, model=model, media_type=request.media_type, edited=True, revised=revised)
        return ImageGenResponse(
            images=images, model=model, provider=self.name, revised_prompt=revised,
            cost_usd=image_cost(model, n=request.n, quality=request.quality),
        )

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


# -- Google Imagen ------------------------------------------------------------


class GoogleImageProvider(ImageProvider):
    """Google Imagen (``:predict``) over httpx."""

    name = "google"

    def __init__(
        self, api_key: str | None = None,
        *, base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client: Any = None

    def _http(self) -> Any:
        import httpx

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def generate_image(
        self, request: ImageGenRequest, *, model: str = "imagen-4.0-generate-001"
    ) -> ImageGenResponse:
        body = {
            "instances": [{"prompt": request.prompt}],
            "parameters": {"sampleCount": request.n},
        }
        response = await self._http().post(
            f"{self.base_url}/models/{model}:predict",
            headers={"x-goog-api-key": self.api_key or ""},
            json=body,
        )
        if response.status_code >= 400:
            raise MediaGenerationError(f"Imagen error {response.status_code}: {response.text[:500]}")
        raw: list[bytes] = []
        for prediction in response.json().get("predictions") or []:
            b64 = prediction.get("bytesBase64Encoded") or prediction.get("image", {}).get("imageBytes")
            if b64:
                raw.append(base64.b64decode(b64))
        images = self._stamp(raw, model=model, media_type=request.media_type, edited=False, revised=None)
        return ImageGenResponse(
            images=images, model=model, provider=self.name,
            cost_usd=image_cost(model, n=request.n, quality=request.quality),
        )

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


# -- Generic HTTP / Replicate -------------------------------------------------


class HTTPImageProvider(ImageProvider):
    """Generic JSON image endpoint (Replicate-style), returning base64 images.

    ``response_path`` is a dotted path to a list of base64 strings in the JSON
    response (default ``output``); ``payload_fn`` builds the request body.
    """

    name = "http-image"

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        response_path: str = "output",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.response_path = response_path
        self.extra_headers = headers or {}
        self._client: Any = None

    def _http(self) -> Any:
        import httpx

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=180.0)
        return self._client

    async def generate_image(
        self, request: ImageGenRequest, *, model: str = ""
    ) -> ImageGenResponse:
        headers = dict(self.extra_headers)
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        body = {"model": model, "prompt": request.prompt, "n": request.n, "size": request.size}
        response = await self._http().post(self.endpoint, headers=headers, json=body)
        if response.status_code >= 400:
            raise MediaGenerationError(f"image endpoint error {response.status_code}: {response.text[:500]}")
        node: Any = response.json()
        for part in self.response_path.split("."):
            node = node.get(part, []) if isinstance(node, dict) else []
        raw = [base64.b64decode(b) for b in (node or []) if isinstance(b, str)]
        images = self._stamp(raw, model=model or self.name, media_type=request.media_type, edited=False, revised=None)
        return ImageGenResponse(
            images=images, model=model or self.name, provider=self.name,
            cost_usd=image_cost(model, n=request.n, quality=request.quality),
        )

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
