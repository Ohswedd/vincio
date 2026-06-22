"""Video generation / editing provider abstraction.

A neutral surface — ``generate_video`` / ``edit_video`` — over OpenAI Sora,
Google Veo, and a generic HTTP adapter, with a deterministic
:class:`MockVideoProvider` for offline tests. Every generated clip carries a
media-aware C2PA manifest bound to its bytes, a usage cost, and (at the app
boundary) is metered against the run budget — so synthetic video is as
tamper-evident and auditable as a generated image, an audio clip, or a text
answer.
"""

from __future__ import annotations

import base64
import hashlib
import struct
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import MediaGenerationError
from ..core.types import ImageRef, VideoRef
from ..governance.transparency import ProvenanceManifest
from .media import attach_media_provenance, video_cost

__all__ = [
    "VideoGenRequest",
    "GeneratedVideo",
    "VideoGenResponse",
    "VideoProvider",
    "MockVideoProvider",
    "OpenAIVideoProvider",
    "GoogleVideoProvider",
    "HTTPVideoProvider",
]

VideoQuality = Literal["low", "medium", "high", "standard", "hd", "auto"]

_VIDEO_MEDIA = {"mp4": "video/mp4", "webm": "video/webm", "mov": "video/quicktime"}


class VideoGenRequest(BaseModel):
    prompt: str
    seconds: float = Field(default=5.0, gt=0)
    fps: int = Field(default=24, ge=1)
    size: str = "1280x720"
    quality: VideoQuality = "auto"
    format: Literal["mp4", "webm", "mov"] = "mp4"
    seed: int | None = None
    # Optional first-frame / reference image to condition the clip, and an input
    # clip for an edit/extend operation.
    reference_image: ImageRef | None = None
    reference_video: VideoRef | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def media_type(self) -> str:
        return _VIDEO_MEDIA.get(self.format, "video/mp4")


class GeneratedVideo(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    data: bytes
    media_type: str = "video/mp4"
    seconds: float = 0.0
    revised_prompt: str | None = None
    seed: int | None = None
    manifest: ProvenanceManifest | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def save(self, path: str) -> str:
        from pathlib import Path

        Path(path).write_bytes(self.data)
        return path

    def to_ref(self, path: str) -> VideoRef:
        """Save and return a :class:`VideoRef` pointing at the saved file."""
        self.save(path)
        return VideoRef(
            path=path, media_type=self.media_type, duration_seconds=self.seconds or None
        )


class VideoGenResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    videos: list[GeneratedVideo] = Field(default_factory=list)
    model: str = ""
    provider: str = ""
    cost_usd: float = 0.0
    seconds: float = 0.0
    revised_prompt: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)


class VideoProvider(ABC):
    """Abstract video generation/editing provider."""

    name: str = "video"

    @abstractmethod
    async def generate_video(self, request: VideoGenRequest, *, model: str) -> VideoGenResponse: ...

    async def edit_video(
        self, video: VideoRef, request: VideoGenRequest, *, model: str
    ) -> VideoGenResponse:
        raise MediaGenerationError(f"provider {self.name!r} does not support video editing")

    async def aclose(self) -> None:
        return None

    # Shared: stamp each produced clip with bound provenance.
    def _stamp(
        self,
        raw: list[bytes],
        *,
        model: str,
        media_type: str,
        seconds: float,
        edited: bool,
        revised: str | None,
    ) -> list[GeneratedVideo]:
        out: list[GeneratedVideo] = []
        for data in raw:
            stamped, manifest = attach_media_provenance(
                data, media_type=media_type, model=model, provider=self.name, edited=edited
            )
            out.append(
                GeneratedVideo(
                    data=stamped,
                    media_type=media_type,
                    seconds=seconds,
                    revised_prompt=revised,
                    manifest=manifest,
                )
            )
        return out


# -- MP4 box helper (dependency-free, for the mock) ---------------------------


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + box_type + payload


def _stub_mp4(seed_bytes: bytes) -> bytes:
    """A deterministic, minimal ISO-BMFF (MP4) box stream for offline tests.

    Structurally a valid ``ftyp`` + ``mdat`` box stream whose payload is derived
    from the prompt/seed (so a given prompt is reproducible and two prompts
    differ). It is a deterministic offline *stub*, not an encoded/playable clip —
    real video comes from a provider; this exists to exercise the generation,
    provenance, and budgeting path without a codec.
    """
    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 0x200) + b"isomiso2mp41")
    mdat = _box(b"mdat", b"vincio-mock-video\x00" + seed_bytes)
    return ftyp + mdat


class MockVideoProvider(VideoProvider):
    """Deterministic offline video provider.

    Produces a deterministic MP4 box stream whose bytes are derived from the
    prompt and seed, with a cost from the overridable media price table. Used by
    tests and offline development.
    """

    name = "mock-video"

    def __init__(self, *, default_model: str = "mock-video") -> None:
        self.default_model = default_model
        self.requests: list[VideoGenRequest] = []

    def _render(self, request: VideoGenRequest, *, salt: str = "") -> bytes:
        digest = hashlib.sha256(
            f"{salt}{request.prompt}{request.seed}{request.seconds}{request.size}".encode()
        ).digest()
        return _stub_mp4(digest)

    async def generate_video(
        self, request: VideoGenRequest, *, model: str = "mock-video"
    ) -> VideoGenResponse:
        self.requests.append(request)
        raw = [self._render(request)]
        videos = self._stamp(
            raw,
            model=model,
            media_type="video/mp4",
            seconds=request.seconds,
            edited=False,
            revised=f"mock revision of: {request.prompt[:80]}",
        )
        return VideoGenResponse(
            videos=videos,
            model=model,
            provider=self.name,
            seconds=request.seconds,
            cost_usd=video_cost(model, seconds=request.seconds, quality=request.quality),
            revised_prompt=videos[0].revised_prompt if videos else None,
        )

    async def edit_video(
        self, video: VideoRef, request: VideoGenRequest, *, model: str = "mock-video"
    ) -> VideoGenResponse:
        self.requests.append(request)
        raw = [self._render(request, salt="edit:")]
        videos = self._stamp(
            raw, model=model, media_type="video/mp4", seconds=request.seconds, edited=True,
            revised=None,
        )
        return VideoGenResponse(
            videos=videos, model=model, provider=self.name, seconds=request.seconds,
            cost_usd=video_cost(model, seconds=request.seconds, quality=request.quality),
        )


# -- shared async job polling -------------------------------------------------


async def _poll_until_done(
    http: Any,
    url: str,
    headers: dict[str, str],
    *,
    is_done: Any,
    max_attempts: int = 60,
    interval_s: float = 5.0,
) -> dict[str, Any]:
    """Poll a long-running generation job until ``is_done(payload)`` is true.

    Video generation is asynchronous on every real provider; this bounds the
    wait so a stuck job raises rather than hanging forever.
    """
    import asyncio

    for _ in range(max_attempts):
        response = await http.get(url, headers=headers)
        if response.status_code >= 400:
            raise MediaGenerationError(
                f"video job poll error {response.status_code}: {response.text[:500]}"
            )
        payload = response.json()
        if is_done(payload):
            return payload
        await asyncio.sleep(interval_s)
    raise MediaGenerationError(f"video job did not complete after {max_attempts} polls")


# -- OpenAI Sora --------------------------------------------------------------


class OpenAIVideoProvider(VideoProvider):
    """OpenAI video (Sora) over httpx: create a job, poll, download the bytes."""

    name = "openai"

    def __init__(
        self, api_key: str | None = None, *, base_url: str = "https://api.openai.com/v1"
    ) -> None:
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

    async def generate_video(
        self, request: VideoGenRequest, *, model: str = "sora-2"
    ) -> VideoGenResponse:
        body = {
            "model": model,
            "prompt": request.prompt,
            "seconds": str(int(request.seconds)),
            "size": request.size,
        }
        http = self._http()
        created = await http.post(f"{self.base_url}/videos", headers=self._headers(), json=body)
        if created.status_code >= 400:
            raise MediaGenerationError(
                f"OpenAI video error {created.status_code}: {created.text[:500]}"
            )
        job = created.json()
        job_id = job.get("id")
        if job.get("status") not in ("completed", "succeeded"):
            job = await _poll_until_done(
                http,
                f"{self.base_url}/videos/{job_id}",
                self._headers(),
                is_done=lambda p: p.get("status") in ("completed", "succeeded", "failed"),
            )
        if job.get("status") == "failed":
            raise MediaGenerationError(f"OpenAI video job failed: {str(job.get('error'))[:300]}")
        content = await http.get(
            f"{self.base_url}/videos/{job_id}/content", headers=self._headers()
        )
        if content.status_code >= 400:
            raise MediaGenerationError(
                f"OpenAI video download error {content.status_code}: {content.text[:300]}"
            )
        videos = self._stamp(
            [content.content], model=model, media_type=request.media_type,
            seconds=request.seconds, edited=False, revised=None,
        )
        return VideoGenResponse(
            videos=videos, model=model, provider=self.name, seconds=request.seconds,
            cost_usd=video_cost(model, seconds=request.seconds, quality=request.quality),
        )

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


# -- Google Veo ---------------------------------------------------------------


class GoogleVideoProvider(VideoProvider):
    """Google Veo (``:predictLongRunning``) over httpx: submit, poll, decode."""

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

    async def generate_video(
        self, request: VideoGenRequest, *, model: str = "veo-3.0-generate-001"
    ) -> VideoGenResponse:
        headers = {"x-goog-api-key": self.api_key or ""}
        body = {
            "instances": [{"prompt": request.prompt}],
            "parameters": {"durationSeconds": int(request.seconds), "sampleCount": 1},
        }
        http = self._http()
        submitted = await http.post(
            f"{self.base_url}/models/{model}:predictLongRunning", headers=headers, json=body
        )
        if submitted.status_code >= 400:
            raise MediaGenerationError(
                f"Veo error {submitted.status_code}: {submitted.text[:500]}"
            )
        op_name = submitted.json().get("name")
        done = await _poll_until_done(
            http, f"{self.base_url}/{op_name}", headers, is_done=lambda p: p.get("done")
        )
        if done.get("error"):
            raise MediaGenerationError(f"Veo job failed: {str(done['error'])[:300]}")
        raw: list[bytes] = []
        predictions = (done.get("response") or {}).get("predictions") or (
            (done.get("response") or {}).get("generatedVideos") or []
        )
        for prediction in predictions:
            b64 = (
                prediction.get("bytesBase64Encoded")
                or prediction.get("video", {}).get("videoBytes")
            )
            if b64:
                raw.append(base64.b64decode(b64))
        if not raw:
            raise MediaGenerationError("Veo returned no video data")
        videos = self._stamp(
            raw, model=model, media_type=request.media_type, seconds=request.seconds,
            edited=False, revised=None,
        )
        return VideoGenResponse(
            videos=videos, model=model, provider=self.name, seconds=request.seconds,
            cost_usd=video_cost(model, seconds=request.seconds, quality=request.quality),
        )

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


# -- Generic HTTP / Replicate -------------------------------------------------


class HTTPVideoProvider(VideoProvider):
    """Generic JSON video endpoint (Replicate-style) returning base64 clips.

    ``response_path`` is a dotted path to a list of base64 strings in the JSON
    response (default ``output``).
    """

    name = "http-video"

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
            self._client = httpx.AsyncClient(timeout=300.0)
        return self._client

    async def generate_video(
        self, request: VideoGenRequest, *, model: str = ""
    ) -> VideoGenResponse:
        headers = dict(self.extra_headers)
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        body = {
            "model": model,
            "prompt": request.prompt,
            "seconds": request.seconds,
            "size": request.size,
        }
        response = await self._http().post(self.endpoint, headers=headers, json=body)
        if response.status_code >= 400:
            raise MediaGenerationError(
                f"video endpoint error {response.status_code}: {response.text[:500]}"
            )
        node: Any = response.json()
        for part in self.response_path.split("."):
            node = node.get(part, []) if isinstance(node, dict) else []
        raw = [base64.b64decode(b) for b in (node or []) if isinstance(b, str)]
        if not raw:
            raise MediaGenerationError("video endpoint returned no base64 clip data")
        videos = self._stamp(
            raw, model=model or self.name, media_type=request.media_type,
            seconds=request.seconds, edited=False, revised=None,
        )
        return VideoGenResponse(
            videos=videos, model=model or self.name, provider=self.name, seconds=request.seconds,
            cost_usd=video_cost(model, seconds=request.seconds, quality=request.quality),
        )

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
