"""Shared media encoding for provider payloads.

One helper turns an :class:`~vincio.core.types.ImageRef` into the base64 / data
URL forms the chat providers and multimodal embedders need, so a local image
path is never emitted as an unreachable ``file://`` URL (the OpenAI bug 1.7
fixes) and the size guardrail is enforced once, in one place.
"""

from __future__ import annotations

import base64
from pathlib import Path

from .errors import InputError
from .types import ImageRef

__all__ = [
    "DEFAULT_MAX_IMAGE_BYTES",
    "encode_image_bytes",
    "image_to_data_url",
]

# Provider payload guardrail: refuse to inline an image larger than this so a
# stray path can't balloon a request (and bill) unbounded. Override per call.
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


def encode_image_bytes(
    image: ImageRef, *, max_bytes: int = DEFAULT_MAX_IMAGE_BYTES
) -> tuple[str, str]:
    """Return ``(media_type, base64_data)`` for a local-path image.

    Reads the file, enforces the byte-size cap, and base64-encodes it. Raises
    :class:`InputError` when there is no local path or the file exceeds the cap.
    (Pixel-dimension capping needs an optional image backend; the byte cap is
    the dependency-free guardrail.)
    """
    if not image.path:
        raise InputError("image has no local path to encode")
    data = Path(image.path).read_bytes()
    if len(data) > max_bytes:
        raise InputError(
            f"image {image.path!r} is {len(data)} bytes, over the {max_bytes}-byte cap"
        )
    media_type = image.media_type or "image/png"
    return media_type, base64.standard_b64encode(data).decode("ascii")


def image_to_data_url(image: ImageRef, *, max_bytes: int = DEFAULT_MAX_IMAGE_BYTES) -> str:
    """A ``data:`` URL for a local image, or the remote ``url`` passthrough.

    Used by the OpenAI/Google-style ``image_url`` payloads. A local path is
    base64-encoded into a data URL (never ``file://``); a remote URL is passed
    through unchanged.
    """
    if image.url:
        return image.url
    media_type, encoded = encode_image_bytes(image, max_bytes=max_bytes)
    return f"data:{media_type};base64,{encoded}"
