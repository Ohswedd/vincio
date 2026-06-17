"""Shared media encoding for provider payloads.

One helper turns an :class:`~vincio.core.types.ImageRef` into the base64 / data
URL forms the chat providers and multimodal embedders need, so a local image
path is never emitted as an unreachable ``file://`` URL (the OpenAI bug 1.7
fixes) and the size guardrail is enforced once, in one place. The audio
companion (1.9) does the same for :class:`~vincio.core.types.AudioRef`, so
``ContentPart.audio`` is finally usable as chat input, and
:func:`media_sha256` gives the C2PA provenance marker one digest that binds
text *or* raw media bytes.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from .errors import InputError
from .types import AudioRef, ImageRef

__all__ = [
    "DEFAULT_MAX_IMAGE_BYTES",
    "DEFAULT_MAX_AUDIO_BYTES",
    "encode_image_bytes",
    "image_to_data_url",
    "encode_audio_bytes",
    "audio_format_label",
    "media_sha256",
]

# Provider payload guardrail: refuse to inline an image larger than this so a
# stray path can't balloon a request (and bill) unbounded. Override per call.
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB

# Audio is more compressible than images and turns are longer, so the byte cap
# is higher; it still bounds an inlined chat-audio part to a sane request size.
DEFAULT_MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB

# Map common audio MIME types to the short format label the chat APIs expect
# (OpenAI ``input_audio.format``; informational elsewhere).
_AUDIO_FORMAT_BY_MIME = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "audio/aac": "aac",
    "audio/pcm": "pcm16",
}


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


def audio_format_label(media_type: str | None) -> str:
    """The short format label (``wav``/``mp3``/…) for an audio MIME type.

    Falls back to the type's subtype, then ``wav``. Used to fill the chat
    providers' ``input_audio.format`` field from an :class:`AudioRef`.
    """
    if not media_type:
        return "wav"
    label = _AUDIO_FORMAT_BY_MIME.get(media_type.lower())
    if label:
        return label
    subtype = media_type.split("/", 1)[-1].lower()
    if subtype.startswith("x-"):  # strip the experimental-subtype prefix, not a char set
        subtype = subtype[2:]
    return subtype or "wav"


def encode_audio_bytes(
    audio: AudioRef, *, max_bytes: int = DEFAULT_MAX_AUDIO_BYTES
) -> tuple[str, str]:
    """Return ``(media_type, base64_data)`` for a local-path audio clip.

    The audio companion to :func:`encode_image_bytes`: reads the file, enforces
    the byte cap, and base64-encodes it, so a typed :class:`AudioRef` becomes a
    chat ``input_audio`` part. Raises :class:`InputError` when there is no local
    path or the file exceeds the cap.
    """
    if not audio.path:
        raise InputError("audio has no local path to encode")
    data = Path(audio.path).read_bytes()
    if len(data) > max_bytes:
        raise InputError(
            f"audio {audio.path!r} is {len(data)} bytes, over the {max_bytes}-byte cap"
        )
    media_type = audio.media_type or "audio/wav"
    return media_type, base64.standard_b64encode(data).decode("ascii")


def media_sha256(content: str | bytes) -> str:
    """SHA-256 hex digest of text *or* raw media bytes.

    The single content-binding digest the C2PA provenance marker uses, so a
    credential binds to a generated image/audio byte stream exactly as it binds
    to a text answer (text is UTF-8 encoded first).
    """
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()
