"""OCR for scanned documents and images (scanned PDFs).

Two engines:

- :class:`TesseractOCR` — local OCR via pytesseract (requires the tesseract
  binary and ``pip install pytesseract pillow``).
- :class:`VisionModelOCR` — uses any vision-capable :class:`ModelProvider`
  to transcribe an image; works wherever a provider is configured.

Both implement :class:`OCREngine`; the document pipeline accepts either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..core.errors import DocumentError
from ..core.types import ContentPart, ImageRef, Message, ModelRequest
from ..providers.base import ModelProvider

__all__ = ["OCREngine", "TesseractOCR", "VisionModelOCR"]


class OCREngine(Protocol):
    async def extract_text(self, image_path: str | Path) -> str:  # pragma: no cover
        ...


class TesseractOCR:
    """Local OCR. Requires the tesseract binary + pytesseract + pillow."""

    def __init__(self, *, language: str = "eng") -> None:
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise DocumentError(
                "TesseractOCR requires: pip install pytesseract pillow "
                "(and the tesseract system binary)"
            ) from exc
        self.language = language

    async def extract_text(self, image_path: str | Path) -> str:
        import asyncio

        import pytesseract
        from PIL import Image

        def run() -> str:
            with Image.open(str(image_path)) as image:
                return pytesseract.image_to_string(image, lang=self.language)

        return await asyncio.get_running_loop().run_in_executor(None, run)


OCR_PROMPT = (
    "Transcribe ALL text visible in this image exactly as written. "
    "Preserve line breaks and table layout. Output only the transcription."
)


class VisionModelOCR:
    """OCR through a vision-capable model provider."""

    def __init__(self, provider: ModelProvider, *, model: str) -> None:
        capabilities = provider.capabilities(model)
        if not capabilities.vision:
            raise DocumentError(f"model {model!r} does not support vision input")
        self.provider = provider
        self.model = model

    async def extract_text(self, image_path: str | Path) -> str:
        request = ModelRequest(
            model=self.model,
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentPart(type="text", text=OCR_PROMPT),
                        ContentPart(type="image", image=ImageRef(path=str(image_path))),
                    ],
                )
            ],
            temperature=0.0,
        )
        response = await self.provider.generate(request)
        return response.text
