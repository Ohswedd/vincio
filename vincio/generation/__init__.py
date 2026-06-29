"""Vincio generation engine — documents & media flow OUT.

Documents and images flow *out* under the same guarantees Vincio applies to
text *in*: cited, structurally-validated, provenance-stamped, budget-metered,
eval-gated deliverables, all on one trace and one audit chain, in-process.

* :class:`DocumentBuilder` renders a validated result into Markdown/HTML
  (dependency-free) or DOCX/PDF/PPTX (extras), checked against a
  :class:`DocumentContract`, with a ``document_generate`` audit event.
* :class:`CitedReportBuilder` resolves ``[E1]`` markers to footnotes and a
  bibliography, with sentence-level citation coverage and optional per-claim
  entailment — every claim cited *and* supported.
* :class:`ImageProvider` / :class:`SpeechProvider` add image generation/editing
  and TTS as first-class output modalities, every asset C2PA-stamped, metered,
  and gated like text.
* Template/form filling (:func:`fill_text_template`, :func:`fill_docx_form`,
  :func:`fill_pdf_form`) and :func:`generate_redline` round out the deliverables.

The native renderers are dependency-free; DOCX/PDF/PPTX output and richer
inputs install behind the ``vincio[gen-docx|gen-pdf|gen-pptx]`` extras.
"""

from .builder import DocumentBuilder, generate_redline, markdown_to_model
from .contracts import (
    DocumentContract,
    DocumentValidationReport,
    TableSpec,
    repair_formatting,
    validate_document,
)
from .image import (
    GeneratedImage,
    GoogleImageProvider,
    HTTPImageProvider,
    ImageGenRequest,
    ImageGenResponse,
    ImageProvider,
    MockImageProvider,
    OpenAIImageProvider,
)
from .media import (
    attach_media_provenance,
    image_cost,
    meter_media_cost,
    speech_cost,
    video_cost,
)
from .model import DocBlock, DocumentModel
from .render import DocumentArtifact, RenderFormat, render
from .report import (
    CitationContract,
    CitationCoverage,
    CitedReport,
    CitedReportBuilder,
    Figure,
    FigureBinding,
    ResolvedCitation,
)
from .speech import (
    ElevenLabsSpeechProvider,
    GeneratedAudio,
    GoogleSpeechProvider,
    MockSpeechProvider,
    OpenAISpeechProvider,
    SpeechProvider,
    SpeechRequest,
    SpeechResponse,
)
from .templates import Slot, fill_docx_form, fill_pdf_form, fill_text_template
from .video import (
    GeneratedVideo,
    GoogleVideoProvider,
    HTTPVideoProvider,
    MockVideoProvider,
    OpenAIVideoProvider,
    VideoGenRequest,
    VideoGenResponse,
    VideoProvider,
)

__all__ = [
    # document engine
    "DocumentBuilder",
    "DocumentModel",
    "DocBlock",
    "DocumentArtifact",
    "RenderFormat",
    "render",
    "markdown_to_model",
    "generate_redline",
    # contracts
    "DocumentContract",
    "DocumentValidationReport",
    "TableSpec",
    "validate_document",
    "repair_formatting",
    # cited reports
    "CitedReportBuilder",
    "CitedReport",
    "CitationContract",
    "CitationCoverage",
    "ResolvedCitation",
    "Figure",
    "FigureBinding",
    # templates / forms
    "Slot",
    "fill_text_template",
    "fill_docx_form",
    "fill_pdf_form",
    # image generation
    "ImageProvider",
    "ImageGenRequest",
    "ImageGenResponse",
    "GeneratedImage",
    "MockImageProvider",
    "OpenAIImageProvider",
    "GoogleImageProvider",
    "HTTPImageProvider",
    # speech synthesis
    "SpeechProvider",
    "SpeechRequest",
    "SpeechResponse",
    "GeneratedAudio",
    "MockSpeechProvider",
    "OpenAISpeechProvider",
    "GoogleSpeechProvider",
    "ElevenLabsSpeechProvider",
    # video generation
    "VideoProvider",
    "VideoGenRequest",
    "VideoGenResponse",
    "GeneratedVideo",
    "MockVideoProvider",
    "OpenAIVideoProvider",
    "GoogleVideoProvider",
    "HTTPVideoProvider",
    # media plumbing
    "meter_media_cost",
    "attach_media_provenance",
    "image_cost",
    "speech_cost",
    "video_cost",
]
