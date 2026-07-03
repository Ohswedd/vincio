"""Document and media flow-out verbs (reports, images, speech, video) — a private mixin of
:class:`~vincio.core.app.ContextApp`.

Extracted verbatim from ``vincio/core/app.py`` (v7.5 structure line): method
source, decorators, comments, and docstrings are unchanged. ``ContextApp``
composes this class, so every method here remains an ``app.*`` verb; the
``self: ContextApp`` annotations keep attribute access type-checked against
the composed app. The standing hygiene lints (:mod:`vincio._error_contract`,
:mod:`vincio._observable_failure`, :mod:`vincio._assert_robustness`)
deliberately keep ``vincio/core/_app_*.py`` in scope despite the private
filename, so the verb surface stays guarded after the split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ..providers.base import run_sync

if TYPE_CHECKING:
    from .app import ContextApp


class _MediaVerbs:
    """Document and media flow-out verbs (reports, images, speech, video). Mixed into :class:`~vincio.core.app.ContextApp`."""

    # -- documents & media flow OUT ------------------------------------

    def build_document(  # type: ignore[misc]
        self: ContextApp,
        source: Any,
        *,
        format: str = "markdown",
        contract: Any | None = None,
        title: str = "",
        evidence_ids: list[str] | None = None,
    ):
        """Render a validated result into a cited, contract-checked artifact.

        Thin wrapper over :class:`~vincio.generation.builder.DocumentBuilder`
        bound to this app's audit log. ``source`` is a validated
        :class:`~vincio.core.types.RunResult`, mapping, or Markdown; ``contract``
        is an optional :class:`~vincio.generation.contracts.DocumentContract`.
        """
        from ..generation.builder import DocumentBuilder

        builder = DocumentBuilder(audit_log=self.audit)
        return builder.build(
            source,
            format=cast("Any", format),
            contract=contract,
            title=title,
            evidence_ids=evidence_ids,
        )

    def cited_report(  # type: ignore[misc]
        self: ContextApp,
        answer: Any,
        evidence: list[Any] | None = None,
        *,
        format: str = "markdown",
        title: str = "",
        contract: Any | None = None,
        entailment: Any | None = None,
        figures: list[Any] | None = None,
        catalog: Any | None = None,
    ):
        """Resolve ``[E1]`` citations into a rendered, footnoted, cited report.

        Synchronous wrapper over
        :class:`~vincio.generation.report.CitedReportBuilder`; use ``acited_report``
        from async code. Evidence defaults to an empty list (markers then resolve
        to nothing and are reported as unresolved). Pass ``figures=`` (a list of
        :class:`~vincio.generation.Figure`) to embed **data-bound** charts/tables —
        each verified to re-derive from its source against ``catalog`` (defaults to
        the app's registered :meth:`data_catalog`)."""
        return run_sync(
            self.acited_report(
                answer,
                evidence,
                format=format,
                title=title,
                contract=contract,
                entailment=entailment,
                figures=figures,
                catalog=catalog,
            )
        )

    async def acited_report(  # type: ignore[misc]
        self: ContextApp,
        answer: Any,
        evidence: list[Any] | None = None,
        *,
        format: str = "markdown",
        title: str = "",
        contract: Any | None = None,
        entailment: Any | None = None,
        figures: list[Any] | None = None,
        catalog: Any | None = None,
    ):
        """Build a cited report from an answer and its evidence (async) → a document artifact."""
        from ..generation.report import CitedReportBuilder

        if catalog is None and figures:
            registered = self.data_catalog()
            catalog = registered if registered.names else None
        builder = CitedReportBuilder(entailment=entailment, audit_log=self.audit)
        return await builder.build(
            answer,
            list(evidence or []),
            format=cast("Any", format),
            title=title,
            contract=contract,
            figures=figures,
            catalog=catalog,
        )

    async def agenerate_image(  # type: ignore[misc]
        self: ContextApp,
        prompt: Any,
        *,
        provider: Any,
        model: str | None = None,
        n: int = 1,
        size: str = "1024x1024",
        budget: Any | None = None,
    ):
        """Generate image(s) through an
        :class:`~vincio.generation.image.ImageProvider`, metered against the
        budget, audited (``image_generate``), and C2PA-stamped per asset."""
        from ..generation.image import ImageGenRequest
        from ..generation.media import meter_media_cost

        request = (
            prompt
            if isinstance(prompt, ImageGenRequest)
            else ImageGenRequest(prompt=str(prompt), n=n, size=size)
        )
        kwargs = {"model": model} if model else {}
        response = await provider.generate_image(request, **kwargs)
        self._meter_and_audit_media(
            "image_generate", response, request.prompt, budget, meter_media_cost
        )
        return response

    def generate_image(  # type: ignore[misc]
        self: ContextApp, prompt: Any, *, provider: Any, model: str | None = None, **kwargs: Any
    ):
        """Synchronous :meth:`agenerate_image`."""
        return run_sync(self.agenerate_image(prompt, provider=provider, model=model, **kwargs))

    async def asynthesize_speech(  # type: ignore[misc]
        self: ContextApp,
        text: str,
        *,
        provider: Any,
        model: str | None = None,
        voice: str = "alloy",
        format: str = "mp3",
        budget: Any | None = None,
    ):
        """Synthesize speech through a
        :class:`~vincio.generation.speech.SpeechProvider`, metered, audited
        (``speech_synthesize``), and audio-provenance-stamped."""
        from ..generation.media import meter_media_cost
        from ..generation.speech import SpeechRequest

        request = SpeechRequest(text=text, voice=voice, format=format)  # type: ignore[arg-type]
        kwargs = {"model": model} if model else {}
        response = await provider.synthesize_speech(request, **kwargs)
        self._meter_and_audit_media("speech_synthesize", response, text, budget, meter_media_cost)
        return response

    def synthesize_speech(  # type: ignore[misc]
        self: ContextApp, text: str, *, provider: Any, model: str | None = None, **kwargs: Any
    ):
        """Synchronous :meth:`asynthesize_speech`."""
        return run_sync(self.asynthesize_speech(text, provider=provider, model=model, **kwargs))

    def _meter_and_audit_media(  # type: ignore[misc]
        self: ContextApp, action: str, response: Any, prompt: str, budget: Any, meter: Any
    ) -> None:
        # Accumulate against the app's cumulative media usage so the cost cap is
        # honored across calls; also record on the cost tracker for cost_report.
        meter(response.cost_usd, budget=budget or self.budget, usage=self._media_usage)
        self.cost_tracker.record_infra(response.cost_usd)
        assets = (
            getattr(response, "images", None)
            or getattr(response, "videos", None)
            or [getattr(response, "audio", None)]
        )
        manifests = [getattr(a, "manifest", None) for a in assets if a is not None]
        self.audit.record(
            action,
            resource=response.model,
            details={
                "provider": response.provider,
                "model": response.model,
                "prompt": prompt[:200],
                "cost_usd": response.cost_usd,
                "assets": len([a for a in assets if a is not None]),
                # frozen audit-detail key — external consumers bind to it.
                "content_sha256": [m.content_hash for m in manifests if m is not None],
            },
        )

    def load_media(self: ContextApp, path: str, *, transcriber: Any, tenant_id: str | None = None):  # type: ignore[misc]
        """Ingest audio/video as a timestamped transcript Document
        (:func:`vincio.documents.load_media`)."""
        from ..documents.loaders import load_media

        return load_media(path, transcriber=transcriber, tenant_id=tenant_id)

    def load_video(self: ContextApp, path: str, *, analyzer: Any, tenant_id: str | None = None):  # type: ignore[misc]
        """Ingest a video as a temporally-segmented Document
        (:func:`vincio.documents.load_video`).

        ``analyzer`` is a :class:`~vincio.documents.video.VideoAnalyzer`
        (``MockVideoAnalyzer`` offline, ``ProviderVideoAnalyzer`` online). Each
        segment becomes a section carrying its ``start`` / ``end`` timestamps, so
        a retrieved claim grounds to a time range, not just a document."""
        from ..documents.loaders import load_video

        return load_video(path, analyzer=analyzer, tenant_id=tenant_id)

    async def agenerate_video(  # type: ignore[misc]
        self: ContextApp,
        prompt: Any,
        *,
        provider: Any,
        model: str | None = None,
        seconds: float = 5.0,
        size: str = "1280x720",
        budget: Any | None = None,
    ):
        """Generate a video through a
        :class:`~vincio.generation.video.VideoProvider`, metered against the
        budget, audited (``video_generate``), and C2PA-stamped per clip."""
        from ..generation.media import meter_media_cost
        from ..generation.video import VideoGenRequest

        request = (
            prompt
            if isinstance(prompt, VideoGenRequest)
            else VideoGenRequest(prompt=str(prompt), seconds=seconds, size=size)
        )
        kwargs = {"model": model} if model else {}
        response = await provider.generate_video(request, **kwargs)
        self._meter_and_audit_media(
            "video_generate", response, request.prompt, budget, meter_media_cost
        )
        return response

    def generate_video(  # type: ignore[misc]
        self: ContextApp, prompt: Any, *, provider: Any, model: str | None = None, **kwargs: Any
    ):
        """Synchronous :meth:`agenerate_video`."""
        return run_sync(self.agenerate_video(prompt, provider=provider, model=model, **kwargs))

    async def aedit_video(  # type: ignore[misc]
        self: ContextApp,
        video: Any,
        prompt: Any,
        *,
        provider: Any,
        model: str | None = None,
        seconds: float = 5.0,
        budget: Any | None = None,
    ):
        """Edit/extend a video through a
        :class:`~vincio.generation.video.VideoProvider`, metered, audited
        (``video_edit``), and C2PA-stamped (the manifest marks it as edited)."""
        from ..generation.media import meter_media_cost
        from ..generation.video import VideoGenRequest

        request = (
            prompt
            if isinstance(prompt, VideoGenRequest)
            else VideoGenRequest(prompt=str(prompt), seconds=seconds)
        )
        kwargs = {"model": model} if model else {}
        response = await provider.edit_video(video, request, **kwargs)
        self._meter_and_audit_media(
            "video_edit", response, request.prompt, budget, meter_media_cost
        )
        return response

    def edit_video(  # type: ignore[misc]
        self: ContextApp, video: Any, prompt: Any, *, provider: Any, model: str | None = None, **kwargs: Any
    ):
        """Synchronous :meth:`aedit_video`."""
        return run_sync(self.aedit_video(video, prompt, provider=provider, model=model, **kwargs))
