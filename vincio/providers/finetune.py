"""Provider fine-tune job backends.

The distillation flywheel curates production traces into grounded,
provider-ready JSONL and gates a *student* model on the eval suite. These
backends are the "trainer" that turns that JSONL into an actual cheaper model:
they submit and poll real
fine-tune jobs on the first-party APIs, returning the trained model id so the
:class:`~vincio.optimize.distill.BootstrapFinetune` loop can register it and
gate-promote it through the same significance swap gate as a model rotation.

Each backend drives one provider's fine-tune surface over its existing
:class:`~vincio.providers.base.HTTPProvider` transport (so auth strategies,
pooling, and the egress DLP scan all apply), mirroring how the Batch backends
(:mod:`vincio.providers.batch`) reuse the same client. Offline, the lifecycle
runs against ``httpx.MockTransport`` cassettes and the wait loop uses a zero
poll interval, so the executed flywheel is fully deterministic in tests.

This is a library capability inside your process — never a hosted training
service. You bring the API key; Vincio submits, polls, and gates.
"""

from __future__ import annotations

import asyncio
import io
import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..core.errors import FineTuneError

if TYPE_CHECKING:
    from .anthropic import AnthropicProvider
    from .google import GoogleProvider
    from .openai import OpenAIProvider

__all__ = [
    "FineTuneStatus",
    "FineTuneJob",
    "FineTuneBackend",
    "OpenAIFineTuneBackend",
    "GoogleFineTuneBackend",
    "AnthropicFineTuneBackend",
    "make_finetune_backend",
    "run_finetune",
]


class FineTuneStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FineTuneJob(BaseModel):
    """One provider fine-tune job, polled to a terminal state."""

    id: str
    backend: str = ""
    status: FineTuneStatus = FineTuneStatus.PENDING
    base_model: str = ""
    fine_tuned_model: str | None = None
    error: str | None = None
    trained_tokens: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.status in (
            FineTuneStatus.SUCCEEDED,
            FineTuneStatus.FAILED,
            FineTuneStatus.CANCELLED,
        )


@runtime_checkable
class FineTuneBackend(Protocol):
    """Submit and poll a fine-tune job on one provider."""

    name: str

    async def submit(
        self, training_jsonl: str, base_model: str, *, suffix: str | None = None
    ) -> FineTuneJob: ...

    async def poll(self, job: FineTuneJob) -> FineTuneJob: ...

    async def cancel(self, job: FineTuneJob) -> FineTuneJob: ...


class OpenAIFineTuneBackend:
    """Drives the OpenAI fine-tuning API over an :class:`OpenAIProvider`.

    Uploads the curated JSONL as a ``purpose="fine-tune"`` file, creates a
    ``/fine_tuning/jobs`` job, and polls it to completion; the resulting
    ``fine_tuned_model`` id is the trained student.
    """

    name = "openai"

    _STATUS = {
        "validating_files": FineTuneStatus.RUNNING,
        "queued": FineTuneStatus.RUNNING,
        "running": FineTuneStatus.RUNNING,
        "succeeded": FineTuneStatus.SUCCEEDED,
        "failed": FineTuneStatus.FAILED,
        "cancelled": FineTuneStatus.CANCELLED,
    }

    def __init__(self, provider: OpenAIProvider) -> None:
        self.provider = provider

    def _job_from(self, data: dict[str, Any]) -> FineTuneJob:
        err = data.get("error")
        return FineTuneJob(
            id=data.get("id", ""),
            backend=self.name,
            status=self._STATUS.get(data.get("status", ""), FineTuneStatus.RUNNING),
            base_model=data.get("model", ""),
            fine_tuned_model=data.get("fine_tuned_model"),
            error=json.dumps(err) if err else None,
            trained_tokens=data.get("trained_tokens"),
            raw=data,
        )

    async def submit(
        self, training_jsonl: str, base_model: str, *, suffix: str | None = None
    ) -> FineTuneJob:
        provider = self.provider
        provider._check_key()
        upload = await provider.client.post(
            f"{provider.base_url}/files",
            headers={"Authorization": f"Bearer {provider.api_key}"},
            files={"file": ("training.jsonl", io.BytesIO(training_jsonl.encode()), "application/jsonl")},
            data={"purpose": "fine-tune"},
        )
        provider._raise_for_status(upload)
        file_id = upload.json().get("id")
        if not file_id:
            raise FineTuneError("OpenAI file upload returned no id", provider=self.name)
        payload: dict[str, Any] = {"training_file": file_id, "model": base_model}
        if suffix:
            payload["suffix"] = suffix
        created = await provider._post_json("/fine_tuning/jobs", payload)
        return self._job_from(created)

    async def poll(self, job: FineTuneJob) -> FineTuneJob:
        data = await self.provider._get_json(f"/fine_tuning/jobs/{job.id}")
        return self._job_from(data)

    async def cancel(self, job: FineTuneJob) -> FineTuneJob:
        data = await self.provider._post_json(f"/fine_tuning/jobs/{job.id}/cancel", {})
        return self._job_from(data)


class GoogleFineTuneBackend:
    """Drives the Gemini *tuned models* API over a :class:`GoogleProvider`.

    Creates a ``tunedModels`` resource from inline examples and polls its
    ``state`` (``CREATING`` → ``ACTIVE``); the tuned model's resource ``name``
    (``tunedModels/...``) is the trained student.
    """

    name = "google"

    _STATE = {
        "CREATING": FineTuneStatus.RUNNING,
        "STATE_UNSPECIFIED": FineTuneStatus.RUNNING,
        "ACTIVE": FineTuneStatus.SUCCEEDED,
        "FAILED": FineTuneStatus.FAILED,
    }

    def __init__(self, provider: GoogleProvider) -> None:
        self.provider = provider

    @staticmethod
    def _examples(training_jsonl: str) -> list[dict[str, str]]:
        """Flatten chat JSONL into Gemini's text-in/text-out tuning examples."""
        examples: list[dict[str, str]] = []
        for line in training_jsonl.splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            text_in = "\n".join(
                m.get("content", "") for m in messages if m.get("role") in ("system", "user")
            )
            text_out = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"),
                "",
            )
            examples.append({"textInput": text_in, "output": text_out})
        return examples

    def _job_from(self, data: dict[str, Any], *, base_model: str = "") -> FineTuneJob:
        state = data.get("state", "")
        name = data.get("name")
        return FineTuneJob(
            id=name or data.get("id", ""),
            backend=self.name,
            status=self._STATE.get(state, FineTuneStatus.RUNNING),
            base_model=data.get("baseModel", base_model),
            fine_tuned_model=name if self._STATE.get(state) == FineTuneStatus.SUCCEEDED else None,
            raw=data,
        )

    async def submit(
        self, training_jsonl: str, base_model: str, *, suffix: str | None = None
    ) -> FineTuneJob:
        payload = {
            "baseModel": base_model if base_model.startswith("models/") else f"models/{base_model}",
            "displayName": suffix or "vincio-distilled",
            "tuningTask": {"trainingData": {"examples": {"examples": self._examples(training_jsonl)}}},
        }
        created = await self.provider._post_json("/tunedModels", payload)
        return self._job_from(created, base_model=base_model)

    async def poll(self, job: FineTuneJob) -> FineTuneJob:
        # The job id is the tunedModels/... resource path.
        path = job.id if job.id.startswith("/") else f"/{job.id}"
        data = await self.provider._get_json(path)
        return self._job_from(data, base_model=job.base_model)

    async def cancel(self, job: FineTuneJob) -> FineTuneJob:
        path = job.id if job.id.startswith("/") else f"/{job.id}"
        await self.provider._post_json(f"{path}:cancel", {})
        job.status = FineTuneStatus.CANCELLED
        return job


class AnthropicFineTuneBackend:
    """Drives Claude fine-tuning over an :class:`AnthropicProvider`'s transport.

    Anthropic offers Claude fine-tuning through Amazon Bedrock and enterprise
    deployments rather than a first-party public REST endpoint, so this backend
    targets a configurable fine-tune surface (``submit_path`` / ``poll_path``)
    that defaults to the Bedrock-style ``/fine_tuning/jobs`` shape. Point it at
    your deployment's endpoint (or an OpenAI-compatible proxy); it submits the
    same grounded JSONL and polls the same lifecycle as the other backends.
    """

    name = "anthropic"

    _STATUS = {
        "queued": FineTuneStatus.RUNNING,
        "in_progress": FineTuneStatus.RUNNING,
        "running": FineTuneStatus.RUNNING,
        "completed": FineTuneStatus.SUCCEEDED,
        "succeeded": FineTuneStatus.SUCCEEDED,
        "failed": FineTuneStatus.FAILED,
        "cancelled": FineTuneStatus.CANCELLED,
    }

    def __init__(
        self,
        provider: AnthropicProvider,
        *,
        submit_path: str = "/fine_tuning/jobs",
        poll_path: str = "/fine_tuning/jobs/{id}",
    ) -> None:
        self.provider = provider
        self.submit_path = submit_path
        self.poll_path = poll_path

    def _job_from(self, data: dict[str, Any], *, base_model: str = "") -> FineTuneJob:
        err = data.get("error")
        return FineTuneJob(
            id=data.get("id", ""),
            backend=self.name,
            status=self._STATUS.get(data.get("status", ""), FineTuneStatus.RUNNING),
            base_model=data.get("model", base_model),
            fine_tuned_model=data.get("fine_tuned_model"),
            error=json.dumps(err) if err else None,
            raw=data,
        )

    async def submit(
        self, training_jsonl: str, base_model: str, *, suffix: str | None = None
    ) -> FineTuneJob:
        payload: dict[str, Any] = {
            "model": base_model,
            "training_data": [json.loads(line) for line in training_jsonl.splitlines() if line.strip()],
        }
        if suffix:
            payload["suffix"] = suffix
        created = await self.provider._post_json(self.submit_path, payload)
        return self._job_from(created, base_model=base_model)

    async def poll(self, job: FineTuneJob) -> FineTuneJob:
        data = await self.provider._get_json(self.poll_path.format(id=job.id))
        return self._job_from(data, base_model=job.base_model)

    async def cancel(self, job: FineTuneJob) -> FineTuneJob:
        await self.provider._post_json(f"{self.poll_path.format(id=job.id)}/cancel", {})
        job.status = FineTuneStatus.CANCELLED
        return job


def make_finetune_backend(provider: Any) -> FineTuneBackend:
    """Build the right fine-tune backend for a provider instance.

    Dispatches on the provider's ``name`` so ``app``-level distillation can pick
    the backend without the caller importing the concrete class.
    """
    name = getattr(provider, "name", "")
    if name == "openai":
        return OpenAIFineTuneBackend(provider)
    if name == "google":
        return GoogleFineTuneBackend(provider)
    if name == "anthropic":
        return AnthropicFineTuneBackend(provider)
    raise FineTuneError(
        f"no fine-tune backend for provider {name!r}; pass a concrete "
        "FineTuneBackend (OpenAI/Google/Anthropic) explicitly",
        provider=name or None,
    )


async def run_finetune(
    backend: FineTuneBackend,
    training_jsonl: str,
    base_model: str,
    *,
    suffix: str | None = None,
    poll_interval_s: float = 5.0,
    max_polls: int = 240,
) -> FineTuneJob:
    """Submit a fine-tune job and poll it to a terminal state.

    Raises :class:`FineTuneError` if the job fails, is cancelled, or does not
    finish within ``max_polls`` polls — never returns a non-terminal job. The
    poll interval is ``0`` in tests so the lifecycle is instantaneous against a
    cassette.
    """
    job = await backend.submit(training_jsonl, base_model, suffix=suffix)
    polls = 0
    while not job.done:
        if polls >= max_polls:
            raise FineTuneError(
                f"fine-tune job {job.id!r} did not finish within {max_polls} polls",
                provider=backend.name,
            )
        if poll_interval_s > 0:
            await asyncio.sleep(poll_interval_s)
        job = await backend.poll(job)
        polls += 1
    if job.status is not FineTuneStatus.SUCCEEDED:
        raise FineTuneError(
            f"fine-tune job {job.id!r} ended {job.status.value}: {job.error or 'no detail'}",
            provider=backend.name,
        )
    if not job.fine_tuned_model:
        raise FineTuneError(
            f"fine-tune job {job.id!r} succeeded but returned no model id",
            provider=backend.name,
        )
    return job
