"""Batch execution against provider Batch APIs (1.3).

Provider Batch APIs trade latency for a flat ~50% price cut, which is the right
trade for latency-tolerant work: offline evals, bulk extraction, synthetic-data
generation, and the improvement loop. :class:`BatchRunner` submits a set of
:class:`ModelRequest`\\ s, polls the job to completion, and reconciles the
responses back **by custom id** — surfacing partial failures rather than losing
them. The same :class:`ModelResponse` objects come back as a sync call, so the
call sites are switchable sync↔batch.

Three backends ship:

- :class:`InProcessBatchBackend` wraps any :class:`ModelProvider` and runs the
  set locally (bounded concurrency). It is the offline/test/default path and
  the one the mock provider uses, modelling the submit→poll→fetch lifecycle
  deterministically.
- :class:`OpenAIBatchBackend` and :class:`AnthropicBatchBackend` drive the real
  OpenAI Batch and Anthropic Message Batches endpoints over the provider's own
  ``httpx`` client, reusing its payload-building and response-parsing so a
  batched call is byte-for-byte the sync one.

Cost is tracked at the discounted batch rate and the run is traced like any
other model call.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from ..core.concurrency import gather_bounded
from ..core.errors import BatchError, ProviderError
from ..core.types import ModelRequest, ModelResponse
from ..core.utils import new_id
from ..observability.costs import PriceTable, default_price_table
from .base import ModelProvider

if TYPE_CHECKING:  # pragma: no cover
    from .anthropic import AnthropicProvider
    from .openai import OpenAIProvider

__all__ = [
    "BatchStatus",
    "BatchRequest",
    "BatchResult",
    "BatchJob",
    "BatchRunResult",
    "BatchBackend",
    "InProcessBatchBackend",
    "OpenAIBatchBackend",
    "AnthropicBatchBackend",
    "BatchRunner",
]

# Provider Batch APIs bill at half the synchronous rate.
DEFAULT_BATCH_DISCOUNT = 0.5


class BatchStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


_TERMINAL = {BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED, BatchStatus.EXPIRED}


class BatchRequest(BaseModel):
    """One request in a batch, tagged with a caller-chosen ``custom_id``."""

    custom_id: str
    request: ModelRequest


class BatchResult(BaseModel):
    """The outcome of one batch request, reconciled by ``custom_id``."""

    custom_id: str
    response: ModelResponse | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.response is not None and self.error is None


class BatchJob(BaseModel):
    """A submitted batch and its provider-reported progress."""

    id: str
    backend: str = ""
    status: BatchStatus = BatchStatus.PENDING
    total: int = 0
    completed: int = 0
    failed: int = 0
    endpoint: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.status in _TERMINAL


class BatchRunResult(BaseModel):
    """A completed batch: the job, every reconciled result, and total cost."""

    job: BatchJob
    results: list[BatchResult] = Field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def succeeded(self) -> list[BatchResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[BatchResult]:
        return [r for r in self.results if not r.ok]

    def by_id(self) -> dict[str, BatchResult]:
        return {r.custom_id: r for r in self.results}


class BatchBackend(Protocol):
    """A provider-batch transport: submit, poll, fetch, cancel."""

    name: str

    async def submit(self, requests: list[BatchRequest]) -> BatchJob: ...

    async def poll(self, job: BatchJob) -> BatchJob: ...

    async def results(
        self, job: BatchJob, requests: dict[str, ModelRequest]
    ) -> list[BatchResult]: ...

    async def cancel(self, job: BatchJob) -> BatchJob: ...

    async def aclose(self) -> None: ...


class InProcessBatchBackend:
    """Offline batch backend: runs the set through a provider locally.

    Models the async lifecycle deterministically — ``submit`` executes the set
    under bounded concurrency and caches the outcomes; ``poll`` reports it
    complete; ``results`` returns the cached outcomes. Ideal for tests, offline
    development, and the mock provider. A per-request ``fail_if`` predicate can
    inject partial failures.
    """

    name = "in_process"

    def __init__(
        self,
        provider: ModelProvider,
        *,
        concurrency: int = 8,
        fail_if: Callable[[BatchRequest], str | None] | None = None,
    ) -> None:
        self.provider = provider
        self.concurrency = max(1, concurrency)
        self.fail_if = fail_if
        self._results: dict[str, list[BatchResult]] = {}

    async def submit(self, requests: list[BatchRequest]) -> BatchJob:
        job_id = new_id("batch")

        async def run_one(item: BatchRequest) -> BatchResult:
            reason = self.fail_if(item) if self.fail_if else None
            if reason is not None:
                return BatchResult(custom_id=item.custom_id, error=reason)
            try:
                response = await self.provider.generate(item.request)
                return BatchResult(custom_id=item.custom_id, response=response)
            except ProviderError as exc:
                return BatchResult(custom_id=item.custom_id, error=exc.message)

        results = await gather_bounded(
            (run_one(item) for item in requests), limit=self.concurrency
        )
        self._results[job_id] = list(results)
        failed = sum(1 for r in results if not r.ok)
        return BatchJob(
            id=job_id,
            backend=self.name,
            status=BatchStatus.COMPLETED,
            total=len(requests),
            completed=len(requests) - failed,
            failed=failed,
        )

    async def poll(self, job: BatchJob) -> BatchJob:
        return job

    async def results(
        self, job: BatchJob, requests: dict[str, ModelRequest]
    ) -> list[BatchResult]:
        return list(self._results.get(job.id, []))

    async def cancel(self, job: BatchJob) -> BatchJob:
        self._results.pop(job.id, None)
        return job.model_copy(update={"status": BatchStatus.CANCELLED})

    async def aclose(self) -> None:
        return None


def _iter_jsonl(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


class OpenAIBatchBackend:
    """Drives the OpenAI Batch API over an :class:`OpenAIProvider`'s client.

    Uploads the request set as a JSONL file, creates a ``/batches`` job, and on
    completion downloads the output file and parses each line with the
    provider's own response parser.
    """

    name = "openai"

    def __init__(self, provider: OpenAIProvider, *, completion_window: str = "24h") -> None:
        self.provider = provider
        self.endpoint = "/chat/completions"
        self.completion_window = completion_window

    def _line(self, item: BatchRequest) -> dict[str, Any]:
        return {
            "custom_id": item.custom_id,
            "method": "POST",
            "url": f"/v1{self.endpoint}",
            "body": self.provider._payload(item.request),
        }

    async def submit(self, requests: list[BatchRequest]) -> BatchJob:
        provider = self.provider
        provider._check_key()
        body = "\n".join(json.dumps(self._line(item)) for item in requests)
        auth = {"Authorization": f"Bearer {provider.api_key}"}
        upload = await provider.client.post(
            f"{provider.base_url}/files",
            headers=auth,
            files={"file": ("batch.jsonl", io.BytesIO(body.encode()), "application/jsonl")},
            data={"purpose": "batch"},
        )
        provider._raise_for_status(upload)
        file_id = upload.json().get("id")
        if not file_id:
            raise BatchError("OpenAI file upload returned no id", provider=self.name)
        created = await provider._post_json(
            "/batches",
            {
                "input_file_id": file_id,
                "endpoint": f"/v1{self.endpoint}",
                "completion_window": self.completion_window,
            },
        )
        return self._job_from(created)

    def _job_from(self, data: dict[str, Any]) -> BatchJob:
        counts = data.get("request_counts") or {}
        status_map = {
            "validating": BatchStatus.PENDING,
            "in_progress": BatchStatus.RUNNING,
            "finalizing": BatchStatus.RUNNING,
            "completed": BatchStatus.COMPLETED,
            "failed": BatchStatus.FAILED,
            "expired": BatchStatus.EXPIRED,
            "cancelling": BatchStatus.RUNNING,
            "cancelled": BatchStatus.CANCELLED,
        }
        return BatchJob(
            id=data.get("id", ""),
            backend=self.name,
            status=status_map.get(data.get("status", ""), BatchStatus.RUNNING),
            total=counts.get("total", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            endpoint=self.endpoint,
            raw=data,
        )

    async def poll(self, job: BatchJob) -> BatchJob:
        data = await self.provider._get_json(f"/batches/{job.id}")
        return self._job_from(data)

    async def results(
        self, job: BatchJob, requests: dict[str, ModelRequest]
    ) -> list[BatchResult]:
        provider = self.provider
        output_file_id = job.raw.get("output_file_id")
        results: list[BatchResult] = []
        if output_file_id:
            text = await provider._get_text(f"/files/{output_file_id}/content")
            for line in _iter_jsonl(text):
                cid = line.get("custom_id", "")
                request = requests.get(cid) or ModelRequest(model="", messages=[])
                body = (line.get("response") or {}).get("body")
                err = line.get("error")
                if err or body is None:
                    results.append(
                        BatchResult(custom_id=cid, error=json.dumps(err) if err else "no body")
                    )
                    continue
                try:
                    response = provider._parse_response(body, request, latency_ms=0)
                    results.append(BatchResult(custom_id=cid, response=response))
                except ProviderError as exc:
                    results.append(BatchResult(custom_id=cid, error=exc.message))
        error_file_id = job.raw.get("error_file_id")
        if error_file_id:
            text = await provider._get_text(f"/files/{error_file_id}/content")
            for line in _iter_jsonl(text):
                results.append(
                    BatchResult(
                        custom_id=line.get("custom_id", ""),
                        error=json.dumps(line.get("error") or line.get("response") or {}),
                    )
                )
        return results

    async def cancel(self, job: BatchJob) -> BatchJob:
        data = await self.provider._post_json(f"/batches/{job.id}/cancel", {})
        return self._job_from(data)

    async def aclose(self) -> None:
        await self.provider.aclose()


class AnthropicBatchBackend:
    """Drives the Anthropic Message Batches API over an :class:`AnthropicProvider`."""

    name = "anthropic"

    def __init__(self, provider: AnthropicProvider) -> None:
        self.provider = provider
        self.endpoint = "/messages"

    async def submit(self, requests: list[BatchRequest]) -> BatchJob:
        payload = {
            "requests": [
                {"custom_id": item.custom_id, "params": self.provider._payload(item.request)}
                for item in requests
            ]
        }
        data = await self.provider._post_json("/messages/batches", payload)
        return self._job_from(data, total=len(requests))

    def _job_from(self, data: dict[str, Any], *, total: int = 0) -> BatchJob:
        counts = data.get("request_counts") or {}
        status = (
            BatchStatus.COMPLETED if data.get("processing_status") == "ended" else BatchStatus.RUNNING
        )
        if data.get("processing_status") == "canceling":
            status = BatchStatus.RUNNING
        return BatchJob(
            id=data.get("id", ""),
            backend=self.name,
            status=status,
            total=total or sum(int(v) for v in counts.values()),
            completed=counts.get("succeeded", 0),
            failed=(counts.get("errored", 0) or 0) + (counts.get("canceled", 0) or 0),
            endpoint=self.endpoint,
            raw=data,
        )

    async def poll(self, job: BatchJob) -> BatchJob:
        data = await self.provider._get_json(f"/messages/batches/{job.id}")
        return self._job_from(data, total=job.total)

    async def results(
        self, job: BatchJob, requests: dict[str, ModelRequest]
    ) -> list[BatchResult]:
        provider = self.provider
        # results_url may be absolute; fall back to the conventional path.
        results_url = job.raw.get("results_url") or f"{provider.base_url}/messages/batches/{job.id}/results"
        path = results_url[len(provider.base_url):] if results_url.startswith(provider.base_url) else results_url
        text = await provider._get_text(path if path.startswith("/") else f"/messages/batches/{job.id}/results")
        results: list[BatchResult] = []
        for line in _iter_jsonl(text):
            cid = line.get("custom_id", "")
            request = requests.get(cid) or ModelRequest(model="", messages=[])
            outcome = line.get("result") or {}
            kind = outcome.get("type")
            if kind == "succeeded" and outcome.get("message"):
                try:
                    response = provider._parse_response(outcome["message"], request, latency_ms=0)
                    results.append(BatchResult(custom_id=cid, response=response))
                except ProviderError as exc:
                    results.append(BatchResult(custom_id=cid, error=exc.message))
            else:
                results.append(
                    BatchResult(custom_id=cid, error=json.dumps(outcome.get("error") or {"type": kind}))
                )
        return results

    async def cancel(self, job: BatchJob) -> BatchJob:
        data = await self.provider._post_json(f"/messages/batches/{job.id}/cancel", {})
        return self._job_from(data, total=job.total)

    async def aclose(self) -> None:
        await self.provider.aclose()


class BatchRunner:
    """Submit a batch, poll it to completion, reconcile, and cost-track.

    ``backend`` is any :class:`BatchBackend`; pass a provider to default to the
    in-process backend. ``await_results`` / ``run`` block until the job reaches
    a terminal state (or ``timeout_s``), and reconcile responses by custom id —
    missing ids surface as failed :class:`BatchResult`\\ s, never silently
    dropped.
    """

    def __init__(
        self,
        backend: BatchBackend | ModelProvider,
        *,
        price_table: PriceTable | None = None,
        tracer: Any | None = None,
        discount: float = DEFAULT_BATCH_DISCOUNT,
        poll_interval_s: float = 2.0,
        timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(backend, ModelProvider):
            backend = InProcessBatchBackend(backend)
        self.backend = backend
        self.price_table = price_table or default_price_table()
        self.tracer = tracer
        self.discount = discount
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self._clock = clock

    @staticmethod
    def _as_requests(
        requests: list[BatchRequest] | list[ModelRequest],
    ) -> list[BatchRequest]:
        out: list[BatchRequest] = []
        for i, item in enumerate(requests):
            if isinstance(item, BatchRequest):
                out.append(item)
            else:
                out.append(BatchRequest(custom_id=f"req-{i}", request=item))
        return out

    async def submit(
        self, requests: list[BatchRequest] | list[ModelRequest]
    ) -> tuple[BatchJob, dict[str, ModelRequest]]:
        items = self._as_requests(requests)
        if len({i.custom_id for i in items}) != len(items):
            raise BatchError("batch custom_ids must be unique", provider="batch")
        job = await self.backend.submit(items)
        return job, {i.custom_id: i.request for i in items}

    async def await_job(self, job: BatchJob, *, timeout_s: float | None = None) -> BatchJob:
        deadline = self._clock() + (timeout_s or self.timeout_s) if (timeout_s or self.timeout_s) else None
        while not job.done:
            if deadline is not None and self._clock() >= deadline:
                raise BatchError(
                    f"batch {job.id!r} did not finish within timeout", provider=self.backend.name
                )
            await asyncio.sleep(self.poll_interval_s)
            job = await self.backend.poll(job)
        return job

    def _reconcile(
        self, results: list[BatchResult], requests: dict[str, ModelRequest]
    ) -> list[BatchResult]:
        by_id = {r.custom_id: r for r in results}
        reconciled: list[BatchResult] = []
        for custom_id in requests:
            reconciled.append(
                by_id.get(custom_id)
                or BatchResult(custom_id=custom_id, error="missing from batch output")
            )
        # Surface any results the provider returned for unknown ids too.
        for result in results:
            if result.custom_id not in requests:
                reconciled.append(result)
        return reconciled

    async def run(
        self,
        requests: list[BatchRequest] | list[ModelRequest],
        *,
        timeout_s: float | None = None,
    ) -> BatchRunResult:
        span_cm = (
            self.tracer.span("batch", type="model_call")
            if self.tracer is not None
            else _NullSpan()
        )
        with span_cm as span:
            job, request_map = await self.submit(requests)
            job = await self.await_job(job, timeout_s=timeout_s)
            raw = await self.backend.results(job, request_map)
            results = self._reconcile(raw, request_map)
            cost = 0.0
            for result in results:
                if result.response is not None:
                    discounted = (
                        self.price_table.cost(result.response.model, result.response.usage)
                        * self.discount
                    )
                    result.response.cost_usd = discounted
                    cost += discounted
            if span is not None:
                span.set(
                    backend=self.backend.name,
                    requests=len(request_map),
                    succeeded=sum(1 for r in results if r.ok),
                    failed=sum(1 for r in results if not r.ok),
                    cost_usd=round(cost, 8),
                    discount=self.discount,
                )
            return BatchRunResult(job=job, results=results, cost_usd=cost)

    async def aclose(self) -> None:
        await self.backend.aclose()


class _NullSpan:
    """Context manager that yields ``None`` when no tracer is configured."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False
