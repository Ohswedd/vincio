"""On-device fine-tuning & continual local adaptation.

The distillation flywheel already turns production traces into executed *hosted*
fine-tune jobs (:mod:`vincio.optimize.distill`), and the in-process GGUF provider
already runs a quantized model air-gapped (:class:`~vincio.providers.local.GGUFProvider`).
The rung this module adds is **local adaptation**: a LoRA-class adapter fit
*on-device* from the flywheel's promoted, grounded dataset and applied to the
in-process model **without the run ever leaving the process** — so an air-gapped
or edge deployment improves on its own traffic with no hosted training round-trip.

Three pieces, all offline-first, deterministic, and gated by the platform's
existing promotion discipline:

* :class:`LocalLoRATrainer` fits a :class:`LocalAdapter` — a parameter-efficient,
  *low-rank* correction over the base model — from a grounded
  :class:`~vincio.optimize.distill.TrainingSet`. The dependency-free default fits
  the adapter in pure Python (a deterministic rank-``r`` example memory); an
  optional :class:`NativeLoRABackend` delegates to a real GGUF/LoRA trainer when
  one is installed, recording the produced ``lora_path`` on the artifact.
* :class:`AdaptedProvider` wraps **any**
  :class:`~vincio.providers.base.ModelProvider` (the GGUF model, the mock, a
  hosted endpoint) and applies a :class:`LocalAdapter` at generation time: a
  request that lands in the adapter's learned, in-distribution region is answered
  the grounded way it was taught; everything else falls through to the base model
  unchanged, so the adapter is **bounded** — it never degrades off-distribution
  traffic.
* :class:`AdapterRegistry` versions adapters and rolls them back, and
  :class:`AdapterGate` is the no-regression gate — the on-device analogue of the
  model-swap gate — that promotes a new adapter version **only** when the adapted
  model is at-least-as-good as its base on a held-out eval set.

:class:`ContinualAdaptation` wires these into the streaming continual loop the
``local_adaptation`` policy drives: gather the promoted dataset, fit a new adapter
version, gate it against the base, and promote or roll back — every version
auditable and reversible, under the same safety gates that gate a hosted
fine-tune job.

This is a library capability inside your process — never a hosted training
service. The model, the adapter, and the training all stay on the device.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..core.errors import OptimizationError
from ..core.tokens import count_tokens
from ..core.types import ModelCapabilities, ModelEvent, ModelRequest, ModelResponse, TokenUsage
from ..providers.base import ModelProvider, run_sync
from .self_improvement import CanarySpec, CanaryVerdict, _canary_verdict

if TYPE_CHECKING:
    from ..core.app import ContextApp
    from ..evals.datasets import Dataset
    from ..evals.reports import EvalReport
    from ..retrieval.embeddings import Embedder
    from .distill import TrainingSet

__all__ = [
    "AdaptationError",
    "LocalAdapter",
    "AdapterHit",
    "NativeLoRABackend",
    "LocalLoRATrainer",
    "AdaptedProvider",
    "AdapterRegistry",
    "AdapterGate",
    "LocalAdaptationPolicy",
    "AdaptationEvent",
    "AdaptationResult",
    "ContinualAdaptation",
]


class AdaptationError(OptimizationError):
    """An on-device adaptation could not proceed (too little data, dim mismatch).

    Inherits :class:`~vincio.core.errors.OptimizationError`'s stable ``.code`` so
    it carries the same remediation surface as every other optimization failure.
    """


# ---------------------------------------------------------------------------
# Low-rank linear algebra (pure-Python, deterministic)
# ---------------------------------------------------------------------------


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm <= 1e-12:
        return [0.0] * len(vector)
    return [x / norm for x in vector]


def _orthonormal_basis(vectors: list[list[float]], rank: int) -> list[list[float]]:
    """A deterministic rank-``r`` orthonormal basis for the example subspace.

    Modified Gram-Schmidt over the example embeddings in order, keeping up to
    ``rank`` independent directions and discarding any whose residual collapses
    to (near) zero. This is the low-rank structure that makes the adapter
    parameter-efficient: the correction is stored as an ``r × d`` basis plus an
    ``n × r`` code matrix instead of the full ``n × d`` example set.
    """
    basis: list[list[float]] = []
    for vector in vectors:
        if len(basis) >= rank:
            break
        residual = list(vector)
        for axis in basis:
            projection = _dot(residual, axis)
            residual = [r - projection * a for r, a in zip(residual, axis, strict=True)]
        norm = math.sqrt(sum(x * x for x in residual))
        if norm > 1e-9:
            basis.append([x / norm for x in residual])
    return basis


def _project(vector: list[float], basis: list[list[float]]) -> list[float]:
    """Coordinates of ``vector`` in the low-rank basis (the adapter's code space)."""
    return [_dot(vector, axis) for axis in basis]


# ---------------------------------------------------------------------------
# The adapter artifact
# ---------------------------------------------------------------------------


class AdapterHit(BaseModel):
    """One adapter activation: the learned answer and how strongly it matched."""

    target: str
    similarity: float
    scale: float
    index: int
    support: float = 1.0


class LocalAdapter(BaseModel):
    """A versioned, content-addressed, portable LoRA-class adapter.

    The on-device analogue of a ``.safetensors`` LoRA file: a small,
    parameter-efficient correction over a base model that travels with the
    deployment and is hot-swapped at generation time. It is *low-rank* by
    construction — an ``r × d`` orthonormal ``basis`` plus an ``n × r`` matrix of
    normalized ``codes`` and their grounded ``targets`` — and *bounded*: it only
    activates when a request projects into its learned region above ``gate``,
    deferring to the base model otherwise.

    Two strength controls mirror real PEFT: ``gate`` is the acceptance threshold
    (how close a request must be to learned support to be answered the adapted
    way) and ``scale`` is the adapter alpha (``0.0`` neutralizes the adapter
    without unloading it — a one-line reversibility knob, and the value passed
    through to a native GGUF backend's LoRA scale when one is used).
    """

    name: str = "local-adapter"
    version: int = 0
    base_model: str = ""
    rank: int = 0
    embed_dim: int = 0
    gate: float = 0.85
    scale: float = 1.0
    basis: list[list[float]] = Field(default_factory=list)
    codes: list[list[float]] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    supports: list[float] = Field(default_factory=list)
    n_examples: int = 0
    training_set_hash: str = ""
    # Set when an optional native backend produced a real GGUF/LoRA file on-device.
    lora_path: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def digest(self) -> str:
        """Content address over the learned parameters — the LoRA file hash analog.

        Two adapters fit from the same data with the same hyper-parameters have
        the same digest; a refit that changes a single learned weight does not.
        Excludes version/provenance/metadata so the address tracks *behaviour*.
        """
        payload = json.dumps(
            {
                "base_model": self.base_model,
                "rank": self.rank,
                "embed_dim": self.embed_dim,
                "gate": self.gate,
                "scale": self.scale,
                "basis": self.basis,
                "codes": self.codes,
                "targets": self.targets,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def size_bytes(self) -> int:
        """Resident footprint of the low-rank parameters, in bytes.

        ``8 bytes`` per float across the basis and code matrices plus the target
        text — the adapter's parameter budget, the analogue of a LoRA's rank-set
        size and what keeps it cheap to ship and hold in memory.
        """
        floats = sum(len(row) for row in self.basis) + sum(len(row) for row in self.codes)
        text = sum(len(t.encode("utf-8")) for t in self.targets)
        return floats * 8 + text

    def apply(self, query_vector: list[float]) -> AdapterHit | None:
        """The adapter's forward pass: shape a request, or stay inert.

        Projects the (unit-norm) request embedding into the low-rank subspace and
        scores it against each learned example by the in-subspace inner product
        ``qᵀ·P·e`` — which equals the true cosine for a request that lives in the
        learned subspace and stays small for one that does not. Returns an
        :class:`AdapterHit` only when the best match clears ``gate`` and ``scale``
        is non-zero; an off-distribution request returns ``None`` and the base
        model answers unchanged. This is what bounds the adapter to the
        distribution it was trained on — it never reshapes traffic it has not
        seen.
        """
        if self.scale <= 0.0 or not self.codes or not self.basis:
            return None
        if len(query_vector) != self.embed_dim:
            raise AdaptationError(
                f"query embedding dim {len(query_vector)} != adapter dim {self.embed_dim}; "
                "apply the adapter with the same embedder it was fit with"
            )
        code = _project(_unit(query_vector), self.basis)
        if not any(code):
            return None
        best_index = -1
        best_similarity = -1.0
        for index, learned in enumerate(self.codes):
            similarity = _dot(code, learned)
            if similarity > best_similarity:
                best_similarity, best_index = similarity, index
        if best_index < 0 or best_similarity < self.gate:
            return None
        support = self.supports[best_index] if best_index < len(self.supports) else 1.0
        return AdapterHit(
            target=self.targets[best_index],
            similarity=round(best_similarity, 6),
            scale=self.scale,
            index=best_index,
            support=support,
        )

    def save(self, path: str | Path) -> Path:
        """Write the adapter to a portable JSON artifact (the LoRA-file analog)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> LocalAdapter:
        """Load an adapter from a JSON artifact written by :meth:`save`."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Training (on-device)
# ---------------------------------------------------------------------------


@runtime_checkable
class NativeLoRABackend(Protocol):
    """Optional native on-device LoRA trainer (e.g. a llama.cpp/GGUF finetune).

    The dependency-free :class:`LocalLoRATrainer` is complete on its own; inject a
    backend to *also* produce a real quantized LoRA file on-device. ``train``
    receives the grounded JSONL and the base model and returns the path to the
    produced ``.gguf``/LoRA, which is recorded on the
    :class:`LocalAdapter` (``lora_path``) for the GGUF provider to load. The
    training still never leaves the process.
    """

    name: str

    async def train(self, training_jsonl: str, base_model: str) -> str: ...


def _message_text(messages: list[dict[str, str]], role: str) -> str:
    """The text of the (last) ``role`` turn in a neutral chat transcript."""
    return next((m.get("content", "") for m in reversed(messages) if m.get("role") == role), "")


class LocalLoRATrainer:
    """Fit a :class:`LocalAdapter` on-device from a grounded training set.

    The dependency-free default embeds each example's user turn, builds a
    deterministic rank-``r`` orthonormal basis over those embeddings, and stores
    the projected codes alongside the grounded assistant turns — a parameter-
    efficient correction the model applies to in-distribution traffic. No network,
    no SDK, fully deterministic given a deterministic embedder, so the whole
    flywheel-to-adapter path is testable offline.

    Inject a :class:`NativeLoRABackend` to additionally produce a real GGUF/LoRA
    file on-device; its path is recorded on the adapter for
    :class:`~vincio.providers.local.GGUFProvider` to load.
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        rank: int = 8,
        gate: float = 0.85,
        scale: float = 1.0,
        backend: NativeLoRABackend | None = None,
    ) -> None:
        self.embedder = embedder
        self.rank = rank
        self.gate = gate
        self.scale = scale
        self.backend = backend

    async def fit(
        self,
        training_set: TrainingSet,
        base_model: str,
        *,
        name: str = "local-adapter",
        min_examples: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> LocalAdapter:
        """Fit a :class:`LocalAdapter` from the grounded ``training_set``.

        Raises :class:`AdaptationError` if there are fewer than ``min_examples``
        examples — refusing to fit an adapter on too little signal rather than
        ship a brittle one.
        """
        examples = list(training_set.examples)
        if len(examples) < min_examples:
            raise AdaptationError(
                f"local adaptation needs at least {min_examples} grounded examples; "
                f"got {len(examples)}"
            )

        embedder = self.embedder
        if embedder is None:
            from ..retrieval.embeddings import LocalHashEmbedder

            embedder = LocalHashEmbedder()

        from ..retrieval.embeddings import embed_texts

        prompts = [_message_text(e.messages, "user") for e in examples]
        targets = [_message_text(e.messages, "assistant") for e in examples]
        supports = [float(e.support) for e in examples]
        vectors = await embed_texts(embedder, prompts)
        embed_dim = len(vectors[0]) if vectors else int(getattr(embedder, "dim", 0))
        basis = _orthonormal_basis(vectors, self.rank)
        codes = [_project(_unit(v), basis) for v in vectors]

        adapter = LocalAdapter(
            name=name,
            base_model=base_model,
            rank=len(basis),
            embed_dim=embed_dim,
            gate=self.gate,
            scale=self.scale,
            basis=basis,
            codes=codes,
            targets=targets,
            supports=supports,
            n_examples=len(examples),
            training_set_hash=hashlib.sha256(
                training_set.to_jsonl().encode("utf-8")
            ).hexdigest(),
            provenance={
                "source": training_set.metadata.get("source", ""),
                "grounded_fraction": training_set.grounded_fraction,
                "embedder": type(embedder).__name__,
            },
            metadata=metadata or {},
        )
        if self.backend is not None:
            adapter.lora_path = await self.backend.train(training_set.to_jsonl(), base_model)
            adapter.provenance["native_backend"] = self.backend.name
        return adapter


# ---------------------------------------------------------------------------
# Applying the adapter (the provider wrapper)
# ---------------------------------------------------------------------------


class AdaptedProvider(ModelProvider):
    """Apply a :class:`LocalAdapter` to any base provider at generation time.

    Wraps a :class:`~vincio.providers.base.ModelProvider` (the in-process GGUF
    model, the deterministic mock, or a hosted endpoint) so a request that lands
    in the adapter's learned region is answered the grounded way it was taught,
    and everything else falls through to the base model unchanged. The wrapper is
    transparent: it reports the base provider's ``name`` and ``capabilities`` so
    residency, provenance marking, and provider lookups are unaffected — the
    adapter is part of the *model*, not a new provider.

    The same embedder the adapter was fit with must project requests at
    inference, so the code space lines up; pass it explicitly or let the wrapper
    build a matching :class:`~vincio.retrieval.embeddings.LocalHashEmbedder`.
    """

    def __init__(
        self,
        base: ModelProvider,
        adapter: LocalAdapter,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self.base = base
        self.adapter = adapter
        self.name = base.name
        self.requires_api_key = getattr(base, "requires_api_key", False)
        if embedder is None:
            from ..retrieval.embeddings import LocalHashEmbedder

            embedder = LocalHashEmbedder(dim=adapter.embed_dim or 256)
        self.embedder = embedder

    async def _match(self, request: ModelRequest) -> AdapterHit | None:
        from ..retrieval.embeddings import embed_texts

        query = next(
            (m.text for m in reversed(request.messages) if m.role == "user"), ""
        )
        if not query:
            return None
        vectors = await embed_texts(self.embedder, [query])
        if not vectors or len(vectors[0]) != self.adapter.embed_dim:
            return None
        return self.adapter.apply(vectors[0])

    def _response_from_hit(self, request: ModelRequest, hit: AdapterHit) -> ModelResponse:
        query = next((m.text for m in reversed(request.messages) if m.role == "user"), "")
        return ModelResponse(
            model=request.model or self.adapter.base_model,
            text=hit.target,
            finish_reason="stop",
            usage=TokenUsage(
                input_tokens=count_tokens(query),
                output_tokens=count_tokens(hit.target),
            ),
            provider=self.name,
            raw={
                "adapter": {
                    "name": self.adapter.name,
                    "version": self.adapter.version,
                    "similarity": hit.similarity,
                    "scale": hit.scale,
                    "matched_index": hit.index,
                }
            },
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        hit = await self._match(request)
        if hit is not None:
            return self._response_from_hit(request, hit)
        return await self.base.generate(request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        hit = await self._match(request)
        if hit is None:
            async for event in self.base.stream(request):
                yield event
            return
        response = self._response_from_hit(request, hit)
        chunk_size = 16
        for start in range(0, len(response.text), chunk_size):
            yield ModelEvent(type="text_delta", text=response.text[start : start + chunk_size])
        yield ModelEvent(type="usage", usage=response.usage)
        yield ModelEvent(type="done", response=response)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return await self.base.embed(texts, model=model)

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.base.capabilities(model)

    async def list_models(self) -> list[Any]:
        return await self.base.list_models()

    async def aclose(self) -> None:
        await self.base.aclose()


# ---------------------------------------------------------------------------
# The adapter registry (versioned & reversible)
# ---------------------------------------------------------------------------


class AdapterRegistry:
    """A versioned, reversible store of on-device adapters.

    Each :meth:`register` assigns the next version under the adapter's name and
    makes it the active head; :meth:`rollback` restores an earlier version as the
    head — the on-device analogue of a prompt-registry rollback, so a regressing
    adapter is reverted without losing its history. With ``directory`` set, every
    version is also persisted as a portable JSON artifact (``name/vN.json``) and
    the head pointer (``name/HEAD``) so the registry survives a restart.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory else None
        self._versions: dict[str, list[LocalAdapter]] = {}
        self._active: dict[str, int] = {}
        if self.directory is not None and self.directory.is_dir():
            self._load()

    def _load(self) -> None:
        assert self.directory is not None  # noqa: S101 - _load runs only when a directory is configured
        for name_dir in sorted(p for p in self.directory.iterdir() if p.is_dir()):
            versions = sorted(
                name_dir.glob("v*.json"), key=lambda p: int(p.stem[1:])
            )
            loaded = [LocalAdapter.load(p) for p in versions]
            if not loaded:
                continue
            self._versions[name_dir.name] = loaded
            head = name_dir / "HEAD"
            if head.is_file():
                self._active[name_dir.name] = int(head.read_text(encoding="utf-8").strip())
            else:
                self._active[name_dir.name] = loaded[-1].version

    def _persist(self, adapter: LocalAdapter) -> None:
        if self.directory is None:
            return
        name_dir = self.directory / adapter.name
        adapter.save(name_dir / f"v{adapter.version}.json")
        (name_dir / "HEAD").write_text(str(self._active[adapter.name]), encoding="utf-8")

    def register(self, adapter: LocalAdapter) -> LocalAdapter:
        """Register ``adapter`` as the next version and make it the active head.

        Stores an independent copy stamped with the assigned version, so the
        caller's object is never mutated and re-registering the same adapter is
        safe.
        """
        versions = self._versions.setdefault(adapter.name, [])
        stored = adapter.model_copy(deep=True)
        stored.version = len(versions) + 1
        versions.append(stored)
        self._active[stored.name] = stored.version
        self._persist(stored)
        return stored

    def versions(self, name: str) -> list[LocalAdapter]:
        """All registered versions of ``name``, oldest first."""
        return list(self._versions.get(name, []))

    def get(self, name: str, version: int) -> LocalAdapter:
        """The adapter registered as ``name`` version ``version``."""
        for adapter in self._versions.get(name, []):
            if adapter.version == version:
                return adapter
        raise AdaptationError(f"no adapter {name!r} version {version}")

    def active(self, name: str) -> LocalAdapter | None:
        """The current head adapter for ``name`` (or ``None`` if none active)."""
        version = self._active.get(name)
        if version is None:
            return None
        return self.get(name, version)

    def latest(self, name: str) -> LocalAdapter | None:
        """The most recently registered version of ``name`` (or ``None``)."""
        versions = self._versions.get(name)
        return versions[-1] if versions else None

    def set_active(self, name: str, version: int) -> LocalAdapter:
        """Make ``version`` the active head for ``name``."""
        adapter = self.get(name, version)
        self._active[name] = version
        self._persist(adapter)
        return adapter

    def rollback(self, name: str, to_version: int) -> LocalAdapter:
        """Restore ``to_version`` as the active head — revert a regressing adapter."""
        return self.set_active(name, to_version)

    def prune(self, name: str, keep: int) -> int:
        """Retain only the most recent ``keep`` in-memory versions of ``name``.

        Bounds the registry's resident footprint; persisted artifacts on disk are
        the durable archive and are left untouched. Returns the number dropped.
        """
        if keep <= 0:
            return 0
        versions = self._versions.get(name, [])
        if len(versions) <= keep:
            return 0
        dropped = len(versions) - keep
        self._versions[name] = versions[-keep:]
        return dropped


# ---------------------------------------------------------------------------
# The no-regression gate (the on-device swap gate)
# ---------------------------------------------------------------------------


class AdapterGate:
    """No-regression gate for an on-device adapter, the model-swap gate's analog.

    Promotes an adapter only when the adapted model is **at-least-as-good** as its
    base on a held-out set: the watched ``metric`` must not regress beyond
    ``regression_threshold`` (default ``0.0`` — a tie passes, a drop does not) and
    no statistically significant regression is detected. Reuses the very same
    verdict machinery a prompt deploy and a model rotation clear
    (:func:`~vincio.optimize.self_improvement._canary_verdict`), so the gate is
    one shared discipline across the platform, not a new one.
    """

    def __init__(
        self,
        *,
        metric: str = "lexical_overlap",
        regression_threshold: float = 0.0,
        require_significance: bool = True,
        min_samples: int = 4,
        alpha: float = 0.05,
    ) -> None:
        self.spec = CanarySpec(
            metric=metric,
            regression_threshold=regression_threshold,
            require_significance=require_significance,
            min_samples=min_samples,
            alpha=alpha,
        )

    def evaluate(
        self, base_report: EvalReport, adapted_report: EvalReport
    ) -> CanaryVerdict:
        """Decide the gate from a base and an adapted eval report."""
        return _canary_verdict(base_report, adapted_report, self.spec)


# ---------------------------------------------------------------------------
# The continual adaptation policy & loop
# ---------------------------------------------------------------------------


class LocalAdaptationPolicy(BaseModel):
    """The opt-in contract for continual on-device adaptation.

    One declarative spec for *how* the local adapter is fit and gated: the
    low-rank ``rank`` and acceptance ``gate``/``scale`` of the adapter, the
    grounding bar the training set must clear, and the no-regression gate's
    ``metric``/``regression_threshold`` (default ``0.0`` enforces the
    at-least-as-good SLO). ``dry_run`` fits and gates without promoting;
    ``keep_versions`` bounds how many adapter versions the registry retains.
    """

    rank: int = 8
    gate: float = 0.85
    scale: float = 1.0
    min_examples: int = 4
    name: str = "local-adapter"
    # Training-set curation (when fitting from runs/traces).
    require_grounding: bool = True
    min_support: float = 0.5
    max_examples: int | None = None
    # No-regression gate.
    metric: str = "lexical_overlap"
    regression_threshold: float = 0.0
    require_significance: bool = True
    min_samples: int = 4
    alpha: float = 0.05
    gates: dict[str, str] | None = None
    # Eval & lifecycle.
    concurrency: int = 4
    keep_versions: int = 10
    dry_run: bool = False


AdaptationPhase = str


class AdaptationEvent(BaseModel):
    """One event in a continual-adaptation cycle — stamped on the audit chain."""

    phase: AdaptationPhase = "observe"
    action: str = ""
    reason: str = ""
    adapter_version: int | None = None
    adapter_digest: str | None = None
    verdict: CanaryVerdict | None = None
    rolled_back_to: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AdaptationResult(BaseModel):
    """The outcome of one gated on-device adaptation cycle."""

    promoted: bool = False
    base_model: str = ""
    adapter_name: str = ""
    adapter_version: int | None = None
    adapter_digest: str | None = None
    training_examples: int = 0
    verdict: CanaryVerdict | None = None
    rolled_back_to: int | None = None
    reason: str = ""


class ContinualAdaptation:
    """Drive continual on-device adaptation as a streaming, gated loop.

    The local analogue of :class:`~vincio.optimize.self_improvement.SelfImprovementController`:
    gather the flywheel's promoted, grounded dataset, fit a new
    :class:`LocalAdapter` version on-device, gate it against the current base on a
    held-out set, and **promote or roll back** — every version versioned in the
    :class:`AdapterRegistry`, every decision on the shared audit chain and event
    bus, and promotion held by the same no-regression discipline a hosted
    fine-tune job clears. Nothing leaves the process.

    Bind a held-out ``dataset`` to enable gating (without one the loop fits but
    refuses to promote, since it cannot prove no regression). Feed the training
    corpus as ``runs`` (the natural output of :meth:`~vincio.core.app.ContextApp.run`),
    a prebuilt ``training_set``, or let it curate the app's captured traces.
    """

    def __init__(
        self,
        app: ContextApp,
        policy: LocalAdaptationPolicy | None = None,
        *,
        dataset: Dataset | None = None,
        registry: AdapterRegistry | None = None,
        embedder: Embedder | None = None,
        base_model: str | None = None,
        trainer: LocalLoRATrainer | None = None,
    ) -> None:
        self.app = app
        self.policy = policy or LocalAdaptationPolicy()
        self.dataset = dataset
        self.registry = registry or AdapterRegistry()
        self.embedder = embedder or app.embedder
        self.base_model = base_model or app.model
        self.trainer = trainer or LocalLoRATrainer(
            embedder=self.embedder,
            rank=self.policy.rank,
            gate=self.policy.gate,
            scale=self.policy.scale,
        )
        self.events: list[AdaptationEvent] = []
        self.result = AdaptationResult(base_model=self.base_model, adapter_name=self.policy.name)

    # -- training-set assembly ----------------------------------------------

    def _build_training_set(
        self, runs: list[Any] | None, training_set: TrainingSet | None
    ) -> TrainingSet:
        if training_set is not None:
            return training_set
        from .distill import export_training_set, export_training_set_from_runs

        system = self.app.prompt_spec.role or self.app.prompt_spec.objective
        if runs is not None:
            return export_training_set_from_runs(
                runs,
                name=self.policy.name,
                system=system,
                require_grounding=self.policy.require_grounding,
                min_support=self.policy.min_support,
                max_examples=self.policy.max_examples,
            )
        traces: list[Any] = []
        exporter = self.app.tracer.exporter
        if hasattr(exporter, "load_all"):
            traces = exporter.load_all(limit=500)
        elif hasattr(exporter, "traces"):
            traces = list(exporter.traces)[-500:]
        return export_training_set(
            traces,
            name=self.policy.name,
            system=system,
            require_grounding=self.policy.require_grounding,
            min_support=self.policy.min_support,
            max_examples=self.policy.max_examples,
        )

    # -- evaluation (provider swap, memory-write-back disabled) -------------

    @staticmethod
    def _unwrap(provider: ModelProvider) -> ModelProvider:
        return provider.base if isinstance(provider, AdaptedProvider) else provider

    async def _eval_report(self, provider: ModelProvider) -> EvalReport:
        from ..evals.runners import EvalRunner

        assert self.dataset is not None  # noqa: S101 - the caller skips evaluation when no dataset is bound
        app = self.app
        original_provider = app._provider_instance
        original_write_back = app.config.memory.write_back
        app._provider_instance = provider
        app.config.memory.write_back = []
        try:
            metrics = [self.policy.metric]
            if self.policy.gates:
                metrics += [m for m in self.policy.gates if m not in metrics]
            runner = EvalRunner(app, metrics=metrics, concurrency=self.policy.concurrency)
            return await runner.arun(self.dataset, name=f"local_adaptation:{self.policy.name}")
        finally:
            app._provider_instance = original_provider
            app.config.memory.write_back = original_write_back

    # -- the streaming cycle ------------------------------------------------

    async def astream(
        self,
        *,
        runs: list[Any] | None = None,
        training_set: TrainingSet | None = None,
        apply: bool = True,
    ) -> AsyncIterator[AdaptationEvent]:
        """Run one gated adaptation cycle, yielding each phase as it lands.

        Sequence: ``observe → train → gate → promote / rollback``. On a pass the
        new adapter is registered, made the active head, and (with ``apply``)
        installed on the app via
        :meth:`~vincio.core.app.ContextApp.use_local_adapter`; on a fail it is
        refused and the registry head stays on the last known-good version.
        """
        policy = self.policy
        base = self._unwrap(self.app._base_provider())
        yield self._emit(AdaptationEvent(phase="observe", reason="cycle started"))

        corpus = self._build_training_set(runs, training_set)
        self.result.training_examples = len(corpus)
        if len(corpus) < policy.min_examples:
            self.result.reason = (
                f"only {len(corpus)} grounded examples (< {policy.min_examples}); "
                "refusing to fit an adapter"
            )
            yield self._emit(
                AdaptationEvent(phase="exhausted", action="skipped", reason=self.result.reason)
            )
            return

        adapter = await self.trainer.fit(
            corpus, self.base_model, name=policy.name, min_examples=policy.min_examples
        )
        self.result.adapter_digest = adapter.digest
        yield self._emit(
            AdaptationEvent(
                phase="train",
                action="fit",
                adapter_digest=adapter.digest,
                reason=f"fit rank-{adapter.rank} adapter from {len(corpus)} examples",
                details={"rank": adapter.rank, "size_bytes": adapter.size_bytes},
            )
        )

        if self.dataset is None:
            self.result.reason = "no held-out dataset bound; cannot gate (not promoting)"
            yield self._emit(
                AdaptationEvent(phase="gate", action="skipped", reason=self.result.reason)
            )
            return

        adapted = AdaptedProvider(base, adapter, embedder=self.embedder)
        base_report = await self._eval_report(base)
        adapted_report = await self._eval_report(adapted)
        gate = AdapterGate(
            metric=policy.metric,
            regression_threshold=policy.regression_threshold,
            require_significance=policy.require_significance,
            min_samples=policy.min_samples,
            alpha=policy.alpha,
        )
        verdict = gate.evaluate(base_report, adapted_report)
        self.result.verdict = verdict

        # Safety/schema overlay: a failing gate blocks promotion regardless of metric.
        if verdict.passed and policy.gates:
            from ..evals.reports import evaluate_gates

            outcomes = evaluate_gates(adapted_report, policy.gates)
            failed = [k for k, v in outcomes.items() if not v["passed"]]
            if failed:
                verdict.passed = False
                verdict.reason = f"adapter safety gates failed: {failed}"

        yield self._emit(
            AdaptationEvent(
                phase="gate",
                action="passed" if verdict.passed else "failed",
                verdict=verdict,
                reason=verdict.reason,
            )
        )

        if not verdict.passed:
            current = self.registry.active(policy.name)
            self.result.promoted = False
            self.result.rolled_back_to = current.version if current else None
            self.result.reason = (
                f"adapter not promoted: {verdict.reason}"
                + (f"; kept v{current.version}" if current else "")
            )
            yield self._emit(
                AdaptationEvent(
                    phase="rollback",
                    action="refused",
                    verdict=verdict,
                    rolled_back_to=self.result.rolled_back_to,
                    reason=self.result.reason,
                )
            )
            return

        if policy.dry_run:
            self.result.reason = f"dry run: adapter would be promoted ({verdict.reason})"
            yield self._emit(
                AdaptationEvent(phase="promote", action="dry_run", verdict=verdict,
                                reason=self.result.reason)
            )
            return

        stored = self.registry.register(adapter)
        self.registry.prune(policy.name, policy.keep_versions)
        if apply:
            self.app.use_local_adapter(stored)
        self.result.promoted = True
        self.result.adapter_version = stored.version
        self.result.reason = f"promoted adapter v{stored.version}: {verdict.reason}"
        yield self._emit(
            AdaptationEvent(
                phase="promote",
                action="promoted",
                adapter_version=stored.version,
                adapter_digest=stored.digest,
                verdict=verdict,
                reason=self.result.reason,
            )
        )

    async def aadapt(
        self,
        *,
        runs: list[Any] | None = None,
        training_set: TrainingSet | None = None,
        apply: bool = True,
    ) -> AdaptationResult:
        """Run one cycle to completion and return its :class:`AdaptationResult`."""
        async for _ in self.astream(runs=runs, training_set=training_set, apply=apply):
            pass
        return self.result

    def adapt(
        self,
        *,
        runs: list[Any] | None = None,
        training_set: TrainingSet | None = None,
        apply: bool = True,
    ) -> AdaptationResult:
        """Sync wrapper over :meth:`aadapt`."""
        return run_sync(self.aadapt(runs=runs, training_set=training_set, apply=apply))

    # -- internals ----------------------------------------------------------

    def _emit(self, event: AdaptationEvent) -> AdaptationEvent:
        self.events.append(event)
        self.app.audit.record(
            "local_adaptation",
            decision="allow" if event.action not in ("skipped", "refused") else "deny",
            resource=self.policy.name,
            details={
                "phase": event.phase,
                "action": event.action,
                "reason": event.reason,
                "adapter_version": event.adapter_version,
                "adapter_digest": event.adapter_digest,
                "rolled_back_to": event.rolled_back_to,
            },
        )
        self.app.events.emit(f"local_adaptation.{event.phase}", event.model_dump())
        return event
