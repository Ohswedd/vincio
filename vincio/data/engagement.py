"""The data-engagement lifecycle facade — the analytics plane as one system.

Seven rungs (4.1–4.7) delivered the data plane's *primitives* — first-class
tabular evidence and the compact encoder, bounded-memory profiling, representative
sampling, fit-in-window, and the data-quality rails, governed text-to-query with
cell-level provenance, the multi-step analysis agent, content- and data-bound
charts, streaming out-of-core processing, and the semantic layer's governed
metrics — each grounded, cited, and offline-verifiable on its own. What was
missing is not an eighth primitive but the **whole**: nothing yet presented the
plane as one coherent system, threaded a real analysis from raw table to cited
deliverable behind one governed call-path, or proved that whole composition
verifies offline. This module is that capstone — the analytics analogue of the
cross-org :class:`~vincio.settlement.CrossOrgEngagement`.

A :class:`DataEngagement` (built by
:meth:`~vincio.core.app.ContextApp.data_engagement`) threads the pipeline behind
one governed, audited call-path — register → profile → sample → fit → screen →
query → analyze → chart → (governed) metric → cite. It is **purely
compositional**: every lifecycle method delegates to the *same*
:class:`~vincio.core.app.ContextApp` entry point a caller would use directly (each
unchanged and still usable on its own), captures the artifact it produced, and
records it as a stage in a single hash-linked narrative.

The :class:`DataNarrative` is that narrative: an ordered chain of
:class:`DataStage`\\ s, each binding the stage's verb, the artifact's own published
commitment (a ``result_hash`` / ``chart_hash`` / ``layer_hash``), and a
deterministic digest of the artifact's bytes into a link that chains to the
previous one. It is content-bound and offline-verifiable the way a
:class:`~vincio.data.QueryResult` is — :meth:`DataNarrative.verify` recomputes the
whole chain from the bytes alone, so a tamper introduced anywhere (a re-ordered
stage, an edited digest, a forged signature) is caught, and re-digesting the live
artifacts against the bound digests proves a tamper to any *underlying* artifact is
caught too.

Beyond the cross-org engagement's digest-binding, a data engagement adds the
data-plane's distinguishing guarantee: every analytical artifact it captured is
**data-bound** — :meth:`DataEngagement.verify` (given the live catalog)
re-executes each query, analysis, chart, and metric against the content-hashed
source and confirms the answer and every cited cell re-derive from the bytes. So
the capstone proves the plane is a *system*, not a pile of primitives: one
continuous, signed, audited narrative from the raw table to the cited deliverable,
every finding re-derivable from the source it cites.

Everything here is dependency-free, deterministic, and offline — never a hosted
query engine, a managed warehouse, or a notebook service, only a mechanical,
verifiable composition of the primitives the data plane already ships.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import DataError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow

if TYPE_CHECKING:
    from ..security.audit import ChainSigner

__all__ = [
    "DataStage",
    "DataEngagementSignature",
    "DataEngagementVerification",
    "DataNarrative",
    "DataEngagement",
]

# The audit action a sealed data-engagement narrative is recorded under — the
# single key a plane-wide engagement roll-up reads back from the chain.
ENGAGEMENT_ACTION = "data_engagement"


def _artifact_wire(artifact: Any) -> Any:
    """A deterministic, JSON-safe projection of any captured artifact.

    Prefers the artifact's own ``to_wire`` (when a primitive publishes one), then a
    pydantic ``model_dump``, falling back to a JSON-safe coercion. A list of
    artifacts projects element-wise, so a multi-artifact stage digests faithfully.
    """
    if isinstance(artifact, (list, tuple)):
        return [_artifact_wire(item) for item in artifact]
    to_wire = getattr(artifact, "to_wire", None)
    if callable(to_wire):
        try:
            return to_wire()
        except Exception:  # pragma: no cover - defensive; fall through to dump
            pass
    dump = getattr(artifact, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:  # pragma: no cover - bytes that don't decode, etc.
            pass
    return to_jsonable(artifact)


def _artifact_digest(artifact: Any) -> str:
    """A content digest of an artifact's bytes — the integrity anchor of a stage."""
    return stable_hash(_artifact_wire(artifact), length=32)


def _artifact_kind(artifact: Any) -> str:
    """The artifact's type name, for a human-legible, audit-friendly stage label."""
    if isinstance(artifact, list):
        inner = _artifact_kind(artifact[0]) if artifact else "object"
        return f"list[{inner}]"
    return type(artifact).__name__


def _artifact_id(artifact: Any) -> str:
    """The artifact's own id, when it carries one (a list/scalar has none)."""
    if isinstance(artifact, (list, tuple)):
        return ""
    return str(getattr(artifact, "id", "") or "")


def _artifact_hash(artifact: Any) -> str:
    """The artifact's own content commitment, best-effort (a digest binds it regardless).

    Reads the published hash a data-plane artifact already exposes — a
    ``result_hash`` (a :class:`~vincio.data.QueryResult` /
    :class:`~vincio.data.AnalysisResult`), a ``chart_hash``
    (:class:`~vincio.data.Chart`), a ``layer_hash``
    (:class:`~vincio.data.MetricResult`), or a generic ``content_hash`` /
    ``head_hash`` — so the narrative binds the *same* commitment the primitive
    publishes, not only an opaque digest.
    """
    if isinstance(artifact, (list, tuple)):
        return ""
    for attr in ("content_hash", "result_hash", "chart_hash", "layer_hash", "head_hash"):
        value = getattr(artifact, attr, None)
        if value:
            return str(value)
    return ""


class DataStage(BaseModel):
    """One step of a data engagement, bound into the narrative's hash chain.

    Each stage records the lifecycle ``stage`` verb (``profile``, ``query``,
    ``analyze``, ``chart`` …), the captured artifact's ``kind`` / ``artifact_id`` /
    ``artifact_hash`` (its own published commitment), a deterministic ``digest`` of
    its bytes (the integrity anchor — a tamper to the artifact changes it), and a
    compact ``summary`` of the analytical facts. ``prev_hash`` links it to the
    preceding stage and ``entry_hash`` binds all of the above, so the stages form a
    tamper-evident chain the way a :class:`~vincio.data.AnalysisResult`'s steps do.
    """

    index: int = 0
    stage: str
    kind: str = ""
    artifact_id: str = ""
    artifact_hash: str = ""
    digest: str = ""
    summary: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=utcnow)
    prev_hash: str = ""
    entry_hash: str = ""

    def link_facts(self) -> dict[str, Any]:
        """The fields the link hash binds (deliberately excludes the timestamp)."""
        return {
            "index": self.index,
            "stage": self.stage,
            "kind": self.kind,
            "artifact_id": self.artifact_id,
            "artifact_hash": self.artifact_hash,
            "digest": self.digest,
            "summary": to_jsonable(self.summary),
            "prev_hash": self.prev_hash,
        }

    def compute_entry_hash(self) -> str:
        """Recompute this stage's chain link from its current fields."""
        return stable_hash(self.link_facts(), length=32)

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        wire: dict[str, Any] = to_jsonable(self.model_dump(mode="json"))
        return wire

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> DataStage:
        return cls.model_validate(data)


class DataEngagementSignature(BaseModel):
    """One party's signature over a data-engagement narrative's content hash."""

    party: str
    signature: str
    key_id: str = ""


class DataEngagementVerification(BaseModel):
    """The (non-raising) outcome of verifying a data engagement offline.

    ``intact`` is whether the stage chain links cleanly, ``head_ok`` whether the
    recorded head matches the chain, ``hash_ok`` whether the content hash recomputes,
    ``digests_ok`` whether the live artifacts (when supplied) still match the bound
    digests, and ``signatures_ok`` whether the required signatures verify.
    ``data_bound`` is whether every verifiable artifact re-executed against the live
    catalog and re-derived from the bytes (``None`` when no catalog was supplied to
    check). ``valid`` is the conjunction. ``broken_at`` pinpoints the first stage
    that fails to chain.
    """

    valid: bool
    intact: bool
    head_ok: bool
    hash_ok: bool
    digests_ok: bool
    signatures_ok: bool
    data_bound: bool | None = None
    signed_by: list[str] = Field(default_factory=list)
    stages: int = 0
    broken_at: int | None = None
    reason: str | None = None


class DataNarrative(BaseModel):
    """A signed, content-bound, hash-chained narrative of a whole data engagement.

    The capstone artifact: the ordered chain of :class:`DataStage`\\ s a
    :class:`DataEngagement` produced as it threaded the plane end-to-end — register
    → profile → sample → screen → query → analyze → chart → metric → cite — sealed
    into a single content hash the analyst signs. It is offline-verifiable the way a
    :class:`~vincio.data.QueryResult` is — :meth:`verify` recomputes the entire chain
    from the bytes alone, so a re-ordered or edited stage, a broken link, a tampered
    head, or a forged signature is caught; pass the live artifacts to :meth:`verify`
    and a tamper to any *underlying* artifact is caught too. One narrative, one
    continuous proof that every primitive composed.
    """

    id: str = Field(default_factory=lambda: new_id("data_engagement"))
    analyst: str
    dataset: str = ""
    question: str = ""
    stages: list[DataStage] = Field(default_factory=list)
    head_hash: str = ""
    started_at: datetime = Field(default_factory=utcnow)
    sealed_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[DataEngagementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- sealing & hashing ----------------------------------------------------

    def narrative_facts(self) -> dict[str, Any]:
        """The facts the content hash binds: the analyst, dataset, question, and chain."""
        return {
            "analyst": self.analyst,
            "dataset": self.dataset,
            "question": self.question,
            "stage_count": len(self.stages),
            "entries": [s.entry_hash for s in self.stages],
            "head_hash": self.head_hash,
        }

    def compute_hash(self) -> str:
        """Recompute the narrative's content hash from the current chain."""
        return stable_hash(self.narrative_facts(), length=32)

    def seal(self) -> DataNarrative:
        """Re-link every stage in order and stamp the head and content hash (idempotent)."""
        prev = ""
        for i, stage in enumerate(self.stages):
            stage.index = i
            stage.prev_hash = prev
            stage.entry_hash = stage.compute_entry_hash()
            prev = stage.entry_hash
        self.head_hash = prev
        self.content_hash = self.compute_hash()
        return self

    # -- signing & verification -----------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> DataNarrative:
        """Add ``party``'s signature over the content hash (sealing first).

        Re-signing for the same party replaces its prior signature, so a narrative
        cannot accumulate stale signatures for one identity.
        """
        if not self.content_hash:
            self.seal()
        sig = DataEngagementSignature(
            party=party,
            signature=signer.sign(self.content_hash),
            key_id=getattr(signer, "key_id", ""),
        )
        self.signatures = [s for s in self.signatures if s.party != party]
        self.signatures.append(sig)
        return self

    def verify(
        self,
        verifier: ChainSigner | None = None,
        *,
        require: list[str] | None = None,
        artifacts: list[Any] | None = None,
    ) -> DataEngagementVerification:
        """Verify the narrative offline: the chain links, the hash recomputes, signatures check.

        Walks the stage chain recomputing each link, confirms the head and content
        hash, and (when ``verifier`` is supplied) checks each signature — ``require``
        names the parties that must have a verified signature (defaults to the
        analyst; pass ``[]`` to check the binding alone). Pass the live ``artifacts``
        the engagement captured (aligned to the stages) to additionally re-digest each
        and confirm it still matches its bound digest, so a tamper to any underlying
        artifact is caught from the bytes alone. (Data-binding — re-executing each
        artifact against the live source — is layered on by
        :meth:`DataEngagement.verify`, which has the catalog.)
        """
        prev = ""
        intact = True
        broken_at: int | None = None
        for i, stage in enumerate(self.stages):
            if (
                stage.index != i
                or stage.prev_hash != prev
                or stage.entry_hash != stage.compute_entry_hash()
            ):
                intact = False
                broken_at = i
                break
            prev = stage.entry_hash
        head_ok = intact and self.head_hash == prev
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()

        digests_ok = True
        if artifacts is not None:
            if len(artifacts) != len(self.stages):
                digests_ok = False
                if broken_at is None:
                    broken_at = min(len(artifacts), len(self.stages))
            else:
                for i, (stage, artifact) in enumerate(zip(self.stages, artifacts, strict=True)):
                    if _artifact_digest(artifact) != stage.digest:
                        digests_ok = False
                        broken_at = i if broken_at is None else broken_at
                        break

        required = [self.analyst] if require is None else require
        verified: list[str] = []
        signatures_ok = True
        for sig in self.signatures:
            if verifier is not None:
                if verifier.verify(self.content_hash, sig.signature):
                    verified.append(sig.party)
                else:
                    signatures_ok = False
            else:
                verified.append(sig.party)
        missing = [p for p in required if p not in verified]
        if missing:
            signatures_ok = False

        valid = (
            intact
            and head_ok
            and hash_ok
            and digests_ok
            and signatures_ok
            and (verifier is not None or not required)
        )
        reason: str | None = None
        if not intact:
            reason = f"chain broken at stage {broken_at}"
        elif not head_ok:
            reason = "head hash mismatch"
        elif not hash_ok:
            reason = "content hash mismatch"
        elif not digests_ok:
            reason = f"artifact digest mismatch at stage {broken_at}"
        elif not signatures_ok:
            reason = (
                f"missing/invalid signatures for {missing}" if missing else "signature mismatch"
            )
        elif verifier is None and required:
            reason = "no verifier supplied — signatures present but not authenticated"
        return DataEngagementVerification(
            valid=valid,
            intact=intact,
            head_ok=head_ok,
            hash_ok=hash_ok,
            digests_ok=digests_ok,
            signatures_ok=signatures_ok,
            signed_by=verified,
            stages=len(self.stages),
            broken_at=broken_at,
            reason=reason,
        )

    def require_valid(
        self,
        verifier: ChainSigner,
        *,
        require: list[str] | None = None,
        artifacts: list[Any] | None = None,
    ) -> DataNarrative:
        """Verify and raise :class:`~vincio.core.errors.DataError` if not valid."""
        result = self.verify(verifier, require=require, artifacts=artifacts)
        if not result.valid:
            raise DataError(
                f"data engagement {self.id} failed verification: {result.reason}",
                details={"engagement_id": self.id, "reason": result.reason},
            )
        return self

    # -- views ----------------------------------------------------------------

    @property
    def stage_names(self) -> list[str]:
        """The lifecycle verbs threaded, in order."""
        return [s.stage for s in self.stages]

    def stage(self, name: str) -> DataStage | None:
        """The first recorded stage with the given verb, if any."""
        for stage in self.stages:
            if stage.stage == name:
                return stage
        return None

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the engagement for the audit chain."""
        details: dict[str, Any] = to_jsonable(
            {
                "engagement_id": self.id,
                "analyst": self.analyst,
                "dataset": self.dataset,
                "question": self.question,
                "stages": self.stage_names,
                "stage_count": len(self.stages),
                "head_hash": self.head_hash,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )
        return details

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        wire: dict[str, Any] = to_jsonable(self.model_dump(mode="json"))
        return wire

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> DataNarrative:
        return cls.model_validate(data)

    def print_summary(self) -> None:
        """Print a one-line-per-stage view of the data-engagement narrative."""
        print(f"Data engagement {self.id} — {self.dataset or '?'} · {self.question or 'analysis'}")
        for stage in self.stages:
            facts = ", ".join(f"{k}={v}" for k, v in stage.summary.items())
            print(f"  {stage.index:>2}. {stage.stage:<14} {stage.kind:<20} {facts}")
        print(f"  head={self.head_hash} signed_by={self.signed_by}")


class DataEngagement:
    """A purely-compositional facade threading the whole data plane in one call-path.

    Build one with :meth:`~vincio.core.app.ContextApp.data_engagement` and call the
    lifecycle verbs in the order an analysis runs — :meth:`register`,
    :meth:`profile`, :meth:`sample`, :meth:`fit`, :meth:`screen`, :meth:`query`,
    :meth:`analyze`, :meth:`chart`, :meth:`query_metric`, and :meth:`cite`. Each
    delegates to the *same* :class:`~vincio.core.app.ContextApp` method a caller would
    use directly — the primitives are unchanged and still usable on their own —
    captures the artifact it produced (exposed as an attribute, e.g.
    :attr:`profile_`, :attr:`result`, :attr:`analysis`, :attr:`chart_`), returns it,
    and records it as a stage in the engagement's hash-linked narrative.

    Call :meth:`seal` to mint the content-bound, signed :class:`DataNarrative`, and
    :meth:`verify` to prove the whole chain — every captured artifact's digest, and
    (given the catalog) every analytical answer's re-derivation from the source it
    cites — verifies offline. The facade adds no new analytical logic; it composes
    and *narrates* the primitives, so the plane reads as one system.
    """

    def __init__(
        self,
        app: Any,
        *,
        dataset: str = "",
        question: str = "",
        analyst: str | None = None,
    ) -> None:
        self.app = app
        self.analyst: str = str(analyst or getattr(app, "name", "analyst"))
        self.table = dataset
        self.question = question
        self._started_at = utcnow()
        self._stages: list[DataStage] = []
        self._artifacts: list[Any] = []
        self._binders: list[Callable[[Any], bool] | None] = []
        self.narrative: DataNarrative | None = None

        # Captured artifacts, by lifecycle stage (None until that stage runs).
        self.dataset_obj: Any = None
        self.profile_: Any = None
        self.sample_: Any = None
        self.window: Any = None
        self.quality: Any = None
        self.result: Any = None
        self.analysis: Any = None
        self.chart_: Any = None
        self.metric: Any = None
        self.report: Any = None

    # -- stage recording ------------------------------------------------------

    def _record(
        self,
        stage: str,
        artifact: Any,
        *,
        artifact_hash: str | None = None,
        binder: Callable[[Any], bool] | None = None,
        **summary: Any,
    ) -> None:
        """Append a stage for ``artifact`` and invalidate any cached narrative.

        ``binder`` is an optional ``catalog -> bool`` closure that re-executes the
        artifact against the live source — recorded so :meth:`verify` can confirm the
        finding re-derives from the bytes (the data-plane's data-binding guarantee,
        on top of the digest-binding every stage carries).
        """
        self._artifacts.append(artifact)
        self._binders.append(binder)
        self._stages.append(
            DataStage(
                index=len(self._stages),
                stage=stage,
                kind=_artifact_kind(artifact),
                artifact_id=_artifact_id(artifact),
                artifact_hash=artifact_hash
                if artifact_hash is not None
                else _artifact_hash(artifact),
                digest=_artifact_digest(artifact),
                summary=to_jsonable({k: v for k, v in summary.items() if v is not None}),
            )
        )
        self.narrative = None

    @property
    def stages(self) -> list[DataStage]:
        """The stages recorded so far, in order (a live view, copied)."""
        return [s.model_copy(deep=True) for s in self._stages]

    def _table_arg(self, table: str | None) -> str | None:
        """The table a query/analysis runs over — the explicit one, else the engagement's."""
        return table if table is not None else (self.table or None)

    # -- register / profile / sample / fit / screen ---------------------------

    def register(
        self,
        data: Any,
        *,
        name: str = "",
        source: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Register the engagement's dataset in the catalog and open the narrative.

        Delegates to :meth:`~vincio.core.app.ContextApp.register_dataset`, sets the
        engagement's :attr:`table` to the resolved name (so later stages default to
        it), captures the registered :class:`~vincio.data.Dataset` on
        :attr:`dataset_obj`, records it as the opening ``register`` stage (binding the
        source bytes the whole narrative cites), and returns the table name.
        """
        table = str(self.app.register_dataset(data, name=name, source=source, **kwargs))
        self.table = table
        dataset = self.app.data_catalog().get(table)
        self.dataset_obj = dataset
        self._record(
            "register",
            dataset,
            name=table,
            rows=getattr(dataset, "row_count", None),
            columns=getattr(dataset, "width", None),
            source=source,
        )
        return table

    def _coerce(self, data: Any) -> Any:
        """Resolve the dataset a stage operates on — the explicit one, else the registered one."""
        if data is not None:
            return data
        if self.dataset_obj is not None:
            return self.dataset_obj
        if self.table and self.table in self.app.data_catalog():
            return self.app.data_catalog().get(self.table)
        raise DataError(
            "no dataset for this stage; call engagement.register(...) first or pass data="
        )

    def profile(self, data: Any | None = None, **kwargs: Any) -> Any:
        """Profile the dataset and record it as a stage.

        Delegates to :meth:`~vincio.core.app.ContextApp.profile_dataset` (defaulting
        ``data`` to the registered dataset), stores the
        :class:`~vincio.data.DatasetProfile` on :attr:`profile_`, and returns it.
        """
        profile = self.app.profile_dataset(self._coerce(data), **kwargs)
        self.profile_ = profile
        self._record(
            "profile",
            profile,
            rows=getattr(profile, "row_count", None),
            columns=len(getattr(profile, "columns", []) or []),
        )
        return profile

    def sample(self, n: int, data: Any | None = None, **kwargs: Any) -> Any:
        """Draw a representative sample and record it as a stage.

        Delegates to :meth:`~vincio.core.app.ContextApp.sample_dataset` (defaulting
        ``data`` to the registered dataset), stores the sampled
        :class:`~vincio.data.Dataset` on :attr:`sample_`, and returns it.
        """
        sample = self.app.sample_dataset(self._coerce(data), n, **kwargs)
        self.sample_ = sample
        self._record(
            "sample",
            sample,
            rows=getattr(sample, "row_count", None),
            method=str((getattr(sample, "metadata", {}) or {}).get("sample", {}).get("method", "")),
        )
        return sample

    def fit(self, *, max_tokens: int, data: Any | None = None, **kwargs: Any) -> Any:
        """Fit the dataset into a fixed token budget and record it as a stage.

        Delegates to :meth:`~vincio.core.app.ContextApp.fit_dataset` (defaulting
        ``data`` to the registered dataset), stores the
        :class:`~vincio.data.WindowFit` on :attr:`window`, and returns it.
        """
        fit = self.app.fit_dataset(self._coerce(data), max_tokens=max_tokens, **kwargs)
        self.window = fit
        self._record(
            "fit",
            fit,
            max_tokens=max_tokens,
            tokens=getattr(fit, "token_count", None),
        )
        return fit

    def screen(self, data: Any | None = None, **kwargs: Any) -> Any:
        """Screen the dataset for schema / quality breaches and record it as a stage.

        Delegates to :meth:`~vincio.core.app.ContextApp.screen_data` (defaulting
        ``data`` to the registered dataset), stores the
        :class:`~vincio.data.DataQualityReport` on :attr:`quality`, and returns it.
        """
        report = self.app.screen_data(self._coerce(data), **kwargs)
        self.quality = report
        self._record(
            "screen",
            report,
            allowed=getattr(report, "allowed", None),
            violations=len(getattr(report, "violations", []) or []),
        )
        return report

    # -- query / analyze / chart / metric -------------------------------------

    def query(
        self, request: str, *, table: str | None = None, dataset: Any | None = None, **kwargs: Any
    ) -> Any:
        """Run a governed, read-only-verified query and record it as a stage.

        Delegates to :meth:`~vincio.core.app.ContextApp.query_data` (defaulting the
        table to the engagement's registered dataset), stores the cited
        :class:`~vincio.data.QueryResult` on :attr:`result`, and returns it. When the
        query ran over the registered catalog (no one-shot ``dataset=``), the stage
        carries a data-binder so :meth:`verify` re-executes it against the source.
        """
        result = self.app.query_data(
            request, table=self._table_arg(table), dataset=dataset, **kwargs
        )
        self.result = result
        binder = (lambda cat: bool(result.verify(cat))) if dataset is None else None
        self._record(
            "query",
            result,
            binder=binder,
            tables=",".join(getattr(result.plan, "tables", []) or []),
            rows=getattr(result, "row_count", None),
            coverage=str(getattr(result, "coverage", "")),
        )
        return result

    def analyze(
        self, objective: str, *, table: str | None = None, dataset: Any | None = None, **kwargs: Any
    ) -> Any:
        """Run the bounded multi-step analysis agent and record it as a stage.

        Delegates to :meth:`~vincio.core.app.ContextApp.analyze_data` (defaulting the
        table to the engagement's registered dataset), stores the cited
        :class:`~vincio.data.AnalysisResult` on :attr:`analysis`, and returns it. When
        run over the registered catalog, the stage carries a data-binder so
        :meth:`verify` re-executes every step against the source.
        """
        analysis = self.app.analyze_data(
            objective, table=self._table_arg(table), dataset=dataset, **kwargs
        )
        self.analysis = analysis
        binder = (lambda cat: bool(analysis.verify(cat))) if dataset is None else None
        findings = getattr(analysis, "findings", None)
        finding_count = len(findings() if callable(findings) else (findings or []))
        self._record(
            "analyze",
            analysis,
            binder=binder,
            steps=len(getattr(analysis, "steps", []) or []),
            findings=finding_count,
        )
        return analysis

    def chart(self, result: Any | None = None, **kwargs: Any) -> Any:
        """Render a content- and data-bound chart and record it as a stage.

        Delegates to :meth:`~vincio.core.app.ContextApp.generate_chart` (defaulting
        ``result`` to the engagement's last :attr:`result`), stores the
        :class:`~vincio.data.Chart` on :attr:`chart_`, and returns it. The stage
        carries a data-binder so :meth:`verify` re-executes the chart's source query
        and re-binds its credential.
        """
        source = result if result is not None else self.result
        if source is None:
            raise DataError(
                "chart() needs a query result; run query()/analyze() first or pass result="
            )
        chart = self.app.generate_chart(source, **kwargs)
        self.chart_ = chart
        self._record(
            "chart",
            chart,
            binder=lambda cat: bool(chart.verify(cat)),
            chart_type=chart.spec.mark.value,
            points=getattr(chart, "point_count", None),
            coverage=str(getattr(chart, "coverage", "")),
        )
        return chart

    def query_metric(
        self,
        request: Any,
        *,
        layer: Any | None = None,
        table: str | None = None,
        dataset: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Compute a governed metric through the semantic layer and record it as a stage.

        Delegates to :meth:`~vincio.core.app.ContextApp.query_metric`, stores the
        :class:`~vincio.data.MetricResult` on :attr:`metric`, and returns it. When the
        layer resolves and the metric ran over the registered catalog, the stage
        carries a data-binder so :meth:`verify` re-proves the answer is the layer's
        canonical compilation and re-derives from the source.
        """
        resolved = None
        resolver = getattr(self.app, "_resolve_layer", None)
        if callable(resolver):
            resolved = resolver(layer, table if table is not None else (self.table or None))
        result = self.app.query_metric(
            request, layer=resolved if resolved is not None else layer, dataset=dataset, **kwargs
        )
        self.metric = result
        binder = (
            (lambda cat: bool(result.verify(resolved, cat)))
            if resolved is not None and dataset is None
            else None
        )
        self._record(
            "metric",
            result,
            binder=binder,
            metrics=list(getattr(result, "metrics", []) or []),
            rows=getattr(result, "row_count", None),
        )
        return result

    # -- cite -----------------------------------------------------------------

    def cite(
        self,
        answer: Any | None = None,
        *,
        figures: list[Any] | None = None,
        title: str = "",
        contract: Any | None = None,
        require_figure_binding: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Assemble the findings into a cited, per-figure data-bound deliverable.

        Delegates to :meth:`~vincio.core.app.ContextApp.cited_report`, embedding the
        engagement's chart (and/or query table) as **data-bound**
        :class:`~vincio.generation.Figure`\\ s verified to re-derive from the source
        against the app's catalog, and records the :class:`~vincio.generation.report.CitedReport`
        as the closing ``cite`` stage. ``answer`` defaults to the analysis narrative
        (then the query's first answer); pass an explicit ``contract`` to override the
        default figure-binding :class:`~vincio.generation.report.CitationContract`.
        """
        from ..generation.report import CitationContract, Figure

        if figures is None:
            figures = []
            if self.chart_ is not None:
                figures.append(
                    Figure.from_chart(self.chart_, caption=self.chart_.spec.title or "Chart")
                )
            elif self.result is not None:
                figures.append(Figure.from_table(self.result, caption="Result"))
        if answer is None:
            if self.analysis is not None:
                answer = getattr(self.analysis, "narrative", "") or "See the attached figures."
            elif self.result is not None:
                answer = "See the attached figures."
            else:
                answer = "Data engagement summary."
        if contract is None:
            # The data engagement's headline citation guarantee is **per-figure data
            # binding** — every embedded chart/table re-derives from its source — not
            # per-claim [E]-marker coverage over the prose narrative (whose grounding
            # is the cell refs the figures carry). A caller wanting per-claim
            # entailment too passes an explicit, stricter contract.
            contract = CitationContract(
                min_coverage=0.0,
                allow_unresolved_markers=True,
                require_figure_binding=require_figure_binding,
            )
        catalog = self.app.data_catalog()
        # Each figure's data-binding is re-checked against the live source here, and
        # the figure-binding contract re-checks it again inside the builder (raising
        # if any figure is unbound) — so a returned deliverable is per-figure
        # data-bound by construction. The figures' underlying chart/query artifacts
        # are themselves recorded stages with their own re-executing binders.
        bound = sum(1 for f in figures if f.verify(catalog))
        report = self.app.cited_report(
            answer,
            title=title or self.question or "Data engagement",
            figures=figures,
            contract=contract,
            catalog=catalog,
            **kwargs,
        )
        self.report = report
        meta = getattr(report, "metadata", {}) or {}
        self._record(
            "cite",
            report,
            figures=len(figures),
            data_bound_figures=bound,
            figure_binding_rate=meta.get("figure_binding_rate"),
            unresolved=len(meta.get("unresolved_markers", []) or []),
        )
        return report

    # -- closure --------------------------------------------------------------

    def record_stage(
        self,
        stage: str,
        artifact: Any,
        *,
        binder: Callable[[Any], bool] | None = None,
        **summary: Any,
    ) -> Any:
        """Record an arbitrary already-produced artifact as a custom engagement stage.

        An escape hatch for a primitive without a dedicated facade method (or a
        caller-built artifact): binds ``artifact``'s digest into the narrative under
        the ``stage`` label exactly as the lifecycle methods do (with an optional
        ``binder`` for data-binding), and returns it.
        """
        self._record(stage, artifact, binder=binder, **summary)
        return artifact

    def seal(self, *, sign: bool = True, record_audit: bool = True) -> DataNarrative:
        """Mint the content-bound, signed :class:`DataNarrative` of the engagement.

        Hash-links every recorded stage, signs the narrative as the analyst, and —
        unless ``record_audit`` is off — lands the sealed engagement on the app's
        hash-chained audit log under ``data_engagement``. Returns the narrative;
        re-sealing after more stages run produces a fresh one.
        """
        narrative = DataNarrative(
            id=new_id("data_engagement"),
            analyst=self.analyst,
            dataset=self.table,
            question=self.question,
            started_at=self._started_at,
            sealed_at=utcnow(),
            stages=[s.model_copy(deep=True) for s in self._stages],
        )
        narrative.seal()
        if sign:
            signer = self._signer()
            if signer is not None:
                narrative.sign(signer, party=self.analyst)
        if record_audit and getattr(self.app, "audit", None) is not None:
            entry = self.app.audit.record(
                ENGAGEMENT_ACTION,
                resource=narrative.id,
                decision="sealed",
                details=narrative.audit_details(),
            )
            narrative.audit_id = getattr(entry, "id", None)
        self.narrative = narrative
        return narrative

    def verify(
        self,
        verifier: Any | None = None,
        *,
        require: list[str] | None = None,
        catalog: Any | None = None,
    ) -> DataEngagementVerification:
        """Verify the whole engagement offline — the chain, every digest, *and* every finding.

        Seals the narrative if needed, then verifies its hash chain from the bytes
        alone and re-digests every captured artifact against its bound digest (so a
        re-ordered stage or an edited underlying artifact is caught). Given the live
        ``catalog`` (defaulting to the app's registered
        :meth:`~vincio.core.app.ContextApp.data_catalog`), it additionally
        **re-executes** every query, analysis, chart, and metric against the
        content-hashed source and confirms each re-derives from the bytes — the
        data-plane's data-binding guarantee, surfaced as ``data_bound``. Pass the
        contract ``verifier`` to additionally authenticate the analyst's signature.
        """
        narrative = self.narrative or self.seal(sign=verifier is not None, record_audit=False)
        base = narrative.verify(verifier, require=require, artifacts=list(self._artifacts))

        if catalog is None:
            registered = self.app.data_catalog() if hasattr(self.app, "data_catalog") else None
            catalog = registered if (registered is not None and registered.names) else None

        data_bound: bool | None = None
        if catalog is not None:
            binders = [b for b in self._binders if b is not None]
            if binders:
                data_bound = all(self._safe_bind(b, catalog) for b in binders)

        valid = base.valid and data_bound is not False
        reason = base.reason
        if data_bound is False and reason is None:
            reason = "an analytical artifact failed to re-derive from its source"
        return base.model_copy(update={"data_bound": data_bound, "valid": valid, "reason": reason})

    @staticmethod
    def _safe_bind(binder: Callable[[Any], bool], catalog: Any) -> bool:
        """Run a data-binder, treating any failure to re-execute as not-bound."""
        try:
            return bool(binder(catalog))
        except Exception:
            return False

    def _signer(self) -> Any:
        """The app's contract signer, when one is resolvable."""
        resolver = getattr(self.app, "_resolve_contract_signer", None)
        if callable(resolver):
            return resolver(None, True)
        return getattr(self.app, "contract_signer", None)
