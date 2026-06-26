"""Data-quality rails: deterministic screening for tabular input.

A tabular input is screened the way a text input already is — on a deterministic
rail path, with no model judgment, every finding explainable and free. Where the
security rails screen text for PII, secrets, and injection,
:class:`DataQualityRails` screen a :class:`~vincio.data.Dataset` for the failure
modes structured data has:

* **schema violations** — a value of the wrong type, a null in a non-nullable
  column, a null rate above a ceiling;
* **constraint breaks** — a value out of range, outside an allowed set, not
  matching a required pattern, a broken uniqueness or monotonicity guarantee;
* **anomalies** — numeric outliers, found with a robust (median/MAD) z-score
  that a few extreme values cannot mask.

The very same security detectors ride this path too: a constraint may run the
PII, secret, or injection detector over a column's string cells, so a leaked
email in a data table is caught exactly as it would be in a prompt. Every finding
becomes a :class:`DataQualityViolation`; a blocking finding fails the screen, and
:meth:`DataQualityReport.raise_for_status` raises a
:class:`~vincio.core.errors.DataQualityError`.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import DataError, DataQualityError
from .core import ColumnSchema, DataSchema, Dataset, DataType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..security.injection import InjectionDetector
    from ..security.pii import PIIDetector
    from ..security.secrets import SecretScanner

__all__ = [
    "ColumnConstraint",
    "DataQualityViolation",
    "DataQualityReport",
    "DataQualityRails",
]

QualityAction = Literal["block", "warn"]
QualitySeverity = Literal["error", "warning", "info"]
Detector = Literal["pii", "secrets", "injection"]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _matches_dtype(value: Any, dtype: DataType) -> bool:
    """Whether a non-null value is consistent with a declared column type."""
    if dtype is DataType.INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if dtype is DataType.FLOAT:
        return _is_number(value)
    if dtype is DataType.BOOL:
        return isinstance(value, bool)
    if dtype is DataType.STR:
        return isinstance(value, str)
    if dtype is DataType.DATE:
        return isinstance(value, (str, _dt.date))
    if dtype is DataType.DATETIME:
        return isinstance(value, (str, _dt.datetime))
    if dtype is DataType.TIME:
        return isinstance(value, (str, _dt.time))
    return True  # NULL / unknown — no constraint


class ColumnConstraint(BaseModel):
    """A declarative quality contract for one column.

    Leave a field unset to skip that check. ``action`` decides whether a breach
    blocks the screen (``block``) or only warns (``warn``).
    """

    column: str
    dtype: DataType | None = None
    nullable: bool | None = None
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] = Field(default_factory=list)
    pattern: str | None = None
    unique: bool = False
    monotonic: Literal["increasing", "decreasing"] | None = None
    max_null_rate: float | None = None
    detectors: list[Detector] = Field(default_factory=list)
    action: QualityAction = "block"


class DataQualityViolation(BaseModel):
    """One data-quality finding: which column, which rule, how many rows, and a
    few example offenders."""

    column: str
    rule: str
    action: QualityAction
    severity: QualitySeverity
    message: str
    count: int = 0
    examples: list[Any] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class DataQualityReport(BaseModel):
    """The outcome of screening a dataset. ``allowed`` is false when any blocking
    rule fired; the violations carry the detail."""

    allowed: bool = True
    row_count: int = 0
    column_count: int = 0
    checked_columns: list[str] = Field(default_factory=list)
    violations: list[DataQualityViolation] = Field(default_factory=list)

    @property
    def blocking(self) -> list[DataQualityViolation]:
        """Violations whose action is ``block``."""
        return [v for v in self.violations if v.action == "block"]

    @property
    def warnings(self) -> list[DataQualityViolation]:
        """Violations whose action is ``warn``."""
        return [v for v in self.violations if v.action == "warn"]

    def raise_for_status(self) -> None:
        """Raise :class:`~vincio.core.errors.DataQualityError` if the screen was
        blocked; otherwise return."""
        if not self.allowed:
            raise DataQualityError(self.summary())

    def summary(self) -> str:
        """A one-line human summary of the screen."""
        if self.allowed and not self.violations:
            return f"data quality OK ({self.row_count} rows × {self.column_count} columns)"
        blocking = self.blocking
        lead = "blocked" if not self.allowed else "passed with warnings"
        head = ", ".join(f"{v.column}:{v.rule}" for v in (blocking or self.violations)[:5])
        return f"data quality {lead} — {len(self.violations)} finding(s): {head}"


class DataQualityRails:
    """Screen tabular data deterministically against a set of column constraints,
    with optional numeric anomaly detection.

    Build it from explicit :class:`ColumnConstraint`s, or derive a baseline from a
    schema or dataset with :meth:`from_schema` / :meth:`from_dataset` (which
    enforces each column's declared type and nullability). :meth:`check` returns a
    :class:`DataQualityReport`.
    """

    def __init__(
        self,
        constraints: list[ColumnConstraint] | None = None,
        *,
        detect_anomalies: bool = False,
        anomaly_threshold: float = 3.5,
        anomaly_action: QualityAction = "warn",
        max_examples: int = 3,
        pii_detector: PIIDetector | None = None,
        secret_scanner: SecretScanner | None = None,
        injection_detector: InjectionDetector | None = None,
    ) -> None:
        self.constraints = list(constraints or [])
        self.detect_anomalies = detect_anomalies
        self.anomaly_threshold = anomaly_threshold
        self.anomaly_action = anomaly_action
        self.max_examples = max_examples
        self._pii = pii_detector
        self._secrets = secret_scanner
        self._injection = injection_detector

    @classmethod
    def from_schema(
        cls, schema: DataSchema | list[ColumnSchema], *, action: QualityAction = "block", **kwargs: Any
    ) -> DataQualityRails:
        """Derive rails that enforce a schema: each column's declared type and,
        for a non-nullable column, the absence of nulls."""
        columns = schema.columns if isinstance(schema, DataSchema) else list(schema)
        constraints = [
            ColumnConstraint(column=c.name, dtype=c.dtype, nullable=c.nullable, action=action)
            for c in columns
        ]
        return cls(constraints, **kwargs)

    @classmethod
    def from_dataset(cls, dataset: Dataset, **kwargs: Any) -> DataQualityRails:
        """Derive schema-enforcing rails from a dataset's own declared schema."""
        return cls.from_schema(dataset.data_schema, **kwargs)

    def check(self, data: Dataset | list[dict[str, Any]]) -> DataQualityReport:
        """Screen a dataset, returning a :class:`DataQualityReport`."""
        dataset = data if isinstance(data, Dataset) else Dataset.from_records(data)
        names = dataset.column_names
        violations: list[DataQualityViolation] = []
        checked: list[str] = []
        for constraint in self.constraints:
            if constraint.column not in names:
                violations.append(
                    DataQualityViolation(
                        column=constraint.column,
                        rule="missing_column",
                        action=constraint.action,
                        severity=_severity(constraint.action),
                        message=f"constrained column {constraint.column!r} is absent",
                    )
                )
                continue
            checked.append(constraint.column)
            values = dataset.column(constraint.column)
            violations.extend(self._check_column(constraint, values))
        if self.detect_anomalies:
            for name in names:
                violations.extend(self._check_anomalies(name, dataset.column(name)))
        allowed = not any(v.action == "block" for v in violations)
        return DataQualityReport(
            allowed=allowed,
            row_count=dataset.row_count,
            column_count=dataset.width,
            checked_columns=checked,
            violations=violations,
        )

    # -- per-column checks -----------------------------------------------------

    def _check_column(self, c: ColumnConstraint, values: list[Any]) -> list[DataQualityViolation]:
        out: list[DataQualityViolation] = []
        nonnull = [v for v in values if v is not None]
        null_count = len(values) - len(nonnull)

        if c.nullable is False and null_count:
            out.append(self._violation(c, "null_in_non_nullable", f"{null_count} null(s) in non-nullable column", null_count))

        if c.dtype is not None:
            offenders = [v for v in nonnull if not _matches_dtype(v, c.dtype)]
            if offenders:
                out.append(self._violation(c, "type_mismatch", f"{len(offenders)} value(s) not {c.dtype.value}", len(offenders), offenders))

        if c.min_value is not None or c.max_value is not None:
            offenders = [
                v
                for v in nonnull
                if _is_number(v)
                and ((c.min_value is not None and v < c.min_value) or (c.max_value is not None and v > c.max_value))
            ]
            if offenders:
                bounds = f"[{c.min_value}, {c.max_value}]"
                out.append(self._violation(c, "out_of_range", f"{len(offenders)} value(s) outside {bounds}", len(offenders), offenders))

        if c.allowed_values:
            allowed = set(c.allowed_values)
            offenders = [v for v in nonnull if v not in allowed]
            if offenders:
                out.append(self._violation(c, "not_allowed", f"{len(offenders)} value(s) outside the allowed set", len(offenders), offenders))

        if c.pattern is not None:
            compiled = re.compile(c.pattern)
            offenders = [v for v in nonnull if not (isinstance(v, str) and compiled.fullmatch(v))]
            if offenders:
                out.append(self._violation(c, "pattern_mismatch", f"{len(offenders)} value(s) do not match {c.pattern!r}", len(offenders), offenders))

        if c.unique:
            seen: set[Any] = set()
            dupes: list[Any] = []
            for v in nonnull:
                if v in seen:
                    dupes.append(v)
                else:
                    seen.add(v)
            if dupes:
                out.append(self._violation(c, "not_unique", f"{len(dupes)} duplicate value(s)", len(dupes), dupes))

        if c.monotonic is not None:
            breaks = _monotonic_breaks(nonnull, c.monotonic)
            if breaks:
                out.append(self._violation(c, "not_monotonic", f"{breaks} value(s) break {c.monotonic} order", breaks))

        if c.max_null_rate is not None and values:
            rate = null_count / len(values)
            if rate > c.max_null_rate:
                out.append(self._violation(c, "null_rate", f"null rate {rate:.3f} exceeds {c.max_null_rate}", null_count, details={"null_rate": round(rate, 6)}))

        if c.detectors:
            out.extend(self._check_detectors(c, nonnull))
        return out

    def _check_detectors(self, c: ColumnConstraint, nonnull: list[Any]) -> list[DataQualityViolation]:
        out: list[DataQualityViolation] = []
        strings = [v for v in nonnull if isinstance(v, str)]
        for detector in c.detectors:
            hits = self._detect(detector, strings)
            if hits:
                out.append(
                    self._violation(
                        c,
                        f"{detector}_detected",
                        f"{detector} detected in {len(hits)} cell(s)",
                        len(hits),
                        hits,
                        details={"detector": detector},
                    )
                )
        return out

    def _detect(self, detector: Detector, strings: list[str]) -> list[str]:
        hits: list[str] = []
        if detector == "pii":
            if self._pii is None:
                from ..security.pii import PIIDetector

                self._pii = PIIDetector()
            for s in strings:
                if self._pii.detect(s):
                    hits.append(s)
        elif detector == "secrets":
            if self._secrets is None:
                from ..security.secrets import SecretScanner

                self._secrets = SecretScanner()
            for s in strings:
                if self._secrets.scan_text(s):
                    hits.append(s)
        elif detector == "injection":
            if self._injection is None:
                from ..security.injection import InjectionDetector

                self._injection = InjectionDetector()
            for s in strings:
                if self._injection.detect(s).detected:
                    hits.append(s)
        return hits

    def _check_anomalies(self, name: str, values: list[Any]) -> list[DataQualityViolation]:
        numbers = [float(v) for v in values if _is_number(v)]
        if len(numbers) < 4:
            return []
        outliers = _mad_outliers(numbers, self.anomaly_threshold)
        if not outliers:
            return []
        return [
            DataQualityViolation(
                column=name,
                rule="anomaly",
                action=self.anomaly_action,
                severity=_severity(self.anomaly_action),
                message=f"{len(outliers)} numeric outlier(s) (robust z > {self.anomaly_threshold})",
                count=len(outliers),
                examples=outliers[: self.max_examples],
                details={"method": "mad", "threshold": self.anomaly_threshold},
            )
        ]

    def _violation(
        self,
        c: ColumnConstraint,
        rule: str,
        message: str,
        count: int,
        examples: list[Any] | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> DataQualityViolation:
        return DataQualityViolation(
            column=c.column,
            rule=rule,
            action=c.action,
            severity=_severity(c.action),
            message=message,
            count=count,
            examples=list(examples or [])[: self.max_examples],
            details=details or {},
        )


def _severity(action: QualityAction) -> QualitySeverity:
    return "error" if action == "block" else "warning"


def _monotonic_breaks(values: list[Any], direction: str) -> int:
    """Count adjacent pairs that violate the requested monotonic order."""
    comparable = [v for v in values if _is_number(v) or isinstance(v, str)]
    breaks = 0
    for prev, cur in zip(comparable, comparable[1:], strict=False):
        try:
            if direction == "increasing" and cur < prev:
                breaks += 1
            elif direction == "decreasing" and cur > prev:
                breaks += 1
        except TypeError as exc:  # mixed, uncomparable types
            raise DataError("cannot check monotonicity over mixed, uncomparable types") from exc
    return breaks


def _mad_outliers(numbers: list[float], threshold: float) -> list[float]:
    """Robust outlier detection via the median absolute deviation (Iglewicz–
    Hoaglin modified z-score). Resistant to the very outliers it looks for."""
    ordered = sorted(numbers)
    median = _median(ordered)
    deviations = sorted(abs(x - median) for x in numbers)
    mad = _median(deviations)
    if mad == 0:
        return []
    out = [x for x in numbers if 0.6745 * abs(x - median) / mad > threshold]
    return sorted(out)


def _median(ordered: list[float]) -> float:
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
