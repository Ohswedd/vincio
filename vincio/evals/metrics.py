"""Eval metrics.

Task metrics (exact match, similarity, F1), grounding metrics (faithfulness,
context precision/recall, citation accuracy, unsupported-claim rate),
operational metrics (tokens/cost/latency), retrieval metrics (recall@K,
precision@K, MRR, NDCG). Deterministic by default; judge-based variants
live in judges.py.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ..context.compression import split_sentences
from ..context.scoring import lexical_similarity
from ..core.types import EvidenceItem, TokenUsage
from .datasets import EvalCase

__all__ = [
    "RunOutput",
    "MetricResult",
    "Metric",
    "METRICS",
    "register_metric",
    "exact_match",
    "semantic_similarity",
    "classification_accuracy",
    "extraction_f1",
    "schema_validity",
    "groundedness",
    "citation_accuracy",
    "citation_recall",
    "unsupported_claim_rate",
    "context_precision",
    "context_recall",
    "cost_metric",
    "latency_metric",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg",
]


class RunOutput(BaseModel):
    """What the system produced for one eval case."""

    model_config = {"arbitrary_types_allowed": True}

    output: Any = None
    raw_text: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    latency_ms: int = 0
    schema_valid: bool | None = None
    parse_success: bool | None = None
    retries: int = 0
    error: str | None = None
    trace_id: str = ""
    agent_metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def output_text(self) -> str:
        if isinstance(self.output, str):
            return self.output
        if self.output is None:
            return self.raw_text
        if hasattr(self.output, "model_dump_json"):
            return self.output.model_dump_json()
        import json

        try:
            return json.dumps(self.output, default=str)
        except (TypeError, ValueError):
            return str(self.output)


class MetricResult(BaseModel):
    name: str
    value: float
    passed: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)


Metric = Callable[[EvalCase, RunOutput], MetricResult]

METRICS: dict[str, Metric] = {}


def register_metric(name: str):
    def decorator(fn: Metric) -> Metric:
        METRICS[name] = fn
        return fn

    return decorator


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def _expected_text(case: EvalCase) -> str:
    if case.expected is None:
        return ""
    if isinstance(case.expected, str):
        return case.expected
    import json

    return json.dumps(case.expected, default=str)


# -- task metrics ---------------------------------------------------------------------


@register_metric("exact_match")
def exact_match(case: EvalCase, run: RunOutput) -> MetricResult:
    expected = _normalize(_expected_text(case))
    got = _normalize(run.output_text)
    value = 1.0 if expected and expected == got else 0.0
    return MetricResult(name="exact_match", value=value, passed=value == 1.0)


@register_metric("semantic_similarity")
def semantic_similarity(case: EvalCase, run: RunOutput) -> MetricResult:
    value = lexical_similarity(_expected_text(case), run.output_text)
    return MetricResult(name="semantic_similarity", value=round(value, 4))


@register_metric("classification_accuracy")
def classification_accuracy(case: EvalCase, run: RunOutput) -> MetricResult:
    expected = case.expected
    if isinstance(expected, dict):
        expected = expected.get("label")
    got: Any = run.output
    if isinstance(got, dict):
        got = got.get("label")
    elif hasattr(got, "label"):
        got = got.label
    elif isinstance(got, str):
        got = got.strip()
    value = 1.0 if _normalize(str(expected)) == _normalize(str(got)) else 0.0
    return MetricResult(
        name="classification_accuracy", value=value, passed=value == 1.0,
        details={"expected": expected, "got": got},
    )


@register_metric("extraction_f1")
def extraction_f1(case: EvalCase, run: RunOutput) -> MetricResult:
    """F1 over extracted item sets (order-insensitive, normalized strings)."""

    def to_set(value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, dict):
            return {f"{k}={_normalize(str(v))}" for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            out: set[str] = set()
            for item in value:
                if isinstance(item, dict):
                    out.add(_normalize(" ".join(f"{k}:{v}" for k, v in sorted(item.items()))))
                else:
                    out.add(_normalize(str(item)))
            return out
        return {_normalize(str(value))}

    expected_set = to_set(case.expected)
    output = run.output
    if hasattr(output, "model_dump"):
        output = output.model_dump()
    got_set = to_set(output)
    if not expected_set and not got_set:
        return MetricResult(name="extraction_f1", value=1.0, passed=True)
    true_positive = len(expected_set & got_set)
    precision = true_positive / len(got_set) if got_set else 0.0
    recall = true_positive / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return MetricResult(
        name="extraction_f1", value=round(f1, 4),
        details={"precision": round(precision, 4), "recall": round(recall, 4)},
    )


@register_metric("schema_validity")
def schema_validity(case: EvalCase, run: RunOutput) -> MetricResult:
    value = 1.0 if run.schema_valid else 0.0
    if run.schema_valid is None:
        value = 1.0 if run.error is None else 0.0
    return MetricResult(name="schema_validity", value=value, passed=value == 1.0)


# -- grounding metrics -----------------------------------------------------

_VERIFIABLE_RE = re.compile(r"\d|%|\$|€|\b(is|are|was|were|has|have|will|must|requires?)\b", re.IGNORECASE)


def _verifiable_claims(text: str) -> list[str]:
    return [
        sentence
        for sentence in split_sentences(text)
        if len(sentence.split()) >= 4 and _VERIFIABLE_RE.search(sentence)
    ]


def _supported(claim: str, evidence: list[EvidenceItem], threshold: float = 0.28) -> bool:
    return any(
        lexical_similarity(claim, item.text or "") >= threshold for item in evidence if item.text
    )


@register_metric("groundedness")
def groundedness(case: EvalCase, run: RunOutput) -> MetricResult:
    """supported_claims / total_verifiable_claims against run evidence."""
    claims = _verifiable_claims(run.output_text)
    if not claims:
        return MetricResult(name="groundedness", value=1.0, details={"claims": 0})
    supported = sum(1 for claim in claims if _supported(claim, run.evidence))
    value = supported / len(claims)
    return MetricResult(
        name="groundedness", value=round(value, 4),
        details={"claims": len(claims), "supported": supported},
    )


@register_metric("unsupported_claim_rate")
def unsupported_claim_rate(case: EvalCase, run: RunOutput) -> MetricResult:
    claims = _verifiable_claims(run.output_text)
    if not claims:
        return MetricResult(name="unsupported_claim_rate", value=0.0)
    unsupported = sum(1 for claim in claims if not _supported(claim, run.evidence))
    return MetricResult(name="unsupported_claim_rate", value=round(unsupported / len(claims), 4))


@register_metric("citation_accuracy")
def citation_accuracy(case: EvalCase, run: RunOutput) -> MetricResult:
    """correct_citations / total_citations."""
    if not run.citations:
        return MetricResult(name="citation_accuracy", value=0.0, details={"citations": 0})
    valid_ids = {e.id for e in run.evidence} | {e.citation_ref for e in run.evidence}
    correct = sum(1 for citation in run.citations if citation in valid_ids)
    return MetricResult(
        name="citation_accuracy", value=round(correct / len(run.citations), 4),
        details={"citations": len(run.citations), "correct": correct},
    )


@register_metric("citation_recall")
def citation_recall(case: EvalCase, run: RunOutput) -> MetricResult:
    """required_evidence_cited / required_evidence; required ids come from
    case.rubric['required_evidence'] or all run evidence as fallback."""
    required = case.rubric.get("required_evidence") or [e.id for e in run.evidence]
    if not required:
        return MetricResult(name="citation_recall", value=1.0)
    cited = set(run.citations)
    hit = sum(1 for ref in required if ref in cited)
    return MetricResult(
        name="citation_recall", value=round(hit / len(required), 4),
        details={"required": len(required), "cited": hit},
    )


@register_metric("context_precision")
def context_precision(case: EvalCase, run: RunOutput) -> MetricResult:
    """Fraction of retrieved evidence relevant to the expected answer/input."""
    if not run.evidence:
        return MetricResult(name="context_precision", value=0.0)
    reference = _expected_text(case) or case.input_text
    relevant = sum(
        1 for item in run.evidence if lexical_similarity(item.text or "", reference) >= 0.15
    )
    return MetricResult(name="context_precision", value=round(relevant / len(run.evidence), 4))


@register_metric("context_recall")
def context_recall(case: EvalCase, run: RunOutput) -> MetricResult:
    """Fraction of expected facts covered by retrieved evidence; expected
    facts come from rubric['facts'] or sentences of the expected answer."""
    facts = case.rubric.get("facts") or _verifiable_claims(_expected_text(case))
    if not facts:
        return MetricResult(name="context_recall", value=1.0)
    covered = sum(1 for fact in facts if _supported(str(fact), run.evidence, threshold=0.25))
    return MetricResult(
        name="context_recall", value=round(covered / len(facts), 4),
        details={"facts": len(facts), "covered": covered},
    )


# -- operational metrics ----------------------------------------------------------------------


@register_metric("cost")
def cost_metric(case: EvalCase, run: RunOutput) -> MetricResult:
    return MetricResult(name="cost", value=round(run.cost_usd, 8))


@register_metric("latency")
def latency_metric(case: EvalCase, run: RunOutput) -> MetricResult:
    return MetricResult(name="latency", value=float(run.latency_ms))


@register_metric("input_tokens")
def input_tokens_metric(case: EvalCase, run: RunOutput) -> MetricResult:
    return MetricResult(name="input_tokens", value=float(run.usage.input_tokens))


@register_metric("output_tokens")
def output_tokens_metric(case: EvalCase, run: RunOutput) -> MetricResult:
    return MetricResult(name="output_tokens", value=float(run.usage.output_tokens))


@register_metric("retries")
def retries_metric(case: EvalCase, run: RunOutput) -> MetricResult:
    return MetricResult(name="retries", value=float(run.retries))


# -- retrieval ranking metrics ------------------------------------------------------


def _relevant_ids(case: EvalCase) -> set[str]:
    return set(case.rubric.get("relevant_ids") or case.context.get("relevant_ids") or [])


@register_metric("recall_at_k")
def recall_at_k(case: EvalCase, run: RunOutput) -> MetricResult:
    relevant = _relevant_ids(case)
    if not relevant:
        return MetricResult(name="recall_at_k", value=1.0)
    retrieved = [e.id for e in run.evidence] + [e.citation_ref for e in run.evidence]
    hit = sum(1 for ref in relevant if ref in retrieved)
    return MetricResult(name="recall_at_k", value=round(hit / len(relevant), 4))


@register_metric("precision_at_k")
def precision_at_k(case: EvalCase, run: RunOutput) -> MetricResult:
    relevant = _relevant_ids(case)
    if not run.evidence:
        return MetricResult(name="precision_at_k", value=0.0 if relevant else 1.0)
    hits = sum(1 for e in run.evidence if e.id in relevant or e.citation_ref in relevant)
    return MetricResult(name="precision_at_k", value=round(hits / len(run.evidence), 4))


@register_metric("mrr")
def mrr(case: EvalCase, run: RunOutput) -> MetricResult:
    relevant = _relevant_ids(case)
    for rank, item in enumerate(run.evidence, start=1):
        if item.id in relevant or item.citation_ref in relevant:
            return MetricResult(name="mrr", value=round(1.0 / rank, 4))
    return MetricResult(name="mrr", value=0.0 if relevant else 1.0)


@register_metric("ndcg")
def ndcg(case: EvalCase, run: RunOutput) -> MetricResult:
    relevant = _relevant_ids(case)
    if not relevant:
        return MetricResult(name="ndcg", value=1.0)
    dcg = 0.0
    for rank, item in enumerate(run.evidence, start=1):
        if item.id in relevant or item.citation_ref in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), len(run.evidence)) + 1))
    return MetricResult(name="ndcg", value=round(dcg / ideal, 4) if ideal else 0.0)
