"""Eval metrics.

Task metrics (exact match, similarity, F1), grounding metrics (faithfulness,
context precision/recall, citation accuracy, unsupported-claim rate),
quality & safety metrics (answer relevance, hallucination, toxicity, bias,
summarization quality), conversational metrics (knowledge retention,
conversation relevance), operational metrics (tokens/cost/latency), retrieval
metrics (recall@K, precision@K, MRR, NDCG). Deterministic by default;
judge-based variants (including rubric-based G-Eval) live in judges.py.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ..context.compression import split_sentences
from ..context.scoring import containment_similarity, lexical_similarity
from ..core.types import EvidenceItem, TokenUsage
from ..retrieval.embeddings import LocalHashEmbedder, cosine
from .datasets import EvalCase
from .trajectory import (
    Trajectory,
    trajectory_from_agent_state,
    trajectory_from_crew_result,
    trajectory_from_trace,
)

__all__ = [
    "RunOutput",
    "MetricResult",
    "Metric",
    "METRICS",
    "LOWER_IS_BETTER",
    "register_metric",
    "set_semantic_embedder",
    "exact_match",
    "lexical_overlap",
    "semantic_similarity",
    "classification_accuracy",
    "extraction_f1",
    "schema_validity",
    "groundedness",
    "citation_accuracy",
    "citation_recall",
    "citation_coverage",
    "claim_entailment",
    "unsupported_claim_rate",
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevance",
    "hallucination",
    "toxicity",
    "bias",
    "summarization_quality",
    "knowledge_retention",
    "conversation_relevance",
    "conversation_outcome",
    "intent_resolution",
    "tool_call_accuracy",
    "tool_call_f1",
    "goal_accuracy",
    "plan_adherence",
    "plan_quality",
    "step_efficiency",
    "topic_adherence",
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
    trajectory: Trajectory | None = None
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

    @classmethod
    def from_agent_state(cls, state: Any, **fields: Any) -> RunOutput:
        """Build a RunOutput from a completed agent run (``AgentState``),
        carrying its trajectory so trajectory metrics can score it. The final
        answer becomes the output; usage/cost ride along. Extra ``fields``
        (e.g. ``trace_id``, ``evidence``) override the derived values."""
        traj = trajectory_from_agent_state(state)
        usage = TokenUsage(
            input_tokens=int(traj.usage.get("input_tokens", 0)),
            output_tokens=int(traj.usage.get("output_tokens", 0)),
        )
        data: dict[str, Any] = {
            "output": state.final_answer,
            "raw_text": state.raw_answer_text or "",
            "evidence": list(getattr(state, "evidence", []) or []),
            "usage": usage,
            "cost_usd": float(traj.usage.get("cost_usd", 0.0)),
            "trajectory": traj,
        }
        data.update(fields)
        return cls(**data)

    @classmethod
    def from_crew_result(cls, result: Any, **fields: Any) -> RunOutput:
        """Build a RunOutput from a :class:`~vincio.agents.crew.CrewResult`."""
        traj = trajectory_from_crew_result(result)
        usage = TokenUsage(
            input_tokens=int(traj.usage.get("input_tokens", 0)),
            output_tokens=int(traj.usage.get("output_tokens", 0)),
        )
        data: dict[str, Any] = {
            "output": result.output,
            "raw_text": str(result.output or ""),
            "usage": usage,
            "cost_usd": float(traj.usage.get("cost_usd", 0.0)),
            "trajectory": traj,
        }
        data.update(fields)
        return cls(**data)

    @classmethod
    def from_trace(cls, trace: Any, **fields: Any) -> RunOutput:
        """Build a RunOutput from a captured ``Trace`` (no re-run needed)."""
        traj = trajectory_from_trace(trace)
        data: dict[str, Any] = {
            "output": trace.attributes.get("output"),
            "raw_text": str(trace.attributes.get("output") or ""),
            "trace_id": trace.id,
            "cost_usd": float(trace.attributes.get("cost_usd", 0.0) or 0.0),
            "trajectory": traj,
        }
        data.update(fields)
        return cls(**data)


class MetricResult(BaseModel):
    name: str
    value: float
    passed: bool | None = None
    # a metric that has no reference to score against (no ground truth, no
    # claims, no trajectory) returns ``skipped=True`` instead of a neutral
    # ``1.0``. The runner excludes skipped results from ``CaseResult.metrics``,
    # so they never inflate a mean or silently pass a gate. ``value`` is still
    # carried for display but must not be aggregated.
    skipped: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


Metric = Callable[[EvalCase, RunOutput], MetricResult]

METRICS: dict[str, Metric] = {}

# Metrics where a LOWER value is better (rates, costs, counts). The single
# source of truth for direction — report diffs, experiment comparison, and
# test assertions all consult this set.
LOWER_IS_BETTER: set[str] = {
    "hallucination",
    "toxicity",
    "bias",
    "unsupported_claim_rate",
    "cost",
    "latency",
    "input_tokens",
    "output_tokens",
    "retries",
}


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


@register_metric("lexical_overlap")
def lexical_overlap(case: EvalCase, run: RunOutput) -> MetricResult:
    """Stemmed bag-of-words overlap between expected and produced text.

    Purely lexical (token Jaccard with stemming and stop-word removal), not
    embedding-based — the name ``semantic_similarity`` is reserved for the
    embedding-backed metric below.
    """
    value = lexical_similarity(_expected_text(case), run.output_text)
    return MetricResult(name="lexical_overlap", value=round(value, 4))


# Offline-deterministic embedder backing ``semantic_similarity`` when no
# semantic embedder is configured. Swap it with :func:`set_semantic_embedder`
# (e.g. a provider embedder) to make the metric truly semantic. Any object with
# a synchronous ``embed_one(text) -> list[float]`` works.
_SEMANTIC_EMBEDDER: Any = LocalHashEmbedder(dim=256)


def set_semantic_embedder(embedder: Any) -> None:
    """Install the embedder used by the ``semantic_similarity`` metric.

    ``embedder`` must expose a synchronous ``embed_one(text) -> list[float]``.
    Defaults to the deterministic offline :class:`LocalHashEmbedder`.
    """
    global _SEMANTIC_EMBEDDER
    _SEMANTIC_EMBEDDER = embedder


@register_metric("semantic_similarity")
def semantic_similarity(case: EvalCase, run: RunOutput) -> MetricResult:
    """Embedding cosine between expected and produced text.

    Unlike the lexical ``lexical_overlap``, this scores vector-space closeness,
    so paraphrases that share little surface wording still score highly when a
    real semantic embedder is installed via :func:`set_semantic_embedder`. The
    default offline embedder is deterministic, so runs stay reproducible.
    Unscoreable when there is no expected reference.
    """
    expected = _expected_text(case)
    got = run.output_text
    if not expected:
        return MetricResult(name="semantic_similarity", value=1.0, skipped=True)
    if not got:
        return MetricResult(name="semantic_similarity", value=0.0)
    va = _SEMANTIC_EMBEDDER.embed_one(expected)
    vb = _SEMANTIC_EMBEDDER.embed_one(got)
    value = max(0.0, min(1.0, cosine(va, vb)))
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
        name="classification_accuracy",
        value=value,
        passed=value == 1.0,
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
    if case.expected is None:
        # No reference to score against — unscoreable, not a free pass.
        return MetricResult(name="extraction_f1", value=1.0, skipped=True)
    if not expected_set and not got_set:
        return MetricResult(name="extraction_f1", value=1.0, passed=True)
    true_positive = len(expected_set & got_set)
    precision = true_positive / len(got_set) if got_set else 0.0
    recall = true_positive / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return MetricResult(
        name="extraction_f1",
        value=round(f1, 4),
        details={"precision": round(precision, 4), "recall": round(recall, 4)},
    )


@register_metric("schema_validity")
def schema_validity(case: EvalCase, run: RunOutput) -> MetricResult:
    value = 1.0 if run.schema_valid else 0.0
    if run.schema_valid is None:
        value = 1.0 if run.error is None else 0.0
    return MetricResult(name="schema_validity", value=value, passed=value == 1.0)


# -- grounding metrics -----------------------------------------------------

_VERIFIABLE_RE = re.compile(
    r"\d|%|\$|€|\b(is|are|was|were|has|have|will|must|requires?)\b", re.IGNORECASE
)
_UNICODE_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _language_neutral_declarative(sentence: str) -> bool:
    letters = [char for char in sentence if char.isalpha()]
    return bool(
        len(letters) >= 8
        and any(ord(char) > 127 for char in letters)
        and not sentence.rstrip().endswith(("?", "？"))
    )


def _verifiable_claims(text: str) -> list[str]:
    claims: list[str] = []
    for sentence in split_sentences(text):
        non_english_declarative = _language_neutral_declarative(sentence)
        enough_content = len(_UNICODE_WORD_RE.findall(sentence)) >= 4 or non_english_declarative
        if enough_content and (_VERIFIABLE_RE.search(sentence) or non_english_declarative):
            claims.append(sentence)
    return claims


def _supported(claim: str, evidence: list[EvidenceItem], threshold: float = 0.28) -> bool:
    return any(
        lexical_similarity(claim, item.text or "") >= threshold for item in evidence if item.text
    )


@register_metric("groundedness")
def groundedness(case: EvalCase, run: RunOutput) -> MetricResult:
    """supported_claims / total_verifiable_claims against run evidence."""
    claims = _verifiable_claims(run.output_text)
    if not claims:
        return MetricResult(name="groundedness", value=1.0, skipped=True, details={"claims": 0})
    supported = sum(1 for claim in claims if _supported(claim, run.evidence))
    value = supported / len(claims)
    return MetricResult(
        name="groundedness",
        value=round(value, 4),
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
        name="citation_accuracy",
        value=round(correct / len(run.citations), 4),
        details={"citations": len(run.citations), "correct": correct},
    )


@register_metric("citation_recall")
def citation_recall(case: EvalCase, run: RunOutput) -> MetricResult:
    """required_evidence_cited / required_evidence; required ids come from
    case.rubric['required_evidence'] or all run evidence as fallback."""
    required = case.rubric.get("required_evidence") or [e.id for e in run.evidence]
    if not required:
        return MetricResult(name="citation_recall", value=1.0, skipped=True)
    cited = set(run.citations)
    hit = sum(1 for ref in required if ref in cited)
    return MetricResult(
        name="citation_recall",
        value=round(hit / len(required), 4),
        details={"required": len(required), "cited": hit},
    )


@register_metric("citation_coverage")
def citation_coverage(case: EvalCase, run: RunOutput) -> MetricResult:
    """Sentence-level citation coverage: fraction of verifiable claims in the
    output whose citation markers resolve to run evidence. Resolution-aware (a
    marker that points at no evidence does not count), matching the cited-report
    builder's coverage so the two never disagree; when no evidence is supplied it
    falls back to marker presence."""
    from ..output.parsers import extract_citations

    claims = _verifiable_claims(run.output_text)
    if not claims:
        return MetricResult(
            name="citation_coverage", value=1.0, skipped=True, details={"claims": 0}
        )
    valid_ids = (
        {e.id for e in run.evidence}
        | {e.citation_ref for e in run.evidence}
        | {e.source_id for e in run.evidence}
    )

    def is_cited(claim: str) -> bool:
        markers = extract_citations(claim)
        if not markers:
            return False
        return any(m in valid_ids for m in markers) if valid_ids else True

    cited = sum(1 for claim in claims if is_cited(claim))
    return MetricResult(
        name="citation_coverage",
        value=round(cited / len(claims), 4),
        details={"claims": len(claims), "cited": cited},
    )


@register_metric("claim_entailment")
def claim_entailment(case: EvalCase, run: RunOutput) -> MetricResult:
    """Of the cited claims, the fraction whose cited evidence actually supports
    them (strict lexical + numeric entailment). Returns 1.0 when nothing is
    cited (no claim to refute), so it pairs with ``citation_coverage``."""
    from ..output.parsers import extract_citations

    by_ref: dict[str, EvidenceItem] = {}
    for item in run.evidence:
        for key in (item.id, item.citation_ref, item.source_id):
            if key:
                by_ref.setdefault(key, item)
    claims = _verifiable_claims(run.output_text)
    cited_claims = [(c, extract_citations(c)) for c in claims if extract_citations(c)]
    if not cited_claims:
        return MetricResult(
            name="claim_entailment", value=1.0, skipped=True, details={"cited_claims": 0}
        )
    supported = 0
    for claim, refs in cited_claims:
        evidence = [by_ref[r] for r in refs if r in by_ref]
        if evidence and _supported_strict(claim, evidence):
            supported += 1
    return MetricResult(
        name="claim_entailment",
        value=round(supported / len(cited_claims), 4),
        details={"cited_claims": len(cited_claims), "supported": supported},
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
        return MetricResult(name="context_recall", value=1.0, skipped=True)
    covered = sum(1 for fact in facts if _supported(str(fact), run.evidence, threshold=0.25))
    return MetricResult(
        name="context_recall",
        value=round(covered / len(facts), 4),
        details={"facts": len(facts), "covered": covered},
    )


# -- quality & safety metrics -----------------------------------------------


def _claims(text: str, *, min_words: int = 3) -> list[str]:
    return [s for s in split_sentences(text) if len(s.split()) >= min_words]


def _reference_evidence(case: EvalCase, run: RunOutput) -> list[EvidenceItem]:
    """Evidence to judge against: the run's evidence, or reference context
    supplied on the case (``context['reference']`` — str or list of str)."""
    if run.evidence:
        return run.evidence
    reference = case.context.get("reference") or case.context.get("source")
    if reference is None:
        return []
    texts = reference if isinstance(reference, list) else [reference]
    return [
        EvidenceItem(id=f"ref_{i}", source_id="case_reference", text=str(t))
        for i, t in enumerate(texts)
    ]


@register_metric("faithfulness")
def faithfulness(case: EvalCase, run: RunOutput) -> MetricResult:
    """Ragas-style faithfulness: fraction of answer claims attributable to the
    retrieved/reference context. 1.0 = every claim supported."""
    claims = _claims(run.output_text)
    if not claims:
        return MetricResult(name="faithfulness", value=1.0, skipped=True, details={"claims": 0})
    evidence = _reference_evidence(case, run)
    supported: list[str] = []
    unsupported: list[str] = []
    for claim in claims:
        (supported if _supported(claim, evidence) else unsupported).append(claim)
    value = len(supported) / len(claims)
    return MetricResult(
        name="faithfulness",
        value=round(value, 4),
        details={
            "claims": len(claims),
            "supported": len(supported),
            "unsupported": [c[:120] for c in unsupported[:5]],
        },
    )


@register_metric("answer_relevance")
def answer_relevance(case: EvalCase, run: RunOutput) -> MetricResult:
    """How directly the answer addresses the question. Penalizes evasive or
    noncommittal answers; uses lexical similarity offline (an LLM/embedding
    judge can replace it via judges)."""
    answer = run.output_text
    if not answer.strip():
        return MetricResult(name="answer_relevance", value=0.0, details={"empty": True})
    question = case.input_text
    similarity = lexical_similarity(question, answer)
    sentences = split_sentences(answer) or [answer]
    covered = sum(1 for s in sentences if lexical_similarity(question, s) >= 0.1)
    coverage = covered / len(sentences)
    noncommittal = bool(
        re.search(r"\b(i don't know|cannot answer|not sure|no idea|unable to)\b", answer.lower())
    )
    value = min(1.0, 0.5 * coverage + 0.5 * min(1.0, similarity * 3))
    if noncommittal:
        value *= 0.25
    return MetricResult(
        name="answer_relevance",
        value=round(value, 4),
        details={
            "similarity": round(similarity, 4),
            "on_topic_sentences": covered,
            "sentences": len(sentences),
            "noncommittal": noncommittal,
        },
    )


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_CITATION_MARKER_RE = re.compile(r"\[[^\]]{1,60}\]")


def _number_entailed(number: str, evidence_numbers: set[str]) -> bool:
    """Whole-token equality, or a dotted-component prefix of an evidence numeral.

    "24" is entailed by "24.11.0" and "3.14" by "3.14.2" (a shorter version line
    is a true statement about a longer one), but "30" never matches "130" and
    "24" never matches "240" — components are compared whole, never substrings.
    """
    if number in evidence_numbers:
        return True
    parts = number.split(".")
    return any(
        candidate.split(".")[: len(parts)] == parts
        for candidate in evidence_numbers
        if "." in candidate
    )


def _supported_strict(claim: str, evidence: list[EvidenceItem], threshold: float = 0.28) -> bool:
    """Lexical support that also requires every number in the claim to appear
    in the supporting evidence — catches numeric contradictions ("90 days"
    against evidence saying "30 days") that bag-of-words similarity misses.
    A short claim checked against a long document is entailment, not
    near-duplication, so support is symmetric similarity **or** strong
    claim-term containment (most of the claim's content terms appear in the
    evidence). Numbers are compared as whole tokens ("30" does not match
    "130"), with one entailment allowance: a dotted-version prefix ("24"
    against "24.11.0") is supported. Citation markers like ``[D1:C0]`` are
    stripped so their ids don't count as numbers."""
    claim = _CITATION_MARKER_RE.sub("", claim)
    numbers = _NUMBER_RE.findall(claim)
    return any(
        (
            lexical_similarity(claim, item.text) >= threshold
            or containment_similarity(claim, item.text) >= 0.55
        )
        and all(
            _number_entailed(number, set(_NUMBER_RE.findall(item.text)))
            for number in set(numbers)
        )
        for item in evidence
        if item.text
    )


@register_metric("hallucination")
def hallucination(case: EvalCase, run: RunOutput) -> MetricResult:
    """Fraction of verifiable claims NOT supported by the context (lower is
    better). The inverse view of faithfulness, restricted to claims that
    assert checkable facts, with strict number checking."""
    claims = _verifiable_claims(run.output_text)
    if not claims:
        return MetricResult(name="hallucination", value=0.0, passed=True, details={"claims": 0})
    evidence = _reference_evidence(case, run)
    unsupported = [c for c in claims if not _supported_strict(c, evidence)]
    value = len(unsupported) / len(claims)
    return MetricResult(
        name="hallucination",
        value=round(value, 4),
        passed=value == 0.0,
        details={"claims": len(claims), "unsupported": [c[:120] for c in unsupported[:5]]},
    )


_TOXIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "insult",
        re.compile(r"\b(idiot|moron|stupid|dumb|pathetic|worthless|loser|trash)\b", re.IGNORECASE),
    ),
    ("profanity", re.compile(r"\b(damn|hell|crap|screw you|shut up)\b", re.IGNORECASE)),
    (
        "threat",
        re.compile(r"\b(kill|hurt|destroy|attack)\s+(you|yourself|them|him|her)\b", re.IGNORECASE),
    ),
    ("hate", re.compile(r"\b(i hate you|you people|go back to|disgusting)\b", re.IGNORECASE)),
    ("dismissal", re.compile(r"\b(nobody cares|waste of (space|time|air))\b", re.IGNORECASE)),
]


def _pattern_rate(
    name: str, patterns: list[tuple[str, re.Pattern[str]]], text: str
) -> MetricResult:
    """Fraction of sentences matching any pattern (first match per sentence)."""
    sentences = split_sentences(text) or [text]
    hits: list[dict[str, str]] = []
    for sentence in sentences:
        for kind, pattern in patterns:
            if pattern.search(sentence):
                hits.append({"kind": kind, "excerpt": sentence[:120]})
                break
    value = len(hits) / len(sentences) if sentences else 0.0
    return MetricResult(
        name=name,
        value=round(value, 4),
        passed=value == 0.0,
        details={"sentences": len(sentences), "hits": hits[:5]},
    )


@register_metric("toxicity")
def toxicity(case: EvalCase, run: RunOutput) -> MetricResult:
    """Heuristic toxicity rate: fraction of sentences with toxic language
    (lower is better). Pattern-based and offline; pair with a model judge for
    nuanced cases."""
    return _pattern_rate("toxicity", _TOXIC_PATTERNS, run.output_text)


_BIAS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "sweeping_generalization",
        re.compile(
            r"\b(all|every|no)\s+(women|men|girls|boys|americans|europeans|asians|africans|"
            r"immigrants|millennials|boomers|engineers|managers|old people|young people)\s+(are|do|can'?t|never|always)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "stereotype_cue",
        re.compile(
            r"\b(typical|naturally|obviously|of course)\s+(for\s+)?(a\s+)?(woman|man|girl|boy|foreigner|immigrant)\b",
            re.IGNORECASE,
        ),
    ),
    ("group_othering", re.compile(r"\b(those people|their kind|that group)\b", re.IGNORECASE)),
    (
        "gendered_assumption",
        re.compile(
            r"\b(women|men)\s+(belong|should stay|are too|aren'?t suited|can'?t handle)\b",
            re.IGNORECASE,
        ),
    ),
]


@register_metric("bias")
def bias(case: EvalCase, run: RunOutput) -> MetricResult:
    """Heuristic bias rate: fraction of sentences with stereotyping or
    sweeping group generalizations (lower is better)."""
    return _pattern_rate("bias", _BIAS_PATTERNS, run.output_text)


@register_metric("summarization_quality")
def summarization_quality(case: EvalCase, run: RunOutput) -> MetricResult:
    """min(coverage, faithfulness) of a summary against its source — a
    summary must cover the source's key content without inventing any.
    Source text comes from ``context['source']`` or the run's evidence."""
    source = case.context.get("source") or " ".join(e.text or "" for e in run.evidence)
    summary = run.output_text
    if not str(source).strip() or not summary.strip():
        return MetricResult(
            name="summarization_quality", value=0.0, details={"missing": "source or summary"}
        )
    source_items = [EvidenceItem(id="src", source_id="summary_source", text=str(source))]
    key_sentences = (
        sorted(_claims(str(source), min_words=5), key=lambda s: len(s.split()), reverse=True)[:8]
        or _claims(str(source))[:8]
    )
    covered = sum(1 for s in key_sentences if lexical_similarity(s, summary) >= 0.2)
    coverage = covered / len(key_sentences) if key_sentences else 0.0
    summary_claims = _claims(summary)
    supported = sum(1 for c in summary_claims if _supported(c, source_items, threshold=0.25))
    faithful = supported / len(summary_claims) if summary_claims else 1.0
    compression = 1.0 - min(1.0, len(summary) / max(1, len(str(source))))
    value = min(coverage, faithful)
    return MetricResult(
        name="summarization_quality",
        value=round(value, 4),
        details={
            "coverage": round(coverage, 4),
            "faithfulness": round(faithful, 4),
            "compression": round(compression, 4),
        },
    )


# -- conversational metrics --------------------------------------------------


def _message_text(content: Any) -> str:
    """Coerce message content to text; supports content-block lists."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content or "")


def _conversation(case: EvalCase) -> list[dict[str, str]]:
    messages = case.context.get("messages") or case.context.get("conversation") or []
    return [
        {"role": str(m.get("role", "")), "content": _message_text(m.get("content"))}
        for m in messages
        if isinstance(m, dict) and m.get("content")
    ]


@register_metric("knowledge_retention")
def knowledge_retention(case: EvalCase, run: RunOutput) -> MetricResult:
    """Whether the output forgets facts the user already stated in the
    session: re-asking for a stated fact is a violation. Facts come from
    ``rubric['session_facts']`` or verifiable claims in prior user turns."""
    facts = [str(f) for f in case.rubric.get("session_facts", [])]
    if not facts:
        for message in _conversation(case):
            if message.get("role") == "user":
                facts.extend(_verifiable_claims(message["content"]))
    if not facts:
        return MetricResult(
            name="knowledge_retention", value=1.0, skipped=True, details={"facts": 0}
        )
    questions = [s for s in split_sentences(run.output_text) if s.rstrip().endswith("?")]
    violations = [
        {"fact": fact[:120], "question": q[:120]}
        for fact in facts
        for q in questions
        if lexical_similarity(fact, q) >= 0.25
    ]
    value = max(0.0, 1.0 - len(violations) / len(facts))
    return MetricResult(
        name="knowledge_retention",
        value=round(value, 4),
        details={"facts": len(facts), "violations": violations[:5]},
    )


@register_metric("conversation_relevance")
def conversation_relevance(case: EvalCase, run: RunOutput) -> MetricResult:
    """Relevance of the output to the latest user turn, with the wider
    session as fallback context (sliding window)."""
    messages = _conversation(case)
    user_turns = [m["content"] for m in messages if m.get("role") == "user"]
    last_turn = user_turns[-1] if user_turns else case.input_text
    window = " ".join(m["content"] for m in messages[-6:]) or case.input_text
    direct = lexical_similarity(last_turn, run.output_text)
    windowed = lexical_similarity(window, run.output_text)
    value = min(1.0, max(direct, windowed) * 3)
    return MetricResult(
        name="conversation_relevance",
        value=round(value, 4),
        details={
            "direct": round(direct, 4),
            "windowed": round(windowed, 4),
            "turns": len(messages),
        },
    )


@register_metric("conversation_outcome")
def conversation_outcome(case: EvalCase, run: RunOutput) -> MetricResult:
    """Whether the multi-turn conversation achieved the user's goal. The goal
    comes from ``rubric['goal']`` / ``rubric['goal_keywords']`` /
    ``context['goal']`` and is measured against the assistant's turns plus the
    final output. The intent-and-outcome view that single-turn relevance misses."""
    goal = str(case.rubric.get("goal") or case.context.get("goal") or "")
    keywords = [str(k) for k in case.rubric.get("goal_keywords", [])]
    messages = _conversation(case)
    assistant_text = " ".join(m["content"] for m in messages if m.get("role") == "assistant")
    assistant_text = f"{assistant_text} {run.output_text}".strip()
    if not goal and not keywords:
        return MetricResult(
            name="conversation_outcome", value=1.0, skipped=True, details={"goal": None}
        )
    if keywords:
        hit = sum(1 for k in keywords if k.lower() in assistant_text.lower())
        value = hit / len(keywords)
        details: dict[str, Any] = {"keywords": len(keywords), "matched": hit}
    else:
        similarity = lexical_similarity(goal, assistant_text)
        value = min(1.0, similarity * 3)
        details = {"similarity": round(similarity, 4)}
    return MetricResult(
        name="conversation_outcome", value=round(value, 4), passed=value >= 0.5, details=details
    )


@register_metric("intent_resolution")
def intent_resolution(case: EvalCase, run: RunOutput) -> MetricResult:
    """Fraction of user intents (turns) the assistant addressed: every user turn
    should be followed by a relevant assistant reply (the last turn falls back to
    the run output)."""
    messages = _conversation(case)
    if not messages:
        similarity = lexical_similarity(case.input_text, run.output_text)
        return MetricResult(
            name="intent_resolution", value=round(min(1.0, similarity * 3), 4), details={"turns": 0}
        )
    intents = 0
    resolved = 0
    for i, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        intents += 1
        reply = next(
            (
                messages[j]["content"]
                for j in range(i + 1, len(messages))
                if messages[j].get("role") == "assistant"
            ),
            run.output_text,
        )
        if lexical_similarity(message["content"], reply) >= 0.1:
            resolved += 1
    value = resolved / intents if intents else 1.0
    return MetricResult(
        name="intent_resolution",
        value=round(value, 4),
        details={"intents": intents, "resolved": resolved},
    )


# -- trajectory & tool-use metrics ------------------------------------------
#
# These read the agent ``trajectory`` carried on the RunOutput (built from an
# AgentState / CrewResult / Trace via ``RunOutput.from_*``). They evaluate *how*
# a run reached its answer — not just the final text — so a crew or StateGraph
# run is scored without re-instrumentation. Expected/optimal references live on
# the case (``rubric['expected_tools' | 'plan' | 'optimal_steps' | 'topic']``).
# When no trajectory is present they return a neutral 1.0 so they can sit
# alongside output-only metrics in the same report.


def _normalize_tool(value: Any) -> tuple[str, frozenset[tuple[str, str]]]:
    """Normalize an expected tool spec (a name, or ``{tool, arguments}``)."""
    if isinstance(value, dict):
        name = value.get("tool") or value.get("name") or value.get("tool_name") or ""
        args = value.get("arguments") or value.get("args") or {}
    else:
        name, args = value, {}
    arg_sig = frozenset((str(k), _normalize(str(v))) for k, v in (args or {}).items())
    return _normalize(str(name)), arg_sig


def _expected_tools(case: EvalCase) -> list[Any]:
    spec = case.rubric.get("expected_tools")
    if spec is None and isinstance(case.expected, dict):
        spec = case.expected.get("tool_calls") or case.expected.get("expected_tools")
    return list(spec or [])


def _actual_tool_sig(step: Any) -> tuple[str, frozenset[tuple[str, str]]]:
    name = _normalize(step.tool_name or step.name)
    args = frozenset((str(k), _normalize(str(v))) for k, v in (step.tool_arguments or {}).items())
    return name, args


def _step_tokens(step: Any) -> set[str]:
    return {_normalize(t) for t in (step.type, step.name, step.tool_name or "") if t}


def _lcs_against_tokens(expected: list[str], token_sets: list[set[str]]) -> int:
    """Longest common subsequence length matching each expected token against a
    step's token set (type / name / tool_name)."""
    n, m = len(expected), len(token_sets)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected[i - 1] in token_sets[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


@register_metric("tool_call_accuracy")
def tool_call_accuracy(case: EvalCase, run: RunOutput) -> MetricResult:
    """Right tool, right args, in the right order. Expected calls come from
    ``rubric['expected_tools']`` (names or ``{tool, arguments}``); positional
    match, args required only when the reference supplies them. With no
    reference, falls back to the fraction of executed tool calls that succeeded."""
    traj = run.trajectory
    actual = traj.tool_calls() if traj else []
    expected = _expected_tools(case)
    if not expected:
        if not actual:
            return MetricResult(
                name="tool_call_accuracy", value=1.0, skipped=True, details={"tools": 0}
            )
        ok = sum(1 for s in actual if s.ok)
        return MetricResult(
            name="tool_call_accuracy",
            value=round(ok / len(actual), 4),
            passed=ok == len(actual),
            details={"tools": len(actual), "ok": ok, "mode": "success_rate"},
        )
    exp_norm = [_normalize_tool(e) for e in expected]
    act_norm = [_actual_tool_sig(s) for s in actual]
    correct = 0
    for i, (name, args) in enumerate(exp_norm):
        if i < len(act_norm):
            act_name, act_args = act_norm[i]
            if act_name == name and (not args or args <= act_args):
                correct += 1
    value = correct / len(exp_norm)
    return MetricResult(
        name="tool_call_accuracy",
        value=round(value, 4),
        passed=value == 1.0,
        details={"expected": len(exp_norm), "correct": correct, "actual": len(act_norm)},
    )


@register_metric("tool_call_f1")
def tool_call_f1(case: EvalCase, run: RunOutput) -> MetricResult:
    """Order-insensitive F1 over tool *names* (multiset) against
    ``rubric['expected_tools']`` — catches missing and spurious tool calls that
    a positional score hides. Returns 1.0 when no reference is given."""
    traj = run.trajectory
    actual = traj.tool_calls() if traj else []
    expected = _expected_tools(case)
    if not expected:
        return MetricResult(
            name="tool_call_f1", value=1.0, skipped=True, details={"reference": None}
        )
    exp = Counter(_normalize_tool(e)[0] for e in expected)
    act = Counter(_normalize(s.tool_name or s.name) for s in actual)
    true_positive = sum((exp & act).values())
    precision = true_positive / sum(act.values()) if act else 0.0
    recall = true_positive / sum(exp.values()) if exp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return MetricResult(
        name="tool_call_f1",
        value=round(f1, 4),
        details={"precision": round(precision, 4), "recall": round(recall, 4), "tp": true_positive},
    )


@register_metric("goal_accuracy")
def goal_accuracy(case: EvalCase, run: RunOutput) -> MetricResult:
    """Did the run achieve its objective? Successful termination, combined with a
    match of the final answer to ``case.expected`` when one is given (half weight
    each), so a run that 'finishes' with the wrong answer does not pass."""
    traj = run.trajectory
    success = bool(traj.success) if traj else (run.error is None)
    expected = _expected_text(case)
    if not expected:
        value = 1.0 if success else 0.0
        return MetricResult(
            name="goal_accuracy",
            value=value,
            passed=success,
            details={"success": success, "expected": False},
        )
    similarity = lexical_similarity(expected, run.output_text)
    answer_match = similarity >= 0.5
    value = 0.5 * (1.0 if success else 0.0) + 0.5 * (1.0 if answer_match else 0.0)
    return MetricResult(
        name="goal_accuracy",
        value=round(value, 4),
        passed=value == 1.0,
        details={
            "success": success,
            "answer_match": answer_match,
            "similarity": round(similarity, 4),
        },
    )


@register_metric("plan_adherence")
def plan_adherence(case: EvalCase, run: RunOutput) -> MetricResult:
    """How closely the executed steps follow the expected plan, as
    LCS-length / plan-length. The plan is ``rubric['plan']`` /
    ``rubric['expected_steps']`` (step types, names, or tool names). Returns 1.0
    with no reference plan."""
    traj = run.trajectory
    plan = case.rubric.get("plan") or case.rubric.get("expected_steps")
    if not plan or not traj or not traj.steps:
        return MetricResult(
            name="plan_adherence", value=1.0, skipped=True, details={"plan": bool(plan)}
        )
    expected = [_normalize(str(p)) for p in plan]
    token_sets = [_step_tokens(s) for s in traj.steps]
    matched = _lcs_against_tokens(expected, token_sets)
    value = matched / len(expected)
    return MetricResult(
        name="plan_adherence",
        value=round(value, 4),
        passed=value == 1.0,
        details={"plan_steps": len(expected), "matched": matched, "actual_steps": len(traj.steps)},
    )


def _step_signature(step: Any) -> tuple[Any, ...]:
    return (
        step.type,
        step.tool_name,
        tuple(sorted((str(k), str(v)) for k, v in (step.tool_arguments or {}).items())),
    )


_FAILED_STATUSES = {"failed", "error", "denied", "timeout"}


@register_metric("plan_quality")
def plan_quality(case: EvalCase, run: RunOutput) -> MetricResult:
    """Structural quality of the executed plan (reference-free): penalizes failed
    steps and redundant back-to-back repeats of the same step/tool/args."""
    traj = run.trajectory
    if not traj or not traj.steps:
        return MetricResult(name="plan_quality", value=1.0, skipped=True, details={"steps": 0})
    steps = traj.steps
    failed = sum(1 for s in steps if s.status in _FAILED_STATUSES)
    redundant = 0
    previous: tuple[Any, ...] | None = None
    for step in steps:
        signature = _step_signature(step)
        if signature == previous:
            redundant += 1
        previous = signature
    value = max(0.0, 1.0 - (failed + redundant) / len(steps))
    return MetricResult(
        name="plan_quality",
        value=round(value, 4),
        passed=value >= 0.999,
        details={"steps": len(steps), "failed": failed, "redundant": redundant},
    )


@register_metric("step_efficiency")
def step_efficiency(case: EvalCase, run: RunOutput) -> MetricResult:
    """Steps taken vs an optimal path (``rubric['optimal_steps']``), capped at
    1.0. Without an optimum, the fraction of non-redundant, successful steps —
    rewarding the shortest correct path."""
    traj = run.trajectory
    if not traj or not traj.steps:
        return MetricResult(name="step_efficiency", value=1.0, skipped=True, details={"steps": 0})
    steps = traj.steps
    optimal = case.rubric.get("optimal_steps")
    if optimal:
        value = min(1.0, float(optimal) / max(1, len(steps)))
        return MetricResult(
            name="step_efficiency",
            value=round(value, 4),
            passed=len(steps) <= optimal,
            details={"steps": len(steps), "optimal": optimal},
        )
    useful = 0
    previous: tuple[Any, ...] | None = None
    for step in steps:
        signature = _step_signature(step)
        if signature != previous and step.status not in _FAILED_STATUSES:
            useful += 1
        previous = signature
    return MetricResult(
        name="step_efficiency",
        value=round(useful / len(steps), 4),
        details={"steps": len(steps), "useful": useful},
    )


@register_metric("topic_adherence")
def topic_adherence(case: EvalCase, run: RunOutput) -> MetricResult:
    """Whether the agent's steps stay on the objective's topic. Each step's text
    is compared to the objective (``rubric['topic']`` / trajectory objective /
    input). Lexical and offline; a model judge can replace it via judges."""
    traj = run.trajectory
    objective = str(case.rubric.get("topic") or (traj.objective if traj else "") or case.input_text)
    if not traj or not traj.steps or not objective:
        return MetricResult(name="topic_adherence", value=1.0, skipped=True, details={"steps": 0})
    on_topic = 0
    for step in traj.steps:
        text = step.text
        if not text or lexical_similarity(objective, text) >= 0.06:
            on_topic += 1
    return MetricResult(
        name="topic_adherence",
        value=round(on_topic / len(traj.steps), 4),
        details={"steps": len(traj.steps), "on_topic": on_topic},
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
        return MetricResult(name="recall_at_k", value=1.0, skipped=True)
    retrieved = [e.id for e in run.evidence] + [e.citation_ref for e in run.evidence]
    hit = sum(1 for ref in relevant if ref in retrieved)
    return MetricResult(name="recall_at_k", value=round(hit / len(relevant), 4))


@register_metric("precision_at_k")
def precision_at_k(case: EvalCase, run: RunOutput) -> MetricResult:
    relevant = _relevant_ids(case)
    if not relevant:
        return MetricResult(name="precision_at_k", value=1.0, skipped=True)
    if not run.evidence:
        return MetricResult(name="precision_at_k", value=0.0)
    hits = sum(1 for e in run.evidence if e.id in relevant or e.citation_ref in relevant)
    return MetricResult(name="precision_at_k", value=round(hits / len(run.evidence), 4))


@register_metric("mrr")
def mrr(case: EvalCase, run: RunOutput) -> MetricResult:
    relevant = _relevant_ids(case)
    if not relevant:
        return MetricResult(name="mrr", value=1.0, skipped=True)
    for rank, item in enumerate(run.evidence, start=1):
        if item.id in relevant or item.citation_ref in relevant:
            return MetricResult(name="mrr", value=round(1.0 / rank, 4))
    return MetricResult(name="mrr", value=0.0)


@register_metric("ndcg")
def ndcg(case: EvalCase, run: RunOutput) -> MetricResult:
    relevant = _relevant_ids(case)
    if not relevant:
        return MetricResult(name="ndcg", value=1.0, skipped=True)
    dcg = 0.0
    for rank, item in enumerate(run.evidence, start=1):
        if item.id in relevant or item.citation_ref in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), len(run.evidence)) + 1)
    )
    return MetricResult(name="ndcg", value=round(dcg / ideal, 4) if ideal else 0.0)
