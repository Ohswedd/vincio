"""Core data contracts shared across all Vincio subsystems.

These are the public, stable Pydantic models (Core Concepts),
§13 (input), §27 (providers), plus run-level types. Subsystem-specific models
live in their own packages and import from here.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .utils import new_id, stable_hash, utcnow

__all__ = [
    "TaskType",
    "TrustLevel",
    "PrivacyClass",
    "Objective",
    "FileRef",
    "ImageRef",
    "AudioRef",
    "VideoRef",
    "UserInput",
    "Budget",
    "BudgetUsage",
    "Instruction",
    "Constraint",
    "Example",
    "EvidenceItem",
    "EvidenceModality",
    "MemoryScope",
    "MemoryType",
    "MemoryItem",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "PolicySet",
    "Document",
    "Chunk",
    "TokenUsage",
    "MessageRole",
    "ContentPart",
    "Message",
    "ToolCallRequest",
    "ModelRequest",
    "ModelResponse",
    "ModelEvent",
    "ModelCapabilities",
    "ModelProfile",
    "ModelLifecycle",
    "RunStatus",
    "RunConfig",
    "RunResult",
    "RunStreamEvent",
]


# ---------------------------------------------------------------------------
# Enums / literals
# ---------------------------------------------------------------------------


class TaskType(StrEnum):
    """Task taxonomy used by the input router."""

    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    DOCUMENT_QA = "document_qa"
    DOCUMENT_COMPARISON = "document_comparison"
    DATA_ANALYSIS = "data_analysis"
    TOOL_ACTION = "tool_action"
    AGENT_WORKFLOW = "agent_workflow"
    PLANNING = "planning"
    CODING = "coding"
    CREATIVE_GENERATION = "creative_generation"
    COMPLIANCE_REVIEW = "compliance_review"
    GENERAL = "general"


class TrustLevel(StrEnum):
    """Trust tags for instruction/data separation."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    UNTRUSTED_DOCUMENT = "untrusted_document"
    UNTRUSTED_TOOL = "untrusted_tool"
    UNTRUSTED_EXTERNAL = "untrusted_external"

    @property
    def allowed_to_instruct_model(self) -> bool:
        return self in (TrustLevel.SYSTEM, TrustLevel.DEVELOPER, TrustLevel.USER)


class PrivacyClass(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PII = "pii"
    SENSITIVE = "sensitive"  # health, financial, government ids


# ---------------------------------------------------------------------------
# Objective / input
# ---------------------------------------------------------------------------


class Objective(BaseModel):
    """What the application is trying to accomplish."""

    id: str = Field(default_factory=lambda: new_id("obj"))
    text: str
    task_type: TaskType = TaskType.GENERAL
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, text: str | None = None, **data: Any) -> None:
        # Allow positional construction: Objective("Review contracts")
        if text is not None:
            data.setdefault("text", text)
        super().__init__(**data)


class FileRef(BaseModel):
    path: str
    name: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageRef(BaseModel):
    path: str | None = None
    url: str | None = None
    media_type: str | None = "image/png"
    detail: Literal["low", "high", "auto"] = "auto"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioRef(BaseModel):
    path: str | None = None
    url: str | None = None
    media_type: str | None = "audio/wav"
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoRef(BaseModel):
    """A reference to a video clip — first-class beside :class:`ImageRef` and
    :class:`AudioRef`.

    ``duration_seconds`` and ``fps`` describe the timeline so deterministic
    frame sampling and temporal segmentation can address the clip without
    decoding it; ``detail`` drives the modality-aware token budget the same way
    it does for images. ``metadata`` carries the scorable surrogate (a
    ``transcript`` / ``caption``) and any sampled-frame references.
    """

    path: str | None = None
    url: str | None = None
    media_type: str | None = "video/mp4"
    duration_seconds: float | None = None
    fps: float | None = None
    detail: Literal["low", "high", "auto"] = "auto"
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserInput(BaseModel):
    """Structured task input."""

    text: str | None = None
    files: list[FileRef] = Field(default_factory=list)
    images: list[ImageRef] = Field(default_factory=list)
    audio: list[AudioRef] = Field(default_factory=list)
    video: list[VideoRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    feature: str | None = None  # product feature/surface for cost attribution
    locale: str | None = None


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


class Budget(BaseModel):
    """Hard resource limits for a run (budgets, termination)."""

    max_input_tokens: int = 100_000
    max_output_tokens: int = 4_096
    max_latency_ms: int = 120_000
    max_cost_usd: float = 1.0
    max_steps: int = 16
    max_tool_calls: int = 32
    max_retries: int = 2

    def scaled(self, fraction: float) -> Budget:
        """A proportional sub-budget (e.g. per agent step)."""
        return Budget(
            max_input_tokens=max(1, int(self.max_input_tokens * fraction)),
            max_output_tokens=max(1, int(self.max_output_tokens * fraction)),
            max_latency_ms=max(1, int(self.max_latency_ms * fraction)),
            max_cost_usd=self.max_cost_usd * fraction,
            max_steps=self.max_steps,
            max_tool_calls=self.max_tool_calls,
            max_retries=self.max_retries,
        )


class BudgetUsage(BaseModel):
    """Running totals checked against a :class:`Budget`."""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    steps: int = 0
    tool_calls: int = 0
    retries: int = 0

    def add(self, other: BudgetUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.latency_ms += other.latency_ms
        self.cost_usd += other.cost_usd
        self.steps += other.steps
        self.tool_calls += other.tool_calls
        self.retries += other.retries

    def exceeds(self, budget: Budget) -> list[str]:
        """Names of exhausted budget dimensions (empty when within budget)."""
        breaches: list[str] = []
        if self.input_tokens > budget.max_input_tokens:
            breaches.append("input_tokens")
        if self.output_tokens > budget.max_output_tokens * max(1, self.steps or 1):
            breaches.append("output_tokens")
        if self.latency_ms > budget.max_latency_ms:
            breaches.append("latency_ms")
        if self.cost_usd > budget.max_cost_usd:
            breaches.append("cost_usd")
        if self.steps > budget.max_steps:
            breaches.append("steps")
        if self.tool_calls > budget.max_tool_calls:
            breaches.append("tool_calls")
        return breaches


# ---------------------------------------------------------------------------
# Prompt building blocks
# ---------------------------------------------------------------------------


class Instruction(BaseModel):
    text: str
    priority: int = 100  # lower sorts first
    category: Literal["role", "objective", "rule", "definition", "safety", "format", "other"] = (
        "rule"
    )
    source: str | None = None

    def __init__(self, text: str | None = None, **data: Any) -> None:
        if text is not None:
            data.setdefault("text", text)
        super().__init__(**data)


class Constraint(BaseModel):
    text: str
    hard: bool = True
    source: str | None = None

    def __init__(self, text: str | None = None, **data: Any) -> None:
        if text is not None:
            data.setdefault("text", text)
        super().__init__(**data)


class Example(BaseModel):
    input: str
    output: str
    explanation: str | None = None
    tags: list[str] = Field(default_factory=list)
    quality: float = 1.0


# ---------------------------------------------------------------------------
# Evidence / memory / tools
# ---------------------------------------------------------------------------


# the modality of an evidence unit. The compiler selects, dedupes, orders,
# budgets, and cites image, table, and video evidence as first-class candidates
# alongside text — not as observations bolted on after the fact.
EvidenceModality = Literal["text", "image", "table", "video"]

# Representative token cost of an image part by requested detail (calibrated to
# the common vision pricing tiers), and per-cell cost for a structured table.
_IMAGE_TOKEN_COST = {"low": 85, "high": 765, "auto": 512}
_TABLE_TOKEN_PER_CELL = 3
# A video segment ships several sampled frames plus its transcript, so its
# floor token cost is a small multiple of a single image's by requested detail.
_VIDEO_TOKEN_COST = {"low": 256, "high": 2048, "auto": 1024}


def _fmt_seconds(value: float) -> str:
    """Compact fixed-point seconds for a temporal citation ref (``12``/``12.5``)."""
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


class EvidenceItem(BaseModel):
    """A provenance-aware unit of evidence (text, image, or table)."""

    id: str = Field(default_factory=lambda: new_id("ev"))
    source_id: str
    source_type: Literal["document", "memory", "tool", "database", "image", "audio", "web"] = (
        "document"
    )
    # typed modality + structured carriers. ``text`` remains the scorable
    # surrogate for every modality (the text for text evidence; a caption / OCR /
    # alt text for an image; a Markdown rendering for a table), so relevance,
    # dedup, ordering, and citation work uniformly. ``image`` / ``table`` carry
    # the actual non-text payload the renderer ships to a vision model.
    modality: EvidenceModality = "text"
    text: str | None = None
    image: ImageRef | None = None
    table: dict[str, Any] | None = None  # {"columns": [...], "rows": [[...]], "markdown": "..."}
    video: VideoRef | None = None
    media_ref: str | None = None
    page: int | None = None
    span: tuple[int, int] | None = None
    # Temporal locator (start, end) in seconds for video / audio evidence, so a
    # claim grounds to a moment in a clip the way ``page`` grounds it to a page.
    # Preserved through to the citation (``citation_ref`` renders ``:t12-18.5``).
    time_range: tuple[float, float] | None = None
    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    trust_level: TrustLevel = TrustLevel.UNTRUSTED_DOCUMENT
    relevance: float = 0.0
    authority: float = 0.5
    freshness: float = 0.5
    provenance: float = 0.5
    token_cost: int = 0

    @property
    def citation_ref(self) -> str:
        """Stable reference for citations, e.g. ``D1:p4`` or, for a clip,
        ``D1:t12-18.5`` (a time range in seconds) style."""
        if self.time_range is not None:
            start, end = self.time_range
            return f"{self.source_id}:t{_fmt_seconds(start)}-{_fmt_seconds(end)}"
        if self.page is not None:
            return f"{self.source_id}:p{self.page}"
        return self.id

    @property
    def scorable_text(self) -> str:
        """The text used for relevance/dedup/ordering across every modality."""
        if self.text:
            return self.text
        if self.modality == "table" and self.table:
            return str(self.table.get("markdown") or self.table.get("caption") or "")
        if self.modality == "image" and self.image is not None:
            return str(self.image.metadata.get("caption") or self.image.metadata.get("alt") or "")
        if self.modality == "video" and self.video is not None:
            return str(
                self.video.metadata.get("transcript") or self.video.metadata.get("caption") or ""
            )
        return ""

    def estimated_token_cost(self) -> int:
        """Modality-aware token cost: a calibrated image budget by detail, a
        per-cell table estimate, or the text token count (computed lazily by the
        caller when 0)."""
        if self.modality == "image" and self.image is not None:
            return _IMAGE_TOKEN_COST.get(self.image.detail, _IMAGE_TOKEN_COST["auto"])
        if self.modality == "table" and self.table:
            rows = self.table.get("rows") or []
            cols = self.table.get("columns") or []
            cells = sum(len(r) for r in rows) if rows else 0
            return max(self.token_cost, _TABLE_TOKEN_PER_CELL * (cells + len(cols)))
        if self.modality == "video" and self.video is not None:
            return _VIDEO_TOKEN_COST.get(self.video.detail, _VIDEO_TOKEN_COST["auto"])
        return self.token_cost


class MemoryScope(StrEnum):
    SESSION = "session"
    USER = "user"
    AGENT = "agent"
    TEAM = "team"  # shared across a team, gated by per-memory ACL
    TENANT = "tenant"
    ORGANIZATION = "organization"
    GLOBAL = "global"


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    DECISION = "decision"
    SUMMARY = "summary"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"


class MemoryItem(BaseModel):
    """A scoped, scored, decaying memory."""

    id: str = Field(default_factory=lambda: new_id("mem"))
    scope: MemoryScope = MemoryScope.USER
    type: MemoryType = MemoryType.FACT
    content: str
    owner_id: str | None = None  # user/tenant/org id the scope binds to
    confidence: float = 0.8
    source_trace_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    privacy_class: PrivacyClass = PrivacyClass.INTERNAL
    status: Literal["candidate", "validated", "active", "decayed", "archived", "deleted"] = (
        "active"
    )
    entities: list[str] = Field(default_factory=list)
    supersedes: str | None = None
    usage_count: int = 0
    confirmations: int = 0
    # Bi-temporal validity. ``created_at`` / ``updated_at`` are *transaction*
    # time (when the system learned the fact); ``valid_from`` / ``valid_to`` are
    # *valid* time (when the fact is true in the world). As-of recall answers
    # "what did we believe was true at time T" without mutating history — a
    # corrected fact closes the old item's ``valid_to`` and opens a new one.
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    # Per-memory access-control list for team-shared memory: reader ids
    # (user / agent / team ids) allowed to recall this item. Empty means the
    # scope's own ownership rule governs (the previous behaviour, unchanged).
    acl: list[str] = Field(default_factory=list)
    # GDPR purpose / consent binding: the purpose this memory was collected
    # for and the consent record that authorises retaining it. Recall and erasure
    # consult these so a withdrawn consent or a purpose mismatch is enforceable.
    purpose: str | None = None
    consent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return min(1.0, max(0.0, v))

    def valid_at(self, moment: datetime) -> bool:
        """True when this memory's *valid time* interval contains ``moment``.

        Items with no ``valid_from`` are treated as valid from creation; items
        with no ``valid_to`` are open-ended. Naive datetimes are read as UTC so a
        mixed-tz store never raises mid-recall.
        """

        def _aware(dt: datetime) -> datetime:
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

        moment = _aware(moment)
        start = _aware(self.valid_from) if self.valid_from is not None else _aware(self.created_at)
        if moment < start:
            return False
        if self.valid_to is not None and moment >= _aware(self.valid_to):
            return False
        return True

    def readable_by(self, reader: str | None) -> bool:
        """ACL check: an empty ACL is scope-governed (open); a populated ACL
        admits only listed readers (``None`` reader is admitted only to open
        items)."""
        if not self.acl:
            return True
        return reader is not None and reader in self.acl


class ToolSpec(BaseModel):
    """Tool contract."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    permissions: list[str] = Field(default_factory=list)
    side_effects: Literal["none", "read", "write", "external"] = "read"
    timeout_ms: int = 30_000
    cost_estimate: float = 0.0
    reliability_score: float = 1.0
    approval_required: bool = False
    cacheable: bool | None = None  # default: read-only tools are cacheable
    idempotent: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_cacheable(self) -> bool:
        if self.cacheable is not None:
            return self.cacheable
        return self.side_effects in ("none", "read")


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: new_id("tc"))
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    requested_by: Literal["model", "planner", "user", "workflow"] = "model"


class ToolResult(BaseModel):
    id: str = Field(default_factory=lambda: new_id("tr"))
    call_id: str
    tool_name: str
    status: Literal["ok", "error", "denied", "timeout", "approval_required"] = "ok"
    output: Any = None
    error: str | None = None
    duration_ms: int = 0
    cached: bool = False
    trust_level: TrustLevel = TrustLevel.UNTRUSTED_TOOL
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class PolicySet(BaseModel):
    """Deterministic per-run policies (policies)."""

    privacy: Literal["open", "tenant_isolated", "user_isolated"] = "tenant_isolated"
    safety: Literal["minimal", "standard", "strict"] = "standard"
    answer_only_from_sources: bool = False
    require_citations: bool = False
    allow_memory_writes: bool = True
    allow_external_tools: bool = True
    redact_pii_in_context: bool = False
    block_untrusted_instructions: bool = True
    retention_days: int | None = None
    custom: dict[str, Any] = Field(default_factory=dict)

    def set(self, name: str, value: Any) -> None:
        if name in type(self).model_fields:
            setattr(self, name, value)
        else:
            self.custom[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        if name in type(self).model_fields:
            return getattr(self, name)
        return self.custom.get(name, default)


# ---------------------------------------------------------------------------
# Documents / chunks
# ---------------------------------------------------------------------------


class Document(BaseModel):
    id: str = Field(default_factory=lambda: new_id("doc"))
    source_uri: str | None = None
    title: str | None = None
    media_type: str = "text/plain"
    text: str = ""
    sections: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    images: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    tenant_id: str | None = None
    permissions: list[str] = Field(default_factory=list)
    trust_level: TrustLevel = TrustLevel.UNTRUSTED_DOCUMENT


class Chunk(BaseModel):
    """Retrieval unit with provenance metadata."""

    id: str = Field(default_factory=lambda: new_id("chk"))
    document_id: str
    text: str
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    token_count: int = 0
    entities: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    source_uri: str | None = None
    permissions: list[str] = Field(default_factory=list)
    tenant_id: str | None = None
    kind: Literal["text", "table", "code", "image_region", "title"] = "text"
    index: int = 0  # position within the document
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def citation_ref(self) -> str:
        return f"{self.document_id}:C{self.index}"


# ---------------------------------------------------------------------------
# Provider messages / requests / responses
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.reasoning_tokens += other.reasoning_tokens


MessageRole = Literal["system", "developer", "user", "assistant", "tool"]


class ContentPart(BaseModel):
    type: Literal["text", "image", "audio", "video", "tool_result"] = "text"
    text: str | None = None
    image: ImageRef | None = None
    audio: AudioRef | None = None
    video: VideoRef | None = None
    tool_call_id: str | None = None
    tool_output: Any = None


class Message(BaseModel):
    role: MessageRole
    content: str | list[ContentPart] = ""
    name: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tool_call_id: str | None = None
    cache_hint: bool = False  # marks the end of a stable, cacheable prefix
    # Provider-cache TTL for the breakpoint at this message (Anthropic
    # cache_control). None falls back to the provider default (5-minute
    # ephemeral); "1h" requests the extended one-hour cache.
    cache_ttl: Literal["5m", "1h"] | None = None

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(p.text or "" for p in self.content if p.type == "text")


class ToolCallRequest(BaseModel):
    id: str = Field(default_factory=lambda: new_id("tcr"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class ModelRequest(BaseModel):
    model: str
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    output_schema_name: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None
    # Unified reasoning control across providers that expose it (OpenAI
    # reasoning models, Anthropic extended/interleaved thinking, Gemini
    # thinking budget). ``reasoning_effort`` is the portable knob; providers
    # that take an explicit thinking-token budget derive it from the effort
    # level unless ``thinking_budget_tokens`` is set. Providers ignore both
    # when the model does not support reasoning.
    reasoning_effort: ReasoningEffort | None = None
    thinking_budget_tokens: int | None = None
    # Provider server-state handle (OpenAI Responses API ``previous_response_id``)
    # so reasoning is preserved across tool calls without resending context.
    previous_response_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def hash(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error"]


class ModelResponse(BaseModel):
    id: str = Field(default_factory=lambda: new_id("resp"))
    model: str = ""
    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    structured: dict[str, Any] | None = None
    finish_reason: FinishReason = "stop"
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    latency_ms: int = 0
    provider: str = ""
    raw: dict[str, Any] | None = None


class ModelEvent(BaseModel):
    """Streaming event."""

    type: Literal["text_delta", "tool_call_delta", "usage", "done", "error"] = "text_delta"
    text: str | None = None
    tool_call: ToolCallRequest | None = None
    usage: TokenUsage | None = None
    error: str | None = None
    response: ModelResponse | None = None


class ModelCapabilities(BaseModel):
    """Provider/model capability matrix."""

    structured_output: bool = False
    tool_calling: bool = False
    vision: bool = False
    audio: bool = False
    prompt_caching: bool = False
    reasoning: bool = False  # exposes a thinking/reasoning-effort control
    max_context_tokens: int = 128_000
    max_output_tokens: int = 8_192
    supports_system_message: bool = True
    supports_developer_message: bool = False
    # Input/output modalities the model accepts/produces. ``vision``/``audio``
    # cover the common input booleans; these lists carry the full picture
    # (e.g. image/audio *output* for generation) without a breaking change.
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    output_modalities: list[str] = Field(default_factory=lambda: ["text"])


# Lifecycle state of a model relative to its published GA / deprecation /
# retirement dates. ``ga`` is serving normally; ``deprecated`` still serves but
# is scheduled for removal; ``retired`` no longer serves.
ModelLifecycle = Literal["ga", "deprecated", "retired"]


class ModelProfile(BaseModel):
    """A named model + provider + pricing + capabilities + lifecycle bundle.

    The :class:`~vincio.providers.registry.ModelRegistry` instantiates one
    profile per exact model id and is the single source capability guards, the
    cost price table, and model rotation all read from.
    """

    name: str
    provider: str
    model: str
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    cached_input_cost_per_mtok: float = 0.0
    # Batch-tier pricing (typically half cost); ``None`` falls back to the
    # standard rate when the provider has no separate batch price.
    batch_input_cost_per_mtok: float | None = None
    batch_output_cost_per_mtok: float | None = None
    tier: Literal["fast", "default", "strong"] = "default"
    # Alternate ids that resolve to this profile (e.g. dated snapshots,
    # provider aliases) so ``lookup`` need not fall back to substring sniffing.
    aliases: list[str] = Field(default_factory=list)
    # Lifecycle dates as ISO strings (provider-published); ``None`` when the
    # provider does not publish that milestone. Deadline-agnostic by design.
    ga_date: str | None = None
    deprecation_date: str | None = None
    retirement_date: str | None = None
    knowledge_cutoff: str | None = None
    # Suggested successor model id once deprecated/retired (consumed by the
    # lifecycle watcher); harmless metadata until then.
    successor: str | None = None

    def lifecycle(self, *, as_of: date | None = None) -> ModelLifecycle:
        """Lifecycle state relative to *as_of* (today by default)."""
        today = as_of or utcnow().date()

        def _parse(value: str | None) -> date | None:
            if not value:
                return None
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None

        retired = _parse(self.retirement_date)
        if retired is not None and today >= retired:
            return "retired"
        deprecated = _parse(self.deprecation_date)
        if deprecated is not None and today >= deprecated:
            return "deprecated"
        return "ga"


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"


class RunConfig(BaseModel):
    """Per-run overrides (A2)."""

    model: str | None = None
    provider: str | None = None
    temperature: float | None = None
    budget: Budget | None = None
    policies: PolicySet | None = None
    retrieval_top_k: int | None = None
    stream: bool = False
    seed: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    thinking_budget_tokens: int | None = None
    # Hard-cap enforcement of the full Budget on the single-shot run path:
    # max_cost_usd / max_input_tokens / max_output_tokens / max_steps become hard
    # caps that raise ``BudgetExceededError``. Set False to restore the previous
    # soft-cap behavior (only latency and tool-count enforced) for one minor.
    enforce_budget_caps: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Result of a ContextApp run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str = Field(default_factory=lambda: new_id("run"))
    status: RunStatus = RunStatus.SUCCEEDED
    output: Any = None
    raw_text: str = ""
    trace_id: str = ""
    context_packet_id: str | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    # Estimated energy (watt-hours) and carbon (grams CO₂e) of the run, accrued
    # from token accounting when energy accounting is enabled (otherwise 0.0).
    # The energy analogue of ``cost_usd``, on the same cost-report surface.
    energy_wh: float = 0.0
    co2e_grams: float = 0.0
    latency_ms: int = 0
    # Estimated resident-memory footprint of the compiled context packet, in
    # bytes. Held under the app's ``memory_budget_mb`` ceiling when one is set.
    memory_bytes: int = 0
    validation: dict[str, Any] = Field(default_factory=dict)
    eval_scores: dict[str, float] = Field(default_factory=dict)
    excluded_context: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunStreamEvent(BaseModel):
    """Event emitted by the streaming run flow (``ContextApp.astream``).

    Types:

    - ``stage`` — a pipeline stage finished (``stage`` + ``data``)
    - ``text_delta`` — a chunk of model output text
    - ``partial_output`` — best-effort parse of the structured output so far,
      with streaming validation: ``valid_prefix`` is False as soon as the
      partial output definitely cannot match the schema (``validation_errors``
      says why), so consumers can abort early
    - ``tool_call`` / ``tool_result`` — tool loop activity
    - ``usage`` — token usage update
    - ``done`` — terminal event carrying the final :class:`RunResult`
    - ``error`` — terminal event carrying the failure message
    """

    type: Literal[
        "stage",
        "text_delta",
        "partial_output",
        "tool_call",
        "tool_result",
        "usage",
        "done",
        "error",
    ] = "text_delta"
    stage: str | None = None
    text: str | None = None
    partial_output: Any = None
    output_complete: bool = False
    valid_prefix: bool | None = None  # streaming validation verdict (schema runs only)
    validation_errors: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    tool_result: ToolResult | None = None
    usage: TokenUsage | None = None
    result: RunResult | None = None
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
