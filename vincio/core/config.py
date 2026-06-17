"""Project configuration (``vincio.yaml``).

Configuration is layered: defaults < config file < environment variables
(``VINCIO_*``) < explicit constructor arguments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from .errors import ConfigError
from .types import Budget, PolicySet

__all__ = [
    "ProviderConfig",
    "StorageConfig",
    "ObservabilityConfig",
    "SecurityConfig",
    "GovernanceConfig",
    "RetrievalConfig",
    "MemoryConfig",
    "CacheConfig",
    "PerformanceConfig",
    "ServerConfig",
    "VincioConfig",
    "load_config",
    "find_config_file",
    "config_json_schema",
]

CONFIG_FILENAMES = ("vincio.yaml", "vincio.yml", "vincio.json")


class ProviderConfig(BaseModel):
    default: str = "openai"
    model: str = "gpt-5.2"
    fallback_models: list[str] = Field(default_factory=list)
    api_keys: dict[str, str] = Field(default_factory=dict)  # provider -> env var name or key
    base_urls: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 120.0
    max_retries: int = 2

    def resolve_api_key(self, provider: str) -> str | None:
        """Resolve an API key: explicit config value, named env var, or standard env var."""
        configured = self.api_keys.get(provider)
        if configured:
            # Treat values that look like env var names as indirection.
            if configured.isupper() and configured in os.environ:
                return os.environ[configured]
            return configured
        standard = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }.get(provider)
        if standard:
            return os.environ.get(standard)
        return os.environ.get(f"{provider.upper()}_API_KEY")


class StorageConfig(BaseModel):
    metadata: str = "sqlite:///.vincio/vincio.db"
    vector: str = "memory://"
    graph: str = "memory://"
    cache: str = "memory://"
    documents_dir: str = ".vincio/documents"
    analytics: str | None = None  # e.g. "duckdb:///.vincio/analytics.duckdb"


class ObservabilityConfig(BaseModel):
    exporter: str = "jsonl"  # jsonl | memory | otel | none
    traces_dir: str = ".vincio/traces"
    redact_pii_in_traces: bool = False
    sample_rate: float = 1.0
    # Opt-in: record the full (untruncated) output and cited evidence on each
    # trace so the distillation flywheel (1.4) can curate faithful, grounded
    # fine-tuning data. Off by default — the span output stays truncated.
    training_capture: bool = False


class SecurityConfig(BaseModel):
    tenant_isolation: bool = True
    pii_detection: bool = True
    injection_detection: bool = True
    audit_log: bool = True
    audit_dir: str = ".vincio/audit"
    retention_days: int | None = None
    # 2.0: HMAC key for tamper-evident audit-chain signatures. When set (e.g.
    # via VINCIO_SECURITY__AUDIT_SIGNING_KEY), every audit entry is signed so a
    # privileged attacker cannot forge a clean chain by recomputing hashes.
    # Empty leaves the chain unsigned (1.x behavior). For asymmetric signing,
    # construct an Ed25519Signer and pass it to AuditLog directly.
    audit_signing_key: str = ""
    audit_signing_key_id: str = "hmac"
    # 2.0: always-on egress DLP scan of the assembled provider request at the
    # provider boundary. "off" disables; "warn" records findings without
    # blocking; "block" raises on a high-confidence leak.
    egress_dlp: Literal["off", "warn", "block"] = "warn"


class GovernanceConfig(BaseModel):
    """Enterprise governance & compliance settings (1.6).

    All fields are off/empty by default, so governance is opt-in and
    backward-compatible. Enable residency by listing ``allowed_regions``; enable
    non-English PII by listing ``locales`` (see :mod:`vincio.security.locales`).
    """

    # Data-residency-aware routing: when non-empty, runs may only egress to
    # these provider regions (else a blocking residency PolicyViolation).
    allowed_regions: list[str] = Field(default_factory=list)
    provider_regions: dict[str, str] = Field(default_factory=dict)
    deny_on_unknown_region: bool = True
    # Synthetic-content (EU AI Act) output marking on every run's output.
    content_marking: bool = False
    # Non-English PII locale packs applied by the security PII detector.
    locales: list[str] = Field(default_factory=list)
    # Default machine-readable card schema (validated at config-load time).
    card_format: Literal["vincio", "open_model_card", "ai_card"] = "vincio"


class RetrievalConfig(BaseModel):
    top_k: int = 8
    candidate_multiplier: int = 4  # candidates fetched per index before merge/rerank
    hybrid_weight_dense: float = 0.5
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50
    chunking: str = "recursive"
    reranker: str | None = "heuristic"
    embedder: str = "local"  # local | jina | voyage | cohere | voyage-context | <provider>
    # Matryoshka (MRL) output-dimension truncation; None keeps the model's
    # native dimension. Hosted embedders that support it truncate server-side.
    embedding_dimensions: int | None = None
    # Query-understanding strategies applied per retrieve():
    # hyde | multi_query | decompose | step_back
    query_strategies: list[str] = Field(default_factory=list)
    # Opt-in embedding-driven context scoring (1.7): when True *and* a semantic
    # ``embedder`` is configured, the context compiler scores relevance, novelty,
    # dedup, and conflict by embedding cosine and selects via MMR. Off by default
    # (the local hash embedder is not semantic), so selection stays lexical.
    semantic_context_scoring: bool = False
    mmr_lambda: float = 0.7  # MMR relevance/diversity trade-off when semantic scoring is on


class MemoryConfig(BaseModel):
    enabled: bool = True
    decay_lambda: float = 0.01  # per day
    min_confidence: float = 0.25
    max_items_per_run: int = 8
    write_policy: str = "guarded"  # guarded | open | off
    hybrid_recall: bool = True  # fuse lexical + vector + graph signals per query
    vector_weight: float = 0.5  # vector share of fused relevance
    retention_weight: float = 0.5  # importance-weighted retention strength
    # Default TTL per scope, in days; scopes not listed never expire.
    ttl_days: dict[str, float] = Field(default_factory=lambda: {"session": 30.0})
    # What step 16 writes back: input | evidence | tools | facts
    write_back: list[str] = Field(default_factory=lambda: ["input"])
    # Auto-memory from runs (0.8, write_back includes "facts"): output claims
    # need this much evidence support to become candidate memories, capped
    # per run. Admission still goes through the guarded write policy.
    fact_min_support: float = 0.5
    max_facts_per_run: int = 5


class CacheConfig(BaseModel):
    response_cache: bool = False
    tool_cache: bool = True
    embedding_cache: bool = True
    retrieval_cache: bool = False
    semantic_cache: bool = False
    semantic_threshold: float = 0.97
    ttl_s: int = 3600
    max_entries: int = 10_000
    # Content-addressed compilation caches (0.2): unchanged inputs are never
    # recomputed. All keys cover every input that affects the output, so
    # these are safe to leave on.
    prompt_compile_cache: bool = True
    chunk_cache: bool = True
    context_compile_cache: bool = True
    # Provider-aware prompt caching (1.3): attach a TTL to the compiler's stable
    # prefix for providers with explicit breakpoints (Anthropic) and record
    # cache-hit-rate telemetry. Auto-cache providers (OpenAI/Gemini) rely on the
    # stable→volatile ordering the compiler already produces.
    provider_cache: bool = True
    provider_cache_ttl: str = "5m"  # 5m | 1h
    provider_cache_min_prefix_tokens: int = 1024


class PerformanceConfig(BaseModel):
    """Concurrency, streaming, and transport tuning (0.2)."""

    max_concurrency: int = 8  # bound for retrieval/tool/embedding fan-out
    tool_parallelism: int = 4  # concurrent tool calls within one model round
    embed_batch_size: int = 64  # max texts per provider embedding call
    embed_window_ms: float = 5.0  # micro-batch coalescing window
    coalesce_requests: bool = True  # in-flight dedup of identical model calls
    max_connections: int = 100  # provider HTTP pool size
    max_keepalive_connections: int = 20
    slim_packets: bool = False  # persist packets without duplicated evidence text
    partial_parse_min_chars: int = 24  # min new chars between partial-JSON parses


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8042
    api_keys: list[str] = Field(default_factory=list)
    jwt_secret: str | None = None
    cors_origins: list[str] = Field(default_factory=list)


class VincioConfig(BaseModel):
    """Top-level project configuration."""

    project: str = "vincio_app"
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    budget: Budget = Field(default_factory=Budget)
    policies: PolicySet = Field(default_factory=PolicySet)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VincioConfig:
        try:
            return cls.model_validate(data)
        except Exception as exc:  # pydantic ValidationError
            raise ConfigError(f"invalid configuration: {exc}") from exc

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


def config_json_schema() -> dict[str, Any]:
    """JSON Schema for ``vincio.yaml``.

    Drives editor completion and validation. Point your editor's YAML language
    server at a file produced by ``vincio config schema`` (a
    ``# yaml-language-server: $schema=...`` line at the top of the config does
    this) to get inline completion and type checks against the typed config.
    """
    schema = VincioConfig.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://vincio.dev/schema/vincio.json"
    schema["title"] = "VincioConfig"
    return schema


def find_config_file(start: str | Path | None = None) -> Path | None:
    """Locate the nearest vincio config file walking up from *start*."""
    directory = Path(start or os.getcwd()).resolve()
    for candidate_dir in (directory, *directory.parents):
        for name in CONFIG_FILENAMES:
            path = candidate_dir / name
            if path.is_file():
                return path
    return None


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply ``VINCIO_SECTION__FIELD=value`` environment overrides."""
    for key, value in os.environ.items():
        if not key.startswith("VINCIO_") or "__" not in key:
            continue
        path = key[len("VINCIO_") :].lower().split("__")
        node = data
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                break
        else:
            parsed: Any = value
            if value.lower() in ("true", "false"):
                parsed = value.lower() == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value
            node[path[-1]] = parsed
    return data


def load_config(
    path: str | Path | None = None, *, overrides: dict[str, Any] | None = None
) -> VincioConfig:
    """Load configuration from a file (or discover it), env vars, and overrides."""
    data: dict[str, Any] = {}
    config_path = Path(path) if path else find_config_file()
    if path and not Path(path).is_file():
        raise ConfigError(f"config file not found: {path}")
    if config_path and config_path.is_file():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse {config_path}: {exc}") from exc
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ConfigError(f"config root must be a mapping: {config_path}")
            data = loaded
    data = _apply_env_overrides(data)
    if overrides:
        _deep_merge(data, overrides)
    return VincioConfig.from_dict(data)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
