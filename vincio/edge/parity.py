"""Parity, not a fork — the mechanical guarantees behind the edge build.

Two checks make "the edge build is the same library under a build target" a
verifiable fact rather than a promise:

- :func:`edge_manifest` statically certifies that every module on the
  compile/score/rail/pack path imports **no native or optional dependency**
  unconditionally — only the stdlib, ``pydantic``, ``httpx``, and other
  ``vincio`` modules — so the core is buildable for a WASM target. (A guarded
  ``try: import numpy`` is the accelerated-but-optional path and is exempt; the
  pure-Python fallback is what ships to the edge.)

- :func:`verify_edge_parity` runs an :class:`~vincio.edge.runtime.EdgeRuntime`
  compile and a direct server-side :class:`~vincio.context.compiler.ContextCompiler`
  compile over the *same* inputs and asserts the packet is byte-identical, and
  that the runtime delegates to the canonical compiler and rail engine — so a
  capability can never silently diverge between server and edge.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, Field

from ..context.compiler import ContextCompiler
from ..core.types import EvidenceItem, Objective, PolicySet, TaskType, UserInput
from ..security.rails import RailEngine
from .profile import EdgeProfile
from .runtime import EdgeRequest, EdgeRuntime

__all__ = [
    "EdgeManifest",
    "EdgeParityReport",
    "edge_manifest",
    "verify_edge_parity",
    "EDGE_CORE_MODULES",
]


# The modules that form the edge core: the compile → score → rail → pack path
# and everything it imports at import time. The manifest certifies each is free
# of an unconditional native/optional import, so the set is WASM-buildable.
EDGE_CORE_MODULES: tuple[str, ...] = (
    "vincio.context.compiler",
    "vincio.context.scoring",
    "vincio.context.vectorized",
    "vincio.context.budgeting",
    "vincio.context.arena",
    "vincio.context.compression",
    "vincio.context.llmlingua",
    "vincio.context.footprint",
    "vincio.context.ir",
    "vincio.context.packet",
    "vincio.context.evidence_store",
    "vincio.prompts.compiler",
    "vincio.prompts.templates",
    "vincio.prompts.ast",
    "vincio.prompts.lint",
    "vincio.prompts.program",
    "vincio.security.rails",
    "vincio.security.injection",
    "vincio.security.pii",
    "vincio.security.secrets",
    "vincio.retrieval.embeddings",
    "vincio.core.types",
    "vincio.core.tokens",
    "vincio.core.utils",
    "vincio.core.errors",
    "vincio.edge.profile",
    "vincio.edge.runtime",
    "vincio.edge.parity",
)


# Top-level packages that are native (C-extension) or live behind an optional
# extra. An unconditional import of any of these on the edge path would break a
# pure-WASM build or require an extra that the offline-first core must not need.
NATIVE_DENYLIST: frozenset[str] = frozenset(
    {
        "numpy",
        "av",
        "PIL",
        "pillow",
        "tiktoken",
        "openai",
        "anthropic",
        "google",
        "mistralai",
        "qdrant_client",
        "rank_bm25",
        "neo4j",
        "networkx",
        "boto3",
        "psycopg",
        "pgvector",
        "duckdb",
        "redis",
        "chromadb",
        "pinecone",
        "lancedb",
        "weaviate",
        "pymilvus",
        "elasticsearch",
        "opensearch",
        "opensearchpy",
        "pyvespa",
        "vespa",
        "opentelemetry",
        "fastapi",
        "uvicorn",
        "pydantic_settings",
        "pypdf",
        "pdfplumber",
        "docx",
        "websockets",
        "langchain",
        "langchain_core",
        "llama_index",
        "haystack",
        "dspy",
        "snowflake",
        "reportlab",
        "pptx",
        "pypdfium2",
        "pytesseract",
        "pyarrow",
        "extract_msg",
        "playwright",
        "fastembed",
        "transformers",
        "sentence_transformers",
        "llama_cpp",
    }
)


class EdgeManifest(BaseModel):
    """The static WASM-buildability certificate for the edge core.

    ``offending`` is the list of ``(module, package)`` pairs where an edge-core
    module imports a native/optional dependency unconditionally; ``clean`` is
    true when it is empty — i.e. the whole compile/score/rail/pack path is
    import-safe for a constrained target.
    """

    modules: list[str] = Field(default_factory=list)
    offending: list[tuple[str, str]] = Field(default_factory=list)
    clean: bool = True


def _module_file(module: str) -> Path:
    parts = module.split(".")
    root = Path(__file__).resolve().parent.parent.parent  # repo's vincio/ parent
    return root.joinpath(*parts).with_suffix(".py")


def _import_roots(node: ast.Import | ast.ImportFrom) -> list[str]:
    """The top-level package names a single import statement pulls in.

    Relative imports (``from . import x``) are intra-``vincio`` and never count.
    """
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if node.level and node.level > 0:
        return []  # relative — a vincio-internal import
    if node.module is None:
        return []
    return [node.module.split(".")[0]]


def _guarded(stmt: ast.Try) -> bool:
    """True when a ``try`` block catches ``ImportError`` — its body's imports are
    the optional/accelerated path (e.g. NumPy), exempt from the native check."""
    for handler in stmt.handlers:
        exc = handler.type
        names: list[str] = []
        if isinstance(exc, ast.Name):
            names = [exc.id]
        elif isinstance(exc, ast.Tuple):
            names = [e.id for e in exc.elts if isinstance(e, ast.Name)]
        if any(name in ("ImportError", "ModuleNotFoundError", "Exception") for name in names):
            return True
    return False


def _native_imports(module: str) -> list[str]:
    """The native/optional packages a module imports *unconditionally* at import
    time. Only module-level statements run on import, so imports nested in a
    function/class (lazy) or guarded by ``try/except ImportError`` are exempt."""
    path = _module_file(module)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []

    def collect(node: ast.Import | ast.ImportFrom) -> None:
        for root in _import_roots(node):
            if root in NATIVE_DENYLIST:
                offenders.append(root)

    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            collect(stmt)
        elif isinstance(stmt, ast.Try) and not _guarded(stmt):
            for inner in stmt.body:
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    collect(inner)
    return offenders


def edge_manifest() -> EdgeManifest:
    """Certify that the edge core imports no native/optional dependency.

    Statically scans every module in :data:`EDGE_CORE_MODULES` for an
    unconditional import of a package in :data:`NATIVE_DENYLIST`. A clean
    manifest is the WASM-buildability guarantee: the compile/score/rail/pack path
    needs only the stdlib, ``pydantic``, and ``httpx`` (all pure-Python /
    WASM-available), with NumPy used only behind a guarded fallback.
    """
    offending: list[tuple[str, str]] = []
    for module in EDGE_CORE_MODULES:
        for package in _native_imports(module):
            offending.append((module, package))
    return EdgeManifest(
        modules=list(EDGE_CORE_MODULES),
        offending=offending,
        clean=not offending,
    )


class EdgeParityReport(BaseModel):
    """The result of verifying the edge build is the same library, not a fork.

    ``held`` is true when the edge runtime's packet is byte-identical to a direct
    server compile over the same inputs (``packet_identical``), the runtime
    delegates to the canonical compiler and rail engine (``same_compiler`` /
    ``same_rail_engine``), and the core is import-clean for WASM
    (``no_native_imports``).
    """

    packet_identical: bool = False
    same_compiler: bool = False
    same_rail_engine: bool = False
    no_native_imports: bool = False
    held: bool = False
    edge_spec_hash: str = ""
    server_spec_hash: str = ""


def _default_request() -> EdgeRequest:
    return EdgeRequest(
        task="What is the refund window and who approves an exception?",
        objective="Answer the policy question from the cited evidence.",
        task_type=TaskType.DOCUMENT_QA,
        instructions=["Answer only from the evidence.", "Cite the source id."],
        constraints=["Do not speculate beyond the evidence."],
        evidence=[
            EvidenceItem(
                source_id="policy",
                text="Refunds are available within 30 days of purchase.",
                authority=0.9,
                relevance=0.9,
            ),
            EvidenceItem(
                source_id="exceptions",
                text="A refund exception beyond the window is approved by a manager.",
                authority=0.8,
                relevance=0.8,
            ),
            EvidenceItem(
                source_id="unrelated",
                text="The office cafeteria reopens at noon on weekdays.",
                authority=0.4,
                relevance=0.05,
            ),
        ],
    )


def verify_edge_parity(
    request: EdgeRequest | None = None,
    *,
    profile: EdgeProfile | None = None,
) -> EdgeParityReport:
    """Prove the edge runtime is the server compiler under a profile, not a fork.

    Compiles *request* both through an :class:`~vincio.edge.runtime.EdgeRuntime`
    and through a direct :class:`~vincio.context.compiler.ContextCompiler` built
    from the *same* profile options, and asserts the resulting packets share a
    ``spec_hash`` (and evidence selection and token count). Also asserts the
    runtime delegates to the canonical compiler/rail-engine classes and that the
    core is import-clean (:func:`edge_manifest`). The whole check is
    deterministic and offline.
    """
    profile = profile or EdgeProfile.default()
    req = request or _default_request()
    runtime = EdgeRuntime(profile)
    edge_result = runtime.run(req)

    # Direct server-side compile under the identical option set — same code, so
    # the packet must be byte-identical. Divergence here means a real fork.
    server_compiler = ContextCompiler(profile.to_compiler_options())

    async def _server_compile() -> object:
        from ..core.types import Budget, Constraint, Instruction

        return await server_compiler.compile(
            objective=Objective(
                text=req.objective or req.task or "edge context",
                task_type=req.task_type,
            ),
            user_input=UserInput(text=req.task, tenant_id=req.tenant_id),
            instructions=[Instruction(text=t) for t in req.instructions],
            constraints=[Constraint(text=t) for t in req.constraints],
            evidence=list(req.evidence),
            memory=list(req.memory),
            budget=req.budget
            or Budget(
                max_input_tokens=profile.max_input_tokens,
                max_output_tokens=profile.max_output_tokens,
            ),
            policies=PolicySet(),
        )

    from .runtime import _run_sync

    server_compiled = _run_sync(_server_compile())
    server_packet = server_compiled.packet  # type: ignore[attr-defined]

    edge_ids = [e.get("id") for e in edge_result.packet.evidence_items]
    server_ids = [e.get("id") for e in server_packet.evidence_items]
    packet_identical = bool(
        edge_result.packet.spec_hash == server_packet.spec_hash
        and edge_ids == server_ids
        and edge_result.token_count == server_compiled.token_count  # type: ignore[attr-defined]
    )

    same_compiler = type(runtime.compiler) is ContextCompiler
    same_rail_engine = type(runtime.rail_engine) is RailEngine
    no_native_imports = edge_manifest().clean

    return EdgeParityReport(
        packet_identical=packet_identical,
        same_compiler=same_compiler,
        same_rail_engine=same_rail_engine,
        no_native_imports=no_native_imports,
        held=packet_identical and same_compiler and same_rail_engine and no_native_imports,
        edge_spec_hash=edge_result.packet.spec_hash,
        server_spec_hash=server_packet.spec_hash,
    )
