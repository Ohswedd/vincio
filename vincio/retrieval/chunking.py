"""Chunking strategies.

Supported: fixed tokens, recursive splitting, semantic chunking,
heading-aware, table-aware, code-aware, adaptive, document-structure.
All chunkers attach provenance metadata.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ..context.compression import split_sentences
from ..context.scoring import lexical_similarity
from ..core.tokens import count_tokens
from ..core.types import Chunk, Document
from ..core.utils import new_id
from ..documents.parsers import TableData, extract_code_symbols

__all__ = ["chunk_document", "extract_entities", "ChunkingStrategy", "CHUNKERS"]

ChunkingStrategy = Callable[[Document, int, int], list[Chunk]]


def _make_chunk(document: Document, text: str, index: int, **kw) -> Chunk:
    return Chunk(
        document_id=document.id,
        text=text,
        token_count=count_tokens(text),
        source_uri=document.source_uri,
        permissions=list(document.permissions),
        tenant_id=document.tenant_id,
        created_at=document.created_at,
        index=index,
        **kw,
    )


# -- fixed token windows ----------------------------------------------------------

def fixed_chunker(document: Document, size: int, overlap: int) -> list[Chunk]:
    words = document.text.split()
    if not words:
        return []
    chunks: list[Chunk] = []
    # Approximate tokens-per-word ratio for window sizing.
    ratio = max(0.5, count_tokens(document.text) / max(1, len(words)))
    window = max(1, int(size / ratio))
    step = max(1, window - int(overlap / ratio))
    index = 0
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + window])
        if not piece.strip():
            continue
        chunks.append(_make_chunk(document, piece, index))
        index += 1
        if start + window >= len(words):
            break
    return chunks


# -- recursive splitting ----------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, size: int, separators: list[str]) -> list[str]:
    if count_tokens(text) <= size or not separators:
        return [text] if text.strip() else []
    separator, *rest = separators
    parts = text.split(separator)
    pieces: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}{separator}{part}" if buffer else part
        if count_tokens(candidate) <= size:
            buffer = candidate
        else:
            if buffer.strip():
                pieces.append(buffer)
            if count_tokens(part) > size:
                pieces.extend(_recursive_split(part, size, rest))
                buffer = ""
            else:
                buffer = part
    if buffer.strip():
        pieces.append(buffer)
    return pieces


def recursive_chunker(document: Document, size: int, overlap: int) -> list[Chunk]:
    pieces = _recursive_split(document.text, size, _SEPARATORS)
    chunks: list[Chunk] = []
    previous_tail = ""
    for index, piece in enumerate(pieces):
        text = piece.strip()
        if not text:
            continue
        if overlap > 0 and previous_tail:
            text = f"{previous_tail} {text}"
        # Tail of this piece becomes overlap for the next.
        words = piece.split()
        tail_words = max(0, int(overlap / 4))
        previous_tail = " ".join(words[-tail_words:]) if tail_words else ""
        chunks.append(_make_chunk(document, text, index))
    return chunks


# -- semantic chunking ---------------------------------------------------------------

def semantic_chunker(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Group sentences by lexical cohesion: a new chunk starts when the next
    sentence's similarity to the current chunk drops below a threshold."""
    sentences = split_sentences(document.text)
    if not sentences:
        return []
    chunks: list[Chunk] = []
    current: list[str] = [sentences[0]]
    index = 0
    for sentence in sentences[1:]:
        current_text = " ".join(current)
        cohesion = lexical_similarity(current_text, sentence)
        if (count_tokens(current_text) + count_tokens(sentence) > size) or (
            cohesion < 0.08 and count_tokens(current_text) > size // 4
        ):
            chunks.append(_make_chunk(document, current_text, index))
            index += 1
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(_make_chunk(document, " ".join(current), index))
    return chunks


# -- heading-aware ----------------------------------------------------------------------

def heading_chunker(document: Document, size: int, overlap: int) -> list[Chunk]:
    """One chunk per section (split oversized sections recursively),
    prefixing the section path for self-contained retrieval units."""
    if not document.sections:
        return recursive_chunker(document, size, overlap)
    chunks: list[Chunk] = []
    index = 0
    for section in document.sections:
        title = section.get("title", "")
        path = section.get("path") or ([title] if title else [])
        text = (section.get("text") or "").strip()
        if not text and not title:
            continue
        header = " > ".join(path)
        body = f"{header}\n{text}" if header else text
        page = section.get("page")
        if count_tokens(body) <= size:
            chunks.append(
                _make_chunk(document, body, index, section_path=path, page=page)
            )
            index += 1
        else:
            for piece in _recursive_split(text, size, _SEPARATORS):
                if not piece.strip():
                    continue
                chunks.append(
                    _make_chunk(
                        document, f"{header}\n{piece.strip()}" if header else piece.strip(),
                        index, section_path=path, page=page,
                    )
                )
                index += 1
    return chunks


# -- table-aware ---------------------------------------------------------------------------

def table_chunker(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Tables become dedicated chunks (table-aware); remaining text
    is chunked heading-aware."""
    chunks = heading_chunker(document, size, overlap)
    base_index = len(chunks)
    for offset, table_dump in enumerate(document.tables):
        table = TableData.model_validate({k: v for k, v in table_dump.items() if k != "formulas"})
        text = table.to_text()
        if not text.strip():
            continue
        chunks.append(
            _make_chunk(
                document,
                text,
                base_index + offset,
                kind="table",
                metadata={"table_id": table.id, "inferred_schema": table.inferred_schema},
            )
        )
    return chunks


# -- code-aware -----------------------------------------------------------------------------

def code_chunker(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Split source by top-level symbols so functions/classes stay intact."""
    language = document.metadata.get("language", "python")
    symbols = [
        s for s in extract_code_symbols(document.text, language=language) if s.kind != "import"
    ]
    lines = document.text.splitlines()
    if not symbols:
        return recursive_chunker(document, size, overlap)
    boundaries = sorted({max(0, s.line - 1) for s in symbols})
    if boundaries and boundaries[0] != 0:
        boundaries.insert(0, 0)
    chunks: list[Chunk] = []
    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else len(lines)
        text = "\n".join(lines[start:end]).strip()
        if not text:
            continue
        symbol = next((s for s in symbols if s.line - 1 >= start and s.line - 1 < end), None)
        chunks.append(
            _make_chunk(
                document,
                text,
                index,
                kind="code",
                section_path=[symbol.name] if symbol else [],
                metadata={"symbol": symbol.name if symbol else None, "language": language},
            )
        )
    return chunks


# -- sentence-window --------------------------------------------------------------------------

def sentence_window_chunker(document: Document, size: int, overlap: int) -> list[Chunk]:
    """One chunk per sentence, carrying a ±2-sentence window in metadata.

    Retrieval scores the precise sentence; the engine swaps in the window
    text when building evidence, so the model sees enough surrounding
    context (sentence-window retrieval)."""
    sentences = [s for s in split_sentences(document.text) if s.strip()]
    chunks: list[Chunk] = []
    window_radius = 2
    for index, sentence in enumerate(sentences):
        lo = max(0, index - window_radius)
        hi = min(len(sentences), index + window_radius + 1)
        window = " ".join(sentences[lo:hi])
        chunks.append(
            _make_chunk(
                document,
                sentence,
                index,
                metadata={"window_text": window, "matched_sentence": sentence},
            )
        )
    return chunks


# -- hierarchical / parent-document ------------------------------------------------------------

def hierarchical_chunker(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Two-level hierarchy for auto-merging / parent-document retrieval.

    Parents are large coherent units (~4× ``size``); each parent splits into
    small children that link back via ``metadata["parent_id"]``. Index the
    children for precision and hand the flat list to
    :class:`~vincio.retrieval.hierarchy.AutoMergingIndex`, which stores the
    parents and merges sibling hits back into them."""
    parent_size = size * 4
    parents = (heading_chunker if document.sections else recursive_chunker)(
        document, parent_size, 0
    )
    chunks: list[Chunk] = []
    index = 0
    for parent in parents:
        parent.metadata = {**parent.metadata, "level": "parent"}
        parent.index = index
        index += 1
        chunks.append(parent)
        pieces = [p.strip() for p in _recursive_split(parent.text, size, _SEPARATORS) if p.strip()]
        if len(pieces) <= 1:
            # The parent is already small; index it directly as its own child.
            child = parent.model_copy(
                update={
                    "id": new_id("chk"),
                    "index": index,
                    "metadata": {**parent.metadata, "level": "child", "parent_id": parent.id},
                }
            )
            index += 1
            chunks.append(child)
            continue
        for piece in pieces:
            chunks.append(
                _make_chunk(
                    document,
                    piece,
                    index,
                    section_path=list(parent.section_path),
                    page=parent.page,
                    metadata={"level": "child", "parent_id": parent.id},
                )
            )
            index += 1
    return chunks


# -- contextual retrieval -----------------------------------------------------------------------

def _document_context(document: Document) -> str:
    lead = split_sentences(document.text)[:1]
    parts = [p for p in (document.title, lead[0] if lead else "") if p]
    return ". ".join(part.rstrip(".") for part in parts)


def contextual_chunker(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Adaptive chunking plus a situating prefix per chunk ("contextual
    retrieval"). The offline prefix is heuristic (title, section path,
    document lead); use :func:`~vincio.retrieval.hierarchy.contextualize_chunks`
    to upgrade prefixes with LLM-written context."""
    chunks = adaptive_chunker(document, size, overlap)
    doc_context = _document_context(document)
    for chunk in chunks:
        scope = " > ".join(chunk.section_path)
        prefix_parts = [p for p in (doc_context, scope) if p]
        prefix = " — ".join(prefix_parts)
        if not prefix or chunk.text.startswith(f"[{prefix}]"):
            continue
        chunk.metadata = {**chunk.metadata, "original_text": chunk.text, "contextualized": "heuristic"}
        chunk.text = f"[{prefix}]\n{chunk.text}"
        chunk.token_count = count_tokens(chunk.text)
    return chunks


# -- adaptive --------------------------------------------------------------------------------

def adaptive_chunker(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Pick the best strategy per document (adaptive)."""
    if document.media_type.startswith("text/x-") or "language" in document.metadata:
        return code_chunker(document, size, overlap)
    if document.tables:
        return table_chunker(document, size, overlap)
    if document.sections:
        return heading_chunker(document, size, overlap)
    if count_tokens(document.text) > size * 4:
        return semantic_chunker(document, size, overlap)
    return recursive_chunker(document, size, overlap)


CHUNKERS: dict[str, ChunkingStrategy] = {
    "fixed": fixed_chunker,
    "recursive": recursive_chunker,
    "semantic": semantic_chunker,
    "heading_aware": heading_chunker,
    "table_aware": table_chunker,
    "code_aware": code_chunker,
    "adaptive": adaptive_chunker,
    "document_structure": heading_chunker,
    "sentence_window": sentence_window_chunker,
    "hierarchical": hierarchical_chunker,
    "parent_document": hierarchical_chunker,
    "contextual": contextual_chunker,
}


def chunk_document(
    document: Document,
    *,
    strategy: str = "adaptive",
    size: int = 400,
    overlap: int = 50,
    cache: Any | None = None,
) -> list[Chunk]:
    """Chunk a document; with a :class:`~vincio.caching.compilation.ChunkCache`
    the result is content-addressed, so unchanged content is never re-chunked."""
    if strategy not in CHUNKERS:
        raise ValueError(f"unknown chunking strategy {strategy!r}; known: {sorted(CHUNKERS)}")
    cache_key: str | None = None
    if cache is not None:
        cache_key = cache.key(content=document.text, strategy=strategy, size=size, overlap=overlap)
        cached = cache.get(cache_key)
        if cached is not None:
            restored = [Chunk.model_validate(dump) for dump in cached]
            # Provenance belongs to the *requesting* document, not the one
            # that originally populated the cache entry.
            for chunk in restored:
                chunk.id = new_id("chk")
                chunk.document_id = document.id
                chunk.source_uri = document.source_uri
                chunk.tenant_id = document.tenant_id
                chunk.permissions = list(document.permissions)
                chunk.created_at = document.created_at
            return restored
    chunks = CHUNKERS[strategy](document, size, overlap)
    # Annotate entities (capitalized multiword phrases + ids) for graph retrieval.
    for chunk in chunks:
        chunk.entities = extract_entities(chunk.text)
    if cache is not None and cache_key is not None:
        cache.set(cache_key, [chunk.model_dump(mode="json") for chunk in chunks])
    return chunks


_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3}|[A-Z]{2,}-\d+|\b\d{4}-\d{2}-\d{2}\b)"
)
_ENTITY_STOP = frozenset(
    "The This That These Those There It If When Where How What Why Who Which In On At For And But Or Not".split()
)


def extract_entities(text: str) -> list[str]:
    """Capitalized multiword phrases, ticket ids, and dates — the same
    entity vocabulary the memory graph and graph retrieval index."""
    entities: list[str] = []
    for match in _ENTITY_RE.finditer(text):
        value = match.group(0).strip()
        if value.split()[0] in _ENTITY_STOP:
            continue
        if value not in entities:
            entities.append(value)
    return entities[:16]
