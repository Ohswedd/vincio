"""Loaders for the formats the classifier advertised but couldn't load.

PPTX, EPUB, RTF, and ODT parse dependency-free (they are OOXML/ODF zips or a
control-word stream); Parquet/Arrow and ``.msg`` use optional extras; mbox uses
the stdlib. Each registers itself on the shared parser registry, so the
classifier's promises and the loader's reality finally agree.
"""

from __future__ import annotations

import html as html_module
import mailbox
import re
import zipfile
from pathlib import Path
from typing import Any

from ..core.errors import LoaderError
from ..core.types import Document
from .parsers import TableData, infer_table_schema, strip_html, table_quality_checks
from .registry import register_loader

__all__ = [
    "load_pptx",
    "load_epub",
    "load_rtf",
    "load_odt",
    "load_parquet",
    "load_mbox",
    "load_msg",
]

_SLIDE_RE = re.compile(r"ppt/slides/slide(\d+)\.xml$")
_A_T_RE = re.compile(r"<a:t>(.*?)</a:t>", re.DOTALL)

# Decompression guardrail: refuse to inflate a single zip entry past this so a
# malicious archive (zip bomb) cannot exhaust memory during a dependency-free
# OOXML/ODF parse. Generous for real documents; override is intentional-only.
_MAX_ZIP_ENTRY_BYTES = 64 * 1024 * 1024  # 64 MB per entry


def _read_zip_entry(archive: zipfile.ZipFile, name: str, *, max_bytes: int = _MAX_ZIP_ENTRY_BYTES) -> bytes:
    """Read one zip entry, refusing entries that inflate past ``max_bytes``."""
    info = archive.getinfo(name)
    if info.file_size > max_bytes:
        raise LoaderError(
            f"zip entry {name!r} inflates to {info.file_size} bytes, over the {max_bytes}-byte cap"
        )
    with archive.open(name) as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise LoaderError(f"zip entry {name!r} exceeds the {max_bytes}-byte decompression cap")
    return data


@register_loader(".pptx")
def load_pptx(path: str | Path, **_: Any) -> Document:
    """Extract slide text from a PPTX (dependency-free OOXML zip parse)."""
    path = Path(path)
    sections: list[dict[str, Any]] = []
    texts: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            slides = sorted(
                (n for n in archive.namelist() if _SLIDE_RE.search(n)),
                key=lambda n: int(_SLIDE_RE.search(n).group(1)),  # type: ignore[union-attr]
            )
            for index, name in enumerate(slides, start=1):
                xml = _read_zip_entry(archive, name).decode("utf-8", "ignore")
                runs = [html_module.unescape(t) for t in _A_T_RE.findall(xml)]
                body = "\n".join(r for r in runs if r.strip())
                texts.append(body)
                sections.append(
                    {"title": f"slide {index}", "level": 1, "path": [f"slide{index}"],
                     "text": body, "start_line": 0}
                )
    except (zipfile.BadZipFile, KeyError) as exc:
        raise LoaderError(f"invalid PPTX {path}: {exc}") from exc
    return Document(
        text="\n\n".join(t for t in texts if t),
        title=path.stem,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        sections=sections,
        metadata={"slide_count": len(sections), "extractor": "pptx-ooxml"},
    )


@register_loader(".epub")
def load_epub(path: str | Path, **_: Any) -> Document:
    """Extract reading-order text from an EPUB (dependency-free zip parse)."""
    path = Path(path)
    sections: list[dict[str, Any]] = []
    texts: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            opf_path = None
            if "META-INF/container.xml" in names:
                container = _read_zip_entry(archive, "META-INF/container.xml").decode("utf-8", "ignore")
                match = re.search(r'full-path="([^"]+)"', container)
                opf_path = match.group(1) if match else None
            ordered: list[str] = []
            if opf_path and opf_path in names:
                opf = _read_zip_entry(archive, opf_path).decode("utf-8", "ignore")
                base = opf_path.rsplit("/", 1)[0] if "/" in opf_path else ""
                # Resolve id→href per <item>, attribute-order independent (EPUB/OPF
                # allows id before *or* after href).
                href_by_id: dict[str, str] = {}
                for tag in re.findall(r"<item\b[^>]*>", opf):
                    id_match = re.search(r'\bid="([^"]+)"', tag)
                    href_match = re.search(r'\bhref="([^"]+)"', tag)
                    if id_match and href_match:
                        href_by_id[id_match.group(1)] = href_match.group(1)
                for idref in re.findall(r'<itemref\b[^>]*idref="([^"]+)"', opf):
                    href = href_by_id.get(idref)
                    if not href:
                        continue
                    full = f"{base}/{href}" if base else href
                    if full in names:
                        ordered.append(full)
            if not ordered:
                ordered = sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))
            for name in ordered:
                body = strip_html(_read_zip_entry(archive, name).decode("utf-8", "ignore"))
                if not body.strip():
                    continue
                texts.append(body)
                sections.append(
                    {"title": f"chapter {len(sections) + 1}", "level": 1,
                     "path": [Path(name).stem], "text": body, "start_line": 0}
                )
    except (zipfile.BadZipFile, KeyError) as exc:
        raise LoaderError(f"invalid EPUB {path}: {exc}") from exc
    return Document(
        text="\n\n".join(texts),
        title=path.stem,
        media_type="application/epub+zip",
        sections=sections,
        metadata={"chapter_count": len(sections), "extractor": "epub"},
    )


_RTF_CONTROL_RE = re.compile(r"\\([a-z]+)(-?\d+)?[ ]?|\\([^a-z])", re.IGNORECASE)


@register_loader(".rtf")
def load_rtf(path: str | Path, **_: Any) -> Document:
    """Strip an RTF document to plain text (dependency-free)."""
    path = Path(path)
    raw = path.read_text(encoding="latin-1", errors="ignore")
    out: list[str] = []
    depth = 0
    i = 0
    n = len(raw)
    while i < n:
        char = raw[i]
        if char == "{":
            depth += 1
            i += 1
        elif char == "}":
            depth = max(0, depth - 1)
            i += 1
        elif char == "\\":
            match = _RTF_CONTROL_RE.match(raw, i)
            if match:
                word = match.group(1)
                if word in ("par", "line", "sect"):
                    out.append("\n")
                elif word == "tab":
                    out.append("\t")
                elif word == "u" and match.group(2) is not None:
                    # RTF \uN Unicode escape — only the exact control word "u"
                    # (not \uc1, \ul0, \up6, … which merely start with 'u').
                    try:
                        out.append(chr(int(match.group(2)) % 0x10000))
                    except (ValueError, OverflowError):
                        pass
                elif match.group(3):  # escaped literal like \{ \} \\
                    out.append(match.group(3))
                i = match.end()
            else:
                i += 1
        else:
            out.append(char)
            i += 1
    text = re.sub(r"[ \t]+", " ", "".join(out))
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return Document(
        text=text, title=path.stem, media_type="application/rtf",
        metadata={"extractor": "rtf"},
    )


_ODT_BLOCK_END_RE = re.compile(r"</text:(p|h)>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


@register_loader(".odt")
def load_odt(path: str | Path, **_: Any) -> Document:
    """Extract text from an ODT (dependency-free ODF zip parse)."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            content = _read_zip_entry(archive, "content.xml").decode("utf-8", "ignore")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise LoaderError(f"invalid ODT {path}: {exc}") from exc
    content = _ODT_BLOCK_END_RE.sub("\n", content)
    content = content.replace("<text:tab/>", "\t").replace("<text:line-break/>", "\n")
    text = html_module.unescape(_TAG_RE.sub("", content))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return Document(
        text=text, title=path.stem,
        media_type="application/vnd.oasis.opendocument.text",
        metadata={"extractor": "odt"},
    )


@register_loader(".parquet", extra="parquet")
def load_parquet(path: str | Path, **_: Any) -> Document:
    """Load a Parquet/Arrow file as a :class:`TableData` (``vincio[parquet]``)."""
    path = Path(path)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dep
        raise LoaderError('Parquet support requires pyarrow: pip install "vincio[parquet]"') from exc
    table = pq.read_table(str(path))
    columns = list(table.column_names)
    pydict = table.to_pydict()
    row_count = table.num_rows
    rows = [
        ["" if pydict[col][i] is None else str(pydict[col][i]) for col in columns]
        for i in range(row_count)
    ]
    data = TableData(id="T1", title=path.stem, columns=columns, rows=rows)
    data.inferred_schema = infer_table_schema(data)
    data.quality = table_quality_checks(data)
    return Document(
        text=data.to_text(), title=path.stem, media_type="application/vnd.apache.parquet",
        tables=[data.model_dump(mode="json")],
        metadata={"row_count": row_count, "extractor": "parquet"},
    )


@register_loader(".mbox")
def load_mbox(path: str | Path, **_: Any) -> Document:
    """Load an mbox thread as one section per message (stdlib mailbox)."""
    path = Path(path)
    box = mailbox.mbox(str(path))
    sections: list[dict[str, Any]] = []
    texts: list[str] = []
    try:
        for index, message in enumerate(box, start=1):
            subject = str(message.get("subject", ""))
            sender = str(message.get("from", ""))
            body = _message_body(message)
            block = f"From: {sender}\nSubject: {subject}\n\n{body}".strip()
            texts.append(block)
            sections.append(
                {"title": subject or f"message {index}", "level": 1,
                 "path": [f"msg{index}"], "text": block, "start_line": 0,
                 "from": sender}
            )
    finally:
        box.close()
    return Document(
        text="\n\n---\n\n".join(texts),
        title=path.stem, media_type="application/mbox",
        sections=sections, metadata={"message_count": len(sections), "extractor": "mbox"},
    )


def _message_body(message: Any) -> str:
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode(part.get_content_charset() or "utf-8", "ignore")
        for part in message.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    return strip_html(payload.decode(part.get_content_charset() or "utf-8", "ignore"))
        return ""
    payload = message.get_payload(decode=True)
    if payload is None:
        return str(message.get_payload())
    text = payload.decode(message.get_content_charset() or "utf-8", "ignore")
    return strip_html(text) if message.get_content_type() == "text/html" else text


@register_loader(".msg", extra="msg")
def load_msg(path: str | Path, **_: Any) -> Document:
    """Load an Outlook ``.msg`` file (``vincio[msg]`` → extract-msg)."""
    path = Path(path)
    try:
        import extract_msg
    except ImportError as exc:  # pragma: no cover - optional dep
        raise LoaderError('.msg support requires extract-msg: pip install "vincio[msg]"') from exc
    message = extract_msg.Message(str(path))
    headers = {
        "from": message.sender or "",
        "to": message.to or "",
        "date": str(message.date or ""),
        "subject": message.subject or "",
    }
    body = message.body or ""
    return Document(
        text=f"Subject: {headers['subject']}\nFrom: {headers['from']}\nTo: {headers['to']}\n\n{body}",
        title=headers["subject"] or path.stem,
        media_type="application/vnd.ms-outlook",
        metadata={**headers, "extractor": "msg"},
    )
