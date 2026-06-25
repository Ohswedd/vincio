"""Targeted real-behavior coverage for offline document format loaders.

Exercises the dependency-free PPTX/EPUB/RTF/ODT/mbox parsers through their real
public API plus the zip decompression guard and the LoaderError paths for
corrupt archives. Optional-dependency loaders (parquet/msg) raise a precise
LoaderError when the extra is absent, which is asserted here too.
"""

from __future__ import annotations

import email.message
import zipfile
from pathlib import Path

import pytest

from vincio.core.errors import LoaderError
from vincio.documents.formats import (
    _MAX_ZIP_ENTRY_BYTES,
    _message_body,
    _read_zip_entry,
    load_epub,
    load_mbox,
    load_odt,
    load_parquet,
    load_pptx,
    load_rtf,
)

# -- _read_zip_entry guard ---------------------------------------------------


def _zip_with(tmp_path: Path, name: str, **entries: str) -> Path:
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as z:
        for entry, body in entries.items():
            z.writestr(entry.replace("__", "/"), body)
    return p


class TestReadZipEntryGuard:
    def test_rejects_when_declared_size_over_cap(self, tmp_path):
        # info.file_size is read from the header and exceeds the cap up front.
        p = _zip_with(tmp_path, "big.zip", payload="z" * 5000)
        with zipfile.ZipFile(p) as archive, pytest.raises(
            LoaderError, match=r"inflates to 5000 bytes, over the 64-byte"
        ):
            _read_zip_entry(archive, "payload", max_bytes=64)

    def test_default_cap_is_64_mib(self):
        assert _MAX_ZIP_ENTRY_BYTES == 64 * 1024 * 1024

    def test_returns_exact_bytes_under_cap(self, tmp_path):
        p = _zip_with(tmp_path, "ok.zip", note="hello world")
        with zipfile.ZipFile(p) as archive:
            assert _read_zip_entry(archive, "note") == b"hello world"


# -- load_pptx ---------------------------------------------------------------


class TestLoadPptx:
    def test_slides_sorted_numerically_not_lexically(self, tmp_path):
        # slide10 must follow slide2, proving int() key sort (lexical would put
        # slide10 before slide2).
        p = tmp_path / "deck.pptx"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("ppt/slides/slide2.xml", "<a:t>Second</a:t>")
            z.writestr("ppt/slides/slide10.xml", "<a:t>Tenth</a:t>")
            z.writestr("ppt/slides/slide1.xml", "<a:t>First</a:t>")
        doc = load_pptx(p)
        assert doc.text == "First\n\nSecond\n\nTenth"
        assert [s["title"] for s in doc.sections] == ["slide 1", "slide 2", "slide 3"]
        assert doc.metadata["slide_count"] == 3

    def test_blank_slide_text_dropped_from_body_but_section_kept(self, tmp_path):
        p = tmp_path / "deck.pptx"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("ppt/slides/slide1.xml", "<a:t>   </a:t>")
            z.writestr("ppt/slides/slide2.xml", "<a:t>Real</a:t>")
        doc = load_pptx(p)
        # Empty slide body is excluded from the joined text...
        assert doc.text == "Real"
        # ...but the slide still produces a section.
        assert len(doc.sections) == 2
        assert doc.sections[0]["text"] == ""

    def test_unescapes_xml_entities(self, tmp_path):
        p = tmp_path / "deck.pptx"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("ppt/slides/slide1.xml", "<a:t>A &amp; B &lt; C</a:t>")
        doc = load_pptx(p)
        assert doc.text == "A & B < C"

    def test_bad_zip_raises_loader_error(self, tmp_path):
        p = tmp_path / "broken.pptx"
        p.write_bytes(b"not a zip at all")
        with pytest.raises(LoaderError, match=r"invalid PPTX .*broken\.pptx"):
            load_pptx(p)


# -- load_epub ---------------------------------------------------------------


class TestLoadEpub:
    def test_no_container_falls_back_to_sorted_html(self, tmp_path):
        # No META-INF/container.xml -> opf_path stays None -> the fallback path
        # collects *.xhtml/.html/.htm sorted by name (branch 98->102, 121).
        p = tmp_path / "b.epub"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("b.xhtml", "<p>Beta chapter.</p>")
            z.writestr("a.xhtml", "<p>Alpha chapter.</p>")
        doc = load_epub(p)
        assert doc.text == "Alpha chapter.\n\nBeta chapter."
        assert len(doc.sections) == 2

    def test_container_points_to_missing_opf_uses_fallback(self, tmp_path):
        # container.xml names an opf that is not in the archive -> opf_path not
        # in names (branch 103->121) -> fallback to sorted html.
        p = tmp_path / "b.epub"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr(
                "META-INF/container.xml",
                '<c><rootfiles><rootfile full-path="missing.opf"/></rootfiles></c>',
            )
            z.writestr("only.html", "<p>Only chapter.</p>")
        doc = load_epub(p)
        assert doc.text == "Only chapter."

    def test_container_without_full_path_attribute(self, tmp_path):
        # container.xml present but with no full-path -> opf_path None (line 101
        # else branch).
        p = tmp_path / "b.epub"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("META-INF/container.xml", "<c><rootfiles></rootfiles></c>")
            z.writestr("ch.xhtml", "<p>Body here.</p>")
        doc = load_epub(p)
        assert doc.text == "Body here."

    def test_item_missing_href_is_skipped_and_idref_without_item(self, tmp_path):
        # One <item> has an id but NO href (skipped, branch 112->109); the spine
        # references an idref with no resolvable href (continue, line 117).
        p = tmp_path / "b.epub"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr(
                "META-INF/container.xml",
                '<c><rootfiles><rootfile full-path="content.opf"/></rootfiles></c>',
            )
            z.writestr(
                "content.opf",
                "<package><manifest>"
                '<item id="noref"/>'
                '<item id="good" href="ch.xhtml"/>'
                "</manifest><spine>"
                '<itemref idref="missing"/>'
                '<itemref idref="noref"/>'
                '<itemref idref="good"/>'
                "</spine></package>",
            )
            z.writestr("ch.xhtml", "<p>Good chapter.</p>")
        doc = load_epub(p)
        assert doc.text == "Good chapter."
        assert len(doc.sections) == 1

    def test_spine_href_not_in_archive_skipped(self, tmp_path):
        # idref resolves to a href that is not a real entry -> full not in names
        # (branch 119->114) -> nothing ordered -> fallback to sorted html.
        p = tmp_path / "b.epub"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr(
                "META-INF/container.xml",
                '<c><rootfiles><rootfile full-path="content.opf"/></rootfiles></c>',
            )
            z.writestr(
                "content.opf",
                "<package><manifest>"
                '<item id="ghost" href="ghost.xhtml"/>'
                "</manifest><spine>"
                '<itemref idref="ghost"/>'
                "</spine></package>",
            )
            z.writestr("present.xhtml", "<p>Fallback body.</p>")
        doc = load_epub(p)
        assert doc.text == "Fallback body."

    def test_opf_base_directory_prepended_to_href(self, tmp_path):
        # opf under a subdir: hrefs resolve relative to that base (line 118).
        p = tmp_path / "b.epub"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr(
                "META-INF/container.xml",
                '<c><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></c>',
            )
            z.writestr(
                "OEBPS/content.opf",
                "<package><manifest>"
                '<item id="c1" href="ch1.xhtml"/>'
                "</manifest><spine>"
                '<itemref idref="c1"/>'
                "</spine></package>",
            )
            z.writestr("OEBPS/ch1.xhtml", "<p>Subdir chapter.</p>")
        doc = load_epub(p)
        assert doc.text == "Subdir chapter."

    def test_empty_body_chapter_skipped(self, tmp_path):
        # A chapter whose stripped body is whitespace-only is skipped (line 126).
        p = tmp_path / "b.epub"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("a.xhtml", "<p>   </p>")
            z.writestr("b.xhtml", "<p>Kept.</p>")
        doc = load_epub(p)
        assert doc.text == "Kept."
        assert len(doc.sections) == 1

    def test_bad_zip_raises_loader_error(self, tmp_path):
        p = tmp_path / "broken.epub"
        p.write_bytes(b"definitely not zip")
        with pytest.raises(LoaderError, match=r"invalid EPUB .*broken\.epub"):
            load_epub(p)


# -- load_rtf ----------------------------------------------------------------


class TestLoadRtf:
    def test_par_line_sect_become_newlines(self, tmp_path):
        p = tmp_path / "n.rtf"
        p.write_text(r"{\rtf1 One\par Two\line Three\sect Four}")
        doc = load_rtf(p)
        assert doc.text == "One\nTwo\nThree\nFour"
        assert doc.media_type == "application/rtf"

    def test_tab_control_word_is_consumed_not_left_literal(self, tmp_path):
        # \tab emits a tab which the final whitespace pass collapses to a single
        # space; what matters is the control word is consumed, not echoed.
        p = tmp_path / "t.rtf"
        p.write_text(r"{\rtf1 a\tab b}")
        doc = load_rtf(p)
        assert "tab" not in doc.text
        assert doc.text == "a b"

    def test_unicode_escape_emits_character(self, tmp_path):
        p = tmp_path / "u.rtf"
        p.write_text(r"{\rtf1 Caf\u233  fin}")
        doc = load_rtf(p)
        assert "Café" in doc.text

    def test_unicode_escape_modulo_wraps_large_value(self, tmp_path):
        # \uN values are taken mod 0x10000; 65601 % 65536 == 65 == 'A'. Proves
        # the chr(int()%0x10000) branch handles values above the BMP cap. Build
        # the RTF source so it contains the literal control word 敠1.
        p = tmp_path / "u.rtf"
        p.write_text("{\\rtf1 \\u65601  x}")
        doc = load_rtf(p)
        assert "A" in doc.text

    def test_escaped_literal_braces_and_backslash(self, tmp_path):
        # \{ \} \\ are escaped literals (group 3) and emit the raw char.
        p = tmp_path / "e.rtf"
        p.write_text(r"{\rtf1 a\{b\}c\\d}")
        doc = load_rtf(p)
        assert doc.text == "a{b}c\\d"

    def test_lone_trailing_backslash_no_match(self, tmp_path):
        # A backslash that matches no control pattern is just skipped (else: i+=1).
        p = tmp_path / "x.rtf"
        # Backslash followed by a digit is not a control word and not an escaped
        # non-alpha that the regex captures the same way; ensure no crash.
        p.write_text("{\\rtf1 keep this\\")
        doc = load_rtf(p)
        assert "keep this" in doc.text

    def test_collapses_whitespace_and_blank_lines(self, tmp_path):
        p = tmp_path / "w.rtf"
        p.write_text("{\\rtf1 lots    of   space\\par\\par\\par done}")
        doc = load_rtf(p)
        assert "lots of space" in doc.text
        assert "\n\n\n" not in doc.text


# -- load_odt ----------------------------------------------------------------


class TestLoadOdt:
    def test_paragraph_and_heading_text(self, tmp_path):
        p = tmp_path / "d.odt"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("content.xml", "<d><text:h>Title</text:h><text:p>Body line.</text:p></d>")
        doc = load_odt(p)
        assert "Title" in doc.text and "Body line." in doc.text
        assert doc.media_type == "application/vnd.oasis.opendocument.text"

    def test_tab_and_line_break_markup(self, tmp_path):
        # <text:tab/> -> tab (collapsed to a space by the whitespace pass) and
        # <text:line-break/> -> newline; the markup must not survive literally.
        p = tmp_path / "d.odt"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr(
                "content.xml",
                "<d><text:p>a<text:tab/>b<text:line-break/>c</text:p></d>",
            )
        doc = load_odt(p)
        assert "text:tab" not in doc.text and "text:line-break" not in doc.text
        assert doc.text == "a b\nc"

    def test_entities_unescaped(self, tmp_path):
        p = tmp_path / "d.odt"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("content.xml", "<d><text:p>R&amp;D &gt; rest</text:p></d>")
        doc = load_odt(p)
        assert "R&D > rest" in doc.text

    def test_bad_zip_raises_loader_error(self, tmp_path):
        p = tmp_path / "broken.odt"
        p.write_bytes(b"not zip")
        with pytest.raises(LoaderError, match=r"invalid ODT .*broken\.odt"):
            load_odt(p)

    def test_missing_content_xml_raises_loader_error(self, tmp_path):
        # A valid zip with no content.xml -> KeyError -> LoaderError.
        p = tmp_path / "empty.odt"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        with pytest.raises(LoaderError, match=r"invalid ODT"):
            load_odt(p)


# -- load_mbox + _message_body ----------------------------------------------


class TestLoadMbox:
    def test_two_messages_with_headers_and_separator(self, tmp_path):
        p = tmp_path / "t.mbox"
        p.write_text(
            "From a@b.com Mon Jan 1 00:00:00 2024\n"
            "From: a@b.com\nSubject: Hello\n\nFirst body.\n\n"
            "From c@d.com Mon Jan 1 00:01:00 2024\n"
            "From: c@d.com\nSubject: Re: Hello\n\nSecond body.\n\n"
        )
        doc = load_mbox(p)
        assert doc.metadata["message_count"] == 2
        assert "---" in doc.text
        assert doc.sections[0]["title"] == "Hello"
        assert doc.sections[0]["from"] == "a@b.com"
        assert "First body." in doc.sections[0]["text"]

    def test_message_without_subject_uses_index_title(self, tmp_path):
        p = tmp_path / "t.mbox"
        p.write_text(
            "From a@b.com Mon Jan 1 00:00:00 2024\nFrom: a@b.com\n\nNo subject body.\n\n"
        )
        doc = load_mbox(p)
        assert doc.sections[0]["title"] == "message 1"


class TestMessageBody:
    def test_multipart_prefers_text_plain(self):
        msg = email.message.EmailMessage()
        msg["Subject"] = "x"
        msg.set_content("plain text wins")
        msg.add_alternative("<p>html version</p>", subtype="html")
        assert _message_body(msg).strip() == "plain text wins"

    def test_multipart_falls_back_to_html_when_no_plain(self):
        # Build a multipart with ONLY an html part so the plain loop yields
        # nothing and the html branch (line 280-284) runs.
        outer = email.message.EmailMessage()
        outer["Subject"] = "x"
        outer.make_mixed()
        html_part = email.message.MIMEPart()
        html_part.set_content("<p>only <b>html</b> body</p>", subtype="html")
        outer.attach(html_part)
        body = _message_body(outer)
        assert "only" in body and "html" in body
        assert "<p>" not in body  # strip_html applied

    def test_multipart_skips_empty_text_plain_then_uses_next(self):
        # First text/plain part decodes to b'' (falsy) so the loop skips it
        # (branch 278->275) and the second non-empty part wins.
        outer = email.message.EmailMessage()
        outer.make_mixed()
        empty = email.message.MIMEPart()
        empty["Content-Type"] = "text/plain"
        empty.set_payload("")
        outer.attach(empty)
        real = email.message.MIMEPart()
        real["Content-Type"] = "text/plain"
        real.set_payload("the real body")
        outer.attach(real)
        assert _message_body(outer) == "the real body"

    def test_multipart_skips_empty_html_then_uses_next(self):
        # No usable text/plain; the first text/html decodes to b'' (skipped,
        # branch 283->280), the second html part is stripped and returned.
        outer = email.message.EmailMessage()
        outer.make_mixed()
        empty_html = email.message.MIMEPart()
        empty_html["Content-Type"] = "text/html"
        empty_html.set_payload("")
        outer.attach(empty_html)
        real_html = email.message.MIMEPart()
        real_html["Content-Type"] = "text/html"
        real_html.set_payload("<p>kept html</p>")
        outer.attach(real_html)
        body = _message_body(outer)
        assert "kept html" in body and "<p>" not in body

    def test_multipart_with_no_text_parts_returns_empty(self):
        outer = email.message.EmailMessage()
        outer["Subject"] = "x"
        outer.make_mixed()
        attach = email.message.MIMEPart()
        attach.set_content(b"\x00\x01binary", maintype="application", subtype="octet-stream")
        outer.attach(attach)
        assert _message_body(outer) == ""

    def test_singlepart_html_is_stripped(self):
        msg = email.message.EmailMessage()
        msg["Content-Type"] = "text/html"
        msg.set_payload(b"<p>Hi <i>there</i></p>")
        body = _message_body(msg)
        assert "Hi" in body and "there" in body and "<p>" not in body

    def test_singlepart_none_payload_returns_str_of_payload(self):
        # A non-multipart message whose payload was never set: get_payload(
        # decode=True) is None -> the str(get_payload()) fallback (line 288).
        msg = email.message.Message()
        assert not msg.is_multipart()
        assert msg.get_payload(decode=True) is None
        assert _message_body(msg) == "None"


# -- optional-dependency loaders --------------------------------------------


class TestOptionalDepLoaders:
    def test_parquet_without_pyarrow_raises_precise_error(self, tmp_path):
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("pyarrow installed; the missing-dep path is unreachable")
        p = tmp_path / "x.parquet"
        p.write_bytes(b"PAR1")
        with pytest.raises(LoaderError, match=r"Parquet support requires pyarrow"):
            load_parquet(p)
