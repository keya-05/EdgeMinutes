"""Build an editable Word (.docx) minutes document with a signature block.

The user edits every field in the app first, then downloads the .docx, which
can be edited again in Word / LibreOffice / Google Docs.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ACCENT = RGBColor(0x1F, 0x3A, 0x5F)
MUTED = RGBColor(0x6B, 0x72, 0x80)


@dataclass
class Signatory:
    name: str
    role: str = ""


@dataclass
class MinutesDocument:
    title: str = "Minutes of Meeting"
    date: str = ""
    venue: str = ""
    attendees: list[str] = field(default_factory=list)
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)   # task / owner / deadline
    open_questions: list[str] = field(default_factory=list)
    signatories: list[Signatory] = field(default_factory=list)
    transcript: str = ""                                       # appended when non-empty


def parse_lines(text: str) -> list[str]:
    """One item per non-empty line; leading bullet characters are dropped."""
    return [re.sub(r"^\s*[-*•]\s*", "", ln).strip() for ln in text.splitlines() if ln.strip()]


def parse_signatories(text: str) -> list[Signatory]:
    """One signatory per line: 'Name, Role' or 'Name - Role' or 'Name | Role'."""
    people = []
    for line in parse_lines(text):
        parts = re.split(r"\s*(?:\||,|\s[-–—]\s)\s*", line, maxsplit=1)
        people.append(Signatory(parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""))
    return people


# ------------------------------------------------------------------ helpers

def _set_cell_border(cell, **edges: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), edges.get(edge, "nil"))
        el.set(qn("w:sz"), "6")
        el.set(qn("w:color"), "444444")
        borders.append(el)


def _shade(cell, fill: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def _heading(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text.upper())
    run.bold = True
    run.font.size = Pt(10.5)
    run.font.color.rgb = ACCENT
    # thin rule under the heading
    p_pr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "4"), ("space", "1"), ("color", "C7CDD6")):
        bottom.set(qn(f"w:{k}"), v)
    bdr.append(bottom)
    p_pr.append(bdr)


def _bullets(doc: Document, items: list[str], empty: str) -> None:
    if not items:
        p = doc.add_paragraph()
        r = p.add_run(empty)
        r.italic = True
        r.font.color.rgb = MUTED
        return
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def _table_header(row, labels: list[str]) -> None:
    for cell, label in zip(row.cells, labels):
        cell.text = ""
        run = cell.paragraphs[0].add_run(label)
        run.bold = True
        run.font.size = Pt(9.5)
        _shade(cell, "E8ECF2")


# --------------------------------------------------------------------- build

def build_docx(m: MinutesDocument) -> bytes:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    r_fonts = normal.element.get_or_add_rPr().get_or_add_rFonts()
    r_fonts.set(qn("w:cs"), "Nirmala UI")       # Devanagari (Hindi / Marathi) fallback
    r_fonts.set(qn("w:eastAsia"), "Calibri")
    normal.paragraph_format.space_after = Pt(4)

    # Title block
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(m.title or "Minutes of Meeting")
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = ACCENT

    meta = [("Date", m.date), ("Venue", m.venue), ("Attendees", ", ".join(m.attendees))]
    meta = [(k, v) for k, v in meta if v.strip()]
    if meta:
        t = doc.add_table(rows=len(meta), cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.LEFT
        for row, (k, v) in zip(t.rows, meta):
            row.cells[0].width, row.cells[1].width = Cm(3), Cm(13.6)
            row.cells[0].text = ""
            lab = row.cells[0].paragraphs[0].add_run(k)
            lab.bold = True
            lab.font.color.rgb = MUTED
            row.cells[1].text = v

    _heading(doc, "Summary")
    doc.add_paragraph(m.summary.strip() or "Not specified")

    _heading(doc, "Key discussion points")
    _bullets(doc, m.key_points, "None identified")

    _heading(doc, "Decisions")
    _bullets(doc, m.decisions, "None recorded")

    _heading(doc, "Action items")
    items = [a for a in m.action_items if str(a.get("task", "")).strip()]
    if items:
        t = doc.add_table(rows=1, cols=3)
        t.style = "Table Grid"
        _table_header(t.rows[0], ["Task", "Owner", "Deadline"])
        for a in items:
            cells = t.add_row().cells
            cells[0].text = str(a.get("task", ""))
            cells[1].text = str(a.get("owner", "") or "Not specified")
            cells[2].text = str(a.get("deadline", "") or "Not specified")
        for row in t.rows:
            row.cells[0].width, row.cells[1].width, row.cells[2].width = Cm(9), Cm(3.8), Cm(3.8)
    else:
        _bullets(doc, [], "None recorded")

    if m.open_questions:
        _heading(doc, "Open questions")
        _bullets(doc, m.open_questions, "")

    # Signature block: two columns, a blank line to sign on, then name and role.
    if m.signatories:
        _heading(doc, "Signatures")
        intro = doc.add_paragraph("Confirmed as an accurate record of the meeting.")
        intro.runs[0].font.color.rgb = MUTED
        people = m.signatories
        rows = (len(people) + 1) // 2
        t = doc.add_table(rows=rows, cols=2)
        for i, person in enumerate(people):
            cell = t.rows[i // 2].cells[i % 2]
            cell.text = ""
            first = cell.paragraphs[0]
            first.paragraph_format.space_before = Pt(34)   # room to sign
            first.add_run("Signature: ").font.color.rgb = MUTED
            _set_cell_border(cell)
            name_p = cell.add_paragraph()
            name_p.paragraph_format.space_before = Pt(0)
            # rule above the name acts as the signature line
            p_pr = name_p._p.get_or_add_pPr()
            bdr = OxmlElement("w:pBdr")
            top = OxmlElement("w:top")
            for k, v in (("val", "single"), ("sz", "6"), ("space", "4"), ("color", "444444")):
                top.set(qn(f"w:{k}"), v)
            bdr.append(top)
            p_pr.append(bdr)
            nr = name_p.add_run(person.name)
            nr.bold = True
            if person.role:
                rp = cell.add_paragraph(person.role)
                rp.runs[0].font.color.rgb = MUTED
            dp = cell.add_paragraph("Date: ")
            dp.runs[0].font.color.rgb = MUTED
            dp.paragraph_format.space_after = Pt(10)
        # keep each signature row on one page
        for row in t.rows:
            tr_pr = row._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement("w:cantSplit"))

    if m.transcript.strip():
        doc.add_page_break()
        _heading(doc, "Appendix: transcript")
        for line in m.transcript.strip().splitlines():
            if line.strip():
                para = doc.add_paragraph(line.strip())
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                para.runs[0].font.size = Pt(9.5)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
