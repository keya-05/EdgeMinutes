import io

from docx import Document

from src.document import MinutesDocument, build_docx, parse_signatories


def test_parse_signatories_variants():
    people = parse_signatories("Asha Rao, Chair\nVik - Secretary\nMeera | Treasurer\nZed")
    assert [(p.name, p.role) for p in people] == [
        ("Asha Rao", "Chair"), ("Vik", "Secretary"), ("Meera", "Treasurer"), ("Zed", "")]


def _text(data: bytes) -> str:
    d = Document(io.BytesIO(data))
    cells = [c.text for t in d.tables for r in t.rows for c in r.cells]
    return "\n".join([p.text for p in d.paragraphs] + cells)


def test_docx_has_signature_block_and_devanagari():
    m = MinutesDocument(
        summary="Budget approved.", decisions=["Approve budget"],
        action_items=[{"task": "Send report", "owner": "Asha", "deadline": "Friday"}],
        signatories=parse_signatories("Asha Rao, Chair\nVik, Secretary"),
        transcript="नमस्ते सर्वांना")
    text = _text(build_docx(m))
    assert "SIGNATURES" in text and "Asha Rao" in text and "Secretary" in text
    assert "Send report" in text and "नमस्ते सर्वांना" in text


def test_docx_without_signatories_has_no_signature_section():
    assert "SIGNATURES" not in _text(build_docx(MinutesDocument(summary="x")))


def test_hallucination_guard():
    from src.transcription import looks_hallucinated
    assert looks_hallucinated("अपने " * 12)
    assert not looks_hallucinated("Aditi will finish the budget report by Friday")


def test_hallucination_report_flags_invented_content():
    from src.evaluation import hallucination_report
    transcript = "Aditi will finish the budget report by Friday. We decided to launch next month."
    ok = hallucination_report(transcript, "Aditi will finish the budget report by Friday.")
    assert ok["hallucination_rate"] == 0.0
    bad = hallucination_report(transcript, "Aditi will finish the budget report by Friday. "
                                           "Rahul approved a 40 percent salary raise in Paris.")
    assert bad["flagged"] == 1 and bad["hallucination_rate"] == 0.5
    assert {"rahul", "paris", "40"} <= set(bad["rows"][1]["unsupported_terms"])


def test_rating_roundtrip(tmp_path):
    from src.evaluation import load_ratings, rating_summary, save_rating
    f = tmp_path / "r.csv"
    for v in (4, 2):
        save_rating(f, {"timestamp": "t", "meeting_id": "m", "rater": "a", "faithfulness": v,
                        "completeness": 3, "fluency": 5, "usefulness": 4})
    s = rating_summary(load_ratings(f))
    assert s["faithfulness"] == {"mean": 3.0, "n": 2} and s["fluency"]["mean"] == 5.0


def test_source_chunks_stay_short_and_complete():
    from src.evaluation import _source_chunks
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = _source_chunks(text, words=40)
    assert len(chunks) > 3 and all(len(c.split()) <= 45 for c in chunks)
    assert " ".join(chunks).count("Sentence") == 60
