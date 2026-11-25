"""Lightweight evaluation: length statistics, compression, timing and ROUGE.

None of these metrics prove factual accuracy. ROUGE measures n-gram overlap
with a reference; a summary can score well and still contain a hallucinated
name or deadline. Human review against the transcript is still required.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w[\w'-]*\b", text or ""))


def compression_ratio(transcript: str, summary: str) -> float:
    """Summary words / transcript words (lower = more compressed)."""
    source = word_count(transcript)
    return round(word_count(summary) / source, 3) if source else 0.0


@contextmanager
def timer(results: dict, key: str):
    """Record elapsed seconds of a block into results[key]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        results[key] = round(time.perf_counter() - start, 2)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _f1(overlap: int, cand: int, ref: int) -> dict:
    p = overlap / cand if cand else 0.0
    r = overlap / ref if ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def _ngram_overlap(c: list[str], r: list[str], n: int) -> dict:
    from collections import Counter

    cg = Counter(tuple(c[i:i + n]) for i in range(len(c) - n + 1))
    rg = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
    return _f1(sum((cg & rg).values()), sum(cg.values()), sum(rg.values()))


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_scores(candidate: str, reference: str) -> tuple[dict, str]:
    """ROUGE-1/2/L. Uses the `rouge-score` package when installed, otherwise
    an equivalent built-in implementation (no stemming). Returns
    (scores, implementation_name)."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        raw = scorer.score(reference, candidate)
        scores = {k: {"precision": round(v.precision, 4), "recall": round(v.recall, 4),
                      "f1": round(v.fmeasure, 4)} for k, v in raw.items()}
        return scores, "rouge-score (Porter stemmer)"
    except ImportError:
        c, r = _tokens(candidate), _tokens(reference)
        scores = {
            "rouge1": _ngram_overlap(c, r, 1),
            "rouge2": _ngram_overlap(c, r, 2),
            "rougeL": _f1(_lcs(c, r), len(c), len(r)),
        }
        return scores, "built-in (no stemming)"


def summary_statistics(transcript: str, summary: str, timings: dict | None = None) -> dict:
    stats = {
        "transcript_words": word_count(transcript),
        "summary_words": word_count(summary),
        "compression_ratio": compression_ratio(transcript, summary),
    }
    stats.update(timings or {})
    return stats


# --------------------------------------------------------------------------
# BERTScore: meaning-based overlap with a reference summary
# --------------------------------------------------------------------------

BERTSCORE_MODEL = "distilbert-base-uncased"   # ~260 MB; roberta-large (default) is 1.4 GB
BERTSCORE_LAYER = 5


def bertscore(candidate: str, reference: str) -> dict:
    """BERTScore P/R/F1 (English), baseline-rescaled. Uses contextual
    embeddings, so paraphrases score higher than with ROUGE. Rescaling spreads
    the raw 0.8-1.0 band: unrelated text lands near 0 and a near-copy near 1."""
    scorer = _bert_scorer()
    p, r, f = scorer.score([candidate], [reference])
    return {"precision": round(float(p[0]), 4), "recall": round(float(r[0]), 4),
            "f1": round(float(f[0]), 4)}


_BERT = {}


def _bert_scorer():
    if "scorer" not in _BERT:
        from bert_score import BERTScorer

        _BERT["scorer"] = BERTScorer(model_type=BERTSCORE_MODEL, num_layers=BERTSCORE_LAYER,
                                     lang="en", rescale_with_baseline=True, device="cpu")
    return _BERT["scorer"]


# --------------------------------------------------------------------------
# Reference-free metrics: summary compared with its own transcript
# --------------------------------------------------------------------------

def _source_chunks(text: str, words: int = 110) -> list[str]:
    """Transcript pieces short enough for DistilBERT's 512-token limit."""
    from src.summarization import split_sentences

    chunks, cur, n = [], [], 0
    for sent in split_sentences(text) or [text]:
        k = len(sent.split())
        if cur and n + k > words:
            chunks.append(" ".join(cur)); cur, n = [], 0
        cur.append(sent); n += k
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def semantic_support(source: str, sentences: list[str]) -> list[float]:
    """For each summary sentence: the best baseline-rescaled BERTScore precision
    against any transcript chunk, i.e. how well its meaning is found in the
    transcript (about 0.8+ = clearly present, below 0.4 = weak or absent)."""
    scorer = _bert_scorer()
    chunks = _source_chunks(source)
    out = []
    for sentence in sentences:
        precision, _, _ = scorer.score([sentence] * len(chunks), chunks)
        out.append(round(max(0.0, float(precision.max())), 3))
    return out


def reference_free_metrics(source: str, summary: str) -> dict:
    """ROUGE and BERTScore computed with no human reference.

    * ROUGE of the summary against the transcript. Read the **precision**:
      the share of summary n-grams that occur in the transcript (an
      extractive summary scores near 1). Recall is low by construction because
      the transcript is much longer than the summary.
    * BERTScore semantic support per summary sentence (see `semantic_support`).
    Either part can fail independently; errors are returned, not raised."""
    out: dict = {"rouge": None, "rouge_impl": "", "bert_support": None,
                 "bert_rows": None, "bert_error": ""}
    out["rouge"], out["rouge_impl"] = rouge_scores(summary, source)
    sentences = _summary_sentences(summary)
    try:
        rows = semantic_support(source, sentences) if sentences else []
        out["bert_rows"] = rows
        out["bert_support"] = round(sum(rows) / len(rows), 3) if rows else None
    except Exception as exc:                       # offline, model missing, ...
        out["bert_error"] = str(exc)
    return out


# --------------------------------------------------------------------------
# Hallucination rate: is each summary sentence supported by the transcript?
# --------------------------------------------------------------------------

SUPPORT_THRESHOLD = 0.6      # a sentence needs >= 60% of its content words in the transcript


def _stem(word: str) -> str:
    return re.sub(r"(?:ing|ed|es|s|ly)$", "", word) if len(word) > 4 else word


def _content_stems(text: str) -> list[str]:
    from src.summarization import STOPWORDS

    return [_stem(w) for w in re.findall(r"[a-z0-9']+", text.lower())
            if w not in STOPWORDS and len(w) > 2]


def _summary_sentences(summary: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary.strip()) if s.strip()]


def hallucination_report(transcript: str, summary: str) -> dict:
    """Lexical faithfulness check of a summary against its source transcript.

    A summary sentence is flagged **unsupported** when fewer than 60% of its
    content words occur in the transcript, or it contains a number or a
    capitalised name that the transcript never mentions. Rate = flagged
    sentences / all sentences.

    This is a proxy, not a proof: abstractive paraphrase can be flagged even
    when correct, and a wrong claim made only from words the transcript does
    use will pass. Use it to find sentences worth checking by hand."""
    source_stems = set(_content_stems(transcript))
    source_tokens = set(re.findall(r"[a-z0-9']+", transcript.lower()))
    rows, total_words, missing_words = [], 0, 0
    for sentence in _summary_sentences(summary):
        stems = _content_stems(sentence)
        missing = [w for w in stems if w not in source_stems]
        numbers = [n for n in re.findall(r"\b\d+(?:[.,:]\d+)*\b", sentence)
                   if n not in source_tokens]
        names = [m for m in re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", sentence)
                 if m.lower() not in source_tokens]
        support = 1 - len(missing) / len(stems) if stems else 1.0
        flagged = support < SUPPORT_THRESHOLD or bool(numbers) or bool(names)
        total_words += len(stems)
        missing_words += len(missing)
        rows.append({"sentence": sentence, "support": round(support, 2),
                     "unsupported_terms": sorted({t.lower() for t in missing + numbers + names}),
                     "flagged": flagged})
    flagged_n = sum(r["flagged"] for r in rows)
    return {
        "sentences": len(rows),
        "flagged": flagged_n,
        "hallucination_rate": round(flagged_n / len(rows), 3) if rows else 0.0,
        "unsupported_word_rate": round(missing_words / total_words, 3) if total_words else 0.0,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# Human ratings (1-5), stored in a CSV so results survive restarts
# --------------------------------------------------------------------------

RATING_FIELDS = ("faithfulness", "completeness", "fluency", "usefulness")
RATING_COLUMNS = ("timestamp", "meeting_id", "rater", *RATING_FIELDS, "language",
                  "summary_method", "comment")


def save_rating(path, record: dict) -> None:
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=RATING_COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(record)


def load_ratings(path) -> list[dict]:
    import csv
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def rating_summary(rows: list[dict]) -> dict:
    """Mean and sample count per criterion over all saved ratings."""
    out = {}
    for field in RATING_FIELDS:
        vals = [float(r[field]) for r in rows if r.get(field) not in (None, "")]
        out[field] = {"mean": round(sum(vals) / len(vals), 2) if vals else None, "n": len(vals)}
    return out
