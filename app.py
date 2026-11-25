"""EdgeMinutes: Streamlit UI. Run with:  streamlit run app.py

One scrolling page: record -> transcript and minutes -> document.
"""

from __future__ import annotations

import platform
import socket
from datetime import date
from functools import partial
from pathlib import Path

import pandas as pd
import streamlit as st

from src.document import MinutesDocument, build_docx, parse_lines, parse_signatories
from src.evaluation import timer
from src.meeting_minutes import (
    MINUTES_PROMPT_TEMPLATE,
    PROMPT_VARIANTS,
    build_prompt,
    generate_minutes,
)
from src.mic_recorder import mic_recorder
from src.summarization import (
    INSTRUCTION_MODEL,
    METHOD_EXTRACTIVE,
    METHOD_TRANSFORMER,
    SUMMARY_MODEL,
    load_summarizer,
    run_instruction_model,
    summarize_transcript,
)
from src.transcription import (
    DEFAULT_MODEL_SIZE,
    MODEL_SIZES,
    SUPPORTED_EXTENSIONS,
    TranscriptionError,
    load_whisper_model,
    transcribe_bytes,
)

ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "sample_data" / "sample_transcript.txt"
AUTO_LABEL = "Auto (English / Hindi / Marathi)"
LANGUAGES = {AUTO_LABEL: None, "English": "en", "Hindi": "hi", "Marathi": "mr",
             "Tamil": "ta", "Telugu": "te", "Bengali": "bn", "Gujarati": "gu",
             "French": "fr", "German": "de", "Spanish": "es"}
SOURCES = ["Record audio", "Upload audio", "Paste transcript", "Demo sample transcript"]

st.set_page_config(page_title="EdgeMinutes", page_icon=":material/graphic_eq:", layout="wide")
state = st.session_state

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap");
:root { --bg:#101013; --panel:#17171b; --panel-2:#1c1c21; --edge:#2a2a31; --ink:#e9e8ee; --muted:#8a8996;
        --accent:#a07bff; --accent-soft:rgba(160,123,255,.14); --warn:#ffb347;
        --mono:"JetBrains Mono",ui-monospace,"Cascadia Mono",Consolas,monospace; }
html { color-scheme: dark; }
html, body, .stApp, [data-testid="stAppViewContainer"] { background: var(--bg); }
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button {
  font-family: "Instrument Sans","Noto Sans Devanagari","Nirmala UI",system-ui,sans-serif; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1180px; padding-top: 2.4rem; padding-bottom: 6rem; }
[data-testid="stSidebar"] { background: #0d0d10; border-right: 1px solid var(--edge); }
[data-testid="stSidebar"] h2 { font-family: var(--mono); font-size: 11px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--muted); font-weight: 500; }

.brand { display:flex; align-items:baseline; gap:14px; margin-bottom:6px; }
.brand b { font-size: 26px; letter-spacing: -.02em; font-weight: 700; }
.brand span { font-family: var(--mono); font-size: 11px; letter-spacing: .14em; color: var(--muted); text-transform: uppercase; }
.lede { color: var(--muted); max-width: 62ch; margin: 0 0 28px; line-height: 1.55; }

.step { display:flex; align-items:center; gap:14px; margin: 56px 0 18px; }
.step .n { font-family: var(--mono); font-size: 11px; color: var(--accent); letter-spacing: .1em; }
.step .t { font-size: 20px; font-weight: 600; letter-spacing: -.01em; margin: 0; padding: 0; line-height: 1.2; }
.step .rule { flex: 1; height: 1px; background: var(--edge); }
.step.first { margin-top: 8px; }

.note { font-family: var(--mono); font-size: 11px; letter-spacing: .04em; color: var(--muted); }

/* inputs */
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input, [data-baseweb="select"] > div {
  background: var(--panel); border-color: var(--edge); }
[data-testid="stTextArea"] textarea { line-height: 1.6; }
[data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus { border-color: var(--accent); }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 6px; border: 1px solid var(--edge); background: var(--panel-2);
  transition: transform .1s, background .15s, border-color .15s; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #55546a; background: #24242b; }
.stButton > button:active, .stDownloadButton > button:active { transform: scale(.985); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
  background: var(--accent); border-color: var(--accent); color: #12101a; font-weight: 600; }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
  background: #b394ff; border-color: #b394ff; }
button:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

/* metric tiles */
[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--edge); border-radius: 6px; padding: 14px 16px; }
[data-testid="stMetricLabel"] p { font-family: var(--mono); font-size: 10.5px; letter-spacing: .08em;
  text-transform: uppercase; color: var(--muted); }
[data-testid="stMetricValue"] { font-family: var(--mono); font-weight: 500; font-size: 26px; }

[data-testid="stDataFrame"] { border: 1px solid var(--edge); border-radius: 6px; }
[data-testid="stExpander"] { border: 1px solid var(--edge); border-radius: 6px; background: var(--panel); }
[data-testid="stForm"] { border: 1px solid var(--edge); border-radius: 6px; background: var(--panel); }
[data-baseweb="tab-list"] { gap: 4px; }
.sub { font-family: var(--mono); font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted); margin: 26px 0 10px; }
iframe[title*="mic_recorder"] { border: 0; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


def step(number: str, title: str, first: bool = False) -> None:
    st.markdown(f'<div class="step{" first" if first else ""}"><span class="n">{number}</span>'
                f'<h2 class="t">{title}</h2><span class="rule"></span></div>',
                unsafe_allow_html=True)


def sub(text: str) -> None:
    st.markdown(f'<div class="sub">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    whisper_size = st.selectbox("Whisper model", MODEL_SIZES, index=MODEL_SIZES.index(DEFAULT_MODEL_SIZE),
                                help="tiny ≈ 75 MB (English only, really), base ≈ 145 MB, "
                                     "small ≈ 480 MB (minimum for Hindi/Marathi), "
                                     "medium ≈ 1.5 GB (best, slow).")
    language_name = st.selectbox(
        "Spoken language", list(LANGUAGES), index=0,
        help="If auto-detect picks the wrong language, choose Hindi or Marathi here.")
    method_label = st.radio(
        "Summarizer",
        ["Transformer (DistilBART, abstractive)", "Extractive fallback (no neural model)"],
        help="The extractive option selects existing sentences and needs no model download.",
    )
    method = METHOD_TRANSFORMER if method_label.startswith("Transformer") else METHOD_EXTRACTIVE
    translate = st.checkbox(
        "Translate non-English speech to English for the summary", value=True,
        help="The summarizer and the decision/action-item rules only understand English. "
             "With this on, Whisper also produces an English translation of Hindi, Marathi etc. "
             "and the minutes are built from it. The original-language transcript is kept.")
    if language_name in (AUTO_LABEL, "Hindi", "Marathi", "Tamil", "Telugu", "Bengali",
                         "Gujarati") and whisper_size in ("tiny", "base"):
        st.warning("Hindi/Marathi accuracy is poor with tiny/base. Use **small** (or **medium**, "
                   "slower) for usable results.")
    max_chunks = st.slider("Max transcript chunks to summarise", 1, 8, 4,
                           help="Each chunk is ~900 tokens. Larger values are slower on CPU.")

    with st.expander("Execution information", expanded=False):
        st.markdown(f"""
**This app's Python process runs on:** `{socket.gethostname()}`
({platform.system()} {platform.release()}, {platform.machine()}).

- **Transcription:** faster-whisper `{whisper_size}` (int8) on **this computer's CPU**.
- **Summarization:** `{SUMMARY_MODEL}` on **this computer's CPU**.
- **Extraction:** rule-based Python on this computer.
- **Remote servers:** none. Audio and transcripts are not sent to any API.
- **Internet:** only for the first download of model weights from Hugging Face.
- **Phone:** if you open this page from a phone, the phone is only a browser
  (upload/record). **All AI inference still runs on this computer**, not on the phone.
""")

# ------------------------------------------------------------------ header
st.markdown('<div class="brand"><b>EdgeMinutes</b><span>Meeting minutes, on this machine</span></div>'
            '<p class="lede">Record a meeting in English, Hindi or Marathi. The transcript and a signable '
            'minutes document appear below, with no cloud and no API keys.</p>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------ 01 record
step("01", "Record", first=True)
source = st.segmented_control("Input source", SOURCES, default="Record audio",
                              label_visibility="collapsed") or "Record audio"
audio_bytes, audio_suffix, typed_text, rec_info = None, None, "", None

if source == "Record audio":
    recording = mic_recorder(key="mic")
    if recording is not None:
        audio_bytes, audio_suffix, rec_info = recording
    st.markdown('<span class="note">Browsers allow the microphone only on localhost or HTTPS pages. '
                'Processing starts by itself when you stop recording.</span>', unsafe_allow_html=True)
    if hasattr(st, "audio_input"):
        with st.expander("Alternative: Streamlit's built-in recorder"):
            # sample_rate=None keeps the browser's native rate: resampling to 16 kHz
            # in the browser fails on some Android phones, and Whisper resamples anyway.
            rec = st.audio_input("Built-in recorder", sample_rate=None)
            if rec is not None:
                audio_bytes, audio_suffix, rec_info = rec.getvalue(), ".wav", None
elif source == "Upload audio":
    up = st.file_uploader("Upload meeting audio", type=list(SUPPORTED_EXTENSIONS))
    if up is not None:
        audio_bytes, audio_suffix = up.getvalue(), Path(up.name).suffix.lower()
        st.audio(audio_bytes)
elif source == "Paste transcript":
    typed_text = st.text_area(
        "Paste a transcript (optionally with `Name: text` speaker labels)", height=220,
        placeholder="Aditi: I will finish the report by Friday.\nRahul: We decided to ...")
else:
    typed_text = SAMPLE_PATH.read_text(encoding="utf-8")
    st.info("**DEMO DATA:** this is a hand-written sample transcript, not recognised speech. "
            "The summary generated from it is real model output.")
    with st.expander("View sample transcript"):
        st.text(typed_text)


# ------------------------------------------------------------------ pipeline
def run_pipeline() -> None:
    timings: dict = {}
    transcript, asr_info = "", None
    try:
        if audio_bytes is not None:
            with st.status("Processing audio…", expanded=True) as status:
                st.write(f"Loading Whisper `{whisper_size}` (first run downloads the model)…")
                load_whisper_model(whisper_size)
                st.write("Transcribing on CPU…")
                asr_info = transcribe_bytes(audio_bytes, audio_suffix, whisper_size,
                                            LANGUAGES[language_name],
                                            translate_to_english=translate)
                transcript = asr_info.text
                timings["transcription_s"] = asr_info.processing_time_s
                status.update(label="Transcription complete", state="complete")
        elif source in ("Record audio", "Upload audio"):
            st.error("No audio yet. Record or upload a file first.")
            return
        else:
            transcript = typed_text.strip()
            if not transcript:
                st.error("The transcript is empty. Paste some text first.")
                return

        if method == METHOD_TRANSFORMER:
            with st.spinner(f"Loading {SUMMARY_MODEL} (first run downloads ~1.2 GB)…"):
                try:
                    load_summarizer()
                except Exception as exc:  # summarize_transcript falls back and notes it
                    st.warning(f"Transformer model unavailable: {exc}")
        with st.spinner("Summarising and extracting decisions and action items…"):
            summarizer = partial(summarize_transcript, method=method, max_chunks=max_chunks)
            # Minutes are built from English text: the translation for non-English audio.
            minutes_source = (asr_info.english_text if asr_info and asr_info.english_text
                              else transcript)
            with timer(timings, "minutes_generation_s"):
                minutes = generate_minutes(minutes_source, summarizer=summarizer)
        if asr_info and asr_info.language != "en" and not asr_info.english_text:
            minutes.notes.append(
                f"Speech was detected as '{asr_info.language}' but not translated, so the "
                "English-only summarizer and extraction rules ran on non-English text. "
                "Enable translation in the sidebar.")
        state.update(transcript=transcript,
                     english_text=minutes_source if minutes_source != transcript else "",
                     minutes=minutes, asr=asr_info, timings=timings,
                     is_demo=source.startswith("Demo"), doc_gen=state.get("doc_gen", 0) + 1)
    except TranscriptionError as exc:
        st.error(f"Transcription error: {exc}")
        st.info("You can still use **Paste transcript** or **Demo sample transcript**.")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")


new_recording = (source == "Record audio" and audio_bytes is not None and rec_info is not None
                 and rec_info.get("id") != state.get("done_rec_id"))
label = "Process again" if source == "Record audio" else "Generate Meeting Minutes"
clicked = False
if source != "Record audio" or audio_bytes is not None:
    clicked = st.button(label, type="primary" if source != "Record audio" else "secondary",
                        width="stretch" if source != "Record audio" else "content")
if new_recording:
    state.done_rec_id = rec_info["id"]       # set first so a failure is not retried on every rerun
if clicked or new_recording:
    run_pipeline()

# ------------------------------------------------------------------ results
if "minutes" in state:
    minutes, transcript = state.minutes, state.transcript
    source_text = state.get("english_text") or transcript   # the text the summary was built from
    if state.is_demo:
        st.info("Results below were generated from the **demo sample transcript**.")

    # ============================================================ 02 transcript
    step("02", "Transcript and minutes")
    col_t, col_m = st.columns([2, 3], gap="large")
    with col_t:
        sub("Transcript")
        if state.asr:
            a = state.asr
            st.caption(f"Whisper {a.model_size} · language `{a.language}` "
                       f"(p={a.language_probability}) · audio {a.audio_duration_s}s · "
                       f"processed in {a.processing_time_s}s")
            if a.language_scores:
                st.caption("Language scores: " + ", ".join(
                    f"{k} {v:.2f}" for k, v in sorted(a.language_scores.items(), key=lambda x: -x[1])))
            if a.filters_relaxed:
                st.info("Whisper's silence filter initially found no speech in this recording, so "
                        "it was re-run with the filter loosened. Check the transcript carefully.")
            if a.peak_level < 0.02:
                st.warning(f"Your recording was very quiet (peak {a.peak_level:.3f}); it was "
                           "amplified automatically, but accuracy suffers. Move closer to the "
                           "mic, check the input device, or upload a clearer file.")
            if (a.language == "en" and language_name != "English"
                    and a.language_scores.get("en", 1.0) < 0.9):
                st.info("Detected **English**. If you actually spoke Hindi or Marathi, pick that "
                        "language in the sidebar and process again.")
        if state.get("english_text"):
            tab_orig, tab_en = st.tabs([f"Original ({state.asr.language})", "English translation"])
            tab_orig.text_area("transcript", transcript, height=380, label_visibility="collapsed")
            tab_en.text_area("english", state.english_text, height=380, label_visibility="collapsed")
            st.caption("The minutes were generated from the English translation. "
                       "Machine translation can be wrong: check names, dates and numbers.")
        else:
            st.text_area("transcript", transcript, height=420, label_visibility="collapsed")

    with col_m:
        sub("Meeting minutes")
        method_name = ("DistilBART transformer (abstractive)" if minutes.summary_method == METHOD_TRANSFORMER
                       else "Extractive fallback — selected sentences, not a neural model")
        for note in minutes.notes:
            st.warning(note)
        st.markdown("**Meeting summary**")
        st.write(minutes.summary)
        st.caption(f"Summary method: {method_name}")
        if minutes.participants:
            st.markdown(f"**Participants:** {', '.join(minutes.participants)}")
        st.markdown("**Key discussion points**")
        st.markdown("\n".join(f"- {p}" for p in minutes.key_points) or "- None identified")
        st.markdown("**Decisions**")
        st.markdown("\n".join(f"- {d}" for d in minutes.decisions) or "- None recorded")
        st.markdown("**Action items**")
        if minutes.action_items:
            st.dataframe(pd.DataFrame([{"Task": i.task, "Owner": i.owner, "Deadline": i.deadline,
                                        "Source sentence": i.source} for i in minutes.action_items]),
                         hide_index=True, width="stretch")
        else:
            st.markdown("- None recorded")
        if minutes.open_questions:
            st.markdown("**Open questions / unresolved**")
            st.markdown("\n".join(f"- {q}" for q in minutes.open_questions))
        st.caption("Decisions, action items and open questions are rule-based extractions "
                   "copied from the transcript. AI-generated draft: review before sharing.")

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download transcript (.txt)", transcript, "transcript.txt", "text/plain",
                       width="stretch")
    d2.download_button("Download minutes (.txt)", minutes.to_text(), "meeting_minutes.txt",
                       "text/plain", width="stretch")
    d3.download_button("Download minutes (.json)", minutes.to_json(), "meeting_minutes.json",
                       "application/json", width="stretch")

    # ============================================================ 03 document
    step("03", "Minutes document")
    st.caption("Edit anything below, list the people who must sign, then download a Word file "
               "you can keep editing.")
    g = state.get("doc_gen", 0)          # new key suffix per generation resets the form
    dc1, dc2, dc3 = st.columns([2, 1, 1])
    doc_title = dc1.text_input("Title", "Minutes of Meeting", key=f"d_title_{g}")
    doc_date = dc2.text_input("Date", date.today().strftime("%d %B %Y"), key=f"d_date_{g}")
    doc_venue = dc3.text_input("Venue", "", key=f"d_venue_{g}")
    doc_attendees = st.text_input("Attendees (comma separated)", ", ".join(minutes.participants),
                                  key=f"d_att_{g}")
    doc_summary = st.text_area("Summary", minutes.summary, height=120, key=f"d_sum_{g}")
    e1, e2 = st.columns(2)
    doc_points = e1.text_area("Key discussion points (one per line)", "\n".join(minutes.key_points),
                              height=140, key=f"d_kp_{g}")
    doc_decisions = e2.text_area("Decisions (one per line)", "\n".join(minutes.decisions),
                                 height=140, key=f"d_dec_{g}")
    st.markdown("**Action items**")
    doc_actions = st.data_editor(
        pd.DataFrame([{"task": i.task, "owner": i.owner, "deadline": i.deadline}
                      for i in minutes.action_items], columns=["task", "owner", "deadline"]),
        num_rows="dynamic", hide_index=True, width="stretch", key=f"d_act_{g}",
        column_config={"task": "Task", "owner": "Owner", "deadline": "Deadline"})
    doc_questions = st.text_area("Open questions (one per line)", "\n".join(minutes.open_questions),
                                 height=90, key=f"d_oq_{g}")
    doc_sign = st.text_area(
        "Signatures required from (one person per line: `Name, Role`)", "", height=110,
        key=f"d_sign_{g}", placeholder="Aditi Sharma, Chairperson\nRahul Verma, Secretary")
    doc_appendix = st.checkbox("Attach full transcript as an appendix", key=f"d_app_{g}")

    appendix = ""
    if doc_appendix:
        appendix = transcript
        if state.get("english_text"):
            appendix += f"\n\n--- English translation ---\n{state.english_text}"
    signatories = parse_signatories(doc_sign)
    docx_bytes = build_docx(MinutesDocument(
        title=doc_title, date=doc_date, venue=doc_venue,
        attendees=[a.strip() for a in doc_attendees.split(",") if a.strip()],
        summary=doc_summary, key_points=parse_lines(doc_points), decisions=parse_lines(doc_decisions),
        action_items=doc_actions.fillna("").to_dict("records"),
        open_questions=parse_lines(doc_questions), signatories=signatories,
        transcript=appendix))
    st.download_button("Download minutes document (.docx)", docx_bytes, "meeting_minutes.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       type="primary", width="stretch")
    if not signatories:
        st.caption("No signatories entered: the document will have no signature block.")

    # ============================================================ prompt lab
    with st.expander("Prompt engineering lab"):
        st.markdown(
            f"`{SUMMARY_MODEL}` is a summarization model **without instruction tuning**: it receives "
            "only the transcript, never a prompt. The structured prompt below is the template "
            f"designed for instruction-tuned models. It is sent, verbatim, to `{INSTRUCTION_MODEL}` "
            "in the comparison below.")
        st.code(build_prompt(source_text, MINUTES_PROMPT_TEMPLATE), language="text")
        if st.button(f"Compare two prompts with {INSTRUCTION_MODEL} (downloads ~300 MB first time)"):
            cols = st.columns(len(PROMPT_VARIANTS))
            for col, (name, template) in zip(cols, PROMPT_VARIANTS.items()):
                with col, st.spinner(f"Running {name}…"):
                    try:
                        out, truncated, secs = run_instruction_model(build_prompt(source_text, template))
                        st.markdown(f"**{name}**")
                        st.code(template.split("\n")[0][:300], language="text")
                        st.write(out or "(empty output)")
                        st.caption(f"{secs}s on CPU" + (" · input truncated to 512 tokens" if truncated else ""))
                    except Exception as exc:
                        st.error(f"Instruction model unavailable: {exc}")
