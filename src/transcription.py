"""Speech-to-text using faster-whisper (CTranslate2 port of OpenAI Whisper).

Whisper is a Transformer encoder-decoder: the encoder reads a log-Mel
spectrogram of the audio, the decoder autoregressively generates text tokens.
All inference here runs locally on the CPU of the machine running this code.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from functools import lru_cache

SUPPORTED_EXTENSIONS = ("wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "mov")
MODEL_SIZES = ("tiny", "base", "small", "medium")
DEFAULT_MODEL_SIZE = "small"
# Auto-detect is limited to these: Whisper's unrestricted detection often picks
# Urdu/Nepali/Romanian etc. for noisy Hindi or Marathi, which wrecks the transcript.
DEFAULT_CANDIDATE_LANGUAGES = ("en", "hi", "mr")


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be transcribed."""


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    audio_duration_s: float
    processing_time_s: float
    model_size: str
    # English translation of the speech (Whisper's "translate" task). Only set
    # when the spoken language is not English and translation was requested.
    english_text: str = ""
    peak_level: float = 0.0          # loudness of the input before normalisation (0..1)
    language_scores: dict = field(default_factory=dict)
    filters_relaxed: int = 0         # 0 = normal, 1-2 = silence filters had to be loosened


@lru_cache(maxsize=2)
def load_whisper_model(model_size: str = DEFAULT_MODEL_SIZE):
    """Load (and cache) a Whisper model. Downloads weights on first use."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc
    try:
        # int8 quantisation keeps memory low (~75 MB for tiny) on CPU.
        return WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as exc:  # network errors, corrupt cache, etc.
        raise TranscriptionError(
            f"Could not load Whisper '{model_size}' model. The first run needs "
            f"internet access to download it. Details: {exc}"
        ) from exc


def decode_audio(audio_path: str, sampling_rate: int = 16000):
    """Decode any PyAV-supported file to mono float32 at 16 kHz.

    faster-whisper's own decoder passes an argument removed in PyAV 19, so we
    decode here and give Whisper a NumPy array instead of a file path.
    """
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    chunks = []
    with av.open(audio_path, mode="r") as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def normalise_audio(audio, target_peak: float = 0.9, max_gain: float = 30.0):
    """Peak-normalise quiet recordings (laptop mics are often very quiet, and
    Whisper then hears noise and hallucinates). Returns (audio, original_peak)."""
    import numpy as np

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-4:
        return audio, peak
    gain = min(target_peak / peak, max_gain)
    return (audio * gain).astype(np.float32), peak


def detect_language(model, audio, candidates=DEFAULT_CANDIDATE_LANGUAGES):
    """Pick the most probable language among `candidates` only."""
    try:
        _, _, probs = model.detect_language(
            audio, vad_filter=True, language_detection_segments=3)
    except Exception:       # VAD found no speech at all
        _, _, probs = model.detect_language(audio, language_detection_segments=3)
    scores = {code: p for code, p in probs if code in candidates}
    if not scores:
        return candidates[0], {}
    best = max(scores, key=scores.get)
    return best, {k: round(v, 3) for k, v in scores.items()}


# Decoding attempts, strictest first. Whisper's silence filters can discard an
# entire quiet or accented recording, which surfaces as "no speech detected";
# each later attempt loosens them.
_PASSES = (
    dict(vad_filter=True),
    dict(vad_filter=True,
         vad_parameters=dict(threshold=0.25, min_silence_duration_ms=1500, speech_pad_ms=500)),
    dict(vad_filter=False, no_speech_threshold=None, log_prob_threshold=None,
         compression_ratio_threshold=None),
)


def looks_hallucinated(text: str) -> bool:
    """Whisper loops on silence/noise ("अपने अपने अपने ..."). Detect that."""
    words = text.split()
    return len(words) >= 6 and len(set(words)) / len(words) < 0.35


def _run_passes(model, audio, language, task="transcribe", first_pass=0):
    """Try decoding with progressively looser filters. Returns (text, info, pass_index)."""
    info = None
    for idx in range(first_pass, len(_PASSES)):
        segments, info = model.transcribe(
            audio, language=language, task=task, beam_size=1,
            condition_on_previous_text=False,   # stops one bad segment poisoning the rest
            **_PASSES[idx])
        # `segments` is a lazy generator; decoding happens while iterating.
        text = " ".join(seg.text.strip() for seg in segments).strip()
        # Loosened passes can invent repeating text from noise: do not accept that.
        if text and not (idx > 0 and looks_hallucinated(text)):
            return text, info, idx
    return "", info, len(_PASSES) - 1


def transcribe_audio(
    audio_path: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str | None = None,
    translate_to_english: bool = False,
    candidate_languages: tuple[str, ...] = DEFAULT_CANDIDATE_LANGUAGES,
) -> TranscriptionResult:
    """Transcribe an audio file. `language=None` auto-detects among
    `candidate_languages` (English, Hindi, Marathi by default).

    With `translate_to_english`, a second Whisper pass (task="translate") adds
    an English version of non-English speech, because the summarizer and the
    extraction rules only understand English."""
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise TranscriptionError("The audio file is empty or missing.")

    model = load_whisper_model(model_size)
    start = time.perf_counter()
    try:
        audio = decode_audio(audio_path)
    except Exception as exc:
        raise TranscriptionError(
            f"Could not decode the audio file (unsupported or corrupt format?). Details: {exc}"
        ) from exc
    if audio.size < 16000 * 0.5:
        raise TranscriptionError("The recording is empty or shorter than half a second.")
    audio, peak = normalise_audio(audio)
    if peak < 0.003:
        raise TranscriptionError(
            f"The recording is silent (peak level {peak:.4f}). Check that the right "
            "microphone is selected and not muted, then record again.")
    try:
        scores: dict = {}
        if language is None:
            language, scores = detect_language(model, audio, candidate_languages)
        text, info, used_pass = _run_passes(model, audio, language)
        english_text = ""
        if translate_to_english and text and info.language != "en":
            english_text, _, _ = _run_passes(model, audio, info.language, "translate", used_pass)
    except Exception as exc:
        raise TranscriptionError(
            f"Speech recognition failed. Details: {exc}"
        ) from exc

    if not text:
        rms = float((audio ** 2).mean() ** 0.5)
        raise TranscriptionError(
            "No speech was detected in the audio, even with the silence filters turned off. "
            f"(duration {audio.size / 16000:.1f}s, original peak level {peak:.3f}, "
            f"language {language}, model {model_size}.) Try the `medium` model, choose the "
            "language manually, or check that the recording really contains your voice.")
    return TranscriptionResult(
        text=text,
        language=info.language,
        filters_relaxed=used_pass,
        language_probability=round(info.language_probability, 3),
        audio_duration_s=round(info.duration, 2),
        processing_time_s=round(time.perf_counter() - start, 2),
        model_size=model_size,
        english_text=english_text,
        peak_level=round(peak, 4),
        language_scores=scores,
    )


def transcribe_bytes(
    data: bytes,
    suffix: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str | None = None,
    translate_to_english: bool = False,
) -> TranscriptionResult:
    """Write uploaded/recorded bytes to a temp file, transcribe, then delete it."""
    if not data:
        raise TranscriptionError("The recording or uploaded file is empty.")
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="edgeminutes_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return transcribe_audio(path, model_size=model_size, language=language,
                                translate_to_english=translate_to_english)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
