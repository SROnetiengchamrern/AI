"""Speech-to-text with Faster-Whisper (multi-language)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_model = None
_model_size: str | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    language: str
    language_probability: float
    segments: list[Segment]

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


def _load_model(model_size: str = "base"):
    global _model, _model_size
    if _model is not None and _model_size == model_size:
        return _model

    from faster_whisper import WhisperModel

    # CPU-friendly defaults; use "cuda" + float16 if a GPU is available
    _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    _model_size = model_size
    return _model


def _run_whisper(
    model,
    audio_path: str,
    *,
    language: str | None,
    fast: bool,
    vad_filter: bool,
    initial_prompt: str | None = None,
) -> Transcript:
    kwargs: dict = {
        "language": language,
        "beam_size": 1 if fast else 5,
        "best_of": 1 if fast else 5,
        "vad_filter": vad_filter,
        "word_timestamps": False,
        "condition_on_previous_text": not fast,
    }
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    # Softer VAD so quiet / noisy speech is less likely to be dropped entirely
    if vad_filter:
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": 700,
            "speech_pad_ms": 400,
            "threshold": 0.35,
        }

    segments_iter, info = model.transcribe(audio_path, **kwargs)

    segments: list[Segment] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if text:
            segments.append(Segment(start=seg.start, end=seg.end, text=text))

    return Transcript(
        language=info.language or "unknown",
        language_probability=float(info.language_probability or 0),
        segments=segments,
    )


def transcribe(
    audio_path: str | Path,
    language: str | None = None,
    model_size: str = "tiny",
    *,
    fast: bool = True,
) -> Transcript:
    """
    Transcribe audio. language=None or 'auto' → auto-detect
    (English, Chinese, Thai, etc.).
    Fast: smaller beam for quicker CPU decode.
    Retries without VAD if the first pass finds no speech.
    """
    model = _load_model(model_size)
    lang = None if not language or language == "auto" else language
    path = str(audio_path)

    # 1) Normal pass with soft VAD
    result = _run_whisper(model, path, language=lang, fast=fast, vad_filter=True)
    if result.segments:
        return result

    # 2) Retry without VAD (helps quiet / music-mixed / soft speech)
    result = _run_whisper(model, path, language=lang, fast=fast, vad_filter=False)
    if result.segments:
        return result

    # 3) One more pass: slower beam, no VAD
    if fast:
        result = _run_whisper(model, path, language=lang, fast=False, vad_filter=False)
    return result


_SONG_PROMPTS: dict[str, str] = {
    "km": "បទចម្រៀង អក្សរខ្មែរ",
    "en": "song lyrics",
    "th": "เพลง เนื้อเพลง",
    "zh": "歌词",
    "ja": "歌詞",
    "ko": "노래 가사",
    "vi": "lời bài hát",
    "fr": "paroles de chanson",
    "es": "letra de canción",
    "de": "Songtext",
    "id": "lirik lagu",
    "hi": "गाने के बोल",
    "ru": "текст песни",
    "ar": "كلمات الأغنية",
    "ms": "lirik lagu",
}


def transcribe_song(
    audio_path: str | Path,
    language: str | None = None,
    model_size: str = "base",
) -> Transcript:
    """
    Transcribe separated vocal stem for Song AI.
    Uses full beam search, no VAD first (music bleed), optional language hint.
    """
    model = _load_model(model_size)
    lang = None if not language or language == "auto" else language
    path = str(audio_path)
    prompt = _SONG_PROMPTS.get(lang or "", "song lyrics vocals")

    if lang:
        result = _run_whisper(
            model,
            path,
            language=lang,
            fast=False,
            vad_filter=False,
            initial_prompt=prompt,
        )
        if result.segments:
            return result

    result = _run_whisper(
        model,
        path,
        language=None,
        fast=False,
        vad_filter=False,
        initial_prompt="song lyrics vocals",
    )
    if result.segments:
        return result

    result = _run_whisper(
        model,
        path,
        language=lang,
        fast=False,
        vad_filter=True,
        initial_prompt=prompt,
    )
    return result
