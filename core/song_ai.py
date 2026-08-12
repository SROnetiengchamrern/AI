"""Video song → remove ALL original audio → AI sings same lyrics (no translate).

Flow:
  1) Extract audio from the music video
  2) Separate vocals vs instrumental (Demucs) — for lyrics + melody only
  3) Discard original singer AND background music from the final mix
  4) Transcribe original lyrics (same language — never translate)
  5) Generate AI singing vocals (melody-matched when possible)
  6) Replace video audio with AI singing only (optional: mix music bed back)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import KHMER_VOICES, run_ffmpeg, sanitize_stem
from .extract import extract_audio, normalize_to_mp4
from .singing import synthesize_singing_track
from .subtitle import write_srt
from .text_clean import prepare_lyric_text
from .transcribe import Segment, transcribe_song
from .translate import merge_nearby_segments, merge_speak_segments, split_into_two_line_cues
from .tts import (
    get_duration_seconds,
    overlay_video_note,
    replace_audio,
    resolve_voice,
    burn_khmer_overlays,
)

# Edge TTS voices for Song AI (re-sing original lyrics)
SONG_AI_VOICES = {
    "Auto (match song language)": "auto",
    "English — Female (Jenny)": "en-US-JennyNeural",
    "English — Male (Guy)": "en-US-GuyNeural",
    "Khmer — Female (Sreymom)": "km-KH-SreymomNeural",
    "Khmer — Male (Piseth)": "km-KH-PisethNeural",
    "Chinese — Female": "zh-CN-XiaoxiaoNeural",
    "Thai — Female": "th-TH-PremwadeeNeural",
    "Vietnamese — Female": "vi-VN-HoaiMyNeural",
    "Japanese — Female": "ja-JP-NanamiNeural",
    "Korean — Female": "ko-KR-SunHiNeural",
    "Spanish — Female": "es-ES-ElviraNeural",
    "French — Female": "fr-FR-DeniseNeural",
}

_LANG_DEFAULT_VOICE = {
    "en": "en-US-JennyNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "th": "th-TH-PremwadeeNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
    "de": "de-DE-KatjaNeural",
    "id": "id-ID-GadisNeural",
    "km": "km-KH-SreymomNeural",
    "hi": "hi-IN-SwaraNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "ms": "ms-MY-YasminNeural",
}


def resolve_song_voice(voice_label: str, detected_language: str) -> str:
    """Pick Edge TTS voice for re-singing (auto = match detected lyric language)."""
    voice_id = SONG_AI_VOICES.get(voice_label, voice_label)
    if voice_id != "auto":
        if voice_label in KHMER_VOICES:
            return resolve_voice(voice_label)
        return voice_id
    lang = (detected_language or "en").split("-")[0].lower()
    return _LANG_DEFAULT_VOICE.get(lang, "en-US-JennyNeural")


def _lyrics_are_khmer(text: str) -> bool:
    return bool(re.search(r"[\u1780-\u17FF]", text or ""))


def _boost_vocal_stem(vocals: Path, out_path: Path) -> Path:
    """Normalize quiet Demucs vocal stem before Whisper."""
    out_path = Path(out_path)
    run_ffmpeg(
        [
            "-i",
            str(vocals),
            "-af",
            "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-y",
            str(out_path),
        ]
    )
    return out_path


def scrub_instrumental(music: Path, out_path: Path) -> Path:
    """
    Further suppress leftover singer bleed in the Demucs instrumental.
    Original vocals must not stay in the final mix — only the music bed.
    """
    music = Path(music)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Soft vocal-range duck + mild mid pullback (keeps drums/bass/harmony)
    attempts = [
        (
            "highpass=f=35,"
            "equalizer=f=900:t=q:w=1.0:g=-2.5,"
            "equalizer=f=2000:t=q:w=1.3:g=-4.0,"
            "equalizer=f=3500:t=q:w=1.2:g=-3.0,"
            "equalizer=f=5500:t=q:w=1.0:g=-1.5,"
            "stereotools=mlev=0.82:slev=1.05,"
            "loudnorm=I=-17:TP=-1.5:LRA=11"
        ),
        # Mono-safe fallback (no stereotools)
        (
            "highpass=f=35,"
            "equalizer=f=900:t=q:w=1.0:g=-2.5,"
            "equalizer=f=2000:t=q:w=1.3:g=-4.0,"
            "equalizer=f=3500:t=q:w=1.2:g=-3.0,"
            "loudnorm=I=-17:TP=-1.5:LRA=11"
        ),
    ]
    for af in attempts:
        try:
            run_ffmpeg(
                [
                    "-i",
                    str(music),
                    "-af",
                    af,
                    "-ar",
                    "48000",
                    "-y",
                    str(out_path),
                ]
            )
            if out_path.exists() and out_path.stat().st_size > 0:
                return out_path
        except Exception:
            if out_path.exists():
                out_path.unlink(missing_ok=True)

    return music


@dataclass
class SongAIResult:
    detected_language: str
    lyrics_original: str
    lyrics_ai: str
    vocals_path: Path | None
    music_path: Path | None
    ai_vocals_path: Path | None
    mix_path: Path | None
    output_video: Path | None
    lyrics_srt: Path | None
    work_dir: Path

    @property
    def lyrics_khmer(self) -> str:
        """Backward compatibility."""
        return self.lyrics_ai

    @property
    def khmer_srt(self) -> Path | None:
        return self.lyrics_srt


def _ensure_demucs():
    try:
        import demucs.separate  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Song AI needs Demucs for vocal/music separation.\n"
            "Install once:\n"
            "  pip install -U demucs torch torchaudio\n"
            f"Detail: {exc}"
        ) from exc


def separate_vocals_and_music(
    audio_path: Path,
    out_dir: Path,
    *,
    model_name: str = "htdemucs",
) -> tuple[Path, Path]:
    """
    Split mix into vocals + instrumental (no_vocals).

    - vocals.wav  → used only to read lyrics + melody (never mixed into output)
    - no_vocals   → music bed after original singer is removed
    """
    _ensure_demucs()
    import demucs.separate

    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Demucs writes: out_dir/<model>/<track>/vocals.wav + no_vocals.wav
    # --shifts 1 = slightly cleaner separation (slower, worth it for songs)
    demucs.separate.main(
        [
            "--two-stems",
            "vocals",
            "-n",
            model_name,
            "--shifts",
            "1",
            "-o",
            str(out_dir),
            str(audio_path),
        ]
    )

    track = audio_path.stem
    stem_dir = out_dir / model_name / track
    vocals = stem_dir / "vocals.wav"
    music = stem_dir / "no_vocals.wav"
    if not vocals.exists() or not music.exists():
        # Some versions nest differently — search
        found_v = list(out_dir.rglob("vocals.wav"))
        found_m = list(out_dir.rglob("no_vocals.wav"))
        if found_v and found_m:
            vocals, music = found_v[0], found_m[0]
        else:
            raise RuntimeError(
                f"Demucs finished but stems missing under {stem_dir}. "
                "Try another song file or reinstall demucs."
            )
    return vocals, music


def mix_ai_vocals_with_music(
    ai_vocals: Path,
    music: Path,
    out_path: Path,
    *,
    vocal_volume: float = 1.6,
    music_volume: float = 0.65,
) -> Path:
    """
    Blend AI singing + cleaned instrumental only.
    Never mix the original singer stem — that track is discarded after
    lyrics/melody extraction.
    """
    from .tts import get_duration_seconds

    ai_vocals = Path(ai_vocals)
    music = Path(music)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    music_dur = max(1.0, get_duration_seconds(music))
    vocal_dur = max(0.1, get_duration_seconds(ai_vocals))

    # Prep stems at same rate; pad/trim vocals to music length
    v_wav = out_path.parent / f"{out_path.stem}_vprep.wav"
    m_wav = out_path.parent / f"{out_path.stem}_mprep.wav"

    # Boost AI vocals so they sit above the bed (original singer is gone)
    v_gain = max(0.8, float(vocal_volume)) * 1.35
    m_gain = max(0.2, float(music_volume))

    run_ffmpeg(
        [
            "-i",
            str(ai_vocals),
            "-af",
            (
                f"aresample=48000,volume={v_gain:.3f},"
                f"apad=whole_dur={music_dur:.3f},atrim=0:{music_dur:.3f},"
                f"loudnorm=I=-14:TP=-1.0:LRA=11"
            ),
            "-ac",
            "1",
            "-y",
            str(v_wav),
        ]
    )
    run_ffmpeg(
        [
            "-i",
            str(music),
            "-af",
            (
                f"aresample=48000,volume={m_gain:.3f},"
                f"atrim=0:{music_dur:.3f},"
                f"loudnorm=I=-17:TP=-1.5:LRA=11"
            ),
            "-ac",
            "2",
            "-y",
            str(m_wav),
        ]
    )

    # Explicit weights: AI singing louder than bed — no original vocals input
    run_ffmpeg(
        [
            "-i",
            str(v_wav),
            "-i",
            str(m_wav),
            "-filter_complex",
            (
                f"[0:a]aformat=channel_layouts=stereo,volume=1.25[v];"
                f"[1:a]aformat=channel_layouts=stereo,volume=0.85[m];"
                f"[v][m]amix=inputs=2:duration=first:dropout_transition=0:"
                f"weights=1.25 0.85:normalize=0,"
                f"alimiter=limit=0.95[a]"
            ),
            "-map",
            "[a]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{music_dur:.3f}",
            "-y",
            str(out_path),
        ]
    )

    _ = vocal_dur
    return out_path


def export_ai_vocals_only(
    ai_vocals: Path,
    out_path: Path,
    *,
    duration: float,
    vocal_volume: float = 2.0,
) -> Path:
    """Normalize AI singing to a full-length stereo AAC track (no music bed)."""
    ai_vocals = Path(ai_vocals)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(1.0, float(duration))
    v_gain = max(0.8, float(vocal_volume))

    run_ffmpeg(
        [
            "-i",
            str(ai_vocals),
            "-af",
            (
                f"aresample=48000,volume={v_gain:.3f},"
                f"apad=whole_dur={dur:.3f},atrim=0:{dur:.3f},"
                f"loudnorm=I=-14:TP=-1.0:LRA=11,alimiter=limit=0.95"
            ),
            "-ac",
            "2",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{dur:.3f}",
            "-y",
            str(out_path),
        ]
    )
    return out_path


def create_song_ai_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    source_language: str = "auto",
    model_size: str = "base",
    voice_label: str = "Auto (match song language)",
    show_captions: bool = False,
    show_note: bool = True,
    video_note: str = "AI Song",
    vocal_volume: float = 2.0,
    music_volume: float = 0.65,
    keep_music_bed: bool = False,
    enable_singing: bool = True,
    fast: bool = True,
    progress_cb=None,
) -> SongAIResult:
    """
    Music video → strip ALL original audio → AI sings the same lyrics.
    No translation. Default final mix is AI vocals only (no background music).
    Set keep_music_bed=True to optionally mix the cleaned instrumental back in.
    """

    def tick(msg: str, frac: float) -> None:
        if progress_cb:
            progress_cb(frac, msg)

    source = Path(video_path)
    stem = sanitize_stem(source.name, max_len=40)
    work = Path(output_dir) / stem
    work.mkdir(parents=True, exist_ok=True)

    tick("Preparing video…", 0.04)
    video = normalize_to_mp4(source, work / f"{stem}_normalized.mp4", fast=fast)

    tick("Extracting song audio…", 0.10)
    mix_wav = extract_audio(video, work / "song_mix.wav")
    duration = get_duration_seconds(mix_wav)

    tick("Removing ALL original sound (singer + background)…", 0.22)
    vocals, music_raw = separate_vocals_and_music(mix_wav, work / "stems")
    # vocals / music_raw are references only unless keep_music_bed is on
    music = scrub_instrumental(music_raw, work / "music_bed_clean.wav")

    tick("Reading lyrics (no translation)…", 0.42)
    vocal_for_asr = _boost_vocal_stem(vocals, work / "vocals_boosted.wav")
    lang_hint = None if source_language == "auto" else source_language
    transcript = transcribe_song(
        vocal_for_asr,
        language=lang_hint,
        model_size=model_size,
    )
    if not transcript.segments:
        tick("Vocals quiet — reading lyrics from full mix…", 0.45)
        transcript = transcribe_song(
            mix_wav,
            language=lang_hint,
            model_size=model_size,
        )
    if not transcript.segments:
        raise RuntimeError(
            "No lyrics/speech detected in this song video.\n"
            "Try a clearer vocal track, set the lyric language, or use Whisper Base/Small."
        )

    tick(
        f"Language: {transcript.language} — same lyrics kept (no translate)…",
        0.55,
    )
    raw_segments = merge_nearby_segments(transcript.segments)
    speak_segments = [
        Segment(s.start, s.end, prepare_lyric_text(s.text))
        for s in raw_segments
        if prepare_lyric_text(s.text)
    ]
    speak_segments = merge_speak_segments(
        speak_segments,
        max_gap=0.8,
        max_chars=120,
        max_dur=8.0,
        prep=prepare_lyric_text,
    )
    if not speak_segments:
        raise RuntimeError("No lyrics detected to re-sing.")

    lyrics_original = "\n".join(s.text for s in transcript.segments)
    lyrics_ai = "\n".join(s.text for s in speak_segments)
    effective_lang = lang_hint or transcript.language
    tts_voice = resolve_song_voice(voice_label, effective_lang)
    tick(f"AI voice: {tts_voice} · lyric lines: {len(speak_segments)}", 0.62)

    tick("Converting sound → AI singing (same lyrics, no translate)…", 0.68)
    ai_vocals, spoken = synthesize_singing_track(
        speak_segments,
        tts_voice,
        vocals,  # pitch reference only — never mixed into final as original audio
        work,
        video_duration=max(duration, get_duration_seconds(music)),
        sing=bool(enable_singing),
        prep_text=prepare_lyric_text,
    )

    if keep_music_bed:
        tick("Optional: mixing AI vocals + music bed…", 0.82)
        mix_path = mix_ai_vocals_with_music(
            ai_vocals,
            music,
            work / f"{stem}_ai_song.m4a",
            vocal_volume=vocal_volume,
            music_volume=music_volume,
        )
    else:
        tick("Building AI singing only (background removed)…", 0.82)
        mix_path = export_ai_vocals_only(
            ai_vocals,
            work / f"{stem}_ai_song.m4a",
            duration=max(duration, get_duration_seconds(ai_vocals)),
            vocal_volume=vocal_volume,
        )

    tick("Replacing ALL video audio with AI singing…", 0.88)
    output_video = replace_audio(
        video,
        mix_path,
        work / f"{stem}_ai_song.mp4",
        fast=fast,
    )

    lyrics_srt: Path | None = None
    caption_segs = split_into_two_line_cues(spoken, max_chars=80)
    lyrics_srt = write_srt(caption_segs, work / f"{stem}_lyrics.srt")

    if show_captions and caption_segs and _lyrics_are_khmer(lyrics_ai):
        tick("Burning lyric captions…", 0.92)
        captioned = work / f"{stem}_ai_song_subs.mp4"
        output_video = burn_khmer_overlays(
            output_video,
            caption_segs,
            captioned,
            fast=fast,
            repage=False,
        )

    note = (video_note or "").strip() if show_note else ""
    if output_video and note:
        tick("Adding video note…", 0.96)
        noted = work / f"{output_video.stem}_noted.mp4"
        output_video = overlay_video_note(output_video, note, noted, fast=fast)

    tick("Done — original audio removed · AI sings (no translate).", 1.0)
    return SongAIResult(
        detected_language=transcript.language,
        lyrics_original=lyrics_original,
        lyrics_ai=lyrics_ai,
        vocals_path=vocals,
        music_path=music,
        ai_vocals_path=ai_vocals,
        mix_path=mix_path,
        output_video=output_video,
        lyrics_srt=lyrics_srt,
        work_dir=work,
    )
