"""Melody-guided speech → singing conversion for Song AI.

Takes Edge TTS speech + original vocal stem pitch, and rebuilds audio so
lyrics follow the song melody (librosa F0 + granular pitch mapping).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from . import run_ffmpeg
from .text_clean import prepare_lyric_text, prepare_speak_text
from .transcribe import Segment
from .tts import (
    _concat_speech_timeline,
    _synthesize_many,
    _tts_clip_ok,
    resolve_voice,
)


def _fix_len(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n]
    return np.pad(y, (0, max(0, n - len(y))))


def _load_mono(path: Path, sr: int, *, offset: float = 0.0, duration: float | None = None):
    import librosa

    y, _ = librosa.load(
        str(path),
        sr=sr,
        mono=True,
        offset=max(0.0, offset),
        duration=duration,
        dtype=np.float32,
    )
    return np.ascontiguousarray(y)


def _save_wav(path: Path, y: np.ndarray, sr: int) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y64 = np.asarray(y, dtype=np.float64)
    peak = float(np.max(np.abs(y64)) + 1e-9)
    y64 = y64 / peak * 0.92
    sf.write(str(path), y64, sr)
    return path


def _safe_f0(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    try:
        f0 = librosa.yin(y, fmin=65, fmax=1000, sr=sr, frame_length=2048)
    except Exception:
        f0, _, _ = librosa.pyin(y, fmin=65, fmax=1000, sr=sr)
        f0 = np.nan_to_num(np.asarray(f0, dtype=np.float32), nan=0.0)
    f0 = np.asarray(f0, dtype=np.float32)
    f0[~np.isfinite(f0)] = 0.0
    return f0


def _median_voiced(f0: np.ndarray, default: float = 220.0) -> float:
    voiced = f0[f0 > 1.0]
    if len(voiced) == 0:
        return default
    return float(np.median(voiced))


def _apply_vibrato(y: np.ndarray, sr: int, *, hz: float = 5.5, depth: float = 0.012) -> np.ndarray:
    """Mild vibrato via time-varying resampling (singing character)."""
    n = len(y)
    if n < sr // 5:
        return y
    t = np.arange(n, dtype=np.float64) / sr
    # Instantaneous playback speed modulation
    speed = 1.0 + depth * np.sin(2.0 * np.pi * hz * t)
    # Cumulative position map
    pos = np.cumsum(speed)
    pos = pos / pos[-1] * (n - 1)
    return np.interp(np.arange(n), pos, y).astype(np.float32)


def _granular_melody_follow(
    speech: np.ndarray,
    ref: np.ndarray,
    sr: int,
    *,
    grain_ms: float = 60.0,
    hop_ms: float = 30.0,
) -> np.ndarray:
    """
    Follow reference melody by pitch-shifting short grains of speech
    toward local reference F0.
    """
    import librosa

    grain = max(512, int(sr * grain_ms / 1000.0))
    hop = max(256, int(sr * hop_ms / 1000.0))
    n = len(speech)
    out = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    window = np.hanning(grain).astype(np.float32)

    f0_ref = _safe_f0(ref, sr)
    f0_src = _safe_f0(speech, sr)
    # Map frame index → sample approx (yin uses hop ~ frame_length//4)
    # Use librosa frames_to_samples with default hop from yin (~512)
    hop_f0 = 512

    def f0_at(f0: np.ndarray, sample_idx: int) -> float:
        fi = int(sample_idx / hop_f0)
        fi = max(0, min(len(f0) - 1, fi))
        v = float(f0[fi])
        return v if v > 1.0 else 0.0

    for start in range(0, max(1, n - grain // 2), hop):
        end = min(n, start + grain)
        chunk = speech[start:end]
        if len(chunk) < grain // 3:
            break
        if len(chunk) < grain:
            pad = np.zeros(grain - len(chunk), dtype=np.float32)
            chunk_w = np.concatenate([chunk, pad])
        else:
            chunk_w = chunk[:grain]

        mid = start + len(chunk_w) // 2
        src_f = f0_at(f0_src, mid)
        ref_f = f0_at(f0_ref, mid)
        if src_f <= 1.0:
            src_f = _median_voiced(f0_src)
        if ref_f <= 1.0:
            ref_f = _median_voiced(f0_ref)

        semitones = 12.0 * np.log2(max(ref_f, 1.0) / max(src_f, 1.0))
        semitones = float(np.clip(semitones, -7.0, 7.0))

        if abs(semitones) >= 0.15:
            try:
                shifted = librosa.effects.pitch_shift(
                    chunk_w, sr=sr, n_steps=semitones, bins_per_octave=12
                )
            except Exception:
                shifted = chunk_w
        else:
            shifted = chunk_w

        shifted = shifted[:grain] * window
        out_end = min(n, start + grain)
        sl = out_end - start
        out[start:out_end] += shifted[:sl]
        weight[start:out_end] += window[:sl]

    weight = np.maximum(weight, 1e-6)
    return out / weight


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y.astype(np.float64))) + 1e-12))


def _ensure_loud(y: np.ndarray, *, target_peak: float = 0.92) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y)) + 1e-9)
    y = y / peak * target_peak
    # If still too quiet on average, soft boost
    r = _rms(y)
    if r < 0.04:
        y = np.clip(y * (0.08 / max(r, 1e-9)), -1.0, 1.0).astype(np.float32)
        peak = float(np.max(np.abs(y)) + 1e-9)
        y = y / peak * target_peak
    return y


def speech_to_singing(
    tts_path: Path,
    ref_vocals_path: Path,
    *,
    start: float,
    end: float,
    out_path: Path,
    sr: int = 22050,
) -> Path:
    """
    Convert one TTS speech clip into a sung phrase using the original
    vocal melody in [start, end].
    Always keeps audible lyric energy (falls back to loud stretched TTS).
    """
    import librosa

    tts_path = Path(tts_path)
    out_path = Path(out_path)
    dur = max(0.25, float(end) - float(start))

    ref = _load_mono(ref_vocals_path, sr, offset=start, duration=dur)
    tts = _load_mono(tts_path, sr)
    if len(ref) < int(0.12 * sr) or len(tts) < int(0.08 * sr):
        return _time_stretch_to_duration(tts_path, dur, out_path, sr=sr)

    target_n = len(ref)
    stretch_rate = max(0.55, min(1.85, len(tts) / max(1, target_n)))
    stretched = librosa.effects.time_stretch(tts, rate=stretch_rate)
    stretched = _fix_len(stretched.astype(np.float32), target_n)
    stretched = _ensure_loud(stretched)
    ref = _fix_len(ref.astype(np.float32), target_n)

    # 1) Phrase-level pitch toward song median
    src_m = _median_voiced(_safe_f0(stretched, sr))
    ref_m = _median_voiced(_safe_f0(ref, sr))
    global_semi = float(np.clip(12.0 * np.log2(ref_m / max(src_m, 1.0)), -8.0, 8.0))
    pitched = stretched
    if abs(global_semi) >= 0.2:
        try:
            pitched = librosa.effects.pitch_shift(
                stretched, sr=sr, n_steps=global_semi, bins_per_octave=12
            )
            pitched = _fix_len(pitched.astype(np.float32), target_n)
        except Exception:
            pitched = stretched

    # 2) Local melody follow
    try:
        sung = _granular_melody_follow(pitched, ref, sr)
        sung = _fix_len(sung.astype(np.float32), target_n)
    except Exception:
        sung = pitched

    # If melody convert wiped energy, blend back loud TTS so singing stays audible
    if _rms(sung) < 0.025 or _rms(sung) < 0.35 * _rms(stretched):
        sung = (0.65 * _ensure_loud(stretched) + 0.35 * _ensure_loud(sung)).astype(np.float32)

    sung = _apply_vibrato(_ensure_loud(sung), sr)
    try:
        sung = librosa.effects.preemphasis(sung, coef=0.10)
    except Exception:
        pass
    sung = _ensure_loud(sung)

    return _save_wav(out_path, sung, sr)


def _time_stretch_to_duration(src: Path, duration: float, out_path: Path, *, sr: int) -> Path:
    import librosa

    y = _load_mono(src, sr)
    target_n = max(1, int(duration * sr))
    rate = max(0.55, min(1.85, len(y) / target_n))
    stretched = librosa.effects.time_stretch(y, rate=rate)
    stretched = _fix_len(stretched.astype(np.float32), target_n)
    stretched = _ensure_loud(stretched)
    return _save_wav(out_path, stretched, sr)


def synthesize_singing_track(
    segments: list[Segment],
    voice: str,
    ref_vocals_path: Path,
    work_dir: str | Path,
    video_duration: float,
    *,
    sing: bool = True,
    prep_text=None,
) -> tuple[Path, list[Segment]]:
    """
    Build a full AI vocal track that sings (melody-matched) over song timing.
    Returns (audio_path, spoken_segments with actual play times).
    """
    clean = prep_text or prepare_lyric_text
    work_dir = Path(work_dir)
    clips_dir = work_dir / "sing_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    ref_vocals_path = Path(ref_vocals_path)

    usable = [s for s in segments if clean(s.text)]
    if not usable:
        raise ValueError("No lyrics available for singing")

    voice_id = resolve_voice(voice)
    jobs = [(clean(s.text), clips_dir / f"tts_{i:04d}.mp3") for i, s in enumerate(usable)]
    asyncio.run(_synthesize_many(jobs, voice_id, rate="-6%", prep=clean))

    placed: list[tuple[Path, float, str]] = []
    missing = 0
    for i, seg in enumerate(usable):
        tts_mp3 = clips_dir / f"tts_{i:04d}.mp3"
        if not _tts_clip_ok(tts_mp3):
            missing += 1
            continue
        sung_wav = clips_dir / f"sing_{i:04d}.wav"
        text = clean(seg.text)
        try:
            if sing:
                speech_to_singing(
                    tts_mp3,
                    ref_vocals_path,
                    start=float(seg.start),
                    end=float(seg.end),
                    out_path=sung_wav,
                )
            else:
                _time_stretch_to_duration(
                    tts_mp3,
                    max(0.25, seg.end - seg.start),
                    sung_wav,
                    sr=22050,
                )
        except Exception:
            run_ffmpeg(
                [
                    "-i",
                    str(tts_mp3),
                    "-ac",
                    "1",
                    "-ar",
                    "22050",
                    "-y",
                    str(sung_wav),
                ]
            )

        if sung_wav.exists() and sung_wav.stat().st_size > 500:
            placed.append((sung_wav, float(seg.start), text))
        else:
            missing += 1

    if not placed:
        raise RuntimeError("Could not build singing clips from lyrics")
    if missing and missing > len(usable) * 0.4:
        raise RuntimeError(
            f"Many singing clips failed ({missing}/{len(usable)}). "
            "Check internet for Edge TTS and try again."
        )

    total = max(video_duration, max(s.end for s in usable) + 2.0, 1.0)
    timeline, spoken = _concat_speech_timeline(placed, total, work_dir, clips_dir)

    # Force audible vocal stem (avoid near-silent timeline after convert)
    boosted = work_dir / "ai_singing_boosted.wav"
    run_ffmpeg(
        [
            "-i",
            str(timeline),
            "-af",
            "loudnorm=I=-14:TP=-1.0:LRA=11,volume=1.25,alimiter=limit=0.97",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-y",
            str(boosted),
        ]
    )

    out_path = work_dir / "ai_singing_vocals.mp3"
    run_ffmpeg(
        [
            "-i",
            str(boosted if boosted.exists() else timeline),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "192k",
            "-y",
            str(out_path),
        ]
    )
    timeline.unlink(missing_ok=True)
    return out_path, spoken
