"""Procedural soft background music + mix under Khmer voice."""

from __future__ import annotations

import math
import random
import re
import struct
import subprocess
import wave
from pathlib import Path

from . import get_ffmpeg, run_ffmpeg

# Soft pentatonic-ish frequencies (Hz) for calm BGM
_SCALE = [196.00, 220.00, 246.94, 293.66, 329.63, 392.00, 440.00]
# Short loop only — never allocate hour-long float arrays in Python
_LOOP_SEC = 24.0


def _write_bgm_loop(duration_sec: float, out_path: Path, *, seed: int | None = None) -> Path:
    """Generate a short ambient WAV loop (streaming write, low RAM)."""
    rng = random.Random(seed if seed is not None else 42)
    sample_rate = 24000
    n_samples = max(1, int(duration_sec * sample_rate))
    amplitude = 0.18

    motif = [_SCALE[rng.randrange(len(_SCALE))] for _ in range(8)]
    note_len = int(0.45 * sample_rate)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        chunk = bytearray()
        t = 0
        note_i = 0
        prev = 0.0
        peak = 1e-6

        raw: list[float] = []
        while t < n_samples:
            freq = motif[note_i % len(motif)]
            if rng.random() < 0.15:
                freq *= 2.0
            length = min(note_len, n_samples - t)
            for i in range(length):
                env = 1.0
                attack = int(0.04 * sample_rate)
                release = int(0.12 * sample_rate)
                if i < attack:
                    env = i / max(1, attack)
                elif i > length - release:
                    env = max(0.0, (length - i) / max(1, release))
                phase = 2 * math.pi * freq * (i / sample_rate)
                val = math.sin(phase) * 0.65 + math.sin(phase * 1.002) * 0.35
                drone = math.sin(2 * math.pi * (freq / 4) * (i / sample_rate)) * 0.15
                sample = (val + drone) * env * amplitude
                smoothed = (prev + sample * 2 + sample) / 4 if i else sample
                prev = sample
                peak = max(peak, abs(smoothed))
                raw.append(smoothed)
            t += int(note_len * 0.85)
            note_i += 1

        gain = min(1.0, 0.55 / peak)
        for x in raw:
            v = int(max(-1.0, min(1.0, x * gain)) * 32767)
            chunk += struct.pack("<h", v)
            if len(chunk) >= 65536:
                wf.writeframes(chunk)
                chunk.clear()
        if chunk:
            wf.writeframes(chunk)

    return out_path


def generate_bgm(duration_sec: float, out_path: str | Path, *, seed: int | None = None) -> Path:
    """
    Generate a gentle looping ambient pad/arpeggio as WAV (no external music API).
    Safe for under-voice mix at low volume.

    For long videos: synthesize a short loop, then extend with ffmpeg
    (avoids MemoryError on 1h+ durations).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_sec = max(1.0, float(duration_sec))

    if duration_sec <= _LOOP_SEC + 1.0:
        return _write_bgm_loop(duration_sec, out_path, seed=seed)

    loop_path = out_path.with_name(out_path.stem + "_loop.wav")
    try:
        _write_bgm_loop(_LOOP_SEC, loop_path, seed=seed)
        run_ffmpeg(
            [
                "-stream_loop",
                "-1",
                "-i",
                str(loop_path),
                "-t",
                f"{duration_sec:.3f}",
                "-c:a",
                "pcm_s16le",
                str(out_path),
            ]
        )
    finally:
        loop_path.unlink(missing_ok=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("Failed to generate background music for this video length.")
    return out_path


def mix_voice_and_music(
    voice_path: str | Path,
    music_path: str | Path,
    out_path: str | Path,
    *,
    music_volume: float = 0.22,
    duck_under_voice: bool = False,
) -> Path:
    """
    Mix Khmer narration (foreground) with music bed (background).
    Duration follows the voice track.

    duck_under_voice=True: soft cinematic dip under spoken lines.
    """
    voice_path = Path(voice_path)
    music_path = Path(music_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cap bed level — high volumes sound muddy / fight Khmer voice
    vol = max(0.08, min(0.55, float(music_volume)))
    if duck_under_voice:
        # Soft duck, no makeup pump; light bed EQ + final limiter
        filt = (
            f"[1:a]highpass=f=50,lowpass=f=14000,"
            f"equalizer=f=250:t=q:w=0.8:g=-1.5,"
            f"volume={vol:.3f},afade=t=in:st=0:d=1.2[mraw];"
            f"[0:a]asplit=2[v][sc];"
            f"[mraw][sc]sidechaincompress="
            f"threshold=0.06:ratio=2.2:attack=80:release=420:makeup=1:level_sc=1[md];"
            f"[v]volume=1.05[vf];"
            f"[vf][md]amix=inputs=2:duration=first:dropout_transition=3:normalize=0,"
            f"alimiter=limit=0.92:attack=5:release=50[a]"
        )
    else:
        filt = (
            f"[1:a]volume={vol:.3f},afade=t=in:st=0:d=1.5[m];"
            f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
            f"alimiter=limit=0.95:attack=5:release=50[a]"
        )

    plain_filt = (
        f"[1:a]highpass=f=50,volume={vol:.3f},afade=t=in:st=0:d=1.2[m];"
        f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=3:normalize=0,"
        f"alimiter=limit=0.93:attack=5:release=50[a]"
    )

    def _do_mix(filter_complex: str) -> None:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        run_ffmpeg(
            [
                "-i",
                str(voice_path),
                "-stream_loop",
                "-1",
                "-i",
                str(music_path),
                "-filter_complex",
                filter_complex,
                "-map",
                "[a]",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(out_path),
            ]
        )

    try:
        _do_mix(filt)
    except Exception:
        if duck_under_voice:
            _do_mix(plain_filt)
        else:
            raise

    if not out_path.exists() or out_path.stat().st_size < 500:
        raise RuntimeError("Failed to mix Khmer voice with music bed.")
    return out_path


def mix_into_video(
    video_path: str | Path,
    music_path: str | Path,
    out_path: str | Path,
    *,
    music_volume: float = 0.18,
) -> Path:
    """Add BGM under existing video audio (for subtitle-only modes)."""
    video_path = Path(video_path)
    music_path = Path(music_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-stream_loop",
            "-1",
            "-i",
            str(music_path),
            "-filter_complex",
            (
                f"[1:a]volume={music_volume:.3f},afade=t=in:st=0:d=1.5[m];"
                f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[a]"
            ),
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_path),
        ]
    )
    return out_path


def _mean_volume_db(audio_path: Path) -> float | None:
    """Return mean_volume from ffmpeg volumedetect, or None."""
    cmd = [
        get_ffmpeg(),
        "-i",
        str(audio_path),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = (result.stderr or "") + (result.stdout or "")
    m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _audio_is_audible(audio_path: Path, *, min_mean_db: float = -38.0) -> bool:
    """True if file exists and has usable energy (not near-silent Demucs bed)."""
    path = Path(audio_path)
    if not path.is_file() or path.stat().st_size < 2000:
        return False
    mean_db = _mean_volume_db(path)
    if mean_db is None:
        return path.stat().st_size > 50_000
    return mean_db >= min_mean_db


def _polish_instrumental_bed(music: Path, out_path: Path) -> Path:
    """
    Light polish for Demucs instrumental — keep music natural / warm.
    Softer than Song-AI scrub (less thin or hollow).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = [
        (
            "highpass=f=40,"
            "equalizer=f=300:t=q:w=0.7:g=-1,"
            "equalizer=f=1800:t=q:w=1.0:g=-2.5,"
            "equalizer=f=3200:t=q:w=1.0:g=-2,"
            "acompressor=threshold=-18dB:ratio=1.8:attack=20:release=200:makeup=1,"
            "loudnorm=I=-22:TP=-2.0:LRA=10"
        ),
        (
            "highpass=f=40,"
            "equalizer=f=1800:t=q:w=1.0:g=-2.5,"
            "equalizer=f=3200:t=q:w=1.0:g=-2,"
            "loudnorm=I=-22:TP=-2.0:LRA=10"
        ),
    ]
    for af in attempts:
        try:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
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
    run_ffmpeg(["-i", str(music), "-ar", "48000", "-y", str(out_path)])
    return out_path


def _ducked_original_bed(mix_audio: Path, out_path: Path) -> Path:
    """
    Fallback when Demucs instrumental is empty (speech-heavy clips).
    Stronger speech-band pull so original talk doesn't fight Khmer voice,
    while keeping bass / pads musical.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-i",
            str(mix_audio),
            "-af",
            (
                "highpass=f=45,"
                "lowpass=f=12000,"
                "equalizer=f=800:t=q:w=1.0:g=-5,"
                "equalizer=f=1600:t=q:w=1.1:g=-8,"
                "equalizer=f=2800:t=q:w=1.2:g=-9,"
                "equalizer=f=4500:t=q:w=1.0:g=-5,"
                "acompressor=threshold=-20dB:ratio=2.5:attack=15:release=250:makeup=1,"
                "volume=0.70,"
                "loudnorm=I=-23:TP=-2.5:LRA=9"
            ),
            "-ar",
            "48000",
            "-y",
            str(out_path),
        ]
    )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            "Could not keep original music from this video. "
            "Try again, or install Demucs:\n"
            "  pip install -U demucs torch torchaudio"
        )
    return out_path


def extract_original_music_bed(
    mix_audio: str | Path,
    work_dir: str | Path,
    *,
    progress_cb=None,
) -> Path:
    """
    Keep the uploaded video's music for Khmer dub (clean underlay).

    1) Prefer Demucs instrumental + light polish when audible.
    2) If missing / silent (speech videos), soft ducked original mix.
    """
    mix_audio = Path(mix_audio)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "original_music_bed.wav"

    def tick(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    try:
        from .song_ai import separate_vocals_and_music

        tick("Separating original music (clean instrumental)…")
        _vocals, music_raw = separate_vocals_and_music(mix_audio, work_dir / "stems")
        tick("Polishing music bed…")
        polished = _polish_instrumental_bed(
            music_raw, work_dir / "instrumental_polish.wav"
        )
        if _audio_is_audible(polished, min_mean_db=-36.0):
            run_ffmpeg(
                [
                    "-i",
                    str(polished),
                    "-ar",
                    "48000",
                    "-y",
                    str(out_path),
                ]
            )
            if _audio_is_audible(out_path, min_mean_db=-38.0):
                tick("Clean instrumental ready.")
                return out_path
        tick("Instrumental too quiet — soft original soundtrack under Khmer…")
    except Exception as exc:
        tick(
            f"Demucs unavailable ({type(exc).__name__}) — "
            "soft original soundtrack under Khmer…"
        )

    return _ducked_original_bed(mix_audio, out_path)
