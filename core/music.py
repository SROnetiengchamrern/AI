"""Procedural soft background music + mix under Khmer voice."""

from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path

from . import run_ffmpeg

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

        # Generate in small chunks so 1h+ videos never allocate ~GB of floats
        chunk = bytearray()
        t = 0
        note_i = 0
        # Simple 1-sample history for light smoothing
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

    # Short clips: write directly. Long clips: loop a short motif with ffmpeg.
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
) -> Path:
    """Mix Khmer narration (loud) with BGM (quiet). Duration follows voice."""
    voice_path = Path(voice_path)
    music_path = Path(music_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_ffmpeg(
        [
            "-i",
            str(voice_path),
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
            "[a]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(out_path),
        ]
    )
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
