"""Rough wall-clock estimate for Video → Khmer conversion (CPU)."""

from __future__ import annotations

from pathlib import Path

from .tts import get_duration_seconds

# Seconds of processing per 1 second of video (CPU ballpark)
_MODEL_FACTOR = {
    "tiny": 0.35,
    "base": 0.55,
    "small": 1.1,
    "medium": 2.2,
    "large-v3": 4.0,
}

_MODE_EXTRA = {
    "srt_only": 0.05,
    "soft_subs": 0.25,
    "burn_subs": 0.55,  # PNG overlays + encode
    "dub": 0.90,  # timed TTS + mux
    "dub_subs": 1.25,  # TTS + shaped overlays + encode
}


def format_duration(seconds: float) -> str:
    """Human time: hours / minutes / seconds."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h} hr {m:02d} min"
    if m > 0:
        return f"{m} min {s:02d} sec"
    return f"{s} sec"


def format_video_length(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def estimate_seconds(
    video_duration: float,
    *,
    mode: str = "dub_subs",
    model_size: str = "tiny",
    generate_story: bool = False,
    add_music: bool = False,
    fast: bool = True,
) -> float:
    """
    Estimate total convert time in seconds.
    Includes normalize, whisper, translate, TTS, burn, music.
    """
    if video_duration <= 0:
        return 0.0

    d = video_duration
    model_f = _MODEL_FACTOR.get(model_size, 0.35)
    mode_f = _MODE_EXTRA.get(mode, 1.0)
    if fast:
        model_f *= 0.7
        mode_f *= 0.45  # one-shot TTS + fewer overlays + remux

    fixed = 25.0 if fast else 45.0
    normalize = d * (0.08 if fast else 0.5)  # remux / ultrafast
    whisper = d * model_f
    translate = max(4.0, (d / 8.0) * 0.25) if fast else max(8.0, (d / 3.0) * 0.35)
    mode_work = d * mode_f

    story = 15.0 if generate_story else 0.0
    music = 5.0 + d * 0.03 if add_music else 0.0

    cushion = 1.1
    total = (fixed + normalize + whisper + translate + mode_work + story + music) * cushion
    return total


def estimate_message(
    video_path: str | Path | None,
    mode_label: str,
    model_label: str,
    generate_story: bool,
    add_music: bool,
    modes_map: dict,
    models_map: dict,
    fast: bool = True,
) -> str:
    if not video_path:
        return (
            "**Estimated time:** upload a video to see estimate\n\n"
            "_Rough CPU estimate — actual time varies by PC speed._"
        )

    path = Path(str(video_path))
    try:
        from . import path_exists_safe

        ok = path_exists_safe(path)
    except Exception:
        ok = False
    if not ok:
        return "**Estimated time:** waiting for valid upload…"

    try:
        duration = get_duration_seconds(path)
    except Exception:
        duration = 0.0

    if duration <= 0:
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
        except OSError:
            try:
                import os
                from . import win_long_path

                size_mb = os.path.getsize(win_long_path(path)) / (1024 * 1024)
            except OSError:
                return "**Estimated time:** could not read this file (path may be too long)"
        duration = max(60.0, size_mb * 8)

    mode = modes_map.get(mode_label, "dub_subs")
    model = models_map.get(model_label, "tiny")
    est = estimate_seconds(
        duration,
        mode=mode,
        model_size=model,
        generate_story=bool(generate_story),
        add_music=bool(add_music),
        fast=bool(fast),
    )
    low = est * 0.75
    high = est * 1.35
    speed_note = "fastest ON" if fast else "quality ON"

    return (
        f"**Video length:** `{format_video_length(duration)}`\n\n"
        f"**Estimated generate time:** `{format_duration(low)}` – `{format_duration(high)}`\n\n"
        f"_About **{format_duration(est)}** on a typical CPU "
        f"(Whisper **{model}**, mode **{mode}**, {speed_note}"
        f"{', story' if generate_story else ''}"
        f"{', music' if add_music else ''})._"
    )
