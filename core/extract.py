"""Extract audio / normalize video files (fast path preferred)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from . import run_ffmpeg

BROWSER_OK = {".mp4", ".webm", ".ogg"}


def extract_audio(video_path: str | Path, output_wav: str | Path) -> Path:
    """Extract mono 16 kHz WAV audio for speech recognition."""
    video_path = Path(video_path)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_wav),
        ]
    )
    if not output_wav.exists() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"Failed to extract audio from {video_path.name}")
    return output_wav


def _probe_stderr(video_path: Path) -> str:
    return run_ffmpeg(["-i", str(video_path)], check=False).stderr or ""


def _has_h264_aac(stderr: str) -> bool:
    # Heuristic: Video stream is h264 and an audio stream is aac (or no audio)
    has_h264 = bool(re.search(r"Video:.*\bh264\b", stderr, re.I))
    has_audio = bool(re.search(r"Audio:", stderr, re.I))
    has_aac = bool(re.search(r"Audio:.*\baac\b", stderr, re.I))
    return has_h264 and (has_aac or not has_audio)


def normalize_to_mp4(
    video_path: str | Path,
    output_mp4: str | Path,
    *,
    fast: bool = True,
) -> Path:
    """
    Prepare an MP4 for processing.
    Fast path: remux/stream-copy when already H.264 (+ AAC), else ultrafast encode
    capped at 720p (much quicker than full-quality re-encode).
    """
    video_path = Path(video_path)
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    stderr = _probe_stderr(video_path)

    # Same-file already mp4 + good codecs → copy bytes (instant)
    if (
        fast
        and video_path.suffix.lower() == ".mp4"
        and _has_h264_aac(stderr)
        and video_path.resolve() != output_mp4.resolve()
    ):
        shutil.copy2(video_path, output_mp4)
        return output_mp4

    # Remux to mp4 without re-encode when codecs are already friendly
    if fast and _has_h264_aac(stderr):
        try:
            run_ffmpeg(
                [
                    "-i",
                    str(video_path),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(output_mp4),
                ]
            )
            if output_mp4.exists() and output_mp4.stat().st_size > 0:
                return output_mp4
        except Exception:
            if output_mp4.exists():
                output_mp4.unlink(missing_ok=True)

    preset = "ultrafast" if fast else "veryfast"
    crf = "28" if fast else "23"
    vf = "scale='min(1280,iw)':'-2'" if fast else None

    args = ["-i", str(video_path)]
    if vf:
        args.extend(["-vf", vf])
    args.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k" if fast else "128k",
            "-movflags",
            "+faststart",
            str(output_mp4),
        ]
    )
    run_ffmpeg(args)
    if not output_mp4.exists() or output_mp4.stat().st_size == 0:
        raise RuntimeError(
            f"Could not convert '{video_path.name}' to MP4. "
            "Try another file (MP4 / WebM recommended)."
        )
    return output_mp4
