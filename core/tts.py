"""Khmer text-to-speech and muxing into video."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import edge_tts

from . import KHMER_VOICES, run_ffmpeg
from .khmer_render import save_subtitle_png
from .text_clean import prepare_speak_text
from .transcribe import Segment


async def _synthesize(text: str, voice: str, out_path: Path, *, rate: str = "-5%") -> None:
    # Slightly slower = clearer Khmer pronunciation
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(str(out_path))


async def _synthesize_with_retry(
    text: str,
    voice: str,
    out_path: Path,
    *,
    rate: str = "-5%",
    attempts: int = 5,
) -> None:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            await _synthesize(text, voice, out_path, rate=rate)
            if _tts_clip_ok(out_path):
                return
            raise RuntimeError("TTS wrote empty or too-short audio")
        except Exception as exc:
            last_exc = exc
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            await asyncio.sleep(0.8 * (i + 1))
    raise RuntimeError(f"TTS failed after {attempts} tries: {last_exc}") from last_exc


def _tts_clip_ok(path: Path) -> bool:
    """Reject silent / truncated Edge TTS outputs (common on long jobs)."""
    if not path.exists():
        return False
    size = path.stat().st_size
    if size < 800:
        return False
    # Typical short Khmer line is well above 2.5 KB; skip slow duration probe
    if size >= 2500:
        return True
    dur = get_duration_seconds(path)
    return dur >= 0.12


def text_to_speech(
    text: str,
    voice: str,
    out_path: str | Path,
    *,
    rate: str = "-5%",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    speak = prepare_speak_text(text)
    if not speak:
        raise ValueError("No text to synthesize")
    asyncio.run(_synthesize_with_retry(speak, voice, out_path, rate=rate))
    return out_path


async def _synthesize_many(
    items: list[tuple[str, Path]],
    voice: str,
    *,
    rate: str = "-5%",
    prep=None,
) -> list[Path]:
    """
    Generate TTS clips in small chunks.
    Long videos used to flood Edge TTS → random missing voice lines.
    """
    prep_fn = prep or prepare_speak_text

    async def one(text: str, path: Path) -> None:
        speak = prep_fn(text)
        if not speak:
            return
        await _synthesize_with_retry(speak, voice, path, rate=rate)

    sem = asyncio.Semaphore(2)
    failed: list[tuple[str, Path]] = []

    async def guarded(text: str, path: Path) -> None:
        async with sem:
            try:
                await one(text, path)
            except Exception:
                failed.append((text, path))

    # Chunked parallel pass — pause between chunks to avoid Edge dropouts
    chunk_size = 10
    for i in range(0, len(items), chunk_size):
        chunk = items[i : i + chunk_size]
        await asyncio.gather(*(guarded(t, p) for t, p in chunk))
        if i + chunk_size < len(items):
            await asyncio.sleep(1.25)

    # Serial retry with longer backoff
    still_bad: list[tuple[str, Path]] = []
    for text, path in failed:
        try:
            if path.exists():
                path.unlink(missing_ok=True)
            await asyncio.sleep(1.0)
            await one(text, path)
        except Exception:
            still_bad.append((text, path))

    # Final attempt for stubborn failures
    last_bad: list[tuple[str, Path]] = []
    for text, path in still_bad:
        try:
            if path.exists():
                path.unlink(missing_ok=True)
            await asyncio.sleep(1.5)
            speak = prep_fn(text)
            if speak:
                await _synthesize_with_retry(speak, voice, path, rate=rate, attempts=4)
        except Exception:
            last_bad.append((text, path))

    ok_paths = [p for _, p in items if _tts_clip_ok(p)]
    if not ok_paths:
        raise RuntimeError("TTS produced no usable audio clips")
    if last_bad and len(ok_paths) < max(1, int(len(items) * 0.75)):
        raise RuntimeError(
            f"Too many TTS clips failed ({len(last_bad)}/{len(items)}). "
            "Check internet / try again — Edge TTS rate-limits long videos."
        )
    return ok_paths


def get_duration_seconds(media_path: str | Path) -> float:
    """Parse duration from ffmpeg stderr (works without ffprobe)."""
    result = run_ffmpeg(["-i", str(media_path)], check=False)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr or "")
    if not match:
        return 0.0
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def get_video_size(media_path: str | Path) -> tuple[int, int]:
    """Return (width, height) from ffmpeg -i stderr."""
    result = run_ffmpeg(["-i", str(media_path)], check=False)
    match = re.search(r"Video:.*?\s(\d{2,5})x(\d{2,5})", result.stderr or "")
    if not match:
        return 1280, 720
    return int(match.group(1)), int(match.group(2))


def _mp3_to_wav(mp3_path: Path, wav_path: Path) -> Path:
    run_ffmpeg(
        [
            "-i",
            str(mp3_path),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "24000",
            "-ac",
            "1",
            str(wav_path),
        ]
    )
    return wav_path


def _write_silence_wav(path: Path, seconds: float) -> Path:
    seconds = max(0.02, float(seconds))
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=24000:cl=mono",
            "-t",
            f"{seconds:.3f}",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )
    return path


def _concat_speech_timeline(
    placed: list[tuple[Path, float, str]],
    total: float,
    work_dir: Path,
    clips_dir: Path,
) -> tuple[Path, list[Segment]]:
    """
    Build narration by concatenating: silence → speech → silence → speech…
    Returns (timeline_wav, segments_with_actual_play_times).
    Prefer hearing every clip; if times overlap, shift the next clip forward.
    """
    ordered = sorted(placed, key=lambda x: x[1])
    pieces: list[Path] = []
    spoken: list[Segment] = []
    cursor = 0.0
    sil_i = 0

    for i, (src, start, text) in enumerate(ordered):
        wav = clips_dir / f"speak_{i:04d}.wav"
        if src.suffix.lower() == ".wav":
            if src.resolve() != wav.resolve():
                run_ffmpeg(
                    ["-i", str(src), "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", str(wav)]
                )
        else:
            _mp3_to_wav(src, wav)
        if not wav.exists() or wav.stat().st_size < 100:
            continue

        dur = max(0.05, get_duration_seconds(wav))
        # Never skip a clip because of overlap — shift so voice stays audible
        if start < cursor:
            start = cursor

        gap = start - cursor
        if gap >= 0.03:
            sil = clips_dir / f"gap_{sil_i:04d}.wav"
            sil_i += 1
            _write_silence_wav(sil, gap)
            pieces.append(sil)

        pieces.append(wav)
        spoken.append(
            Segment(
                start=float(start),
                end=float(start + dur),
                text=prepare_speak_text(text) or text,
            )
        )
        cursor = start + dur
        if src.suffix.lower() != ".wav" and src.exists():
            src.unlink(missing_ok=True)

    if cursor < total - 0.05:
        sil = clips_dir / "gap_end.wav"
        _write_silence_wav(sil, total - cursor)
        pieces.append(sil)

    if not pieces:
        raise RuntimeError("No speech pieces to concatenate")

    list_file = clips_dir / "concat_list.txt"
    lines: list[str] = []
    for p in pieces:
        posix = p.resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{posix}'")
    list_file.write_text("\n".join(lines), encoding="utf-8")

    timeline = work_dir / "timeline.wav"
    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(timeline),
        ]
    )
    return timeline, spoken


def build_timed_khmer_track(
    segments: list[Segment],
    voice: str,
    work_dir: str | Path,
    video_duration: float,
) -> tuple[Path, list[Segment]]:
    """
    Timed Khmer narration for long videos.

    Returns (audio_path, spoken_segments) where spoken_segments use the
    **actual** times the voice plays (so captions can walk with sound).
    """
    work_dir = Path(work_dir)
    clips_dir = work_dir / "tts_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    usable = [s for s in segments if prepare_speak_text(s.text)]
    if not usable:
        raise ValueError("No Khmer text available for TTS")

    total = max(video_duration, max(s.end for s in usable) + 2.0, 1.0)

    jobs: list[tuple[str, Path]] = []
    for i, seg in enumerate(usable):
        jobs.append((prepare_speak_text(seg.text), clips_dir / f"seg_{i:04d}.mp3"))
    asyncio.run(_synthesize_many(jobs, voice, rate="-5%"))

    placed: list[tuple[Path, float, str]] = []
    missing = 0
    for i, seg in enumerate(usable):
        mp3_path = clips_dir / f"seg_{i:04d}.mp3"
        if not _tts_clip_ok(mp3_path):
            missing += 1
            continue

        clip_dur = get_duration_seconds(mp3_path)
        slot = max(seg.end - seg.start, 0.7)
        fitted = mp3_path
        # Fit voice into the Whisper slot when possible so captions stay aligned
        if clip_dur > slot * 1.12 and slot > 0.55:
            speed = min(1.25, max(1.02, clip_dur / slot))
            fitted = clips_dir / f"seg_{i:04d}_fast.mp3"
            run_ffmpeg(
                [
                    "-i",
                    str(mp3_path),
                    "-filter:a",
                    f"atempo={speed:.3f}",
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "128k",
                    str(fitted),
                ]
            )
            mp3_path.unlink(missing_ok=True)

        placed.append((fitted, float(seg.start), prepare_speak_text(seg.text)))

    if not placed:
        raise RuntimeError("TTS produced no audio clips")
    if missing and missing > len(usable) * 0.25:
        raise RuntimeError(
            f"Many voice clips missing ({missing}/{len(usable)}). "
            "Internet/Edge TTS may be unstable — please retry."
        )

    timeline, spoken = _concat_speech_timeline(placed, total, work_dir, clips_dir)

    out_path = work_dir / "khmer_narration.mp3"
    run_ffmpeg(
        [
            "-i",
            str(timeline),
            "-filter:a",
            "loudnorm=I=-16:TP=-1.5:LRA=11,alimiter=limit=0.95",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(out_path),
        ]
    )
    timeline.unlink(missing_ok=True)
    return out_path, spoken


def synthesize_segments(
    segments: list[Segment],
    voice: str,
    work_dir: str | Path,
    video_duration: float | None = None,
    *,
    fast: bool = True,
) -> tuple[Path, list[Segment]]:
    """
    Generate Khmer voice track.
    Returns (audio_path, caption_timing_segments).
    Timing segments match when the voice actually plays (for caption sync).
    """
    work_dir = Path(work_dir)
    usable = [s for s in segments if prepare_speak_text(s.text)]
    if not usable:
        raise ValueError("No Khmer text available for TTS")

    if video_duration and video_duration > 0:
        return build_timed_khmer_track(usable, voice, work_dir, video_duration)

    full_text = "។ ".join(prepare_speak_text(s.text) for s in usable)
    out = text_to_speech(full_text, voice, work_dir / "khmer_narration.mp3", rate="-8%")
    audio_dur = max(1.0, get_duration_seconds(out))
    # Spread original cues across the continuous narration
    span = max(s.end for s in usable) - min(s.start for s in usable)
    origin = min(s.start for s in usable)
    if span <= 0.1:
        slot = audio_dur / max(1, len(usable))
        timed = [
            Segment(start=i * slot, end=(i + 1) * slot, text=prepare_speak_text(s.text))
            for i, s in enumerate(usable)
        ]
    else:
        scale = audio_dur / span
        timed = [
            Segment(
                start=(s.start - origin) * scale,
                end=max((s.start - origin) * scale + 0.35, (s.end - origin) * scale),
                text=prepare_speak_text(s.text),
            )
            for s in usable
        ]
    return out, timed


def _parse_srt_segments(srt_path: Path) -> list[Segment]:
    """Minimal SRT parser → Segment list."""
    raw = srt_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    blocks = re.split(r"\n\s*\n", raw)
    segments: list[Segment] = []
    time_re = re.compile(
        r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)"
    )

    def to_sec(h, m, s, ms) -> float:
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        # first line may be index
        idx = 0
        if lines[0].strip().isdigit():
            idx = 1
        if idx >= len(lines):
            continue
        m = time_re.search(lines[idx])
        if not m:
            continue
        start = to_sec(*m.groups()[:4])
        end = to_sec(*m.groups()[4:])
        text = " ".join(lines[idx + 1 :]).strip()
        if text:
            segments.append(Segment(start=start, end=end, text=text))
    return segments


# Windows CreateProcess cmdline ~8191 chars; also avoid hundreds of hour-long -i loops.
_OVERLAY_BATCH = 35


def _escape_movie_path(path: Path) -> str:
    """Escape a path for ffmpeg movie= inside a filter script."""
    s = path.resolve().as_posix()
    # movie filter: escape \ and : ; keep forward slashes
    s = s.replace("\\", "/").replace(":", "\\:").replace("'", r"\'")
    return s


def _build_movie_overlay_filter(
    png_paths: list[Path],
    segments: list[Segment],
    *,
    margin_bottom: int = 40,
    fps: int = 15,
) -> str:
    """
    Load PNGs via movie= (keeps cmdline short) and chain overlays.
    Each still is looped; enable= shows it only during its cue window.
    """
    if not png_paths:
        raise ValueError("No overlay PNGs")
    if len(png_paths) != len(segments):
        raise ValueError("PNG count must match segment count")

    parts: list[str] = []
    for i, png in enumerate(png_paths):
        esc = _escape_movie_path(png)
        # Infinite still → timestamps; enable= clips visibility on the timeline
        parts.append(
            f"movie='{esc}',loop=loop=-1:size=1,format=rgba,"
            f"setpts=N/{fps}/TB[p{i}]"
        )

    last = "[0:v]"
    n = len(segments)
    for i, seg in enumerate(segments):
        out = f"[v{i}]" if i < n - 1 else "[vout]"
        start = max(0.0, float(seg.start))
        end = max(start + 0.05, float(seg.end))
        parts.append(
            f"{last}[p{i}]overlay=x=(W-w)/2:y=H-h-{margin_bottom}:"
            f"enable='between(t,{start:.3f},{end:.3f})'{out}"
        )
        last = out
    return ";\n".join(parts)


def _burn_overlay_batch(
    video_path: Path,
    png_paths: list[Path],
    segments: list[Segment],
    output_path: Path,
    *,
    duration: float,
    audio_path: Path | None = None,
    fast: bool = True,
    filter_script: Path,
) -> Path:
    """One ffmpeg pass: video (+ optional audio) + movie-loaded PNG overlays."""
    fps = 15 if fast else 25
    filt = _build_movie_overlay_filter(png_paths, segments, fps=fps)
    filter_script.write_text(filt, encoding="utf-8")

    cmd: list[str] = ["-i", str(video_path)]
    if audio_path:
        cmd.extend(["-i", str(audio_path)])

    cmd.extend(
        [
            "-filter_complex_script",
            str(filter_script),
            "-map",
            "[vout]",
        ]
    )
    if audio_path:
        cmd.extend(["-map", "1:a:0"])
    else:
        cmd.extend(["-map", "0:a:0?"])

    preset = "ultrafast" if fast else "veryfast"
    crf = "28" if fast else "23"
    cmd.extend(
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
            "160k",
            "-t",
            f"{duration:.3f}",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    try:
        run_ffmpeg(cmd)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "too long" in msg or "8191" in msg or "error creating process" in msg:
            raise RuntimeError(
                "Subtitle burn failed on a long video (Windows command limit). "
                "Retry, or use Soft subtitles / Subtitles file only for very long videos."
            ) from exc
        raise
    return output_path


def burn_khmer_overlays(
    video_path: Path,
    segments: list[Segment],
    output_path: Path,
    *,
    audio_path: Path | None = None,
    fast: bool = True,
    repage: bool = True,
) -> Path:
    """
    Burn Khmer subtitles as HarfBuzz-shaped PNG overlays (correct feet/hands).
    If audio_path is set, replace video audio with that track (dub + subs).
    Set repage=False when segments are already timed to match spoken audio.

    Long videos (1h+) are burned in batches so Windows ffmpeg does not hit the
    ~8191-char command-line limit or open hundreds of full-length PNG inputs.
    """
    from .translate import split_into_two_line_cues

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    usable = [s for s in segments if s.text and s.text.strip()]
    if not usable:
        raise ValueError("No subtitle segments to burn")

    # Always keep captions to ~1–2 lines from cleaned cues
    if repage:
        usable = split_into_two_line_cues(usable, max_chars=80)
    width, _height = get_video_size(video_path)
    duration = get_duration_seconds(video_path)
    if duration <= 0:
        duration = max(s.end for s in usable) + 0.5

    png_dir = output_path.parent / "sub_pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    png_paths: list[Path] = []
    for i, seg in enumerate(usable):
        png = png_dir / f"sub_{i:04d}.png"
        # Slightly smaller font in fast mode = quicker rasterize
        save_subtitle_png(
            seg.text,
            png,
            video_width=width,
            font_size=max(24, width // 34),
            max_lines=2,
        )
        png_paths.append(png)

    batch_size = _OVERLAY_BATCH
    current = video_path
    temps: list[Path] = []
    n = len(png_paths)
    n_batches = max(1, (n + batch_size - 1) // batch_size)

    try:
        for bi in range(n_batches):
            start = bi * batch_size
            end = min(n, start + batch_size)
            batch_pngs = png_paths[start:end]
            batch_segs = usable[start:end]
            is_last = bi == n_batches - 1
            out = output_path if is_last else (output_path.parent / f"_ov_batch_{bi:03d}.mp4")
            if not is_last:
                temps.append(out)
            script = output_path.parent / f"overlay_filter_{bi:03d}.txt"
            _burn_overlay_batch(
                current,
                batch_pngs,
                batch_segs,
                out,
                duration=duration,
                # Attach replacement audio only on the final pass
                audio_path=audio_path if is_last else None,
                fast=fast,
                filter_script=script,
            )
            current = out
    finally:
        for tmp in temps:
            if tmp.exists() and tmp.resolve() != output_path.resolve():
                tmp.unlink(missing_ok=True)

    return output_path


def burn_subtitles(
    video_path: str | Path,
    srt_path: str | Path,
    output_path: str | Path,
    *,
    fast: bool = True,
) -> Path:
    """Hard-burn Khmer subtitles with correct Battambang shaping."""
    segments = _parse_srt_segments(Path(srt_path))
    return burn_khmer_overlays(Path(video_path), segments, Path(output_path), fast=fast)


def replace_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    fast: bool = True,
) -> Path:
    """Replace video audio with Khmer TTS narration (browser-playable MP4)."""
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preset = "ultrafast" if fast else "veryfast"
    # Fast: copy video stream when possible (no re-encode)
    if fast:
        try:
            run_ffmpeg(
                [
                    "-i",
                    str(video_path),
                    "-i",
                    str(audio_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "160k",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
            )
            return output_path
        except Exception:
            if output_path.exists():
                output_path.unlink(missing_ok=True)

    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            "28" if fast else "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return output_path


def dub_and_burn(
    video_path: str | Path,
    audio_path: str | Path,
    srt_path: str | Path | None,
    output_path: str | Path,
    *,
    segments: list[Segment] | None = None,
    fast: bool = True,
) -> Path:
    """Khmer voice + HarfBuzz-shaped Battambang subtitles in one MP4."""
    if segments is None:
        if not srt_path:
            raise ValueError("dub_and_burn needs segments or srt_path")
        segments = _parse_srt_segments(Path(srt_path))
        repage = True
    else:
        # Already timed to match TTS playback
        repage = False
    return burn_khmer_overlays(
        Path(video_path),
        segments,
        Path(output_path),
        audio_path=Path(audio_path),
        fast=fast,
        repage=repage,
    )


def soft_sub_video(
    video_path: str | Path,
    srt_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Soft subs still use mov_text (player-dependent shaping).
    Prefer burn / dub_subs modes for correct on-screen Khmer.
    """
    video_path = Path(video_path)
    srt_path = Path(srt_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-i",
            str(srt_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=khm",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return output_path


def _render_note_badge(text: str, *, video_width: int) -> "Image.Image":
    """Top-right note badge (Latin or Khmer)."""
    from PIL import Image, ImageDraw, ImageFont

    from .fonts import ensure_battambang_fonts
    from .khmer_render import render_khmer_line

    text = (text or "").strip()
    if not text:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    # Prefer Khmer shaping when the note contains Khmer letters
    if re.search(r"[\u1780-\u17FF]", text):
        text_img = render_khmer_line(
            text,
            font_size=max(18, video_width // 50),
            max_width=int(video_width * 0.42),
            max_lines=1,
            prefer_two_lines=False,
            stroke_width=2,
        )
        pad = 10
        badge = Image.new(
            "RGBA",
            (text_img.width + pad * 2, text_img.height + pad * 2),
            (0, 0, 0, 0),
        )
        draw = ImageDraw.Draw(badge)
        draw.rounded_rectangle(
            [0, 0, badge.width, badge.height],
            radius=6,
            fill=(20, 20, 24, 175),
            outline=(255, 255, 255, 90),
            width=1,
        )
        badge.alpha_composite(text_img, (pad, pad))
        return badge

    fonts_dir = ensure_battambang_fonts()
    font_path = fonts_dir / "Battambang-Bold.ttf"
    font_size = max(17, video_width // 48)
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font = ImageFont.load_default()

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x, pad_y = 16, 10
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2
    badge = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge)
    draw.rounded_rectangle(
        [0, 0, box_w, box_h],
        radius=6,
        fill=(20, 20, 24, 175),
        outline=(255, 255, 255, 90),
        width=1,
    )
    draw.text(
        (box_w // 2, box_h // 2),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        anchor="mm",
    )
    return badge


def overlay_video_note(
    video_path: str | Path,
    note_text: str,
    output_path: str | Path,
    *,
    fast: bool = True,
) -> Path:
    """
    Burn a top-right video note badge (e.g. 'Cinema Summary') for the full duration.
    Supports English and Khmer note text.
    """
    from PIL import Image

    video_path = Path(video_path)
    output_path = Path(output_path)
    note_text = (note_text or "").strip()
    if not note_text:
        if Path(video_path).resolve() != output_path.resolve():
            import shutil

            shutil.copy2(video_path, output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, _height = get_video_size(video_path)
    duration = max(0.5, get_duration_seconds(video_path))

    badge = _render_note_badge(note_text, video_width=width)
    # Full-frame transparent overlay with badge top-right
    overlay = Image.new("RGBA", (width, max(1, _height)), (0, 0, 0, 0))
    margin = max(16, width // 50)
    x = max(0, width - badge.width - margin)
    y = margin
    overlay.alpha_composite(badge, (x, y))

    png = output_path.parent / f"{output_path.stem}_note.png"
    overlay.save(png, "PNG")

    preset = "ultrafast" if fast else "veryfast"
    crf = "28" if fast else "23"
    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-i",
            str(png),
            "-filter_complex",
            "[0:v][1:v]overlay=0:0[vout]",
            "-map",
            "[vout]",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-t",
            f"{duration:.3f}",
            "-movflags",
            "+faststart",
            "-y",
            str(output_path),
        ]
    )
    return output_path


def resolve_voice(label: str) -> str:
    return KHMER_VOICES.get(label, label)
