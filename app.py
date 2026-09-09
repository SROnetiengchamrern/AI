"""
Video → Khmer converter

Upload a video spoken in English, Chinese, Thai, etc.
The tool will:
  1. Transcribe speech (Whisper, auto language detect)
  2. Translate to Khmer (optional story rewrite)
  3. Export Khmer SRT + optional dubbed / subtitled video (+ optional BGM)
"""

from __future__ import annotations

import os
import queue
import re
import tempfile
import threading
from pathlib import Path

from core import KHMER_VOICES, SOURCE_LANGUAGES, ensure_ffmpeg_on_path
from core.firebase_config import is_firebase_configured, load_firebase_config
from core.fonts import ensure_battambang_fonts

ensure_ffmpeg_on_path()
ensure_battambang_fonts()

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from core.estimate import estimate_message
from core.pipeline import convert_video_to_khmer
from core.song_ai import SONG_AI_VOICES, create_song_ai_video
from core.text_video import create_video_from_text
from core import copy_upload_to_short_path, path_exists_safe, preferred_temp_root

_ROOT = Path(__file__).resolve().parent
_AUTH_DIR = _ROOT / "static" / "auth"
# Guests always start at login unless FIREBASE_REQUIRE_LOGIN=0
_REQUIRE_LOGIN = os.environ.get("FIREBASE_REQUIRE_LOGIN", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_APP_PATH = "/app"
MODES = {
    "Khmer voice + subtitles (SPEAKS Khmer)": "dub_subs",
    "Khmer voice only (SPEAKS Khmer)": "dub",
    "Burned subtitles only (keeps original speech)": "burn_subs",
    "Soft subtitles only (keeps original speech)": "soft_subs",
    "Subtitles file only (.srt)": "srt_only",
}

MODELS = {
    "Tiny (faster, less accurate)": "tiny",
    "Base (recommended quality)": "base",
    "Small (more accurate)": "small",
    "Medium (best CPU accuracy)": "medium",
}

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".mpeg", ".mpg", ".3gp"}

SOURCE_LANGUAGES_REVERSE = {v: k for k, v in SOURCE_LANGUAGES.items()}


def resolve_upload(file_obj) -> Path:
    """Resolve Gradio File upload value to a local Path."""
    if file_obj is None:
        raise RuntimeError("Please upload a video sample first.")

    path: Path | None = None
    if isinstance(file_obj, (str, Path)):
        path = Path(file_obj)
    elif isinstance(file_obj, dict):
        for key in ("path", "name", "video"):
            if file_obj.get(key):
                path = Path(file_obj[key])
                break
    else:
        candidate = getattr(file_obj, "name", None) or str(file_obj)
        path = Path(candidate)

    if path is None:
        raise RuntimeError("Could not read the uploaded file path. Re-upload the video.")

    # Gradio may keep titles with `/` `|` → broken Windows paths; try parent search
    if not path_exists_safe(path):
        parent = path.parent
        stem_hint = re.sub(r"[\\/|:*?\"<>]+", "", path.stem)[:24]
        found = None
        try:
            if parent.is_dir():
                for cand in parent.rglob("*"):
                    if not cand.is_file():
                        continue
                    if cand.suffix.lower() in VIDEO_EXTS and (
                        stem_hint.lower() in cand.name.lower()
                        or cand.stat().st_size > 0
                    ):
                        # Prefer exact-ish name match; else largest video in folder
                        if found is None or cand.stat().st_size > found.stat().st_size:
                            if stem_hint and stem_hint.lower() in cand.name.lower():
                                found = cand
                                break
                            if found is None:
                                found = cand
        except OSError:
            found = None
        if found is not None:
            path = found

    if not path_exists_safe(path):
        raise RuntimeError(
            "Uploaded file not found (path too long or invalid characters like / |). "
            "Rename the video to a short name like video.mkv and upload again."
        )

    if path.suffix.lower() not in VIDEO_EXTS:
        raise RuntimeError(
            f"Unsupported file type '{path.suffix}'. "
            "Please upload MP4, MOV, MKV, AVI, or WebM."
        )
    return path


def _progress_status(pct: float, msg: str) -> str:
    pct_i = max(0, min(100, int(round(pct))))
    return f"**Progress: {pct_i}%** — {msg}"


def _loading_html(active: bool, pct: float = 0, msg: str = "Ready") -> str:
    """Visible spinner panel while conversion runs (or idle/error/done panel)."""
    pct_i = max(0, min(100, int(round(pct))))
    safe_msg = (msg or "Ready").replace("<", "&lt;").replace(">", "&gt;")
    low = safe_msg.lower()
    is_error = (not active) and low.startswith("error")
    is_done = (not active) and (low.startswith("done") or pct_i >= 100)

    if is_error:
        return (
            '<div class="vk-loading vk-idle" style="border-color:#c44;background:#2a1515;">'
            '<div class="vk-dot" style="background:#e55;"></div>'
            "<div><strong>Failed</strong>"
            f'<div class="vk-msg">{safe_msg}</div></div></div>'
        )
    if is_done:
        return (
            '<div class="vk-loading vk-idle">'
            '<div class="vk-dot"></div>'
            "<div><strong>Done</strong>"
            f'<div class="vk-msg">{safe_msg}</div></div></div>'
        )
    if not active:
        return (
            '<div class="vk-loading vk-idle">'
            '<div class="vk-dot"></div>'
            "<div><strong>Ready</strong>"
            "<div class=\"vk-msg\">Upload a video, then click Convert to Khmer.</div></div>"
            "</div>"
        )
    return (
        '<div class="vk-loading vk-active">'
        '<div class="vk-spinner" aria-hidden="true"></div>'
        f"<div><strong>Generating… {pct_i}%</strong>"
        f'<div class="vk-msg">{safe_msg}</div>'
        '<div class="vk-bar"><span style="width:'
        f'{pct_i}%"></span></div></div></div>'
    )


def _btn_busy():
    return gr.update(interactive=False, value="Generating… please wait")


def _btn_ready():
    return gr.update(interactive=True, value="Convert to Khmer")


def _fail_convert(exc) -> tuple:
    """
    Friendly failed outputs for Video → Khmer.
    Do NOT raise gr.Error after yield — Gradio then paints every File as 'Error'.
    Must return exactly 10 values matching run_btn.click outputs.
    """
    msg = str(exc)
    summary = (
        f"**Status:** Conversion failed\n\n"
        f"**Error:** {msg}\n\n"
        f"_Tip:_ Rename long titles (especially with `/` or `|`) to a short name like "
        f"`video.mkv`, then upload again. Keep the tab open and check internet for TTS._"
    )
    return (
        _loading_html(False, 0, f"Error: {msg}"),
        0,
        _progress_status(0, f"Error: {msg}"),
        _btn_ready(),
        summary,
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
    )


def _progress_tuple(pct: float, msg: str, *, busy: bool = True) -> tuple:
    """Exactly 10 outputs: loading, bar, status, button, summary + 5 files (hold)."""
    return (
        _loading_html(True, pct, msg),
        pct,
        _progress_status(pct, msg),
        _btn_busy() if busy else _btn_ready(),
        gr.update(),  # summary
        gr.update(),  # khmer_srt
        gr.update(),  # original_srt
        gr.update(),  # story
        gr.update(),  # music
        gr.update(),  # video
    )


def process(
    video,
    source_language,
    mode_label,
    model_label,
    voice_label,
    generate_story,
    add_music,
    keep_original_music,
    show_note,
    video_note,
    fast_mode,
):
    """
    Generator so the loading spinner + progress bar update live while converting.
    Yields: loading_html, progress_pct, progress_status, run_btn,
            summary, khmer_srt, original_srt, story_file, music_file, video_download
    """
    try:
        video_path = resolve_upload(video)
    except Exception as exc:
        yield _fail_convert(exc)
        return

    out_root = Path(tempfile.mkdtemp(prefix="vk_", dir=str(preferred_temp_root())))

    yield _progress_tuple(1, "Staging upload (short path)…")

    try:
        # Long titles / Hindi/Chinese chars / `/` `|` → WinError 206; stage as input.mkv
        video_path = copy_upload_to_short_path(video_path, out_root)
    except Exception as exc:
        yield _fail_convert(exc)
        return

    yield _progress_tuple(2, "Starting…")

    q: queue.Queue = queue.Queue()
    holder: dict = {}

    def on_progress(frac: float, msg: str) -> None:
        q.put(("progress", float(frac), str(msg)))

    def worker() -> None:
        try:
            lang = SOURCE_LANGUAGES_REVERSE.get(source_language, "auto")
            holder["result"] = convert_video_to_khmer(
                video_path=video_path,
                output_dir=out_root,
                source_language=lang,
                model_size=MODELS.get(model_label, "base"),
                mode=MODES.get(mode_label, "dub_subs"),
                voice_label=voice_label,
                generate_story=bool(generate_story),
                add_music=bool(add_music),
                keep_original_music=bool(keep_original_music),
                show_note=bool(show_note),
                video_note=(video_note or "").strip() or "Cinema Summary",
                fast=bool(fast_mode),
                progress_cb=on_progress,
            )
            q.put(("done", None, None))
        except Exception as exc:
            msg = str(exc)
            low = msg.lower()
            if "no space left" in low or "errno 28" in low:
                msg = (
                    "Disk full while generating Khmer voice. "
                    "Free space on C: (or clear old .work / Temp folders) and try again. "
                    f"Detail: {exc}"
                )
            elif "206" in msg or ("too long" in low and "path" in low) or "cannot find the path" in low:
                msg = (
                    "Windows path / filename problem (long title or characters like / |). "
                    "Rename to a short name like video.mkv and upload again. "
                    f"Detail: {exc}"
                )
            elif (
                "memory" in low
                or "out of memory" in low
                or "8191" in msg
                or "error creating process" in low
                or "command limit" in low
            ):
                msg = (
                    "This video is too long for one burn pass (common on 1h+ files). "
                    "Retry Convert — long videos now burn in batches. "
                    "Or use Soft subtitles / Subtitles file only. "
                    f"Detail: {exc}"
                )
            elif "edge tts" in low or ("tts" in low and ("fail" in low or "missing" in low)):
                msg = (
                    "Khmer voice failed (Edge TTS rate limit / network). "
                    "Wait a minute and retry, or use Burned subtitles only / Soft subtitles. "
                    f"Detail: {exc}"
                )
            elif "no speech" in low:
                msg = str(exc)
            elif "translation" in low or "translator" in low:
                msg = (
                    "Translation to Khmer failed (Google Translate). "
                    "Check internet, set Source language to Auto detect or English, then retry. "
                    f"Detail: {exc}"
                )
            q.put(("error", RuntimeError(msg), None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, a, b = q.get()
        if kind == "progress":
            pct = max(0.0, min(100.0, float(a) * 100.0))
            msg = b or ""
            yield _progress_tuple(pct, msg)
        elif kind == "error":
            # Yield friendly UI and return — do not raise gr.Error (hides the real message)
            yield _fail_convert(a)
            return
        elif kind == "done":
            result = holder["result"]
            extras = []
            if fast_mode:
                extras.append("fastest ON")
            if generate_story:
                extras.append("story ON")
            if keep_original_music:
                extras.append("keep original music ON")
            elif add_music:
                extras.append("procedural BGM ON")
            extra_line = f"\n\n**Options:** {', '.join(extras)}" if extras else ""
            body_label = "Khmer story" if generate_story else "Khmer translation"
            # Cap preview text — 1h+ transcripts can break Gradio File outputs
            orig_preview = _preview_text(result.original_text)
            khmer_preview = _preview_text(result.khmer_text)
            summary = (
                f"**Status:** Conversion complete{extra_line}\n\n"
                f"**Detected language:** `{result.detected_language}`\n\n"
                f"**Original transcript**\n\n{orig_preview}\n\n"
                f"**{body_label}**\n\n{khmer_preview}\n\n"
            )
            if result.output_video:
                summary += f"**Download:** `{Path(result.output_video).name}`\n\n"
            summary += "_Download files below (open the video in VLC / your player)._"
            yield (
                _loading_html(False, 100, "Done"),
                100,
                _progress_status(100, "Done — download files below."),
                _btn_ready(),
                summary,
                str(result.khmer_srt),
                str(result.original_srt),
                str(result.story_path) if result.story_path else None,
                str(result.music_path) if result.music_path else None,
                str(result.output_video) if result.output_video else None,
            )
            return


VIDEO_STYLES = {
    "Cinematic": "cinematic",
    "Storybook": "storybook",
    "Nature": "nature",
    "Modern": "modern",
}

TEXT_VIDEO_MODES = {
    "Title story (from title only)": "title",
    "Auto (detect prompt vs script)": "auto",
    "Visual prompt (match my description)": "visual",
    "Narration script (story text)": "script",
}


def _btn_busy_text():
    return gr.update(interactive=False, value="Generating… please wait")


def _btn_ready_text():
    return gr.update(interactive=True, value="Generate AI video")


def process_text_video(
    video_title,
    script_text,
    source_language,
    voice_label,
    style_label,
    mode_label,
    duration_hours,
    duration_minutes,
    add_music,
    show_captions_kh,
    show_captions_en,
    show_note,
    video_note,
    show_title_en,
    show_title_kh,
    fast_mode,
):
    """Text → AI voice + AI scene video (title / visual / script)."""
    title = (video_title or "").strip()
    script = (script_text or "").strip()
    if not title and not script:
        raise gr.Error("Please write a video title (example: A Poor Cat Becomes a Millionaire).")

    out_root = Path(tempfile.mkdtemp(prefix="tv_", dir=str(preferred_temp_root())))

    yield (
        _loading_html(True, 1, "Starting text → video…"),
        1,
        _progress_status(1, "Starting text → video…"),
        _btn_busy_text(),
        gr.update(),
        gr.update(),
        gr.update(),
    )

    q: queue.Queue = queue.Queue()
    holder: dict = {}

    def on_progress(frac: float, msg: str) -> None:
        q.put(("progress", float(frac), str(msg)))

    def worker() -> None:
        try:
            holder["result"] = create_video_from_text(
                script,
                output_dir=out_root,
                video_title=title,
                duration_hours=int(duration_hours or 0),
                duration_minutes=int(duration_minutes or 0),
                source_language=SOURCE_LANGUAGES_REVERSE.get(source_language, "auto"),
                voice_label=voice_label,
                style=VIDEO_STYLES.get(style_label, "cinematic"),
                mode=TEXT_VIDEO_MODES.get(mode_label, "auto"),
                add_music=bool(add_music),
                show_captions_kh=bool(show_captions_kh),
                show_captions_en=bool(show_captions_en),
                show_note=bool(show_note),
                video_note=(video_note or "").strip() or "Cinema Summary",
                show_title_en=bool(show_title_en),
                show_title_kh=bool(show_title_kh),
                fast=bool(fast_mode),
                progress_cb=on_progress,
            )
            q.put(("done", None, None))
        except Exception as exc:
            q.put(("error", exc, None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, a, b = q.get()
        if kind == "progress":
            pct = max(0.0, min(100.0, float(a) * 100.0))
            msg = b or ""
            yield (
                _loading_html(True, pct, msg),
                pct,
                _progress_status(pct, msg),
                _btn_busy_text(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
        elif kind == "error":
            yield (
                _loading_html(False, 0, f"Error: {a}"),
                0,
                _progress_status(0, f"Error: {a}"),
                _btn_ready_text(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
            raise gr.Error(f"Text → video failed: {a}") from a
        elif kind == "done":
            result = holder["result"]
            scene_preview = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(result.scenes[:12]))
            if len(result.scenes) > 12:
                scene_preview += f"\n… (+{len(result.scenes) - 12} more)"
            mode_labels = {
                "visual": "visual prompt → matched AI scenes",
                "title": "title story → AI auto scenes",
                "script": "narration script",
            }
            mode_note = mode_labels.get(result.mode, result.mode)
            hours = int(duration_hours or 0)
            minutes = int(duration_minutes or 0)
            if hours <= 0 and minutes <= 0:
                minutes = 1
            dur_note = f"{hours}h {minutes}m" if hours else f"{minutes} min"
            summary = (
                f"**Status:** AI video ready ({mode_note})\n\n"
                f"**Title:** {title or '(from script)'}\n\n"
                f"**Target length:** {dur_note}\n\n"
                f"**Scenes:** {len(result.scenes)}\n\n"
                f"**Khmer voice script**\n\n{result.khmer_text}\n\n"
                f"**Scene list**\n\n{scene_preview}\n\n"
                f"**Download:** `{Path(result.video_path).name}`\n\n"
                f"_Download the MP4 / voice below._"
            )
            yield (
                _loading_html(False, 100, "Done"),
                100,
                _progress_status(100, "Done — download files below."),
                _btn_ready_text(),
                summary,
                str(result.video_path),
                str(result.audio_path),
            )
            return


def _btn_busy_song():
    return gr.update(interactive=False, value="Building AI song… please wait")


def _btn_ready_song():
    return gr.update(interactive=True, value="Build AI Song")


def process_song_ai(
    video,
    source_language,
    model_label,
    voice_label,
    enable_singing,
    keep_music_bed,
    show_captions,
    show_note,
    video_note,
    music_volume,
    vocal_volume,
    fast_mode,
):
    """Song video → strip all original audio → AI sings same lyrics (no translate)."""
    video_path = resolve_upload(video)
    out_root = Path(tempfile.mkdtemp(prefix="song_", dir=str(preferred_temp_root())))

    yield (
        _loading_html(True, 1, "Starting Song AI…"),
        1,
        _progress_status(1, "Starting Song AI…"),
        _btn_busy_song(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )

    try:
        video_path = copy_upload_to_short_path(video_path, out_root)
    except Exception as exc:
        msg = str(exc)
        summary = f"**Status:** Song AI failed\n\n**Error:** {msg}"
        yield (
            _loading_html(False, 0, f"Error: {msg}"),
            0,
            _progress_status(0, f"Error: {msg}"),
            _btn_ready_song(),
            summary,
            None,
            None,
            None,
            None,
            None,
        )
        return

    q: queue.Queue = queue.Queue()
    holder: dict = {}

    def on_progress(frac: float, msg: str) -> None:
        q.put(("progress", float(frac), str(msg)))

    def worker() -> None:
        try:
            holder["result"] = create_song_ai_video(
                video_path,
                output_dir=out_root,
                source_language=SOURCE_LANGUAGES_REVERSE.get(source_language, "auto"),
                model_size=MODELS.get(model_label, "base"),
                voice_label=voice_label,
                show_captions=bool(show_captions),
                show_note=bool(show_note),
                video_note=(video_note or "").strip() or "AI Song",
                vocal_volume=float(vocal_volume or 2.0),
                music_volume=float(music_volume or 0.65),
                keep_music_bed=bool(keep_music_bed),
                enable_singing=bool(enable_singing),
                fast=bool(fast_mode),
                progress_cb=on_progress,
            )
            q.put(("done", None, None))
        except Exception as exc:
            q.put(("error", exc, None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, a, b = q.get()
        if kind == "progress":
            pct = max(0.0, min(100.0, float(a) * 100.0))
            msg = b or ""
            yield (
                _loading_html(True, pct, msg),
                pct,
                _progress_status(pct, msg),
                _btn_busy_song(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
        elif kind == "error":
            msg = str(a)
            summary = f"**Status:** Song AI failed\n\n**Error:** {msg}"
            yield (
                _loading_html(False, 0, f"Error: {msg}"),
                0,
                _progress_status(0, f"Error: {msg}"),
                _btn_ready_song(),
                summary,
                None,
                None,
                None,
                None,
                None,
            )
            return
        elif kind == "done":
            result = holder["result"]
            lyrics_preview = (result.lyrics_ai or result.lyrics_khmer or "")[:1200]
            full_lyrics = result.lyrics_ai or result.lyrics_khmer or ""
            if len(full_lyrics) > 1200:
                lyrics_preview += "…"
            bed_note = (
                "AI singing + music bed (original singer removed)"
                if keep_music_bed
                else "AI singing only (no music bed)"
            )
            summary = (
                f"**Status:** AI Song ready\n\n"
                f"**Original singer:** removed → replaced with **AI voice**\n\n"
                f"**Output audio:** {bed_note}\n\n"
                f"**Lyrics:** same language · **no translation**\n\n"
                f"**Detected language:** `{result.detected_language}`\n\n"
                f"**Lyrics (AI voice)**\n\n{lyrics_preview}\n\n"
                f"**Download:** `{Path(result.output_video).name if result.output_video else ''}`"
            )
            yield (
                _loading_html(False, 100, "Done"),
                100,
                _progress_status(100, "Done — download AI song below."),
                _btn_ready_song(),
                summary,
                str(result.output_video) if result.output_video else None,
                str(result.ai_vocals_path) if result.ai_vocals_path else None,
                str(result.mix_path) if result.mix_path else None,
                str(result.music_path) if result.music_path else None,
                str(result.lyrics_srt or result.khmer_srt) if (result.lyrics_srt or result.khmer_srt) else None,
            )
            return


def _preview_text(text: str, limit: int = 3500) -> str:
    """Keep Gradio markdown payloads small for hour-long transcripts."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n_…truncated — full text is in the .srt download._"


def update_estimate(
    video, mode_label, model_label, generate_story, add_music, keep_original_music, fast_mode
):
    try:
        msg = estimate_message(
            video,
            mode_label,
            model_label,
            generate_story,
            add_music,
            MODES,
            MODELS,
            fast=bool(fast_mode),
            keep_original_music=bool(keep_original_music),
        )
        # Friendly note when the sample is very long
        try:
            from core.tts import get_duration_seconds

            path = None
            if isinstance(video, (str, Path)):
                path = Path(video)
            elif isinstance(video, dict):
                for key in ("path", "name", "video"):
                    if video.get(key):
                        path = Path(video[key])
                        break
            elif video is not None:
                path = Path(getattr(video, "name", None) or str(video))
            if path and path_exists_safe(path):
                dur = get_duration_seconds(path)
                if dur >= 3600:
                    msg += (
                        "\n\n**Long video (1h+):** conversion can take a long time. "
                        "Keep the tab open. Soft subtitles / SRT-only modes are faster."
                    )
        except Exception:
            pass
        return msg
    except Exception as exc:
        return f"**Estimated time:** could not read video ({exc})"


_UI_CSS = """
.vk-auth-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  padding: 0;
  margin: 0 0 8px;
  border: 0;
  background: transparent;
}
.vk-auth-warn {
  color: #9a6b12 !important;
  font-size: 0.86rem;
  margin-right: auto;
}
.vk-auth-actions {
  display: flex;
  gap: 8px;
  align-items: center;
  position: relative;
  margin-left: auto;
  width: fit-content;
}
.vk-auth-guest,
.vk-auth-dropdown {
  display: none;
  align-items: center;
  gap: 8px;
  position: relative;
}
.vk-auth-guest.is-visible,
.vk-auth-dropdown.is-visible {
  display: flex;
}
.vk-auth-bar a.vk-auth-link,
.vk-auth-bar button.vk-auth-trigger {
  display: inline-flex !important;
  align-items: center;
  gap: 8px;
  text-decoration: none !important;
  border: 1px solid #c5d5cb !important;
  border-radius: 999px !important;
  padding: 8px 16px !important;
  font-size: 0.92rem !important;
  line-height: 1.2 !important;
  color: #14241c !important;
  background: #ffffff !important;
  cursor: pointer;
  font-family: inherit !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}
.vk-auth-bar a.vk-auth-primary {
  border-color: #c9a227 !important;
  background: #f6e7b0 !important;
  color: #14241c !important;
  font-weight: 700 !important;
}
.vk-auth-bar button.vk-auth-trigger {
  max-width: min(340px, 78vw);
  border-color: #c9a227 !important;
  background: #fffdf4 !important;
  color: #14241c !important;
}
.vk-auth-bar .vk-auth-trigger-label,
.vk-auth-bar .vk-auth-caret {
  color: #14241c !important;
  opacity: 1 !important;
}
.vk-auth-trigger-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 260px;
}
.vk-auth-caret {
  font-size: 0.75rem;
}
.vk-auth-menu {
  display: none;
  position: absolute;
  right: 0;
  top: calc(100% + 8px);
  min-width: 168px;
  padding: 6px;
  border-radius: 12px;
  border: 1px solid #d5e3db;
  background: #ffffff !important;
  box-shadow: 0 12px 28px rgba(20, 40, 30, 0.16);
  z-index: 1000;
}
.vk-auth-menu.is-open {
  display: block;
}
.vk-auth-bar button.vk-auth-menu-item {
  width: 100%;
  border: 0 !important;
  background: transparent !important;
  text-align: left !important;
  padding: 10px 12px !important;
  border-radius: 8px !important;
  font: inherit !important;
  font-size: 0.92rem !important;
  font-weight: 600 !important;
  color: #8b2e2e !important;
  cursor: pointer;
  box-shadow: none !important;
}
.vk-auth-bar button.vk-auth-menu-item:hover {
  background: #f8eeee !important;
}
.vk-loading {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  padding: 14px 16px;
  border-radius: 10px;
  border: 1px solid #c5d9ce;
  background: linear-gradient(120deg, #f3faf6 0%, #eef5fb 100%);
  margin: 8px 0 4px;
}
.vk-loading.vk-idle {
  border-color: #d5dce3;
  background: #f7f8fa;
  opacity: 0.9;
}
.vk-loading.vk-active {
  border-color: #7eb89a;
  box-shadow: 0 0 0 1px rgba(26, 127, 90, 0.08);
}
.vk-spinner {
  width: 28px;
  height: 28px;
  margin-top: 2px;
  border: 3px solid #c9ddd2;
  border-top-color: #1a7f5a;
  border-radius: 50%;
  animation: vkspin 0.75s linear infinite;
  flex-shrink: 0;
}
.vk-dot {
  width: 12px;
  height: 12px;
  margin-top: 6px;
  border-radius: 50%;
  background: #9aa7b5;
  flex-shrink: 0;
}
.vk-msg {
  margin-top: 4px;
  font-size: 0.92em;
  color: #3d4a57;
  line-height: 1.35;
}
.vk-bar {
  margin-top: 10px;
  height: 8px;
  width: 100%;
  background: #d9e6df;
  border-radius: 999px;
  overflow: hidden;
}
.vk-bar > span {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #1a7f5a, #3aa0c8);
  border-radius: 999px;
  transition: width 0.25s ease;
}
@keyframes vkspin {
  to { transform: rotate(360deg); }
}
"""


def _auth_bar_html() -> str:
    """Right-aligned account control: guest links or name dropdown + Log out."""
    configured = is_firebase_configured()
    setup = (
        ""
        if configured
        else '<span class="vk-auth-warn">Firebase not configured — see AUTH.md</span>'
    )
    return f"""
<div class="vk-auth-bar" id="vk-auth-bar">
  {setup}
  <div class="vk-auth-actions">
    <div id="vk-auth-guest" class="vk-auth-guest">
      <a class="vk-auth-link" href="/auth/login.html?next=/app">Log in</a>
      <a class="vk-auth-link vk-auth-primary" href="/auth/register.html?next=/app">Register</a>
    </div>
    <div id="vk-auth-dropdown" class="vk-auth-dropdown">
      <button type="button" class="vk-auth-trigger" id="vk-auth-trigger" aria-haspopup="menu" aria-expanded="false">
        <span class="vk-auth-trigger-label" id="vk-auth-user">Account</span>
        <span class="vk-auth-caret" aria-hidden="true">▾</span>
      </button>
      <div class="vk-auth-menu" id="vk-auth-menu" role="menu">
        <button type="button" class="vk-auth-menu-item" id="vk-auth-logout" role="menuitem">Log out</button>
      </div>
    </div>
  </div>
</div>
"""


def _auth_head() -> str:
    require = "true" if _REQUIRE_LOGIN else "false"
    return (
        f"<script>window.__VK_REQUIRE_LOGIN__={require};</script>"
        '<script type="module" src="/auth/app_auth_bar.js"></script>'
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Khmer AI Video Tools") as demo:
        gr.HTML(_auth_bar_html(), js_on_load=None, container=False, padding=False)
        gr.Markdown(
            """
            # Khmer AI Video Tools
            **Tab 1:** Upload a video → Khmer voice / subtitles (translates).  
            **Tab 2:** Write text → **AI voice + AI scene video**.  
            **Tab 3:** Upload a **song video** → AI singing voice + keep music (**no translate**).
            """
        )

        with gr.Tabs():
            with gr.Tab("Video → Khmer"):
                gr.Markdown(
                    """
                    Upload a video in **English, Chinese, Thai,** or other languages.
                    Transcribe → translate to **Khmer (ខ្មែរ)** → subtitles / dub / story.
                    **Keep original music** is ON by default for Khmer voice modes.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        video_in = gr.File(
                            label="Upload video (MP4/MOV/MKV/WebM… · up to 5 GB · 1h+ OK, needs time)",
                            file_types=["video", ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv"],
                            type="filepath",
                        )
                        source_lang = gr.Dropdown(
                            choices=list(SOURCE_LANGUAGES.values()),
                            value="Auto detect",
                            label="Source spoken language",
                        )
                        mode = gr.Dropdown(
                            choices=list(MODES.keys()),
                            value="Khmer voice + subtitles (SPEAKS Khmer)",
                            label="Output mode — pick a SPEAKS Khmer option to hear Khmer",
                        )
                        model = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="Base (recommended quality)",
                            label="Whisper model size",
                        )
                        voice = gr.Dropdown(
                            choices=list(KHMER_VOICES.keys()),
                            value="Female (Sreymom)",
                            label="Khmer TTS voice (for dub mode)",
                        )

                        gr.Markdown("### Extra options")
                        fast_mode = gr.Checkbox(
                            label="Faster encode (remux / ultrafast) — voice stays timed for quality",
                            value=True,
                            info="Keeps clearer timed Khmer voice. Uncheck only if encode fails.",
                        )
                        generate_story = gr.Checkbox(
                            label="Generate story (Khmer narrative script + voice/subs from story)",
                            value=False,
                            info="Turns the video transcript into a short Khmer story before speaking.",
                        )
                        keep_original_music = gr.Checkbox(
                            label="Keep original music — ON = music from upload stays under Khmer voice",
                            value=True,
                            info="For SPEAKS Khmer modes: keeps the uploaded video soundtrack. Soft/burned subs already keep full original audio.",
                        )
                        add_music = gr.Checkbox(
                            label="Add soft procedural BGM (only if Keep original music is OFF)",
                            value=False,
                            info="Synthetic calm music — ignored when Keep original music is ON.",
                        )
                        show_note = gr.Checkbox(
                            label="Show video note (top right)",
                            value=True,
                        )
                        video_note = gr.Textbox(
                            label="Video note text",
                            value="Cinema Summary",
                            placeholder="Cinema Summary",
                        )

                        run_btn = gr.Button("Convert to Khmer", variant="primary")
                        estimate_box = gr.Markdown(
                            value=update_estimate(
                                None,
                                "Khmer voice + subtitles (SPEAKS Khmer)",
                                "Base (recommended quality)",
                                False,
                                False,
                                True,
                                True,
                            ),
                            elem_id="estimate_time",
                        )

                        gr.Markdown("### Progress")
                        loading_panel = gr.HTML(
                            value=_loading_html(False),
                            elem_id="vk_loading",
                        )
                        progress_bar = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=1,
                            interactive=False,
                            label="Generate progress (%)",
                            elem_id="bottom_progress",
                        )
                        progress_status = gr.Markdown("**Progress: 0%** — Ready")

                    with gr.Column(scale=1):
                        summary = gr.Markdown(label="Result")
                        khmer_srt = gr.File(label="Download Khmer subtitles (.srt)")
                        original_srt = gr.File(label="Download original-language subtitles (.srt)")
                        story_file = gr.File(label="Download Khmer story (.txt)")
                        music_file = gr.File(label="Download music bed (.wav)")
                        video_download = gr.File(label="Download output video (.mp4)")

                estimate_inputs = [
                    video_in,
                    mode,
                    model,
                    generate_story,
                    add_music,
                    keep_original_music,
                    fast_mode,
                ]
                for comp in estimate_inputs:
                    comp.change(
                        fn=update_estimate,
                        inputs=estimate_inputs,
                        outputs=estimate_box,
                    )

                run_btn.click(
                    fn=process,
                    inputs=[
                        video_in,
                        source_lang,
                        mode,
                        model,
                        voice,
                        generate_story,
                        add_music,
                        keep_original_music,
                        show_note,
                        video_note,
                        fast_mode,
                    ],
                    outputs=[
                        loading_panel,
                        progress_bar,
                        progress_status,
                        run_btn,
                        summary,
                        khmer_srt,
                        original_srt,
                        story_file,
                        music_file,
                        video_download,
                    ],
                    show_progress="full",
                )

                gr.Markdown(
                    """
                    ### Tips (Video → Khmer)
                    - **Keep original music** (default ON): Khmer voice + soundtrack from your upload.
                    - Whisper **Base** = better text. Keep a stable internet for Edge TTS.
                    - **Video note** (default: *Cinema Summary*) shows in the **top right** of the output video.
                    - Long Chinese filenames are auto-staged to a short path.
                    - Download MP4 and open in VLC.
                    """
                )

            with gr.Tab("Text → AI Video"):
                gr.Markdown(
                    """
                    Write a **short title** — AI auto-builds the story video (script detail optional).  
                    Example title: **A Poor Cat Becomes a Millionaire**  
                    Set **Hours / Minutes** for target video length.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        title_in = gr.Textbox(
                            label="Video title (main)",
                            lines=2,
                            value="A Poor Cat Becomes a Millionaire",
                            placeholder="A Poor Cat Becomes a Millionaire",
                        )
                        script_in = gr.Textbox(
                            label="Optional script / visual prompt (not required for title mode)",
                            lines=6,
                            placeholder=(
                                "Leave empty for title-only AI story.\n"
                                "Or paste a visual prompt / narration if you want more control."
                            ),
                        )
                        with gr.Row():
                            tv_hours = gr.Number(
                                label="Length — Hours",
                                value=0,
                                minimum=0,
                                maximum=2,
                                precision=0,
                            )
                            tv_minutes = gr.Number(
                                label="Length — Minutes",
                                value=1,
                                minimum=0,
                                maximum=59,
                                precision=0,
                            )
                        tv_mode = gr.Dropdown(
                            choices=list(TEXT_VIDEO_MODES.keys()),
                            value="Title story (from title only)",
                            label="Input type",
                        )
                        tv_source = gr.Dropdown(
                            choices=list(SOURCE_LANGUAGES.values()),
                            value="Auto detect",
                            label="Script language (for narration scripts)",
                        )
                        tv_voice = gr.Dropdown(
                            choices=list(KHMER_VOICES.keys()),
                            value="Female (Sreymom)",
                            label="AI voice (Khmer TTS)",
                        )
                        tv_style = gr.Dropdown(
                            choices=list(VIDEO_STYLES.keys()),
                            value="Cinematic",
                            label="Extra style (script mode / soft boost)",
                        )
                        tv_music = gr.Checkbox(
                            label="Add soft background music",
                            value=False,
                        )
                        tv_captions = gr.Checkbox(
                            label="Show Khmer captions on video (turn OFF for animal/visual prompts)",
                            value=False,
                        )
                        tv_captions_en = gr.Checkbox(
                            label="Show English captions on video (turn OFF for animal/visual prompts)",
                            value=False,
                        )
                        tv_show_note = gr.Checkbox(
                            label="Show video note (top right)",
                            value=True,
                        )
                        tv_note = gr.Textbox(
                            label="Video note text",
                            value="Cinema Summary",
                            placeholder="Cinema Summary",
                        )
                        tv_title_en = gr.Checkbox(
                            label="Show English title (top right)",
                            value=True,
                        )
                        tv_title_kh = gr.Checkbox(
                            label="Show Khmer title (top right)",
                            value=False,
                        )
                        tv_fast = gr.Checkbox(
                            label="Faster encode",
                            value=True,
                        )
                        tv_btn = gr.Button("Generate AI video", variant="primary")

                        gr.Markdown("### Progress")
                        tv_loading = gr.HTML(value=_loading_html(False))
                        tv_bar = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=1,
                            interactive=False,
                            label="Generate progress (%)",
                        )
                        tv_status = gr.Markdown("**Progress: 0%** — Ready")

                    with gr.Column(scale=1):
                        tv_summary = gr.Markdown(label="Result")
                        tv_video = gr.File(label="Download AI video (.mp4)")
                        tv_audio = gr.File(label="Download AI voice (.mp3 / .m4a)")

                tv_btn.click(
                    fn=process_text_video,
                    inputs=[
                        title_in,
                        script_in,
                        tv_source,
                        tv_voice,
                        tv_style,
                        tv_mode,
                        tv_hours,
                        tv_minutes,
                        tv_music,
                        tv_captions,
                        tv_captions_en,
                        tv_show_note,
                        tv_note,
                        tv_title_en,
                        tv_title_kh,
                        tv_fast,
                    ],
                    outputs=[
                        tv_loading,
                        tv_bar,
                        tv_status,
                        tv_btn,
                        tv_summary,
                        tv_video,
                        tv_audio,
                    ],
                    show_progress="full",
                )

                gr.Markdown(
                    """
                    ### Standard generate (title → AI video)
                    1. Title: **A Poor Cat Becomes a Millionaire** (or your own)
                    2. Input type: **Title story (from title only)**
                    3. Length: **0 hours + 1–3 minutes** (start short; max 2 hours)
                    4. Show English title: **ON**
                    5. Captions: **OFF** (optional later)
                    6. Faster encode: **ON** → Generate AI video

                    Script detail is **not required** — AI expands the title into scenes + Khmer voice.

                    | Length tip | Scenes (approx.) |
                    |------------|------------------|
                    | 1 minute | ~6 scenes |
                    | 5 minutes | ~30 scenes |
                    | 1 hour | ~40 longer-held scenes |

                    ### Tips
                    - Optional script/prompt only if you want extra control.
                    - Needs internet for AI images + Edge TTS.
                    """
                )

            with gr.Tab("Video → Song AI"):
                gr.Markdown(
                    """
                    Upload a **song video**. The app will:
                    1. **Remove** the original singer  
                    2. **Keep** the music / instrumental  
                    3. **Change the singing voice to AI** (same lyrics — **no translation**)
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        song_video = gr.File(
                            label="Upload song video (MP4, MOV, MKV, WebM…)",
                            file_types=["video", ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"],
                            type="filepath",
                        )
                        song_source = gr.Dropdown(
                            choices=list(SOURCE_LANGUAGES.values()),
                            value="Auto detect",
                            label="Lyric language (read lyrics only — not translated)",
                        )
                        song_model = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="Small (more accurate)",
                            label="Whisper model (lyrics)",
                        )
                        song_voice = gr.Dropdown(
                            choices=list(SONG_AI_VOICES.keys()),
                            value="Auto (match song language)",
                            label="AI singing voice (replaces original singer)",
                        )
                        song_sing = gr.Checkbox(
                            label="Sing mode (follow original melody) — keep ON for real singing",
                            value=True,
                            info="Uses original pitch as a guide. Original singer is still removed.",
                        )
                        song_keep_music = gr.Checkbox(
                            label="Keep music bed — ON = AI voice + music (recommended)",
                            value=True,
                            info="ON: music stays under AI singing. OFF: AI singing only.",
                        )
                        song_captions = gr.Checkbox(
                            label="Burn lyric captions on video (Khmer lyrics only)",
                            value=False,
                            info="Optional captions — still no translation.",
                        )
                        song_show_note = gr.Checkbox(
                            label="Show video note (top right)",
                            value=True,
                        )
                        song_note = gr.Textbox(
                            label="Video note text",
                            value="AI Song",
                            placeholder="AI Song",
                        )
                        song_music_vol = gr.Slider(
                            minimum=0.3,
                            maximum=1.2,
                            value=0.65,
                            step=0.02,
                            label="Music bed volume",
                        )
                        song_vocal_vol = gr.Slider(
                            minimum=0.8,
                            maximum=2.5,
                            value=2.0,
                            step=0.05,
                            label="AI singing volume",
                        )
                        song_fast = gr.Checkbox(label="Faster encode", value=True)
                        song_btn = gr.Button("Build AI Song", variant="primary")

                        gr.Markdown("### Progress")
                        song_loading = gr.HTML(value=_loading_html(False))
                        song_bar = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=1,
                            interactive=False,
                            label="Generate progress (%)",
                        )
                        song_status = gr.Markdown("**Progress: 0%** — Ready")

                    with gr.Column(scale=1):
                        song_summary = gr.Markdown(label="Result")
                        song_out_video = gr.File(label="Download AI Song video (.mp4)")
                        song_out_vocals = gr.File(label="Download AI vocals only (.mp3)")
                        song_out_mix = gr.File(label="Download AI audio (.m4a)")
                        song_out_music = gr.File(label="Download separated instrumental (.wav)")
                        song_out_srt = gr.File(label="Download lyrics (.srt)")

                song_btn.click(
                    fn=process_song_ai,
                    inputs=[
                        song_video,
                        song_source,
                        song_model,
                        song_voice,
                        song_sing,
                        song_keep_music,
                        song_captions,
                        song_show_note,
                        song_note,
                        song_music_vol,
                        song_vocal_vol,
                        song_fast,
                    ],
                    outputs=[
                        song_loading,
                        song_bar,
                        song_status,
                        song_btn,
                        song_summary,
                        song_out_video,
                        song_out_vocals,
                        song_out_mix,
                        song_out_music,
                        song_out_srt,
                    ],
                    show_progress="full",
                )

                gr.Markdown(
                    """
                    ### Tips (Video → Song AI)
                    - **Replace singer with AI voice** · **keep music** · **no translation**.
                    - English stays English, Chinese stays Chinese, Urdu stays Urdu, etc.
                    - Use **Tab 1 (Video → Khmer)** only if you want speech translated to Khmer.
                    - Adjust **AI singing volume** / **Music bed volume** to balance the mix.
                    - Needs **Demucs** + singing libs: `pip install -U demucs torch torchaudio librosa soundfile`
                    - Use **Auto** voice to match the song language, or pick a voice manually.
                    - Keep **Sing mode** ON so AI vocals follow the original melody.
                    - Use **Small** Whisper model for clearer lyric transcription.
                    - First Demucs run downloads a model (can take a few minutes).
                    """
                )

    return demo


def create_app() -> FastAPI:
    """FastAPI host: login first at /, Gradio tools at /app."""
    api = FastAPI(title="Khmer AI Video Tools")

    @api.get("/")
    def root():
        # First open → login page (already-signed-in users are bounced to /app by login.html)
        return RedirectResponse(url="/auth/login.html?next=/app")

    @api.get("/auth/config.json")
    def auth_config():
        cfg = load_firebase_config()
        if not cfg:
            return JSONResponse(
                {
                    "apiKey": "YOUR_FIREBASE_API_KEY",
                    "authDomain": "YOUR_PROJECT.firebaseapp.com",
                    "projectId": "YOUR_PROJECT_ID",
                    "storageBucket": "YOUR_PROJECT.appspot.com",
                    "messagingSenderId": "YOUR_SENDER_ID",
                    "appId": "YOUR_APP_ID",
                    "configured": False,
                },
                status_code=200,
            )
        return {**cfg, "configured": True}

    @api.get("/auth")
    @api.get("/auth/")
    def auth_root():
        return RedirectResponse(url="/auth/login.html?next=/app")

    # Static login/register assets (html/css/js). config.json is served above.
    if _AUTH_DIR.is_dir():
        api.mount(
            "/auth",
            StaticFiles(directory=str(_AUTH_DIR), html=True),
            name="auth-static",
        )

    demo = build_ui()
    return gr.mount_gradio_app(
        api,
        demo,
        path=_APP_PATH,
        css=_UI_CSS,
        head=_auth_head(),
        max_file_size="5gb",
    )


if __name__ == "__main__":
    import socket

    import uvicorn

    def _port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    port = next((p for p in range(7860, 7871) if _port_free(p)), None)
    if port is None:
        raise SystemExit("No free port in 7860–7870. Close the old app and try again.")

    app = create_app()
    print(f"Open (login first): http://127.0.0.1:{port}/")
    print(f"Login page:         http://127.0.0.1:{port}/auth/login.html")
    print(f"Register page:      http://127.0.0.1:{port}/auth/register.html")
    print(f"App (after login):  http://127.0.0.1:{port}/app")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
