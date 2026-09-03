"""Shared helpers for video → Khmer conversion."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg

# Language codes Whisper / Google Translate understand
SOURCE_LANGUAGES = {
    "auto": "Auto detect",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
    "km": "Khmer",
    "vi": "Vietnamese",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "id": "Indonesian",
    "ms": "Malay",
    "hi": "Hindi",
    "ur": "Urdu (Pakistan)",
    "ru": "Russian",
    "ar": "Arabic",
}

KHMER_VOICES = {
    "Female (Sreymom)": "km-KH-SreymomNeural",
    "Male (Piseth)": "km-KH-PisethNeural",
}

TARGET_LANG = "km"

_TOOLS_DIR = Path(__file__).resolve().parent.parent / ".tools"
_ffmpeg_shim_ready = False


def ensure_ffmpeg_on_path() -> Path:
    """
    Gradio's Video preview looks for a binary named `ffmpeg` on PATH.
    imageio-ffmpeg ships a differently named exe — copy it as ffmpeg.exe
    into .tools/ and prepend that folder to PATH.
    """
    global _ffmpeg_shim_ready
    _TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    shim = _TOOLS_DIR / "ffmpeg.exe"

    if not shim.exists() or shim.stat().st_size == 0:
        src = Path(imageio_ffmpeg.get_ffmpeg_exe())
        shutil.copy2(src, shim)

    tools = str(_TOOLS_DIR)
    path = os.environ.get("PATH", "")
    if tools not in path.split(os.pathsep):
        os.environ["PATH"] = tools + os.pathsep + path

    _ffmpeg_shim_ready = True
    return shim


def sanitize_stem(name: str, max_len: int = 24) -> str:
    """
    Safe short ASCII folder/file stem.
    Long Chinese/TikTok titles caused WinError 206 (path too long) under Temp.
    """
    import hashlib

    stem = Path(name).stem
    ascii_part = re.sub(r"[^A-Za-z0-9\-]+", "_", stem)
    ascii_part = re.sub(r"_+", "_", ascii_part).strip("._-")
    if len(ascii_part) < 3:
        ascii_part = "v_" + hashlib.md5(name.encode("utf-8", errors="replace")).hexdigest()[:10]
    else:
        # Keep readable prefix + short hash so collisions stay rare
        digest = hashlib.md5(name.encode("utf-8", errors="replace")).hexdigest()[:6]
        ascii_part = f"{ascii_part[: max(8, max_len - 7)]}_{digest}"
    return ascii_part[:max_len]


def download_video_stem(
    source_text: str,
    *,
    label: str = "",
    max_len: int = 55,
) -> str:
    """
    Readable download filename stem from prompt/script text.
    Example: label 'Cinema Summary' + 'young woman rain' -> cinema-summary-young-woman-rain
    """
    import hashlib

    line = (source_text or "").strip().splitlines()[0][:140]
    label = (label or "").strip()

    def _slug_words(s: str, limit: int = 6) -> str:
        words = re.findall(r"[A-Za-z0-9]{2,}", s)
        if not words:
            return ""
        return "-".join(w.lower() for w in words[:limit])

    label_part = re.sub(r"[^A-Za-z0-9\-]+", "-", label).strip("-").lower()
    label_part = re.sub(r"-+", "-", label_part)
    content_part = _slug_words(line, 5)

    parts = [p for p in [label_part, content_part] if p]
    stem = "-".join(parts) if parts else ""
    stem = re.sub(r"-+", "-", stem).strip("-")
    if len(stem) < 3:
        digest = hashlib.md5(line.encode("utf-8", errors="replace")).hexdigest()[:8]
        stem = f"ai-video-{digest}"
    return stem[:max_len].rstrip("-")


def win_long_path(path: str | Path) -> str:
    """
    Windows extended-length path (\\\\?\\…) so APIs can open >260 char paths.
    Needed for Gradio uploads that keep long Chinese / TikTok filenames.
    """
    p = str(Path(path))
    if os.name != "nt":
        return p
    # Absolute / normalized without resolving symlinks (resolve can also hit 206)
    p = os.path.abspath(p)
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


def path_exists_safe(path: str | Path) -> bool:
    """exists() that tolerates WinError 206 on oversized paths."""
    try:
        return Path(path).exists()
    except OSError:
        try:
            return os.path.exists(win_long_path(path))
        except OSError:
            return False


def copy_upload_to_short_path(src: str | Path, dest_dir: str | Path) -> Path:
    """
    Copy (or hardlink) an uploaded video to a short ASCII path.

    Gradio keeps the original filename. Long Chinese titles + Temp folders
    exceed Windows MAX_PATH → WinError 206. ffmpeg also fails on those paths.
    """
    src = Path(src)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    ext = src.suffix.lower() if src.suffix else ".mp4"
    if len(ext) > 8 or not re.match(r"^\.[A-Za-z0-9]+$", ext):
        ext = ".mp4"
    dest = dest_dir / f"input{ext}"

    src_long = win_long_path(src)
    dest_long = win_long_path(dest)

    # Same volume: try hardlink (instant, no extra 1GB disk use)
    try:
        if os.name == "nt":
            os.link(src_long, dest_long)
        else:
            os.link(str(src), str(dest))
        return dest
    except OSError:
        pass

    try:
        shutil.copy2(src_long, dest_long)
    except OSError:
        # Last resort: open via extended path and stream copy
        with open(src_long, "rb") as rf, open(dest_long, "wb") as wf:
            shutil.copyfileobj(rf, wf, length=1024 * 1024 * 8)
        try:
            shutil.copystat(src_long, dest_long)
        except OSError:
            pass

    if not path_exists_safe(dest) or dest.stat().st_size <= 0:
        raise RuntimeError(
            "Could not stage upload to a short path (WinError 206 / copy failed). "
            "Rename the video to a short English name (e.g. video.mp4) and try again."
        )
    return dest


def preferred_temp_root() -> Path:
    """Prefer project .work on D: over C:\\Temp (C: filled up during TTS)."""
    root = Path(__file__).resolve().parent.parent / ".work"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return root
    except OSError:
        import tempfile

        return Path(tempfile.gettempdir())


def get_ffmpeg() -> str:
    """Return path to ffmpeg (shim / system / imageio-ffmpeg)."""
    ensure_ffmpeg_on_path()
    system = shutil.which("ffmpeg")
    if system:
        return system
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = [get_ffmpeg(), "-y", *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        low = err.lower()
        if "no space left" in low or "errno 28" in low:
            raise RuntimeError(
                "Disk full (ffmpeg: no space left on device). "
                "Free disk space and delete old folders under .work / Temp, then retry."
            )
        if "filename or extension is too long" in low or "winerror 206" in low:
            raise RuntimeError(
                "Path too long for Windows. Re-upload or rename the video to a short English name."
            )
        tail = err[-1200:] if err else "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg failed:\n{tail}")
    return result


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_timestamp(seconds: float) -> str:
    """SRT timestamp: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
