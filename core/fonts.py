"""Ensure Battambang (Khmer) font is available for burned-in subtitles."""

from __future__ import annotations

import urllib.request
from pathlib import Path

FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"

# Google Fonts OFL — used for ffmpeg libass subtitle burn-in
_FONT_FILES = {
    "Battambang-Regular.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/battambang/Battambang-Regular.ttf"
    ),
    "Battambang-Bold.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/battambang/Battambang-Bold.ttf"
    ),
}

FONT_FAMILY = "Battambang"


def ensure_battambang_fonts() -> Path:
    """Download Battambang TTFs into ./fonts if missing. Returns fonts directory."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in _FONT_FILES.items():
        dest = FONTS_DIR / name
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            if not dest.exists():
                raise RuntimeError(
                    f"Could not download Khmer font {name}. "
                    f"Check your internet connection.\n{exc}"
                ) from exc
    return FONTS_DIR


def fontsdir_for_ffmpeg() -> str:
    """Path escaped for ffmpeg subtitles filter fontsdir= on Windows."""
    fonts = ensure_battambang_fonts()
    # ffmpeg wants forward slashes; escape drive colon for filter graph
    return str(fonts).replace("\\", "/").replace(":", "\\:")


def khmer_subtitle_filter(srt_escaped: str, font_size: int = 26) -> str:
    """
    ffmpeg -vf subtitles=... using Battambang for Khmer text.
    PrimaryColour is ASS BGR: white text, black outline.
    """
    fontsdir = fontsdir_for_ffmpeg()
    style = (
        f"FontName={FONT_FAMILY},"
        f"FontSize={font_size},"
        "PrimaryColour=&H00FFFFFF&,"
        "OutlineColour=&H00000000&,"
        "BorderStyle=1,"
        "Outline=2,"
        "Shadow=1,"
        "MarginV=28"
    )
    return f"subtitles='{srt_escaped}':fontsdir='{fontsdir}':force_style='{style}'"
