"""Render Khmer text with correct coeng (feet) / vowel (hands) shaping via HarfBuzz."""

from __future__ import annotations

import re
from pathlib import Path

import freetype
import uharfbuzz as hb
from PIL import Image, ImageDraw

from .fonts import ensure_battambang_fonts


def _shape(text: str, font_path: Path, font_size: int):
    """Return (glyph_ids, x_advances, x_offsets, y_offsets) in 26.6 / font units scaled to px."""
    data = font_path.read_bytes()
    face = hb.Face(data)
    font = hb.Font(face)
    upem = face.upem
    scale = int(font_size * 64 * (upem / upem))  # pt-ish
    # HarfBuzz expects scale in 26.6 for pixel size roughly: font_size * 64
    font.scale = (font_size << 6, font_size << 6)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)

    infos = buf.glyph_infos
    positions = buf.glyph_positions
    return infos, positions, font_size, upem


def render_khmer_line(
    text: str,
    *,
    font_size: int = 42,
    max_width: int = 1000,
    max_lines: int = 2,
    prefer_two_lines: bool = True,
    fill=(255, 255, 255, 255),
    stroke=(0, 0, 0, 255),
    stroke_width: int = 2,
    bold: bool = True,
) -> Image.Image:
    """
    Render Khmer string to a transparent RGBA image with proper shaping.
    Always fits within max_width (shrink font / truncate if needed).
    """
    text = (text or "").replace("\u200b", "").strip()
    if not text:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    fonts = ensure_battambang_fonts()
    font_path = fonts / ("Battambang-Bold.ttf" if bold else "Battambang-Regular.ttf")
    if not font_path.exists():
        font_path = fonts / "Battambang-Regular.ttf"

    # Leave room for outline padding so rendered line stays inside the frame
    fit_width = max(80, int(max_width) - (stroke_width + 4) * 2)

    size = max(18, int(font_size))
    lines: list[str] = [text]
    for try_size in range(size, 17, -2):
        if prefer_two_lines and max_lines >= 2:
            lines = _balance_two_lines(text, font_path, try_size, fit_width)
        else:
            lines = _wrap_khmer(text, font_path, try_size, fit_width, max_lines=max_lines)
        lines = [_clamp_line_width(ln, font_path, try_size, fit_width) for ln in lines if ln]
        if not lines:
            lines = [text]
        if all(_measure_width(ln, font_path, try_size) <= fit_width for ln in lines):
            size = try_size
            break
        size = try_size

    line_images = [
        _render_single_line(line, font_path, size, fill, stroke, stroke_width)
        for line in lines
        if line
    ]
    if not line_images:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    # Extra gap so Khmer feet / vowel marks on adjacent lines don't collide
    gap = max(18, size // 2)
    width = max(im.width for im in line_images)
    height = sum(im.height for im in line_images) + gap * (len(line_images) - 1)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    y = 0
    for im in line_images:
        x = (width - im.width) // 2
        canvas.alpha_composite(im, (x, y))
        y += im.height + gap
    return canvas


def _clamp_line_width(
    text: str,
    font_path: Path,
    font_size: int,
    max_width: int,
) -> str:
    """Truncate a single line with ... so it never exceeds max_width."""
    text = (text or "").strip()
    if not text:
        return text
    if _measure_width(text, font_path, font_size) <= max_width:
        return text
    ell = "..."
    tokens = _caption_tokens(text)
    if len(tokens) >= 2:
        kept = ""
        for tok in tokens:
            trial = kept + tok
            if kept and _measure_width(trial + ell, font_path, font_size) > max_width:
                break
            kept = trial
        if kept:
            return kept.rstrip() + ell
    # Cluster / char trim fallback
    out = text
    while out and _measure_width(out + ell, font_path, font_size) > max_width:
        out = out[:-1]
    return (out.rstrip() + ell) if out else ell


def _caption_tokens(text: str) -> list[str]:
    """
    Break caption text into wrap units.
    Prefer spaces; otherwise Khmer orthographic clusters (keep coeng/vowels intact).
    """
    spaced = [p for p in text.replace(" ", "\u200b").split("\u200b") if p]
    if len(spaced) >= 2:
        return spaced
    return _khmer_clusters(text) or [text]


def _khmer_clusters(text: str) -> list[str]:
    """Split Khmer into wrap-safe clusters (base + coeng/subscript + marks)."""
    import unicodedata

    clusters: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        unit = ch
        i += 1
        while i < n:
            nxt = text[i]
            # COENG + following consonant (subscript)
            if nxt == "\u17d2" and i + 1 < n:
                unit += nxt + text[i + 1]
                i += 2
                continue
            # Dependent vowels / signs / combining marks
            if ("\u17b6" <= nxt <= "\u17d3") or unicodedata.category(nxt) in ("Mn", "Mc"):
                unit += nxt
                i += 1
                continue
            break
        clusters.append(unit)
    return clusters


def _balance_two_lines(
    text: str,
    font_path: Path,
    font_size: int,
    max_width: int,
) -> list[str]:
    """
    Prefer exactly **2 lines** on screen, each within max_width.
    Falls back to width-aware wrap (never overflows the frame).
    """
    text = text.strip()
    tokens = _caption_tokens(text)
    if not tokens:
        return [text]

    # Too short for 2 lines
    if len(text) < 22 and _measure_width(text, font_path, font_size) <= max_width * 0.7:
        return [text]

    if len(tokens) < 2:
        return _wrap_khmer(text, font_path, font_size, max_width, max_lines=2)

    best: list[str] | None = None
    best_score = 10**18
    for i in range(1, len(tokens)):
        line1 = "".join(tokens[:i])
        line2 = "".join(tokens[i:])
        if not line1.strip() or not line2.strip():
            continue
        w1 = _measure_width(line1, font_path, font_size)
        w2 = _measure_width(line2, font_path, font_size)
        if w1 > max_width or w2 > max_width:
            continue
        mid_bias = abs(i - len(tokens) / 2) * 2
        score = abs(w1 - w2) + abs(len(line1) - len(line2)) + mid_bias
        if score < best_score:
            best_score = score
            best = [line1, line2]

    if best:
        return best

    # No balanced split fits — wrap by pixel width (truncate 3rd+ line)
    return _wrap_khmer(text, font_path, font_size, max_width, max_lines=2)

def _wrap_khmer(
    text: str,
    font_path: Path,
    font_size: int,
    max_width: int,
    max_lines: int = 2,
) -> list[str]:
    """Wrap to width, hard-capped at max_lines (1–2 for video captions)."""
    max_lines = max(1, min(2, int(max_lines)))
    parts = _caption_tokens(text)
    if not parts:
        return [text]

    lines: list[str] = []
    current = ""
    truncated = False

    for part in parts:
        # Single token wider than frame → clamp immediately
        if not current and _measure_width(part, font_path, font_size) > max_width:
            lines.append(_clamp_line_width(part, font_path, font_size, max_width))
            if len(lines) >= max_lines:
                truncated = True
                current = ""
                break
            continue

        trial = part if not current else (current + part)
        w = _measure_width(trial, font_path, font_size)
        if current and w > max_width:
            lines.append(current)
            current = part
            if len(lines) >= max_lines:
                truncated = True
                current = ""
                break
        else:
            current = trial

    if current and len(lines) < max_lines:
        lines.append(current)
    elif current and len(lines) >= max_lines:
        truncated = True

    if not lines:
        lines = [_clamp_line_width(text, font_path, font_size, max_width)]

    lines = lines[:max_lines]
    lines = [_clamp_line_width(ln, font_path, font_size, max_width) for ln in lines]

    if truncated and lines and not lines[-1].endswith("..."):
        last = lines[-1].rstrip(".")
        while last and _measure_width(last + "...", font_path, font_size) > max_width:
            last = last[:-1]
        lines[-1] = (last.rstrip() + "...") if last else "..."

    return lines


def _measure_width(text: str, font_path: Path, font_size: int) -> int:
    infos, positions, _, _ = _shape(text, font_path, font_size)
    x = 0
    for pos in positions:
        x += pos.x_advance
    return max(1, x >> 6)


def _render_single_line(
    text: str,
    font_path: Path,
    font_size: int,
    fill,
    stroke,
    stroke_width: int,
) -> Image.Image:
    infos, positions, _, upem = _shape(text, font_path, font_size)

    ft_face = freetype.Face(str(font_path))
    ft_face.set_char_size(font_size * 64)

    # Compute bounding box
    pen_x = 0
    pen_y = 0
    min_x = min_y = 10**9
    max_x = max_y = -10**9
    glyphs: list[tuple] = []

    for info, pos in zip(infos, positions):
        gx = pen_x + pos.x_offset
        gy = pen_y + pos.y_offset
        ft_face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_FORCE_AUTOHINT)
        bitmap = ft_face.glyph.bitmap
        top = ft_face.glyph.bitmap_top
        left = ft_face.glyph.bitmap_left
        w, h = bitmap.width, bitmap.rows
        pitch = bitmap.pitch

        x0 = (gx >> 6) + left
        y0 = -(gy >> 6) - top
        x1 = x0 + w
        y1 = y0 + h
        min_x = min(min_x, x0)
        min_y = min(min_y, y0)
        max_x = max(max_x, x1)
        max_y = max(max_y, y1)

        if w > 0 and h > 0:
            # FreeType pitch may be larger than width (row padding)
            raw = bytes(bitmap.buffer)
            if pitch == w:
                buf = raw
            else:
                rows = []
                for row in range(h):
                    start = row * abs(pitch)
                    rows.append(raw[start : start + w])
                buf = b"".join(rows)
            glyphs.append((buf, x0, y0, w, h))

        pen_x += pos.x_advance
        pen_y += pos.y_advance

    if min_x > max_x:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    pad = stroke_width + 4
    width = max_x - min_x + pad * 2
    height = max_y - min_y + pad * 2
    stroke_img = Image.new("L", (width, height), 0)
    fill_img = Image.new("L", (width, height), 0)

    for buf, x0, y0, w, h in glyphs:
        glyph_im = Image.frombytes("L", (w, h), buf)
        px = x0 - min_x + pad
        py = y0 - min_y + pad
        fill_img.paste(glyph_im, (px, py), glyph_im)

    if stroke_width > 0:
        # Expand fill mask for outline
        from PIL import ImageFilter

        stroke_img = fill_img.filter(ImageFilter.MaxFilter(stroke_width * 2 + 1))

    rgba = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    # stroke behind
    if stroke_width > 0:
        stroke_rgba = Image.new("RGBA", (width, height), stroke)
        stroke_rgba.putalpha(stroke_img)
        rgba = Image.alpha_composite(rgba, stroke_rgba)
    fill_rgba = Image.new("RGBA", (width, height), fill)
    fill_rgba.putalpha(fill_img)
    rgba = Image.alpha_composite(rgba, fill_rgba)
    return rgba


def save_subtitle_png(
    text: str,
    out_path: str | Path,
    *,
    video_width: int = 1280,
    font_size: int | None = None,
    max_lines: int = 2,
) -> Path:
    """Render Khmer subtitle PNG as **2 lines** when text is long enough."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if font_size is None:
        font_size = max(26, video_width // 34)
    max_width = int(video_width * 0.82)
    img = render_khmer_line(
        text,
        font_size=font_size,
        max_width=max_width,
        max_lines=max_lines,
        prefer_two_lines=True,
        stroke_width=2,
    )
    # Never wider than the video frame
    max_cap_w = int(video_width * 0.92)
    if img.width > max_cap_w:
        ratio = max_cap_w / img.width
        img = img.resize(
            (max_cap_w, max(1, int(img.height * ratio))),
            Image.Resampling.LANCZOS,
        )
    img.save(out_path, "PNG")
    return out_path
