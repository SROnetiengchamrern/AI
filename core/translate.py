"""Translate text segments to Khmer (quality-focused)."""

from __future__ import annotations

import re
import time

from deep_translator import GoogleTranslator

from . import TARGET_LANG
from .text_clean import clean_khmer_text, clean_source_text
from .transcribe import Segment, Transcript


def _chunk_text(text: str, max_chars: int = 4200) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in text.split():
        add = len(word) + (1 if current else 0)
        if length + add > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += add
    if current:
        chunks.append(" ".join(current))
    return chunks


def translate_text(text: str, source: str = "auto") -> str:
    text = clean_source_text(text)
    if not text:
        return ""
    src = "auto" if not source or source == "auto" else source
    if src == "zh":
        src = "zh-CN"

    translator = GoogleTranslator(source=src, target=TARGET_LANG)
    parts: list[str] = []
    for chunk in _chunk_text(text):
        translated = translator.translate(chunk) or ""
        parts.append(clean_khmer_text(translated))
        time.sleep(0.08)
    return " ".join(p for p in parts if p)


def _translate_batch(texts: list[str], source: str) -> list[str]:
    """Translate a small batch with a strong separator; fall back per-line."""
    cleaned = [clean_source_text(t) for t in texts]
    if not any(cleaned):
        return [""] * len(texts)

    sep = "\n¶\n"
    src = "auto" if not source or source == "auto" else source
    if src == "zh":
        src = "zh-CN"

    try:
        translator = GoogleTranslator(source=src, target=TARGET_LANG)
        raw = translator.translate(sep.join(cleaned)) or ""
        parts = [clean_khmer_text(p) for p in re.split(r"\s*¶\s*", raw)]
        if len(parts) == len(texts) and all(parts):
            return parts
    except Exception:
        pass

    return [translate_text(t, source=source) for t in cleaned]


def merge_nearby_segments(
    segments: list[Segment],
    *,
    max_gap: float = 0.55,
    max_chars: int = 110,
) -> list[Segment]:
    """Merge close Whisper fragments into fuller sentences (better translation)."""
    usable = [s for s in segments if clean_source_text(s.text)]
    if not usable:
        return []

    merged: list[Segment] = []
    cur = Segment(start=usable[0].start, end=usable[0].end, text=clean_source_text(usable[0].text))
    for s in usable[1:]:
        t = clean_source_text(s.text)
        gap = s.start - cur.end
        if gap <= max_gap and len(cur.text) + 1 + len(t) <= max_chars:
            cur = Segment(start=cur.start, end=s.end, text=f"{cur.text} {t}".strip())
        else:
            merged.append(cur)
            cur = Segment(start=s.start, end=s.end, text=t)
    merged.append(cur)
    return merged


def merge_speak_segments(
    segments: list[Segment],
    *,
    max_gap: float = 0.65,
    max_chars: int = 140,
    max_dur: float = 9.0,
    prep=None,
) -> list[Segment]:
    """
    Merge nearby lines for TTS / singing.
    Fewer Edge TTS calls → fewer missing clips / silent gaps.
    """
    from .text_clean import prepare_speak_text

    clean = prep or prepare_speak_text
    usable = [s for s in segments if clean(s.text)]
    if not usable:
        return []

    merged: list[Segment] = []
    cur_text = clean(usable[0].text)
    cur = Segment(start=usable[0].start, end=usable[0].end, text=cur_text)
    for s in usable[1:]:
        t = clean(s.text)
        gap = s.start - cur.end
        new_dur = s.end - cur.start
        joined = f"{cur.text} {t}".strip()
        if gap <= max_gap and len(joined) <= max_chars and new_dur <= max_dur:
            cur = Segment(start=cur.start, end=s.end, text=joined)
        else:
            merged.append(cur)
            cur = Segment(start=s.start, end=s.end, text=t)
    merged.append(cur)
    return merged


def translate_transcript(
    transcript: Transcript,
    source: str | None = None,
    *,
    fast: bool = True,
) -> list[Segment]:
    """
    Translate to Khmer with timestamps.
    Quality: merge nearby lines, then translate in small batches (or per line).
    """
    src = source or transcript.language or "auto"
    segs = merge_nearby_segments(transcript.segments)
    if not segs:
        return []

    out: list[Segment] = []
    batch_size = 6 if fast else 3
    for i in range(0, len(segs), batch_size):
        batch = segs[i : i + batch_size]
        khmer_parts = _translate_batch([s.text for s in batch], src)
        for seg, kh in zip(batch, khmer_parts):
            text = clean_khmer_text(kh) or clean_khmer_text(translate_text(seg.text, source=src))
            if text:
                out.append(Segment(start=seg.start, end=seg.end, text=text))
        time.sleep(0.05)
    return out


def split_into_two_line_cues(
    segments: list[Segment],
    *,
    max_chars: int = 80,
) -> list[Segment]:
    """Split captions into timed cues sized for **2 on-screen lines**."""
    out: list[Segment] = []
    for seg in segments:
        text = clean_khmer_text(seg.text).rstrip("។").strip()
        if not text:
            continue
        pieces = _chunk_caption_text(text, max_chars=max_chars)
        if len(pieces) == 1:
            out.append(Segment(start=seg.start, end=seg.end, text=clean_khmer_text(pieces[0])))
            continue

        span = max(seg.end - seg.start, 0.45 * len(pieces))
        weights = [max(8, len(p)) for p in pieces]
        total_w = float(sum(weights))
        t = seg.start
        for i, piece in enumerate(pieces):
            dur = span * (weights[i] / total_w)
            start = t
            end = seg.end if i == len(pieces) - 1 else min(seg.end, start + dur)
            end = max(start + 0.28, end)
            out.append(Segment(start=start, end=end, text=clean_khmer_text(piece)))
            t = end
    return out


def merge_segments_for_speed(segments: list[Segment], max_groups: int = 40) -> list[Segment]:
    """Keep captions sized for 2 lines."""
    return split_into_two_line_cues(segments, max_chars=80)


def _chunk_caption_text(text: str, max_chars: int = 80) -> list[str]:
    """Break text into chunks that typically fill 2 subtitle lines."""
    raw_parts = re.split(r"(?<=[។!?\.])\s+|\s+", text)
    parts = [p for p in raw_parts if p]
    if not parts:
        return [text[:max_chars]]

    chunks: list[str] = []
    current = ""
    for part in parts:
        trial = part if not current else f"{current} {part}"
        if current and len(trial) > max_chars:
            chunks.append(current.strip())
            current = part
        else:
            current = trial
    if current.strip():
        chunks.append(current.strip())

    final: list[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            mid = max_chars
            for i in range(0, len(c), mid):
                final.append(c[i : i + mid].strip())
    return [c for c in final if c] or [text[:max_chars]]
