"""Translate text segments to Khmer (quality-focused, resilient)."""

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


def _normalize_source(source: str | None) -> str:
    src = "auto" if not source or source == "auto" else source
    if src == "zh":
        return "zh-CN"
    return src


def _looks_latin(text: str) -> bool:
    """True when text is mostly Latin letters (English-like)."""
    letters = re.findall(r"[A-Za-z\u00C0-\u024F]", text or "")
    other = re.findall(
        r"[\u0900-\u097F\u0600-\u06FF\u0E00-\u0E7F\u1780-\u17FF\u4E00-\u9FFF]",
        text or "",
    )
    if not letters and not other:
        return False
    return len(letters) >= max(8, len(other) * 2)


def _source_candidates(source: str | None, sample_text: str = "") -> list[str]:
    """
    Prefer requested source, then auto/en.
    If UI says Hindi/Urdu but lines look English, try English first.
    """
    primary = _normalize_source(source)
    if primary not in ("auto", "en") and _looks_latin(sample_text):
        ordered = ["en", "auto", primary]
    else:
        ordered = [primary, "auto", "en"]
    out: list[str] = []
    for s in ordered:
        if s and s not in out:
            out.append(s)
    return out


def _google_once(text: str, source: str) -> str:
    translator = GoogleTranslator(source=source, target=TARGET_LANG)
    return (translator.translate(text) or "").strip()


def translate_text(text: str, source: str = "auto") -> str:
    """
    Translate one string to Khmer with retries + source fallbacks.
    Returns "" if all attempts fail (does not raise for TranslationNotFound).
    """
    text = clean_source_text(text)
    if not text:
        return ""

    for src in _source_candidates(source, text):
        parts: list[str] = []
        failed = False
        for chunk in _chunk_text(text):
            got = ""
            for attempt in range(4):
                try:
                    got = clean_khmer_text(_google_once(chunk, src))
                    if got:
                        break
                except Exception:
                    time.sleep(0.4 * (attempt + 1))
            if got:
                parts.append(got)
            else:
                failed = True
                break
            time.sleep(0.08)
        if parts and not failed:
            return " ".join(parts)
        if parts:
            return " ".join(parts)
    return ""


def _translate_batch(texts: list[str], source: str) -> list[str]:
    """Translate a small batch; fall back per-line. Never aborts the job."""
    cleaned = [clean_source_text(t) for t in texts]
    if not any(cleaned):
        return [""] * len(texts)

    sep = "\n¶\n"
    sample = " ".join(t for t in cleaned if t)[:240]

    for src in _source_candidates(source, sample):
        for attempt in range(3):
            try:
                raw = _google_once(sep.join(cleaned), src)
                parts = [clean_khmer_text(p) for p in re.split(r"\s*¶\s*", raw)]
                if len(parts) == len(texts) and sum(1 for p in parts if p) >= max(
                    1, len(texts) // 2
                ):
                    out: list[str] = []
                    for i, p in enumerate(parts):
                        out.append(p if p else translate_text(cleaned[i], source=src))
                    return out
            except Exception:
                time.sleep(0.45 * (attempt + 1))

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
    cur = Segment(
        start=usable[0].start,
        end=usable[0].end,
        text=clean_source_text(usable[0].text),
    )
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
    Resilient: Google Translate misses skip a line instead of killing the job.
    """
    src = source or transcript.language or "auto"
    sample = " ".join(s.text for s in transcript.segments[:8])
    # Title may say Hindi/Urdu but narration is often English
    if src in ("hi", "ur") and _looks_latin(sample):
        src = "en"
    elif (
        src not in ("auto", None, "en")
        and _looks_latin(sample)
        and (transcript.language or "").startswith("en")
    ):
        src = "en"

    segs = merge_nearby_segments(transcript.segments)
    if not segs:
        return []

    out: list[Segment] = []
    batch_size = 4 if fast else 2
    misses = 0
    for i in range(0, len(segs), batch_size):
        batch = segs[i : i + batch_size]
        try:
            khmer_parts = _translate_batch([s.text for s in batch], src)
        except Exception:
            khmer_parts = [""] * len(batch)

        for seg, kh in zip(batch, khmer_parts):
            text = clean_khmer_text(kh) or clean_khmer_text(
                translate_text(seg.text, source=src)
            )
            if text:
                out.append(Segment(start=seg.start, end=seg.end, text=text))
            else:
                misses += 1
        time.sleep(0.12)

    if not out:
        raise RuntimeError(
            "Translation to Khmer failed (Google Translate returned no results).\n"
            "Tips:\n"
            "• Check internet connection\n"
            "• Set Source language to Auto detect or English "
            "(this video may be English narration)\n"
            "• Wait a minute and retry (rate limit)\n"
            f"Detail: {misses} lines could not be translated."
        )
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
            out.append(
                Segment(start=seg.start, end=seg.end, text=clean_khmer_text(pieces[0]))
            )
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
