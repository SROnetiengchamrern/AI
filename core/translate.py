"""Translate text segments to Khmer (quality-focused, resilient).

Handles short clips (few lines) and long videos (many lines) with:
  - Google Translate retries + rate-limit backoff
  - MyMemory fallback when Google is blocked / rate-limited
  - Partial success (skip failed lines) instead of killing the whole job
"""

from __future__ import annotations

import re
import time

from deep_translator import GoogleTranslator
from deep_translator.exceptions import TooManyRequests, TranslationNotFound

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
    Prefer requested source, then auto.
    Avoid useless EN fallback for Chinese/CJK (that made 45% feel stuck).
    """
    primary = _normalize_source(source)
    sample = sample_text or ""
    has_cjk = bool(re.search(r"[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]", sample))

    if primary in ("zh-CN", "zh") or (has_cjk and primary == "auto"):
        ordered = ["zh-CN", "auto"]
    elif primary not in ("auto", "en") and _looks_latin(sample):
        ordered = ["en", "auto", primary]
    elif primary in ("ja", "ko", "th", "vi", "hi", "ur", "ar", "ru"):
        ordered = [primary, "auto"]
    else:
        ordered = [primary, "auto", "en"]

    out: list[str] = []
    for s in ordered:
        s = _normalize_source(s)
        if s and s not in out:
            out.append(s)
    return out


def _is_rate_limit(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        isinstance(exc, TooManyRequests)
        or "toomanyrequests" in name
        or "too many requests" in msg
        or "rate limit" in msg
        or "429" in msg
    )


def _google_once(text: str, source: str) -> str:
    translator = GoogleTranslator(source=source, target=TARGET_LANG)
    return (translator.translate(text) or "").strip()


def _mymemory_lang(source: str) -> str:
    """MyMemory expects locale pairs like en-GB / km-KH."""
    src = _normalize_source(source)
    mapping = {
        "en": "en-GB",
        "auto": "en-GB",
        "zh-CN": "zh-CN",
        "zh": "zh-CN",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "th": "th-TH",
        "vi": "vi-VN",
        "fr": "fr-FR",
        "es": "es-ES",
        "de": "de-DE",
        "id": "id-ID",
        "ms": "ms-MY",
        "hi": "hi-IN",
        "ur": "ur-PK",
        "ru": "ru-RU",
        "ar": "ar-SA",
        "km": "km-KH",
    }
    return mapping.get(src, "en-GB")


def _mymemory_once(text: str, source: str) -> str:
    from deep_translator import MyMemoryTranslator

    # Keep chunks small — MyMemory free tier is strict
    if len(text) > 450:
        text = text[:450]
    tr = MyMemoryTranslator(source=_mymemory_lang(source), target="km-KH")
    return (tr.translate(text) or "").strip()


def _translate_once(
    text: str,
    source: str,
    *,
    allow_mymemory: bool = True,
    rate_limited: bool = False,
) -> tuple[str, bool]:
    """
    One chunk → Khmer via Google, then MyMemory if needed.
    Returns (khmer_text, hit_rate_limit).
    """
    hit_rl = rate_limited
    attempts = 2 if not rate_limited else 4
    for attempt in range(attempts):
        try:
            got = clean_khmer_text(_google_once(text, source))
            if got:
                return got, hit_rl
        except Exception as exc:
            if _is_rate_limit(exc):
                hit_rl = True
                time.sleep(min(8.0, 1.2 * (attempt + 1) ** 1.3))
            else:
                time.sleep(0.25 * (attempt + 1))

    if allow_mymemory:
        # MyMemory after Google miss (helps when Google 429 on zh→km)
        for attempt in range(2):
            try:
                got = clean_khmer_text(_mymemory_once(text, source))
                if got:
                    return got, hit_rl
            except Exception:
                time.sleep(0.4 * (attempt + 1))
    return "", hit_rl


def translate_text(text: str, source: str = "auto") -> str:
    """
    Translate one string to Khmer with retries + source fallbacks + MyMemory.
    Returns "" if all attempts fail (does not raise for TranslationNotFound).
    """
    text = clean_source_text(text)
    if not text:
        return ""

    rate_limited = False
    for src in _source_candidates(source, text):
        parts: list[str] = []
        failed = False
        for chunk in _chunk_text(text, max_chars=3500):
            got, rate_limited = _translate_once(
                chunk, src, rate_limited=rate_limited
            )
            if got:
                parts.append(got)
            else:
                failed = True
                break
            time.sleep(0.08 if not rate_limited else 0.25)
        if parts and not failed:
            return " ".join(parts)
        if parts:
            return " ".join(parts)
    return ""


def _translate_batch(texts: list[str], source: str) -> tuple[list[str], bool]:
    """Translate a small batch; fall back per-line. Returns (parts, rate_limited)."""
    cleaned = [clean_source_text(t) for t in texts]
    if not any(cleaned):
        return [""] * len(texts), False

    sep = "\n¶\n"
    sample = " ".join(t for t in cleaned if t)[:240]
    rate_limited = False

    for src in _source_candidates(source, sample):
        for attempt in range(2 if not rate_limited else 3):
            try:
                raw = _google_once(sep.join(cleaned), src)
                parts = [clean_khmer_text(p) for p in re.split(r"\s*¶\s*", raw)]
                if len(parts) == len(texts) and sum(1 for p in parts if p) >= max(
                    1, len(texts) // 2
                ):
                    out: list[str] = []
                    for i, p in enumerate(parts):
                        if p:
                            out.append(p)
                        else:
                            one, rate_limited = _translate_once(
                                cleaned[i], src, rate_limited=rate_limited
                            )
                            out.append(one)
                    return out, rate_limited
            except Exception as exc:
                if _is_rate_limit(exc):
                    rate_limited = True
                    time.sleep(min(6.0, 1.5 * (attempt + 1)))
                else:
                    time.sleep(0.35 * (attempt + 1))

    out = []
    for t in cleaned:
        one, rate_limited = _translate_once(t, source, rate_limited=rate_limited)
        out.append(one)
    return out, rate_limited


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
    progress_cb=None,
    progress_start: float = 0.45,
    progress_end: float = 0.62,
) -> list[Segment]:
    """
    Translate to Khmer with timestamps.
    Reports live progress (so UI does not freeze at 45%).
    Missed lines are skipped; job only fails if NOTHING translates.
    """

    def tick(msg: str, frac: float) -> None:
        if progress_cb:
            progress_cb(max(progress_start, min(progress_end, frac)), msg)

    src = source or transcript.language or "auto"
    # Whisper often returns "zh"; UI may also pass "zh"
    if (src or "").startswith("zh"):
        src = "zh"
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

    # Chinese/CJK: merge more aggressively → fewer Google calls (faster past 45%)
    is_cjk = (src or "").startswith("zh") or bool(
        re.search(r"[\u4E00-\u9FFF]", sample)
    )
    merge_chars = 200 if is_cjk else 110
    segs = merge_nearby_segments(transcript.segments, max_chars=merge_chars)
    if not segs:
        return []

    n = len(segs)
    # Bigger batches = fewer HTTP round-trips (main reason 45% felt stuck)
    if n <= 8:
        batch_size = 3 if fast else 2
        pause = 0.12
    elif n <= 60:
        batch_size = 5 if fast else 3
        pause = 0.15
    else:
        batch_size = 6 if fast else 3
        pause = 0.22

    tick(
        f"Translating to Khmer… 0/{n} lines"
        + (" (Chinese → Khmer)" if is_cjk else ""),
        progress_start,
    )

    out: list[Segment] = []
    missed: list[Segment] = []
    rate_limited = False
    done = 0
    for i in range(0, len(segs), batch_size):
        batch = segs[i : i + batch_size]
        try:
            khmer_parts, rl = _translate_batch([s.text for s in batch], src)
            rate_limited = rate_limited or rl
        except Exception:
            khmer_parts = [""] * len(batch)

        for seg, kh in zip(batch, khmer_parts):
            text = clean_khmer_text(kh)
            if not text:
                one, rl = _translate_once(seg.text, _normalize_source(src), rate_limited=rate_limited)
                rate_limited = rate_limited or rl
                text = clean_khmer_text(one)
            if text:
                out.append(Segment(start=seg.start, end=seg.end, text=text))
            else:
                missed.append(seg)
        done = min(n, i + len(batch))
        # Map line progress into 45% → ~60%
        frac = progress_start + (progress_end - progress_start - 0.03) * (done / max(1, n))
        tick(f"Translating to Khmer… {done}/{n} lines", frac)
        time.sleep(pause if not rate_limited else pause + 0.35)

    # Second pass only for misses (avoid always waiting 1.2s)
    if missed:
        tick(f"Retrying {len(missed)} missed lines…", progress_end - 0.025)
        time.sleep(0.4 if not rate_limited else 1.0)
        still: list[Segment] = []
        for idx, seg in enumerate(missed):
            one, rate_limited = _translate_once(
                seg.text, _normalize_source(src), rate_limited=rate_limited
            )
            text = clean_khmer_text(one)
            if text:
                out.append(Segment(start=seg.start, end=seg.end, text=text))
            else:
                still.append(seg)
            if idx % 3 == 0:
                tick(
                    f"Retrying missed lines… {idx + 1}/{len(missed)}",
                    progress_end - 0.02,
                )
            time.sleep(0.2 if not rate_limited else 0.45)
        missed = still
        out.sort(key=lambda s: s.start)

    tick(f"Translation done ({len(out)}/{n} lines)", progress_end - 0.005)

    if not out:
        kind = "short" if n <= 8 else "long"
        raise RuntimeError(
            "Translation to Khmer failed (Google Translate rate limit / no results).\n"
            f"This looked like a {kind} video ({n} speech lines).\n"
            "Tips:\n"
            "• Wait 1–2 minutes and retry (Google free limit is easy to hit)\n"
            "• Set Source language to Chinese or Auto detect\n"
            "• Check internet connection\n"
            "• For long videos: use Faster encode OFF, or convert in shorter parts\n"
            f"Detail: {n} lines could not be translated."
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
