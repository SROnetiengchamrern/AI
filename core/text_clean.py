"""Clean source / Khmer text for better TTS and subtitles."""

from __future__ import annotations

import re

from .khmer_numbers import apply_khmer_digits_in_text, expand_numbers_for_speak


_EN_PAREN = re.compile(r"\([^)]*[A-Za-z][^)]*\)")
_BARE_ENGLISH = re.compile(r"\b[A-Za-z]{2,}(?:'[A-Za-z]+)?\b")
_MULTI_SPACE = re.compile(r"[\u200b\u200c\u200d\s]+")
_BAD_PUNCT = re.compile(r"[|]{2,}|_{2,}|\*{2,}")


def clean_source_text(text: str) -> str:
    """Normalize original transcript before translation."""
    t = (text or "").strip()
    t = _MULTI_SPACE.sub(" ", t)
    t = _BAD_PUNCT.sub(" ", t)
    # Drop stage directions like (CLIMBED), (laughs)
    t = _EN_PAREN.sub(" ", t)
    t = _MULTI_SPACE.sub(" ", t).strip(" -–,.")
    return t


def clean_khmer_text(text: str) -> str:
    """
    Clean Khmer for TTS + on-screen captions.
    Removes leftover English words/parentheticals and normalizes punctuation.
    Converts Western digits (0-9) → Khmer digits (០-៩) for on-screen text.
    """
    t = (text or "").strip()
    t = _EN_PAREN.sub(" ", t)
    t = _BARE_ENGLISH.sub(" ", t)
    t = _BAD_PUNCT.sub(" ", t)
    t = t.replace("...", "។").replace("…", "។")
    # Prefer Khmer full stop spacing
    t = re.sub(r"\s*។\s*", "។ ", t)
    t = _MULTI_SPACE.sub(" ", t).strip()
    t = t.strip(" |/-")
    # Video numbers → Khmer digits on captions (១២៣ …)
    t = apply_khmer_digits_in_text(t)
    # Ensure sentence ends cleanly for TTS
    if t and not re.search(r"[។!?]$", t):
        t += "។"
    return t


def prepare_speak_text(text: str) -> str:
    """
    Short pause-friendly text for Edge TTS (Khmer).
    Numbers are spoken as Khmer words (មួយរយ…) per place-value chart,
    while captions keep Khmer digits via clean_khmer_text.
    """
    t = clean_khmer_text(text)
    # Speak: ១២៣ / 123 → មួយរយម្ភៃបី (clearer than digit glyphs)
    t = expand_numbers_for_speak(t)
    # Edge TTS reads better with a space after Khmer stop
    t = re.sub(r"។+", "។ ", t)
    return _MULTI_SPACE.sub(" ", t).strip()


def prepare_english_speak_text(text: str) -> str:
    """Pause-friendly English text for Edge TTS (kids / EN speak)."""
    t = clean_source_text(text)
    t = t.replace("…", "...").replace("—", "-").replace("–", "-")
    t = re.sub(r"\s+", " ", t).strip(" |/-")
    if t and not re.search(r"[.!?]$", t):
        t += "."
    return t


def prepare_lyric_text(text: str) -> str:
    """Original-language song lyrics for AI re-sing (no translation)."""
    t = clean_source_text(text)
    # Khmer songs: clean script but skip sentence-end ។ (causes pauses when sung)
    if re.search(r"[\u1780-\u17FF]", t):
        t = clean_khmer_text(t)
        t = re.sub(r"។+\s*$", "", t.strip())
    t = re.sub(r"\s+", " ", t).strip()
    return t
