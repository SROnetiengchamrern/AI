"""Build a Khmer narrative story from the video transcript."""

from __future__ import annotations

import re

from .transcribe import Segment
from .translate import translate_text


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    # Split on Khmer / Latin sentence ends
    parts = re.split(r"(?<=[។!?\.])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def build_english_story_draft(original_text: str) -> str:
    """Heuristic short story draft from dialogue/narration (no LLM required)."""
    cleaned = re.sub(r"\s+", " ", (original_text or "").strip())
    if not cleaned:
        return ""

    # Keep a readable length for TTS
    if len(cleaned) > 1800:
        cleaned = cleaned[:1800].rsplit(" ", 1)[0] + "…"

    return (
        "This is a short story based on the video. "
        f"{cleaned} "
        "That is how this story unfolds."
    )


def generate_khmer_story(
    original_text: str,
    khmer_fallback: str,
    source_lang: str = "auto",
) -> str:
    """
    Create a Khmer story script.
    Drafts a simple English narrative from the transcript, then translates to Khmer.
    Falls back to joining existing Khmer lines if translation fails.
    """
    draft = build_english_story_draft(original_text)
    if draft:
        try:
            story = translate_text(draft, source="en")
            if story and story.strip():
                # Soft story framing in Khmer
                intro = "នេះជារឿងខ្លីដែលបានមកពីវីដេអូ។"
                outro = "អស់ហើយសម្រាប់រឿងនេះ។"
                body = story.strip()
                # Avoid double framing if translator already added similar lines
                if "រឿង" not in body[:40]:
                    body = f"{intro} {body}"
                if not body.rstrip().endswith(("។", "!", "?")):
                    body = body.rstrip() + "។"
                if "អស់ហើយ" not in body[-40:]:
                    body = f"{body} {outro}"
                return body
        except Exception:
            pass

    fallback = re.sub(r"\s+", " ", (khmer_fallback or "").strip())
    if not fallback:
        return "នេះជារឿងខ្លីពីវីដេអូ។"
    return f"នេះជារឿងខ្លីដែលបានមកពីវីដេអូ។ {fallback} អស់ហើយសម្រាប់រឿងនេះ។"


def story_to_timed_segments(story_text: str, video_duration: float) -> list[Segment]:
    """Spread story sentences evenly across the video timeline for TTS + overlays."""
    sentences = _split_sentences(story_text)
    if not sentences:
        sentences = [story_text.strip() or "រឿង"]

    duration = max(video_duration, 3.0)
    # Leave a tiny gap between lines
    n = len(sentences)
    slot = duration / n
    segments: list[Segment] = []
    for i, sent in enumerate(sentences):
        start = i * slot
        end = min(duration, (i + 1) * slot - 0.08)
        if end <= start:
            end = start + 0.5
        segments.append(Segment(start=start, end=end, text=sent))
    return segments


def write_story_file(story_text: str, path) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(story_text.strip() + "\n", encoding="utf-8")
