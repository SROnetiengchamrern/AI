"""Kids / nursery-rhyme video: CoComelon-style titles + sing-along scripts → AI video.

Title pattern (like CoComelon):
  "{Song Name} Song {emoji} | Nursery Rhymes & Kids Songs"

Script style: short, repetitive, singable lines + simple toddler moments
(inspired by Humpty Dumpty / Bananaphone / Yummy Peas style videos).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .text_video import (
    SceneSpec,
    TextVideoResult,
    create_video_from_text,
    parse_target_duration,
    prepare_khmer_script,
)
from .text_clean import clean_khmer_text, prepare_speak_text
from .translate import translate_text
from .scene_images import IMAGE_SOURCE_CHOICES, parse_image_urls

# Speak language for kids narration (default English like CoComelon)
KIDS_SPEAK_LANGUAGES = {
    "English (default)": "en",
    "Khmer": "km",
}

# Edge TTS voices — English first / default
KIDS_VOICES = {
    "English — Female (Jenny)": "en-US-JennyNeural",
    "English — Female (Aria)": "en-US-AriaNeural",
    "English — Female (Ana)": "en-US-AnaNeural",
    "English — Male (Guy)": "en-US-GuyNeural",
    "English — Male (Christopher)": "en-US-ChristopherNeural",
    "Khmer — Female (Sreymom)": "km-KH-SreymomNeural",
    "Khmer — Male (Piseth)": "km-KH-PisethNeural",
}

KIDS_DEFAULT_VOICE = "English — Female (Jenny)"
KIDS_DEFAULT_SPEAK = "English (default)"
KIDS_DEFAULT_IMAGE_SOURCE = "AI + stock mix (better variety)"
KIDS_IMAGE_SOURCES = IMAGE_SOURCE_CHOICES

# Preset themes → display name, emoji, short topic for lyrics/scenes
KIDS_THEMES: dict[str, tuple[str, str, str]] = {
    "Humpty Dumpty": ("Humpty Dumpty", "🥚", "egg on a wall, great fall, put together again"),
    "Bananaphone": ("Bananaphone", "🍌", "yellow banana phone, call friends, giggle and sing"),
    "Peekaboo": ("Peekaboo", "👀", "peekaboo with dad and toddler, hide and find, happy laughs"),
    "Yummy Veggies": ("Yummy Peas", "🥦", "green peas on a plate, try a bite, yummy veggies"),
    "Camping Adventure": ("I Can Do It — Camping", "⛺", "camping with dad, tent, flashlight, I can do it"),
    "ABC Song": ("ABC", "🔤", "alphabet letters A to Z, bright colorful letters for toddlers"),
    "Colors": ("Rainbow Colors", "🌈", "red orange yellow green blue purple, learn colors"),
    "Animals": ("Farm Animals", "🐄", "cow pig duck chicken, animal sounds moo oink quack"),
    "Bedtime": ("Goodnight", "🌙", "stars moon bedtime, soft lullaby, sweet dreams"),
    "Wheels on the Bus": ("Wheels on the Bus", "🚌", "yellow school bus, wheels go round, kids wave"),
    "Twinkle Twinkle": ("Twinkle Twinkle Little Star", "⭐", "little star in night sky, twinkle bright"),
    "Brush Teeth": ("Brush Your Teeth", "🦷", "brush teeth morning and night, sparkle smile"),
    "Custom / my idea": ("", "🎵", ""),
}

KIDS_THEME_CHOICES = list(KIDS_THEMES.keys())

_BRAND_SUFFIX = "Nursery Rhymes & Kids Songs"

# Soft interjections like toddler nursery videos (spoken lightly between verses)
_INTERJECTIONS = ["Wow!", "Yay!", "Ha ha!", "Ooh!", "Yeah!"]


@dataclass
class KidsDraft:
    """Generated title + English lyrics + Khmer voice script."""

    title: str
    lyrics_en: str
    script_kh: str
    theme_key: str


def _clean_song_name(name: str) -> str:
    name = re.sub(r"\s+", " ", (name or "").strip())
    name = re.sub(r"\s*[|·•].*$", "", name).strip()
    name = re.sub(
        r"\s*(song|nursery rhymes?|& kids songs?|cocomelon)\s*$",
        "",
        name,
        flags=re.I,
    ).strip(" -–—|")
    return name or "Happy Kids"


def format_kids_title(song_name: str, emoji: str = "🎵") -> str:
    """
    CoComelon-like YouTube title.
    Example: Humpty Dumpty Song 🥚 | Nursery Rhymes & Kids Songs
    """
    base = _clean_song_name(song_name)
    if not re.search(r"\bsong\b", base, re.I):
        base = f"{base} Song"
    em = (emoji or "🎵").strip() or "🎵"
    return f"{base} {em} | {_BRAND_SUFFIX}"


def resolve_kids_inputs(
    theme_label: str,
    custom_title: str,
    custom_script: str,
) -> tuple[str, str, str]:
    """
    Resolve theme / custom fields → (display_song_name, emoji, topic_hint).
    """
    theme_label = (theme_label or "").strip() or "Custom / my idea"
    custom_title = (custom_title or "").strip()
    custom_script = (custom_script or "").strip()

    name, emoji, topic = KIDS_THEMES.get(theme_label, ("", "🎵", ""))
    if theme_label == "Custom / my idea" or not name:
        name = _clean_song_name(custom_title) if custom_title else "Happy Kids"
        if custom_title and not emoji:
            emoji = "🎵"
        topic = custom_script[:180] if custom_script else f"fun toddler song about {name}"
    elif custom_title:
        # User overrides the song name but keeps theme emoji/topic flavor
        name = _clean_song_name(custom_title)
    return name, emoji, topic


def _classic_lyrics(song_name: str) -> list[str] | None:
    """Known nursery verses when the theme matches a classic rhyme."""
    key = song_name.lower()
    if "humpty" in key:
        verse = [
            "Humpty Dumpty sat on a wall.",
            "Humpty Dumpty had a great fall.",
            "All the king's horses and all the king's men",
            "Couldn't put Humpty together again.",
        ]
        return verse
    if "twinkle" in key or "little star" in key:
        return [
            "Twinkle, twinkle, little star,",
            "How I wonder what you are.",
            "Up above the world so high,",
            "Like a diamond in the sky.",
            "Twinkle, twinkle, little star,",
            "How I wonder what you are.",
        ]
    if "wheels on the bus" in key or "bus" in key and "wheel" in key:
        return [
            "The wheels on the bus go round and round,",
            "Round and round, round and round.",
            "The wheels on the bus go round and round,",
            "All through the town.",
            "The wipers on the bus go swish, swish, swish,",
            "All through the town.",
            "The horn on the bus goes beep, beep, beep,",
            "All through the town.",
        ]
    if "abc" in key or key.strip() in {"a b c", "alphabet"}:
        return [
            "A B C D E F G,",
            "H I J K L M N O P,",
            "Q R S, T U V,",
            "W X Y and Z.",
            "Now I know my ABCs,",
            "Next time won't you sing with me?",
        ]
    return None


def _themed_verse_lines(song_name: str, topic: str) -> list[str]:
    """Simple original toddler verse when not a classic rhyme."""
    subject = _clean_song_name(song_name)
    topic = (topic or subject).strip()
    return [
        f"Come on kids, let's sing today!",
        f"This is the {subject} song — hip hip hooray!",
        f"Look around — {topic}.",
        "Clap your hands and sing along.",
        f"{subject}, {subject}, one, two, three!",
        "Happy friends for you and me.",
        "We can learn and we can play.",
        f"Sing the {subject} song every day!",
        "Wow! Yay! Let's try again!",
        f"{subject} makes us smile and grin.",
    ]


def build_kids_lyrics(
    song_name: str,
    topic: str = "",
    *,
    target_seconds: float = 150.0,
    user_script: str = "",
) -> str:
    """
    Build English nursery lyrics: classic verse when known, else themed lines.
    Repeats verses to roughly fill target length (~2.5–3 min default like CoComelon shorts).
    """
    user_script = (user_script or "").strip()
    if user_script and len(user_script) > 40:
        # User pasted a full script — keep it (optionally pad with repeats)
        base_block = user_script
        lines = [ln.strip() for ln in re.split(r"\n+", base_block) if ln.strip()]
    else:
        classic = _classic_lyrics(song_name)
        lines = list(classic) if classic else _themed_verse_lines(song_name, topic)

    # ~12–14 spoken chars/sec for toddler pace; aim for verse repeats
    chars_needed = max(280, int(target_seconds * 11))
    out: list[str] = []
    # Opening toddler energy (like CoComelon cold open)
    out.extend(["Wow!", "Yay!", "Let's sing!"])
    i = 0
    while sum(len(x) for x in out) < chars_needed and i < 24:
        out.extend(lines)
        if i % 2 == 1:
            out.append(_INTERJECTIONS[i % len(_INTERJECTIONS)])
        i += 1
    out.append("Yay! Great job singing!")
    return "\n".join(out)


def lyrics_to_scene_beats(lyrics: str, *, max_beats: int = 24) -> list[str]:
    """Split lyrics into short scene beats for images + TTS chunks."""
    raw = [ln.strip() for ln in re.split(r"\n+", lyrics or "") if ln.strip()]
    # Drop ultra-short interjections as solo scenes — merge with next
    beats: list[str] = []
    buf = ""
    for ln in raw:
        core = ln.rstrip("!.?").strip()
        if len(ln) <= 8 and (not core or len(core) <= 5):
            buf = f"{buf} {ln}".strip()
            continue
        if buf:
            beats.append(f"{buf} {ln}".strip())
            buf = ""
        else:
            beats.append(ln)
    if buf:
        if beats:
            beats[-1] = f"{beats[-1]} {buf}".strip()
        else:
            beats.append(buf)

    if len(beats) <= max_beats:
        return beats or [lyrics[:120] or "Happy kids song"]
    # Merge evenly
    chunk = max(1, len(beats) // max_beats)
    merged: list[str] = []
    for i in range(0, len(beats), chunk):
        merged.append(" ".join(beats[i : i + chunk]))
    return merged[:max_beats]


def _kids_image_prompt(beat: str, title: str, topic: str) -> str:
    """Bright 3D preschool cartoon style (CoComelon-like look, original characters)."""
    beat_s = re.sub(r"\s+", " ", (beat or "")[:140])
    topic_s = re.sub(r"\s+", " ", (topic or title)[:100])
    return (
        f"bright colorful 3D preschool cartoon for toddlers, cute friendly child with big eyes, "
        f"soft rounded shapes, highly saturated happy colors, sunny lighting, "
        f"scene: {beat_s}, theme: {topic_s}, song title '{title}', "
        f"nursery rhyme animation still, wholesome family-friendly, sharp focus, "
        f"no text, no watermark, no letters, no logos"
    )


def expand_kids_to_scenes(
    title: str,
    lyrics_en: str,
    *,
    topic: str = "",
    target_seconds: float = 150.0,
) -> tuple[str, list[SceneSpec]]:
    """English kids lyrics → Khmer speak lines + cartoon scene prompts."""
    n_beats = max(6, min(24, int(round(max(60.0, target_seconds) / 12.0))))
    beats = lyrics_to_scene_beats(lyrics_en, max_beats=n_beats)
    scenes: list[SceneSpec] = []
    narrate_kh: list[str] = []

    for beat in beats:
        try:
            kh = clean_khmer_text(translate_text(beat, source="en"))
        except Exception:
            kh = ""
        speak = prepare_speak_text(kh) if kh else ""
        img = _kids_image_prompt(beat, title, topic)
        scenes.append(
            SceneSpec(
                image_prompt=img,
                speak_text=speak or "។",
                caption=speak,
                caption_en=beat,
            )
        )
        if speak:
            narrate_kh.append(speak)

    khmer = " ".join(narrate_kh) if narrate_kh else prepare_khmer_script(lyrics_en[:500], "en")
    return khmer, scenes


def draft_kids_content(
    theme_label: str,
    custom_title: str = "",
    custom_script: str = "",
    *,
    duration_hours: int = 0,
    duration_minutes: int = 3,
) -> KidsDraft:
    """Preview/generate title + lyrics + Khmer script without rendering video."""
    name, emoji, topic = resolve_kids_inputs(theme_label, custom_title, custom_script)
    title = format_kids_title(name, emoji)
    target = parse_target_duration(duration_hours, duration_minutes)
    # Kids clips default ~2–5 min like CoComelon song uploads
    if target < 90:
        target = 150.0
    lyrics = build_kids_lyrics(
        name,
        topic,
        target_seconds=target,
        user_script=custom_script,
    )
    try:
        script_kh = prepare_khmer_script(lyrics, source_language="en")
    except Exception:
        script_kh = ""
    return KidsDraft(
        title=title,
        lyrics_en=lyrics,
        script_kh=script_kh,
        theme_key=theme_label,
    )


def create_kids_video(
    theme_label: str,
    custom_title: str = "",
    custom_script: str = "",
    *,
    output_dir: str | Path | None = None,
    duration_hours: int = 0,
    duration_minutes: int = 3,
    speak_language: str = "en",
    voice_label: str = KIDS_DEFAULT_VOICE,
    image_source: str = KIDS_DEFAULT_IMAGE_SOURCE,
    image_urls: str = "",
    add_music: bool = True,
    show_captions_kh: bool = False,
    show_captions_en: bool = True,
    show_note: bool = True,
    video_note: str = "Kids Song",
    show_title_en: bool = True,
    show_title_kh: bool = False,
    fast: bool = True,
    progress_cb=None,
) -> TextVideoResult:
    """
    Theme/title → CoComelon-style title + nursery script → AI kids video.
    Default: English speak + English Jenny voice (like nursery channels).
    """

    def tick(msg: str, frac: float) -> None:
        if progress_cb:
            progress_cb(frac, msg)

    speak_lang = (speak_language or "en").split("-")[0].lower()
    if speak_lang not in ("en", "km"):
        speak_lang = "en"
    # Map UI labels → codes
    speak_lang = KIDS_SPEAK_LANGUAGES.get(speak_language, speak_lang)
    if speak_lang not in ("en", "km"):
        speak_lang = "en"

    # Keep voice language aligned with speak language when mismatched
    voice = voice_label or KIDS_DEFAULT_VOICE
    voice_id = KIDS_VOICES.get(voice, voice)
    if speak_lang == "en" and str(voice_id).startswith("km-"):
        voice = KIDS_DEFAULT_VOICE
    elif speak_lang == "km" and str(voice_id).startswith("en-"):
        voice = "Khmer — Female (Sreymom)"

    img_label = image_source or KIDS_DEFAULT_IMAGE_SOURCE
    img_code = IMAGE_SOURCE_CHOICES.get(img_label, img_label)
    if img_code not in ("ai", "mix", "stock", "urls"):
        img_code = "mix"
    refs = parse_image_urls(image_urls or "")

    tick("Writing kids title & nursery lyrics…", 0.03)
    draft = draft_kids_content(
        theme_label,
        custom_title,
        custom_script,
        duration_hours=duration_hours,
        duration_minutes=duration_minutes,
    )

    # Feed lyrics as narration script; force kids visual style via "kids" style key
    tick(
        f"Building sing-along scenes ({'English' if speak_lang == 'en' else 'Khmer'} speak)…",
        0.08,
    )
    result = create_video_from_text(
        draft.lyrics_en,
        output_dir=output_dir,
        video_title=draft.title,
        duration_hours=duration_hours,
        duration_minutes=duration_minutes if (duration_hours or duration_minutes) else 3,
        source_language="en",
        voice_label=voice,
        style="kids",
        mode="script",
        speak_language=speak_lang,
        image_source=img_code,
        image_urls=refs,
        add_music=add_music,
        show_captions_kh=show_captions_kh,
        show_captions_en=show_captions_en,
        show_note=show_note,
        video_note=(video_note or "").strip() or "Kids Song",
        show_title_en=show_title_en,
        show_title_kh=show_title_kh,
        fast=fast,
        progress_cb=progress_cb,
    )
    # Attach draft lyrics into result mode label for UI
    result.mode = "kids"
    if speak_lang == "en":
        result.khmer_text = draft.lyrics_en
    else:
        result.khmer_text = draft.script_kh or result.khmer_text
    return result
