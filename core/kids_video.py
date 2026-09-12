"""Kids story video: short spoken beats → 3D cartoon AI video.

Title pattern:
  "{Name} Story {emoji} | Kids Stories & Friends"

Script style: short spoken lines (not Scene 1 headers, not sing-along).
Images use locked 3D preschool CGI look — original characters.
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
from .text_clean import clean_khmer_text, prepare_speak_text, prepare_english_speak_text
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
# AI-only keeps 3D cartoon look; mix often falls back to photo stock
KIDS_DEFAULT_IMAGE_SOURCE = "AI cartoon (recommended)"
KIDS_IMAGE_SOURCES = IMAGE_SOURCE_CHOICES

# Preset themes → display name, emoji, short topic for lyrics/scenes
KIDS_THEMES: dict[str, tuple[str, str, str]] = {
    "Humpty Dumpty": ("Humpty Dumpty", "🥚", "egg on a wall, great fall, put together again"),
    "Bananaphone": ("Bananaphone", "🍌", "yellow banana phone, call friends, giggle and play"),
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
    "New Kid, New Friend": (
        "New Kid, New Friend",
        "🏫",
        "shy new child at school, kind friend says hello, playground ball, make friends",
    ),
    "Custom / my idea": ("", "🎵", ""),
}

KIDS_THEME_CHOICES = list(KIDS_THEMES.keys())

_BRAND_SUFFIX = "Kids Stories & Friends"

# Soft interjections (spoken energy — not singing)
_INTERJECTIONS = ["Wow!", "Yay!", "Ha ha!", "Ooh!", "Yeah!"]

# Shared 3D look for every kids still (original characters — not any brand IP)
_CGI_LOOK = (
    "polished 3D CGI preschool animation still, soft rounded plastic toy look, "
    "big expressive eyes, smooth subsurface skin, candy saturated colors, "
    "sunny soft key light, shallow depth of field, wholesome family-friendly, "
    "kids story cartoon quality, sharp focus"
)
_CGI_NEG = "no photoreal photo, no live action, no text, no watermark, no logo, no letters, no subtitles"


@dataclass
class KidsDraft:
    """Generated title + English speak script + Khmer voice script."""

    title: str
    lyrics_en: str
    script_kh: str
    theme_key: str


def _clean_song_name(name: str) -> str:
    name = re.sub(r"\s+", " ", (name or "").strip())
    name = re.sub(r"\s*[|·•].*$", "", name).strip()
    name = re.sub(
        r"\s*(song|story|nursery rhymes?|& kids songs?|& friends|cocomelon)\s*$",
        "",
        name,
        flags=re.I,
    ).strip(" -–—|")
    return name or "Happy Kids"


def format_kids_title(song_name: str, emoji: str = "🎵") -> str:
    """
    Kids story YouTube-style title (spoken story — not a sing-along).
    Example: New Kid, New Friend Story 🏫 | Kids Stories & Friends
    """
    base = _clean_song_name(song_name)
    if not re.search(r"\b(story|song)\b", base, re.I):
        base = f"{base} Story"
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
        topic = custom_script[:180] if custom_script else f"fun toddler story about {name}"
    elif custom_title:
        # User overrides the song name but keeps theme emoji/topic flavor
        name = _clean_song_name(custom_title)
    return name, emoji, topic


def kids_character_lock(song_name: str, topic: str = "") -> str:
    """Stable cast description so scenes look like one 3D show."""
    key = f"{song_name} {topic}".lower()
    if "humpty" in key:
        return (
            "same cute 3D egg character with blue suspenders tiny arms and legs, "
            "big friendly smile, consistent design every shot"
        )
    if "new kid" in key or "friend" in key or "school" in key:
        return (
            "same shy toddler boy with soft orange-brown hair red backpack, "
            "same kind toddler girl with teal hair and pink bow, "
            "consistent 3D character designs every shot"
        )
    if "bus" in key:
        return "same bright yellow 3D school bus and cute waving toddlers, consistent designs"
    if "banana" in key:
        return "same cute yellow 3D banana phone toy character, consistent design"
    if "animal" in key or "farm" in key:
        return "same cute 3D farm animals cow pig duck chicken, consistent toy-like designs"
    return (
        "same cute 3D preschool toddler characters with big eyes soft cheeks, "
        "consistent character designs every shot"
    )


def clean_kids_script_lines(raw: str) -> list[str]:
    """
    Turn pasted storyboards into speakable beats.
    Strips 'Scene 1 — …' headers, emoji-only lines, and empty noise.
    """
    lines: list[str] = []
    for ln in re.split(r"\n+", raw or ""):
        s = ln.strip()
        if not s:
            continue
        # Drop scene headers / stage directions
        if re.match(r"^(scene\s*\d+|ending\s*message|storyboard|title)\b", s, re.I):
            # Keep text after em-dash / colon only if it looks like a spoken sentence
            after = re.split(r"[—\-–:|]", s, maxsplit=1)
            if len(after) > 1:
                tail = after[1].strip()
                # Skip short Title-Case labels like "Sitting Alone"
                words = tail.split()
                looks_label = (
                    len(tail) < 40
                    and len(words) <= 5
                    and not re.search(r"[.!?]$", tail)
                    and sum(1 for w in words if w[:1].isupper()) >= max(1, len(words) - 1)
                )
                if looks_label or len(tail) <= 12:
                    continue
                s = tail
            else:
                continue
        # Strip leading emoji clusters for cleaner TTS
        s = re.sub(
            r"^[\U0001F300-\U0001FAFF\U00002700-\U000027BF\s]+",
            "",
            s,
        ).strip()
        # Remove leftover header crumbs
        s = re.sub(r"^\d+[\).]\s*", "", s).strip()
        if len(s) < 2:
            continue
        # Skip pure emoji / symbol lines
        if not re.search(r"[A-Za-z\u1780-\u17FF]", s):
            continue
        lines.append(s)
    return lines


def _is_story_script(lines: list[str]) -> bool:
    """Narrative friendship/school scripts vs short rhyme verses."""
    if len(lines) >= 10:
        return True
    blob = " ".join(lines).lower()
    story_words = (
        "school", "friend", "recess", "playground", "hello", "alone",
        "bench", "walk home", "new kid", "welcome", "be kind",
    )
    hits = sum(1 for w in story_words if w in blob)
    return hits >= 3


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
    if "wheels on the bus" in key or ("bus" in key and "wheel" in key):
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
            "Next time won't you learn with me?",
        ]
    if "new kid" in key or "new friend" in key:
        return [
            "A new school day!",
            "A shy child walks in.",
            "Other kids are playing.",
            "The new kid feels nervous.",
            "Recess time!",
            "The new kid sits alone.",
            "Look! A kind friend sees.",
            "Hello! Hello!",
            "Come play with us!",
            "Yes! Let's play together!",
            "Run and laugh!",
            "Happy friends!",
            "Welcome, new friend!",
            "Walk home with a friend.",
            "Be kind. Say hello. Make a new friend!",
        ]
    return None


def _themed_verse_lines(song_name: str, topic: str) -> list[str]:
    """Simple original toddler speak lines when not a classic rhyme."""
    subject = _clean_song_name(song_name)
    topic = (topic or subject).strip()
    return [
        f"Come on kids, let's begin today!",
        f"This is the {subject} story — hip hip hooray!",
        f"Look around — {topic}.",
        "Clap your hands and smile along.",
        f"{subject}, {subject}, one, two, three!",
        "Happy friends for you and me.",
        "We can learn and we can play.",
        f"Watch the {subject} story every day!",
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
    Build English speak script / story beats (spoken voice — not singing).
    Story scripts: clean headers, light pad only.
    Rhyme verses: classic/themed + toddler energy repeats.
    """
    user_script = (user_script or "").strip()
    cleaned = clean_kids_script_lines(user_script) if user_script else []

    if cleaned and (len(" ".join(cleaned)) > 40 or _is_story_script(cleaned)):
        lines = cleaned
        is_story = _is_story_script(cleaned)
    else:
        classic = _classic_lyrics(song_name)
        lines = list(classic) if classic else _themed_verse_lines(song_name, topic)
        is_story = _is_story_script(lines)

    chars_needed = max(280, int(target_seconds * 11))
    out: list[str] = []

    if is_story:
        # Narrative: open soft, do not drown story in Wow/Yay loops
        if not any("new school" in ln.lower() or "come on" in ln.lower() for ln in lines[:2]):
            out.append("Let's begin!")
        out.extend(lines)
        # Light repeat once if much shorter than target
        if sum(len(x) for x in out) < chars_needed * 0.55 and len(lines) >= 6:
            out.append("Let's see again!")
            out.extend(lines)
        if not any("be kind" in ln.lower() or "make a new friend" in ln.lower() for ln in out[-3:]):
            out.append("Yay! Great job!")
        return "\n".join(out)

    # Rhyme / short theme: toddler cold open + verse repeats (spoken, not sung)
    out.extend(["Wow!", "Yay!", "Let's begin!"])
    i = 0
    while sum(len(x) for x in out) < chars_needed and i < 24:
        out.extend(lines)
        if i % 2 == 1:
            out.append(_INTERJECTIONS[i % len(_INTERJECTIONS)])
        i += 1
    out.append("Yay! Great job!")
    return "\n".join(out)


def lyrics_to_scene_beats(lyrics: str, *, max_beats: int = 24) -> list[str]:
    """Split lyrics into short scene beats for images + TTS chunks."""
    raw = clean_kids_script_lines(lyrics) or [
        ln.strip() for ln in re.split(r"\n+", lyrics or "") if ln.strip()
    ]
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
        return beats or [lyrics[:120] or "Happy kids story"]
    # Merge evenly
    chunk = max(1, len(beats) // max_beats)
    merged: list[str] = []
    for i in range(0, len(beats), chunk):
        merged.append(" ".join(beats[i : i + chunk]))
    return merged[:max_beats]


def _kids_image_prompt(beat: str, title: str, topic: str, *, cast: str = "") -> str:
    """Bright 3D preschool CGI still (original characters, channel-quality look)."""
    beat_s = re.sub(r"\s+", " ", (beat or "")[:160])
    topic_s = re.sub(r"\s+", " ", (topic or title)[:90])
    cast_s = re.sub(r"\s+", " ", (cast or kids_character_lock(title, topic))[:180])
    return (
        f"{_CGI_LOOK}, {cast_s}, "
        f"action scene: {beat_s}, theme: {topic_s}, "
        f"story '{title}', {_CGI_NEG}"
    )


def expand_kids_to_scenes(
    title: str,
    lyrics_en: str,
    *,
    topic: str = "",
    target_seconds: float = 150.0,
    speak_language: str = "en",
) -> tuple[str, list[SceneSpec]]:
    """English kids lyrics → speak lines + locked 3D cartoon scene prompts."""
    # ~1 beat / 10–12s keeps images changing like nursery edits
    n_beats = max(8, min(28, int(round(max(60.0, target_seconds) / 10.0))))
    beats = lyrics_to_scene_beats(lyrics_en, max_beats=n_beats)
    cast = kids_character_lock(title, topic)
    scenes: list[SceneSpec] = []
    narrate_kh: list[str] = []
    speak_lang = (speak_language or "en").split("-")[0].lower()

    for beat in beats:
        speak_en = prepare_english_speak_text(beat) or beat
        try:
            kh = clean_khmer_text(translate_text(beat, source="en"))
        except Exception:
            kh = ""
        speak_kh = prepare_speak_text(kh) if kh else ""
        img = _kids_image_prompt(beat, title, topic, cast=cast)
        if speak_lang == "en":
            speak = speak_en
            caption = speak_kh
        else:
            speak = speak_kh or "។"
            caption = speak_kh
        scenes.append(
            SceneSpec(
                image_prompt=img,
                speak_text=speak or "។",
                caption=caption,
                caption_en=beat,
            )
        )
        if speak_kh:
            narrate_kh.append(speak_kh)

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
    # Kids clips default ~2–5 min like toddler story uploads
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
    video_note: str = "Kids Story",
    show_title_en: bool = True,
    show_title_kh: bool = False,
    fast: bool = True,
    progress_cb=None,
) -> TextVideoResult:
    """
    Theme/title → kids story title + speak beats → 3D cartoon AI video.
    Default: English speak + Jenny + AI cartoon images (spoken story — not singing).
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
        img_code = "ai"
    # Kids 3D look: prefer pure AI over mix (stock photos break cartoon style)
    if img_code == "mix":
        img_code = "ai"
    refs = parse_image_urls(image_urls or "")

    tick("Writing kids title & 3D story beats…", 0.03)
    name, emoji, topic = resolve_kids_inputs(theme_label, custom_title, custom_script)
    draft = draft_kids_content(
        theme_label,
        custom_title,
        custom_script,
        duration_hours=duration_hours,
        duration_minutes=duration_minutes,
    )
    target = parse_target_duration(
        duration_hours,
        duration_minutes if (duration_hours or duration_minutes) else 3,
    )
    if target < 90:
        target = 150.0

    tick(
        f"Building 3D cartoon scenes ({'English' if speak_lang == 'en' else 'Khmer'} speak)…",
        0.08,
    )
    _khmer, scene_specs = expand_kids_to_scenes(
        draft.title,
        draft.lyrics_en,
        topic=topic or name,
        target_seconds=target,
        speak_language=speak_lang,
    )
    if not scene_specs:
        raise ValueError("No kids scenes could be built from this script.")

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
        video_note=(video_note or "").strip() or "Kids Story",
        show_title_en=show_title_en,
        show_title_kh=show_title_kh,
        fast=fast,
        progress_cb=progress_cb,
        scene_specs=scene_specs,
    )
    result.mode = "kids"
    if speak_lang == "en":
        result.khmer_text = draft.lyrics_en
    else:
        result.khmer_text = draft.script_kh or result.khmer_text
    return result
