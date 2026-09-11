"""Text → AI video + AI voice.

Supports three input styles:
  1) Title story (short title → AI expands scenes to target length)
  2) Visual prompt (cinematic / wildlife description) → images match the prompt
  3) Narration script (story text) → Khmer VO + scene cards
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from . import download_video_stem, preferred_temp_root, run_ffmpeg
from .fonts import ensure_battambang_fonts
from .khmer_render import render_khmer_line
from .music import generate_bgm, mix_voice_and_music
from .text_clean import clean_khmer_text, prepare_english_speak_text, prepare_speak_text
from .translate import translate_text
from .tts import get_duration_seconds, resolve_voice, text_to_speech


@dataclass
class SceneSpec:
    """One video beat."""

    image_prompt: str  # English — sent to AI image model
    speak_text: str  # Khmer (or cleaned) for TTS
    caption: str  # on-screen Khmer caption; empty = hidden
    caption_en: str = ""  # on-screen English caption; empty = hidden


@dataclass
class TextVideoResult:
    khmer_text: str
    scenes: list[str]
    audio_path: Path
    video_path: Path
    work_dir: Path
    mode: str = "script"


_SENTENCE_SPLIT = re.compile(r"(?<=[។!?\.])\s+|\n+")
_VISUAL_HINTS = re.compile(
    r"\b("
    r"cinematic|ultra-?realistic|photorealistic|4k|8k|documentary|"
    r"natural lighting|smooth camera|wildlife|detailed water|"
    r"lily pad|hiding on|watching a|suddenly jumps|camera movement|"
    r"film still|shallow depth|golden hour|portrait|close-?up|"
    r"walking in|standing in|looking at"
    r")\b",
    re.I,
)
_STYLE_TAIL = re.compile(
    r"(?:^|[.!?]\s+|,\s*)("
    r"(?:Cinematic|Ultra-?realistic|Photorealistic|4\s*K|8\s*K|"
    r"wildlife documentary|natural lighting|smooth camera).+"
    r")\s*$",
    re.I,
)
_STYLE_ONLY = re.compile(
    r"^\s*(cinematic|ultra-?realistic|photorealistic|4\s*k|wildlife documentary|"
    r"natural lighting|smooth camera|detailed water)",
    re.I,
)

# Animals / creatures to force into AI images when mentioned in the text
_ANIMAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("green frog", re.compile(r"\bgreen\s+frogs?\b", re.I)),
    ("small fish", re.compile(r"\bsmall\s+fish\b", re.I)),
    ("polar bear", re.compile(r"\bpolar\s+bears?\b", re.I)),
    ("brown bear", re.compile(r"\bbrown\s+bears?\b", re.I)),
    ("sea turtle", re.compile(r"\bsea\s+turtles?\b", re.I)),
    ("golden eagle", re.compile(r"\bgolden\s+eagles?\b", re.I)),
    ("butterfly", re.compile(r"\bbutterfl(?:y|ies)\b", re.I)),
    ("dragonfly", re.compile(r"\bdragonfl(?:y|ies)\b", re.I)),
    ("crocodile", re.compile(r"\bcrocodiles?\b", re.I)),
    ("alligator", re.compile(r"\balligators?\b", re.I)),
    ("chameleon", re.compile(r"\bchameleons?\b", re.I)),
    ("scorpion", re.compile(r"\bscorpions?\b", re.I)),
    ("jellyfish", re.compile(r"\bjelly(?:fish)?\b", re.I)),
    ("seahorse", re.compile(r"\bseahorses?\b", re.I)),
    ("starfish", re.compile(r"\bstarf(?:ish|ishes)\b", re.I)),
    ("flamingo", re.compile(r"\bflamingos?\b", re.I)),
    ("peacock", re.compile(r"\bpeacocks?\b", re.I)),
    ("sparrow", re.compile(r"\bsparrows?\b", re.I)),
    ("pigeon", re.compile(r"\bpigeons?\b", re.I)),
    ("crow", re.compile(r"\bcrows?\b", re.I)),
    ("raven", re.compile(r"\bravens?\b", re.I)),
    ("swan", re.compile(r"\bswans?\b", re.I)),
    ("goose", re.compile(r"\b(?:goose|geese)\b", re.I)),
    ("chicken", re.compile(r"\bchickens?\b|\broosters?\b|\bhens?\b", re.I)),
    ("duck", re.compile(r"\bducks?\b", re.I)),
    ("parrot", re.compile(r"\bparrots?\b", re.I)),
    ("eagle", re.compile(r"\beagles?\b", re.I)),
    ("owl", re.compile(r"\bowls?\b", re.I)),
    ("hawk", re.compile(r"\bhawks?\b", re.I)),
    ("bird", re.compile(r"\bbirds?\b", re.I)),
    ("penguin", re.compile(r"\bpenguins?\b", re.I)),
    ("dolphin", re.compile(r"\bdolphins?\b", re.I)),
    ("whale", re.compile(r"\bwhales?\b", re.I)),
    ("shark", re.compile(r"\bsharks?\b", re.I)),
    ("octopus", re.compile(r"\boctopus(?:es)?\b", re.I)),
    ("crab", re.compile(r"\bcrabs?\b", re.I)),
    ("lobster", re.compile(r"\blobsters?\b", re.I)),
    ("fish", re.compile(r"\bfish\b", re.I)),
    ("frog", re.compile(r"\bfrogs?\b", re.I)),
    ("toad", re.compile(r"\btoads?\b", re.I)),
    ("snake", re.compile(r"\bsnakes?\b|\bserpents?\b", re.I)),
    ("lizard", re.compile(r"\blizards?\b", re.I)),
    ("turtle", re.compile(r"\bturtles?\b", re.I)),
    ("spider", re.compile(r"\bspiders?\b", re.I)),
    ("ant", re.compile(r"\bants?\b", re.I)),
    ("bee", re.compile(r"\bbees?\b", re.I)),
    ("wasp", re.compile(r"\bwasps?\b", re.I)),
    ("mosquito", re.compile(r"\bmosquitos?(?:es)?\b", re.I)),
    ("puppy", re.compile(r"\bpupp(?:y|ies)\b", re.I)),
    ("kitten", re.compile(r"\bkittens?\b", re.I)),
    ("dog", re.compile(r"\bdogs?\b", re.I)),
    ("cat", re.compile(r"\bcats?\b", re.I)),
    ("horse", re.compile(r"\bhorses?\b|\bpon(?:y|ies)\b|\bstallions?\b", re.I)),
    ("cow", re.compile(r"\bcows?\b|\bcattle\b|\bbulls?\b", re.I)),
    ("pig", re.compile(r"\bpigs?\b|\bboars?\b", re.I)),
    ("sheep", re.compile(r"\bsheep\b|\blambs?\b", re.I)),
    ("goat", re.compile(r"\bgoats?\b", re.I)),
    ("rabbit", re.compile(r"\brabbits?\b|\bbunn(?:y|ies)\b", re.I)),
    ("mouse", re.compile(r"\b(?:mouse|mice)\b", re.I)),
    ("rat", re.compile(r"\brats?\b", re.I)),
    ("hamster", re.compile(r"\bhamsters?\b", re.I)),
    ("lion", re.compile(r"\blions?\b", re.I)),
    ("tiger", re.compile(r"\btigers?\b", re.I)),
    ("leopard", re.compile(r"\bleopards?\b|\bcheetahs?\b|\bpanthers?\b", re.I)),
    ("bear", re.compile(r"\bbears?\b", re.I)),
    ("wolf", re.compile(r"\b(?:wolf|wolves)\b", re.I)),
    ("fox", re.compile(r"\bfox(?:es)?\b", re.I)),
    ("deer", re.compile(r"\bdeer\b|\bstags?\b", re.I)),
    ("elephant", re.compile(r"\belephants?\b", re.I)),
    ("giraffe", re.compile(r"\bgiraffes?\b", re.I)),
    ("zebra", re.compile(r"\bzebras?\b", re.I)),
    ("rhino", re.compile(r"\brhinos?(?:ceros)?\b", re.I)),
    ("hippo", re.compile(r"\bhippos?(?:potamus(?:es)?)?\b", re.I)),
    ("monkey", re.compile(r"\bmonkeys?\b", re.I)),
    ("gorilla", re.compile(r"\bgorillas?\b|\bapes?\b|\bchimpanzees?\b|\bchimps?\b", re.I)),
    ("panda", re.compile(r"\bpandas?\b", re.I)),
    ("koala", re.compile(r"\bkoalas?\b", re.I)),
    ("kangaroo", re.compile(r"\bkangaroos?\b", re.I)),
    ("camel", re.compile(r"\bcamels?\b", re.I)),
    ("buffalo", re.compile(r"\bbuffalos?\b|\bbison\b", re.I)),
    ("bat", re.compile(r"\bbats?\b", re.I)),
    ("squirrel", re.compile(r"\bsquirrels?\b", re.I)),
    ("hedgehog", re.compile(r"\bhedgehogs?\b", re.I)),
    ("dragon", re.compile(r"\bdragons?\b", re.I)),
    ("unicorn", re.compile(r"\bunicorns?\b", re.I)),
    ("dinosaur", re.compile(r"\bdinosaurs?\b|\btrex\b|\bt[\-\s]?rex\b", re.I)),
    ("animal", re.compile(r"\banimals?\b|\bcreature[s]?\b|\bwildlife\b", re.I)),
]

# Khmer animal words → English for image prompts
_KHMER_ANIMALS: list[tuple[str, str]] = [
    ("កង្កែប", "frog"),
    ("ត្រី", "fish"),
    ("ឆ្មា", "cat"),
    ("ឆ្កែ", "dog"),
    ("សត្វស្លាប", "bird"),
    ("បក្សី", "bird"),
    ("មាន់", "chicken"),
    ("ទា", "duck"),
    ("សេះ", "horse"),
    ("គោ", "cow"),
    ("ក្របី", "buffalo"),
    ("ជ្រូក", "pig"),
    ("ពស់", "snake"),
    ("អណ្តើក", "turtle"),
    ("ក្រពើ", "crocodile"),
    ("សីហ៍", "lion"),
    ("ខ្លា", "tiger"),
    ("ខ្លាឃ្មុំ", "bear"),
    ("ដំរី", "elephant"),
    ("ស្វា", "monkey"),
    ("ទន្សាយ", "rabbit"),
    ("សត្វ", "animal"),
]

# People / characters to force into AI images when mentioned
_PEOPLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("young woman", re.compile(r"\byoung\s+wom[ae]n\b", re.I)),
    ("young man", re.compile(r"\byoung\s+m[ae]n\b", re.I)),
    ("little girl", re.compile(r"\blittle\s+girls?\b", re.I)),
    ("little boy", re.compile(r"\blittle\s+boys?\b", re.I)),
    ("old woman", re.compile(r"\bold\s+wom[ae]n\b|\bgrandma\b|\bgrandmother\b", re.I)),
    ("old man", re.compile(r"\bold\s+m[ae]n\b|\bgrandpa\b|\bgrandfather\b", re.I)),
    ("beautiful woman", re.compile(r"\bbeautiful\s+wom[ae]n\b", re.I)),
    ("handsome man", re.compile(r"\bhandsome\s+m[ae]n\b", re.I)),
    ("pregnant woman", re.compile(r"\bpregnant\s+wom[ae]n\b", re.I)),
    ("asian woman", re.compile(r"\basian\s+wom[ae]n\b", re.I)),
    ("asian man", re.compile(r"\basian\s+m[ae]n\b", re.I)),
    ("asian girl", re.compile(r"\basian\s+girls?\b", re.I)),
    ("businessman", re.compile(r"\bbusinessm[ae]n\b", re.I)),
    ("businesswoman", re.compile(r"\bbusinesswom[ae]n\b", re.I)),
    ("police officer", re.compile(r"\bpolice\s+officers?\b|\bpolicem[ae]n\b|\bcops?\b", re.I)),
    ("firefighter", re.compile(r"\bfirefighters?\b|\bfiremen\b", re.I)),
    ("soldier", re.compile(r"\bsoldiers?\b", re.I)),
    ("doctor", re.compile(r"\bdoctors?\b", re.I)),
    ("nurse", re.compile(r"\bnurses?\b", re.I)),
    ("teacher", re.compile(r"\bteachers?\b", re.I)),
    ("student", re.compile(r"\bstudents?\b", re.I)),
    ("farmer", re.compile(r"\bfarmers?\b", re.I)),
    ("fisherman", re.compile(r"\bfisherm[ae]n\b", re.I)),
    ("chef", re.compile(r"\bchefs?\b|\bcooks?\b", re.I)),
    ("singer", re.compile(r"\bsingers?\b", re.I)),
    ("dancer", re.compile(r"\bdancers?\b", re.I)),
    ("actor", re.compile(r"\bactors?\b|\bactress(?:es)?\b", re.I)),
    ("pilot", re.compile(r"\bpilots?\b", re.I)),
    ("driver", re.compile(r"\bdrivers?\b", re.I)),
    ("worker", re.compile(r"\bworkers?\b", re.I)),
    ("scientist", re.compile(r"\bscientists?\b", re.I)),
    ("mother", re.compile(r"\bmothers?\b|\bmom\b|\bmommy\b|\bmum\b", re.I)),
    ("father", re.compile(r"\bfathers?\b|\bdad\b|\bdaddy\b", re.I)),
    ("sister", re.compile(r"\bsisters?\b", re.I)),
    ("brother", re.compile(r"\bbrothers?\b", re.I)),
    ("daughter", re.compile(r"\bdaughters?\b", re.I)),
    ("son", re.compile(r"\bsons?\b", re.I)),
    ("wife", re.compile(r"\bwife\b|\bwives\b", re.I)),
    ("husband", re.compile(r"\bhusbands?\b", re.I)),
    ("bride", re.compile(r"\bbrides?\b", re.I)),
    ("groom", re.compile(r"\bgrooms?\b", re.I)),
    ("lover", re.compile(r"\blovers?\b", re.I)),
    ("heroine", re.compile(r"\bheroines?\b", re.I)),
    ("hero", re.compile(r"\bheroes?\b|\bhero\b", re.I)),
    ("villain", re.compile(r"\bvillains?\b", re.I)),
    ("king", re.compile(r"\bkings?\b", re.I)),
    ("queen", re.compile(r"\bqueens?\b", re.I)),
    ("prince", re.compile(r"\bprinces?\b", re.I)),
    ("princess", re.compile(r"\bprincess(?:es)?\b", re.I)),
    ("warrior", re.compile(r"\bwarriors?\b|\bknights?\b|\bsamurai\b", re.I)),
    ("monk", re.compile(r"\bmonks?\b|\bnuns?\b", re.I)),
    ("couple", re.compile(r"\bcouples?\b", re.I)),
    ("family", re.compile(r"\bfamil(?:y|ies)\b", re.I)),
    ("friend", re.compile(r"\bfriends?\b", re.I)),
    ("stranger", re.compile(r"\bstrangers?\b", re.I)),
    ("crowd", re.compile(r"\bcrowds?\b", re.I)),
    ("people", re.compile(r"\bpeople\b|\bhumans?\b|\bfolks?\b", re.I)),
    ("woman", re.compile(r"\bwom[ae]n\b|\bladies\b|\blady\b|\bgirlfriend\b|\bfemale\b", re.I)),
    ("man", re.compile(r"\bm[ae]n\b(?!\s*kind)|\bboyfriend\b|\bguy\b|\bguys\b|\bmale\b", re.I)),
    ("girl", re.compile(r"\bgirls?\b", re.I)),
    ("boy", re.compile(r"\bboys?\b", re.I)),
    ("child", re.compile(r"\bchild(?:ren)?\b|\bkids?\b|\btoddlers?\b", re.I)),
    ("baby", re.compile(r"\bbab(?:y|ies)\b|\binfant\b|\bnewborn\b", re.I)),
    ("teenager", re.compile(r"\bteenagers?\b|\bteens?\b", re.I)),
    ("person", re.compile(r"\bpersons?\b|\bhuman\b|\bsomeone\b|\bsomebody\b|\bcharacter\b|\bfigure\b", re.I)),
]

# Pronouns → default people subject when no noun was found
_PRONOUN_WOMAN = re.compile(r"\b(she|her|hers|herself)\b", re.I)
_PRONOUN_MAN = re.compile(r"\b(he|him|his|himself)\b", re.I)
_PRONOUN_PEOPLE = re.compile(r"\b(they|them|their|themselves|we|us|our)\b", re.I)

_KHMER_PEOPLE: list[tuple[str, str]] = [
    ("មនុស្ស", "person"),
    ("នារី", "woman"),
    ("ស្ត្រី", "woman"),
    ("ស្រី", "woman"),
    ("បុរស", "man"),
    ("ប្រុស", "man"),
    ("ក្មេងស្រី", "girl"),
    ("ក្មេងប្រុស", "boy"),
    ("ក្មេង", "child"),
    ("ទារក", "baby"),
    ("ម្តាយ", "mother"),
    ("ឪពុក", "father"),
    ("បងស្រី", "sister"),
    ("បងប្រុស", "brother"),
    ("គ្រួសារ", "family"),
    ("ព្រះសង្ឃ", "monk"),
    ("ទាហាន", "soldier"),
    ("គ្រូពេទ្យ", "doctor"),
    ("គ្រូ", "teacher"),
    ("សិស្ស", "student"),
    ("ព្រះមហាក្សត្រ", "king"),
    ("ព្រះនាង", "princess"),
    ("នាង", "woman"),
    ("គាត់", "person"),
]


# Ghosts / spirits to force into AI images when mentioned
_GHOST_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ghost girl", re.compile(r"\bghost\s+girls?\b", re.I)),
    ("ghost boy", re.compile(r"\bghost\s+boys?\b", re.I)),
    ("ghost woman", re.compile(r"\bghost\s+wom[ae]n\b", re.I)),
    ("ghost man", re.compile(r"\bghost\s+m[ae]n\b", re.I)),
    ("ghost child", re.compile(r"\bghost\s+child(?:ren)?\b", re.I)),
    ("female ghost", re.compile(r"\bfemale\s+ghosts?\b", re.I)),
    ("male ghost", re.compile(r"\bmale\s+ghosts?\b", re.I)),
    ("white ghost", re.compile(r"\bwhite\s+ghosts?\b", re.I)),
    ("scary ghost", re.compile(r"\bscary\s+ghosts?\b", re.I)),
    ("poltergeist", re.compile(r"\bpoltergeists?\b", re.I)),
    ("apparition", re.compile(r"\bapparitions?\b", re.I)),
    ("phantom", re.compile(r"\bphantoms?\b", re.I)),
    ("spectre", re.compile(r"\bspectres?\b|\bspecters?\b", re.I)),
    ("wraith", re.compile(r"\bwraiths?\b", re.I)),
    ("vampire", re.compile(r"\bvampires?\b", re.I)),
    ("skeleton", re.compile(r"\bskeletons?\b", re.I)),
    ("undead", re.compile(r"\bundead\b", re.I)),
    ("soul", re.compile(r"\bsouls?\b", re.I)),
    ("spirit", re.compile(r"\bspirits?\b(?!\s+animal)", re.I)),
    ("ghost", re.compile(r"\bghosts?\b|\bhaunted\b|\bghastly\b|\bparanormal\b", re.I)),
    ("demon", re.compile(r"\bdemons?\b|\bdevil\b|\bfiend\b", re.I)),
    ("zombie", re.compile(r"\bzombies?\b", re.I)),
]

_KHMER_GHOSTS: list[tuple[str, str]] = [
    ("ខ្មោច", "ghost"),
    ("ព្រាយ", "ghost"),
    ("អារក្ស", "spirit"),
    ("បិសាច", "demon"),
    ("វិញ្ញាណ", "spirit"),
    ("ព្រលឹង", "soul"),
]


def extract_animals(text: str) -> list[str]:
    """
    Find animals mentioned in the text.
    Prefer specific phrases (green frog) over generic (frog) when both match.
    """
    text = text or ""
    found: list[str] = []
    seen_roots: set[str] = set()

    # Longer / more specific patterns first
    for label, pat in sorted(_ANIMAL_PATTERNS, key=lambda x: -len(x[0])):
        if not pat.search(text):
            continue
        root = label.split()[-1].lower()
        # Skip generic if a more specific form already added (e.g. green frog → skip frog)
        if root in seen_roots and " " not in label:
            continue
        if any(label in f or f in label for f in found):
            # Avoid duplicates like both "small fish" and "fish"
            if " " not in label and any(f.endswith(label) or label in f for f in found):
                continue
        found.append(label)
        seen_roots.add(root)

    for kh, en in _KHMER_ANIMALS:
        if kh in text and en not in seen_roots and not any(en in f for f in found):
            found.append(en)
            seen_roots.add(en)

    return found


def extract_people(text: str) -> list[str]:
    """Find people / characters mentioned in the text (nouns + pronouns)."""
    text = text or ""
    found: list[str] = []
    seen_roots: set[str] = set()

    for label, pat in sorted(_PEOPLE_PATTERNS, key=lambda x: -len(x[0])):
        if not pat.search(text):
            continue
        root = label.split()[-1].lower().rstrip("s")
        if root in seen_roots and " " not in label:
            continue
        if any(label in f or f.endswith(label) for f in found):
            if " " not in label and any(label in f for f in found):
                continue
        found.append(label)
        seen_roots.add(root)

    for kh, en in _KHMER_PEOPLE:
        if kh in text and en not in seen_roots and not any(en in f for f in found):
            found.append(en)
            seen_roots.add(en)

    # Pronouns: "She walks…" / "He opens…" → still force a human in the image
    if not found:
        if _PRONOUN_WOMAN.search(text):
            found.append("woman")
        elif _PRONOUN_MAN.search(text):
            found.append("man")
        elif _PRONOUN_PEOPLE.search(text):
            found.append("people")

    # If only baby/child and mother/father pronouns missing but "mother" pattern got baby —
    # also: "Two lovers" already covered

    cleaned: list[str] = []
    for item in found:
        if any(item != other and item in other for other in found):
            continue
        cleaned.append(item)
    return cleaned


def extract_ghosts(text: str) -> list[str]:
    """Find ghosts / spirits mentioned in the text."""
    text = text or ""
    found: list[str] = []
    seen_roots: set[str] = set()

    for label, pat in sorted(_GHOST_PATTERNS, key=lambda x: -len(x[0])):
        if not pat.search(text):
            continue
        root = label.split()[-1].lower().rstrip("s")
        if root in seen_roots and " " not in label:
            continue
        if any(label in f or f.endswith(label) for f in found):
            if " " not in label and any(label in f for f in found):
                continue
        found.append(label)
        seen_roots.add(root)

    for kh, en in _KHMER_GHOSTS:
        if kh in text and en not in seen_roots and not any(en in f for f in found):
            found.append(en)
            seen_roots.add(en)

    cleaned: list[str] = []
    for item in found:
        if any(item != other and item in other for other in found):
            continue
        cleaned.append(item)
    return cleaned


def extract_subjects(text: str) -> list[str]:
    """Ghosts + people + animals that must appear in AI scenes."""
    ghosts = extract_ghosts(text)
    people = extract_people(text)
    animals = extract_animals(text)
    out: list[str] = []
    for s in ghosts + people + animals:
        if s not in out:
            out.append(s)
    return out[:6]


def mentions_people(text: str) -> bool:
    """True if the description is about human(s)."""
    return bool(extract_people(text or ""))


def mentions_ghosts(text: str) -> bool:
    """True if the description is about ghost(s) / spirits."""
    return bool(extract_ghosts(text or ""))


def detect_motion_kind(text: str) -> str:
    """
    Pick animation style for the scene:
      ghost | people | animal | scene
    """
    blob = text or ""
    if extract_ghosts(blob):
        return "ghost"
    if extract_people(blob):
        return "people"
    if extract_animals(blob):
        return "animal"
    return "scene"


def _motion_prompt_tail(kind: str) -> str:
    """Extra prompt words so stills look ready for motion."""
    return {
        "ghost": "ghost drifting in air, floating motion, eerie atmosphere",
        "people": "natural body pose mid-action, living human presence, cinematic motion",
        "animal": "animal mid-movement, alive and active, natural motion",
        "scene": "cinematic camera movement, living atmosphere",
    }.get(kind, "cinematic camera movement")


def _people_display_name(primary: str) -> str:
    """Map roles/pronouns to words Flux understands as humans."""
    return {
        "lover": "romantic couple of two people",
        "hero": "heroic young man",
        "heroine": "heroic young woman",
        "people": "group of people",
        "person": "person",
        "couple": "romantic couple",
        "family": "family of people",
        "crowd": "crowd of people",
        "friend": "friends",
        "stranger": "person",
    }.get(primary, primary)


def _ghost_display_name(primary: str) -> str:
    """Map ghost labels to clear Flux-friendly wording."""
    return {
        "ghost": "translucent glowing ghost",
        "ghost girl": "translucent ghost girl",
        "ghost boy": "translucent ghost boy",
        "ghost woman": "translucent ghost woman",
        "ghost man": "translucent ghost man",
        "ghost child": "translucent ghost child",
        "female ghost": "translucent female ghost",
        "male ghost": "translucent male ghost",
        "white ghost": "pale white glowing ghost",
        "scary ghost": "terrifying glowing ghost",
        "spirit": "glowing spirit apparition",
        "phantom": "translucent phantom ghost",
        "spectre": "spectral ghost figure",
        "apparition": "ghostly apparition",
        "wraith": "wraith ghost figure",
        "poltergeist": "poltergeist ghost",
        "demon": "demonic spirit",
        "zombie": "undead zombie",
        "vampire": "pale vampire figure",
        "skeleton": "glowing skeleton spirit",
        "undead": "undead spirit",
        "soul": "glowing soul spirit",
    }.get(primary, f"translucent glowing {primary}")


def _portrait_style(style_tail: str) -> str:
    """People scenes must never inherit wildlife / empty-nature style."""
    style = re.sub(r"\s+", " ", (style_tail or "").strip(" ,."))
    style = re.sub(
        r"\bwildlife(?:\s+documentary)?(?:\s+style)?\b",
        "portrait photography",
        style,
        flags=re.I,
    )
    if not style or "wildlife" in style.lower() or len(style) > 85:
        return "cinematic portrait photography, ultra-realistic human, natural lighting, 4K"
    if "portrait" not in style.lower():
        style = f"cinematic portrait, {style}"
    return style[:90]


def _ghost_style(style_tail: str) -> str:
    """Horror / supernatural look for ghost scenes."""
    style = re.sub(r"\s+", " ", (style_tail or "").strip(" ,."))
    style = re.sub(
        r"\bwildlife(?:\s+documentary)?(?:\s+style)?\b",
        "horror atmosphere",
        style,
        flags=re.I,
    )
    if not style or "wildlife" in style.lower() or len(style) > 85:
        return "cinematic horror, dark moody lighting, volumetric fog, 4K"
    if "horror" not in style.lower() and "dark" not in style.lower():
        style = f"cinematic horror, {style}"
    return style[:90]


def build_focused_image_prompt(
    beat: str,
    *,
    full_text: str,
    style_tail: str,
) -> str:
    """
    Short English prompt for AI images.
    Ghosts / people / animals mentioned in text are forced into every scene.
    """
    blob = f"{full_text} {beat}"
    subjects = extract_subjects(blob)
    ghosts = extract_ghosts(blob)
    people = extract_people(blob)
    action = re.sub(r"\s+", " ", (beat or "").strip(" ,."))
    if len(action) > 100:
        action = action[:100].rsplit(" ", 1)[0]

    if ghosts:
        display = _ghost_display_name(ghosts[0])
        if not extract_ghosts(action):
            action = f"{display} {action}"
        style = _ghost_style(style_tail)
        with_people = ""
        if people:
            with_people = f", {_people_display_name(people[0])} also in the scene"
        prompt = (
            f"cinematic horror shot of {display} clearly visible in frame, "
            f"{display} as the main supernatural subject{with_people}, {action}, "
            f"ethereal translucent ghost body, pale glowing face, spectral mist, "
            f"{_motion_prompt_tail('ghost')}, sharp focus on the ghost, "
            f"not empty room, not empty landscape, {style}, no text, no watermark"
        )
        return prompt[:420]

    if people:
        display = _people_display_name(people[0])
        # Re-state the person inside the action if this beat lost them (comma splits)
        if not extract_people(action):
            action = f"{display} {action}"
        style = _portrait_style(style_tail)
        prompt = (
            f"photorealistic medium shot of {display}, "
            f"{display} in the foreground filling the frame, "
            f"{action}, face and body clearly visible, "
            f"detailed human skin, {_motion_prompt_tail('people')}, "
            f"sharp focus on {display}, shallow depth of field, "
            f"{style}, no text, no watermark"
        )
        return prompt[:420]

    style = re.sub(r"\s+", " ", (style_tail or "").strip(" ,."))
    if len(style) > 90:
        style = "cinematic wildlife documentary, ultra-realistic, natural lighting, 4K"

    if subjects:
        listed = ", ".join(subjects)
        prompt = (
            f"{listed} as main subjects, {action}, "
            f"must clearly show {listed}, {_motion_prompt_tail('animal')}, "
            f"sharp focus on the animals, detailed realistic {subjects[0]}, "
            f"{style}, no text, no watermark"
        )
    else:
        prompt = (
            f"{action}, cinematic still, {_motion_prompt_tail('scene')}, "
            f"sharp focus on the main subject, {style}, no text, no watermark"
        )
    return prompt[:420]


def _subject_lock_clause(subjects: list[str]) -> str:
    """Short lock phrase so AI images keep animals/people/ghosts in frame."""
    if not subjects:
        return ""
    listed = ", ".join(subjects)
    people_like = {
        "woman", "man", "girl", "boy", "child", "baby", "person", "people",
        "mother", "father", "lover", "hero", "heroine", "couple", "family",
    }
    ghost_like = {
        "ghost", "spirit", "phantom", "spectre", "apparition", "wraith",
        "poltergeist", "demon", "zombie",
    }
    roots = {s.split()[-1].rstrip("s") for s in subjects} | set(subjects)
    if roots & ghost_like or any("ghost" in s for s in subjects):
        return (
            f"{listed} clearly visible as a glowing translucent ghost in frame, "
            f"not empty scenery"
        )
    if any(s.split()[-1].rstrip("s") in people_like or s in people_like for s in subjects):
        return (
            f"{listed} clearly visible with face and body in frame, "
            f"not empty scenery"
        )
    return f"{listed} clearly visible as the main subjects in frame"


def reinforce_animals_in_prompt(image_prompt: str, full_text: str, beat: str = "") -> str:
    """Force animals + people + ghosts from text into an image prompt."""
    subjects = extract_subjects(f"{full_text} {beat} {image_prompt}")
    ghosts = extract_ghosts(f"{full_text} {beat}")
    people = extract_people(f"{full_text} {beat}")
    if not subjects:
        return image_prompt
    lock = _subject_lock_clause(subjects)
    if ghosts:
        head = (
            f"cinematic horror shot of {_ghost_display_name(ghosts[0])}, "
            f"{lock}. "
        )
    elif people:
        head = (
            f"photorealistic medium shot of {_people_display_name(people[0])}, "
            f"{lock}. "
        )
    else:
        head = f"Photorealistic {', '.join(subjects)}: {lock}. "
    tail = f", must depict {', '.join(subjects)}"
    return f"{head}{image_prompt}{tail}"


def _looks_khmer(text: str) -> bool:
    return bool(re.search(r"[\u1780-\u17FF]", text or ""))


def is_visual_prompt(text: str) -> bool:
    """True when text looks like an image/video generation prompt."""
    t = (text or "").strip()
    if not t or _looks_khmer(t):
        return False
    hits = len(_VISUAL_HINTS.findall(t))
    if hits >= 2:
        return True
    if hits >= 1 and len(t) > 100 and t.count(",") >= 2:
        return True
    if re.search(r"\b(4[kK]|ultra-?realistic|cinematic)\b", t) and len(t) > 80:
        return True
    return False


def _extract_style_tail(prompt: str) -> tuple[str, str]:
    """Split body actions from trailing style keywords."""
    prompt = re.sub(r"\s+", " ", (prompt or "").strip())
    m = _STYLE_TAIL.search(prompt)
    if m:
        style = m.group(1).strip(" ,.")
        body = prompt[: m.start()].strip(" ,.")
    else:
        body, style = prompt, "cinematic still, ultra-realistic, natural lighting, detailed, 4K"

    # People / ghost descriptions must not get a wildlife default (empty landscapes)
    if mentions_ghosts(prompt):
        style = _ghost_style(style)
    elif mentions_people(prompt):
        style = _portrait_style(style)
    return body, style


def split_visual_beats(prompt: str, *, max_scenes: int = 8) -> list[tuple[str, str]]:
    """
    Break a cinematic prompt into action beats.
    Returns list of (image_prompt_en, narrate_en).
    """
    body, style = _extract_style_tail(prompt)
    chunks = re.split(r"(?<=[.!?])\s+|(?<=,)\s+then\s+|\s+then\s+", body, flags=re.I)
    chunks = [c.strip(" ,.") for c in chunks if c and c.strip(" ,.")]

    beats: list[str] = []
    for ch in chunks:
        if _STYLE_ONLY.match(ch) and not re.search(
            r"\b(frog|fish|ghost|spirit|phantom)\b", ch, re.I
        ):
            continue
        if len(ch) > 140 and ch.count(",") >= 2:
            bits = [b.strip() for b in ch.split(",") if b.strip()]
            buf = ""
            for b in bits:
                trial = f"{buf}, {b}" if buf else b
                if buf and len(trial) > 100:
                    beats.append(buf)
                    buf = b
                else:
                    buf = trial
            if buf:
                beats.append(buf)
        else:
            beats.append(ch)

    beats = [
        b
        for b in beats
        if b
        and not (
            _STYLE_ONLY.match(b)
            and not re.search(
                r"\b(frog|fish|animal|bird|cat|dog|person|girl|boy|"
                r"woman|man|mother|father|child|baby|hero|lover|people|"
                r"ghost|spirit|phantom|demon|zombie)\b",
                b,
                re.I,
            )
        )
    ]
    if not beats:
        beats = [body or prompt]

    beats = beats[:max_scenes]
    _body, style = _extract_style_tail(prompt)
    out: list[tuple[str, str]] = []
    for beat in beats:
        img = build_focused_image_prompt(beat, full_text=prompt, style_tail=style)
        out.append((img, beat))
    return out


def split_script_scenes(text: str, *, max_chars: int = 90, max_scenes: int = 24) -> list[str]:
    """Split narration script into short scenes."""
    raw = clean_khmer_text(text) if _looks_khmer(text) else (text or "").strip()
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return []

    parts = [p.strip() for p in _SENTENCE_SPLIT.split(raw) if p and p.strip()]
    if not parts:
        parts = [raw]

    scenes: list[str] = []
    buf = ""
    for part in parts:
        trial = part if not buf else f"{buf} {part}"
        if buf and len(trial) > max_chars:
            scenes.append(buf.strip())
            buf = part
        else:
            buf = trial
    if buf.strip():
        scenes.append(buf.strip())

    if len(scenes) > max_scenes:
        merged: list[str] = []
        chunk = max(1, len(scenes) // max_scenes)
        for i in range(0, len(scenes), chunk):
            merged.append(" ".join(scenes[i : i + chunk]))
        scenes = merged[:max_scenes]

    return [clean_khmer_text(s) if _looks_khmer(s) else s for s in scenes if s.strip()]


def prepare_khmer_script(text: str, source_language: str = "auto") -> str:
    """Ensure script is Khmer (translate when needed)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Please write some text first.")
    if _looks_khmer(text) or source_language == "km":
        return clean_khmer_text(text)

    src = "auto" if not source_language or source_language == "auto" else source_language
    translated = translate_text(text, source=src)
    khmer = clean_khmer_text(translated)
    if not khmer:
        raise RuntimeError("Could not translate text to Khmer. Try writing in Khmer directly.")
    return khmer


def _style_suffix(style: str) -> str:
    styles = {
        "cinematic": "cinematic still, dramatic lighting, shallow depth of field, film grain",
        "storybook": "soft storybook illustration, warm colors, gentle light",
        "nature": "nature wildlife photography, golden hour, serene",
        "modern": "modern cinematic color grade, clean aesthetic",
        "kids": (
            "bright colorful 3D preschool cartoon, cute toddler with big friendly eyes, "
            "soft rounded shapes, highly saturated happy colors, sunny wholesome lighting, "
            "nursery rhyme animation still"
        ),
        "match": "",  # visual prompt already has full style
    }
    return styles.get(style, styles["cinematic"])


def _fit_caption(text: str, max_chars: int = 52) -> str:
    """Keep on-screen Khmer short enough for 1–2 readable lines."""
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in (" ", "\u200b", "។", "៕", ","):
        idx = cut.rfind(sep)
        if idx >= max_chars // 2:
            return cut[: idx + (1 if sep == "។" else 0)].rstrip() + "..."
    return cut.rstrip() + "..."


def _video_title_en(text: str) -> str:
    """Short English title from prompt/script (first sentence)."""
    line = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip(), maxsplit=1)[0]
    line = re.sub(r"\s+", " ", line).strip(" ,.")
    if not line or _looks_khmer(line):
        return ""
    if len(line) > 58:
        line = line[:58].rsplit(" ", 1)[0].rstrip() + "..."
    return line


def _video_title_kh(text: str, khmer_fallback: str = "") -> str:
    """Short Khmer title from script or translated narration."""
    source = (text or "").strip()
    if _looks_khmer(source):
        line = re.split(r"(?<=[។!?])\s+|\n+", source, maxsplit=1)[0]
    else:
        line = re.split(r"(?<=[។!?])\s+|\n+", (khmer_fallback or "").strip(), maxsplit=1)[0]
    line = clean_khmer_text(line)
    return _fit_caption(line, max_chars=42)


def parse_target_duration(hours: float | int | None, minutes: float | int | None) -> float:
    """
    Hours + minutes → total seconds.
    Default 60s when both empty/zero. Cap at 2 hours (practical AI scene limit).
    """
    try:
        h = max(0, int(hours or 0))
    except (TypeError, ValueError):
        h = 0
    try:
        m = max(0, int(minutes or 0))
    except (TypeError, ValueError):
        m = 0
    total = h * 3600 + m * 60
    if total <= 0:
        total = 60
    return float(min(total, 2 * 3600))


def _title_subject(title: str) -> str:
    """Main character / subject from a short title."""
    t = (title or "").strip()
    animals = extract_subjects(t)
    people = extract_people(t)
    if animals:
        return animals[0]
    if people:
        return _people_display_name(people[0])
    # First meaningful word(s) from title
    cleaned = re.sub(r"[^\w\s'-]", " ", t, flags=re.UNICODE)
    words = [w for w in cleaned.split() if len(w) > 2 and w.lower() not in {
        "the", "and", "becomes", "become", "a", "an", "to", "of", "in", "on",
        "with", "from", "into", "how", "why", "what", "when", "who", "poor",
        "rich", "new", "old", "very", "true", "story",
    }]
    if words:
        return " ".join(words[:3])
    return t[:40] or "hero"


def _title_story_beats(title: str, *, n_beats: int) -> list[str]:
    """
    Expand a short title into narration beats (no LLM).
    Example title: "A Poor Cat Becomes a Millionaire"
    """
    title = re.sub(r"\s+", " ", (title or "").strip())
    subject = _title_subject(title)
    arc = [
        f"This story is called {title}.",
        f"Meet {subject}, living a hard and humble life.",
        f"{subject} dreams of a better future every day.",
        f"One day, a surprising chance appears for {subject}.",
        f"{subject} faces fear, hunger, and doubt — but keeps going.",
        f"A small kindness and bold courage open a new door.",
        f"Step by step, luck begins to turn for {subject}.",
        f"Hard work finally brings money, respect, and hope.",
        f"Against all odds, {subject} becomes a true millionaire.",
        f"Now {subject} enjoys a new life of comfort and freedom.",
        f"Friends gather as {subject} celebrates this miracle.",
        f"And so ends the story: {title}.",
    ]
    n = max(4, min(int(n_beats), 48))
    if n <= len(arc):
        # Evenly sample arc points
        idxs = [round(i * (len(arc) - 1) / (n - 1)) for i in range(n)]
        return [arc[i] for i in idxs]
    # Stretch: repeat arc with scene variations
    extras = [
        f"{subject} walks through crowded streets searching for opportunity.",
        f"Night falls, and {subject} still refuses to give up.",
        f"A stranger notices {subject} and offers unexpected help.",
        f"{subject} learns a clever skill that changes everything.",
        f"Danger comes close, but {subject} escapes with courage.",
        f"The first coins appear — proof that destiny can change.",
        f"{subject} invests wisely and grows wealth little by little.",
        f"Rivals laugh, until {subject} proves them wrong.",
        f"A golden moment arrives; {subject} seizes it.",
        f"Luxury replaces struggle as {subject} rises.",
    ]
    out = list(arc)
    i = 0
    while len(out) < n:
        out.insert(len(out) - 1, extras[i % len(extras)])
        i += 1
    return out[:n]


def expand_title_to_scenes(
    title: str,
    *,
    style: str = "cinematic",
    target_seconds: float = 60.0,
) -> tuple[str, list[SceneSpec]]:
    """
    Title-only → auto story scenes + Khmer voice lines.
    Script detail is optional; AI fills the journey from the title.
    """
    title = re.sub(r"\s+", " ", (title or "").strip())
    if not title:
        raise ValueError("Please write a video title first.")

    # ~1 beat per 10s, min 4, max 40 (long videos hold each scene longer)
    n_beats = max(4, min(40, int(round(max(30.0, target_seconds) / 10.0))))
    beats_en = _title_story_beats(title, n_beats=n_beats)
    subject = _title_subject(title)
    subjects = extract_subjects(title)
    people = extract_people(title)
    ghosts = extract_ghosts(title)

    if ghosts:
        subject_prefix = (
            f"cinematic horror shot of {_ghost_display_name(ghosts[0])} clearly visible, "
        )
    elif people:
        subject_prefix = (
            f"photorealistic medium shot of {_people_display_name(people[0])}, "
            f"face clearly visible, "
        )
    elif subjects:
        subject_prefix = f"Photorealistic {', '.join(subjects)} as main subjects, "
    else:
        subject_prefix = f"cinematic story scene featuring {subject}, "

    scenes: list[SceneSpec] = []
    narrate_kh: list[str] = []
    for beat in beats_en:
        try:
            kh = clean_khmer_text(translate_text(beat, source="en"))
        except Exception:
            kh = ""
        speak = prepare_speak_text(kh) if kh else ""
        img = (
            f"{subject_prefix}illustrating: {beat[:120]}, "
            f"story title '{title}', {_style_suffix(style)}, "
            f"emotional atmosphere, sharp focus, no text, no watermark, no letters"
        )
        img = reinforce_animals_in_prompt(img, title, beat)
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

    khmer = " ".join(narrate_kh) if narrate_kh else title
    return khmer, scenes


def _english_from_khmer(text: str) -> str:
    """Khmer script → short English title for on-screen badge."""
    try:
        from deep_translator import GoogleTranslator

        src = clean_khmer_text((text or "").strip())[:220]
        if not src:
            return ""
        en = GoogleTranslator(source="km", target="en").translate(src) or ""
        return _video_title_en(en)
    except Exception:
        return ""


def _render_latin_badge(text: str, *, width: int) -> Image.Image:
    """Small top-right badge for English / Latin text."""
    text = (text or "").strip()
    if not text:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    fonts_dir = ensure_battambang_fonts()
    font_path = fonts_dir / "Battambang-Bold.ttf"
    font_size = max(17, width // 48)
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font = ImageFont.load_default()

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x, pad_y = 16, 10
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2
    badge = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge)
    draw.rounded_rectangle(
        [0, 0, box_w, box_h],
        radius=6,
        fill=(20, 20, 24, 175),
        outline=(255, 255, 255, 90),
        width=1,
    )
    draw.text((box_w // 2, box_h // 2), text, font=font, fill=(255, 255, 255, 255), anchor="mm")
    return badge


def _render_khmer_badge(text: str, *, width: int) -> Image.Image:
    """Small top-right badge for Khmer title."""
    text = (text or "").strip()
    if not text:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    text_img = render_khmer_line(
        text,
        font_size=max(18, width // 50),
        max_width=int(width * 0.42),
        max_lines=1,
        prefer_two_lines=False,
        stroke_width=2,
    )
    pad = 10
    badge = Image.new("RGBA", (text_img.width + pad * 2, text_img.height + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge)
    draw.rounded_rectangle(
        [0, 0, badge.width, badge.height],
        radius=6,
        fill=(20, 20, 24, 175),
        outline=(255, 255, 255, 90),
        width=1,
    )
    badge.alpha_composite(text_img, (pad, pad))
    return badge


def _video_header_overlay_layer(
    *,
    width: int,
    height: int,
    note_text: str = "",
    title_en: str = "",
    title_kh: str = "",
) -> Image.Image:
    """Top-right stack: note, English title, Khmer title."""
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    badges: list[Image.Image] = []
    if (note_text or "").strip():
        badges.append(_render_latin_badge(note_text, width=width))
    if (title_en or "").strip():
        badges.append(_render_latin_badge(title_en, width=width))
    if (title_kh or "").strip():
        badges.append(_render_khmer_badge(title_kh, width=width))
    if not badges:
        return overlay

    margin = max(16, width // 50)
    gap = 8
    y = margin
    for badge in badges:
        x = width - margin - badge.width
        overlay.alpha_composite(badge, (x, y))
        y += badge.height + gap
    return overlay


def _video_note_overlay_layer(
    note_text: str,
    *,
    width: int,
    height: int,
) -> Image.Image:
    """Backward-compatible single-note overlay."""
    return _video_header_overlay_layer(
        width=width,
        height=height,
        note_text=note_text,
    )


def _caption_pages(text: str, max_chars: int = 44) -> list[str]:
    """
    Split spoken Khmer into short on-screen pages.
    Pages use the **same words in order** as TTS so text walks with the voice.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [p.strip() for p in re.split(r"(?<=[។!?])\s*", text) if p and p.strip()]
    if not sentences:
        sentences = [text]

    pages: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            pages.append(buf.strip())
        buf = ""

    for sent in sentences:
        if len(sent) <= max_chars:
            trial = f"{buf} {sent}".strip() if buf else sent
            if buf and len(trial) > max_chars:
                flush()
                buf = sent
            else:
                buf = trial
            continue

        if buf:
            flush()
        try:
            from .khmer_render import _caption_tokens

            tokens = _caption_tokens(sent)
        except Exception:
            tokens = list(sent)

        chunk = ""
        for tok in tokens:
            trial = chunk + tok
            if chunk and len(trial) > max_chars:
                pages.append(chunk)
                chunk = tok
            else:
                chunk = trial
        if chunk:
            buf = chunk

    flush()
    return pages or [text]


def _caption_pages_en(text: str, max_chars: int = 52) -> list[str]:
    """Split English captions into short on-screen pages (word-safe)."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p and p.strip()]
    if not sentences:
        sentences = [text]

    pages: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            pages.append(buf.strip())
        buf = ""

    for sent in sentences:
        if len(sent) <= max_chars:
            trial = f"{buf} {sent}".strip() if buf else sent
            if buf and len(trial) > max_chars:
                flush()
                buf = sent
            else:
                buf = trial
            continue

        if buf:
            flush()
        words = sent.split()
        chunk = ""
        for word in words:
            trial = f"{chunk} {word}".strip() if chunk else word
            if chunk and len(trial) > max_chars:
                pages.append(chunk)
                chunk = word
            else:
                chunk = trial
        if chunk:
            buf = chunk

    flush()
    return pages or [text]


def _aligned_caption_pages(caption_kh: str, caption_en: str) -> list[tuple[str, str]]:
    """Pair Khmer + English caption pages with matching timings."""
    kh_pages = _caption_pages(caption_kh, max_chars=44) if (caption_kh or "").strip() else []
    en_pages = _caption_pages_en(caption_en, max_chars=52) if (caption_en or "").strip() else []
    n = max(len(kh_pages), len(en_pages))
    if n == 0:
        return []
    while len(kh_pages) < n:
        kh_pages.append("")
    while len(en_pages) < n:
        en_pages.append("")
    return list(zip(kh_pages, en_pages))


def _latin_caption_font(font_size: int, *, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Readable Latin font for English captions (system UI font when available)."""
    candidates: list[Path] = []
    if sys.platform == "win32":
        windir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        candidates.extend(
            [
                windir / ("segoeuib.ttf" if bold else "segoeui.ttf"),
                windir / ("arialbd.ttf" if bold else "arial.ttf"),
            ]
        )
    fonts_dir = ensure_battambang_fonts()
    candidates.append(fonts_dir / ("Battambang-Bold.ttf" if bold else "Battambang-Regular.ttf"))
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_english_display_lines(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    draw: ImageDraw.ImageDraw,
    *,
    max_width: int,
    max_lines: int = 2,
    stroke_width: int = 2,
) -> list[str]:
    """Word-wrap English caption text to fit on screen (1–2 lines)."""
    words = re.sub(r"\s+", " ", (text or "").strip()).split()
    if not words:
        return []

    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip() if current else word
        bbox = draw.textbbox((0, 0), trial, font=font, stroke_width=stroke_width)
        if current and (bbox[2] - bbox[0]) > max_width:
            lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break
        else:
            current = trial
    if current and len(lines) < max_lines:
        lines.append(current)
    elif current and lines:
        # Last resort: squeeze overflow onto the final line (may be tight).
        lines[-1] = f"{lines[-1]} {current}".strip()
    return lines[:max_lines]


def _render_english_caption(text: str, *, width: int) -> Image.Image:
    """Render 1–2 lines of English caption text (white + black stroke)."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    font_size = max(22, width // 34)
    font = _latin_caption_font(font_size, bold=True)
    stroke = 2
    pad = stroke + 3
    max_width = int(width * 0.82)

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = _wrap_english_display_lines(
        text,
        font,
        probe,
        max_width=max_width,
        max_lines=2,
        stroke_width=stroke,
    )
    if not lines:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    line_boxes: list[tuple[int, int, int, int]] = []
    line_widths: list[int] = []
    line_heights: list[int] = []
    for line in lines:
        bbox = probe.textbbox((0, 0), line, font=font, stroke_width=stroke)
        line_boxes.append(bbox)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    gap = max(6, font_size // 5)
    canvas_w = min(max_width + pad * 2, max(line_widths) + pad * 2)
    canvas_h = sum(line_heights) + gap * (len(lines) - 1) + pad * 2
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    y = pad
    for line, bbox, lw, lh in zip(lines, line_boxes, line_widths, line_heights):
        x = (canvas_w - lw) // 2 - bbox[0]
        draw_y = y - bbox[1]
        draw.text(
            (x, draw_y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=stroke,
            stroke_fill=(0, 0, 0, 255),
        )
        y += lh + gap
    return canvas


def build_scene_specs(
    text: str,
    *,
    mode: str = "auto",
    source_language: str = "auto",
    style: str = "cinematic",
    video_title: str = "",
    target_seconds: float = 60.0,
    speak_language: str = "km",
) -> tuple[str, list[SceneSpec], str]:
    """
    Build scene list from user text.
    Returns (khmer_summary, scenes, resolved_mode).

    speak_language: "km" (default) speaks Khmer; "en" speaks English lyrics/script.
    """
    title = re.sub(r"\s+", " ", (video_title or "").strip())
    text = (text or "").strip()
    speak_lang = (speak_language or "km").split("-")[0].lower()
    if speak_lang not in ("en", "km"):
        speak_lang = "km"

    # Title-only / title mode: AI expands story from short title
    use_title = mode == "title" or (
        mode == "auto" and title and (not text or len(text) < 80)
    )
    if use_title and title:
        khmer, scenes = expand_title_to_scenes(
            title, style=style, target_seconds=target_seconds
        )
        if speak_lang == "en":
            scenes = _scenes_speak_english_from_captions(scenes)
        return khmer, scenes, "title"

    if not text and title:
        khmer, scenes = expand_title_to_scenes(
            title, style=style, target_seconds=target_seconds
        )
        if speak_lang == "en":
            scenes = _scenes_speak_english_from_captions(scenes)
        return khmer, scenes, "title"

    if not text:
        raise ValueError("Please write a title, prompt, or script first.")

    resolved = mode
    if mode == "auto":
        resolved = "visual" if is_visual_prompt(text) else "script"

    scenes: list[SceneSpec] = []

    if resolved == "visual":
        beats = split_visual_beats(text)
        narrate_parts: list[str] = []
        for img_prompt, narrate_en in beats:
            caption_en = re.sub(r"\s+", " ", (narrate_en or "").strip())
            if speak_lang == "en":
                speak = prepare_english_speak_text(narrate_en) if narrate_en else ""
                try:
                    kh = clean_khmer_text(translate_text(narrate_en, source="en")) if narrate_en else ""
                except Exception:
                    kh = ""
                scenes.append(
                    SceneSpec(
                        image_prompt=img_prompt,
                        speak_text=speak or ".",
                        caption=prepare_speak_text(kh) if kh else "",
                        caption_en=caption_en,
                    )
                )
                if speak:
                    narrate_parts.append(speak)
            else:
                try:
                    kh = clean_khmer_text(translate_text(narrate_en, source="en"))
                except Exception:
                    kh = ""
                if not kh and narrate_en:
                    try:
                        kh = clean_khmer_text(translate_text(narrate_en[:200], source="auto"))
                    except Exception:
                        kh = ""

                speak = prepare_speak_text(kh) if kh else ""
                scenes.append(
                    SceneSpec(
                        image_prompt=img_prompt,
                        speak_text=speak or "។",
                        caption=speak if speak else "",
                        caption_en=caption_en,
                    )
                )
                if speak:
                    narrate_parts.append(speak)

        if speak_lang == "en":
            summary = " ".join(narrate_parts) if narrate_parts else text[:500]
        else:
            summary = " ".join(narrate_parts) if narrate_parts else prepare_khmer_script(text[:500], "en")
        return summary, scenes, "visual"

    # Script / story mode — English speak keeps original lines for TTS
    if speak_lang == "en":
        en_source = text if not _looks_khmer(text) else ""
        if not en_source:
            # Khmer script but user asked English voice — translate to EN for speak
            try:
                from deep_translator import GoogleTranslator

                en_source = GoogleTranslator(source="km", target="en").translate(text[:4000]) or text
            except Exception:
                en_source = text
        en_scenes = split_script_scenes(en_source)
        try:
            khmer = prepare_khmer_script(text, source_language=source_language)
        except Exception:
            khmer = ""
        subjects = extract_subjects(en_source or text)
        ghosts = extract_ghosts(en_source or text)
        people = extract_people(en_source or text)
        if ghosts:
            display = _ghost_display_name(ghosts[0])
            subject_prefix = (
                f"cinematic horror shot of {display} clearly visible, "
                f"{display} as main supernatural subject, "
            )
        elif people and style != "kids":
            display = _people_display_name(people[0])
            subject_prefix = (
                f"photorealistic medium shot of {display}, "
                f"{display} in the foreground filling the frame, "
            )
        elif subjects:
            subject_prefix = f"Photorealistic {', '.join(subjects)} as main subjects, "
        else:
            subject_prefix = ""

        narrate_en: list[str] = []
        for idx, scene in enumerate(en_scenes):
            speak = prepare_english_speak_text(scene)
            if not speak:
                continue
            caption_en = re.sub(r"\s+", " ", scene.strip())
            # Optional Khmer caption line
            try:
                cap_kh = prepare_speak_text(
                    clean_khmer_text(translate_text(caption_en, source="en"))
                )
            except Exception:
                cap_kh = ""
            if style == "kids":
                img = (
                    f"{subject_prefix}{caption_en[:140]}, "
                    f"{_style_suffix(style)}, no text, no watermark, no letters"
                )
            elif ghosts:
                img = (
                    f"{subject_prefix}{caption_en[:180]}, "
                    f"ethereal translucent ghost, spectral mist, {_style_suffix(style)}, "
                    f"no text, no watermark"
                )
            elif people:
                img = (
                    f"{subject_prefix}{caption_en[:180]}, "
                    f"face and body clearly visible, {_style_suffix(style)}, "
                    f"no text, no watermark"
                )
            else:
                img = (
                    f"{subject_prefix}{caption_en[:220]}, "
                    f"{_style_suffix(style)}, no text, no watermark"
                )
            img = reinforce_animals_in_prompt(img, text, speak)
            scenes.append(
                SceneSpec(
                    image_prompt=img,
                    speak_text=speak,
                    caption=cap_kh,
                    caption_en=caption_en,
                )
            )
            narrate_en.append(speak)
        summary = " ".join(narrate_en) if narrate_en else en_source
        # Keep Khmer translation available in summary field when present
        return (khmer or summary), scenes, "script"

    # Script / story mode (Khmer speak — original behavior)
    khmer = prepare_khmer_script(text, source_language=source_language)
    en_scenes = split_script_scenes(text) if not _looks_khmer(text) else []
    subjects = extract_subjects(text)
    ghosts = extract_ghosts(text)
    people = extract_people(text)
    if ghosts:
        display = _ghost_display_name(ghosts[0])
        subject_prefix = (
            f"cinematic horror shot of {display} clearly visible, "
            f"{display} as main supernatural subject, "
        )
    elif people:
        display = _people_display_name(people[0])
        subject_prefix = (
            f"photorealistic medium shot of {display}, "
            f"{display} in the foreground filling the frame, "
        )
    elif subjects:
        subject_prefix = f"Photorealistic {', '.join(subjects)} as main subjects, "
    else:
        subject_prefix = ""
    kh_scenes = split_script_scenes(khmer)
    for idx, scene in enumerate(kh_scenes):
        speak = prepare_speak_text(scene)
        if not speak:
            continue
        if idx < len(en_scenes):
            caption_en = re.sub(r"\s+", " ", en_scenes[idx].strip())
        elif not _looks_khmer(text):
            caption_en = re.sub(r"\s+", " ", scene.strip())
        else:
            caption_en = _english_from_khmer(speak)
        if ghosts:
            img = (
                f"{subject_prefix}scene: {speak[:80]}, "
                f"ethereal translucent ghost, spectral mist, {_style_suffix(style)}, "
                f"sharp focus on the ghost, no text, no watermark"
            )
            if not _looks_khmer(text):
                img = (
                    f"{subject_prefix}{text[:180]}, "
                    f"ethereal translucent ghost, spectral mist, {_style_suffix(style)}, "
                    f"no text, no watermark"
                )
        elif people:
            img = (
                f"{subject_prefix}scene: {speak[:80]}, "
                f"face and body clearly visible, {_style_suffix(style)}, "
                f"sharp focus on the person, no text, no watermark"
            )
            if not _looks_khmer(text):
                img = (
                    f"{subject_prefix}{text[:180]}, "
                    f"face and body clearly visible, {_style_suffix(style)}, "
                    f"no text, no watermark"
                )
        else:
            img = (
                f"{subject_prefix}cinematic scene illustrating: {scene[:100]}, "
                f"{_style_suffix(style)}, emotional atmosphere, no text, no watermark, no letters"
            )
            # Prefer original English text snippet for image model when available
            if not _looks_khmer(text):
                if style == "kids" and idx < len(en_scenes):
                    focus = en_scenes[idx][:140]
                    img = (
                        f"{subject_prefix}{focus}, "
                        f"{_style_suffix(style)}, no text, no watermark, no letters"
                    )
                else:
                    img = (
                        f"{subject_prefix}{text[:220]}, scene focus: {speak[:60]}, "
                        f"{_style_suffix(style)}, no text, no watermark"
                    )
        img = reinforce_animals_in_prompt(img, text, speak)
        scenes.append(
            SceneSpec(
                image_prompt=img,
                speak_text=speak,
                caption=speak,
                caption_en=caption_en,
            )
        )
    return khmer, scenes, "script"


def _scenes_speak_english_from_captions(scenes: list[SceneSpec]) -> list[SceneSpec]:
    """Prefer English caption_en as TTS line (title-mode kids / EN speak)."""
    out: list[SceneSpec] = []
    for s in scenes:
        en = prepare_english_speak_text(s.caption_en or "")
        out.append(
            SceneSpec(
                image_prompt=s.image_prompt,
                speak_text=en or s.speak_text,
                caption=s.caption,
                caption_en=s.caption_en or en,
            )
        )
    return out


def _fallback_scene_image(width: int, height: int, seed: str, style: str) -> Image.Image:
    h = int(hashlib.md5(seed.encode("utf-8")).hexdigest()[:8], 16)
    palettes = {
        "cinematic": [(20, 24, 48), (80, 40, 60), (30, 60, 90)],
        "storybook": [(255, 236, 210), (255, 180, 140), (120, 170, 200)],
        "nature": [(20, 60, 40), (60, 120, 70), (200, 180, 100)],
        "modern": [(15, 20, 35), (40, 80, 120), (180, 200, 220)],
        "kids": [(255, 200, 80), (120, 210, 255), (255, 140, 180), (100, 220, 140)],
        "match": [(18, 55, 40), (40, 100, 70), (120, 160, 90)],
    }
    colors = palettes.get(style, palettes["cinematic"])
    c0 = colors[h % len(colors)]
    c1 = colors[(h // 7) % len(colors)]
    c2 = colors[(h // 13) % len(colors)]

    img = Image.new("RGB", (width, height), c0)
    px = img.load()
    for y in range(height):
        t = y / max(1, height - 1)
        r = int(c0[0] * (1 - t) + c1[0] * t)
        g = int(c0[1] * (1 - t) + c1[1] * t)
        b = int(c0[2] * (1 - t) + c1[2] * t)
        for x in range(0, width, 2):
            u = x / max(1, width - 1)
            px[x, y] = (
                int(r * (1 - u) + c2[0] * u),
                int(g * (1 - u) + c2[1] * u),
                int(b * (1 - u) + c2[2] * u),
            )
            if x + 1 < width:
                px[x + 1, y] = px[x, y]
    return img.filter(ImageFilter.GaussianBlur(radius=1.2))


def fetch_ai_scene_image(
    prompt: str,
    out_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    style: str = "cinematic",
    seed: int | None = None,
    timeout: int = 90,
    image_source: str = "ai",
    ref_urls: list[str] | None = None,
) -> Path:
    """
    Download scene image via AI / stock / user URLs.
    Kids style uses stronger cartoon prompts + multi-source fallbacks.
    """
    from .scene_images import fetch_scene_image

    # Legacy people/ghost hard prompts still help AI fidelity
    people = extract_people(prompt)
    ghosts = extract_ghosts(prompt)
    subjects = extract_subjects(prompt)
    use_prompt = prompt
    if style == "kids":
        use_prompt = prompt
    elif ghosts:
        display = _ghost_display_name(ghosts[0])
        use_prompt = (
            f"cinematic horror photo of {display}, "
            f"translucent glowing ghost clearly visible, pale face, "
            f"spectral mist, dark atmosphere, ghost filling the frame, 4K, {prompt[:120]}"
        )
    elif people:
        display = _people_display_name(people[0])
        use_prompt = (
            f"photorealistic portrait photo of {display}, "
            f"face clearly visible, upper body, detailed human skin, "
            f"person filling the frame, natural light, 4K, {prompt[:100]}"
        )
    elif subjects and style != "kids":
        use_prompt = (
            f"{', '.join(subjects)}, realistic photo, "
            f"sharp focus, natural lighting, 4K, no text, {prompt[:100]}"
        )

    return fetch_scene_image(
        use_prompt,
        Path(out_path),
        width=width,
        height=height,
        style=style if style in ("kids", "storybook", "cinematic", "nature", "modern") else "cinematic",
        seed=seed,
        source=image_source or "ai",
        ref_urls=ref_urls,
        timeout=timeout,
    )

def _compose_captioned_frame(
    background: Image.Image,
    caption_kh: str = "",
    caption_en: str = "",
    *,
    width: int,
    height: int,
) -> Image.Image:
    base = background.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
    if not (caption_kh or "").strip() and not (caption_en or "").strip():
        return base.convert("RGB")

    overlay = _caption_overlay_layer(
        caption_kh, caption_en, width=width, height=height
    )
    composed = Image.alpha_composite(base, overlay)
    return composed.convert("RGB")


def _caption_overlay_layer(
    caption_kh: str = "",
    caption_en: str = "",
    *,
    width: int,
    height: int,
) -> Image.Image:
    """Transparent full-frame layer with bottom English + Khmer captions."""
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    caption_kh = (caption_kh or "").strip()
    caption_en = (caption_en or "").strip()
    if not caption_kh and not caption_en:
        return overlay

    draw = ImageDraw.Draw(overlay)
    has_both = bool(caption_kh and caption_en)
    band_top = int(height * (0.62 if has_both else 0.70))
    for y in range(band_top, height):
        a = int(120 * ((y - band_top) / max(1, height - band_top)))
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, a))

    blocks: list[Image.Image] = []
    if caption_en:
        blocks.append(_render_english_caption(caption_en, width=width))
    if caption_kh:
        kh_img = render_khmer_line(
            caption_kh,
            font_size=max(22, width // 36),
            max_width=int(width * 0.82),
            max_lines=2,
            prefer_two_lines=True,
            stroke_width=2,
        )
        max_cap_w = int(width * 0.92)
        if kh_img.width > max_cap_w:
            ratio = max_cap_w / kh_img.width
            kh_img = kh_img.resize(
                (max_cap_w, max(1, int(kh_img.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        blocks.append(kh_img)

    gap = max(8, width // 140)
    total_h = sum(b.height for b in blocks) + gap * (len(blocks) - 1)
    y = max(24, height - total_h - max(24, height // 24))
    for block in blocks:
        x = max(0, (width - block.width) // 2)
        overlay.alpha_composite(block, (x, y))
        y += block.height + gap
    return overlay


def _subject_motion_vf(kind: str, width: int, height: int, frames: int) -> str:
    """
    Subject-aware Ken Burns / float animation.
    ghost → drift+float, people → push-in, animal → track pan, scene → slow zoom.
    """
    frames = max(24, int(frames))
    # Oversized canvas so pan/zoom has room without black edges
    pre = (
        f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,"
        f"crop={width * 2}:{height * 2},"
    )
    if kind == "ghost":
        # Floating / swaying spectral motion
        zp = (
            f"zoompan=z='1.14+0.04*sin(on/18)':"
            f"x='iw/2-(iw/zoom/2)+55*sin(on/28)':"
            f"y='ih/2-(ih/zoom/2)-35*cos(on/22)':"
            f"d={frames}:s={width}x{height}:fps=24,"
            f"eq=brightness='0.025*sin(2*PI*t/1.8)':saturation=1.08"
        )
    elif kind == "people":
        # Slow cinematic push-in toward the person
        zp = (
            f"zoompan=z='min(1.06+on*0.00042,1.20)':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)-on*0.12':"
            f"d={frames}:s={width}x{height}:fps=24"
        )
    elif kind == "animal":
        # Tracking pan + gentle zoom (alive wildlife feel)
        zp = (
            f"zoompan=z='min(1.08+on*0.00038,1.16)':"
            f"x='iw/2-(iw/zoom/2)+45*sin(on/40)':"
            f"y='ih/2-(ih/zoom/2)+12*cos(on/55)':"
            f"d={frames}:s={width}x{height}:fps=24"
        )
    else:
        zp = (
            f"zoompan=z='min(1.04+on*0.00032,1.12)':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={width}x{height}:fps=24"
        )
    return pre + zp


def _make_scene_clip(
    image_path: Path,
    audio_path: Path | None,
    caption_kh: str,
    out_mp4: Path,
    *,
    caption_en: str = "",
    duration: float | None = None,
    width: int = 1280,
    height: int = 720,
    fast: bool = True,
    motion_kind: str = "scene",
    video_note: str = "",
    title_en: str = "",
    title_kh: str = "",
) -> Path:
    """
    One scene: AI still + subject animation + voice.
    Caption pages advance with spoken words; video length matches audio.
    """
    has_audio = bool(audio_path and Path(audio_path).exists())
    if has_audio:
        audio_dur = max(1.2, get_duration_seconds(audio_path))
        # Optional target hold (title/duration mode) — pad voice if scene must run longer
        hold = float(duration) if duration and float(duration) > 0 else 0.0
        dur = max(audio_dur, hold) if hold else audio_dur
    else:
        dur = max(2.8, float(duration or 3.2))

    frames = max(30, int(round(dur * 24)))
    kind = motion_kind if motion_kind in ("ghost", "people", "animal", "scene") else "scene"
    preset = "veryfast" if fast else "medium"
    work = out_mp4.parent / f"{out_mp4.stem}_anim"
    work.mkdir(parents=True, exist_ok=True)

    # 1) Animate the still (subject-aware motion)
    still = work / "still.jpg"
    Image.open(image_path).convert("RGB").resize(
        (width, height), Image.Resampling.LANCZOS
    ).save(still, "JPEG", quality=94)

    anim = work / "motion.mp4"
    motion_vf = _subject_motion_vf(kind, width, height, frames)
    fade_vf = (
        f"{motion_vf},"
        f"fade=t=in:st=0:d=0.25,"
        f"fade=t=out:st={max(0.3, dur - 0.3):.2f}:d=0.25"
    )
    run_ffmpeg(
        [
            "-loop",
            "1",
            "-i",
            str(still),
            "-vf",
            fade_vf,
            "-t",
            f"{dur:.3f}",
            "-r",
            "24",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            "22" if fast else "20",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-y",
            str(anim),
        ]
    )

    pages = _aligned_caption_pages(caption_kh, caption_en)
    note_text = (video_note or "").strip()
    header_text_en = (title_en or "").strip()
    header_text_kh = (title_kh or "").strip()
    header_path: Path | None = None
    if note_text or header_text_en or header_text_kh:
        header_path = work / "header.png"
        _video_header_overlay_layer(
            width=width,
            height=height,
            note_text=note_text,
            title_en=header_text_en,
            title_kh=header_text_kh,
        ).save(header_path, "PNG")

    margin = max(16, width // 50)

    if not pages:
        # No captions — mux voice + optional top-right header onto animation
        if header_path:
            filter_complex = f"[0:v][1:v]overlay=W-w-{margin}:{margin}[vout]"
            cmd = ["-i", str(anim), "-i", str(header_path), "-filter_complex", filter_complex]
        else:
            cmd = ["-i", str(anim)]
        if has_audio:
            cmd.extend(["-i", str(audio_path)])
        if header_path:
            cmd.extend(
                [
                    "-map",
                    "[vout]",
                    "-c:v",
                    "libx264",
                    "-preset",
                    preset,
                    "-crf",
                    "22" if fast else "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    "24",
                    "-t",
                    f"{dur:.3f}",
                ]
            )
        else:
            cmd.extend(
                [
                    "-c:v",
                    "copy",
                    "-t",
                    f"{dur:.3f}",
                ]
            )
        if has_audio:
            audio_idx = 2 if header_path else 1
            cmd.extend(
                [
                    "-map",
                    f"{audio_idx}:a:0",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-af",
                    f"apad=whole_dur={dur:.3f}",
                    "-shortest",
                ]
            )
        else:
            cmd.extend(["-an"])
        cmd.extend(["-movflags", "+faststart", "-y", str(out_mp4)])
        run_ffmpeg(cmd)
        return out_mp4

    # 2) Timed caption overlays on top of animation (+ note on every page)
    weights = [max(8, len(kh) + len(en)) for kh, en in pages]
    total_w = float(sum(weights))
    page_durs = [dur * (w / total_w) for w in weights]
    starts: list[float] = []
    t = 0.0
    for pd in page_durs:
        starts.append(t)
        t += pd

    overlay_paths: list[Path] = []
    for i, (kh_page, en_page) in enumerate(pages):
        layer = _caption_overlay_layer(kh_page, en_page, width=width, height=height)
        op = work / f"cap_{i:03d}.png"
        layer.save(op, "PNG")
        overlay_paths.append(op)

    # Note overlay once on top (not baked into caption PNGs — avoids double badge)
    filter_parts: list[str] = []
    last = "[0:v]"
    for i, (op, st, pd) in enumerate(zip(overlay_paths, starts, page_durs)):
        end = st + pd + 0.05
        out_label = f"[v{i}]"
        filter_parts.append(
            f"{last}[{i + 1}:v]overlay=0:0:enable='between(t\\,{st:.3f}\\,{end:.3f})'{out_label}"
        )
        last = out_label
    if header_path:
        header_idx = 1 + len(overlay_paths)
        filter_parts.append(
            f"{last}[{header_idx}:v]overlay=W-w-{margin}:{margin}[vout]"
        )
    else:
        filter_parts.append(f"{last}null[vout]")
    filter_complex = ";".join(filter_parts)

    cmd = ["-i", str(anim)]
    for op in overlay_paths:
        cmd.extend(["-i", str(op)])
    if header_path:
        cmd.extend(["-i", str(header_path)])
    if has_audio:
        cmd.extend(["-i", str(audio_path)])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            "22" if fast else "20",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "24",
            "-t",
            f"{dur:.3f}",
        ]
    )
    if has_audio:
        audio_idx = 1 + len(overlay_paths) + (1 if header_path else 0)
        cmd.extend(
            [
                "-map",
                f"{audio_idx}:a:0",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-af",
                f"apad=whole_dur={dur:.3f}",
                "-shortest",
            ]
        )
    else:
        cmd.extend(["-an"])
    cmd.extend(["-movflags", "+faststart", "-y", str(out_mp4)])
    run_ffmpeg(cmd)
    return out_mp4



def create_video_from_text(
    text: str,
    output_dir: str | Path | None = None,
    *,
    video_title: str = "",
    duration_hours: int = 0,
    duration_minutes: int = 1,
    source_language: str = "auto",
    voice_label: str = "Female (Sreymom)",
    style: str = "cinematic",
    mode: str = "auto",
    speak_language: str = "km",
    image_source: str = "ai",
    image_urls: str | list[str] | None = None,
    add_music: bool = False,
    show_captions_kh: bool = False,
    show_captions_en: bool = False,
    video_note: str = "Cinema Summary",
    show_note: bool = True,
    show_title_en: bool = False,
    show_title_kh: bool = False,
    width: int = 1280,
    height: int = 720,
    fast: bool = True,
    progress_cb=None,
) -> TextVideoResult:
    """
    text/title → AI images + AI voice → MP4.

    mode: auto | title | visual | script
    speak_language: km (Khmer TTS) | en (English TTS)
    image_source: ai | mix | stock | urls
    image_urls: optional direct image links (paste or list)
    duration_hours + duration_minutes set target length (especially title mode).
    """

    def tick(msg: str, frac: float) -> None:
        if progress_cb:
            progress_cb(frac, msg)

    from .scene_images import IMAGE_SOURCE_CHOICES, parse_image_urls

    title = re.sub(r"\s+", " ", (video_title or "").strip())
    text = (text or "").strip()
    target_sec = parse_target_duration(duration_hours, duration_minutes)
    speak_lang = (speak_language or "km").split("-")[0].lower()
    if speak_lang not in ("en", "km"):
        speak_lang = "km"
    img_src = IMAGE_SOURCE_CHOICES.get(image_source or "", image_source or "ai")
    if img_src not in ("ai", "mix", "stock", "urls"):
        img_src = "ai"
    if isinstance(image_urls, list):
        ref_urls = [u for u in image_urls if u]
    else:
        ref_urls = parse_image_urls(image_urls or "")

    tick("Understanding your title / text…", 0.04)
    visual_like = mode == "visual" or (
        mode == "auto" and text and is_visual_prompt(text) and not title
    )
    use_style = "match" if visual_like else style
    # Captions: Khmer and/or English (independent toggles)
    effective_captions_kh = bool(show_captions_kh)
    effective_captions_en = bool(show_captions_en)
    effective_note = (video_note or "").strip() if show_note else ""

    khmer, specs, resolved = build_scene_specs(
        text,
        mode=mode,
        source_language=source_language,
        style=use_style if use_style != "match" else style,
        video_title=title,
        target_seconds=target_sec,
        speak_language=speak_lang,
    )
    if not specs:
        raise ValueError("No usable scenes from this text.")

    # Explicit title wins; otherwise derive from text when toggles on
    title_en = ""
    title_kh = ""
    if show_title_en:
        if title and not _looks_khmer(title):
            title_en = title if len(title) <= 58 else title[:58].rsplit(" ", 1)[0] + "..."
        elif title and _looks_khmer(title):
            title_en = _english_from_khmer(title)
        elif _looks_khmer(text):
            title_en = _english_from_khmer(text)
        else:
            title_en = _video_title_en(text)
    if show_title_kh:
        if title and _looks_khmer(title):
            title_kh = _fit_caption(clean_khmer_text(title), max_chars=42)
        elif title:
            try:
                title_kh = _fit_caption(
                    clean_khmer_text(translate_text(title, source="en")), max_chars=42
                )
            except Exception:
                title_kh = _video_title_kh(text, khmer_fallback=khmer)
        else:
            title_kh = _video_title_kh(text, khmer_fallback=khmer)

    # Even scene holds so joined video ≈ target length (title mode)
    scene_hold = max(3.0, target_sec / max(1, len(specs))) if resolved == "title" else None

    seed_src = title or text
    root = Path(output_dir) if output_dir else preferred_temp_root() / "text_video"
    root.mkdir(parents=True, exist_ok=True)
    work = root / f"tv_{hashlib.md5(seed_src.encode('utf-8')).hexdigest()[:10]}"
    work.mkdir(parents=True, exist_ok=True)
    scenes_dir = work / "scenes"
    scenes_dir.mkdir(exist_ok=True)

    (work / "image_prompts.txt").write_text(
        "\n\n".join(f"[{i + 1}]\n{s.image_prompt}" for i, s in enumerate(specs)),
        encoding="utf-8",
    )

    voice = resolve_voice(voice_label)
    clip_paths: list[Path] = []
    audio_parts: list[Path] = []
    n = len(specs)
    # Same seed family → more consistent animals across scenes
    base_seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest()[:8], 16) % 90000
    empty_mark = "." if speak_lang == "en" else "។"

    for i, spec in enumerate(specs):
        base = 0.08 + 0.75 * (i / max(1, n))
        tick(f"AI image + voice {i + 1}/{n}…", base)

        mp3 = scenes_dir / f"voice_{i:03d}.mp3"
        if speak_lang == "en":
            speak = prepare_english_speak_text(spec.speak_text)
        else:
            speak = prepare_speak_text(spec.speak_text)
        if speak and speak not in (empty_mark, "។", "."):
            try:
                text_to_speech(speak, voice, mp3, rate="-5%")
            except Exception:
                mp3 = None  # type: ignore
        else:
            mp3 = None  # type: ignore

        jpg = scenes_dir / f"scene_{i:03d}.jpg"
        fetch_ai_scene_image(
            spec.image_prompt,
            jpg,
            width=width,
            height=height,
            style=use_style if use_style != "match" else "nature",
            seed=base_seed + i * 17,
            image_source=img_src,
            ref_urls=ref_urls,
        )

        clip = scenes_dir / f"clip_{i:03d}.mp4"
        cap_kh = spec.caption if effective_captions_kh else ""
        cap_en = spec.caption_en if effective_captions_en else ""
        motion_kind = detect_motion_kind(f"{seed_src} {spec.image_prompt} {spec.speak_text}")
        _make_scene_clip(
            jpg,
            mp3,
            cap_kh,
            clip,
            caption_en=cap_en,
            duration=scene_hold if scene_hold else (3.2 if not mp3 else None),
            width=width,
            height=height,
            fast=fast,
            motion_kind=motion_kind,
            video_note=effective_note,
            title_en=title_en,
            title_kh=title_kh,
        )
        clip_paths.append(clip)
        if mp3 and Path(mp3).exists():
            audio_parts.append(Path(mp3))

    if not clip_paths:
        raise RuntimeError("Could not build any video scenes from this text.")

    tick("Joining scenes…", 0.88)
    concat_list = work / "concat.txt"
    lines = [f"file '{p.resolve().as_posix().replace(chr(39), r"'\\''")}'" for p in clip_paths]
    concat_list.write_text("\n".join(lines), encoding="utf-8")

    video_raw = work / "text_video_raw.mp4"
    run_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(video_raw)]
    )

    tick("Building voice track…", 0.92)
    narration = work / "narration.mp3"
    if audio_parts:
        audio_list = work / "audio_concat.txt"
        audio_list.write_text(
            "\n".join(
                f"file '{p.resolve().as_posix().replace(chr(39), r"'\\''")}'" for p in audio_parts
            ),
            encoding="utf-8",
        )
        run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", str(audio_list), "-c", "copy", str(narration)]
        )
    else:
        # Silent placeholder
        run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=24000:cl=mono",
                "-t",
                "1",
                "-c:a",
                "libmp3lame",
                str(narration),
            ]
        )

    download_label = title or effective_note or title_en or title_kh or "ai-video"
    download_stem = download_video_stem(seed_src, label=download_label)
    final = work / f"{download_stem}.mp4"
    if add_music and audio_parts:
        tick("Mixing background music…", 0.95)
        dur = get_duration_seconds(video_raw)
        bgm = generate_bgm(max(dur, 3.0), work / "bgm.wav")
        mixed_audio = mix_voice_and_music(narration, bgm, work / "voice_music.m4a")
        run_ffmpeg(
            [
                "-i",
                str(video_raw),
                "-i",
                str(mixed_audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(final),
            ]
        )
        narration = Path(mixed_audio)
    else:
        run_ffmpeg(
            ["-i", str(video_raw), "-c", "copy", "-movflags", "+faststart", str(final)]
        )

    voice_download = work / f"{download_stem}_voice{Path(narration).suffix}"
    if Path(narration).exists() and voice_download.resolve() != Path(narration).resolve():
        shutil.copy2(narration, voice_download)
        narration = voice_download

    tick("Done.", 1.0)
    return TextVideoResult(
        khmer_text=khmer,
        scenes=[s.caption or s.speak_text for s in specs],
        audio_path=narration,
        video_path=final,
        work_dir=work,
        mode=resolved,
    )
