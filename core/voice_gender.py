"""Detect who is speaking → recommended clear Khmer TTS.

People types (pitch / speech):
  baby · young girl · young boy/son · young woman · young man ·
  mother/woman · father/man

Khmer Edge TTS has only Sreymom + Piseth — we map each role to the
best voice and tune rate/pitch so speech sounds clear and matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# UI labels (must match KHMER_VOICES keys where listed)
VOICE_FEMALE = "Female (Sreymom)"
VOICE_MALE = "Male (Piseth)"
VOICE_AUTO = "Auto ★ recommended (match people)"


@dataclass(frozen=True)
class SpeakProfile:
    """Concrete TTS settings for clear Khmer speech."""

    voice_label: str
    role: str
    rate: str
    pitch: str
    recommend: str  # short tip shown in UI / logs


# Clearer = slightly slower; young/kids = higher pitch
_ROLE_PROFILES: dict[str, SpeakProfile] = {
    "baby": SpeakProfile(
        VOICE_FEMALE, "baby", "-12%", "+18Hz",
        "Baby → Female (Sreymom), slow + soft high",
    ),
    "girl": SpeakProfile(
        VOICE_FEMALE, "girl", "-10%", "+12Hz",
        "Young girl / daughter → Female (Sreymom), clear young",
    ),
    "boy": SpeakProfile(
        VOICE_MALE, "boy", "-10%", "+12Hz",
        "Young boy / son → Male (Piseth), clear young",
    ),
    "young_woman": SpeakProfile(
        VOICE_FEMALE, "young_woman", "-9%", "+6Hz",
        "Young woman → Female (Sreymom), clear",
    ),
    "young_man": SpeakProfile(
        VOICE_MALE, "young_man", "-9%", "+4Hz",
        "Young man → Male (Piseth), clear",
    ),
    "woman": SpeakProfile(
        VOICE_FEMALE, "woman", "-8%", "+0Hz",
        "Mother / woman → Female (Sreymom), clear adult",
    ),
    "man": SpeakProfile(
        VOICE_MALE, "man", "-8%", "+0Hz",
        "Father / man → Male (Piseth), clear adult",
    ),
}

# Friendly names in convert summary
ROLE_DISPLAY = {
    "baby": "baby",
    "girl": "young girl / daughter",
    "boy": "young boy / son",
    "young_woman": "young woman",
    "young_man": "young man",
    "woman": "mother / woman",
    "man": "father / man",
}

# Manual dropdown choices → role
_LABEL_TO_ROLE: dict[str, str] = {
    VOICE_AUTO: "auto",
    "Auto (match video gender)": "auto",
    "Auto (match people in video)": "auto",
    VOICE_FEMALE: "woman",
    VOICE_MALE: "man",
    "Father / Man (Piseth)": "man",
    "Mother / Woman (Sreymom)": "woman",
    "Young man (Piseth)": "young_man",
    "Young woman (Sreymom)": "young_woman",
    "Young boy / Son (Piseth)": "boy",
    "Young girl / Daughter (Sreymom)": "girl",
    "Baby (Sreymom)": "baby",
    "Khmer — Female (Sreymom)": "woman",
    "Khmer — Male (Piseth)": "man",
}

# Gradio order: recommended Auto first, then roles
RECOMMENDED_VOICE_CHOICES: list[str] = [
    VOICE_AUTO,
    "Father / Man (Piseth)",
    "Mother / Woman (Sreymom)",
    "Young man (Piseth)",
    "Young woman (Sreymom)",
    "Young boy / Son (Piseth)",
    "Young girl / Daughter (Sreymom)",
    "Baby (Sreymom)",
]


def is_auto_voice(label: str | None) -> bool:
    t = (label or "").strip().lower()
    return (
        t.startswith("auto")
        or "match video" in t
        or "match gender" in t
        or "match people" in t
        or "★ recommended" in t
        or "recommended" in t and "auto" in t
    )


def _pitch_median_hz(y, sr: int) -> float | None:
    """Median F0 (Hz) from louder frames; None if not enough voiced speech."""
    import librosa
    import numpy as np

    if y is None or len(y) < sr // 2:
        return None

    try:
        y = librosa.effects.preemphasis(y)
    except Exception:
        pass

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    if rms.size == 0:
        return None
    thresh = float(np.percentile(rms, 65))

    f0, _, _ = librosa.pyin(
        y,
        fmin=85,
        fmax=400,
        sr=sr,
        hop_length=hop,
    )
    if f0 is None:
        return None

    voiced: list[float] = []
    for i, hz in enumerate(f0):
        if hz is None or not np.isfinite(hz):
            continue
        if i < len(rms) and float(rms[i]) < thresh:
            continue
        hz_f = float(hz)
        if 90.0 <= hz_f <= 380.0:
            voiced.append(hz_f)

    if len(voiced) < 15:
        voiced = [
            float(hz)
            for hz in f0
            if hz is not None and np.isfinite(hz) and 90.0 <= float(hz) <= 380.0
        ]
    if len(voiced) < 10:
        return None

    return float(np.median(voiced))


def role_from_pitch_hz(median_hz: float) -> str:
    """
    Map median F0 → recommended person type.

      father/man     ~ 85–140 Hz
      young man      ~ 140–155 Hz
      mother/woman   ~ 155–175 Hz
      young woman    ~ 175–195 Hz
      young boy/son  ~ 195–215 Hz
      young girl     ~ 215–255 Hz
      baby           ~ 255+ Hz
    """
    if median_hz >= 255.0:
        return "baby"
    if median_hz >= 215.0:
        return "girl"
    if median_hz >= 195.0:
        return "boy"
    if median_hz >= 175.0:
        return "young_woman"
    if median_hz >= 155.0:
        return "woman"
    if median_hz >= 140.0:
        return "young_man"
    return "man"


def detect_speaker_role(audio_path: str | Path) -> str:
    """
    Return role key for recommended speak profile.
    Samples several windows (skips intro music) and majority-votes.
    Falls back to 'woman' if analysis fails.
    """
    path = Path(audio_path)
    if not path.is_file():
        return "woman"

    try:
        import librosa
        import numpy as np
    except ImportError:
        return "woman"

    try:
        y, sr = librosa.load(str(path), sr=16000, mono=True, duration=180.0)
        if y is None or len(y) < sr:
            return "woman"

        total_sec = len(y) / float(sr)
        starts: list[float] = []
        if total_sec > 25:
            starts.append(12.0)
        if total_sec > 55:
            starts.append(min(40.0, total_sec * 0.35))
        if total_sec > 90:
            starts.append(min(70.0, total_sec * 0.55))
        if not starts:
            starts.append(0.0)

        votes: dict[str, int] = {}
        medians: list[float] = []
        win = int(22 * sr)

        for start_sec in starts:
            i0 = int(start_sec * sr)
            i1 = min(len(y), i0 + win)
            if i1 - i0 < sr:
                continue
            med = _pitch_median_hz(y[i0:i1], sr)
            if med is None:
                continue
            medians.append(med)
            role = role_from_pitch_hz(med)
            votes[role] = votes.get(role, 0) + 1

        if not votes:
            return "woman"
        best = max(votes.items(), key=lambda kv: kv[1])
        top_score = best[1]
        tied = [r for r, c in votes.items() if c == top_score]
        if len(tied) == 1:
            return tied[0]
        overall = float(np.median(medians)) if medians else 170.0
        return role_from_pitch_hz(overall)
    except Exception:
        return "woman"


def detect_speaker_gender(audio_path: str | Path) -> str:
    """Backward-compatible: 'male' or 'female'."""
    role = detect_speaker_role(audio_path)
    if role in ("man", "boy", "young_man"):
        return "male"
    return "female"


def profile_for_role(role: str) -> SpeakProfile:
    return _ROLE_PROFILES.get(role, _ROLE_PROFILES["woman"])


def _role_from_label(label: str) -> str | None:
    if label in _LABEL_TO_ROLE:
        return _LABEL_TO_ROLE[label]
    low = label.lower()
    if "baby" in low:
        return "baby"
    if "girl" in low or "daughter" in low:
        return "girl"
    if "boy" in low or "son" in low:
        return "boy"
    if "young woman" in low or "young female" in low:
        return "young_woman"
    if "young man" in low or "young male" in low:
        return "young_man"
    if "mother" in low or "woman" in low:
        return "woman"
    if "father" in low or ("man" in low and "woman" not in low):
        return "man"
    if "female" in low or "sreymom" in low:
        return "woman"
    if "male" in low or "piseth" in low:
        return "man"
    return None


def resolve_khmer_voice_for_audio(
    voice_label: str | None,
    audio_path: str | Path | None = None,
) -> tuple[str, str | None]:
    """Map UI voice → Khmer voice label + detected/chosen role display."""
    profile, detected = resolve_speak_profile(voice_label, audio_path)
    return profile.voice_label, detected


def resolve_speak_profile(
    voice_label: str | None,
    audio_path: str | Path | None = None,
) -> tuple[SpeakProfile, str | None]:
    """
    Auto: detect people in video → recommended speak profile.
    Manual role / Male / Female: use that recommended profile.
    """
    label = (voice_label or "").strip() or VOICE_AUTO

    if is_auto_voice(label):
        role = detect_speaker_role(audio_path) if audio_path else "woman"
        profile = profile_for_role(role)
        return profile, ROLE_DISPLAY.get(role, role)

    role = _role_from_label(label)
    if role and role != "auto":
        profile = profile_for_role(role)
        return profile, None

    return SpeakProfile(label, "woman", "-8%", "+0Hz", "Custom voice"), None
