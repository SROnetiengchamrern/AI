"""Detect who is speaking in the video → clear matching Khmer TTS.

People types (from pitch / speech):
  baby, girl, boy/son, woman/mother, man/father

Khmer Edge TTS only has two voices — we map roles + tune rate/pitch
so speech stays clear and closer to the person on screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# UI labels (must match KHMER_VOICES keys)
VOICE_FEMALE = "Female (Sreymom)"
VOICE_MALE = "Male (Piseth)"
VOICE_AUTO = "Auto (match video gender)"


@dataclass(frozen=True)
class SpeakProfile:
    """Concrete TTS settings for clear Khmer speech."""

    voice_label: str
    role: str  # baby | girl | boy | woman | man
    rate: str
    pitch: str


# Clearer = slightly slower; kids = higher pitch
_ROLE_PROFILES: dict[str, SpeakProfile] = {
    "baby": SpeakProfile(VOICE_FEMALE, "baby", "-12%", "+20Hz"),
    "girl": SpeakProfile(VOICE_FEMALE, "girl", "-10%", "+12Hz"),
    "boy": SpeakProfile(VOICE_MALE, "boy", "-10%", "+14Hz"),
    "woman": SpeakProfile(VOICE_FEMALE, "woman", "-8%", "+0Hz"),
    "man": SpeakProfile(VOICE_MALE, "man", "-8%", "+0Hz"),
}

# Friendly names shown in the UI summary
ROLE_DISPLAY = {
    "baby": "baby",
    "girl": "girl",
    "boy": "boy/son",
    "woman": "woman/mother",
    "man": "man/father",
}


def is_auto_voice(label: str | None) -> bool:
    t = (label or "").strip().lower()
    return t.startswith("auto") or "match video" in t or "match gender" in t or "match people" in t


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
        fmax=400,  # allow child / baby range
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
    Map median F0 → person type.

    Typical speech:
      man/father   ~ 85–150 Hz
      woman/mother ~ 150–200 Hz
      boy/son      ~ 180–230 Hz (pre-adult; often overlaps)
      girl         ~ 210–260 Hz
      baby         ~ 260+ Hz
    """
    if median_hz >= 255.0:
        return "baby"
    if median_hz >= 210.0:
        return "girl"
    if median_hz >= 175.0:
        # Mid-high: young male (son) vs soft female — prefer boy when
        # closer to male band edge; else girl for clarity on kids clips
        return "boy" if median_hz < 195.0 else "girl"
    if median_hz >= 150.0:
        return "woman"
    return "man"


def detect_speaker_role(audio_path: str | Path) -> str:
    """
    Return role: baby | girl | boy | woman | man.
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
        # Majority vote; tie → overall median
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
    if role in ("man", "boy"):
        return "male"
    return "female"


def profile_for_role(role: str) -> SpeakProfile:
    return _ROLE_PROFILES.get(role, _ROLE_PROFILES["woman"])


def resolve_khmer_voice_for_audio(
    voice_label: str | None,
    audio_path: str | Path | None = None,
) -> tuple[str, str | None]:
    """
    Map UI voice choice → concrete Khmer voice label.
    Returns (voice_label, detected_role_display_or_None).

    Prefer resolve_speak_profile() when rate/pitch are needed.
    """
    profile, detected = resolve_speak_profile(voice_label, audio_path)
    return profile.voice_label, detected


def resolve_speak_profile(
    voice_label: str | None,
    audio_path: str | Path | None = None,
) -> tuple[SpeakProfile, str | None]:
    """
    Auto: detect baby / girl / boy / woman / man → clear TTS profile.
    Manual Male/Female: clear adult rate, no role detect.
    """
    label = (voice_label or "").strip() or VOICE_AUTO

    if not is_auto_voice(label):
        low = label.lower()
        if "female" in low or "sreymom" in low:
            return _ROLE_PROFILES["woman"], None
        if "male" in low or "piseth" in low:
            return _ROLE_PROFILES["man"], None
        # Unknown label — clear female default
        return SpeakProfile(label, "woman", "-8%", "+0Hz"), None

    role = detect_speaker_role(audio_path) if audio_path else "woman"
    profile = profile_for_role(role)
    display = ROLE_DISPLAY.get(role, role)
    return profile, display
