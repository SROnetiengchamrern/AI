"""Detect speaker gender from audio for matching Khmer TTS voice.

Uses pitch (F0) heuristics:
  - lower median pitch → male → Piseth
  - higher median pitch → female → Sreymom
"""

from __future__ import annotations

from pathlib import Path

# UI labels (must match KHMER_VOICES keys)
VOICE_FEMALE = "Female (Sreymom)"
VOICE_MALE = "Male (Piseth)"
VOICE_AUTO = "Auto (match video gender)"


def is_auto_voice(label: str | None) -> bool:
    t = (label or "").strip().lower()
    return t.startswith("auto") or "match video" in t or "match gender" in t


def detect_speaker_gender(audio_path: str | Path) -> str:
    """
    Return 'male' or 'female' from speech audio.
    Falls back to 'female' if analysis fails (safer default for kids/narration).
    """
    path = Path(audio_path)
    if not path.is_file():
        return "female"

    try:
        import librosa
        import numpy as np
    except ImportError:
        return "female"

    try:
        # ~90s is enough for gender; keeps long videos fast
        y, sr = librosa.load(str(path), sr=16000, mono=True, duration=90.0)
        if y is None or len(y) < sr // 2:
            return "female"

        # Focus on louder speech frames (skip silence / music beds)
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        if rms.size == 0:
            return "female"
        thresh = float(np.percentile(rms, 55))
        f0, _, _ = librosa.pyin(
            y,
            fmin=70,
            fmax=320,
            sr=sr,
            hop_length=hop,
        )
        if f0 is None:
            return "female"

        voiced = []
        for i, hz in enumerate(f0):
            if hz is None or not np.isfinite(hz):
                continue
            if i < len(rms) and rms[i] < thresh:
                continue
            if 75 <= float(hz) <= 300:
                voiced.append(float(hz))

        if len(voiced) < 12:
            # Retry without RMS gate
            voiced = [
                float(hz)
                for hz in f0
                if hz is not None and np.isfinite(hz) and 75 <= float(hz) <= 300
            ]
        if len(voiced) < 8:
            return "female"

        median_hz = float(np.median(voiced))
        # Typical adult speech F0: male ~85–155, female ~165–255
        # Mid band ~155–175 is ambiguous → lean female for this product default
        if median_hz < 160:
            return "male"
        return "female"
    except Exception:
        return "female"


def resolve_khmer_voice_for_audio(
    voice_label: str | None,
    audio_path: str | Path | None = None,
) -> tuple[str, str | None]:
    """
    Map UI voice choice → concrete Khmer voice label.
    Returns (voice_label, detected_gender_or_None).
    """
    label = (voice_label or "").strip() or VOICE_FEMALE
    if not is_auto_voice(label):
        if "male" in label.lower() and "female" not in label.lower():
            return VOICE_MALE, None
        if "female" in label.lower() or "sreymom" in label.lower():
            return VOICE_FEMALE, None
        if "piseth" in label.lower():
            return VOICE_MALE, None
        return label, None

    gender = detect_speaker_gender(audio_path) if audio_path else "female"
    if gender == "male":
        return VOICE_MALE, gender
    return VOICE_FEMALE, gender
