"""End-to-end pipeline: video (any language) → Khmer subtitles / dubbed video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .extract import extract_audio, normalize_to_mp4
from .music import (
    extract_original_music_bed,
    generate_bgm,
    mix_into_video,
    mix_voice_and_music,
)
from .story import generate_khmer_story, story_to_timed_segments, write_story_file
from .subtitle import write_srt
from .transcribe import Segment, Transcript, transcribe
from .translate import merge_speak_segments, split_into_two_line_cues, translate_transcript
from .text_clean import clean_khmer_text
from .tts import (
    burn_subtitles,
    dub_and_burn,
    get_duration_seconds,
    overlay_video_note,
    replace_audio,
    resolve_voice,
    soft_sub_video,
    synthesize_segments,
)


@dataclass
class PipelineResult:
    detected_language: str
    original_text: str
    khmer_text: str
    story_text: str | None
    original_srt: Path
    khmer_srt: Path
    story_path: Path | None
    music_path: Path | None
    output_video: Path | None
    work_dir: Path


def convert_video_to_khmer(
    video_path: str | Path,
    output_dir: str | Path,
    source_language: str = "auto",
    model_size: str = "base",
    mode: str = "dub_subs",
    voice_label: str = "Female (Sreymom)",
    generate_story: bool = False,
    add_music: bool = False,
    keep_original_music: bool = True,
    show_note: bool = True,
    video_note: str = "Cinema Summary",
    fast: bool = True,
    progress_cb=None,
) -> PipelineResult:
    """
    Modes:
      - dub_subs: Khmer speaking voice + burned Khmer subtitles (recommended)
      - dub: Khmer speaking voice only (replaces original audio)
      - soft_subs / burn_subs: Khmer text only — original language still spoken
      - srt_only: subtitle files only

    Options:
      - generate_story: rewrite as a Khmer narrative story for voice/subs
      - keep_original_music: keep uploaded video music under Khmer dub (default)
      - add_music: mix soft procedural background music (if not keeping original)
      - show_note / video_note: top-right badge on the output video
      - fast: faster remux/encode; voice still uses timed TTS for quality
    """
    source_path = Path(video_path)
    # Short readable stem from original upload name (safe on Windows)
    from . import sanitize_stem

    stem = sanitize_stem(source_path.name, max_len=40)
    output_dir = Path(output_dir)
    work = output_dir / stem
    work.mkdir(parents=True, exist_ok=True)

    def tick(msg: str, frac: float) -> None:
        if progress_cb:
            progress_cb(frac, msg)

    tick("Preparing video…", 0.02)
    video_path = normalize_to_mp4(
        source_path,
        work / f"{stem}_normalized.mp4",
        fast=fast,
    )

    tick("Extracting audio…", 0.08)
    wav = extract_audio(video_path, work / "audio.wav")

    tick("Transcribing speech…", 0.2)
    lang = None if source_language == "auto" else source_language
    transcript: Transcript = transcribe(
        wav,
        language=lang,
        model_size=model_size,
        fast=fast,
    )

    if not transcript.segments:
        raise RuntimeError(
            "No speech detected in this video.\n"
            "Tips:\n"
            "• Make sure the video has spoken audio (not music-only / muted).\n"
            "• Set source language instead of Auto detect.\n"
            "• Try Whisper model Base or Small.\n"
            "• Uncheck Faster encode and try again."
        )

    tick(f"Detected language: {transcript.language} — translating to Khmer…", 0.45)
    khmer_segments: list[Segment] = translate_transcript(
        transcript,
        source=source_language,
        fast=fast,
    )
    # Clean + merge nearby lines for stable voice on long videos; split captions separately
    speak_segments = [
        Segment(s.start, s.end, clean_khmer_text(s.text))
        for s in khmer_segments
        if clean_khmer_text(s.text)
    ]
    speak_segments = merge_speak_segments(speak_segments)
    overlay_srt_segments = split_into_two_line_cues(speak_segments, max_chars=80)

    original_text = "\n".join(s.text for s in transcript.segments)
    khmer_text = "\n".join(s.text for s in speak_segments)

    story_text: str | None = None
    story_path: Path | None = None

    if generate_story:
        tick("Generating Khmer story script…", 0.55)
        story_text = clean_khmer_text(
            generate_khmer_story(
                original_text,
                khmer_text,
                source_lang=source_language,
            )
        )
        story_path = work / f"{stem}_khmer_story.txt"
        write_story_file(story_text, story_path)
        duration_hint = get_duration_seconds(video_path)
        speak_segments = story_to_timed_segments(story_text, duration_hint)
        speak_segments = [
            Segment(s.start, s.end, clean_khmer_text(s.text))
            for s in speak_segments
            if clean_khmer_text(s.text)
        ]
        speak_segments = merge_speak_segments(speak_segments)
        overlay_srt_segments = split_into_two_line_cues(speak_segments, max_chars=80)
        khmer_text = story_text

    tick("Writing subtitle files…", 0.62)
    original_srt = write_srt(transcript.segments, work / f"{stem}_original.srt")
    khmer_srt = write_srt(overlay_srt_segments, work / f"{stem}_khmer.srt")

    music_path: Path | None = None
    output_video: Path | None = None
    duration = get_duration_seconds(video_path)
    dub_mode = mode in ("dub", "dub_subs")

    # Dub modes strip original audio — keep the video's music under Khmer by default
    if keep_original_music and dub_mode:
        tick("Keeping original music from upload…", 0.66)

        def _music_progress(msg: str) -> None:
            tick(msg, 0.68)

        music_path = extract_original_music_bed(
            wav,
            work / "music_keep",
            progress_cb=_music_progress,
        )
    elif add_music:
        tick("Generating background music…", 0.68)
        music_path = generate_bgm(max(duration, 5.0), work / f"{stem}_bgm.wav")

    if mode == "srt_only":
        tick("Done (subtitles only).", 1.0)
    elif mode == "soft_subs":
        tick("Muxing soft Khmer subtitles…", 0.8)
        output_video = soft_sub_video(video_path, khmer_srt, work / f"{stem}_khmer.mp4")
        # Soft subs already keep original audio (speech + music)
        if add_music and not keep_original_music and music_path and output_video:
            tick("Mixing background music…", 0.92)
            mixed = work / f"{stem}_khmer_music.mp4"
            output_video = mix_into_video(output_video, music_path, mixed)
    elif mode == "burn_subs":
        tick("Burning Khmer subtitles…", 0.8)
        output_video = burn_subtitles(
            video_path,
            khmer_srt,
            work / f"{stem}_khmer_subs.mp4",
            fast=fast,
        )
        # Burned subs already keep original audio (speech + music)
        if add_music and not keep_original_music and music_path and output_video:
            tick("Mixing background music…", 0.92)
            mixed = work / f"{stem}_khmer_subs_music.mp4"
            output_video = mix_into_video(output_video, music_path, mixed)
    elif dub_mode:
        tick("Generating timed Khmer voice…", 0.72)
        voice = resolve_voice(voice_label)
        narration, spoken_segments = synthesize_segments(
            speak_segments,
            voice,
            work,
            video_duration=duration,
            fast=fast,
        )
        # Captions must use the same windows as the spoken audio
        overlay_srt_segments = split_into_two_line_cues(spoken_segments, max_chars=80)
        khmer_srt = write_srt(overlay_srt_segments, work / f"{stem}_khmer.srt")

        if music_path:
            tick(
                "Mixing Khmer voice + original music…"
                if keep_original_music
                else "Mixing Khmer voice + music…",
                0.82,
            )
            narration = mix_voice_and_music(
                narration,
                music_path,
                work / "khmer_voice_music.m4a",
                # Original soundtrack a bit louder than soft procedural BGM
                music_volume=0.48 if keep_original_music else 0.22,
            )

        if mode == "dub_subs":
            tick("Muxing Khmer voice + burned subtitles…", 0.88)
            output_video = dub_and_burn(
                video_path,
                narration,
                khmer_srt,
                work / f"{stem}_khmer_dub.mp4",
                segments=overlay_srt_segments,
                fast=fast,
            )
        else:
            tick("Replacing audio with Khmer voice…", 0.9)
            output_video = replace_audio(
                video_path,
                narration,
                work / f"{stem}_khmer_dub.mp4",
                fast=fast,
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    note = (video_note or "").strip() if show_note else ""
    if output_video and note:
        tick("Adding video note…", 0.96)
        noted = work / f"{output_video.stem}_noted.mp4"
        output_video = overlay_video_note(
            output_video,
            note,
            noted,
            fast=fast,
        )

    tick("Done.", 1.0)

    return PipelineResult(
        detected_language=transcript.language,
        original_text=original_text,
        khmer_text=khmer_text,
        story_text=story_text,
        original_srt=original_srt,
        khmer_srt=khmer_srt,
        story_path=story_path,
        music_path=music_path,
        output_video=output_video,
        work_dir=work,
    )
