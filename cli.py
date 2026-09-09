#!/usr/bin/env python3
"""CLI: convert a video file to Khmer subtitles / dubbed video."""

from __future__ import annotations

import argparse
from pathlib import Path

from core import KHMER_VOICES, SOURCE_LANGUAGES
from core.pipeline import convert_video_to_khmer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe multi-language video and convert to Khmer."
    )
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "-l",
        "--language",
        choices=list(SOURCE_LANGUAGES.keys()),
        default="auto",
        help="Source spoken language (default: auto)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["dub_subs", "dub", "soft_subs", "burn_subs", "srt_only"],
        default="dub_subs",
        help="Output mode (default: dub_subs = Khmer voice + subtitles)",
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="base",
        help="Whisper model size (default: base = better text quality)",
    )
    parser.add_argument(
        "--voice",
        choices=list(KHMER_VOICES.values()),
        default="km-KH-SreymomNeural",
        help="Khmer Edge TTS voice for dub mode",
    )
    parser.add_argument(
        "--story",
        action="store_true",
        help="Generate a Khmer narrative story from the transcript",
    )
    parser.add_argument(
        "--music",
        action="store_true",
        help="Add soft procedural background music under the voice / audio",
    )
    parser.add_argument(
        "--keep-music",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep original music from the upload under Khmer dub (default: on)",
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fastest pipeline (default: on). Use --no-fast for quality/sync.",
    )
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    voice_label = next(
        (k for k, v in KHMER_VOICES.items() if v == args.voice),
        args.voice,
    )

    def progress(frac, msg):
        print(f"[{frac:5.0%}] {msg}")

    result = convert_video_to_khmer(
        video_path=args.video,
        output_dir=args.output,
        source_language=args.language,
        model_size=args.model,
        mode=args.mode,
        voice_label=voice_label,
        generate_story=args.story,
        add_music=args.music,
        keep_original_music=args.keep_music,
        fast=args.fast,
        progress_cb=progress,
    )

    print("\n=== Result ===")
    print(f"Detected language : {result.detected_language}")
    print(f"Khmer SRT         : {result.khmer_srt}")
    print(f"Original SRT      : {result.original_srt}")
    if result.story_path:
        print(f"Story text        : {result.story_path}")
    if result.music_path:
        print(f"Background music  : {result.music_path}")
    if result.output_video:
        print(f"Output video      : {result.output_video}")
    print("\n--- Khmer text ---")
    print(result.khmer_text)


if __name__ == "__main__":
    main()
