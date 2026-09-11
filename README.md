# Khmer AI Video Tools

Two tools in one app:

1. **Video → Khmer** — upload a video → Khmer subtitles / voice dub  
2. **Text → AI Video** — write a script → **AI voice + AI scene video**

## What it does

### Video → Khmer
1. **Extract** audio from your video  
2. **Transcribe** speech with Whisper (auto-detect language)  
3. **Translate** text to Khmer  
4. **Export** Khmer `.srt` and optionally dubbed / subtitled video (+ story / BGM)

### Text → AI Video
1. Write a script (Khmer or another language)  
2. Translate to Khmer if needed  
3. Generate **AI voice** (Edge TTS: Sreymom / Piseth)  
4. Generate **AI scene images** + Khmer captions → MP4  

## Setup

```bash
cd video-to-khmer
pip install -r requirements.txt
```

Internet is required for:
- Google Translate → Khmer
- Microsoft Edge TTS (voice)
- AI scene images (Pollinations; falls back to gradients offline)
- First-time Whisper model download (Video → Khmer only)

`ffmpeg` is bundled via `imageio-ffmpeg` (no system install required).

## Free AI MCP (Cursor)

This project ships Cursor MCP config in `.cursor/mcp.json`:

| Server | Cost | Use for |
|--------|------|---------|
| **pollinations** | Free tier (optional API key) | Images, short video, 3D via MCP chat |
| **krea-ai** | Credits | Higher-quality video / 3D when you have balance |

1. Restart Cursor (or reload MCP servers) after cloning.
2. Optional: copy `.env.example` → `.env` and set `POLLINATIONS_API_KEY` from [enter.pollinations.ai/keys](https://enter.pollinations.ai/keys) for better rate limits.
3. Same key is picked up by Kids / Text → AI Video scene images (`POLLINATIONS_API_KEY`).

In chat you can ask: *“Generate a CoComelon-style toddler nursery still with Pollinations.”*

## Web UI

```bash
python app.py
```

Open http://127.0.0.1:7860

- **Tab: Video → Khmer** — upload a sample and convert  
- **Tab: Text → AI Video** — paste a script, pick style/voice, click **Generate AI video**

## Command line (Video → Khmer)

```bash
# Soft Khmer subtitles
python cli.py sample.mp4 -o output -m soft_subs

# English → Khmer burned subtitles
python cli.py english_clip.mp4 -l en -m burn_subs

# Chinese → Khmer voice dub
python cli.py china_clip.mp4 -l zh -m dub --voice km-KH-PisethNeural

# Subtitles only
python cli.py any.mp4 -m srt_only
```

## Output

### Video → Khmer
For input `sample.mp4`, files are written under `output/sample/`:

| File | Description |
|------|-------------|
| `sample_khmer.srt` | Khmer subtitles |
| `sample_original.srt` | Original-language subtitles |
| `sample_khmer_dub.mp4` | Khmer narrated dub |

### Text → AI Video
Downloads from the UI: AI video (`.mp4`) + AI voice track.

## Supported source languages

Auto-detect, English, Chinese, Japanese, Korean, Thai, Vietnamese, French, Spanish, German, Indonesian, Malay, Hindi, Urdu (Pakistan), Russian, Arabic, and other languages Whisper understands.
