# Render.com — Khmer AI Video Tools (CPU)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FIREBASE_REQUIRE_LOGIN=0

WORKDIR /app

# ffmpeg + audio libs (Whisper / Demucs / librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only PyTorch (smaller image; no GPU on Render web services)
RUN pip install --upgrade pip && \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE '^(torch|torchaudio)\b' requirements.txt > /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt && \
    pip install "uvicorn[standard]>=0.30.0" "fastapi>=0.115.0"

COPY . .

# Temp files + Whisper/Demucs caches
RUN mkdir -p .work .tools

EXPOSE 10000

# Render sets $PORT automatically
CMD uvicorn app:create_app --factory --host 0.0.0.0 --port ${PORT:-10000}
