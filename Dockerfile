# voice-agent: one container, CPU only. Models for the free floor are baked in at build time so the
# running container needs no model downloads (asgard containers reach the internet, but builds are the
# place to pin what runs). Pinned 16 Oct 2026 (D11).
FROM python:3.12-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    VOICE_MODELS_DIR=/models VOICE_DATA_DIR=/data HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Free floor models: faster-whisper small (int8, ~250 MB) and one Piper voice (~65 MB).
ARG WHISPER_MODEL=small
ARG PIPER_VOICE=en_GB-alan-medium
RUN mkdir -p /models/whisper /models/piper && \
    python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8', download_root='/models/whisper')" && \
    python -m piper.download_voices "${PIPER_VOICE}" --data-dir /models/piper && \
    chmod -R a+rX /models

COPY voice_agent ./voice_agent
COPY config ./config
COPY web ./web
COPY docs ./docs
RUN mkdir -p /data && chmod 777 /data
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -sf http://127.0.0.1:8090/healthz || exit 1
CMD ["python", "-m", "voice_agent.app"]
