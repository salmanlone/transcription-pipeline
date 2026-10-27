# docker build -t transcriber .
# docker run -p 8000:8000 transcriber
# Optional: bake a model into the image so the first request doesn't download it:
#   docker build --build-arg PRELOAD_MODEL=small -t transcriber .
FROM python:3.11-slim-bookworm

# ffmpeg/ffprobe decode every input format; it is a system package, not a pip dependency
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY transcriber/ ./transcriber/

# Run as a non-root user: ffmpeg parses untrusted uploads
RUN useradd --create-home --uid 10001 app
USER app
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

ARG PRELOAD_MODEL=""
RUN if [ -n "$PRELOAD_MODEL" ]; then \
      python -c "from faster_whisper import WhisperModel; WhisperModel('$PRELOAD_MODEL', device='cpu', compute_type='int8')"; \
    fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "transcriber.api:app", "--host", "0.0.0.0", "--port", "8000"]
