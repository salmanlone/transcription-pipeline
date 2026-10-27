# Transcription Service

Audio file in, transcript with per-segment timestamps out (JSON, SRT subtitles or plain text), through a CLI or an
HTTP API. Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and ffmpeg.

**Full design, rejected alternatives and measurements: [DESIGN.md](DESIGN.md).**

## Design decisions

The guiding rule: do the cheapest check first, bound every resource, make every failure explicit, and measure
instead of guessing.

```
upload ─► cheap checks ─► fair queue ─► ffprobe check ─► ffmpeg to 16 kHz mono WAV ─► 10-min chunks
       (type, size, options)  (2 run, 10 wait)                                          │
JSON / SRT / text ◄── merge: file-wide times, drop overlap repeats and weak text ◄── faster-whisper per chunk
```

| Decision | Why | Trade-off |
|---|---|---|
| **faster-whisper** (open-source Whisper on CTranslate2), model size as a parameter, default `small` | Its maintainers report up to 4x faster than openai-whisper with comparable accuracy; int8 on CPU, GPU when available; built-in silence filter | Accuracy depends on model size; GPU path not tested on real hardware |
| **Convert every format once with ffmpeg** to 16 kHz mono 16-bit WAV | One format for everything after it; a new format is a list entry; exact, fast chunk cuts | One extra pass over the audio |
| **Cheapest checks first**: extension (415), model/language (422), size (413), then real content with ffprobe (422) | Bad requests cost almost nothing; expensive work only for valid files | The extension alone proves nothing, so the content check is required |
| **Lock down ffmpeg**: only our audio formats, local files only, a timeout on every call | I reproduced an attack: a fake `.mp3` that is really a playlist made ffmpeg read another file on the server | A new format needs its ffmpeg reader name added |
| **10-minute chunks with 2 s overlap** for long audio | The engine loads its whole input into memory (about 230 MB per hour); chunks keep it to about 38 MB | A segment crossing a cut can repeat a few words |
| **Merge rule**: chunk times become file times; a segment mostly inside already-covered audio is a repeat; times only move forward | Ordered, valid timestamps without needing word-level times | Segment-level, not word-level |
| **Clean output for downstream use**: drop empty and silence-like text, keep confidence scores, JSON / SRT / text | Other systems get clean data and can still flag doubtful parts | Confidence thresholds are defaults, not tuned on real data |
| **Model pool**: at most one model in memory, never removed while in use | A plain cache can briefly hold two large models | A little concurrency code, covered by tests |
| **Fair FIFO job queue** per worker (2 running, 10 waiting, then 503 + Retry-After) | Load test: a burst of 20 doubled throughput (1.03 to 2.12 jobs/s) with no client retries | Per process and in memory |
| **Synchronous API for now** | One node, no message broker to run; the queue is one replaceable component | Long files hold a connection open |
| **Only bad input is a client error** (422); anything else is 500 | A bug is never blamed on the client | Rejecting code must raise the specific error type |
| **Request id on every log line**, one summary line per job, transcript text never logged | Any request can be traced; transcripts can be private | None worth noting |

**Measured** (laptop CPU, Apple M3 Pro): the default `small` model runs at about 1.5x real time (80 s of speech in
52 s, 1.2 GB memory). Running more jobs at once on one worker did not add speed, so capacity comes from more
machines or GPUs.

**Known limits and next steps:** the API is synchronous and the queue lives in one process, so the next step is an
async job API (upload to object storage, a durable shared queue, GPU workers, poll or webhook) with per-chunk
retries. Then word-level timestamps, audio track selection, authentication and rate limits.

## Quick start

```bash
brew install ffmpeg                    # or: apt install ffmpeg
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt    # runtime dependencies + test, lint and type-check tools

python -m transcriber sample.mp3 --model tiny --srt sample.srt     # CLI
uvicorn transcriber.api:app --port 8000                            # HTTP service, docs at /docs
```

With Docker (ffmpeg included, runs as a non-root user):

```bash
docker build -t transcriber .          # add --build-arg PRELOAD_MODEL=small to bake the model into the image
docker run -p 8000:8000 transcriber
```

## API

```bash
curl -F "file=@meeting.mp3" "http://localhost:8000/transcribe"                       # JSON
curl -F "file=@meeting.mp3" "http://localhost:8000/transcribe?format=srt"            # subtitles
curl -F "file=@meeting.mp3" "http://localhost:8000/transcribe?model=small&language=en&format=text"
curl http://localhost:8000/health                                                     # status + queue depth
```

| Parameter | Values | Default |
|---|---|---|
| `model` | `tiny`, `base`, `small`, `medium` (and `.en` versions), `large-v3`, `large-v3-turbo` | `small` |
| `language` | Whisper language code (`en`, `ar`, `ur`, ...); omitted = detect | detect |
| `format` | `json`, `srt`, `text` | `json` |

Supported files: `.wav .mp3 .m4a .aac .flac .ogg .opus .webm .mp4 .wma .amr` (video files are accepted; only the
sound is used).

| Status | Meaning |
|---|---|
| 200 | Transcript |
| 400 / 413 | Invalid `Content-Length` / file too large |
| 415 | Unsupported file type |
| 422 | Not readable audio, or invalid model / language / format |
| 503 + `Retry-After` | Busy: the waiting line is full, the wait was too long, or the requested model is busy |
| 500 | Server error (logged with the request id) |

Every response carries an `X-Request-ID` header; the same id is on every server log line for that request.

Example output (real run, `tiny` model, the included `sample.mp3`):

```json
{
  "source_file": "sample.mp3",
  "language": "en",
  "language_probability": 0.998,
  "duration_sec": 7.5,
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.5, "text": "Hello, this is a test of the transcription pipeline.",
     "avg_logprob": -0.346, "no_speech_prob": 0.003},
    {"id": 1, "start": 3.5, "end": 7.5, "text": "A Convert's audio into text with timestamps for every segment.",
     "avg_logprob": -0.346, "no_speech_prob": 0.003}
  ],
  "text": "Hello, this is a test of the transcription pipeline. A Convert's audio into text with timestamps for every segment.",
  "stats": {"chunks": 1, "segments": 2, "dropped_segments": 0, "processing_sec": 1.89, "real_time_factor": 0.252, "model": "tiny"}
}
```

(`tiny` misheard "It converts" as "A Convert's"; the default `small` model is more accurate.)

## Configuration

All settings are environment variables, read once and validated at startup (`transcriber/config.py`).

| Variable | Default | Controls |
|---|---|---|
| `MAX_UPLOAD_MB` | 500 | Largest file accepted. Also drives the generated nginx limit. |
| `MAX_CONCURRENT_JOBS` | 2 | Transcriptions one worker runs at once. More does not add throughput. |
| `MAX_QUEUED_JOBS` | 10 | Requests that may wait in line for a slot. |
| `QUEUE_WAIT_SEC` | 300 | Longest wait in line before 503. |
| `MODEL_CACHE_SIZE` | 1 | Models kept in memory per worker. |
| `MODEL_WAIT_SEC` | 300 | Longest wait for a busy model before 503. |
| `LOG_LEVEL` | `INFO` | Log level. |
| `LOG_FORMAT` | `text` | `text`, or `json` (one object per line) for log tools. |

## Deployment

nginx sits in front of the service. Its body-size limit is generated from the same `MAX_UPLOAD_MB` setting, so the
proxy and the app cannot disagree:

```bash
MAX_UPLOAD_MB=800 python deploy/render_nginx.py      # writes deploy/nginx.conf
```

## Development

```bash
pytest -q          # 151 tests; the speech model is faked, ffmpeg runs for real
ruff check .       # lint
mypy               # type check
python tools/loadtest.py sample.mp3 --requests 20 --concurrency 20 --base http://127.0.0.1:8000
```

CI runs lint, type check and tests on Linux, then builds the Docker image and smoke-tests it
(`.github/workflows/ci.yml`).
