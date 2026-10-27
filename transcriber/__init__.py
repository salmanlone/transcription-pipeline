"""Audio file in -> timestamped transcript out (JSON / SRT / text), via CLI or HTTP API.

Modules, in the order a request flows through them:
  api            HTTP endpoint (FastAPI) and the wiring of the layers below
  observability  request ids, log formats, one summary line per job
  limits         request-size limit around the whole app
  jobs           FIFO job queue: how many transcriptions run at once, how many wait
  pipeline       the job for one file: probe -> normalize -> chunk -> transcribe -> merge
  media          ffprobe / ffmpeg: validation, conversion to 16 kHz mono WAV, chunking
  engine         speech model interface, allowed models, device choice, model pool
  merge          chunk times -> file times, overlap de-duplication, confidence filter
  result         Segment / TranscriptionResult and SRT output
  config         all settings, read once from environment variables
"""
