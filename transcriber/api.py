"""HTTP service.

  uvicorn transcriber.api:app --port 8000
  curl -F "file=@sample.mp3" "http://localhost:8000/transcribe?model=small"
  curl -F "file=@sample.mp3" "http://localhost:8000/transcribe?format=srt"
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse

from .config import settings
from .engine import validate_options
from .errors import InvalidInputError, ModelBusyError
from .jobs import JobQueue
from .limits import MaxBodySizeMiddleware
from .media import SUPPORTED_EXT
from .observability import RequestIdMiddleware, job_log, setup_logging
from .pipeline import transcribe_file
from .result import to_srt

setup_logging()

JOBS = JobQueue(settings.max_concurrent_jobs, settings.max_queued_jobs, settings.queue_wait_sec)

api_app = FastAPI(title="Transcription Service")


@api_app.get("/health")
async def health() -> dict[str, Any]:   # async: a plain def would queue behind transcriptions on the thread pool
    # jobs.waiting is the number a load balancer or autoscaler should watch
    return {"status": "ok", "ffmpeg": bool(shutil.which("ffmpeg")), "jobs": JOBS.stats()}


@api_app.post("/transcribe", response_model=None)
async def transcribe(
    request: Request,
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Query()] = "small",
    language: Annotated[str | None, Query()] = None,
    format: Annotated[str, Query(pattern="^(json|srt|text)$")] = "json",
) -> dict[str, Any] | PlainTextResponse:
    with job_log(file=file.filename, model=model, language=language, format=format) as job:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXT:   # also rejects files with no extension
            raise HTTPException(415, f"Unsupported file type '{ext or 'none'}'. Supported: {sorted(SUPPORTED_EXT)}")
        try:
            validate_options(model, language)          # before writing any temp file of our own
        except InvalidInputError as e:
            raise HTTPException(422, str(e)) from e

        # Wait in line for a slot (FIFO), before any temp file of our own; 503 if the line is full
        # or the wait is too long, 499 if the client hangs up while waiting.
        async with JOBS.slot(request.is_disconnected) as waited:
            job["queue_sec"] = waited
            fd, tmp_path = tempfile.mkstemp(suffix=ext)
            try:
                # Everything after the temp file exists sits inside try/finally, so it is
                # deleted on every exit path: size-limit 413, unreadable-audio 422, or success.
                with os.fdopen(fd, "wb") as tmp:
                    size = 0
                    while chunk := await file.read(1024 * 1024):   # stream to disk, never hold whole file in RAM
                        size += len(chunk)
                        job["bytes"] = size
                        if size > settings.max_upload_bytes:
                            raise HTTPException(413, "File too large")
                        tmp.write(chunk)

                result = await run_in_threadpool(transcribe_file, tmp_path, model, language)
            except InvalidInputError as e:             # only the client's fault -> 422; other errors -> 500
                raise HTTPException(422, str(e)) from e
            except ModelBusyError as e:
                raise HTTPException(503, str(e), headers={"Retry-After": "30"}) from e
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        stats = result.stats
        job.update(audio_sec=result.duration_sec, language=result.language, chunks=stats.get("chunks"),
                   segments=stats.get("segments"), dropped=stats.get("dropped_segments"),
                   processing_sec=stats.get("processing_sec"), rtf=stats.get("real_time_factor"))
        result.source_file = file.filename or ""
        if format == "srt":
            return PlainTextResponse(to_srt(result))
        if format == "text":
            return PlainTextResponse(result.text)
        return result.to_dict()


# The size limiter wraps the WHOLE FastAPI app (outside its error-handling middleware), and the
# request-id wrapper sits outside both so even a 413 from the limiter carries an id.
# `uvicorn transcriber.api:app` serves this wrapped app.
app = RequestIdMiddleware(MaxBodySizeMiddleware(api_app, max_bytes=settings.max_request_bytes))
