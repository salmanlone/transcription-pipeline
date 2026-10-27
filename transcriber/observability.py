"""Logging: request ids on every line, text or JSON format, and one summary line per job."""
from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO, Any

from fastapi import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import settings

log = logging.getLogger("transcriber.api")

# Every log line carries the request id of the request that produced it. A ContextVar follows the
# request into the worker thread (run_in_threadpool copies the context), so pipeline logs get it too.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class TextFormatter(logging.Formatter):
    """2026-10-27 13:03:01 INFO transcriber.api [a1b2c3d4e5f6] job finished status=200 file="a.mp3" ..."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage().replace("\n", "\\n")    # one event = one line (ffmpeg output has newlines)
        line = (f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} {record.levelname} {record.name} "
                f"[{request_id_var.get()}] {message}")
        fields = getattr(record, "fields", None)
        if fields:   # json.dumps quotes strings, so spaces / newlines in a file name can't break the line
            line += " " + " ".join(f"{k}={json.dumps(v) if isinstance(v, str) else v}" for k, v in fields.items())
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log tools (CloudWatch, Datadog, Loki) to index and search."""

    def format(self, record: logging.LogRecord) -> str:
        out = {"time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
               "logger": record.name, "request_id": request_id_var.get(), "message": record.getMessage()}
        out.update(getattr(record, "fields", None) or {})
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


class _AppHandler(logging.StreamHandler):
    """Marks the handler we installed, so setup_logging() can replace it instead of stacking."""


def setup_logging(level: str | None = None, fmt: str | None = None, stream: IO[str] | None = None) -> None:
    """Configure the app's "transcriber" loggers (uvicorn keeps its own). Safe to call again."""
    level = (level or settings.log_level).upper()
    fmt = (fmt or settings.log_format).lower()
    logger = logging.getLogger("transcriber")
    for h in [h for h in logger.handlers if isinstance(h, _AppHandler)]:
        logger.removeHandler(h)
    handler = _AppHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)


REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestIdMiddleware:
    """Outermost ASGI wrapper: gives each request an id, puts it in every log line and in the
    X-Request-ID response header (also on 413s from the size limiter). A sane incoming
    X-Request-ID (e.g. from nginx) is reused so proxy and app logs can be joined."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), (b"x-request-id", rid.encode())]}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)


@contextmanager
def job_log(**fields: Any) -> Iterator[dict[str, Any]]:
    """Write exactly one summary line per /transcribe request, whatever the outcome.
    Never put transcript text in `fields`: it can be private."""
    started = time.monotonic()
    fields["status"] = 500                      # stays 500 if an unexpected error escapes
    try:
        yield fields
        fields["status"] = 200
    except HTTPException as e:
        fields["status"] = e.status_code
        fields["error"] = str(e.detail)[:200]
        raise
    finally:
        fields["total_sec"] = round(time.monotonic() - started, 3)
        status = fields["status"]
        level = logging.INFO if status < 400 else logging.WARNING if status < 500 or status == 503 else logging.ERROR
        log.log(level, "job finished", extra={"fields": fields})
