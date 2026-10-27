"""All runtime settings in one place, read once from environment variables and validated at startup."""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

MIB = 1024 * 1024


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int = 500 * MIB   # MAX_UPLOAD_MB: largest audio file accepted
    max_concurrent_jobs: int = 2        # MAX_CONCURRENT_JOBS: transcriptions one worker runs at once
    max_queued_jobs: int = 10           # MAX_QUEUED_JOBS: requests that may wait in line for a slot
    queue_wait_sec: float = 300.0       # QUEUE_WAIT_SEC: longest wait in line before 503
    model_cache_size: int = 1           # MODEL_CACHE_SIZE: models resident in memory per worker
    model_wait_sec: float = 300.0       # MODEL_WAIT_SEC: longest wait for a busy model before 503
    log_level: str = "INFO"             # LOG_LEVEL
    log_format: str = "text"            # LOG_FORMAT: text | json

    @property
    def max_request_bytes(self) -> int:
        """Whole-request cap: the file plus multipart overhead. nginx is generated from this."""
        return self.max_upload_bytes + MIB

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        def number(name: str, default: float, minimum: float, cast: Callable[[str], float] = int) -> float:
            raw = env.get(name)
            if raw is None:
                return default
            try:
                value = cast(raw)
            except ValueError:
                raise ValueError(f"{name}={raw!r} is not a valid {getattr(cast, '__name__', 'number')}") from None
            if value < minimum:
                raise ValueError(f"{name}={raw!r} must be at least {minimum}")
            return value

        log_format = env.get("LOG_FORMAT", cls.log_format).lower()
        if log_format not in ("text", "json"):
            raise ValueError(f"LOG_FORMAT={log_format!r} must be 'text' or 'json'")
        return cls(
            max_upload_bytes=int(number("MAX_UPLOAD_MB", 500, 1)) * MIB,
            max_concurrent_jobs=int(number("MAX_CONCURRENT_JOBS", cls.max_concurrent_jobs, 1)),
            max_queued_jobs=int(number("MAX_QUEUED_JOBS", cls.max_queued_jobs, 0)),
            queue_wait_sec=number("QUEUE_WAIT_SEC", cls.queue_wait_sec, 0, float),
            model_cache_size=int(number("MODEL_CACHE_SIZE", cls.model_cache_size, 1)),
            model_wait_sec=number("MODEL_WAIT_SEC", cls.model_wait_sec, 0, float),
            log_level=env.get("LOG_LEVEL", cls.log_level).upper(),
            log_format=log_format,
        )


settings = Settings.from_env()
