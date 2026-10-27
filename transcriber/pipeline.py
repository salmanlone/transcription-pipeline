"""The whole job for one file: probe -> normalize -> chunk -> transcribe -> merge -> result."""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from . import engine
from .engine import RawSegment, transcribe_chunk, validate_options
from .media import make_chunks, normalize_audio, probe_duration
from .merge import merge_and_clean
from .result import TranscriptionResult

log = logging.getLogger("transcriber")


def transcribe_file(path: str | Path, model_size: str = "small", language: str | None = None,
                    chunk_sec: int = 600) -> TranscriptionResult:
    validate_options(model_size, language)
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        duration = probe_duration(src)
        wav = normalize_audio(src, tmpd, duration)
        chunks = make_chunks(wav, duration, tmpd, chunk_sec=chunk_sec)
        log.info("%s: %.1fs audio, %d chunk(s)", src.name, duration, len(chunks))

        raw: list[tuple[float, list[RawSegment]]] = []
        info: Any = None
        with engine.use_model(model_size) as model:     # model is released (evictable) on exit
            for cpath, offset in chunks:
                segs, cinfo = transcribe_chunk(model, cpath, language)
                info = info or cinfo
                language = language or cinfo.language   # lock language after first chunk
                raw.append((offset, segs))

    segments, dropped = merge_and_clean(raw)
    elapsed = time.time() - t0
    return TranscriptionResult(
        source_file=src.name,
        language=info.language if info else (language or "unknown"),
        language_probability=round(info.language_probability, 3) if info else 0.0,
        duration_sec=round(duration, 2),
        segments=segments,
        text=" ".join(s.text for s in segments),
        stats={"chunks": len(chunks), "segments": len(segments), "dropped_segments": dropped,
               "processing_sec": round(elapsed, 2), "real_time_factor": round(elapsed / max(duration, 1e-6), 3),
               "model": model_size},
    )
