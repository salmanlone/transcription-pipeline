"""Speech-to-text engine: which models may be used, on which device, and how many stay in memory."""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Protocol

from .config import settings
from .errors import InvalidInputError, ModelBusyError

log = logging.getLogger("transcriber")


class RawSegment(Protocol):
    """What the merge step needs from each segment an engine returns (faster-whisper's Segment fits)."""
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


class SpeechModel(Protocol):
    """The engine interface. faster-whisper's WhisperModel fits it; another engine (whisper.cpp,
    a hosted API) can be plugged in through ModelPool's `loader` without touching the pipeline."""
    def transcribe(self, audio: str, **options: Any) -> tuple[Iterable[RawSegment], Any]: ...


# Only these names are accepted. WhisperModel() also accepts Hugging Face repo ids and local
# paths, so passing user input straight through would let a caller make the server download
# arbitrary models or load files from its disk.
ALLOWED_MODELS = {"tiny", "tiny.en", "base", "base.en", "small", "small.en",
                  "medium", "medium.en", "large-v3", "large-v3-turbo"}

SUPPORTED_LANGUAGES: frozenset[str] | None
try:  # Whisper's own list of language codes (en, ar, ur, fr, ...)
    from faster_whisper.tokenizer import _LANGUAGE_CODES
    SUPPORTED_LANGUAGES = frozenset(_LANGUAGE_CODES)
except Exception:  # pragma: no cover - library layout changed; skip the check rather than block
    SUPPORTED_LANGUAGES = None


def validate_options(model_size: str, language: str | None) -> None:
    """Reject bad model / language values with a clear error before any work is done."""
    if model_size not in ALLOWED_MODELS:
        raise InvalidInputError(f"Unknown model '{model_size}'. Allowed: {sorted(ALLOWED_MODELS)}")
    if language is None:
        return
    if model_size.endswith(".en") and language != "en":       # English-only models
        raise InvalidInputError(f"Model '{model_size}' is English-only; language must be 'en' or omitted")
    if SUPPORTED_LANGUAGES is not None and language not in SUPPORTED_LANGUAGES:
        raise InvalidInputError(f"Unsupported language code '{language}' (use e.g. 'en', 'ar', 'ur')")


def resolve_device(device: str = "auto") -> tuple[str, str]:
    """Pick the real device first, THEN the compute type for it.
    faster-whisper's documented defaults: float16 on GPU, int8 on CPU."""
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device '{device}' (use auto, cpu or cuda)")
    return device, ("float16" if device == "cuda" else "int8")


def _load_whisper(size: str, device: str, compute: str) -> SpeechModel:
    from faster_whisper import WhisperModel
    model: SpeechModel = WhisperModel(size, device=device, compute_type=compute)
    return model


class ModelPool:
    """Hard limit on how many models are resident in memory at once.

    A plain LRU cache only limits cache *entries*: if model A is evicted while a request is
    still transcribing with it, that request keeps A alive while B loads, so memory briefly
    holds both. This pool counts active users per model and:
      - never evicts a model that is in use,
      - reserves a slot BEFORE loading (so two requests never load the same model twice),
      - if every slot is busy with a different model, waits for one to become idle,
        and raises ModelBusyError after `wait_timeout` seconds instead of loading anyway.
    Requests must use models via `with pool.use(...)` and must not keep references afterwards.
    """

    def __init__(self, capacity: int, loader: Callable[[str, str, str], SpeechModel] = _load_whisper,
                 wait_timeout: float = 300.0):
        self.capacity, self.loader, self.wait_timeout = max(1, capacity), loader, wait_timeout
        self._entries: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()   # key -> {model, users, ready}
        self._cond = threading.Condition()

    def resident(self) -> int:
        with self._cond:
            return len(self._entries)

    def _acquire(self, size: str, device: str) -> tuple[tuple[str, str], SpeechModel]:
        validate_options(size, None)
        real_device, compute = resolve_device(device)
        key = (size, real_device)
        deadline = time.monotonic() + self.wait_timeout
        with self._cond:
            while True:
                entry = self._entries.get(key)
                if entry is not None:
                    if entry["ready"]:
                        entry["users"] += 1
                        self._entries.move_to_end(key)
                        return key, entry["model"]
                    # another request is loading this model right now -> wait for it
                elif len(self._entries) < self.capacity:
                    self._entries[key] = {"model": None, "users": 1, "ready": False}   # reserve slot
                    break
                else:
                    idle = next((k for k, e in self._entries.items() if e["ready"] and e["users"] == 0), None)
                    if idle is not None:
                        log.info("Evicting idle model %s", idle)
                        del self._entries[idle]          # no users -> no other references -> memory freed
                        continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ModelBusyError(f"All {self.capacity} model slot(s) busy; retry later")
                self._cond.wait(remaining)

        # Load outside the lock so requests using other, already-loaded models are not blocked.
        try:
            log.info("Loading model %s on %s (%s)", size, real_device, compute)
            model = self.loader(size, real_device, compute)
        except BaseException:
            with self._cond:
                del self._entries[key]
                self._cond.notify_all()
            raise
        with self._cond:
            self._entries[key].update(model=model, ready=True)
            self._cond.notify_all()
        return key, model

    def _release(self, key: tuple[str, str]) -> None:
        with self._cond:
            self._entries[key]["users"] -= 1
            self._cond.notify_all()

    @contextmanager
    def use(self, size: str = "small", device: str = "auto") -> Iterator[SpeechModel]:
        key, model = self._acquire(size, device)
        try:
            yield model
        finally:
            self._release(key)


# Default: ONE resident model per worker. In production, run one worker pool per model size
# so requests never force model switching at all.
_POOL = ModelPool(settings.model_cache_size, wait_timeout=settings.model_wait_sec)


def use_model(size: str = "small", device: str = "auto") -> AbstractContextManager[SpeechModel]:
    return _POOL.use(size, device)


def transcribe_chunk(model: SpeechModel, path: Path, language: str | None) -> tuple[list[RawSegment], Any]:
    segments, info = model.transcribe(
        str(path),
        language=language,
        vad_filter=True,                               # skip silence -> faster, can reduce hallucinations
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        condition_on_previous_text=False,              # limits error propagation on long audio
    )
    return list(segments), info
