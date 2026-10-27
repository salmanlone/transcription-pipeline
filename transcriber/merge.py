"""Chunk-local segments -> one clean, ordered list of segments in file time."""
from __future__ import annotations

from collections.abc import Sequence

from .engine import RawSegment
from .result import Segment

# Confidence filter. A segment is dropped when the text is very unlikely (avg_logprob below
# MIN_AVG_LOGPROB), or when the model thinks there is no speech AND is unsure of the text.
# The second rule has the same shape as Whisper's own silence check (no_speech_prob high AND
# avg_logprob low), so confident speech is never dropped just because no_speech_prob is high.
MIN_AVG_LOGPROB = -1.5
MAX_NO_SPEECH_PROB = 0.8
SILENCE_LOGPROB = -1.0


def merge_and_clean(raw: Sequence[tuple[float, Sequence[RawSegment]]], min_logprob: float = MIN_AVG_LOGPROB,
                    max_no_speech: float = MAX_NO_SPEECH_PROB,
                    silence_logprob: float = SILENCE_LOGPROB) -> tuple[list[Segment], int]:
    """Shift chunk-local timestamps to global time, remove overlap duplicates, drop weak segments.
    Guarantees for the output: start <= end, and each segment starts at or after the previous end."""
    out: list[Segment] = []
    dropped = 0
    last_end = 0.0
    for offset, segs in raw:
        covered = last_end                         # audio already transcribed by earlier chunks
        for s in segs:
            start, end = round(s.start + offset, 2), round(s.end + offset, 2)
            text = " ".join(s.text.split())
            if offset > 0 and (start + end) / 2 <= covered:   # mostly inside the overlap -> duplicate
                continue
            if (not text or s.avg_logprob < min_logprob
                    or (s.no_speech_prob > max_no_speech and s.avg_logprob < silence_logprob)):
                dropped += 1
                continue
            start = max(start, last_end)           # never go back in time
            end = max(end, start)                  # never end before start
            out.append(Segment(len(out), start, end, text,
                               round(s.avg_logprob, 3), round(s.no_speech_prob, 3)))
            last_end = end
    return out, dropped
