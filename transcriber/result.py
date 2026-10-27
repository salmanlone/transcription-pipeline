"""The output: timestamped segments plus the full text, and its SRT rendering."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Segment:
    id: int
    start: float          # seconds, global
    end: float
    text: str
    avg_logprob: float    # model confidence proxy
    no_speech_prob: float


@dataclass
class TranscriptionResult:
    source_file: str
    language: str
    language_probability: float
    duration_sec: float
    segments: list[Segment] = field(default_factory=list)
    text: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_srt(result: TranscriptionResult) -> str:
    def ts(t: float) -> str:
        ms = round(t * 1000)                       # whole milliseconds; float math gives 1.15 -> 1,149
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    return "\n".join(f"{s.id + 1}\n{ts(s.start)} --> {ts(s.end)}\n{s.text}\n" for s in result.segments)
