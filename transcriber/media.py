"""Audio in: validate with ffprobe, convert to one standard WAV with ffmpeg, cut long files into chunks."""
from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path

from .errors import InvalidInputError

log = logging.getLogger("transcriber")

SAMPLE_RATE = 16_000
# Timeouts for external tools. A crafted or unusual file must never hang a worker forever.
PROBE_TIMEOUT_SEC = 30            # ffprobe only reads headers; this is generous
SPLIT_TIMEOUT_SEC = 120           # per chunk; stream copy of a WAV slice, no re-encoding
NORMALIZE_BASE_SEC = 60           # decode/resample: base allowance ...
NORMALIZE_PER_AUDIO_SEC = 0.5     # ... plus 0.5 s per second of audio (ffmpeg is normally ~100x real time)

SUPPORTED_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".wma", ".amr"}

# ffmpeg picks the demuxer from the file's CONTENT, not its extension. Without these options an
# upload named .mp3 that is really an HLS playlist makes ffmpeg open the local files or URLs listed
# inside it (local file read / SSRF). So: only the demuxers SUPPORTED_EXT needs (mov covers
# mp4/m4a, ogg covers opus, matroska covers webm, asf covers wma), and only the local-file protocol.
ALLOWED_DEMUXERS = "wav,mp3,mov,aac,flac,ogg,matroska,asf,amr,amrnb,amrwb"
SAFE_INPUT_ARGS = ["-protocol_whitelist", "file", "-format_whitelist", ALLOWED_DEMUXERS]


def run_tool(cmd: list[str], timeout: float, what: str) -> subprocess.CompletedProcess[str]:
    """Run ffprobe/ffmpeg with a hard timeout. subprocess.run kills the child process
    when the timeout expires; we turn that into InvalidInputError so the API returns 422."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise InvalidInputError(f"{what} timed out after {timeout:.0f}s - file may be malformed") from None


def probe_duration(path: Path) -> float:
    """Validate the file's contents with ffprobe and return its duration in seconds.
    Rejects: unreadable files, files with no audio stream (e.g. silent video),
    and missing / non-numeric / non-finite / non-positive durations."""
    out = run_tool(
        ["ffprobe", "-v", "error", *SAFE_INPUT_ARGS,
         "-show_entries", "format=duration:stream=codec_type",
         "-of", "json", str(path)],
        timeout=PROBE_TIMEOUT_SEC, what="Reading the file",
    )
    if out.returncode != 0 or not out.stdout.strip():
        # Full ffmpeg output goes to the server log only: it contains server paths and tool details.
        log.warning("ffprobe rejected %s: %s", path.name, out.stderr.strip()[:500])
        raise InvalidInputError("Not a readable audio file")
    try:
        info = json.loads(out.stdout)
    except json.JSONDecodeError:
        raise InvalidInputError("Not a readable audio file") from None

    # Valid JSON can still have the wrong shape (e.g. [] or {"streams": "x"}); treat that as unreadable.
    if not isinstance(info, dict):
        raise InvalidInputError("Not a readable audio file")
    streams = info.get("streams", [])
    fmt = info.get("format", {})
    if not isinstance(streams, list) or not isinstance(fmt, dict):
        raise InvalidInputError("Not a readable audio file")

    if not any(isinstance(s, dict) and s.get("codec_type") == "audio" for s in streams):
        raise InvalidInputError("No audio stream found in the file")

    raw = str(fmt.get("duration", ""))
    try:
        duration = float(raw)                      # ffprobe can report "N/A" or omit it for broken files
    except ValueError:
        raise InvalidInputError(f"Not a readable audio file (invalid duration {raw!r})") from None
    if not math.isfinite(duration) or duration <= 0:   # reject nan, inf, zero, negative
        raise InvalidInputError(f"Not a readable audio file (invalid duration {raw!r})")
    return duration


def normalize_audio(src: Path, dst_dir: Path, duration: float = 0.0) -> Path:
    """Convert any input format to 16 kHz, mono, 16-bit PCM WAV (what Whisper expects)."""
    if src.suffix.lower() not in SUPPORTED_EXT:
        log.warning("Unrecognized extension %s - trying ffmpeg anyway", src.suffix)
    dst = dst_dir / "normalized.wav"
    cmd = ["ffmpeg", "-y", "-v", "error", *SAFE_INPUT_ARGS, "-i", str(src),
           "-vn",                      # drop video stream if present
           "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-af", "loudnorm",          # even out volume across recordings
           "-c:a", "pcm_s16le", str(dst)]
    timeout = NORMALIZE_BASE_SEC + NORMALIZE_PER_AUDIO_SEC * duration
    res = run_tool(cmd, timeout=timeout, what="Converting the file")
    if res.returncode != 0:
        log.warning("ffmpeg failed to decode %s: %s", src.name, res.stderr.strip()[:500])
        raise InvalidInputError("Could not decode the audio")
    return dst


def make_chunks(wav: Path, duration: float, out_dir: Path,
                chunk_sec: int = 600, overlap_sec: int = 2) -> list[tuple[Path, float]]:
    """Split into fixed windows with a small overlap so words on boundaries aren't cut.
    Returns [(chunk_path, offset_seconds)]. Short files return a single chunk."""
    if duration <= chunk_sec:
        return [(wav, 0.0)]
    chunks, start, i = [], 0.0, 0
    while start < duration:
        p = out_dir / f"chunk_{i:04d}.wav"
        res = run_tool(["ffmpeg", "-y", "-v", "error", "-ss", f"{start}", "-t", f"{chunk_sec + overlap_sec}",
                        "-i", str(wav), "-c", "copy", str(p)],
                       timeout=SPLIT_TIMEOUT_SEC, what=f"Splitting chunk {i}")
        if res.returncode != 0:
            # input here is our own normalized WAV, so a failure is a server problem (e.g. disk full) -> 500
            raise RuntimeError(f"ffmpeg failed to split chunk {i}: {res.stderr.strip()[:300]}")
        chunks.append((p, start))
        start += chunk_sec
        i += 1
    return chunks
