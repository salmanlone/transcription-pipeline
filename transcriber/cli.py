"""Command line: python -m transcriber meeting.mp3 --model small --out result.json --srt result.srt"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from .engine import ALLOWED_MODELS
from .pipeline import transcribe_file
from .result import to_srt


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m transcriber",
                                 description="Transcribe an audio file with segment timestamps")
    ap.add_argument("audio")
    ap.add_argument("--model", default="small", choices=sorted(ALLOWED_MODELS))
    ap.add_argument("--language", default=None, help="force language code, e.g. en, ar")
    ap.add_argument("--chunk-sec", type=int, default=600)
    ap.add_argument("--out", default=None, help="write JSON here (default: stdout)")
    ap.add_argument("--srt", default=None, help="also write an .srt subtitle file")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required (brew install ffmpeg / apt install ffmpeg)")

    result = transcribe_file(a.audio, a.model, a.language, a.chunk_sec)
    payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    if a.srt:
        Path(a.srt).write_text(to_srt(result), encoding="utf-8")
