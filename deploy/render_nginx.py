"""Render deploy/nginx.conf from the SAME limit the app uses, so proxy and app never disagree.

    MAX_UPLOAD_MB=800 python deploy/render_nginx.py      # writes deploy/nginx.conf
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from transcriber.config import MIB, Settings  # noqa: E402  (reads MAX_UPLOAD_MB exactly like the service)


def nginx_body_size(settings: Settings) -> str:
    """nginx 'm' means MiB, matching the app's 1024*1024 arithmetic. Round up so nginx is never stricter."""
    return f"{-(-settings.max_request_bytes // MIB)}m"


def render(out: Path = ROOT / "nginx.conf", settings: Settings | None = None) -> Path:
    settings = settings or Settings.from_env()
    text = (ROOT / "nginx.conf.template").read_text().replace("{{CLIENT_MAX_BODY_SIZE}}", nginx_body_size(settings))
    out.write_text(text)
    return out


if __name__ == "__main__":
    s = Settings.from_env()
    path = render(settings=s)
    print(f"Wrote {path} with client_max_body_size {nginx_body_size(s)} "
          f"(app file limit {s.max_upload_bytes // MIB} MiB + 1 MiB overhead)")
