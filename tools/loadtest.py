"""Load test for the running service. Standard library only, so it needs no extra install.

    python tools/loadtest.py sample.mp3 --requests 20 --concurrency 20            # burst
    python tools/loadtest.py sample.mp3 --requests 12 --concurrency 2 --retry      # steady client that obeys 503

Sends --requests uploads using --concurrency parallel clients, polls /health during the run,
optionally samples the server's memory (--pid), and prints status counts, latency percentiles,
throughput and health-check latency.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def multipart(path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n")
    body = head.encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def upload(url: str, body: bytes, ctype: str, retry: bool, max_wait: float) -> dict:
    """One logical job. With retry, a 503 is retried after Retry-After (capped) until max_wait."""
    started = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                status, headers, payload = r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            status, headers, payload = e.code, e.headers, e.read()
        except Exception as e:                      # connection refused / reset / timeout
            return {"status": f"error:{type(e).__name__}", "latency": time.monotonic() - started, "attempts": attempts}
        if status == 503 and retry and time.monotonic() - started < max_wait:
            time.sleep(min(float(headers.get("Retry-After", "1")), 2.0))   # capped so the test stays short
            continue
        out = {"status": status, "latency": time.monotonic() - started, "attempts": attempts,
               "request_id": headers.get("X-Request-ID")}
        if status == 200:
            out["audio_sec"] = json.loads(payload)["duration_sec"]
        return out


def poll_health(base: str, stop: threading.Event, samples: list) -> None:
    while not stop.is_set():
        t = time.monotonic()
        try:
            urllib.request.urlopen(f"{base}/health", timeout=10).read()
            samples.append(time.monotonic() - t)
        except Exception:
            samples.append(float("inf"))
        stop.wait(0.25)


def poll_rss(pid: int, stop: threading.Event, samples: list) -> None:
    while not stop.is_set():
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        if out:
            samples.append(int(out) / 1024)          # MB
        stop.wait(0.5)


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--retry", action="store_true", help="retry 503s after Retry-After (capped at 2 s)")
    ap.add_argument("--max-wait", type=float, default=600)
    ap.add_argument("--pid", type=int, help="server process id, to sample its memory")
    a = ap.parse_args()

    body, ctype = multipart(a.audio)
    url = f"{a.base}/transcribe?model={a.model}"
    stop, health, rss = threading.Event(), [], []
    threading.Thread(target=poll_health, args=(a.base, stop, health), daemon=True).start()
    if a.pid:
        threading.Thread(target=poll_rss, args=(a.pid, stop, rss), daemon=True).start()

    t0 = time.monotonic()
    with ThreadPoolExecutor(a.concurrency) as pool:
        results = list(pool.map(lambda _: upload(url, body, ctype, a.retry, a.max_wait), range(a.requests)))
    wall = time.monotonic() - t0
    stop.set()

    by_status: dict = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r)
    ok = by_status.get(200, [])
    print(f"requests={a.requests} concurrency={a.concurrency} retry={a.retry} model={a.model} wall={wall:.1f}s")
    for status, rs in sorted(by_status.items(), key=lambda kv: str(kv[0])):
        lat = [r["latency"] for r in rs]
        print(f"  {status}: {len(rs):3d}   latency p50={pct(lat, 50):.2f}s p95={pct(lat, 95):.2f}s max={max(lat):.2f}s")
    if ok:
        audio = sum(r["audio_sec"] for r in ok)
        print(f"  throughput: {len(ok) / wall:.2f} jobs/s, {audio / wall:.1f}x real time "
              f"(audio seconds processed per wall-clock second)")
        if a.retry:
            attempts = [r["attempts"] for r in ok]
            print(f"  attempts per job: mean={statistics.mean(attempts):.1f} max={max(attempts)}")
    if health:
        finite = [h for h in health if h != float("inf")]
        if finite:
            print(f"  /health during run: {len(health)} checks, failed={len(health) - len(finite)}, "
                  f"p95={pct(finite, 95) * 1000:.0f}ms max={max(finite) * 1000:.0f}ms")
        else:
            print("  /health: all failed")
    if rss:
        print(f"  server memory: start={rss[0]:.0f}MB peak={max(rss):.0f}MB end={rss[-1]:.0f}MB")


if __name__ == "__main__":
    main()
