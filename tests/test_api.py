"""API tests: validation errors and temp-file cleanup on every exit path. Run: pytest -q"""
import re
import tempfile
from dataclasses import replace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from transcriber import api, errors, jobs, limits, media, observability, pipeline, result


@pytest.fixture
def tracked_tmp(monkeypatch):
    """Record every temp file the endpoint creates so tests can assert it was deleted."""
    created = []
    real_mkstemp = tempfile.mkstemp

    def spy(*a, **k):
        fd, path = real_mkstemp(*a, **k)
        created.append(path)
        return fd, path

    monkeypatch.setattr(api.tempfile, "mkstemp", spy)
    return created


client = TestClient(api.app)


def test_unsupported_extension_returns_415(tracked_tmp):
    r = client.post("/transcribe", files={"file": ("notes.txt", b"hello")})
    assert r.status_code == 415
    assert tracked_tmp == []                      # rejected before any temp file is made


def test_extensionless_upload_returns_415(tracked_tmp):
    r = client.post("/transcribe", files={"file": ("recording", b"data")})
    assert r.status_code == 415
    assert tracked_tmp == []


def test_oversize_upload_returns_413_and_cleans_up(tracked_tmp, monkeypatch):
    monkeypatch.setattr(api, "settings", replace(api.settings, max_upload_bytes=10))
    r = client.post("/transcribe", files={"file": ("big.mp3", b"x" * 100)})
    assert r.status_code == 413
    assert len(tracked_tmp) == 1
    assert not __import__("os").path.exists(tracked_tmp[0])   # temp file removed


def test_wrong_probe_shape_returns_422_not_500(tracked_tmp, monkeypatch):
    from types import SimpleNamespace as NS
    monkeypatch.setattr(media.subprocess, "run", lambda *a, **k: NS(returncode=0, stdout="[]", stderr=""))
    r = client.post("/transcribe", files={"file": ("a.mp3", b"data")})
    assert r.status_code == 422


@pytest.mark.parametrize("fmt,ctype,check", [
    ("json", "application/json", lambda r: r.json()["segments"][0]["text"] == "hello"),
    ("srt", "text/plain", lambda r: "00:00:00,000 --> 00:00:01,500" in r.text),
    ("text", "text/plain", lambda r: r.text == "hello"),
])
def test_format_parameter_changes_response(monkeypatch, fmt, ctype, check):
    fake = result.TranscriptionResult("a.mp3", "en", 1.0, 1.5,
                                        [result.Segment(0, 0.0, 1.5, "hello", -0.1, 0.0)], text="hello")
    monkeypatch.setattr(api, "transcribe_file", lambda *a, **k: fake)
    r = client.post(f"/transcribe?format={fmt}", files={"file": ("a.mp3", b"data")})
    assert r.status_code == 200 and r.headers["content-type"].startswith(ctype) and check(r)


def test_invalid_format_rejected():
    r = client.post("/transcribe?format=xml", files={"file": ("a.mp3", b"data")})
    assert r.status_code == 422


def test_fake_audio_returns_422_and_cleans_up(tracked_tmp):
    r = client.post("/transcribe", files={"file": ("fake.mp3", b"this is not audio")})
    assert r.status_code == 422                   # ffprobe content check rejects it
    assert "Not a readable audio file" in r.json()["detail"]
    assert not __import__("os").path.exists(tracked_tmp[0])


# ---------------------------------------------------------------- request-size limit at the server boundary
import asyncio


def test_limiter_wraps_the_whole_app():
    assert isinstance(api.app, observability.RequestIdMiddleware)      # outermost: ids on every response
    assert isinstance(api.app.app, limits.MaxBodySizeMiddleware)
    assert api.app.app.app is api.api_app                  # outside FastAPI's own middleware


def _run_raw(app, headers, chunks):
    """Drive the ASGI app directly so we control exactly which headers are sent.
    Returns (messages sent to client, number of body chunks the app pulled)."""
    sent, pulled = [], []
    queue = list(chunks)

    async def receive():
        if not queue:
            return {"type": "http.disconnect"}
        body = queue.pop(0)
        pulled.append(body)
        return {"type": "http.request", "body": body, "more_body": bool(queue)}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": "/transcribe", "raw_path": b"/transcribe", "query_string": b"",
             "root_path": "", "headers": headers, "client": ("test", 1), "server": ("test", 80)}
    asyncio.run(app(scope, receive, send))
    return sent, len(pulled)


def _multipart_chunks(n_chunks, chunk_size):
    head = (b"--zzz\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp3\"\r\n"
            b"Content-Type: audio/mpeg\r\n\r\n")
    return [head] + [b"x" * chunk_size] * n_chunks + [b"\r\n--zzz--\r\n"]


def test_chunked_oversize_gets_exactly_one_413_and_stops_reading(monkeypatch):
    reached = []
    monkeypatch.setattr(api, "transcribe_file", lambda *a, **k: reached.append(True))
    wrapped = limits.MaxBodySizeMiddleware(api.api_app, max_bytes=1000)
    headers = [(b"content-type", b"multipart/form-data; boundary=zzz"),
               (b"transfer-encoding", b"chunked")]                      # NO content-length
    assert not any(k == b"content-length" for k, _ in headers)

    sent, pulled = _run_raw(wrapped, headers, _multipart_chunks(n_chunks=50, chunk_size=100))

    starts = [m for m in sent if m["type"] == "http.response.start"]
    bodies = [m for m in sent if m["type"] == "http.response.body"]
    assert len(starts) == 1 and starts[0]["status"] == 413            # one response, and it is 413
    assert dict(starts[0]["headers"])[b"content-type"] == b"application/json"
    assert b"".join(m["body"] for m in bodies) == b'{"detail": "Request too large"}'
    assert pulled < 52                                                 # stopped early, not all 52 chunks
    assert reached == []                                               # endpoint never ran


def test_declared_oversize_rejected_without_reading_body():
    wrapped = limits.MaxBodySizeMiddleware(api.api_app, max_bytes=1000)
    headers = [(b"content-type", b"multipart/form-data; boundary=zzz"), (b"content-length", b"999999")]
    sent, pulled = _run_raw(wrapped, headers, _multipart_chunks(5, 100))
    assert sent[0]["status"] == 413 and pulled == 0


def test_invalid_content_length_returns_400():
    wrapped = limits.MaxBodySizeMiddleware(api.api_app, max_bytes=1000)
    sent, pulled = _run_raw(wrapped, [(b"content-length", b"abc")], [b""])
    assert sent[0]["status"] == 400 and pulled == 0


def test_declared_oversize_via_http_client():
    c = TestClient(limits.MaxBodySizeMiddleware(api.api_app, max_bytes=1000))
    r = c.post("/transcribe", files={"file": ("a.mp3", b"x" * 5000)})
    assert r.status_code == 413 and r.json() == {"detail": "Request too large"}


def test_request_under_limit_passes_through():
    c = TestClient(limits.MaxBodySizeMiddleware(api.api_app, max_bytes=10_000))
    r = c.post("/transcribe", files={"file": ("a.mp3", b"not audio")})
    assert r.status_code == 422                            # reached the endpoint, failed content check


def test_unrelated_app_errors_still_raise(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bug")
    monkeypatch.setattr(api, "transcribe_file", boom)
    c = TestClient(api.app, raise_server_exceptions=False)
    r = c.post("/transcribe", files={"file": ("a.mp3", b"data")})
    assert r.status_code == 500                            # limiter does not hide real bugs


def test_probe_timeout_returns_422(monkeypatch):
    import subprocess
    def hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd=a[0], timeout=k.get("timeout"))
    monkeypatch.setattr(media.subprocess, "run", hang)
    r = TestClient(api.app).post("/transcribe", files={"file": ("a.mp3", b"data")})
    assert r.status_code == 422 and "timed out" in r.json()["detail"]


# ---------------------------------------------------------------- limiter + routes that respond before reading the body
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

CHUNKED = [(b"content-type", b"application/octet-stream"), (b"transfer-encoding", b"chunked")]


def _run_raw_route(app, path, method, chunks):
    sent, pulled = [], []
    queue = list(chunks)

    async def receive():
        if not queue:
            return {"type": "http.disconnect"}
        body = queue.pop(0)
        pulled.append(body)
        return {"type": "http.request", "body": body, "more_body": bool(queue)}

    async def send(message):
        sent.append(message)

    # spec_version 2.4: Starlette's StreamingResponse then doesn't run a parallel disconnect
    # listener that would steal body chunks from this single-queue test harness.
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1", "method": method,
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "", "headers": CHUNKED, "client": ("test", 1), "server": ("test", 80)}
    asyncio.run(app(scope, receive, send))
    return sent, len(pulled)


def test_route_that_never_reads_body_is_unaffected():
    """Real /health route with an oversized chunked body: normal 200, body never pulled."""
    wrapped = limits.MaxBodySizeMiddleware(api.api_app, max_bytes=1000)
    sent, pulled = _run_raw_route(wrapped, "/health", "GET", [b"x" * 600] * 10)
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert [m["status"] for m in starts] == [200]
    assert pulled == 0


def test_response_already_started_then_oversize_body_is_cut_off_not_413(caplog):
    """A route that sends headers first, then reads the body (streaming echo)."""
    demo = FastAPI()
    chunks_seen = []

    @demo.post("/echo")
    async def echo(request: Request):
        async def gen():
            yield b"started\n"                       # response headers go out before the body is read
            async for part in request.stream():
                chunks_seen.append(len(part))
                yield b"."
            yield b"\ndone"
        return StreamingResponse(gen())

    wrapped = limits.MaxBodySizeMiddleware(demo, max_bytes=1000)
    sent, pulled = _run_raw_route(wrapped, "/echo", "POST", [b"x" * 300] * 20)   # 6000 bytes

    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert [m["status"] for m in starts] == [200]        # exactly one status line; no late 413 injected
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert b"done" not in body                           # response is cut off, never looks complete
    assert not any(m["type"] == "http.response.body" and not m.get("more_body", False)
                   for m in sent)                        # final "complete" body message never forwarded
    assert pulled <= 5 and sum(chunks_seen) <= 1000      # stopped reading at the limit (4 chunks + the one over)
    assert "already started its response" in caplog.text  # operators can see why it was cut off


def test_response_started_route_under_limit_completes_normally():
    demo = FastAPI()

    @demo.post("/echo")
    async def echo(request: Request):
        async def gen():
            yield b"started\n"
            async for _ in request.stream():
                yield b"."
            yield b"\ndone"
        return StreamingResponse(gen())

    wrapped = limits.MaxBodySizeMiddleware(demo, max_bytes=10_000)
    sent, _ = _run_raw_route(wrapped, "/echo", "POST", [b"x" * 300] * 5)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert body.endswith(b"done")


@pytest.mark.parametrize("query", ["model=someone/evil", "model=huge", "language=xx"])
def test_bad_model_or_language_returns_422_before_any_temp_file(tracked_tmp, query):
    r = client.post(f"/transcribe?{query}", files={"file": ("a.mp3", b"data")})
    assert r.status_code == 422
    assert tracked_tmp == []


def test_english_only_model_with_other_language_returns_422(tracked_tmp):
    r = client.post("/transcribe?model=small.en&language=ar", files={"file": ("a.mp3", b"data")})
    assert r.status_code == 422 and "English-only" in r.json()["detail"]
    assert tracked_tmp == []


def test_model_busy_returns_503_with_retry_after(monkeypatch):
    def busy(*a, **k):
        raise errors.ModelBusyError("All 1 model slot(s) busy; retry later")
    monkeypatch.setattr(api, "transcribe_file", busy)
    r = client.post("/transcribe", files={"file": ("a.mp3", b"data")})
    assert r.status_code == 503 and r.headers["retry-after"] == "30"


# ---------------------------------------------------------------- health, job limit, error mapping
def test_health_is_async_so_it_never_waits_for_the_thread_pool():
    assert asyncio.iscoroutinefunction(api.health)
    assert client.get("/health").json()["status"] == "ok"


def _full_queue():
    """Every slot busy and no room in the line."""
    q = jobs.JobQueue(capacity=1, max_waiting=0, wait_timeout=5)
    q.running = 1
    return q


def test_full_queue_returns_503_before_any_temp_file(tracked_tmp, monkeypatch):
    monkeypatch.setattr(api, "JOBS", _full_queue())
    r = client.post("/transcribe", files={"file": ("a.mp3", b"data")})
    assert r.status_code == 503 and r.headers["retry-after"] == "30"
    assert tracked_tmp == []


def _boom(exc):
    def raiser(*a, **k):
        raise exc
    return raiser


@pytest.mark.parametrize("outcome,expected", [
    (None, 200),                                                        # success
    (_boom(errors.InvalidInputError("bad file")), 422),
    (_boom(errors.ModelBusyError("busy")), 503),
    (_boom(RuntimeError("bug")), 500),
])
def test_job_slot_is_released_on_every_exit_path(monkeypatch, outcome, expected):
    ok = result.TranscriptionResult("a.mp3", "en", 1.0, 1.0, [], text="")
    monkeypatch.setattr(api, "transcribe_file", outcome or (lambda *a, **k: ok))
    c = TestClient(api.app, raise_server_exceptions=False)
    assert c.post("/transcribe", files={"file": ("a.mp3", b"data")}).status_code == expected
    assert api.JOBS.running == 0


def test_job_slot_is_released_after_413(monkeypatch):
    monkeypatch.setattr(api, "settings", replace(api.settings, max_upload_bytes=10))
    assert client.post("/transcribe", files={"file": ("a.mp3", b"x" * 100)}).status_code == 413
    assert api.JOBS.running == 0


def test_unexpected_value_error_is_500_not_blamed_on_client(tracked_tmp, monkeypatch):
    """A ValueError from a bug in our code or a library must not turn into a 422."""
    monkeypatch.setattr(api, "transcribe_file", _boom(ValueError("internal bug")))
    c = TestClient(api.app, raise_server_exceptions=False)
    r = c.post("/transcribe", files={"file": ("a.mp3", b"data")})
    assert r.status_code == 500
    assert not __import__("os").path.exists(tracked_tmp[0])            # still cleaned up


def test_422_detail_does_not_leak_server_paths_or_ffmpeg_output(tracked_tmp):
    r = client.post("/transcribe", files={"file": ("fake.mp3", b"this is not audio")})
    detail = r.json()["detail"]
    assert r.status_code == 422 and detail == "Not a readable audio file"
    assert "tmp" not in detail and "/" not in detail


# ---------------------------------------------------------------- logging: request ids and one summary line per job
import io
import json
import logging


@pytest.fixture
def json_logs():
    buf = io.StringIO()
    observability.setup_logging(level="INFO", fmt="json", stream=buf)
    yield lambda: [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    observability.setup_logging()


def _fake_result(text="hello"):
    return result.TranscriptionResult(
        "a.mp3", "en", 0.99, 7.5, [result.Segment(0, 0.0, 1.5, text, -0.1, 0.0)], text=text,
        stats={"chunks": 1, "segments": 1, "dropped_segments": 0, "processing_sec": 1.9, "real_time_factor": 0.25})


def test_every_response_gets_a_request_id_and_a_sane_incoming_one_is_reused():
    r = client.get("/health")
    assert re.fullmatch(r"[0-9a-f]{12}", r.headers["x-request-id"])
    assert client.get("/health", headers={"X-Request-ID": "nginx-abc.123"}).headers["x-request-id"] == "nginx-abc.123"
    for bad in ["has space", "x" * 65, "new\tline"]:
        assert client.get("/health", headers={"X-Request-ID": bad}).headers["x-request-id"] != bad


def test_413_from_the_size_limiter_also_has_a_request_id_and_is_logged(json_logs):
    small = TestClient(observability.RequestIdMiddleware(limits.MaxBodySizeMiddleware(api.api_app, max_bytes=10)))
    r = small.post("/transcribe", files={"file": ("a.mp3", b"x" * 100)})
    assert r.status_code == 413 and r.headers["x-request-id"]
    assert any(e["request_id"] == r.headers["x-request-id"] and "413" in e["message"] for e in json_logs())


def test_successful_job_writes_one_summary_line_with_timings(json_logs, monkeypatch):
    monkeypatch.setattr(api, "transcribe_file", lambda *a, **k: _fake_result())
    r = client.post("/transcribe?model=tiny", files={"file": ("talk.mp3", b"data")})
    jobs = [e for e in json_logs() if e["message"] == "job finished"]
    assert len(jobs) == 1
    job = jobs[0]
    assert job["request_id"] == r.headers["x-request-id"] and job["level"] == "INFO"
    assert job["status"] == 200 and job["file"] == "talk.mp3" and job["model"] == "tiny" and job["bytes"] == 4
    assert job["audio_sec"] == 7.5 and job["processing_sec"] == 1.9 and job["rtf"] == 0.25 and job["total_sec"] >= 0


def _raise(exc):
    def f(*a, **k):
        raise exc
    return f


@pytest.mark.parametrize("filename,query,patch,status,level", [
    ("notes.txt", "", None, 415, "WARNING"),
    ("a.mp3", "?model=huge", None, 422, "WARNING"),
    ("a.mp3", "", _raise(errors.InvalidInputError("Not a readable audio file")), 422, "WARNING"),
    ("a.mp3", "", _raise(errors.ModelBusyError("busy")), 503, "WARNING"),
    ("a.mp3", "", _raise(RuntimeError("bug")), 500, "ERROR"),
])
def test_failed_jobs_also_write_one_summary_line(json_logs, monkeypatch, filename, query, patch, status, level):
    if patch:
        monkeypatch.setattr(api, "transcribe_file", patch)
    c = TestClient(api.app, raise_server_exceptions=False)
    r = c.post(f"/transcribe{query}", files={"file": (filename, b"data")})
    assert r.status_code == status
    jobs = [e for e in json_logs() if e["message"] == "job finished"]
    assert len(jobs) == 1 and jobs[0]["status"] == status and jobs[0]["level"] == level
    assert jobs[0]["request_id"] == r.headers["x-request-id"]
    if status != 500:
        assert jobs[0]["error"]


def test_full_server_503_is_logged(json_logs, monkeypatch):
    monkeypatch.setattr(api, "JOBS", _full_queue())
    client.post("/transcribe", files={"file": ("a.mp3", b"data")})
    assert [e["status"] for e in json_logs() if e["message"] == "job finished"] == [503]


def test_pipeline_logs_in_the_worker_thread_carry_the_request_id(json_logs, monkeypatch):

    def fake(*a, **k):
        pipeline.log.info("inside the worker thread")
        return _fake_result()
    monkeypatch.setattr(api, "transcribe_file", fake)
    r = client.post("/transcribe", files={"file": ("a.mp3", b"data")})
    inside = [e for e in json_logs() if e["message"] == "inside the worker thread"]
    assert inside and inside[0]["request_id"] == r.headers["x-request-id"]


def test_transcript_text_is_never_logged(json_logs, monkeypatch):
    monkeypatch.setattr(api, "transcribe_file", lambda *a, **k: _fake_result("my private medical details"))
    client.post("/transcribe", files={"file": ("a.mp3", b"data")})
    assert json_logs() and "private medical" not in json.dumps(json_logs())


def test_text_format_keeps_one_event_on_one_line():
    rec = logging.LogRecord("transcriber.api", logging.INFO, __file__, 1, "job finished", None, None)
    rec.fields = {"file": "evil\nINFO fake line.mp3", "status": 200}
    out = observability.TextFormatter().format(rec)
    assert "\n" not in out and 'file="evil\\nINFO fake line.mp3"' in out and "status=200" in out

    rec = logging.LogRecord("transcriber", logging.WARNING, __file__, 1, "ffprobe rejected %s: %s",
                            ("a.mp3", "[mp3] bad frames\n/tmp/a.mp3: Invalid data"), None)
    out = observability.TextFormatter().format(rec)
    assert "\n" not in out and "bad frames\\n/tmp/a.mp3" in out


def test_setup_logging_is_idempotent_and_reads_the_level():
    observability.setup_logging(level="debug")
    observability.setup_logging(level="warning")
    logger = logging.getLogger("transcriber")
    assert sum(isinstance(h, observability._AppHandler) for h in logger.handlers) == 1
    assert logger.level == logging.WARNING
    observability.setup_logging()


# ---------------------------------------------------------------- job queue: wait in line instead of 503
def _run(coro):
    return asyncio.run(coro)


def test_queue_serves_waiting_jobs_in_arrival_order():
    async def scenario():
        q = jobs.JobQueue(capacity=1, max_waiting=10, wait_timeout=5)
        order = []

        async def job(i):
            async with q.slot():
                order.append(i)
                await asyncio.sleep(0.01)

        tasks = []
        for i in range(6):
            tasks.append(asyncio.create_task(job(i)))
            await asyncio.sleep(0)               # make arrival order explicit
        await asyncio.gather(*tasks)
        return order, q.stats()

    order, stats = _run(scenario())
    assert order == [0, 1, 2, 3, 4, 5]
    assert stats["running"] == 0 and stats["waiting"] == 0


def test_queue_never_runs_more_than_capacity():
    async def scenario():
        q = jobs.JobQueue(capacity=2, max_waiting=20, wait_timeout=5)
        live = peak = 0

        async def job():
            nonlocal live, peak
            async with q.slot():
                live += 1
                peak = max(peak, live)
                await asyncio.sleep(0.01)
                live -= 1

        await asyncio.gather(*(job() for _ in range(12)))
        return peak, q.running

    assert _run(scenario()) == (2, 0)


def test_full_line_is_rejected_immediately_with_503():
    async def scenario():
        q = jobs.JobQueue(capacity=1, max_waiting=1, wait_timeout=5)
        release = asyncio.Event()

        async def holder():
            async with q.slot():
                await release.wait()

        tasks = [asyncio.create_task(holder()), asyncio.create_task(holder())]   # 1 running, 1 waiting
        await asyncio.sleep(0.01)
        with pytest.raises(HTTPException) as e:
            async with q.slot():
                pass
        release.set()
        await asyncio.gather(*tasks)
        return e.value, q.stats()

    err, stats = _run(scenario())
    assert err.status_code == 503 and err.headers["Retry-After"] == "30"
    assert stats["running"] == 0 and stats["waiting"] == 0


def test_waiting_too_long_gives_503_and_leaves_the_line():
    async def scenario():
        q = jobs.JobQueue(capacity=1, max_waiting=5, wait_timeout=0.05)
        q.running = 1                                 # slot held forever
        with pytest.raises(HTTPException) as e:
            async with q.slot():
                pass
        return e.value, q.stats()

    err, stats = _run(scenario())
    assert err.status_code == 503 and "in line" in err.detail
    assert stats["waiting"] == 0 and stats["running"] == 1


def test_client_that_disconnects_leaves_the_line_without_taking_a_slot():
    async def scenario():
        q = jobs.JobQueue(capacity=1, max_waiting=5, wait_timeout=30)
        q.running = 1

        async def gone():
            return True

        with pytest.raises(HTTPException) as e:
            async with q.slot(gone):
                pass
        return e.value, q.stats()

    err, stats = _run(scenario())
    assert err.status_code == 499 and stats["waiting"] == 0 and stats["running"] == 1


def test_slot_handed_over_at_the_moment_a_waiter_gives_up_is_passed_on_not_leaked():
    async def scenario():
        q = jobs.JobQueue(capacity=1, max_waiting=5, wait_timeout=30)
        await q._acquire(None)                        # A holds the only slot
        b = asyncio.create_task(q._acquire(None))     # B waits
        c = asyncio.create_task(q._acquire(None))     # C waits behind B
        await asyncio.sleep(0.01)
        q._release()                                  # A done: slot handed to B ...
        b.cancel()                                    # ... but B is cancelled before it resumes
        with pytest.raises(asyncio.CancelledError):
            await b
        await asyncio.wait_for(c, 1)                  # C must get the slot B gave up
        running_with_c = q.running
        q._release()
        return running_with_c, q.stats()

    running_with_c, stats = _run(scenario())
    assert running_with_c == 1 and stats["running"] == 0 and stats["waiting"] == 0


def _multipart_request(app, name="a.mp3", body=b"data"):
    """Drive one full /transcribe request through the ASGI app; returns (status, headers)."""
    boundary = "qqq"
    payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
               f"Content-Type: audio/mpeg\r\n\r\n").encode() + body + f"\r\n--{boundary}--\r\n".encode()
    sent, chunks = [], [payload]

    async def receive():
        if chunks:
            return {"type": "http.request", "body": chunks.pop(), "more_body": False}
        await asyncio.sleep(3600)                     # client stays connected until the response is sent

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/transcribe", "raw_path": b"/transcribe",
             "query_string": b"", "root_path": "", "client": ("test", 1), "server": ("test", 80),
             "headers": [(b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
                         (b"content-length", str(len(payload)).encode())]}

    async def run():
        await app(scope, receive, send)
        start = next(m for m in sent if m["type"] == "http.response.start")
        return start["status"], dict(start["headers"])
    return run()


def test_api_queues_a_second_request_instead_of_rejecting_it(json_logs, monkeypatch):
    """One slot: the second upload waits for the first, then succeeds. Before the queue it got 503."""
    import time as _time
    monkeypatch.setattr(api, "JOBS", jobs.JobQueue(capacity=1, max_waiting=5, wait_timeout=30))

    def slow(*a, **k):
        _time.sleep(0.3)
        return _fake_result()
    monkeypatch.setattr(api, "transcribe_file", slow)

    async def scenario():
        return await asyncio.gather(_multipart_request(api.app), _multipart_request(api.app))

    results = _run(scenario())
    assert [status for status, _ in results] == [200, 200]
    waits = sorted(e["queue_sec"] for e in json_logs() if e["message"] == "job finished")
    assert waits[0] == 0.0 and waits[1] >= 0.25           # the second one waited for the first
    assert api.JOBS.running == 0


def test_health_reports_queue_depth():
    jobs = client.get("/health").json()["jobs"]
    assert set(jobs) == {"running", "waiting", "capacity", "max_waiting"}
