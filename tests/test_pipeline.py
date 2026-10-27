"""Tests run without downloading a model: the Whisper model is mocked. Run: pytest -q"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from transcriber import config, engine, errors, media, merge, pipeline, result


def make_audio(path: Path, seconds: int, fmt_args=()):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                    *fmt_args, str(path)], check=True)
    return path


class FakeModel:
    """Returns one segment every 4 s of each chunk, like a real model would (chunk-local timestamps)."""
    def transcribe(self, path, **kw):
        dur = media.probe_duration(Path(path))
        segs = [NS(start=t, end=min(t + 4, dur), text=f"  words at {t}  ", avg_logprob=-0.3, no_speech_prob=0.05)
                for t in range(0, int(dur), 4)]
        return iter(segs), NS(language="en", language_probability=0.99)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def fake_use(*a, **k):
        yield FakeModel()

    monkeypatch.setattr(engine, "use_model", fake_use)


@pytest.mark.parametrize("ext,args", [
    (".wav", ()), (".mp3", ()), (".flac", ()),
    (".ogg", ["-c:a", "vorbis", "-strict", "-2", "-ac", "2"]),   # built-in encoder; libvorbis isn't in every build
    (".m4a", ["-c:a", "aac"]),
])
def test_accepts_multiple_formats(tmp_path, ext, args):
    f = make_audio(tmp_path / f"a{ext}", 6, args)
    r = pipeline.transcribe_file(f)
    assert r.duration_sec == pytest.approx(6, abs=0.2)
    assert r.segments and r.segments[0].start == 0


def test_long_audio_is_chunked_and_timestamps_are_global_and_monotonic(tmp_path):
    f = make_audio(tmp_path / "long.wav", 25)
    r = pipeline.transcribe_file(f, chunk_sec=10)
    assert r.stats["chunks"] == 3
    starts = [s.start for s in r.segments]
    assert starts == sorted(starts)                       # monotonic
    pairs = zip(r.segments, r.segments[1:], strict=False)
    assert all(b.start >= a.end - 1e-6 for a, b in pairs)    # no overlap duplicates
    assert r.segments[-1].end == pytest.approx(25, abs=0.5)  # reaches the end -> offsets applied


def test_text_is_cleaned():
    segs, dropped = merge.merge_and_clean([(0, [
        NS(start=0, end=1, text="  hello   world ", avg_logprob=-0.2, no_speech_prob=0.1),
        NS(start=1, end=2, text="", avg_logprob=-0.2, no_speech_prob=0.1),
        NS(start=2, end=3, text="noise", avg_logprob=-3.0, no_speech_prob=0.1),
    ])])
    assert [s.text for s in segs] == ["hello world"] and dropped == 2


def test_rejects_non_audio(tmp_path):
    bad = tmp_path / "x.mp3"
    bad.write_text("not audio")
    with pytest.raises(ValueError):
        pipeline.transcribe_file(bad)


def _fake_probe(monkeypatch, payload):
    stdout = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(media.subprocess, "run", lambda *a, **k: NS(returncode=0, stdout=stdout, stderr=""))


@pytest.mark.parametrize("duration", ["N/A", "nan", "inf", "-1", "0", "abc", None])
def test_probe_rejects_malformed_duration(monkeypatch, tmp_path, duration):
    fmt = {} if duration is None else {"duration": duration}
    _fake_probe(monkeypatch, {"streams": [{"codec_type": "audio"}], "format": fmt})
    with pytest.raises(ValueError, match="invalid duration"):
        media.probe_duration(tmp_path / "x.mp3")


def test_probe_rejects_unparseable_output(monkeypatch, tmp_path):
    _fake_probe(monkeypatch, "not json")
    with pytest.raises(ValueError, match="Not a readable audio file"):
        media.probe_duration(tmp_path / "x.mp3")


@pytest.mark.parametrize("payload", [
    [], "null", 42, "\"text\"",
    {"streams": "audio", "format": {"duration": "5"}},
    {"streams": [{"codec_type": "audio"}], "format": ["5"]},
])
def test_probe_rejects_wrong_json_shape(monkeypatch, tmp_path, payload):
    _fake_probe(monkeypatch, payload)
    with pytest.raises(ValueError, match="Not a readable audio file"):
        media.probe_duration(tmp_path / "x.mp3")


def test_probe_ignores_non_dict_stream_entries(monkeypatch, tmp_path):
    _fake_probe(monkeypatch, {"streams": ["junk", None], "format": {"duration": "5"}})
    with pytest.raises(ValueError, match="No audio stream"):
        media.probe_duration(tmp_path / "x.mp3")


def test_video_without_audio_stream_is_rejected(tmp_path):
    """Real file: a valid MP4 with only a video stream."""
    f = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
                    "-pix_fmt", "yuv420p", str(f)], check=True)
    with pytest.raises(ValueError, match="No audio stream"):
        media.probe_duration(f)


def test_video_with_audio_stream_is_accepted(tmp_path):
    f = tmp_path / "talk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
                    "-f", "lavfi", "-i", "sine=duration=2", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(f)], check=True)
    assert media.probe_duration(f) == pytest.approx(2, abs=0.2)


def _seg(start, end, text="x", avg_logprob=-0.2, no_speech_prob=0.1):
    return NS(start=start, end=end, text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)


def test_overlap_duplicate_is_dropped_and_times_stay_valid():
    """Chunk 2 starts at 600 s; its first segment repeats the end of chunk 1 and ends just
    before chunk 1's last segment did. Previously this produced start > end and a duplicate."""
    segs, dropped = merge.merge_and_clean([
        (0, [_seg(598, 601.5, "hello world")]),
        (600, [_seg(0, 1.4, "world"), _seg(1.4, 4, "next bit")]),
    ])
    assert [s.text for s in segs] == ["hello world", "next bit"] and dropped == 0
    assert (segs[1].start, segs[1].end) == (601.5, 604)          # clamped to previous end
    assert [s.id for s in segs] == [0, 1]


def test_segment_mostly_after_overlap_is_kept():
    segs, _ = merge.merge_and_clean([
        (0, [_seg(598, 601.0, "a")]),
        (600, [_seg(0.5, 3, "b")]),                                # 600.5-603, midpoint 601.75 > 601
    ])
    assert [(s.start, s.end, s.text) for s in segs] == [(598, 601.0, "a"), (601.0, 603, "b")]


def test_overlap_rule_does_not_apply_inside_one_chunk():
    """Whisper timestamps that overlap inside a chunk are clamped, never silently deleted."""
    segs, _ = merge.merge_and_clean([(0, [_seg(0, 5, "a"), _seg(3, 4, "b"), _seg(4, 8, "c")])])
    assert [s.text for s in segs] == ["a", "b", "c"]
    assert [(s.start, s.end) for s in segs] == [(0, 5), (5, 5), (5, 8)]


@pytest.mark.parametrize("raw", [
    [(0, [_seg(0, 3), _seg(2.9, 6), _seg(5, 9.99)]), (10, [_seg(-0.5, 0.1), _seg(0, 1.9), _seg(1.8, 4)])],
    [(0, [_seg(0, 10.4)]), (10, [_seg(0, 0.3), _seg(0.3, 0.38), _seg(0.38, 2)])],
    [(0, [_seg(8.123, 9.876)]), (10, [_seg(0.004, 0.006)])],
])
def test_output_times_are_ordered_and_non_negative_length(raw):
    segs, _ = merge.merge_and_clean(raw)
    assert all(s.start <= s.end for s in segs)
    assert all(b.start >= a.end for a, b in zip(segs, segs[1:], strict=False))


def test_confident_speech_with_high_no_speech_prob_is_kept():
    segs, dropped = merge.merge_and_clean([(0, [
        _seg(0, 2, "yes", avg_logprob=-0.1, no_speech_prob=0.85),   # confident text -> keep
        _seg(2, 4, "uh", avg_logprob=-1.2, no_speech_prob=0.85),    # silence-like -> drop
        _seg(4, 6, "ok", avg_logprob=-1.2, no_speech_prob=0.5),     # unsure but speech -> keep
    ])])
    assert [s.text for s in segs] == ["yes", "ok"] and dropped == 1


def test_confidence_thresholds_are_configurable():
    raw = [(0, [_seg(0, 1, "a", avg_logprob=-0.9)])]
    assert merge.merge_and_clean(raw)[1] == 0
    assert merge.merge_and_clean(raw, min_logprob=-0.5)[1] == 1


def test_srt_format():
    r = result.TranscriptionResult("f", "en", 1.0, 70, [result.Segment(0, 61.5, 63.25, "hi", -0.1, 0.0)])
    assert "00:01:01,500 --> 00:01:03,250" in result.to_srt(r)


@pytest.mark.parametrize("t,expected", [
    (1.15, "00:00:01,150"), (4.35, "00:00:04,350"), (0.29, "00:00:00,290"),   # float truncation gave ,149 / ,349 / ,289
    (59.9999, "00:01:00,000"), (3599.9996, "01:00:00,000"), (0, "00:00:00,000"),
    (36000.5, "10:00:00,500"),
])
def test_srt_timestamps_use_whole_milliseconds(t, expected):
    r = result.TranscriptionResult("f", "en", 1.0, t, [result.Segment(0, t, t, "x", 0, 0)])
    assert f"{expected} --> {expected}" in result.to_srt(r)


def test_srt_blocks_are_numbered_and_blank_line_separated():
    r = result.TranscriptionResult("f", "en", 1.0, 5, [result.Segment(0, 0, 1, "a", 0, 0),
                                                         result.Segment(1, 1, 2, "b", 0, 0)])
    assert result.to_srt(r) == "1\n00:00:00,000 --> 00:00:01,000\na\n\n2\n00:00:01,000 --> 00:00:02,000\nb\n"


# ---------------------------------------------------------------- external tool timeouts
def _hang(*a, **k):
    raise subprocess.TimeoutExpired(cmd=a[0], timeout=k.get("timeout"))


def test_every_tool_call_has_a_timeout(monkeypatch, tmp_path):
    seen = []
    real = subprocess.run

    def spy(cmd, *a, **k):
        seen.append((cmd[0], k.get("timeout")))
        return real(cmd, *a, **k)

    f = make_audio(tmp_path / "a.wav", 25)              # test setup, before spying
    monkeypatch.setattr(media.subprocess, "run", spy)
    pipeline.transcribe_file(f, chunk_sec=10)            # probe + normalize + 3 splits
    assert {c for c, _ in seen} == {"ffprobe", "ffmpeg"}
    assert all(t is not None and t > 0 for _, t in seen), seen


def test_probe_timeout_becomes_value_error(monkeypatch, tmp_path):
    monkeypatch.setattr(media.subprocess, "run", _hang)
    with pytest.raises(ValueError, match="timed out"):
        media.probe_duration(tmp_path / "x.mp3")


def test_normalize_timeout_becomes_value_error(monkeypatch, tmp_path):
    monkeypatch.setattr(media.subprocess, "run", _hang)
    with pytest.raises(ValueError, match="timed out"):
        media.normalize_audio(tmp_path / "x.mp3", tmp_path, duration=10)


def test_normalize_timeout_scales_with_duration(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(media.subprocess, "run",
                        lambda *a, **k: seen.append(k["timeout"]) or NS(returncode=0, stdout="", stderr=""))
    media.normalize_audio(tmp_path / "x.mp3", tmp_path, duration=3600)
    assert seen == [media.NORMALIZE_BASE_SEC + media.NORMALIZE_PER_AUDIO_SEC * 3600]


def test_real_hung_process_is_killed(monkeypatch):
    """End-to-end: a genuinely stuck child process is killed at the timeout."""
    import time
    t0 = time.time()
    with pytest.raises(ValueError, match="timed out"):
        media.run_tool(["sleep", "30"], timeout=1, what="Sleep")
    assert time.time() - t0 < 5


# ---------------------------------------------------------------- model / language validation and model cache
@pytest.mark.parametrize("bad", ["huge", "../../etc", "/tmp/model", "someone/evil-model", ""])
def test_unknown_or_path_like_model_rejected(bad):
    with pytest.raises(ValueError, match="Unknown model"):
        engine.validate_options(bad, None)


def test_bad_language_rejected_good_ones_pass():
    with pytest.raises(ValueError, match="Unsupported language"):
        engine.validate_options("small", "xx")
    for code in ("en", "ar", "ur", None):
        engine.validate_options("small", code)



# ---------------------------------------------------------------- device / compute type, .en models, bounded cache
def _fake_ct2(monkeypatch, gpus):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: gpus))


def test_auto_picks_gpu_and_float16_when_cuda_available(monkeypatch):
    _fake_ct2(monkeypatch, gpus=1)
    assert engine.resolve_device("auto") == ("cuda", "float16")


def test_auto_picks_cpu_and_int8_without_gpu(monkeypatch):
    _fake_ct2(monkeypatch, gpus=0)
    assert engine.resolve_device("auto") == ("cpu", "int8")


def test_explicit_devices_and_bad_device():
    assert engine.resolve_device("cuda") == ("cuda", "float16")
    assert engine.resolve_device("cpu") == ("cpu", "int8")
    with pytest.raises(ValueError):
        engine.resolve_device("tpu")


@pytest.mark.parametrize("model", ["tiny.en", "base.en", "small.en", "medium.en"])
def test_english_only_models_reject_other_languages(model):
    with pytest.raises(ValueError, match="English-only"):
        engine.validate_options(model, "ar")
    engine.validate_options(model, "en")
    engine.validate_options(model, None)


def test_default_cache_keeps_one_model():
    assert config.settings.model_cache_size == 1 and engine._POOL.capacity == 1


# ---------------------------------------------------------------- ModelPool: hard bound on resident models
import threading
import time as _time


class Loader:
    """Fake model loader that tracks how many models are alive at once."""
    def __init__(self, delay=0.0):
        self.delay, self.loads, self.alive, self.max_alive = delay, [], 0, 0
        self.lock = threading.Lock()

    def __call__(self, size, device, compute):
        loader = self

        class M:
            def __init__(self):
                with loader.lock:
                    loader.alive += 1
                    loader.max_alive = max(loader.max_alive, loader.alive)
                self.args = (size, device, compute)

            def __del__(self):
                with loader.lock:
                    loader.alive -= 1

        _time.sleep(self.delay)
        self.loads.append(size)
        return M()


def test_pool_rejects_unknown_model_before_loading():
    ld = Loader()
    with pytest.raises(ValueError, match="Unknown model"):
        with engine.ModelPool(1, loader=ld).use("someone/evil-model", "cpu"):
            pass
    assert ld.loads == []


def test_pool_passes_resolved_device_and_compute(monkeypatch):
    _fake_ct2(monkeypatch, gpus=2)
    ld = Loader()
    with engine.ModelPool(1, loader=ld).use("small", "auto") as m:
        assert m.args == ("small", "cuda", "float16")      # not int8 on a GPU


def test_same_model_loaded_once_under_concurrency():
    ld = Loader(delay=0.2)
    pool = engine.ModelPool(1, loader=ld)
    got, errors = [], []

    def worker():
        try:
            with pool.use("tiny", "cpu") as m:
                got.append(id(m))
                _time.sleep(0.05)
        except Exception as e:
            errors.append(e)

    ts = [threading.Thread(target=worker) for _ in range(10)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errors == [] and len(got) == 10
    assert ld.loads == ["tiny"] and len(set(got)) == 1


def test_in_use_model_is_never_evicted_hard_bound():
    """Request A is transcribing with 'tiny'; request B wants 'base'. B must WAIT until A is
    done, so at most ONE model is ever alive (the case a plain LRU cache gets wrong)."""
    ld = Loader()
    pool = engine.ModelPool(1, loader=ld, wait_timeout=5)
    a_inside, a_release = threading.Event(), threading.Event()
    order = []

    def request_a():
        with pool.use("tiny", "cpu"):
            order.append("A start")
            a_inside.set()
            a_release.wait(2)
            order.append("A end")

    def request_b():
        a_inside.wait(2)
        with pool.use("base", "cpu"):
            order.append("B start")

    ta, tb = threading.Thread(target=request_a), threading.Thread(target=request_b)
    ta.start()
    tb.start()
    _time.sleep(0.3)
    assert order == ["A start"] and ld.loads == ["tiny"]   # B is waiting, not loading
    a_release.set()
    ta.join()
    tb.join()
    assert order == ["A start", "A end", "B start"]
    assert ld.max_alive == 1                               # never two models in memory
    assert pool.resident() == 1


def test_busy_pool_times_out_with_model_busy_error():
    ld = Loader()
    pool = engine.ModelPool(1, loader=ld, wait_timeout=0.3)
    with pool.use("tiny", "cpu"):
        t0 = _time.monotonic()
        with pytest.raises(errors.ModelBusyError):
            with pool.use("base", "cpu"):
                pass
        assert 0.25 < _time.monotonic() - t0 < 2
    assert ld.loads == ["tiny"]                            # base was never loaded


def test_idle_models_are_evicted_lru():
    ld = Loader()
    pool = engine.ModelPool(2, loader=ld)
    for m in ["tiny", "base", "tiny", "small", "medium"]:
        with pool.use(m, "cpu"):
            assert pool.resident() <= 2
    assert ld.loads == ["tiny", "base", "small", "medium"]  # tiny reused, not reloaded
    assert ld.max_alive <= 2


def test_failed_load_frees_the_slot():
    calls = []

    def flaky(size, device, compute):
        calls.append(size)
        if len(calls) == 1:
            raise RuntimeError("download failed")
        return object()

    pool = engine.ModelPool(1, loader=flaky, wait_timeout=1)
    with pytest.raises(RuntimeError):
        with pool.use("tiny", "cpu"):
            pass
    assert pool.resident() == 0
    with pool.use("tiny", "cpu"):                          # retry works, slot was released
        pass


# ---------------------------------------------------------------- client errors vs server errors
def test_bad_client_input_raises_invalid_input_error(tmp_path):
    bad = tmp_path / "x.mp3"
    bad.write_text("not audio")
    with pytest.raises(errors.InvalidInputError):
        pipeline.transcribe_file(bad)
    with pytest.raises(errors.InvalidInputError):
        engine.validate_options("someone/evil", None)


def test_split_failure_is_a_server_error_not_client_error(monkeypatch, tmp_path):
    """make_chunks reads our own normalized WAV; if ffmpeg fails there, it is not the client's fault."""
    monkeypatch.setattr(media, "run_tool", lambda *a, **k: NS(returncode=1, stdout="", stderr="disk full"))
    with pytest.raises(RuntimeError) as e:
        media.make_chunks(tmp_path / "n.wav", duration=1300, out_dir=tmp_path, chunk_sec=600)
    assert not isinstance(e.value, errors.InvalidInputError)


# ---------------------------------------------------------------- untrusted input: ffmpeg must not follow playlists
def _playlist_to(path: Path, segment: Path) -> Path:
    path.write_text(f"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\n{segment}\n#EXT-X-ENDLIST\n")
    return path


@pytest.mark.parametrize("name", ["list.m3u8", "disguised.mp3"])
def test_playlist_pointing_at_local_file_is_rejected(tmp_path, name):
    """Without the demuxer whitelist, ffprobe follows the playlist and reads the server's own file
    (reproduced with the .m3u8 name on ffmpeg 9 and the .mp3 name on Debian's ffmpeg 5.1)."""
    secret = make_audio(tmp_path / "secret.mp3", 3)
    with pytest.raises(errors.InvalidInputError):
        pipeline.transcribe_file(_playlist_to(tmp_path / name, secret))


@pytest.mark.parametrize("ext,args", [
    (".wav", ()), (".mp3", ()), (".flac", ()), (".m4a", ["-c:a", "aac"]), (".mp4", ["-c:a", "aac"]),
    (".aac", ["-c:a", "aac"]), (".ogg", ["-c:a", "vorbis", "-strict", "-2", "-ac", "2"]),
    (".opus", ["-c:a", "opus", "-strict", "-2", "-ar", "48000"]),
    (".webm", ["-c:a", "opus", "-strict", "-2", "-ar", "48000"]), (".wma", ["-c:a", "wmav2"]),
])
def test_every_supported_format_passes_the_demuxer_whitelist(tmp_path, ext, args):
    f = make_audio(tmp_path / f"a{ext}", 2, args)
    assert media.probe_duration(f) == pytest.approx(2, abs=0.2)


def test_ffmpeg_and_ffprobe_both_get_the_safe_input_args(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(media, "run_tool", lambda cmd, **k: calls.append(cmd) or NS(
        returncode=0, stdout='{"streams":[{"codec_type":"audio"}],"format":{"duration":"5"}}', stderr=""))
    media.probe_duration(tmp_path / "a.mp3")
    media.normalize_audio(tmp_path / "a.mp3", tmp_path, 5)
    for cmd in calls:
        i = cmd.index("-i") if "-i" in cmd else len(cmd) - 1      # input options must come before the input
        assert cmd[:i].count("-format_whitelist") == 1 and cmd[:i].count("-protocol_whitelist") == 1
