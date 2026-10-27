"""The nginx limit must follow MAX_UPLOAD_MB, never a hard-coded number."""
import importlib.util
import re
from pathlib import Path

import pytest

from transcriber.config import MIB, Settings

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
_spec = importlib.util.spec_from_file_location("render_nginx", DEPLOY / "render_nginx.py")
assert _spec and _spec.loader
render_nginx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_nginx)


@pytest.mark.parametrize("mb", ["1", "500", "800", "2048"])
def test_nginx_limit_matches_app_limit(tmp_path, mb):
    settings = Settings.from_env({"MAX_UPLOAD_MB": mb})
    conf = render_nginx.render(tmp_path / "nginx.conf", settings).read_text()
    size = re.search(r"client_max_body_size (\d+)m;", conf)
    assert size, conf
    assert int(size.group(1)) * MIB == settings.max_request_bytes      # exactly aligned with the app cap
    assert settings.max_request_bytes == (int(mb) + 1) * MIB
    assert "{{" not in conf


def test_template_has_no_hardcoded_size():
    tpl = (DEPLOY / "nginx.conf.template").read_text()
    assert re.search(r"client_max_body_size\s+\d", tpl) is None


def test_render_reads_the_environment_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "64")
    conf = render_nginx.render(tmp_path / "nginx.conf").read_text()
    assert "client_max_body_size 65m;" in conf


# ---------------------------------------------------------------- settings validation
def test_settings_defaults():
    s = Settings.from_env({})
    assert (s.max_upload_bytes, s.max_concurrent_jobs, s.max_queued_jobs, s.queue_wait_sec) == (500 * MIB, 2, 10, 300)
    assert (s.model_cache_size, s.model_wait_sec, s.log_level, s.log_format) == (1, 300, "INFO", "text")


@pytest.mark.parametrize("env,message", [
    ({"MAX_UPLOAD_MB": "lots"}, "not a valid int"),
    ({"MAX_UPLOAD_MB": "0"}, "at least 1"),
    ({"MAX_CONCURRENT_JOBS": "0"}, "at least 1"),
    ({"MAX_QUEUED_JOBS": "-1"}, "at least 0"),
    ({"QUEUE_WAIT_SEC": "soon"}, "not a valid float"),
    ({"LOG_FORMAT": "xml"}, "'text' or 'json'"),
])
def test_bad_settings_fail_at_startup_with_a_clear_message(env, message):
    with pytest.raises(ValueError, match=message):
        Settings.from_env(env)
