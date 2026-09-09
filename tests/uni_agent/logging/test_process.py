from __future__ import annotations

import logging
import time

import pytest

from uni_agent.logging import handlers as logging_handlers
from uni_agent.logging import sample_logging, setup_console_logging
from uni_agent.logging.context import _QUIET_LOGGERS
from uni_agent.logging.handlers import _ConsoleFilter, _dispatch
from uni_agent.logging.session import _process_logging_ready


@pytest.fixture(autouse=True)
def _isolated_logging_state():
    """Snapshot every piece of logging global state these tests touch — root
    handlers/level, the console-sink global, third-party logger levels, the
    per-process ready flag, and dispatch registrations — and restore it all
    afterward so nothing leaks into other tests in the same pytest process."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_console = logging_handlers._console_handler
    saved_quiet_levels = {name: logging.getLogger(name).level for name in _QUIET_LOGGERS}
    saved_ready = _process_logging_ready
    root.handlers = []
    logging_handlers._console_handler = None
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)
    logging_handlers._console_handler = saved_console
    for name, level in saved_quiet_levels.items():
        logging.getLogger(name).setLevel(level)
    import uni_agent.logging.session as session_module

    session_module._process_logging_ready = saved_ready
    with _dispatch._lock:
        _dispatch._log_ids.clear()


def _console_filter(handler: logging.Handler) -> _ConsoleFilter:
    filters = [f for f in handler.filters if isinstance(f, _ConsoleFilter)]
    assert filters, "console handler must carry the _ConsoleFilter"
    return filters[0]


@pytest.mark.cpu
@pytest.mark.level0
def test_setup_console_logging_installs_console_sink():
    """First call installs the console sink: handler on root at INFO with the
    shared formatter, and root level set to INFO."""
    setup_console_logging()

    root = logging.getLogger()
    console = logging_handlers._console_handler
    assert console is not None
    assert console in root.handlers
    assert console.level == logging.INFO
    assert root.level == logging.INFO
    _console_filter(console)
    assert console.formatter is logging_handlers._formatter


@pytest.mark.cpu
@pytest.mark.level0
def test_setup_console_logging_emits_to_stdout(capsys):
    """Behavior: a record outside any LogContext reaches stdout through the sink."""
    setup_console_logging()

    logging.getLogger("uni_agent.test").info("hello-console")

    assert "hello-console" in capsys.readouterr().out


@pytest.mark.cpu
@pytest.mark.level0
def test_setup_console_logging_is_idempotent(capsys):
    """Repeated calls install no second handler and re-apply the level to the
    existing sink — the record is emitted exactly once, at the newest level."""
    setup_console_logging()
    console = logging_handlers._console_handler

    setup_console_logging("DEBUG")

    root = logging.getLogger()
    assert root.handlers.count(console) == 1
    assert console.level == logging.DEBUG

    logging.getLogger("uni_agent.test").debug("hello-once")
    assert capsys.readouterr().out.count("hello-once") == 1


@pytest.mark.cpu
@pytest.mark.level0
def test_setup_console_logging_preserves_host_handlers():
    """Handlers installed by the embedding process survive setup and repeated calls."""
    root = logging.getLogger()
    host_handler = logging.Handler()
    root.addHandler(host_handler)

    setup_console_logging()
    setup_console_logging()

    assert host_handler in root.handlers


@pytest.mark.cpu
@pytest.mark.level0
def test_setup_console_logging_quiets_third_party_loggers():
    """Every _QUIET_LOGGERS entry (httpx, modal, ...) is pinned to WARNING."""
    noisy = logging.getLogger("httpx")
    noisy.setLevel(logging.DEBUG)

    setup_console_logging()

    for name in _QUIET_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


@pytest.mark.cpu
@pytest.mark.level0
def test_setup_console_logging_explicit_level_wins(capsys, monkeypatch):
    """The explicit level beats the env and accepts a logging-level int: with
    UNI_AGENT_LOG_LEVEL=ERROR, passing logging.DEBUG still lets a DEBUG record
    through to stdout."""
    monkeypatch.setenv("UNI_AGENT_LOG_LEVEL", "ERROR")
    setup_console_logging(logging.DEBUG)

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert logging_handlers._console_handler.level == logging.DEBUG

    logging.getLogger("uni_agent.test").debug("debug-visible")
    assert "debug-visible" in capsys.readouterr().out


@pytest.mark.cpu
@pytest.mark.level0
def test_setup_console_logging_reads_env_level(monkeypatch):
    """With no explicit level, UNI_AGENT_LOG_LEVEL is honored."""
    monkeypatch.setenv("UNI_AGENT_LOG_LEVEL", "WARNING")
    setup_console_logging()

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert logging_handlers._console_handler.level == logging.WARNING


@pytest.mark.cpu
@pytest.mark.level0
def test_setup_console_logging_invalid_level_falls_back_to_info(capsys, monkeypatch):
    """An invalid level name falls back to INFO and warns on stderr."""
    monkeypatch.setenv("UNI_AGENT_LOG_LEVEL", "NOT_A_LEVEL")
    setup_console_logging()

    root = logging.getLogger()
    assert root.level == logging.INFO
    assert "invalid log level" in capsys.readouterr().err


@pytest.mark.cpu
@pytest.mark.level0
def test_setup_console_logging_composes_with_sample_logging(capsys, tmp_path):
    """Composition: inside a sample_logging block, scoped INFO is filtered from the
    console (readable next to a progress bar), WARNING passes, and both reach the
    run's log file."""
    setup_console_logging()
    log_file = tmp_path / "run.log"

    with sample_logging("compose-1", log_path=log_file):
        scoped = logging.getLogger("uni_agent.compose")
        scoped.info("scoped-info-quiet")
        scoped.warning("scoped-warn-loud")

    out = capsys.readouterr().out
    assert "scoped-info-quiet" not in out
    assert "scoped-warn-loud" in out

    content = _poll_file(log_file)
    assert "scoped-info-quiet" in content
    assert "scoped-warn-loud" in content


def _poll_file(path, timeout: float = 5.0) -> str:
    """The file writer is a background thread; wait for the close on context exit
    to land before asserting on the file contents."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        if "scoped-info-quiet" in content and "scoped-warn-loud" in content:
            return content
        time.sleep(0.05)
    return path.read_text(encoding="utf-8") if path.exists() else ""
