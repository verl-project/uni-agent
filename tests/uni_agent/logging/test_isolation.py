"""Regression tests for the namespace-isolation redesign (design doc
``docs/design/logging-isolation.md`` §5.2).

These tests pin the isolation contract: uni-agent logging lives entirely on
the ``uni_agent`` mount point, the host's root logger is never read or written,
and a session bootstrap never overrides an earlier explicit configuration.
"""

from __future__ import annotations

import logging

import pytest

from uni_agent.logging import handlers as logging_handlers
from uni_agent.logging import sample_logging
from uni_agent.logging.handlers import _dispatch, _mount
from uni_agent.logging.session import _ensure_process_logging, _setup_console_logging


@pytest.fixture(autouse=True)
def _isolated_logging_state():
    """Same snapshot contract as test_session.py, but scoped to what these
    isolation tests touch: root, mount, console global, ready flag."""
    root = logging.getLogger()
    saved_root_handlers = root.handlers[:]
    saved_root_level = root.level
    mount = _mount()
    saved_mount_handlers = mount.handlers[:]
    saved_mount_level = mount.level
    saved_mount_propagate = mount.propagate
    saved_console = logging_handlers._console_handler
    import uni_agent.logging.session as session_module

    saved_ready = session_module._process_logging_ready
    root.handlers = []
    mount.handlers = []
    mount.setLevel(logging.NOTSET)
    mount.propagate = True
    logging_handlers._console_handler = None
    # Reset the ready flag too: a True saved value only means a *previous*
    # module (in another test file) ran the bootstrap in this same process —
    # leaving it set here would skip the bootstrap and leave the mount
    # unconfigured for every test below.
    session_module._process_logging_ready = False
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for handler in saved_root_handlers:
        root.addHandler(handler)
    root.setLevel(saved_root_level)
    for handler in mount.handlers[:]:
        mount.removeHandler(handler)
    for handler in saved_mount_handlers:
        mount.addHandler(handler)
    mount.setLevel(saved_mount_level)
    mount.propagate = saved_mount_propagate
    logging_handlers._console_handler = saved_console
    session_module._process_logging_ready = saved_ready
    with _dispatch._lock:
        _dispatch._log_ids.clear()


@pytest.mark.cpu
@pytest.mark.level0
def test_host_root_untouched_by_setup_and_session(capsys):
    """Regression: the old code cleared root handlers and pinned root to INFO.
    Now neither _setup_console_logging nor the session bootstrap reads or writes
    the host's root logger — its handler keeps collecting the host's own
    records, and its level survives."""
    root = logging.getLogger()
    seen: list[logging.LogRecord] = []
    host_handler = logging.Handler()
    host_handler.emit = seen.append  # type: ignore[method-assign]
    host_handler.setLevel(logging.DEBUG)
    root.addHandler(host_handler)
    root.setLevel(logging.DEBUG)

    _setup_console_logging("DEBUG")
    with sample_logging("iso-host", log_path=None):
        logging.getLogger("host.app").warning("host-record")
        logging.getLogger("uni_agent.test").warning("namespace-record")

    assert host_handler in root.handlers
    assert root.level == logging.DEBUG
    assert any(getattr(record, "msg", "") == "host-record" for record in seen)


@pytest.mark.cpu
@pytest.mark.level0
def test_propagate_false_keeps_records_out_of_host_root(capsys):
    """Records on the uni_agent namespace stop at the mount (propagate=False):
    even a host root handler at DEBUG never receives them, so host collectors
    see only the host's own records."""
    root = logging.getLogger()
    seen: list[logging.LogRecord] = []
    host_handler = logging.Handler()
    host_handler.emit = seen.append  # type: ignore[method-assign]
    host_handler.setLevel(logging.DEBUG)
    root.addHandler(host_handler)
    root.setLevel(logging.DEBUG)

    _setup_console_logging("DEBUG")
    logging.getLogger("uni_agent.test").debug("namespace-only")

    assert "namespace-only" in capsys.readouterr().out
    assert not any(getattr(record, "message", "") == "namespace-only" for record in seen)


@pytest.mark.cpu
@pytest.mark.level0
def test_session_bootstrap_does_not_override_explicit_level(capsys, monkeypatch):
    """Regression: _ensure_process_logging used to re-run the default config
    with no explicit level, so the first sample_logging entry reset an earlier
    _setup_console_logging("DEBUG") back to INFO and DEBUG went silent. Now the
    bootstrap only fills in when the mount was never configured. DEBUG_MODE
    opens the console filter for scoped records."""
    monkeypatch.setenv("DEBUG_MODE", "1")
    _setup_console_logging("DEBUG")

    with sample_logging("iso-debug", log_path=None):
        assert _mount().level == logging.DEBUG
        logging.getLogger("uni_agent.test").debug("debug-inside-session")
        out_inside = capsys.readouterr().out
    logging.getLogger("uni_agent.test").debug("debug-after-session")

    out = out_inside + capsys.readouterr().out
    assert _mount().level == logging.DEBUG
    assert "debug-inside-session" in out
    assert "debug-after-session" in out


@pytest.mark.cpu
@pytest.mark.level0
def test_ensure_bootstrap_skipped_when_mount_configured():
    """Once the mount has handlers or a level, _ensure_process_logging adds only
    the dispatch handler — it must not re-apply the default INFO or reinstall
    anything else (dual-path idempotency)."""
    _setup_console_logging("WARNING")
    mount = _mount()
    handlers_before = mount.handlers[:]
    import uni_agent.logging.session as session_module

    session_module._process_logging_ready = False

    _ensure_process_logging()

    assert mount.level == logging.WARNING
    assert mount.handlers[: len(handlers_before)] == handlers_before
    assert _dispatch in mount.handlers


@pytest.mark.cpu
@pytest.mark.level0
def test_session_without_setup_defaults_to_info_and_isolation():
    """The default bootstrap path (sample_logging without any prior setup) still
    lands on mount INFO with propagate off — equivalent to the pre-refactor
    default, and still isolated from the host root."""
    with sample_logging("iso-default", log_path=None):
        pass

    mount = _mount()
    assert mount.level == logging.INFO
    assert mount.propagate is False
    assert _dispatch in mount.handlers


@pytest.mark.cpu
@pytest.mark.level0
def test_smoke_host_config_and_uni_agent_compose(capsys, monkeypatch):
    """End-to-end composition scenario (design §5.1 smoke test): a host process
    configures logging first, then uni-agent is set up and used both outside
    and inside a session. All four uni-agent record classes reach stdout (with
    DEBUG_MODE opening scoped DEBUG), while the host keeps its own records,
    its level, its handler, and its format untouched — and never sees a
    uni-agent record."""
    monkeypatch.setenv("DEBUG_MODE", "1")

    root = logging.getLogger()
    host_records: list[logging.LogRecord] = []
    host_handler = logging.Handler()
    host_handler.emit = host_records.append  # type: ignore[method-assign]
    host_handler.setLevel(logging.DEBUG)
    host_handler.setFormatter(logging.Formatter("HOST %(levelname)s %(name)s: %(message)s"))
    root.addHandler(host_handler)
    root.setLevel(logging.DEBUG)

    logging.getLogger("host.driver").info("host-INFO-alive")

    _setup_console_logging("DEBUG")

    logger = logging.getLogger("uni_agent.router.demo")
    logger.debug("A-debug-outside")
    with sample_logging("smoke-rid", log_path=None):
        logger.debug("B-debug-inside")
        logger.warning("C-warn-inside")
    logger.debug("D-debug-after")

    out = capsys.readouterr().out
    assert "A-debug-outside" in out
    assert "B-debug-inside" in out
    assert "C-warn-inside" in out
    assert "D-debug-after" in out

    assert [getattr(record, "msg", "") for record in host_records] == ["host-INFO-alive"]
    assert host_handler in root.handlers
    assert root.level == logging.DEBUG
    assert "HOST %(levelname)s" in host_handler.formatter._fmt
    assert _mount().propagate is False
