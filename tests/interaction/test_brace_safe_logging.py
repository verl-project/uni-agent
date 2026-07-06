"""Regression tests for the stdlib per-sample logging package.

Covers the two failure modes the routing has to survive: messages containing braces
or stray ``%`` (which must be logged verbatim, never format-crash) and correct
routing to the right run's file via both the ambient and explicit run_id paths.
"""

from __future__ import annotations

import logging

import pytest

from uni_agent.logging import add_file_handler, cleanup_handlers, get_logger, sample_logging

_TRICKY_MESSAGES = [
    "ValueError: searcher_re must be a compiled re: re.compile('{action}')",
    'Fail to parse: {"thought": "I will edit',
    "ConfigError: got {'tool': 'bash', 'args': {'cmd': 'ls'",
    "Model Output: ```bash\ncat <<EOF\n{anything}",
    "stray percent %s %d with no args",
]


@pytest.mark.parametrize("msg", _TRICKY_MESSAGES)
def test_ambient_run_id_routes_verbatim(tmp_path, msg: str) -> None:
    log_path = tmp_path / "ambient.log"
    with sample_logging("ambient-run", log_path):
        logging.getLogger("uni_agent.test").error(msg)
    assert msg in log_path.read_text()


def test_explicit_run_id_routes_verbatim(tmp_path) -> None:
    log_path = tmp_path / "explicit.log"
    add_file_handler(log_path, "explicit-run")
    try:
        get_logger("agent-loop", "explicit-run").error("boom {routed_experts} %s")
    finally:
        cleanup_handlers("explicit-run")
    assert "boom {routed_experts} %s" in log_path.read_text()


def test_records_route_to_their_own_run_file(tmp_path) -> None:
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    with sample_logging("run-a", a):
        logging.getLogger("uni_agent.test").warning("only in A")
    with sample_logging("run-b", b):
        logging.getLogger("uni_agent.test").warning("only in B")
    assert "only in A" in a.read_text() and "only in B" not in a.read_text()
    assert "only in B" in b.read_text() and "only in A" not in b.read_text()
