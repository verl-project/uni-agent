"""Regression tests for brace-safe loguru logging."""

from __future__ import annotations

import pytest
from loguru import logger

_BRACE_MESSAGES = [
    "ValueError: searcher_re must be a compiled re: re.compile('\\{action\\}')",
    'Fail to parse: {"thought": "I will edit',
    "ConfigError: got {'tool': 'bash', 'args': {'cmd': 'ls'",
    "Model Output: ```bash\ncat <<EOF\n{anything}",
    "RuntimeError: f'hello {name}' is invalid here",
]


@pytest.mark.parametrize("brace_msg", _BRACE_MESSAGES)
def test_safe_template_does_not_raise_on_brace_bearing_message(brace_msg: str) -> None:
    logger.error("{}", brace_msg)
    logger.critical("{}", brace_msg)


def test_safe_template_with_opt_exception_does_not_raise() -> None:
    brace_msg = "[step1] unknown_error: KeyError: '{routed_experts}' missing"
    try:
        raise KeyError("'{routed_experts}'")
    except KeyError:
        logger.opt(exception=True).critical("{}", brace_msg)


def test_safe_template_with_bound_logger_does_not_raise() -> None:
    bound = logger.bind(name="test-interaction", run_id="brace-safety-check")
    bound.error("{}", "ValueError: bad regex re.compile('{action}')")
    bound.critical("{}", '{"json": "{nested"')
