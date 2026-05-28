"""Regression test for brace-bearing error messages in loguru-style logging.

`AgentInteraction` previously passed exception reprs and LLM-generated text
directly into the *template* arg of `self.logger.error(...)` via f-strings.
Loguru runs `.format()` on the final rendered string, so any unbalanced
'{' / '}' in the substituted content (typical of regex reprs, JSON, dict
literals, code snippets) makes the log call itself raise
`ValueError("Single '}' encountered in format string")` — which then
cascades into the rollout-worker outer except handlers (also f-string-based)
and kills the worker.

The fix is to use the positional-arg pattern: `logger.error("{}", msg)`.
The template field is the literal `"{}"`, which is always safe, and the
brace-prone content rides in the positional argument and is NOT re-parsed
by `.format()`.

This file covers the contract that pattern relies on.
"""

from __future__ import annotations

import pytest
from loguru import logger

# Real-world brace-bearing strings we have actually seen in production
# rollout failures on Qwen3-235B SWE-bench training (2026-05).
_BRACE_MESSAGES = [
    # regex repr from a config validation failure
    "ValueError: searcher_re must be a compiled re: re.compile('\\{action\\}')",
    # LLM output snippet with JSON
    'Fail to parse: {"thought": "I will edit',  # unbalanced — worst case
    # exception repr containing a dict literal
    "ConfigError: got {'tool': 'bash', 'args': {'cmd': 'ls'",
    # bash heredoc preview
    "Model Output: ```bash\ncat <<EOF\n{anything}",
    # Python f-string in code output
    "RuntimeError: f'hello {name}' is invalid here",
]


@pytest.mark.parametrize("brace_msg", _BRACE_MESSAGES)
def test_safe_template_does_not_raise_on_brace_bearing_message(brace_msg: str) -> None:
    """`logger.error("{}", msg)` must NOT raise when `msg` contains
    unbalanced or stray '{' / '}'. This is the pattern the patched
    handlers in `uni_agent/interaction/interaction.py` use."""
    # Must not raise:
    logger.error("{}", brace_msg)
    logger.critical("{}", brace_msg)


def test_safe_template_with_opt_exception_does_not_raise() -> None:
    """The `run()` outer except handler uses
    `logger.opt(exception=True).critical("{}", msg)` to also dump the
    stack trace. Verify that chained `.opt(exception=True)` does not
    break the brace-safety contract."""
    brace_msg = "[step1] unknown_error: KeyError: '{routed_experts}' missing"
    try:
        raise KeyError("'{routed_experts}'")
    except KeyError:
        logger.opt(exception=True).critical("{}", brace_msg)


def test_safe_template_with_bound_logger_does_not_raise() -> None:
    """`AgentInteraction.logger` is a `logger.bind(name=..., run_id=...)`
    so the safe-template contract must hold across bound loggers too."""
    bound = logger.bind(name="test-interaction", run_id="brace-safety-check")
    bound.error("{}", "ValueError: bad regex re.compile('{action}')")
    bound.critical("{}", '{"json": "{nested"')
