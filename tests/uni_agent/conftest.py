"""Bootstrap uni-agent logging during test collection."""

import logging
from collections.abc import Iterator

import pytest

from uni_agent.logging.session import _ensure_process_logging

_ensure_process_logging()


@pytest.fixture
def caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture the namespace that intentionally does not propagate to root."""
    logger = logging.getLogger("uni_agent")
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
