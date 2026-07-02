"""Root conftest for the llm_router test suite.

Registers the marker vocabulary so pytest recognises them and does not emit
``PytestUnknownMarkWarning``.  Two orthogonal dimensions:

  type     — ``ut`` (pure unit tests) / ``st`` (system/integration) / ``e2e`` (end-to-end)
  resource — ``cpu`` (no GPU) / ``gpu`` (needs a real vLLM + GPU)

Select tests with ``-m``, e.g.::

    pytest -m "ut and cpu"        # unit tests
    pytest -m "st and cpu"        # Ray actor integration
    pytest -m "st and gpu"        # collector integration (conftest vLLM)
    pytest -m "e2e and gpu"       # end-to-end (run_infer.sh)
"""

from __future__ import annotations


def pytest_configure(config):  # type: ignore[no-untyped-def]
    for marker, desc in (
        ("ut", "pure unit test — no Ray, no GPU, no external services"),
        ("st", "system / integration test — exercises real subsystems"),
        ("e2e", "end-to-end test via run_infer.sh; standalone vLLM (no conftest sharing)"),
        ("cpu", "runs on CPU (no GPU required)"),
        ("gpu", "needs a real GPU + vLLM service"),
    ):
        config.addinivalue_line("markers", f"{marker}: {desc}")
