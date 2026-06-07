"""Test-log parsers for SWE-rebench-V2 (Python subset).

SWE-rebench-V2 is language-agnostic, but we currently only process the Python
slice of the dataset, and every Python instance is graded with the single
``parse_log_pytest`` parser. We therefore keep just that parser here (vendored
verbatim from the official harness,
https://github.com/SWE-rebench/SWE-rebench-V2 ``lib/agent/log_parsers.py``)
plus the ``TestStatus`` enum.

``NAME_TO_PARSER`` is the extension point: the ``swe_rebench_v2`` reward spec
resolves ``install_config.log_parser`` through it. To enable another language,
copy its parser(s) from the upstream file and register them here -- nothing else
in the reward spec needs to change.
"""

from enum import Enum


class TestStatus(str, Enum):
    """Inlined from the official ``lib/agent/swe_constants.py``."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


def parse_log_pytest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if any(line.startswith(x.value) for x in TestStatus):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


# Maps install_config.log_parser -> parser fn. Python-only for now; add other
# languages' parsers here (see module docstring) to extend coverage.
NAME_TO_PARSER = {
    "parse_log_pytest": parse_log_pytest,
}
