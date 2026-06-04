"""Vendored from SWE-rebench/SWE-rebench-V2 (lib/agent/swe_constants.py).

Kept verbatim so the vendored ``log_parsers`` produce identical statuses to the
upstream evaluator. Do not edit by hand; refresh from upstream when bumping.
"""

from enum import Enum


class TestStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
