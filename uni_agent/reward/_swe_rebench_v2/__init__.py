"""Vendored SWE-rebench-V2 evaluation helpers.

The log parsers and ``TestStatus`` are copied verbatim from
https://github.com/SWE-rebench/SWE-rebench-V2 (``lib/agent/``) so that our reward
grades trajectories with the exact same, language-agnostic parsers the official
evaluator uses. Only the intra-package import path was adapted.
"""

from uni_agent.reward._swe_rebench_v2.log_parsers import NAME_TO_PARSER
from uni_agent.reward._swe_rebench_v2.swe_constants import TestStatus

__all__ = ["NAME_TO_PARSER", "TestStatus"]
