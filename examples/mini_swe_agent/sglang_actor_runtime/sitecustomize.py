"""Keep optional training-only CUDA packages out of SGLang actor subprocesses."""

from __future__ import annotations

import importlib.abc
import sys

_BLOCKED_OPTIONAL_MODULES = ("megatron", "transformer_engine")


class _BlockOptionalTrainingPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in _BLOCKED_OPTIONAL_MODULES):
            raise ModuleNotFoundError(f"optional training package blocked in SGLang actor: {fullname}")
        return None


sys.meta_path.insert(0, _BlockOptionalTrainingPackages())
