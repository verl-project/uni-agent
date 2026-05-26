"""Plan E environment contract.

This test intentionally does not fail on missing optional host capabilities.
It prints a compact capability report so Plan E logs show whether a run
actually exercised GPU, vLLM, verl, and Mooncake.
"""
from __future__ import annotations

import importlib


def _probe_module(name: str) -> tuple[bool, str, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return False, type(exc).__name__, str(exc)
    return True, str(getattr(module, "__version__", "unknown")), str(
        getattr(module, "__file__", "")
    )


def test_plan_e_environment_contract(capsys):
    report: list[str] = []
    for name in ["torch", "ray", "vllm", "verl", "omegaconf", "mooncake.engine"]:
        ok, version, detail = _probe_module(name)
        report.append(f"{name}: {'OK' if ok else 'MISSING'} {version} {detail}")

    torch_ok, _, _ = _probe_module("torch")
    if torch_ok:
        import torch

        report.append(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        report.append(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            report.append(f"torch.cuda.device_names: {names}")

    print("\n".join(report))
    captured = capsys.readouterr()
    assert "torch:" in captured.out
