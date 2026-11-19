"""The E2B prebuild example must build exactly the templates ``E2BSandbox.start()`` looks up."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from uni_agent.sandbox import SandboxConfig
from uni_agent.tasks import TaskConfigResolver

_SCRIPT = Path(__file__).resolve().parents[3] / "examples/quickstart/sandbox/prebuild_e2b_templates.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prebuild_e2b_templates", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(image):
    return {"extra_info": {"tools_kwargs": {"task": {"name": "swe_bench", "sandbox": {"image": image}}}}}


@pytest.mark.cpu
@pytest.mark.level0
def test_prebuild_matches_the_templates_start_uses(monkeypatch):
    import uni_agent.sandbox.e2b as e2b_module

    resolver = TaskConfigResolver(
        defaults_by_name={
            "swe_bench": {
                "sandbox": {
                    "provider": "modal",  # overridden to e2b, as SANDBOX_PROVIDER=e2b does in the runners
                    "image_map": [{"from": "swebench/**", "to": "registry.example.com/swebench/**"}],
                    "sandbox_kwargs": {"cpu_count": 4, "memory_mb": 8192},
                }
            }
        }
    )
    samples = [_sample("swebench/a:latest"), _sample("swebench/b:latest"), _sample("swebench/a:latest")]

    collected = _load_script().collect_templates(samples, resolver)

    started = []

    async def ensure(*args):
        started.append(args)
        return "tpl"

    monkeypatch.setattr(e2b_module, "_ensure_template", ensure)

    class _AsyncSandbox:
        @classmethod
        async def create(cls, **kwargs):
            return SimpleNamespace(sandbox_id="sb-1")

    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(AsyncSandbox=_AsyncSandbox))
    for sample in samples:
        config = resolver.resolve(sample["extra_info"]["tools_kwargs"]["task"])
        sandbox_config = SandboxConfig.model_validate({**config["sandbox"], "provider": "e2b"})
        asyncio.run(e2b_module.E2BSandbox.from_config(sandbox_config).start())

    assert collected == [
        ("registry.example.com/swebench/a:latest", 4, 8192, None),
        ("registry.example.com/swebench/b:latest", 4, 8192, None),
    ]
    assert sorted(set(started)) == collected


@pytest.mark.cpu
@pytest.mark.level0
def test_prebuild_skips_samples_with_a_ready_template():
    resolver = TaskConfigResolver(
        defaults_by_name={"swe_bench": {"sandbox": {"provider": "e2b", "sandbox_kwargs": {"template": "custom"}}}}
    )

    assert _load_script().collect_templates([_sample("swebench/a:latest")], resolver) == []


@pytest.mark.cpu
@pytest.mark.level0
def test_prebuild_reports_failures_and_respects_concurrency(monkeypatch, capsys):
    import uni_agent.sandbox.e2b as e2b_module

    running, peak = 0, 0

    async def ensure(image, cpu_count, memory_mb, proxy):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0)
        running -= 1
        if image == "bad":
            raise RuntimeError("build failed")
        return f"tpl-{image}"

    monkeypatch.setattr(e2b_module, "_ensure_template", ensure)
    specs = [(image, 2, 4096, None) for image in ("a", "bad", "b", "c")]

    failed = asyncio.run(_load_script().build_all(specs, concurrency=2))

    assert failed == 1
    assert peak == 2
    out = capsys.readouterr().out
    assert "failed  bad: RuntimeError: build failed" in out and out.count("ready   tpl-") == 3
