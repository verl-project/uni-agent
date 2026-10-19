import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.inference.parallel_infer_verl import init_config


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "engine,device,text_only",
    [("vllm", "cuda", True), ("sglang", "cuda", False), ("sglang", "cuda", True), ("sglang", "npu", True)],
)
def test_backend_config(monkeypatch, engine, device, text_only):
    from verl.plugin import platform

    monkeypatch.setattr(platform, "get_platform", lambda: SimpleNamespace(device_name=device))
    args = Namespace(
        temperature=0.4,
        top_p=0.85,
        top_k=20,
        n=1,
        nnodes=1,
        n_gpus_per_node=8,
        model_path="/model",
        engine=engine,
        language_model_only=text_only,
        enforce_eager=True,
        disable_thinking=True,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.9,
        enable_rollout_routing_replay=False,
        response_length=1024,
        tool_parser="qwen3_coder",
        gateway_count=1,
        concurrency=1,
        allowed_request_sampling_param_keys=[],
        task_config="/task.yaml",
        log_dir="/logs",
    )
    config = init_config(args, served_model_name="policy")
    assert config.data.apply_chat_template_kwargs.enable_thinking is False
    rollout = config.actor_rollout_ref.rollout

    assert rollout.name == engine
    assert rollout.enforce_eager is True
    if engine == "vllm":
        assert rollout.skip_tokenizer_init is True
        assert rollout.engine_kwargs.vllm.kv_cache_dtype == "auto"
        assert rollout.engine_kwargs.vllm.language_model_only is True
        assert "json_model_override_args" not in rollout.engine_kwargs.vllm
        return

    assert rollout.skip_tokenizer_init is False
    kwargs = rollout.engine_kwargs.sglang
    assert kwargs.random_seed == rollout.seed
    assert "language_model_only" not in kwargs
    if text_only:
        assert json.loads(kwargs.json_model_override_args) == {"language_model_only": True}
    else:
        assert "json_model_override_args" not in kwargs
    if device == "npu":
        assert (kwargs.device, kwargs.attention_backend, kwargs.enable_memory_saver) == ("npu", "ascend", False)
    else:
        assert not {"device", "attention_backend", "enable_memory_saver"}.intersection(kwargs)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("engine", [None, "sglang"], ids=["default-vllm", "explicit-sglang"])
def test_mini_swe_launcher(tmp_path, monkeypatch, engine):
    repo_root = Path(__file__).resolve().parents[3]
    ray = tmp_path / "ray"
    ray.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
    ray.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("REPO_ROOT", str(repo_root))
    monkeypatch.setenv("OPENYUANRONG_SERVER_ADDRESS", "https://sandbox.example")
    monkeypatch.setenv("OPENYUANRONG_TOKEN", "synthetic-test-token")
    for name in ("ENGINE", "CUDA_VISIBLE_DEVICES", "TENSOR_PARALLEL_SIZE", "N_GPUS_PER_NODE", "GATEWAY_COUNT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGUAGE_MODEL_ONLY", str(int(engine == "sglang")))
    monkeypatch.setenv("ENFORCE_EAGER", str(int(engine == "sglang")))

    result = subprocess.run(
        ["bash", "examples/mini_swe_agent/run_infer_mini_swe_agent.sh"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env={**os.environ, **({"ENGINE": engine} if engine else {})},
    )
    argv = json.loads(result.stdout)
    assert argv[argv.index("--engine") + 1] == (engine or "vllm")
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--n-gpus-per-node") + 1] == "2"
    assert argv[argv.index("--gateway-count") + 1] == "4"
    assert ("--language-model-only" in argv) is (engine == "sglang")
    assert ("--enforce-eager" in argv) is (engine == "sglang")
    runtime = json.loads(argv[argv.index("--runtime-env-json") + 1])["env_vars"]
    assert runtime["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert not any(key.startswith("UNI_AGENT_SGLANG_") for key in runtime)
