from __future__ import annotations

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import hydra
import pytest
from omegaconf import OmegaConf, open_dict

pytest.importorskip("ray")

from examples.agent_aware_router import run_infer
from examples.agent_aware_router.run_infer import init_config as init_router_config
from examples.inference import parallel_infer_verl
from examples.inference.parallel_infer_verl import init_config


def _parse_inference_args(monkeypatch, *argv):
    parse_args = ArgumentParser.parse_args
    captured = None

    class ParsedArgs(Exception):
        pass

    def capture(parser):
        nonlocal captured
        captured = parse_args(parser, ["--task-config", "/unused.yaml", *argv])

        def parsed():
            raise ParsedArgs

        monkeypatch.setattr(parallel_infer_verl.ray, "init", parsed)
        return captured

    monkeypatch.setattr(ArgumentParser, "parse_args", capture)
    with pytest.raises(ParsedArgs):
        parallel_infer_verl.main()
    return captured


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("override", [False, True])
def test_prompt_length_cli_can_override_invalid_environment(monkeypatch, capsys, override):
    monkeypatch.setenv("PROMPT_LENGTH", "invalid")
    if override:
        args = _parse_inference_args(monkeypatch, "--prompt-length", "2048")
        assert args.prompt_length == 2048
    else:
        with pytest.raises(SystemExit) as exc:
            _parse_inference_args(monkeypatch)
        assert exc.value.code == 2
        assert "--prompt-length: invalid int value" in capsys.readouterr().err


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("value", [0, -1, -65537])
def test_prompt_length_rejects_non_positive_values(monkeypatch, capsys, value):
    with pytest.raises(SystemExit) as exc:
        _parse_inference_args(monkeypatch, "--prompt-length", str(value))
    assert exc.value.code == 2
    assert "must be >= 1" in capsys.readouterr().err


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("value", [0, -1])
def test_prompt_length_rejects_non_positive_environment_values(monkeypatch, capsys, value):
    monkeypatch.setenv("PROMPT_LENGTH", str(value))
    with pytest.raises(SystemExit) as exc:
        _parse_inference_args(monkeypatch)
    assert exc.value.code == 2
    assert "must be >= 1" in capsys.readouterr().err


@pytest.mark.cpu
@pytest.mark.level0
def test_shared_inference_preserves_defaults_and_accepts_recipe_overrides(monkeypatch):
    monkeypatch.setenv("LIMIT", "7")
    with monkeypatch.context() as context:
        args = _parse_inference_args(context)
    assert args.limit is None
    assert args.concurrency == int(os.getenv("GLOBAL_CONCURRENCY", 128))
    assert (args.n_gpus_per_node, args.tensor_parallel_size, args.gateway_count) == (8, 4, 4)

    args = _parse_inference_args(
        monkeypatch,
        "--limit",
        "1",
        "--concurrency",
        "1",
        "--n-gpus-per-node",
        "1",
        "--tensor-parallel-size",
        "1",
        "--gateway-count",
        "1",
    )
    assert (args.limit, args.concurrency, args.n_gpus_per_node, args.tensor_parallel_size, args.gateway_count) == (
        1,
        1,
        1,
        1,
        1,
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("configured", [None, False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_language_model_only_is_opt_in_and_preserves_config(monkeypatch, configured, enabled):
    args = _parse_inference_args(monkeypatch, *(["--language-model-only"] if enabled else []))
    compose = hydra.compose

    def compose_with_engine_setting(*args, **kwargs):
        config = compose(*args, **kwargs)
        engine = config.actor_rollout_ref.rollout.engine_kwargs.vllm
        with open_dict(engine):
            if configured is None:
                engine.pop("language_model_only", None)
            else:
                engine.language_model_only = configured
        return config

    monkeypatch.setattr(hydra, "compose", compose_with_engine_setting)
    config = init_config(args, served_model_name="policy")
    engine = config.actor_rollout_ref.rollout.engine_kwargs.vllm
    if enabled:
        assert engine.language_model_only is True
    elif configured is None:
        assert "language_model_only" not in engine
    else:
        assert OmegaConf.select(engine, "language_model_only") is configured


@pytest.fixture
def router_args():
    return Namespace(
        temperature=0.4,
        top_p=0.85,
        top_k=20,
        allowed_request_sampling_param_keys=["temperature", "top_p", "top_k"],
        n=2,
        nnodes=1,
        n_gpus_per_node=1,
        model_path="/tmp/test-model",
        engine="vllm",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        enable_rollout_routing_replay=False,
        tool_parser="qwen3_coder",
        gateway_count=1,
        concurrency=4,
        task_config="/not-loaded-until-task-preparation.yaml",
        log_dir="/tmp/test-inference",
        language_model_only=False,
        num_workers=1,
        max_model_len=32768,
        max_num_seqs=16,
        mooncake_config_path=None,
        mooncake_save_decode_cache=False,
        device="gpu",
        kv_events=False,
        router_config_path="uni_agent/agent_aware_router/configs/agent_aware_router.yaml",
        simulated_runner_fqn=None,
        load_threshold=0.5,
        prompt_length=2048,
        response_length=8192,
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "entrypoint,simulated_runner",
    [(init_config, None), (init_router_config, None), (init_router_config, "example.simulated_runner")],
)
def test_inference_sampling_uses_run_options_and_preserves_length_configuration(
    entrypoint, simulated_runner, router_args
):
    args = router_args
    args.simulated_runner_fqn = simulated_runner
    config = entrypoint(
        args,
        served_model_name="policy",
    )

    rollout = config.actor_rollout_ref.rollout
    for sampling in (rollout, rollout.val_kwargs):
        assert sampling.temperature == 0.4
        assert sampling.top_p == 0.85
        assert sampling.top_k == 20
    assert rollout.val_kwargs.do_sample is True
    expected_prompt_length = args.prompt_length
    expected_response_length = args.response_length
    assert rollout.prompt_length == config.data.max_prompt_length == expected_prompt_length
    assert rollout.response_length == config.data.max_response_length == expected_response_length
    if entrypoint is init_config:
        assert rollout.max_model_len == expected_prompt_length + expected_response_length
    assert rollout.custom.agent_framework.allowed_request_sampling_param_keys == ["temperature", "top_p", "top_k"]
    task_runner = rollout.custom.agent_framework.agent_runners.task
    if simulated_runner:
        assert task_runner.runner_fqn == simulated_runner
        assert not task_runner.runner_kwargs
    else:
        assert task_runner.runner_kwargs.task_config_path == args.task_config


@pytest.mark.cpu
@pytest.mark.level0
def test_inference_language_model_only_wires_vllm_and_rejects_other_engines():
    args = Namespace(
        temperature=0.4,
        top_p=0.85,
        top_k=20,
        allowed_request_sampling_param_keys=["temperature", "top_p", "top_k"],
        n=1,
        nnodes=1,
        n_gpus_per_node=1,
        model_path="/tmp/test-model",
        engine="vllm",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        enable_rollout_routing_replay=False,
        language_model_only=True,
        tool_parser="qwen3_coder",
        gateway_count=1,
        concurrency=1,
        task_config="/not-loaded-until-task-preparation.yaml",
        log_dir="/tmp/test-inference",
        num_workers=1,
        max_model_len=32768,
        max_num_seqs=16,
        enable_mooncake=False,
        kv_events=False,
        router_config_path="uni_agent/agent_aware_router/configs/agent_aware_router.yaml",
        simulated_runner_fqn=None,
        load_threshold=0.5,
        prompt_length=2048,
        response_length=8192,
    )

    config = init_config(args, served_model_name="policy")
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only is True

    args.engine = "unsupported"
    with pytest.raises(ValueError, match="only with --engine vllm or sglang"):
        init_config(args, served_model_name="policy")


@pytest.mark.cpu
@pytest.mark.level0
def test_shared_inference_cli_preserves_existing_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["parallel_infer_verl.py", "--task-config", "unused.yaml"])
    monkeypatch.delenv("LIMIT", raising=False)
    parse_args = parallel_infer_verl.argparse.ArgumentParser.parse_args

    class ParsedDefaults(Exception):
        pass

    def check_defaults(parser):
        args = parse_args(parser)
        assert args.limit is None
        assert args.concurrency == int(os.getenv("GLOBAL_CONCURRENCY", 128))
        assert args.n_gpus_per_node == 8
        assert args.tensor_parallel_size == 4
        assert args.gateway_count == 4
        assert args.language_model_only is False
        assert args.disable_thinking is False
        raise ParsedDefaults

    monkeypatch.setattr(parallel_infer_verl.argparse.ArgumentParser, "parse_args", check_defaults)
    with pytest.raises(ParsedDefaults):
        parallel_infer_verl.main()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "enabled,save_decode,device",
    [(False, False, "gpu"), (True, False, "gpu"), (True, True, "gpu"), (True, False, "ascend")],
)
def test_mooncake_engine_config(router_args, enabled, save_decode, device):
    router_args.mooncake_config_path = "/tmp/store.json" if enabled else None
    router_args.mooncake_save_decode_cache = save_decode
    router_args.device = device
    rollout = init_router_config(router_args, served_model_name="policy").actor_rollout_ref.rollout
    assert rollout.enable_prefix_caching
    if not enabled:
        assert "kv_transfer_config" not in rollout.engine_kwargs.vllm
    else:
        assert rollout.engine_kwargs.vllm.kv_transfer_config == {
            "kv_connector": "MooncakeStoreConnector" if device == "gpu" else "MooncakeConnectorStoreV1",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"save_decode_cache": save_decode} if device == "gpu" else {},
        }


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("address", [None, "auto"])
@pytest.mark.parametrize("enabled", [False, True])
def test_mooncake_ray_job_env(router_args, tmp_path, monkeypatch, address, enabled):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "store.json").write_text("{}")
    router_args.mooncake_config_path = "store.json" if enabled else None
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "stale-driver-path")
    monkeypatch.setenv("MOONCAKE_PREFERRED_SEGMENT", "127.0.0.1:50053")
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    if address:
        monkeypatch.setenv("RAY_ADDRESS", address)
    else:
        monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setattr(run_infer.ray, "is_initialized", lambda: False)
    init = Mock()
    monkeypatch.setattr(run_infer.ray, "init", init)
    env_vars = run_infer._prepare_mooncake_env(router_args)
    run_infer._init_ray(env_vars)
    kwargs = init.call_args.kwargs
    if enabled:
        env = kwargs["runtime_env"]["env_vars"]
        assert env["MOONCAKE_CONFIG_PATH"] == str(tmp_path / "store.json")
        assert router_args.mooncake_config_path == "store.json"
        assert env["MOONCAKE_PREFERRED_SEGMENT"] == "127.0.0.1:50053"
        assert env["PYTHONHASHSEED"] == "42"
    else:
        assert "runtime_env" not in kwargs
    if address:
        assert kwargs["address"] == "auto"
        assert "_system_config" not in kwargs
    else:
        assert "idle_worker_killing_time_threshold_ms" in kwargs["_system_config"]
    assert os.environ["MOONCAKE_CONFIG_PATH"] == "stale-driver-path"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("contents", [None, "not json", "[]", "null"])
def test_invalid_mooncake_config_fails_before_ray(router_args, tmp_path, monkeypatch, contents):
    path = tmp_path / "store.json"
    router_args.mooncake_config_path = str(path)
    if contents is not None:
        path.write_text(contents)
    init = Mock()
    monkeypatch.setattr(run_infer.ray, "init", init)
    monkeypatch.setattr(run_infer, "_parse_args", lambda: router_args)
    with pytest.raises(ValueError, match="Mooncake"):
        run_infer.main()
    init.assert_not_called()
    assert router_args.mooncake_config_path == str(path)


@pytest.mark.cpu
@pytest.mark.level0
def test_mooncake_expands_home(router_args, tmp_path, monkeypatch):
    # Patch expanduser rather than changing the process HOME.
    path = tmp_path / "store.json"
    path.write_text("{}")
    router_args.mooncake_config_path = "~/store.json"
    monkeypatch.setattr(run_infer.Path, "expanduser", lambda self: path)
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "old")
    assert run_infer._prepare_mooncake_env(router_args)["MOONCAKE_CONFIG_PATH"] == str(path)
    assert router_args.mooncake_config_path == "~/store.json"
    assert os.environ["MOONCAKE_CONFIG_PATH"] == "old"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("matches", [False, True])
def test_already_initialized_ray_requires_matching_env(router_args, tmp_path, monkeypatch, matches):
    path = tmp_path / "store.json"
    path.write_text("{}")
    router_args.mooncake_config_path = str(path)
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "old")
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    env = run_infer._prepare_mooncake_env(router_args)
    monkeypatch.setattr(run_infer.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        run_infer.ray,
        "get_runtime_context",
        lambda: Namespace(runtime_env={"env_vars": env if matches else {}}),
    )
    init = Mock()
    monkeypatch.setattr(run_infer.ray, "init", init)
    if matches:
        run_infer._init_ray(env)
    else:
        with pytest.raises(ValueError, match="already initialized") as error:
            run_infer._init_ray(env)
        assert f"MOONCAKE_CONFIG_PATH: requested={str(path)!r}, active=<unset>" in str(error.value)
        assert "PYTHONHASHSEED: requested='42', active=<unset>" in str(error.value)
    init.assert_not_called()
    assert os.environ["MOONCAKE_CONFIG_PATH"] == "old"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("enabled", [False, True])
def test_report_records_submitted_mooncake_config(router_args, tmp_path, monkeypatch, enabled):
    router_args.mooncake_config_path = "/tmp/store.json" if enabled else None
    router_args.mooncake_save_decode_cache = enabled
    router_args.result_path = str(tmp_path / "result.json")
    router_args.data_path = "/tmp/data.parquet"
    build_config = Mock(wraps=run_infer._mooncake_transfer_config)
    monkeypatch.setattr(run_infer, "_mooncake_transfer_config", build_config)
    config = init_router_config(router_args, served_model_name="policy")
    snapshot = run_infer._mooncake_metadata(config, router_args.mooncake_config_path)
    router_args.device = "ascend"
    router_args.mooncake_save_decode_cache = True
    router_args.mooncake_config_path = "/changed.json"
    if enabled:
        transfer = config.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config
        transfer.kv_connector_extra_config.save_decode_cache = False
    run_infer._report(
        {"scores": [1.0], "per_uid": {"task": [1.0]}, "uid_status": {"task": "finished"}},
        wall=1.0,
        num_prompts=1,
        n=1,
        args=router_args,
        served_model_name="policy",
        mooncake_metadata=snapshot,
    )
    metadata = json.loads((tmp_path / "result.json").read_text())["mooncake"]
    assert metadata["enabled"] == enabled
    if enabled:
        assert metadata["kv_transfer_config"]["kv_connector"] == "MooncakeStoreConnector"
        assert metadata["kv_transfer_config"]["kv_connector_extra_config"] == {"save_decode_cache": True}
        assert metadata["config_path"] == "/tmp/store.json"
    else:
        assert metadata["config_path"] is None and metadata["kv_transfer_config"] is None
    build_config.assert_called_once()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "options,path,save_decode",
    [
        ([], None, False),
        (["--mooncake-config-path", "store.json"], "store.json", False),
        (["--mooncake-config-path", "store.json", "--mooncake-save-decode-cache"], "store.json", True),
    ],
)
def test_mooncake_cli_uses_optional_path_and_store_true(options, path, save_decode, monkeypatch):
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "/stale-driver.json")
    args = run_infer._parse_args(["--task-config", "unused.yaml", *options])
    assert args.mooncake_config_path == path
    assert args.mooncake_save_decode_cache == save_decode
    assert not hasattr(args, "enable_mooncake")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("option", ["--enable-mooncake", "--no-mooncake-save-decode-cache"])
def test_removed_mooncake_cli_options_are_rejected(option, capsys):
    with pytest.raises(SystemExit) as error:
        run_infer._parse_args(["--task-config", "unused.yaml", option])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "engine,device,enabled,save_decode,message",
    [
        ("vllm", "gpu", False, True, "requires --mooncake-config-path"),
        ("vllm", "ascend", True, True, "only supported.*gpu"),
        ("sglang", "gpu", True, False, "requires --engine vllm"),
    ],
)
def test_invalid_mooncake_options_fail_before_ray(
    router_args, tmp_path, monkeypatch, engine, device, enabled, save_decode, message
):
    path = tmp_path / "store.json"
    path.write_text("{}")
    router_args.mooncake_config_path = str(path) if enabled else None
    router_args.engine, router_args.device = engine, device
    router_args.mooncake_save_decode_cache = save_decode
    monkeypatch.setattr(run_infer, "_parse_args", lambda: router_args)
    init = Mock()
    monkeypatch.setattr(run_infer.ray, "init", init)
    with pytest.raises(ValueError, match=message):
        run_infer.main()
    init.assert_not_called()


@pytest.mark.cpu
@pytest.mark.level0
def test_ray_env_error_reports_only_mismatching_variables(monkeypatch):
    env = {"MOONCAKE_CONFIG_PATH": "/store.json", "PYTHONHASHSEED": "42"}
    active = {**env, "PYTHONHASHSEED": "123", "UNRELATED": "not-in-error"}
    monkeypatch.setattr(run_infer.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(run_infer.ray, "get_runtime_context", lambda: Namespace(runtime_env={"env_vars": active}))
    with pytest.raises(ValueError, match="already initialized") as error:
        run_infer._init_ray(env)
    text = str(error.value)
    assert "PYTHONHASHSEED: requested='42', active='123'" in text
    assert "MOONCAKE_CONFIG_PATH" not in text and "UNRELATED" not in text


@pytest.mark.cpu
@pytest.mark.level0
def test_mooncake_forwarded_keys_are_logged_without_values(monkeypatch, caplog):
    monkeypatch.setattr(run_infer.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(run_infer.ray, "init", Mock())
    module_logger = logging.getLogger(run_infer.__name__)
    monkeypatch.setattr(module_logger, "propagate", False)
    module_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger=run_infer.__name__):
            run_infer._init_ray({"MOONCAKE_CONFIG_PATH": "/private-config-value.json", "PYTHONHASHSEED": "42"})
    finally:
        module_logger.removeHandler(caplog.handler)
    messages = [record.getMessage() for record in caplog.records if record.name == run_infer.__name__]
    assert len(messages) == 1
    assert "Mooncake Ray job environment keys: MOONCAKE_CONFIG_PATH, PYTHONHASHSEED" in messages[0]
    assert "/private-config-value.json" not in messages[0]


@pytest.mark.cpu
@pytest.mark.level0
def test_main_passes_resolved_path_and_original_connector_snapshot(router_args, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "store.json").write_text("{}")
    router_args.mooncake_config_path = "store.json"
    router_args.result_path = str(tmp_path / "result.json")
    router_args.data_path, router_args.shuffle, router_args.limit = "data.parquet", False, None
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "old-driver-path")
    monkeypatch.setattr(run_infer, "_parse_args", lambda: router_args)
    init_ray = Mock()
    monkeypatch.setattr(run_infer, "_init_ray", init_ray)
    monkeypatch.setattr(run_infer, "TaskConfigResolver", SimpleNamespace(from_file=Mock()))
    monkeypatch.setattr(run_infer, "load_dataset", lambda *a, **kw: SimpleNamespace(to_list=lambda: [{}]))
    monkeypatch.setattr(run_infer, "tq", SimpleNamespace(init=Mock()))
    monkeypatch.setattr(run_infer, "_build_prompts", Mock(return_value="batch"))
    monkeypatch.setattr(
        run_infer,
        "_read_rm_scores",
        lambda *a, **kw: {"scores": [1.0], "per_uid": {"task": [1.0]}, "uid_status": {"task": "finished"}},
    )
    server = Mock()

    def create_engine(*, config):
        assert router_args.mooncake_config_path == str(tmp_path / "store.json")
        router_args.device, router_args.mooncake_save_decode_cache = "ascend", True
        router_args.mooncake_config_path = "/changed.json"
        transfer = config.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config
        transfer.kv_connector_extra_config.save_decode_cache = True
        return server

    monkeypatch.setattr(run_infer, "LLMServerManager", SimpleNamespace(create=create_engine))
    monkeypatch.setattr(run_infer, "AgentFrameworkRolloutAdapter", SimpleNamespace(create=Mock(return_value=Mock())))
    build_config = Mock(wraps=run_infer._mooncake_transfer_config)
    monkeypatch.setattr(run_infer, "_mooncake_transfer_config", build_config)
    run_infer.main()
    build_config.assert_called_once()
    assert init_ray.call_args.args[0]["MOONCAKE_CONFIG_PATH"] == str(tmp_path / "store.json")
    metadata = json.loads((tmp_path / "result.json").read_text())["mooncake"]
    assert metadata["config_path"] == str(tmp_path / "store.json")
    assert metadata["kv_transfer_config"]["kv_connector"] == "MooncakeStoreConnector"
    assert metadata["kv_transfer_config"]["kv_connector_extra_config"] == {"save_decode_cache": False}
    assert os.environ["MOONCAKE_CONFIG_PATH"] == "old-driver-path"
