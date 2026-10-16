import asyncio
import copy
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from uni_agent.inference.server import InferenceServerManager
from uni_agent.inference.sglang import SGLangInferenceReplica, SGLangInferenceServer, _engine_args
from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import RolloutMode


def _backend(**overrides):
    config = RolloutConfig(name="sglang", max_model_len=1024, tensor_model_parallel_size=8)
    config.engine_kwargs = {"sglang": {"json_model_override_args": '{"language_model_only": true}'}}
    backend = SimpleNamespace(
        config=config,
        model_config=SimpleNamespace(
            local_path="/weights",
            local_tokenizer_path="/tokenizer",
            tokenizer_path=None,
            trust_remote_code=True,
        ),
        nnodes=1,
        node_rank=0,
        base_gpu_id=0,
        rollout_mode=RolloutMode.STANDALONE,
        _server_address="127.0.0.1",
    )
    backend.__dict__.update(overrides)
    return backend


@pytest.mark.cpu
@pytest.mark.level0
def test_manager_selects_recipe_replica_without_replacing_verl_registry(monkeypatch):
    from verl.workers.rollout import llm_server
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    registry = dict(RolloutReplicaRegistry._registry)
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"name": "sglang", "nnodes": 1, "disaggregation": {"enabled": False}},
                "model": {},
            }
        }
    )
    assert InferenceServerManager(config).rollout_replica_class is SGLangInferenceReplica
    default_replica = object()
    calls = []

    def resolve(name, **kwargs):
        calls.append((name, kwargs))
        return default_replica

    monkeypatch.setattr(llm_server, "get_rollout_replica_class", resolve)
    config.actor_rollout_ref.rollout.name = "vllm"
    assert InferenceServerManager(config).rollout_replica_class is default_replica
    assert calls == [("vllm", {"disaggregation_enabled": False})]
    assert RolloutReplicaRegistry._registry == registry


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("nnodes,host,expected", [(1, None, None), (2, "::1", "[::1]:12345")])
def test_native_server_args_keep_config_and_separate_tokenizer(nnodes, host, expected):
    backend = _backend(nnodes=nnodes, _master_address=host, _master_port=12345)
    original = copy.deepcopy(backend.config.engine_kwargs)
    args = _engine_args(backend)
    assert args["tp_size"] == 8
    assert args["tokenizer_path"] == "/tokenizer"
    assert args["skip_tokenizer_init"] is True
    assert args["enable_weights_cpu_backup"] is False
    assert args.get("dist_init_addr") == expected
    assert args["json_model_override_args"] == '{"language_model_only": true}'
    assert backend.config.engine_kwargs == original


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("device", ["cuda", "npu"])
def test_native_server_args_use_platform_defaults(monkeypatch, device):
    from uni_agent.inference import sglang as adapter

    monkeypatch.setattr(adapter, "get_platform", lambda: SimpleNamespace(device_name=device))
    args = _engine_args(_backend())
    assert args["enable_memory_saver"] is (device != "npu")
    assert args.get("device") == ("npu" if device == "npu" else None)
    assert args.get("attention_backend") == ("ascend" if device == "npu" else None)


@pytest.mark.cpu
@pytest.mark.level0
def test_native_server_args_preserve_explicit_platform_overrides(monkeypatch):
    from uni_agent.inference import sglang as adapter

    monkeypatch.setattr(adapter, "get_platform", lambda: SimpleNamespace(device_name="npu"))
    backend = _backend()
    overrides = {"device": "cuda", "attention_backend": "flashinfer", "enable_memory_saver": True}
    backend.config.engine_kwargs["sglang"].update(overrides)
    original = copy.deepcopy(backend.config.engine_kwargs)
    args = _engine_args(backend)
    assert {name: args[name] for name in overrides} == overrides
    assert backend.config.engine_kwargs == original


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [None, RuntimeError("backend failure"), asyncio.CancelledError()])
async def test_generation_delegates_tokens_metadata_errors_and_cancellation(exception):
    actor = object.__new__(SGLangInferenceServer)
    output = object()
    generate = AsyncMock(return_value=output, side_effect=exception)
    actor._server = SimpleNamespace(generate=generate)
    params = {"stop_token_ids": [248046], "ignore_eos": False}
    kwargs = dict(prompt_ids=[1, 2], sampling_params=params, request_id="turn-id", image_data=None)
    if exception is None:
        assert await actor.generate(**kwargs) is output
    else:
        with pytest.raises(type(exception)):
            await actor.generate(**kwargs)
    generate.assert_awaited_once_with(**kwargs)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method",
    ["wake_up", "sleep", "clear_kv_cache", "release_kv_cache", "resume_kv_cache", "resume_generation", "stop_profile"],
)
async def test_control_methods_use_verl_contract(method):
    actor = object.__new__(SGLangInferenceServer)
    delegated = AsyncMock(return_value="ok")
    actor._server = SimpleNamespace(**{method: delegated})
    assert await getattr(actor, method)() == "ok"
    delegated.assert_awaited_once_with()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("reject_request", [False, True])
async def test_abort_accepts_verl_keyword_and_preserves_sglang_pause_contract(monkeypatch, reject_request):
    from uni_agent.inference import sglang as adapter

    actor = object.__new__(SGLangInferenceServer)
    output = object()
    abort = AsyncMock(return_value=output)
    actor._server = SimpleNamespace(abort_all_requests=abort)
    warning = Mock()
    monkeypatch.setattr(adapter.logger, "warning", warning)

    assert await actor.abort_all_requests(reject_request=reject_request) is output
    abort.assert_awaited_once_with()
    if reject_request:
        warning.assert_called_once()
        assert "reject_request=True" in warning.call_args.args[0]
    else:
        warning.assert_not_called()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("fail_startup", [False, True])
async def test_native_engine_launch_uses_scheduler_metadata_and_cleans_up_on_error(monkeypatch, fail_startup):
    from uni_agent.inference import sglang as adapter
    from verl.utils.tracking import RLInsightLogger

    captured = {}

    class ServerArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.enable_metrics = False

    class Engine:
        def __init__(self, *, server_args):
            captured["args"] = server_args
            self.tokenizer_manager = SimpleNamespace(server_status=None)
            self.template_manager = object()
            self._scheduler_init_result = SimpleNamespace(scheduler_infos=[{"max_req_input_len": 1024}])

        def shutdown(self):
            captured["shutdown"] = True

    def set_global_state(value):
        captured["state"] = value

    class GlobalState:
        def __init__(self, tokenizer, template, scheduler_info):
            self.tokenizer_manager, self.template_manager, self.scheduler_info = tokenizer, template, scheduler_info

    for name, exports in {
        "sglang": {},
        "sglang.srt": {},
        "sglang.srt.entrypoints": {},
        "sglang.srt.entrypoints.engine": {"Engine": Engine},
        "sglang.srt.entrypoints.http_server": {
            "ServerArgs": ServerArgs,
            "_GlobalState": GlobalState,
            "app": SimpleNamespace(),
            "set_global_state": set_global_state,
        },
        "sglang.srt.managers": {},
        "sglang.srt.managers.tokenizer_manager": {"ServerStatus": SimpleNamespace(Up="up")},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(RLInsightLogger, "enabled", lambda: False)

    async def serve(*args):
        if fail_startup:
            raise RuntimeError("HTTP startup failed")
        return 18000, object()

    monkeypatch.setattr(adapter, "run_uvicorn", serve)
    actor = object.__new__(SGLangInferenceServer)
    actor._server, actor._engine = _backend(), None
    if fail_startup:
        with pytest.raises(RuntimeError, match="HTTP startup failed"):
            await actor.launch_server()
        assert captured["shutdown"]
    else:
        await actor.launch_server()
        assert actor._server._server_port == 18000
        assert actor._server.tokenizer_manager.server_status == "up"
    assert captured["state"].scheduler_info == {"max_req_input_len": 1024}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_real_unmodified_verl_sglang_generation_contract():
    pytest.importorskip("sglang")
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer

    backend = object.__new__(SGLangHttpServer)
    backend.config = RolloutConfig(name="sglang", max_model_len=32)
    backend.model_config = SimpleNamespace(lora_rank=0, lora={})
    backend.replica_rank = 0
    backend.global_steps = None
    backend._disaggregation_role = "null"

    async def generate_request(request, _):
        assert request.input_ids == [1, 2]
        assert request.sampling_params["stop_token_ids"] == [248046]
        yield {
            "output_ids": [42],
            "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [(-0.25, 42, None)]},
        }

    backend.tokenizer_manager = SimpleNamespace(generate_request=generate_request)
    actor = object.__new__(SGLangInferenceServer)
    actor._server = backend
    output = await actor.generate(
        prompt_ids=[1, 2],
        sampling_params={"max_tokens": 4, "stop_token_ids": [248046], "logprobs": True},
        request_id="actual-verl",
    )
    assert output.token_ids == [42]
    assert output.log_probs == [-0.25]
    assert output.stop_reason == "stop"


@pytest.mark.cpu
@pytest.mark.level0
def test_real_sglang_qwen35_hf_override_is_separate_from_top_level_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    pytest.importorskip("sglang", minversion="0.5.20")
    from sglang.srt.arg_groups.model_hook import handle_language_model_only
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "max_position_embeddings": 128,
            "vocab_size": 64,
            "full_attention_interval": 1,
        },
        "vision_config": {"depth": 1, "hidden_size": 32, "num_heads": 2, "intermediate_size": 64},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    defaults = {"model_path": str(tmp_path), "device": "cpu", "dtype": "bfloat16"}
    plain = ModelConfig.from_server_args(ServerArgs(**defaults))
    assert plain.is_lm_only is False
    assert plain.is_multimodal is True

    args = ServerArgs(**defaults, json_model_override_args=json.dumps({"language_model_only": True}))
    model = ModelConfig.from_server_args(args)
    assert model.hf_config.language_model_only is True
    assert model.is_lm_only is True
    assert model.is_multimodal is False
    assert args.language_model_only is False
    assert resolving_view(args).language_model_only is False
    assert config["architectures"][0] not in args.LANGUAGE_MODEL_ONLY_ARCHITECTURES
    handle_language_model_only(args)

    # The native top-level flag takes a different validation path for this model.
    flagged = ServerArgs(**defaults, language_model_only=True)
    with pytest.raises(ValueError, match="--language-model-only does not support"):
        handle_language_model_only(flagged)
