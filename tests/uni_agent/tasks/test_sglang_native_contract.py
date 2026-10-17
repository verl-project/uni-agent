import json
from types import SimpleNamespace

import pytest


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_real_unmodified_verl_sglang_generation_contract(monkeypatch):
    pytest.importorskip("sglang")
    from verl.workers.config import RolloutConfig
    from verl.workers.rollout.sglang_rollout import async_sglang_server

    def refuse_ray_init(*args, **kwargs):
        raise AssertionError("the native CPU contract must not initialize Ray")

    monkeypatch.setattr(async_sglang_server.ray, "init", refuse_ray_init)
    monkeypatch.setattr(
        async_sglang_server.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_actor_name=lambda: "native-cpu-contract"),
    )
    backend = object.__new__(async_sglang_server.SGLangHttpServer)
    backend.config = RolloutConfig(name="sglang", max_model_len=32)
    backend.model_config = SimpleNamespace(lora_rank=0, lora={})
    backend.replica_rank = 0
    backend.global_steps = None
    backend._disaggregation_role = "null"

    async def generate_request(request, _):
        assert request.input_ids == [1, 2]
        assert request.rid == "actual-verl"
        assert request.sampling_params["stop_token_ids"] == [248046]
        assert request.return_logprob is True
        yield {
            "output_ids": [42],
            "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [(-0.25, 42, None)]},
        }

    backend.tokenizer_manager = SimpleNamespace(generate_request=generate_request)
    output = await backend.generate(
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
