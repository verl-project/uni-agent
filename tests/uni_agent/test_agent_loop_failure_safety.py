"""Tests for the failure-path safety layer in `UniAgentLoop`.

Covers:
  * `_synth_failed_routed_experts` returns the correct shape
    `(response_length, num_layers, topk)` when routing replay is on
    AND the MoE shape cache is populated.
  * It returns `None` when routing replay is off (the dominant
    deployment path -- failure should NOT synthesize a tensor).
  * It returns `None` when the MoE shape cache is missing or 0
    (graceful degradation: better None than wrong shape).
  * `_make_minimal_output` returns a coherent `AgentLoopOutput` with
    pad-token + masked structure even when the loop is only partially
    initialized -- so the Layer-2 safety net in `run()` actually has
    a viable fallback.

These tests bypass `UniAgentLoop.__init__` (which would invoke the
verl `AgentLoopBase` setup -- chat model, tokenizer load, etc.) by
constructing the instance via `object.__new__` and manually assigning
the few attributes the methods under test read. This mirrors the
pattern used in `tests/deployment/test_modal_starting_limiter.py`.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# uni_agent.agent_loop transitively imports verl (via uni_agent.reward.base),
# which is intentionally NOT a hard dependency of the uni-agent package and
# is therefore absent from the lean CI environment that runs only ruff/mypy.
# Skip the whole module if verl is not installed; on developer machines
# (and any environment that has run `pip install -e verl`) the tests run.
pytest.importorskip("verl.experimental.agent_loop.agent_loop")

from uni_agent.agent_loop import UniAgentLoop  # noqa: E402

# -------------------- fixtures --------------------


def _make_loop(
    *,
    routing_replay: bool,
    moe_num_layers: int | None,
    moe_topk: int | None,
    pad_token_id: int | None = 0,
) -> UniAgentLoop:
    """Build a UniAgentLoop that skips the verl AgentLoopBase init.

    `routing_replay` controls
    `config.actor_rollout_ref.rollout.enable_rollout_routing_replay`.
    The MoE shape cache is set on the *class*, not the instance, to
    mirror production (it is a class-level singleton populated by
    `_ensure_moe_shape_cached`).
    """
    UniAgentLoop._moe_num_layers = moe_num_layers
    UniAgentLoop._moe_topk = moe_topk

    self = object.__new__(UniAgentLoop)
    self.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(enable_rollout_routing_replay=routing_replay))
    )
    self.tokenizer = SimpleNamespace(pad_token_id=pad_token_id, eos_token_id=2)
    return self


@pytest.fixture(autouse=True)
def _reset_moe_cache():
    """Reset the class-level MoE shape cache before/after each test
    so cross-test pollution does not produce false positives."""
    UniAgentLoop._moe_num_layers = None
    UniAgentLoop._moe_topk = None
    yield
    UniAgentLoop._moe_num_layers = None
    UniAgentLoop._moe_topk = None


# -------------------- _synth_failed_routed_experts --------------------


def test_synth_returns_zero_tensor_with_correct_shape_when_replay_on_and_cache_populated():
    """When routing replay is on and the MoE shape is cached, the
    failure path must return a zero tensor with the exact
    `(response_length, num_layers, topk)` shape that the normal path
    writes (verl/experimental/agent_loop/agent_loop.py:715 destructures
    `length, layer_num, topk_num = output.routed_experts.shape`)."""
    loop = _make_loop(routing_replay=True, moe_num_layers=64, moe_topk=8)

    result = loop._synth_failed_routed_experts(response_length=512)

    assert isinstance(result, np.ndarray)
    assert result.shape == (512, 64, 8)
    assert result.dtype == np.int64
    assert (result == 0).all()


def test_synth_returns_none_when_routing_replay_disabled():
    """`enable_rollout_routing_replay=False` is the default deployment
    state; the normal path never writes `routed_experts` there, so the
    failure path must also return `None` to keep the batch
    homogeneous."""
    loop = _make_loop(routing_replay=False, moe_num_layers=64, moe_topk=8)
    assert loop._synth_failed_routed_experts(response_length=512) is None


def test_synth_returns_none_when_moe_cache_is_unpopulated():
    """`_ensure_moe_shape_cached` swallows failures (non-MoE model,
    bad HF cache, schema change) and leaves the cache at `None`. The
    failure path must then return `None` rather than synthesizing a
    wrong-shape tensor."""
    loop = _make_loop(routing_replay=True, moe_num_layers=None, moe_topk=None)
    assert loop._synth_failed_routed_experts(response_length=512) is None


def test_synth_returns_none_when_moe_cache_has_invalid_zero_values():
    """Defensive: a buggy cache populated with 0 would otherwise
    produce a `(N, 0, 0)` tensor that explodes downstream. Treat 0 as
    'cache unavailable'."""
    loop = _make_loop(routing_replay=True, moe_num_layers=0, moe_topk=8)
    assert loop._synth_failed_routed_experts(response_length=512) is None

    loop2 = _make_loop(routing_replay=True, moe_num_layers=64, moe_topk=0)
    assert loop2._synth_failed_routed_experts(response_length=512) is None


def test_synth_response_length_propagates_to_first_axis():
    """The shape is `(response_length, num_layers, topk)` -- verify
    the response_length axis is wired through correctly so a failure
    sample matches whatever response length the failure builder picked
    (e.g. dummy_response_length = min(512, response_length))."""
    loop = _make_loop(routing_replay=True, moe_num_layers=48, moe_topk=4)
    for n in (1, 8, 256, 512):
        result = loop._synth_failed_routed_experts(response_length=n)
        assert result is not None
        assert result.shape == (n, 48, 4)


# -------------------- _make_minimal_output --------------------


def test_minimal_output_returns_valid_agent_loop_output_with_pad_token():
    """The Layer-2 safety net must produce a usable `AgentLoopOutput`
    even when only `self.tokenizer` is set (the partial init scenario
    -- failure happened before `chat_model` / `interaction` /
    `output_dir` were set)."""
    loop = _make_loop(routing_replay=False, moe_num_layers=None, moe_topk=None, pad_token_id=42)

    output = loop._make_minimal_output()

    assert output.prompt_ids == [42]
    assert output.response_ids == [42]
    assert output.response_mask == [0]
    assert output.response_logprobs is None
    assert output.routed_experts is None
    assert output.reward_score == 0
    assert output.num_turns == 0
    # Extra fields must include traj_masked + traj_exit_reason so
    # downstream verl reward tracking knows to ignore this sample.
    assert output.extra_fields["traj_masked"] == 1
    assert output.extra_fields["traj_exit_reason"] == "build_failed"


def test_minimal_output_falls_back_to_eos_when_pad_token_missing():
    """Some tokenizers (notably older Llama configs) have no
    pad_token_id. Use eos_token_id as the fallback, matching the
    handling already in `_build_empty_agent_output`."""
    loop = _make_loop(routing_replay=False, moe_num_layers=None, moe_topk=None, pad_token_id=None)
    output = loop._make_minimal_output()
    # tokenizer.eos_token_id == 2 from the fixture default
    assert output.prompt_ids == [2]
    assert output.response_ids == [2]


def test_minimal_output_handles_list_pad_token_id():
    """A few tokenizers (multi-modal Qwen variants) expose
    `pad_token_id` as a list -- pick the first element rather than
    crashing inside AgentLoopOutput's int validation."""
    loop = _make_loop(routing_replay=False, moe_num_layers=None, moe_topk=None, pad_token_id=[7, 8, 9])
    output = loop._make_minimal_output()
    assert output.prompt_ids == [7]
