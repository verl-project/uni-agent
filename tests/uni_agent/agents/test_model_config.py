import pytest

from uni_agent.agents.base import ModelConfig, WhiteBoxSamplingConfig


@pytest.mark.cpu
@pytest.mark.level0
def test_unconfigured_sampling_params_delegate_to_endpoint():
    sampling = WhiteBoxSamplingConfig()
    assert sampling.sampling_params() == {}
    assert ModelConfig().model_name is None


@pytest.mark.cpu
@pytest.mark.level0
def test_explicit_sampling_params_are_forwarded():
    sampling = WhiteBoxSamplingConfig(temperature=0.2, top_p=0.8, top_k=-1)
    assert sampling.sampling_params() == {"temperature": 0.2, "top_p": 0.8, "top_k": -1}
