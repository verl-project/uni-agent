import pytest

from uni_agent.agents.base import WhiteBoxModelConfig


@pytest.mark.cpu
@pytest.mark.level0
def test_unconfigured_sampling_params_delegate_to_endpoint():
    model = WhiteBoxModelConfig()
    assert model.sampling_params_override == {}
    assert model.sampling_params() == {}


@pytest.mark.cpu
@pytest.mark.level0
def test_explicit_sampling_params_are_forwarded():
    model = WhiteBoxModelConfig(sampling_params_override={"temperature": 0.2, "top_p": 0.8, "top_k": -1})
    assert model.sampling_params() == {"temperature": 0.2, "top_p": 0.8, "top_k": -1}
