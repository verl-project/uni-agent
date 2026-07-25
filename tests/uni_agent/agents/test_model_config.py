from uni_agent.agents.base import ModelConfig


def test_unconfigured_sampling_params_delegate_to_endpoint():
    model = ModelConfig()

    assert model.temperature is None
    assert model.top_p is None
    assert model.top_k is None
    assert model.sampling_params() == {}


def test_explicit_sampling_params_are_forwarded():
    model = ModelConfig(temperature=0.2, top_p=0.8, top_k=-1)

    assert model.sampling_params() == {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": -1,
    }
