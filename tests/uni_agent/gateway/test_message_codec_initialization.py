import pytest


class _UnstableGenerationPromptTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=True, **kwargs):
        del messages, kwargs
        return [10, 20] if not add_generation_prompt else [99, 30, 40]


def test_initialize_generation_prompt_returns_generation_suffix():
    from tests.uni_agent.support import FakeTokenizer
    from uni_agent.gateway.session import codec as codec_module

    assert codec_module.initialize_generation_prompt(FakeTokenizer()) == [ord(char) for char in "assistant:"]


def test_initialize_generation_prompt_rejects_non_prefix_template():
    from uni_agent.gateway.session import codec as codec_module

    with pytest.raises(ValueError, match="stable token suffix"):
        codec_module.initialize_generation_prompt(_UnstableGenerationPromptTokenizer())
