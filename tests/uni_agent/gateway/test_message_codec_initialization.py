import pytest


class _StableGenerationPromptTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=True, **kwargs):
        del messages, kwargs
        ids = [10, 20]
        if add_generation_prompt:
            ids += [30, 40]
        return ids


class _UnstableGenerationPromptTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=True, **kwargs):
        del messages, kwargs
        return [10, 20] if not add_generation_prompt else [99, 30, 40]


class _TypedContentOnlyProcessor:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=True, **kwargs):
        del kwargs
        if any(not isinstance(message["content"], list) for message in messages):
            raise TypeError("content must use typed parts")
        ids = [10, 20]
        if add_generation_prompt:
            ids += [30, 40]
        return ids


def test_initialize_generation_prompt_returns_generation_suffix():
    from uni_agent.gateway.session import codec as codec_module

    assert codec_module.initialize_generation_prompt(_StableGenerationPromptTokenizer()) == [30, 40]


def test_initialize_generation_prompt_supports_typed_content_processors():
    from uni_agent.gateway.session import codec as codec_module

    assert codec_module.initialize_generation_prompt(_TypedContentOnlyProcessor()) == [30, 40]


def test_initialize_generation_prompt_rejects_non_prefix_template():
    from uni_agent.gateway.session import codec as codec_module

    with pytest.raises(ValueError, match="stable token suffix"):
        codec_module.initialize_generation_prompt(_UnstableGenerationPromptTokenizer())
