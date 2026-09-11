from __future__ import annotations

import pytest

from uni_agent.gateway.session import codec as codec_module
from uni_agent.gateway.session.codec import MessageCodec

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class StrictQwenTokenizer:
    """Minimal Qwen3.5-like template used with an unpatched verl adapter."""

    eos_token_id = 100_000
    name_or_path = "Qwen/Qwen3.5-9B"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        del kwargs
        if not any(message.get("role") == "user" for message in messages):
            raise ValueError("Qwen3.5 requires at least one user message")
        if any(message.get("role") == "system" for message in messages[1:]):
            raise ValueError("System message must be at the beginning")

        text = "".join(f"<{message['role']}>{self._content(message.get('content', ''))}<end>" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        encoded = []
        index = 0
        while index < len(text):
            if text.startswith("<end>", index):
                encoded.append(self.eos_token_id)
                index += len("<end>")
            else:
                encoded.append(ord(text[index]))
                index += 1
        return encoded

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join("<end>" if token_id == self.eos_token_id else chr(token_id) for token_id in token_ids)

    @staticmethod
    def _content(content):
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content)
        return str(content)


def test_system_only_full_encode_inserts_dummy_user_after_system(monkeypatch):
    calls = []

    def clean_verl_apply(processing_class, messages, **kwargs):
        calls.append([message["role"] for message in messages])
        return processing_class.apply_chat_template(messages, **kwargs)

    monkeypatch.setattr(codec_module, "_apply_chat_template", clean_verl_apply)
    tokenizer = StrictQwenTokenizer()
    codec = MessageCodec(tokenizer)
    messages = [{"role": "system", "content": "system prompt"}]

    encoded = codec.encode_full(messages)

    assert messages == [{"role": "system", "content": "system prompt"}]
    assert calls[-2:] == [["system"], ["system", "user"]]
    assert "<system>system prompt<end><user><end><assistant>" == tokenizer.decode(encoded)
