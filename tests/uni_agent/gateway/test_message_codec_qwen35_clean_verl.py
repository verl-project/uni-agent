from __future__ import annotations

import pytest

from uni_agent.gateway.session.codec import MessageCodec, _normalize_messages_for_continuous_tokens

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


@pytest.mark.cpu
@pytest.mark.level0
def test_system_only_initial_tokens_insert_dummy_user_after_system():
    tokenizer = StrictQwenTokenizer()
    codec = MessageCodec(tokenizer)
    messages = [{"role": "system", "content": "system prompt"}]

    normalized = _normalize_messages_for_continuous_tokens(messages)
    encoded = codec.build_initial_tokens(messages)

    assert messages == [{"role": "system", "content": "system prompt"}]
    assert [message["role"] for message in normalized] == ["system", "user"]
    assert "<system>system prompt<end><user><end><assistant>" == tokenizer.decode(encoded)

    updated = messages + [{"role": "user", "content": "continue"}]
    previous_for_merge = _normalize_messages_for_continuous_tokens(messages, force_system_anchor=True)
    updated_for_merge = _normalize_messages_for_continuous_tokens(updated, force_system_anchor=True)
    assert updated_for_merge[: len(previous_for_merge)] == previous_for_merge
