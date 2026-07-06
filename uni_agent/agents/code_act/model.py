"""Thin async client for an OpenAI-compatible chat endpoint (the policy server).

Wraps a single ``chat.completions`` call: normalizes the running conversation to
the API shape, sends the tool schemas, and returns the assistant text plus any
structured tool calls for the CodeAct loop to execute.
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

#: Sampling keys the OpenAI Python SDK accepts as top-level kwargs. Anything else
#: (e.g. ``top_k``, ``repetition_penalty``, ``min_p`` -- vLLM/SGLang extensions the
#: SDK rejects) is forwarded through ``extra_body`` so the server still receives it.
_OPENAI_SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "seed",
        "logit_bias",
        "response_format",
    }
)


class OpenAICompatibleChatModel:
    """One-shot chat client against an OpenAI-compatible server (e.g. vLLM / SGLang)."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "EMPTY",
        model_name: str | None = None,
        sampling_params: dict[str, Any] | None = None,
        tools_schemas: list[dict] | None = None,
        timeout: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.sampling_params = sampling_params or {}
        self.tools_schemas = tools_schemas
        self.timeout = timeout
        self.client = AsyncOpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout)

    def _normalize_messages_for_api(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip locally-added fields the OpenAI API doesn't accept.

        Keeps only the role, content, assistant ``tool_calls`` and tool
        ``tool_call_id`` / ``name`` so a transcript carrying extra bookkeeping
        still serializes cleanly.
        """
        normalized_messages = []
        for message in messages:
            normalized_message: dict[str, Any] = {"role": message["role"]}
            if message.get("content") is not None:
                normalized_message["content"] = message["content"]
            if message["role"] == "assistant" and message.get("tool_calls"):
                normalized_message["tool_calls"] = message["tool_calls"]
            if message["role"] == "tool":
                if message.get("tool_call_id") is not None:
                    normalized_message["tool_call_id"] = message["tool_call_id"]
                if message.get("name") is not None:
                    normalized_message["name"] = message["name"]
            normalized_messages.append(normalized_message)
        return normalized_messages

    async def query(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling_params: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict], dict[str, int]]:
        """Run one chat-completion call.

        Returns ``(text, tool_calls, generation_info)``. ``tool_calls`` is the
        OpenAI ``{"id", "type", "function": {"name", "arguments"}}`` shape (one
        entry per parallel call; ``[]`` when the model answered with plain text).
        """
        params = dict(sampling_params if sampling_params is not None else self.sampling_params)
        # Model name is an endpoint attribute, but tolerate it riding in sampling params.
        model_name = params.pop("model", None) or self.model_name

        # Split standard knobs (passed as top-level kwargs) from server extensions
        # like top_k, which the SDK forwards only via ``extra_body``.
        standard = {key: value for key, value in params.items() if key in _OPENAI_SAMPLING_KEYS}
        extra_body = {key: value for key, value in params.items() if key not in _OPENAI_SAMPLING_KEYS}

        create_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": self._normalize_messages_for_api(messages),
            **standard,
        }
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        if self.tools_schemas:
            create_kwargs["tools"] = self.tools_schemas

        chat_completion = await self.client.chat.completions.create(**create_kwargs)

        response_message = chat_completion.choices[0].message
        response_content = response_message.content or ""
        serialized_tool_calls: list[dict] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
            }
            for tool_call in (response_message.tool_calls or [])
        ]

        usage = chat_completion.usage
        generation_info = {
            "prompt_tokens": usage.prompt_tokens if usage is not None else 0,
            "completion_tokens": usage.completion_tokens if usage is not None else 0,
        }
        return response_content, serialized_tool_calls, generation_info
