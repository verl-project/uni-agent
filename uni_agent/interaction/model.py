import asyncio
import uuid
from typing import Any

from uni_agent.utils import get_event_loop
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - handled at runtime
    AsyncOpenAI = None


class MaxTokenExceededError(Exception):
    pass


class AgentChatModel:
    client: Any
    """AsyncLLM server manager"""

    tokenizer: Any
    """Tokenizer for the model"""

    max_model_len: int
    """Max model context length"""

    sampling_params: dict[str, Any]
    """Sampling parameters for the model"""

    tools_schemas: list[dict] = None

    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)
        self.loop = asyncio.get_running_loop()

    def set_tools_schemas(self, tools_schemas: list[dict]) -> None:
        self.tools_schemas = tools_schemas

    async def prepare_rollout_cache(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                tools=self.tools_schemas,
            ),
        )
        prompt_ids = normalize_token_ids(prompt_ids)
        return {
            "request_id": str(uuid.uuid4()),
            "prompt_ids": prompt_ids,
            "response_mask": [],
            "response_logprobs": [],
            "routed_experts": None,
            "metrics": {},
            "extra_fields": {},
        }

    async def append_messages_to_rollout_cache(
        self,
        new_messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
    ):
        """Append newly added user/tool messages to the rollout cache."""

        assert new_messages[-1]["role"] in ["user", "tool"], (
            f"Last message must be user or tool, but got {new_messages[-1]['role']}"
        )

        # encode tool response
        tool_response_ids = await self._get_new_message_ids(new_messages)

        # append tool response to prompt
        rollout_cache["prompt_ids"] += tool_response_ids
        rollout_cache["response_mask"] += [0] * len(tool_response_ids)
        if rollout_cache["response_logprobs"]:
            rollout_cache["response_logprobs"] += [0.0] * len(tool_response_ids)

        return rollout_cache

    async def query(
        self,
        messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
        **kwargs,
    ) -> list[dict] | dict:
        request_id = rollout_cache["request_id"]
        prompt_ids = rollout_cache["prompt_ids"]
        metrics = rollout_cache["metrics"]

        if len(prompt_ids) >= self.max_model_len:
            raise MaxTokenExceededError(
                f"prompt_ids length {len(rollout_cache['prompt_ids'])} exceeds max_model_len {self.max_model_len}\n"
                f"Last tool response: {messages[-1]['content']}"
            )

        sampling_params = kwargs.get("sampling_params", self.sampling_params)

        with simple_timer("generate_sequences", metrics):
            token_output = await self.client.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1
        else:
            metrics["num_preempted"] += token_output.num_preempted if token_output.num_preempted is not None else 0
        generation_info = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(token_output.token_ids),
        }
        response_ids = token_output.token_ids
        rollout_cache["prompt_ids"] += response_ids
        rollout_cache["response_mask"] += [1] * len(response_ids)
        if token_output.log_probs is not None:
            rollout_cache["response_logprobs"] += token_output.log_probs
        if token_output.routed_experts is not None:
            rollout_cache["routed_experts"] = token_output.routed_experts
        if not rollout_cache["extra_fields"]:
            rollout_cache["extra_fields"].update(token_output.extra_fields)
        else:
            max_global_steps = token_output.extra_fields.get("max_global_steps", None)
            if max_global_steps is not None:
                rollout_cache["extra_fields"]["max_global_steps"] = max_global_steps
        response_str = await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(response_ids))

        if len(rollout_cache["prompt_ids"]) >= self.max_model_len:
            raise MaxTokenExceededError(
                f"prompt_ids length {len(rollout_cache['prompt_ids'])} exceeds max_model_len {self.max_model_len}\n"
                f"Generated response:\n{response_str}"
            )
        return response_str, rollout_cache, generation_info

    async def _get_new_message_ids(self, new_messages: list[dict[str, str]]) -> list[int]:
        messages = [
            {"role": "system", "content": "mock system"},
            {"role": "user", "content": "mock user"},
            {"role": "assistant", "content": "mock assistant"},
        ]
        base_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                tools=self.tools_schemas,
            ),
        )
        base_ids = normalize_token_ids(base_ids)

        messages.extend(new_messages)
        full_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                tools=self.tools_schemas,
            ),
        )
        full_ids = normalize_token_ids(full_ids)

        # Drop trailing whitespace ("\n") the template appends after the
        # assistant's eos, so base_ids ends exactly where real generation stops.
        eos_id = self.tokenizer.eos_token_id
        cut = 0
        for i in range(len(base_ids) - 1, -1, -1):
            if base_ids[i] == eos_id:
                cut = i + 1
                break
        base_ids = base_ids[:cut]

        assert full_ids[: len(base_ids)] == base_ids, (
            "base_ids must be an exact prefix of full_ids; "
            "chat template produced inconsistent rendering between baseline and full."
        )
        return full_ids[len(base_ids) :]


# this class is only used for Inference-Only Scenario
class OpenAICompatibleChatModel:
    base_url: str
    """OpenAI-compatible API base URL, for example http://127.0.0.1:8000/v1"""

    api_key: str
    """API key for the chat completion endpoint"""

    model_name: str
    """Model name sent to the OpenAI-compatible endpoint"""

    sampling_params: dict[str, Any]
    """Default sampling parameters passed to the endpoint"""

    timeout: int | float
    """HTTP timeout in seconds"""

    tools_schemas: list[dict] = None

    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)
        if not hasattr(self, "sampling_params"):
            self.sampling_params = {}
        if not hasattr(self, "timeout"):
            self.timeout = 300
        self.base_url = self.base_url.rstrip("/")
        self.loop = get_event_loop()
        if AsyncOpenAI is None:
            raise ImportError(
                "openai is required for OpenAICompatibleChatModel. Please install it with `pip install openai`."
            )
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def set_tools_schemas(self, tools_schemas: list[dict]) -> None:
        self.tools_schemas = tools_schemas

    async def prepare_rollout_cache(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "metrics": {},
            "extra_fields": {
                "backend": "openai-compatible",
                "api_messages": [dict(message) for message in messages],
                "last_tool_calls": [],
            },
        }

    async def append_messages_to_rollout_cache(
        self,
        new_messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
    ):
        api_messages = rollout_cache["extra_fields"]["api_messages"]
        last_tool_calls = rollout_cache["extra_fields"].get("last_tool_calls", [])
        last_tool_call = last_tool_calls[0] if last_tool_calls else None

        for message in new_messages:
            if message["role"] == "tool":
                tool_message = {
                    "role": "tool",
                    "content": message["content"],
                }
                if last_tool_call is not None:
                    tool_message["tool_call_id"] = last_tool_call["id"]
                    tool_message["name"] = last_tool_call["function"]["name"]
                api_messages.append(tool_message)
            else:
                api_messages.append(dict(message))

        return rollout_cache

    def _normalize_messages_for_api(self, api_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized_messages = []
        for message in api_messages:
            normalized_message = {"role": message["role"]}
            if message.get("content") is not None:
                normalized_message["content"] = message["content"]
            if message["role"] == "assistant" and message.get("tool_calls"):
                normalized_message["tool_calls"] = message["tool_calls"]
            if message["role"] == "tool":
                normalized_message["tool_call_id"] = message["tool_call_id"]
                if message.get("name") is not None:
                    normalized_message["name"] = message["name"]
            normalized_messages.append(normalized_message)
        return normalized_messages

    async def query(
        self,
        messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
        **kwargs,
    ) -> list[dict] | dict:
        sampling_params = kwargs.get("sampling_params", self.sampling_params)
        api_messages = self._normalize_messages_for_api(rollout_cache["extra_fields"]["api_messages"])

        with simple_timer("generate_sequences", rollout_cache["metrics"]):
            chat_completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                tools=self.tools_schemas,
                temperature=sampling_params.get("temperature", 0.0),
            )

        response_message = chat_completion.choices[0].message
        response_content = response_message.content or ""
        response_tool_calls = list(response_message.tool_calls or [])

        if response_tool_calls:
            serialized_tool_calls = []
            for tool_call in response_tool_calls:
                serialized_tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
            rollout_cache["extra_fields"]["last_tool_calls"] = serialized_tool_calls
            rollout_cache["extra_fields"]["api_messages"].append(
                {
                    "role": "assistant",
                    "content": response_content,
                    "tool_calls": serialized_tool_calls,
                }
            )
        else:
            rollout_cache["extra_fields"]["last_tool_calls"] = []
            rollout_cache["extra_fields"]["api_messages"].append(
                {
                    "role": "assistant",
                    "content": response_content,
                }
            )

        usage = chat_completion.usage
        completion_tokens = usage.completion_tokens if usage is not None else max(len(response_content.split()), 1)
        prompt_tokens = usage.prompt_tokens if usage is not None else 0
        return (
            response_content,
            rollout_cache,
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
