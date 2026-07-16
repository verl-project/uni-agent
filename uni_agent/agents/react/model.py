"""Thin async client for an OpenAI-compatible chat endpoint (the policy server).

Talks to ``{base_url}/chat/completions`` directly over ``aiohttp`` (no OpenAI SDK):
normalizes the running conversation to the API shape, sends the tool schemas, and
returns the assistant text plus any structured tool calls for the ReAct loop to
execute. One :class:`aiohttp.ClientSession` is reused across calls (connections are not
kept alive), so callers should :meth:`aclose` the model (or use it as an async context manager).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class _TransientHTTPError(Exception):
    """A 429 / 5xx response worth retrying (server busy or a transient hiccup)."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status


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
        timeout: float | None = None,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.sampling_params = sampling_params or {}
        self.tools_schemas = tools_schemas
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self._session: aiohttp.ClientSession | None = None

    # ----- session lifecycle -----
    def _session_for_call(self) -> aiohttp.ClientSession:
        """Lazily open (and reuse) one session, bound to the running loop (connections not kept alive)."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(force_close=True),
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        return self._session

    async def aclose(self) -> None:
        """Close the underlying session (idempotent; safe if it was never opened)."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> OpenAICompatibleChatModel:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

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

        # Raw HTTP: every sampling knob (incl. server extensions like top_k) goes
        # straight into the request body -- there's no SDK to reject unknown keys.
        body: dict[str, Any] = {
            "model": model_name,
            "messages": self._normalize_messages_for_api(messages),
            **params,
        }
        if self.tools_schemas:
            body["tools"] = self.tools_schemas

        data = await self._post_chat_completion(body)

        response_message = data["choices"][0]["message"]
        response_content = response_message.get("content") or ""
        serialized_tool_calls: list[dict] = [
            {
                "id": tool_call["id"],
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"],
                },
            }
            for tool_call in (response_message.get("tool_calls") or [])
        ]

        usage = data.get("usage") or {}
        generation_info = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
        return response_content, serialized_tool_calls, generation_info

    async def _post_chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST ``body`` to ``/chat/completions``, retrying transient failures.

        Retries up to ``max_retries`` (exponential backoff) on connection errors,
        timeouts, and 429/5xx; any other 4xx fails fast with the server's response text.
        """
        url = f"{self.base_url}/chat/completions"
        session = self._session_for_call()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with session.post(url, json=body) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        raise _TransientHTTPError(resp.status, await resp.text())
                    if resp.status >= 400:
                        raise RuntimeError(
                            f"chat/completions returned HTTP {resp.status}: {(await resp.text())[:1000]}"
                        )
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError, _TransientHTTPError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = 2.0 * (2**attempt)
                logger.warning(
                    "chat/completions attempt %d/%d failed (%r); retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc
