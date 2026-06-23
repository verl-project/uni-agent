"""Internal generation request: the gateway's wire-agnostic request canonical.

Provider adapters (openai / anthropic) lower their wire request into this shape
before it reaches MessageCodec / GatewaySession. The messages/tools shape is
OpenAI-like template-facing canonical (what HF chat templates consume), not a
provider-neutral block model. No wire envelope concept (choices, stream, error
shape, raw tool_choice) appears here.
"""

from __future__ import annotations

from typing import Any, TypedDict


class InternalGenerationRequest(TypedDict):
    """Lowered request consumed by GatewaySession.run_generation.

    messages: template-facing canonical chat messages (role/content/tool_calls).
    tools: OpenAI-function-schema tools, or None.
    chat_template_kwargs: per-request chat template overrides.
    sampling_params: generation params already keyed by codec sampling names.
    """

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    chat_template_kwargs: dict[str, Any]
    sampling_params: dict[str, Any]
