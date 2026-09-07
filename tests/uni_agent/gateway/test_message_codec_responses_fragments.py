import pytest

from tests.uni_agent.support import FakeTokenizer
from uni_agent.gateway.session.codec import MessageCodec

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_incremental_codec_coalesces_responses_assistant_fragments():
    codec = MessageCodec(FakeTokenizer())
    codec.encode_incremental(
        [
            {"role": "assistant", "content": "", "reasoning_content": "thinking"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
    )
