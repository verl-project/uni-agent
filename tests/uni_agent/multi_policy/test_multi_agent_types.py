"""Single-session and session-group handles have distinct public fields."""

import pickle

import pytest

from examples.multi_policy.framework import SessionGroupHandle
from uni_agent.gateway.session import SessionHandle


@pytest.mark.cpu
@pytest.mark.level0
def test_group_handle_round_trip_preserves_role_addresses():
    handle = SessionGroupHandle("group", {"planner": "http://gateway/sessions/group:planner/v1"})
    restored = pickle.loads(pickle.dumps(handle))
    assert restored == handle
    assert not hasattr(restored, "base_url")
    assert not hasattr(restored, "session_id")


@pytest.mark.cpu
@pytest.mark.level0
def test_session_handle_has_no_group_fields():
    handle = SessionHandle("group:planner", "http://gateway/sessions/group:planner/v1")
    assert not hasattr(handle, "role_base_urls")
    assert not hasattr(handle, "session_group_id")
