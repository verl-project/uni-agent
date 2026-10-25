"""CPU tests for the multi-policy verl replica lifecycle wrappers."""

from __future__ import annotations

import pytest

from examples.multi_policy import verl_patch


def _install_fakes(monkeypatch):
    calls = []

    def fake_orig(self, *args, **kwargs):
        calls.append(("orig", self, args, kwargs))
        return "original-result"

    def fake_suffix(self):
        calls.append(("suffix", self))

    monkeypatch.setattr(
        verl_patch, "_ORIG_REPLICA_INIT_COLOCATED", fake_orig
    )
    monkeypatch.setattr(
        verl_patch, "_ORIG_REPLICA_INIT_STANDALONE", fake_orig
    )
    monkeypatch.setattr(
        verl_patch, "_policy_suffixed_replica", fake_suffix
    )
    return calls


@pytest.mark.cpu
@pytest.mark.level0
def test_patched_init_colocated_forwards_positional_resource_pool(monkeypatch):
    calls = _install_fakes(monkeypatch)
    replica = object()
    resource_pool = object()

    result = verl_patch._patched_replica_init_colocated(replica, resource_pool)

    assert result == "original-result"
    assert calls == [
        ("suffix", replica),
        ("orig", replica, (resource_pool,), {}),
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_patched_init_colocated_forwards_keyword_resource_pool(monkeypatch):
    calls = _install_fakes(monkeypatch)
    replica = object()
    resource_pool = object()

    result = verl_patch._patched_replica_init_colocated(
        replica, resource_pool=resource_pool
    )

    assert result == "original-result"
    assert calls == [
        ("suffix", replica),
        ("orig", replica, (), {"resource_pool": resource_pool}),
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_patched_init_standalone_keeps_no_arg_calls(monkeypatch):
    calls = _install_fakes(monkeypatch)
    replica = object()

    result = verl_patch._patched_replica_init_standalone(replica)

    assert result == "original-result"
    assert calls == [
        ("suffix", replica),
        ("orig", replica, (), {}),
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_patched_init_standalone_forwards_resource_pool(monkeypatch):
    calls = _install_fakes(monkeypatch)
    replica = object()
    resource_pool = object()

    result = verl_patch._patched_replica_init_standalone(replica, resource_pool)

    assert result == "original-result"
    assert calls == [
        ("suffix", replica),
        ("orig", replica, (resource_pool,), {}),
    ]
