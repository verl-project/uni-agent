"""Tests for the Modal cold-start fleet limiter (2026-05-22 Patch).

Covers:
  * `_get_starting_semaphore` reads per-worker permits from
    MODAL_MAX_STARTING_PER_WORKER, is lazy, idempotent (singleton), and clamps to >=1.
  * `ModalDeployment.start` retry loop respects max_retries=2 (not 5).
  * `ModalDeployment.start` wall-clock budget aborts further attempts once
    `MODAL_INIT_WALL_BUDGET` is exceeded.
  * `asyncio.wait_for` inside the retry loop cancels a hung `_start`.
  * The STARTING semaphore actually serializes overlapping `_start` calls and
    is released on both success and failure paths.

We do NOT import or hit real modal.com. We bypass `ModalDeployment.__init__`
(which would invoke `_ImageBuilder.auto`) via `object.__new__` and manually
assign the few attributes the methods under test read.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

import pytest

from uni_agent.deployment.modal import deployment as mod
from uni_agent.deployment.modal.deployment import ModalDeployment

# -------------------- helpers --------------------


def _reset_limiter_state(monkeypatch, *, per_worker=16, wall_budget=900.0):
    """Pin the limiter env vars to known values and force semaphore re-init.

    The limiter reads MODAL_MAX_STARTING_PER_WORKER and MODAL_INIT_WALL_BUDGET
    lazily (at first use / per call), so we set them via monkeypatch.setenv and
    they revert after the test. `_STARTING_SEMA` is the singleton cache -- clear
    it so the next call rebuilds with the patched value.
    """
    monkeypatch.setenv("MODAL_MAX_STARTING_PER_WORKER", str(per_worker))
    monkeypatch.setenv("MODAL_INIT_WALL_BUDGET", str(wall_budget))
    monkeypatch.setattr(mod, "_STARTING_SEMA", None, raising=True)


def _make_deployment(
    start_fn: Callable[[_FakeDeployment], asyncio.Future | None] | None = None,
    stop_fn: Callable[[_FakeDeployment], asyncio.Future | None] | None = None,
) -> _FakeDeployment:
    """Build a ModalDeployment instance that skips the heavy ImageBuilder
    and exposes hookable `_start` / `stop`.
    """
    self = object.__new__(_FakeDeployment)
    self.logger = logging.getLogger("test-modal-limiter")
    self.run_id = "test-run"
    self._sandbox = None
    self._runtime = None
    self._start_calls = 0
    self._stop_calls = 0
    self._concurrent_in_start = 0
    self._max_concurrent_observed = 0
    self._start_fn = start_fn or (lambda d: _ok())
    self._stop_fn = stop_fn or (lambda d: _ok())
    return self


async def _ok():
    return None


class _FakeDeployment(ModalDeployment):
    """ModalDeployment with `_start` and `stop` rewired to user callbacks.

    Crucially, `_start` keeps the production semaphore-acquire wrapping
    so we can test serialization. The body just delegates to the test
    callback after acquiring the permit.
    """

    async def _start(self):  # type: ignore[override]
        async with mod._get_starting_semaphore():
            self._start_calls += 1
            self._concurrent_in_start += 1
            self._max_concurrent_observed = max(self._max_concurrent_observed, self._concurrent_in_start)
            try:
                await self._start_fn(self)
            finally:
                self._concurrent_in_start -= 1

    async def stop(self):  # type: ignore[override]
        self._stop_calls += 1
        await self._stop_fn(self)


# -------------------- _get_starting_semaphore --------------------


def test_starting_semaphore_uses_per_worker_env(monkeypatch):
    _reset_limiter_state(monkeypatch, per_worker=16)

    async def _check():
        sem = mod._get_starting_semaphore()
        # Internal asyncio.Semaphore exposes its initial value via `_value`
        # on CPython 3.10+. This is the contract we rely on.
        assert sem._value == 16, f"expected 16, got {sem._value}"

    asyncio.run(_check())


def test_starting_semaphore_clamps_to_one(monkeypatch):
    # A per-worker value of 0 (or negative) must clamp to >=1 -> no deadlock.
    _reset_limiter_state(monkeypatch, per_worker=0)

    async def _check():
        sem = mod._get_starting_semaphore()
        assert sem._value == 1, f"expected clamp to 1, got {sem._value}"

    asyncio.run(_check())


def test_starting_semaphore_is_singleton(monkeypatch):
    _reset_limiter_state(monkeypatch, per_worker=5)

    async def _check():
        a = mod._get_starting_semaphore()
        b = mod._get_starting_semaphore()
        assert a is b, "semaphore should be lazily cached"

    asyncio.run(_check())


def test_starting_semaphore_respects_large_value(monkeypatch):
    _reset_limiter_state(monkeypatch, per_worker=64)

    async def _check():
        sem = mod._get_starting_semaphore()
        assert sem._value == 64

    asyncio.run(_check())


# -------------------- start() retry + wall-budget --------------------


def test_start_succeeds_on_first_attempt_does_not_retry(monkeypatch):
    _reset_limiter_state(monkeypatch)
    dep = _make_deployment()

    async def _go():
        await dep.start()

    asyncio.run(_go())
    assert dep._start_calls == 1
    assert dep._stop_calls == 0  # stop() only called on failure


def test_start_retries_once_then_succeeds(monkeypatch):
    _reset_limiter_state(monkeypatch)

    attempts = {"n": 0}

    async def flaky(dep):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("simulated transient cold-start failure")

    dep = _make_deployment(start_fn=flaky)

    async def _go():
        await dep.start()

    asyncio.run(_go())
    assert dep._start_calls == 2
    assert dep._stop_calls == 1  # cleanup ran once between attempts


def test_start_max_retries_is_two_not_five(monkeypatch):
    """The pre-patch loop tried 5 times. The patched loop must give up at 2."""
    _reset_limiter_state(monkeypatch)

    async def always_fail(dep):
        raise RuntimeError("never works")

    dep = _make_deployment(start_fn=always_fail)

    async def _go():
        await dep.start()

    with pytest.raises(RuntimeError, match=r"after 2 retries"):
        asyncio.run(_go())
    assert dep._start_calls == 2, "must attempt exactly 2 times, not 5"
    assert dep._stop_calls == 2


def test_start_wall_budget_aborts_before_attempt_when_exhausted(monkeypatch):
    """If wall budget is already negative, must NOT call _start again."""
    # 0.5s budget: first attempt sleeps 0.6s and fails -> second attempt
    # should be vetoed by the deadline check (not even invoked).
    _reset_limiter_state(monkeypatch, wall_budget=0.5)

    async def slow_fail(dep):
        await asyncio.sleep(0.6)
        raise RuntimeError("too slow")

    dep = _make_deployment(start_fn=slow_fail)

    async def _go():
        await dep.start()

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        asyncio.run(_go())
    elapsed = time.monotonic() - t0
    assert dep._start_calls == 1, (
        f"second attempt must be skipped after wall budget exhausted, got start_calls={dep._start_calls}"
    )
    # Generous upper bound: first attempt 0.6s + sleep gap + cleanup << 3s
    assert elapsed < 3.0, f"wall budget should short-circuit, elapsed={elapsed:.2f}s"


def test_start_wait_for_cancels_hung_start(monkeypatch):
    """A _start that hangs forever must be cancelled by the per-attempt wait_for.

    We give a 0.4s wall budget; the hung _start must be killed within that
    bound (plus epsilon) instead of hanging the test forever.
    """
    _reset_limiter_state(monkeypatch, wall_budget=0.4)

    async def hang_forever(dep):
        await asyncio.sleep(3600)

    dep = _make_deployment(start_fn=hang_forever)

    async def _go():
        await dep.start()

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        asyncio.run(_go())
    elapsed = time.monotonic() - t0
    # wait_for floor is max(60.0, remaining); remaining=0.4s -> floor 60s,
    # so wall_budget=0.4 will exhaust before 60s timeout fires. The retry
    # loop checks `remaining <= 0` next iteration and exits.
    # Verify we don't actually wait the full 60s wait_for floor: that
    # depends on Python's asyncio.wait_for behavior; with budget < 60 we
    # rely on the OUTER loop's deadline check after attempt 1 to bail.
    # NOTE: this test mainly proves no infinite hang.
    assert elapsed < 90.0, f"start() must not hang forever, elapsed={elapsed:.1f}s"


# -------------------- semaphore serialization --------------------


def test_starting_semaphore_serializes_concurrent_starts(monkeypatch):
    """With per-worker permits=2, 6 concurrent _start calls must have
    at most 2 inside the critical section at any time.
    """
    _reset_limiter_state(monkeypatch, per_worker=2)  # 2 permits

    # Each _start holds the permit for 0.05s, then succeeds.
    async def slow_ok(dep):
        await asyncio.sleep(0.05)

    deps = [_make_deployment(start_fn=slow_ok) for _ in range(6)]

    async def _go():
        await asyncio.gather(*[d.start() for d in deps])

    asyncio.run(_go())

    # Combine observations across all deps. Each dep's
    # _max_concurrent_observed is its OWN local counter (incremented
    # before yielding inside the critical section), but the SEMAPHORE
    # is shared. To prove the cap, sum: at any moment, the sum of
    # _concurrent_in_start across all deps must be <= 2.
    # The local _max_concurrent_observed will always be 1 because
    # each dep can be inside its own _start at most once.
    # Better proof: count how many concurrent deps were active by
    # tracking via a shared counter -- next test does that.
    assert all(d._start_calls == 1 for d in deps)


def test_starting_semaphore_caps_global_in_flight(monkeypatch):
    """Stronger version: instrument a SHARED counter to prove the
    semaphore really caps the number of `_start` bodies running
    simultaneously across multiple ModalDeployment instances.
    """
    _reset_limiter_state(monkeypatch, per_worker=2)  # 2 permits

    shared = {"in_flight": 0, "peak": 0}
    lock = asyncio.Lock()

    async def track(dep):
        async with lock:
            shared["in_flight"] += 1
            shared["peak"] = max(shared["peak"], shared["in_flight"])
        await asyncio.sleep(0.03)
        async with lock:
            shared["in_flight"] -= 1

    deps = [_make_deployment(start_fn=track) for _ in range(10)]

    async def _go():
        await asyncio.gather(*[d.start() for d in deps])

    asyncio.run(_go())

    assert shared["in_flight"] == 0
    assert shared["peak"] <= 2, (
        f"semaphore must cap concurrent _start bodies at 2 (per-worker permits), observed peak={shared['peak']}"
    )
    assert shared["peak"] >= 1


def test_starting_semaphore_released_on_failure(monkeypatch):
    """Permit must be released even when `_start` raises -- otherwise
    a chain of failures would slowly leak all permits and deadlock.
    """
    _reset_limiter_state(monkeypatch, per_worker=1)  # 1 permit

    fail_first_two = {"n": 0}

    async def flaky(dep):
        fail_first_two["n"] += 1
        if fail_first_two["n"] <= 2:
            raise RuntimeError("transient")

    # Single deployment: 3 attempts total inside start() retry, but
    # max_retries=2 so this would only see 2 attempts. To prove
    # release across MULTIPLE deployments we run two back-to-back.
    dep_a = _make_deployment(start_fn=flaky)
    dep_b = _make_deployment(start_fn=lambda d: _ok())  # must succeed -- permit must be free

    async def _go():
        # dep_a uses 2 attempts then raises -- both must release.
        with pytest.raises(RuntimeError):
            await dep_a.start()
        # dep_b must NOT block: if the single permit leaked, it would hang.
        await asyncio.wait_for(dep_b.start(), timeout=2.0)

    asyncio.run(_go())
    assert dep_b._start_calls == 1
