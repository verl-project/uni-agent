from __future__ import annotations

import asyncio
import gc

import pytest

from tests.deployment.support import fail_start, fast_backoff, hang_start, make_loop, reset_limiter
from uni_agent.deployment.local import deployment as mod
from uni_agent.deployment.local.deployment import LocalDeployment


def test_cache_free(monkeypatch):
    """A limiter cache entry must not outlive its event loop."""
    reset_limiter(monkeypatch, per_worker=1)

    async def _use():
        mod._get_limiter()

    for _ in range(3):
        asyncio.run(_use())
    gc.collect()
    assert not mod._LIMITERS


@pytest.mark.parametrize(
    ("var", "raw"),
    [
        ("LOCAL_MAX_STARTING_PER_WORKER", "0"),
        ("LOCAL_MAX_STARTING_PER_WORKER", "junk"),
        ("LOCAL_INIT_WALL_BUDGET", "nan"),
        ("LOCAL_INIT_WALL_BUDGET", "0"),
        ("LOCAL_INIT_WALL_BUDGET", "junk"),
    ],
)
def test_env_reject(monkeypatch, var, raw):
    """Invalid limits fail fast before any attempt starts."""
    reset_limiter(monkeypatch)
    monkeypatch.setenv(var, raw)
    dep = make_loop()

    with pytest.raises(ValueError, match="invalid"):
        asyncio.run(dep.start())
    assert dep._starts == 0


def test_nap_clamp(monkeypatch):
    """Retry backoff never sleeps past the wall deadline."""
    reset_limiter(monkeypatch, wall_budget=0.5)
    naps = []

    async def spy(seconds):
        naps.append(seconds)

    monkeypatch.setattr(mod, "_retrydelay", spy)
    dep = make_loop(starter=fail_start)

    with pytest.raises(RuntimeError):
        asyncio.run(dep.start())
    assert naps
    assert all(n <= 0.5 for n in naps)


def test_no_reentry(monkeypatch):
    """A deployment's lifecycle is serial: neither an overlapping start nor one over a live sandbox."""
    reset_limiter(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    async def hang(dep):
        started.set()
        await release.wait()

    dep = make_loop(starter=hang, stopper=LocalDeployment._cleanup)

    async def _go():
        first = asyncio.create_task(dep.start())
        await started.wait()
        with pytest.raises(RuntimeError, match="lifecycle is serial"):
            await dep.start()
        release.set()
        await first
        # The first start left a runtime behind, so a second one must go through stop() first.
        with pytest.raises(RuntimeError, match="stop\\(\\) must reclaim them first"):
            await dep.start()
        assert dep._busy is None
        await dep.stop()
        await dep.start()

    asyncio.run(asyncio.wait_for(_go(), timeout=5))
    assert dep._starts == 2


def test_hung_cancel(monkeypatch):
    """A hung attempt is cancelled in bounded time and frees its permit."""
    reset_limiter(monkeypatch, per_worker=1, wall_budget=0.5)
    dep_a = make_loop(starter=hang_start)
    dep_b = make_loop()

    async def _go():
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(dep_a.start(), timeout=10.0)
        # The single permit is free again, so a second deployment can take it.
        await asyncio.wait_for(dep_b.start(), timeout=2.0)

    asyncio.run(_go())
    assert dep_a._starts == 1
    assert dep_a._abandoned is True
    assert dep_b._starts == 1


@pytest.mark.parametrize("pinned", [False, True])
def test_repick(monkeypatch, pinned):
    """Refresh unpinned names and ports between retries; keep pinned ones."""
    reset_limiter(monkeypatch)
    fast_backoff(monkeypatch)
    picks = {"n": 0}

    def next_port():
        picks["n"] += 1
        return 10000 + picks["n"]

    monkeypatch.setattr(mod, "_pick_free_port", next_port)
    config = {"published_port": 4567, "container_name": "pinned"} if pinned else {}
    dep = make_loop(starter=fail_start, **config)

    with pytest.raises(RuntimeError, match=r"after 2 attempt"):
        asyncio.run(dep.start(max_retries=2))
    if pinned:
        assert picks["n"] == 0
        assert dep._ports == [4567, 4567]
        assert dep._names == ["pinned", "pinned"]
    else:
        base = "uni-agent-test-run"
        assert dep._ports == [10001, 10002]
        assert dep._names == [base, f"{base}-r1"]
