import asyncio

import pytest

from uni_agent.efficiency import collect_efficiency, measure_efficiency, record_efficiency

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_measure_accumulates_and_restores_nested_collectors(monkeypatch):
    ticks = iter([1.0, 3.0, 5.0, 8.0])
    monkeypatch.setattr("uni_agent.efficiency.time.perf_counter", lambda: next(ticks))
    with collect_efficiency() as outer:
        with measure_efficiency("model"):
            pass
        with collect_efficiency() as inner:
            record_efficiency("only_inner")
        with measure_efficiency("model"):
            pass
    record_efficiency("outside")
    assert {k: v for k, v in outer.items() if k.startswith("model/")} == {
        "model/count": 2,
        "model/total_s": 5,
        "model/error_count": 0,
        "model/cancelled_count": 0,
    }
    assert "only_inner" not in outer
    assert "outside" not in outer
    assert inner["only_inner"] == 1
    assert "model/count" not in inner


@pytest.mark.asyncio
async def test_overlapping_episode_collectors_are_isolated():
    async def episode(stage):
        with collect_efficiency() as metrics:
            with measure_efficiency(stage):
                await asyncio.sleep(0)
            return metrics

    one, two = await asyncio.gather(episode("first"), episode("second"))
    assert one["first/count"] == two["second/count"] == 1
    assert "second/count" not in one
    assert "first/count" not in two
