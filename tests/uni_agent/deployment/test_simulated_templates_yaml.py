"""Template-pool YAML loading tests.

The observation templates live in a standalone YAML so perf runs can tune
weights / add long-tail entries without touching code. ``SimulatedRuntime`` loads
the bundled default; an explicit ``templates_path`` overrides it.
"""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("swerex")

from swerex.runtime.abstract import BashAction  # noqa: E402

from uni_agent.deployment.simulated.deployment import SimulatedRuntime, load_templates  # noqa: E402


def _bash(command: str, timeout: int = 10) -> BashAction:
    return BashAction(command=command, timeout=10)


def test_load_default_templates_has_all_route_keys() -> None:
    """The bundled default pool covers every route key the router can emit."""
    pool = load_templates()
    for key in (
        "editor:view",
        "editor:create",
        "editor:str_replace",
        "editor:insert",
        "editor:undo_edit",
        "test_output",
        "python_script",
        "listing",
        "search",
        "file_view",
        "default",
    ):
        assert key in pool, key
        assert len(pool[key]) >= 1
        # each entry is (weight:int, text:str)
        for entry in pool[key]:
            assert isinstance(entry, tuple) and len(entry) == 2
            assert isinstance(entry[0], int) and entry[0] > 0
            assert isinstance(entry[1], str)


def test_each_pool_entry_has_positive_weight() -> None:
    pool = load_templates()
    for key, entries in pool.items():
        assert sum(w for w, _ in entries) > 0, key


def test_templates_path_overrides_pool(tmp_path) -> None:
    """An external YAML replaces the default pool; its weights/text win."""
    yaml_path = tmp_path / "obs.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            python_script:
              - weight: 10
                text: "MARKER_PY_OK"
              - weight: 1
                text: "MARKER_PY_FAIL"
            default:
              - weight: 1
                text: "MARKER_DEFAULT"
            """
        )
    )
    rt = SimulatedRuntime(run_id="t", seed=1, templates_path=str(yaml_path))
    # weight 10:1 -> in 60 pulls the ok marker must dominate
    draws = [await_marker(rt) for _ in range(60)]
    assert draws.count("MARKER_PY_OK") > draws.count("MARKER_PY_FAIL")


def await_marker(rt: SimulatedRuntime) -> str:
    import asyncio

    return asyncio.run(_one_draw(rt))


async def _one_draw(rt: SimulatedRuntime) -> str:
    obs = await rt.run_in_session(_bash("python /testbed/repro.py"))
    return obs.output.strip()


@pytest.mark.asyncio
async def test_overridden_pool_seed_reproducible(tmp_path) -> None:
    yaml_path = tmp_path / "obs.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            python_script:
              - weight: 5
                text: "A"
              - weight: 5
                text: "B"
              - weight: 5
                text: "C"
            default:
              - weight: 1
                text: "D"
            """
        )
    )
    rt1 = SimulatedRuntime(run_id="t", seed=9, templates_path=str(yaml_path))
    rt2 = SimulatedRuntime(run_id="t", seed=9, templates_path=str(yaml_path))
    seq1 = [o.output.strip() for o in [await rt1.run_in_session(_bash(f"python /r{i}.py")) for i in range(20)]]
    seq2 = [o.output.strip() for o in [await rt2.run_in_session(_bash(f"python /r{i}.py")) for i in range(20)]]
    assert seq1 == seq2
