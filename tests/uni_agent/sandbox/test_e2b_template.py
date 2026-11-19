"""E2B image-backed templates: build once, and only treat a successful build as ready."""

import asyncio
import sys
import time
from types import SimpleNamespace

import pytest


class _Templates:
    """Stateful stand-in for ``e2b.AsyncTemplate``.

    The alias and its ``default`` tag appear as soon as a build is requested, as on
    E2B; only the build status says whether the template is usable.
    """

    def __init__(self, *, fail_builds=0):
        self.builds = []
        self.status = {}  # name -> "building" | "ready" | "error"
        self.fail_builds = fail_builds
        self.release_build = None  # asyncio.Event: the build blocks until it is set
        self.build_started = None  # asyncio.Event: set when a build request lands
        self.saw_building = None  # asyncio.Event: set when a status poll returns "building"
        self.race_winner = False  # another process owns the alias and is still building
        self.status_error = None  # exception raised by status polls, once set

    def __call__(self):
        return SimpleNamespace(from_image=lambda image: ("image", image))

    def template_id(self, name):
        return name if name in self.status else None

    async def get_tags(self, name, **kwargs):
        return [SimpleNamespace(tag="default", build_id=f"{name}-build")]

    async def get_build_status(self, build_info, **kwargs):
        if self.status_error is not None:
            raise self.status_error
        status = self.status[build_info.template_id]
        if status == "building" and self.saw_building is not None:
            self.saw_building.set()
        return SimpleNamespace(status=SimpleNamespace(value=status))

    async def build(self, template, name, **kwargs):
        self.builds.append(name)
        self.status[name] = "building"
        if self.build_started is not None:
            self.build_started.set()
        if self.race_winner:
            raise RuntimeError("500: duplicate key value violates unique constraint")
        if self.release_build is not None:
            await self.release_build.wait()
        if self.fail_builds > 0:
            self.fail_builds -= 1
            self.status[name] = "error"
            raise RuntimeError("build failed")
        self.status[name] = "ready"


@pytest.fixture
def e2b_module(monkeypatch):
    import uni_agent.sandbox.e2b as module

    monkeypatch.setattr(module, "_BUILD_POLL_INTERVAL", 0.01)

    def install(templates):
        monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(AsyncTemplate=templates))

        async def template_id(name, kwargs):
            # Like a network call: the answer reflects the state when the request
            # was sent, and other tasks run before it returns.
            answer = templates.template_id(name)
            await asyncio.sleep(0)
            return answer

        monkeypatch.setattr(module, "_template_id", template_id)
        return module

    return install


def _name(module, image="python:3.12"):
    async def main():
        return await module._ensure_template(image, 2, 4096)

    return asyncio.run(main())


@pytest.mark.cpu
@pytest.mark.level0
def test_builds_once_then_reuses_ready_template(e2b_module):
    templates = _Templates()
    module = e2b_module(templates)

    first = _name(module)
    second = _name(module)

    assert first == second
    assert templates.builds == [first]


@pytest.mark.cpu
@pytest.mark.level0
def test_failed_build_is_not_treated_as_ready(e2b_module):
    templates = _Templates(fail_builds=1)
    module = e2b_module(templates)

    with pytest.raises(RuntimeError, match="build failed"):
        _name(module)
    # The alias and default tag now exist, but the build failed: the next call rebuilds.
    name = _name(module)

    assert templates.builds == [name, name]
    assert templates.status[name] == "ready"


@pytest.mark.cpu
@pytest.mark.level0
def test_waits_while_another_process_is_building(e2b_module):
    templates = _Templates()
    module = e2b_module(templates)
    name = _name(module)
    templates.builds.clear()
    templates.status[name] = "building"

    async def main():
        templates.saw_building = asyncio.Event()
        call = asyncio.create_task(module._ensure_template("python:3.12", 2, 4096))
        await asyncio.wait_for(templates.saw_building.wait(), 2)
        assert not call.done()  # it is waiting on the running build, not returning early
        templates.status[name] = "ready"
        return await asyncio.wait_for(call, 2)

    assert asyncio.run(main()) == name
    assert templates.builds == []


@pytest.mark.cpu
@pytest.mark.level0
def test_concurrent_callers_share_one_build(e2b_module):
    templates = _Templates()
    module = e2b_module(templates)

    async def main():
        templates.release_build = asyncio.Event()
        templates.build_started = asyncio.Event()
        calls = [asyncio.create_task(module._ensure_template("python:3.12", 2, 4096)) for _ in range(8)]
        await asyncio.wait_for(templates.build_started.wait(), 2)
        # The build request is created after every caller has joined the shared
        # attempt (tasks run in creation order), so all eight are waiting on it now.
        assert len(templates.builds) == 1 and not any(call.done() for call in calls)
        templates.release_build.set()
        return await asyncio.wait_for(asyncio.gather(*calls), 2)

    results = asyncio.run(main())

    assert len(set(results)) == 1
    assert templates.builds == [results[0]]


@pytest.mark.cpu
@pytest.mark.level0
def test_callers_that_joined_a_failed_build_get_its_error_and_the_next_call_rebuilds(e2b_module):
    templates = _Templates(fail_builds=1)
    module = e2b_module(templates)

    async def main():
        templates.release_build = asyncio.Event()
        templates.build_started = asyncio.Event()
        calls = [asyncio.create_task(module._ensure_template("python:3.12", 2, 4096)) for _ in range(4)]
        await asyncio.wait_for(templates.build_started.wait(), 2)
        templates.release_build.set()
        results = await asyncio.gather(*calls, return_exceptions=True)
        assert all(isinstance(r, RuntimeError) and "build failed" in str(r) for r in results)
        assert len(templates.builds) == 1
        return await module._ensure_template("python:3.12", 2, 4096)

    name = asyncio.run(main())

    assert templates.builds == [name, name]


@pytest.mark.cpu
@pytest.mark.level0
def test_cancelled_caller_does_not_cancel_the_shared_build(e2b_module):
    templates = _Templates()
    module = e2b_module(templates)

    async def main():
        templates.release_build = asyncio.Event()
        templates.build_started = asyncio.Event()
        first = asyncio.create_task(module._ensure_template("python:3.12", 2, 4096))
        second = asyncio.create_task(module._ensure_template("python:3.12", 2, 4096))
        await asyncio.wait_for(templates.build_started.wait(), 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        templates.release_build.set()
        return await asyncio.wait_for(second, 2)

    name = asyncio.run(main())

    assert templates.builds == [name]
    assert templates.status[name] == "ready"


@pytest.mark.cpu
@pytest.mark.level0
def test_lost_alias_race_waits_for_the_running_winner(e2b_module):
    templates = _Templates()
    templates.race_winner = True
    module = e2b_module(templates)

    async def main():
        templates.saw_building = asyncio.Event()
        call = asyncio.create_task(module._ensure_template("python:3.12", 2, 4096))
        await asyncio.wait_for(templates.saw_building.wait(), 2)
        assert not call.done()  # the winner is still building; the loser waits
        name = next(iter(templates.status))
        templates.status[name] = "ready"
        return await asyncio.wait_for(call, 2)

    name = asyncio.run(main())

    assert templates.builds == [name]


@pytest.mark.cpu
@pytest.mark.level0
def test_build_error_is_kept_when_the_recovery_status_check_fails(e2b_module):
    templates = _Templates(fail_builds=1)
    module = e2b_module(templates)
    real_build = templates.build

    async def build_then_break_status(*args, **kwargs):
        try:
            await real_build(*args, **kwargs)
        finally:
            templates.status_error = RuntimeError("503: service unavailable")

    templates.build = build_then_break_status

    with pytest.raises(RuntimeError, match="build failed"):
        _name(module)


@pytest.mark.cpu
@pytest.mark.level0
def test_alias_owned_by_another_team_is_not_reported_ready(e2b_module, monkeypatch):
    templates = _Templates()
    module = e2b_module(templates)

    async def foreign(name, kwargs):
        return ""

    monkeypatch.setattr(module, "_template_id", foreign)

    with pytest.raises(RuntimeError, match="belongs to another team"):
        _name(module)
    assert templates.builds == []


@pytest.mark.cpu
@pytest.mark.level0
def test_status_errors_other_than_missing_template_propagate(e2b_module, monkeypatch):
    templates = _Templates()
    module = e2b_module(templates)

    async def unauthorized(name, kwargs):
        raise RuntimeError("401: invalid API key")

    monkeypatch.setattr(module, "_template_id", unauthorized)

    with pytest.raises(RuntimeError, match="401"):
        _name(module)
    assert templates.builds == []


@pytest.mark.cpu
@pytest.mark.level0
def test_build_status_lookup_errors_propagate_but_not_found_means_building(e2b_module):
    templates = _Templates()
    module = e2b_module(templates)
    name = _name(module)

    templates.status_error = RuntimeError("400: Build 'x' not found")
    assert asyncio.run(module._build_status(name, {})) == "building"
    templates.status_error = RuntimeError("401: invalid API key")
    with pytest.raises(RuntimeError, match="401"):
        asyncio.run(module._build_status(name, {}))


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("startup_timeout", ["0", "-1"])
def test_non_positive_startup_timeout_waits_for_a_running_build(e2b_module, monkeypatch, startup_timeout):
    templates = _Templates()
    module = e2b_module(templates)
    name = _name(module)
    templates.builds.clear()
    templates.status[name] = "building"
    monkeypatch.setenv("SANDBOX_STARTUP_TIMEOUT", startup_timeout)

    async def main():
        templates.saw_building = asyncio.Event()
        call = asyncio.create_task(module._ensure_template("python:3.12", 2, 4096))
        await asyncio.wait_for(templates.saw_building.wait(), 2)
        templates.status[name] = "ready"
        return await asyncio.wait_for(call, 2)

    assert asyncio.run(main()) == name
    assert templates.builds == []


@pytest.mark.cpu
@pytest.mark.level0
def test_build_wait_stops_at_the_startup_timeout_not_at_the_next_poll(e2b_module, monkeypatch):
    templates = _Templates()
    module = e2b_module(templates)
    name = _name(module)
    templates.status[name] = "building"
    monkeypatch.setattr(module, "_BUILD_POLL_INTERVAL", 5.0)
    monkeypatch.setenv("SANDBOX_STARTUP_TIMEOUT", "0.05")

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="still building"):
        asyncio.run(module._wait_for_build(name, {}))
    assert time.monotonic() - started < 1.0


@pytest.mark.cpu
@pytest.mark.level0
def test_start_does_not_use_the_default_executor_for_templates(e2b_module, monkeypatch):
    templates = _Templates()
    module = e2b_module(templates)

    async def no_threads(*args, **kwargs):
        raise AssertionError("template handling must not run in the default executor")

    class _AsyncSandbox:
        @classmethod
        async def create(cls, **kwargs):
            return SimpleNamespace(sandbox_id="sb-1")

    monkeypatch.setattr(module.asyncio, "to_thread", no_threads)
    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(AsyncTemplate=templates, AsyncSandbox=_AsyncSandbox))

    sandbox = module.E2BSandbox(image="python:3.12")
    asyncio.run(sandbox.start())

    assert sandbox.sandbox_id == "sb-1"
    assert len(templates.builds) == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(("status_code", "expected"), [(200, "tpl-id"), (404, None), (403, "")])
def test_template_id_uses_the_installed_sdk_alias_lookup(monkeypatch, status_code, expected):
    pytest.importorskip("e2b")
    from e2b.api.client.api.templates import get_templates_aliases_alias

    import uni_agent.sandbox.e2b as module

    calls = []

    async def asyncio_detailed(*, alias, client):
        calls.append((alias, client))
        parsed = SimpleNamespace(template_id="tpl-id") if status_code == 200 else None
        return SimpleNamespace(status_code=status_code, parsed=parsed)

    monkeypatch.setenv("E2B_API_KEY", "e2b_" + "0" * 40)
    monkeypatch.setattr(get_templates_aliases_alias, "asyncio_detailed", asyncio_detailed)

    assert asyncio.run(module._template_id("uni-agent-x", {})) == expected
    assert calls and calls[0][0] == "uni-agent-x"
