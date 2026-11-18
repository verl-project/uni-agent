"""E2B image-backed templates: build once, and only treat a successful build as ready."""

import sys
import threading
import time
from types import SimpleNamespace

import pytest


class _Templates:
    """Stateful stand-in for ``e2b.Template``: alias and ``default`` tag appear when a build is requested."""

    def __init__(self, *, fail_builds=0, block=None, alias_race=False, request_delay=0.0):
        self.builds = []
        self.status = {}  # name -> "building" | "ready" | "error"
        self.fail_builds = fail_builds
        self.block = block
        self.alias_race = alias_race
        self.request_delay = request_delay
        self._lock = threading.Lock()

    # e2b.Template() / .from_image()
    def __call__(self):
        return SimpleNamespace(from_image=lambda image: ("image", image))

    def template_id(self, name):
        """Stand-in for the alias lookup: the template ID, or None if the alias does not exist."""
        return name if name in self.status else None

    def get_tags(self, name, **kwargs):
        if name not in self.status:
            raise RuntimeError("404: template not found")
        return [SimpleNamespace(tag="default", build_id=f"{name}-build")]

    def get_build_status(self, build_info, **kwargs):
        return SimpleNamespace(status=SimpleNamespace(value=self.status[build_info.template_id]))

    def build(self, template, name, **kwargs):
        time.sleep(self.request_delay)  # the alias appears only after the build request lands
        with self._lock:
            self.builds.append(name)
            self.status[name] = "building"
            fail = self.fail_builds > 0
            self.fail_builds -= 1
        if self.alias_race:
            # Another process won the alias insert and finishes its build.
            self.status[name] = "ready"
            raise RuntimeError("500: duplicate key value violates unique constraint")
        if self.block is not None:
            self.block.wait(5)
        self.status[name] = "error" if fail else "ready"
        if fail:
            raise RuntimeError("build failed")


@pytest.fixture
def e2b_module(monkeypatch):
    import uni_agent.sandbox.e2b as module

    monkeypatch.setattr(module, "_BUILD_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(module, "_build_locks", {})
    monkeypatch.setattr(module, "_failed_builds", {})

    def install(templates):
        monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(Template=templates))
        monkeypatch.setattr(module, "_template_id", lambda name, kwargs: templates.template_id(name))
        return module

    return install


@pytest.mark.cpu
@pytest.mark.level0
def test_builds_once_then_reuses_ready_template(e2b_module):
    templates = _Templates()
    module = e2b_module(templates)

    first = module._ensure_template("python:3.12", 2, 4096)
    second = module._ensure_template("python:3.12", 2, 4096)

    assert first == second
    assert templates.builds == [first]


@pytest.mark.cpu
@pytest.mark.level0
def test_failed_build_is_not_treated_as_ready(e2b_module):
    templates = _Templates(fail_builds=1)
    module = e2b_module(templates)

    with pytest.raises(RuntimeError, match="build failed"):
        module._ensure_template("python:3.12", 2, 4096)
    # The alias and default tag now exist, but the build failed: the next call rebuilds.
    name = module._ensure_template("python:3.12", 2, 4096)

    assert templates.builds == [name, name]
    assert templates.status[name] == "ready"


@pytest.mark.cpu
@pytest.mark.level0
def test_in_progress_build_from_another_process_is_awaited(e2b_module):
    templates = _Templates()
    module = e2b_module(templates)
    name = module._ensure_template("python:3.12", 2, 4096)
    templates.builds.clear()
    templates.status[name] = "building"
    threading.Timer(0.1, lambda: templates.status.__setitem__(name, "ready")).start()

    assert module._ensure_template("python:3.12", 2, 4096) == name
    assert templates.builds == []


@pytest.mark.cpu
@pytest.mark.level0
def test_concurrent_calls_in_one_process_build_once(e2b_module):
    release = threading.Event()
    templates = _Templates(block=release, request_delay=0.05)
    module = e2b_module(templates)
    results, errors = [], []

    def call():
        try:
            results.append(module._ensure_template("python:3.12", 2, 4096))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    threading.Timer(0.2, release.set).start()
    for thread in threads:
        thread.join(5)

    assert errors == []
    assert len(results) == 8 and len(set(results)) == 1
    assert templates.builds == [results[0]]


@pytest.mark.cpu
@pytest.mark.level0
def test_waiters_behind_a_failed_build_reraise_instead_of_rebuilding(e2b_module):
    release = threading.Event()
    templates = _Templates(fail_builds=1, block=release)
    module = e2b_module(templates)
    errors = []

    def call():
        try:
            module._ensure_template("python:3.12", 2, 4096)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for thread in threads:
        thread.start()
    threading.Timer(0.2, release.set).start()
    for thread in threads:
        thread.join(5)

    assert len(errors) == 4
    assert len(templates.builds) == 1


@pytest.mark.cpu
@pytest.mark.level0
def test_lost_alias_race_waits_for_the_winning_build(e2b_module):
    templates = _Templates(alias_race=True)
    module = e2b_module(templates)

    name = module._ensure_template("python:3.12", 2, 4096)

    assert templates.status[name] == "ready"
    assert templates.builds == [name]


@pytest.mark.cpu
@pytest.mark.level0
def test_status_errors_other_than_missing_template_propagate(e2b_module):
    class _Unauthorized(_Templates):
        def template_id(self, name):
            raise RuntimeError("401: invalid API key")

    templates = _Unauthorized()
    module = e2b_module(templates)

    with pytest.raises(RuntimeError, match="401"):
        module._ensure_template("python:3.12", 2, 4096)
    assert templates.builds == []


@pytest.mark.cpu
@pytest.mark.level0
def test_build_status_lookup_errors_propagate_but_not_found_means_building(e2b_module):
    templates = _Templates()
    module = e2b_module(templates)
    name = module._ensure_template("python:3.12", 2, 4096)
    errors = iter([RuntimeError("400: Build 'x' not found"), RuntimeError("401: invalid API key")])

    def get_build_status(build_info, **kwargs):
        raise next(errors)

    templates.get_build_status = get_build_status

    assert module._build_status(name, {}) == "building"
    with pytest.raises(RuntimeError, match="401"):
        module._build_status(name, {})


@pytest.mark.cpu
@pytest.mark.level0
def test_alias_owned_by_another_team_is_used_as_is(e2b_module, monkeypatch):
    templates = _Templates()
    module = e2b_module(templates)
    monkeypatch.setattr(module, "_template_id", lambda name, kwargs: "")

    module._ensure_template("python:3.12", 2, 4096)

    assert templates.builds == []


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(("status_code", "expected"), [(200, "tpl-id"), (404, None), (403, "")])
def test_template_id_uses_the_installed_sdk_alias_lookup(monkeypatch, status_code, expected):
    pytest.importorskip("e2b")
    from e2b.api.client.api.templates import get_templates_aliases_alias

    import uni_agent.sandbox.e2b as module

    calls = []

    def sync_detailed(*, alias, client):
        calls.append((alias, client))
        parsed = SimpleNamespace(template_id="tpl-id") if status_code == 200 else None
        return SimpleNamespace(status_code=status_code, parsed=parsed)

    monkeypatch.setenv("E2B_API_KEY", "e2b_" + "0" * 40)
    monkeypatch.setattr(get_templates_aliases_alias, "sync_detailed", sync_detailed)

    assert module._template_id("uni-agent-x", {}) == expected
    assert calls and calls[0][0] == "uni-agent-x"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("startup_timeout", ["0", "-1"])
def test_non_positive_startup_timeout_waits_for_a_running_build(e2b_module, monkeypatch, startup_timeout):
    templates = _Templates()
    module = e2b_module(templates)
    name = module._ensure_template("python:3.12", 2, 4096)
    templates.builds.clear()
    templates.status[name] = "building"
    monkeypatch.setenv("SANDBOX_STARTUP_TIMEOUT", startup_timeout)
    threading.Timer(0.1, lambda: templates.status.__setitem__(name, "ready")).start()

    assert module._ensure_template("python:3.12", 2, 4096) == name
    assert templates.builds == []


@pytest.mark.cpu
@pytest.mark.level0
def test_build_wait_is_bounded_by_the_startup_timeout(e2b_module, monkeypatch):
    templates = _Templates()
    module = e2b_module(templates)
    name = module._ensure_template("python:3.12", 2, 4096)
    templates.status[name] = "building"
    monkeypatch.setenv("SANDBOX_STARTUP_TIMEOUT", "0.05")

    with pytest.raises(TimeoutError, match="still building"):
        module._wait_for_build(name, {})
