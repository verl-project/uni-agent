"""Unit tests for the shared sandbox ``exec`` error policy and provider wiring.

The refactor split every provider's data plane into a private :meth:`Sandbox._exec`
primitive plus one public :meth:`Sandbox.exec` wrapper that applies a single error
policy for all backends:

* a timeout (per :meth:`Sandbox._is_timeout_error`) -> ``exit_code == -1``;
* any other failure re-raises when the sandbox is no longer alive
  (:meth:`Sandbox.is_alive` -> ``False``), i.e. an infra fault worth surfacing;
* otherwise (sandbox still alive) -> ``exit_code == 127`` carrying the error text
  on ``stderr`` so one bad command can't crash the caller.

These pin that policy with a tiny in-memory fake, then check the real providers
wire their overrides in (``_exec`` primitive, ``is_alive`` liveness probe, and the
``_is_timeout_error`` name-matching). No real SDK / network -- runs fast under
``pytest`` (or ``python`` on this file).
"""

from __future__ import annotations

import asyncio

import pytest

from uni_agent.sandbox.base import ExecResult, Sandbox
from uni_agent.sandbox.registry import get_sandbox_cls


class _FakeSandbox(Sandbox):
    """Drive :meth:`Sandbox.exec` with a preset ``_exec`` result / error / liveness."""

    provider = "fake"

    def __init__(self, *, result: ExecResult | None = None, error: BaseException | None = None, alive: bool = True):
        self._result = result
        self._error = error
        self._alive = alive
        self.calls: list[dict] = []

    async def start(self) -> None:  # pragma: no cover - trivial
        pass

    async def stop(self) -> None:  # pragma: no cover - trivial
        pass

    async def is_alive(self) -> bool:
        return self._alive

    async def _exec(self, argv, *, timeout=None, workdir=None, env=None) -> ExecResult:
        self.calls.append({"argv": list(argv), "timeout": timeout, "workdir": workdir, "env": env})
        if self._error is not None:
            raise self._error
        return self._result if self._result is not None else ExecResult(exit_code=0, stdout="ok", stderr="")


# --------------------------- exec() error policy ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
def test_success_passes_through_and_forwards_args():
    sb = _FakeSandbox(result=ExecResult(exit_code=0, stdout="hi", stderr=""))
    res = asyncio.run(sb.exec(["echo", "hi"], timeout=7, workdir="/w", env={"A": "1"}))
    assert (res.exit_code, res.stdout, res.stderr) == (0, "hi", "")
    assert sb.calls == [{"argv": ["echo", "hi"], "timeout": 7, "workdir": "/w", "env": {"A": "1"}}]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("exc", [asyncio.TimeoutError(), TimeoutError("slow")])
def test_timeout_becomes_exit_code_minus_one(exc):
    sb = _FakeSandbox(error=exc)
    res = asyncio.run(sb.exec(["sleep", "100"], timeout=5))
    assert res.exit_code == -1
    assert res.stdout == ""
    assert "exec timed out after 5" in res.stderr


@pytest.mark.cpu
@pytest.mark.level0
def test_non_timeout_error_downgrades_to_127_when_alive():
    sb = _FakeSandbox(error=RuntimeError("no such file"), alive=True)
    res = asyncio.run(sb.exec(["missing-bin"]))
    assert res.exit_code == 127
    assert res.stdout == ""
    assert res.stderr == "no such file"  # exact backend message, verbatim


@pytest.mark.cpu
@pytest.mark.level0
def test_non_timeout_error_reraises_when_sandbox_dead():
    boom = RuntimeError("backend gone")
    sb = _FakeSandbox(error=boom, alive=False)
    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(sb.exec(["whoami"]))
    assert excinfo.value is boom  # the original infra fault, re-raised unchanged


@pytest.mark.cpu
@pytest.mark.level0
def test_timeout_wins_over_liveness_even_when_dead():
    # A timeout is classified before the liveness check, so it never re-raises.
    sb = _FakeSandbox(error=TimeoutError(), alive=False)
    res = asyncio.run(sb.exec(["x"], timeout=3))
    assert res.exit_code == -1


@pytest.mark.cpu
@pytest.mark.level0
def test_exec_shell_wraps_in_non_login_bash():
    sb = _FakeSandbox(result=ExecResult(exit_code=0, stdout="", stderr=""))
    asyncio.run(sb.exec_shell("echo hi", timeout=3, workdir="/tmp"))
    assert sb.calls[0]["argv"] == ["bash", "-c", "echo hi"]
    assert sb.calls[0]["timeout"] == 3
    assert sb.calls[0]["workdir"] == "/tmp"


# --------------------------- overridable hooks ---------------------------


class _NoIsAlive(Sandbox):
    """Uses the base defaults for ``is_alive`` / ``_is_timeout_error`` (no override)."""

    provider = "noisalive"

    async def start(self) -> None:  # pragma: no cover - trivial
        pass

    async def stop(self) -> None:  # pragma: no cover - trivial
        pass

    async def _exec(self, argv, *, timeout=None, workdir=None, env=None) -> ExecResult:
        raise RuntimeError("kaboom")


@pytest.mark.cpu
@pytest.mark.level0
def test_base_is_alive_defaults_true_so_errors_downgrade():
    sb = _NoIsAlive()
    assert asyncio.run(sb.is_alive()) is True
    res = asyncio.run(sb.exec(["x"]))
    assert res.exit_code == 127 and res.stderr == "kaboom"


class _CustomTimeout(Exception):
    pass


class _CustomTimeoutSandbox(_FakeSandbox):
    def _is_timeout_error(self, exc: BaseException) -> bool:
        return isinstance(exc, _CustomTimeout) or super()._is_timeout_error(exc)


@pytest.mark.cpu
@pytest.mark.level0
def test_is_timeout_error_override_is_honored_by_exec():
    sb = _CustomTimeoutSandbox(error=_CustomTimeout("slow"))
    res = asyncio.run(sb.exec(["x"], timeout=9))
    assert res.exit_code == -1
    assert "exec timed out after 9" in res.stderr


# --------------------------- provider wiring ---------------------------

_PROVIDERS = ["local", "docker", "modal", "vefaas", "openyuanrong"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("name", _PROVIDERS)
def test_provider_implements_private_exec_and_shares_public_exec(name):
    cls = get_sandbox_cls(name)
    assert "_exec" in cls.__dict__, f"{name} must implement the _exec primitive"
    assert "exec" not in cls.__dict__, f"{name} must not override the shared public exec()"
    assert cls.exec is Sandbox.exec


@pytest.mark.cpu
@pytest.mark.level0
def test_local_file_operations_use_host_filesystem(tmp_path):
    from uni_agent.sandbox.local import LocalSandbox

    sandbox = LocalSandbox()
    sandbox_file = tmp_path / "sandbox" / "data.bin"
    upload_source = tmp_path / "upload.bin"
    uploaded_file = tmp_path / "sandbox" / "uploaded.bin"
    downloaded_file = tmp_path / "download" / "result.bin"
    upload_source.write_bytes(b"\x00upload\xff")

    async def run() -> None:
        await sandbox.write_file(str(sandbox_file), b"\x00sandbox\xff")
        assert await sandbox.read_file(str(sandbox_file)) == b"\x00sandbox\xff"

        await sandbox.upload_file(upload_source, str(uploaded_file))
        assert await sandbox.read_file(str(uploaded_file)) == b"\x00upload\xff"

        await sandbox.download_file(str(uploaded_file), downloaded_file)

    asyncio.run(run())
    assert downloaded_file.read_bytes() == b"\x00upload\xff"


def _named_exc(name: str) -> Exception:
    """An exception whose class name matches a provider's SDK timeout type."""
    return type(name, (Exception,), {})()


@pytest.mark.cpu
@pytest.mark.level0
def test_vefaas_recognizes_its_timeout_by_name(monkeypatch):
    monkeypatch.setenv("VEFAAS_FUNCTION_ID", "fid")
    monkeypatch.setenv("VEFAAS_FUNCTION_ROUTE", "route")
    from uni_agent.sandbox.vefaas import VefaasSandbox

    sb = VefaasSandbox()
    assert sb._is_timeout_error(_named_exc("CommandTimeoutError")) is True
    assert sb._is_timeout_error(TimeoutError()) is True
    assert sb._is_timeout_error(RuntimeError("other")) is False


@pytest.mark.cpu
@pytest.mark.level0
def test_modal_inherits_base_timeout_check():
    from uni_agent.sandbox.modal import ModalSandbox

    assert "_is_timeout_error" not in ModalSandbox.__dict__
    sb = ModalSandbox()
    assert sb._is_timeout_error(TimeoutError()) is True
    assert sb._is_timeout_error(RuntimeError("other")) is False


@pytest.mark.cpu
@pytest.mark.level0
def test_openyuanrong_recognizes_its_timeout():
    from uni_agent.sandbox.openyuanrong import OpenyuanrongSandbox

    sb = OpenyuanrongSandbox(image="python:3.12")
    assert sb._is_timeout_error(RuntimeError("Command timed out after 60 seconds")) is True
    assert sb._is_timeout_error(TimeoutError()) is True
    assert sb._is_timeout_error(RuntimeError("other")) is False


# --------------------------- provider is_alive() liveness ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
def test_modal_is_alive_false_before_start():
    from uni_agent.sandbox.modal import ModalSandbox

    assert asyncio.run(ModalSandbox().is_alive()) is False


@pytest.mark.cpu
@pytest.mark.level0
def test_docker_is_alive_false_before_start():
    from uni_agent.sandbox.docker import DockerSandbox

    assert asyncio.run(DockerSandbox().is_alive()) is False


@pytest.mark.cpu
@pytest.mark.level0
def test_vefaas_is_alive_false_before_start(monkeypatch):
    monkeypatch.setenv("VEFAAS_FUNCTION_ID", "fid")
    monkeypatch.setenv("VEFAAS_FUNCTION_ROUTE", "route")
    from uni_agent.sandbox.vefaas import VefaasSandbox

    assert asyncio.run(VefaasSandbox().is_alive()) is False


@pytest.mark.cpu
@pytest.mark.level0
def test_openyuanrong_is_alive_false_before_start():
    from uni_agent.sandbox.openyuanrong import OpenyuanrongSandbox

    assert asyncio.run(OpenyuanrongSandbox(image="python:3.12").is_alive()) is False


def _fake_modal_sandbox(*, poll_returns=None, poll_raises: BaseException | None = None):
    """Stand-in for ``modal.Sandbox`` exposing just ``poll.aio()`` (what is_alive uses)."""

    class _Poll:
        async def aio(self):
            if poll_raises is not None:
                raise poll_raises
            return poll_returns

    class _Sandbox:
        poll = _Poll()

    return _Sandbox()


@pytest.mark.cpu
@pytest.mark.level0
def test_modal_is_alive_true_while_task_running():
    from uni_agent.sandbox.modal import ModalSandbox

    sb = ModalSandbox()
    sb._sandbox = _fake_modal_sandbox(poll_returns=None)  # None -> still running
    assert asyncio.run(sb.is_alive()) is True


@pytest.mark.cpu
@pytest.mark.level0
def test_modal_is_alive_false_when_terminated():
    from uni_agent.sandbox.modal import ModalSandbox

    sb = ModalSandbox()
    sb._sandbox = _fake_modal_sandbox(poll_returns=0)  # exit code -> terminated
    assert asyncio.run(sb.is_alive()) is False


@pytest.mark.cpu
@pytest.mark.level0
def test_modal_is_alive_false_and_swallows_poll_errors():
    from uni_agent.sandbox.modal import ModalSandbox

    sb = ModalSandbox()
    sb._sandbox = _fake_modal_sandbox(poll_raises=RuntimeError("connection lost"))
    assert asyncio.run(sb.is_alive()) is False  # must never raise


# --------------------------- veFaaS multi-function selection ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
def test_vefaas_single_function_pair(monkeypatch):
    monkeypatch.setenv("VEFAAS_FUNCTION_ID", "fid")
    monkeypatch.setenv("VEFAAS_FUNCTION_ROUTE", "route")
    from uni_agent.sandbox.vefaas import VefaasSandbox

    sb = VefaasSandbox()
    assert (sb._function_id, sb._function_route) == ("fid", "route")


@pytest.mark.cpu
@pytest.mark.level0
def test_vefaas_multi_function_pairs_by_index(monkeypatch):
    # Comma-separated ids/routes pair up one-to-one; whitespace is trimmed.
    monkeypatch.setenv("VEFAAS_FUNCTION_ID", "fid1, fid2 ,fid3")
    monkeypatch.setenv("VEFAAS_FUNCTION_ROUTE", "route1,route2,route3")
    from uni_agent.sandbox import vefaas as vefaas_mod

    expected = [("fid1", "route1"), ("fid2", "route2"), ("fid3", "route3")]
    choices = iter(expected)

    def pick(items):
        assert items == expected
        return next(choices)

    monkeypatch.setattr(vefaas_mod.random, "choice", pick)
    seen = []
    for fid, route in expected:
        sb = vefaas_mod.VefaasSandbox()
        assert (sb._function_id, sb._function_route) == (fid, route)
        seen.append((sb._function_id, sb._function_route))
    assert seen == expected


@pytest.mark.cpu
@pytest.mark.level0
def test_vefaas_mismatched_pair_counts_raise(monkeypatch):
    monkeypatch.setenv("VEFAAS_FUNCTION_ID", "fid1,fid2")
    monkeypatch.setenv("VEFAAS_FUNCTION_ROUTE", "route1")
    from uni_agent.sandbox.vefaas import VefaasSandbox

    with pytest.raises(ValueError, match="pair up one-to-one"):
        VefaasSandbox()


@pytest.mark.cpu
@pytest.mark.level0
def test_vefaas_missing_function_env_raises(monkeypatch):
    monkeypatch.delenv("VEFAAS_FUNCTION_ID", raising=False)
    monkeypatch.setenv("VEFAAS_FUNCTION_ROUTE", "route")
    from uni_agent.sandbox.vefaas import VefaasSandbox

    with pytest.raises(ValueError, match="VEFAAS_FUNCTION_ID is not set"):
        VefaasSandbox()


@pytest.mark.cpu
@pytest.mark.level0
def test_vefaas_install_command_downloads_then_execs():
    from uni_agent.sandbox.vefaas import _install_command

    command = _install_command("token123")
    assert command.startswith("curl -fsSL ")
    assert " -o /tmp/swe-rex-install.sh && exec bash /tmp/swe-rex-install.sh token123" in command
    assert "| bash" not in command
    assert "bash -s" not in command


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
