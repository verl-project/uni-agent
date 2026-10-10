import asyncio

import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("swerex")

from aiohttp import web  # noqa: E402
from swerex.runtime.abstract import Command  # noqa: E402

from uni_agent.sandbox.vefaas import VefaasSandbox, _VefaasRuntime  # noqa: E402

pytestmark = [pytest.mark.cpu, pytest.mark.level0, pytest.mark.asyncio]


async def serve(handler):
    app = web.Application()
    app.router.add_route("*", "/{endpoint}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def test_keepalive_routes_headers_and_close():
    calls = []

    async def handler(request):
        calls.append((request.path, request.transport, dict(request.headers)))
        if request.path == "/execute":
            return web.json_response({"exit_code": 0, "stdout": "ok", "stderr": ""})
        return web.json_response({})

    server, url = await serve(handler)
    a = _VefaasRuntime(base_url=url, auth_token="a-token", instance_name="a")
    b = _VefaasRuntime(base_url=url, auth_token="b-token", instance_name="b")
    try:
        assert await a.is_alive()
        assert (await a.execute(Command(command=["true"]))).stdout == "ok"
        await a.execute(Command(command=["true"]))
        assert len({id(call[1]) for call in calls}) == 1
        assert await b.is_alive()
        assert calls[-1][1] is not calls[0][1]
        assert calls[0][2]["X-API-Key"] == "a-token"
        assert calls[-1][2]["X-API-Key"] == "b-token"
        assert calls[-1][2]["X-Faas-Instance-Name"] == "b"
        assert calls[1][2]["X-Request-ID"] != calls[2][2]["X-Request-ID"]
        await asyncio.gather(a.close(), a.close())
        assert a._session.closed
        assert sum(path == "/close" for path, _, _ in calls) == 1
        with pytest.raises(RuntimeError, match="closed"):
            await a.execute(Command(command=["true"]))
        assert not await a.is_alive()
    finally:
        await a.close()
        await b.close()
        await server.cleanup()


async def test_errors_and_timeout_do_not_leak_session_or_retry_post():
    calls = []

    async def handler(request):
        calls.append(request.path)
        if request.path == "/execute":
            return web.json_response(
                {"swerexception": {"class_path": "swerex.exceptions.CommandTimeoutError", "message": "slow"}},
                status=511,
            )
        return web.Response(status=500)

    server, url = await serve(handler)
    runtime = _VefaasRuntime(base_url=url, auth_token="", instance_name="a")
    try:
        from swerex.exceptions import CommandTimeoutError

        with pytest.raises(CommandTimeoutError):
            await runtime.execute(Command(command=["true"]))
        assert calls == ["/execute"]
        await runtime.close()
        assert runtime._session.closed
    finally:
        await runtime.close()
        await server.cleanup()


async def test_cancel_close_releases_connection():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        entered.set()
        await release.wait()
        return web.json_response({})

    server, url = await serve(handler)
    runtime = _VefaasRuntime(base_url=url, auth_token="", instance_name="a")
    task = asyncio.create_task(runtime.close())
    try:
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime._session.closed and runtime._closed
    finally:
        release.set()
        await runtime.close()
        await server.cleanup()


async def test_server_closed_connection_is_replaced_without_replaying_command():
    transports = []

    async def handler(request):
        transports.append(request.transport)
        response = web.json_response({"exit_code": 0, "stdout": "", "stderr": ""})
        response.force_close()
        return response

    server, url = await serve(handler)
    runtime = _VefaasRuntime(base_url=url, auth_token="", instance_name="a")
    try:
        for _ in range(2):
            assert (await runtime.execute(Command(command=["true"]))).exit_code == 0
        assert len(transports) == 2 and transports[0] is not transports[1]
    finally:
        await runtime.close()
        await server.cleanup()


async def test_partial_start_does_not_silently_report_success(monkeypatch):
    monkeypatch.setenv("VEFAAS_FUNCTION_ID", "function")
    monkeypatch.setenv("VEFAAS_FUNCTION_ROUTE", "https://example.invalid")
    sandbox = VefaasSandbox()
    sandbox._runtime = object()
    with pytest.raises(RuntimeError, match="startup is incomplete"):
        await sandbox.start()
    sandbox._started = True
    await sandbox.start()


@pytest.mark.parametrize(
    ("server_env", "expected"),
    [
        ({"LD_LIBRARY_PATH": "/tmp/_MEIabc"}, "unset|unset"),
        ({"LD_LIBRARY_PATH": "/tmp/_MEIabc", "LD_LIBRARY_PATH_ORIG": "/opt/lib"}, "/opt/lib|unset"),
        ({"LD_LIBRARY_PATH": "/opt/keep"}, "/opt/keep|unset"),
    ],
)
async def test_open_shell_restores_library_path_and_reads_bashrc(monkeypatch, tmp_path, server_env, expected):
    import os

    from uni_agent.sandbox.base import ExecResult

    monkeypatch.setenv("VEFAAS_FUNCTION_ID", "function")
    monkeypatch.setenv("VEFAAS_FUNCTION_ROUTE", "https://example.invalid")
    (tmp_path / ".bashrc").write_text("case $- in *i*) ;; *) return;; esac\nexport FROM_RC=1\n")
    sandbox = VefaasSandbox()
    sandbox._runtime = object()

    async def local_exec(argv, *, timeout=None, workdir=None, env=None):
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "HOME": str(tmp_path), **server_env},
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        return ExecResult(proc.returncode, stdout.decode(), stderr.decode())

    monkeypatch.setattr(sandbox, "_exec", local_exec)
    shell = await sandbox.open_shell()
    res = await shell.run('printf "%s|%s|%s" "${LD_LIBRARY_PATH-unset}" "${LD_LIBRARY_PATH_ORIG-unset}" "$FROM_RC"')
    assert res.stdout == f"{expected}|1"
    sandbox._runtime = object()
    with pytest.raises(RuntimeError, match="replaced"):
        await shell.run("true")
    await shell.close()
