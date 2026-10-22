"""Run the real optional SDK against isolated HTTP responses, without a cluster."""

from __future__ import annotations

import json

import pytest

from uni_agent.sandbox.agent_dx import AgentDxSandbox, _connection_config

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.asyncio
async def test_actual_sdk_create_commands_shell_files_ports_and_delete(monkeypatch):
    sdk = pytest.importorskip("adx_environment", reason="install the Agent DX SDK to run its HTTP contract")
    httpx = pytest.importorskip("httpx")
    from adx_environment import _transport, tunnel_client

    exchanges = []
    clients = []
    stored = {}
    submitted = {}
    tunnels = []

    class Tunnel:
        def __init__(self, upstream, **kwargs):
            self.info = {"upstream": upstream, **kwargs}
            tunnels.append(self.info)

        def start(self, url, timeout):
            self.info.update(url=url, timeout=timeout)
            return True

        def stop(self):
            self.info["stopped"] = True

    monkeypatch.setattr(tunnel_client, "TunnelClient", Tunnel)

    def respond(request):
        exchanges.append(request)
        path = request.url.path
        if path == "/api/environment/v1/environments" and request.method == "POST":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=(
                    'event: final\ndata: {"environmentId":"sdk-contract","status":"running",'
                    '"tunnel":{"path":"/tunnel/server-selected","proxyUrl":"http://127.0.0.1:38197"}}\n\n'
                ),
            )
        if path == "/api/environment/v1/environments/sdk-contract" and request.method == "DELETE":
            return httpx.Response(204)
        if path == "/direct/sdk-contract/invoke":
            body = json.loads(request.content)
            action, args = body["action"], body["args"]
            if action == "process.capabilities":
                return httpx.Response(
                    200,
                    json={
                        "protocol_version": 1,
                        "capabilities": [
                            "stable-command-id",
                            "recoverable-command-result",
                            "multiplexed-command-watch",
                        ],
                    },
                )
            if action == "process.start":
                submitted[args["command_id"]] = args
                return httpx.Response(200, json={"pid": 123})
            if action == "process.get":
                assert args["command_id"] in submitted
                return httpx.Response(200, json={"status": "SUCCEEDED", "exit_code": 0, "stdout": "sdk-ok\n"})
            if action in {"shell.create", "shell.run", "shell.close"}:
                return httpx.Response(200, json={})
            if action == "shell.poll":
                return httpx.Response(200, json={"status": "done", "exit_code": 0, "stdout": "echo\nshell-ok\n"})
        if path == "/direct/sdk-contract/upload":
            stored[request.url.params["path"]] = request.content
            return httpx.Response(200, json={"path": request.url.params["path"], "size": len(request.content)})
        if path == "/direct/sdk-contract/download":
            return httpx.Response(200, content=stored[request.url.params["path"]])
        raise AssertionError(f"unexpected SDK HTTP call: {request.method} {path}")

    def client_factory(*args, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setattr(_transport, "acquire_shared_http_client", client_factory)
    connection = sdk.ConnectionConfig(server_address="sdk.invalid:7443", token="fixture-token", verify_tls=True)
    sandbox = AgentDxSandbox(
        image="python:3.12",
        cwd="/work dir",
        connection=connection,
        name="sdk-contract-name",
        port_forwardings=[8080],
        mounts=[{"target": "/tools", "image_url": "example/tools:1"}],
        upstream="http://127.0.0.1:8000",
        proxy_port=38197,
    )
    try:
        async with sandbox:
            assert (await sandbox.exec(["printf", "sdk-ok"])).stdout == "sdk-ok\n"
            shell = await sandbox.open_shell(env={"VALUE": "a'b"})
            assert (await shell.run("pwd")).exit_code == 0
            await shell.close()
            await sandbox.write_file("/tmp/data", b"\0value\xff")
            assert await sandbox.read_file("/tmp/data") == b"\0value\xff"
            assert await sandbox.expose_port(8080) == "https://sdk.invalid:7443/sdk-contract/8080"
            assert sandbox.get_tunnel_url() == "http://127.0.0.1:38197"
        body = json.loads(exchanges[0].content)
        assert body["rootfs"]["imageurl"] == "python:3.12"
        assert body["cwd"] == "/work dir" and body["name"] == "sdk-contract-name"
        assert body["mounts"][0]["image_url"] == "example/tools:1"
        assert body["tunnel"] == {"enabled": True, "proxyPort": 38197}
        assert tunnels[0]["url"] == "wss://sdk.invalid:7443/tunnel/server-selected"
        assert tunnels[0]["stopped"] is True
        assert any(request.method == "DELETE" for request in exchanges)
        assert list(submitted.values())[0]["cwd"] == "/work dir"
    finally:
        await sandbox.stop()
        for client in clients:
            client.close()


def test_real_connection_uses_only_adx_variables_and_enables_tls_verification(monkeypatch):
    sdk = pytest.importorskip("adx_environment")
    monkeypatch.setenv("ADX_SERVER_ADDRESS", "sdk.invalid:7443")
    monkeypatch.setenv("ADX_TOKEN", "fixture-token")
    monkeypatch.setenv("OPENYUANRONG_SERVER_ADDRESS", "wrong.invalid")
    monkeypatch.setenv("ADX_TLS", "1")
    monkeypatch.delenv("ADX_TLS_VERIFY", raising=False)
    connection = _connection_config(sdk)
    assert connection.server_address == "sdk.invalid:7443"
    assert connection.use_tls and connection.verify_tls
    monkeypatch.setenv("ADX_TLS_VERIFY", "0")
    assert not _connection_config(sdk).verify_tls
