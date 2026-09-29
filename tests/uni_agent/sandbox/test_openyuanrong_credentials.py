from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from uni_agent.sandbox.openyuanrong import _connection_config


def _clear_connection_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENYUANRONG_CREDENTIAL_FILE",
        "OPENYUANRONG_SERVER_ADDRESS",
        "OPENYUANRONG_TOKEN",
        "OPENYUANRONG_TLS",
        "OPENYUANRONG_TLS_VERIFY",
        "OPENYUANRONG_GATEWAY_ADDRESS",
        "OPENYUANRONG_GATEWAY_TLS",
        "OPENYUANRONG_TUNNEL_SSL_VERIFY",
        "TUNNEL_SSL_VERIFY",
        "AKERNEL_SERVER_ADDRESS",
        "AKERNEL_TOKEN",
        "YR_TUNNEL_SSL_VERIFY",
    ):
        monkeypatch.delenv(name, raising=False)


def _write_credentials(path: Path, text: str, mode: int = 0o600) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


@pytest.mark.cpu
@pytest.mark.level0
def test_connection_config_loads_protected_file_and_compatible_legacy_token(tmp_path: Path, monkeypatch):
    _clear_connection_env(monkeypatch)
    credential_file = _write_credentials(
        tmp_path / "openyuanrong.env",
        "\n".join(
            [
                'OPENYUANRONG_SERVER_ADDRESS="https://sandbox.example"',
                'AKERNEL_SERVER_ADDRESS="https://sandbox.example"',
                'OPENYUANRONG_TOKEN="invalid-placeholder"',
                'AKERNEL_TOKEN="header.payload.signature"',
                "TUNNEL_SSL_VERIFY=1",
            ]
        ),
    )
    monkeypatch.setenv("OPENYUANRONG_CREDENTIAL_FILE", str(credential_file))
    sdk = SimpleNamespace(ConnectionConfig=lambda **kwargs: kwargs)

    result = _connection_config(sdk)

    assert result["server_address"] == "https://sandbox.example"
    assert result["token"] == "header.payload.signature"
    assert result["gateway_use_tls"] is True
    assert os.environ["YR_TUNNEL_SSL_VERIFY"] == "1"


@pytest.mark.cpu
@pytest.mark.level0
def test_connection_config_prefers_valid_current_token(tmp_path: Path, monkeypatch):
    _clear_connection_env(monkeypatch)
    credential_file = _write_credentials(
        tmp_path / "openyuanrong.env",
        "\n".join(
            [
                "OPENYUANRONG_SERVER_ADDRESS=https://sandbox.example",
                "AKERNEL_SERVER_ADDRESS=https://sandbox.example",
                "OPENYUANRONG_TOKEN=current.payload.signature",
                "AKERNEL_TOKEN=legacy.payload.signature",
            ]
        ),
    )
    monkeypatch.setenv("OPENYUANRONG_CREDENTIAL_FILE", str(credential_file))
    sdk = SimpleNamespace(ConnectionConfig=lambda **kwargs: kwargs)

    result = _connection_config(sdk)

    assert result["token"] == "current.payload.signature"


@pytest.mark.cpu
@pytest.mark.level0
def test_connection_config_never_pairs_legacy_token_with_a_different_endpoint(tmp_path: Path, monkeypatch):
    _clear_connection_env(monkeypatch)
    credential_file = _write_credentials(
        tmp_path / "openyuanrong.env",
        "\n".join(
            [
                "OPENYUANRONG_SERVER_ADDRESS=https://current.example",
                "AKERNEL_SERVER_ADDRESS=https://legacy.example",
                "OPENYUANRONG_TOKEN=invalid-current-token",
                "AKERNEL_TOKEN=legacy.payload.signature",
            ]
        ),
    )
    monkeypatch.setenv("OPENYUANRONG_CREDENTIAL_FILE", str(credential_file))
    sdk = SimpleNamespace(ConnectionConfig=lambda **kwargs: kwargs)

    result = _connection_config(sdk)

    assert result["server_address"] == "https://current.example"
    assert result["token"] == "invalid-current-token"


@pytest.mark.cpu
@pytest.mark.level0
def test_connection_config_rejects_unprotected_file(tmp_path: Path, monkeypatch):
    _clear_connection_env(monkeypatch)
    credential_file = _write_credentials(
        tmp_path / "openyuanrong.env",
        "OPENYUANRONG_SERVER_ADDRESS=https://sandbox.example\n"
        "OPENYUANRONG_TOKEN=header.payload.signature\n",
        mode=0o644,
    )
    monkeypatch.setenv("OPENYUANRONG_CREDENTIAL_FILE", str(credential_file))

    with pytest.raises(PermissionError, match="mode 600"):
        _connection_config(SimpleNamespace(ConnectionConfig=lambda **kwargs: kwargs))


@pytest.mark.cpu
@pytest.mark.level0
def test_connection_config_still_accepts_direct_environment_variables(monkeypatch):
    _clear_connection_env(monkeypatch)
    monkeypatch.setenv("OPENYUANRONG_SERVER_ADDRESS", "https://sandbox.example")
    monkeypatch.setenv("OPENYUANRONG_TOKEN", "header.payload.signature")
    sdk = SimpleNamespace(ConnectionConfig=lambda **kwargs: kwargs)

    result = _connection_config(sdk)

    assert result == {
        "server_address": "https://sandbox.example",
        "token": "header.payload.signature",
        "gateway_use_tls": True,
    }


@pytest.mark.cpu
@pytest.mark.level0
def test_same_server_gateway_inherits_control_plane_tls(monkeypatch):
    _clear_connection_env(monkeypatch)
    monkeypatch.setenv("OPENYUANRONG_SERVER_ADDRESS", "https://sandbox.example")
    monkeypatch.setenv("OPENYUANRONG_TOKEN", "header.payload.signature")
    sdk = SimpleNamespace(ConnectionConfig=lambda **kwargs: kwargs)

    result = _connection_config(sdk)

    assert result["gateway_use_tls"] is True


@pytest.mark.cpu
@pytest.mark.level0
def test_explicit_gateway_tls_overrides_inherited_tls(monkeypatch):
    _clear_connection_env(monkeypatch)
    monkeypatch.setenv("OPENYUANRONG_SERVER_ADDRESS", "https://sandbox.example")
    monkeypatch.setenv("OPENYUANRONG_TOKEN", "header.payload.signature")
    monkeypatch.setenv("OPENYUANRONG_GATEWAY_TLS", "false")
    sdk = SimpleNamespace(ConnectionConfig=lambda **kwargs: kwargs)

    result = _connection_config(sdk)

    assert result["gateway_use_tls"] is False


@pytest.mark.cpu
@pytest.mark.level0
def test_separate_gateway_does_not_inherit_server_tls(monkeypatch):
    _clear_connection_env(monkeypatch)
    monkeypatch.setenv("OPENYUANRONG_SERVER_ADDRESS", "https://sandbox.example")
    monkeypatch.setenv("OPENYUANRONG_TOKEN", "header.payload.signature")
    monkeypatch.setenv("OPENYUANRONG_GATEWAY_ADDRESS", "gateway.example")
    sdk = SimpleNamespace(ConnectionConfig=lambda **kwargs: kwargs)

    result = _connection_config(sdk)

    assert result["gateway_address"] == "gateway.example"
    assert "gateway_use_tls" not in result
