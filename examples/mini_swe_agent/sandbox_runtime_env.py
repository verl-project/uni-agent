"""Write a private Ray runtime-env file without putting sandbox tokens in argv."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

PROVIDER_ENV = {
    "openyuanrong": (
        "OPENYUANRONG_SERVER_ADDRESS",
        "OPENYUANRONG_TOKEN",
        "OPENYUANRONG_TLS",
        "OPENYUANRONG_GATEWAY_ADDRESS",
        "OPENYUANRONG_GATEWAY_TLS",
        "OPENYUANRONG_TLS_VERIFY",
        "OPENYUANRONG_TUNNEL_SSL_VERIFY",
    ),
    "agent_dx": (
        "ADX_SERVER_ADDRESS",
        "ADX_TOKEN",
        "ADX_TLS",
        "ADX_TLS_VERIFY",
        "ADX_GATEWAY_ADDRESS",
        "ADX_GATEWAY_TLS",
        "ADX_REQUEST_MAX_ATTEMPTS",
        "ADX_COMMUNICATION_MARGIN_SECONDS",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    ),
}


def runtime_env(provider: str, base: Path | None = None) -> dict:
    import yaml

    config = yaml.safe_load(base.read_text()) if base else {}
    config = config or {}
    if not isinstance(config, dict) or not isinstance(config.get("env_vars", {}), dict):
        raise ValueError("runtime environment must be a mapping with an env_vars mapping")
    names = PROVIDER_ENV[provider]
    for name in names[:2]:
        if not os.environ.get(name, "").strip():
            raise ValueError(f"{name} must be set")
    env = dict(config.get("env_vars", {}))
    env.update({name: os.environ[name] for name in names if os.environ.get(name)})
    config["env_vars"] = env
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDER_ENV, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--inference", action="store_true")
    args = parser.parse_args()
    try:
        config = runtime_env(args.provider, args.base)
    except ValueError as exc:
        parser.error(str(exc))
    if args.inference:
        config["env_vars"].update(
            {
                name: os.environ[name]
                for name in ("DEPLOYMENT", "CUDA_VISIBLE_DEVICES", "TUNNEL_SSL_VERIFY")
                if name in os.environ
            }
        )
        config["env_vars"]["PYTHONPATH"] = "verl"
        config.setdefault("excludes", ["/.git/"])
    descriptor, name = tempfile.mkstemp(prefix="uni-agent-sandbox-runtime-", suffix=".json")
    with os.fdopen(descriptor, "w") as stream:
        json.dump(config, stream)
    print(name)


if __name__ == "__main__":
    main()
