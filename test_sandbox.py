"""One sidecar smoke test: mount, run the declared probe, check health, and clean up."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import secrets
import shlex
import ssl
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from examples.sidecar.build import load_recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Task image with bash, python3, and pip")
    parser.add_argument("--sidecar-image", required=True, help="Published sidecar image or digest")
    parser.add_argument("--sidecar-recipe", required=True, help="Recipe name in the manifest")
    parser.add_argument("--recipe-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("sidecar-validation.json"))
    parser.add_argument("--pip-index", default="", help="Optional public package index")
    args = parser.parse_args()
    index = urlsplit(args.pip_index)
    if index.username or index.password or index.query or index.fragment:
        parser.error("--pip-index must not contain credentials or query parameters")
    try:
        recipe = load_recipe(args.sidecar_recipe, args.recipe_file)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        parser.error(str(exc))

    health_key = secrets.token_hex(24)
    private_values = [health_key] + [
        value
        for key, value in os.environ.items()
        if value and any(word in key.upper() for word in ("TOKEN", "PASSWORD", "SECRET", "API_KEY"))
    ]
    report = {
        "success": False,
        "image": args.image,
        "sidecar_image": args.sidecar_image,
        "recipe": args.sidecar_recipe,
        "stages": {},
    }
    sandbox = None
    stage = "configuration"
    started = time.monotonic()
    logging.disable(logging.CRITICAL)

    def redact(value):
        text = str(value)
        for secret in private_values:
            text = text.replace(secret, "[REDACTED]")
        return text

    def passed(name, **details):
        report["stages"][name] = {"success": True, **details}
        print(f"PASS {name}", flush=True)

    def run(command, timeout=60):
        result = sandbox.commands.run(command, timeout=timeout)
        if result.exit_code != 0:
            raise RuntimeError(f"Command exited {result.exit_code}: {(result.stderr or result.stdout or '')[-1500:]}")
        return result.stdout or ""

    try:
        server = os.getenv("AKERNEL_SERVER_ADDRESS") or os.getenv("OPENYUANRONG_SERVER_ADDRESS")
        token = os.getenv("AKERNEL_TOKEN") or os.getenv("OPENYUANRONG_TOKEN")
        if not server or not token or "://" in server:
            raise ValueError("Set AKERNEL_SERVER_ADDRESS=host:port and AKERNEL_TOKEN to the frontend JWT")
        verify = os.getenv(
            "AKERNEL_TUNNEL_SSL_VERIFY",
            os.getenv("OPENYUANRONG_TUNNEL_SSL_VERIFY", os.getenv("TUNNEL_SSL_VERIFY", "1")),
        ).lower() not in ("0", "false")
        passed(stage, tls_verification=verify)
        stage = "sdk"
        import yr_sandbox

        version = importlib.metadata.version("openyuanrong-sandbox")
        if version != "0.10.2rc9":
            raise RuntimeError(f"Expected openyuanrong-sandbox==0.10.2rc9, found {version}")
        sdk_file = Path(yr_sandbox.__file__)
        passed(stage, version=version, file=str(sdk_file), sha256=hashlib.sha256(sdk_file.read_bytes()).hexdigest())
        connection = yr_sandbox.ConnectionConfig(
            server_address=server,
            token=token,
            use_tls=True,
            gateway_use_tls=server.endswith(":443"),
            verify_tls=verify,
        )
        stage = "create"
        sandbox = yr_sandbox.Sandbox(
            image=args.image,
            cpu=2000,
            memory=4096,
            idle_timeout=600,
            create_timeout=180,
            connection=connection,
            port_forwardings=[8000],
            mounts=[yr_sandbox.Mount(target=recipe["mount_target"], image_url=args.sidecar_image)],
        )
        report["sandbox_id"] = sandbox.sandbox_id
        passed(stage, cpu_millicores=2000, memory_mb=4096)
        stage = "sidecar_probe"
        run("command -v bash && python3 --version")
        passed(stage, output=run(shlex.join(recipe["probe_argv"])).strip()[:1500])

        stage = "swe_rex_install"
        package_index = f"--index-url {shlex.quote(args.pip_index)} " if args.pip_index else ""
        run(f"python3 -m pip install --disable-pip-version-check --no-input {package_index}swe-rex==1.4.0", 300)
        passed(stage, version="1.4.0")
        sandbox.commands.run(
            "python3 -m swerex.server --host 0.0.0.0 --port 8000 "
            f"--auth-token {health_key} > /tmp/sidecar-swerex.log 2>&1",
            background=True,
        )
        stage = "internal_health"
        health_code = (
            "import json,urllib.request; "
            "req=urllib.request.Request('http://127.0.0.1:8000/is_alive', "
            f"headers={{'X-API-Key':{health_key!r}}}); "
            "assert json.load(urllib.request.urlopen(req,timeout=5)).get('is_alive') is True"
        )
        for attempt in range(30):
            try:
                run(f"python3 -c {shlex.quote(health_code)}", 15)
                break
            except RuntimeError:
                if attempt == 29:
                    raise
                time.sleep(1)
        passed(stage, port=8000)
        stage = "external_health"
        url = urlsplit(sandbox.get_port_url(8000))
        if server.endswith(":443") and url.scheme == "http":
            url = url._replace(scheme="https")
        context = ssl.create_default_context()
        if not verify:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        request = urllib.request.Request(url.geturl().rstrip("/") + "/is_alive", headers={"X-API-Key": health_key})
        with urllib.request.urlopen(request, context=context, timeout=30) as response:
            if json.load(response).get("is_alive") is not True:
                raise RuntimeError("Forwarded server is not healthy")
        passed(stage, port=8000)
        report["success"] = True
    except Exception as exc:
        report["stages"][stage] = {"success": False, "error": redact(exc)}
        print(f"FAIL {stage}: {redact(exc)}", flush=True)
    finally:
        if sandbox is not None:
            try:
                sandbox.kill()
                for _ in range(10):
                    if not sandbox.is_running():
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError("Sandbox is still running after kill")
                passed("cleanup", confirmed=True)
            except Exception as exc:
                report["success"] = False
                report["stages"]["cleanup"] = {"success": False, "error": redact(exc)}
                print(f"FAIL cleanup: {redact(exc)}", flush=True)
        else:
            report["stages"]["cleanup"] = {"success": None, "reason": "No sandbox handle returned"}
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(redact(json.dumps(report, indent=2)) + "\n", encoding="utf-8")
        print(f"Report: {args.report}", flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
