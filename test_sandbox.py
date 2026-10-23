"""Validate a remote sandbox and an optional sidecar without the training stack.

Run ``python test_sandbox.py --help`` for the documented validation interface.
Credentials are read only from the process environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import secrets
import shlex
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from examples.sidecar.validation import add_check_arguments, check_result, load_check


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Task image with bash, python3 and pip")
    parser.add_argument("--sidecar-recipe", help="Recipe name in the sidecar manifest")
    parser.add_argument("--sidecar-image", help="Published sidecar image accessible to the provider")
    parser.add_argument("--report", type=Path, default=Path("sidecar-validation.json"))
    parser.add_argument("--pip-index", default="", help="Public index URL for the sandbox pip install")
    parser.add_argument("--swe-rex-spec", default="swe-rex", help="Package requirement, e.g. swe-rex==1.4.0")
    parser.add_argument("--cpu", type=int, default=2000, help="Sandbox CPU request in millicores")
    parser.add_argument("--memory", type=int, default=4096, help="Sandbox memory request in MB")
    add_check_arguments(parser)
    args = parser.parse_args(argv)
    if bool(args.sidecar_recipe) != bool(args.sidecar_image):
        parser.error("--sidecar-recipe and --sidecar-image must be supplied together")
    check_options = (args.recipe_file, args.stdin_file, args.expected_stdout_file, args.check_script)
    if not args.sidecar_recipe and any(check_options):
        parser.error("sidecar check options require --sidecar-recipe and --sidecar-image")
    args.entrypoint_check = None
    if args.sidecar_recipe:
        try:
            args.entrypoint_check = load_check(
                args.sidecar_recipe,
                recipe_file=args.recipe_file,
                stdin_file=args.stdin_file,
                expected_stdout_file=args.expected_stdout_file,
                check_script=args.check_script,
            )
        except (ValueError, TypeError, OSError, KeyError) as exc:
            parser.error(str(exc))
    if args.cpu <= 0 or args.memory <= 0:
        parser.error("--cpu and --memory must be positive")
    if args.pip_index and (urlsplit(args.pip_index).username or urlsplit(args.pip_index).password):
        parser.error("--pip-index must not contain credentials")
    return args


def redact(value, sensitive_values):
    text = str(value)
    for secret in sensitive_values:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)
    return text


def configure_environment():
    server = os.getenv("AKERNEL_SERVER_ADDRESS") or os.getenv("OPENYUANRONG_SERVER_ADDRESS")
    token = os.getenv("AKERNEL_TOKEN") or os.getenv("OPENYUANRONG_TOKEN")
    if not server or not token:
        raise ValueError("Set AKERNEL_SERVER_ADDRESS and AKERNEL_TOKEN (or OPENYUANRONG equivalents)")
    if "://" in server:
        raise ValueError("The service address must use host:port, without a URL scheme")
    os.environ["AKERNEL_SERVER_ADDRESS"] = server
    os.environ["AKERNEL_TOKEN"] = token
    verify = os.getenv(
        "AKERNEL_TUNNEL_SSL_VERIFY",
        os.getenv("OPENYUANRONG_TUNNEL_SSL_VERIFY", os.getenv("TUNNEL_SSL_VERIFY", "1")),
    )
    os.environ["TUNNEL_SSL_VERIFY"] = verify
    os.environ["YR_LOG_LEVEL"] = "ERROR"
    logging.disable(logging.CRITICAL)
    return server, verify not in ("0", "false", "False")


def main(argv=None):
    args = parse_args(argv)
    # The report deliberately excludes environment values and command strings.
    sensitive_values = [
        value
        for key, value in os.environ.items()
        if re.search(r"TOKEN|PASSWORD|SECRET|API_KEY", key, re.IGNORECASE) and value
    ]
    health_token = secrets.token_hex(24)
    sensitive_values.append(health_token)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "image": args.image,
        "sidecar_recipe": args.sidecar_recipe,
        "sidecar_image": args.sidecar_image,
        "resources": {"cpu_millicores": args.cpu, "memory_mb": args.memory},
        "stages": {},
        "success": False,
    }
    sandbox = None
    stage = "configuration"
    started = time.monotonic()

    def record(name, **details):
        report["stages"][name] = {"success": True, **details}
        print(f"PASS {name}", flush=True)

    def run(command, timeout=60):
        result = sandbox.commands.run(command, timeout=timeout)
        if result.exit_code != 0:
            detail = redact((result.stderr or result.stdout or "")[-3000:], sensitive_values)
            raise RuntimeError(f"Sandbox command exited {result.exit_code}: {detail}")
        return result.stdout or ""

    try:
        server, verify_tls = configure_environment()
        record("configuration", tls_verification=verify_tls)
        stage = "sdk"
        import yr_sandbox

        expected_sdk = "0.10.2rc9"
        sdk_version = importlib.metadata.version("openyuanrong-sandbox")
        if sdk_version != expected_sdk:
            raise RuntimeError(f"Expected openyuanrong-sandbox=={expected_sdk}, found {sdk_version}")
        sdk_file = Path(yr_sandbox.__file__)
        sdk_source = sdk_file.read_bytes()
        sdk_info = {
            "openyuanrong_sandbox": sdk_version,
            "sandbox_source": str(sdk_file),
            "sandbox_source_sha256": hashlib.sha256(sdk_source).hexdigest(),
        }
        report["sdk"] = sdk_info
        connection = yr_sandbox.ConnectionConfig(
            server_address=server,
            token=os.environ["AKERNEL_TOKEN"],
            use_tls=True,
            gateway_use_tls=server.rsplit(":", 1)[-1] == "443",
            verify_tls=verify_tls,
        )
        record("sdk", **sdk_info)

        stage = "create"
        kwargs = {
            "image": args.image,
            "cpu": args.cpu,
            "memory": args.memory,
            "port_forwardings": [8000],
            "idle_timeout": 600,
            "create_timeout": 180,
            "connection": connection,
        }
        if args.sidecar_image:
            mount_target = args.entrypoint_check.recipe["mount_target"]
            kwargs["mounts"] = [yr_sandbox.Mount(target=mount_target, image_url=args.sidecar_image)]
        create_start = time.monotonic()
        sandbox = yr_sandbox.Sandbox(**kwargs)
        report["sandbox_id"] = sandbox.sandbox_id
        record("create", seconds=round(time.monotonic() - create_start, 3))

        stage = "command"
        output = run("printf 'SANDBOX_OK\\n'; command -v bash; python3 --version")
        if "SANDBOX_OK" not in output:
            raise RuntimeError("Sandbox command marker is missing")
        record("command", output=output.strip())

        if args.sidecar_recipe:
            stage = "sidecar_entrypoint"
            check = args.entrypoint_check
            check_directory = f"/tmp/sidecar-check-{secrets.token_hex(6)}"
            run(f"mkdir -p {shlex.quote(check_directory)}")
            for filename, content in check.files.items():
                sandbox.files.write(f"{check_directory}/{filename}", content)
            output = run(shlex.join(check.command(check_directory)), timeout=180)
            check_result(output)
            record("sidecar_entrypoint", output=output.strip())

        stage = "swe_rex_install"
        install = "python3 -m pip install --disable-pip-version-check --no-input "
        if args.pip_index:
            install += f"--index-url {shlex.quote(args.pip_index)} "
        install += shlex.quote(args.swe_rex_spec)
        run(install, timeout=300)
        version = run("python3 -c 'import importlib.metadata; print(importlib.metadata.version(\"swe-rex\"))'")
        record("swe_rex_install", version=version.strip())

        stage = "server_start"
        server_command = (
            "python3 -m swerex.server --host 0.0.0.0 --port 8000 "
            f"--auth-token {shlex.quote(health_token)} > /tmp/sidecar-swerex.log 2>&1"
        )
        sandbox.commands.run(server_command, background=True)
        record("server_start", port=8000)

        stage = "internal_health"
        health_code = (
            "import json, urllib.request; "
            "req=urllib.request.Request('http://127.0.0.1:8000/is_alive', "
            f"headers={{'X-API-Key': {health_token!r}}}); "
            "data=json.load(urllib.request.urlopen(req, timeout=5)); "
            "assert data.get('is_alive') is True, data; print(json.dumps(data))"
        )
        last_error = None
        for _ in range(30):
            try:
                output = run(f"python3 -c {shlex.quote(health_code)}", timeout=15)
                break
            except Exception as exc:  # noqa: BLE001 - persist the validation failure
                last_error = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"SWE-ReX did not become healthy: {last_error}")
        record("internal_health", response=json.loads(output))

        stage = "external_health"
        url = sandbox.get_port_url(8000)
        parts = urlsplit(url)
        if server.rsplit(":", 1)[-1] == "443" and parts.scheme == "http":
            url = urlunsplit(("https", parts.netloc, parts.path, parts.query, parts.fragment))
        context = ssl.create_default_context()
        if not verify_tls:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        request = urllib.request.Request(url.rstrip("/") + "/is_alive", headers={"X-API-Key": health_token})
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            health = json.load(response)
        if health.get("is_alive") is not True:
            raise RuntimeError("Forwarded SWE-ReX health did not return is_alive=true")
        record("external_health", response=health)
        report["success"] = True
    except Exception as exc:  # noqa: BLE001 - persist the validation failure
        report["stages"][stage] = {
            "success": False,
            "error_type": type(exc).__name__,
            "error": redact(exc, sensitive_values),
        }
        print(f"FAIL {stage}: {redact(exc, sensitive_values)}", file=sys.stderr, flush=True)
    finally:
        if sandbox is not None:
            try:
                sandbox.kill()
                for _ in range(10):
                    if not sandbox.is_running():
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError("SDK still reports the sandbox as running after kill")
                record("cleanup", sandbox_id=report.get("sandbox_id"), confirmed=True)
            except Exception as exc:  # noqa: BLE001 - persist the validation failure
                report["success"] = False
                report["stages"]["cleanup"] = {
                    "success": False,
                    "error": redact(exc, sensitive_values),
                }
                print(f"FAIL cleanup: {redact(exc, sensitive_values)}", file=sys.stderr, flush=True)
        else:
            report["stages"]["cleanup"] = {"success": None, "reason": "No sandbox handle was returned"}
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        content = redact(json.dumps(report, ensure_ascii=False, indent=2), sensitive_values)
        args.report.write_text(content + "\n", encoding="utf-8")
        print(f"Report: {args.report}", flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
