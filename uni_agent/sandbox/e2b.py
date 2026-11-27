from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
import shlex
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from .base import _DEFAULT_STARTUP_TIMEOUT, ExecResult, Sandbox, _env_number, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import ImageMount, SandboxConfig

logger = logging.getLogger(__name__)


async def _ensure_template(
    image: str,
    cpu_count: int,
    memory_mb: int,
    proxy: str | None = None,
) -> str:
    """Return the deterministic name of the image-backed template, building it if it is not ready.

    Concurrent callers on the same event loop share one in-flight attempt per
    template name and connection. The attempt runs on the event loop through the
    async E2B SDK, so waiting for a build holds no executor thread. A cancelled
    caller does not cancel the shared attempt; callers that joined an attempt that
    fails receive its error, and the next caller starts a new attempt. The whole
    attempt is bounded by ``SANDBOX_STARTUP_TIMEOUT`` (``<= 0`` disables the bound,
    as for ``start()``).
    """
    image = image.strip()
    if not image:
        raise ValueError("image must be a non-empty string")

    slug = re.sub(r"[^a-z0-9_-]+", "-", image.lower()).strip("-_")[:64] or "image"
    digest = hashlib.sha256(image.encode()).hexdigest()[:12]
    name = f"uni-agent-{slug}-{cpu_count}c-{memory_mb}mb-{digest}"
    connection_kwargs = {"proxy": proxy} if proxy is not None else {}

    timeout = _env_number("SANDBOX_STARTUP_TIMEOUT", _DEFAULT_STARTUP_TIMEOUT)
    attempts = _attempts_for_running_loop()
    key = (name, proxy)
    attempt = attempts.get(key)
    if attempt is None:
        build = _build_if_not_ready(image, name, cpu_count, memory_mb, connection_kwargs)
        attempt = asyncio.ensure_future(asyncio.wait_for(build, timeout if timeout > 0 else None))
        attempts[key] = attempt
        attempt.add_done_callback(lambda done: _forget_attempt(attempts, key, done))
    try:
        return await asyncio.shield(attempt)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"E2B template {name!r} is not ready within SANDBOX_STARTUP_TIMEOUT={timeout:g}s") from exc


async def ensure_template(image: str, cpu_count: int = 2, memory_mb: int = 4096, proxy: str | None = None) -> str:
    """Build the E2B template that ``E2BSandbox(image=...)`` uses, if it is not ready yet.

    Returns the template name. Await it ahead of a run (for example from
    ``examples/quickstart/sandbox/prebuild_e2b_templates.py``) so that sandbox
    startup does not wait for first-time template builds. The defaults match
    ``E2BSandbox``.
    """
    return await _ensure_template(image, cpu_count, memory_mb, proxy)


# ponytail: attempts are shared per event loop only. Workers in other processes
# may build the same name concurrently; E2B then rejects all but one alias
# insert, and the loser waits for the winning build. An attempt keeps running
# after its last caller is cancelled, at most until SANDBOX_STARTUP_TIMEOUT, so
# a retried start() joins it; the remote build is not stopped either way.
_attempts_by_loop: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[str, str | None], asyncio.Future]
] = weakref.WeakKeyDictionary()
_BUILD_POLL_INTERVAL = 2.0


def _attempts_for_running_loop() -> dict[tuple[str, str | None], asyncio.Future]:
    return _attempts_by_loop.setdefault(asyncio.get_running_loop(), {})


def _forget_attempt(attempts: dict, key: tuple[str, str | None], done: asyncio.Future) -> None:
    if attempts.get(key) is done:
        del attempts[key]
    if not done.cancelled():
        done.exception()  # retrieved here, so a failed attempt with no callers left is not logged as unhandled


async def _build_if_not_ready(
    image: str, name: str, cpu_count: int, memory_mb: int, connection_kwargs: dict[str, Any]
) -> str:
    import e2b

    if await _wait_for_build(name, connection_kwargs) == "ready":
        return name
    try:
        await e2b.AsyncTemplate.build(
            e2b.AsyncTemplate().from_image(image),
            name,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            on_build_logs=None,
            **connection_kwargs,
        )
    except Exception as build_error:
        # Another process may have created the alias first and be building it.
        try:
            recovered = await _wait_for_build(name, connection_kwargs) == "ready"
        except Exception:
            logger.warning("E2B template %s: status check after a failed build also failed", name, exc_info=True)
            recovered = False
        if recovered:
            return name
        raise build_error
    return name


async def _build_status(name: str, connection_kwargs: dict[str, Any]) -> str | None:
    """Status of the build behind the template's ``default`` tag, or ``None`` if there is none.

    The alias and the ``default`` tag exist as soon as a build is requested, even
    while it is running or after it failed, so only the build status says whether
    sandboxes can be created from the template.
    """
    import e2b

    template_id = await _template_id(name, connection_kwargs)
    if template_id is None:
        return None
    if template_id == "":
        raise RuntimeError(
            f"E2B template name {name!r} belongs to another team, so its build cannot be verified; "
            "use another image or set sandbox_kwargs.template"
        )
    tags = await e2b.AsyncTemplate.get_tags(name, **connection_kwargs)
    build_id = next((t.build_id for t in tags if t.tag == "default"), None)
    if build_id is None:
        return None
    try:
        build = await e2b.AsyncTemplate.get_build_status(
            SimpleNamespace(template_id=template_id, build_id=build_id), **connection_kwargs
        )
    except Exception as exc:
        if "not found" in str(exc).lower():
            # Right after a build is requested its status can briefly be missing.
            return "building"
        raise
    status = build.status
    return str(getattr(status, "value", status)).lower()


async def _template_id(name: str, connection_kwargs: dict[str, Any]) -> str | None:
    """Template ID behind alias ``name``; ``None`` if there is no such alias, ``""`` if it is not ours.

    The build-status endpoint needs the template ID; the SDK has no public lookup
    from name to ID, so this uses the same generated API call as
    ``AsyncTemplate.exists``.
    """
    from e2b.api.client.api.templates import get_templates_aliases_alias
    from e2b.api.client_async import get_api_client
    from e2b.connection_config import ConnectionConfig

    client = get_api_client(ConnectionConfig(**connection_kwargs))
    res = await get_templates_aliases_alias.asyncio_detailed(alias=name, client=client)
    if res.status_code == 404:
        return None
    if res.status_code == 403:
        return ""
    if res.status_code >= 300 or res.parsed is None:
        raise RuntimeError(f"E2B template alias lookup for {name!r} failed: HTTP {res.status_code}")
    return str(res.parsed.template_id)


async def _wait_for_build(name: str, connection_kwargs: dict[str, Any]) -> str | None:
    """Wait while the template's build is running; return its final status (``None`` if no build)."""
    while (status := await _build_status(name, connection_kwargs)) in ("building", "waiting"):
        await asyncio.sleep(_BUILD_POLL_INTERVAL)
    return status


@register_sandbox("e2b")
class E2BSandbox(Sandbox):
    """Create an E2B sandbox from an image-backed template."""

    def __init__(
        self,
        *,
        image: str | None = None,
        template: str | None = None,
        cpu_count: int = 2,
        memory_mb: int = 4096,
        runtime_timeout: float = 3600.0,
        user: str | None = "root",
        metadata: dict[str, str] | None = None,
        envs: dict[str, str] | None = None,
        proxy: str | None = None,
        secure: bool = True,
        allow_internet_access: bool = True,
        network: dict[str, Any] | None = None,
        lifecycle: dict[str, Any] | None = None,
        file_request_timeout: float = 600.0,
        image_mounts: list[ImageMount] | None = None,
        executable_paths: dict[str, str] | None = None,
    ) -> None:
        if runtime_timeout <= 0:
            raise ValueError("runtime_timeout must be positive")
        if file_request_timeout <= 0:
            raise ValueError("file_request_timeout must be positive")
        if cpu_count < 1:
            raise ValueError("cpu_count must be at least 1")
        if memory_mb < 512:
            raise ValueError("memory_mb must be at least 512")
        if image is not None and not image.strip():
            raise ValueError("image must be non-empty when provided")
        if template is not None and not template.strip():
            raise ValueError("template must be non-empty when provided")
        if image is None and template is None:
            raise ValueError("either image or template must be provided")
        if proxy is not None and not proxy.strip():
            raise ValueError("proxy must be non-empty when provided")
        if image_mounts:
            raise NotImplementedError(
                "E2BSandbox does not support image mounts; build a composite E2B template instead"
            )

        self.image = image.strip() if image is not None else None
        self.template = template.strip() if template is not None else None
        self.cpu_count = cpu_count
        self.memory_mb = memory_mb
        self.runtime_timeout = float(runtime_timeout)
        self.user = user
        self.metadata = dict(metadata or {})
        self.envs = dict(envs or {})
        self.proxy = proxy.strip() if proxy is not None else None
        self.secure = secure
        self.allow_internet_access = allow_internet_access
        self.network = network
        self.lifecycle = lifecycle
        self.file_request_timeout = float(file_request_timeout)
        self.image_mounts = list(image_mounts or [])
        self.executable_paths = dict(executable_paths or {})

        self._sandbox: Any | None = None
        self.sandbox_id: str | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> E2BSandbox:
        return cls(
            image=config.image,
            runtime_timeout=config.runtime_timeout,
            image_mounts=config.image_mounts,
            executable_paths=config.executable_paths,
            **config.sandbox_kwargs,
        )

    async def start(self) -> None:
        if self._sandbox is not None:
            return

        from e2b import AsyncSandbox

        template = self.template
        if template is None:
            assert self.image is not None
            template = await _ensure_template(self.image, self.cpu_count, self.memory_mb, self.proxy)

        create_kwargs: dict[str, Any] = {
            "template": template,
            "timeout": int(self.runtime_timeout),
            "metadata": self.metadata,
            "envs": self.envs,
            "secure": self.secure,
            "allow_internet_access": self.allow_internet_access,
        }
        if self.network is not None:
            create_kwargs["network"] = self.network
        if self.lifecycle is not None:
            create_kwargs["lifecycle"] = self.lifecycle
        if self.proxy is not None:
            create_kwargs["proxy"] = self.proxy

        sandbox = await AsyncSandbox.create(**create_kwargs)
        self._sandbox = sandbox
        self.sandbox_id = str(sandbox.sandbox_id)
        await self._setup_executable_paths(self.executable_paths)

    async def stop(self) -> None:
        sandbox, self._sandbox = self._sandbox, None
        sandbox_id, self.sandbox_id = self.sandbox_id, None
        if sandbox is None:
            return
        try:
            await sandbox.kill()
        except Exception as exc:
            logger.warning("Failed to kill E2B sandbox %s: %s", sandbox_id or "?", exc)

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise RuntimeError("E2BSandbox not started; call start() first")
        return self._sandbox

    async def is_alive(self) -> bool:
        if self._sandbox is None:
            return False
        try:
            return bool(await self._sandbox.is_running())
        except Exception:
            return False

    def _is_timeout_error(self, exc: BaseException) -> bool:
        return type(exc).__name__ == "TimeoutException" or super()._is_timeout_error(exc)

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        from e2b import CommandExitException

        # E2B executes command strings through a shell. ``exec`` replaces that
        # wrapper with the requested process so handle.kill() targets the real
        # command rather than leaving an orphaned child behind.
        command = f"exec {shlex.join(argv)}"
        command_timeout = timeout if timeout is not None and timeout > 0 else 0
        handle = await self._require_sandbox().commands.run(
            command,
            background=True,
            timeout=command_timeout,
            user=self.user,
            envs=env,
            cwd=workdir,
        )
        try:
            result = await handle.wait()
        except CommandExitException as exc:
            return ExecResult(
                exit_code=int(exc.exit_code),
                stdout=_to_str(exc.stdout),
                stderr=_to_str(exc.stderr),
            )
        except BaseException:
            try:
                await asyncio.shield(handle.kill())
            except Exception:
                logger.warning("Failed to kill E2B command in sandbox %s", self.sandbox_id)
            raise
        return ExecResult(
            exit_code=int(result.exit_code or 0),
            stdout=_to_str(result.stdout),
            stderr=_to_str(result.stderr),
        )

    async def read_file(self, path: str) -> bytes:
        data = await self._require_sandbox().files.read(
            path,
            format="bytes",
            user=self.user,
            request_timeout=self.file_request_timeout,
        )
        return bytes(data)

    async def write_file(self, path: str, content: bytes | str) -> None:
        await self._require_sandbox().files.write(
            path,
            content,
            user=self.user,
            request_timeout=self.file_request_timeout,
            use_octet_stream=isinstance(content, bytes),
        )

    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        with Path(local_file).open("rb") as stream:
            await self._require_sandbox().files.write(
                remote_file,
                stream,
                user=self.user,
                request_timeout=self.file_request_timeout,
                use_octet_stream=True,
            )

    async def download_file(self, remote_file: str, local_file: Path | str) -> None:
        data = await self.read_file(remote_file)
        destination = Path(local_file)

        def _write() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

        await asyncio.to_thread(_write)

    async def expose_port(self, port: int) -> str:
        host = self._require_sandbox().get_host(port)
        if inspect.isawaitable(host):
            host = await host
        host = str(host)
        return host if "://" in host else f"https://{host}"
