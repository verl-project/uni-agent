from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import ImageMount, SandboxConfig

logger = logging.getLogger(__name__)

#: Machine states in which a boxd machine still answers exec (standby and hibernated wake on exec).
_ALIVE_STATES = frozenset({"pending", "starting", "running", "suspended", "hibernated", "migrating"})
#: Seconds between the first delete attempt and the retry in ``stop()``.
_DELETE_RETRY_DELAY = 1.0
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _positive(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number of seconds, got {value!r}")
    return value


@register_sandbox("boxd")
class BoxdSandbox(Sandbox):
    """Run a sandbox on a boxd machine (https://boxd.sh?ref=uni-agent).

    Each sandbox is one new machine, created isolated: it does not join the
    org's private network and holds no boxd credentials. With ``image`` the
    task image runs as a container inside the machine through its Docker
    daemon, which uses the containerd image store, so ``image_mounts`` work as
    with the Docker provider. Without ``image`` commands run on the machine
    itself as its default user. ``stop()`` deletes the machine, and the
    machine is created with a delete-after timer of ``runtime_timeout`` so
    boxd removes it even if ``stop()`` never runs.
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        runtime_timeout: float = 3600.0,
        api_key: str | None = None,
        org: str | None = None,
        region: str | None = None,
        from_snapshot: str | None = None,
        vcpu: int | None = None,
        memory: str | int | None = None,
        name_prefix: str = "uni-agent",
        labels: dict[str, str] | None = None,
        auto_suspend_timeout: int = 0,
        ready_timeout: float = 180.0,
        pull_policy: str = "missing",
        pull_timeout: float | None = 600.0,
        run_args: list[str] | None = None,
        image_mounts: list[ImageMount] | None = None,
        executable_paths: dict[str, str] | None = None,
    ) -> None:
        if not math.isfinite(runtime_timeout) or runtime_timeout <= 0:
            raise ValueError("runtime_timeout must be finite and positive")
        if image is not None and not image.strip():
            raise ValueError("image must be non-empty when provided")
        if image_mounts and image is None:
            raise ValueError("BoxdSandbox image_mounts require an image")
        if executable_paths and image is None:
            raise ValueError("BoxdSandbox executable_paths require an image")
        if pull_policy not in {"always", "missing", "never"}:
            raise ValueError("pull_policy must be one of: 'always', 'missing', 'never'")
        if auto_suspend_timeout < 0:
            raise ValueError("auto_suspend_timeout must be zero (off) or a positive number of seconds")
        if from_snapshot is not None and (vcpu is not None or memory is not None):
            raise ValueError("from_snapshot keeps the snapshot's machine size; drop vcpu and memory")
        if not name_prefix or not name_prefix.strip():
            raise ValueError("name_prefix must be non-empty")

        self.image = image.strip() if image is not None else None
        self.runtime_timeout = float(runtime_timeout)
        self.api_key = api_key
        self.org = org
        self.region = region
        self.from_snapshot = from_snapshot
        self.vcpu = vcpu
        self.memory = memory
        self.name_prefix = name_prefix.strip()
        self.labels = dict(labels or {})
        self.auto_suspend_timeout = int(auto_suspend_timeout)
        if ready_timeout is None:
            raise ValueError("ready_timeout must be a positive number of seconds")
        self.ready_timeout = _positive("ready_timeout", ready_timeout)
        self.pull_policy = pull_policy
        self.pull_timeout = _positive("pull_timeout", pull_timeout)
        self.run_args = list(run_args or [])
        self.image_mounts = list(image_mounts or [])
        self.executable_paths = dict(executable_paths or {})

        self._client: Any | None = None
        self.machine_id: str | None = None
        self.machine_name: str | None = None
        self._container: str | None = None
        #: The container's working directory, which relative file paths resolve against.
        self._container_workdir: str | None = None
        #: Name of a machine whose create call has not returned an id yet.
        self._pending_name: str | None = None
        #: Machines (ids, or names of interrupted creates) that an earlier ``stop()`` could not
        #: confirm deleted. They are kept apart from ``machine_id`` so ``start()`` never reuses one.
        self._cleanup_ids: list[str] = []
        self._cleanup_names: list[str] = []

    @classmethod
    def from_config(cls, config: SandboxConfig) -> BoxdSandbox:
        return cls(
            image=config.image,
            runtime_timeout=config.runtime_timeout,
            image_mounts=config.image_mounts,
            executable_paths=config.executable_paths,
            **config.sandbox_kwargs,
        )

    # ----- control plane -----
    def _make_client(self) -> Any:
        api_key = self.api_key or os.getenv("BOXD_API_KEY")
        if not api_key:
            raise ValueError(
                "BoxdSandbox needs a boxd API key: set BOXD_API_KEY or sandbox_kwargs.api_key "
                "(create one with `boxd auth keys create <name> --org <org>`)"
            )
        from boxd import AsyncBoxd

        return AsyncBoxd(api_key=api_key, region=self.region)

    async def start(self) -> None:
        if self.machine_id is not None:
            return
        if self._client is None:
            self._client = self._make_client()
        machines = self._client.machines

        name = f"{self.name_prefix}-{uuid.uuid4().hex[:12]}"
        create_kwargs: dict[str, Any] = {
            "org": self.org,
            "isolated": True,
            "labels": {"app": "uni-agent", **self.labels},
            "auto_suspend_timeout": self.auto_suspend_timeout,
            "delete_after": math.ceil(self.runtime_timeout),
            "region": self.region,
        }
        if self.from_snapshot is not None:
            create_kwargs["from_snapshot"] = self.from_snapshot
        if self.vcpu is not None:
            create_kwargs["vcpu"] = self.vcpu
        if self.memory is not None:
            create_kwargs["memory"] = self.memory

        # Keep the name while create is in flight, so stop() can find a machine
        # whose create was cancelled before it returned an id.
        self._pending_name = name
        machine = await machines.create(name, **create_kwargs)
        # Record the id first: a failed start is cleaned up by stop().
        self.machine_id = str(machine.id)
        self.machine_name = name
        self._pending_name = None
        await machines.wait_until_ready(self.machine_id, timeout=self.ready_timeout, region=self.region)

        if self.image is not None:
            await self._start_container()
        await self._setup_executable_paths(self.executable_paths)

    async def _machine_exec(
        self,
        command: list[str],
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecResult:
        """Run ``command`` on the machine itself and return its result."""
        if self._client is None or self.machine_id is None:
            raise RuntimeError("BoxdSandbox not started; call start() first")
        from boxd.errors import BoxdError

        for key in env or {}:
            if not _ENV_NAME_RE.fullmatch(key):
                raise ValueError(f"invalid environment variable name {key!r}")
        started = time.monotonic()
        try:
            result = await self._client.machines.exec(
                self.machine_id,
                command,
                env=env or None,
                cwd=cwd,
                timeout=timeout,
                region=self.region,
            )
        except BoxdError as exc:
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(f"boxd exec exceeded {timeout:g}s") from exc
            raise
        exit_code = result.exit_code
        return ExecResult(
            exit_code=int(exit_code if exit_code is not None else -1),
            stdout=_to_str(result.stdout),
            stderr=_to_str(result.stderr),
        )

    async def _checked(self, command: list[str], what: str, *, timeout: float | None = None) -> ExecResult:
        result = await self._machine_exec(command, timeout=timeout)
        if result.exit_code != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"{what} failed on boxd machine {self.machine_id}: {detail}")
        return result

    async def _prepare_image(self, image: str) -> None:
        """Apply ``pull_policy`` to one task or mounted image inside the machine."""
        if self.pull_policy != "always":
            if (await self._machine_exec(["docker", "image", "inspect", image])).exit_code == 0:
                return
            if self.pull_policy == "never":
                raise RuntimeError(f"Docker image {image!r} is not available on boxd machine {self.machine_id}")
        try:
            await self._checked(["docker", "pull", "-q", image], f"pulling {image!r}", timeout=self.pull_timeout)
        except TimeoutError as exc:
            raise TimeoutError(f"Pulling Docker image {image!r} exceeded pull_timeout={self.pull_timeout:g}s") from exc

    async def _start_container(self) -> None:
        assert self.image is not None
        for image in dict.fromkeys([self.image, *(mount.image for mount in self.image_mounts)]):
            await self._prepare_image(image)
        name = f"uni-agent-{uuid.uuid4().hex[:12]}"
        args = ["docker", "run", "-d", "--name", name, "--pull", "never", "--entrypoint", "sleep"]
        for mount in self.image_mounts:
            args.extend(["--mount", f"type=image,source={mount.image},destination={mount.mount_path}"])
        args.extend(self.run_args)
        args.extend([self.image, str(math.ceil(self.runtime_timeout))])
        await self._checked(args, f"starting a container from {self.image!r}")
        self._container = name

    async def stop(self) -> None:
        """Delete the machine. A machine whose delete is not confirmed is kept for the next ``stop()``."""
        client, self._client = self._client, None
        machine_id, self.machine_id = self.machine_id, None
        pending_name, self._pending_name = self._pending_name, None
        ids, self._cleanup_ids = self._cleanup_ids, []
        names, self._cleanup_names = self._cleanup_names, []
        self._container = None
        self._container_workdir = None
        if machine_id is not None:
            ids.append(machine_id)
        elif pending_name is not None:
            names.append(pending_name)
        if not ids and not names:
            if client is not None:
                await self._close_client(client)
            return
        if client is None:
            try:
                client = self._make_client()
            except Exception:
                self._cleanup_ids, self._cleanup_names = ids, names
                raise
        try:
            for name in names:
                found = await self._find_machine(client, name)
                if found is not None:
                    ids.append(found)
            for mid in ids:
                if not await self._delete_machine(client, mid):
                    self._cleanup_ids.append(mid)
        finally:
            await self._close_client(client)

    async def _close_client(self, client: Any) -> None:
        try:
            await client.close()
        except Exception:
            logger.debug("Failed to close the boxd client", exc_info=True)

    async def _find_machine(self, client: Any, name: str) -> str | None:
        """Return the id of machine ``name`` if it exists (a create interrupted before it returned).

        If the lookup fails for another reason than the machine not existing, the name is kept
        for the next ``stop()``.
        """
        from boxd.errors import NotFoundError

        try:
            machine = await client.machines.get(name, region=self.region)
        except NotFoundError:
            return None
        except Exception as exc:
            logger.warning("Failed to look up boxd machine %s: %s", name, exc)
            self._cleanup_names.append(name)
            return None
        return str(machine.id)

    async def _delete_machine(self, client: Any, machine_id: str) -> bool:
        """Delete the machine; after an error, check whether it is gone and retry once if not.

        Return whether the machine is confirmed gone.
        """
        from boxd.errors import NotFoundError

        error: Exception | None = None
        for attempt in range(2):
            try:
                await client.machines.delete(machine_id, region=self.region)
                return True
            except NotFoundError:
                return True
            except Exception as exc:
                error = exc
            try:
                await client.machines.get(machine_id, region=self.region)
            except NotFoundError:
                return True
            except Exception:
                pass
            if attempt == 0:
                await asyncio.sleep(_DELETE_RETRY_DELAY)
        logger.warning("Failed to delete boxd machine %s: %s (a later stop() tries again)", machine_id, error)
        return False

    async def is_alive(self) -> bool:
        if self._client is None or self.machine_id is None:
            return False
        try:
            machine = await self._client.machines.get(self.machine_id, region=self.region)
        except Exception:
            return False
        return str(machine.status) in _ALIVE_STATES

    # ----- data plane -----
    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if self._container is None:
            return await self._machine_exec(list(argv), timeout=timeout, env=env, cwd=workdir)
        args = ["docker", "exec"]
        if workdir:
            args.extend(["--workdir", workdir])
        # ``--env NAME`` takes the value from the docker client's environment,
        # which the machine exec sets, so values are not in the docker exec arguments.
        for key in env or {}:
            args.extend(["--env", key])
        args.append(self._container)
        if timeout is not None:
            # The machine exec stops the docker client on timeout, not the process in the
            # container, so bound it there too with the image's ``timeout``; the +2s margin
            # lets the outer timer fire first.
            args.extend(["timeout", "-k", "10", str(math.ceil(timeout) + 2)])
        args.extend(argv)
        return await self._machine_exec(args, timeout=timeout, env=env)

    def _staging_path(self) -> str:
        return f"/tmp/uni-agent-xfer-{uuid.uuid4().hex}"

    async def _remove_staged(self, staged: str) -> None:
        """Best-effort cleanup that never replaces the caller's exception."""
        try:
            await self._machine_exec(["rm", "-rf", staged])
        except Exception:
            logger.debug("Failed to remove %s on boxd machine %s", staged, self.machine_id, exc_info=True)

    async def _container_path(self, path: str) -> str:
        """Return ``path`` in the container as an absolute path.

        ``docker cp`` resolves a relative path against ``/``; a relative path here is under the
        container's working directory, as for ``exec``.
        """
        if PurePosixPath(path).is_absolute():
            return path
        if self._container_workdir is None:
            assert self._container is not None
            result = await self._checked(
                ["docker", "inspect", "--format", "{{.Config.WorkingDir}}", self._container],
                "reading the container's working directory",
            )
            self._container_workdir = result.stdout.strip() or "/"
        return str(PurePosixPath(self._container_workdir) / path)

    async def read_file(self, path: str) -> bytes:
        if self._container is None:
            return await self._download_bytes(path)
        source = await self._container_path(path)
        staged = self._staging_path()
        try:
            # -L copies what a symlink points to, as reading the path in the container would.
            await self._checked(["docker", "cp", "-L", f"{self._container}:{source}", staged], f"read_file {path!r}")
            return await self._download_bytes(staged)
        finally:
            await self._remove_staged(staged)

    async def write_file(self, path: str, content: bytes | str) -> None:
        data = content.encode("utf-8") if isinstance(content, str) else content
        if self._container is None:
            parent = str(PurePosixPath(path).parent)
            await self._checked(["mkdir", "-p", parent], f"write_file {path!r}")
            await self._upload_bytes(path, data)
            return
        target = await self._container_path(path)
        # docker cp copies a file into a destination directory under the file's own name.
        if (await self.exec(["test", "-d", target])).exit_code == 0:
            raise RuntimeError(f"write_file {path!r} failed: it is a directory")
        created = await self.exec(["mkdir", "-p", str(PurePosixPath(target).parent)])
        if created.exit_code != 0:
            raise RuntimeError(f"write_file {path!r} failed: {created.stderr.strip()}")
        staged = self._staging_path()
        try:
            await self._upload_bytes(staged, data)
            await self._checked(["docker", "cp", staged, f"{self._container}:{target}"], f"write_file {path!r}")
        finally:
            await self._remove_staged(staged)

    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        data = await asyncio.to_thread(Path(local_file).read_bytes)
        await self.write_file(remote_file, data)

    async def download_file(self, remote_file: str, local_file: Path | str) -> None:
        data = await self.read_file(remote_file)
        destination = Path(local_file)

        def _write() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

        await asyncio.to_thread(_write)

    async def _upload_bytes(self, path: str, data: bytes) -> None:
        if self._client is None or self.machine_id is None:
            raise RuntimeError("BoxdSandbox not started; call start() first")
        await self._client.machines.files.upload(self.machine_id, path, data, region=self.region)

    async def _download_bytes(self, path: str) -> bytes:
        if self._client is None or self.machine_id is None:
            raise RuntimeError("BoxdSandbox not started; call start() first")
        return bytes(await self._client.machines.files.download(self.machine_id, path, region=self.region))
