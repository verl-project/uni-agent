import asyncio
import concurrent.futures
import math
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import Command, CreateBashSessionRequest, IsAliveResponse, UploadRequest
from swerex.utils.wait import _wait_until_alive

from uni_agent.async_logging import get_logger
from uni_agent.deployment.config import LocalDeploymentConfig
from uni_agent.deployment.remote_runtime import RemoteRuntime as LocalRuntime
from uni_agent.deployment.remote_runtime import RemoteRuntimeConfig as LocalRuntimeConfig

_APPTAINER_RUNTIMES = {"apptainer", "singularity"}
_CONTAINER_RUNTIME_ENV_VARS = ("UNI_AGENT_CONTAINER_RUNTIME", "LOCAL_CONTAINER_RUNTIME")
_DEFAULT_CONTAINER_RUNTIME_CANDIDATES = ("apptainer", "singularity", "docker", "podman")
_IMAGE_URI_PREFIXES = (
    "docker://",
    "oras://",
    "library://",
    "shub://",
    "instance://",
    "http://",
    "https://",
    "file://",
)

_STARTCAP = 16
_BUDGET = 600.0
_LOGWAIT = 30.0
_CLEANWAIT = 60.0
_CLOSEWAIT = 10.0
_RMWAIT = 60.0
# A cancelled coroutine cannot stop a running subprocess.run, so this timeout is its only deadline.
_CLIWAIT = 600.0
_PROCWAIT = 10.0
_REAPWAIT = 2.0
_DELWAIT = 10.0
_STARTPAD = 120.0
_RETRYCAP = 10
_BACKBASE = 2
_RMTRIES = 2
_SLOWQUEUE = 1.0
_RMPAUSE = 1.0
_LOGTAIL = 4000
_IDLEN = 12
# The runtime prints a container's id in full, so anything shorter is a partial write, not an id --
# claiming one would be a guess.
_IDRE = re.compile(r"^[0-9a-f]{64}$")
# A random, non-sensitive tag on every container we launch, so one whose id we never learned can still
# be found. Never the auth token, which must not reach a `docker inspect` away from us.
_OWNERTAG = "uni-agent.owner"
# Keyed on the loop itself, so an entry dies with it; the value is weak so an idle semaphore can go.
_LIMITERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
# Dedicated CLI pool: stranded container runs must not exhaust asyncio's default executor.
_CLIPOOL: concurrent.futures.ThreadPoolExecutor | None = None
_POOLLOCK = threading.Lock()


def _env_cap() -> int:
    raw = os.getenv("LOCAL_MAX_STARTING_PER_WORKER", "")
    if not raw:
        return _STARTCAP
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"invalid LOCAL_MAX_STARTING_PER_WORKER={raw!r}; expected a positive integer")
    return value


def _clipool() -> concurrent.futures.ThreadPoolExecutor:
    global _CLIPOOL
    with _POOLLOCK:
        if _CLIPOOL is None:
            _CLIPOOL = concurrent.futures.ThreadPoolExecutor(max_workers=_env_cap(), thread_name_prefix="local-cli")
        return _CLIPOOL


def _get_limiter() -> asyncio.Semaphore:
    """Per-event-loop STARTING semaphore, lazy so the env var is read at first use."""
    loop = asyncio.get_running_loop()
    ref = _LIMITERS.get(loop)
    sem = ref() if ref is not None else None
    if sem is None:
        sem = asyncio.Semaphore(_env_cap())
        _LIMITERS[loop] = weakref.ref(sem)
    return sem


def _get_budget() -> float:
    raw = os.getenv("LOCAL_INIT_WALL_BUDGET", "")
    if not raw:
        return _BUDGET
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid LOCAL_INIT_WALL_BUDGET={raw!r}; expected a finite positive number")
    return value


# Retry-delay seam so tests can skip real backoff without patching asyncio.
_retrydelay = asyncio.sleep


# Redact live auth tokens by value before logging or raising CLI errors.
class CliError(RuntimeError):
    """A container CLI failure that never carries the command it ran."""


class CliTimeout(CliError):
    """The CLI was killed before it returned. It may have created a container first."""


class CliNoRun(CliError):
    """The runtime binary is not there, so nothing ran and nothing was created."""


def _confirmed_gone(result: subprocess.CompletedProcess[str]) -> bool:
    stderr = (result.stderr or "").lower()
    return result.returncode == 0 or "no such container" in stderr or "no container with" in stderr


def _valid_id(text: str) -> str:
    """The text if it is a full runtime container id, else "" -- a partial write is not an id."""
    candidate = text.strip()
    return candidate if _IDRE.match(candidate) else ""


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()
    return sanitized or "uni-agent-local"


def _is_running_in_container() -> bool:
    return Path("/.dockerenv").exists()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _default_container_runtime() -> str:
    for env_var in _CONTAINER_RUNTIME_ENV_VARS:
        runtime = os.getenv(env_var)
        if runtime:
            return runtime
    for runtime in _DEFAULT_CONTAINER_RUNTIME_CANDIDATES:
        runtime_path = shutil.which(runtime)
        if runtime_path:
            return runtime_path
    return "apptainer"


def _runtime_basename(runtime: str) -> str:
    return Path(runtime).name.lower()


def _is_apptainer_runtime(runtime: str) -> bool:
    return _runtime_basename(runtime) in _APPTAINER_RUNTIMES


def _normalize_apptainer_image(image: str) -> str:
    if image.startswith(_IMAGE_URI_PREFIXES):
        return image
    image_path = Path(image)
    if image.startswith(("/", ".")) or image_path.exists() or image_path.suffix in {".sif", ".sqsh", ".img"}:
        return image
    return f"docker://{image}"


class LocalDeployment(AbstractDeployment):
    """A local container sandbox. Concurrency is across deployments (the STARTING limiter caps it);
    one deployment's lifecycle is serial, so overlapping start/stop calls -- or a start over
    resources no stop reclaimed -- raise. An uncancellable container CLI call holds a single slot
    until someone takes its result back: nothing may start, retry or submit over it, and stop()
    drains it first."""

    def __init__(self, run_id: str, **kwargs: Any):
        self.run_id = run_id
        config_kwargs = dict(kwargs)
        if not config_kwargs.get("container_runtime"):
            config_kwargs["container_runtime"] = _default_container_runtime()
        self._config = LocalDeploymentConfig(**config_kwargs)
        self._runtime: LocalRuntime | None = None
        self.logger = get_logger("deployment", run_id)
        self._hooks = CombinedDeploymentHook()
        self._container_id: str | None = None
        self._secrets: set[str] = set()  # live auth tokens
        # (cidfile, owner, settled) of a run that may have created a container but named no id.
        # Nothing may start over it. `settled` says the runtime ran, returned and reported failure --
        # only then does "no container carries this owner" mean it, rather than "not yet".
        self._orphan: tuple[Path, str, bool] | None = None
        # The one container this deployment owns, as (id, name); the state machine allows no more.
        self._owned: tuple[str, str] | None = None
        # A late container result self-reaps after its attempt is abandoned.
        self._abandoned = False
        self._lock = threading.Lock()
        self._busy: str | None = None
        self._server_process: subprocess.Popen[str] | None = None
        self._future: concurrent.futures.Future | None = None
        self._server_log_path: Path | None = None
        self._server_log_handle: Any | None = None

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: LocalDeploymentConfig, run_id: str | None = None) -> Self:
        if not run_id:
            run_id = str(uuid.uuid4())
        config_kwargs = config.model_dump()
        if "container_runtime" not in config.model_fields_set:
            config_kwargs["container_runtime"] = _default_container_runtime()
        return cls(run_id=run_id, **config_kwargs)

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None:
            raise DeploymentNotStartedError("Runtime not started")
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float) -> IsAliveResponse:
        try:
            return await _wait_until_alive(self.is_alive, timeout=timeout, function_timeout=0.5)
        except TimeoutError as exc:
            # start() owns diagnostics and failure logging.
            raise TimeoutError(f"runtime not alive within {timeout:.0f}s") from exc

    def _get_token(self) -> str:
        return str(uuid.uuid4())

    def _redact(self, text: str) -> str:
        with self._lock:
            secrets = tuple(self._secrets)
        for secret in secrets:
            text = text.replace(secret, "***")
        return text

    def _why(self, exc: BaseException) -> str:
        """Render an exception into a redacted summary for a record. See the note above CliError."""
        return self._redact(f"{type(exc).__name__}: {exc}")

    def _runtime_exec(
        self, args: list[str], check: bool = True, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        action = f"{_runtime_basename(args[0])} {args[1]}"
        target = args[args.index("--name") + 1] if "--name" in args else ""
        self._say("debug", f"container runtime call: {action} {target}".rstrip())
        if timeout is None:
            timeout = _CLIWAIT
        try:
            result = subprocess.run(args, check=False, text=True, capture_output=True, timeout=timeout)
        except FileNotFoundError:
            failure: Exception = CliNoRun(f"container runtime {self._config.container_runtime!r} not found in PATH")
        except subprocess.TimeoutExpired:
            failure = CliTimeout(f"{action}: timed out after {timeout:.0f}s")
        else:
            if check and result.returncode != 0:
                detail = (
                    self._redact(result.stderr or "")
                    or self._redact(result.stdout or "")
                    or f"exit code {result.returncode}"
                ).strip()
                raise CliError(f"{action}: {detail}")
            return result
        # Raised out here, never inside the except: `raise ... from None` only hides the original,
        # which stays on __context__ with the whole argv -- and the auth token -- in .cmd.
        raise failure

    def _own_reap(self, container_id: str, name: str) -> None:
        """Own a confirmed id, or try to reap it if abandoned (a failed reap re-owns it for the sweep)."""
        with self._lock:
            reap = self._abandoned
            if not reap:
                self._owned = (container_id, name)
                self._container_id = container_id  # so diagnostics can fetch its logs
        if reap:
            self._reap_late(container_id, name)

    def _read_cid(self, cidfile: Path) -> str:
        """The id in the cidfile, or "" -- unreadable, empty and absent are one answer here.

        None of them proves that no container exists: the daemon creates it before the CLI writes the
        file. Only the caller knows whether the call could have created one, so it does the judging
        and the logging; this only reports what is on disk.
        """
        try:
            return cidfile.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _drop_cid(self, cidfile: Path) -> None:
        try:
            cidfile.unlink(missing_ok=True)
            cidfile.parent.rmdir()
        except FileNotFoundError:
            pass  # already gone: the finalizer drops the file of an orphan it still keeps the barrier for
        except OSError as exc:
            self._say("warning", "failed to remove cidfile {}: {}", cidfile, self._why(exc))

    def _claim_cid(self, cidfile: Path, name: str, fallback: str, *, owner: str, outcome: str) -> None:
        """Commit the id, or fall back to the owner tag, or raise the barrier. No exit code proves a
        container was not created; an empty tag query is proof only for outcome=="failed" (the runtime
        ran, returned, reported failure) -- every other outcome leaves the question open."""
        container_id = _valid_id(self._read_cid(cidfile)) or _valid_id(fallback)
        if container_id:
            self._own_reap(container_id, name)
            self._orphan = None
            self._drop_cid(cidfile)
            return
        if outcome == "never-ran":
            self._drop_cid(cidfile)  # the runtime binary is not there, so nothing ran
            return
        settled = outcome == "failed"
        found = self._find_owned(owner, _RMWAIT)
        if found is not None and len(found) == 1:
            self._own_reap(found[0], name)
            self._orphan = None
            self._drop_cid(cidfile)
            return
        if found == [] and settled:
            # It ran, it failed, and it carries nothing of ours: a name clash, a missing image.
            self._drop_cid(cidfile)
            return
        self._orphan = (cidfile, owner, settled)
        self.logger.error(
            "the container runtime named no container (outcome={}); one may exist tagged {}={}, and "
            "only that tag or {} can still find it, so nothing may start over it",
            outcome,
            _OWNERTAG,
            owner,
            cidfile,
        )

    def _run_container(self, args: list[str], name: str) -> subprocess.CompletedProcess[str]:
        # The CLI thread owns the cidfile end to end: an undispatched run allocates nothing.
        cidfile = Path(tempfile.mkdtemp(prefix="uni-agent-cid-")) / "cid"
        owner = str(uuid.uuid4())
        result: subprocess.CompletedProcess[str] | None = None
        # Nothing has been handed to the runtime yet, so a failure here created nothing. The moment
        # the command is built and about to be run, that stops being true: from then on the default
        # is "unknown", because anything we did not foresee may land after the daemon created it.
        outcome = "never-ran"
        try:
            tagged = self._with_cid(args, cidfile, owner)
            outcome = "unknown"
            result = self._runtime_exec(tagged)
            outcome = "created"
        except CliNoRun:
            outcome = "never-ran"
            raise
        except CliTimeout:
            outcome = "killed"  # it may already have created the container
            raise
        except CliError:
            outcome = "failed"  # it ran and reported failure -- which still proves nothing on its own
            raise
        finally:
            stdout_id = result.stdout.strip() if result is not None else ""
            self._claim_cid(cidfile, name, stdout_id, owner=owner, outcome=outcome)
        if self._orphan is not None:
            # Nothing to inspect and nothing to connect to: we cannot name what we just made.
            raise CliError("container runtime returned without a container id")
        return result

    def _kill_server(self) -> None:
        """Clear the process reference only after confirmed exit; a survivor is kept for the next stop()."""
        proc = self._server_process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=_REAPWAIT)
        except Exception as exc:
            self._say(
                "warning",
                "could not reap the apptainer server process, keeping it for the next stop(): {}",
                self._why(exc),
            )
            return
        self._server_process = None

    def _has_cli(self) -> bool:
        """A CLI call nobody has taken the result back from -- running, or done but unconsumed."""
        return getattr(self, "_future", None) is not None

    def _probe_orphan(self, timeout: float) -> None:
        """Re-probe for the unnamed container (cidfile, then owner tag). Only an answer lifts the
        barrier: an id to remove by, or an empty that is authoritative (settled)."""
        if self._orphan is None:
            return
        cidfile, owner, settled = self._orphan
        container_id = _valid_id(self._read_cid(cidfile))
        found = [container_id] if container_id else self._find_owned(owner, timeout)
        if found is None:
            return  # _find_owned already recorded why it could not answer; the barrier stays
        if len(found) > 1 or (found == [] and not settled):
            self._say("warning", "owner {} still names no single container; keeping the barrier", owner)
            return
        if found:
            with self._lock:
                self._owned = (found[0], found[0][:_IDLEN])
        else:
            self._say("warning", "no container carries owner {}; releasing the barrier", owner)
        self._orphan = None
        self._drop_cid(cidfile)

    def _residue(self) -> str:
        """What an earlier attempt left behind. No container may be started over any of it: doing so
        is how one failed cleanup turns into two live containers."""
        if self._has_cli():
            return "a container runtime call is still in flight"
        if self._orphan is not None:
            return f"a container may exist that no id names; only owner {self._orphan[1]} can find it"
        with self._lock:
            owned = self._owned
        if owned is not None:
            return f"a container is still owned: {owned[1]}"
        if self._server_process is not None:
            return "an apptainer server process was not reaped"
        if self._server_log_handle is not None or self._server_log_path is not None:
            return "apptainer log resources were not cleaned"
        return ""

    def _has_cleanup(self) -> bool:
        return bool(
            getattr(self, "_owned", None)
            or getattr(self, "_orphan", None) is not None
            or self._has_cli()
            or getattr(self, "_server_process", None) is not None
            or getattr(self, "_runtime", None) is not None
            or getattr(self, "_server_log_handle", None) is not None
            or getattr(self, "_server_log_path", None) is not None
        )

    def _reap_late(self, container_id: str, name: str) -> None:
        cid = container_id[:_IDLEN]
        self._say("warning", "container runtime call finished after its attempt was abandoned; removing id={}", cid)
        if not self._force_remove(container_id):
            # _force_remove reported why; keep the id so the next sweep tries again.
            with self._lock:
                self._owned = (container_id, name)

    def _get_current_container_network(self) -> str | None:
        if not _is_running_in_container():
            return None

        container_id = os.getenv("HOSTNAME")
        if not container_id:
            return None

        try:
            result = self._runtime_exec(
                [
                    self._config.container_runtime,
                    "inspect",
                    container_id,
                    "--format",
                    "{{range $name, $_ := .NetworkSettings.Networks}}{{println $name}}{{end}}",
                ]
            )
        except CliError as exc:
            self.logger.warning("Failed to inspect current container network: {}", self._why(exc))
            return None

        networks = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not networks:
            return None
        return networks[0]

    def _get_container_ip(self, container_name: str) -> str | None:
        try:
            result = self._runtime_exec(
                [
                    self._config.container_runtime,
                    "inspect",
                    container_name,
                    "--format",
                    "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                ]
            )
        except CliError as exc:
            self.logger.warning("Failed to inspect sandbox IP: {}", self._why(exc))
            return None

        ip_address = result.stdout.strip()
        return ip_address or None

    def _get_runtime_host(self, container_name: str) -> str:
        if self._config.host:
            return self._config.host

        if _is_running_in_container() or self._config.network:
            container_ip = self._get_container_ip(container_name)
            if container_ip:
                return f"http://{container_ip}"

        return "http://127.0.0.1"

    def _format_command(self, token: str, port: int) -> str:
        return self._config.command.format(token=token, port=port)

    def _build_run_command(self, container_name: str, published_port: int, command: str) -> list[str]:
        network = self._config.network or self._get_current_container_network()

        args = [
            self._config.container_runtime,
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "--entrypoint",
            self._config.shell,
        ]
        if network:
            args.extend(["--network", network])
        args.extend(["-p", f"{published_port}:{self._config.runtime_port}"])
        args.extend(self._config.extra_run_args)
        args.extend([self._config.image, "-lc", command])
        return args

    def _build_apptainer_command(self, command: str) -> list[str]:
        args = [
            self._config.container_runtime,
            "exec",
            "--cleanenv",
            "--compat",
        ]
        args.extend(self._config.extra_run_args)
        args.extend([_normalize_apptainer_image(self._config.image), self._config.shell, "-lc", command])
        return args

    def _get_container_logs(self, container_name: str) -> str:
        if _is_apptainer_runtime(self._config.container_runtime):
            return self._get_apptainer_logs()

        try:
            result = self._runtime_exec(
                [self._config.container_runtime, "logs", container_name], check=False, timeout=_LOGWAIT
            )
        except Exception as exc:
            return f"<failed to fetch logs: {self._why(exc)}>"
        return self._redact((result.stdout or result.stderr).strip())

    def _get_apptainer_logs(self) -> str:
        if self._server_log_handle:
            try:
                self._server_log_handle.flush()
            except Exception:
                pass
        if not self._server_log_path:
            return ""
        try:
            return self._redact(self._server_log_path.read_text(encoding="utf-8", errors="replace").strip())
        except Exception as exc:
            return f"<failed to fetch logs: {self._why(exc)}>"

    def _start_apptainer_process(self, args: list[str]) -> subprocess.Popen[str]:
        fd, log_path = tempfile.mkstemp(prefix=f"uni-agent-local-{_sanitize_name(self.run_id)}-", suffix=".log")
        os.close(fd)
        self._server_log_path = Path(log_path)
        self._server_log_handle = self._server_log_path.open("w", encoding="utf-8", errors="replace")
        self.logger.debug("starting apptainer server process")
        return subprocess.Popen(
            args,
            stdout=self._server_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    async def _start_apptainer(self, token: str, published_port: int) -> None:
        command = self._format_command(token=token, port=published_port)
        self._server_process = self._start_apptainer_process(self._build_apptainer_command(command))
        runtime_config = LocalRuntimeConfig(
            auth_token=token,
            host=self._config.host or "http://127.0.0.1",
            port=published_port,
            timeout=self._config.timeout,
        )
        self._runtime = LocalRuntime.from_config(runtime_config, run_id=self.run_id)

    async def _exec_bg(self, call: Callable[[], Any]) -> Any:
        """Run a CLI call on the tracked thread; the slot holds it until its result is handed back."""
        if self._has_cli():
            raise RuntimeError("a container runtime call from a previous attempt has not been handed back")
        fut = _clipool().submit(call)
        self._future = fut
        try:
            return await asyncio.wrap_future(fut)
        finally:
            if fut.done() and self._future is fut:
                self._future = None

    def _with_cid(self, args: list[str], cidfile: Path, owner: str) -> list[str]:
        """Inject the two handles right before the image, last so a repeated `--cidfile` or same-key
        label from extra_run_args cannot override them. Refuse an ambiguous bare `--` rather than
        guess where the options end (a literal goes as `--flag=value`)."""
        if len(args) < 4 or args[-2] != "-lc":
            raise RuntimeError("unexpected run command shape; the container handles have nowhere safe to go")
        at = len(args) - 3  # the image
        if "--" in args[:at]:
            raise RuntimeError("a bare '--' in the run command is ambiguous; pass a literal value as --flag=value")
        handles = ["--cidfile", str(cidfile), "--label", f"{_OWNERTAG}={owner}"]
        return [*args[:at], *handles, *args[at:]]

    def _find_owned(self, owner: str, timeout: float) -> list[str] | None:
        """The ids carrying our owner tag, or None if the runtime could not tell us -- a failed query
        or an unreadable answer is not proof that no container exists, so it is None, not []."""
        try:
            result = self._runtime_exec(
                [self._config.container_runtime, "ps", "-aq", "--no-trunc", "--filter", f"label={_OWNERTAG}={owner}"],
                timeout=timeout,
            )
        except Exception as exc:
            self._say("warning", "could not list containers by owner {}: {}", owner, self._why(exc))
            return None
        found = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            container_id = _valid_id(line)
            if not container_id:
                self._say("warning", "the runtime listed something that is not a container id for owner {}", owner)
                return None
            found.append(container_id)
        return found

    async def _start_oci_container(self, token: str, container_name: str, published_port: int) -> None:
        command = self._format_command(token=token, port=self._config.runtime_port)
        # Phase 1 may run a network inspect; a cancellation here never submits docker run.
        args = await self._exec_bg(lambda: self._build_run_command(container_name, published_port, command))
        await self._exec_bg(lambda: self._run_container(args, container_name))
        host = await self._exec_bg(lambda: self._get_runtime_host(container_name))
        runtime_config = LocalRuntimeConfig(
            auth_token=token,
            host=host,
            port=self._config.runtime_port,
            timeout=self._config.timeout,
        )
        self._runtime = LocalRuntime.from_config(runtime_config, run_id=self.run_id)

    async def _start_once(self, token: str, container_name: str, published_port: int, *, attempt: int) -> None:
        queue_t0 = time.monotonic()
        sem = _get_limiter()
        await sem.acquire()
        try:
            waited = time.monotonic() - queue_t0
            if waited > _SLOWQUEUE:
                self.logger.info(f"STARTING permit acquired after {waited:.1f}s")
            await self._boot_once(token, container_name, published_port, attempt=attempt)
        finally:
            # The permit follows the CLI thread, not this cancellable coroutine.
            fut = self._future
            if fut is not None and not fut.done():
                loop = asyncio.get_running_loop()

                def _free(_, loop=loop):
                    try:
                        loop.call_soon_threadsafe(sem.release)
                    except RuntimeError:
                        pass  # Loop already closed; its limiter died with it.

                fut.add_done_callback(_free)
            else:
                sem.release()

    async def _boot_once(self, token: str, container_name: str, published_port: int, *, attempt: int) -> None:
        start_t0 = time.monotonic()
        self._hooks.on_custom_step("Creating local sandbox")
        if residue := self._residue():
            raise RuntimeError(f"cannot start a container: {residue}")

        if _is_apptainer_runtime(self._config.container_runtime):
            await self._start_apptainer(token=token, published_port=published_port)
        else:
            await self._start_oci_container(
                token=token,
                container_name=container_name,
                published_port=published_port,
            )

        await self._wait_until_alive(timeout=self._config.startup_timeout)
        try:
            await self.runtime.create_session(
                CreateBashSessionRequest(startup_source=["/root/.bashrc"], startup_timeout=60)
            )
        except Exception as exc:
            raise RuntimeError("bash session creation failed after sandbox became alive") from exc
        container_id = self._container_id[:_IDLEN] if self._container_id else "n/a"
        self.logger.info(
            f"local sandbox alive in {time.monotonic() - start_t0:.2f}s (attempt {attempt + 1}); "
            f"container={container_name} id={container_id}"
        )

    def _begin_attempt(self, retry: int, max_retries: int, base_name: str) -> tuple[str, int, str]:
        # Register only after the announcement succeeds; a raise before this strands no token.
        published_port = self._config.published_port or _pick_free_port()
        container_name = self._config.container_name or (base_name if retry == 0 else f"{base_name}-r{retry}")
        token = self._get_token()
        self.logger.info(
            f"[attempt {retry + 1}/{max_retries}] starting local sandbox "
            f"container={container_name} port={published_port} "
            f"runtime={self._config.container_runtime} image={self._config.image}"
        )
        with self._lock:
            self._secrets.add(token)
            self._abandoned = False
        return token, published_port, container_name

    def _abandon_attempt(self) -> None:
        with self._lock:
            self._abandoned = True
        self._kill_server()
        self._close_logs()

    def _release_token(self, token: str) -> None:
        """Drop the token once nothing it authorised remains: a later stop() may still log an
        exception carrying it, but only while some resource it opened is still here to fail."""
        if not self._has_cleanup():
            with self._lock:
                self._secrets.discard(token)

    async def _backoff(self, retry: int, max_retries: int, deadline: float) -> None:
        nap_room = deadline - time.monotonic()
        if retry < max_retries - 1 and nap_room > 0:
            sleep_time = min(_RETRYCAP, _BACKBASE**retry, nap_room)
            self.logger.info(f"Retrying local deployment startup in {sleep_time:.1f} seconds...")
            await _retrydelay(sleep_time)

    def _claim(self, phase: str) -> None:
        with self._lock:
            if self._busy is not None:
                raise RuntimeError(f"{phase}() overlaps an in-flight {self._busy}(); the lifecycle is serial")
            if phase == "start" and self._has_cleanup():
                raise RuntimeError("the previous lifecycle left resources behind; stop() must reclaim them first")
            self._busy = phase

    def _release(self) -> None:
        with self._lock:
            self._busy = None

    async def start(self, max_retries: int = 5) -> None:
        self._claim("start")
        try:
            await self._start_loop(max_retries)
        finally:
            self._release()

    async def _start_loop(self, max_retries: int) -> None:
        """Stop launching retries after the deadline; refresh unpinned ports and names."""
        base_name = f"uni-agent-{_sanitize_name(self.run_id)}"
        # Redacted at the point of failure: _handle_failure cleans up, and cleanup drops the tokens.
        summary = "no attempt ran"
        wall_budget = _get_budget()
        # Reject a bad cap here; inside the loop it would be retried per attempt.
        _env_cap()
        deadline = time.monotonic() + wall_budget
        attempts = 0
        reason = "retry limit reached"
        for retry in range(max_retries):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = "wall budget exhausted"
                break
            attempts += 1
            token, published_port, container_name = self._begin_attempt(retry, max_retries, base_name)
            attempt_timeout = min(remaining, self._config.startup_timeout + _STARTPAD)
            try:
                await asyncio.wait_for(
                    self._start_once(token, container_name, published_port, attempt=retry), timeout=attempt_timeout
                )
                return
            except asyncio.CancelledError:
                self._abandon_attempt()
                raise
            except Exception as exc:
                summary = self._why(exc)
                await self._handle_failure(
                    exc,
                    attempt_label=f"[attempt {retry + 1}/{max_retries}]",
                    container_name=container_name,
                    published_port=published_port,
                    attempt_timeout=attempt_timeout,
                )
                if residue := self._residue():
                    reason = f"not retrying: {residue}"
                    break
                await self._backoff(retry, max_retries, deadline)
            finally:
                self._release_token(token)

        # Unchained: _handle_failure logged the full chain, redacted; a raw cause reaching the
        # caller's own logger would undo that.
        raise RuntimeError(
            f"Failed to create local sandbox after {attempts} attempt(s): {reason} "
            f"(max_retries={max_retries}, wall budget {wall_budget}s); last error: {summary}"
        )

    async def _collect_logs(self, container_name: str) -> str:
        """Never query OCI logs without a confirmed container ID, and never off the tracked pool: a
        log fetch that outlives its timeout must hold the slot so no rm can overtake it."""
        if self._has_cli():
            return "<not collected: a container runtime call is still in flight>"
        log_ref = container_name if _is_apptainer_runtime(self._config.container_runtime) else self._container_id
        if not log_ref:
            return "<not collected: container creation unconfirmed>"
        try:
            return await asyncio.wait_for(self._exec_bg(lambda: self._get_container_logs(log_ref)), timeout=_LOGWAIT)
        except Exception:
            return "<log collection failed or timed out>"

    async def _handle_failure(
        self,
        exc: Exception,
        *,
        attempt_label: str,
        container_name: str,
        published_port: int,
        attempt_timeout: float,
    ) -> None:
        if isinstance(exc, TimeoutError | asyncio.TimeoutError):
            # Mark timeouts so a late run self-reaps.
            with self._lock:
                self._abandoned = True
        logs = await self._collect_logs(container_name)
        container_id = self._container_id[:_IDLEN] if self._container_id else "n/a"
        reason = self._why(exc)
        if isinstance(exc, TimeoutError | asyncio.TimeoutError) and not str(exc):
            reason = f"{type(exc).__name__}: attempt timed out after {attempt_timeout:.0f}s"
        # format_exception, not logger.opt(exception=...): see the note above CliError.
        chain = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._say(
            "error",
            "{}",
            self._redact(
                f"{attempt_label} failed to start local sandbox "
                f"container={container_name} id={container_id} port={published_port}: {reason}\n"
                f"container logs (last {_LOGTAIL} chars):\n{logs[-_LOGTAIL:]}\n{chain}"
            ),
        )
        try:
            await asyncio.wait_for(self._cleanup(), timeout=_CLEANWAIT)
        except (TimeoutError, asyncio.TimeoutError):
            self._say(
                "warning",
                f"teardown of failed attempt timed out after {_CLEANWAIT}s; container {container_name} may leak",
            )

    async def copy_to_container(self, src: Path, tgt: Path):
        await self.runtime.execute(Command(command=["mkdir", "-p", str(tgt.parent)]))
        await self.runtime.upload(UploadRequest(source_path=str(src), target_path=str(tgt)))

    @property
    def tool_install_dir(self) -> Path:
        """Directory inside the container where tool scripts are installed."""
        return Path("/usr/local/bin")

    def _release_id(self, container_id: str) -> None:
        """Drop a confirmed-gone id and its diagnostics pointer in one critical section, so a cancelled
        sweep leaves neither the id for a second rm nor a pointer at a container that is not there."""
        with self._lock:
            if self._owned is not None and self._owned[0] == container_id:
                self._owned = None
            if self._container_id == container_id:
                self._container_id = None

    def _force_remove(self, container_id: str) -> bool:
        """Force-remove with one retry. An id leaves the ledger only on a confirmed removal."""
        detail = ""
        for attempt in range(_RMTRIES):
            try:
                result = self._runtime_exec(
                    [self._config.container_runtime, "rm", "-f", container_id], check=False, timeout=_RMWAIT
                )
            except Exception as exc:
                detail = self._why(exc)
            else:
                if _confirmed_gone(result):
                    self._release_id(container_id)
                    self._say("debug", f"local sandbox container {container_id} is gone")
                    return True
                detail = self._redact((result.stderr or result.stdout or f"exit code {result.returncode}").strip())
            if attempt == 0:
                self._say("warning", "rm -f {} failed ({}); retrying once", container_id, detail)
                time.sleep(_RMPAUSE)
        self._say("error", "rm -f {} failed twice ({}); the container stays owned and may leak", container_id, detail)
        return False

    async def _close_runtime(self) -> None:
        """Drop the non-resource handle after one bounded close attempt."""
        runtime = self._runtime
        if runtime is None:
            return
        try:
            await asyncio.wait_for(runtime.close(), timeout=_CLOSEWAIT)
        except Exception as exc:
            self._say("warning", "runtime.close() failed; dropping the handle anyway: {}", self._why(exc))
        finally:
            self._runtime = None

    async def _stop_server(self) -> None:
        """Terminate then kill the server process; a failure keeps the slot so a later stop retries."""
        proc = self._server_process
        if proc is None:
            return
        exited = False
        try:
            if proc.poll() is None:
                proc.terminate()
                await asyncio.to_thread(proc.wait, _PROCWAIT)
            exited = True
        except subprocess.TimeoutExpired:
            exited = await self._kill_proc(proc)
        except Exception as exc:
            self._say("error", "Failed to stop local Apptainer process: {}", self._why(exc))
            exited = await self._kill_proc(proc)
        if exited:
            self._server_process = None

    async def _kill_proc(self, proc: subprocess.Popen[str]) -> bool:
        """SIGKILL the captured process with a bounded wait; keep the reference if it survives."""
        try:
            if proc.poll() is None:
                proc.kill()
            # SIGKILL does not guarantee that wait() returns.
            await asyncio.to_thread(proc.wait, _PROCWAIT)
            return True
        except subprocess.TimeoutExpired:
            self._say("error", "apptainer server process did not exit after SIGKILL; keeping reference")
        except Exception as exc:
            self._say("error", "SIGKILL escalation failed: {}; keeping reference", self._why(exc))
        return False

    def _close_logs(self) -> None:
        # As in _stop_server: a failure keeps its slot so the next stop() retries it.
        handle, log_path = self._server_log_handle, self._server_log_path
        if handle is not None:
            try:
                handle.close()
            except Exception as exc:
                self._say("warning", "failed to close apptainer log handle: {}", self._why(exc))
            else:
                self._server_log_handle = None
        if log_path is not None:
            try:
                log_path.unlink(missing_ok=True)
            except Exception as exc:
                self._say("warning", "failed to unlink apptainer log file: {}", self._why(exc))
            else:
                self._server_log_path = None

    async def _drain(self) -> None:
        """Take back the call that outlived its attempt; subprocess.run is its only deadline."""
        fut = self._future
        if fut is None:
            return
        try:
            await asyncio.wrap_future(fut)
        except Exception as exc:
            self._say("warning", "container runtime call failed after its attempt ended: {}", self._why(exc))
        finally:
            if fut.done() and self._future is fut:
                self._future = None

    async def stop(self):
        self._claim("stop")
        try:
            await self._cleanup()
        finally:
            self._release()

    async def _cleanup(self) -> None:
        """Reclaim everything this deployment owns. Runs under the lifecycle, never claims it."""
        await self._drain()
        if self._orphan is not None:
            await self._exec_bg(lambda: self._probe_orphan(_RMWAIT))
        with self._lock:
            pending = self._owned

        await self._close_runtime()
        await self._stop_server()
        self._close_logs()
        if _is_apptainer_runtime(self._config.container_runtime):
            self._container_id = None
        elif pending is not None:
            # On the tracked pool, so a cancelled sweep leaves the rm in the slot, not adrift.
            await self._exec_bg(lambda: self._force_remove(pending[0]))
        if not self._has_cleanup():
            with self._lock:
                self._secrets.clear()

    @property
    def runtime(self) -> LocalRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    def __del__(self):
        # Finalizers cannot rely on the original event loop.
        if not self._has_cleanup():
            return
        try:
            self._kill_server()
        except Exception:
            pass
        try:
            self._close_logs()
        except Exception:
            pass
        try:
            self._del_sweep()
        except Exception:
            pass

    def _del_sweep(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        # Synchronous: a finalizer cannot await the tracked pool. Each OCI call is bounded by _DELWAIT.
        self._probe_orphan(_DELWAIT)
        if self._orphan is not None:
            cidfile, owner, _ = self._orphan
            # Drop the cidfile but keep the owner barrier for a later stop.
            self._say(
                "error",
                "a container may still exist that nothing here can name; remove it by hand with "
                "`{} rm -f $({} ps -aq --filter label={}={})`",
                self._config.container_runtime,
                self._config.container_runtime,
                _OWNERTAG,
                owner,
            )
            self._drop_cid(cidfile)
        with lock:
            owned = self._owned
        runtime = self._config.container_runtime
        if owned is None or _is_apptainer_runtime(runtime):
            return
        container_id, name = owned
        self._say("warning", "__del__ emergency teardown for container {}", name)
        try:
            result = subprocess.run(
                [runtime, "rm", "-f", container_id],
                check=False,
                capture_output=True,
                text=True,
                timeout=_DELWAIT,
            )
        except Exception:
            result = None
        if result is not None and _confirmed_gone(result):
            self._release_id(container_id)
        if self._owned is not None:
            self._say("error", "__del__ could not remove container {}; it may leak", name)

    def _say(self, level: str, message: str, *args: object) -> None:
        """Log; swallow any sink failure so it cannot abort the caller (e.g. a teardown sweep)."""
        try:
            getattr(self.logger, level)(message, *args)
        except Exception:
            pass
