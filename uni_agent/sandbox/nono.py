"""Nono local sandbox provider."""

from __future__ import annotations

import asyncio
import math
import os
import secrets
import shutil
import sqlite3
import string
import tempfile
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from nono_py import sandboxed_exec

from uni_agent.tools.base import ToolError

from .base import ExecResult, Sandbox, SandboxConfig
from .registry import register_sandbox
from .utils import (
    extract_dir_from_file,
    pack_dir_to_file,
    remote_pack_command,
    remote_unpack_command,
)

if TYPE_CHECKING:
    from nono_py import CapabilitySet

# Before starting Ray: export RAY_worker_num_grpc_internal_threads=1

NONO_CACHE_ROOT = Path("/tmp/n")
NONO_SHELL_ROOT = Path("/tmp/uni-agent-shell")
NONO_SYSTEM_READ_DIRS = (
    "/usr",
    "/lib",
    "/lib64",
    "/etc/ld.so.conf.d",
    "/etc/fonts",
    "/etc/texmf",
    "/var/cache/fontconfig",
    "/var/cache/texmf",
    "/var/lib/texmf",
    "/proc",
)
NONO_SYSTEM_READ_FILES = (
    "/etc/bash.bashrc",
    "/etc/debian_version",
    "/etc/passwd",
    "/etc/group",
    "/etc/mime.types",
    "/etc/ld.so.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/resolv.conf",
    "/etc/gai.conf",
    "/proc/meminfo",
)


@register_sandbox("nono")
class NonoSandbox(Sandbox):
    """Local task isolation and one-shot command execution through nono-py."""

    def __init__(
        self,
        *,
        repo_dir: str | Path,
        conda_dir: str | Path,
        startup_commands: list[str],
        instance_id: str | None = None,
        runtime_timeout: float = 3600.0,
    ) -> None:
        self.instance_id = instance_id
        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.conda_dir = Path(conda_dir).expanduser().resolve()
        self.startup_commands = startup_commands.copy()
        self.runtime_timeout = runtime_timeout
        self.registry = SessionRegistry(NONO_CACHE_ROOT / "sessions.db")

        self.session_id: str | None = None
        self.session_dir: Path | None = None
        self.workdir: Path | None = None
        self.env: dict[str, str] = {}
        self.caps: CapabilitySet | None = None

        self._started = False
        self._started_at: float | None = None
        self._creation_claim: SessionRegistry.Record | None = None
        self._deletion_claim: SessionRegistry.Record | None = None
        self._delete_task: asyncio.Task[None] | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> NonoSandbox:
        return cls(
            runtime_timeout=config.runtime_timeout,
            **config.sandbox_kwargs,
        )

    def create_session_id(self, instance_id: str) -> tuple[str, Path]:
        """Register a short ID and directory, retrying name conflicts up to five times.

        The first attempt and each retry use one random letter or digit. Keep
        the creation claim internally; the caller creates the directory later.
        """
        if self.session_id is not None or self._creation_claim is not None:
            raise SessionRegistry.ConflictError("NonoSandbox already has a session awaiting completion or recovery")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("instance_id must be a non-empty string")

        name = Path(instance_id).name.split("__", 1)[-1] or "nono"
        alphabet = string.ascii_letters + string.digits
        for attempt in range(6):
            session_id = f"{name}-{secrets.choice(alphabet)}"
            session_dir = NONO_CACHE_ROOT / "s" / session_id
            try:
                claim = self.registry.begin_create(session_id, session_dir)
            except SessionRegistry.ConflictError:
                if attempt == 5:
                    raise
                continue
            self._creation_claim = claim
            return session_id, session_dir

    async def start(self) -> None:
        """Copy the prepared repo and make one isolated session ready."""
        from nono_py import AccessMode, CapabilitySet

        if self._started:
            return
        if self.session_id is not None:
            raise SessionRegistry.ConflictError("NonoSandbox already has a session awaiting completion or recovery")

        # 1. Register ownership before creating any session files.
        session_id, session_dir = self.create_session_id(self.instance_id or "nono")
        claim = cast(SessionRegistry.Record, self._creation_claim)
        self.session_id = session_id
        self.session_dir = session_dir
        self.workdir = session_dir / "repo"

        # 2. Keep the repo copy, user packages, and temporary files together.
        session_dir.mkdir(parents=True, exist_ok=False)
        self.workdir.mkdir()
        home_dir = session_dir / "home"
        tmp_dir = session_dir / "tmp"
        home_dir.mkdir()
        tmp_dir.mkdir()

        process = await asyncio.create_subprocess_exec(
            "cp",
            "-a",
            "--reflink=auto",
            "--",
            f"{self.repo_dir}/.",
            str(self.workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output = asyncio.create_task(process.communicate())
        try:
            _, stderr = await asyncio.shield(output)
        finally:
            # Finish the copy process before start returns, even on repeated cancellation.
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                reaped = asyncio.create_task(process.wait())
                while not reaped.done():
                    try:
                        await asyncio.shield(reaped)
                    except asyncio.CancelledError:
                        pass
                reaped.result()
        if process.returncode != 0:
            raise RuntimeError(f"Failed to copy repo: {stderr.decode(errors='replace').strip()}")

        # 3. Prepare the base environment for subsequent commands.
        self.env = {
            "PATH": f"{self.conda_dir}/bin:/usr/bin:/bin",
            "HOME": str(home_dir),
            "TMPDIR": str(tmp_dir),
            "CONDA_PREFIX": str(self.conda_dir),
        }

        # 4. Match the filesystem permissions used by the existing nono runner.
        # Store capabilities here; only sandboxed child processes apply them.
        self.caps = CapabilitySet()
        self.caps.allow_path(str(session_dir), AccessMode.READ_WRITE)
        self.caps.allow_path(str(self.conda_dir), AccessMode.READ)
        NONO_SHELL_ROOT.mkdir(parents=True, exist_ok=True)
        self.caps.allow_path(str(NONO_SHELL_ROOT), AccessMode.READ_WRITE)
        for path in NONO_SYSTEM_READ_DIRS:
            if Path(path).is_dir():
                self.caps.allow_path(path, AccessMode.READ)
        for path in NONO_SYSTEM_READ_FILES:
            if Path(path).is_file():
                self.caps.allow_file(path, AccessMode.READ)
        self.caps.allow_file("/dev/null", AccessMode.READ_WRITE)
        self.caps.allow_path("/dev/pts", AccessMode.READ_WRITE)
        self.caps.allow_file("/dev/ptmx", AccessMode.READ_WRITE)
        if Path("/dev/shm").is_dir():
            self.caps.allow_path("/dev/shm", AccessMode.READ_WRITE)

        # 5. Capture startup exports and unsets as the session's base environment.
        if self.startup_commands:
            script = "{\nset -e\n" + "\n".join(self.startup_commands) + "\n} >&2\nexec /usr/bin/env -0"
            execution = asyncio.create_task(
                asyncio.to_thread(
                    sandboxed_exec,
                    self.caps,
                    ["/bin/bash", "--noprofile", "--norc", "-c", script],
                    cwd=str(self.workdir),
                    env=list(self.env.items()),
                    inherit_env=False,
                    timeout_secs=self.runtime_timeout,
                )
            )
            try:
                result = await asyncio.shield(execution)
            except asyncio.CancelledError:
                while not execution.done():
                    try:
                        await asyncio.shield(execution)
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        break
                if not execution.cancelled():
                    execution.exception()
                raise
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Startup commands failed (exit {result.exit_code}): "
                    f"{result.stderr.decode(errors='replace').strip()}"
                )
            self.env = dict(entry.split("=", 1) for entry in os.fsdecode(result.stdout).split("\0") if entry)

        # 6. Publish readiness only after all preparation has succeeded.
        self.registry.finish_create(session_id, cast(str, claim.owner_token))
        self._creation_claim = None
        self._started_at = time.monotonic()
        self._started = True

    async def is_alive(self) -> bool:
        """Check whether this session is still usable without claiming it."""
        if not self._started:
            return False

        elapsed = time.monotonic() - cast(float, self._started_at)
        if elapsed >= self.runtime_timeout:
            return False

        try:
            record = await asyncio.to_thread(
                self.registry.get,
                cast(str, self.session_id),
            )
            return (
                self._started
                and record is not None
                and record.state in ("idle", "in_use")
                and self.workdir is not None
                and self.workdir.is_dir()
            )
        except Exception:
            return False

    def _append_permission_hint(self, stderr: str) -> str:
        """Add workspace guidance to permission-denied command errors."""
        if "Permission denied" in stderr:
            return f"{stderr.rstrip()}\nPlease return to {self.workdir}."
        return stderr

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run argv once with this session's capabilities and base environment."""
        if not self._started:
            raise RuntimeError("Nono sandbox has not been started")

        session_id = cast(str, self.session_id)
        claim = self.registry.acquire(session_id)
        try:
            command_dir = str(self.workdir) if workdir is None else workdir
            command_env = self.env.copy()
            command_env.update(env or {})
            remaining = self.runtime_timeout - (time.monotonic() - cast(float, self._started_at))
            if remaining <= 0:
                raise TimeoutError("Nono sandbox runtime_timeout exceeded")
            effective_timeout = remaining if timeout is None else min(timeout, remaining)

            execution = asyncio.create_task(
                asyncio.to_thread(
                    sandboxed_exec,
                    self.caps,
                    argv.copy(),
                    cwd=command_dir,
                    env=list(command_env.items()),
                    inherit_env=False,
                    timeout_secs=effective_timeout,
                )
            )
            try:
                result = await asyncio.shield(execution)
            except asyncio.CancelledError:
                # Cancelling the await does not stop the SDK's worker thread.
                while not execution.done():
                    try:
                        await asyncio.shield(execution)
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        break
                if not execution.cancelled():
                    execution.exception()
                raise

            return ExecResult(
                exit_code=result.exit_code,
                stdout=result.stdout.decode(errors="replace"),
                stderr=self._append_permission_hint(result.stderr.decode(errors="replace")),
            )
        finally:
            self.registry.release(session_id, cast(str, claim.owner_token))

    async def write_file(self, path: str, content: bytes | str) -> None:
        """Stage bytes locally, then write the target with sandbox permissions."""
        if not self._started:
            raise RuntimeError("Nono sandbox has not been started")
        data = content.encode("utf-8") if isinstance(content, str) else content
        # Keep payloads out of argv: even a modest source file can exceed the
        # kernel's per-argument limit after base64 encoding. The session root
        # cannot be replaced by sandbox commands, unlike its TMPDIR child.
        with tempfile.NamedTemporaryFile(dir=self.session_dir, prefix="uni-agent-write-") as staged:
            staged.write(data)
            staged.flush()
            result = await self.exec(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    'mkdir -p -- "$(dirname -- "$2")" && cat -- "$1" > "$2"',
                    "nono-write",
                    staged.name,
                    path,
                ]
            )
            if result.exit_code != 0:
                message = f"write_file {path!r} failed: {result.stderr.strip()}"
                if "Permission denied" in result.stderr:
                    raise ToolError(message)
                raise RuntimeError(message)

    async def stop(self) -> None:
        """Delete this session's files and registry record."""
        session_id = self.session_id
        if session_id is None:
            return

        # 1. Claim an idle session; creating and in_use sessions raise a conflict.
        claim = self._deletion_claim
        if claim is None:
            claim = self.registry.begin_delete(session_id)
            self._deletion_claim = claim
        elif self.registry.get(session_id) != claim:
            raise SessionRegistry.ConflictError("Session deletion ownership has changed")
        self._started = False
        self._started_at = None

        # 2. Keep the task on cancellation so a retry waits for the same deletion.
        session_dir = Path(claim.workspace)
        if self._delete_task is None and session_dir.exists():
            self._delete_task = asyncio.create_task(asyncio.to_thread(shutil.rmtree, session_dir))
        delete_task = self._delete_task
        if delete_task is not None:
            try:
                await asyncio.shield(delete_task)
            finally:
                if delete_task.done() and self._delete_task is delete_task:
                    self._delete_task = None
        if self.session_id != session_id:
            return

        # 3. Keep the deleting record and claim if filesystem or DB cleanup fails.
        self.registry.finish_delete(session_id, cast(str, claim.owner_token))
        self.session_id = None
        self.session_dir = None
        self.workdir = None
        self.env.clear()
        self.caps = None
        self._creation_claim = None
        self._deletion_claim = None

    async def _upload_tree(self, local_dir: Path, remote_dir: str) -> None:
        """Upload a directory using archives staged in this session's tmp directory."""
        tmp_dir = Path(self.env["TMPDIR"])
        remote_archive = str(tmp_dir / f"uni-agent-upload-{uuid.uuid4().hex}.tar.gz")
        try:
            with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
                archive = Path(tmp) / "upload.tar.gz"
                pack_dir_to_file(local_dir, archive)
                await self.upload_file(archive, remote_archive)
            res = await self.exec_shell(remote_unpack_command(remote_archive, remote_dir))
            if res.exit_code != 0:
                raise RuntimeError(
                    f"upload into {remote_dir!r} failed (sandbox needs tar and gzip): {res.stderr.strip()}"
                )
        finally:
            await self.exec(["rm", "-f", remote_archive])

    async def _download_tree(self, remote_dir: str, local_dir: Path | str) -> None:
        """Download a directory using archives staged in this session's tmp directory."""
        dst = Path(local_dir)
        dst.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(self.env["TMPDIR"])
        remote_archive = str(tmp_dir / f"uni-agent-download-{uuid.uuid4().hex}.tar.gz")
        try:
            res = await self.exec_shell(remote_pack_command(remote_dir, remote_archive))
            if res.exit_code != 0:
                raise RuntimeError(
                    f"download of {remote_dir!r} failed (sandbox needs tar and gzip): {res.stderr.strip()}"
                )
            with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
                archive = Path(tmp) / "download.tar.gz"
                await self.download_file(remote_archive, archive)
                extract_dir_from_file(archive, dst)
        finally:
            await self.exec(["rm", "-f", remote_archive])


class SessionRegistry:
    """Track local sandbox sessions across processes using SQLite.

    Lifecycle:
        absent -> creating -> idle <-> in_use
        idle -> deleting -> absent

    The backend manages files and processes outside database transactions.
    Failed operations retain their state for explicit recovery; sessions are
    never unlocked automatically on timeout.

    Share one local database (not NFS) outside sandbox-accessible paths.
    Callers must keep session workspaces disjoint and hold ownership until
    all workspace operations and their child processes have stopped.

    Each call opens its own connection. ``timeout`` bounds database lock waits,
    which raise sqlite3.OperationalError on expiry. Busy sessions raise
    ConflictError without waiting for session availability.
    """

    State = Literal["creating", "idle", "in_use", "deleting"]
    _STATES = ("creating", "idle", "in_use", "deleting")

    class Error(RuntimeError):
        """Registry operation error; SQLite exceptions propagate unchanged."""

    class NotFoundError(Error):
        """The requested session does not exist."""

    class ConflictError(Error):
        """The session already exists, or its state or ownership token conflicts."""

    @dataclass(frozen=True)
    class Record:
        """Committed session snapshot with Unix timestamps.

        Tokens from begin_create/acquire/begin_delete authorize one operation
        and may be shared by trusted backend processes; they are not PID-bound.
        Query snapshots do not acquire ownership.
        """

        session_id: str
        workspace: str
        state: SessionRegistry.State
        owner_token: str | None
        created_at: float
        updated_at: float

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        *,
        timeout: float = 30.0,
    ) -> None:
        raw_path = os.fspath(db_path)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("db_path 必须是非空文件路径")
        if raw_path == ":memory:":
            raise ValueError("多进程共享注册表必须使用本地数据库文件")
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout 必须是非负有限秒数")
        self.db_path = str(Path(raw_path).expanduser().resolve())
        self.timeout = float(timeout)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def begin_create(self, session_id: str, workspace: str | os.PathLike[str]) -> SessionRegistry.Record:
        """Claim a new session in creating without creating its directory.

        Duplicate IDs or workspace paths raise ConflictError. Failed setup
        leaves the record in creating until explicitly recovered for deletion.
        """
        self._validate_id(session_id)
        raw_workspace = os.fspath(workspace)
        if not isinstance(raw_workspace, str) or not raw_workspace.strip():
            raise ValueError("workspace 必须是非空目录路径")
        workspace_path = Path(raw_workspace).expanduser().resolve()
        if Path(self.db_path).is_relative_to(workspace_path):
            raise ValueError("数据库文件不能位于 session 工作目录内")
        token = secrets.token_hex(32)
        now = time.time()
        with self._transaction() as connection:
            try:
                connection.execute(
                    """INSERT INTO sessions
                       (session_id, workspace, state, owner_token,
                        created_at, updated_at)
                       VALUES (?, ?, 'creating', ?, ?, ?)""",
                    (session_id, str(workspace_path), token, now, now),
                )
            except sqlite3.IntegrityError as error:
                raise self.ConflictError(f"无法创建 {session_id!r}：session_id 或 workspace 已登记") from error
            record = self._fetch(connection, session_id)
        return record

    def finish_create(self, session_id: str, owner_token: str) -> SessionRegistry.Record:
        """Use the creation token to move creating -> idle after initialization succeeds."""
        self._validate_token(owner_token)
        return self._transition(session_id, "creating", "idle", owner_token, None)

    def acquire(self, session_id: str) -> SessionRegistry.Record:
        """Atomically claim idle -> in_use; raise ConflictError if the session is busy."""
        return self._transition(session_id, "idle", "in_use", None, secrets.token_hex(32))

    def release(self, session_id: str, owner_token: str) -> SessionRegistry.Record:
        """Return in_use -> idle using the claim token after all users have stopped."""
        self._validate_token(owner_token)
        return self._transition(session_id, "in_use", "idle", owner_token, None)

    @contextmanager
    def use(self, session_id: str) -> Generator[SessionRegistry.Record, None, None]:
        """Acquire the session and release it on both normal and exceptional exit.

        Stop and reap all workspace users before leaving. If that cannot be
        guaranteed, acquire/release explicitly and retain the in_use state.
        Nested use of the same session is not supported.
        """
        record = self.acquire(session_id)
        try:
            yield record
        finally:
            self.release(record.session_id, cast(str, record.owner_token))

    def begin_delete(self, session_id: str) -> SessionRegistry.Record:
        """Claim idle -> deleting before removing session resources."""
        return self._transition(session_id, "idle", "deleting", None, secrets.token_hex(32))

    def finish_delete(self, session_id: str, owner_token: str) -> None:
        """Remove the record using its deletion token after cleanup succeeds.

        Keep the record in deleting if cleanup fails, for retry or recovery.
        """
        self._validate_id(session_id)
        self._validate_token(owner_token)
        with self._transaction() as connection:
            cursor = connection.execute(
                """DELETE FROM sessions
                   WHERE session_id = ? AND state = 'deleting'
                     AND owner_token = ?""",
                (session_id, owner_token),
            )
            if cursor.rowcount != 1:
                self._raise_conflict(connection, session_id, "deleting")

    def recover_for_delete(self, snapshot: SessionRegistry.Record) -> SessionRegistry.Record:
        """Claim deletion using the snapshot's current state and ownership token.

        Stop and reap the previous operation and its children first; this method
        neither checks nor terminates processes. A stale snapshot is rejected.
        The new token invalidates old completion and release requests.

        Recover creating, in_use, or deleting; use begin_delete for idle sessions.
        Clean up remaining resources before calling finish_delete.
        """
        if snapshot.state not in ("creating", "in_use", "deleting"):
            raise ValueError("仅恢复非空闲状态；idle 请使用 begin_delete")
        self._validate_token(snapshot.owner_token)
        return self._transition(
            snapshot.session_id,
            snapshot.state,
            "deleting",
            snapshot.owner_token,
            secrets.token_hex(32),
        )

    def get(self, session_id: str) -> SessionRegistry.Record | None:
        """Return a committed snapshot, or None; querying does not acquire ownership."""
        self._validate_id(session_id)
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return self._record(row) if row is not None else None

    def list_sessions(self, state: SessionRegistry.State | None = None) -> list[SessionRegistry.Record]:
        """Return snapshots ordered by session_id, optionally filtered by state."""
        if state is not None and state not in self._STATES:
            raise ValueError(f"无效 session 状态：{state!r}")
        with self._connection() as connection:
            if state is None:
                rows = connection.execute("SELECT * FROM sessions ORDER BY session_id").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM sessions WHERE state = ? ORDER BY session_id",
                    (state,),
                ).fetchall()
        return [self._record(row) for row in rows]

    def _initialize(self) -> None:
        with self._connection() as connection:
            mode = connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
            if mode.lower() != "delete":
                raise self.Error("无法启用 SQLite DELETE 日志模式")
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY NOT NULL,
                        workspace TEXT UNIQUE NOT NULL,
                        state TEXT NOT NULL CHECK (
                            state IN ('creating', 'idle', 'in_use', 'deleting')
                        ),
                        owner_token TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        CHECK (
                            (state = 'idle' AND owner_token IS NULL)
                            OR (state <> 'idle' AND owner_token IS NOT NULL)
                        )
                    )"""
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.db_path, timeout=self.timeout, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA synchronous = FULL")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _transition(
        self,
        session_id: str,
        expected_state: SessionRegistry.State,
        next_state: SessionRegistry.State,
        expected_token: str | None,
        next_token: str | None,
    ) -> SessionRegistry.Record:
        self._validate_id(session_id)
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE sessions
                   SET state = ?, owner_token = ?, updated_at = ?
                   WHERE session_id = ? AND state = ? AND owner_token IS ?""",
                (
                    next_state,
                    next_token,
                    time.time(),
                    session_id,
                    expected_state,
                    expected_token,
                ),
            )
            if cursor.rowcount != 1:
                self._raise_conflict(connection, session_id, expected_state)
            record = self._fetch(connection, session_id)
        return record

    def _raise_conflict(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        expected_state: SessionRegistry.State,
    ) -> None:
        record = self._fetch(connection, session_id)
        raise self.ConflictError(
            f"session {session_id!r} 操作失败：要求状态 {expected_state!r} "
            f"且凭据匹配，当前状态 {record.state!r}；状态或凭据不匹配"
        )

    @classmethod
    def _fetch(cls, connection: sqlite3.Connection, session_id: str) -> SessionRegistry.Record:
        row = connection.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            raise cls.NotFoundError(f"session {session_id!r} 不存在")
        return cls._record(row)

    @staticmethod
    def _record(row: sqlite3.Row) -> SessionRegistry.Record:
        return SessionRegistry.Record(
            session_id=row["session_id"],
            workspace=row["workspace"],
            state=cast(SessionRegistry.State, row["state"]),
            owner_token=row["owner_token"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _validate_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id 必须是非空字符串")

    @staticmethod
    def _validate_token(owner_token: str | None) -> None:
        if not isinstance(owner_token, str) or not owner_token:
            raise ValueError("owner_token 必须是占用操作返回的非空凭据")
