# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node-local ext4 image ownership and supervised fuse2fs mounts.

The caller must stop all runtime users before release/check/delete. Controller
authorization and placement are separate from this filesystem boundary.
"""

import fcntl
import json
import os
from pathlib import Path
import re
import select
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid

from .workspace import WorkspaceConfig


class WorkspaceError(RuntimeError):
    pass


def new_workspace_id():
    return "ws-" + uuid.uuid4().hex


def probe_configuration(config, node_id, pool):
    root = Path(config.image_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    probe_root = tempfile.mkdtemp(prefix=".cape-workspace-probe-", dir=str(root))
    values = config.to_dict()
    values.update(image_root=probe_root, image_size_bytes=32 * 1024 ** 2)
    values["max_image_size_bytes"] = max(config.max_image_size_bytes, 64 * 1024 ** 2)
    manager = WorkspaceManager(WorkspaceConfig.from_dict(values), node_id, pool)
    identity = new_workspace_id()
    try:
        handle = manager.prepare(identity, "__cape_probe__", "__cape_probe__", create=True)
        marker = Path(handle.path) / "probe"
        with marker.open("xb") as stream:
            stream.write(b"cape-workspace-ready\n")
            stream.flush()
            os.fsync(stream.fileno())
        manager.release(identity)
        from .workspace_maintenance import probe_disposable_tools
        probe_disposable_tools(manager, identity)
        handle = manager.prepare(identity, "__cape_probe__", "__cape_probe__")
        if (Path(handle.path) / "probe").read_bytes() != b"cape-workspace-ready\n":
            raise WorkspaceError("workspace maintenance probe content mismatch")
        manager.release(identity)
        manager.delete(identity, "__cape_probe__")
        storage = manager.inspect(identity, "__cape_probe__")["space"]
        if storage.get("error"):
            raise WorkspaceError("cannot inspect workspace backing storage: " + storage["error"])
        if any(entry["target"].startswith(probe_root + "/") for entry in mount_entries()):
            raise WorkspaceError("workspace probe still has mount references")
        shutil.rmtree(probe_root)
        from .workspace_reclaim import maintenance_host_key
        return {"state": "applied", "applied_version": config.version, "error": None,
                "maintenance_host_key": maintenance_host_key(),
                "config": config.to_dict(), "protocol": 1, "maintenance_protocol": 1, "storage": storage["backing"]}
    except BaseException as error:
        try:
            manager.release(identity)
        except Exception:
            pass
        raise WorkspaceError("workspace probe failed; diagnostic image retained at {}: {}".format(probe_root, error))


def mount_entries():
    def decode(value):
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)

    for _ in range(5):
        with open("/proc/self/mountinfo") as stream:
            poll = select.poll()
            poll.register(stream.fileno(), select.POLLPRI)
            poll.poll(0)
            contents = stream.read()
            if not poll.poll(0):
                break
        time.sleep(0.01)
    else:
        raise WorkspaceError("mount table kept changing during workspace inspection")
    entries = []
    for line in contents.splitlines():
        left, separator, right = line.partition(" - ")
        fields, filesystem = left.split(), right.split()
        if separator and len(fields) >= 6 and len(filesystem) >= 2:
            entries.append({"mount_id": int(fields[0]), "target": decode(fields[4]),
                            "fstype": filesystem[0], "source": decode(filesystem[1])})
    return entries


class WorkspaceHandle:
    def __init__(self, directory, manifest, lease_fd, process):
        self.directory = directory
        self.manifest = manifest
        self.lease_fd = lease_fd
        self.process = process

    @property
    def path(self):
        return str(self.directory / "mnt")


class WorkspaceManager:
    def __init__(self, config, node_id, pool, tools=None):
        self.config = WorkspaceConfig.from_dict(config.to_dict())
        self.node_id = node_id
        self.pool = pool
        self.tools = dict(tools or {})
        self.handles = {}
        self._mutex = threading.Lock()
        self._locks = {}
        self._space_cache = {}

    def _guard(self, workspace_id):
        if not isinstance(workspace_id, str) or re.fullmatch(r"ws-[0-9a-f]{32}", workspace_id) is None:
            raise WorkspaceError("invalid workspace_id")
        with self._mutex:
            return self._locks.setdefault(workspace_id, threading.RLock())

    def _tool(self, name):
        variables = {"fuse2fs": "FUSE2FS", "mkfs.ext4": "MKFS", "e2fsck": "E2FSCK", "fusermount": "FUSERMOUNT", "timeout": "TIMEOUT", "stat": "STAT",
                     "resize2fs": "RESIZE2FS", "dumpe2fs": "DUMPE2FS", "cp": "CP", "sha256sum": "SHA256SUM", "nice": "NICE"}
        configured = self.tools.get(name) or os.environ.get("CAPE_WORKSPACE_" + variables[name] + "_BIN")
        resolved = shutil.which(configured or name)
        if resolved is None and name == "fusermount" and not configured:
            resolved = shutil.which("fusermount3")
        if resolved is None:
            raise WorkspaceError("workspace tool is unavailable: " + (configured or name))
        return resolved

    @staticmethod
    def _private_directory(path, create):
        if create:
            try:
                path.mkdir(mode=0o700)
            except FileExistsError:
                pass
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise WorkspaceError("workspace directory must be private, owned, and not a symlink: " + str(path))

    def _directory(self, workspace_id, create):
        root = Path(self.config.image_root)
        if create:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root = root.resolve(strict=True)
        WorkspaceConfig(image_root=str(root))
        parent = root / ("cape-workspaces-" + str(os.geteuid()))
        self._private_directory(parent, create)
        directory = parent / workspace_id
        self._private_directory(directory, create)
        return directory

    @staticmethod
    def _lease(directory):
        fd = os.open(str(directory / "writer.lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1 or info.st_mode & 0o077:
                raise WorkspaceError("unsafe workspace writer lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise WorkspaceError("workspace already has a writer; takeover is forbidden")
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _write(directory, manifest):
        revision = manifest.get("revision", 0)
        if type(revision) is not int or not 0 <= revision < 2 ** 63 - 1:
            raise WorkspaceError("workspace manifest revision overflow")
        manifest["revision"] = revision + 1
        fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=str(directory))
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(manifest, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, str(directory / "manifest.json"))
            parent_fd = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _unlock_completed_lease(fd):
        # Normal completion releases the open-file-description lock explicitly.
        # Crash/error cleanup must only close: live helpers still need their fence.
        fcntl.flock(fd, fcntl.LOCK_UN)

    def _read(self, directory, workspace_id, user_id):
        fd = os.open(str(directory / "manifest.json"), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1 or info.st_size > 1024 ** 2:
                raise WorkspaceError("unsafe workspace manifest")
            manifest = json.load(stream)
        if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 3 or manifest.get("workspace_id") != workspace_id:
            raise WorkspaceError("workspace manifest identity/schema mismatch")
        if type(manifest.get("revision", 0)) is not int or not 0 <= manifest.get("revision", 0) < 2 ** 63:
            raise WorkspaceError("invalid workspace manifest revision")
        if manifest.get("user_id") != user_id or manifest.get("pool") != self.pool:
            raise WorkspaceError("workspace belongs to another user or pool")
        if manifest.get("node_id") != self.node_id:
            raise WorkspaceError("workspace belongs to another node; migration is not supported")
        if manifest.get("state") not in ("creating", "retained", "mounting", "mounted", "needs_check", "deleting", "deleted", "maintenance", "needs_recovery"):
            raise WorkspaceError("invalid workspace lifecycle state")
        if WorkspaceConfig.from_dict(manifest["config"]).to_dict() != self.config.to_dict():
            raise WorkspaceError("workspace creation configuration differs from its controller record")
        if type(manifest.get("usage_generation", 0)) is not int or not 0 <= manifest.get("usage_generation", 0) < 2 ** 63:
            raise WorkspaceError("invalid workspace usage generation")
        return manifest

    @staticmethod
    def _image_fd(directory, manifest):
        fd = os.open(str(directory / "workspace.img"), os.O_RDONLY | os.O_NOFOLLOW)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1
                or info.st_mode & 0o077 or info.st_dev != manifest.get("image_device")
                or info.st_ino != manifest.get("image_inode")
                or info.st_size != manifest.get("current_image_size_bytes", manifest["config"]["image_size_bytes"])):
            os.close(fd)
            raise WorkspaceError("workspace backing image identity/size/ownership changed")
        return fd

    def _run(self, name, arguments, timeout, check=True, lease_fd=None):
        argv = [self._tool("timeout"), "--signal=TERM", "--kill-after=5s", str(timeout) + "s", self._tool(name), *map(str, arguments)]
        priority = getattr(self, "maintenance_nice", 0)
        if priority:
            argv = [self._tool("nice"), "-n", str(priority), *argv]
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            completed = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=timeout + 10,
                                       start_new_session=True, pass_fds=() if lease_fd is None else (lease_fd,))
            captured = []
            for stream in (stdout, stderr):
                size = stream.tell()
                stream.seek(max(0, size - 65536))
                captured.append(stream.read().decode("utf-8", errors="replace"))
            result = subprocess.CompletedProcess(argv, completed.returncode, *captured)
        if check and result.returncode:
            raise WorkspaceError("{} failed ({}): {}".format(name, result.returncode, result.stderr[-4000:]))
        return result

    @staticmethod
    def _mount_at(directory):
        target = str(directory / "mnt")
        matches = [entry for entry in mount_entries() if entry["target"] == target]
        unique = {}
        for entry in matches:
            previous = unique.get(entry["mount_id"])
            if previous is not None and previous != entry:
                raise WorkspaceError("workspace mount table changed during inspection")
            unique[entry["mount_id"]] = entry
        matches = list(unique.values())
        if len(matches) > 1:
            raise WorkspaceError("stacked workspace mounts are not supported: " + json.dumps(matches, sort_keys=True))
        return matches[0] if matches else None

    @staticmethod
    def _is_image_mount(directory, entry):
        return (entry is not None and entry["fstype"].startswith("fuse")
                and entry["source"] == str(directory / "workspace.img"))

    def prepare(self, workspace_id, user_id, execution_session_id, create=False):
        with self._guard(workspace_id):
            existing = self.handles.get(workspace_id)
            if existing is not None:
                if existing.manifest["execution_session_id"] != execution_session_id or existing.manifest["user_id"] != user_id:
                    raise WorkspaceError("workspace is in use by another session")
                self.verify(existing)
                return existing
            directory = self._directory(workspace_id, create)
            lease = self._lease(directory)
            process = None
            manifest = None
            changed = False
            try:
                if (directory / "manifest.json").exists():
                    manifest = self._read(directory, workspace_id, user_id)
                    if manifest["state"] in ("maintenance", "needs_recovery"):
                        raise WorkspaceError("workspace maintenance requires explicit recovery")
                    if any(os.path.lexists(directory / name) for name in ("maintenance-backup.img", "maintenance-candidate.img")):
                        raise WorkspaceError("maintenance recovery files require cleanup before reuse")
                    if manifest["state"] == "deleted":
                        raise WorkspaceError("workspace was deleted")
                    if manifest["state"] == "deleting":
                        raise WorkspaceError("workspace deletion is incomplete; retry deletion before reuse")
                    if create and manifest["execution_session_id"] != execution_session_id:
                        raise WorkspaceError("existing workspace requires explicit reuse")
                elif create:
                    create_started = time.monotonic()
                    manifest = {"schema_version": 3, "workspace_id": workspace_id, "user_id": user_id,
                                "pool": self.pool, "node_id": self.node_id, "execution_session_id": execution_session_id,
                                "config": self.config.to_dict(), "state": "creating", "last_error": None,
                                "image_device": None, "image_inode": None, "mount_id": None, "daemon_pid": None}
                    # An orphaned image must never be truncated or reformatted.
                    fd = os.open(str(directory / "workspace.img"), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                    changed = True
                    try:
                        os.ftruncate(fd, self.config.image_size_bytes)
                        os.fsync(fd)
                        info = os.fstat(fd)
                        manifest.update(image_device=info.st_dev, image_inode=info.st_ino)
                    finally:
                        os.close(fd)
                    self._write(directory, manifest)
                    manifest["timings"] = {"create_seconds": time.monotonic() - create_started}
                    format_started = time.monotonic()
                    self._run("mkfs.ext4", ["-F", "-m", "0", "-i", self.config.bytes_per_inode,
                                           "-E", "root_owner={}:{}".format(os.geteuid(), os.getegid()),
                                           directory / "workspace.img"], self.config.mount_timeout_seconds, lease_fd=lease)
                    manifest["timings"]["format_seconds"] = time.monotonic() - format_started
                    from .workspace_maintenance import geometry
                    manifest["geometry"] = geometry(self, directory / "workspace.img", lease, self.config.mount_timeout_seconds)
                    from .workspace_reclaim import reclaim_sample
                    fd = self._image_fd(directory, manifest)
                    try:
                        manifest["reclaim_reference"] = reclaim_sample(os.fstat(fd).st_blocks * 512, manifest["geometry"], 0, 0)
                    finally:
                        os.close(fd)
                    manifest["state"] = "retained"
                    self._write(directory, manifest)
                else:
                    raise WorkspaceError("workspace does not exist")
                if self._mount_at(directory) is not None:
                    raise WorkspaceError("workspace has an existing mount; reconciliation required")
                self._private_directory(directory / "mnt", True)
                fd = self._image_fd(directory, manifest)
                os.close(fd)
                config = WorkspaceConfig.from_dict(manifest["config"])
                if manifest["state"] != "retained":
                    result = self._run("e2fsck", ["-fn", directory / "workspace.img"], config.mount_timeout_seconds, check=False, lease_fd=lease)
                    if result.returncode:
                        raise WorkspaceError("workspace needs offline maintenance; automatic repair is forbidden")
                if not manifest.get("geometry"):
                    from .workspace_maintenance import geometry
                    manifest["geometry"] = geometry(self, directory / "workspace.img", lease, config.mount_timeout_seconds)
                usage = manifest.get("usage_generation", 0) + 1
                if usage >= 2 ** 63:
                    raise WorkspaceError("workspace usage generation exhausted")
                manifest.update(state="mounting", execution_session_id=execution_session_id, last_error=None,
                                usage_generation=usage, reclaim_sample=None)
                changed = True
                self._write(directory, manifest)
                mount_started = time.monotonic()
                log_fd = os.open(str(directory / "fuse2fs.log"), os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                try:
                    log_info = os.fstat(log_fd)
                    if not stat.S_ISREG(log_info.st_mode) or log_info.st_uid != os.geteuid() or log_info.st_nlink != 1:
                        raise WorkspaceError("unsafe workspace daemon log")
                    os.ftruncate(log_fd, 0)
                    process = subprocess.Popen([self._tool("fuse2fs"), str(directory / "workspace.img"),
                                                str(directory / "mnt"), "-o", "rw", "-f"],
                                               stdin=subprocess.DEVNULL, stdout=log_fd, stderr=log_fd,
                                               start_new_session=True, pass_fds=(lease,))
                finally:
                    os.close(log_fd)
                deadline = time.monotonic() + config.mount_timeout_seconds
                poll_delay = 0.005
                while process.poll() is None:
                    entry = self._mount_at(directory)
                    if entry is not None:
                        if not self._is_image_mount(directory, entry):
                            raise WorkspaceError("unexpected filesystem mounted at workspace target")
                        manifest.update(state="mounted", mount_id=entry["mount_id"], daemon_pid=process.pid)
                        manifest.setdefault("timings", {})["mount_seconds"] = time.monotonic() - mount_started
                        self._write(directory, manifest)
                        handle = WorkspaceHandle(directory, manifest, lease, process)
                        self.handles[workspace_id] = handle
                        return handle
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(poll_delay)
                    poll_delay = min(0.05, poll_delay * 2)
                raise WorkspaceError("fuse2fs failed to establish a verified workspace mount")
            except BaseException as error:
                try:
                    cleanup_error = None
                    try:
                        if process is not None:
                            if process.poll() is None:
                                process.terminate()
                                try:
                                    process.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait(timeout=5)
                            entry = self._mount_at(directory)
                            if self._is_image_mount(directory, entry):
                                self._run("fusermount", ["-u", directory / "mnt"], 10, lease_fd=lease)
                    except Exception as failure:
                        cleanup_error = str(failure)
                    if manifest is not None and changed:
                        manifest.update(state="needs_check", last_error=str(error))
                        if cleanup_error:
                            manifest["last_error"] += "; cleanup: " + cleanup_error
                        self._write(directory, manifest)
                finally:
                    os.close(lease)
                raise

    def verify(self, handle):
        identity = handle.manifest["workspace_id"]
        with self._guard(identity):
            if self.handles.get(identity) is not handle:
                raise WorkspaceError("workspace session was released")
            try:
                entry = self._mount_at(handle.directory)
                if (handle.process.poll() is not None or not self._is_image_mount(handle.directory, entry)
                        or entry["mount_id"] != handle.manifest["mount_id"]):
                    raise WorkspaceError("workspace mount is no longer healthy; ordinary-directory fallback is forbidden")
                fd = self._image_fd(handle.directory, handle.manifest)
                os.close(fd)
            except Exception as error:
                if handle.manifest["state"] != "needs_check":
                    handle.manifest.update(state="needs_check", last_error=str(error))
                    self._write(handle.directory, handle.manifest)
                raise

    def release(self, workspace_id):
        with self._guard(workspace_id):
            handle = self.handles.get(workspace_id)
            if handle is None:
                return
            manifest = handle.manifest
            unmount_started = time.monotonic()
            config = WorkspaceConfig.from_dict(manifest["config"])
            entry = self._mount_at(handle.directory)
            if entry is not None:
                if not self._is_image_mount(handle.directory, entry) or entry["mount_id"] != manifest["mount_id"]:
                    raise WorkspaceError("refusing to unmount an unexpected filesystem")
                try:
                    self._run("fusermount", ["-u", handle.directory / "mnt"], config.mount_timeout_seconds, lease_fd=handle.lease_fd)
                except Exception as error:
                    if manifest.get("last_error") != str(error):
                        manifest["last_error"] = str(error)
                        self._write(handle.directory, manifest)
                    raise
            handle.process.wait(timeout=config.mount_timeout_seconds)
            if self._mount_at(handle.directory) is not None:
                raise WorkspaceError("workspace remains mounted")
            healthy = handle.process.returncode == 0 and entry is not None
            try:
                fd = self._image_fd(handle.directory, manifest)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                manifest.update(state="retained" if healthy else "needs_check", mount_id=None, daemon_pid=None,
                                last_error=None if healthy else "workspace daemon or mount disappeared unexpectedly")
                manifest["reclaim_sample"] = None
                if healthy:
                    from .workspace_maintenance import geometry
                    from .workspace_reclaim import reclaim_sample
                    try:
                        current_geometry = geometry(self, handle.directory / "workspace.img", handle.lease_fd, config.mount_timeout_seconds)
                        fd = self._image_fd(handle.directory, manifest)
                        try:
                            manifest["reclaim_sample"] = reclaim_sample(os.fstat(fd).st_blocks * 512, current_geometry,
                                                                         manifest.get("generation", 0), manifest.get("usage_generation", 0))
                        finally:
                            os.close(fd)
                        manifest.pop("reclaim_sample_error", None)
                    except Exception as error:
                        # Optional admission telemetry must not turn a clean save into a failure.
                        manifest["reclaim_sample_error"] = str(error)[-1000:]
                manifest.setdefault("timings", {})["unmount_sync_seconds"] = time.monotonic() - unmount_started
                self._write(handle.directory, manifest)
                self._unlock_completed_lease(handle.lease_fd)
            finally:
                os.close(handle.lease_fd)
                self.handles.pop(workspace_id, None)
            if not healthy:
                raise WorkspaceError("workspace was not cleanly unmounted; offline check required")

    def inspect(self, workspace_id, user_id):
        with self._guard(workspace_id):
            directory = self._directory(workspace_id, False)
            manifest = self._read(directory, workspace_id, user_id)
            entry = self._mount_at(directory)
            mounted = entry is not None
            verified = self._is_image_mount(directory, entry) and entry["mount_id"] == manifest.get("mount_id")
            if mounted and not verified:
                manifest.update(state="needs_check", last_error="workspace has an unowned or unexpected mount")
            cached = self._space_cache.get(workspace_id)
            if cached is not None and cached[1] == manifest["revision"] and time.monotonic() - cached[0] < 1 and (not mounted or verified):
                return dict(manifest, path=str(directory / "mnt"), mounted=mounted, space=cached[2])
            space = {"internal": None, "backing": None, "logical_bytes": None, "allocated_bytes": None}
            try:
                if manifest["state"] != "deleted":
                    fd = self._image_fd(directory, manifest)
                    try:
                        info = os.fstat(fd)
                        space.update(logical_bytes=info.st_size, allocated_bytes=info.st_blocks * 512)
                    finally:
                        os.close(fd)
                for name, path in (("backing", directory), ("internal", directory / "mnt")):
                    if name == "internal" and not verified:
                        continue
                    result = self._run("stat", ["-f", "-c", "%S %b %a %c %d %T", path], 5)
                    unit, blocks, available, inodes, free_inodes, filesystem = result.stdout.split()
                    space[name] = {"total_bytes": int(unit) * int(blocks), "available_bytes": int(unit) * int(available),
                                   "total_inodes": int(inodes), "free_inodes": int(free_inodes), "filesystem": filesystem}
            except Exception as error:
                space["error"] = str(error)
            self._space_cache[workspace_id] = (time.monotonic(), manifest["revision"], space)
            return dict(manifest, path=str(directory / "mnt"), mounted=mounted, space=space)

    def reconcile_released(self, workspace_id, user_id):
        """Fence an unowned predecessor; never claim its mount was cleanly saved."""
        with self._guard(workspace_id):
            try:
                directory = self._directory(workspace_id, False)
            except FileNotFoundError:
                return
            lease = self._lease(directory)
            try:
                manifest = self._read(directory, workspace_id, user_id)
                if self._mount_at(directory) is not None:
                    raise WorkspaceError("workspace predecessor remains mounted; takeover is forbidden")
                if manifest["state"] not in ("retained", "deleted", "needs_check", "maintenance", "needs_recovery"):
                    manifest.update(state="needs_check", mount_id=None, daemon_pid=None,
                                    last_error="predecessor ownership ended without a verified clean unmount")
                    self._write(directory, manifest)
                self._unlock_completed_lease(lease)
            finally:
                os.close(lease)

    def check(self, workspace_id, user_id):
        with self._guard(workspace_id):
            directory = self._directory(workspace_id, False)
            lease = self._lease(directory)
            try:
                manifest = self._read(directory, workspace_id, user_id)
                if self._mount_at(directory) is not None or workspace_id in self.handles:
                    raise WorkspaceError("offline check requires an unmounted workspace")
                if manifest["state"] in ("deleting", "deleted", "maintenance", "needs_recovery"):
                    raise WorkspaceError("cannot check a deleted/deleting workspace")
                fd = self._image_fd(directory, manifest)
                os.close(fd)
                result = self._run("e2fsck", ["-fn", directory / "workspace.img"],
                                   manifest["config"]["mount_timeout_seconds"], check=False, lease_fd=lease)
                manifest.update(state="retained" if result.returncode == 0 else "needs_check",
                                mount_id=None, daemon_pid=None,
                                last_error=None if result.returncode == 0 else result.stderr[-4000:])
                self._write(directory, manifest)
                self._unlock_completed_lease(lease)
                return {"workspace_id": workspace_id, "exit_code": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr, "state": manifest["state"]}
            finally:
                os.close(lease)

    def delete(self, workspace_id, user_id):
        with self._guard(workspace_id):
            directory = self._directory(workspace_id, False)
            lease = self._lease(directory)
            try:
                manifest = self._read(directory, workspace_id, user_id)
                if manifest["state"] in ("maintenance", "needs_recovery") or any(os.path.lexists(directory / name) for name in ("maintenance-backup.img", "maintenance-candidate.img")):
                    raise WorkspaceError("finalize maintenance recovery before deleting workspace data")
                if manifest["state"] == "deleted":
                    self._unlock_completed_lease(lease)
                    return manifest
                if self._mount_at(directory) is not None or workspace_id in self.handles:
                    raise WorkspaceError("refusing to delete a mounted workspace")
                image_exists = os.path.lexists(str(directory / "workspace.img"))
                if manifest["state"] != "deleting" or image_exists:
                    fd = self._image_fd(directory, manifest)
                    os.close(fd)
                manifest["state"] = "deleting"
                self._write(directory, manifest)
                if image_exists:
                    os.unlink(str(directory / "workspace.img"))
                manifest.update(state="deleted", mount_id=None, daemon_pid=None, last_error=None)
                self._write(directory, manifest)
                self._unlock_completed_lease(lease)
                return manifest
            finally:
                os.close(lease)

    def maintain(self, workspace_id, user_id, action, target_bytes, operation_id, policy=None, maximum_bytes=None):
        from .workspace_maintenance import execute
        return execute(self, workspace_id, user_id, action, target_bytes, operation_id, policy, maximum_bytes)

    def recover_maintenance(self, workspace_id, user_id):
        from .workspace_maintenance import recover
        return recover(self, workspace_id, user_id)
