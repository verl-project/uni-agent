# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import glob
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time

from .models import EnforcementMode, MountSpec, RuntimeCapabilities


class GPUVisibilityError(ValueError):
    """A caller requested visibility outside the assigned GPU lease."""


def leased_gpu_visibility(gpu_uuids, environment):
    if not gpu_uuids:
        return ""
    value = environment.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return ",".join(gpu_uuids)
    leased = {uuid.strip() for uuid in gpu_uuids}
    requested = {entry.strip() for entry in value.split(",") if entry.strip()}
    invalid = sorted(requested - leased)
    if invalid:
        raise GPUVisibilityError(
            "CUDA_VISIBLE_DEVICES contains entries outside the GPU lease: "
            + ", ".join(invalid)
            + "; only leased UUIDs are allowed (ordinal indices are not supported)"
        )
    return value


class RuntimeExecution(object):
    def __init__(
        self,
        phase,
        argv,
        cwd=None,
        env=None,
        exit_code=None,
    ):
        if not phase:
            raise ValueError("runtime execution phase is required")
        if not argv:
            raise ValueError("runtime execution argv must not be empty")
        self.phase = phase
        self.argv = list(argv)
        self.cwd = cwd
        self.env = dict(env or {})
        self.exit_code = exit_code


class RuntimeResult(object):
    def __init__(
        self,
        exit_code,
        stdout="",
        stderr="",
        argv=None,
        cwd=None,
        env=None,
        phase="command",
        executions=(),
        failure_kind=None,
    ):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.failure_kind = failure_kind
        recorded = list(executions or ())
        if argv:
            recorded.append(
                RuntimeExecution(
                    phase,
                    argv,
                    cwd=cwd,
                    env=env,
                    exit_code=exit_code,
                )
            )
        for execution in recorded:
            if not isinstance(execution, RuntimeExecution):
                raise TypeError("executions must contain RuntimeExecution objects")
        self.executions = tuple(recorded)
        representative = None
        for execution in reversed(self.executions):
            if execution.phase == "command":
                representative = execution
                break
        if representative is None and self.executions:
            representative = self.executions[-1]
        self.argv = list(argv) if argv else (
            list(representative.argv) if representative is not None else []
        )
        self.cwd = cwd if argv else (
            representative.cwd if representative is not None else cwd
        )
        self.env = dict(env or {}) if argv else (
            dict(representative.env) if representative is not None else dict(env or {})
        )


class RuntimeLaunch(object):
    def __init__(self, argv, env=None, enforcement_mode=EnforcementMode.NONE, enforcement_message="", cwd=None, phase="command"):
        if not argv:
            raise ValueError("runtime launch argv must not be empty")
        self.argv = list(argv)
        self.env = dict(env or {})
        self.enforcement_mode = enforcement_mode
        self.enforcement_message = enforcement_message
        self.cwd = cwd
        self.phase = phase


def combine_runtime_results(*results):
    results = tuple(
        result for result in results if isinstance(result, RuntimeResult)
    )
    if not results:
        return RuntimeResult(0, "", "")
    exit_code = results[-1].exit_code
    for result in results:
        if result.exit_code != 0:
            exit_code = result.exit_code
            break
    return RuntimeResult(
        exit_code,
        "".join(result.stdout or "" for result in results),
        "".join(result.stderr or "" for result in results),
        executions=tuple(
            execution
            for result in results
            for execution in result.executions
        ),
        failure_kind=next(
            (
                result.failure_kind
                for result in results
                if result.failure_kind is not None
            ),
            None,
        ),
    )


def _deduplicated_paths(paths):
    result = []
    seen = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _append_search_paths(paths, current):
    entries = current.split(os.pathsep) if current else []
    entries.extend(paths)
    return os.pathsep.join(_deduplicated_paths(entries))


_BUBBLEWRAP_DEFAULT_SYSTEM_PATHS = ("/usr", "/etc", "/opt")
_BUBBLEWRAP_SYSTEM_ALIASES = ("/bin", "/sbin", "/lib", "/lib64", "/lib32")
_BUBBLEWRAP_DEFAULT_PATH = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
_BUBBLEWRAP_VIRTUAL_ROOTS = ("/proc", "/sys", "/dev", "/run")


def _normalize_virtual_subpaths(paths):
    result = []
    for configured in paths or ():
        path = os.path.normpath(os.fspath(configured))
        root = next((root for root in _BUBBLEWRAP_VIRTUAL_ROOTS
                     if path.startswith(root + os.sep)), None)
        if root is None:
            raise ValueError("read-only virtual path must be strictly under /proc, /sys, /dev, or /run: " + path)
        resolved = os.path.realpath(path)
        if not resolved.startswith(root + os.sep):
            raise ValueError("read-only virtual path escapes its virtual root: " + path)
        try:
            os.stat(path)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("virtual path contains a symlink loop: " + path)
            if exc.errno not in (errno.ENOENT, errno.ENOTDIR):
                raise
        if path not in result:
            result.append(path)
    return tuple(result)


_BUBBLEWRAP_FORBIDDEN_EXTRA_ARGS = frozenset(
    (
        "--args",
        "--bind",
        "--bind-try",
        "--dev-bind",
        "--dev-bind-try",
        "--ro-bind",
        "--ro-bind-try",
        "--bind-fd",
        "--ro-bind-fd",
        "--overlay-src",
        "--overlay",
        "--tmp-overlay",
        "--ro-overlay",
        "--proc",
        "--dev",
        "--tmpfs",
        "--dir",
        "--file",
        "--bind-data",
        "--ro-bind-data",
        "--symlink",
        "--cap-add",
    )
)


def _normalize_bubblewrap_system_paths(paths):
    candidates = (
        tuple(paths)
        if paths is not None
        else tuple(
            path
            for path in (
                _BUBBLEWRAP_DEFAULT_SYSTEM_PATHS
                + tuple(
                    alias
                    for alias in _BUBBLEWRAP_SYSTEM_ALIASES
                    if os.path.isdir(alias) and not os.path.islink(alias)
                )
            )
            if os.path.isdir(path)
        )
    )
    if not candidates:
        raise ValueError("Bubblewrap system-path allowlist must not be empty")
    normalized = []
    for value in candidates:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("Bubblewrap system paths must be nonempty strings")
        if not os.path.isabs(value):
            raise ValueError("Bubblewrap system paths must be absolute: " + value)
        path = os.path.normpath(value)
        if path == "/":
            raise ValueError("Bubblewrap system paths must never include host root /")
        path = os.path.realpath(path)
        if path == "/":
            raise ValueError("Bubblewrap system path resolves to host root: " + value)
        if any(
            path == root or path.startswith(root + os.sep)
            for root in _BUBBLEWRAP_VIRTUAL_ROOTS
        ):
            raise ValueError(
                "Bubblewrap system path may not expose a virtual kernel tree: "
                + value
            )
        if not os.path.isdir(path):
            raise ValueError("Bubblewrap system path is not a directory: " + value)
        if path in normalized:
            continue
        if any(path.startswith(parent + os.sep) for parent in normalized):
            continue
        normalized = [
            child
            for child in normalized
            if not child.startswith(path + os.sep)
        ]
        normalized.append(path)
    if not normalized:
        raise ValueError("Bubblewrap system-path allowlist must not be empty")
    return tuple(normalized)


def _bubblewrap_system_aliases(system_paths):
    aliases = []
    for path in _BUBBLEWRAP_SYSTEM_ALIASES:
        if not os.path.islink(path):
            continue
        resolved = os.path.realpath(path)
        if any(
            resolved == root or resolved.startswith(root + os.sep)
            for root in system_paths
        ):
            aliases.append((os.readlink(path), path))
    return tuple(aliases)


class ContainerCommandSpec(object):
    def __init__(self, command, cwd=None, env=None):
        if not command:
            raise ValueError("container command must not be empty")
        self.command = tuple(command)
        self.cwd = cwd
        self.env = dict(env or {})


class ContainerSessionSpec(object):
    def __init__(
        self,
        execution_session_id,
        image=None,
        mounts=(),
        cwd=None,
        env=None,
        gpu_uuids=(),
        privileged=False,
        cache_dir=None,
        tmp_dir=None,
        runtime_dir=None,
        oci_bundle=None,
        isolation_policy="default",
        state_layer_id=None,
        overlay_path=None,
        overlay_upper=None,
        overlay_work=None,
    ):
        if not execution_session_id:
            raise ValueError("execution_session_id is required")
        self.execution_session_id = execution_session_id
        self.image = image
        self.mounts = tuple(mounts)
        self.cwd = cwd
        self.env = dict(env or {})
        self.gpu_uuids = tuple(gpu_uuids)
        self.privileged = privileged
        self.cache_dir = cache_dir
        self.tmp_dir = tmp_dir
        self.runtime_dir = runtime_dir
        self.oci_bundle = oci_bundle
        self.isolation_policy = isolation_policy
        self.state_layer_id = state_layer_id
        self.overlay_path = overlay_path
        self.overlay_upper = overlay_upper
        self.overlay_work = overlay_work
        for mount in self.mounts:
            if not isinstance(mount, MountSpec):
                raise TypeError("mounts must contain MountSpec objects")

    def resolved_image(self, default_image):
        return self.image or default_image


class ManagedProcess(object):
    def __init__(self, process, launch=None, cancel_callback=None):
        self.process = process
        self.launch = launch
        self.cancel_callback = cancel_callback

    def wait(self, timeout=None, terminate_grace_seconds=2.0, reap_process_group=False):
        timed_out = False
        try:
            stdout, stderr = self.process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_tree(self.process, terminate_grace_seconds)
            stdout, stderr = self.process.communicate()
            if self.process.returncode is None:
                self.process.returncode = 124
        stdout = stdout or ""
        stderr = stderr or ""
        lingering_reaped = False
        if reap_process_group:
            lingering_reaped = reap_lingering_process_group(self.process, terminate_grace_seconds)
        exit_code = self.process.returncode
        if timed_out and exit_code == 0:
            exit_code = 124
        if lingering_reaped:
            if exit_code == 0:
                exit_code = 125
            if stderr and not stderr.endswith("\n"):
                stderr += "\n"
            stderr += "CAPE reaped lingering workload process group after top-level command exit"
        if self.launch is None:
            return RuntimeResult(
                exit_code=exit_code, stdout=stdout, stderr=stderr
            )
        return RuntimeResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            argv=self.launch.argv,
            cwd=self.launch.cwd,
            env=self.launch.env,
            phase=self.launch.phase,
        )

    def terminate_tree(self, grace_seconds=2.0):
        if self.cancel_callback is not None:
            try:
                self.cancel_callback()
            except Exception:
                pass
        terminate_process_tree(self.process, grace_seconds)


class FinalizingManagedProcess(object):
    def __init__(self, handle, finalize, prefix=None):
        self.handle = handle
        self.process = handle.process
        self.launch = handle.launch
        self.finalize = finalize
        self.prefix = prefix

    def wait(self, **kwargs):
        result = None
        finalized = RuntimeResult(0, "", "")
        try:
            result = self.handle.wait(**kwargs)
        finally:
            finalized = self.finalize()
        combined = combine_runtime_results(self.prefix, result, finalized)
        stderr = result.stderr
        if finalized.exit_code != 0:
            if stderr and not stderr.endswith("\n"):
                stderr += "\n"
            stderr += finalized.stderr or "failed to finalize runtime command"
        return RuntimeResult(
            combined.exit_code,
            result.stdout,
            stderr,
            executions=combined.executions,
        )

    def terminate_tree(self, grace_seconds=2.0):
        self.handle.terminate_tree(grace_seconds)


def _process_group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    except Exception:
        return False


def reap_lingering_process_group(process, grace_seconds=2.0):
    pgid = process.pid
    if not _process_group_exists(pgid):
        return False
    terminate_process_tree(process, grace_seconds)
    return True


def terminate_process_tree(process, grace_seconds=2.0):
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        if process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if process.poll() is not None and not _process_group_exists(pgid):
            return
        time.sleep(0.02)
    if process.poll() is not None and not _process_group_exists(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        if process.poll() is None:
            try:
                process.kill()
            except Exception:
                pass


class FakeRuntimeAdapter(object):
    def __init__(self, name="fake"):
        self.name = name
        self.calls = []
        self.retained_sessions = {}

    def prepare_image(self, image_ref, cache_config="default"):
        self.calls.append("prepare_image:" + image_ref + ":" + cache_config)

    def start_session(self, session_spec):
        execution_session_id = getattr(session_spec, "execution_session_id", session_spec)
        self.retained_sessions[execution_session_id] = session_spec
        self.calls.append("start_session:" + execution_session_id)
        return RuntimeResult(exit_code=0, stdout="", stderr="")

    def run_command(self, execution_session_id, command, cwd=None, env=None, timeout=None):
        self.calls.append("run_command:" + execution_session_id)
        return RuntimeResult(exit_code=0, stdout=" ".join(command), stderr="")

    def stop_session(self, execution_session_id):
        self.calls.append("stop_session:" + execution_session_id)

    def suspend_session(self, execution_session_id):
        self.calls.append("suspend_session:" + execution_session_id)
        return RuntimeResult(0, "", "")

    def resume_session(self, session_spec):
        if session_spec.execution_session_id not in self.retained_sessions:
            return RuntimeResult(127, "", "retained metadata missing")
        self.calls.append("resume_session:" + session_spec.execution_session_id)
        return RuntimeResult(0, "", "")

    def cleanup_session(self, execution_session_id):
        self.calls.append("cleanup_session:" + execution_session_id)


class LocalProcessRuntimeAdapter(object):
    """CPU-only local process adapter for smoke tests and demos."""

    name = "local-process"

    def run_command(self, execution_session_id, command, cwd=None, env=None, timeout=None):
        handle = self.start_process(command, cwd=cwd, env=env)
        return handle.wait(timeout=timeout, reap_process_group=True)

    def start_process(self, command, cwd=None, env=None):
        merged_env = os.environ.copy()
        merged_env.update(env or {})
        proc = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            start_new_session=True,
        )
        return ManagedProcess(
            proc,
            RuntimeLaunch(
                list(command), env=dict(env or {}), cwd=cwd, phase="command"
            ),
        )


class ContainerRuntimeAdapter(object):
    name = "container"
    image_formats = ("default", "docker", "oci")
    cache_env_key = "PYCAPE_RUNTIME_CACHE"
    tmp_env_key = "TMPDIR"
    runtime_env_key = "PYCAPE_RUNTIME_DIR"
    isolation_policies = ("default",)

    def __init__(
        self,
        binary=None,
        default_image="docker://ubuntu:latest",
        cache_dir=None,
        tmp_dir=None,
        runtime_dir=None,
        extra_args=(),
        gpu_enforcement=EnforcementMode.ADVISORY,
        warm_sessions=True,
    ):
        self.binary = binary or self.name
        self.default_image = default_image
        self.cache_dir = cache_dir
        self.tmp_dir = tmp_dir
        self.runtime_dir = runtime_dir
        self.extra_args = tuple(extra_args)
        self.gpu_enforcement = gpu_enforcement
        self.warm_sessions = warm_sessions
        self.sessions = {}
        self.execution_state_dir = None
        self.execution_state_min_free_bytes = 512 * 1024 ** 2
        self.execution_state_max_bytes = 0

    def probe(self, execute=True):
        executable = self.binary_available()
        functional = executable
        stderr = ""
        if executable and execute:
            result = self.version(timeout=5)
            functional = result.exit_code == 0
            stderr = result.stderr
        return RuntimeCapabilities(
            adapter=self.name,
            image_formats=self.image_formats,
            isolation_policies=self.isolation_policies if functional else (),
            nested_container=functional,
            can_bind_mounts=True,
            can_select_gpus=True,
            gpu_enforcement=self.gpu_enforcement,
            can_enforce_cpu_memory=False,
            node_local_cache=bool(self.cache_dir or self.tmp_dir or self.runtime_dir),
            supports_warm_sessions=self.warm_sessions,
            privileged_allowed=False,
        )

    def probe_details(self):
        result = self.version(timeout=5) if self.binary_available() else RuntimeResult(127, "", "runtime binary not found")
        caps = self.probe(execute=False)
        return {"capabilities": caps, "version_exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr}

    def binary_available(self):
        if os.path.isabs(self.binary) or os.sep in self.binary:
            return os.path.exists(self.binary) and os.access(self.binary, os.X_OK)
        return shutil.which(self.binary) is not None

    def version(self, timeout=5):
        try:
            proc = subprocess.Popen(
                [self.binary, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                start_new_session=True,
            )
        except OSError as exc:
            return RuntimeResult(127, "", str(exc))
        return ManagedProcess(proc).wait(timeout=timeout)

    def prepare_image(self, image_ref, cache_config="default"):
        return image_ref or self.default_image

    def start_session(self, session_spec):
        if not isinstance(session_spec, ContainerSessionSpec):
            raise TypeError("session_spec must be a ContainerSessionSpec")
        self.sessions[session_spec.execution_session_id] = session_spec
        try:
            self._ensure_runtime_dirs(session_spec)
        except OSError as exc:
            return RuntimeResult(127, "", "failed to create runtime directory: " + str(exc))
        provisioned = self._provision_session_state(session_spec)
        if isinstance(provisioned, RuntimeResult) and provisioned.exit_code != 0:
            self.sessions.pop(session_spec.execution_session_id, None)
            return provisioned
        if self._uses_live_session(session_spec):
            launch = self.build_start_session_command(session_spec)
            launch.phase = "session-start"
            started = self._run_launch(launch)
            return combine_runtime_results(provisioned, started)
        return provisioned

    def run_command(self, execution_session_id, command, cwd=None, env=None, timeout=None):
        try:
            handle = self.start_command_process(
                execution_session_id, command, cwd=cwd, env=env
            )
        except Exception as exc:
            return RuntimeResult(
                127,
                "",
                str(exc),
                failure_kind="infrastructure",
            )
        return handle.wait(timeout=timeout, reap_process_group=True)

    def start_command_process(
        self, execution_session_id, command, cwd=None, env=None, command_id=None
    ):
        if execution_session_id not in self.sessions:
            raise KeyError("unknown runtime session: " + execution_session_id)
        command_spec = ContainerCommandSpec(command, cwd=cwd, env=env)
        session_spec = self.sessions[execution_session_id]
        try:
            self._ensure_runtime_dirs(session_spec)
        except OSError as exc:
            raise RuntimeError(
                "failed to create runtime directory: " + str(exc)
            )
        launch = self.build_run_command(session_spec, command_spec)
        return self._start_launch(launch)

    def purge_session(self, execution_session_id):
        session_spec = self.sessions.get(execution_session_id)
        if session_spec is None:
            return RuntimeResult(exit_code=0, stdout="", stderr="")
        self._verify_layer_cleanup_scope(session_spec)
        if self._layer_has_mounts(session_spec):
            return RuntimeResult(127, "", "execution layer is still mounted; purge refused")
        if self._uses_live_session(session_spec):
            launch = self.build_stop_session_command(session_spec)
            launch.phase = "session-stop"
            result = self._run_launch(launch)
        else:
            result = RuntimeResult(exit_code=0, stdout="", stderr="")
        cleanup = self._cleanup_session_state(session_spec)
        if isinstance(cleanup, RuntimeResult):
            result = combine_runtime_results(result, cleanup)
        self.sessions.pop(execution_session_id, None)
        return result

    def stop_session(self, execution_session_id):
        return self.suspend_session(execution_session_id)

    def restore_session(self, session_spec):
        if not isinstance(session_spec, ContainerSessionSpec):
            raise TypeError("session_spec must be a ContainerSessionSpec")
        self.sessions[session_spec.execution_session_id] = session_spec

    def suspend_session(self, execution_session_id):
        """Detach an idle instance without invoking destructive layer cleanup.

        The caller owns process/mount fencing and durable session_spec storage.
        Stateful live containers are deliberately not a supported retention path.
        """
        spec = self.sessions.get(execution_session_id)
        if spec is None:
            return RuntimeResult(0, "", "")
        if self._uses_live_session(spec):
            return RuntimeResult(127, "", "live-container suspension is not supported")
        if self._layer_has_mounts(spec):
            return RuntimeResult(127, "", "execution layer is still mounted; suspension deferred")
        self.sessions.pop(execution_session_id, None)
        if not hasattr(self, "retained_sessions"):
            self.retained_sessions = {}
        self.retained_sessions[execution_session_id] = spec
        return RuntimeResult(0, "", "")

    def _verify_layer_cleanup_scope(self, spec):
        root = self.execution_state_dir or spec.runtime_dir or self.runtime_dir or tempfile.gettempdir()
        identity = _safe_name(spec.state_layer_id or spec.execution_session_id)
        if self.name == "bubblewrap":
            expected = os.path.join(root, "bubblewrap-overlays", identity)
            for actual, leaf in ((spec.overlay_upper, "upper"), (spec.overlay_work, "work")):
                if actual and os.path.abspath(actual) != os.path.abspath(os.path.join(expected, leaf)):
                    raise ValueError("execution layer cleanup path does not match its identity")
        if self.name == "apptainer" and spec.overlay_path:
            expected = os.path.abspath(os.path.join(root, "apptainer-overlays", identity))
            if os.path.abspath(spec.overlay_path) not in (expected, expected + ".img"):
                raise ValueError("execution layer cleanup path does not match its identity")

    @staticmethod
    def _layer_has_mounts(spec):
        from .workspace_fs import mount_entries
        roots = [os.path.realpath(os.path.dirname(path)) for path in (spec.overlay_upper, spec.overlay_work) if path]
        if spec.overlay_path:
            roots.append(os.path.realpath(spec.overlay_path))
        return any(entry["target"] == root or entry["target"].startswith(root + os.sep)
                   for entry in mount_entries() for root in roots)

    def retention_fingerprint(self, spec):
        def identity(path):
            path = os.path.realpath(path)
            info = os.stat(path)
            return {"path": path, "device": info.st_dev, "inode": info.st_ino,
                    "size": info.st_size, "mtime_ns": str(info.st_mtime_ns)}
        policy = self._resolved_policy(spec) if hasattr(self, "_resolved_policy") else spec.isolation_policy
        roots = [identity(path) for path in self.system_paths] if self.name == "bubblewrap" else []
        return {"adapter": self.name, "policy": policy, "image": spec.image,
                "image_stat": identity(spec.image) if spec.image and os.path.isfile(spec.image) else None,
                "system_roots": roots, "overlay_mode": getattr(self, "overlay_mode", None)}

    def resume_session(self, session_spec):
        """Reattach retained layer paths; never provision missing state."""
        policy = self._resolved_policy(session_spec) if hasattr(self, "_resolved_policy") else session_spec.isolation_policy
        if self.name == "podman" and policy == "image-stateful":
            return RuntimeResult(127, "", "Podman image-stateful sessions are not supported")
        self.restore_session(session_spec)
        try:
            self._verify_layer_cleanup_scope(session_spec)
        except ValueError as error:
            self.sessions.pop(session_spec.execution_session_id, None)
            return RuntimeResult(127, "", str(error))
        if policy == "host-stateful":
            paths = (session_spec.overlay_upper, session_spec.overlay_work)
        elif policy == "image-stateful":
            paths = (session_spec.overlay_path,)
        else:
            paths = ()
        if any(not path or not os.path.exists(path) or os.path.islink(path) for path in paths):
            self.sessions.pop(session_spec.execution_session_id, None)
            return RuntimeResult(127, "", "retained execution state is missing or unsafe; explicit rebuild required")
        if policy == "host-stateful":
            for index, lower in enumerate(self.system_paths):
                upper, work, _ = self._system_layer_paths(session_spec, index, lower)
                if not os.path.isdir(upper) or not os.path.isdir(work):
                    self.sessions.pop(session_spec.execution_session_id, None)
                    return RuntimeResult(127, "", "retained system layer is incomplete; explicit rebuild required")
        return RuntimeResult(0, "", "")

    def cleanup_session(self, execution_session_id):
        return self.stop_session(execution_session_id)

    def suspend_workspace(self, execution_session_id):
        spec = self.sessions.get(execution_session_id)
        if spec is not None and self._uses_live_session(spec):
            raise RuntimeError("runtime cannot safely suspend live workspace references")

    def resume_workspace(self, execution_session_id):
        return None

    def build_start_session_command(self, session_spec):
        raise NotImplementedError

    def build_run_command(self, session_spec, command_spec):
        raise NotImplementedError

    def build_stop_session_command(self, session_spec):
        raise NotImplementedError

    def _uses_live_session(self, session_spec):
        return self.warm_sessions

    def _provision_session_state(self, session_spec):
        return RuntimeResult(0, "", "")

    def _cleanup_session_state(self, session_spec):
        return RuntimeResult(0, "", "")

    def _state_layer_directory(self, session_spec, namespace):
        root = self.execution_state_dir or session_spec.runtime_dir or self.runtime_dir or tempfile.gettempdir()
        identity = session_spec.state_layer_id or session_spec.execution_session_id
        path = os.path.join(root, namespace, _safe_name(identity))
        if self.execution_state_dir:
            if os.path.exists(path) or os.path.isfile(path + ".img"):
                from .credential_file import _private_parent
                with _private_parent(os.path.join(root, ".state-owner")):
                    pass
            else:
                from .execution_storage import check_storage
                check_storage(root, self.execution_state_min_free_bytes, self.execution_state_max_bytes)
        return path

    def _run_launch(self, launch, timeout=None, reap_process_group=False):
        try:
            handle = self._start_launch(launch)
        except OSError as exc:
            return RuntimeResult(
                127,
                "",
                str(exc),
                argv=launch.argv,
                cwd=launch.cwd,
                env=launch.env,
                phase=launch.phase,
                failure_kind="infrastructure",
            )
        return handle.wait(
            timeout=timeout, reap_process_group=reap_process_group
        )

    def _start_launch(self, launch, cancel_callback=None):
        env = os.environ.copy()
        env.update(launch.env)
        proc = subprocess.Popen(
            launch.argv,
            cwd=launch.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            start_new_session=True,
        )
        return ManagedProcess(
            proc, launch=launch, cancel_callback=cancel_callback
        )

    def _merged_env(self, session_spec, command_spec=None):
        env = {}
        if self.cache_dir or session_spec.cache_dir:
            env[self.cache_env_key] = session_spec.cache_dir or self.cache_dir
        if self.tmp_dir or session_spec.tmp_dir:
            env[self.tmp_env_key] = session_spec.tmp_dir or self.tmp_dir
        if self.runtime_dir or session_spec.runtime_dir:
            env[self.runtime_env_key] = session_spec.runtime_dir or self.runtime_dir
        env.update(session_spec.env)
        if command_spec is not None:
            env.update(command_spec.env)
        env["CUDA_VISIBLE_DEVICES"] = leased_gpu_visibility(session_spec.gpu_uuids, env)
        return env

    def _ensure_runtime_dirs(self, session_spec):
        seen = set()
        for path in (
            session_spec.cache_dir or self.cache_dir,
            session_spec.tmp_dir or self.tmp_dir,
            session_spec.runtime_dir or self.runtime_dir,
        ):
            if not path or path in seen:
                continue
            seen.add(path)
            try:
                os.makedirs(path)
            except OSError as exc:
                if exc.errno != errno.EEXIST or not os.path.isdir(path):
                    raise

    def _enforcement_message(self, session_spec):
        if not session_spec.gpu_uuids:
            return "no GPU devices requested"
        if self.gpu_enforcement == EnforcementMode.HARD:
            return "runtime reports hard GPU/device filtering for selected devices"
        if self.gpu_enforcement == EnforcementMode.ADVISORY:
            return "GPU selection is advisory unless backed by runtime device filters or cgroups"
        return "runtime does not report GPU/device enforcement"


class ApptainerRuntimeAdapter(ContainerRuntimeAdapter):
    name = "apptainer"
    image_formats = ("default", "docker", "sif", "sandbox", "oci")
    cache_env_key = "APPTAINER_CACHEDIR"
    tmp_env_key = "APPTAINER_TMPDIR"
    runtime_env_key = "APPTAINER_LOCALCACHEDIR"

    def __init__(
        self,
        binary=None,
        cleanenv=False,
        contain=False,
        fakeroot=False,
        nv=True,
        unsquash=False,
        overlay_supported=None,
        overlay_mode=None,
        overlay_size_mb=256,
        probe_image=None,
        stateful_sessions=None,
        **kwargs
    ):
        legacy_warm_sessions = kwargs.pop("warm_sessions", None)
        if stateful_sessions is None:
            stateful_sessions = legacy_warm_sessions is not False
        ContainerRuntimeAdapter.__init__(
            self, binary=binary or "apptainer", warm_sessions=False, **kwargs
        )
        self.cleanenv = cleanenv
        self.contain = contain
        self.fakeroot = fakeroot
        self.nv = nv
        self.unsquash = unsquash
        self.overlay_supported = overlay_supported
        self.overlay_mode = overlay_mode or (
            "directory" if overlay_supported else None
        )
        self.overlay_size_mb = overlay_size_mb
        self.probe_image = probe_image
        self.stateful_sessions = stateful_sessions

    def probe(self, execute=True):
        executable = self.binary_available()
        functional = executable
        if executable and execute:
            functional = self.version(timeout=5).exit_code == 0
            if functional:
                functional = self._probe_stateless_execution()
        if self.overlay_supported is None:
            self.overlay_supported = False
            if functional and execute:
                for candidate in ("directory", "ext3"):
                    self.overlay_mode = candidate
                    if self._probe_overlay_persistence():
                        self.overlay_supported = True
                        break
            if not self.overlay_supported:
                self.overlay_mode = None
        policies = ()
        if functional:
            policies = ["default", "image-stateless"]
            if self.stateful_sessions and self.overlay_supported:
                policies.append("image-stateful")
        return RuntimeCapabilities(
            adapter=self.name,
            image_formats=self.image_formats,
            isolation_policies=tuple(policies),
            nested_container=functional,
            can_bind_mounts=True,
            can_select_gpus=True,
            gpu_enforcement=self.gpu_enforcement,
            can_enforce_cpu_memory=False,
            node_local_cache=bool(self.cache_dir or self.tmp_dir or self.runtime_dir),
            supports_warm_sessions=False,
            privileged_allowed=False,
        )

    def _probe_stateless_execution(self):
        if not self.probe_image:
            return False
        probe_root = tempfile.mkdtemp(
            prefix="pycape-apptainer-stateless-probe-",
            dir=self.runtime_dir if self.runtime_dir and os.path.isdir(self.runtime_dir) else None,
        )
        execution_session_id = "stateless-probe-{}".format(os.getpid())
        session = ContainerSessionSpec(
            execution_session_id,
            image=self.probe_image,
            runtime_dir=probe_root,
            isolation_policy="image-stateless",
        )
        try:
            started = self.start_session(session)
            if started.exit_code != 0:
                return False
            result = self.run_command(execution_session_id, ("true",), timeout=300)
            return result.exit_code == 0
        except Exception:
            return False
        finally:
            self.purge_session(execution_session_id)
            shutil.rmtree(probe_root, ignore_errors=True)

    def _probe_overlay_persistence(self):
        if not self.probe_image:
            return False
        probe_root = tempfile.mkdtemp(
            prefix="pycape-apptainer-probe-",
            dir=self.runtime_dir if self.runtime_dir and os.path.isdir(self.runtime_dir) else None,
        )
        previous = self.overlay_supported
        self.overlay_supported = True
        execution_session_id = "overlay-probe-{}".format(os.getpid())
        target = "/opt/.pycape-overlay-probe-{}".format(os.getpid())
        session = ContainerSessionSpec(
            execution_session_id,
            image=self.probe_image,
            runtime_dir=probe_root,
            isolation_policy="image-stateful",
            state_layer_id=execution_session_id,
        )
        try:
            started = self.start_session(session)
            if started.exit_code != 0:
                return False
            first = self.run_command(
                execution_session_id,
                ("sh", "-c", "mkdir -p {0} && printf cape > {0}/value".format(target)),
                timeout=30,
            )
            if first.exit_code != 0:
                return False
            second = self.run_command(
                execution_session_id, ("cat", target + "/value"), timeout=30
            )
            return second.exit_code == 0 and second.stdout == "cape"
        except Exception:
            return False
        finally:
            self.purge_session(execution_session_id)
            self.overlay_supported = previous
            shutil.rmtree(probe_root, ignore_errors=True)

    def build_start_session_command(self, session_spec):
        return RuntimeLaunch(["/bin/true"])

    def build_run_command(self, session_spec, command_spec):
        argv = [self.binary, "exec"]
        argv.extend(self._run_flags(session_spec, command_spec, include_image=True))
        argv.extend(self.extra_args)
        argv.extend(command_spec.command)
        return RuntimeLaunch(argv, self._merged_env(session_spec, command_spec), self.gpu_enforcement, self._enforcement_message(session_spec))

    def build_stop_session_command(self, session_spec):
        return RuntimeLaunch(["/bin/true"])

    def restore_session(self, session_spec):
        if (
            self._resolved_policy(session_spec) == "image-stateful"
            and not session_spec.overlay_path
        ):
            overlay_path = self._state_layer_directory(
                session_spec, "apptainer-overlays"
            )
            if self.overlay_mode == "ext3":
                overlay_path += ".img"
            session_spec.overlay_path = overlay_path
        ContainerRuntimeAdapter.restore_session(self, session_spec)

    def _resolved_policy(self, session_spec):
        if session_spec.isolation_policy == "default":
            if self.stateful_sessions and self.overlay_supported:
                return "image-stateful"
            return "image-stateless"
        if session_spec.isolation_policy not in (
            "image-stateless",
            "image-stateful",
        ):
            raise ValueError(
                "apptainer cannot satisfy isolation policy: "
                + session_spec.isolation_policy
            )
        return session_spec.isolation_policy

    def _provision_session_state(self, session_spec):
        try:
            policy = self._resolved_policy(session_spec)
        except ValueError as exc:
            return RuntimeResult(125, "", str(exc))
        if policy != "image-stateful":
            return RuntimeResult(0, "", "")
        if not self.overlay_supported:
            return RuntimeResult(
                125,
                "",
                "rootless Apptainer writable overlay was not proven on this node",
            )
        if not session_spec.overlay_path:
            layer_path = self._state_layer_directory(
                session_spec, "apptainer-overlays"
            )
            if self.overlay_mode == "ext3":
                layer_path += ".img"
            session_spec.overlay_path = layer_path
        if self.overlay_mode == "ext3":
            parent = os.path.dirname(session_spec.overlay_path)
            try:
                os.makedirs(parent)
            except OSError as exc:
                if exc.errno != errno.EEXIST or not os.path.isdir(parent):
                    return RuntimeResult(127, "", "failed to create Apptainer overlay parent: " + str(exc))
            if os.path.isfile(session_spec.overlay_path):
                return RuntimeResult(0, "", "")
            launch = RuntimeLaunch(
                [
                    self.binary,
                    "overlay",
                    "create",
                    "--sparse",
                    "--size",
                    str(self.overlay_size_mb),
                    session_spec.overlay_path,
                ],
                self._merged_env(session_spec),
                phase="session-provision",
            )
            result = self._run_launch(launch)
            if result.exit_code != 0:
                try:
                    os.remove(session_spec.overlay_path)
                except OSError:
                    pass
            return result
        try:
            os.makedirs(session_spec.overlay_path, mode=0o777)
        except OSError as exc:
            if exc.errno != errno.EEXIST or not os.path.isdir(
                session_spec.overlay_path
            ):
                return RuntimeResult(127, "", "failed to create Apptainer overlay: " + str(exc))
        try:
            os.chmod(session_spec.overlay_path, 0o777)
        except OSError as exc:
            return RuntimeResult(127, "", "failed to prepare Apptainer overlay: " + str(exc))
        return RuntimeResult(0, "", "")

    def _cleanup_session_state(self, session_spec):
        if session_spec.overlay_path:
            if os.path.isdir(session_spec.overlay_path):
                shutil.rmtree(session_spec.overlay_path, ignore_errors=True)
            else:
                try:
                    os.remove(session_spec.overlay_path)
                except OSError:
                    pass
        return RuntimeResult(0, "", "")

    def _run_flags(self, session_spec, command_spec=None, include_image=True, include_cwd=True):
        flags = []
        if self.cleanenv:
            flags.append("--cleanenv")
        if self.contain:
            flags.append("--contain")
        if self.unsquash:
            flags.append("--unsquash")
        if self._resolved_policy(session_spec) == "image-stateful":
            if not session_spec.overlay_path:
                raise ValueError("image-stateful session has no overlay_path")
            flags.extend(["--overlay", session_spec.overlay_path])
        if session_spec.privileged or self.fakeroot:
            flags.append("--fakeroot")
        if self.nv and session_spec.gpu_uuids:
            flags.append("--nv")
        for mount in session_spec.mounts:
            flags.extend(["--bind", self._mount_arg(mount)])
        cwd = command_spec.cwd if command_spec is not None and command_spec.cwd else session_spec.cwd
        if include_cwd and cwd:
            flags.extend(["--pwd", cwd])
        flags.extend(self._env_flags(self._merged_env(session_spec, command_spec), session_spec.gpu_uuids))
        if include_image:
            flags.append(session_spec.resolved_image(self.default_image))
        return flags

    def _env_flags(self, env, gpu_uuids):
        flags = []
        merged = dict(env or {})
        merged["CUDA_VISIBLE_DEVICES"] = leased_gpu_visibility(gpu_uuids, merged)
        for key in sorted(merged):
            flags.extend(["--env", "{}={}".format(key, merged[key])])
        return flags

    def _mount_arg(self, mount):
        mode = "ro" if mount.read_only else "rw"
        return "{}:{}:{}".format(mount.source, mount.target, mode)


class _FuseOverlayMount(object):
    def __init__(self, process, result, merged):
        self.process = process
        self.result = result
        self.merged = merged


class _FuseOverlayMounts(object):
    def __init__(self, mounts):
        self.mounts = tuple(mounts)
        self.result = combine_runtime_results(
            *(mount.result for mount in self.mounts)
        )


class BubbleWrapRuntimeAdapter(ContainerRuntimeAdapter):
    name = "bubblewrap"
    image_formats = ("host",)

    def __init__(
        self,
        binary=None,
        overlay_supported=None,
        fuse_overlayfs_binary=None,
        fusermount_binary=None,
        stateful_sessions=None,
        host_view_mode=None,
        user_namespace_mode=None,
        system_paths=None,
        read_only_paths=(),
        node_gpu_uuids=None,
        gpu_device_map=None,
        gpu_shared_device_paths=None,
        nvidia_smi_binary=None,
        nvidia_driver_paths=None,
        **kwargs
    ):
        legacy_warm_sessions = kwargs.pop("warm_sessions", None)
        if stateful_sessions is None:
            stateful_sessions = legacy_warm_sessions is not False
        ContainerRuntimeAdapter.__init__(
            self,
            binary=binary or "bwrap",
            warm_sessions=False,
            gpu_enforcement=kwargs.pop(
                "gpu_enforcement", EnforcementMode.HARD
            ),
            **kwargs
        )
        self.overlay_supported = overlay_supported
        self.overlay_mode = "native" if overlay_supported else None
        self.fuse_overlayfs_binary = fuse_overlayfs_binary or shutil.which(
            "fuse-overlayfs"
        )
        self.fusermount_binary = fusermount_binary or shutil.which(
            "fusermount3"
        ) or shutil.which("fusermount")
        self.stateful_sessions = stateful_sessions
        if host_view_mode not in (None, False, True):
            raise ValueError("host_view_mode must be true, false, or None")
        if host_view_mode is True:
            raise ValueError(
                "Bubblewrap host-view/outer-proc mode cannot satisfy confined host policies"
            )
        if user_namespace_mode not in (None, False, True):
            raise ValueError(
                "user_namespace_mode must be true, false, or None"
            )
        self.host_view_mode = host_view_mode
        self.user_namespace_mode = user_namespace_mode
        self.system_paths = _normalize_bubblewrap_system_paths(system_paths)
        self.read_only_paths = _normalize_virtual_subpaths(read_only_paths)
        self.system_aliases = _bubblewrap_system_aliases(self.system_paths)
        forbidden_extra_args = tuple(
            value
            for value in self.extra_args
            if value in _BUBBLEWRAP_FORBIDDEN_EXTRA_ARGS
        )
        if forbidden_extra_args:
            raise ValueError(
                "Bubblewrap extra_args may not alter filesystem or capabilities: "
                + ", ".join(forbidden_extra_args)
            )
        self.node_gpu_uuids = (
            None if node_gpu_uuids is None else tuple(node_gpu_uuids)
        )
        self._configured_gpu_device_map = (
            None
            if gpu_device_map is None
            else {
                str(uuid): int(minor)
                for uuid, minor in dict(gpu_device_map).items()
            }
        )
        self._configured_gpu_shared_device_paths = (
            None
            if gpu_shared_device_paths is None
            else tuple(gpu_shared_device_paths)
        )
        self._configured_nvidia_smi_binary = nvidia_smi_binary
        self._configured_nvidia_driver_paths = (
            None
            if nvidia_driver_paths is None
            else tuple(nvidia_driver_paths)
        )
        self.gpu_probe_failures = {}

    def probe(self, execute=True):
        executable = self.binary_available()
        functional = executable
        if executable and execute:
            functional = self.version(timeout=5).exit_code == 0
            if functional:
                functional = self._probe_stateless()
        if self.overlay_supported is None:
            self.overlay_supported = False
            if functional and execute:
                if self._native_overlay_flags_available():
                    self.overlay_mode = "native"
                    self.overlay_supported = self._probe_overlay_persistence()
                if (
                    not self.overlay_supported
                    and self.fuse_overlayfs_binary
                    and self.fusermount_binary
                ):
                    self.overlay_mode = "fuse"
                    self.overlay_supported = self._probe_overlay_persistence()
                if not self.overlay_supported:
                    self.overlay_mode = None
        policies = ()
        if functional:
            policies = ["default", "host-stateless"]
            if self.stateful_sessions and self.overlay_supported:
                policies.append("host-stateful")
        gpu_policies = []
        if functional:
            if self.node_gpu_uuids is None:
                gpu_policies = list(policies)
            elif self.node_gpu_uuids and execute:
                for policy in ("host-stateless", "host-stateful"):
                    if policy in policies and self._probe_gpu_policy(policy):
                        gpu_policies.append(policy)
                default_policy = (
                    "host-stateful"
                    if self.stateful_sessions and self.overlay_supported
                    else "host-stateless"
                )
                if default_policy in gpu_policies:
                    gpu_policies.insert(0, "default")
        reduced_policies = (
            tuple(policies) if self.user_namespace_mode is False else ()
        )
        return RuntimeCapabilities(
            adapter=self.name,
            image_formats=self.image_formats,
            isolation_policies=tuple(policies),
            nested_container=functional,
            can_bind_mounts=True,
            can_select_gpus=bool(gpu_policies),
            gpu_enforcement=self.gpu_enforcement,
            can_enforce_cpu_memory=False,
            node_local_cache=bool(self.runtime_dir),
            supports_warm_sessions=False,
            privileged_allowed=False,
            gpu_isolation_policies=tuple(gpu_policies),
            reduced_isolation_policies=reduced_policies,
        )

    def _probe_stateless(self):
        session = ContainerSessionSpec(
            "bubblewrap-stateless-probe", isolation_policy="host-stateless"
        )
        configured_host_view = self.host_view_mode
        configured_user_namespace = self.user_namespace_mode
        user_namespace_modes = (
            (True, False)
            if configured_user_namespace is None and os.geteuid() == 0
            else (
                (True,)
                if configured_user_namespace is None
                else (configured_user_namespace,)
            )
        )
        for user_namespace_mode in user_namespace_modes:
            self.host_view_mode = False
            self.user_namespace_mode = user_namespace_mode
            try:
                result = self._run_launch(
                    self.build_run_command(
                        session,
                        ContainerCommandSpec(
                            ("sh", "-c", "printf cape-bubblewrap-probe")
                        ),
                    ),
                    timeout=10,
                    reap_process_group=True,
                )
            except Exception:
                continue
            if result.exit_code == 0 and result.stdout == "cape-bubblewrap-probe":
                return True
        self.host_view_mode = configured_host_view
        self.user_namespace_mode = configured_user_namespace
        return False

    def _probe_gpu_policy(self, isolation_policy):
        self.gpu_probe_failures.pop(isolation_policy, None)
        if not self.node_gpu_uuids:
            return self._gpu_probe_failed(
                isolation_policy, "node has no configured GPU UUIDs"
            )
        nvidia_smi = self._nvidia_smi_binary()
        if not nvidia_smi:
            return self._gpu_probe_failed(
                isolation_policy, "nvidia-smi was not found"
            )
        execution_session_id = "bubblewrap-gpu-{}-probe-{}".format(
            isolation_policy, os.getpid()
        )
        session = ContainerSessionSpec(
            execution_session_id,
            runtime_dir=self.runtime_dir,
            isolation_policy=isolation_policy,
            state_layer_id=execution_session_id,
            gpu_uuids=self.node_gpu_uuids,
        )
        try:
            started = self.start_session(session)
            if started.exit_code != 0:
                return self._gpu_probe_failed(
                    isolation_policy,
                    "session start failed: " + (started.stderr or "unknown error"),
                )
            visibility = self.run_command(
                execution_session_id,
                (
                    nvidia_smi,
                    "--query-gpu=uuid",
                    "--format=csv,noheader",
                ),
                timeout=30,
            )
            if visibility.exit_code != 0:
                return self._gpu_probe_failed(
                    isolation_policy,
                    "nvidia-smi failed: "
                    + (visibility.stderr or visibility.stdout or "unknown error"),
                )
            visible = tuple(
                line.strip()
                for line in visibility.stdout.splitlines()
                if line.strip()
            )
            if len(visible) != len(self.node_gpu_uuids) or set(visible) != set(
                self.node_gpu_uuids
            ):
                return self._gpu_probe_failed(
                    isolation_policy,
                    "nvidia-smi reported {!r}; expected {!r}".format(
                        visible, self.node_gpu_uuids
                    ),
                )
            probe_python = self._probe_python_executable()
            if not probe_python:
                return self._gpu_probe_failed(
                    isolation_policy,
                    "no executable Python interpreter is visible beneath the "
                    "Bubblewrap system-path allowlist",
                )
            cuda_probe = (
                "import ctypes,sys; "
                "cuda=ctypes.CDLL('libcuda.so.1'); "
                "count=ctypes.c_int(); "
                "rc=cuda.cuInit(0); "
                "rc2=cuda.cuDeviceGetCount(ctypes.byref(count)) if rc == 0 else rc; "
                "print(rc,rc2,count.value); "
                "sys.exit(0 if rc == 0 and rc2 == 0 and count.value > 0 else 1)"
            )
            initialized = self.run_command(
                execution_session_id,
                (probe_python, "-c", cuda_probe),
                timeout=30,
            )
            if initialized.exit_code != 0:
                return self._gpu_probe_failed(
                    isolation_policy,
                    "cuInit failed: "
                    + (
                        initialized.stderr
                        or initialized.stdout
                        or "unknown error"
                    ),
                )
            kernel_probe = (
                "import torch; "
                "assert torch.cuda.device_count() == {count}; "
                "x=torch.arange(32,device='cuda'); "
                "assert int((x*x).sum().item()) == 10416; "
                "print('cuda-kernel-ok')"
            ).format(count=len(self.node_gpu_uuids))
            kernel = self.run_command(
                execution_session_id,
                (probe_python, "-c", kernel_probe),
                timeout=120,
            )
            if kernel.exit_code != 0:
                return self._gpu_probe_failed(
                    isolation_policy,
                    "CUDA kernel failed: "
                    + (
                        kernel.stderr
                        or kernel.stdout
                        or "unknown error"
                    ),
                )
            self.gpu_probe_failures.pop(isolation_policy, None)
            return True
        except Exception as exc:
            return self._gpu_probe_failed(isolation_policy, str(exc))
        finally:
            self.purge_session(execution_session_id)

    def _gpu_probe_failed(self, isolation_policy, reason):
        self.gpu_probe_failures[isolation_policy] = reason.strip()
        return False

    def _probe_python_executable(self):
        candidates = []
        for root in self.system_paths:
            candidates.extend(
                (
                    os.path.join(root, "bin", "python3"),
                    os.path.join(root, "bin", "python"),
                    os.path.join(root, "local", "bin", "python3"),
                    os.path.join(root, "local", "bin", "python"),
                )
            )
        candidates.extend(("/usr/bin/python3", "/usr/local/bin/python3"))
        seen = set()
        for candidate in candidates:
            if not candidate:
                continue
            candidate = os.path.realpath(candidate)
            if candidate in seen:
                continue
            seen.add(candidate)
            if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
                continue
            if any(
                candidate == root or candidate.startswith(root + os.sep)
                for root in self.system_paths
            ):
                return candidate
        return None

    def _native_overlay_flags_available(self):
        try:
            proc = subprocess.Popen(
                [self.binary, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                start_new_session=True,
            )
        except OSError:
            return False
        result = ManagedProcess(proc).wait(timeout=5)
        help_text = (result.stdout or "") + (result.stderr or "")
        return result.exit_code == 0 and "--overlay-src" in help_text and "--overlay" in help_text

    def _probe_overlay_persistence(self):
        probe_root = tempfile.mkdtemp(
            prefix="pycape-bwrap-probe-",
            dir=self.runtime_dir if self.runtime_dir and os.path.isdir(self.runtime_dir) else None,
        )
        previous = self.overlay_supported
        self.overlay_supported = True
        execution_session_id = "bubblewrap-overlay-probe-{}".format(os.getpid())
        target = os.path.join(
            self.system_paths[-1],
            ".pycape-bwrap-probe-{}".format(os.getpid()),
        )
        session = ContainerSessionSpec(
            execution_session_id,
            runtime_dir=probe_root,
            isolation_policy="host-stateful",
            state_layer_id=execution_session_id,
        )
        try:
            started = self.start_session(session)
            if started.exit_code != 0:
                return False
            first = self.run_command(
                execution_session_id,
                ("sh", "-c", "mkdir -p {0} && printf cape > {0}/value".format(target)),
                timeout=10,
            )
            if first.exit_code != 0:
                return False
            second = self.run_command(
                execution_session_id, ("cat", target + "/value"), timeout=10
            )
            return second.exit_code == 0 and second.stdout == "cape"
        except Exception:
            return False
        finally:
            self.purge_session(execution_session_id)
            self.overlay_supported = previous
            shutil.rmtree(probe_root, ignore_errors=True)

    def build_start_session_command(self, session_spec):
        return RuntimeLaunch(["/bin/true"])

    def _system_layer_paths(self, session_spec, index, system_path):
        if not session_spec.overlay_upper or not session_spec.overlay_work:
            raise ValueError("host-stateful session has no overlay directories")
        key = "{:02d}-{}".format(
            index, _safe_name(system_path.strip(os.sep) or "root")
        )
        layer_dir = os.path.dirname(session_spec.overlay_upper)
        return (
            os.path.join(session_spec.overlay_upper, key),
            os.path.join(session_spec.overlay_work, key),
            os.path.join(layer_dir, "merged", key),
        )

    def _system_mount_flags(self, session_spec, policy):
        flags = []
        for index, path in enumerate(self.system_paths):
            if policy == "host-stateless":
                flags.extend(["--ro-bind", path, path])
                continue
            upper, work, merged = self._system_layer_paths(
                session_spec, index, path
            )
            if self.overlay_mode == "fuse":
                flags.extend(["--bind", merged, path])
            else:
                flags.extend(["--overlay-src", path])
                flags.extend(["--overlay", upper, work, path])
        for target, link in self.system_aliases:
            flags.extend(["--symlink", target, link])
        return flags

    def _sandbox_environment(self, session_spec, command_spec, cwd):
        environment = {
            "HOME": "/home/cape",
            "PATH": _BUBBLEWRAP_DEFAULT_PATH,
        }
        environment.update(session_spec.env)
        environment.update(command_spec.env)
        if cwd:
            environment["PWD"] = cwd
        environment["CUDA_VISIBLE_DEVICES"] = leased_gpu_visibility(session_spec.gpu_uuids, environment)
        if session_spec.gpu_uuids:
            environment.update(
                self._gpu_search_path_environment(environment)
            )
        return environment

    def build_run_command(self, session_spec, command_spec):
        policy = self._resolved_policy(session_spec)
        if self.host_view_mode:
            raise ValueError(
                "Bubblewrap host-view/outer-proc mode cannot execute confined host policies"
            )
        argv = [
            self.binary,
            "--die-with-parent",
            "--new-session",
        ]
        if self.user_namespace_mode is not False:
            argv.extend(["--unshare-user", "--uid", "0", "--gid", "0"])
        argv.extend(["--unshare-pid", "--unshare-ipc", "--unshare-uts"])
        argv.extend(["--clearenv", "--tmpfs", "/"])
        argv.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"])
        argv.extend(
            [
                "--dir",
                "/home",
                "--tmpfs",
                "/home",
                "--dir",
                "/home/cape",
            ]
        )
        argv.extend(self._system_mount_flags(session_spec, policy))
        if session_spec.gpu_uuids:
            argv.extend(self._gpu_mount_flags(session_spec.gpu_uuids))
        for path in _normalize_virtual_subpaths(self.read_only_paths):
            if os.path.exists(path):
                argv.extend(["--ro-bind", path, path])
        for mount in session_spec.mounts:
            flag = "--ro-bind" if mount.read_only else "--bind"
            argv.extend([flag, mount.source, mount.target])
        cwd = command_spec.cwd or session_spec.cwd
        if cwd:
            argv.extend(["--chdir", cwd])
        sandbox_env = self._sandbox_environment(
            session_spec, command_spec, cwd
        )
        for key in sorted(sandbox_env):
            argv.extend(["--setenv", key, str(sandbox_env[key])])
        if self.user_namespace_mode is False:
            # A rootful fresh-proc fallback is useful only for a root-mapped
            # outer container that can create PID/mount namespaces but whose
            # kernel rejects procfs mounts from a nested user namespace.  Do
            # not pass the outer container's capabilities to the workload.
            argv.extend(["--cap-drop", "ALL"])
        argv.extend(self.extra_args)
        argv.append("--")
        argv.extend(command_spec.command)
        launch_env = self._merged_env(session_spec, command_spec)
        launch_env.update(sandbox_env)
        return RuntimeLaunch(
            argv,
            launch_env,
            self.gpu_enforcement,
            self._enforcement_message(session_spec),
        )

    def _gpu_mount_flags(self, gpu_uuids):
        device_map = self._gpu_device_map()
        unresolved = [uuid for uuid in gpu_uuids if uuid not in device_map]
        if unresolved:
            raise ValueError(
                "cannot resolve leased GPU UUID(s) to NVIDIA device minors: "
                + ", ".join(unresolved)
            )
        flags = []
        devices = [
            "/dev/nvidia{}".format(device_map[uuid]) for uuid in gpu_uuids
        ]
        devices.extend(self._gpu_shared_device_paths())
        for path in _deduplicated_paths(devices):
            flags.extend(["--dev-bind", path, path])
        nvidia_smi = self._nvidia_smi_binary()
        if not nvidia_smi:
            raise ValueError(
                "leased GPU exposure requires a host nvidia-smi executable"
            )
        driver_paths = self._nvidia_driver_paths()
        if not driver_paths:
            raise ValueError(
                "leased GPU exposure requires host NVIDIA driver libraries"
            )
        # Bind canonical files.  NVIDIA images commonly expose compatibility
        # libraries through a symlinked directory such as
        # /usr/local/cuda/compat/lib -> lib.real.  Bubblewrap deliberately
        # refuses to create a bind destination through a symlinked parent, so
        # rebinding the alias path makes an otherwise valid GPU launch fail.
        # The aliases already come from an allowlisted read-only system root;
        # rebinding the canonical binary and library files is both explicit
        # and mountable.
        nvidia_smi_mount = os.path.realpath(nvidia_smi)
        for path in _deduplicated_paths((nvidia_smi_mount,) + driver_paths):
            flags.extend(["--ro-bind", path, path])
        if not self.host_view_mode and os.path.exists("/proc/driver/nvidia"):
            flags.extend(
                [
                    "--ro-bind",
                    "/proc/driver/nvidia",
                    "/proc/driver/nvidia",
                ]
            )
        return flags

    def _gpu_device_map(self):
        if self._configured_gpu_device_map is not None:
            return dict(self._configured_gpu_device_map)
        resolved = {}
        for information in sorted(
            glob.glob("/proc/driver/nvidia/gpus/*/information")
        ):
            uuid = None
            minor = None
            try:
                with open(information) as fh:
                    for raw_line in fh:
                        key, separator, value = raw_line.partition(":")
                        if not separator:
                            continue
                        key = key.strip().lower()
                        value = value.strip()
                        if key == "gpu uuid":
                            uuid = value
                        elif key == "device minor":
                            minor = int(value)
            except (OSError, ValueError):
                continue
            if uuid is not None and minor is not None:
                resolved[uuid] = minor
        nvidia_smi = self._nvidia_smi_binary()
        if not nvidia_smi:
            return resolved
        for query in ("uuid,minor_number", "uuid,index"):
            try:
                output = subprocess.check_output(
                    [
                        nvidia_smi,
                        "--query-gpu=" + query,
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    timeout=10,
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
            for line in output.splitlines():
                values = [value.strip() for value in line.split(",")]
                if len(values) != 2:
                    continue
                try:
                    resolved.setdefault(values[0], int(values[1]))
                except ValueError:
                    continue
            if resolved:
                return resolved
        return {}

    def _gpu_shared_device_paths(self):
        if self._configured_gpu_shared_device_paths is not None:
            return self._configured_gpu_shared_device_paths
        candidates = (
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
            "/dev/nvidia-uvm-tools",
            "/dev/nvidia-caps",
        )
        return tuple(path for path in candidates if os.path.exists(path))

    def _nvidia_smi_binary(self):
        if self._configured_nvidia_smi_binary is not None:
            return self._configured_nvidia_smi_binary
        return shutil.which("nvidia-smi")

    def _nvidia_driver_paths(self):
        if self._configured_nvidia_driver_paths is not None:
            return self._configured_nvidia_driver_paths
        paths = []
        machine = os.uname().machine
        multiarch = {
            "aarch64": "aarch64-linux-gnu",
            "arm64": "aarch64-linux-gnu",
            "x86_64": "x86_64-linux-gnu",
            "amd64": "x86_64-linux-gnu",
        }.get(machine)
        directories = ["/usr/lib", "/usr/lib64", "/lib64"]
        if multiarch:
            directories.append("/usr/lib/" + multiarch)
        directories.extend(
            [
                "/usr/local/nvidia/lib",
                "/usr/local/nvidia/lib64",
                "/usr/local/cuda/compat/lib",
                "/usr/local/cuda/compat/lib.real",
            ]
        )
        patterns = []
        for directory in directories:
            patterns.extend(
                [
                    os.path.join(directory, "libcuda.so*"),
                    os.path.join(directory, "libnvidia-*.so*"),
                ]
            )
        for pattern in tuple(patterns):
            paths.extend(glob.glob(pattern))
        expanded = []
        for path in paths:
            target = os.path.realpath(path)
            if os.path.isfile(target):
                expanded.append(target)
        return tuple(_deduplicated_paths(expanded))

    def _gpu_search_path_environment(self, merged_env):
        nvidia_smi = self._nvidia_smi_binary()
        driver_paths = self._nvidia_driver_paths()
        binary_dirs = [os.path.dirname(nvidia_smi)] if nvidia_smi else []
        library_dirs = [os.path.dirname(path) for path in driver_paths]
        current_path = merged_env.get("PATH", "")
        current_library_path = merged_env.get("LD_LIBRARY_PATH", "")
        return {
            "PATH": _append_search_paths(binary_dirs, current_path),
            "LD_LIBRARY_PATH": _append_search_paths(
                library_dirs, current_library_path
            ),
        }

    def build_stop_session_command(self, session_spec):
        return RuntimeLaunch(["/bin/true"])

    def restore_session(self, session_spec):
        if self._resolved_policy(session_spec) == "host-stateful":
            layer_dir = self._state_layer_directory(
                session_spec, "bubblewrap-overlays"
            )
            session_spec.overlay_upper = (
                session_spec.overlay_upper
                or os.path.join(layer_dir, "upper")
            )
            session_spec.overlay_work = (
                session_spec.overlay_work
                or os.path.join(layer_dir, "work")
            )
        ContainerRuntimeAdapter.restore_session(self, session_spec)

    def run_command(self, execution_session_id, command, cwd=None, env=None, timeout=None):
        session_spec = self.sessions.get(execution_session_id)
        if (
            session_spec is None
            or self._resolved_policy(session_spec) != "host-stateful"
            or self.overlay_mode != "fuse"
        ):
            return ContainerRuntimeAdapter.run_command(
                self, execution_session_id, command, cwd=cwd, env=env, timeout=timeout
            )
        mounted = self._mount_fuse_overlays(session_spec)
        if isinstance(mounted, RuntimeResult):
            return mounted
        result = None
        try:
            result = ContainerRuntimeAdapter.run_command(
                self, execution_session_id, command, cwd=cwd, env=env, timeout=timeout
            )
        finally:
            unmounted = self._unmount_fuse_overlays(mounted)
        combined = combine_runtime_results(mounted.result, result, unmounted)
        stderr = result.stderr
        if unmounted.exit_code != 0:
            if stderr and not stderr.endswith("\n"):
                stderr += "\n"
            stderr += unmounted.stderr or "failed to unmount FUSE overlay"
        return RuntimeResult(
            combined.exit_code,
            result.stdout,
            stderr,
            executions=combined.executions,
        )

    def start_command_process(
        self, execution_session_id, command, cwd=None, env=None, command_id=None
    ):
        session_spec = self.sessions.get(execution_session_id)
        if (
            session_spec is None
            or self._resolved_policy(session_spec) != "host-stateful"
            or self.overlay_mode != "fuse"
        ):
            return ContainerRuntimeAdapter.start_command_process(
                self, execution_session_id, command, cwd=cwd, env=env,
                command_id=command_id,
            )
        mounted = self._mount_fuse_overlays(session_spec)
        if isinstance(mounted, RuntimeResult):
            raise RuntimeError(mounted.stderr or "failed to mount FUSE overlays")
        try:
            handle = ContainerRuntimeAdapter.start_command_process(
                self, execution_session_id, command, cwd=cwd, env=env,
                command_id=command_id,
            )
        except Exception:
            self._unmount_fuse_overlays(mounted)
            raise
        return FinalizingManagedProcess(
            handle,
            lambda: self._unmount_fuse_overlays(mounted),
            prefix=mounted.result,
        )

    def _mount_one_fuse_overlay(self, lower, upper, work, merged):
        options = "lowerdir={},upperdir={},workdir={}".format(
            lower, upper, work
        )
        argv = [self.fuse_overlayfs_binary, "-f", "-o", options, merged]
        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                start_new_session=True,
            )
        except OSError as exc:
            return RuntimeResult(
                127,
                "",
                "failed to start fuse-overlayfs: " + str(exc),
                argv=argv,
                phase="session-mount",
            )
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if os.path.ismount(merged):
                return _FuseOverlayMount(
                    process,
                    RuntimeResult(
                        0,
                        "",
                        "",
                        argv=argv,
                        phase="session-mount",
                    ),
                    merged,
                )
            if process.poll() is not None:
                _stdout, stderr = process.communicate()
                return RuntimeResult(
                    process.returncode or 125,
                    "",
                    "fuse-overlayfs mount failed: " + stderr,
                    argv=argv,
                    phase="session-mount",
                )
            time.sleep(0.02)
        terminate_process_tree(process, 0.5)
        try:
            process.communicate(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        return RuntimeResult(
            124,
            "",
            "timed out mounting fuse-overlayfs",
            argv=argv,
            phase="session-mount",
        )

    def _mount_fuse_overlays(self, session_spec):
        mounts = []
        for index, path in enumerate(self.system_paths):
            upper, work, merged = self._system_layer_paths(
                session_spec, index, path
            )
            mounted = self._mount_one_fuse_overlay(
                path, upper, work, merged
            )
            if isinstance(mounted, RuntimeResult):
                cleanup = self._unmount_fuse_overlays(
                    _FuseOverlayMounts(mounts)
                )
                return combine_runtime_results(mounted, cleanup)
            mounts.append(mounted)
        return _FuseOverlayMounts(mounts)

    def _unmount_fuse_overlays(self, mounted):
        results = []
        for mount in reversed(mounted.mounts):
            try:
                results.append(
                    self._run_launch(
                        RuntimeLaunch(
                            [self.fusermount_binary, "-u", mount.merged],
                            phase="session-unmount",
                        ),
                        timeout=10,
                    )
                )
            finally:
                terminate_process_tree(mount.process, 0.5)
                try:
                    mount.process.communicate(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        return combine_runtime_results(*results)

    def _resolved_policy(self, session_spec):
        if session_spec.isolation_policy == "default":
            if self.stateful_sessions and self.overlay_supported:
                return "host-stateful"
            return "host-stateless"
        if session_spec.isolation_policy not in ("host-stateless", "host-stateful"):
            raise ValueError(
                "bubblewrap cannot satisfy isolation policy: "
                + session_spec.isolation_policy
            )
        return session_spec.isolation_policy

    def _provision_session_state(self, session_spec):
        try:
            policy = self._resolved_policy(session_spec)
        except ValueError as exc:
            return RuntimeResult(125, "", str(exc))
        if policy != "host-stateful":
            return RuntimeResult(0, "", "")
        if not self.overlay_supported:
            return RuntimeResult(
                125,
                "",
                "rootless Bubblewrap writable overlay was not proven on this node",
            )
        layer_dir = self._state_layer_directory(
            session_spec, "bubblewrap-overlays"
        )
        session_spec.overlay_upper = session_spec.overlay_upper or os.path.join(
            layer_dir, "upper"
        )
        session_spec.overlay_work = session_spec.overlay_work or os.path.join(
            layer_dir, "work"
        )
        try:
            os.makedirs(session_spec.overlay_upper)
        except OSError as exc:
            if exc.errno != errno.EEXIST or not os.path.isdir(
                session_spec.overlay_upper
            ):
                return RuntimeResult(127, "", "failed to create Bubblewrap upper: " + str(exc))
        try:
            os.makedirs(session_spec.overlay_work)
        except OSError as exc:
            if exc.errno != errno.EEXIST or not os.path.isdir(
                session_spec.overlay_work
            ):
                return RuntimeResult(127, "", "failed to create Bubblewrap work: " + str(exc))
        layer_paths = []
        for index, path in enumerate(self.system_paths):
            upper, work, merged = self._system_layer_paths(
                session_spec, index, path
            )
            layer_paths.extend((upper, work))
            if self.overlay_mode == "fuse":
                layer_paths.append(merged)
        for path in layer_paths:
            try:
                os.makedirs(path)
            except OSError as exc:
                if exc.errno != errno.EEXIST or not os.path.isdir(path):
                    return RuntimeResult(
                        127,
                        "",
                        "failed to create Bubblewrap system layer: "
                        + str(exc),
                    )
        return RuntimeResult(0, "", "")

    def _cleanup_session_state(self, session_spec):
        paths = [session_spec.overlay_upper, session_spec.overlay_work]
        parents = [os.path.dirname(path) for path in paths if path]
        for parent in set(parents):
            shutil.rmtree(parent, ignore_errors=True)
        return RuntimeResult(0, "", "")


class PodmanRuntimeAdapter(ContainerRuntimeAdapter):
    name = "podman"
    image_formats = ("default", "docker", "oci")
    cache_env_key = "PYCAPE_PODMAN_ROOT"
    tmp_env_key = "TMPDIR"
    runtime_env_key = "PYCAPE_PODMAN_RUNROOT"

    def __init__(
        self,
        binary=None,
        keep_id=True,
        run_as_current_user=True,
        gpu_device_mode="advisory",
        graphroot=None,
        runroot=None,
        tmpdir=None,
        probe_image=None,
        **kwargs
    ):
        ContainerRuntimeAdapter.__init__(
            self,
            binary=binary or "podman",
            cache_dir=graphroot,
            tmp_dir=tmpdir,
            runtime_dir=runroot,
            **kwargs
        )
        self.keep_id = keep_id
        self.run_as_current_user = run_as_current_user
        self.gpu_device_mode = gpu_device_mode
        self.graphroot = graphroot
        self.runroot = runroot
        self.tmpdir = tmpdir
        self.probe_image = probe_image

    def probe(self, execute=True):
        caps = ContainerRuntimeAdapter.probe(self, execute=execute)
        policies = ()
        stateless_functional = caps.nested_container
        stateful_functional = False
        if caps.nested_container and execute:
            stateless_functional = bool(self.probe_image) and self._probe_policy(
                "image-stateless"
            )
        if stateless_functional:
            policies = ["default", "image-stateless"]
            if stateful_functional:
                policies.append("image-stateful")
        return RuntimeCapabilities(
            adapter=caps.adapter,
            image_formats=caps.image_formats,
            isolation_policies=tuple(policies),
            nested_container=stateless_functional,
            can_bind_mounts=caps.can_bind_mounts,
            can_select_gpus=caps.can_select_gpus,
            gpu_enforcement=caps.gpu_enforcement,
            can_enforce_cpu_memory=caps.can_enforce_cpu_memory,
            node_local_cache=caps.node_local_cache,
            supports_warm_sessions=stateful_functional,
            privileged_allowed=caps.privileged_allowed,
        )

    def _probe_policy(self, isolation_policy):
        image = self.probe_image
        if image.startswith("docker://"):
            image = image[len("docker://"):]
        execution_session_id = "podman-{}-probe-{}".format(
            isolation_policy, os.getpid()
        )
        session = ContainerSessionSpec(
            execution_session_id,
            image=image,
            isolation_policy=isolation_policy,
        )
        try:
            started = self.start_session(session)
            if started.exit_code != 0:
                return False
            result = self.run_command(
                execution_session_id,
                ("true",),
                timeout=300,
            )
            return result.exit_code == 0
        except Exception:
            return False
        finally:
            self.purge_session(execution_session_id)

    def build_start_session_command(self, session_spec):
        self._uses_live_session(session_spec)
        return RuntimeLaunch(["/bin/true"])

    def build_run_command(self, session_spec, command_spec):
        if self._uses_live_session(session_spec):
            argv = self._base_args()
            argv.append("exec")
            command_env = dict(command_spec.env)
            visibility_env = dict(session_spec.env)
            visibility_env.update(command_env)
            command_env["CUDA_VISIBLE_DEVICES"] = leased_gpu_visibility(session_spec.gpu_uuids, visibility_env)
            for key in sorted(command_env):
                argv.extend(["--env", "{}={}".format(key, command_env[key])])
            if command_spec.cwd or session_spec.cwd:
                argv.extend(["--workdir", command_spec.cwd or session_spec.cwd])
            argv.append(self._container_name(session_spec))
            argv.extend(command_spec.command)
        else:
            argv = self._base_args()
            argv.extend(["run", "--rm", "--name", self._container_name(session_spec)])
            argv.extend(self._container_flags(session_spec, command_spec))
            argv.extend(self.extra_args)
            argv.append(session_spec.resolved_image(self.default_image))
            argv.extend(command_spec.command)
        return RuntimeLaunch(argv, self._merged_env(session_spec, command_spec), self.gpu_enforcement, self._enforcement_message(session_spec))

    def start_command_process(
        self, execution_session_id, command, cwd=None, env=None, command_id=None
    ):
        if execution_session_id not in self.sessions:
            raise KeyError("unknown runtime session: " + execution_session_id)
        session_spec = self.sessions[execution_session_id]
        command_spec = ContainerCommandSpec(command, cwd=cwd, env=env)
        self._ensure_runtime_dirs(session_spec)
        stateful = self._uses_live_session(session_spec)
        pidfile = None
        if stateful:
            pidfile = "/tmp/.cape-command-{}.pid".format(
                _safe_name(command_id or "command")
            )
            wrapped = (
                "/bin/sh",
                "-c",
                'pidfile="$1"; shift; echo "$$" > "$pidfile"; exec "$@"',
                "cape-command",
                pidfile,
            ) + tuple(command_spec.command)
            command_spec = ContainerCommandSpec(
                wrapped, cwd=command_spec.cwd, env=command_spec.env
            )
        launch = self.build_run_command(session_spec, command_spec)
        container = self._container_name(session_spec)
        if stateful:
            callback = lambda: self._cancel_inner_process(container, pidfile)
        else:
            callback = lambda: self._force_remove_container(container)
        return self._start_launch(launch, cancel_callback=callback)

    def _force_remove_container(self, container):
        subprocess.call(
            self._base_args() + ["rm", "-f", container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )

    def _cancel_inner_process(self, container, pidfile):
        script = r'''
i=0
while [ ! -s "$1" ] && [ "$i" -lt 100 ]; do i=$((i + 1)); sleep 0.02; done
[ -s "$1" ] || exit 0
root=$(cat "$1")
descend() {
  for child in $(cat "/proc/$1/task/$1/children" 2>/dev/null); do descend "$child"; done
  echo "$1"
}
pids=$(descend "$root")
for pid in $pids; do kill -TERM "$pid" 2>/dev/null || true; done
sleep 0.2
for pid in $pids; do kill -KILL "$pid" 2>/dev/null || true; done
rm -f "$1"
'''
        subprocess.call(
            self._base_args()
            + ["exec", container, "/bin/sh", "-c", script, "cape-cancel", pidfile],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )

    def build_stop_session_command(self, session_spec):
        argv = self._base_args()
        argv.extend(["rm", "-f", "--ignore", self._container_name(session_spec)])
        return RuntimeLaunch(argv)

    def suspend_workspace(self, execution_session_id):
        spec = self.sessions.get(execution_session_id)
        if spec is not None and self._uses_live_session(spec):
            result = self._run_launch(RuntimeLaunch(self._base_args() + ["stop", "--time", "5", self._container_name(spec)]))
            if result.exit_code:
                raise RuntimeError("workspace runtime suspension failed: " + result.stderr)

    def resume_workspace(self, execution_session_id):
        spec = self.sessions.get(execution_session_id)
        if spec is not None and self._uses_live_session(spec):
            result = self._run_launch(RuntimeLaunch(self._base_args() + ["start", self._container_name(spec)]))
            if result.exit_code:
                raise RuntimeError("workspace runtime resume failed: " + result.stderr)

    def _uses_live_session(self, session_spec):
        policy = session_spec.isolation_policy
        if policy == "image-stateful":
            raise ValueError("Podman image-stateful sessions are not supported")
        if policy in ("default", "image-stateless"):
            return False
        raise ValueError("podman cannot satisfy isolation policy: " + policy)

    def _base_args(self):
        args = [self.binary]
        if self.graphroot:
            args.extend(["--root", self.graphroot])
        if self.runroot:
            args.extend(["--runroot", self.runroot])
        if self.tmpdir:
            args.extend(["--tmpdir", self.tmpdir])
        return args

    def _container_flags(self, session_spec, command_spec):
        flags = []
        if self.keep_id:
            flags.append("--userns=keep-id")
        if self.run_as_current_user:
            flags.extend(["--user", "{}:{}".format(os.getuid(), os.getgid())])
        if session_spec.privileged:
            flags.append("--privileged")
        for mount in session_spec.mounts:
            mode = "ro" if mount.read_only else "rw"
            flags.extend(["-v", "{}:{}:{}".format(mount.source, mount.target, mode)])
        cwd = command_spec.cwd if command_spec is not None and command_spec.cwd else session_spec.cwd
        if cwd:
            flags.extend(["--workdir", cwd])
        for key, value in sorted(self._merged_env(session_spec, command_spec).items()):
            flags.extend(["--env", "{}={}".format(key, value)])
        if session_spec.gpu_uuids and self.gpu_device_mode == "cdi":
            for gpu_uuid in session_spec.gpu_uuids:
                flags.extend(["--device", "nvidia.com/gpu={}".format(gpu_uuid)])
        return flags

    def _container_name(self, session_spec):
        return "pycape-" + _safe_name(session_spec.execution_session_id)


class CrunRuntimeAdapter(ContainerRuntimeAdapter):
    name = "crun"
    image_formats = ("oci-bundle",)
    cache_env_key = "PYCAPE_CRUN_CACHE"
    tmp_env_key = "TMPDIR"
    runtime_env_key = "PYCAPE_CRUN_RUNTIME"

    def __init__(self, binary=None, **kwargs):
        ContainerRuntimeAdapter.__init__(self, binary=binary or "crun", warm_sessions=False, **kwargs)

    def build_start_session_command(self, session_spec):
        return self.build_run_command(session_spec, ContainerCommandSpec(("__oci_bundle_process__",)))

    def build_run_command(self, session_spec, command_spec):
        if not session_spec.oci_bundle:
            raise ValueError("crun requires a prepared OCI bundle path")
        argv = [self.binary, "run", "--bundle", session_spec.oci_bundle, self._container_id(session_spec)]
        return RuntimeLaunch(argv, self._merged_env(session_spec, command_spec), self.gpu_enforcement, self._enforcement_message(session_spec))

    def build_stop_session_command(self, session_spec):
        return RuntimeLaunch([self.binary, "kill", self._container_id(session_spec), "KILL"])

    def _container_id(self, session_spec):
        return "pycape-" + _safe_name(session_spec.execution_session_id)


def _safe_name(value):
    clean = []
    for char in value:
        if char.isalnum() or char in ("-", "_"):
            clean.append(char)
        else:
            clean.append("-")
    return "".join(clean)[:56]
