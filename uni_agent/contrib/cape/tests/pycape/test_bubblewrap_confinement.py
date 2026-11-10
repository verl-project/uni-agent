# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import errno
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


MATRIX_ENABLED = os.environ.get("PYCAPE_BUBBLEWRAP_CONFINEMENT_MATRIX") == "1"


def _deduplicated_existing(paths):
    result = []
    seen = set()
    for path in paths:
        canonical = os.path.realpath(path)
        if canonical in seen or not os.path.exists(canonical):
            continue
        seen.add(canonical)
        result.append(canonical)
    return tuple(result)


def _system_path_layout():
    roots = tuple(path for path in ("/usr", "/etc", "/opt") if os.path.exists(path))
    aliases = []
    extra_roots = []
    for path in ("/bin", "/sbin", "/lib", "/lib64", "/lib32"):
        if os.path.islink(path):
            aliases.append((os.readlink(path), path))
        elif os.path.exists(path) and not any(
            path == root or path.startswith(root + os.sep) for root in roots
        ):
            extra_roots.append(path)
    return roots + tuple(extra_roots), tuple(aliases)


def _nvidia_resources(gpu_uuid):
    device_map = {}
    for information in sorted(glob.glob("/proc/driver/nvidia/gpus/*/information")):
        uuid = None
        minor = None
        with contextlib.suppress(OSError, ValueError):
            with open(information) as handle:
                for raw_line in handle:
                    key, separator, value = raw_line.partition(":")
                    if not separator:
                        continue
                    if key.strip().lower() == "gpu uuid":
                        uuid = value.strip()
                    elif key.strip().lower() == "device minor":
                        minor = int(value.strip())
        if uuid is not None and minor is not None:
            device_map[uuid] = minor
    if gpu_uuid not in device_map:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,minor_number",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=20,
        )
        for line in output.splitlines():
            fields = tuple(field.strip() for field in line.split(","))
            if len(fields) == 2:
                with contextlib.suppress(ValueError):
                    device_map[fields[0]] = int(fields[1])
    if gpu_uuid not in device_map:
        raise RuntimeError("cannot resolve GPU UUID to an NVIDIA device minor: " + gpu_uuid)

    devices = ["/dev/nvidia{}".format(device_map[gpu_uuid])]
    devices.extend(
        path
        for path in (
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
            "/dev/nvidia-uvm-tools",
            "/dev/nvidia-caps",
        )
        if os.path.exists(path)
    )

    machine = platform.machine()
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
        (
            "/usr/local/nvidia/lib",
            "/usr/local/nvidia/lib64",
            "/usr/local/cuda/compat/lib",
            "/usr/local/cuda/compat/lib.real",
        )
    )
    libraries = []
    for directory in directories:
        libraries.extend(glob.glob(os.path.join(directory, "libcuda.so*")))
        libraries.extend(glob.glob(os.path.join(directory, "libnvidia-*.so*")))
    binary = shutil.which("nvidia-smi")
    if not binary:
        raise RuntimeError("nvidia-smi is unavailable")
    assets = _deduplicated_existing((os.path.realpath(binary),) + tuple(libraries))
    if not assets:
        raise RuntimeError("NVIDIA user-space assets are unavailable")
    return {
        "minor": device_map[gpu_uuid],
        "devices": _deduplicated_existing(devices),
        "assets": assets,
        "nvidia_smi": os.path.realpath(binary),
    }


@unittest.skipUnless(
    MATRIX_ENABLED,
    "set PYCAPE_BUBBLEWRAP_CONFINEMENT_MATRIX=1 for the real raw-option matrix",
)
class RealBubblewrapConfinementMatrixTests(unittest.TestCase):
    records = []

    @classmethod
    def setUpClass(cls):
        cls.binary = os.environ.get("PYCAPE_BUBBLEWRAP_BINARY", "bwrap")
        cls.legacy_binary = os.environ.get("PYCAPE_BUBBLEWRAP_LEGACY_BINARY")
        cls.gpu_uuid = os.environ.get("PYCAPE_BUBBLEWRAP_TEST_GPU_UUID")
        cls.output = os.environ.get("PYCAPE_BUBBLEWRAP_MATRIX_OUTPUT")
        cls.base = Path(
            tempfile.mkdtemp(
                prefix="pycape-bwrap-confinement-",
                dir=os.environ.get("PYCAPE_BUBBLEWRAP_MATRIX_TMPDIR"),
            )
        )
        cls.workspace = cls.base / "granted-workspace"
        cls.cache_source = cls.base / "cache-source"
        cls.outer_home = cls.base / "outer-home"
        for directory in (cls.workspace, cls.cache_source, cls.outer_home):
            directory.mkdir()
        cls.marker_token = "cape-foreign-process-{}".format(os.getpid())
        (cls.workspace / "granted.txt").write_text("granted\n")
        (cls.workspace / "marker-token.txt").write_text(cls.marker_token + "\n")
        (cls.cache_source / "cache.txt").write_text("cache-mounted\n")
        cls.sibling_canary = cls.base / "ground-truth-sibling.txt"
        cls.sibling_canary.write_text("ground-truth\n")
        cls.home_canary = cls.outer_home / "home-secret.txt"
        cls.home_canary.write_text("home-secret\n")
        cls.root_canary = Path("/root") / (
            "cape-bwrap-ground-truth-{}".format(os.getpid())
        )
        cls.root_canary.write_text("root-secret\n")
        cls.marker = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)", cls.marker_token],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        cls.system_roots, cls.system_aliases = _system_path_layout()
        cls.explicit_environment = {
            "CAPE_MATRIX_ENV": "explicit",
            "HOME": "/home/cape",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PWD": "/workspace",
        }
        cls._write_probe_payload()

    @classmethod
    def tearDownClass(cls):
        cls.marker.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            cls.marker.wait(timeout=2)
        if cls.marker.poll() is None:
            cls.marker.kill()
            cls.marker.wait()
        with contextlib.suppress(OSError):
            cls.root_canary.unlink()
        evidence = {
            "schema_version": 3,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": {
                "node": platform.node(),
                "machine": platform.machine(),
                "kernel": platform.release(),
                "python": platform.python_version(),
                "outer_image": os.environ.get("PYCAPE_BUBBLEWRAP_MATRIX_OUTER_IMAGE"),
            },
            "configuration": {
                "binary": cls.binary,
                "legacy_binary": cls.legacy_binary,
                "gpu_uuid": cls.gpu_uuid,
                "system_roots": cls.system_roots,
                "system_aliases": cls.system_aliases,
                "explicit_environment": cls.explicit_environment,
            },
            "records": cls.records,
        }
        if cls.output:
            output = Path(cls.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        shutil.rmtree(cls.base, ignore_errors=True)

    @classmethod
    def _write_probe_payload(cls):
        probe = r'''import errno
import json
import os
import sys

mode, sibling, home_canary, root_canary, marker_pid = sys.argv[1:]
with open("/workspace/marker-token.txt") as handle:
    marker_token = handle.read().strip()

def path_state(path):
    try:
        os.stat(path)
        return "visible"
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return "enoent"
        return "errno-{}".format(exc.errno)

cmdlines = {}
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    try:
        with open("/proc/{}/cmdline".format(name), "rb") as handle:
            cmdlines[name] = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        pass

status = {}
with open("/proc/self/status") as handle:
    for line in handle:
        key, separator, value = line.partition(":")
        if separator and key in ("CapEff", "NoNewPrivs"):
            status[key] = value.strip()

result = {
    "sibling": path_state(sibling),
    "home_canary": path_state(home_canary),
    "root_canary": path_state(root_canary),
    "granted": path_state("/workspace/granted.txt"),
    "new_bind_target": path_state("/cache/cache.txt"),
    "marker_pid": path_state("/proc/{}".format(marker_pid)),
    "marker_in_enumeration": any(marker_token in value for value in cmdlines.values()),
    "pid_1_cmdline": cmdlines.get("1", ""),
    "proc_pids": sorted(int(value) for value in cmdlines),
    "cap_eff": status.get("CapEff"),
    "no_new_privs": status.get("NoNewPrivs"),
}

if mode == "isolated":
    ok = (
        result["sibling"] == "enoent"
        and result["home_canary"] == "enoent"
        and result["root_canary"] == "enoent"
        and result["granted"] == "visible"
        and result["new_bind_target"] == "visible"
        and result["marker_pid"] == "enoent"
        and not result["marker_in_enumeration"]
        and bool(result["pid_1_cmdline"])
        and marker_token not in result["pid_1_cmdline"]
        and result["cap_eff"] == "0000000000000000"
        and result["no_new_privs"] == "1"
    )
elif mode == "leaky-files-private-processes":
    ok = (
        result["sibling"] == "visible"
        and result["home_canary"] == "visible"
        and result["root_canary"] == "visible"
        and result["marker_pid"] == "enoent"
        and not result["marker_in_enumeration"]
    )
elif mode == "leaky-files-and-processes":
    ok = (
        result["sibling"] == "visible"
        and result["home_canary"] == "visible"
        and result["root_canary"] == "visible"
        and result["marker_pid"] == "visible"
        and result["marker_in_enumeration"]
    )
else:
    raise SystemExit("unknown probe mode: " + mode)

result["ok"] = ok
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if ok else 1)
'''
        (cls.workspace / "confinement_probe.py").write_text(probe)

    @classmethod
    def _record_process(cls, name, argv, expected, timeout=120, env=None):
        started = time.monotonic()
        try:
            process = subprocess.run(
                list(argv),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=env,
            )
            exit_code = process.returncode
            stdout = process.stdout
            stderr = process.stderr
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + "\nTIMEOUT"
        record = {
            "name": name,
            "argv": list(argv),
            "expected": expected,
            "exit_code": exit_code,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": stdout,
            "stderr": stderr,
        }
        cls.records.append(record)
        return record

    @classmethod
    def _root_flags(
        cls,
        namespace_mode,
        system_mounts=None,
        sys_paths=(),
        gpu_resources=None,
        binary=None,
    ):
        flags = [binary or cls.binary, "--die-with-parent", "--new-session"]
        if namespace_mode == "userns":
            flags.extend(("--unshare-user", "--uid", "0", "--gid", "0"))
        elif namespace_mode != "rootful":
            raise ValueError("unknown namespace mode: " + namespace_mode)
        flags.extend(("--unshare-pid", "--unshare-ipc", "--unshare-uts", "--clearenv"))
        flags.extend(("--tmpfs", "/"))
        if system_mounts is None:
            system_mounts = tuple(("--ro-bind", path, path) for path in cls.system_roots)
        for mount in system_mounts:
            flags.extend(mount)
        for target, link in cls.system_aliases:
            flags.extend(("--symlink", target, link))
        flags.extend(("--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"))
        flags.extend(("--dir", "/home", "--tmpfs", "/home", "--dir", "/home/cape"))
        if sys_paths:
            flags.extend(("--dir", "/sys"))
            parents = set()
            for path in sys_paths:
                parent = os.path.dirname(path)
                while parent not in ("", "/", "/sys"):
                    parents.add(parent)
                    parent = os.path.dirname(parent)
            for parent in sorted(parents, key=lambda item: (item.count("/"), item)):
                flags.extend(("--dir", parent))
            for path in sys_paths:
                if os.path.exists(path):
                    flags.extend(("--ro-bind", path, path))
        if gpu_resources:
            for path in gpu_resources["devices"]:
                flags.extend(("--dev-bind", path, path))
            for path in gpu_resources["assets"]:
                flags.extend(("--ro-bind", path, path))
            if os.path.exists("/proc/driver/nvidia"):
                flags.extend(("--ro-bind", "/proc/driver/nvidia", "/proc/driver/nvidia"))
        flags.extend(("--bind", str(cls.workspace), "/workspace"))
        flags.extend(("--bind", str(cls.cache_source), "/cache"))
        flags.extend(("--chdir", "/workspace"))
        environment = dict(cls.explicit_environment)
        if gpu_resources:
            environment["CUDA_VISIBLE_DEVICES"] = cls.gpu_uuid
            library_dirs = []
            for path in gpu_resources["assets"]:
                if ".so" in os.path.basename(path):
                    library_dirs.append(os.path.dirname(path))
            outer_library_path = os.environ.get("LD_LIBRARY_PATH", "")
            library_dirs.extend(
                part for part in outer_library_path.split(":") if part
            )
            environment["LD_LIBRARY_PATH"] = ":".join(
                dict.fromkeys(library_dirs)
            )
        for key in sorted(environment):
            flags.extend(("--setenv", key, environment[key]))
        if namespace_mode == "rootful":
            flags.extend(("--cap-drop", "ALL"))
        flags.append("--")
        return flags, environment

    @classmethod
    def _confinement_probe_command(cls, mode):
        return (
            "/usr/bin/python3",
            "/workspace/confinement_probe.py",
            mode,
            str(cls.sibling_canary),
            str(cls.home_canary),
            str(cls.root_canary),
            str(cls.marker.pid),
        )

    @classmethod
    def _assembled_run(
        cls,
        name,
        command,
        namespace_mode="rootful",
        system_mounts=None,
        sys_paths=(),
        gpu_resources=None,
        binary=None,
        expected="success",
        timeout=120,
    ):
        flags, environment = cls._root_flags(
            namespace_mode,
            system_mounts=system_mounts,
            sys_paths=sys_paths,
            gpu_resources=gpu_resources,
            binary=binary,
        )
        record = cls._record_process(
            name,
            flags + list(command),
            expected,
            timeout=timeout,
            env=dict(os.environ, CAPE_MATRIX_OUTER_SECRET="must-not-cross-clearenv"),
        )
        record["namespace_mode"] = namespace_mode
        record["sandbox_environment"] = environment
        record["system_mounts"] = system_mounts
        record["sys_paths"] = list(sys_paths)
        return record

    def test_01_versions_and_assembled_root_namespace_matrix(self):
        version = self._record_process(
            "bubblewrap-under-test-version",
            (self.binary, "--version"),
            "exit 0 and version 0.11.2",
        )
        self.assertEqual(version["exit_code"], 0, version["stderr"])
        help_record = self._record_process(
            "bubblewrap-under-test-help",
            (self.binary, "--help"),
            "native overlay flags present",
        )
        self.assertEqual(help_record["exit_code"], 0, help_record["stderr"])
        self.assertIn("--overlay-src", help_record["stdout"] + help_record["stderr"])

        if self.legacy_binary:
            legacy_help = self._record_process(
                "legacy-bubblewrap-help",
                (self.legacy_binary, "--help"),
                "record whether native overlay flags exist",
            )
            legacy_help["native_overlay_flags"] = "--overlay-src" in (
                legacy_help["stdout"] + legacy_help["stderr"]
            )
            legacy_confined = self._assembled_run(
                "legacy-bubblewrap-assembled-root-rootful-confinement",
                self._confinement_probe_command("isolated"),
                namespace_mode="rootful",
                binary=self.legacy_binary,
            )
            self.assertEqual(
                legacy_confined["exit_code"],
                0,
                legacy_confined["stderr"] + legacy_confined["stdout"],
            )
            legacy_confined["observations"] = json.loads(
                legacy_confined["stdout"]
            )

        commands = (
            ("true", ("/usr/bin/true",)),
            ("shell", ("/bin/sh", "-c", "test x$(printf y) = xy")),
            ("python", ("/usr/bin/python3", "-c", "print(1)")),
        )
        for label, command in commands:
            record = self._assembled_run(
                "assembled-root-rootful-" + label,
                command,
                namespace_mode="rootful",
            )
            self.assertEqual(record["exit_code"], 0, record["stderr"])

        environment = self._assembled_run(
            "assembled-root-rootful-clear-environment",
            ("/usr/bin/env", "-0"),
            namespace_mode="rootful",
        )
        self.assertEqual(environment["exit_code"], 0, environment["stderr"])
        observed_environment = {}
        for entry in environment["stdout"].split("\0"):
            if entry:
                key, value = entry.split("=", 1)
                observed_environment[key] = value
        environment["observed_environment"] = observed_environment
        self.assertEqual(observed_environment, self.explicit_environment)

        confined = self._assembled_run(
            "assembled-root-rootful-file-process-confinement",
            self._confinement_probe_command("isolated"),
            namespace_mode="rootful",
        )
        self.assertEqual(confined["exit_code"], 0, confined["stderr"] + confined["stdout"])
        confined["observations"] = json.loads(confined["stdout"])

        userns = self._assembled_run(
            "assembled-root-userns-file-process-confinement",
            self._confinement_probe_command("isolated"),
            namespace_mode="userns",
            expected="record kernel/outer-container support; success is not assumed",
        )
        if userns["exit_code"] == 0:
            userns["observations"] = json.loads(userns["stdout"])

    def test_02_legacy_whole_root_leak_matrix(self):
        base = [self.binary, "--die-with-parent", "--new-session", "--clearenv"]
        environment = dict(self.explicit_environment)
        env_flags = []
        for key in sorted(environment):
            env_flags.extend(("--setenv", key, environment[key]))

        nonexistent_target = base + [
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            str(self.cache_source),
            "/cape-matrix-target-that-does-not-exist",
        ] + env_flags + ["--cap-drop", "ALL", "--", "/usr/bin/true"]
        nonexistent_record = self._record_process(
            "legacy-whole-root-nonexistent-bind-target",
            nonexistent_target,
            "must fail because the read-only host root cannot create the target",
        )
        self.assertNotEqual(nonexistent_record["exit_code"], 0)
        self.assertIn("Read-only file system", nonexistent_record["stderr"])

        fresh = base + [
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            str(self.workspace),
            "/workspace",
        ] + env_flags + ["--cap-drop", "ALL", "--"]
        fresh_record = self._record_process(
            "legacy-whole-root-fresh-pid",
            fresh + list(self._confinement_probe_command("leaky-files-private-processes")),
            "must demonstrate file leak despite private process view",
        )
        self.assertEqual(fresh_record["exit_code"], 0, fresh_record["stderr"] + fresh_record["stdout"])
        fresh_record["observations"] = json.loads(fresh_record["stdout"])

        host_view = base + [
            "--ro-bind",
            "/",
            "/",
            "--ro-bind",
            "/proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            str(self.workspace),
            "/workspace",
        ] + env_flags + ["--cap-drop", "ALL", "--"]
        host_record = self._record_process(
            "legacy-whole-root-host-proc-view",
            host_view + list(self._confinement_probe_command("leaky-files-and-processes")),
            "must demonstrate both file and foreign-process leaks",
        )
        self.assertEqual(host_record["exit_code"], 0, host_record["stderr"] + host_record["stdout"])
        host_record["observations"] = json.loads(host_record["stdout"])

    def test_03_native_overlay_stateful_assembled_root(self):
        session = self.base / "native-session"
        fresh_session = self.base / "native-fresh-session"
        for root in (session, fresh_session):
            root.mkdir()

        def mounts(state_root):
            result = []
            for index, path in enumerate(self.system_roots):
                upper = state_root / ("upper-{}".format(index))
                work = state_root / ("work-{}".format(index))
                upper.mkdir()
                work.mkdir()
                result.append(("--overlay-src", path, "--overlay", str(upper), str(work), path))
            return tuple(result)

        persistent_mounts = mounts(session)
        first = self._assembled_run(
            "native-overlay-stateful-write",
            ("/bin/sh", "-c", "mkdir -p /opt/cape-matrix-state && printf persisted >/opt/cape-matrix-state/value"),
            system_mounts=persistent_mounts,
        )
        self.assertEqual(first["exit_code"], 0, first["stderr"])
        second = self._assembled_run(
            "native-overlay-stateful-second-run",
            ("/bin/cat", "/opt/cape-matrix-state/value"),
            system_mounts=persistent_mounts,
        )
        self.assertEqual(second["exit_code"], 0, second["stderr"])
        self.assertEqual(second["stdout"], "persisted")
        confined = self._assembled_run(
            "native-overlay-stateful-file-process-confinement",
            self._confinement_probe_command("isolated"),
            system_mounts=persistent_mounts,
        )
        self.assertEqual(confined["exit_code"], 0, confined["stderr"] + confined["stdout"])
        confined["observations"] = json.loads(confined["stdout"])

        fresh_mounts = mounts(fresh_session)
        absent = self._assembled_run(
            "native-overlay-fresh-session-does-not-share-state",
            ("/bin/sh", "-c", "test ! -e /opt/cape-matrix-state/value"),
            system_mounts=fresh_mounts,
        )
        self.assertEqual(absent["exit_code"], 0, absent["stderr"])

    @classmethod
    @contextlib.contextmanager
    def _fuse_system_mounts(cls, state_root):
        processes = []
        mount_specs = []
        mount_records = []
        try:
            for index, path in enumerate(cls.system_roots):
                upper = state_root / ("upper-{}".format(index))
                work = state_root / ("work-{}".format(index))
                merged = state_root / ("merged-{}".format(index))
                for directory in (upper, work, merged):
                    directory.mkdir(exist_ok=True)
                argv = [
                    os.environ.get("PYCAPE_FUSE_OVERLAYFS_BINARY", "fuse-overlayfs"),
                    "-f",
                    "-o",
                    "lowerdir={},upperdir={},workdir={}".format(path, upper, work),
                    str(merged),
                ]
                process = subprocess.Popen(
                    argv,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if os.path.ismount(merged):
                        break
                    if process.poll() is not None:
                        _stdout, stderr = process.communicate()
                        raise RuntimeError("fuse-overlayfs failed: " + stderr)
                    time.sleep(0.02)
                if not os.path.ismount(merged):
                    process.terminate()
                    raise RuntimeError("timed out mounting fuse-overlayfs at " + str(merged))
                processes.append((process, merged))
                mount_specs.append(("--bind", str(merged), path))
                mount_records.append({"argv": argv, "path": path})
            yield tuple(mount_specs), mount_records
        finally:
            unmount = os.environ.get("PYCAPE_FUSERMOUNT_BINARY") or shutil.which("fusermount3") or shutil.which("fusermount")
            for process, merged in reversed(processes):
                if unmount:
                    subprocess.run(
                        [unmount, "-u", str(merged)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=20,
                    )
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=2)
                if process.poll() is None:
                    process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.communicate(timeout=2)

    def test_04_fuse_overlay_stateful_assembled_root(self):
        if not shutil.which(os.environ.get("PYCAPE_FUSE_OVERLAYFS_BINARY", "fuse-overlayfs")):
            self.skipTest("fuse-overlayfs is unavailable")
        session = self.base / "fuse-session"
        fresh_session = self.base / "fuse-fresh-session"
        session.mkdir()
        fresh_session.mkdir()
        fallback_binary = self.legacy_binary or self.binary

        try:
            with self._fuse_system_mounts(session) as (mounts, mount_records):
                first = self._assembled_run(
                    "fuse-overlay-stateful-write",
                    ("/bin/sh", "-c", "mkdir -p /opt/cape-matrix-fuse && printf persisted >/opt/cape-matrix-fuse/value"),
                    system_mounts=mounts,
                    binary=fallback_binary,
                )
                first["fuse_mounts"] = mount_records
                self.assertEqual(first["exit_code"], 0, first["stderr"])
            with self._fuse_system_mounts(session) as (mounts, mount_records):
                second = self._assembled_run(
                    "fuse-overlay-stateful-second-run",
                    ("/bin/cat", "/opt/cape-matrix-fuse/value"),
                    system_mounts=mounts,
                    binary=fallback_binary,
                )
                second["fuse_mounts"] = mount_records
                self.assertEqual(second["exit_code"], 0, second["stderr"])
                self.assertEqual(second["stdout"], "persisted")
                confined = self._assembled_run(
                    "fuse-overlay-stateful-file-process-confinement",
                    self._confinement_probe_command("isolated"),
                    system_mounts=mounts,
                    binary=fallback_binary,
                )
                self.assertEqual(confined["exit_code"], 0, confined["stderr"] + confined["stdout"])
                confined["observations"] = json.loads(confined["stdout"])
            with self._fuse_system_mounts(fresh_session) as (mounts, mount_records):
                absent = self._assembled_run(
                    "fuse-overlay-fresh-session-does-not-share-state",
                    ("/bin/sh", "-c", "test ! -e /opt/cape-matrix-fuse/value"),
                    system_mounts=mounts,
                    binary=fallback_binary,
                )
                absent["fuse_mounts"] = mount_records
                self.assertEqual(absent["exit_code"], 0, absent["stderr"])
        except RuntimeError as exc:
            self.records.append(
                {
                    "name": "fuse-overlay-stateful-setup",
                    "expected": "record whether per-system-path FUSE overlays work",
                    "exit_code": 125,
                    "stderr": str(exc),
                }
            )
            self.skipTest(str(exc))

    def test_05_gpu_sysfs_minimization_and_ordered_gates(self):
        if not self.gpu_uuid:
            self.skipTest("PYCAPE_BUBBLEWRAP_TEST_GPU_UUID is not configured")
        resources = _nvidia_resources(self.gpu_uuid)
        baseline_environment = dict(os.environ)
        baseline_environment["CUDA_VISIBLE_DEVICES"] = self.gpu_uuid
        visibility_command = (
            resources["nvidia_smi"],
            "--query-gpu=uuid",
            "--format=csv,noheader",
        )
        direct_visibility = self._record_process(
            "outer-baseline-nvidia-smi",
            visibility_command,
            "exact selected UUID before Bubblewrap",
            env=baseline_environment,
        )
        self.assertEqual(direct_visibility["exit_code"], 0, direct_visibility["stderr"])
        direct_visibility["visible_uuids"] = tuple(
            line.strip() for line in direct_visibility["stdout"].splitlines() if line.strip()
        )

        cuda_probe = (
            sys.executable,
            "-c",
            "import ctypes,sys; c=ctypes.CDLL('libcuda.so.1'); n=ctypes.c_int(); "
            "a=c.cuInit(0); b=c.cuDeviceGetCount(ctypes.byref(n)) if a == 0 else a; "
            "print(a,b,n.value); sys.exit(0 if a == 0 and b == 0 and n.value == 1 else 1)",
        )
        direct_cuda = self._record_process(
            "outer-baseline-cuinit",
            cuda_probe,
            "cuInit and exactly one visible GPU before Bubblewrap",
            env=baseline_environment,
        )
        if direct_cuda["exit_code"] != 0:
            self.skipTest(
                "outer exact-image CUDA baseline failed; Bubblewrap GPU attribution is invalid: "
                + direct_cuda["stderr"]
                + direct_cuda["stdout"]
            )

        variants = (
            ("no-sysfs", ()),
            ("devices", ("/sys/devices",)),
            ("devices-bus-pci-class", ("/sys/devices", "/sys/bus/pci", "/sys/class")),
        )
        selected = None
        kernel_command = (
            sys.executable,
            "-c",
            "import torch; assert torch.cuda.device_count() == 1; "
            "x=torch.arange(32,device='cuda'); assert int((x*x).sum().item()) == 10416; "
            "print('cuda-kernel-ok')",
        )
        for label, sys_paths in variants:
            visibility = self._assembled_run(
                "gpu-{}-nvidia-smi".format(label),
                visibility_command,
                sys_paths=sys_paths,
                gpu_resources=resources,
                timeout=120,
            )
            if visibility["exit_code"] != 0:
                continue
            visible = tuple(line.strip() for line in visibility["stdout"].splitlines() if line.strip())
            visibility["visible_uuids"] = visible
            if visible != (self.gpu_uuid,):
                continue
            initialized = self._assembled_run(
                "gpu-{}-cuinit".format(label),
                cuda_probe,
                sys_paths=sys_paths,
                gpu_resources=resources,
                timeout=120,
            )
            if initialized["exit_code"] != 0:
                continue
            kernel = self._assembled_run(
                "gpu-{}-torch-kernel".format(label),
                kernel_command,
                sys_paths=sys_paths,
                gpu_resources=resources,
                timeout=180,
            )
            if kernel["exit_code"] == 0:
                selected = (label, sys_paths)
                break
        self.assertIsNotNone(selected, "no narrow-sysfs Bubblewrap GPU variant passed all ordered gates")
        self.records.append(
            {
                "name": "gpu-minimal-sysfs-selection",
                "expected": "first variant passing nvidia-smi, cuInit, and a real Torch kernel",
                "exit_code": 0,
                "selected_variant": selected[0],
                "sys_paths": list(selected[1]),
                "gpu_minor": resources["minor"],
            }
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
