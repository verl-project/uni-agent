# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import subprocess
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parent / "manual"
    / "prerequisites.sh"
)


class ManualPrerequisitesScriptTests(unittest.TestCase):
    def test_workspace_tools_and_environment_are_explicit(self):
        source = SCRIPT.read_text(encoding="utf-8")
        packages = source.split("PACKAGES=(", 1)[1].split(")", 1)[0].split()
        for package in ("fuse2fs", "e2fsprogs", "util-linux", "coreutils", "fuse3"):
            self.assertIn(package, packages)
        for helper in ("fuse2fs", "mkfs.ext4", "e2fsck", "dumpe2fs", "findmnt", "flock", "truncate"):
            self.assertIn(helper, source)
        for name in ("FUSE2FS", "MKFS", "E2FSCK", "FUSERMOUNT", "FINDMNT"):
            self.assertIn("export CAPE_WORKSPACE_{}_BIN=".format(name), source)
        self.assertIn("system_tool fusermount || system_tool fusermount3", source)
        self.assertIn("workspace_mount_capability=not-tested-use-cape-node-probe", source)
        self.assertIn("workspace_toolchain_installed=yes", source)
        self.assertIn("workspace_fuse_library_version", source)
        self.assertIn("Apptainer unsquash is not a workspace fallback", source)
        self.assertNotIn("e2fsck -y", source)
        self.assertNotIn("mkfs.ext4 -F", source)

    def test_help_documents_workspace_prerequisites_and_pinned_image(self):
        result = subprocess.run([str(SCRIPT), "--help"], check=True, text=True, capture_output=True)
        for value in ("nvcr.io/nvidia/pytorch:26.09-py3", "fuse2fs", "e2fsprogs", "existing workspace image"):
            self.assertIn(value, result.stdout)

    def test_entrypoint_and_help_are_cape_node_specific(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(source.splitlines()[0], "#!/usr/bin/env bash")
        self.assertTrue(stat.S_IMODE(SCRIPT.stat().st_mode) & stat.S_IXUSR)

        result = subprocess.run(
            [str(SCRIPT), "--help"],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("Usage: prerequisites.sh", result.stdout)
        self.assertIn("probe all supported installed container runtimes", result.stdout)
        self.assertIn("local-process is never an automatic fallback", result.stdout)
        self.assertIn("CAPE_NODE_RUNTIME_ENV_FILE", result.stdout)
        self.assertIn("CAPE_NODE_RUNTIME_REPORT_FILE", result.stdout)
        self.assertIn("--check", result.stdout)
        self.assertNotIn("install-node-runtimes.sh", result.stdout)
        self.assertNotIn("run_all.sh", result.stdout)

    def test_uses_node_wrapper_exports_not_overhead_harness_exports(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for variable in (
            "CAPE_APPTAINER_BIN",
            "CAPE_BUBBLEWRAP_BIN",
            "CAPE_BUBBLEWRAP_SYSTEM_PATHS",
            "CAPE_PODMAN_BIN",
            "CAPE_APPTAINER_UNSQUASH",
        ):
            self.assertIn("export {}=".format(variable), source)
        self.assertNotIn("CRO_", source)
        self.assertNotIn("export BWRAP=", source)
        self.assertNotIn("export APPTAINER=", source)
        self.assertNotIn("export PODMAN=", source)

    def test_check_and_force_are_mutually_exclusive(self):
        result = subprocess.run(
            [str(SCRIPT), "--check", "--force"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--check and --force are mutually exclusive", result.stderr)

    def test_rejects_virtual_kernel_roots_before_runtime_checks(self):
        for path in ("/", "/proc", "/sys", "/dev", "/run"):
            with self.subTest(path=path):
                env = dict(os.environ, CAPE_BUBBLEWRAP_SYSTEM_PATHS=path)
                result = subprocess.run(
                    [str(SCRIPT), "--check"],
                    env=env,
                    check=False,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("must not expose", result.stderr)

    def test_records_installation_separately_from_outer_capabilities(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for report_field in (
            "bubblewrap_namespace_smoke",
            "dev_fuse_usable",
            "cgroup_subtree_control_writable",
            "podman_standard_cgroups_likely_usable",
            "apptainer_unsquash_enabled",
        ):
            self.assertIn(report_field, source)
        self.assertIn("packages are installed but the current outer environment", source)

    def test_bounds_and_distinguishes_bubblewrap_envelope_probes(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("CAPE_NODE_RUNTIME_PROBE_TIMEOUT_SECONDS", source)
        self.assertIn("timeout --signal=TERM --kill-after=2s", source)
        for report_field in (
            "bubblewrap_resolved_envelope",
            "bubblewrap_userns_fresh_proc_smoke",
            "bubblewrap_rootful_fresh_proc_smoke",
            "bubblewrap_assembled_root_smoke",
            "bubblewrap_gpu_capability=not-tested-use-cape-node-probe",
            "runtime_probe_timeout_seconds",
        ):
            self.assertIn(report_field, source)
        self.assertIn("--unshare-pid --unshare-ipc --unshare-uts", source)
        self.assertIn("--proc /proc", source)
        self.assertIn("--cap-drop ALL", source)
        self.assertIn("--clearenv", source)
        self.assertIn("--tmpfs /", source)
        self.assertNotIn("--ro-bind /proc /proc", source)
        self.assertNotIn("--ro-bind / /", source)
        self.assertNotIn("--bind /proc /proc", source)

    def test_bootstraps_dpkg_tables_removed_from_the_outer_image(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("bootstrap_dpkg_architecture_tables", source)
        self.assertIn("local table_dir=/usr/share/dpkg", source)
        self.assertIn('"$table_dir/cputable"', source)
        self.assertIn("base-gnu-linux-<cpu>", source)
        self.assertIn("/var/lib/apt/lists/partial", source)
        self.assertIn("/var/cache/apt/archives/partial", source)
        self.assertIn("ubuntu-keyring_2023.11.28.1_all.deb", source)
        self.assertIn(
            "36de43b15853ccae0028e9a767613770c704833f82586f28eb262f0311adb8a8",
            source,
        )
        self.assertNotIn("trusted=yes", source)
        self.assertNotIn("AllowUnauthenticated", source)
        self.assertIn("Acquire::ForceIPv4=true", source)
        self.assertIn("Acquire::Retries=5", source)
        self.assertIn("Acquire::http::Timeout=30", source)
        self.assertIn("Acquire::https::Timeout=30", source)
        self.assertIn("prefer_https_ubuntu_sources", source)
        self.assertIn("https://ports.ubuntu.com/", source)
        self.assertIn("https://archive.ubuntu.com/", source)
        self.assertIn("https://security.ubuntu.com/", source)
        self.assertIn("detach_mounted_localtime_target", source)
        self.assertIn('readlink -f /etc/localtime', source)
        self.assertIn('/proc/self/mountinfo', source)
        self.assertIn('umount "$timezone_path"', source)
        self.assertIn('dpkg --configure -a', source)
        self.assertIn(
            'if ! DEBIAN_FRONTEND=noninteractive dpkg --configure -a; then',
            source,
        )
        self.assertIn("APT will resolve remaining dependencies", source)
        self.assertIn("dpkg_repair_needed=1", source)
        self.assertIn('"${apt_get[@]}" --fix-broken install', source)
        self.assertIn("libc-bin", source)
        self.assertIn("/usr/sbin/ldconfig.real", source)
        self.assertIn("mkdir -p /var/tmp", source)
        self.assertIn("chmod 1777 /var/tmp", source)
        self.assertLess(
            source.index('DEBIAN_FRONTEND=noninteractive "${apt_get[@]}" update'),
            source.index('"${apt_get[@]}" --fix-broken install'),
        )
        self.assertLess(
            source.index("detach_mounted_localtime_target"),
            source.index('DEBIAN_FRONTEND=noninteractive "${apt_get[@]}" update'),
        )
        self.assertLess(
            source.index("bootstrap_dpkg_architecture_tables"),
            source.index('DPKG_ARCH="$(dpkg --print-architecture)"'),
        )


if __name__ == "__main__":
    unittest.main()
