# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installer control flow with isolated package/network/compiler command doubles."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUTER = r'''
import json, os, pathlib, shutil, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
root = pathlib.Path(os.environ["FIXTURE_ROOT"])
with (root / "calls").open("a") as out:
    out.write(json.dumps([name] + args) + "\n")
fixed = (root / "packages-fixed").exists()
if name in os.environ.get("BROKEN", "").split(",") and not fixed:
    sys.exit(127)
if name == "uname":
    print(os.environ.get("MOCK_OS", "Linux") if args == ["-s"] else os.environ.get("MOCK_ARCH", "x86_64"))
elif name == "cc" and args == ["-print-file-name=libc.a"]:
    print(root / "libc.a")
elif name == "rustc":
    if args == ["-vV"]: print("host: x86_64-unknown-linux-gnu")
    else: print("rustc " + ("1.90.0" if (root / "rust-fixed").exists() else os.environ.get("RUST_VERSION", "1.71.1")))
elif name == "cargo":
    if args == ["--version"]:
        print("cargo " + ("1.90.0" if (root / "rust-fixed").exists() else os.environ.get("RUST_VERSION", "1.71.1")))
    elif args[0] == "build":
        if os.environ.get("BUILD_FAIL"): sys.exit(42)
        assert args == ["build", "--release", "--locked", "--target", "x86_64-unknown-linux-gnu"]
        dest = pathlib.Path(os.environ["CARGO_TARGET_DIR"]) / "x86_64-unknown-linux-gnu/release/rscape"
        dest.parent.mkdir(parents=True, exist_ok=True)
        source = pathlib.Path("source-marker").read_text()
        dest.write_text("#!/bin/sh\n# " + source + "\nexit 0\n")
        dest.chmod(0o755)
elif name == "resize2fs":
    print("resize2fs 1.47.0 (fixture)")
    sys.exit(1)
elif name == "unsquashfs":
    assert args == ["-version"]
    print(os.environ.get("UNSQUASHFS_OUTPUT", "unsquashfs version 4.3 (2021/10/8)"))
    sys.exit(int(os.environ.get("UNSQUASHFS_EXIT", "1")))
elif name == "squashfuse":
    print("squashfuse 0.5.2\nUsage: squashfuse ARCHIVE MOUNTPOINT")
    sys.exit(254)
elif name == "sudo":
    os.execvp(args[0], args)
elif name == "apt-get":
    if os.environ.get("APT_FAIL"): sys.exit(1)
    if args[0] == "install": (root / "packages-fixed").touch()
elif name == "apt-cache":
    print("Candidate: " + ("(none)" if os.environ.get("NO_APPTAINER") and not (root / "ppa").exists() else "1.4.0"))
elif name == "add-apt-repository":
    (root / "ppa").touch()
elif name == "curl" and "-o" in args:
    target = pathlib.Path(args[args.index("-o") + 1])
    target.write_text('#!/bin/sh\nmkdir -p "$CARGO_HOME/bin"\ncp "$FIXTURE_ROOT/bin/router" "$CARGO_HOME/bin/rustup"\nchmod 755 "$CARGO_HOME/bin/rustup"\n"$CARGO_HOME/bin/rustup" toolchain install stable --profile minimal\n')
elif name == "rustup":
    assert args == ["toolchain", "install", "stable", "--profile", "minimal"]
    (root / "rust-fixed").touch()
    directory = pathlib.Path(sys.argv[0]).parent
    for tool in ("rustc", "cargo"):
        if directory != root / "bin": shutil.copyfile(root / "bin/router", directory / tool)
        (directory / tool).chmod(0o755)
elif name in ("dcgmi", "nv-hostengine", "prometheus", "telegraf"):
    print(name + " version " + {"dcgmi":"4.5.2", "nv-hostengine":"4.5.2", "prometheus":"3.7.3", "telegraf":"1.36.3"}[name])
else:
    print(name + " 1.47.0 (fixture)")
'''


class SourceInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cape-installer-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.checkout = self.root / "checkout with spaces"
        crate = self.checkout / "rscape"
        crate.mkdir(parents=True)
        (crate / "Cargo.toml").write_text('rust-version = "1.71"\n')
        (crate / "Cargo.lock").write_text("")
        (crate / "source-marker").write_text("first")
        self.os_release = self.root / "os-release"
        self.os_release.write_text("ID=ubuntu\n")
        # Replace only the OS metadata read; system-writing commands remain doubles.
        script = (ROOT / "install.sh").read_text().replace("/etc/os-release", str(self.os_release))
        self.script = self.checkout / "install.sh"
        self.script.write_text(script)
        shutil.copytree(ROOT / "install", self.checkout / "install")
        (self.root / "libc.a").touch()
        router = self.bin / "router"
        router.write_text("#!" + sys.executable + "\n" + ROUTER)
        router.chmod(0o755)
        real = ("bash", "sh", "sed", "head", "cat", "dirname", "mkdir", "mktemp", "cp", "chmod", "mv", "rm", "stat", "id", "sha256sum")
        for name in real:
            (self.bin / name).symlink_to(shutil.which(name))
        mocks = ("uname cc make ar pkg-config bwrap apptainer fuse-overlayfs squashfuse mksquashfs unsquashfs "
                 "fuse2fs mkfs.ext4 e2fsck resize2fs dumpe2fs findmnt flock mount umount truncate timeout nice "
                 "fusermount3 fusermount rustc cargo rustup curl sudo apt-get apt-cache add-apt-repository dcgmi nv-hostengine prometheus telegraf").split()
        for name in mocks:
            (self.bin / name).symlink_to(router)
        self.prefix = self.root / "install with spaces"
        self.env = dict(os.environ, PATH=str(self.bin), FIXTURE_ROOT=str(self.root),
                        CARGO_HOME=str(self.root / "cargo-home"), CARGO_TARGET_DIR=str(self.root / "build"))

    def run_install(self, *args, success=True, **env):
        result = subprocess.run([str(self.bin / "bash"), str(self.script), "--prefix", str(self.prefix), *args],
                                cwd=self.root, env=dict(self.env, **env), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result

    def calls(self):
        path = self.root / "calls"
        return [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines()]

    def test_always_build_current_source_reuses_dependencies_and_backs_up(self):
        self.run_install("--yes")
        first = (self.prefix / "bin/rscape").read_bytes()
        (self.checkout / "rscape/source-marker").write_text("second")
        self.run_install("--yes")
        self.assertNotEqual(first, (self.prefix / "bin/rscape").read_bytes())
        self.assertEqual(first, (self.prefix / "bin/rscape.previous").read_bytes())
        self.assertEqual(sum(call[:2] == ["cargo", "build"] for call in self.calls()), 2)
        self.assertFalse(any(call[0] in ("apt-get", "curl", "rustup", "sudo") for call in self.calls()))

    def test_check_is_read_only_and_reports_missing(self):
        self.run_install("--check")
        self.run_install("--check", success=False, BROKEN="bwrap")
        self.assertFalse(self.prefix.exists())
        self.assertFalse(any(call[0] in ("apt-get", "curl", "rustup") or call[:2] == ["cargo", "build"] for call in self.calls()))

    def test_only_missing_package_is_installed(self):
        self.run_install("--yes", BROKEN="bwrap")
        installs = [call for call in self.calls() if call[:2] == ["apt-get", "install"]]
        self.assertEqual(installs, [["apt-get", "install", "-y", "--no-remove", "--no-install-recommends", "bubblewrap"]])

    def test_ppa_requires_explicit_unattended_consent(self):
        self.run_install("--yes", success=False, BROKEN="apptainer", NO_APPTAINER="1")
        self.assertFalse((self.root / "ppa").exists())
        self.run_install("--yes", "--allow-apptainer-ppa", BROKEN="apptainer", NO_APPTAINER="1")
        self.assertTrue((self.root / "ppa").exists())

    def test_old_rust_is_upgraded_without_shell_changes(self):
        self.run_install("--yes", RUST_VERSION="1.70.0")
        self.assertIn(["rustup", "toolchain", "install", "stable", "--profile", "minimal"], self.calls())
        self.assertFalse(any(call[0] == "curl" and "-o" in call for call in self.calls()))

    def test_rust_bootstrap_download_then_build(self):
        (self.bin / "rustup").unlink()
        self.run_install("--yes", RUST_VERSION="1.70.0")
        self.assertTrue(any(call[0] == "curl" and "https://sh.rustup.rs" in call for call in self.calls()))
        self.assertTrue((self.prefix / "bin/rscape").exists())

    def test_rustup_path_change_does_not_hide_caller_reminder(self):
        (self.bin / "rustup").unlink()
        result = self.run_install("--yes", "--prefix", self.env["CARGO_HOME"], RUST_VERSION="1.70.0")
        self.assertTrue(result.stdout.rstrip().splitlines()[-1].startswith("export PATH="))

    def test_build_failure_preserves_installed_binary(self):
        self.run_install("--yes")
        previous = (self.prefix / "bin/rscape").read_bytes()
        self.run_install("--yes", success=False, BUILD_FAIL="1")
        self.assertEqual(previous, (self.prefix / "bin/rscape").read_bytes())
        self.assertFalse((self.prefix / "bin/rscape.previous").exists())

    def test_package_failure_never_builds(self):
        self.run_install("--yes", success=False, BROKEN="bwrap", APT_FAIL="1")
        self.assertFalse(any(call[:2] == ["cargo", "build"] for call in self.calls()))

    def test_unsupported_distro_requires_manual_missing_packages(self):
        self.os_release.write_text("ID=fedora\n")
        self.run_install("--yes", success=False, BROKEN="bwrap")
        self.assertFalse(any(call[0] == "apt-get" for call in self.calls()))

    def test_no_sudo_has_actionable_failure(self):
        if os.geteuid() == 0:
            self.skipTest("root needs no sudo")
        (self.bin / "sudo").unlink()
        result = self.run_install("--yes", success=False, BROKEN="bwrap")
        self.assertIn("root/sudo required", result.stderr)

    def test_symlink_destination_is_not_overwritten(self):
        self.prefix.mkdir()
        (self.prefix / "bin").mkdir()
        victim = self.root / "untouched"
        victim.write_text("keep")
        (self.prefix / "bin/rscape").symlink_to(victim)
        self.run_install("--yes", success=False)
        self.assertEqual(victim.read_text(), "keep")

    def test_platform_and_arguments_fail_early(self):
        self.run_install("--bogus", success=False)
        self.run_install("--yes", success=False, MOCK_ARCH="riscv64")
        self.run_install("--yes", success=False, MOCK_OS="Darwin")
        self.assertEqual([call for call in self.calls() if call[0] == "cargo"], [])

    def test_native_compiler_architecture_must_match_machine(self):
        self.run_install("--yes", success=False, MOCK_ARCH="aarch64")
        self.assertFalse(any(call[:2] == ["cargo", "build"] for call in self.calls()))

    def test_path_reminder_is_last_and_omitted_when_directory_is_present(self):
        result = self.run_install("--yes")
        lines = result.stdout.rstrip().splitlines()
        self.assertIn("ACTION REQUIRED", lines[-2])
        self.assertTrue(lines[-1].startswith("export PATH="))
        self.assertIn('"$PATH"', lines[-1])
        result = self.run_install("--yes", PATH=str(self.prefix / "bin") + ":" + self.env["PATH"])
        self.assertNotIn("ACTION REQUIRED", result.stdout)
        self.assertNotIn("export PATH=", result.stdout)
        older = self.bin / "rscape"
        older.write_text("#!/bin/sh\nexit 0\n")
        older.chmod(0o755)
        result = self.run_install("--yes", PATH=self.env["PATH"] + ":" + str(self.prefix / "bin"))
        self.assertTrue(result.stdout.rstrip().splitlines()[-1].startswith("export PATH="))

    def test_unsquashfs_version_exit_zero_and_one_are_usable(self):
        for status in ("0", "1"):
            with self.subTest(status=status):
                self.run_install("--check", UNSQUASHFS_EXIT=status)
        self.assertFalse(any(call[0] == "apt-get" for call in self.calls()))

    def test_unsquashfs_bad_output_or_execution_failure_is_not_usable(self):
        for status, output in (("0", ""), ("1", "loader error"),
                               ("0", "unsquashfs version 4.bad"),
                               ("2", "unsquashfs version 4.3"),
                               ("126", "unsquashfs version 4.3"),
                               ("127", "unsquashfs version 4.3")):
            with self.subTest(status=status, output=output):
                result = self.run_install("--check", success=False,
                    UNSQUASHFS_EXIT=status, UNSQUASHFS_OUTPUT=output)
                self.assertIn("missing/unusable: unsquashfs", result.stdout)
