#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install container runtimes and workspace tools consumed by manual/register_node.sh into the writable
# root filesystem of an allocated node's outer container.
set -euo pipefail

EXPECTED_OUTER_IMAGE=nvcr.io/nvidia/pytorch:26.09-py3
NODE_ARCH="$(uname -m)"
STATE_DIR="${CAPE_NODE_RUNTIME_STATE_DIR:-${TMPDIR:-/tmp}/cape-manual-runtimes-$(id -u)-$NODE_ARCH}"
ENV_FILE="${CAPE_NODE_RUNTIME_ENV_FILE:-$STATE_DIR/node-runtime.env}"
REPORT_FILE="${CAPE_NODE_RUNTIME_REPORT_FILE:-$STATE_DIR/node-runtime-report.txt}"
FORCE=0
CHECK_ONLY=0
PROBE_TIMEOUT_SECONDS="${CAPE_NODE_RUNTIME_PROBE_TIMEOUT_SECONDS:-10}"
BWRAP_SYSTEM_PATH_TEXT="${CAPE_BUBBLEWRAP_SYSTEM_PATHS:-/usr,/etc,/opt}"

usage() {
  cat <<EOF
Usage: prerequisites.sh [OPTIONS]

Install native Apptainer, Bubblewrap, Podman, fuse-overlayfs, fuse2fs,
e2fsprogs (mkfs.ext4/e2fsck), util-linux, coreutils, and required
helpers into the writable Ubuntu root of the current allocated-node outer
container. This script never calls sudo and does not modify the bare node host.

Run it once inside a root-mapped $EXPECTED_OUTER_IMAGE container. Reusing the
same writable Enroot/Pyxis root filesystem reuses the packages; a fresh root
filesystem needs another installation.

Options:
  --env-file PATH       Write sourceable CAPE node runtime exports
                        (default: CAPE_NODE_RUNTIME_ENV_FILE or
                         $STATE_DIR/node-runtime.env)
  --report-file PATH    Write versions and outer-capability observations
                        (default: CAPE_NODE_RUNTIME_REPORT_FILE or
                         $STATE_DIR/node-runtime-report.txt)
  --check               Verify packages and refresh artifacts without APT
  --force               Run APT even when every package is already ready
  -h, --help            Show this help

Environment:
  CAPE_NODE_RUNTIME_STATE_DIR   Node-local artifact directory
  CAPE_NODE_RUNTIME_ENV_FILE    Default --env-file value
  CAPE_NODE_RUNTIME_REPORT_FILE Default --report-file value
  CAPE_NODE_RUNTIME_PROBE_TIMEOUT_SECONDS
                               Bound each live runtime probe (default: 10)
  CAPE_BUBBLEWRAP_SYSTEM_PATHS Comma/newline-separated node-local system
                               paths (default: /usr,/etc,/opt). Host root and
                               /proc, /sys, /dev, /run are forbidden.
  CAPE_NODE_OUTER_IMAGE         Actual outer-image identity for the report

Workspace images use ext4 + fuse2fs. Package installation does not prove
mount capability: /dev/fuse, namespaces, and runtime access are verified
by the node's workspace probe. This installer never formats or repairs an
existing workspace image and does not change host FUSE configuration.

After success, source the generated environment before starting register_node.sh. CAPE
will probe all supported installed container runtimes by default. CAPE_RUNTIME_ADAPTERS
explicitly selects required adapters; local-process is never an automatic fallback.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

require_value() {
  [ "$2" -ge 2 ] && [ -n "$3" ] || die "$1 requires a value"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --env-file)
      require_value "$1" "$#" "${2:-}"
      ENV_FILE="$2"
      shift 2
      ;;
    --env-file=*)
      ENV_FILE="${1#*=}"
      [ -n "$ENV_FILE" ] || die "--env-file requires a value"
      shift
      ;;
    --report-file)
      require_value "$1" "$#" "${2:-}"
      REPORT_FILE="$2"
      shift 2
      ;;
    --report-file=*)
      REPORT_FILE="${1#*=}"
      [ -n "$REPORT_FILE" ] || die "--report-file requires a value"
      shift
      ;;
    --check)
      CHECK_ONLY=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[ "$CHECK_ONLY" != 1 ] || [ "$FORCE" != 1 ] || \
  die "--check and --force are mutually exclusive"
case "$PROBE_TIMEOUT_SECONDS" in
  ''|*[!0-9]*) die "CAPE_NODE_RUNTIME_PROBE_TIMEOUT_SECONDS must be a positive integer" ;;
esac
[ "$PROBE_TIMEOUT_SECONDS" -ge 1 ] || \
  die "CAPE_NODE_RUNTIME_PROBE_TIMEOUT_SECONDS must be at least 1"

BWRAP_SYSTEM_PATHS=()
declare -A bwrap_system_path_seen=()
bwrap_system_path_lines="${BWRAP_SYSTEM_PATH_TEXT//,/$'\n'}"
command -v readlink >/dev/null 2>&1 || die "readlink was not found"
while IFS= read -r bwrap_system_path; do
  bwrap_system_path="${bwrap_system_path%$'\r'}"
  bwrap_system_path="${bwrap_system_path#"${bwrap_system_path%%[![:space:]]*}"}"
  bwrap_system_path="${bwrap_system_path%"${bwrap_system_path##*[![:space:]]}"}"
  [ -n "$bwrap_system_path" ] || continue
  case "$bwrap_system_path" in
    /*) ;;
    *) die "CAPE_BUBBLEWRAP_SYSTEM_PATHS entries must be absolute paths" ;;
  esac
  [ -d "$bwrap_system_path" ] || \
    die "Bubblewrap system path is not a directory: $bwrap_system_path"
  bwrap_system_path_input="$bwrap_system_path"
  bwrap_system_path="$(readlink -f -- "$bwrap_system_path")" || \
    die "failed to resolve Bubblewrap system path: $bwrap_system_path_input"
  case "$bwrap_system_path" in
    /|/proc|/proc/*|/sys|/sys/*|/dev|/dev/*|/run|/run/*)
      die "CAPE_BUBBLEWRAP_SYSTEM_PATHS must not expose host root or virtual kernel trees: $bwrap_system_path_input"
      ;;
  esac
  if [ -z "${bwrap_system_path_seen[$bwrap_system_path]:-}" ]; then
    BWRAP_SYSTEM_PATHS+=("$bwrap_system_path")
    bwrap_system_path_seen["$bwrap_system_path"]=1
  fi
done <<< "$bwrap_system_path_lines"
[ "${#BWRAP_SYSTEM_PATHS[@]}" -gt 0 ] || \
  die "CAPE_BUBBLEWRAP_SYSTEM_PATHS must not be empty"
BWRAP_SYSTEM_PATHS_CSV="$(IFS=,; printf '%s' "${BWRAP_SYSTEM_PATHS[*]}")"

case "$ENV_FILE" in
  /*) ;;
  *) ENV_FILE="$PWD/${ENV_FILE#./}" ;;
esac
case "$REPORT_FILE" in
  /*) ;;
  *) REPORT_FILE="$PWD/${REPORT_FILE#./}" ;;
esac
[ "$ENV_FILE" != "$REPORT_FILE" ] || \
  die "--env-file and --report-file must be different paths"

[ -r /etc/os-release ] || die "/etc/os-release is unavailable"
# shellcheck disable=SC1091
. /etc/os-release
[ "${ID:-}" = ubuntu ] || die "Ubuntu is required; found ${ID:-unknown}"

for command_name in apt-cache apt-get awk dpkg dpkg-query readlink sed stat timeout; do
  command -v "$command_name" >/dev/null 2>&1 || die "$command_name was not found"
done

bootstrap_dpkg_architecture_tables() {
  local table_dir=/usr/share/dpkg
  local missing=0
  local table
  for table in cputable ostable tupletable abitable; do
    [ -s "$table_dir/$table" ] || missing=1
  done
  [ "$missing" -eq 1 ] || return 0
  [ "$CHECK_ONLY" -eq 0 ] || die \
    "the outer image stripped dpkg architecture tables; rerun without --check in a writable root-mapped container"
  [ "$(id -u)" -eq 0 ] || die \
    "the outer image stripped dpkg architecture tables; root mapping is required to restore them"

  mkdir -p "$table_dir"
  cat >"$table_dir/cputable" <<'EOF'
# Version=1.0
# <Debian name> <GNU name> <config.guess regex> <Bits> <Endian>
amd64		x86_64		(amd64|x86_64)		64	little
arm64		aarch64		aarch64			64	little
ppc64el		powerpc64le	powerpc64le		64	little
EOF
  cat >"$table_dir/ostable" <<'EOF'
# Version=2.0
# <Debian name> <GNU name> <config.guess regex>
base-gnu-linux	linux-gnu	linux[^-]*(-gnu.*)?
EOF
  cat >"$table_dir/tupletable" <<'EOF'
# Version=1.0
# <Debian arch tuple> <Debian arch name>
base-gnu-linux-<cpu>	<cpu>
EOF
  cat >"$table_dir/abitable" <<'EOF'
# Version=2.0
# No ABI overrides are required for the supported native architectures.
EOF
  chmod 0644 \
    "$table_dir/cputable" "$table_dir/ostable" \
    "$table_dir/tupletable" "$table_dir/abitable"
  dpkg --print-architecture >/dev/null 2>&1 || die \
    "failed to restore dpkg architecture tables in the writable outer image"
  printf '%s\n' \
    "restored dpkg architecture tables stripped from the outer image"
}

bootstrap_dpkg_architecture_tables
DPKG_ARCH="$(dpkg --print-architecture)"
case "$NODE_ARCH:$DPKG_ARCH" in
  x86_64:amd64|aarch64:arm64|ppc64le:ppc64el) ;;
  *) die "node architecture $NODE_ARCH does not match Debian architecture $DPKG_ARCH" ;;
esac

OUTER_IMAGE="${CAPE_NODE_OUTER_IMAGE:-unknown}"
case "${NVIDIA_PYTORCH_VERSION:-}" in
  26.09*) [ "$OUTER_IMAGE" != unknown ] || OUTER_IMAGE="$EXPECTED_OUTER_IMAGE" ;;
esac

PACKAGES=(
  apptainer
  bubblewrap
  conmon
  containernetworking-plugins
  crun
  coreutils
  e2fsprogs
  fuse-overlayfs
  fuse2fs
  fuse3
  libc-bin
  podman
  slirp4netns
  squashfs-tools
  uidmap
  util-linux
)

system_tool() {
  local path
  for path in "/usr/sbin/$1" "/usr/bin/$1"; do
    if [ -x "$path" ]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

package_is_native() {
  local package="$1"
  local architecture
  local status
  status="$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)"
  [ "$status" = "install ok installed" ] || return 1
  architecture="$(dpkg-query -W -f='${Architecture}' "$package" 2>/dev/null || true)"
  [ "$architecture" = all ] || [ "$architecture" = "$DPKG_ARCH" ]
}

packages_ready() {
  local package
  for package in "${PACKAGES[@]}"; do
    package_is_native "$package" || return 1
  done
  local helper
  for helper in fuse2fs mkfs.ext4 e2fsck resize2fs dumpe2fs findmnt flock truncate cp sha256sum; do
    system_tool "$helper" >/dev/null || return 1
  done
  [ -x /usr/bin/apptainer ] &&
    [ -x /usr/bin/bwrap ] &&
    [ -x /usr/bin/podman ] &&
    [ -x /usr/bin/crun ] &&
    [ -x /usr/bin/fuse-overlayfs ] &&
    [ -x /usr/bin/fusermount3 ] &&
    [ -x /usr/bin/unsquashfs ] &&
    [ -s /usr/sbin/ldconfig.real ] &&
    [ -x /usr/sbin/ldconfig.real ] &&
    [ -d /var/tmp ] &&
    [ "$(stat -c %a /var/tmp 2>/dev/null || true)" = 1777 ]
}

bootstrap_ubuntu_archive_keyring() {
  local keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg
  local package_url=https://archive.ubuntu.com/ubuntu/pool/main/u/ubuntu-keyring/ubuntu-keyring_2023.11.28.1_all.deb
  local package_sha256=36de43b15853ccae0028e9a767613770c704833f82586f28eb262f0311adb8a8
  local package_file
  local package_root
  local actual_sha256
  [ -s "$keyring" ] && return 0
  command -v curl >/dev/null 2>&1 || die \
    "the outer image stripped its Ubuntu archive keyring and curl is unavailable"
  command -v dpkg-deb >/dev/null 2>&1 || die "dpkg-deb was not found"
  command -v sha256sum >/dev/null 2>&1 || die "sha256sum was not found"

  package_file="$(mktemp /tmp/ubuntu-keyring.XXXXXX.deb)"
  package_root="$(mktemp -d /tmp/ubuntu-keyring.XXXXXX)"
  if ! curl --fail --location --silent --show-error \
      "$package_url" --output "$package_file"; then
    rm -rf "$package_file" "$package_root"
    die "failed to download the Ubuntu archive keyring over HTTPS"
  fi
  actual_sha256="$(sha256sum "$package_file" | awk '{print $1}')"
  if [ "$actual_sha256" != "$package_sha256" ]; then
    rm -rf "$package_file" "$package_root"
    die "downloaded Ubuntu archive keyring checksum does not match the pinned Noble package"
  fi
  [ "$(dpkg-deb --field "$package_file" Package)" = ubuntu-keyring ] || \
    die "downloaded archive-keyring package has an unexpected package name"
  [ "$(dpkg-deb --field "$package_file" Version)" = 2023.11.28.1 ] || \
    die "downloaded archive-keyring package has an unexpected version"
  [ "$(dpkg-deb --field "$package_file" Architecture)" = all ] || \
    die "downloaded archive-keyring package has an unexpected architecture"
  dpkg-deb --extract "$package_file" "$package_root"
  [ -s "$package_root$keyring" ] || die \
    "downloaded Ubuntu package did not contain its archive keyring"
  mkdir -p /usr/share/keyrings
  cp -a "$package_root/usr/share/keyrings/." /usr/share/keyrings/
  rm -rf "$package_file" "$package_root"
  [ -s "$keyring" ] || die "failed to restore the Ubuntu archive keyring"
  printf '%s\n' "restored the checksum-pinned Ubuntu Noble archive keyring"
}

prefer_https_ubuntu_sources() {
  local source_file
  for source_file in \
      /etc/apt/sources.list \
      /etc/apt/sources.list.d/*.list \
      /etc/apt/sources.list.d/*.sources; do
    [ -f "$source_file" ] || continue
    sed -i \
      -e 's#http://ports\.ubuntu\.com/#https://ports.ubuntu.com/#g' \
      -e 's#http://archive\.ubuntu\.com/#https://archive.ubuntu.com/#g' \
      -e 's#http://security\.ubuntu\.com/#https://security.ubuntu.com/#g' \
      "$source_file"
  done
}

detach_mounted_localtime_target() {
  local timezone_path
  timezone_path="$(readlink -f /etc/localtime 2>/dev/null || true)"
  case "$timezone_path" in
    /usr/share/zoneinfo/*) ;;
    *) return 0 ;;
  esac

  awk -v path="$timezone_path" \
    '$5 == path { found = 1 } END { exit !found }' \
    /proc/self/mountinfo || return 0
  command -v umount >/dev/null 2>&1 || die \
    "the outer container mounted $timezone_path but umount is unavailable"
  umount "$timezone_path" || die \
    "failed to detach the outer container timezone mount at $timezone_path"
  if awk -v path="$timezone_path" \
      '$5 == path { found = 1 } END { exit !found }' \
      /proc/self/mountinfo; then
    die "the outer container timezone mount remains at $timezone_path"
  fi
  printf 'detached outer-container timezone mount at %s for package installation\n' \
    "$timezone_path"
}

install_packages() {
  local -a apt_get=(
    apt-get
    -o Acquire::ForceIPv4=true
    -o Acquire::Retries=5
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
  )
  local dpkg_repair_needed=0
  [ "$(id -u)" -eq 0 ] || die \
    "runtime packages are missing; run inside a writable root-mapped $EXPECTED_OUTER_IMAGE container"

  mkdir -p \
    /var/lib/apt/lists/partial \
    /var/cache/apt/archives/partial \
    /var/lib/dpkg/updates \
    /var/log/apt
  mkdir -p /var/tmp
  chmod 1777 /var/tmp
  bootstrap_ubuntu_archive_keyring
  prefer_https_ubuntu_sources
  export DEBIAN_FRONTEND=noninteractive
  export NEEDRESTART_MODE=a
  export TZ=Etc/UTC

  # Enroot resolves its /etc/localtime bind through the image symlink and can
  # mount the resolved zoneinfo file itself. dpkg cannot replace that mount or
  # create its backup hard link, so detach it only in this package namespace.
  detach_mounted_localtime_target
  if ! DEBIAN_FRONTEND=noninteractive dpkg --configure -a; then
    dpkg_repair_needed=1
    printf '%s\n' \
      "warning: dpkg repair is partial; APT will resolve remaining dependencies" >&2
  fi

  DEBIAN_FRONTEND=noninteractive "${apt_get[@]}" update
  if [ "$dpkg_repair_needed" -eq 1 ]; then
    DEBIAN_FRONTEND=noninteractive "${apt_get[@]}" --fix-broken install \
      -y --no-install-recommends || die \
      "APT could not repair the interrupted package transaction"
  fi
  if ! DEBIAN_FRONTEND=noninteractive "${apt_get[@]}" install -y --no-install-recommends \
      ca-certificates curl gnupg software-properties-common; then
    command -v curl >/dev/null 2>&1 &&
      command -v gpg >/dev/null 2>&1 &&
      command -v add-apt-repository >/dev/null 2>&1 || \
      die "APT failed before the Apptainer repository tools became usable"
    printf '%s\n' \
      "warning: APT reported a bootstrap-package error, but its required commands are usable" >&2
  fi

  if ! apt-cache show apptainer >/dev/null 2>&1; then
    command -v add-apt-repository >/dev/null 2>&1 || \
      die "add-apt-repository was not installed"
    add-apt-repository -y ppa:apptainer/ppa
    DEBIAN_FRONTEND=noninteractive "${apt_get[@]}" update
  fi

  if ! DEBIAN_FRONTEND=noninteractive "${apt_get[@]}" install -y --no-install-recommends \
      "${PACKAGES[@]}"; then
    packages_ready || die "APT failed and one or more native runtime packages are incomplete"
    printf '%s\n' \
      "warning: APT reported a post-install error, but every runtime passed verification" >&2
  fi
}

if [ "$CHECK_ONLY" -eq 1 ]; then
  packages_ready || die \
    "runtime packages are incomplete; rerun without --check inside the root-mapped outer container"
elif [ "$FORCE" -eq 1 ] || ! packages_ready; then
  install_packages
else
  printf '%s\n' "node runtime packages are already installed; skipping APT"
fi

packages_ready || die "one or more native runtime packages failed final verification"

APPTAINER_PATH=/usr/bin/apptainer
BWRAP_PATH=/usr/bin/bwrap
PODMAN_PATH=/usr/bin/podman
CRUN_PATH=/usr/bin/crun
FUSE_OVERLAYFS_PATH=/usr/bin/fuse-overlayfs
FUSERMOUNT_PATH=/usr/bin/fusermount3
UNSQUASHFS_PATH=/usr/bin/unsquashfs
FUSE2FS_PATH="$(system_tool fuse2fs)"
MKFS_EXT4_PATH="$(system_tool mkfs.ext4)"
E2FSCK_PATH="$(system_tool e2fsck)"
RESIZE2FS_PATH="$(system_tool resize2fs)"
DUMPE2FS_PATH="$(system_tool dumpe2fs)"
FINDMNT_PATH="$(system_tool findmnt)"
WORKSPACE_FUSERMOUNT_PATH="$(system_tool fusermount || system_tool fusermount3)"

first_version_line() {
  run_probe "$1" --version 2>&1 | sed -n '1p'
}

run_probe() {
  timeout --signal=TERM --kill-after=2s \
    "${PROBE_TIMEOUT_SECONDS}s" "$@"
}

APPTAINER_VERSION="$(first_version_line "$APPTAINER_PATH")" || \
  die "Apptainer version probe failed"
BWRAP_VERSION="$(first_version_line "$BWRAP_PATH")" || \
  die "Bubblewrap version probe failed"
PODMAN_VERSION="$(first_version_line "$PODMAN_PATH")" || \
  die "Podman version probe failed"
CRUN_VERSION="$(first_version_line "$CRUN_PATH")" || \
  die "crun version probe failed"
FUSE_OVERLAYFS_VERSION="$(
  run_probe "$FUSE_OVERLAYFS_PATH" --version 2>&1 | \
    sed -n '/^fuse-overlayfs:/ {p;q;}' || true
)"
FUSERMOUNT_VERSION="$(first_version_line "$FUSERMOUNT_PATH")" || \
  die "fusermount3 version probe failed"
UNSQUASHFS_VERSION="$(
  run_probe "$UNSQUASHFS_PATH" -version 2>&1 | sed -n '1p' || true
)"
FUSE2FS_VERSION_OUTPUT="$(run_probe "$FUSE2FS_PATH" -V 2>&1)" || \
  die "fuse2fs version probe failed"
FUSE2FS_VERSION="$(printf '%s\n' "$FUSE2FS_VERSION_OUTPUT" | sed -n '1p')"
FUSE2FS_LIBRARY_VERSION="$(printf '%s\n' "$FUSE2FS_VERSION_OUTPUT" | sed -n '/^FUSE library version:/p')"
MKFS_EXT4_VERSION="$(run_probe "$MKFS_EXT4_PATH" -V 2>&1 | sed -n '1p')" || \
  die "mkfs.ext4 version probe failed"
E2FSCK_VERSION="$(run_probe "$E2FSCK_PATH" -V 2>&1 | sed -n '1p')" || \
  die "e2fsck version probe failed"
WORKSPACE_FUSERMOUNT_VERSION="$(first_version_line "$WORKSPACE_FUSERMOUNT_PATH")" || \
  die "workspace unmount utility version probe failed"
FINDMNT_VERSION="$(first_version_line "$FINDMNT_PATH")" || \
  die "findmnt version probe failed"

for version_value in \
    "$APPTAINER_VERSION" "$BWRAP_VERSION" "$PODMAN_VERSION" \
    "$CRUN_VERSION" "$FUSE_OVERLAYFS_VERSION" "$FUSERMOUNT_VERSION" \
    "$UNSQUASHFS_VERSION" "$FUSE2FS_VERSION" "$MKFS_EXT4_VERSION" \
    "$E2FSCK_VERSION" "$WORKSPACE_FUSERMOUNT_VERSION" "$FINDMNT_VERSION"; do
  [ -n "$version_value" ] || die "a runtime version probe returned no output"
done

if [ -e /dev/fuse ] && [ -r /dev/fuse ] && [ -w /dev/fuse ]; then
  DEV_FUSE_USABLE=yes
  APPTAINER_UNSQUASH_ENABLED=0
else
  DEV_FUSE_USABLE=no
  APPTAINER_UNSQUASH_ENABLED=1
fi

BWRAP_USERNS_FRESH_PROC_SMOKE=no
BWRAP_ROOTFUL_FRESH_PROC_SMOKE=no
BWRAP_ASSEMBLED_ROOT_SMOKE=no
BWRAP_RESOLVED_ENVELOPE=unavailable
bwrap_fresh_args=(
  --die-with-parent --new-session
  --unshare-pid --unshare-ipc --unshare-uts
  --clearenv --tmpfs /
)
for bwrap_system_path in "${BWRAP_SYSTEM_PATHS[@]}"; do
  bwrap_fresh_args+=(--ro-bind "$bwrap_system_path" "$bwrap_system_path")
done
for bwrap_alias in /bin /sbin /lib /lib64 /lib32; do
  if [ -L "$bwrap_alias" ]; then
    bwrap_fresh_args+=(--symlink "$(readlink "$bwrap_alias")" "$bwrap_alias")
  elif [ -d "$bwrap_alias" ]; then
    bwrap_fresh_args+=(--ro-bind "$bwrap_alias" "$bwrap_alias")
  fi
done
bwrap_fresh_args+=(
  --proc /proc --dev /dev --tmpfs /tmp
  --dir /home --tmpfs /home --dir /home/cape
  --setenv HOME /home/cape
  --setenv PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
)

if run_probe "$BWRAP_PATH" \
    --unshare-user --uid 0 --gid 0 \
    "${bwrap_fresh_args[@]}" -- /bin/true >/dev/null 2>&1; then
  BWRAP_USERNS_FRESH_PROC_SMOKE=yes
  BWRAP_ASSEMBLED_ROOT_SMOKE=yes
  BWRAP_RESOLVED_ENVELOPE=userns-assembled-root-fresh-proc
elif [ "$(id -u)" -eq 0 ] && run_probe "$BWRAP_PATH" \
    "${bwrap_fresh_args[@]}" --cap-drop ALL -- /bin/true \
    >/dev/null 2>&1; then
  BWRAP_ROOTFUL_FRESH_PROC_SMOKE=yes
  BWRAP_ASSEMBLED_ROOT_SMOKE=yes
  BWRAP_RESOLVED_ENVELOPE=rootful-assembled-root-fresh-proc-cap-drop
fi
if [ "$BWRAP_RESOLVED_ENVELOPE" = unavailable ]; then
  BWRAP_SMOKE=no
else
  BWRAP_SMOKE=yes
fi

if command -v unshare >/dev/null 2>&1 && \
    run_probe unshare -Ur true >/dev/null 2>&1; then
  USER_NAMESPACE_SMOKE=yes
else
  USER_NAMESPACE_SMOKE=no
fi

CGROUP_CONTROL=/sys/fs/cgroup/cgroup.subtree_control
if [ -e "$CGROUP_CONTROL" ]; then
  if [ -w "$CGROUP_CONTROL" ]; then
    CGROUP_SUBTREE_WRITABLE=yes
    PODMAN_STANDARD_CGROUPS_LIKELY_USABLE=yes
  else
    CGROUP_SUBTREE_WRITABLE=no
    PODMAN_STANDARD_CGROUPS_LIKELY_USABLE=no
  fi
else
  CGROUP_SUBTREE_WRITABLE=not-applicable
  PODMAN_STANDARD_CGROUPS_LIKELY_USABLE=unknown
fi
CGROUP_MOUNT_OPTIONS="$(awk '$2 == "/sys/fs/cgroup" {print $4; exit}' /proc/mounts)"
CGROUP_MOUNT_OPTIONS="${CGROUP_MOUNT_OPTIONS:-missing}"

write_environment() {
  local directory
  local temporary
  directory="$(dirname -- "$ENV_FILE")"
  mkdir -p "$directory"
  temporary="$(mktemp "$directory/.cape-node-runtime.env.XXXXXX")"
  {
    printf '# Generated by manual/prerequisites.sh on %s (%s).\n' "$(hostname)" "$NODE_ARCH"
    printf '# Source before manual/register_node.sh; contains no credentials.\n'
    printf 'export CAPE_APPTAINER_BIN=%q\n' "$APPTAINER_PATH"
    printf 'export CAPE_BUBBLEWRAP_BIN=%q\n' "$BWRAP_PATH"
    printf 'export CAPE_BUBBLEWRAP_SYSTEM_PATHS=%q\n' \
      "$BWRAP_SYSTEM_PATHS_CSV"
    printf 'export CAPE_PODMAN_BIN=%q\n' "$PODMAN_PATH"
    printf 'export CAPE_APPTAINER_UNSQUASH=%q\n' "$APPTAINER_UNSQUASH_ENABLED"
    printf 'export CAPE_WORKSPACE_FUSE2FS_BIN=%q\n' "$FUSE2FS_PATH"
    printf 'export CAPE_WORKSPACE_MKFS_BIN=%q\n' "$MKFS_EXT4_PATH"
    printf 'export CAPE_WORKSPACE_E2FSCK_BIN=%q\n' "$E2FSCK_PATH"
    printf 'export CAPE_WORKSPACE_RESIZE2FS_BIN=%q\n' "$RESIZE2FS_PATH"
    printf 'export CAPE_WORKSPACE_DUMPE2FS_BIN=%q\n' "$DUMPE2FS_PATH"
    printf 'export CAPE_WORKSPACE_FUSERMOUNT_BIN=%q\n' "$WORKSPACE_FUSERMOUNT_PATH"
    printf 'export CAPE_WORKSPACE_FINDMNT_BIN=%q\n' "$FINDMNT_PATH"
  } >"$temporary"
  chmod 0644 "$temporary"
  mv -f "$temporary" "$ENV_FILE"
}

write_report() {
  local directory
  local temporary
  directory="$(dirname -- "$REPORT_FILE")"
  mkdir -p "$directory"
  temporary="$(mktemp "$directory/.cape-node-runtime-report.XXXXXX")"
  {
    printf 'packages_installed=yes\n'
    printf 'outer_image=%s\n' "$OUTER_IMAGE"
    printf 'expected_outer_image=%s\n' "$EXPECTED_OUTER_IMAGE"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'node_architecture=%s\n' "$NODE_ARCH"
    printf 'package_architecture=%s\n' "$DPKG_ARCH"
    printf 'ubuntu_version=%s\n' "${VERSION_ID:-unknown}"
    printf 'apptainer_path=%s\n' "$APPTAINER_PATH"
    printf 'apptainer_version=%s\n' "$APPTAINER_VERSION"
    printf 'bubblewrap_path=%s\n' "$BWRAP_PATH"
    printf 'bubblewrap_version=%s\n' "$BWRAP_VERSION"
    printf 'podman_path=%s\n' "$PODMAN_PATH"
    printf 'podman_version=%s\n' "$PODMAN_VERSION"
    printf 'crun_path=%s\n' "$CRUN_PATH"
    printf 'crun_version=%s\n' "$CRUN_VERSION"
    printf 'fuse_overlayfs_path=%s\n' "$FUSE_OVERLAYFS_PATH"
    printf 'fuse_overlayfs_version=%s\n' "$FUSE_OVERLAYFS_VERSION"
    printf 'fusermount_path=%s\n' "$FUSERMOUNT_PATH"
    printf 'fusermount_version=%s\n' "$FUSERMOUNT_VERSION"
    printf 'unsquashfs_path=%s\n' "$UNSQUASHFS_PATH"
    printf 'unsquashfs_version=%s\n' "$UNSQUASHFS_VERSION"
    printf 'workspace_toolchain_installed=yes\n'
    printf 'workspace_mount_capability=not-tested-use-cape-node-probe\n'
    printf 'workspace_fuse2fs_path=%s\n' "$FUSE2FS_PATH"
    printf 'workspace_fuse2fs_version=%s\n' "$FUSE2FS_VERSION"
    printf 'workspace_fuse_library_version=%s\n' "${FUSE2FS_LIBRARY_VERSION:-unknown-probe-required}"
    printf 'workspace_mkfs_ext4_path=%s\n' "$MKFS_EXT4_PATH"
    printf 'workspace_mkfs_ext4_version=%s\n' "$MKFS_EXT4_VERSION"
    printf 'workspace_e2fsck_path=%s\n' "$E2FSCK_PATH"
    printf 'workspace_e2fsck_version=%s\n' "$E2FSCK_VERSION"
    printf 'workspace_resize2fs_path=%s\n' "$RESIZE2FS_PATH"
    printf 'workspace_dumpe2fs_path=%s\n' "$DUMPE2FS_PATH"
    printf 'workspace_maintenance_capability=not-tested-use-cape-node-probe\n'
    printf 'workspace_fusermount_path=%s\n' "$WORKSPACE_FUSERMOUNT_PATH"
    printf 'workspace_fusermount_version=%s\n' "$WORKSPACE_FUSERMOUNT_VERSION"
    printf 'workspace_findmnt_path=%s\n' "$FINDMNT_PATH"
    printf 'workspace_findmnt_version=%s\n' "$FINDMNT_VERSION"
    printf 'bubblewrap_namespace_smoke=%s\n' "$BWRAP_SMOKE"
    printf 'bubblewrap_resolved_envelope=%s\n' "$BWRAP_RESOLVED_ENVELOPE"
    printf 'bubblewrap_userns_fresh_proc_smoke=%s\n' \
      "$BWRAP_USERNS_FRESH_PROC_SMOKE"
    printf 'bubblewrap_rootful_fresh_proc_smoke=%s\n' \
      "$BWRAP_ROOTFUL_FRESH_PROC_SMOKE"
    printf 'bubblewrap_assembled_root_smoke=%s\n' \
      "$BWRAP_ASSEMBLED_ROOT_SMOKE"
    printf 'bubblewrap_system_paths=%s\n' "$BWRAP_SYSTEM_PATHS_CSV"
    printf 'bubblewrap_gpu_capability=not-tested-use-cape-node-probe\n'
    printf 'runtime_probe_timeout_seconds=%s\n' "$PROBE_TIMEOUT_SECONDS"
    printf 'user_namespace_smoke=%s\n' "$USER_NAMESPACE_SMOKE"
    printf 'dev_fuse_usable=%s\n' "$DEV_FUSE_USABLE"
    printf 'apptainer_unsquash_enabled=%s\n' "$APPTAINER_UNSQUASH_ENABLED"
    printf 'cgroup_mount_options=%s\n' "$CGROUP_MOUNT_OPTIONS"
    printf 'cgroup_subtree_control_writable=%s\n' "$CGROUP_SUBTREE_WRITABLE"
    printf 'podman_standard_cgroups_likely_usable=%s\n' \
      "$PODMAN_STANDARD_CGROUPS_LIKELY_USABLE"
  } >"$temporary"
  chmod 0644 "$temporary"
  mv -f "$temporary" "$REPORT_FILE"
}

write_environment
write_report

printf 'verified native CAPE node runtimes for %s (%s)\n' "$NODE_ARCH" "$DPKG_ARCH"
printf 'runtime environment: %s\n' "$ENV_FILE"
printf 'capability report:   %s\n' "$REPORT_FILE"
printf 'before register_node.sh, use: source %q\n' "$ENV_FILE"

if [ "$OUTER_IMAGE" != "$EXPECTED_OUTER_IMAGE" ]; then
  printf 'warning: CAPE_NODE_OUTER_IMAGE does not identify %s\n' \
    "$EXPECTED_OUTER_IMAGE" >&2
fi
if [ "$BWRAP_SMOKE" != yes ]; then
  printf '%s\n' \
    "warning: packages are installed but the current outer environment rejected Bubblewrap namespace launch" >&2
elif [ "$BWRAP_RESOLVED_ENVELOPE" = rootful-assembled-root-fresh-proc-cap-drop ]; then
  printf '%s\n' \
    "warning: Bubblewrap needs CAPE's capability-dropped root-mapped fresh-proc fallback; node capability reports mark this reduced isolation" >&2
fi
if [ "$DEV_FUSE_USABLE" != yes ]; then
  printf '%s\n' \
    "warning: packages are installed but the current outer environment lacks usable /dev/fuse; CAPE_APPTAINER_UNSQUASH=1 was emitted and stateful FUSE overlays may remain unavailable" >&2
  printf '%s\n' \
    "warning: managed workspaces require usable /dev/fuse; Apptainer unsquash is not a workspace fallback" >&2
fi
if [ "$PODMAN_STANDARD_CGROUPS_LIKELY_USABLE" = no ]; then
  printf '%s\n' \
    "warning: packages are installed but the current outer environment does not delegate writable cgroups; standard nested Podman workloads may fail" >&2
fi
