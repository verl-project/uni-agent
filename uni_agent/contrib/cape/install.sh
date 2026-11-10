#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash install.sh [--prefix DIR] [--yes] [--allow-apptainer-ppa] [--check]

Build and install rscape from THIS checkout on Linux x86_64/aarch64, and install
missing Rust, Bubblewrap, Apptainer, workspace tools, DCGM, Prometheus and Telegraf. Every normal invocation
runs cargo build --release --locked, even if rscape is already installed.

  --prefix DIR           Install to DIR/bin/rscape (default: $HOME/.local)
  --yes                  Approve missing packages and Rust bootstrap noninteractively
  --allow-apptainer-ppa  Explicitly allow the official Ubuntu Apptainer PPA if needed
  --check                Inspect prerequisites only; no install/build/network writes
  -h, --help             Show help

Missing system packages use Ubuntu/Debian APT and require root or sudo. Other
Linux distributions can reuse installed tools, but must provision missing packages
themselves. Existing usable tools are not upgraded. Rust must meet Cargo.toml's
rust-version. Rustup bootstrap uses HTTPS and does not edit shell startup files.

CARGO_TARGET_DIR may select a reusable build directory (prefer node-local storage).
Otherwise a private temporary build directory is created and retained for reuse.
An existing regular rscape binary is backed up to rscape.previous before atomic
replacement. No controller/agent is started or restarted. No workspace images,
GPU drivers, FUSE settings, user namespace policy or existing credentials are changed.
Package checks do not prove mount/runtime readiness: node enroll performs live probes.
Run as your normal user; only package operations use sudo. Do not sudo the whole script.
EOF
}

die() { printf '[install] error: %s\n' "$*" >&2; exit 1; }
note() { printf '[install] %s\n' "$*"; }
prefix=${HOME:?HOME must identify the installing user}/.local
yes=0 check=0 allow_ppa=0
while (($#)); do
    case "$1" in
        --prefix) (($# >= 2)) && [[ -n $2 ]] || die '--prefix requires a directory'; prefix=$2; shift 2 ;;
        --yes) yes=1; shift ;;
        --check) check=1; shift ;;
        --allow-apptainer-ppa) allow_ppa=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done
caller_path=${PATH:-/usr/bin:/bin}
export PATH="$prefix/bin:$caller_path:/usr/sbin:/sbin"
[[ $(uname -s) == Linux ]] || die 'Linux is required'
case $(uname -m) in x86_64|aarch64) ;; *) die 'only x86_64 and aarch64 are supported' ;; esac
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
[[ -f $source_dir/rscape/Cargo.toml && -f $source_dir/rscape/Cargo.lock ]] || die 'run the install.sh supplied with a complete CAPE checkout'
[[ $prefix == /* ]] || prefix=$PWD/$prefix
export PATH="$prefix/bin:$caller_path:/usr/sbin:/sbin"
cd "$source_dir/rscape"
minimum=$(sed -n 's/^rust-version *= *"\([^"]*\)".*/\1/p' Cargo.toml)
[[ $minimum =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || die 'invalid rust-version in Cargo.toml'

confirm() {
    local answer
    if ((yes)); then return; fi
    if ! { printf '[install] %s [y/N] ' "$1" >/dev/tty; read -r answer </dev/tty; } 2>/dev/null; then
        die 'confirmation requires a terminal; review --help and use --yes for unattended installation'
    fi
    [[ $answer == y || $answer == Y ]] || die 'installation declined'
}

version_at_least() {
    local actual=$1 required=$2 am an ap rm rn rp
    [[ $actual =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
    IFS=. read -r am an ap <<< "$actual"
    IFS=. read -r rm rn rp <<< "$required"
    ((am > rm || (am == rm && an > rn) || (am == rm && an == rn && ap >= ${rp:-0})))
}

rust_ready() {
    local rv cv
    command -v rustc >/dev/null && command -v cargo >/dev/null || return 1
    rv=$(rustc --version 2>/dev/null) && cv=$(cargo --version 2>/dev/null) || return 1
    rv=${rv#rustc }; rv=${rv%% *}; cv=${cv#cargo }; cv=${cv%% *}
    version_at_least "$rv" "$minimum" && version_at_least "$cv" "$minimum"
}

# Test launchability, not merely file presence. resize2fs prints its version with
# usage and exits 1 when called without an image; never give a real image to probes.
tool_ready() {
    local tool=$1 output rc=0
    command -v "$tool" >/dev/null 2>&1 || return 1
    case $tool in
        resize2fs)
            output=$("$tool" 2>&1) || rc=$?
            [[ $rc -le 1 && $output =~ resize2fs[[:space:]]+[0-9]+\. ]] ;;
        squashfuse)
            output=$("$tool" 2>&1) || rc=$?
            # Upstream prints PACKAGE_STRING in usage and returns -2 without an image.
            [[ ( $rc == 0 || $rc == 1 || $rc == 254 ) && $output =~ [Ss]quashfuse[[:space:]]+[0-9]+\. ]] ;;
        mkfs.ext4|e2fsck|dumpe2fs|fuse2fs) "$tool" -V >/dev/null 2>&1 ;;
        unsquashfs)
            output=$("$tool" -version 2>&1) || rc=$?
            # Older releases print a valid version but exit 1 without an image.
            [[ $rc -le 1 && $output =~ ^unsquashfs[[:blank:]]+version[[:blank:]]+[0-9]+\.[0-9]+([.][0-9]+)?([[:space:]]|$) ]] ;;
        mksquashfs) "$tool" -version >/dev/null 2>&1 ;;
        *) "$tool" --version >/dev/null 2>&1 ;;
    esac
}

tools=(cc make ar pkg-config bwrap apptainer fuse-overlayfs squashfuse mksquashfs unsquashfs
       fuse2fs mkfs.ext4 e2fsck resize2fs dumpe2fs findmnt flock mount umount
       truncate timeout nice cp chmod mv mktemp stat)
packages=(build-essential build-essential binutils pkg-config bubblewrap apptainer fuse-overlayfs
          squashfuse squashfs-tools squashfs-tools fuse2fs e2fsprogs e2fsprogs e2fsprogs e2fsprogs
          util-linux util-linux mount mount coreutils coreutils coreutils coreutils coreutils coreutils coreutils coreutils)
missing=()
declare -A wanted=()
scan() {
    local i library
    missing=(); wanted=()
    for i in "${!tools[@]}"; do
        if tool_ready "${tools[i]}"; then
            note "reuse ${tools[i]}: $(command -v "${tools[i]}")"
        else
            missing+=("${tools[i]}"); wanted[${packages[i]}]=1
        fi
    done
    if tool_ready fusermount3; then note "reuse fusermount3: $(command -v fusermount3)"
    elif tool_ready fusermount; then note "reuse fusermount: $(command -v fusermount)"
    else missing+=(fusermount3/fusermount); wanted[fuse3]=1
    fi
    library=$(cc -print-file-name=libc.a 2>/dev/null) || library=
    if [[ $library != /* || ! -r $library ]]; then
        missing+=(static-libc); wanted[libc6-dev]=1
    fi
}

source "$source_dir/install/metrics.sh"
scan
metrics_scan
need_rust=0
if rust_ready; then note "reuse $(rustc --version); $(cargo --version)"
else need_rust=1; note "Rust/cargo >= $minimum required"
fi
if ((check)); then
    ((${#missing[@]} == 0)) || note "missing/unusable: ${missing[*]}"
    note 'Runtime execution and workspace mount readiness: not tested; use node enroll probes.'
    ((${#metrics_missing[@]} == 0)) || note "missing/unusable metrics tools: ${metrics_missing[*]}"
    ((${#missing[@]} == 0 && ${#metrics_missing[@]} == 0 && need_rust == 0))
    exit $?
fi

root_command=()
apt_ready=0
apt_prepare() {
    ((apt_ready == 0)) || return 0
    [[ -r /etc/os-release ]] || die 'cannot identify distribution; install missing dependencies manually'
    # Distribution-owned metadata, not repository or network input.
    . /etc/os-release
    case ${ID:-} in ubuntu|debian) ;; *) die "automatic packages support Ubuntu/Debian only; missing: ${missing[*]}" ;; esac
    command -v apt-get >/dev/null && command -v apt-cache >/dev/null || die 'APT is unavailable'
    if ((EUID != 0)); then
        command -v sudo >/dev/null || die "root/sudo required for missing packages: ${!wanted[*]}"
        root_command=(sudo)
    fi
    confirm "Install missing system packages: ${!wanted[*]}?"
    "${root_command[@]}" apt-get update
    apt_ready=1
}
apt_install() {
    "${root_command[@]}" apt-get install -y --no-remove --no-install-recommends "$@"
}
candidate_available() {
    local policy
    policy=$(LC_ALL=C apt-cache policy "$1") || return 1
    [[ $policy =~ Candidate:[[:space:]]+([^[:space:]]+) && ${BASH_REMATCH[1]} != '(none)' ]]
}

if ((need_rust)) && ! tool_ready curl; then wanted[curl]=1; wanted[ca-certificates]=1; fi
if ((${#metrics_missing[@]})) && ! tool_ready curl; then wanted[curl]=1; wanted[ca-certificates]=1; fi
if ((${#wanted[@]})); then
    apt_prepare
    if [[ ${wanted[apptainer]:-} == 1 ]] && ! candidate_available apptainer; then
        [[ ${ID:-} == ubuntu ]] || die 'Apptainer is unavailable in configured repositories; install its official Debian package, then rerun'
        if ((allow_ppa == 0)); then
            ((yes == 0)) || die 'Apptainer needs a new package source; explicitly pass --allow-apptainer-ppa'
            confirm 'Add the official ppa:apptainer/ppa repository (a system-wide package trust change)?'
        fi
        if ! command -v add-apt-repository >/dev/null; then apt_install software-properties-common; fi
        "${root_command[@]}" add-apt-repository -y ppa:apptainer/ppa
        "${root_command[@]}" apt-get update
        candidate_available apptainer || die 'Apptainer still has no repository candidate'
    fi
    apt_install "${!wanted[@]}"
    scan
    ((${#missing[@]} == 0)) || die "tools remain unusable (check PATH shadowing): ${missing[*]}"
fi

metrics_install

bootstrap= stage= backup_stage=
cleanup() {
    [[ -z $bootstrap ]] || rm -f -- "$bootstrap"
    [[ -z $stage ]] || rm -f -- "$stage"
    [[ -z $backup_stage ]] || rm -f -- "$backup_stage"
    return 0
}
trap cleanup EXIT
if ((need_rust)); then
    confirm "Install a user Rust stable toolchain (required >= $minimum)?"
    rustup_bin=$(command -v rustup || true)
    if [[ -z $rustup_bin && -x ${CARGO_HOME:-$HOME/.cargo}/bin/rustup ]]; then
        rustup_bin=${CARGO_HOME:-$HOME/.cargo}/bin/rustup
    fi
    if [[ -z $rustup_bin ]]; then
        bootstrap=$(mktemp "${TMPDIR:-/tmp}/cape-rustup.XXXXXX")
        curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o "$bootstrap"
        sh "$bootstrap" -y --profile minimal --no-modify-path --default-toolchain stable
        rustup_bin=${CARGO_HOME:-$HOME/.cargo}/bin/rustup
    else
        "$rustup_bin" toolchain install stable --profile minimal
    fi
    export PATH="$(dirname -- "$rustup_bin"):$PATH" RUSTUP_TOOLCHAIN=stable
    rust_ready || die "Rust bootstrap did not provide rustc/cargo >= $minimum"
fi

if [[ -n ${CARGO_TARGET_DIR:-} ]]; then
    [[ $CARGO_TARGET_DIR == /* ]] || die 'CARGO_TARGET_DIR must be absolute'
else
    CARGO_TARGET_DIR=$(mktemp -d "${TMPDIR:-/tmp}/cape-build.XXXXXX")
    export CARGO_TARGET_DIR
fi
note "build current checkout: $source_dir (build cache: $CARGO_TARGET_DIR)"
host=$(rustc -vV | sed -n 's/^host: //p')
case "$(uname -m):$host" in
    x86_64:x86_64-unknown-linux-gnu|aarch64:aarch64-unknown-linux-gnu) ;;
    *) die "unsupported/mismatched native Rust host: $host" ;;
esac
cargo build --release --locked --target "$host"
binary=$CARGO_TARGET_DIR/$host/release/rscape
[[ -x $binary ]] || die "build did not produce $binary"
"$binary" --help >/dev/null
mkdir -p -- "$prefix/bin"
destination=$prefix/bin/rscape
[[ ! -L $destination && ! -L $destination.previous ]] || die 'refusing to replace an install/backup symlink'
[[ ! -e $destination || -f $destination ]] || die 'installation target is not a regular file'
[[ ! -e $destination.previous || -f $destination.previous ]] || die 'backup target is not a regular file'
stage=$(mktemp "$prefix/bin/.rscape.install.XXXXXX")
cp -- "$binary" "$stage"
chmod 755 "$stage"
"$stage" --help >/dev/null
if [[ -f $destination ]]; then
    backup_stage=$(mktemp "$prefix/bin/.rscape.backup.XXXXXX")
    cp -p -- "$destination" "$backup_stage"
    mv -fT -- "$backup_stage" "$destination.previous"
    backup_stage=
fi
mv -fT -- "$stage" "$destination"
stage=
note "installed current-source rscape: $destination"
note "Build cache retained: $CARGO_TARGET_DIR"
if [[ ! -c /dev/fuse || ! -r /dev/fuse || ! -w /dev/fuse ]]; then
    note 'WARNING: /dev/fuse is unavailable/inaccessible; workspace mounts are not ready in this environment.'
fi
note 'Tools installed; mount/namespace/container readiness is NOT certified. node enroll runs live probes.'
note 'Metrics remain disabled by default; DCGM permissions and receiver reachability are not certified by installation.'
if [[ $(PATH="$caller_path" command -v rscape || true) != "$destination" ]]; then
    printf '\n[install] ACTION REQUIRED: put the installed rscape first in this shell PATH:\n'
    printf 'export PATH=%q:"$PATH"\n' "$prefix/bin"
fi
