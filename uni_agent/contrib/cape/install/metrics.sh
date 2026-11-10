#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sourced by install.sh: installation is not service activation.
metrics_ready() {
    local tool=$1 version output minimum
    command -v "$tool" >/dev/null 2>&1 || return 1
    output=$("$tool" --version 2>&1) || return 1
    version=$(sed -n 's/^[^0-9]*\([0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/p' <<< "$output" | head -1)
    case "$tool" in prometheus) minimum=2.33.0 ;; telegraf) minimum=1.36.3 ;; *) minimum=4.0.0 ;; esac
    version_at_least "$version" "$minimum"
}

metrics_scan() {
    metrics_missing=()
    local tool
    for tool in dcgmi nv-hostengine prometheus telegraf; do
        if metrics_ready "$tool"; then note "reuse $tool: $(command -v "$tool")"
        else metrics_missing+=("$tool"); fi
    done
}

metrics_binary() {
    local name=$1 url=$2 checksum=$3 member=$4 cache extracted stage
    metrics_ready "$name" && return 0
    confirm "Install verified $name binary under $prefix/bin?"
    cache=$(mktemp -d "${TMPDIR:-/tmp}/cape-metrics-download.XXXXXX")
    curl --proto '=https' --tlsv1.2 -fsSL --connect-timeout 15 --max-time 180 --retry 2 "$url" -o "$cache/package.tar.gz"
    printf '%s  %s\n' "$checksum" "$cache/package.tar.gz" | sha256sum --check --status || die "$name checksum mismatch; retained $cache"
    tar -xzf "$cache/package.tar.gz" -C "$cache" "$member"
    extracted=$cache/$member
    [[ -f $extracted && ! -L $extracted ]] || die "invalid $name archive member"
    mkdir -p "$prefix/bin"
    [[ ! -L $prefix/bin/$name && ! -L $prefix/bin/$name.previous ]] || die "refusing $name install symlink"
    stage=$(mktemp "$prefix/bin/.$name.install.XXXXXX")
    cp "$extracted" "$stage"; chmod 755 "$stage"
    "$stage" --version >/dev/null 2>&1 || die "$name is not runnable on this machine"
    if [[ -f $prefix/bin/$name ]]; then cp -p "$prefix/bin/$name" "$prefix/bin/$name.previous"; fi
    mv -fT "$stage" "$prefix/bin/$name"
    note "$name installed; verified download cache retained: $cache"
}

metrics_install_dcgm() {
    metrics_ready dcgmi && metrics_ready nv-hostengine && return 0
    apt_prepare
    local arch repo keyring cache package target libraries binary wrapper relation dependency rest
    case $(uname -m) in x86_64) arch=x86_64 ;; aarch64) arch=sbsa ;; esac
    if ! candidate_available datacenter-gpu-manager-4-core; then
        case "$ID:${VERSION_ID:-}" in
            ubuntu:22.04|ubuntu:24.04|debian:12|debian:13) repo=${ID}${VERSION_ID//./} ;;
            *) die 'DCGM requires a supported NVIDIA repository; provision DCGM >=4 manually on this distribution' ;;
        esac
        confirm "Add NVIDIA CUDA package keyring/repository for DCGM ($repo/$arch), without installing GPU drivers?"
        keyring=$(mktemp "${TMPDIR:-/tmp}/cape-cuda-keyring.XXXXXX.deb")
        curl --proto '=https' --tlsv1.2 -fsSL --max-time 60 "https://developer.download.nvidia.com/compute/cuda/repos/$repo/$arch/cuda-keyring_1.1-1_all.deb" -o "$keyring"
        "${root_command[@]}" dpkg -i "$keyring"
        "${root_command[@]}" apt-get update
    fi
    command -v dpkg-deb >/dev/null || die 'dpkg-deb is required for private DCGM installation'
    mkdir -p "$prefix/libexec" "$prefix/bin"
    target=$(mktemp -d "$prefix/libexec/cape-dcgm.XXXXXX")
    cache=$(mktemp -d "${TMPDIR:-/tmp}/cape-dcgm-packages.XXXXXX")
    # APT verifies downloads against configured signed repository metadata. Extract
    # only DCGM packages; do not execute postinst, enable systemd or install drivers.
    local -a queue=(datacenter-gpu-manager-4-core datacenter-gpu-manager-4-cuda12 datacenter-gpu-manager-4-proprietary datacenter-gpu-manager-4-proprietary-cuda12)
    local -A visited=()
    while ((${#queue[@]})); do
        package=${queue[0]}; queue=("${queue[@]:1}")
        [[ ${visited[$package]:-} == 1 ]] && continue
        visited[$package]=1
        ((${#visited[@]} <= 64)) || die 'unexpectedly large DCGM package dependency closure'
        candidate_available "$package" || die "missing DCGM package candidate: $package"
        while read -r relation dependency rest; do
            case "$relation" in Depends:|PreDepends:|Recommends:) ;; *) continue ;; esac
            case "$dependency" in datacenter-gpu-manager-4*|libdcgm|dcgmi|nv-hostengine) queue+=("$dependency") ;; esac
        done < <(LC_ALL=C apt-cache depends "$package")
    done
    (cd "$cache" && apt-get download "${!visited[@]}")
    for package in "$cache"/*.deb; do dpkg-deb -x "$package" "$target"; done
    libraries=$(find "$target/usr" -type f -name '*.so*' -printf '%h\n' | sort -u | paste -sd: -)
    for binary in dcgmi nv-hostengine; do
        metrics_ready "$binary" && continue
        wrapper=$prefix/bin/$binary
        [[ ! -L $wrapper ]] || die "refusing DCGM wrapper symlink"
        [[ -x $target/usr/bin/$binary ]] || die "DCGM archive did not contain $binary"
        printf '#!/usr/bin/env bash\nexport LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:"$LD_LIBRARY_PATH"}\nexec %q "$@"\n' "$libraries" "$target/usr/bin/$binary" > "$wrapper"
        chmod 755 "$wrapper"
    done
    note "DCGM installed privately at $target; package cache retained: $cache; no service was started"
}

metrics_install() {
    local arch prom_sha telegraf_sha
    case $(uname -m) in
        x86_64) arch=amd64; prom_sha=fc9e5da126817438cf2820d9a7206c75e3122802ba8d20add3a6219a59ca6913; telegraf_sha=9b723764456638f5c6f2a9e0c9381a6867feb0ce70049a10a0215d5b60e33b59 ;;
        aarch64) arch=arm64; prom_sha=6af448fa4f3640eb9bd51ce71575689a1ef79843b1cf08ef1fcc7ff9a1928bfd; telegraf_sha=0b5bce3b27fe5e4f0eb665b9f9946815043a28c3e628670597da314ef694d87d ;;
    esac
    metrics_binary prometheus "https://github.com/prometheus/prometheus/releases/download/v3.7.3/prometheus-3.7.3.linux-$arch.tar.gz" "$prom_sha" "prometheus-3.7.3.linux-$arch/prometheus"
    metrics_binary telegraf "https://dl.influxdata.com/telegraf/releases/telegraf-1.36.3_linux_$arch.tar.gz" "$telegraf_sha" './telegraf-1.36.3/usr/bin/telegraf'
    metrics_install_dcgm
    metrics_scan
    ((${#metrics_missing[@]} == 0)) || die "metrics tools remain unusable: ${metrics_missing[*]}"
}
