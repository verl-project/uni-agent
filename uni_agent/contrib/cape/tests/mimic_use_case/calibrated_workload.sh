#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Calibration workload: sleep 1 second of CPU work, then run a GPU kernel that
# occupies the whole device. The default GPU mode is `vecadd`, a work-bound
# element-wise add whose wall time grows when co-tenants take bandwidth, so a
# shared pool is measurable. `--gpu-mode sleep` restores the older wall-clock
# kernel, whose duration is constant regardless of contention.
# The launch geometry is derived from the CUDA occupancy calculator: the grid is
# exactly one full wave, blocksPerSM * multiProcessorCount, so every SM is
# resident-full and the kernel still finishes in one sleep duration. Theoretical
# actual processing time: 2 seconds. Run it directly (outside CAPE) to measure
# the real per-node processing constant; submit the same script through CAPE so
# both paths produce identical marker output.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  calibrated_workload.sh [OPTIONS]

Performs, in order:
  1. CPU work: sleep CPU_SECONDS (default 1)
  2. GPU work: a device-filling kernel. The grid is one full wave sized from
     the occupancy calculator (blocksPerSM * multiProcessorCount), so every SM
     is resident-full. Default mode `vecadd` runs a fixed amount of work, so
     its duration reflects the bandwidth actually available. Mode `sleep` runs
     for GPU_SECONDS of wall clock instead.

Prints monotonic evidence markers on stdout, one per line:
  CAPE_CALIBRATED work_start <unix-epoch-seconds.ns>
  CAPE_CALIBRATED cpu_done   <unix-epoch-seconds.ns>
  CAPE_CALIBRATED gpu_launch device=<n> sms=<n> block=<threads>
                             blocks_per_sm=<n> grid=<ctas> occupancy=<fraction>
  CAPE_CALIBRATED gpu_done   <unix-epoch-seconds.ns>
  CAPE_CALIBRATED work_end   <unix-epoch-seconds.ns>
  CAPE_CALIBRATED elapsed_ms cpu=<ms> gpu=<ms> total=<ms>
  CAPE_CALIBRATED result     ok cpu=<s> gpu=<s> total_theoretical=<s>

The elapsed_ms line is the measured wall-clock decomposition: cpu is
work_start->cpu_done, gpu is cpu_done->gpu_done (device sleep plus CUDA
context/synchronize overhead), total is work_start->work_end. Use the total
from a few direct runs (median) as the --processing-constant for
concurrency.sh.

Options:
  --cpu-seconds N     CPU sleep duration (default: 1)
  --gpu-seconds N     On-device sleep duration for --gpu-mode sleep
                      (default: 1). Ignored by --gpu-mode vecadd, whose
                      duration is set by --gpu-iterations and by how much
                      bandwidth co-tenants leave available.
  --cache-dir DIR     Where the compiled GPU helper is cached
                      (default: CAPE_CALIBRATED_CACHE or
                       ${TMPDIR:-/tmp}/cape-calibrated-$(id -u))
  --gpu-lock-dir DIR  Cooperatively lock DIR/<single leased GPU UUID>.lock
                      before CUDA initialization through synchronized execution
                      and teardown; requires the same shared node-local directory.
  --gpu-mode MODE     vecadd (default) or sleep. sleep bounds the kernel by
                      wall clock, so its duration is constant and unaffected by
                      co-tenants -- correct for measuring CAPE overhead against
                      a known constant, useless for measuring contention.
                      vecadd bounds it by work done (element-wise c += a + b
                      over a grid-stride range), so a co-tenant that halves the
                      achievable bandwidth roughly doubles the wall time.
  --gpu-bytes N       vecadd: bytes per input array (default 268435456; three
                      arrays are allocated)
  --gpu-iterations N  vecadd: passes over the arrays (default 5000, which is
                      about one second uncontended on a GB200-class device). Calibrate
                      once per GPU model so an uncontended run lands near
                      --gpu-seconds; the count must stay fixed across a
                      comparison or contention is absorbed rather than measured.
  --gpu-block-size N  Threads per CTA (default: chosen by
                      cudaOccupancyMaxPotentialBlockSize)
  --gpu-blocks-per-sm N
                      Resident CTAs per SM (default: chosen by
                      cudaOccupancyMaxActiveBlocksPerMultiprocessor)
  --gpu-grid N        Total CTAs, overriding blocks_per_sm * SM count. Values
                      above one full wave serialize into extra waves and
                      multiply the measured GPU time; use only for diagnostics.
  --gpu-single-cta    Legacy <<<1,1>>> launch, for comparison against the
                      device-filling default
  --skip-gpu          Skip the GPU step (CPU-only calibration)
  -h, --help          Show this help

Environment:
  CAPE_CALIBRATED_CACHE     Default cache directory for the compiled kernel
  CAPE_CALIBRATED_GPU_CMD   Override the GPU step with an arbitrary command
                            (executed via bash -c; test hook)

The compiled GPU helper takes no required arguments: run it directly from the
cache directory and it sleeps one second across one full wave of the device.
NANOSECONDS, BLOCK_SIZE, BLOCKS_PER_SM and GRID are optional positional
overrides, and 0 means "derive it". This script always passes them explicitly.

The GPU helper is a few lines of CUDA C compiled once per cache directory with
nvcc, preferring -arch=native (SASS for the local GPU, independent of the
driver's PTX JIT) and falling back to the default arch. nvcc runs with
TMPDIR set to the cache directory so the build works inside containers whose
/tmp is a fresh tmpfs masking the caller's TMPDIR path. The kernel bounds its
sleep with %globaltimer and uses __nanosleep on sm_70+, so occupancy is one
CTA / one thread. Because the binary is environment-specific, use one cache
directory per execution environment (e.g. per isolation policy); a binary
built inside one container image may not run in another environment's driver
view. Exit code is nonzero if any step fails; the GPU step failing means the
environment is not GPU-calibratable (missing device, driver, or nvcc and no
override), never a silent CPU-only pass.
EOF
}

die() {
  printf 'calibrated_workload.sh: %s\n' "$*" >&2
  exit 1
}

marker() {
  local name="$1" stamp
  stamp="$(date +%s.%N)"
  printf 'CAPE_CALIBRATED %s %s\n' "$name" "$stamp"
  printf -v "MARKER_$name" '%s' "$stamp"
}

cpu_seconds=1
gpu_seconds=1
cache_dir="${CAPE_CALIBRATED_CACHE:-}"
skip_gpu=0
gpu_lock_dir=""
gpu_mode=vecadd
gpu_bytes=$((256 * 1024 * 1024))
gpu_iterations=5000
gpu_block_size=0
gpu_blocks_per_sm=0
gpu_grid=0
# Bump when the kernel source changes: the compiled helper is cached by this
# name, so an unversioned rename would let an existing cache serve a stale
# binary with the old launch geometry.
GPU_HELPER_VERSION=v7-lock-timeline

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cpu-seconds) [ "$#" -ge 2 ] || die "--cpu-seconds requires a value"; cpu_seconds="$2"; shift 2 ;;
    --cpu-seconds=*) cpu_seconds="${1#*=}"; shift ;;
    --gpu-seconds) [ "$#" -ge 2 ] || die "--gpu-seconds requires a value"; gpu_seconds="$2"; shift 2 ;;
    --gpu-seconds=*) gpu_seconds="${1#*=}"; shift ;;
    --cache-dir) [ "$#" -ge 2 ] || die "--cache-dir requires a value"; cache_dir="$2"; shift 2 ;;
    --cache-dir=*) cache_dir="${1#*=}"; shift ;;
    --gpu-lock-dir) [ "$#" -ge 2 ] || die "--gpu-lock-dir requires a value"; gpu_lock_dir="$2"; shift 2 ;;
    --gpu-lock-dir=*) gpu_lock_dir="${1#*=}"; shift ;;
    --gpu-block-size) [ "$#" -ge 2 ] || die "--gpu-block-size requires a value"; gpu_block_size="$2"; shift 2 ;;
    --gpu-block-size=*) gpu_block_size="${1#*=}"; shift ;;
    --gpu-blocks-per-sm) [ "$#" -ge 2 ] || die "--gpu-blocks-per-sm requires a value"; gpu_blocks_per_sm="$2"; shift 2 ;;
    --gpu-blocks-per-sm=*) gpu_blocks_per_sm="${1#*=}"; shift ;;
    --gpu-grid) [ "$#" -ge 2 ] || die "--gpu-grid requires a value"; gpu_grid="$2"; shift 2 ;;
    --gpu-grid=*) gpu_grid="${1#*=}"; shift ;;
    --gpu-mode) [ "$#" -ge 2 ] || die "--gpu-mode requires a value"; gpu_mode="$2"; shift 2 ;;
    --gpu-mode=*) gpu_mode="${1#*=}"; shift ;;
    --gpu-bytes) [ "$#" -ge 2 ] || die "--gpu-bytes requires a value"; gpu_bytes="$2"; shift 2 ;;
    --gpu-bytes=*) gpu_bytes="${1#*=}"; shift ;;
    --gpu-iterations) [ "$#" -ge 2 ] || die "--gpu-iterations requires a value"; gpu_iterations="$2"; shift 2 ;;
    --gpu-iterations=*) gpu_iterations="${1#*=}"; shift ;;
    --gpu-single-cta) gpu_block_size=1; gpu_blocks_per_sm=1; gpu_grid=1; shift ;;
    --skip-gpu) skip_gpu=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

for value in "$cpu_seconds" "$gpu_seconds"; do
  [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
    die "durations must be non-negative numbers: $value"
done
case "$gpu_mode" in
  sleep|vecadd) ;;
  *) die "--gpu-mode must be sleep or vecadd: $gpu_mode" ;;
esac
for value in "$gpu_bytes" "$gpu_iterations"; do
  [[ "$value" =~ ^[0-9]+$ ]] && [ "$value" -gt 0 ] || \
    die "--gpu-bytes and --gpu-iterations must be positive integers: $value"
done
for value in "$gpu_block_size" "$gpu_blocks_per_sm" "$gpu_grid"; do
  [[ "$value" =~ ^[0-9]+$ ]] || \
    die "GPU geometry overrides must be non-negative integers: $value"
done

[ -n "$cache_dir" ] || cache_dir="${TMPDIR:-/tmp}/cape-calibrated-$(id -u)"

write_kernel_source() {
  cat > "$1" <<'EOF'
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

struct TimingLock {
    int fd = -1;
    ~TimingLock() {
        // Keep CUDA teardown from interfering with the next participating holder.
        cudaDeviceSynchronize();
        cudaDeviceReset();
        if (fd >= 0) close(fd);
    }
};

__device__ unsigned long long global_timer_ns() {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// Every thread sleeps for the same wall duration. Sleeping (rather than
// spinning on an ALU) keeps a resident-full grid cheap in issue bandwidth
// while still holding every warp slot on every SM.
__global__ void device_sleep(unsigned long long duration_ns) {
    const unsigned long long start = global_timer_ns();
    while (global_timer_ns() - start < duration_ns) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(1000000U);
#endif
    }
}

// Work-bound alternative. Completion requires a fixed number of memory
// operations rather than a fixed wall duration, so a co-tenant that halves the
// achievable bandwidth doubles the wall time -- the property device_sleep
// deliberately does not have.
//
// float4 is 128 bits of fp32, so each access compiles to a single LDG.128 /
// STG.128. That is the widest access every CUDA architecture supports, and it
// is what lets one thread saturate its share of memory bandwidth: scalar
// accesses issue four times as many instructions for the same bytes and become
// issue-limited before they become bandwidth-limited. cudaMalloc returns
// 256-byte aligned pointers, so the 16-byte alignment float4 requires holds.
//
// Accumulating into c makes every iteration depend on the previous one, so the
// compiler cannot hoist the loop and leave an empty kernel behind.
__global__ void vector_add(const float4 *__restrict__ a,
                           const float4 *__restrict__ b,
                           float4 *__restrict__ c,
                           unsigned long long n4,
                           unsigned long long iterations) {
    const unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    const unsigned long long base =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    for (unsigned long long it = 0; it < iterations; ++it) {
        for (unsigned long long i = base; i < n4; i += stride) {
            const float4 av = a[i];   // LDG.128
            const float4 bv = b[i];   // LDG.128
            float4 cv = c[i];         // LDG.128
            cv.x += av.x + bv.x;
            cv.y += av.y + bv.y;
            cv.z += av.z + bv.z;
            cv.w += av.w + bv.w;
            c[i] = cv;                // STG.128
        }
    }
}

__global__ void fill_const(float *__restrict__ p, unsigned long long n, float value) {
    const unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    for (unsigned long long i =
             (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         i < n; i += stride) {
        p[i] = value;
    }
}

static int fail(const char *what, cudaError_t status) {
    fprintf(stderr, "gpu_sleep: %s: %s\n", what, cudaGetErrorString(status));
    return 1;
}

// Every argument is optional so the cached helper is directly runnable with no
// arguments: one second of sleep across one full wave of the device.
#define GPU_SLEEP_DEFAULT_NS 1000000000ULL

int main(int argc, char **argv) {
    if (argc > 5 || (argc > 1 && strcmp(argv[1], "--help") == 0)) {
        fprintf(stderr,
                "usage: gpu_sleep [NANOSECONDS] [BLOCK_SIZE] [BLOCKS_PER_SM] [GRID]\n"
                "  no arguments: sleep 1 second on one full wave of the device\n"
                "  NANOSECONDS:  sleep duration (default 1000000000); 0 launches\n"
                "                the grid without sleeping\n"
                "  BLOCK_SIZE / BLOCKS_PER_SM / GRID: 0 or omitted means 'derive\n"
                "                it from the occupancy calculator'\n");
        return argc > 5 ? 2 : 0;
    }
    const unsigned long long duration_ns =
        (argc > 1) ? strtoull(argv[1], NULL, 10) : GPU_SLEEP_DEFAULT_NS;
    // Mode and work sizing come from the environment so the positional
    // argument contract stays what it was.
    const char *mode_env = getenv("GPU_SLEEP_MODE");
    const int work_mode = (mode_env != NULL && strcmp(mode_env, "vecadd") == 0);
    const char *bytes_env = getenv("GPU_SLEEP_BYTES");
    const char *iters_env = getenv("GPU_SLEEP_ITERATIONS");
    unsigned long long bytes_per_array =
        (bytes_env != NULL) ? strtoull(bytes_env, NULL, 10) : (256ULL << 20);
    unsigned long long work_iterations =
        (iters_env != NULL) ? strtoull(iters_env, NULL, 10) : 5000ULL;
    int want_block = (argc > 2) ? atoi(argv[2]) : 0;
    int want_per_sm = (argc > 3) ? atoi(argv[3]) : 0;
    int want_grid = (argc > 4) ? atoi(argv[4]) : 0;

    TimingLock timing_lock;
    const char *lock_path = getenv("CAPE_CALIBRATED_LOCK_PATH");
    auto lock_begin = std::chrono::steady_clock::now();
    double lock_wait_start = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (lock_path && *lock_path) {
        timing_lock.fd = open(lock_path, O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0660);
        if (timing_lock.fd < 0 || flock(timing_lock.fd, LOCK_EX) != 0) {
            perror("gpu timing lock"); return 2;
        }
    }
    double lock_wait_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - lock_begin).count();
    if (timing_lock.fd >= 0) {
        double lock_acquired = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        printf("CAPE_CALIBRATED lock_wait_start %.9f\n", lock_wait_start);
        printf("CAPE_CALIBRATED lock_acquired %.9f\n", lock_acquired);
    }
    printf("gpu_lock enabled=%d wait_ms=%.3f\n", timing_lock.fd >= 0, lock_wait_ms);
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return fail("cudaGetDevice", status);
    cudaDeviceProp prop;
    status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) return fail("cudaGetDeviceProperties", status);

    // Block size: let the occupancy calculator pick the size that maximises
    // resident warps for this kernel's register and shared-memory footprint.
    int block_size = want_block;
    if (block_size <= 0) {
        int min_grid = 0;
        status = work_mode
            ? cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, vector_add, 0, 0)
            : cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, device_sleep, 0, 0);
        if (status != cudaSuccess)
            return fail("cudaOccupancyMaxPotentialBlockSize", status);
    }
    if (block_size <= 0) block_size = 1;
    if (block_size > prop.maxThreadsPerBlock) block_size = prop.maxThreadsPerBlock;

    // Resident CTAs per SM at that block size -- this is the hardware
    // occupancy limit derived from the kernel's own characteristics.
    int blocks_per_sm = want_per_sm;
    if (blocks_per_sm <= 0) {
        status = work_mode
            ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks_per_sm, vector_add, block_size, 0)
            : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks_per_sm, device_sleep, block_size, 0);
        if (status != cudaSuccess)
            return fail("cudaOccupancyMaxActiveBlocksPerMultiprocessor", status);
    }
    if (blocks_per_sm <= 0) blocks_per_sm = 1;

    // One full wave: every SM resident-full, nothing queued behind it.
    int grid = (want_grid > 0) ? want_grid : blocks_per_sm * prop.multiProcessorCount;
    if (grid <= 0) grid = 1;

    const double occupancy =
        prop.maxThreadsPerMultiProcessor > 0
            ? (double)(blocks_per_sm * block_size) / (double)prop.maxThreadsPerMultiProcessor
            : 0.0;

    printf("gpu_sleep launch device=%d sms=%d block=%d blocks_per_sm=%d grid=%d occupancy=%.4f mode=%s\n",
           device, prop.multiProcessorCount, block_size, blocks_per_sm, grid, occupancy,
           work_mode ? "vecadd" : "sleep");
    fflush(stdout);

    if (!work_mode) {
        device_sleep<<<grid, block_size>>>(duration_ns);
        status = cudaGetLastError();
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
        if (status != cudaSuccess) return fail("launch", status);
        printf("gpu_sleep ok %llu\n", duration_ns);
        return 0;
    }

    // Round down to a whole number of float4 so every access is a clean
    // 128-bit transaction with no scalar tail.
    const unsigned long long n4 = bytes_per_array / sizeof(float4);
    const unsigned long long n = n4 * 4ULL;
    if (n4 == 0) {
        fprintf(stderr, "gpu_sleep: GPU_SLEEP_BYTES must be at least %zu\n", sizeof(float4));
        return 2;
    }
    float *a = NULL, *b = NULL, *c = NULL;
    if ((status = cudaMalloc(&a, n * sizeof(float))) != cudaSuccess) return fail("cudaMalloc a", status);
    if ((status = cudaMalloc(&b, n * sizeof(float))) != cudaSuccess) return fail("cudaMalloc b", status);
    if ((status = cudaMalloc(&c, n * sizeof(float))) != cudaSuccess) return fail("cudaMalloc c", status);
    fill_const<<<grid, block_size>>>(a, n, 1.0f);
    fill_const<<<grid, block_size>>>(b, n, 2.0f);
    fill_const<<<grid, block_size>>>(c, n, 0.0f);
    if ((status = cudaDeviceSynchronize()) != cudaSuccess) return fail("fill", status);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin); cudaEventCreate(&end);
    cudaEventRecord(begin);
    vector_add<<<grid, block_size>>>((const float4 *)a, (const float4 *)b,
                                     (float4 *)c, n4, work_iterations);
    cudaEventRecord(end);
    status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status != cudaSuccess) return fail("launch", status);
    float device_ms = 0.0f;
    cudaEventElapsedTime(&device_ms, begin, end);

    // Read one element back: proves the work actually happened rather than
    // being optimised away or silently skipped.
    float sample = 0.0f;
    if ((status = cudaMemcpy(&sample, c, sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess)
        return fail("cudaMemcpy", status);
    const double expected = 3.0 * (double)work_iterations;
    if (sample != (float)expected) {
        fprintf(stderr, "gpu_sleep: vecadd result %f != expected %f\n", sample, expected);
        return 1;
    }
    // 4 memory operations per element per iteration: read a, read b, read+write c.
    const double gb = (double)n * (double)work_iterations * 4.0 * sizeof(float) / 1e9;
    printf("gpu_sleep ok vecadd bytes_per_array=%llu elements=%llu vec4=%llu iterations=%llu device_ms=%.1f gbps=%.1f\n",
           bytes_per_array, n, n4, work_iterations, device_ms,
           device_ms > 0.0f ? gb / (device_ms / 1000.0) : 0.0);
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
EOF
}

ensure_gpu_helper() {
  GPU_HELPER="$cache_dir/gpu_sleep-$GPU_HELPER_VERSION"
  [ -x "$GPU_HELPER" ] && return 0
  command -v nvcc >/dev/null 2>&1 || \
    die "GPU step needs nvcc to build the sleep kernel (or set CAPE_CALIBRATED_GPU_CMD); cache: $cache_dir"
  mkdir -p "$cache_dir"
  local lock="$cache_dir/.build-lock"
  local waited=0
  until mkdir "$lock" 2>/dev/null; do
    [ -x "$GPU_HELPER" ] && return 0
    waited=$((waited + 1))
    [ "$waited" -le 120 ] || die "timed out waiting for concurrent kernel build: $lock"
    sleep 1
  done
  if [ -x "$GPU_HELPER" ]; then
    rmdir "$lock" 2>/dev/null || true
    return 0
  fi
  write_kernel_source "$cache_dir/gpu_sleep-$GPU_HELPER_VERSION.cu"
  # TMPDIR: the caller's value may point at a path that does not exist in this
  # container view (e.g. a CAPE session tmp masked by Bubblewrap's fresh
  # tmpfs); the cache directory is writable wherever this cold path runs.
  # -arch=native: emit SASS for the local GPU so execution never depends on
  # the driver's PTX JIT (a driver older than the toolkit rejects its PTX).
  if ! TMPDIR="$cache_dir" nvcc -O2 -arch=native \
      -o "$GPU_HELPER.tmp.$$" "$cache_dir/gpu_sleep-$GPU_HELPER_VERSION.cu" \
      2> "$cache_dir/nvcc-build.log"; then
    if ! TMPDIR="$cache_dir" nvcc -O2 \
        -o "$GPU_HELPER.tmp.$$" "$cache_dir/gpu_sleep-$GPU_HELPER_VERSION.cu" \
        2>> "$cache_dir/nvcc-build.log"; then
      cat "$cache_dir/nvcc-build.log" >&2
      rmdir "$lock" 2>/dev/null || true
      die "nvcc failed to build the GPU sleep kernel"
    fi
    printf 'calibrated_workload.sh: note: -arch=native unavailable; built with the default arch (PTX JIT)\n' >&2
  fi
  mv "$GPU_HELPER.tmp.$$" "$GPU_HELPER"
  rmdir "$lock" 2>/dev/null || true
}

run_gpu_step() {
  local nanos
  local lock_path=""
  if [ -n "$gpu_lock_dir" ]; then
    [[ ${CUDA_VISIBLE_DEVICES:-} =~ ^GPU-[[:xdigit:]]{8}(-[[:xdigit:]]{4}){3}-[[:xdigit:]]{12}$ ]] || \
      die "GPU timing lock requires exactly one full leased GPU UUID; no empty/numeric fallback"
    [ -d "$gpu_lock_dir" ] && [ -w "$gpu_lock_dir" ] || die "GPU lock directory is not writable: $gpu_lock_dir"
    lock_path="$gpu_lock_dir/$CUDA_VISIBLE_DEVICES.lock"
    [ -z "${CAPE_CALIBRATED_GPU_CMD:-}" ] || die "GPU command override cannot be combined with --gpu-lock-dir"
  fi
  nanos="$(awk -v s="$gpu_seconds" 'BEGIN {printf "%.0f", s * 1e9}')"
  if [ -n "${CAPE_CALIBRATED_GPU_CMD:-}" ]; then
    GPU_SLEEP_SECONDS="$gpu_seconds" GPU_SLEEP_NANOS="$nanos" \
      bash -c "$CAPE_CALIBRATED_GPU_CMD" || die "override GPU command failed"
    return 0
  fi
  ensure_gpu_helper
  local output
  output="$(CAPE_CALIBRATED_LOCK_PATH="$lock_path" GPU_SLEEP_MODE="$gpu_mode" GPU_SLEEP_BYTES="$gpu_bytes" \
    GPU_SLEEP_ITERATIONS="$gpu_iterations" \
    "$GPU_HELPER" "$nanos" "$gpu_block_size" "$gpu_blocks_per_sm" "$gpu_grid")" || \
    die "GPU kernel failed"
  # Promote the helper's geometry line to a CAPE marker so the achieved
  # occupancy is recorded alongside the timing evidence.
  local launch
  launch="$(printf '%s\n' "$output" | sed -n 's/^gpu_sleep launch //p')"
  [ -z "$launch" ] || printf 'CAPE_CALIBRATED gpu_launch %s\n' "$launch"
  printf '%s\n' "$output" | grep -v '^gpu_sleep launch ' || true
  if [ -n "${CAPE_CONCURRENCY_RUN_ID:-}" ] && [ -n "${CAPE_CONCURRENCY_INDEX:-}" ]; then
    [[ $CAPE_CONCURRENCY_RUN_ID =~ ^[A-Za-z0-9_.-]+$ && $CAPE_CONCURRENCY_INDEX =~ ^[0-9]+$ ]] || die "invalid measurement identity"
    local timing_dir="$cache_dir/timing/$CAPE_CONCURRENCY_RUN_ID"
    mkdir -p "$timing_dir"
    printf '%s\n' "$output" > "$timing_dir/$CAPE_CONCURRENCY_INDEX.txt"
  fi
}

marker work_start
sleep "$cpu_seconds"
marker cpu_done
if [ "$skip_gpu" = 1 ]; then
  printf 'CAPE_CALIBRATED gpu_skipped requested\n'
else
  run_gpu_step
fi
marker gpu_done
marker work_end
awk -v ws="$MARKER_work_start" -v cd="$MARKER_cpu_done" \
    -v gd="$MARKER_gpu_done" -v we="$MARKER_work_end" 'BEGIN {
  printf "CAPE_CALIBRATED elapsed_ms cpu=%.1f gpu=%.1f total=%.1f\n", \
    (cd - ws) * 1000, (gd - cd) * 1000, (we - ws) * 1000
}'
if [ "$skip_gpu" != 1 ] && [ "$gpu_mode" = vecadd ]; then
  # vecadd is work-bound: there is no theoretical duration to state, because the
  # wall time is exactly what the available bandwidth produced. Reporting one
  # would invite readers to treat contention as measurement error.
  printf 'CAPE_CALIBRATED result ok cpu=%s gpu_mode=vecadd gpu_iterations=%s gpu_bytes=%s total_theoretical=none\n' \
    "$cpu_seconds" "$gpu_iterations" "$gpu_bytes"
else
  printf 'CAPE_CALIBRATED result ok cpu=%s gpu=%s total_theoretical=%s\n' \
    "$cpu_seconds" "$gpu_seconds" \
    "$(awk -v a="$cpu_seconds" -v b="$gpu_seconds" 'BEGIN {printf "%.3f", a + b}')"
fi
