#!/usr/bin/env bash
# Single-task mini-swe-agent inference through the Uni-Agent/verl rollout path.
# The sandbox credential file path is sent to workers; credential values are
# loaded by the OpenYuanRong provider inside each worker.
set -euo pipefail
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

: "${DATA_PATH:?DATA_PATH must point to the selected one-row SWE-bench parquet}"
: "${MODEL_PATH:?MODEL_PATH must point to the local Qwen3.5-9B weights}"
: "${OPENYUANRONG_CREDENTIAL_FILE:?Set only the protected credential-file path}"

if [[ ! -f "${OPENYUANRONG_CREDENTIAL_FILE}" || -L "${OPENYUANRONG_CREDENTIAL_FILE}" || ! -r "${OPENYUANRONG_CREDENTIAL_FILE}" ]]; then
    echo "OpenYuanRong credential file is unreadable" >&2
    exit 2
fi
credential_mode="$(stat -c '%a' "${OPENYUANRONG_CREDENTIAL_FILE}")"
if [[ "${credential_mode}" != "600" ]]; then
    echo "OpenYuanRong credential file must have mode 600" >&2
    exit 2
fi

ENGINE="${ENGINE:-vllm}"
case "${ENGINE}" in
    vllm|sglang) ;;
    *) echo "ENGINE must be vllm or sglang" >&2; exit 2 ;;
esac
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
case "${ENFORCE_EAGER}" in
    0|1|false|true|False|True) ;;
    *) echo "ENFORCE_EAGER must be 0, 1, false, or true" >&2; exit 2 ;;
esac

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
LIMIT="${LIMIT:-1}"
N="${N:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:--1}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-}"
RAY_OBJECT_STORE_MEMORY_BYTES="${RAY_OBJECT_STORE_MEMORY_BYTES:-}"
RAY_idle_worker_killing_memory_threshold_bytes="${RAY_idle_worker_killing_memory_threshold_bytes:-}"
TASK_CONFIG="${TASK_CONFIG:-examples/mini_swe_agent/task_config_mini_swe_agent.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
RAY_BIN="${RAY_BIN:-$(command -v ray)}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${ENGINE}-$$}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
    exit 2
fi
OUTPUT_ROOT="${REPO_ROOT}/artifacts"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"
LOG_DIR="${OUTPUT_DIR}/logs"
RESULT_PATH="${OUTPUT_DIR}/result.json"
RAY_TMPDIR="${REPO_ROOT}/.tmp/r${BASHPID:-$$}"
XDG_CACHE_HOME="${REPO_ROOT}/.cache"
HF_HOME="${XDG_CACHE_HOME}/huggingface"
TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
FLASHINFER_WORKSPACE_BASE="${REPO_ROOT}"
FLASHINFER_CUBIN_DIR="${XDG_CACHE_HOME}/flashinfer/cubins"
TMPDIR="${RAY_TMPDIR}"
# Keep Python/SGLang's implicit home cache inside the task workspace.  In
# particular, SGLang's JIT loader uses ~/.cache/sglang even when XDG_CACHE_HOME
# is set, so leaving HOME inherited would leak build products into /root.
TASK_HOME="${XDG_CACHE_HOME}/home"

resolved_output_dir="$(realpath -m "${OUTPUT_DIR}")"
resolved_output_root="$(realpath -m "${OUTPUT_ROOT}")"
if [[ "${resolved_output_dir}" != "${resolved_output_root}/${RUN_ID}" ]]; then
    echo "OUTPUT_DIR must resolve to exactly ${OUTPUT_ROOT}/${RUN_ID}" >&2
    exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "OUTPUT_DIR already exists; select a new run_id" >&2
    exit 2
fi
mkdir -p "${LOG_DIR}" "${RAY_TMPDIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${FLASHINFER_CUBIN_DIR}" "${TASK_HOME}"
export HOME="${TASK_HOME}"

JOB_PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl"
if [[ "${ENGINE}" == "sglang" ]]; then
    SGLANG_ENV_DIR="${SGLANG_ENV_DIR:-${REPO_ROOT}/.envs/sglang-v0.5.20}"
    SGLANG_PYTHON="${SGLANG_PYTHON:-${SGLANG_ENV_DIR}/bin/python}"
    DRIVER_SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    if [[ ! -x "${SGLANG_PYTHON}" ]]; then
        echo "SGLang isolated Python is not executable: ${SGLANG_PYTHON}" >&2
        exit 2
    fi
    SGLANG_SITE_PACKAGES="$("${SGLANG_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    if [[ ! -d "${SGLANG_SITE_PACKAGES}" ]]; then
        echo "SGLang site-packages directory is missing: ${SGLANG_SITE_PACKAGES}" >&2
        exit 2
    fi
    SGLANG_CUDA_SOURCE="${SGLANG_SITE_PACKAGES}/nvidia/cu13"
    SGLANG_CUDA_DRIVER_LIB="${DRIVER_SITE_PACKAGES}/nvidia/cu13/lib"
    SGLANG_CUDA_INCLUDE_PATH="${DRIVER_SITE_PACKAGES}/nvidia/cu13/include"
    SGLANG_HOST_COMPILER_DIR="${REPO_ROOT}/.envs/gcc11-toolchain/bin"
    SGLANG_HOST_C_COMPILER="${SGLANG_HOST_COMPILER_DIR}/x86_64-conda-linux-gnu-gcc"
    SGLANG_HOST_CXX_COMPILER="${SGLANG_HOST_COMPILER_DIR}/x86_64-conda-linux-gnu-g++"
    SGLANG_HOST_SHIM_DIR="${REPO_ROOT}/.tmp/sglang-host-toolchain/bin"
    SGLANG_CUDA_HOME="${REPO_ROOT}/.tmp/sglang-cuda13"
    SGLANG_NVCC_HOME="${SGLANG_CUDA_HOME}"
    mkdir -p "${SGLANG_CUDA_HOME}"
    ln -sfnT "${SGLANG_CUDA_SOURCE}/bin" "${SGLANG_CUDA_HOME}/bin"
    ln -sfnT "${SGLANG_CUDA_INCLUDE_PATH}" "${SGLANG_CUDA_HOME}/include"
    for header in "${SGLANG_CUDA_INCLUDE_PATH}"/*.h "${SGLANG_CUDA_INCLUDE_PATH}"/*.hpp; do
        [[ -f "${header}" ]] || continue
        header_name="$(basename "${header}")"
        if [[ ! -e "${SGLANG_CUDA_SOURCE}/include/${header_name}" ]]; then
            ln -s "${header}" "${SGLANG_CUDA_SOURCE}/include/${header_name}"
        fi
    done
    ln -sfnT "${SGLANG_CUDA_DRIVER_LIB}" "${SGLANG_CUDA_HOME}/lib"
    if [[ -L "${SGLANG_CUDA_HOME}/lib64" ]]; then
        unlink "${SGLANG_CUDA_HOME}/lib64"
    fi
    mkdir -p "${SGLANG_CUDA_HOME}/lib64/stubs"
    ln -sfn "${SGLANG_CUDA_DRIVER_LIB}/libcudart.so.13" "${SGLANG_CUDA_HOME}/lib64/libcudart.so.13"
    ln -sfn "${SGLANG_CUDA_DRIVER_LIB}/libcudart.so.13" "${SGLANG_CUDA_HOME}/lib64/libcudart.so"
    # Do not exit awk early: with pipefail, that would SIGPIPE ldconfig and
    # abort the launcher with status 141 when multiple libcuda entries exist.
    SGLANG_LIBCUDA_PATH="$(ldconfig -p 2>/dev/null | awk '$1 == "libcuda.so" && path == "" {path = $NF} END {print path}')"
    if [[ -z "${SGLANG_LIBCUDA_PATH}" || ! -e "${SGLANG_LIBCUDA_PATH}" ]]; then
        echo "system libcuda.so linker library is unavailable" >&2
        exit 2
    fi
    ln -sfn "${SGLANG_LIBCUDA_PATH}" "${SGLANG_CUDA_HOME}/lib64/stubs/libcuda.so"
    ln -sfnT "${SGLANG_CUDA_SOURCE}/nvvm" "${SGLANG_CUDA_HOME}/nvvm"
    if [[ ! -x "${SGLANG_NVCC_HOME}/bin/nvcc" || ! -f "${SGLANG_CUDA_INCLUDE_PATH}/cuda_runtime.h" || ! -x "${SGLANG_HOST_CXX_COMPILER}" ]]; then
        echo "matching CUDA 13 compiler or headers are missing: nvcc=${SGLANG_NVCC_HOME}/bin/nvcc include=${SGLANG_CUDA_INCLUDE_PATH}" >&2
        exit 2
    fi
    mkdir -p "${SGLANG_HOST_SHIM_DIR}"
    ln -sfn "${SGLANG_HOST_C_COMPILER}" "${SGLANG_HOST_SHIM_DIR}/gcc"
    ln -sfn "${SGLANG_HOST_C_COMPILER}" "${SGLANG_HOST_SHIM_DIR}/cc"
    ln -sfn "${SGLANG_HOST_CXX_COMPILER}" "${SGLANG_HOST_SHIM_DIR}/g++"
    ln -sfn "${SGLANG_HOST_CXX_COMPILER}" "${SGLANG_HOST_SHIM_DIR}/c++"
    SGLANG_ACTOR_PATH="${SGLANG_NVCC_HOME}/bin:${SGLANG_HOST_SHIM_DIR}:${SGLANG_HOST_COMPILER_DIR}:${PATH}"
    SGLANG_ACTOR_CPATH="${SGLANG_CUDA_INCLUDE_PATH}${CPATH:+:${CPATH}}"
    SGLANG_SYSTEM_SITE_PACKAGES="$("${SGLANG_PYTHON}" -c 'import os, site, sysconfig; own = os.path.realpath(sysconfig.get_path("purelib")); print(os.pathsep.join(path for path in site.getsitepackages() if os.path.realpath(path) != own))')"
    if [[ -z "${SGLANG_SYSTEM_SITE_PACKAGES}" ]]; then
        echo "SGLang environment has no inherited system site-packages" >&2
        exit 2
    fi
    SGLANG_ACTOR_LD_LIBRARY_PATH="$(SGLANG_SITE_PACKAGES="${SGLANG_SITE_PACKAGES}" DRIVER_SITE_PACKAGES="${DRIVER_SITE_PACKAGES}" "${SGLANG_PYTHON}" -c 'import os; from pathlib import Path; roots = [Path(os.environ["SGLANG_SITE_PACKAGES"]), Path(os.environ["DRIVER_SITE_PACKAGES"])]; candidates = [root / "nvidia" / "cu13" / "lib" for root in roots] + [root / "torch" / "lib" for root in roots]; print(os.pathsep.join(dict.fromkeys(str(path) for path in candidates if path.is_dir())))')"
    if [[ -z "${SGLANG_ACTOR_LD_LIBRARY_PATH}" ]]; then
        echo "SGLang actor CUDA library path is empty" >&2
        exit 2
    fi
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        SGLANG_ACTOR_LD_LIBRARY_PATH="${SGLANG_ACTOR_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
    fi
    SGLANG_ACTOR_PYTHONPATH="${REPO_ROOT}/examples/mini_swe_agent/sglang_actor_runtime:${JOB_PYTHONPATH}:${SGLANG_SITE_PACKAGES}:${SGLANG_SYSTEM_SITE_PACKAGES}:${DRIVER_SITE_PACKAGES}"
    export SGLANG_ENV_DIR SGLANG_PYTHON SGLANG_SITE_PACKAGES SGLANG_SYSTEM_SITE_PACKAGES
    export SGLANG_CUDA_HOME SGLANG_CUDA_INCLUDE_PATH SGLANG_ACTOR_PATH SGLANG_ACTOR_CPATH
    export SGLANG_HOST_COMPILER_DIR SGLANG_HOST_C_COMPILER SGLANG_HOST_CXX_COMPILER SGLANG_HOST_SHIM_DIR
    export SGLANG_ACTOR_PYTHONPATH SGLANG_ACTOR_LD_LIBRARY_PATH
fi
export PYTHONPATH="${JOB_PYTHONPATH}"
if [[ "${ENGINE}" == "sglang" ]]; then
    env \
        FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE}" \
        FLASHINFER_CUBIN_DIR="${FLASHINFER_CUBIN_DIR}" \
        HF_HOME="${HF_HOME}" \
        CUDA_HOME="${SGLANG_CUDA_HOME}" \
        CUDA_PATH="${SGLANG_CUDA_HOME}" \
        CUDAHOSTCXX="${SGLANG_HOST_CXX_COMPILER}" \
        CC="${SGLANG_HOST_C_COMPILER}" \
        CXX="${SGLANG_HOST_CXX_COMPILER}" \
        CPATH="${SGLANG_ACTOR_CPATH}" \
        CPLUS_INCLUDE_PATH="${SGLANG_ACTOR_CPATH}" \
        LD_LIBRARY_PATH="${SGLANG_ACTOR_LD_LIBRARY_PATH}" \
        PATH="${SGLANG_ACTOR_PATH}" \
        TMPDIR="${TMPDIR}" \
        XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
        PYTHONPATH="${SGLANG_ACTOR_PYTHONPATH}" \
        "${SGLANG_PYTHON}" -c 'import ray, sglang, torch, transformers, sys; from sglang.srt.entrypoints.http_server import ServerArgs; print(f"SGLang actor runtime preflight: python={sys.executable}, ray={ray.__version__} ({ray.__file__}), sglang={sglang.__version__} ({sglang.__file__}), torch={torch.__version__} ({torch.__file__}), transformers={transformers.__version__} ({transformers.__file__}), ServerArgs=ok")'
fi

export REPO_ROOT PYTHON_BIN RAY_BIN RUN_ID ENGINE TENSOR_PARALLEL_SIZE NNODES N_GPUS_PER_NODE
export CUDA_VISIBLE_DEVICES LANGUAGE_MODEL_ONLY ENFORCE_EAGER LIMIT N CONCURRENCY GATEWAY_COUNT
export TEMPERATURE TOP_P TOP_K RESPONSE_LENGTH GPU_MEMORY_UTILIZATION
export TOOL_PARSER TASK_CONFIG DATA_PATH MODEL_PATH OUTPUT_DIR LOG_DIR RESULT_PATH
export OPENYUANRONG_CREDENTIAL_FILE RAY_TMPDIR TMPDIR XDG_CACHE_HOME HF_HOME
export TASK_HOME TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR FLASHINFER_WORKSPACE_BASE FLASHINFER_CUBIN_DIR

# Credential values only enter the provider process from the protected file.
unset OPENYUANRONG_SERVER_ADDRESS OPENYUANRONG_TOKEN AKERNEL_SERVER_ADDRESS AKERNEL_TOKEN
unset OPENYUANRONG_TUNNEL_SSL_VERIFY OPENYUANRONG_TLS_VERIFY TUNNEL_SSL_VERIFY
if [[ -n "${RAY_idle_worker_killing_memory_threshold_bytes}" ]]; then
    export RAY_idle_worker_killing_memory_threshold_bytes
else
    unset RAY_idle_worker_killing_memory_threshold_bytes
fi

"${PYTHON_BIN}" "${REPO_ROOT}/examples/mini_swe_agent/prepare_single_swe_manifest.py"

exec 9>"${REPO_ROOT}/.tmp/ray-head.lock"
if ! flock -n 9; then
    echo "Another Uni-Agent Ray run holds the task lock" >&2
    exit 2
fi
if "${RAY_BIN}" status --address=auto >/dev/null 2>&1; then
    echo "An existing Ray cluster is active; refusing to attach or stop it" >&2
    exit 2
fi

RAY_HEAD_STARTED=0
JOB_SUBMIT_PID=""
GPU_MONITOR_PID=""
RAY_START_ARGS=(start --head --num-gpus "${N_GPUS_PER_NODE}")
if [[ -n "${RAY_NUM_CPUS}" ]]; then
    RAY_START_ARGS+=(--num-cpus "${RAY_NUM_CPUS}")
fi
if [[ -n "${RAY_OBJECT_STORE_MEMORY_BYTES}" ]]; then
    RAY_START_ARGS+=(--object-store-memory "${RAY_OBJECT_STORE_MEMORY_BYTES}")
fi
cleanup() {
    exit_code=$?
    trap - EXIT
    if [[ -n "${JOB_SUBMIT_PID}" ]]; then
        "${RAY_BIN}" job stop --address http://127.0.0.1:8265 --no-wait "${RUN_ID}" >/dev/null 2>&1 || true
        kill -TERM "${JOB_SUBMIT_PID}" >/dev/null 2>&1 || true
        wait "${JOB_SUBMIT_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
        kill -TERM "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
        wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    fi
    if [[ "${RAY_HEAD_STARTED}" == "1" ]]; then
        "${RAY_BIN}" stop --grace-period 30 >/dev/null 2>&1 || true
    fi
    exit "${exit_code}"
}
trap cleanup EXIT

"${RAY_BIN}" "${RAY_START_ARGS[@]}" \
    --include-dashboard true \
    --dashboard-host 127.0.0.1 \
    --dashboard-port 8265 \
    --port 6379 \
    --temp-dir "${RAY_TMPDIR}" \
    --disable-usage-stats
RAY_HEAD_STARTED=1

RUNTIME_ENV_JSON="$("${PYTHON_BIN}" - <<'PY'
import json
import os

root = os.environ["REPO_ROOT"]
env = {
    "PYTHONPATH": os.environ["PYTHONPATH"],
    "RUN_ID": os.environ["RUN_ID"],
    "OPENYUANRONG_CREDENTIAL_FILE": os.environ["OPENYUANRONG_CREDENTIAL_FILE"],
    "SANDBOX_NAME_PREFIX": "uniagentsglang-" + os.environ["RUN_ID"] + "-",
    "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
    "TMPDIR": os.environ["RAY_TMPDIR"],
    "RAY_TMPDIR": os.environ["RAY_TMPDIR"],
    "HOME": os.environ["HOME"],
    "XDG_CACHE_HOME": os.environ["XDG_CACHE_HOME"],
    "HF_HOME": os.environ["HF_HOME"],
    "TRITON_CACHE_DIR": os.environ["TRITON_CACHE_DIR"],
    "TORCH_EXTENSIONS_DIR": os.environ["TORCH_EXTENSIONS_DIR"],
    "FLASHINFER_WORKSPACE_BASE": os.environ["FLASHINFER_WORKSPACE_BASE"],
    "FLASHINFER_CUBIN_DIR": os.environ["FLASHINFER_CUBIN_DIR"],
}
if os.environ["ENGINE"] == "sglang":
    env["UNI_AGENT_SGLANG_PYTHON"] = os.environ["SGLANG_PYTHON"]
    env["UNI_AGENT_SGLANG_PYTHONPATH"] = os.environ["SGLANG_ACTOR_PYTHONPATH"]
    env["UNI_AGENT_SGLANG_LD_LIBRARY_PATH"] = os.environ["SGLANG_ACTOR_LD_LIBRARY_PATH"]
    env["UNI_AGENT_SGLANG_CUDA_HOME"] = os.environ["SGLANG_CUDA_HOME"]
    env["UNI_AGENT_SGLANG_CUDA_INCLUDE_PATH"] = os.environ["SGLANG_CUDA_INCLUDE_PATH"]
    env["UNI_AGENT_SGLANG_CPATH"] = os.environ["SGLANG_ACTOR_CPATH"]
    env["UNI_AGENT_SGLANG_PATH"] = os.environ["SGLANG_ACTOR_PATH"]
    env["UNI_AGENT_SGLANG_CUDAHOSTCXX"] = os.environ["SGLANG_HOST_CXX_COMPILER"]
    env["UNI_AGENT_SGLANG_CC"] = os.environ["SGLANG_HOST_C_COMPILER"]
    env["UNI_AGENT_SGLANG_CXX"] = os.environ["SGLANG_HOST_CXX_COMPILER"]
if os.environ.get("UNI_AGENT_SGLANG_RESPONSE_DIAGNOSTICS") == "1":
    env["UNI_AGENT_SGLANG_RESPONSE_DIAGNOSTICS"] = "1"
print(json.dumps({
    "env_vars": env,
    "excludes": [
        "/.git/",
        "/.pytest_cache/",
        "/.research/",
        "/.envs/",
        "/.tmp/",
        "/.cache/",
        "/artifacts/",
        "/execution-state.json",
    ],
}))
PY
)"

INFER_ARGS=(
    --data-path "${DATA_PATH}"
    --model-path "${MODEL_PATH}"
    --task-config "${TASK_CONFIG}"
    --tool-parser "${TOOL_PARSER}"
    --engine "${ENGINE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --nnodes "${NNODES}"
    --n-gpus-per-node "${N_GPUS_PER_NODE}"
    --limit "${LIMIT}"
    --n "${N}"
    --concurrency "${CONCURRENCY}"
    --gateway-count "${GATEWAY_COUNT}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --response-length "${RESPONSE_LENGTH}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --log-dir "${LOG_DIR}"
    --result-path "${RESULT_PATH}"
)
if [[ "${LANGUAGE_MODEL_ONLY}" == "1" || "${LANGUAGE_MODEL_ONLY}" == "true" ]]; then
    INFER_ARGS+=(--language-model-only)
fi
if [[ "${ENFORCE_EAGER}" == "1" || "${ENFORCE_EAGER}" == "true" ]]; then
    INFER_ARGS+=(--enforce-eager)
fi

"${RAY_BIN}" job submit \
    --address http://127.0.0.1:8265 \
    --submission-id "${RUN_ID}" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    --working-dir "${REPO_ROOT}" \
    -- "${PYTHON_BIN}" "${REPO_ROOT}/examples/inference/parallel_infer_verl.py" "${INFER_ARGS[@]}" &
JOB_SUBMIT_PID=$!

(
    gpu_snapshot_tmp="${OUTPUT_DIR}/.gpu-snapshot.csv.tmp"
    gpu_processes_tmp="${OUTPUT_DIR}/.gpu-processes.csv.tmp"
    best_total_memory=0
    capture_gpu_state() {
        if ! env -u CUDA_VISIBLE_DEVICES nvidia-smi \
            --query-gpu=index,uuid,name,memory.used --format=csv,noheader,nounits \
            >"${gpu_snapshot_tmp}" 2>/dev/null; then
            return
        fi
        if ! env -u CUDA_VISIBLE_DEVICES nvidia-smi \
            --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits \
            >"${gpu_processes_tmp}" 2>/dev/null; then
            return
        fi
        current_total_memory="$(awk -F, '{gsub(/[[:space:]]/, "", $4); if ($4 ~ /^[0-9]+$/) total += $4} END {print total + 0}' "${gpu_snapshot_tmp}")"
        if (( current_total_memory > best_total_memory )); then
            mv -f "${gpu_snapshot_tmp}" "${OUTPUT_DIR}/gpu-snapshot.csv"
            mv -f "${gpu_processes_tmp}" "${OUTPUT_DIR}/gpu-processes.csv"
            best_total_memory="${current_total_memory}"
        fi
    }
    while kill -0 "${JOB_SUBMIT_PID}" >/dev/null 2>&1; do
        capture_gpu_state
        sleep 3
    done
    capture_gpu_state
) &
GPU_MONITOR_PID=$!

if wait "${JOB_SUBMIT_PID}"; then
    JOB_EXIT_CODE=0
else
    JOB_EXIT_CODE=$?
fi
wait "${GPU_MONITOR_PID}" || true
JOB_SUBMIT_PID=""
GPU_MONITOR_PID=""
exit "${JOB_EXIT_CODE}"
