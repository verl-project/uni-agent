#!/usr/bin/env bash
set -uo pipefail

TOOLS_DIR=/opt/triton-agent-tools
if (( EUID != 0 )); then
  exec /usr/bin/sudo -n -H "${TOOLS_DIR}/verify_once.sh" "$@"
fi

# sudo removes linker variables such as LD_LIBRARY_PATH. Rebuild the CANN
# runtime environment before importing torch_npu.
CANN_ENV=/usr/local/Ascend/cann/set_env.sh
if [[ ! -r "${CANN_ENV}" ]]; then
  echo "[verify-once] missing CANN environment: ${CANN_ENV}" >&2
  exit 2
fi
set +u
# shellcheck disable=SC1091
source "${CANN_ENV}"
set -u

VERIFIER_DIR="${TOOLS_DIR}/verifier"
WORKSPACE="$(pwd -P)"
OP_NAME=${1:-${OPERATOR_NAME:-}}
VERIFY_TIMEOUT=${2:-${TRITON_EVAL_TIMEOUT:-900}}
PYTHON=${OPERATOR_PYTHON:-/usr/local/bin/python3}
[[ -x "${PYTHON}" ]] || PYTHON=python3

if [[ ! "${OP_NAME}" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "usage: tools/verify_once.sh <safe_op_name> [timeout_sec]" >&2
  exit 2
fi

reference="src/${OP_NAME}.py"
implementation="src/${OP_NAME}_triton_ascend_impl.py"
case_source=".triton_case_sidecars/${OP_NAME}.json"
verify_dir="output/verify"
verify_result="${verify_dir}/verify_result.json"
summary_result="${verify_dir}/verify_result_summary.json"
perf_result="${verify_dir}/perf_result.json"

for required in "${reference}" "${implementation}" "${case_source}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[verify-once] missing required file: ${required}" >&2
    exit 2
  fi
done

rm -rf "${verify_dir}"
mkdir -p "${verify_dir}"
rm -f src/*.json
cp "${reference}" "${verify_dir}/${OP_NAME}_torch.py"
cp "${implementation}" "${verify_dir}/${OP_NAME}_triton_ascend_impl.py"
cp "${case_source}" "src/${OP_NAME}.json"
cp "${case_source}" "${verify_dir}/${OP_NAME}_torch.json"
chmod 0444 "src/${OP_NAME}.json"

"${PYTHON}" "${VERIFIER_DIR}/validate_triton_impl.py" "${implementation}" --json || exit $?

verify_status=0
"${TOOLS_DIR}/with_npu_lease.py" -- \
  "${PYTHON}" "${VERIFIER_DIR}/verify.py" \
  --op_name "${OP_NAME}" \
  --verify_dir "${verify_dir}" \
  --triton_impl_name triton_ascend_impl \
  --timeout "${VERIFY_TIMEOUT}" \
  --output "${verify_result}" \
  >"${verify_dir}/verify_result.raw.log" 2>&1 || verify_status=$?

"${PYTHON}" "${TOOLS_DIR}/summarize_verify_result.py" \
  "${verify_result}" --exit-code "${verify_status}" --write-json "${summary_result}"

if "${PYTHON}" - "${verify_result}" <<'PY'
import json
import sys

try:
    result = json.load(open(sys.argv[1], encoding="utf-8-sig"))
    total = int(result.get("total_cases") or 0)
    passed = int(result.get("passed_cases") or 0)
    failed = int(result.get("failed_cases") or max(total - passed, 0))
except (OSError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if total > 0 and passed == total and failed == 0 else 1)
PY
then
  benchmark_timeout=${TRITON_LATENCY_BENCHMARK_TIMEOUT:-1800}
  timeout --signal=TERM --kill-after=30s "${benchmark_timeout}s" \
    "${TOOLS_DIR}/with_npu_lease.py" -- \
    "${PYTHON}" "${VERIFIER_DIR}/benchmark.py" \
    --op_name "${OP_NAME}" \
    --verify_dir "${verify_dir}" \
    --triton_impl_name triton_ascend_impl \
    --warmup "${TRITON_LATENCY_BENCHMARK_WARMUP:-2}" \
    --repeats "${TRITON_LATENCY_BENCHMARK_REPEATS:-10}" \
    --output "${perf_result}" \
    >"${verify_dir}/perf_result.raw.log" 2>&1 || rm -f "${perf_result}"
fi

"${PYTHON}" "${TOOLS_DIR}/snapshot_verify_best.py" \
  --op-name "${OP_NAME}" \
  --verify-result "${verify_result}" \
  --summary "${summary_result}" \
  --perf-result "${perf_result}"

exit "${verify_status}"
