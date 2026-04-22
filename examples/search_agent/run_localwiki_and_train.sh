#!/usr/bin/env bash
# Wrapper that starts the localwiki retrieval service on the Ray head and
# then submits the fully-async training job. The localwiki process must
# stay alive for the whole training run because `WikiRetrievalManager` is
# a non-detached Ray actor owned by this driver.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LOCALWIKI_SCRIPT="${REPO_ROOT}/uni_agent/tools/search/localwiki/run_localwiki.sh"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_fully_async_128K.sh"

LOCALWIKI_PORT="${LOCALWIKI_PORT:-8001}"
LOCALWIKI_READY_TIMEOUT="${LOCALWIKI_READY_TIMEOUT:-300}"  # seconds. Expect 1-3 minutes.
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOCALWIKI_LOG="${LOG_DIR}/localwiki_$(date +%Y%m%d_%H%M%S).log"

cleanup() {
    local exit_code=$?
    if [[ -n "${LOCALWIKI_PID:-}" ]] && kill -0 "${LOCALWIKI_PID}" 2>/dev/null; then
        echo "[run_all] Stopping localwiki (pid=${LOCALWIKI_PID})..."
        kill "${LOCALWIKI_PID}" 2>/dev/null || true
        wait "${LOCALWIKI_PID}" 2>/dev/null || true
    fi
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

# Refuse to start if something already serves localwiki on this port: a stale
# instance from a previous run would silently shadow the new one.
if curl -fsS -o /dev/null --max-time 3 "http://127.0.0.1:${LOCALWIKI_PORT}/docs" 2>/dev/null \
   || curl -fsS -g -o /dev/null --max-time 3 "http://[::1]:${LOCALWIKI_PORT}/docs" 2>/dev/null; then
    echo "[run_all] ERROR: port ${LOCALWIKI_PORT} is already serving localwiki. A previous instance is still alive." >&2
    echo "[run_all] Find and kill it before retrying. On the node owning port ${LOCALWIKI_PORT} (look at past Ray logs):" >&2
    echo "[run_all]     pkill -f 'uvicorn.*retrieval_server'" >&2
    echo "[run_all] You may also want to run 'ray list actors' and ray.kill any stray WikiRetrievalManager." >&2
    exit 1
fi

echo "[run_all] Launching localwiki service: ${LOCALWIKI_SCRIPT}"
echo "[run_all] localwiki logs -> ${LOCALWIKI_LOG}"
bash "${LOCALWIKI_SCRIPT}" >"${LOCALWIKI_LOG}" 2>&1 &
LOCALWIKI_PID=$!
echo "[run_all] localwiki pid=${LOCALWIKI_PID}"

echo "[run_all] Waiting for localwiki HTTP server on port ${LOCALWIKI_PORT} (timeout=${LOCALWIKI_READY_TIMEOUT}s)."
echo "[run_all] First-time startup loads FAISS + corpus.pkl, expect 1-3 minutes."
SECONDS=0
LAST_TAIL=0
TAIL_INTERVAL=30
while true; do
    if ! kill -0 "${LOCALWIKI_PID}" 2>/dev/null; then
        echo "[run_all] ERROR: localwiki process exited before becoming ready. Last log lines:" >&2
        tail -n 100 "${LOCALWIKI_LOG}" >&2 || true
        exit 1
    fi
    # Try both IPv4 and IPv6 loopback: uvicorn binds `::` and on systems
    # with bindv6only=1 the IPv4 probe will never succeed.
    if curl -fsS -o /dev/null --max-time 5 "http://127.0.0.1:${LOCALWIKI_PORT}/docs" 2>/dev/null \
       || curl -fsS -g -o /dev/null --max-time 5 "http://[::1]:${LOCALWIKI_PORT}/docs" 2>/dev/null; then
        echo "[run_all] localwiki is ready after ${SECONDS}s."
        break
    fi
    if (( SECONDS >= LOCALWIKI_READY_TIMEOUT )); then
        echo "[run_all] ERROR: timed out waiting for localwiki on port ${LOCALWIKI_PORT}. Last log lines:" >&2
        tail -n 100 "${LOCALWIKI_LOG}" >&2 || true
        exit 1
    fi
    if (( SECONDS - LAST_TAIL >= TAIL_INTERVAL )); then
        echo "[run_all] still waiting (${SECONDS}s). Last localwiki log lines:"
        tail -n 5 "${LOCALWIKI_LOG}" 2>/dev/null | sed 's/^/[localwiki] /' || true
        LAST_TAIL=${SECONDS}
    fi
    sleep 5
done

echo "[run_all] Submitting training job: ${TRAIN_SCRIPT}"
bash "${TRAIN_SCRIPT}" "$@"

echo "[run_all] Training submitted. Keeping localwiki alive (pid=${LOCALWIKI_PID})."
echo "[run_all] Press Ctrl+C to stop the localwiki service."
wait "${LOCALWIKI_PID}"
