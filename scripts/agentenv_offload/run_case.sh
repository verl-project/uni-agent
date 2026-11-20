#!/usr/bin/env bash
# Run one explicitly named experiment from the remote186 client.
set -euo pipefail
umask 077
if (($# < 7)); then
    printf 'usage: %s LABEL MODE WAIT_SECONDS MEMORY_MIB CYCLES PAIRS OBSERVATION_SECONDS [demo options]\n' "$0" >&2
    exit 2
fi
label="$1"; mode="$2"; wait_seconds="$3"; memory_mib="$4"
cycles="$5"; pairs="$6"; observation_seconds="$7"
shift 7
[[ "$label" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { printf 'Invalid label\n' >&2; exit 2; }
experiment_root="${AGENTENV_EXPERIMENT_ROOT:-/home/zxh/uni-agent-sandbox-offload}"
python_path="${AGENTENV_CLIENT_PYTHON:-/home/zxh/miniconda3/envs/uni-agent/bin/python}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
access_dir="$experiment_root/node-access"
ssh_config="$access_dir/ssh_config"
node_alias=agentenv-node
mkdir -p "$experiment_root/results" "$experiment_root/logs"
exec 9>"$experiment_root/case.lock"
flock -n 9 || { printf 'Another experiment is running\n' >&2; exit 1; }
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_name="$label-$stamp"
run_dir="$experiment_root/results/$run_name"
log_prefix="$experiment_root/logs/$run_name"

# Fail before create when the dedicated API tunnel is not healthy.
"$python_path" - <<'PY'
from urllib.request import ProxyHandler, build_opener
with build_opener(ProxyHandler({})).open('http://127.0.0.1:18800/health', timeout=10) as response:
    if response.status not in (200, 204):
        raise RuntimeError('API tunnel health failed')
PY
ssh -n -F "$ssh_config" "$node_alias" \
    'cat /opt/agentenv-offload/pool-variant-bound1.json' > "$log_prefix.deployment.json"
ssh -n -F "$ssh_config" "$node_alias" \
    'sha256sum /opt/agentenv-offload/config.toml' > "$log_prefix.config-before.sha256"
printf '%s\n' "$run_dir" > "$experiment_root/latest-run.txt"
printf 'RUN_DIR=%s\n' "$run_dir"
if "$python_path" "$script_dir/demo.py" \
    --api-url http://127.0.0.1:18800 --credentials-file "$access_dir/api-credentials.toml" \
    --node-ssh "$node_alias" --ssh-config "$ssh_config" \
    --node-root /opt/agentenv-offload/runs --aenv-bin /usr/local/bin/aenv \
    --serial-dir /opt/agentenv-offload/state/serial --state-dir /opt/agentenv-offload/state \
    --runtime-dir /run/agentenv-offload --provider-pid-file /opt/agentenv-offload/provider-pids.json \
    --guest-image docker.io/library/python@sha256:8fb0b110b022c70cbe09d3ff1fb313654f31f9c814abb0d48180af56d906a199 \
    --output "$run_dir" --mode "$mode" --wait-seconds "$wait_seconds" --memory-mib "$memory_mib" \
    --sandbox-memory-mib 1024 --cycles "$cycles" --pairs "$pairs" --entropy low --dirty-ratio 0 \
    --seed 1 --cpus 1 --operation-timeout 300 --sample-interval 0.1 --io-tail-seconds 2 \
    --observation-seconds "$observation_seconds" "$@" 2>&1 | tee "$log_prefix.runner.log"; then
    result=0
else
    result=$?
fi
if test -d "$run_dir"; then
    cp "$log_prefix.deployment.json" "$run_dir/deployment.json"
    cp "$log_prefix.config-before.sha256" "$run_dir/config-before.sha256"
    ssh -n -F "$ssh_config" "$node_alias" \
        'sha256sum /opt/agentenv-offload/config.toml' > "$run_dir/config-after.sha256"
    if ! cmp -s "$run_dir/config-before.sha256" "$run_dir/config-after.sha256"; then
        printf 'Node configuration changed during experiment\n' >&2
        result=1
    fi
    cp "$0" "$run_dir/source/run_case.sh"
    "$python_path" "$run_dir/source/analyze.py" "$run_dir"
fi
exit "$result"
