#!/usr/bin/env bash
# 在专用实验节点部署固定版本 AgentEnv；禁止用于共享宿主机。
set -Eeuo pipefail
umask 027

ROOT=/opt/agentenv-offload
REV=a27b2db909b035f58696ba8798ea0e285c7bfa17
SRC="$ROOT/source"
STATE="$ROOT/state"
RUNTIME=/run/agentenv-offload
CONFIG="$ROOT/config.toml"
# 原始部署使用 stable 通道，实际 rustc 为 1.99.1；复现默认固定该版本。
AENV_RUST_TOOLCHAIN="${AENV_RUST_TOOLCHAIN:-1.99.1}"
AENV_POOL_PRESET="${AENV_POOL_PRESET:-off}"
export RUSTUP_TOOLCHAIN="$AENV_RUST_TOOLCHAIN"
case "$AENV_POOL_PRESET" in
  off|bound1) ;;
  *) echo 'AENV_POOL_PRESET 仅支持 off 或 bound1'; exit 1 ;;
esac

require_service_stopped() {
  local service_state
  service_state=$(systemctl show agentenv-offload.service --property=ActiveState --value 2>/dev/null || true)
  case "$service_state" in
    active|activating|deactivating|reloading)
      echo '服务正在运行或切换状态，拒绝重新部署；请先完成任务并明确停止服务'
      exit 1
      ;;
  esac
}

[[ $(id -u) == 0 ]] || { echo '需要专用节点 root 权限'; exit 1; }
[[ $(hostname) == agentenv-offload-node ]] || { echo '拒绝在非专用实验节点运行'; exit 1; }
[[ $(uname -m) == x86_64 ]] || { echo '此安装脚本仅验证 x86_64'; exit 1; }
[[ -e /dev/kvm ]] || { echo '/dev/kvm 不可用'; exit 1; }
require_service_stopped
install -d -m 0755 "$ROOT"
exec 9>"$ROOT/provision.lock"
flock -n 9 || { echo '已有安装进程，拒绝重复启动'; exit 1; }
if [[ -f "$ROOT/provision.exit" ]]; then
  mv "$ROOT/provision.exit" "$ROOT/provision.previous-exit"
fi
echo "$$" >"$ROOT/provision.pid"
trap 'rc=$?; printf "%s\n" "$rc" >"$ROOT/provision.exit"; exit "$rc"' EXIT
phase() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; printf '%s\n' "$*" >"$ROOT/provision.phase"; }

phase 'install-packages'
export DEBIAN_FRONTEND=noninteractive
if [[ ! -f "$ROOT/packages.ready" ]]; then
  env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy apt-get update
  env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy apt-get install -y build-essential pkg-config cmake clang libclang-dev \
    libprotobuf-dev protobuf-compiler libssl-dev git ca-certificates curl \
    e2fsprogs iproute2 iptables jq sudo umoci zstd kmod util-linux python3
  touch "$ROOT/packages.ready"
fi
if ! modinfo ublk_drv >/dev/null 2>&1; then
  apt-get install -y "linux-modules-extra-$(uname -r)"
fi
modprobe ublk_drv
[[ -e /dev/ublk-control ]] || { echo 'ublk 驱动加载后没有控制设备'; exit 1; }

phase 'source-and-toolchain'
if [[ ! -d "$SRC/.git" ]]; then
  [[ ! -e "$SRC" ]] || { echo '源码目录存在且不是 Git 仓库，拒绝覆盖'; exit 1; }
  git clone --filter=blob:none https://github.com/kvcache-ai/AgentENV.git "$SRC"
fi
[[ -z $(git -C "$SRC" status --porcelain) ]] || { echo '源码目录不干净，拒绝覆盖'; exit 1; }
if ! git -C "$SRC" cat-file -e "$REV^{commit}" 2>/dev/null; then
  git -C "$SRC" fetch origin "$REV"
fi
git -C "$SRC" checkout --detach "$REV"
[[ $(git -C "$SRC" rev-parse HEAD) == "$REV" ]]
if [[ ! -x /root/.cargo/bin/cargo ]]; then
  curl -fL --retry 3 --retry-all-errors \
    https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init \
    -o "$ROOT/rustup-init"
  chmod 0700 "$ROOT/rustup-init"
  "$ROOT/rustup-init" -y --profile minimal --default-toolchain "$AENV_RUST_TOOLCHAIN"
fi
export PATH="/root/.cargo/bin:$PATH"
rustup toolchain install "$AENV_RUST_TOOLCHAIN" --profile minimal
rustc --version
cargo --version

phase 'build'
cd "$SRC"
rustc_version=$(rustc --version)
if [[ ! -s "$ROOT/binary-sha256.txt" ]] || \
   [[ ! -f "$ROOT/build-rustc.txt" ]] || \
   [[ $(cat "$ROOT/build-rustc.txt") != "$rustc_version" ]] || \
   ! sha256sum -c "$ROOT/binary-sha256.txt" >/dev/null 2>&1; then
  cargo build --release --locked -j 8 \
    -p agentenv --bin server \
    -p uvm-ublk-daemon --bin uvm-ublk-daemon \
    -p aenv --bin aenv >"$ROOT/build.log" 2>&1
  sha256sum target/release/server target/release/uvm-ublk-daemon target/release/aenv >"$ROOT/binary-sha256.txt"
  printf '%s\n' "$rustc_version" >"$ROOT/build-rustc.txt"
fi
# 构建以受限 umask 运行；服务账户只需遍历目标目录并执行这三个程序。
chmod 0755 "$SRC" "$SRC/target" "$SRC/target/release"
chmod 0755 target/release/server target/release/uvm-ublk-daemon target/release/aenv

phase 'configuration'
require_service_stopped
id aenv >/dev/null 2>&1 || useradd --system --user-group --home-dir /var/lib/aenv --create-home --shell /usr/sbin/nologin aenv
getent group kvm >/dev/null || groupadd --system kvm
install -d -o aenv -g aenv -m 0700 "$STATE"
install -d -o aenv -g aenv -m 0750 "$RUNTIME"
python3 - "$SRC/config/default.toml" "$CONFIG" "$AENV_POOL_PRESET" <<'PY'
import pathlib, re, sys
source, output = map(pathlib.Path, sys.argv[1:3])
preset = sys.argv[3]
if preset not in ('off', 'bound1'):
    raise ValueError('invalid pool preset')
text = source.read_text()
changes = {
    'firecracker': {
        'work_dir': '"/opt/agentenv-offload/state/firecracker-work"',
        'serial_dir': '"/opt/agentenv-offload/state/serial"',
        'log_level': '"Info"',
    },
    'orchestrator': {
        'metrics_interval_secs': '1',
        'metrics_retention_secs': '3600',
        'default_sandbox_timeout_secs': '600',
        'persisted_sandbox_store_path': '"/opt/agentenv-offload/state/persisted-sandboxes"',
    },
    'pool.firecracker': {'enabled': 'false', 'startup_prewarm': 'false'},
    'pool.block': {'enabled': 'false'},
    'pool.network': {'maintenance_enabled': 'false'},
    'observability': {'enabled': 'true'},
    'observability.scheduler_report': {'enabled': 'false'},
}
if preset == 'bound1':
    changes['pool'] = {'low_watermark': '1', 'high_watermark': '1'}
    changes['pool.firecracker'] = {
        'enabled': 'true', 'maintenance_enabled': 'true',
        'startup_prewarm': 'true', 'fill_concurrency': '1',
    }
for section, fields in changes.items():
    pattern = r'(?ms)^\[' + re.escape(section) + r'\]\n(.*?)(?=^\[|\Z)'
    found = re.search(pattern, text)
    if found is None:
        raise RuntimeError(f'missing section: {section}')
    body = found.group(1)
    for key, value in fields.items():
        field = r'(?m)^\s*(?:#\s*)?' + re.escape(key) + r'\s*=.*$'
        body, count = re.subn(field, key + ' = ' + value, body)
        if count == 0:
            body += key + ' = ' + value + '\n'
        elif count != 1:
            raise RuntimeError(f'duplicate field: {section}.{key}')
    text = text[:found.start(1)] + body + text[found.end(1):]
output.write_text(text)
PY
chown root:aenv "$CONFIG"
chmod 0640 "$CONFIG"

export AENV_CONFIG_PATH="$CONFIG"
export AENV_HOME_PATH="$STATE"
export AENV_RUNTIME_PATH="$RUNTIME"
export AENV_UBLK_DAEMON_BINARY_PATH="$SRC/target/release/uvm-ublk-daemon"
export AENV_VIRTUALIZATION_MODE=kvm
phase 'setup'
# overlaybd-apply 在解析自定义路径前仍检查默认配置路径。
install -d -m 0755 /etc/overlaybd
if [[ ! -e /etc/overlaybd/overlaybd.json ]]; then
  printf '{}\n' >/etc/overlaybd/overlaybd.json
  chmod 0644 /etc/overlaybd/overlaybd.json
fi
"$SRC/target/release/server" --setup-only >>"$ROOT/setup.log" 2>&1
"$SRC/target/release/server" --setup-host --runtime-user aenv --runtime-group aenv >>"$ROOT/setup.log" 2>&1
chown -R aenv:aenv "$STATE"
install -d -o aenv -g aenv -m 0750 "$RUNTIME"

phase 'service'
if [[ -n ${HTTPS_PROXY:-} ]]; then
  {
    printf 'HTTPS_PROXY=%s\nHTTP_PROXY=%s\n' "$HTTPS_PROXY" "${HTTP_PROXY:-$HTTPS_PROXY}"
    printf 'https_proxy=%s\nhttp_proxy=%s\n' "$HTTPS_PROXY" "${HTTP_PROXY:-$HTTPS_PROXY}"
    printf 'NO_PROXY=%s\nno_proxy=%s\n' "${NO_PROXY:-localhost,127.0.0.1,::1}" "${NO_PROXY:-localhost,127.0.0.1,::1}"
  } >"$ROOT/download.env"
  chown root:aenv "$ROOT/download.env"
  chmod 0640 "$ROOT/download.env"
fi
cat > /etc/systemd/system/agentenv-offload.service <<'UNIT'
[Unit]
Description=AgentEnv offload experiment node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=aenv
Group=aenv
SupplementaryGroups=kvm
WorkingDirectory=/opt/agentenv-offload/source
Environment=AENV_CONFIG_PATH=/opt/agentenv-offload/config.toml
Environment=AENV_HOME_PATH=/opt/agentenv-offload/state
Environment=AENV_RUNTIME_PATH=/run/agentenv-offload
Environment=AENV_UBLK_DAEMON_BINARY_PATH=/opt/agentenv-offload/source/target/release/uvm-ublk-daemon
Environment=AENV_VIRTUALIZATION_MODE=kvm
Environment=API_ADDR=0.0.0.0:8000
Environment=RUST_LOG=info
EnvironmentFile=-/opt/agentenv-offload/download.env
AmbientCapabilities=CAP_NET_ADMIN CAP_SYS_ADMIN
CapabilityBoundingSet=CAP_NET_ADMIN CAP_SYS_ADMIN
LimitNOFILE=65536
LimitMEMLOCK=infinity
RuntimeDirectory=agentenv-offload
RuntimeDirectoryMode=0750
UMask=0077
ExecStart=/opt/agentenv-offload/source/target/release/server
Restart=no
TimeoutStopSec=120
StandardOutput=append:/opt/agentenv-offload/server.log
StandardError=append:/opt/agentenv-offload/server.log

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl start agentenv-offload.service
for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null; then break; fi
  systemctl is-active --quiet agentenv-offload.service || { echo '服务启动失败，检查 server.log'; exit 1; }
  sleep 1
done
curl -fsS http://127.0.0.1:8000/health >/dev/null

phase 'credential-and-verification'
python3 - <<'PY'
import json, os, pathlib, urllib.request
key = pathlib.Path('/opt/agentenv-offload/state/secrets/api-key').read_text().strip()
folder = pathlib.Path('/root/.config/aenv')
folder.mkdir(parents=True, exist_ok=True)
os.chmod(folder, 0o700)
path = folder / 'credentials'
with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as f:
    f.write('url = "http://127.0.0.1:8000"\napi_key = ' + json.dumps(key) + '\n')
req = urllib.request.Request('http://127.0.0.1:8000/nodes', headers={'X-API-Key': key})
with urllib.request.urlopen(req, timeout=15) as response:
    nodes = json.load(response)
if len(nodes) != 1 or not nodes[0].get('commit', '').startswith('a27b2db'):
    raise RuntimeError('node commit verification failed')
summary = [{'id': n['id'], 'commit': n['commit'], 'version': n['version'],
            'sandboxCount': n['sandboxCount'], 'sandboxPausedCount': n['sandboxPausedCount']} for n in nodes]
pathlib.Path('/opt/agentenv-offload/node-verification.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
PY
ln -sfn "$SRC/target/release/aenv" /usr/local/bin/aenv
systemctl show agentenv-offload.service --property=MainPID --property=ActiveState
phase 'ready'
