#!/usr/bin/env bash
# Prepare an isolated modern Linux node on a host with nested KVM enabled.
set -euo pipefail

task_root=""
image=""
ssh_port=18422
api_port=18800
memory_mib=32768
cpus=8
while (($#)); do
    case "$1" in
        --root) task_root="$2"; shift 2 ;;
        --image) image="$2"; shift 2 ;;
        --ssh-port) ssh_port="$2"; shift 2 ;;
        --api-port) api_port="$2"; shift 2 ;;
        --memory-mib) memory_mib="$2"; shift 2 ;;
        --cpus) cpus="$2"; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ "$task_root" = /* && "$image" = /* ]] || { printf 'Absolute --root and --image required\n' >&2; exit 2; }
test -r "$image"
test -c /dev/kvm
for binary in qemu-img qemu-system-x86_64 cloud-localds ssh-keygen; do command -v "$binary" >/dev/null; done
mkdir -p "$task_root"
task_root="$(realpath "$task_root")"
image="$(realpath "$image")"
cd "$task_root"
umask 077
if test -f vm.pid; then
    prior_pid="$(cat vm.pid)"
    if [[ "$prior_pid" =~ ^[0-9]+$ ]] && test -r "/proc/$prior_pid/cmdline"; then
        if tr '\0' '\n' < "/proc/$prior_pid/cmdline" | grep -Fq "file=$task_root/system.qcow2,"; then
            printf 'Existing experiment VM is running, pid=%s\n' "$prior_pid"
            exit 0
        fi
        printf 'PID file does not identify the expected VM; inspect it before proceeding\n' >&2
        exit 1
    fi
fi
test -f node_key || ssh-keygen -q -t ed25519 -N '' -C 'agentenv-offload-node' -f node_key
public_key="$(cat node_key.pub)"
cat > user-data <<EOF
#cloud-config
hostname: agentenv-offload-node
manage_etc_hosts: true
disable_root: false
ssh_pwauth: false
users:
  - name: root
    lock_passwd: true
    shell: /bin/bash
    ssh_authorized_keys:
      - $public_key
write_files:
  - path: /etc/ssh/sshd_config.d/70-experiment.conf
    permissions: '0600'
    content: |
      PermitRootLogin prohibit-password
      PasswordAuthentication no
runcmd:
  - [systemctl, restart, ssh]
EOF
cat > meta-data <<EOF
instance-id: agentenv-offload-node-v1
local-hostname: agentenv-offload-node
EOF
test -f system.qcow2 || qemu-img create -f qcow2 -F qcow2 -b "$image" "$task_root/system.qcow2" 100G
cloud-localds seed.img user-data meta-data
balloon_args=()
if qemu-system-x86_64 -device virtio-balloon-pci,help 2>&1 | grep -q free-page-reporting; then
    balloon_args=(-device virtio-balloon-pci,free-page-reporting=on)
fi
qemu-system-x86_64 \
    -name agentenv-offload-node \
    -machine q35,accel=kvm -cpu host -smp "$cpus" -m "$memory_mib" \
    -drive "file=$task_root/system.qcow2,if=virtio,format=qcow2,cache=none" \
    -drive "file=$task_root/seed.img,if=virtio,format=raw,readonly=on" \
    -netdev "user,id=net0,hostfwd=tcp:0.0.0.0:$ssh_port-:22,hostfwd=tcp:127.0.0.1:$api_port-:8000" \
    -device virtio-net-pci,netdev=net0 \
    "${balloon_args[@]}" \
    -display none -serial "file:$task_root/console.log" \
    -monitor "unix:$task_root/monitor.sock,server,nowait" \
    -daemonize -pidfile "$task_root/vm.pid"
printf 'VM started, pid=%s, ssh_port=%s, api_port=%s\n' "$(cat vm.pid)" "$ssh_port" "$api_port"
qemu-system-x86_64 --version | head -1
sha256sum "$image" > image.sha256
printf '%s\n' "${balloon_args[*]:-free-page-reporting unavailable}" > balloon-mode.txt
