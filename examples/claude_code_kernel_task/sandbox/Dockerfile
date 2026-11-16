ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SANDBOX_USER=claude
USER root

COPY template/ /opt/triton-agent-template/
COPY tools/ /opt/triton-agent-tools/

RUN set -eux; \
    id -u "${SANDBOX_USER}" >/dev/null; \
    test -x /usr/bin/sudo; \
    mkdir -p \
      /etc/sudoers.d \
      /opt/triton-agent-template/tools \
      /opt/triton-agent-template/.claude/skills/triton-op-verifier \
      /workspace; \
    ln -s /opt/triton-agent-tools/verify_once.sh \
      /opt/triton-agent-template/tools/verify_once.sh; \
    ln -s /opt/triton-agent-tools/verifier \
      /opt/triton-agent-template/.claude/skills/triton-op-verifier/scripts; \
    chmod -R a+rX,a-w /opt/triton-agent-template /opt/triton-agent-tools; \
    chmod 0555 \
      /opt/triton-agent-tools/cleanup_task_processes.sh \
      /opt/triton-agent-tools/verify_once.sh \
      /opt/triton-agent-tools/with_npu_lease.py; \
    printf '%s\n' \
      "Defaults:${SANDBOX_USER} env_keep += \"TRITON_EVAL_DEVICE_IDS TRITON_EVAL_LOCK_DIR TRITON_EVAL_LOCK_TIMEOUT TRITON_EVAL_TIMEOUT TRITON_LATENCY_BENCHMARK_TIMEOUT TRITON_LATENCY_BENCHMARK_WARMUP TRITON_LATENCY_BENCHMARK_REPEATS\"" \
      "${SANDBOX_USER} ALL=(root) NOPASSWD: /opt/triton-agent-tools/verify_once.sh *, /opt/triton-agent-tools/cleanup_task_processes.sh \"\"" \
      > /etc/sudoers.d/triton-agent; \
    chmod 0440 /etc/sudoers.d/triton-agent; \
    visudo -cf /etc/sudoers.d/triton-agent; \
    chown "${SANDBOX_USER}:" /workspace

ENV HOME=/home/${SANDBOX_USER}
USER ${SANDBOX_USER}
WORKDIR /workspace
