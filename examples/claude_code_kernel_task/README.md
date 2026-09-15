# Triton Ascend operator-generation recipe

Train an LLM through UniAgent's Claude Code agent, with operator verification on remote Ascend hosts.

## Layout

```text
uni_agent/tasks/kernel_bench/
  task.py                    # workspace, agent execution, result collection
  workspace_snapshot.py      # bounded sandbox file collection
  reward.py                  # verifier metrics and reward
  preprocess.py              # DrKernel preparation
  track_verify_snapshot.py   # best-prefix and early-stop hook

examples/claude_code_kernel_task/
  task_config_kernel_bench.yaml
  runner.py                  # per-session bindings and Claude output budget
  remote_docker.py           # Docker provider, host selection and NPU settings
  trajectory_processor.py    # trajectory postprocessor
  run_train_gpu.sh           # GPU training, remote NPU verification
  sandbox/                   # image template, pinned upstream assets and RL patch
```

The Task is registered as `npu_triton_kernel`. The runner delegates to the stock `run_task`; the Task uses `uni_agent.agents.claude_code`.

## 1. Prepare the sandbox hosts

The `sandbox/` directory contains image build files and local RL adaptations. `build_image.sh` assembles skills and verifier scripts from a pinned upstream commit plus a local patch; it also accepts a local upstream repository for offline builds. The base image must already contain Claude Code, Python, Ascend/CANN, torch-npu, Triton Ascend, sudo and the `claude` user. See [sandbox setup](sandbox/README.md); the image must match `task_config_kernel_bench.yaml`.

Build the image on each remote Docker daemon:

```bash
cd examples/claude_code_kernel_task/sandbox
DOCKER_HOST=ssh://root@npu-host-01 \
BASE_IMAGE='<your-prepared-ascend-claude-image>:<tag>' \
bash build_image.sh
```

Replace the placeholder with your prepared base image. The output defaults to `triton-claude-code-env:latest`, matching `task_config_kernel_bench.yaml`; if overriding `OUTPUT_IMAGE`, update the YAML too. Check all device/driver bind-mount sources in that file against each host.

### External sources

- Triton/NPU skills originate from [CANNBot Skills PR #205](https://gitcode.com/cann/cannbot-skills/pull/205), with local RL adaptations described in the [sandbox README](sandbox/README.md#skills-source).
- DrKernel dataset: [AhNr/dr-kernel-RL](https://huggingface.co/datasets/AhNr/dr-kernel-RL). Download the dataset separately and pass its local parquet files to preprocessing.

On each NPU host, create the shared lock files:

```bash
sudo install -d -o root -g root -m 1777 /var/lock/triton-agent-npu
for device in 0 1 2 3 4 5 6 7; do
  sudo touch "/var/lock/triton-agent-npu/device-${device}.lock"
  sudo chown root:root "/var/lock/triton-agent-npu/device-${device}.lock"
  sudo chmod 0666 "/var/lock/triton-agent-npu/device-${device}.lock"
done
```

Configure SSH access and trusted host keys from every Ray node that may run tasks, then check:

```bash
docker --host ssh://root@npu-host-01 info
```

`REMOTE_DOCKER_HOSTS` is a comma-separated string, for example `ssh://root@npu-host-01,ssh://root@npu-host-02:2222`. All hosts must provide the configured image and device IDs. The sandbox must reach the Gateway's advertised LAN address; no reverse tunnel is used. Do not expose an unauthenticated Docker TCP endpoint.

Each session creates a fresh container and destroys it after execution. Containers on one host share that host's lock directory; different hosts have independent device pools. Do not use a global NFS lock directory. Verifier calls acquire a device only while verifying/benchmarking, not for the whole session.

Claude runs as a non-root user; the fixed verifier/cleanup commands use sudo for NPU access. Privileged containers and advisory locks are a cooperative deployment model, not an isolation boundary against malicious code. Container TTL is the fallback for hard-killed workers; it cannot guarantee immediate cleanup after host/daemon failures.

## 2. Prepare data

For DrKernel parquet files (`training_*.parquet` and `validation_level*.parquet`):

```bash
python -m uni_agent.tasks.kernel_bench.preprocess \
  --train-source /data/drkernel \
  --validation-source /data/drkernel \
  --drkernel-validation-levels 1,2 \
  --output-dir /data/triton-agent
```

Preprocessing produces `train.parquet`, `validation.parquet`, and `dataset_summary.json`. DrKernel defaults to ten input groups; use `--drkernel-num-cases` to change it. See `--help` for validation-level selection, seeded shuffling and sample limits.

## 3. Start training

Run from `examples/claude_code_kernel_task` in the configured training environment:

```bash
MODEL_PATH=/models/your-model \
TRAIN_FILE=/data/triton-agent/train.parquet \
VAL_FILE=/data/triton-agent/validation.parquet \
REMOTE_DOCKER_HOSTS=ssh://root@npu-host-01,ssh://root@npu-host-02 \
EVALUATOR_NPU_DEVICE_IDS=0,1,2,3,4,5,6,7 \
bash run_train_gpu.sh
```

Adjust model parallelism and concurrency to the actual cluster.

The launcher submits the repository with Ray `--working-dir`; `RUNTIME_ENV` is optional for deployment-specific dependencies/environment. The installed verl and accelerator packages must match the selected source revision.

Training uses the resolved Task configuration's `agent.max_turns` and `agent.run_timeout` (YAML defaults: 100 and 7200s). Samples with `metadata.split: validation` instead use `VALIDATION_MAX_TURNS` (default 120) and `VALIDATION_RUN_TIMEOUT` (default 10800s). This selects by dataset split, not the Framework execution phase. Both share `SESSION_TIMEOUT_SECONDS` (default 11400s). If you increase either agent budget, adjust the session timeout and `sandbox.runtime_timeout` (currently 14400s) to leave time for setup and cleanup.

## Task results and trajectories

The workspace template supplies `CLAUDE.md`, `INSTRUCTIONS.md` and skills. The stock agent passes the user message to `claude -p` and forwards non-empty system messages.

During execution, `tools/verify_once.sh` performs AST checking, NPU correctness testing and, after full correctness, latency benchmarking. It maintains the best implementation/metrics pair. After Claude exits, the Task stops remaining verifier processes and reads that pair, falling back to staged verifier artifacts or current metrics.

Rewards retain AST, compilation, correctness and speedup components. `finished=True` describes agent completion, not correctness; use accuracy/pass rate to judge verification.

The Claude hook records best-snapshot assistant indices and cooperatively stops searches after configured non-improvement counts. Defaults: seven verify calls before correctness (best reward at least 0.15), three latency calls after correctness. Set a patience to zero to disable that phase. Do not remove the hook if best-prefix or early-stop is needed.

Each session runs once in one sandbox. A missing implementation remains a `missing_impl` result with zero reward, not a reason to rerun or discard the trajectory.

The postprocessor exposes only `selection`, defaulting to `best`. Missing/invalid best hints and multiple chains always fall back to `all_final`, since a scalar assistant index cannot identify a Gateway chain. Partial-credit best hints are accepted. Empty inputs or trajectories without an assistant span produce no prefix; token-array misalignment raises. Tokens, masks, logprobs and R3 routes are cropped together without modifying the input.

Task `timing_ms` records `setup`, `agent`, `evaluate`, and `total` durations in milliseconds. Setting `artifact_dir` enables artifact exports to `<artifact_dir>/<uid>/<session_id>/`.

## Logs

Each completed Task prints one `[triton-result]` line to the Ray job's main log and session logger, including sample/session identity, operator name, passed/total cases, reward, metric source and completion status. This reports the selected verification snapshot, not whether the framework later retains the trajectory. Tasks that raise before returning a result are reported through the existing exception logs.

The GPU launcher saves the main log under `LOG_DIR/<experiment>.log`. The launcher saves per-session logs under `LOG_DIR/<experiment>/step_<n>/<session_id>/`: `task.log`, `framework.log`, and, when trajectory dumping is reached, `trajectory.json` / `trajectory.npz`. `AGENT_LOG_DIR` is derived from `LOG_DIR` and is not a separate environment override. Ray workers must have access to the same filesystem for these files to appear together.

Claude stdout/stderr logging contains only tails, not a full transcript. `artifact_dir` is disabled by default; enable it in the task YAML to export selected metrics JSON and implementation files before sandbox destruction. It does not export full Claude transcripts or raw verifier logs. In multi-node runs, use shared storage or collect artifacts from each runner host.

## Current limitations

The task sets `CLAUDE_CODE_SKIP_PROMPT_HISTORY=0` to allow transcript persistence for best-prefix attribution. A valid snapshot hint is still required; unavailable hints fall back to finalized trajectories.

Early-stop markers are terminal for a session: subsequent tool admissions are denied, and late verifier completions cannot clear them. This is cooperative admission control, not a hard kill of an in-flight tool or model request.

## Validation

Recipe tests use the registered `cpu` and `level0` markers:

```bash
python -m pytest -q -m "cpu and level0" tests/uni_agent/tasks/kernel_bench
bash -n examples/claude_code_kernel_task/run_train_gpu.sh
```

Common deployment settings are environment variables in the launcher. Additional training settings can be passed as Hydra overrides after the script name; they are appended last. A running Ray cluster and a compatible installed verl/Megatron/vLLM environment are required.
