# Repository agent instructions

## Codex recipe entrypoint and acceptance

For Codex recipe integration checks and SWE-bench acceptance, use
`examples/codex/run_infer_codex.sh` as the entrypoint. The script starts the
verl-managed inference path and dispatches the task through
`AgentFrameworkRolloutAdapter` and `task_runner/run_task`.

The acceptance run is one SWE-bench sample and one rollout session. A run is
complete only when the script exits successfully after verifying the result
file contains exactly one scored trajectory with reward `1.0`.

The entrypoint selects `SAMPLE_INDEX=0` by default and normalizes legacy parquet
rows into the current `extra_info.tools_kwargs.task` shape before submission.
It passes `AKERNEL_SDK_LD_PRELOAD` through the Ray runtime environment for the
OpenYuanRong worker's libffi ABI compatibility.
