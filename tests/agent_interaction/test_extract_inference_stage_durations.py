import importlib.util
import json
import sys
from pathlib import Path


_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_EXTRACTOR_SCRIPT = _WORKSPACE_ROOT / "scripts" / "extract_inference_stage_durations.py"


def _load_extractor_module():
    module_name = "_extract_inference_stage_durations_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, _EXTRACTOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _stage_by_name(stages):
    return {stage.name: stage for stage in stages}


def test_summarize_vllm_stage_durations_from_terminal_log(tmp_path):
    module = _load_extractor_module()
    terminal_log = tmp_path / "inference-20260603T000000Z-123.log"
    terminal_log.write_text(
        "\n".join(
            [
                "Terminal log: /tmp/inference.log",
                "2026-06-03 00:00:01,000\tINFO worker.py:2012 -- Started a local Ray instance.",
                "2026-06-03 00:00:03,000 - __main__ - INFO - Initializing configuration and AgentLoopManager...",
                "(vLLMHttpServer pid=1) INFO 06-03 00:00:05 [model.py:617] Resolved architecture: Qwen3_5MoeForConditionalGeneration",
                "(EngineCore pid=2) INFO 06-03 00:00:08 [core.py:112] Initializing a V1 LLM engine (v0.22.0) with config: model='model'",
                "(Worker_TP0 pid=3) INFO 06-03 00:00:10 [gpu_model_runner.py:5037] Starting to load model model...",
                "(Worker_TP0 pid=3) INFO 06-03 00:00:20 [default_loader.py:397] Loading weights took 8.50 seconds",
                "(Worker_TP0 pid=3) INFO 06-03 00:00:21 [gpu_model_runner.py:5132] Model loading took 16.52 GiB memory and 11.250000 seconds",
                "(Worker_TP0 pid=3) 2026-06-03 00:00:30,000 - INFO - autotuner.py:615 - flashinfer.jit: [Autotuner]: Autotuning process starts ...",
                "(Worker_TP1 pid=4) 2026-06-03 00:00:45,500 - INFO - autotuner.py:634 - flashinfer.jit: [Autotuner]: Autotuning process ends",
                "(Worker_TP0 pid=3) INFO 06-03 00:00:46 [monitor.py:53] torch.compile and initial profiling/warmup run together took 12.34 s in total",
                "(Worker_TP0 pid=3) INFO 06-03 00:01:00 [gpu_model_runner.py:6456] Graph capturing finished in 18 secs, took 1.17 GiB",
                "(EngineCore pid=2) INFO 06-03 00:01:02 [core.py:302] init engine (profile, create kv cache, warmup model) took 80.00 s (compilation: 12.34 s)",
                "(vLLMHttpServer pid=1) INFO 06-03 00:01:15 [base.py:224] Multi-modal warmup completed in 12.535s",
                "=> Mean RM Score: 1.0000",
            ]
        ),
        encoding="utf-8",
    )

    sources, stages = module.summarize_logs([terminal_log])
    by_name = _stage_by_name(stages)

    assert sources == [str(terminal_log)]
    assert by_name["Weight loading"].duration_s == 8.5
    assert by_name["Model loading"].duration_s == 11.25
    assert by_name["torch.compile and initial profiling/warmup"].duration_s == 12.34
    assert by_name["FlashInfer autotuning"].duration_s == 15.5
    assert by_name["CUDA graph capture"].duration_s == 18.0
    assert by_name["Engine profile/create KV/warmup"].duration_s == 80.0
    assert by_name["Final RM score"].status == "observed"


def test_summarize_agent_run_aggregate_durations(tmp_path):
    module = _load_extractor_module()
    run_dir = tmp_path / "agent-run"
    run_dir.mkdir()
    run_log = run_dir / "run.log"
    run_log.write_text(
        "\n".join(
            [
                "2026-06-03 00:02:00 | agent-loop   | INFO     | model name: model",
                "2026-06-03 00:02:00 | environment  | INFO     | Beginning environment startup...",
                "2026-06-03 00:02:08 | environment  | INFO     | Runtime initialized",
                "2026-06-03 00:02:15 | environment  | INFO     | Tool submit successfully installed",
                "2026-06-03 00:02:20 | interaction  | INFO     | Inital Prompt:",
                "2026-06-03 00:03:00 | agent-loop   | INFO     | reward_score: True",
                "2026-06-03 00:03:02 | environment  | INFO     | Beginning environment shutdown...",
                "2026-06-03 00:03:05 | environment  | INFO     | Environment shutdown completed",
            ]
        ),
        encoding="utf-8",
    )

    sources, stages = module.summarize_logs([], [run_log])
    by_name = _stage_by_name(stages)

    assert sources == [str(run_log)]
    assert by_name["Agent environment startup"].duration_s == 15.0
    assert by_name["Agent interaction to reward"].duration_s == 40.0
    assert by_name["Agent environment shutdown"].duration_s == 3.0
    assert by_name["Agent rollout wall time"].duration_s == 65.0


def test_cli_writes_json_without_modifying_source_log(tmp_path, capsys):
    module = _load_extractor_module()
    terminal_log = tmp_path / "inference-20260603T000000Z-123.log"
    terminal_log.write_text(
        "\n".join(
            [
                "2026-06-03 00:00:01,000\tINFO worker.py:2012 -- Started a local Ray instance.",
                "(Worker_TP0 pid=3) INFO 06-03 00:00:20 [default_loader.py:397] Loading weights took 8.50 seconds",
            ]
        ),
        encoding="utf-8",
    )
    before_mtime = terminal_log.stat().st_mtime_ns
    output = tmp_path / "summary.json"

    result = module.main(
        [
            "--workspace-root",
            str(tmp_path),
            "--terminal-log",
            str(terminal_log),
            "--no-agent-logs",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    written = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(captured.out)
    assert result == 0
    assert written == printed
    assert written["sources"] == [str(terminal_log)]
    assert terminal_log.stat().st_mtime_ns == before_mtime
