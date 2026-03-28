import sys
from pathlib import Path

from dashboard.server import DashboardMonitor, extract_current_step, infer_status, parse_args


def test_infer_status_from_uni_agent_files() -> None:
    assert infer_status({"run.log"}, "") == "queued"
    assert infer_status({"run.log"}, "Beginning environment startup...") == "running"
    assert infer_status({"run.log"}, "2026-01-01 | reward_spec | INFO | Eval report: {}") == "verify"
    assert infer_status({"run.log", "interaction_result.json"}, "anything") == "completed"


def test_extract_current_step_from_log_excerpt() -> None:
    log_excerpt = "\n".join(
        [
            "2026-01-01 | interaction | INFO | ========================= STEP 1 =========================",
            "2026-01-01 | interaction | INFO | ========================= STEP 7 =========================",
        ]
    )
    assert extract_current_step(log_excerpt) == 7


def test_monitor_reads_run_log_incrementally(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    run_dir = log_dir / "sample-run"
    run_dir.mkdir(parents=True)

    run_log = run_dir / "run.log"
    run_log.write_text("Beginning environment startup...\n", encoding="utf-8")

    monitor = DashboardMonitor(log_dir, poll_interval=60)
    monitor.scan_once()
    first_snapshot = monitor.snapshot()
    run = first_snapshot["runs"][0]
    assert run["status"] == "running"
    assert "Beginning environment startup..." in run["log_excerpt"]

    run_log.write_text(
        "Beginning environment startup...\n2026-01-01 00:00:01 | interaction  | INFO     | STEP 1\n",
        encoding="utf-8",
    )
    monitor.scan_once()
    second_snapshot = monitor.snapshot()
    updated_run = second_snapshot["runs"][0]
    assert "STEP 1" in updated_run["log_excerpt"]
    assert updated_run["current_step"] == 1
    appended_events = [event for event in monitor.events if event["type"] == "log_append"]
    assert appended_events


def test_monitor_marks_completed_when_result_file_exists(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    run_dir = log_dir / "finished-run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text("Beginning environment startup...\n", encoding="utf-8")
    (run_dir / "interaction_result.json").write_text("{}", encoding="utf-8")

    monitor = DashboardMonitor(log_dir, poll_interval=60)
    monitor.scan_once()
    run = monitor.snapshot()["runs"][0]
    assert run["status"] == "completed"


def test_monitor_exposes_verify_log_source_when_present(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    run_dir = log_dir / "verify-run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text("Beginning environment startup...\n", encoding="utf-8")
    (run_dir / "verify.log").write_text("verify output\n", encoding="utf-8")

    monitor = DashboardMonitor(log_dir, poll_interval=60)
    monitor.scan_once()
    run = monitor.snapshot()["runs"][0]
    log_labels = {item["label"] for item in run["log_sources"]}

    assert "Run log" in log_labels
    assert "Verify log" in log_labels
    assert run["log_contents"]["verify.log"] == "verify output\n"


def test_monitor_clears_deleted_log_state(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    run_dir = log_dir / "deleted-log-run"
    run_dir.mkdir(parents=True)
    run_log = run_dir / "run.log"
    run_log.write_text("Beginning environment startup...\nSTEP 3\n", encoding="utf-8")

    monitor = DashboardMonitor(log_dir, poll_interval=60)
    monitor.scan_once()
    first_run = monitor.snapshot()["runs"][0]
    assert first_run["log_excerpt"]
    assert first_run["current_step"] == 3

    run_log.unlink()
    monitor.scan_once()
    updated_run = monitor.snapshot()["runs"][0]
    assert updated_run["status"] == "queued"
    assert updated_run["log_excerpt"] == ""
    assert updated_run["current_step"] is None
    assert updated_run["log_sources"] == []


def test_monitor_reads_large_log_in_chunks(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    run_dir = log_dir / "chunked-run"
    run_dir.mkdir(parents=True)
    lines = [f"line-{idx:03d}" for idx in range(120)]
    payload = "\n".join(lines) + "\n"
    (run_dir / "run.log").write_text(payload, encoding="utf-8")

    monitor = DashboardMonitor(log_dir, poll_interval=60, max_log_tail_chars=40)
    monitor.scan_once()

    latest_chunk = monitor.read_log_chunk("chunked-run", "run.log", chunk_size=80)
    assert latest_chunk["text"].endswith("line-119\n")
    assert latest_chunk["has_more"] is True

    combined = latest_chunk["text"]
    before = latest_chunk["start_offset"]
    while before > 0:
        older_chunk = monitor.read_log_chunk("chunked-run", "run.log", before=before, chunk_size=80)
        combined = older_chunk["text"] + combined
        before = older_chunk["start_offset"]

    assert "line-000\n" not in latest_chunk["text"]
    assert "line-000\n" in combined
    assert combined.endswith("line-119\n")


def test_parse_args_defaults_to_all_interfaces(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["dashboard.server"])
    args = parse_args()
    assert args.host == "0.0.0.0"
