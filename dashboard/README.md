# Dashboard

This directory contains a lightweight streaming dashboard for `uni_agent` run directories such as `/tmp/swebench_qwen3_coder`.

## Features

- Per-run status inferred from the existing `uni_agent` log flow:
  - `queued`: `run.log` exists, but no actual run markers yet
  - `running`: environment startup or interaction steps have begun
  - `verify`: reward or shutdown phase has started, but the final result file is not written yet
  - `completed`: `interaction_result.json` exists
- Parallel run cards so multiple samples can be watched together
- Incremental log streaming from `run.log`
- Recent file create, modify, and delete events inside each run directory
- Search, status filtering, and overall counters

## Start the Dashboard

From the repository root, run:

```bash
python -m dashboard.server --log-dir /tmp/swebench_qwen3_coder --port 8765
```

By default, the server listens on:

```text
http://0.0.0.0:8765
```

## Open the Page

If you are running locally, open:

```text
http://127.0.0.1:8765
```

## Remote Server Access

If Cursor is connected to a remote server, this default bind mode is closer to `python -m http.server`, so a single command is usually enough for port forwarding or remote preview tools to detect the service:

```bash
python -m dashboard.server --log-dir /tmp/swebench_qwen3_coder --port 8765
```

If you still need manual forwarding, you can use:

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<remote-host>
```

Then open:

```text
http://127.0.0.1:8765
```

If you want to bind only to localhost instead, run:

```bash
python -m dashboard.server --host 127.0.0.1 --log-dir /tmp/swebench_qwen3_coder --port 8765
```

## Notes

- The server only uses the Python standard library.
- It polls the filesystem incrementally and only reads the appended part of `run.log`.
- The page animates newly appended log text word-by-word so active runs feel live.
